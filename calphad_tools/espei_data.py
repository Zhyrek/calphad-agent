"""ESPEI thermochemical datasets (non-ZPF) and a goodness-of-fit report.

Dataset kinds (verified against the installed ESPEI 0.9.1 source)
-----------------------------------------------------------------
* **Fixed-configuration ("non-equilibrium") data** - has a ``solver`` key.  Fitted by
  ``generate_parameters`` and scored in MCMC by ``FixedConfigurationPropertyResidual``.
  Outputs: HM_FORM, SM_FORM, CPM_FORM, HM_MIX, SM_MIX, CPM_MIX (volume V0/VM/VM_MIX are read by
  parameter generation only).  ``values`` shape = (n_P, n_T, n_configurations).
  _FORM = relative to the pure elements in their TDB reference (SER) phases at the same T;
  _MIX  = relative to the phase's own endmembers (pycalphad ``*_MIX``).  _MIX data need
  ``sublattice_occupancies`` (parameter generation raises otherwise); interacting species in a
  sublattice must be sorted alphabetically (check_dataset).  Data with T < 298.15 K are dropped by
  ESPEI (``filter_temperatures``) and configurations containing species that the phase does not have
  are dropped silently (``filter_sublattice_configurations``) - both are checked here.
* **Equilibrium property data** - no ``solver``, output not ZPF/ACR.  ``EquilibriumPropertyResidual``
  evaluates any pycalphad property string (HM, SM, CPM, GM, ``*_MIX``; ``*R`` such as HMR only with
  ``reference_states``) at equilibrium among ``phases``.  ``_FORM`` is NOT accepted by pycalphad here
  (use HMR with SER reference states).  ``values`` shape = (n_P, n_T, n_X); composition conditions
  ``X_<EL>`` are lists (one entry per point, never broadcast).  ESPEI 0.9.1 computes the points in
  X-major order but flattens ``values`` P/T-major, so a dataset with several T (or P) *and* several X
  is compared in the wrong order - the writers here therefore split datasets so that at most one of
  n_P, n_T, n_X exceeds 1.  Standard deviations: HM 500 J/mol, SM 0.2 J/K/mol, CPM 0.2 J/K/mol
  (anything else: 1.0 -> effectively a huge weight).
* **Activity data** - output ``ACR_<EL>`` + ``reference_state`` {"phases": [...], "conditions":
  {P, T, X_...}}.  ``ActivityResidual`` compares chemical potentials (sigma 500 J/mol):
  mu_target = R T ln(a) + mu_ref, with ONE reference equilibrium -> one temperature per dataset.
  All phases of the system are allowed at the data points (the dataset ``phases`` are not used).

Units: ESPEI data are per mole of ATOMS (J/mol-atom, J/K/mol-atom); parameter generation multiplies
by atoms-per-formula itself (``fitting_steps``: "our data are all in [qty]/mole-atoms").  Vacancies
do not count as atoms.  Temperatures in K, pressures in Pa.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

R_GAS = 8.3145
FIXED_OUTPUTS = ["HM_FORM", "SM_FORM", "CPM_FORM", "HM_MIX", "SM_MIX", "CPM_MIX", "V0", "VM", "VM_MIX"]
MCMC_FIXED_OUTPUTS = FIXED_OUTPUTS[:6]
EQ_BASES = ("HM", "SM", "CPM", "GM")
SIGMA = {"HM": 500.0, "SM": 0.2, "CPM": 0.2, "ACR": 500.0, "ZPF": 1000.0}
UNIT = {"HM": "J/mol-atom", "GM": "J/mol-atom", "SM": "J/K/mol-atom", "CPM": "J/K/mol-atom",
        "V0": "m3/mol-atom", "VM": "m3/mol-atom", "ACR": "J/mol (mu)", "ZPF": "J/mol (driving force)"}
DEFAULT_P = 101325.0


class ThermoDataError(ValueError):
    pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _base(output: str) -> str:
    o = output.upper()
    if o.startswith("ACR"):
        return "ACR"
    if o == "ZPF":
        return "ZPF"
    return re.match(r"^(CPM|HM|SM|GM|VM|V0)", o).group(1) if re.match(r"^(CPM|HM|SM|GM|VM|V0)", o) else o


def unit_of(output: str) -> str:
    return UNIT.get(_base(output), "?")


def _unit_factor(value_unit: str) -> Tuple[float, bool]:
    """'kJ/mol-formula' -> (1000, True).  Returns (scale to J, per_formula)."""
    u = (value_unit or "J/mol-atom").strip().lower().replace(" ", "")
    per_formula = "formula" in u or "f.u" in u or "/fu" in u
    if u.startswith("kcal"):
        s = 4184.0
    elif u.startswith("cal"):
        s = 4.184
    elif u.startswith("kj"):
        s = 1000.0
    elif u.startswith("j") or u.startswith("m3") or u in ("", "si"):
        s = 1.0
    else:
        raise ThermoDataError(f"unknown value_unit {value_unit!r} (use e.g. J/mol-atom, kJ/mol-atom, "
                              f"J/mol-formula, kJ/mol-formula, cal/mol-atom)")
    return s, per_formula


def _tk(T, unit: str = "K"):
    off = 273.15 if str(unit).upper().startswith("C") else 0.0
    if isinstance(T, (list, tuple, np.ndarray)):
        return [float(t) + off for t in T]
    return float(T) + off


def _as_list(x) -> List[float]:
    return [float(t) for t in np.atleast_1d(np.asarray(x, dtype=float))]


def _scalar_or_list(xs: Sequence[float]):
    return float(xs[0]) if len(xs) == 1 else [float(x) for x in xs]


def _norm_x_key(k: str) -> Optional[str]:
    m = re.fullmatch(r"\s*X[_(]\s*([A-Za-z]{1,2})\s*\)?\s*", k)
    return f"X_{m.group(1).upper()}" if m else None


def _check_espei(ds: Dict) -> None:
    from espei.datasets import DatasetError, check_dataset, clean_dataset
    try:
        check_dataset(ds)
        clean_dataset(json.loads(json.dumps(ds)))
    except (DatasetError, KeyError, IndexError, TypeError, ValueError) as e:
        raise ThermoDataError(f"ESPEI rejected the dataset: {type(e).__name__}: {e}") from None


def write_dataset(ds: Dict, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ds, indent=2))
    return str(p)


def _sanity(base: str, output: str, vals: np.ndarray, warns: List[str]) -> None:
    a = np.abs(vals[np.isfinite(vals)])
    if a.size == 0:
        raise ThermoDataError("no finite values")
    if base == "HM":
        if a.max() > 2.0e5:
            warns.append(f"|{output}| up to {a.max():.3g} J/mol-atom is very large - check units "
                         f"(J/mol-ATOM expected, not kJ or cal, not per formula unit)")
        elif 0 < a.max() < 60:
            warns.append(f"|{output}| <= {a.max():.3g}: looks like kJ/mol - values must be J/mol-atom "
                         f"(pass value_unit='kJ/mol-atom')")
    elif base == "SM" and a.max() > 60:
        warns.append(f"|{output}| up to {a.max():.3g} J/K/mol-atom is implausible for an entropy - check units")
    elif base == "CPM" and a.max() > 150:
        warns.append(f"|{output}| up to {a.max():.3g} J/K/mol-atom is implausible for a heat capacity")


# ---------------------------------------------------------------------------
# phase information (sublattice model / site ratios)
# ---------------------------------------------------------------------------
def phase_info(phase: str, tdb: Optional[str] = None, phase_models: Optional[Dict] = None,
               components: Optional[Sequence[str]] = None) -> Optional[Dict]:
    """{'sublattice_model': [[...]], 'site_ratios': [...]} from a phase-models dict or a TDB."""
    phase = phase.upper()
    comps = {c.upper() for c in components} | {"VA"} if components else None
    if phase_models:
        ph = phase_models.get("phases", {}).get(phase)
        if ph is None:
            raise ThermoDataError(f"phase {phase} not in phase_models ({sorted(phase_models.get('phases', {}))})")
        return {"sublattice_model": [[s.upper() for s in sl] for sl in ph["sublattice_model"]],
                "site_ratios": [float(r) for r in ph["sublattice_site_ratios"]]}
    if tdb:
        from .calc import get_db
        db = get_db(tdb)
        if phase not in db.phases:
            raise ThermoDataError(f"phase {phase} not in {Path(tdb).name}; phases: {sorted(db.phases)}")
        p = db.phases[phase]
        model = [sorted(str(s.name).upper() for s in sl if comps is None or str(s.name).upper() in comps)
                 for sl in p.constituents]
        return {"sublattice_model": model, "site_ratios": [float(r) for r in p.sublattices]}
    return None


def ser_phases_from_tdb(tdb: str) -> Dict[str, str]:
    from .calc import get_db
    db = get_db(tdb)
    return {str(el).upper(): str(d.get("phase", "")).upper() for el, d in db.refstates.items()
            if d.get("phase") and str(el).upper() not in ("VA", "/-")}


# ---------------------------------------------------------------------------
# 1a. fixed-configuration datasets
# ---------------------------------------------------------------------------
def _norm_config(cfg) -> list:
    out = []
    for sl in cfg:
        if isinstance(sl, (list, tuple)):
            out.append([str(s).upper() for s in sl] if len(sl) > 1 else str(sl[0]).upper())
        else:
            out.append(str(sl).upper())
    return out


def _sort_config(cfg: list, occ: Optional[list]) -> Tuple[list, Optional[list]]:
    """Sort interacting species alphabetically (ESPEI requirement) and permute occupancies with them."""
    new_c, new_o = [], []
    for i, sl in enumerate(cfg):
        if isinstance(sl, list):
            o = occ[i] if occ is not None else None
            order = sorted(range(len(sl)), key=lambda k: sl[k])
            new_c.append([sl[k] for k in order])
            if o is not None:
                if not isinstance(o, (list, tuple)) or len(o) != len(sl):
                    raise ThermoDataError(f"occupancies {occ} do not match the shape of configuration {cfg}")
                new_o.append([float(o[k]) for k in order])
        else:
            new_c.append(sl)
            if occ is not None:
                o = occ[i]
                new_o.append(float(o[0] if isinstance(o, (list, tuple)) else o))
    return new_c, (new_o if occ is not None else None)


def _atoms_per_formula(cfg: list, occ: Optional[list], ratios: Sequence[float]) -> float:
    n = 0.0
    for i, (sl, r) in enumerate(zip(cfg, ratios)):
        if isinstance(sl, list):
            o = occ[i] if occ is not None else [1.0 / len(sl)] * len(sl)
            n += r * (1.0 - sum(f for s, f in zip(sl, o) if s == "VA"))
        else:
            n += 0.0 if sl == "VA" else r
    return n


def mixing_configurations(phase_model: Dict, components: Sequence[str], x: Sequence[float],
                          x_component: Optional[str] = None) -> Tuple[list, list, str]:
    """Configurations/occupancies of a *disordered* binary phase at mole fractions ``x`` of ``x_component``.

    Every sublattice holding both elements gets occupancy [1-x, x]; all other sublattices must hold a
    single species (e.g. VA)."""
    els = sorted(c.upper() for c in components if c.upper() != "VA")
    if len(els) != 2:
        raise ThermoDataError(f"mixing tables need exactly 2 elements, got {els}")
    xc = (x_component or els[1]).upper()
    if xc not in els:
        raise ThermoDataError(f"x_component {xc} not in {els}")
    other = [e for e in els if e != xc][0]
    model = phase_model["sublattice_model"]
    mixing = [i for i, sl in enumerate(model) if xc in sl and other in sl]
    if not mixing:
        raise ThermoDataError(f"no sublattice of the phase contains both {els}: {model}")
    cfg = []
    for i, sl in enumerate(model):
        if i in mixing:
            cfg.append(sorted([other, xc]))
        else:
            rest = [s for s in sl if s in set(els) | {"VA"}]
            if len(rest) != 1:
                raise ThermoDataError(f"sublattice {i} {sl} is neither mixing {els} nor single-species; give "
                                      f"configurations/occupancies explicitly")
            cfg.append(rest[0])
    configs, occs = [], []
    for xv in x:
        xv = float(xv)
        o = []
        for i, sl in enumerate(cfg):
            if isinstance(sl, list):
                o.append([xv if s == xc else 1.0 - xv for s in sl])
            else:
                o.append(1.0)
        configs.append(copy.deepcopy(cfg))
        occs.append(o)
    return configs, occs, xc


def make_thermo_dataset(components: Sequence[str], phase: str, output: str, values, T=298.15, P=DEFAULT_P,
                        configurations: Optional[list] = None, occupancies: Optional[list] = None,
                        site_ratios: Optional[Sequence[float]] = None, phase_model: Optional[Dict] = None,
                        mixing_x: Optional[Sequence[float]] = None, x_component: Optional[str] = None,
                        reference: str = "", comment: str = "", excluded_model_contributions=None,
                        weight: Optional[float] = None, value_unit: str = "J/mol-atom",
                        temperature_unit: str = "K", bibtex: str = "") -> Tuple[Dict, List[str]]:
    """Build + validate a fixed-configuration ESPEI dataset.  Returns (dataset, warnings).

    values: one per configuration (single T/P), [T][config], [T] (one configuration), or full [P][T][config].
    phase_model: {'sublattice_model', 'site_ratios'} (from :func:`phase_info`) - used for defaults and to
    verify the configurations.  mixing_x: build disordered binary configurations from mole fractions."""
    warns: List[str] = []
    output = output.upper().strip()
    if output not in FIXED_OUTPUTS:
        raise ThermoDataError(f"output {output!r} is not a fixed-configuration property; use one of "
                              f"{FIXED_OUTPUTS} (for HMR/ACR/HM at equilibrium use write_equilibrium_dataset)")
    if output not in MCMC_FIXED_OUTPUTS:
        warns.append(f"{output}: used by generate_parameters only (no MCMC residual in ESPEI 0.9)")
    phase = phase.upper()
    comps = [c.upper() for c in components]
    if mixing_x is not None:
        if phase_model is None:
            raise ThermoDataError("mixing_x needs the phase's sublattice model (give tdb or phase_models)")
        configurations, occupancies, xc = mixing_configurations(phase_model, comps, mixing_x, x_component)
        keep = [i for i, xv in enumerate(mixing_x) if 0.0 < float(xv) < 1.0]
        if len(keep) < len(mixing_x):
            warns.append(f"dropped {len(mixing_x) - len(keep)} point(s) at x=0 or 1 (endmembers: a _MIX value is 0)")
        vals_in = np.asarray(values, dtype=float)
        if vals_in.shape[-1] != len(mixing_x):
            raise ThermoDataError(f"{len(mixing_x)} mole fractions but values have shape {vals_in.shape}")
        values = vals_in[..., keep].tolist()
        configurations = [configurations[i] for i in keep]
        occupancies = [occupancies[i] for i in keep]
        comment = (comment + "; " if comment else "") + f"x({xc}) = " + ", ".join(f"{float(mixing_x[i]):.4g}" for i in keep)
    if not configurations:
        raise ThermoDataError("give configurations (per-sublattice, e.g. [['CR','NI'],'VA'] or ['CR','NI']) or mixing_x")
    configs = [_norm_config(c) for c in configurations]
    is_mixing = any(isinstance(sl, list) for c in configs for sl in c)
    if occupancies is None and (is_mixing or output.endswith("_MIX")):
        if is_mixing:
            raise ThermoDataError("configurations with interacting species need occupancies, e.g. [[0.3, 0.7], 1]")
    occs = None
    if occupancies is not None or output.endswith("_MIX"):
        if occupancies is None:
            occupancies = [[1.0] * len(c) for c in configs]
        if len(occupancies) != len(configs):
            raise ThermoDataError(f"{len(configs)} configurations but {len(occupancies)} occupancies")
        occs = []
        new_cfg = []
        for c, o in zip(configs, occupancies):
            if len(o) != len(c):
                raise ThermoDataError(f"occupancy {o} and configuration {c} have different numbers of sublattices")
            c2, o2 = _sort_config(c, list(o))
            new_cfg.append(c2)
            occs.append(o2)
        configs = new_cfg
        for c, o in zip(configs, occs):
            for sl, so in zip(c, o):
                if isinstance(sl, list):
                    if any(f < -1e-9 or f > 1 + 1e-9 for f in so) or abs(sum(so) - 1.0) > 1e-4:
                        raise ThermoDataError(f"site fractions {so} of sublattice {sl} must be in [0,1] and sum to 1")
    else:
        configs = [_sort_config(c, None)[0] for c in configs]
    # sublattice model checks
    ratios = [float(r) for r in site_ratios] if site_ratios is not None else None
    if phase_model is not None:
        model_ratios = phase_model["site_ratios"]
        if ratios is None:
            ratios = model_ratios
        elif len(ratios) != len(model_ratios) or not np.allclose(ratios, model_ratios):
            warns.append(f"site_ratios {ratios} differ from the phase model {model_ratios}")
        model = phase_model["sublattice_model"]
        for c in configs:
            if len(c) != len(model):
                raise ThermoDataError(f"configuration {c} has {len(c)} sublattices; {phase} has {len(model)}: {model}")
            for sl, msl in zip(c, model):
                for s in (sl if isinstance(sl, list) else [sl]):
                    if s not in msl:
                        raise ThermoDataError(f"{s} is not a constituent of sublattice {msl} of {phase} "
                                              f"(ESPEI would silently drop configuration {c})")
    if ratios is None:
        raise ThermoDataError("site_ratios unknown: give site_ratios, or a tdb/phase_models defining the phase")
    used = sorted({s for c in configs for sl in c for s in (sl if isinstance(sl, list) else [sl])})
    missing = [s for s in used if s not in comps and s != "VA"]
    if missing:
        raise ThermoDataError(f"configurations use {missing} which are not in components {comps}")
    ds_comps = sorted({c for c in comps if c != "VA"} & set(used) | ({"VA"} if "VA" in used else set()))
    # conditions / values
    Ts = _as_list(_tk(T, temperature_unit))
    Ps = _as_list(P)
    v = np.asarray(values, dtype=float)
    nP, nT, nC = len(Ps), len(Ts), len(configs)
    if v.shape == (nP, nT, nC):
        pass
    elif v.ndim == 2 and v.shape == (nT, nC) and nP == 1:
        v = v[None]
    elif v.ndim == 1 and nP == 1 and nT == 1 and v.size == nC:
        v = v.reshape(1, 1, nC)
    elif v.ndim == 1 and nP == 1 and nC == 1 and v.size == nT:
        v = v.reshape(1, nT, 1)
    elif v.ndim == 0 and nP == nT == nC == 1:
        v = v.reshape(1, 1, 1)
    else:
        raise ThermoDataError(f"values shape {v.shape} does not fit (P, T, configurations) = ({nP}, {nT}, {nC})")
    if not np.all(np.isfinite(v)):
        raise ThermoDataError("values contain NaN/inf")
    scale, per_formula = _unit_factor(value_unit)
    base = _base(output)
    if base in ("V0", "VM"):
        scale = 1.0
    v = v * scale
    if per_formula:
        apf = np.array([_atoms_per_formula(c, o, ratios) for c, o in
                        zip(configs, occs if occs is not None else [None] * nC)])
        if np.any(apf <= 0):
            raise ThermoDataError("cannot convert per-formula values: a configuration has no atoms")
        v = v / apf[None, None, :]
        warns.append("converted per-formula-unit values to per mole of atoms (divided by "
                     + ", ".join(f"{a:g}" for a in apf) + " atoms/formula)")
    if min(Ts) < 298.15:
        warns.append("ESPEI drops fixed-configuration data below 298.15 K (filter_temperatures)")
    if base in ("HM", "SM", "CPM"):
        _sanity(base, output, v, warns)
    if output.endswith("_MIX"):
        for k, c in enumerate(configs):
            if not any(isinstance(sl, list) for sl in c) and np.any(np.abs(v[..., k]) > 1e-9):
                warns.append(f"{output} of the endmember {c} should be 0 (mixing property of a pure configuration)")
    ds = {"components": ds_comps, "phases": [phase], "solver": {"mode": "manual", "sublattice_site_ratios": ratios,
                                                                  "sublattice_configurations": configs},
          "conditions": {"P": _scalar_or_list(Ps), "T": _scalar_or_list(Ts)}, "output": output,
          "values": v.tolist(), "reference": reference or "unknown", "comment": comment}
    if occs is not None:
        ds["solver"]["sublattice_occupancies"] = occs
    if excluded_model_contributions:
        ds["excluded_model_contributions"] = list(excluded_model_contributions)
    if weight is not None:
        ds["weight"] = float(weight)
    if bibtex:
        ds["bibtex"] = bibtex
    _check_espei(ds)
    return ds, warns


# ---------------------------------------------------------------------------
# 1b. equilibrium property / activity datasets
# ---------------------------------------------------------------------------
def normalize_output(output: str) -> str:
    o = output.upper().replace(" ", "")
    m = re.fullmatch(r"ACR[(_]([A-Z]{1,2})\)?", o)
    if m:
        return f"ACR_{m.group(1)}"
    return o


def _norm_refstates(reference_states: Optional[Dict], comps: Sequence[str]) -> Optional[Dict]:
    if not reference_states:
        return None
    out = {}
    for el, val in reference_states.items():
        el = el.upper()
        if isinstance(val, str):
            out[el] = {"phase": val.upper()}
        else:
            d = {"phase": str(val["phase"]).upper()}
            fsv = val.get("fixed_state_variables") or {k: val[k] for k in ("T", "P") if k in val}
            if fsv:
                d["fixed_state_variables"] = {k.upper(): float(x) for k, x in fsv.items()}
            out[el] = d
    need = [c for c in comps if c != "VA"]
    miss = [c for c in need if c not in out]
    if miss:
        raise ThermoDataError(f"reference_states must define every element {need}; missing {miss}")
    return out


def make_equilibrium_datasets(components: Sequence[str], phases: Sequence[str], output: str, conditions: Dict,
                              values, reference_states: Optional[Dict] = None,
                              reference_state: Optional[Dict] = None, reference: str = "", comment: str = "",
                              weight: Optional[float] = None, value_unit: str = "J/mol-atom",
                              temperature_unit: str = "K", ser_phases: Optional[Dict[str, str]] = None,
                              all_phases: Optional[Sequence[str]] = None, bibtex: str = ""
                              ) -> Tuple[List[Dict], List[str]]:
    """Build + validate equilibrium-property or activity datasets (split so ESPEI evaluates them in order).

    conditions: {"T": scalar|list, "P": scalar|list, "X_NI" (or "X(NI)"): list, ...} with n_elements-1
    composition keys.  values: flat list (when only one condition varies) or nested [P][T][X].
    reference_states (for *R outputs): {"CR": "LIQUID", ...} or {"CR": {"phase":..., "T": 298.15}}.
    ACR_<EL>: reference_state {"phases": [...], "conditions": {...}} or reference_states {EL: phase};
    default = the element's SER phase (ser_phases) at the data temperature."""
    warns: List[str] = []
    comps = sorted({c.upper() for c in components} | {"VA"})
    els = [c for c in comps if c != "VA"]
    output = normalize_output(output)
    phases = [p.upper() for p in phases]
    if all_phases is not None:
        bad = [p for p in phases if p not in {q.upper() for q in all_phases}]
        if bad:
            raise ThermoDataError(f"phases {bad} not in the database ({sorted(all_phases)})")
    is_acr = output.startswith("ACR_")
    refs = None
    if is_acr:
        el = output[4:]
        if el not in els:
            raise ThermoDataError(f"{output}: {el} not in components {els}")
    else:
        m = re.fullmatch(r"(HM|SM|CPM|GM)(R|_MIX|_FORM)?", output)
        if not m:
            raise ThermoDataError(f"unsupported equilibrium output {output!r}; use HM, SM, CPM, GM, HMR/SMR/CPMR/GMR "
                                  f"(+reference_states), *_MIX, or ACR_<EL>")
        suffix = m.group(2) or ""
        refs = _norm_refstates(reference_states, comps)
        if suffix == "_FORM":
            if refs is None:
                if not ser_phases:
                    raise ThermoDataError(f"{output} at equilibrium needs reference_states (or a tdb for SER phases)")
                refs = {e: {"phase": ser_phases[e]} for e in els}
            output = m.group(1) + "R"
            warns.append(f"{m.group(0)} -> {output} with reference_states {refs} (pycalphad has no _FORM at equilibrium)")
        elif suffix == "R" and refs is None:
            output = m.group(1)
            warns.append(f"{m.group(0)} without reference_states is the SER-referenced {output} (as in Thermo-Calc)")
        elif suffix != "R" and refs is not None:
            raise ThermoDataError(f"reference_states only apply to *R outputs (e.g. {m.group(1)}R), not {output}")
        if m.group(1) == "GM":
            warns.append("ESPEI has no standard deviation for GM data (uses 1.0 -> very high weight)")
    # conditions
    cond = {}
    xconds = {}
    for k, val in conditions.items():
        kk = k.strip().upper()
        if kk in ("T", "T_K"):
            cond["T"] = _as_list(_tk(val, temperature_unit))
        elif kk == "T_C":
            cond["T"] = _as_list(_tk(val, "C"))
        elif kk == "P":
            cond["P"] = _as_list(val)
        elif kk == "N":
            continue
        else:
            xk = _norm_x_key(k)
            if xk is None:
                raise ThermoDataError(f"unknown condition {k!r} (use T, P, X_<EL>)")
            xconds[xk] = _as_list(val)
    if "T" not in cond:
        raise ThermoDataError("conditions need T")
    Ts, Ps = cond["T"], cond.get("P", [DEFAULT_P])
    if len(xconds) != len(els) - 1:
        raise ThermoDataError(f"give exactly {len(els) - 1} composition condition(s) X_<EL> for {els}, got {list(xconds)}")
    for xk in xconds:
        if xk[2:] not in els:
            raise ThermoDataError(f"{xk}: {xk[2:]} not in components {els}")
    nX = {len(x) for x in xconds.values()}
    if len(nX) != 1:
        raise ThermoDataError(f"all X_ lists must have the same length: { {k: len(x) for k, x in xconds.items()} }")
    nX = nX.pop()
    X = np.array([xconds[k] for k in sorted(xconds)]).T      # (nX, n_indep)
    if np.any(X < 0) or np.any(X > 1) or np.any(X.sum(axis=1) > 1 + 1e-9):
        raise ThermoDataError("mole fractions must be in [0,1] and sum to <= 1 (convert at.%/wt.% first)")
    v = np.asarray(values, dtype=float)
    shape = (len(Ps), len(Ts), nX)
    if v.shape != shape:
        if v.size == int(np.prod(shape)) and sum(s > 1 for s in shape) <= 1:
            v = v.reshape(shape)
        else:
            raise ThermoDataError(f"values shape {v.shape} does not fit (P, T, X) = {shape}")
    if not np.all(np.isfinite(v)):
        raise ThermoDataError("values contain NaN/inf")
    base = _base(output)
    if is_acr:
        if np.any(v <= 0):
            raise ThermoDataError("activities must be > 0")
        if np.any(v > 1.0 + 1e-9):
            warns.append("activities > 1: only possible if the reference phase is metastable - check reference state")
    else:
        scale, per_formula = _unit_factor(value_unit)
        if per_formula:
            raise ThermoDataError("equilibrium properties are per mole of atoms; convert per-formula values first")
        v = v * scale
        if base in ("HM", "SM", "CPM"):
            _sanity(base, output, v, warns)
    # split so that at most one of nP, nT, nX > 1 (ESPEI ordering) and one T per activity dataset
    blocks = []
    if is_acr or (nX > 1 and (len(Ps) > 1 or len(Ts) > 1)):
        for i, p in enumerate(Ps):
            for j, t in enumerate(Ts):
                blocks.append(([p], [t], list(range(nX)), v[i:i + 1, j:j + 1, :]))
    elif len(Ps) > 1 and len(Ts) > 1:
        for i, p in enumerate(Ps):
            blocks.append(([p], Ts, list(range(nX)), v[i:i + 1]))
    else:
        blocks.append((Ps, Ts, list(range(nX)), v))
    if len(blocks) > 1:
        warns.append(f"split into {len(blocks)} datasets (ESPEI needs one T per activity dataset and at most one "
                     f"varying condition per equilibrium dataset)")
    out = []
    indep = sorted(xconds)
    for bP, bT, idx, bv in blocks:
        ds = {"components": comps, "phases": sorted(set(phases)),
              "conditions": {"P": _scalar_or_list(bP), "T": _scalar_or_list(bT),
                             **{k: [float(X[i, indep.index(k)]) for i in idx] for k in indep}},
              "output": output, "values": bv.tolist(), "reference": reference or "unknown", "comment": comment}
        if is_acr:
            el = output[4:]
            if reference_state:
                rs = copy.deepcopy(reference_state)
                rs["phases"] = [p.upper() for p in rs["phases"]]
                rc = {k.upper() if k.upper() in ("T", "P") else (_norm_x_key(k) or k): x
                      for k, x in rs.get("conditions", {}).items()}
                rc.setdefault("T", bT[0])
                rc.setdefault("P", bP[0])
                if not any(k.startswith("X_") for k in rc):
                    rc.update({k: (1.0 if k[2:] == el else 0.0) for k in indep})
                rs["conditions"] = rc
            else:
                rphase = None
                if reference_states:
                    r = reference_states.get(el) or reference_states.get(el.lower())
                    rphase = (r if isinstance(r, str) else (r or {}).get("phase"))
                if rphase is None:
                    if not ser_phases or el not in ser_phases:
                        raise ThermoDataError(f"{output}: give the reference phase of {el} (reference_states) or a tdb")
                    rphase = ser_phases[el]
                    warns.append(f"{output}: reference = pure {el} in {rphase} (SER phase) at the data temperature")
                rs = {"phases": [rphase.upper()],
                      "conditions": {"P": bP[0], "T": bT[0], **{k: (1.0 if k[2:] == el else 0.0) for k in indep}}}
            if all_phases is not None and any(p not in {q.upper() for q in all_phases} for p in rs["phases"]):
                raise ThermoDataError(f"reference phase {rs['phases']} not in the database")
            ds["reference_state"] = rs
        elif refs is not None:
            ds["reference_states"] = refs
        if weight is not None:
            ds["weight"] = float(weight)
        if bibtex:
            ds["bibtex"] = bibtex
        _check_espei(ds)
        out.append(ds)
    return out, warns


# ---------------------------------------------------------------------------
# 1c. POP thermochemical experiments
# ---------------------------------------------------------------------------
_PROP_RE = re.compile(r"^(LNACR|LNAC|ACR|AC|MUR|MU|HMR|HM|H|SMR|SM|S|GMR|GM|G|CPM|CP)(\.T)?(?:\(([^)]*)\))?(\.T)?$")


def _pop_extras(path: str) -> List[Dict]:
    """Per equilibrium (same order as expdata.parse_pop): reference states and experiments with uncertainties."""
    from .expdata import _expand_pop_tables, _pop_kw
    raw = Path(path).read_text(errors="replace").replace("\r", "")
    lines = []
    for ln in raw.split("\n"):
        lines.append(ln.split("$", 1)[0].strip() if not ln.strip().startswith("$") else "")
    lines = _expand_pop_tables([ln for ln in lines if ln])
    out: List[Dict] = []
    cur = None
    for ln in lines:
        w = ln.split(None, 1)
        kw = _pop_kw(w[0])
        rest = w[1] if len(w) > 1 else ""
        if kw == "CREATE_NEW_EQUILIBRIUM":
            cur = {"refstates": {}, "experiments": {}}
            out.append(cur)
        elif cur is None:
            continue
        elif kw == "SET_REFERENCE_STATE":
            t = [x for x in re.split(r"[\s,]+", rest.strip()) if x]
            if len(t) >= 2:
                T = None if len(t) < 3 or t[2] == "*" else float(t[2])
                P = None if len(t) < 4 or t[3] == "*" else float(t[3])
                cur["refstates"][t[0].upper()] = (t[1].upper(), T, P)
        elif kw in ("EXPERIMENT", "EXPERIMENTAL"):
            for m in re.finditer(r"([A-Za-z][\w.]*(?:\([^)]*\))?(?:\.T)?)\s*=\s*([-\d.Ee+]+)(?:\s*:\s*([-\d.Ee+]+)(%?))?",
                                 rest):
                key = m.group(1).upper().replace(" ", "")
                val = float(m.group(2))
                sig = None
                if m.group(3):
                    sig = float(m.group(3)) * (abs(val) / 100.0 if m.group(4) else 1.0)
                cur["experiments"][key] = (val, sig)
    return out


def pop_to_thermo(path: str, components: Sequence[str], tdb_phases: Optional[Sequence[str]] = None,
                  phase_map: Optional[Dict[str, str]] = None, ser_phases: Optional[Dict[str, str]] = None,
                  reference: str = "", weights_from_uncertainty: bool = True) -> Tuple[List[Dict], List[str]]:
    """Convert the thermochemical EXPERIMENTs of a POP file (ACR, HMR/HM, SM, CP/HM.T, ...) to ESPEI
    equilibrium/activity datasets.  Returns (datasets, notes).  Phase-boundary data -> expdata.pop_to_zpf."""
    from .expdata import parse_pop, phase_resolver, to_mole_fraction
    resolve = phase_resolver(tdb_phases, phase_map)
    eqs = parse_pop(path)
    extras = _pop_extras(path)
    if len(extras) != len(eqs):
        raise ThermoDataError("internal: POP equilibrium count mismatch")
    comps = sorted({c.upper() for c in components if c.upper() != "VA"})
    notes: List[str] = []
    points = []
    for n, (eq, ex) in enumerate(zip(eqs, extras), 1):
        tag = f"eq {n} ({eq.label or 'no label'})"
        T = eq.conditions.get("T")
        P = eq.conditions.get("P", DEFAULT_P)
        xs = {}
        for k, val in eq.conditions.items():
            m = re.fullmatch(r"(X|W)(?:%)?\((\w+)\)", k)
            if m:
                el = m.group(2).upper()
                if m.group(1) == "W":
                    if len(comps) != 2:
                        xs = None
                        break
                    other = [c for c in comps if c != el][0]
                    xs[el] = to_mole_fraction(val, "wt_fraction" if val <= 1 else "wt%", el, other)
                else:
                    xs[el] = val if val <= 1 else val / 100.0
        fixed = [resolve(p) for p, amt in eq.fixed.items() if amt > 0]
        for key, (val, sig) in ex["experiments"].items():
            m = _PROP_RE.match(key)
            if not m:
                if not re.match(r"^(T|X|W)\b|^(T|X\(|W\()", key):
                    notes.append(f"{tag}: experiment {key} not supported by ESPEI - skipped")
                continue
            prop, arg, dT = m.group(1), m.group(3), m.group(2) or m.group(4)
            if T is None:
                notes.append(f"{tag}: {key} without a T condition - skipped")
                continue
            if xs is None or len(xs) < len(comps) - 1 or any(e not in comps for e in xs):
                notes.append(f"{tag}: {key}: composition conditions {list(eq.conditions)} do not give "
                             f"{len(comps) - 1} mole fractions of {comps} - skipped")
                continue
            indep = sorted(xs)[: len(comps) - 1]
            X = tuple(float(xs[e]) for e in indep)
            ph_arg = None
            if prop.startswith(("AC", "LNAC", "MU")):
                a = [s.strip().upper() for s in (arg or "").split(",") if s.strip()]
                if not a:
                    notes.append(f"{tag}: {key} lacks the component - skipped")
                    continue
                el = a[0]
                ph_arg = resolve(a[1]) if len(a) > 1 else None
                if el not in comps:
                    notes.append(f"{tag}: {key}: {el} not in {comps} - skipped")
                    continue
                if prop in ("AC", "LNAC", "MU"):
                    notes.append(f"{tag}: {key} is relative to SER at 298.15 K (absolute); ESPEI activities need a "
                                 f"reference phase - use ACR with SET_REFERENCE_STATE - skipped")
                    continue
                a_val = val if prop == "ACR" else math.exp(val) if prop == "LNACR" else math.exp(val / (R_GAS * T))
                rs = ex["refstates"].get(el)
                if rs is None and not (ser_phases and el in ser_phases):
                    notes.append(f"{tag}: {key}: no SET_REFERENCE_STATE for {el} and no tdb for its SER phase - skipped")
                    continue
                rphase = resolve(rs[0]) if rs else ser_phases[el]
                if rs and rs[1] is not None and abs(rs[1] - T) > 1e-6:
                    notes.append(f"{tag}: {key}: reference state at fixed T={rs[1]} not supported for activities - skipped")
                    continue
                if not rs:
                    notes.append(f"{tag}: {key}: no SET_REFERENCE_STATE - assumed pure {el} in {rphase} at T")
                sig_mu = (R_GAS * T * (sig / a_val if prop == "ACR" else sig if prop == "LNACR" else sig / (R_GAS * T))
                          if sig else None)
                phases = [ph_arg] if ph_arg else (fixed or list(tdb_phases or []) or [rphase])
                points.append(dict(label=eq.label, output=f"ACR_{el}", phases=phases, refkey=rphase, T=T, P=P,
                                   X=X, indep=indep, value=a_val, sigma=sig_mu, base="ACR"))
                continue
            base = {"H": "HM", "HM": "HM", "HMR": "HM", "S": "SM", "SM": "SM", "SMR": "SM", "G": "GM", "GM": "GM",
                    "GMR": "GM", "CP": "CPM", "CPM": "CPM"}[prop]
            if dT:
                if base != "HM":
                    notes.append(f"{tag}: {key} not supported - skipped")
                    continue
                base, prop = "CPM", "CPM"
            ph_arg = resolve(arg.split(",")[0]) if arg else None
            if prop in ("H", "S", "G"):
                notes.append(f"{tag}: {key}: extensive property taken per mole (N=1 assumed)")
            refs = None
            out = base
            if prop.endswith("R") and ex["refstates"] and base != "CPM":
                refs = {}
                for el in comps:
                    rs = ex["refstates"].get(el)
                    if rs:
                        d = {"phase": resolve(rs[0])}
                        if rs[1] is not None:
                            d["fixed_state_variables"] = {"T": rs[1], "P": rs[2] or DEFAULT_P}
                    elif ser_phases and el in ser_phases:
                        d = {"phase": ser_phases[el], "fixed_state_variables": {"T": 298.15, "P": DEFAULT_P}}
                    else:
                        refs = None
                        break
                    refs[el] = d
                if refs is None:
                    notes.append(f"{tag}: {key}: incomplete reference states and no tdb for SER phases - skipped")
                    continue
                out = base + "R"
            phases = [ph_arg] if ph_arg else (fixed or list(tdb_phases or []))
            if not phases:
                notes.append(f"{tag}: {key}: no phase fixed/named and no tdb for the phase list - skipped")
                continue
            if not ph_arg and not fixed:
                notes.append(f"{tag}: {key}: no phase fixed - equilibrium over all {len(phases)} TDB phases")
            points.append(dict(label=eq.label, output=out, phases=phases, refkey=json.dumps(refs, sort_keys=True),
                               refs=refs, T=T, P=P, X=X, indep=indep, value=val, sigma=sig, base=base))
    datasets = []
    groups: Dict[Tuple, List[Dict]] = {}
    for p in points:
        k = (p["label"], p["output"], tuple(sorted(p["phases"])), p["refkey"], tuple(p["indep"]), p["P"])
        groups.setdefault(k, []).append(p)
    for (label, output, phases, refkey, indep, P), pts in groups.items():
        for sub in _split_points(pts, per_T=output.startswith("ACR")):
            Ts = sorted({q["T"] for q in sub}) if len({q["X"] for q in sub}) == 1 else [sub[0]["T"]]
            if len(Ts) > 1:
                sub = sorted(sub, key=lambda q: q["T"])
                conds = {"T": [q["T"] for q in sub], "P": P, **{f"X_{e}": [sub[0]["X"][i]] for i, e in enumerate(indep)}}
                vals = [[[q["value"]] for q in sub]]
            else:
                conds = {"T": Ts[0], "P": P, **{f"X_{e}": [q["X"][i] for q in sub] for i, e in enumerate(indep)}}
                vals = [[[q["value"] for q in sub]]]
            sigs = [q["sigma"] for q in sub if q["sigma"]]
            weight = None
            cm = f"from {Path(path).name}" + (f", label {label}" if label else "")
            if weights_from_uncertainty and sigs:
                weight = round(SIGMA.get(sub[0]["base"], 1.0) / float(np.median(sigs)), 4)
                cm += f"; median uncertainty {np.median(sigs):.3g} {unit_of(output)} -> weight {weight}"
            try:
                kw = {}
                if output.startswith("ACR"):
                    kw["reference_states"] = {output[4:]: refkey}
                elif sub[0].get("refs"):
                    kw["reference_states"] = sub[0]["refs"]
                dss, w = make_equilibrium_datasets(comps, list(phases), output, conds, vals, reference=reference or Path(path).name,
                                                   comment=cm, weight=weight, ser_phases=ser_phases, all_phases=tdb_phases, **kw)
                datasets += dss
                notes += [f"label {label!r} {output}: {x}" for x in w if "reference =" not in x]
            except ThermoDataError as e:
                notes.append(f"label {label!r} {output}: {e}")
    return datasets, notes


def _split_points(pts: List[Dict], per_T: bool = False) -> List[List[Dict]]:
    """Group points into ESPEI-safe blocks: one composition with several T, or one T with several compositions."""
    xs = {p["X"] for p in pts}
    if len(xs) == 1 and not per_T:
        return [pts]
    by_t: Dict[float, List[Dict]] = {}
    for p in pts:
        by_t.setdefault(p["T"], []).append(p)
    return list(by_t.values())


# ---------------------------------------------------------------------------
# 1d. CSV tables
# ---------------------------------------------------------------------------
_CSV_FIXED = set(FIXED_OUTPUTS)


def _parse_cfg(s: str):
    return [([x.strip().upper() for x in sl.split(",")] if "," in sl else sl.strip().upper()) for sl in s.split(":")]


def _parse_occ(s: str):
    return [([float(x) for x in sl.split(",")] if "," in sl else float(sl)) for sl in s.split(":")]


def csv_to_thermo(path: str, components: Sequence[str], phase: Optional[str] = None,
                  phase_model_fn=None, ser_phases: Optional[Dict[str, str]] = None,
                  all_phases: Optional[Sequence[str]] = None, reference_phase: Optional[str] = None,
                  value_unit: str = "J/mol-atom", composition_unit: str = "mole_fraction",
                  reference: str = "", weights_from_uncertainty: bool = True) -> Tuple[List[Dict], List[str]]:
    """Calorimetry/activity table -> ESPEI datasets.  See the convert_csv_to_thermo tool for the columns."""
    from .expdata import to_mole_fraction
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ThermoDataError("empty CSV")
    cols = [c for c in rows[0].keys() if c is not None]
    ncol = {c: c.strip() for c in cols}
    tcol = next((c for c in cols if re.fullmatch(r"T(_K|_C|\(K\)|\(C\))?", ncol[c], re.I)), None)
    celsius = bool(tcol and "C" in ncol[tcol].upper()[1:])
    pcol = next((c for c in cols if re.fullmatch(r"P(_Pa)?", ncol[c], re.I)), None)
    xcols = {c: re.fullmatch(r"X[(_](\w+)\)?", ncol[c], re.I).group(1).upper() for c in cols
             if re.fullmatch(r"X[(_](\w+)\)?", ncol[c], re.I)}
    props = {}
    for c in cols:
        u = ncol[c].upper()
        if u in _CSV_FIXED or re.fullmatch(r"(HM|SM|CPM|GM)(R)?", u) or re.fullmatch(r"ACR[(_]\w+\)?", u):
            props[c] = normalize_output(u)
    if not props:
        raise ThermoDataError(f"no property column found in {cols}; name it e.g. HM_MIX, HM_FORM, HMR, HM, CPM, SM, ACR(CR)")
    comps = sorted({c.upper() for c in components if c.upper() != "VA"})

    def col(*names):
        for c in cols:
            if ncol[c].lower() in names:
                return c
        return None
    phcol, refcol, rpcol = col("phase"), col("reference", "ref"), col("ref_phase", "reference_phase")
    cfgcol, occcol = col("configuration", "config"), col("occupancy", "occupancies")
    notes: List[str] = []
    groups: Dict[Tuple, List[Dict]] = {}
    for n, r in enumerate(rows, 2):
        if tcol is None or not (r.get(tcol) or "").strip():
            if tcol is None:
                raise ThermoDataError("CSV needs a T_K (or T_C) column")
            continue
        T = float(r[tcol]) + (273.15 if celsius else 0.0)
        P = float(r[pcol]) if pcol and (r.get(pcol) or "").strip() else DEFAULT_P
        xs = {}
        for c, el in xcols.items():
            s = (r.get(c) or "").strip()
            if s:
                other = [k for k in comps if k != el]
                xs[el] = to_mole_fraction(float(s), composition_unit, el, other[0]) if other else float(s)
        ph = ((r.get(phcol) or "").strip().upper() if phcol else "") or (phase or "").upper()
        ref = ((r.get(refcol) or "").strip() if refcol else "") or reference or Path(path).name
        for c, out in props.items():
            s = (r.get(c) or "").strip()
            if not s:
                continue
            sc = next((k for k in cols if ncol[k].lower() in (f"{ncol[c].lower()}_err", f"{ncol[c].lower()}_sigma",
                                                             "uncertainty", "sigma", "err", "error")), None)
            sig = float(r[sc]) if sc and (r.get(sc) or "").strip() else None
            rp = ((r.get(rpcol) or "").strip().upper() if rpcol else "") or (reference_phase or "").upper()
            p = dict(row=n, T=T, P=P, xs=xs, value=float(s), sigma=sig, phase=ph, ref_phase=rp,
                     cfg=(r.get(cfgcol) or "").strip() if cfgcol else "", occ=(r.get(occcol) or "").strip() if occcol else "")
            groups.setdefault((ref, out, ph, rp, P), []).append(p)
    datasets: List[Dict] = []
    for (ref, out, ph, rp, P), pts in groups.items():
        base = _base(out)
        sigs = [p["sigma"] for p in pts if p["sigma"]]
        weight, cm = None, f"from {Path(path).name}"
        if weights_from_uncertainty and sigs:
            s_med = float(np.median(sigs)) * _unit_factor(value_unit)[0] if base != "ACR" else float(np.median(
                [R_GAS * p["T"] * p["sigma"] / p["value"] for p in pts if p["sigma"]]))
            weight = round(SIGMA.get(base, 1.0) / s_med, 4)
            cm += f"; median uncertainty -> weight {weight}"
        try:
            if out in _CSV_FIXED:
                if not ph:
                    raise ThermoDataError(f"{out} needs a phase (column 'phase' or the phase argument)")
                pm = phase_model_fn(ph) if phase_model_fn else None
                same_cfg = len({(p["cfg"], p["occ"], tuple(sorted(p["xs"].items()))) for p in pts}) == 1
                blocks = [pts] if same_cfg else list(_group_by(pts, "T").values())
                for blk in blocks:
                    blk_T = [p["T"] for p in blk] if same_cfg else blk[0]["T"]
                    if blk[0]["cfg"]:
                        cfgs = [_parse_cfg(p["cfg"]) for p in (blk[:1] if same_cfg else blk)]
                        occs = ([_parse_occ(p["occ"]) for p in (blk[:1] if same_cfg else blk)]
                                if blk[0]["occ"] else None)
                        if occs is None and any(isinstance(sl, list) for c in cfgs for sl in c) and len(comps) == 2:
                            occs = []
                            for p, c in zip(blk[:1] if same_cfg else blk, cfgs):
                                xe = sorted(comps)[1]
                                x = p["xs"].get(xe, 1 - p["xs"].get(sorted(comps)[0], float("nan")))
                                occs.append([[x if s == xe else 1 - x for s in sl] if isinstance(sl, list) else 1.0
                                             for sl in c])
                        vals = [p["value"] for p in blk]
                        ds, w = make_thermo_dataset(comps, ph, out, vals, T=blk_T, P=P, configurations=cfgs,
                                                    occupancies=occs, phase_model=pm, reference=ref, comment=cm,
                                                    weight=weight, value_unit=value_unit)
                    else:
                        if len(comps) != 2:
                            raise ThermoDataError(f"{out} without a configuration column needs a binary system")
                        xe = sorted(comps)[1]
                        xv = [p["xs"].get(xe, 1 - p["xs"].get(sorted(comps)[0], float("nan"))) for p in blk]
                        if any(math.isnan(x) for x in xv):
                            raise ThermoDataError(f"{out}: rows need X({xe}) (or a configuration column)")
                        if same_cfg:
                            ds, w = make_thermo_dataset(comps, ph, out, [[p["value"]] for p in blk], T=blk_T, P=P,
                                                        phase_model=pm, mixing_x=[xv[0]], x_component=xe,
                                                        reference=ref, comment=cm, weight=weight, value_unit=value_unit)
                        else:
                            ds, w = make_thermo_dataset(comps, ph, out, [p["value"] for p in blk], T=blk_T, P=P,
                                                        phase_model=pm, mixing_x=xv, x_component=xe, reference=ref,
                                                        comment=cm, weight=weight, value_unit=value_unit)
                    datasets.append(ds)
                    notes += [f"{ref} {out} {ph}: {x}" for x in w]
                continue
            # equilibrium / activity
            indep = sorted(comps)[1:] if len(comps) > 1 else []
            for p in pts:
                if len(p["xs"]) < len(comps) - 1:
                    raise ThermoDataError(f"row {p['row']}: needs X() columns for {len(comps) - 1} of {comps}")
                if not set(indep) <= set(p["xs"]):
                    known = p["xs"]
                    if len(comps) == 2:
                        p["xs"] = {indep[0]: 1 - known[comps[0]]}
                    else:
                        raise ThermoDataError(f"row {p['row']}: give X of {indep}")
                p["X"] = tuple(p["xs"][e] for e in indep)
            phases = [ph] if ph else list(all_phases or [])
            if not phases:
                raise ThermoDataError(f"{out}: give a phase (column/argument) or a tdb")
            kw: Dict[str, Any] = {}
            if out.startswith("ACR"):
                if rp:
                    kw["reference_states"] = {out[4:]: rp}
            elif out.endswith("R") and out != "HM" and re.fullmatch(r"(HM|SM|CPM|GM)R", out):
                if not rp:
                    raise ThermoDataError(f"{out} needs ref_phase (column or reference_phase argument), e.g. LIQUID for "
                                          f"mixing enthalpies relative to the pure liquids")
                kw["reference_states"] = {e: rp for e in comps}
            for blk in _split_points(pts, per_T=out.startswith("ACR")):
                if len({p["X"] for p in blk}) == 1 and len(blk) > 1:
                    blk = sorted(blk, key=lambda q: q["T"])
                    conds = {"T": [p["T"] for p in blk], "P": P, **{f"X_{e}": [blk[0]["X"][i]] for i, e in enumerate(indep)}}
                    vals = [[[p["value"]] for p in blk]]
                else:
                    conds = {"T": blk[0]["T"], "P": P, **{f"X_{e}": [p["X"][i] for p in blk] for i, e in enumerate(indep)}}
                    vals = [[[p["value"] for p in blk]]]
                dss, w = make_equilibrium_datasets(comps, phases, out, conds, vals, reference=ref, comment=cm,
                                                   weight=weight, value_unit=value_unit, ser_phases=ser_phases,
                                                   all_phases=all_phases, **kw)
                datasets += dss
                notes += [f"{ref} {out}: {x}" for x in w]
        except ThermoDataError as e:
            notes.append(f"{ref} {out} {ph}: NOT CONVERTED - {e}")
    return datasets, notes


def _group_by(pts, key):
    g: Dict[Any, List] = {}
    for p in pts:
        g.setdefault(p[key], []).append(p)
    return g


# ---------------------------------------------------------------------------
# 2. goodness of fit
# ---------------------------------------------------------------------------
def _load_one(path: str) -> Optional[Dict]:
    from espei.datasets import check_dataset, clean_dataset
    d = json.loads(Path(path).read_text())
    if d.get("disabled"):
        return None
    check_dataset(d)
    return clean_dataset(d)


def _tinydb(ds: Dict):
    from espei.utils import PickleableTinyDB
    from tinydb.storages import MemoryStorage
    db = PickleableTinyDB(storage=MemoryStorage)
    db.insert(copy.deepcopy(ds))
    return db


def _kind(ds: Dict) -> str:
    if ds["output"] == "ZPF":
        return "ZPF"
    if "solver" in ds:
        return "fixed"
    if ds["output"].startswith("ACR"):
        return "activity"
    return "equilibrium"


def _cfg_str(cfg, occ) -> str:
    parts = []
    for i, sl in enumerate(cfg):
        if isinstance(sl, (list, tuple)):
            o = occ[i] if occ is not None else None
            parts.append(",".join(f"{s}{'' if o is None else f'={float(f):.3g}'}" for s, f in zip(sl, o or [None] * len(sl))))
        else:
            parts.append(str(sl))
    return ":".join(parts)


def _spec(comps, phase_models):
    from espei.phase_models import PhaseModelSpecification
    phases = (phase_models or {}).get("phases", {})
    return PhaseModelSpecification(components=sorted(comps), phases=phases)


def evaluate_dataset(dbf, ds: Dict, comps: Sequence[str], phase_models: Optional[Dict] = None,
                     data_weights: Optional[Dict] = None) -> Tuple[List[Dict], List[str]]:
    """Per-point comparison of one (cleaned) dataset using ESPEI's residual functions (parameters as in the TDB)."""
    from espei.core_utils import ravel_conditions
    data_weights = data_weights or {}
    kind = _kind(ds)
    spec = _spec(comps, phase_models)
    db1 = _tinydb(ds)
    notes: List[str] = []
    rows: List[Dict] = []
    empty = np.array([])
    w_ds = float(ds.get("weight", 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "fixed":
            from espei.error_functions.non_equilibrium_thermochemical_error import (
                FixedConfigurationPropertyResidual, filter_sublattice_configurations)
            from espei.core_utils import filter_temperatures
            ph = ds["phases"][0]
            if ds["output"] not in MCMC_FIXED_OUTPUTS:
                return [], [f"{ds['output']} is not scored by ESPEI's MCMC residuals"]
            res = FixedConfigurationPropertyResidual(dbf, db1, spec, symbols_to_fit=[], weight=data_weights)
            if not res.thermochemical_data:
                return [], [f"not used by ESPEI (phase {ph} not in the TDB/system or no matching configuration)"]
            r, sig = res.get_residuals(empty)
            r = np.asarray(r, dtype=float).ravel()
            sig = np.asarray(sig, dtype=float).ravel()
            d = copy.deepcopy(ds)
            constituents = res.thermochemical_data[0]["constituents"]
            n_before = len(d["solver"]["sublattice_configurations"])
            d = filter_temperatures(filter_sublattice_configurations([d], constituents))[0]
            cfgs = d["solver"]["sublattice_configurations"]
            if len(cfgs) < n_before:
                notes.append(f"{n_before - len(cfgs)} configuration(s) not representable by the TDB model - ignored by ESPEI")
            vals = np.asarray(d["values"], dtype=float)
            occs = d["solver"].get("sublattice_occupancies") or [None] * len(cfgs)
            P, T, ci = ravel_conditions(vals, np.atleast_1d(d["conditions"]["P"]), np.atleast_1d(d["conditions"]["T"]),
                                        np.arange(len(cfgs)))
            exp = vals.ravel()
            if exp.size != r.size:
                return [], [f"internal: {r.size} residuals for {exp.size} values"]
            for i in range(exp.size):
                c = int(ci[i])
                rows.append(dict(T=float(T[i]), P=float(P[i]), point=f"{ph}({_cfg_str(cfgs[c], occs[c])})",
                                 exp=float(exp[i]), calc=float(exp[i] + r[i]), residual=float(r[i]), sigma=float(sig[i])))
        elif kind == "equilibrium":
            from espei.error_functions.equilibrium_thermochemical_error import EquilibriumPropertyResidual
            res = EquilibriumPropertyResidual(dbf, db1, spec, symbols_to_fit=[], weight=data_weights)
            if not res.property_data:
                return [], ["not used by ESPEI (phases/components not in the system)"]
            r, sig = res.get_residuals(empty)
            r = np.asarray(r, dtype=float)
            sig = np.asarray(sig, dtype=float)
            vals = np.asarray(ds["values"], dtype=float)
            nP, nT, nX = vals.shape
            exp = vals.ravel()
            calc_espei_order = r + exp                     # ESPEI: X-major, then (P, T)
            calc = calc_espei_order.reshape(nX, nP, nT).transpose(1, 2, 0).ravel()
            if nX > 1 and nP * nT > 1:
                notes.append("ESPEI 0.9.1 compares this dataset in the wrong order (several T/P AND several X) - "
                             "residuals below are re-ordered correctly; split the dataset for fitting")
            xk = sorted(k for k in ds["conditions"] if k.startswith("X_"))
            P, T, xi = ravel_conditions(vals, np.atleast_1d(ds["conditions"]["P"]), np.atleast_1d(ds["conditions"]["T"]),
                                        np.arange(nX))
            for i in range(exp.size):
                comp = ", ".join(f"{k[2:]}={np.atleast_1d(ds['conditions'][k])[int(xi[i])]:.4g}" for k in xk)
                rows.append(dict(T=float(T[i]), P=float(P[i]), point=f"{'+'.join(ds['phases'])} x({comp})",
                                 exp=float(exp[i]), calc=float(calc[i]), residual=float(calc[i] - exp[i]),
                                 sigma=float(sig[i])))
        elif kind == "activity":
            from espei.error_functions.activity_error import ActivityResidual
            res = ActivityResidual(dbf, db1, spec, symbols_to_fit=[], weight=data_weights)
            r, sig = res.get_residuals(empty)
            r = np.asarray(r, dtype=float)
            vals = np.asarray(ds["values"], dtype=float)
            if r.size != vals.size:
                return [], ["not used by ESPEI (components not in the system)"]
            xk = [k for k in ds["conditions"] if k not in ("P", "T")]
            P, T, xi = ravel_conditions(vals, np.atleast_1d(ds["conditions"]["P"]), np.atleast_1d(ds["conditions"]["T"]),
                                        np.arange(vals.shape[2]))
            exp = vals.ravel()
            for i in range(exp.size):
                comp = ", ".join(f"{k[2:]}={np.atleast_1d(ds['conditions'][k])[int(xi[i])]:.4g}" for k in xk)
                calc = float(exp[i] * math.exp(r[i] / (R_GAS * T[i]))) if np.isfinite(r[i]) else float("nan")
                rows.append(dict(T=float(T[i]), P=float(P[i]), point=f"x({comp})", exp=float(exp[i]), calc=calc,
                                 residual=float(r[i]), sigma=float(sig[i])))
            notes.append(f"activity reference: {ds['reference_state']['phases']} at {ds['reference_state']['conditions']}")
        else:
            from espei.error_functions.zpf_error import (driving_force_to_hyperplane, estimate_hyperplane,
                                                         get_zpf_data)
            from pycalphad.core.utils import filter_phases, unpack_species
            phases = sorted(filter_phases(dbf, unpack_species(dbf, sorted(comps)), dbf.phases.keys()))
            missing = [p for p in ds["phases"] if p not in phases]
            if missing:
                return [], [f"phases {missing} not in the TDB/system - ESPEI cannot evaluate this dataset"]
            zd = get_zpf_data(dbf, sorted(comps), phases, db1, {}, spec.get_model_dict())
            if not zd:
                return [], ["not used by ESPEI (components/phases not in the system)"]
            zd = zd[0]
            sig = SIGMA["ZPF"] / float(data_weights.get("ZPF", 1.0)) / w_ds
            Tc = np.atleast_1d(ds["conditions"]["T"])
            n_undet = 0
            for i, region in enumerate(zd["phase_regions"]):
                raw = [vx for vx in ds["values"][i] if vx[0].upper() != "__HYPERPLANE__"]
                try:
                    hp = estimate_hyperplane(region, dbf, empty)
                except Exception as e:  # noqa: BLE001
                    hp = None
                    notes.append(f"point {i}: hyperplane failed ({type(e).__name__})")
                undet = hp is None or np.any(np.isnan(hp))
                n_undet += bool(undet)
                ph_all = "+".join(vx[0] for vx in raw)
                for vtx, rv in zip(region.vertices, raw):
                    if undet:
                        df = float("nan")
                    else:
                        try:
                            df = float(driving_force_to_hyperplane(hp, region, dbf, zd["parameter_dict"], vtx, empty))
                        except Exception:  # noqa: BLE001
                            df = float("nan")
                    xs = ", ".join(f"{c}={x:.4g}" if x is not None else f"{c}=?" for c, x in zip(rv[1], rv[2]))
                    rows.append(dict(T=float(Tc[i] if Tc.size > 1 else Tc[0]), P=float(np.atleast_1d(ds["conditions"]["P"])[0]),
                                     point=f"{vtx.phase_name} x({xs}) in {ph_all}", exp=None, calc=None, residual=df,
                                     sigma=sig, phase=vtx.phase_name, zpf_index=i,
                                     composition=[float(x) for x in vtx.composition], undetermined=undet))
            if n_undet:
                notes.append(f"{n_undet} tie-line(s) with undetermined target hyperplane (ESPEI scores 0 there)")
    return rows, notes


def fit_report(tdb: str, datasets_dir: str, out_prefix: str, elements: Optional[Sequence[str]] = None,
               phase_models: Optional[Dict] = None, data_weights: Optional[Dict] = None, diagram: bool = True,
               pptx: bool = False, progress=None) -> Dict:
    """Compare every dataset under ``datasets_dir`` with ``tdb``.  Writes CSVs + PNGs (+ PPTX); returns a summary."""
    from espei.datasets import recursive_glob
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dbf = Database(tdb)
    db_els = {str(e).upper() for e in dbf.elements} - {"/-"}
    system = None
    if phase_models:
        system = {c.upper() for c in phase_models["components"]}
    if elements:
        system = {e.upper() for e in elements} | {"VA"}
    files = recursive_glob(datasets_dir, "*.json")
    if not files:
        raise FileNotFoundError(f"no *.json datasets under {datasets_dir}")
    all_rows, summary, skipped = [], [], []
    for f in files:
        rel = os.path.relpath(f, datasets_dir)
        try:
            ds = _load_one(f)
        except Exception as e:  # noqa: BLE001
            skipped.append((rel, f"invalid: {e}"))
            continue
        if ds is None:
            skipped.append((rel, "disabled"))
            continue
        dcomps = {c.upper() for c in ds["components"]}
        if system is not None and not dcomps <= system:
            skipped.append((rel, f"components {sorted(dcomps)} outside the system"))
            continue
        if not dcomps - {"VA"} <= db_els:
            skipped.append((rel, f"elements {sorted(dcomps - db_els - {'VA'})} not in the TDB"))
            continue
        comps = sorted(dcomps | ({"VA"} if "VA" in db_els else set()))
        if progress:
            progress(f"fit_report: {rel}")
        try:
            rows, notes = evaluate_dataset(dbf, ds, comps, phase_models, data_weights)
        except Exception as e:  # noqa: BLE001
            skipped.append((rel, f"evaluation failed: {type(e).__name__}: {e}"))
            continue
        kind = _kind(ds)
        unit = unit_of(ds["output"])
        for r in rows:
            r.update(file=rel, reference=ds.get("reference", ""), kind=kind, output=ds["output"], unit=unit,
                     components=sorted(dcomps - {"VA"}))
            r["z"] = r["residual"] / r["sigma"] if r["sigma"] and np.isfinite(r["residual"]) else float("nan")
        all_rows += rows
        res = np.array([r["residual"] for r in rows], dtype=float)
        ok = np.isfinite(res)
        z = np.array([r["z"] for r in rows], dtype=float)
        worst = sorted([r for r in rows if np.isfinite(r["residual"])], key=lambda r: -abs(r["z"]))[:3]
        summary.append(dict(file=rel, reference=ds.get("reference", ""), kind=kind, output=ds["output"],
                            phases="+".join(ds.get("phases", [])), n=len(rows), n_failed=int((~ok).sum()), unit=unit,
                            rms=float(np.sqrt(np.mean(res[ok] ** 2))) if ok.any() else float("nan"),
                            max_abs=float(np.max(np.abs(res[ok]))) if ok.any() else float("nan"),
                            bias=float(np.mean(res[ok])) if ok.any() else float("nan"),
                            rms_z=float(np.sqrt(np.mean(z[ok] ** 2))) if ok.any() else float("nan"),
                            sigma=float(np.median([r["sigma"] for r in rows])) if rows else float("nan"),
                            worst=[f"T={w['T']:.1f} {w['point']}: " + (f"exp {w['exp']:.4g} calc {w['calc']:.4g} "
                                   if w.get("exp") is not None else "") + f"res {w['residual']:.4g}" for w in worst],
                            notes=notes))
    summary.sort(key=lambda s: -(s["rms_z"] if np.isfinite(s["rms_z"]) else 1e300))
    for i, s in enumerate(summary, 1):
        s["rank"] = i
    out = {"summary": summary, "skipped": skipped, "files": {}}
    pre = Path(out_prefix)
    pre.parent.mkdir(parents=True, exist_ok=True)
    rcsv = str(pre) + "_residuals.csv"
    with open(rcsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "reference", "kind", "output", "T_K", "P_Pa", "point", "experimental", "calculated",
                    "residual", "unit", "sigma", "residual_over_sigma"])
        for r in all_rows:
            w.writerow([r["file"], r["reference"], r["kind"], r["output"], f"{r['T']:.2f}", f"{r['P']:.6g}", r["point"],
                        "" if r["exp"] is None else f"{r['exp']:.6g}", "" if r["calc"] is None else f"{r['calc']:.6g}",
                        f"{r['residual']:.6g}", r["unit"], f"{r['sigma']:.4g}", f"{r['z']:.4g}"])
    scsv = str(pre) + "_summary.csv"
    with open(scsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "file", "reference", "kind", "output", "phases", "n", "n_failed", "rms", "max_abs", "bias",
                    "unit", "sigma", "rms_over_sigma", "worst_points", "notes"])
        for s in summary:
            w.writerow([s["rank"], s["file"], s["reference"], s["kind"], s["output"], s["phases"], s["n"], s["n_failed"],
                        f"{s['rms']:.4g}", f"{s['max_abs']:.4g}", f"{s['bias']:.4g}", s["unit"], f"{s['sigma']:.4g}",
                        f"{s['rms_z']:.4g}", " | ".join(s["worst"]), " | ".join(s["notes"])])
    out["files"].update(residuals_csv=rcsv, summary_csv=scsv)
    images = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = parity_figure(all_rows)
    if fig is not None:
        p = str(pre) + "_parity.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        images.append(p)
    fig = histogram_figure(all_rows)
    if fig is not None:
        p = str(pre) + "_residual_hist.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        images.append(p)
    if diagram:
        systems = sorted({tuple(r["components"]) for r in all_rows if r["kind"] == "ZPF"})
        for sysc in systems:
            if len(sysc) != 2:
                continue
            try:
                fig = zpf_overlay_figure(dbf, list(sysc), [r for r in all_rows if r["kind"] == "ZPF"
                                                            and tuple(r["components"]) == sysc],
                                         title=f"{Path(tdb).name} vs ZPF data")
                p = str(pre) + f"_zpf_{'-'.join(sysc)}.png"
                fig.savefig(p, dpi=140, bbox_inches="tight")
                plt.close(fig)
                images.append(p)
            except Exception as e:  # noqa: BLE001
                out.setdefault("warnings", []).append(f"ZPF overlay {'-'.join(sysc)} failed: {type(e).__name__}: {e}")
    out["images"] = images
    if pptx:
        out["files"]["pptx"] = _write_pptx(str(pre) + "_fit_report.pptx", tdb, summary, skipped, images)
    out["text"] = report_text(summary, skipped, Path(tdb).name)
    out["n_points"] = len(all_rows)
    out["rows"] = all_rows
    return out


def report_text(summary: List[Dict], skipped, tdb_name: str, top: int = 5) -> str:
    lines = [f"Goodness of fit of {tdb_name}: {len(summary)} datasets (ranked worst first by RMS/sigma; sigma = "
             f"ESPEI's likelihood std: HM 500 J/mol, SM/CPM 0.2 J/K/mol, activity 500 J/mol in mu, ZPF 1000 J/mol, "
             f"divided by weights)",
             f"{'#':>2} {'RMS/sig':>8} {'RMS':>10} {'max':>10} {'unit':<22} {'n':>4} output/kind          file (reference)"]
    for s in summary:
        lines.append(f"{s['rank']:>2} {s['rms_z']:8.3g} {s['rms']:10.4g} {s['max_abs']:10.4g} {s['unit']:<22} "
                     f"{s['n']:>4} {s['output'] + '/' + s['kind']:<20} {s['file']} ({s['reference']})"
                     + (f" [{s['n_failed']} failed]" if s["n_failed"] else ""))
    lines.append("")
    for s in summary[:top]:
        if s["worst"] or s["notes"]:
            lines.append(f"#{s['rank']} {s['file']}: worst: " + "; ".join(s["worst"]))
            for n in s["notes"]:
                lines.append(f"    note: {n}")
    for s in summary[top:]:
        for n in s["notes"]:
            lines.append(f"#{s['rank']} {s['file']} note: {n}")
    if skipped:
        lines.append("skipped: " + "; ".join(f"{f}: {why}" for f, why in skipped))
    return "\n".join(lines)


# ------------------------------------------------------------------ plots
def _group_key(r) -> str:
    return "ACR" if r["output"].startswith("ACR") else r["output"]


def parity_figure(rows: List[Dict]):
    import matplotlib.pyplot as plt
    pts = [r for r in rows if r["exp"] is not None and r["calc"] is not None and np.isfinite(r["calc"])]
    if not pts:
        return None
    groups: Dict[str, List[Dict]] = {}
    for r in pts:
        groups.setdefault(_group_key(r), []).append(r)
    n = len(groups)
    nc = min(3, n)
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.3 * nr), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (g, rs) in zip(axes.flat, groups.items()):
        files = sorted({r["file"] for r in rs})
        for k, f in enumerate(files):
            sel = [r for r in rs if r["file"] == f]
            label = (sel[0]["reference"] or f)[:28] + (f" ({sel[0]['output']})" if g == "ACR" else "")
            ax.scatter([r["exp"] for r in sel], [r["calc"] for r in sel], s=26, color=cmap(k % 10), edgecolor="k",
                       linewidth=0.4, label=label, zorder=3)
        allv = np.array([r["exp"] for r in rs] + [r["calc"] for r in rs], dtype=float)
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
        pad = 0.06 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, zorder=1)
        if g != "ACR":
            s = float(np.median([r["sigma"] for r in rs]))
            ax.fill_between([lo, hi], [lo - s, hi - s], [lo + s, hi + s], color="0.85", zorder=0,
                            label=f"±σ ({s:.3g})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        unit = "activity" if g == "ACR" else unit_of(g)
        ax.set_xlabel(f"experimental {g} [{unit}]")
        ax.set_ylabel(f"calculated {g} [{unit}]")
        ax.set_title(f"{g}: {len(rs)} points")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def histogram_figure(rows: List[Dict]):
    import matplotlib.pyplot as plt
    groups: Dict[str, List[float]] = {}
    for r in rows:
        if not np.isfinite(r["residual"]):
            continue
        g = "ZPF" if r["kind"] == "ZPF" else _group_key(r)
        groups.setdefault(g, []).append(r["residual"] if g == "ZPF" else r["z"])
    if not groups:
        return None
    n = len(groups)
    nc = min(3, n)
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.3 * nr), squeeze=False)
    for ax, (g, vals) in zip(axes.flat, groups.items()):
        vals = np.asarray(vals, dtype=float)
        ax.hist(vals, bins=max(5, min(30, int(np.sqrt(vals.size)) * 2)), color="#4C72B0", edgecolor="white")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("driving force [J/mol]" if g == "ZPF" else f"(calc - exp)/σ")
        ax.set_ylabel("count")
        ax.set_title(f"{g}: n={vals.size}, RMS={np.sqrt(np.mean(vals ** 2)):.3g}")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def zpf_overlay_figure(dbf, elements: Sequence[str], rows: List[Dict], title: str = "", x_step: float = 0.02,
                       t_step: float = 10.0):
    """Calculated binary diagram with the ZPF data points coloured by |driving force|."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from pycalphad import binplot, variables as v
    from .diagrams import TIELINE_COLOR, phases_for
    a, b = sorted(e.upper() for e in elements)
    pts = [r for r in rows if r.get("composition") is not None and len(r["composition"]) == 2
           and np.all(np.isfinite(r["composition"]))]
    Ts = [r["T"] for r in rows]
    lo = max(200.0, math.floor((min(Ts) - 150) / 50) * 50)
    hi = math.ceil((max(Ts) + 150) / 50) * 50
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ph = phases_for(dbf, [a, b])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        binplot(dbf, [a, b, "VA"], ph, {v.P: DEFAULT_P, v.T: (lo, hi, t_step), v.X(b): (0, 1, x_step), v.N: 1},
                plot_kwargs=dict(ax=ax, tielines=False, tieline_color=TIELINE_COLOR))
    leg = ax.get_legend()
    handles = list(leg.legend_handles) if leg is not None and hasattr(leg, "legend_handles") else []
    df = np.array([abs(r["residual"]) if np.isfinite(r["residual"]) else np.nan for r in pts])
    vmax = max(500.0, float(np.nanpercentile(df, 95)) if np.isfinite(df).any() else 500.0)
    norm = Normalize(vmin=0, vmax=vmax)
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    ties: Dict[Tuple, List[Dict]] = {}
    for r in pts:
        ties.setdefault((r.get("file"), r.get("zpf_index")), []).append(r)
    for tie in ties.values():
        if len(tie) >= 2:
            ax.plot([r["composition"][1] for r in tie], [r["T"] for r in tie], color="0.55", lw=0.8, ls="--",
                    zorder=4)
    phases = sorted({r["phase"] for r in pts})
    sc = None
    for k, p in enumerate(phases):
        sel = [r for r in pts if r["phase"] == p]
        d = np.array([abs(r["residual"]) if np.isfinite(r["residual"]) else np.nan for r in sel])
        good = np.isfinite(d)
        xs = np.array([r["composition"][1] for r in sel])
        ts = np.array([r["T"] for r in sel])
        if good.any():
            sc = ax.scatter(xs[good], ts[good], c=d[good], cmap="plasma", norm=norm, marker=markers[k % len(markers)],
                            s=46, edgecolor="k", linewidth=0.6, zorder=5)
        if (~good).any():
            ax.scatter(xs[~good], ts[~good], facecolor="none", edgecolor="0.4", marker=markers[k % len(markers)],
                       s=46, linewidth=0.8, zorder=5)
        handles.append(Line2D([], [], marker=markers[k % len(markers)], ls="none", mfc="0.75", mec="k",
                              label=f"data: {p}"))
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("|driving force| [J/mol]")
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.18, 0.5), fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Mole fraction {b}")
    ax.set_ylabel("Temperature [K]")
    n_unk = len(rows) - len(pts)
    ax.set_title((title or f"{a}-{b}") + (f"  ({n_unk} vertices without composition not shown)" if n_unk else ""),
                 fontsize=10)
    return fig


def _write_pptx(path: str, tdb: str, summary, skipped, images) -> str:
    from . import pptx_utils as pu
    prs = pu.new_presentation()
    pu.add_title_slide(prs, f"Goodness of fit: {Path(tdb).name}",
                       f"{len(summary)} ESPEI datasets, ranked by RMS/σ")
    rows = [[str(s["rank"]), s["file"][:40], (s["reference"] or "")[:28], s["output"], str(s["n"]), f"{s['rms']:.4g}",
             f"{s['max_abs']:.4g}", s["unit"], f"{s['rms_z']:.3g}"] for s in summary]
    pu.add_table_slides(prs, "Datasets (worst first)", ["#", "file", "reference", "output", "n", "RMS", "max", "unit",
                                                        "RMS/σ"], rows, [0.4, 3.0, 2.2, 1.1, 0.5, 1.1, 1.1, 1.9, 0.8],
                        font_size=9)
    wrows = [[str(s["rank"]), s["file"][:36], w[:110]] for s in summary[:8] for w in s["worst"]]
    if wrows:
        pu.add_table_slides(prs, "Worst points", ["#", "file", "point"], wrows, [0.4, 3.2, 8.5], font_size=9)
    for im in images:
        pu.add_image_slide(prs, Path(im).stem.replace("_", " "), im)
    if skipped:
        pu.add_bullets_slide(prs, "Skipped datasets", [f"{f}: {w}" for f, w in skipped][:14])
    return pu.save(prs, path)


# ---------------------------------------------------------------------------
# agent tools
# ---------------------------------------------------------------------------
def register(tb) -> None:
    S = {"type": "string"}
    N = {"type": "number"}
    B = {"type": "boolean"}
    STRS = {"type": "array", "items": S}
    NUMS = {"type": "array", "items": N}
    NUM_OR_LIST = {"anyOf": [N, NUMS]}
    VU = {"type": "string", "description": "J/mol-atom (default), kJ/mol-atom, J/mol-formula, cal/mol-atom..."}

    def _pm(phase_models):
        if not phase_models:
            return None
        d = json.loads(Path(tb.p(phase_models)).read_text())
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def _file(file: Optional[str], default: str) -> str:
        name = file if (file or "").endswith(".json") else (file + ".json" if file else None)
        return tb.outp(name, default)

    def _write_many(dss, od: Path, stem: str) -> List[str]:
        files = []
        for i, d in enumerate(dss, 1):
            tag = d["output"] + "_" + "-".join(d["phases"])[:30]
            files.append(write_dataset(d, str(od / f"{stem}_{tag}_{i}.json")))
        return files

    @tb.add("write_thermo_dataset",
            "Write an ESPEI fixed-configuration dataset (fitted by generate_parameters): output HM_FORM|SM_FORM|"
            "CPM_FORM|HM_MIX|SM_MIX|CPM_MIX, per mole of ATOMS (J/mol-atom, J/K/mol-atom); _FORM vs elements in SER "
            "phases, _MIX vs the phase's endmembers. configurations per sublattice, e.g. [\"CR\",\"NI\"] or "
            "[[\"CR\",\"NI\"],\"VA\"] + occupancies [[0.3,0.7],1]; or mixing_x (fractions of x_component) for a "
            "disordered binary phase. values: per configuration, [T][config], or per T (one configuration). Site "
            "ratios from tdb/phase_models. ESPEI-validated.",
            {"components": STRS, "phase": S, "output": S, "values": {"type": "array"}, "T": NUM_OR_LIST, "P": N,
             "temperature_unit": {"type": "string", "enum": ["K", "C"]}, "configurations": {"type": "array"},
             "occupancies": {"type": "array"}, "mixing_x": NUMS, "x_component": S, "site_ratios": NUMS, "tdb": S,
             "phase_models": S, "reference": S, "comment": S, "excluded_model_contributions": STRS, "weight": N,
             "value_unit": VU, "file": {"type": "string", "description": "output .json path"}},
            ["components", "phase", "output", "values", "reference", "file"])
    def write_thermo_dataset(components, phase, output, values, reference, file, T=298.15, P=DEFAULT_P,
                             temperature_unit="K", configurations=None, occupancies=None, mixing_x=None,
                             x_component=None, site_ratios=None, tdb=None, phase_models=None, comment="",
                             excluded_model_contributions=None, weight=None, value_unit="J/mol-atom"):
        pm = phase_info(phase, tb._find_tdb(tdb) if tdb else None, _pm(phase_models), components)
        ds, warns = make_thermo_dataset(components, phase, output, values, T=T, P=P, configurations=configurations,
                                        occupancies=occupancies, site_ratios=site_ratios, phase_model=pm,
                                        mixing_x=mixing_x, x_component=x_component, reference=reference,
                                        comment=comment, excluded_model_contributions=excluded_model_contributions,
                                        weight=weight, value_unit=value_unit, temperature_unit=temperature_unit)
        out = _file(file, "thermo.json")
        write_dataset(ds, out)
        n = int(np.asarray(ds["values"]).size)
        return (f"wrote {tb.rel(out)}: {ds['output']} {ds['phases'][0]}, {n} values "
                f"({len(ds['solver']['sublattice_configurations'])} configurations), site ratios "
                f"{ds['solver']['sublattice_site_ratios']}" + ("\nwarnings:\n  " + "\n  ".join(warns) if warns else "")), []

    @tb.add("write_equilibrium_dataset",
            "Write ESPEI equilibrium-property/activity datasets. output: HM, SM, CPM, HM_MIX, HMR/SMR (needs "
            "reference_states {EL: phase or {phase, T}}), ACR_<EL> (reference_states {EL: phase}; default SER phase "
            "at data T). conditions {T, P, X_<EL>: list} (n_elements-1 X lists). values J/mol-atom, flat if one "
            "condition varies, else [P][T][X]. phases: allowed in the equilibrium. Split per T as ESPEI needs; "
            "ESPEI-validated.",
            {"components": STRS, "phases": STRS, "output": S, "conditions": {"type": "object"},
             "values": {"type": "array"}, "reference_states": {"type": "object"}, "reference_state": {"type": "object"},
             "temperature_unit": {"type": "string", "enum": ["K", "C"]}, "tdb": S, "reference": S, "comment": S,
             "weight": N, "value_unit": VU, "file": {"type": "string", "description": "output .json path"}},
            ["components", "phases", "output", "conditions", "values", "reference", "file"])
    def write_equilibrium_dataset(components, phases, output, conditions, values, reference, file,
                                  reference_states=None, reference_state=None, temperature_unit="K", tdb=None,
                                  comment="", weight=None, value_unit="J/mol-atom"):
        ser, allp = None, None
        if tdb:
            t = tb._find_tdb(tdb)
            ser = ser_phases_from_tdb(t)
            from .calc import get_db
            allp = sorted(get_db(t).phases)
        dss, warns = make_equilibrium_datasets(components, phases, output, conditions, values,
                                               reference_states=reference_states, reference_state=reference_state,
                                               reference=reference, comment=comment, weight=weight,
                                               value_unit=value_unit, temperature_unit=temperature_unit,
                                               ser_phases=ser, all_phases=allp)
        out = _file(file, "equilibrium.json")
        files = [write_dataset(dss[0], out)] if len(dss) == 1 else \
            [write_dataset(d, out[:-5] + f"_{i}.json") for i, d in enumerate(dss, 1)]
        return (f"wrote {len(files)} dataset(s) ({dss[0]['output']}):\n" + "\n".join(tb.rel(f) for f in files)
                + ("\nwarnings:\n  " + "\n  ".join(warns) if warns else "")), []

    @tb.add("convert_pop_to_thermo",
            "Convert thermochemical EXPERIMENTs of a Thermo-Calc POP file (ACR/LNACR + SET_REFERENCE_STATE, HM/HMR, "
            "SM, CP/HM.T, phase-specific HMR(LIQ)) to ESPEI datasets; lists unsupported ones. Phase boundaries: "
            "convert_pop_to_zpf. Give tdb. value:err uncertainties set weights.",
            {"path": S, "components": STRS, "tdb": S, "phase_map": {"type": "object", "additionalProperties": S},
             "output_dir": S, "weights_from_uncertainty": B}, ["path", "components"])
    def convert_pop_to_thermo(path, components, tdb=None, phase_map=None, output_dir=None,
                              weights_from_uncertainty=True):
        phases, ser = None, None
        if tdb:
            from .calc import get_db
            t = tb._find_tdb(tdb)
            phases, ser = sorted(get_db(t).phases), ser_phases_from_tdb(t)
        dss, notes = pop_to_thermo(tb.p(path), components, tdb_phases=phases, phase_map=phase_map, ser_phases=ser,
                                   weights_from_uncertainty=weights_from_uncertainty)
        od = tb.outdir(output_dir, "datasets")
        files = _write_many(dss, od, Path(path).stem)
        return (f"wrote {len(files)} datasets:\n" + "\n".join(tb.rel(f) for f in files)
                + ("\nnotes:\n" + "\n".join(notes) if notes else "")), []

    @tb.add("convert_csv_to_thermo",
            "CSV of calorimetry/activity data -> ESPEI datasets. Columns: T_K|T_C; X(EL); property columns "
            "HM_MIX/SM_MIX/CPM_MIX/HM_FORM/... (fixed-configuration: binary disordered phase from X, or "
            "'configuration' CR,NI:VA + 'occupancy' 0.3,0.7:1), HMR/SMR (needs ref_phase), HM/SM/CPM, ACR(EL) "
            "(ref_phase default SER). Optional: phase, reference, ref_phase, P, uncertainty|<PROP>_err (-> weight). "
            "Values J/mol-atom unless value_unit.",
            {"path": S, "components": STRS, "phase": S, "tdb": S, "phase_models": S, "reference_phase": S,
             "value_unit": VU, "composition_unit": {"type": "string", "enum": ["mole_fraction", "at%", "wt%", "wt_fraction"]},
             "reference": S, "output_dir": S}, ["path", "components"])
    def convert_csv_to_thermo(path, components, phase=None, tdb=None, phase_models=None, reference_phase=None,
                              value_unit="J/mol-atom", composition_unit="mole_fraction", reference="", output_dir=None):
        t = tb._find_tdb(tdb) if tdb else None
        pmd = _pm(phase_models)
        allp = ser = None
        if t:
            from .calc import get_db
            allp, ser = sorted(get_db(t).phases), ser_phases_from_tdb(t)
        dss, notes = csv_to_thermo(tb.p(path), components, phase=phase,
                                   phase_model_fn=(lambda ph: phase_info(ph, t, pmd, components)) if (t or pmd) else None,
                                   ser_phases=ser, all_phases=allp, reference_phase=reference_phase,
                                   value_unit=value_unit, composition_unit=composition_unit, reference=reference)
        od = tb.outdir(output_dir, "datasets")
        files = _write_many(dss, od, Path(path).stem)
        return (f"wrote {len(files)} datasets:\n" + "\n".join(tb.rel(f) for f in files)
                + ("\nnotes:\n" + "\n".join(notes) if notes else "")), []

    @tb.add("fit_report",
            "Goodness of fit of a TDB vs all ESPEI datasets in a directory via ESPEI's residual functions (ZPF "
            "driving force, equilibrium/activity, fixed-configuration). Ranked per-dataset summary (RMS/sigma), "
            "residual CSVs, parity plots, histograms, binary ZPF overlays coloured by driving force, optional PPTX.",
            {"tdb": S, "datasets_dir": S, "elements": STRS, "phase_models": S, "output_prefix": S,
             "data_weights": {"type": "object", "description": "ESPEI mcmc data_weights, e.g. {\"ZPF\": 1, \"HM\": 1}"},
             "diagram": B, "pptx": B}, ["tdb", "datasets_dir"])
    def fit_report_tool(tdb, datasets_dir, elements=None, phase_models=None, output_prefix=None, data_weights=None,
                        diagram=True, pptx=False):
        t = tb._find_tdb(tdb)
        pre = tb.outp(output_prefix, f"fit_{Path(t).stem}/fit_{Path(t).stem}")
        r = fit_report(t, tb.p(datasets_dir), pre, elements=elements, phase_models=_pm(phase_models),
                       data_weights=data_weights, diagram=diagram, pptx=pptx, progress=tb.ui.progress)
        files = "\n".join(f"{k}: {tb.rel(v)}" for k, v in r["files"].items())
        imgs = "\n".join(tb.rel(i) for i in r["images"])
        extra = ("\nwarnings: " + "; ".join(r["warnings"])) if r.get("warnings") else ""
        return f"{r['text']}\n\n{files}\nimages:\n{imgs}{extra}", r["images"][:4]
