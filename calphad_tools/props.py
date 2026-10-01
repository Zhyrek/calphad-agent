"""Equilibrium-property, single-phase-property and raw-data tools for the chat agent.

Built on pycalphad 0.11 (``pycalphad.core.workspace.Workspace`` + ``pycalphad.property_framework``)
so that everything pycalphad can compute can be requested *without* running arbitrary Python:

* ``equilibrium_property_plot`` - any equilibrium property (phase fractions, phase compositions,
  site fractions, degree of order, G/H/S/Cp, chemical potentials, activities, driving forces of
  dormant phases, isolated-phase (metastable) properties, T0, Jansson derivatives such as
  ``HM.T``) over a 1-D or 2-D condition grid -> CSV (+netCDF) + line plot / map + summary.
* ``phase_property_plot`` - single-phase properties (pycalphad ``calculate``) of each phase vs
  composition (binary, one T) or vs T (fixed composition), incl. mixing properties, custom
  pure-element reference states, all sampled points, and the common-tangent construction.
* ``export_equilibrium_data`` - raw ``pycalphad.equilibrium`` Dataset to netCDF + a tidy CSV
  (one row per stable phase per grid point), optionally with the legacy ``eqplot`` figure.
* ``plot_csv`` - re-plot / overlay columns of CSV files (tool outputs or experimental data).

Every function can also be used directly from Python; ``register(tb)`` wires them into the ToolBox.
"""
from __future__ import annotations

import math
import re
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KELVIN = 273.15
MAX_POINTS = 5000            # equilibrium grid points per call
MAX_POINTS_SLOW = 1500       # ... when DF / ISOLATED / DORMANT / T0 quantities are requested
MAX_PANELS = 12
R_GAS = 8.314462618

QUANTITY_HELP = (
    "Quantity syntax (pycalphad property framework, case-insensitive): "
    "GM HM SM CPM (system, per mole of atoms); GM(PH) HM(PH) ... (one phase); GM_MIX(PH) HM_MIX(PH) SM_MIX(PH); "
    "NP(PH) or NP(*) (phase fractions); X(EL) overall; X(PH,EL) or X(*,EL) (phase composition); W(PH,EL) mass "
    "fraction; Y(PH,SUBL,SPECIES) site fraction (SUBL from 0, '*' allowed); DOO(PH) degree of order; MU(EL) "
    "chemical potential (SER); MUR(EL[,REFPH]) chemical potential relative to pure EL in REFPH at the same T,P; "
    "ACR(EL[,REFPH]) activity (default REFPH = SER phase of EL from the TDB); DF(PH) driving force of PH as "
    "a dormant phase (J/mol, >0 means PH would precipitate; leave PH out of `phases` to get its driving force "
    "while suspended); ISOLATED(PH):PROP property of PH alone (metastable single-phase equilibrium), e.g. "
    "ISOLATED(FCC_A1):GM; DORMANT(PH):PROP property of dormant PH at the equilibrium chemical potentials, e.g. "
    "DORMANT(SIGMA):X(CR) (nucleus composition); T0(PH1,PH2) T0 temperature; derivatives 'A.B' (e.g. HM.T = "
    "apparent heat capacity, NP(LIQUID).T); T, P. Optional units suffix in brackets: HM[kJ/mol], HM[J/g], GM[kJ/mol].")


from . import pcache  # noqa: E402  (process-wide model / compiled-code cache)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _get_db(tdb: str):
    from .calc import get_db
    return get_db(tdb)


def _phases_for(db, els, include=None, exclude=None):
    from .diagrams import phases_for
    return phases_for(db, els, include=include, exclude=exclude)


def _db_elements(db) -> List[str]:
    return sorted(str(e) for e in db.elements if str(e) not in ("VA", "/-"))


def _check_elements(db, elements) -> List[str]:
    els = [str(e).upper() for e in elements]
    known = _db_elements(db)
    bad = [e for e in els if e not in known]
    if bad:
        raise ValueError(f"element(s) {bad} not in the database; available elements: {known}")
    if len(set(els)) != len(els):
        raise ValueError(f"duplicate elements in {els}")
    return els


def _comps(db, els):
    return list(els) + (["VA"] if "VA" in {str(e) for e in db.elements} else [])


def _resolve_phases(db, els, phases=None, exclude_phases=None) -> List[str]:
    avail = _phases_for(db, els)
    if phases:
        want = [p.upper() for p in phases]
        bad = [p for p in want if p not in db.phases]
        if bad:
            raise ValueError(f"unknown phase(s) {bad}; phases available for {'-'.join(els)}: {avail}")
        not_here = [p for p in want if p not in avail]
        if not_here:
            raise ValueError(f"phase(s) {not_here} cannot form from elements {els} (no suitable constituents); "
                             f"available: {avail}")
        ph = want
    else:
        ph = list(avail)
    if exclude_phases:
        ex = {p.upper() for p in exclude_phases}
        bad = [p for p in ex if p not in db.phases]
        if bad:
            raise ValueError(f"unknown phase(s) in exclude_phases {bad}; available: {avail}")
        ph = [p for p in ph if p not in ex]
    if not ph:
        raise ValueError("no phases left to calculate")
    return ph


def _is_celsius(unit: Optional[str]) -> bool:
    return bool(unit) and str(unit).strip().upper().startswith("C")


def _fmt(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "nan"
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        return f"{x:.4g}"
    return f"{x:.4g}" if abs(x) < 10 else f"{x:.1f}".rstrip("0").rstrip(".")


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:80] or "x"


def _pretty_units(u) -> str:
    s = str(u or "")
    rep = {"kelvin": "K", "pascal": "Pa", "fraction": "", "dimensionless": "", "joule": "J", "mole": "mol",
           "gram": "g", "kilojoule": "kJ", "degree_Celsius": "°C"}
    for a, b in rep.items():
        s = re.sub(rf"\b{a}\b", b, s)
    s = s.replace(" / ", "/").replace(" * ", "·").strip()
    return s


def _values(val, what: str) -> np.ndarray:
    """number -> [number]; [start, stop, step] -> inclusive range; other lists -> explicit values."""
    if isinstance(val, dict):
        if {"start", "stop"} <= set(val):
            val = [val["start"], val["stop"], val.get("step", (val["stop"] - val["start"]) / 20.0)]
        elif "values" in val:
            return np.asarray([float(x) for x in val["values"]], dtype=float)
        else:
            raise ValueError(f"condition {what}: use a number, [start, stop, step] or a list of values")
    if isinstance(val, (list, tuple, np.ndarray)):
        vals = [float(x) for x in val]
        if len(vals) == 0:
            raise ValueError(f"condition {what}: empty list")
        if len(vals) == 3:
            a, b, st = vals
            if st == 0 or (b - a) / st < 0:
                raise ValueError(f"condition {what}: [start, stop, step] = {vals} is not a valid range "
                                 "(step must be non-zero and point from start to stop)")
            n = int(math.floor((b - a) / st + 1e-9)) + 1
            if n > 100000:
                raise ValueError(f"condition {what}: range has {n} points - use a coarser step")
            return a + st * np.arange(n)
        return np.asarray(vals, dtype=float)
    return np.asarray([float(val)], dtype=float)


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------
_COND_RE = re.compile(r"(X|W|MU)\(([A-Z][A-Z]?)\)")


def parse_conditions(conditions: Dict, els: Sequence[str], unit: str = "K"):
    """Return (conds {prop: array}, labels {label: prop}, varying [label...], values {label: array (impl units)})."""
    from pycalphad import variables as v
    from pycalphad.property_framework import as_property
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("conditions must be an object like {\"T\": [800, 1400, 25], \"X(CR)\": 0.2}")
    conds, labels, values = OrderedDict(), OrderedDict(), OrderedDict()
    for k, val in conditions.items():
        ku = str(k).strip().upper().replace(" ", "")
        if ku in ("T", "T_K", "T_C", "TEMPERATURE"):
            prop, lab = v.T, "T"
        elif ku in ("P", "PRESSURE", "P_PA"):
            prop, lab = v.P, "P"
        elif ku == "N":
            prop, lab = v.N, "N"
        elif ku in els:
            prop, lab = v.X(ku), f"X({ku})"
        else:
            m = _COND_RE.fullmatch(ku)
            if m:
                if m.group(2) not in els:
                    raise ValueError(f"condition {k}: {m.group(2)} is not one of the elements {list(els)}")
                prop, lab = getattr(v, m.group(1))(m.group(2)), f"{m.group(1)}({m.group(2)})"
            else:
                try:
                    prop, lab = as_property(str(k).strip()), str(k).strip().upper()
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"unknown condition {k!r}: use T, P, N, X(EL), W(EL), MU(EL) "
                                     f"(elements {list(els)})") from e
        arr = _values(val, lab)
        if lab == "T":
            if _is_celsius(unit) or ku == "T_C":
                arr = arr + KELVIN
            if np.any(arr <= 0):
                raise ValueError(f"temperature must be > 0 K (got {arr.min():g} K); check temperature_unit")
        if lab.startswith(("X(", "W(")) and (np.any(arr < 0) or np.any(arr > 1)):
            raise ValueError(f"{lab} must be a fraction between 0 and 1 (got {arr.min():g}..{arr.max():g}); "
                             "convert at.% / wt.% to fractions")
        if lab in labels:
            raise ValueError(f"condition {lab} given twice")
        conds[prop], labels[lab], values[lab] = arr, prop, arr
    if "T" not in labels:
        raise ValueError("a temperature condition T is required")
    if "P" not in labels:
        conds[v.P], labels["P"], values["P"] = np.array([101325.0]), v.P, np.array([101325.0])
    if "N" not in labels:
        conds[v.N], labels["N"], values["N"] = np.array([1.0]), v.N, np.array([1.0])
    varying = [lab for lab, arr in values.items() if len(arr) > 1]
    if len(varying) > 2:
        raise ValueError(f"at most two conditions may vary (got {varying}); make the others single numbers")
    return conds, labels, varying, values


def _w_to_x(db, els, conds, labels, values):
    """Scalar mass-fraction conditions -> mole fractions (pycalphad's W conditions converge poorly)."""
    from pycalphad import variables as v
    wl = [lab for lab in labels if lab.startswith("W(")]
    if not wl or any(len(values[lab]) > 1 for lab in wl) or any(lab.startswith("X(") for lab in labels):
        return None
    w = {lab[2:-1]: float(values[lab][0]) for lab in wl}
    bal = [e for e in els if e not in w]
    if len(bal) != 1 or sum(w.values()) >= 1:
        return None
    w[bal[0]] = 1 - sum(w.values())
    mass = {e: float(db.refstates.get(e, {}).get("mass", 0) or 0) for e in els}
    if any(m <= 0 for m in mass.values()):
        return None
    tot = sum(w[e] / mass[e] for e in els)
    note = []
    for lab in wl:
        el = lab[2:-1]
        x = (w[el] / mass[el]) / tot
        del conds[labels[lab]], labels[lab], values[lab]
        conds[v.X(el)], labels[f"X({el})"], values[f"X({el})"] = np.array([x]), v.X(el), np.array([x])
        note.append(f"{lab}={w[el]:g} -> X({el})={x:.5f}")
    return "mass fractions converted to mole fractions: " + ", ".join(note)


def _cond_axis_label(lab: str, unit: str) -> str:
    if lab == "T":
        return "Temperature (°C)" if _is_celsius(unit) else "Temperature (K)"
    if lab == "P":
        return "Pressure (Pa)"
    if lab.startswith("X("):
        return f"Mole fraction {lab[2:-1].capitalize()}"
    if lab.startswith("W("):
        return f"Mass fraction {lab[2:-1].capitalize()}"
    if lab.startswith("MU("):
        return f"μ({lab[3:-1].capitalize()}) (J/mol)"
    return lab


def _cond_display(lab: str, arr: np.ndarray, unit: str) -> np.ndarray:
    return arr - KELVIN if (lab == "T" and _is_celsius(unit)) else arr


def _describe_conditions(values, unit) -> str:
    out = []
    for lab, arr in values.items():
        if lab == "N" and len(arr) == 1 and arr[0] == 1:
            continue
        d = _cond_display(lab, arr, unit)
        u = ("°C" if _is_celsius(unit) else "K") if lab == "T" else ("Pa" if lab == "P" else "")
        if len(d) == 1:
            out.append(f"{lab}={_fmt(d[0])}{(' ' + u) if u else ''}")
        else:
            out.append(f"{lab}={_fmt(d[0])}..{_fmt(d[-1])}{(' ' + u) if u else ''} ({len(d)} pts)")
    return ", ".join(out)


# ---------------------------------------------------------------------------
# quantities
# ---------------------------------------------------------------------------
class _Guarded:
    """Wraps a meta-property so failed points (no compsets / solver errors) give NaN instead of aborting."""

    def __init__(self, inner, label):
        self.inner = inner
        self.label = label
        self.shape = getattr(inner, "shape", tuple())
        self.implementation_units = inner.implementation_units
        self.display_units = inner.display_units
        self.display_name = getattr(inner, "display_name", label)
        self.failures = 0

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        if len(compsets) == 0:
            return np.nan
        try:
            return self.inner.compute_property(compsets, cur_conds, chemical_potentials)
        except Exception:  # noqa: BLE001
            self.failures += 1
            return np.nan

    def __str__(self):
        return self.label

    def __hash__(self):
        return id(self)


def _label_of(p) -> str:
    from pycalphad import variables as v
    from pycalphad.property_framework import JanssonDerivative
    if isinstance(p, _Guarded):
        return p.label
    if isinstance(p, v.PhaseFraction):
        return f"NP({p.phase_name})"
    if isinstance(p, v.MoleFraction):
        return f"X({p.phase_name},{p.species.name})" if p.phase_name else f"X({p.species.name})"
    if isinstance(p, v.MassFraction):
        ph = getattr(p, "phase_name", None)
        return f"W({ph},{p.species.name})" if ph else f"W({p.species.name})"
    if isinstance(p, v.ChemicalPotential):
        return f"MU({p.species.name})"
    if isinstance(p, v.SiteFraction):
        return f"Y({p.phase_name},{p.sublattice_index},{p.species.name})"
    if isinstance(p, JanssonDerivative):
        return f"{_label_of(p.numerator)}.{_label_of(p.denominator)}"
    return str(p)


def _split_units(q: str) -> Tuple[str, Optional[str]]:
    m = re.fullmatch(r"\s*(.*?)\s*\[\s*([^\]]+)\s*\]\s*", q)
    if m:
        return m.group(1), m.group(2)
    return q.strip(), None


def _normalize_q(q: str) -> str:
    return re.sub(r"\s+", "", q).upper()


def _with_units(prop, units):
    if not units:
        return prop
    try:
        return prop[units]
    except Exception:  # noqa: BLE001
        prop.display_units = units
        return prop


def _phase_prop(inner: str, ph: str) -> str:
    """Rewrite a bare property for use inside ISOLATED(PH): / DORMANT(PH):."""
    m = re.fullmatch(r"([A-Z_]+)(?:\((.*)\))?", inner)
    if not m:
        return inner
    name, args = m.group(1), m.group(2)
    if args is None:
        if name in ("T", "P", "N"):
            return inner
        if name in ("NP",):
            return f"NP({ph})"
        return f"{name}({ph})"
    parts = [a for a in args.split(",") if a]
    if name in ("X", "W") and len(parts) == 1:
        return f"{name}({ph},{parts[0]})"
    return inner


class _Spec:
    def __init__(self, text: str):
        self.text = text
        self.kind = "pc"          # pc | acr | mur
        self.prop = None
        self.units = None
        self.el = None
        self.ref_phase = None
        self.wildcard = "*" in text
        self.slow = False


def parse_quantities(quantities: Sequence[str], els: Sequence[str], db, wks_factory) -> List[_Spec]:
    """Parse quantity strings.  wks_factory() returns the (computed) Workspace (needed by meta-properties)."""
    from pycalphad.property_framework import DormantPhase, IsolatedPhase, T0, as_property
    from pycalphad.property_framework.metaproperties import DrivingForce, find_first_compset
    if isinstance(quantities, str):
        quantities = [quantities]
    if not quantities:
        raise ValueError("give at least one quantity. " + QUANTITY_HELP)
    specs = []
    for raw in quantities:
        body, units = _split_units(str(raw))
        q = _normalize_q(body)
        aliases = {"NP": "NP(*)", "PHASE_FRACTIONS": "NP(*)", "CP": "CPM", "H": "HM", "G": "GM", "S": "SM",
                   "DOO": "DOO(*)"}
        q = aliases.get(q, q)
        sp = _Spec(q if not units else f"{q}[{units}]")
        sp.units = units
        try:
            m_meta = re.fullmatch(r"(ISOLATED|DORMANT)\(([A-Z0-9_]+)\):(.+)", q)
            m_df = re.fullmatch(r"(?:DF|DRIVINGFORCE|DGM)\(([A-Z0-9_]+)\)", q)
            m_t0 = re.fullmatch(r"T0\(([A-Z0-9_]+),([A-Z0-9_]+)\)", q)
            m_acr = re.fullmatch(r"(ACR|MUR|AC)\(([A-Z][A-Z]?)(?:,([A-Z0-9_]+))?\)", q)
            if m_acr:
                sp.kind = "mur" if m_acr.group(1) == "MUR" else "acr"
                sp.el = m_acr.group(2)
                if sp.el not in els:
                    raise ValueError(f"{sp.el} is not one of the elements {list(els)}")
                sp.ref_phase = m_acr.group(3)
                if sp.ref_phase and sp.ref_phase not in db.phases:
                    raise ValueError(f"unknown reference phase {sp.ref_phase}")
                sp.prop = as_property(f"MU({sp.el})")
            elif m_df or m_meta or m_t0:
                sp.slow = True
                wks = wks_factory()
                if m_df:
                    ph = m_df.group(1)
                    _check_phase(db, ph)
                    dp = DormantPhase(ph, wks)
                    if dp._compset is None:
                        raise ValueError(f"could not find a starting configuration for {ph}")
                    dfp = DrivingForce(ph)
                    if units:
                        dfp.display_units = units
                    sp.prop = _Guarded(dp(dfp), f"DF({ph})")
                elif m_meta:
                    which, ph, inner = m_meta.groups()
                    _check_phase(db, ph)
                    inner_s = _phase_prop(inner, ph)
                    if inner_s in ("DF", "DRIVINGFORCE"):
                        raise ValueError("use DF(PHASE) for driving forces")
                    cls = IsolatedPhase if which == "ISOLATED" else DormantPhase
                    meta = cls(ph, wks)
                    if meta._compset is None:
                        raise ValueError(f"could not find a starting configuration for {ph}")
                    inner_p = _with_units(as_property(inner_s), units)
                    sp.prop = _Guarded(meta(inner_p), f"{which}({ph}):{_label_of(as_property(inner_s))}")
                else:
                    p1, p2 = m_t0.groups()
                    _check_phase(db, p1)
                    _check_phase(db, p2)
                    c1, c2 = find_first_compset(p1, wks), find_first_compset(p2, wks)
                    if c1 is None or c2 is None:
                        raise ValueError(f"could not find a starting configuration for {p1 if c1 is None else p2}")
                    sp.prop = _Guarded(T0(c1, c2, None), f"T0({p1},{p2})")
            else:
                sp.prop = _with_units(as_property(q), units)
                _check_model_attrs(sp.prop)
                # cheap validation of phase names inside the property
                for ph in re.findall(r"\(([A-Z][A-Z0-9_]*)(?:#\d+)?[,)]", q):
                    if ph in els or ph in ("T", "P"):
                        continue
                    if ph not in db.phases:
                        raise ValueError(f"unknown phase or element {ph!r}")
        except ValueError as e:
            raise ValueError(f"quantity {raw!r}: {e}. " + QUANTITY_HELP) from None
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"cannot parse quantity {raw!r} ({type(e).__name__}: {e}). " + QUANTITY_HELP) from None
        specs.append(sp)
    return specs


def _check_model_attrs(p):
    from pycalphad import Model
    from pycalphad.property_framework import JanssonDerivative, ModelComputedProperty
    if isinstance(p, JanssonDerivative):
        _check_model_attrs(p.numerator)
        _check_model_attrs(p.denominator)
    elif isinstance(p, ModelComputedProperty) and not (hasattr(Model, p.model_attr_name)
                                                      or p.model_attr_name in _model_instance_attrs()):
        raise ValueError(f"{p.model_attr_name!r} is neither a state variable nor a Model property")


_INSTANCE_ATTRS: Optional[set] = None


def _model_instance_attrs() -> set:
    """Model properties that only exist on instances (set in Model.__init__, e.g. VM, V0, VA, VC, VK, TC)."""
    global _INSTANCE_ATTRS
    if _INSTANCE_ATTRS is None:
        import inspect
        import re as _re
        from pycalphad import Model
        try:
            src = inspect.getsource(Model.__init__)
            _INSTANCE_ATTRS = set(_re.findall(r"self\.([A-Z][A-Z0-9_]*)\s*=", src))
        except (OSError, TypeError):
            _INSTANCE_ATTRS = set()
        _INSTANCE_ATTRS |= {"VM", "V0", "VA", "VC", "VK", "TC", "BMAG", "NT"}
    return _INSTANCE_ATTRS


def _check_phase(db, ph):
    if ph not in db.phases:
        raise ValueError(f"unknown phase {ph}; database phases: {sorted(db.phases)}")


def _pure_ref(db, el: str, phase: Optional[str], T: np.ndarray, P: np.ndarray, prop: str = "GM",
              candidates: Optional[Sequence[str]] = None):
    """Property of pure `el` in `phase` (lowest-G internal configuration) at each (T, P).  Returns (values, phase)."""
    from pycalphad import calculate
    comps = _comps(db, [el])
    ser = str(db.refstates.get(el, {}).get("phase", "")).upper()
    if phase is None and ser in _phases_for(db, [el]):
        phase = ser                                         # SER phase from the TDB ELEMENT line
    if phase is None:
        cands = [p for p in (candidates or _phases_for(db, [el])) if p in _phases_for(db, [el])]
        scored = []
        for ph in cands:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = calculate(db, comps, ph, T=298.15, P=101325, N=1, output="GM", pdens=20)
                g = float(np.nanmin(r.GM.values))
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(g):
                # ties (e.g. disordered B2 == A2) -> prefer the phase without an ordering description
                scored.append((round(g, 2), bool(db.phases[ph].model_hints.get("ordered_phase")), ph))
        if not scored:
            raise ValueError(f"no phase can hold pure {el}")
        phase = sorted(scored)[0][2]
    if phase not in _phases_for(db, [el]):
        raise ValueError(f"phase {phase} cannot hold pure {el}; candidates: {_phases_for(db, [el])}")
    T = np.asarray(T, dtype=float).ravel()
    P = np.broadcast_to(np.asarray(P, dtype=float).ravel(), T.shape) if np.size(P) > 1 else np.full(T.shape, float(np.ravel(P)[0]))
    out = np.full(T.shape, np.nan)
    for pv in np.unique(P):
        sel = P == pv
        tu, inv = np.unique(T[sel], return_inverse=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = calculate(db, comps, phase, T=tu, P=pv, N=1, output="GM", pdens=50)
            g = r.GM.values.reshape(len(tu), -1)
            idx = np.nanargmin(g, axis=1)
            if prop == "GM":
                vals = g[np.arange(len(tu)), idx]
            else:
                y = r.Y.values.reshape(len(tu), g.shape[1], -1)
                vals = np.empty(len(tu))
                for i, t in enumerate(tu):
                    rr = calculate(db, comps, phase, T=t, P=pv, N=1, output=prop, points=y[i, idx[i]][None, :])
                    vals[i] = float(np.ravel(rr[prop].values)[0])
        out[sel] = vals[inv]
    return out, phase


# ---------------------------------------------------------------------------
# Workspace evaluation
# ---------------------------------------------------------------------------
def _make_workspace(db, els, phases, conds, pdens=None):
    from pycalphad.core.workspace import Workspace
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wks = Workspace(database=db, components=_comps(db, els), phases=phases, conditions=conds,
                        calc_opts={"pdens": int(pdens)} if pdens else None)
        _ = wks.eq  # compute now
    return wks


def _assemblages(wks, shape) -> np.ndarray:
    ph = np.asarray(wks.eq.Phase)
    nv = ph.shape[-1]
    ph = ph.reshape(tuple(shape) + (nv,))
    gm = np.asarray(wks.eq.GM).reshape(shape)
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
        if not np.isfinite(gm[idx]):
            out[idx] = "(failed)"
            continue
        names = [str(p) for p in ph[idx] if str(p) and str(p) != "_FAKE_"]
        cnt = Counter(names)
        parts = []
        for n in sorted(cnt):
            parts += [n] if cnt[n] == 1 else [f"{n}#{i + 1}" for i in range(cnt[n])]
        out[idx] = "+".join(parts) if parts else "(none)"
    return out


def _merge_disordered(wks, db, phases, assem):
    """Ordered phases (order/disorder partitioned models) that are present only in their disordered state are the
    same solution as the disordered phase (e.g. L12_FCC with DOO=0 == FCC_A1).  Rename them for the summary/plots and
    return an actionable note."""
    present = set()
    for a in np.ravel(assem):
        present |= {re.sub(r"#\d+$", "", n) for n in str(a).split("+")}
    ordered = [p for p in phases if p in present and db.phases[p].model_hints.get("ordered_phase") == p]
    if not ordered:
        return assem, [], {}
    disordered = []
    for p in ordered:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = wks.get_dict(f"DOO({p})")
            mx = max(float(np.nanmax(np.abs(np.asarray(a, dtype=float)))) for a in d.values())
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(mx) and mx < 1e-3:
            disordered.append(p)
    if not disordered:
        return assem, [], {}
    lattice = ("FCC", "BCC", "HCP")
    plain = [p for p in phases if not db.phases[p].model_hints.get("ordered_phase")]
    ren = {}
    for p in disordered:
        tok = set(p.split("_")) & set(lattice)
        cands = [q for q in plain if tok and tok <= set(q.split("_"))]
        ren[p] = cands[0] if len(cands) == 1 else f"{p}(disordered)"

    def fix(a):
        names = [re.sub(r"#\d+$", "", n) for n in str(a).split("+")]
        names = [ren.get(n, n) for n in names]
        cnt = Counter(names)
        parts = []
        for n in sorted(cnt):
            parts += [n] if cnt[n] == 1 else [f"{n}#{i + 1}" for i in range(cnt[n])]
        return "+".join(parts)
    out = np.vectorize(fix, otypes=[object])(assem) if np.ndim(assem) else np.array(fix(assem), dtype=object)
    note = ("NOTE: " + ", ".join(disordered) + " only occur(s) with degree of order DOO=0 here - the same solution as "
            + ", ".join(f"{p}->{ren[p]}" for p in disordered)
            + " (renamed in the phase-region summary; their columns were merged into the equivalent phase's columns "
              f"where it exists). To avoid this, re-run with exclude_phases={disordered}.")
    return out, [note], {p: q for p, q in ren.items() if not q.endswith("(disordered)")}


def _merge_columns(columns, units, ren):
    for p, q in ren.items():
        for lab in list(columns):
            lab2 = re.sub(rf"(?<=[(,]){re.escape(p)}(?=[,)#])", q, lab)
            if lab2 == lab or lab2 not in columns:
                continue
            a, b = np.asarray(columns[lab2], dtype=float), np.asarray(columns[lab], dtype=float)
            if lab.startswith("NP("):
                columns[lab2] = np.where(np.isfinite(a) | np.isfinite(b), np.nan_to_num(a) + np.nan_to_num(b), np.nan)
            else:
                columns[lab2] = np.where(np.isfinite(a), a, b)
            del columns[lab]
            units.pop(lab, None)


def _expand_site_fractions(wks, prop):
    """Y(PH,*,SP) / Y(PH,SUBL,*) / Y(*,..): expand against the phase's actual site-fraction variables (pycalphad's own
    wildcard expansion fails when a species is absent from one sublattice)."""
    from pycalphad import variables as v
    if not (isinstance(prop, v.SiteFraction) and "*" in (str(prop.phase_name), str(prop.sublattice_index),
                                                          str(prop.species.name))):
        return [prop]
    out = []
    prf = wks.phase_record_factory
    for ph in sorted(prf.keys()):
        if prop.phase_name not in ("*", ph):
            continue
        for sf in prf[ph].variables:
            if prop.sublattice_index not in ("*", sf.sublattice_index):
                continue
            if prop.species.name not in ("*", sf.species.name):
                continue
            out.append(v.Y(ph, sf.sublattice_index, sf.species))
    if not out:
        raise ValueError(f"no site fractions match {prop}")
    return out


def _order_composition_sets(wks, els, labels, columns, shape) -> str:
    """Make miscibility-gap composition sets consistent along the grid.

    pycalphad numbers composition sets (PH#1, PH#2) in whatever order the solver found them, so the
    Fe-rich and Cr-rich branches of a gap can swap labels between neighbouring points.  Renumber them at
    every point by increasing mole fraction of a reference element (#1 = leanest), applied to every
    column that refers to those sets.
    """
    import re as _re
    multi: Dict[str, List[int]] = {}
    for lab in columns:
        for ph, n in _re.findall(r"([A-Z0-9_]+)#(\d+)", lab):
            multi.setdefault(ph, [])
            if int(n) not in multi[ph]:
                multi[ph].append(int(n))
    multi = {ph: sorted(ns) for ph, ns in multi.items() if len(ns) > 1}
    if not multi or not shape:
        return ""
    x_conds = [lab[2:-1] for lab in labels if lab.startswith("X(")]
    ref = x_conds[0] if x_conds else sorted(els)[0]
    done = []
    for ph, ns in multi.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xd = wks.get_dict(f"X({ph},{ref})")
        except Exception:  # noqa: BLE001 - ordering is cosmetic; never fail the calculation
            continue
        xs = {}
        for key, arr in xd.items():
            m = _re.search(rf"{ph}#(\d+)", _label_of(key))
            if m:
                xs[int(m.group(1))] = np.asarray(arr, dtype=float).reshape(shape)
        if sorted(xs) != ns:
            continue
        stack = np.stack([np.where(np.isfinite(xs[n]), xs[n], np.inf) for n in ns])   # absent sets sort last
        order = np.argsort(stack, axis=0, kind="stable")          # order[k] = which set becomes #(k+1)
        if np.all(order == np.arange(len(ns)).reshape((-1,) + (1,) * len(shape))):
            continue
        groups: Dict[str, Dict[int, str]] = {}                   # template -> {set number: column label}
        for lab in list(columns):
            m = _re.search(rf"{ph}#(\d+)", lab)
            if m and int(m.group(1)) in ns:
                groups.setdefault(lab.replace(m.group(0), f"{ph}#{{n}}"), {})[int(m.group(1))] = lab
        for tmpl, members in groups.items():
            if sorted(members) != ns:
                continue
            src = np.stack([columns[members[n]] for n in ns])
            new = np.take_along_axis(src, order, axis=0)
            for k, n in enumerate(ns):
                columns[members[n]] = new[k]
        done.append(ph)
    return (f"composition sets of {', '.join(done)} ordered by X({ref}) (#1 = lowest)" if done else "")


def evaluate_quantities(db, els, phases, conditions, quantities, unit="K", pdens=None):
    """Core of equilibrium_property_plot.  Returns a dict with grid, columns, units, assemblages, ..."""
    conds, labels, varying, values = parse_conditions(conditions, els, unit)
    wnote = _w_to_x(db, els, conds, labels, values)
    shape = tuple(len(values[lab]) for lab in varying)
    npts = int(np.prod(shape)) if shape else 1
    if npts > MAX_POINTS:
        raise ValueError(f"{npts} equilibrium points requested (limit {MAX_POINTS}); use coarser steps")
    t0 = time.time()
    state = {}

    def wks_factory():
        if "wks" not in state:
            state["wks"] = _make_workspace(db, els, phases, conds, pdens)
        return state["wks"]

    specs = parse_quantities(quantities, els, db, wks_factory)
    if any(s.slow for s in specs) and npts > MAX_POINTS_SLOW:
        raise ValueError(f"{npts} points requested; with DF/ISOLATED/DORMANT/T0 quantities the limit is "
                         f"{MAX_POINTS_SLOW}. Use coarser steps.")
    wks = wks_factory()
    # axis order of Workspace results = order of eq coords (sorted keys); map to our labels
    str2lab = {str(p): lab for lab, p in labels.items()}
    axis_labels = [str2lab.get(str(k), str(k)) for k in wks.condition_axis_order]
    shape = tuple(len(values[lab]) for lab in axis_labels)
    gm = np.asarray(wks.eq.GM).reshape(shape)
    converged = np.isfinite(gm)
    grids = {}
    mesh = np.meshgrid(*[values[lab] for lab in axis_labels], indexing="ij") if axis_labels else []
    for lab in values:
        if lab in axis_labels:
            grids[lab] = mesh[axis_labels.index(lab)]
        else:
            grids[lab] = np.full(shape, values[lab][0])

    columns, col_units, notes = OrderedDict(), OrderedDict(), ([wnote] if wnote else [])
    for sp in specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = wks.get_dict(*_expand_site_fractions(wks, sp.prop))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"computing {sp.text} failed ({type(e).__name__}: {e}). " + QUANTITY_HELP) from None
        if isinstance(sp.prop, _Guarded) and sp.prop.failures:
            notes.append(f"{sp.prop.label}: solver failed at {sp.prop.failures} point(s) (NaN)")
        for key, arr in res.items():
            arr = np.asarray(arr, dtype=float).reshape(shape) if shape else np.asarray(arr, dtype=float).reshape(())
            lab = _label_of(key)
            u = getattr(key, "display_units", "")
            if sp.units and sp.wildcard and str(u) != sp.units:
                from pycalphad.property_framework.units import Q_
                try:
                    arr = Q_(arr, u).to(sp.units).magnitude
                    u = sp.units
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"cannot convert {lab} to {sp.units} for a wildcard quantity ({e}); "
                                     "list the phases explicitly instead of '*'") from None
            if sp.kind in ("acr", "mur"):
                gref, refph = _pure_ref(db, sp.el, sp.ref_phase, grids["T"], grids["P"], "GM", candidates=None)
                gref = gref.reshape(arr.shape)
                if sp.kind == "mur":
                    lab, u = f"MUR({sp.el},{refph})", "J/mol"
                    arr = arr - gref
                else:
                    lab, u = f"ACR({sp.el},{refph})", ""
                    arr = np.exp((arr - gref) / (R_GAS * grids["T"].reshape(arr.shape)))
            if lab.startswith("NP("):
                arr = np.where(converged & ~np.isfinite(arr), 0.0, arr)
            if sp.wildcard and (np.all(~np.isfinite(arr)) or (lab.startswith("NP(") and np.nanmax(np.abs(arr)) < 1e-12)):
                continue
            if lab.startswith(("T0(",)) or lab == "T":
                u = "°C" if _is_celsius(unit) else "K"
                arr = arr - KELVIN if _is_celsius(unit) else arr
            if lab in columns:
                continue
            columns[lab] = arr
            col_units[lab] = _pretty_units(u)
    if not columns:
        raise ValueError("all requested quantities are NaN/absent over the whole grid (e.g. a phase that never "
                         "forms). Check phase names or use DF(PHASE)/ISOLATED(PHASE):PROP for metastable phases.")
    assem_raw = _assemblages(wks, shape) if shape else _assemblages(wks, (1,)).reshape(())
    xs = [g for lab, g in grids.items() if lab.startswith("X(")]
    if xs and shape:
        bad = np.sum(xs, axis=0) > 1 + 1e-9
        if bad.any():
            assem_raw = assem_raw.copy()
            assem_raw[bad] = "(X sum > 1)"
            notes.append(f"{int(bad.sum())} grid point(s) have mole fractions summing to > 1 (outside the "
                         "composition triangle) - left as NaN")
    assem, dis_notes, ren = _merge_disordered(wks, db, phases, assem_raw)
    _merge_columns(columns, col_units, ren)
    notes = dis_notes + notes
    order_note = _order_composition_sets(wks, els, labels, columns, shape)
    if order_note:
        notes.append(order_note)
    return {"wks": wks, "axis_labels": axis_labels, "values": values, "grids": grids, "shape": shape,
            "columns": columns, "units": col_units, "assemblages": assem, "assemblages_raw": assem_raw,
            "converged": converged,
            "notes": notes, "seconds": round(time.time() - t0, 2), "specs": specs}


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def _family(lab: str, units: str) -> Tuple[str, str]:
    """(group key, y-axis label) for 1-D plots."""
    if lab.startswith("NP("):
        return "NP", "Phase fraction"
    m = re.fullmatch(r"(X|W)\(([^,()]+),([^,()]+)\)", lab)
    if m:
        return f"{m.group(1)}ph_{m.group(3)}", f"{'Mole' if m.group(1) == 'X' else 'Mass'} fraction {m.group(3).capitalize()} in phase"
    if re.fullmatch(r"(X|W)\(([^,()]+)\)", lab):
        return "Xsys", "Overall composition"
    if lab.startswith("Y("):
        return "Y", "Site fraction"
    if lab.startswith(("MU(", "MUR(")):
        return "MU", f"Chemical potential ({units or 'J/mol'})"
    if lab.startswith("ACR("):
        return "ACR", "Activity"
    if lab.startswith("DF("):
        return "DF", f"Driving force ({units or 'J/mol'})"
    if lab.startswith("DOO"):
        return "DOO", "Degree of order"
    if lab.startswith("T0(") or lab == "T":
        return "T", f"Temperature ({units})"
    return f"u:{units}", f"{units}" if units else "value"


def _short_label(lab: str, fam: str) -> str:
    if fam == "NP":
        return lab[3:-1]
    m = re.fullmatch(r"(X|W)\(([^,()]+),([^,()]+)\)", lab)
    if m and fam.endswith("ph_" + m.group(3)):
        return m.group(2)
    if fam == "DF":
        return lab[3:-1]
    if fam == "DOO" and lab.startswith("DOO("):
        return lab[4:-1]
    return lab


def _plot_1d(res, x_lab, xvals, xlabel, out_png, title, unit):
    cols, units = res["columns"], res["units"]
    groups = OrderedDict()
    for lab in cols:
        if lab == x_lab:
            continue
        key, ylab = _family(lab, units[lab])
        groups.setdefault(key, (ylab, []))[1].append(lab)
    keys = list(groups)[:8]
    n = len(keys)
    if n == 0:
        raise ValueError("nothing to plot besides the x-axis quantity")
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.3 + 2.7 * n), sharex=True, squeeze=False)
    assem = res["assemblages"]
    for ax, key in zip(axes[:, 0], keys):
        ylab, labs = groups[key]
        many = len(labs) > 10
        cmap = plt.get_cmap("tab20" if len(labs) > 10 else "tab10")
        for i, lab in enumerate(labs):
            y = np.asarray(cols[lab], dtype=float)
            ax.plot(xvals, y, lw=1.8, color=cmap(i % cmap.N), label=_short_label(lab, key),
                    marker="o" if len(xvals) < 12 else None, ms=3)
        if key == "NP":
            ax.set_ylim(-0.02, 1.02)
        if key == "ACR":
            ax.set_ylim(bottom=0)
        if key == "DF":
            ax.axhline(0, color="k", lw=0.8, ls="--")
        if key.startswith("u:") and len(labs) == 1:
            ylab = f"{labs[0]}" + (f" ({units[labs[0]]})" if units[labs[0]] else "")
        elif key.startswith("u:"):
            ylab = f"{', '.join(labs[:3])}{'…' if len(labs) > 3 else ''}" + (f" ({units[labs[0]]})" if units[labs[0]] else "")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2 if many else 1, loc="best", framealpha=0.8)
    # shade phase-assemblage changes along the x axis (only when x is a condition)
    if x_lab in res["axis_labels"] and assem.ndim == 1:
        ch = [i for i in range(1, len(assem)) if assem[i] != assem[i - 1]]
        for ax in axes[:, 0]:
            for i in ch:
                ax.axvline(0.5 * (xvals[i] + xvals[i - 1]), color="0.6", lw=0.6, ls=":")
    axes[-1, 0].set_xlabel(xlabel)
    axes[0, 0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _region_map(ax, xv, yv, assem2d, label_regions=True):
    """Draw stable-phase regions (categorical) + boundaries; assem2d shape (ny, nx)."""
    names, ids = _assem_ids(assem2d)
    ids = np.ma.masked_invalid(ids)
    cmap = plt.get_cmap("tab20", max(len(names), 2))
    ax.pcolormesh(xv, yv, ids, cmap=cmap, shading="auto", vmin=-0.5, vmax=len(names) - 0.5, alpha=0.55)
    _boundaries(ax, xv, yv, ids, len(names))
    if label_regions:
        for i, nm in enumerate(names):
            jj, ii = np.nonzero(ids == i)
            if len(ii) >= 2:
                k = np.argsort(np.hypot(ii - ii.mean(), jj - jj.mean()))[0]
                ax.text(xv[ii[k]], yv[jj[k]], nm.replace("+", "+\n") if len(nm) > 18 else nm, fontsize=6.5,
                        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    return names


def _assem_ids(assem):
    """Integer id per assemblage; failed / invalid points ('(...)') -> NaN."""
    names = sorted({a for a in np.ravel(assem) if not str(a).startswith("(")})
    ids = np.array([names.index(a) if a in names else np.nan for a in np.ravel(assem)], dtype=float)
    return names, ids.reshape(np.shape(assem))


def _boundaries(ax, xv, yv, ids, n):
    ids = np.ma.filled(ids, np.nan) if np.ma.isMaskedArray(ids) else ids
    if len(xv) < 2 or len(yv) < 2:
        return
    for i in range(n):
        m = (ids == i).astype(float)
        if 0 < m.sum() < m.size:
            ax.contour(xv, yv, m, levels=[0.5], colors="k", linewidths=0.6)


def _plot_2d(res, x_lab, y_lab, out_png, title, unit, boundaries=True):
    cols, units = res["columns"], res["units"]
    ax_labels = res["axis_labels"]
    xv = _cond_display(x_lab, res["values"][x_lab], unit)
    yv = _cond_display(y_lab, res["values"][y_lab], unit)
    transpose = ax_labels.index(x_lab) == 0          # arrays are (a, b); want (ny, nx)

    def orient(a):
        a = np.asarray(a)
        return a.T if transpose else a

    labs = list(cols)[:MAX_PANELS - 1]
    assem = orient(res["assemblages"])
    names, ids = _assem_ids(assem)
    n = len(labs) + 1
    ncol = min(3, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)
    axs = axes.ravel()
    _region_map(axs[0], xv, yv, assem)
    axs[0].set_title("Stable phases", fontsize=9)
    for ax, lab in zip(axs[1:], labs):
        z = orient(cols[lab]).astype(float)
        zm = np.ma.masked_invalid(z)
        if lab.startswith("NP("):
            vmin, vmax = 0.0, 1.0
        else:
            fin = z[np.isfinite(z)]
            vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-9
        cmap = "RdBu_r" if lab.startswith("DF(") and vmin < 0 < vmax else "viridis"
        if cmap == "RdBu_r":
            lim = max(abs(vmin), abs(vmax))
            vmin, vmax = -lim, lim
        ax.set_facecolor("0.85")
        pm = ax.pcolormesh(xv, yv, zm, cmap=cmap, shading="gouraud" if min(z.shape) > 1 else "auto",
                           vmin=vmin, vmax=vmax)
        if min(z.shape) > 2 and np.isfinite(z).sum() > 4:
            try:
                ax.contour(xv, yv, zm, levels=8, colors="w", linewidths=0.4, alpha=0.7)
            except Exception:  # noqa: BLE001
                pass
        if boundaries:
            _boundaries(ax, xv, yv, ids, len(names))
        cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.03)
        if units[lab]:
            cb.set_label(units[lab], fontsize=8)
        ax.set_title(lab, fontsize=9)
    for ax in axs[:n]:
        ax.set_xlabel(_cond_axis_label(x_lab, unit), fontsize=8)
        ax.set_ylabel(_cond_axis_label(y_lab, unit), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(cols) - len(labs)


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def _runs(seq):
    out, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            out.append((start, i - 1, seq[start]))
            start = i
    return out


def _summary_1d(res, x_lab, xvals, xunit):
    lines = []
    assem = res["assemblages"]
    if x_lab in res["axis_labels"]:
        lines.append(f"Stable phases along {x_lab}:")
        for a, b, nm in _runs(list(assem)):
            lines.append(f"  {_fmt(xvals[a])}–{_fmt(xvals[b])} {xunit}: {nm}")
    lines.append("Quantities:")
    for lab, arr in res["columns"].items():
        u = res["units"][lab]
        y = np.asarray(arr, dtype=float)
        fin = np.isfinite(y)
        head = f"  {lab}{(' [' + u + ']') if u else ''}: "
        if not fin.any():
            lines.append(head + "all NaN")
            continue
        imin, imax = np.nanargmin(y), np.nanargmax(y)
        s = head + f"min {_fmt(y[imin])} at {_fmt(xvals[imin])}, max {_fmt(y[imax])} at {_fmt(xvals[imax])}"
        if lab.startswith("NP("):
            pres = np.nonzero(fin & (y > 1e-6))[0]
            s = head + (f"present {_fmt(xvals[pres[0]])}–{_fmt(xvals[pres[-1]])} {xunit}, max {_fmt(y[imax])} at "
                        f"{_fmt(xvals[imax])}" if len(pres) else "never present")
        elif lab.startswith("DF("):
            pos = np.nonzero(fin & (y > -1e-6))[0]
            s += (f"; DF>=0 (stable / would precipitate) for {x_lab} in {_fmt(xvals[pos[0]])}–{_fmt(xvals[pos[-1]])} "
                  f"{xunit}" if len(pos) else "; DF<0 everywhere (phase not favoured)")
        nn = int((~fin).sum())
        if nn and not lab.startswith(("X(", "Y(", "DOO", "W(", "GM(", "HM(", "SM(", "CPM(")):
            s += f"; {nn} NaN point(s)"
        lines.append(s)
    return lines


def _summary_2d(res, unit):
    lines = []
    a_lab, b_lab = res["axis_labels"]
    va = _cond_display(a_lab, res["values"][a_lab], unit)
    vb = _cond_display(b_lab, res["values"][b_lab], unit)
    assem = res["assemblages"]
    lines.append("Stable phase regions (grid points; ranges of each axis where they occur):")
    for nm, cnt in Counter(assem.ravel()).most_common(25):
        ii, jj = np.nonzero(assem == nm)
        lines.append(f"  {nm}: {cnt} pts; {a_lab} {_fmt(va[ii.min()])}–{_fmt(va[ii.max()])}, "
                     f"{b_lab} {_fmt(vb[jj.min()])}–{_fmt(vb[jj.max()])}")
    lines.append("Quantities (min / max with location):")
    for lab, arr in res["columns"].items():
        y = np.asarray(arr, dtype=float)
        if not np.isfinite(y).any():
            lines.append(f"  {lab}: all NaN")
            continue
        i1 = np.unravel_index(np.nanargmin(y), y.shape)
        i2 = np.unravel_index(np.nanargmax(y), y.shape)
        u = res["units"][lab]
        lines.append(f"  {lab}{(' [' + u + ']') if u else ''}: min {_fmt(y[i1])} at ({a_lab}={_fmt(va[i1[0]])}, "
                     f"{b_lab}={_fmt(vb[i1[1]])}), max {_fmt(y[i2])} at ({a_lab}={_fmt(va[i2[0]])}, "
                     f"{b_lab}={_fmt(vb[i2[1]])})")
    return lines


def _to_frame(res, unit):
    import pandas as pd
    shape = res["shape"]
    data = OrderedDict()
    for lab in res["axis_labels"]:
        g = res["grids"][lab].ravel()
        if lab == "T":
            data["T [K]"] = g
            if _is_celsius(unit):
                data["T [°C]"] = g - KELVIN
        else:
            data[lab + (" [Pa]" if lab == "P" else "")] = g
    for lab, arr in res["columns"].items():
        u = res["units"][lab]
        data[f"{lab} [{u}]" if u else lab] = np.asarray(arr, dtype=float).ravel() if shape else [float(arr)]
    data["stable_phases"] = np.asarray(res["assemblages"], dtype=object).ravel() if shape else [str(res["assemblages"])]
    return pd.DataFrame(data)


def _write_netcdf(res, path, unit, attrs):
    import xarray as xr
    coords = OrderedDict()
    dims = []
    for lab in res["axis_labels"]:
        nm = _safe_name(lab)
        dims.append(nm)
        coords[nm] = (nm, res["values"][lab], {"long_name": lab, "units": {"T": "K", "P": "Pa"}.get(lab, "")})
    dv = OrderedDict()
    used = set()
    for lab, arr in res["columns"].items():
        nm = _safe_name(lab.replace("(", "_").replace(")", "").replace(",", "_"))
        while nm in used or nm in coords:
            nm += "_"
        used.add(nm)
        dv[nm] = (dims, np.asarray(arr, dtype=float), {"long_name": lab, "units": res["units"][lab]})
    dv["stable_phases"] = (dims, np.asarray(res["assemblages"], dtype=str), {"long_name": "stable phase assemblage"})
    ds = xr.Dataset(dv, coords=coords, attrs={k: str(v) for k, v in attrs.items()})
    ds.to_netcdf(path, engine="scipy")
    return path


# ---------------------------------------------------------------------------
# tool 1: equilibrium properties
# ---------------------------------------------------------------------------
@pcache.cached
def equilibrium_property_plot(tdb: str, elements: Sequence[str], conditions: Dict, quantities: Sequence[str],
                              out_base: str, phases=None, exclude_phases=None, temperature_unit: str = "K",
                              x_axis: Optional[str] = None, plot: bool = True, write_netcdf: bool = False,
                              title: Optional[str] = None, pdens: Optional[int] = None) -> Dict:
    db = _get_db(tdb)
    els = _check_elements(db, elements)
    ph = _resolve_phases(db, els, phases, exclude_phases)
    res = evaluate_quantities(db, els, ph, conditions, quantities, unit=temperature_unit, pdens=pdens)
    unit = temperature_unit
    out = {"csv": out_base + ".csv", "png": None, "nc": None}
    df = _to_frame(res, unit)
    df.to_csv(out["csv"], index=False, float_format="%.8g")
    sysname = "-".join(e.capitalize() for e in els)
    fixed = ", ".join(f"{lab}={_fmt(_cond_display(lab, arr, unit)[0])}{' °C' if lab == 'T' and _is_celsius(unit) else (' K' if lab == 'T' else '')}"
                      for lab, arr in res["values"].items() if len(arr) == 1 and lab not in ("N", "P"))
    ttl = title or f"{sysname}" + (f"   {fixed}" if fixed else "")
    lines = [f"System {sysname}; phases considered: {', '.join(ph)}",
             f"Conditions: {_describe_conditions(res['values'], unit)}  ({res['seconds']} s)"]
    nfail = int((np.asarray(res["assemblages"]) == "(failed)").sum())
    if nfail:
        lines.append(f"WARNING: equilibrium did not converge at {nfail} point(s) (NaN in the CSV).")
    xunit = ("°C" if _is_celsius(unit) else "K")
    nd = len(res["axis_labels"])
    if nd == 1:
        clab = res["axis_labels"][0]
        x_lab = clab
        if x_axis:
            xa = _normalize_q(_split_units(x_axis)[0])
            xa = {"TEMPERATURE": "T"}.get(xa, xa)
            if xa in res["values"] and xa in res["axis_labels"]:
                x_lab = xa
            elif xa in res["columns"]:
                x_lab = xa
            else:
                raise ValueError(f"x_axis {x_axis!r} must be the varying condition {clab} or one of the computed "
                                 f"quantities {list(res['columns'])}")
        if x_lab == clab:
            xvals = _cond_display(clab, res["values"][clab], unit)
            xlabel = _cond_axis_label(clab, unit)
            xu = xunit if clab == "T" else ""
        else:
            xvals = np.asarray(res["columns"][x_lab], dtype=float)
            xu = res["units"][x_lab]
            xlabel = x_lab + (f" ({xu})" if xu else "")
        lines += _summary_1d(res, clab, _cond_display(clab, res["values"][clab], unit), xunit if clab == "T" else "")
        if plot:
            out["png"] = out_base + ".png"
            _plot_1d(res, x_lab, xvals, xlabel, out["png"], ttl, unit)
    elif nd == 2:
        a, b = res["axis_labels"]
        if x_axis:
            xa = _normalize_q(_split_units(x_axis)[0])
            xa = {"TEMPERATURE": "T"}.get(xa, xa)
            if xa not in (a, b):
                raise ValueError(f"for a 2-D grid x_axis must be one of the varying conditions {a}, {b}")
            x_lab = xa
        else:
            x_lab = a if not a.startswith(("T", "P")) or b in ("T", "P") else b
            if a in ("T", "P") and b not in ("T", "P"):
                x_lab = b
        y_lab = b if x_lab == a else a
        lines += _summary_2d(res, unit)
        if plot:
            out["png"] = out_base + ".png"
            skipped = _plot_2d(res, x_lab, y_lab, out["png"], ttl, unit)
            if skipped:
                lines.append(f"(plot shows the first {MAX_PANELS - 1} quantities; {skipped} more are only in the CSV)")
    else:
        lines.append("Single point (no condition varies):")
        for lab, arr in res["columns"].items():
            u = res["units"][lab]
            lines.append(f"  {lab} = {_fmt(float(arr))}{(' ' + u) if u else ''}")
        lines.append(f"  stable phases: {res['assemblages']}")
    lines += res["notes"]
    if write_netcdf:
        out["nc"] = _write_netcdf(res, out_base + ".nc", unit, {"tdb": Path(tdb).name, "elements": " ".join(els),
                                                                 "phases": " ".join(ph),
                                                                 "created_by": "calphad_tools equilibrium_property_plot"})
    out["summary"] = "\n".join(lines)
    out["columns"] = list(df.columns)
    out["n_points"] = len(df)
    return out


# ---------------------------------------------------------------------------
# tool 2: single-phase properties (no equilibrium)
# ---------------------------------------------------------------------------
_PHASE_PROPS = ("GM", "HM", "SM", "CPM", "G", "H", "GM_MIX", "HM_MIX", "SM_MIX", "CPM_MIX", "DOO", "TC", "BMAG",
                "NT")
_REFERENCABLE = ("GM", "HM", "SM", "CPM")


def _prop_units(prop: str) -> Tuple[str, float]:
    """(axis unit label, scale) for plotting."""
    if prop in ("GM", "HM", "G", "H", "GM_MIX", "HM_MIX"):
        return "kJ/mol", 1e-3
    if prop in ("SM", "CPM", "SM_MIX", "CPM_MIX"):
        return "J/(mol·K)", 1.0
    if prop in ("TC", "NT"):
        return "K", 1.0
    return "", 1.0


def _lower_hull(x, g):
    """Indices of the lower convex hull of points (x, g), sorted by x."""
    o = np.lexsort((g, x))
    pts = np.column_stack([x[o], g[o]])
    hull = []
    for i, p in enumerate(pts):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = pts[hull[-2]], pts[hull[-1]]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return o[hull]


@pcache.cached
def phase_property_plot(tdb: str, elements: Sequence[str], prop: str, out_base: str, temperature=None,
                        composition: Optional[Dict[str, float]] = None, phases=None, exclude_phases=None,
                        temperature_unit: str = "K", x_element: Optional[str] = None,
                        reference: Optional[Dict[str, str]] = None, show_all_points: bool = False,
                        common_tangent: bool = False, pressure: float = 101325.0, points: int = 2000,
                        title: Optional[str] = None) -> Dict:
    from pycalphad import calculate
    db = _get_db(tdb)
    els = _check_elements(db, elements)
    prop = prop.strip().upper()
    prop = {"G_MIX": "GM_MIX", "H_MIX": "HM_MIX", "S_MIX": "SM_MIX", "CP": "CPM", "CP_MIX": "CPM_MIX"}.get(prop, prop)
    if prop not in _PHASE_PROPS:
        raise ValueError(f"property must be one of {list(_PHASE_PROPS)} (per mole of atoms; _MIX = relative to the "
                         "phase's own endmembers)")
    ph_list = _resolve_phases(db, els, phases, exclude_phases)
    comps = _comps(db, els)
    ref = {k.upper(): v.upper() for k, v in (reference or {}).items()}
    if ref:
        if prop not in _REFERENCABLE:
            raise ValueError(f"reference states apply to {list(_REFERENCABLE)} only (got {prop})")
        missing = [e for e in els if e not in ref]
        if missing:
            raise ValueError(f"reference needs a phase for every element; missing {missing} "
                             f"(e.g. {{\"{els[0]}\": \"BCC_A2\"}})")
        for e, p in ref.items():
            if e not in els:
                raise ValueError(f"reference element {e} not in {els}")
            if p not in _phases_for(db, [e]):
                raise ValueError(f"reference phase {p} cannot hold pure {e}; candidates {_phases_for(db, [e])}")
    unit_lab, scale = _prop_units(prop)
    raw_u = {"kJ/mol": "J/mol", "J/(mol·K)": "J/mol/K"}.get(unit_lab, unit_lab)
    vcol = f"{prop} [{raw_u}]" if raw_u else prop
    celsius = _is_celsius(temperature_unit)
    t0 = time.time()
    lines = []
    out = {"csv": out_base + ".csv", "png": out_base + ".png"}
    sysname = "-".join(e.capitalize() for e in els)
    ref_txt = (" relative to " + ", ".join(f"{e.capitalize()}({p})" for e, p in ref.items())) if ref else ""
    tvals = _values(temperature, "temperature") if temperature is not None else None
    if tvals is None:
        raise ValueError("give temperature (a number for curves vs composition, or [start, stop, step] for "
                         "curves vs temperature at a fixed composition)")
    if celsius:
        tvals = tvals + KELVIN
    import pandas as pd
    if len(tvals) == 1:
        # ------------------------------------------------ vs composition (binary)
        if len(els) != 2:
            raise ValueError("property-vs-composition curves need exactly 2 elements (for more elements give a "
                             "temperature range and a fixed composition, or use equilibrium_property_plot with "
                             "ISOLATED(PHASE):PROP)")
        T = float(tvals[0])
        xel = (x_element or sorted(els)[1]).upper()
        if xel not in els:
            raise ValueError(f"x_element must be one of {els}")
        other = [e for e in els if e != xel][0]
        refvals = {}
        if ref:
            for e in els:
                refvals[e] = float(_pure_ref(db, e, ref[e], np.array([T]), np.array([pressure]), prop)[0][0])
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        cmap = plt.get_cmap("tab10" if len(ph_list) <= 10 else "tab20")
        rows, env_all = [], []
        bins = np.linspace(0, 1, 201)
        skipped = []
        for i, name in enumerate(ph_list):
            col = cmap(i % cmap.N)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rg = calculate(db, comps, name, T=T, P=pressure, N=1, output="GM", pdens=points)
                    x = rg.X.sel(component=xel).values.ravel()
                    g = rg.GM.values.ravel()
                    if prop == "GM":
                        val = g.copy()
                    else:
                        y = rg.Y.values.reshape(len(g), -1)
                        rp = calculate(db, comps, name, T=T, P=pressure, N=1, output=prop, points=y)
                        val = rp[prop].values.ravel()
                        if len(val) != len(g):
                            raise RuntimeError("point mismatch")
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{name} ({type(e).__name__})")
                continue
            ok = np.isfinite(x) & np.isfinite(g)
            x, g, val = x[ok], g[ok], val[ok]
            if ref:
                val = val - (x * refvals[xel] + (1 - x) * refvals[other])
            if show_all_points:
                ax.scatter(x, val * scale, s=2, color=col, alpha=0.25, edgecolors="none")
            idx = np.digitize(x, bins)
            ex, ev, eg = [], [], []
            for k in np.unique(idx):
                m = np.nonzero(idx == k)[0]
                j = m[np.argmin(g[m])]
                ex.append(x[j]), ev.append(val[j]), eg.append(g[j])
            o = np.argsort(ex)
            ex, ev, eg = np.array(ex)[o], np.array(ev)[o], np.array(eg)[o]
            if np.all(~np.isfinite(ev)):
                skipped.append(f"{name} (NaN - e.g. _MIX of a partitioned order/disorder model; use GM with a "
                               "reference instead)")
                continue
            if np.ptp(ex) < 0.011:
                ax.plot(ex, ev * scale, "o", color=col, ms=6, label=name)
            else:
                ax.plot(ex, ev * scale, "-", color=col, lw=1.8, label=name)
            for a, b, c in zip(ex, ev, eg):
                rows.append({"phase": name, f"X({xel})": a, vcol: b, **({} if prop == "GM" else {"GM [J/mol]": c})})
                env_all.append((a, b, name))
        if not rows:
            raise ValueError(f"no phase could be calculated: {skipped}")
        tangent_lines = []
        if common_tangent:
            if prop != "GM":
                raise ValueError("common_tangent needs prop='GM' (a common reference is required)")
            xa = np.array([r[0] for r in env_all])
            ga = np.array([r[1] for r in env_all])
            pa = [r[2] for r in env_all]
            hidx = _lower_hull(xa, ga)
            # walk the hull: long segments are tie-lines; everything between them is a single-phase field
            regions, cur = [], [hidx[0]]
            for i1, i2 in zip(hidx[:-1], hidx[1:]):
                if xa[i2] - xa[i1] > 0.012:
                    regions.append(cur)
                    cur = [i2]
                    ax.plot([xa[i1], xa[i2]], [ga[i1] * scale, ga[i2] * scale], "k--", lw=1.1, zorder=5)
                    ax.plot([xa[i1], xa[i2]], [ga[i1] * scale, ga[i2] * scale], "ko", ms=3.5, zorder=6)
                    tangent_lines.append(f"  two-phase {pa[i1]} + {pa[i2]}: X({xel}) {xa[i1]:.4f} – {xa[i2]:.4f} "
                                         f"(tie-line)")
                else:
                    cur.append(i2)
            regions.append(cur)
            lines.append(f"Common-tangent construction (lower convex hull of all phases) at {T:.2f} K:")
            for reg in regions:
                cnt = Counter(pa[i] for i in reg)
                # ignore sampling noise: phases touching the hull at <10 % of a field's vertices
                names = [n for n, c in cnt.most_common() if c >= 0.1 * len(reg)] or [cnt.most_common(1)[0][0]]
                nm = names[0] if len(names) == 1 else "/".join(names) + " (equal energies - same solution)"
                lines.append(f"  single-phase {nm}: X({xel}) {xa[reg[0]]:.4f} – {xa[reg[-1]]:.4f}")
            lines += tangent_lines or ["  no two-phase regions (one phase is stable at all compositions)"]
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Mole fraction {xel.capitalize()}")
        ax.set_ylabel(f"{prop}{ref_txt and ' (ref.)'}" + (f" ({unit_lab})" if unit_lab else ""))
        tt = f"{T - KELVIN:.0f} °C" if celsius else f"{T:.0f} K"
        ax.set_title(title or f"{sysname}: {prop} of each phase at {tt}{ref_txt}", fontsize=10)
        ax.axhline(0, color="0.5", lw=0.6) if (ref or prop.endswith("_MIX")) else None
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2 if len(ph_list) > 6 else 1)
        fig.savefig(out["png"], dpi=150, bbox_inches="tight")
        plt.close(fig)
        df = pd.DataFrame(rows)
        df.insert(0, "T [K]", T)
        df.to_csv(out["csv"], index=False, float_format="%.8g")
        lines.insert(0, f"{prop} vs X({xel}) at {T:.2f} K for {len(set(df.phase))} phases{ref_txt} "
                        f"(lowest-G internal configuration per composition bin; {round(time.time() - t0, 1)} s)")
        for name, grp in df.groupby("phase", sort=False):
            j = grp[vcol].idxmin()
            lines.append(f"  {name}: X({xel}) {grp[f'X({xel})'].min():.3f}–{grp[f'X({xel})'].max():.3f}; "
                         f"min {prop} {_fmt(grp.loc[j, vcol])} at X={grp.loc[j, f'X({xel})']:.3f}")
    else:
        # ------------------------------------------------ vs temperature (fixed composition)
        from pycalphad import variables as v
        from pycalphad.property_framework import IsolatedPhase
        comp = {k.upper().replace("X(", "").replace(")", ""): float(x) for k, x in (composition or {}).items()}
        if len(els) == 1:
            comp = {}
        elif len(comp) != len(els) - 1 or any(k not in els for k in comp):
            raise ValueError(f"composition must give mole fractions of all elements but one (balance) of {els}, "
                             f"e.g. {{\"{els[0]}\": 0.3}}")
        if sum(comp.values()) >= 1:
            raise ValueError("mole fractions in composition sum to >= 1")
        if len(tvals) > 2000:
            raise ValueError("too many temperatures (max 2000)")
        conds = {v.T: tvals, v.P: pressure, v.N: 1, **{v.X(k): x for k, x in comp.items()}}
        xfull = dict(comp)
        bal = [e for e in els if e not in comp][0]
        xfull[bal] = 1 - sum(comp.values())
        shift = np.zeros_like(tvals)
        if ref:
            for e in els:
                shift = shift + xfull[e] * _pure_ref(db, e, ref[e], tvals, np.array([pressure]), prop)[0]
        tdisp = tvals - KELVIN if celsius else tvals
        data = OrderedDict([("T [K]", tvals)])
        if celsius:
            data["T [°C]"] = tdisp
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        cmap = plt.get_cmap("tab10" if len(ph_list) <= 10 else "tab20")
        skipped = []
        for i, name in enumerate(ph_list):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = _make_workspace(db, els, [name], conds)
                    ip = IsolatedPhase(name, w)
                    if ip._compset is None:
                        raise RuntimeError("phase cannot reach this composition")
                    arr = np.asarray(w.get(_Guarded(ip(f"{prop}({name})"), name)), dtype=float).ravel()
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{name} ({type(e).__name__}: {str(e)[:60]})")
                continue
            arr = arr - shift
            if not np.isfinite(arr).any():
                skipped.append(f"{name} (cannot reach this composition / NaN)")
                continue
            data[f"{prop}({name})"] = arr
            ax.plot(tdisp, arr * scale, lw=1.8, color=cmap(i % cmap.N), label=name)
        if len(data) <= (2 if celsius else 1):
            raise ValueError(f"no phase could be calculated at this composition: {skipped}")
        ax.set_xlabel("Temperature (°C)" if celsius else "Temperature (K)")
        ax.set_ylabel(f"{prop}{ref_txt and ' (ref.)'}" + (f" ({unit_lab})" if unit_lab else ""))
        cs = ", ".join(f"X({k})={x:g}" for k, x in comp.items())
        ax.set_title(title or f"{sysname} {cs}: {prop} of each phase (single-phase, internally relaxed){ref_txt}",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2 if len(ph_list) > 6 else 1)
        fig.savefig(out["png"], dpi=150, bbox_inches="tight")
        plt.close(fig)
        df = pd.DataFrame(data)
        df.to_csv(out["csv"], index=False, float_format="%.8g")
        lines.append(f"{prop} vs T for each phase at {cs or 'pure ' + els[0]}{ref_txt} "
                     f"(IsolatedPhase: each phase alone, internal DOF relaxed; {round(time.time() - t0, 1)} s)")
        for c in df.columns:
            if c.startswith(prop + "("):
                y = df[c].values
                lines.append(f"  {c[len(prop) + 1:-1]}: {_fmt(np.nanmin(y))} .. {_fmt(np.nanmax(y))} {raw_u}")
        # lowest-G phase by T (only meaningful for GM)
        if prop == "GM":
            gcols = [c for c in df.columns if c.startswith("GM(")]
            best = [gcols[int(np.nanargmin(r))][3:-1] if np.isfinite(r).any() else "-" for r in df[gcols].values]
            lines.append("Lowest-G single phase (ignores phase separation):")
            for a, b, nm in _runs(best):
                lines.append(f"  {_fmt(tdisp[a])}–{_fmt(tdisp[b])} {'°C' if celsius else 'K'}: {nm}")
    if skipped:
        lines.append("skipped: " + "; ".join(skipped))
    out["summary"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------------------
# tool 3: raw equilibrium dataset export (+ legacy eqplot)
# ---------------------------------------------------------------------------
@pcache.cached
def export_equilibrium_data(tdb: str, elements: Sequence[str], conditions: Dict, out_base: str, phases=None,
                            exclude_phases=None, extra_outputs: Optional[Sequence[str]] = None,
                            temperature_unit: str = "K", formats: Sequence[str] = ("nc", "csv"),
                            eqplot: bool = False) -> Dict:
    from pycalphad import equilibrium
    db = _get_db(tdb)
    els = _check_elements(db, elements)
    ph = _resolve_phases(db, els, phases, exclude_phases)
    conds, labels, varying, values = parse_conditions(conditions, els, temperature_unit)
    wnote = _w_to_x(db, els, conds, labels, values)
    npts = int(np.prod([len(a) for a in values.values()]))
    if npts > MAX_POINTS:
        raise ValueError(f"{npts} grid points requested (limit {MAX_POINTS}); use coarser steps")
    extra = [e.strip().upper() for e in (extra_outputs or [])]
    for e in extra:
        if e not in _PHASE_PROPS:
            raise ValueError(f"extra_outputs must be Model properties such as {list(_PHASE_PROPS)} (got {e}); "
                             "use equilibrium_property_plot for anything else")
    formats = [f.lower().lstrip(".") for f in (formats or ["nc", "csv"])]
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, _comps(db, els), ph, conds, output=extra or None)
    secs = round(time.time() - t0, 1)
    out = {"nc": None, "csv": None, "png": None}
    lines = [f"pycalphad.equilibrium for {'-'.join(els)} with {len(ph)} phases, {npts} grid points ({secs} s)",
             f"Conditions: {_describe_conditions(values, temperature_unit)}" + (f" ({wnote})" if wnote else ""),
             "Dataset dims: " + ", ".join(f"{k}={v}" for k, v in eq.sizes.items()),
             "Variables: " + ", ".join(f"{k}{tuple(eq[k].dims)}" for k in eq.data_vars)]
    if "nc" in formats:
        out["nc"] = out_base + ".nc"
        ds = eq.copy()
        ds.attrs = {k: str(v) for k, v in ds.attrs.items()}
        ds.attrs["tdb"] = Path(tdb).name
        ds.attrs["phases"] = " ".join(ph)
        ds.to_netcdf(out["nc"], engine="scipy")
        lines.append("netCDF written with engine='scipy' (netCDF3; netCDF4/h5netcdf are not installed). "
                     "Re-open with xarray.open_dataset(path) - Phase/component come back as object strings.")
    # tidy table: one row per grid point x stable phase
    cond_dims = [d for d in eq.GM.dims]
    comp_names = [str(c) for c in eq.component.values]
    phase_arr = np.asarray(eq.Phase.values, dtype=str)
    npv = eq.NP.values
    xv = eq.X.values
    yv = eq.Y.values
    gm = eq.GM.values
    mu = eq.MU.values
    extra_arrs = {e: eq[e].values for e in extra if e in eq}
    rows = []
    stable_counter = Counter()
    for idx in np.ndindex(*gm.shape):
        base = OrderedDict()
        for d, i in zip(cond_dims, idx):
            val = float(eq.coords[d].values[i])
            if d == "T":
                base["T [K]"] = val
                if _is_celsius(temperature_unit):
                    base["T [°C]"] = val - KELVIN
            elif d == "P":
                base["P [Pa]"] = val
            elif d == "N":
                continue
            else:
                base[d.replace("X_", "X(") + ")" if d.startswith("X_") else d] = val
        base["GM [J/mol]"] = float(gm[idx])
        for j, c in enumerate(comp_names):
            base[f"MU({c}) [J/mol]"] = float(mu[idx + (j,)])
        for e, arr in extra_arrs.items():
            base[e] = float(arr[idx])
        names = [p for p in phase_arr[idx] if p]
        stable_counter["+".join(sorted(names)) or "(failed)"] += 1
        for vtx, pname in enumerate(phase_arr[idx]):
            if not pname:
                continue
            r = OrderedDict(base)
            r["phase"] = pname
            r["NP"] = float(npv[idx + (vtx,)])
            for j, c in enumerate(comp_names):
                r[f"X(phase,{c})"] = float(xv[idx + (vtx, j)])
            yy = yv[idx + (vtx,)]
            for k, yk in enumerate(yy):
                if np.isfinite(yk):
                    r[f"Y{k}"] = float(yk)
            rows.append(r)
    if "csv" in formats:
        import pandas as pd
        out["csv"] = out_base + ".csv"
        pd.DataFrame(rows).to_csv(out["csv"], index=False, float_format="%.8g")
        # site-fraction column meaning per stable phase
        from pycalphad import Model
        ylab = []
        for pname in sorted({r["phase"] for r in rows})[:15]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = Model(db, _comps(db, els), pname)
                ylab.append(f"  {pname}: " + ", ".join(f"Y{k}={sf.species.name}@{sf.sublattice_index}"
                                                        for k, sf in enumerate(m.site_fractions)))
            except Exception:  # noqa: BLE001
                pass
        lines.append(f"CSV: one row per grid point and stable phase ({len(rows)} rows); columns: conditions, GM, "
                     "MU(el), extra outputs, phase, NP, X(phase,el) = phase composition, Y0.. = site fractions:")
        lines += ylab
    lines.append("Stable phase assemblages (grid-point counts): " +
                 ", ".join(f"{k}: {v}" for k, v in stable_counter.most_common(20)))
    if eqplot:
        indep_x = [lab for lab in varying if lab.startswith("X(")]
        indep_p = [lab for lab in varying if lab in ("T", "P")]
        if not ((len(indep_x) == 1 and len(indep_p) == 1) or (len(indep_x) == 2 and not indep_p)):
            lines.append("eqplot skipped: needs one varying X + T (binary grid diagram) or two varying X at fixed T "
                         "(ternary grid diagram)")
        else:
            from pycalphad.plot.eqplot import eqplot as _eqplot
            out["png"] = out_base + ".png"
            proj = "triangular" if len(indep_x) == 2 else None
            fig = plt.figure(figsize=(7, 5.5) if proj is None else (7, 6.5))
            ax = fig.add_subplot(111, projection=proj)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _eqplot(eq.copy(), ax=ax)
            if proj is None and indep_p == ["T"] and _is_celsius(temperature_unit):
                from .diagrams import to_celsius_axis
                to_celsius_axis(ax)
                ax.set_ylabel("Temperature (°C)")
            ax.set_title(f"{'-'.join(e.capitalize() for e in els)} equilibrium grid (eqplot: tie-line end points)",
                         fontsize=10)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)
            fig.savefig(out["png"], dpi=150, bbox_inches="tight")
            plt.close(fig)
    out["summary"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------------------
# tool 4: generic CSV plotter
# ---------------------------------------------------------------------------
def plot_csv(series: List[Dict], out_png: str, resolve=lambda p: p, xlabel=None, ylabel=None, title=None,
             logx=False, logy=False, xlim=None, ylim=None) -> Dict:
    import pandas as pd
    if not series:
        raise ValueError("series must list at least one {csv, x, y}")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    n = 0
    info = []
    cyc = plt.get_cmap("tab10")
    for s in series:
        path = resolve(s["csv"])
        df = pd.read_csv(path, sep=None, engine="python", comment=s.get("comment", "#") or None)
        cols = list(df.columns)

        def col(name):
            if name in df.columns:
                return name
            low = {c.lower(): c for c in cols}
            if name.lower() in low:
                return low[name.lower()]
            pref = [c for c in cols if c.lower().startswith(name.lower() + " [") or c.lower().startswith(name.lower())]
            if len(pref) == 1:
                return pref[0]
            raise ValueError(f"column {name!r} not in {path}; columns: {cols}")
        for k, val in (s.get("where") or {}).items():
            c = col(k)
            df = df[df[c].astype(str) == str(val)]
        xs = col(s["x"])
        ys = s["y"] if isinstance(s["y"], list) else [s["y"]]
        style = s.get("style", "line")
        groups = [(None, df)]
        if s.get("group_by"):
            g = col(s["group_by"])
            groups = list(df.groupby(g, sort=False))
        for yname in ys:
            yc = col(yname)
            for gname, sub in groups:
                sub = sub[[xs, yc]].apply(pd.to_numeric, errors="coerce").dropna()
                if s.get("sort_x", style != "scatter"):
                    sub = sub.sort_values(xs)
                lab = s.get("label") or (yc if gname is None else f"{gname}")
                if gname is not None and len(ys) > 1:
                    lab = f"{gname}: {yc}"
                c = cyc(n % 10)
                kw = dict(color=s.get("color") or c, label=lab)
                if style in ("scatter", "points"):
                    ax.scatter(sub[xs], sub[yc], s=s.get("size", 18), marker=s.get("marker", "o"), **kw)
                elif style == "both":
                    ax.plot(sub[xs], sub[yc], marker=s.get("marker", "o"), ms=4, lw=1.5, **kw)
                else:
                    ax.plot(sub[xs], sub[yc], lw=1.8, ls=s.get("linestyle", "-"), **kw)
                info.append(f"{Path(path).name}: {lab} ({len(sub)} pts)")
                n += 1
        if xlabel is None:
            xlabel = xs
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel or "")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if n > 1 or any(s.get("label") for s in series):
        ax.legend(fontsize=8, ncol=2 if n > 8 else 1)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"png": out_png, "series": info}


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def register(tb):  # noqa: C901
    S = {"type": "string"}
    N = {"type": "number"}
    B = {"type": "boolean"}
    STRS = {"type": "array", "items": {"type": "string"}}
    UNIT = {"type": "string", "enum": ["K", "C"], "description": "unit of temperatures given and shown (default K)"}
    COND = {"type": "object", "description": (
        "Equilibrium conditions. Keys: T, P (Pa, default 101325), X(EL) mole fraction, W(EL) mass fraction, MU(EL). "
        "Value: a number, or [start, stop, step] (inclusive) for a varying condition (a list of != 3 numbers = "
        "explicit values). One or two conditions may vary. Give fractions for all elements but the balance one. "
        "E.g. {\"T\": [600, 1400, 10], \"X(CR)\": 0.18, \"X(NI)\": 0.08}")}

    def base(name: Optional[str], default: str) -> str:
        p = tb.outp(name, default)
        return str(Path(p).with_suffix("")) if Path(p).suffix.lower() in (".png", ".csv", ".nc") else p

    @tb.add("equilibrium_property_plot",
            "General equilibrium property calculator (pycalphad Workspace + property framework): computes any list "
            "of quantities over a 1-D (line plot, one panel per kind of quantity) or 2-D (maps with stable-phase "
            "boundaries) condition grid; writes CSV (+ optional netCDF) and returns a summary incl. where phases "
            "appear/disappear. Use for: phase fractions vs T or X (NP(*)), phase compositions (X(*,CR)), site "
            "fractions / degree of order (Y(..), DOO(PH)), GM/HM/SM/CPM vs T, apparent Cp (HM.T), chemical "
            "potentials and activities vs composition (MU(CR), ACR(CR,BCC_A2)), driving force of a phase, e.g. "
            "sigma precipitation vs T (DF(SIGMA)), metastable single-phase properties (ISOLATED(FCC_A1):GM), T0. "
            + QUANTITY_HELP,
            {"tdb": S, "elements": {**STRS, "description": "all elements, e.g. [\"FE\",\"CR\",\"NI\"]"},
             "conditions": COND,
             "quantities": {**STRS, "description": "e.g. [\"NP(*)\"], [\"HM\",\"CPM\"], [\"ACR(CR,BCC_A2)\"], "
                            "[\"DF(SIGMA)\"], [\"X(*,CR)\"]"},
             "phases": {**STRS, "description": "optional phase subset (default: all phases possible for elements)"},
             "exclude_phases": {**STRS, "description": "phases to suspend (e.g. to get DF of a suppressed phase)"},
             "temperature_unit": UNIT,
             "x_axis": {"type": "string", "description": "1-D: plot vs this computed quantity instead of the varying "
                        "condition (e.g. \"X(FCC_A1,CR)\"); 2-D: which varying condition goes on the x axis"},
             "plot": {"type": "boolean", "description": "make the PNG (default true)"},
             "write_netcdf": {"type": "boolean", "description": "also write an .nc file (default false)"},
             "output_name": {"type": "string", "description": "base file name (no extension)"},
             "title": S},
            ["tdb", "elements", "conditions", "quantities"])
    def _eq_prop(tdb, elements, conditions, quantities, phases=None, exclude_phases=None, temperature_unit="K",
                 x_axis=None, plot=True, write_netcdf=False, output_name=None, title=None):
        t = tb._find_tdb(tdb)
        els = sorted(e.upper() for e in elements)
        qs = [quantities] if isinstance(quantities, str) else list(quantities)
        stem = _safe_name(f"eqprop_{'-'.join(els)}_{'_'.join(_normalize_q(q) for q in qs)[:40]}")
        r = equilibrium_property_plot(t, elements, conditions, qs, base(output_name, stem), phases=phases,
                                      exclude_phases=exclude_phases, temperature_unit=temperature_unit or "K",
                                      x_axis=x_axis, plot=plot, write_netcdf=write_netcdf, title=title)
        files = [f"CSV: {tb.rel(r['csv'])} ({r['n_points']} rows; columns {', '.join(r['columns'][:14])}"
                 f"{'…' if len(r['columns']) > 14 else ''})"]
        if r["png"]:
            files.append(f"PNG: {tb.rel(r['png'])}")
        if r["nc"]:
            files.append(f"netCDF: {tb.rel(r['nc'])} (scipy engine, netCDF3)")
        return r["summary"] + "\n" + "\n".join(files), [r["png"]] if r["png"] else []

    @tb.add("phase_property_plot",
            "Single-phase properties WITHOUT global equilibrium (pycalphad calculate / IsolatedPhase): for each "
            "phase, prop vs mole fraction in a binary at one temperature (lowest-G internal configuration per "
            "composition, optional scatter of all sampled configurations and common-tangent/convex-hull "
            "construction with tie-line compositions), or prop vs temperature at a fixed composition "
            "(temperature=[start, stop, step]). prop: GM HM SM CPM (J/mol-atoms, SER), GM_MIX HM_MIX SM_MIX "
            "CPM_MIX (relative to each phase's own endmembers), DOO, TC, BMAG. `reference` re-references GM/HM/SM/CPM "
            "to pure elements in chosen phases at the same T, e.g. {\"CR\":\"BCC_A2\",\"NI\":\"FCC_A1\"} (gives "
            "Gibbs energy of mixing/formation curves vs a common reference). Writes PNG + CSV.",
            {"tdb": S, "elements": STRS, "prop": S,
             "temperature": {"description": "number (curves vs composition, binary) or [start, stop, step] (curves "
                             "vs T at fixed composition)", "anyOf": [N, {"type": "array", "items": N}]},
             "temperature_unit": UNIT,
             "composition": {"type": "object", "additionalProperties": N,
                             "description": "for vs-T mode: mole fractions of all elements but the balance"},
             "phases": STRS, "exclude_phases": STRS,
             "x_element": {"type": "string", "description": "binary: element on the x axis (default: 2nd alphabetically)"},
             "reference": {"type": "object", "additionalProperties": S,
                           "description": "element -> reference phase for every element"},
             "show_all_points": {"type": "boolean", "description": "scatter every sampled internal configuration"},
             "common_tangent": {"type": "boolean", "description": "prop=GM only: draw lower convex hull / common "
                                "tangents and report stable phases and tie-lines"},
             "output_name": S, "title": S},
            ["tdb", "elements", "prop", "temperature"])
    def _phase_prop(tdb, elements, prop, temperature, temperature_unit="K", composition=None, phases=None,
                    exclude_phases=None, x_element=None, reference=None, show_all_points=False,
                    common_tangent=False, output_name=None, title=None):
        t = tb._find_tdb(tdb)
        els = sorted(e.upper() for e in elements)
        tv = temperature if not isinstance(temperature, list) else "T" + "-".join(f"{x:g}" for x in temperature[:2])
        stem = _safe_name(f"phaseprop_{'-'.join(els)}_{prop.upper()}_{tv}{temperature_unit or 'K'}")
        r = phase_property_plot(t, elements, prop, base(output_name, stem), temperature=temperature,
                                composition=composition, phases=phases, exclude_phases=exclude_phases,
                                temperature_unit=temperature_unit or "K", x_element=x_element, reference=reference,
                                show_all_points=show_all_points, common_tangent=common_tangent, title=title)
        return f"{r['summary']}\nPNG: {tb.rel(r['png'])}\nCSV: {tb.rel(r['csv'])}", [r["png"]]

    @tb.add("export_equilibrium_data",
            "Run pycalphad.equilibrium over a condition grid (<=5000 points) and save the raw xarray Dataset "
            "(NP, GM, MU, X, Y, Phase + extra Model outputs such as HM, SM, CPM, DOO) to netCDF (.nc) and a tidy CSV "
            "(one row per grid point and stable phase: conditions, GM, MU, phase, NP, phase composition, site "
            "fractions). Optional eqplot=true draws pycalphad's legacy eqplot grid diagram (tie-line end points) "
            "for an X-T grid (binary) or X-X grid at fixed T (ternary) - a robust brute-force fallback when "
            "mapping-based phase diagrams fail.",
            {"tdb": S, "elements": STRS, "conditions": COND, "phases": STRS, "exclude_phases": STRS,
             "extra_outputs": {**STRS, "description": "e.g. [\"HM\",\"SM\",\"CPM\",\"DOO\"]"},
             "temperature_unit": UNIT,
             "formats": {"type": "array", "items": {"type": "string", "enum": ["nc", "csv"]},
                         "description": "default both"},
             "eqplot": B, "output_name": S},
            ["tdb", "elements", "conditions"])
    def _export(tdb, elements, conditions, phases=None, exclude_phases=None, extra_outputs=None,
                temperature_unit="K", formats=None, eqplot=False, output_name=None):
        t = tb._find_tdb(tdb)
        els = sorted(e.upper() for e in elements)
        r = export_equilibrium_data(t, elements, conditions, base(output_name, f"eqdata_{'-'.join(els)}"),
                                    phases=phases, exclude_phases=exclude_phases, extra_outputs=extra_outputs,
                                    temperature_unit=temperature_unit or "K", formats=formats or ("nc", "csv"),
                                    eqplot=eqplot)
        files = [f"{k}: {tb.rel(r[k])}" for k in ("nc", "csv", "png") if r[k]]
        return r["summary"] + "\n" + "\n".join(files), [r["png"]] if r["png"] else []

    series_schema = {"type": "object", "properties": {
        "csv": S, "x": {"type": "string", "description": "column name (prefix match allowed, e.g. 'T' or 'NP(SIGMA)')"},
        "y": {"description": "column name or list of names", "anyOf": [S, STRS]},
        "label": S, "style": {"type": "string", "enum": ["line", "scatter", "both"]},
        "where": {"type": "object", "description": "row filter {column: value}"},
        "group_by": {"type": "string", "description": "one curve per distinct value of this column (e.g. phase)"},
        "color": S, "marker": S, "linestyle": S}, "required": ["csv", "x", "y"]}

    # (CSV re-plotting/overlays: see make_plot / edit_figure in calphad_tools/figures.py)
