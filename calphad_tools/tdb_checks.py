"""Sanity checks for CALPHAD databases (``check_tdb`` tool).

Looks for the usual assessment / database-assembly pathologies and returns a prioritized
report (ERROR / WARNING / INFO) with evidence numbers, the responsible parameter(s) or
function(s), a "what it usually means / how to fix" hint, a JSON file of the findings and
diagnostic plots.

Checks (``checks`` names):

* ``syntax``       undefined / unused / duplicate functions, duplicate parameters (pycalphad *adds*
                   duplicates, Thermo-Calc keeps the last one), parameters for undefined phases or with
                   undeclared / wrong constituents, TYPE_DEFINITIONs pointing at missing phases, order /
                   disorder (DIS_PART) pairs whose sublattices do not match, parser warnings.
* ``ranges``       T-range order (overlapping ranges), lower limits above tmin, upper limits below
                   tmax (extrapolation).
* ``breakpoints``  jumps of G and dG/dT (= -S) at the breakpoints of piecewise functions/parameters.
* ``load``         pycalphad Database load + Model build for every phase (``merge.validate``).
* ``unary``        per element: is the SER phase the most stable solid at 298.15 K, melting point
                   (vs the SGTE value when the unary is SER based), liquid stable above Tm, no solid
                   re-stabilised above Tm up to tmax (inverse melting), Cp of every pure-element
                   end-member positive and < 100 J/mol/K.
* ``interactions`` very large interaction parameters or temperature coefficients and large excess Cp
                   (-T d2L/dT2, from T*LN(T), T**2 ... terms).
* ``binary``       per binary on coarse grids: liquid must be stable at all compositions well above
                   the melting points (solid phases re-appearing at high T), liquid (and simple solution
                   phase) miscibility gaps at high T / inverted gaps, negative or huge liquid Cp.

All pycalphad calculations use pycalphad's semantics (duplicate parameters summed, expressions
extrapolated beyond their T limits).
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .tdbparse import (TDB, ExpressionError, Function, Piecewise, _key_tuple, _split_commands,  # noqa: E402
                       parse_tdb, resolve_keyword)

R_GAS = 8.3145
INTERSTITIALS = {"C", "N", "O", "B", "H"}     # sublattices holding only these (+VA) may be mostly vacant
ALL_CHECKS = ("syntax", "ranges", "breakpoints", "load", "unary", "interactions", "binary")
SEVERITIES = ("ERROR", "WARNING", "INFO")

# Melting points (K) of the SGTE unary database (Dinsdale 1991 / SGTE unary 5.0).
SGTE_TM = {
    "AG": 1234.93, "AL": 933.47, "AU": 1337.33, "BI": 544.55, "CD": 594.22, "CO": 1768.0, "CR": 2180.0,
    "CU": 1357.77, "FE": 1811.0, "GA": 302.91, "GE": 1211.4, "HF": 2506.0, "IN": 429.75, "MG": 923.0,
    "MN": 1519.0, "MO": 2896.0, "NB": 2750.0, "NI": 1728.0, "PB": 600.61, "PD": 1828.05, "PT": 2041.4,
    "RE": 3458.0, "SB": 903.78, "SI": 1687.0, "SN": 505.08, "TA": 3290.0, "TI": 1941.0, "V": 2183.0,
    "W": 3695.0, "ZN": 692.68, "ZR": 2128.0,
}

THRESHOLDS = {
    "breakpoint_dG_warning_J": 1.0, "breakpoint_dG_error_J": 100.0,
    "breakpoint_dS_warning_J_K": 0.1, "breakpoint_dS_error_J_K": 5.0,
    "low_limit_info_K": 5.0,                  # lower limit above tmin by <= this: INFO, else WARNING
    "tm_mismatch_K": 5.0,
    "ser_H298_tol_J": 200.0,                  # |H(298.15)| of the reference phase for a SER-based unary
    "cp_max_J_K": 100.0, "cp_min_J_K": 0.0,
    "restabilisation_tol_J": 1.0,
    "L_max_J_per_mol_atoms": 200e3, "dLdT_max_J_K": 100.0, "excess_cp_max_J_K": 15.0,
    "high_T_margin_K": 500.0,                 # binary checks start at max(Tm) + margin
    "high_T_error_margin_K": 1500.0,          # high-T artefacts starting below max(Tm)+this -> ERROR, above -> WARNING
    "vacancy_max_site_fraction": 0.1,         # sampled points with more VA on a metal sublattice are ignored
    "hull_tol_J": 1.0,
    "liquid_cp_xs_max_J_K": 15.0,
}

FIX = {
    "parse_warning": "The line could not be parsed; Thermo-Calc/pycalphad may reject or misread it.",
    "undefined_function": "Define the missing FUNCTION (typo, or it lives in a unary/other database) or remove the "
                          "reference; pycalphad silently evaluates an undefined function as 0, so the affected phase is "
                          "badly wrong without any error message.",
    "unused_function": "Harmless (often left over from a unary database); remove to tidy up.",
    "duplicate_function": "Only the last definition is used; delete the other(s) so readers are not misled.",
    "duplicate_parameter": "Thermo-Calc keeps the LAST definition but pycalphad ADDS all of them, so the two programs "
                           "disagree (and pycalphad double-counts even identical copies). Keep exactly one.",
    "param_undefined_phase": "The parameter is silently ignored. Add the PHASE (and CONSTITUENT) definition or fix "
                             "the phase name.",
    "undeclared_species": "Declare the ELEMENT/SPECIES or fix the constituent name (typo?).",
    "param_constituent_not_in_phase": "The parameter is ignored because the constituent is not listed for that "
                                      "sublattice in CONSTITUENT; add it there or fix the parameter.",
    "param_sublattice_mismatch": "The number of sublattices in the parameter does not match the PHASE definition.",
    "phase_no_constituents": "Add a CONSTITUENT command for the phase.",
    "constituent_undefined_phase": "CONSTITUENT given for a phase that has no PHASE command.",
    "typedef_missing_phase": "The TYPE_DEFINITION amends a phase that does not exist (typo or phase removed).",
    "typedef_undefined_code": "The PHASE uses a type code with no TYPE_DEFINITION; usually harmless (ignored).",
    "dis_part_missing_phase": "The disordered phase named in DIS_PART does not exist; the ordered phase loses its "
                              "disordered contribution.",
    "dis_part_mismatch": "Ordered and disordered phases must have matching sublattices (ordered sublattices summing to "
                         "the disordered one, same constituents) or the order/disorder model is wrong.",
    "dis_part_unused": "No phase uses this TYPE_DEFINITION code, so the DIS_PART is not applied.",
    "pycalphad_load": "pycalphad cannot load the database or build a phase model; the other checks may be "
                      "incomplete.",
    "range_order": "T limits must increase; overlapping or reversed ranges are undefined behaviour.",
    "range_gap_low": "Below the first limit the expression is extrapolated (pycalphad silently, Thermo-Calc with a "
                     "warning). 300 vs 298.15 K is harmless; a larger gap usually is a typo in the limits.",
    "range_extrapolated_high": "Above the last limit the expression is extrapolated; Cp terms (T*LN(T), T**2, "
                               "T**3, T**-1) often misbehave there. Extend the description (SGTE uses T**-9 above "
                               "Tm) or keep calculations below the limit.",
    "breakpoint_jump": "G (and normally H and S) must be continuous at a breakpoint; a jump makes steps in H/S and "
                       "phase-boundary kinks. Usually a typo in a coefficient or mismatched ranges.",
    "breakpoint_slope": "dG/dT = -S jumps at the breakpoint (latent heat without a transition). Usually a typo or "
                        "an expression fitted independently on each side.",
    "non_ser_reference": "G of the pure element is not referred to SER (e.g. Kaufman lattice stabilities with one "
                         "phase = 0); absolute Cp/S checks are skipped and the database cannot be combined with "
                         "SER-based data without re-referencing.",
    "ser_offset": "Under SER, H of the reference phase at 298.15 K is 0 by definition. An offset means a constant (or "
                  "extra terms) was added to the reference phase; all phases of the element are affected relative to "
                  "other databases.",
    "ser_entropy_mismatch": "S298 in the ELEMENT line differs from -dG/dT of the reference phase; only the ELEMENT "
                            "line data is affected (used for H/S output in some software).",
    "ref_phase_missing": "The ELEMENT reference phase is not a phase of the database.",
    "ref_phase_not_stable": "Another solid is more stable than the SER reference phase at 298.15 K: wrong lattice "
                            "stability (sign or magnitude) or wrong reference phase in the ELEMENT line.",
    "liquid_stable_at_tmin": "Liquid is the most stable phase already at tmin: lattice stabilities are wrong (or this "
                             "is a liquid metal like Hg with a solid reference).",
    "no_melting": "The liquid never becomes stable up to tmax: a solid lattice stability or the liquid description "
                  "is wrong (often a +/- sign or a T coefficient typo).",
    "inverse_melting": "A solid becomes more stable than the liquid again above Tm (inverse melting). Typical cause: "
                       "Cp terms extrapolated beyond their range, missing T**(-9) terms, or a metastable end-member "
                       "with a too negative T coefficient. The solid will appear in high-T equilibria.",
    "tm_mismatch": "The unary looks SGTE/SER-based but the melting point differs from SGTE; check the liquid or solid "
                   "description (mixed unary versions, modified lattice stability).",
    "tm_info": "Computed melting point for information.",
    "negative_cp": "Negative heat capacity is unphysical (usually a T**2 / T**3 / T**-1 coefficient with the wrong "
                   "sign or an expression used outside its range).",
    "huge_cp": "Cp > 100 J/mol/K for a pure element is not physical (except narrow magnetic/ordering peaks); check "
               "T*LN(T)/T**2 coefficients and extrapolation.",
    "large_interaction": "Interaction parameters above ~200 kJ/mol are extreme; they often produce artificial "
                         "miscibility gaps or phases stabilised far from their range. Check units/typos.",
    "large_tcoef": "|dL/dT| > 100 J/mol/K (excess entropy) is extreme; at high T such terms invert stability "
                   "(e.g. liquid gaps opening on heating).",
    "excess_cp": "Large excess Cp of a solution (from T*LN(T), T**2 ... in interaction parameters) makes the phase "
                 "misbehave on extrapolation to high T. Prefer a + b*T interactions unless Cp data require more.",
    "binary_solid_above_liquid": "A solid phase is more stable than the liquid far above the melting points of both "
                                 "elements (inverted liquidus / phase re-appearing at high T). Typical causes: "
                                 "compound G with Cp terms extrapolated to high T, too negative entropy of the liquid "
                                 "interaction, or too negative T coefficient of the solid. Unless the compound is a "
                                 "genuinely refractory one (carbide/boride/nitride/oxide) this is an artefact.",
    "liquid_miscibility_gap": "The liquid separates into two liquids at high T. Real liquid gaps (monotectic "
                              "systems) close at a critical temperature; one that persists to very high T or opens "
                              "on heating (inverted) comes from a large positive or T-increasing interaction "
                              "(L = a + b*T with large positive b).",
    "liquid_gap_low_T": "Liquid miscibility gap (possibly metastable, below the liquidus). Fine if the system is a "
                        "known monotectic, otherwise the liquid interactions are too positive.",
    "solution_gap_high_T": "A solution phase shows a miscibility gap far above the melting points (artificial "
                           "positive interaction / temperature coefficient). Harmless for stable equilibria while "
                           "the liquid is stable, but it misbehaves in metastable calculations and extrapolations.",
    "liquid_negative_cp": "The liquid has a negative Cp at high T because of excess Cp in its interaction parameters.",
    "liquid_excess_cp": "The liquid's excess Cp is large at high T (T*LN(T) or similar terms in its interactions).",
    "missing_endmember": "Usual for ordered phases described by an ordering contribution only; if the phase should "
                         "include the disordered state, add a DIS_PART type definition or the end-member parameters.",
    "eval_failed": "The phase could not be evaluated with pycalphad (see the load/syntax findings).",
    "skipped": "Part of the check was skipped (time limit or missing data).",
}


@dataclass
class Finding:
    severity: str
    check: str
    code: str
    message: str
    responsible: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def fix(self) -> str:
        return FIX.get(self.code, "")

    def to_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity, "check": self.check, "code": self.code, "message": self.message,
                "responsible": self.responsible, "evidence": self.evidence, "fix": self.fix}


class Findings(list):
    def add(self, severity: str, check: str, code: str, message: str, responsible=None, **evidence) -> Finding:
        f = Finding(severity, check, code, message, list(responsible or []), _clean(evidence))
        self.append(f)
        return f


def _clean(obj):
    """JSON-friendly rounding."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return float(f"{v:.6g}")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _fmt_list(names: Sequence[str], n: int = 8) -> str:
    names = list(names)
    s = ", ".join(names[:n])
    return s + (f" (+{len(names) - n} more)" if len(names) > n else "")


# ---------------------------------------------------------------------------
# phase classification / expression helpers
# ---------------------------------------------------------------------------
def _is_gas(tdb: TDB, name: str) -> bool:
    ph = tdb.phases.get(name)
    return (ph is not None and ph.suffix.upper() == ":G") or name.upper() in ("GAS", "GAS_PHASE", "IDEAL_GAS")


def _is_liquid(tdb: TDB, name: str) -> bool:
    ph = tdb.phases.get(name)
    n = name.upper()
    return (ph is not None and ph.suffix.upper() == ":L") or n.startswith("LIQ") or n.startswith("IONIC_LIQ")


def _extend(pw: Piecewise) -> Piecewise:
    """pycalphad semantics: first range extrapolated downwards, last one upwards."""
    r = list(pw.ranges)
    if not r:
        return pw
    r[0] = (0.0, r[0][1], r[0][2])
    r[-1] = (r[-1][0], 1e7, r[-1][2])
    return Piecewise(r, pw.reference)


def _extended_tdb(tdb: TDB) -> TDB:
    t2 = copy.copy(tdb)
    t2.functions = {k: Function(k, _extend(f.expr)) for k, f in tdb.functions.items()}
    return t2


def _func_closure(tdb: TDB, names: Sequence[str]) -> List[str]:
    out, stack = [], list(names)
    while stack:
        n = stack.pop()
        if n in out or n not in tdb.functions:
            continue
        out.append(n)
        stack.extend(tdb.functions[n].expr.symbols())
    return out


def _breakpoints(tdb: TDB, pw: Piecewise) -> List[float]:
    bps = [lo for lo, _, _ in pw.ranges[1:]]
    for fn in _func_closure(tdb, pw.symbols()):
        bps += [lo for lo, _, _ in tdb.functions[fn].expr.ranges[1:]]
    return sorted(set(bps))


def _dis_part_of(tdb: TDB) -> Dict[str, str]:
    """ordered phase -> disordered phase."""
    out = {}
    for ph in tdb.phases.values():
        for td in tdb.phase_typedefs(ph):
            m = re.search(r"DIS\w*_P\w*\s+([A-Z0-9_]+)", td.text, re.I)
            if m:
                out[ph.name] = m.group(1).upper()
    return out


def _phase_params(tdb: TDB, phase: str, allowed: Set[str], ptypes=("G",)) -> List[str]:
    """Raw names of parameters of ``phase`` (and its DIS_PART) whose constituents are all in ``allowed``."""
    phases = {phase}
    dp = _dis_part_of(tdb).get(phase)
    if dp:
        phases.add(dp)
    out = []
    for p in tdb.parameters:
        if p.key.phase in phases and p.key.ptype in ptypes:
            if all(s in allowed or s == "*" for sl in p.key.constituents for s in sl):
                out.append(p.raw_name)
    return out


def _with_functions(tdb: TDB, raw_names: Sequence[str], nmax: int = 4) -> List[str]:
    """Parameter names followed by the functions they reference (first level)."""
    out = list(raw_names)
    byname = {p.raw_name: p for p in tdb.parameters}
    for rn in raw_names:
        p = byname.get(rn)
        if p is None:
            continue
        for s in p.expr.symbols()[:nmax]:
            if s in tdb.functions and f"FUNCTION {s}" not in out:
                out.append(f"FUNCTION {s}")
    return out


# ---------------------------------------------------------------------------
# 1. syntax / consistency
# ---------------------------------------------------------------------------
def check_syntax(tdb: TDB, text: str, F: Findings) -> None:
    for w in tdb.warnings:
        sev = "ERROR" if w.startswith("CONSTITUENT for undefined phase") else "WARNING"
        code = "constituent_undefined_phase" if sev == "ERROR" else "parse_warning"
        F.add(sev, "syntax", code, w)

    # --- function references
    users: Dict[str, List[str]] = {}
    for fn in tdb.functions.values():
        for s in fn.expr.symbols():
            users.setdefault(s, []).append(f"FUNCTION {fn.name}")
    for p in tdb.parameters:
        for s in p.expr.symbols():
            users.setdefault(s, []).append(p.raw_name)
    for s, us in sorted(users.items()):
        if s not in tdb.functions and s != "R":
            F.add("ERROR", "syntax", "undefined_function", f"undefined function {s} referenced by {_fmt_list(us, 5)}",
                  [s] + us, referenced_by=us)
    used = set(_func_closure(tdb, [s for p in tdb.parameters for s in p.expr.symbols()]))
    unused = sorted(set(tdb.functions) - used)
    if unused:
        F.add("INFO", "syntax", "unused_function", f"{len(unused)} function(s) not used by any parameter: "
              f"{_fmt_list(unused, 12)}", unused, functions=unused)

    # --- duplicate FUNCTIONs (parse_tdb keeps only the last one)
    cmds, _ = _split_commands(text)
    fcount: Dict[str, int] = {}
    for c in cmds:
        w = c.split()
        if len(w) > 1 and resolve_keyword(w[0]) == "FUNCTION":
            nm = w[1].upper().rstrip("#")
            fcount[nm] = fcount.get(nm, 0) + 1
    for nm, n in sorted(fcount.items()):
        if n > 1:
            F.add("WARNING", "syntax", "duplicate_function", f"FUNCTION {nm} is defined {n} times (the last one "
                  "is used)", [f"FUNCTION {nm}"], count=n)

    # --- duplicate PARAMETERs
    groups: Dict[Tuple, List] = {}
    for p in tdb.parameters:
        groups.setdefault(_key_tuple(p.key), []).append(p)
    ext = _extended_tdb(tdb)
    for key, ps in groups.items():
        if len(ps) < 2:
            continue
        vals = []
        for p in ps:
            try:
                vals.append(float(ext.evaluate(p.expr, [1000.0])[0] * p.sign))
            except ExpressionError:
                vals.append(float("nan"))
        same = len({round(v, 3) for v in vals if math.isfinite(v)}) <= 1
        F.add("ERROR", "syntax", "duplicate_parameter",
              f"{ps[0].key.label()} defined {len(ps)} times (values at 1000 K: "
              f"{', '.join(f'{v:.6g}' for v in vals)} J/mol{'; identical' if same else ''}); pycalphad uses the sum "
              f"{sum(vals):.6g}, Thermo-Calc the last {vals[-1]:.6g}", [p.raw_name for p in ps],
              values_1000K=vals, identical=same)

    # --- phases / constituents
    declared = set(tdb.elements) | set(tdb.species)
    for ph in tdb.phases.values():
        if not ph.constituents:
            F.add("ERROR", "syntax", "phase_no_constituents", f"phase {ph.name} has no CONSTITUENT command",
                  [ph.name])
            continue
        if len(ph.constituents) != ph.n_sublattices:
            F.add("ERROR", "syntax", "param_sublattice_mismatch", f"phase {ph.name}: {ph.n_sublattices} sublattices "
                  f"declared but CONSTITUENT lists {len(ph.constituents)}", [ph.name])
        bad = sorted({s for sl in ph.constituents for s in sl if s not in declared})
        if bad:
            F.add("ERROR", "syntax", "undeclared_species", f"phase {ph.name}: constituent(s) {bad} not declared as "
                  "ELEMENT or SPECIES", [ph.name], species=bad)

    undefined_phase: Dict[str, List[str]] = {}
    for p in tdb.parameters:
        ph = tdb.phases.get(p.key.phase)
        if ph is None:
            undefined_phase.setdefault(p.key.phase, []).append(p.raw_name)
            continue
        if not ph.constituents:
            continue
        cons = p.key.constituents
        if len(cons) != ph.n_sublattices:
            F.add("ERROR", "syntax", "param_sublattice_mismatch", f"{p.raw_name}: {len(cons)} sublattice(s) given, "
                  f"phase {ph.name} has {ph.n_sublattices}", [p.raw_name])
            continue
        for i, sl in enumerate(cons):
            for s in sl:
                if s == "*":
                    continue
                if s not in declared:
                    F.add("ERROR", "syntax", "undeclared_species", f"{p.raw_name}: constituent {s} is not a declared "
                          "ELEMENT/SPECIES", [p.raw_name], species=s)
                elif i < len(ph.constituents) and s not in ph.constituents[i]:
                    F.add("WARNING", "syntax", "param_constituent_not_in_phase",
                          f"{p.raw_name}: {s} is not a constituent of sublattice {i + 1} of {ph.name} "
                          f"({','.join(ph.constituents[i])}); the parameter is ignored", [p.raw_name])
    for phn, names in sorted(undefined_phase.items()):
        F.add("ERROR", "syntax", "param_undefined_phase", f"{len(names)} parameter(s) for phase {phn}, which has no "
              f"PHASE definition: {_fmt_list(names, 6)}", names, phase=phn)

    # --- type definitions
    used_codes = {c for ph in tdb.phases.values() for c in ph.type_codes}
    for code, td in tdb.typedefs.items():
        tp = td.target_phase
        if tp and tp != "@" and tp not in tdb.phases:
            F.add("WARNING", "syntax", "typedef_missing_phase", f"TYPE_DEFINITION {code} amends undefined phase {tp}: "
                  f"{td.text[:80]}", [f"TYPE_DEFINITION {code}"], phase=tp)
        m = re.search(r"DIS\w*_P\w*\s+([A-Z0-9_]+)", td.text, re.I)
        if m:
            dis = m.group(1).upper()
            if dis not in tdb.phases:
                F.add("ERROR", "syntax", "dis_part_missing_phase", f"TYPE_DEFINITION {code}: disordered phase {dis} "
                      "is not defined", [f"TYPE_DEFINITION {code}"], phase=dis)
            if code not in used_codes:
                F.add("WARNING", "syntax", "dis_part_unused", f"TYPE_DEFINITION {code} ({td.text[:60]}) is not used by "
                      "any PHASE", [f"TYPE_DEFINITION {code}"])
    for ph in tdb.phases.values():
        for c in ph.type_codes:
            if c != "%" and c not in tdb.typedefs:
                F.add("INFO", "syntax", "typedef_undefined_code", f"phase {ph.name} uses type code {c!r} which has no "
                      "TYPE_DEFINITION", [ph.name])
    for ordered, dis in _dis_part_of(tdb).items():
        o, d = tdb.phases[ordered], tdb.phases.get(dis)
        if d is None:
            continue
        k = o.n_sublattices - d.n_sublattices + 1
        problems = []
        if k < 1:
            problems.append(f"disordered phase has more sublattices ({d.n_sublattices}) than the ordered one "
                            f"({o.n_sublattices})")
        else:
            if abs(sum(o.site_ratios[:k]) - d.site_ratios[0]) > 1e-6 * max(1.0, d.site_ratios[0]):
                problems.append(f"site ratios {o.site_ratios[:k]} of {ordered} sum to {sum(o.site_ratios[:k]):g}, "
                                f"not {d.site_ratios[0]:g} as in {dis}")
            if [round(x, 6) for x in o.site_ratios[k:]] != [round(x, 6) for x in d.site_ratios[1:]]:
                problems.append(f"remaining site ratios {o.site_ratios[k:]} != {d.site_ratios[1:]}")
            if o.constituents and d.constituents and len(o.constituents) == o.n_sublattices:
                ou = set().union(*o.constituents[:k])
                extra = ou - set(d.constituents[0])
                if extra:
                    problems.append(f"{ordered} has constituents {sorted(extra)} that {dis} lacks")
                for i, (a, b) in enumerate(zip(o.constituents[k:], d.constituents[1:])):
                    if set(a) != set(b):
                        problems.append(f"sublattice {k + i + 1} constituents differ: {a} vs {b}")
        if problems:
            F.add("ERROR", "syntax", "dis_part_mismatch", f"order/disorder pair {ordered}/{dis}: " + "; ".join(problems),
                  [ordered, dis])


# ---------------------------------------------------------------------------
# 1b. temperature ranges and breakpoints
# ---------------------------------------------------------------------------
def _all_exprs(tdb: TDB):
    for fn in tdb.functions.values():
        yield f"FUNCTION {fn.name}", fn.expr, 1.0
    for p in tdb.parameters:
        yield p.raw_name, p.expr, p.sign


def check_ranges(tdb: TDB, tmin: float, tmax: float, F: Findings) -> None:
    low_info, low_warn, high = [], [], []
    for name, pw, _ in _all_exprs(tdb):
        if not pw.ranges:
            continue
        for (lo, hi, _e) in pw.ranges:
            if hi <= lo:
                F.add("ERROR", "ranges", "range_order", f"{name}: T range [{lo:g}, {hi:g}] is empty or reversed "
                      "(overlapping ranges)", [name], low=lo, high=hi)
        lo0, hiN = pw.ranges[0][0], pw.ranges[-1][1]
        if lo0 > tmin + 0.011:
            (low_info if lo0 - tmin <= THRESHOLDS["low_limit_info_K"] else low_warn).append((name, lo0))
        if hiN < tmax - 0.011:
            high.append((name, hiN))
    if low_warn:
        F.add("WARNING", "ranges", "range_gap_low", f"{len(low_warn)} expression(s) start well above tmin={tmin:g} K "
              f"(gap below the first limit): " + _fmt_list([f"{n} from {t:g} K" for n, t in low_warn], 8),
              [n for n, _ in low_warn], limits={n: t for n, t in low_warn})
    if low_info:
        F.add("INFO", "ranges", "range_gap_low", f"{len(low_info)} expression(s) start slightly above tmin={tmin:g} K "
              f"(e.g. 300 K): " + _fmt_list([f"{n} ({t:g} K)" for n, t in low_info], 6),
              [n for n, _ in low_info], limits={n: t for n, t in low_info})
    if high:
        F.add("WARNING", "ranges", "range_extrapolated_high", f"{len(high)} expression(s) end below tmax={tmax:g} K and "
              "are extrapolated above: " + _fmt_list([f"{n} (to {t:g} K)" for n, t in sorted(high, key=lambda x: x[1])],
                                                     10),
              [n for n, _ in high], limits={n: t for n, t in high})


def check_breakpoints(tdb: TDB, F: Findings, delta: float = 1e-6, h: float = 1e-2) -> None:
    lookup = tdb.evaluator()
    th = THRESHOLDS
    used = set(_func_closure(tdb, [s for p in tdb.parameters for s in p.expr.symbols()]))
    for name, pw, _ in _all_exprs(tdb):
        if len(pw.ranges) < 2:
            continue
        unused = name.startswith("FUNCTION ") and name.split()[1] not in used
        for i in range(len(pw.ranges) - 1):
            lo_l, hi_l, e_l = pw.ranges[i]
            lo_r, hi_r, e_r = pw.ranges[i + 1]
            bp = hi_l
            if hi_l <= lo_l or hi_r <= lo_r:
                continue
            try:
                left = Piecewise([(lo_l, hi_l, e_l)]).evaluate(np.array([bp - delta - h, bp - delta]), lookup)
                right = Piecewise([(lo_r, hi_r, e_r)]).evaluate(np.array([bp + delta, bp + delta + h]), lookup)
            except ExpressionError:
                continue
            if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
                continue
            dG = right[0] - left[1]
            dS = -((right[1] - right[0]) / h - (left[1] - left[0]) / h)      # jump of S = -dG/dT
            sev_g = ("ERROR" if abs(dG) > th["breakpoint_dG_error_J"] else
                     "WARNING" if abs(dG) > th["breakpoint_dG_warning_J"] else None)
            sev_s = ("ERROR" if abs(dS) > th["breakpoint_dS_error_J_K"] else
                     "WARNING" if abs(dS) > th["breakpoint_dS_warning_J_K"] else None)
            note = " (function not used by any parameter)" if unused else ""
            if unused:
                sev_g = sev_g and "INFO"
                sev_s = sev_s and "INFO"
            if sev_g:
                F.add(sev_g, "breakpoints", "breakpoint_jump", f"{name}: G jumps by {dG:+.4g} J/mol at the "
                      f"breakpoint T={bp:g} K (dS = {dS:+.3g} J/mol/K){note}", [name], T=bp, dG_J=dG, dS_J_K=dS)
            elif sev_s:
                F.add(sev_s, "breakpoints", "breakpoint_slope", f"{name}: dG/dT jumps at T={bp:g} K: entropy jump "
                      f"dS = {dS:+.4g} J/mol/K (latent heat T*dS = {bp * dS:+.4g} J/mol; G jump {dG:+.2g} J/mol){note}",
                      [name], T=bp, dG_J=dG, dS_J_K=dS)


def check_load(tdb_path: str, F: Findings) -> Dict:
    from .merge import validate
    r = validate(tdb_path)
    for e in r.get("errors", []):
        if e.startswith("undefined functions referenced"):
            continue      # reported (with the users) by the syntax check
        F.add("ERROR", "load", "pycalphad_load", f"pycalphad: {e}", [e.split(":")[0]] if ":" in e else [])
    return r


# ---------------------------------------------------------------------------
# pycalphad helpers
# ---------------------------------------------------------------------------
def _calc(db, comps, phase, model, T, points, output="GM") -> np.ndarray:
    from pycalphad import calculate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = calculate(db, comps, phase, T=np.asarray(T, float), P=101325, N=1, points=np.atleast_2d(points),
                      model={phase: model}, output=output)
    return np.asarray(getattr(r, output).values).reshape(len(np.atleast_1d(T)), -1)


def _calc_sampled(db, comps, phase, model, T, pdens: int = 200):
    """Sampled GM of a phase: returns (x_B, GM) with shape (nT, npts).

    Points with more than THRESHOLDS["vacancy_max_site_fraction"] vacancies on a sublattice that also holds a
    metal are dropped: per mole of atoms the ideal entropy of such nearly empty lattices diverges, which makes
    any model with VA on a substitutional sublattice look "stable" at high T (a vacancy-gas artefact).
    """
    from pycalphad import calculate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = calculate(db, comps, phase, T=np.asarray(T, float), P=101325, N=1, pdens=pdens, model={phase: model})
    g = np.asarray(r.GM.values).reshape(len(T), -1)
    comp_names = [str(c) for c in r.component.values]
    X = np.asarray(r.X.values)
    X = X.reshape(len(T), -1, len(comp_names))
    x = X[..., comp_names.index(comps[1])]
    metal_subl = [i for i, subl in enumerate(model.constituents)
                  if "VA" in {s.name for s in subl} and any(s.name not in INTERSTITIALS | {"VA"} for s in subl)]
    if metal_subl:
        Y = np.asarray(r.Y.values).reshape(len(T), g.shape[1], -1)
        cols = [j for j, sf in enumerate(model.site_fractions)
                if sf.species.name == "VA" and sf.sublattice_index in metal_subl]
        bad = (Y[0][:, cols] > THRESHOLDS["vacancy_max_site_fraction"]).any(axis=1)
        g = np.where(bad[None, :], np.nan, g)
    return x, g


def _endmember_point(model, el: str) -> Optional[np.ndarray]:
    picks = []
    for subl in model.constituents:
        names = {s.name for s in subl}
        if el in names:
            picks.append(el)
        elif "VA" in names:
            picks.append("VA")
        else:
            return None
    if all(p == "VA" for p in picks):
        return None
    return np.array([1.0 if picks[sf.sublattice_index] == sf.species.name else 0.0 for sf in model.site_fractions])


def _binary_points(model, a: str, b: str, x: np.ndarray) -> Optional[np.ndarray]:
    """Site fractions for a simple (A,B)(VA)... solution at X(B)=x, or None if not that simple."""
    mix = []
    for i, subl in enumerate(model.constituents):
        names = {s.name for s in subl}
        if names == {a, b}:
            mix.append(i)
        elif names != {"VA"}:
            return None
    if len(mix) != 1:
        return None
    k = mix[0]
    pts = np.zeros((len(x), len(model.site_fractions)))
    for j, sf in enumerate(model.site_fractions):
        if sf.sublattice_index == k:
            pts[:, j] = (1 - x) if sf.species.name == a else x
        else:
            pts[:, j] = 1.0
    return pts


def _lower_hull(x: np.ndarray, g: np.ndarray, tags: Optional[np.ndarray] = None):
    ok = np.isfinite(x) & np.isfinite(g)
    x, g = x[ok], g[ok]
    tags = tags[ok] if tags is not None else np.zeros(len(x), int)
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    xr = np.round(x, 9)
    order = np.lexsort((g, xr))
    xr, g, tags = xr[order], g[order], tags[order]
    first = np.r_[True, xr[1:] != xr[:-1]]       # min G for each distinct x (sorted by g within x)
    xr, g, tags = xr[first], g[first], tags[first]
    hull: List[int] = []
    for i in range(len(xr)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (xr[a] - xr[o]) * (g[i] - g[o]) - (g[a] - g[o]) * (xr[i] - xr[o])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return xr[hull], g[hull], tags[hull]


def _hull_on(xg, hx, hg, htags=None):
    val = np.interp(xg, hx, hg, left=np.nan, right=np.nan)
    if htags is None:
        return val, None
    seg = np.clip(np.searchsorted(hx, xg, side="right") - 1, 0, len(hx) - 1)
    seg2 = np.clip(seg + 1, 0, len(hx) - 1)
    return val, (htags[seg], htags[seg2])


def _ranges_of(mask: np.ndarray, vals: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous True runs of mask -> [(vals[start], vals[end])]."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((float(vals[start]), float(vals[i - 1])))
            start = None
    if start is not None:
        out.append((float(vals[start]), float(vals[len(mask) - 1])))
    return out


def _fmt_ranges(rs, unit=" K", fmt="{:.0f}") -> str:
    return ", ".join((fmt.format(a) + unit) if a == b else (fmt.format(a) + "-" + fmt.format(b) + unit)
                     for a, b in rs)


# ---------------------------------------------------------------------------
# 2. unaries
# ---------------------------------------------------------------------------
def compute_unaries(db, tdb: TDB, elements: Sequence[str], tmin: float, tmax: float, dT: float = 5.0,
                    deadline: float = float("inf")) -> Dict[str, Dict]:
    from pycalphad import Model
    from .diagrams import phases_for
    T = np.arange(tmin, tmax, dT)
    T = np.r_[T, tmax - 0.01]       # never exactly at an upper limit (some pycalphad Piecewise give 0 there)
    out: Dict[str, Dict] = {}
    for el in elements:
        if time.time() > deadline:
            out[el] = {"skipped": "time limit"}
            continue
        rec: Dict[str, Any] = {"T": T, "G": {}, "Cp": {}, "failed": {}, "solids": [], "liquids": [], "gases": []}
        try:
            phs = phases_for(db, [el])
        except Exception as e:  # noqa: BLE001
            rec["failed"]["*"] = str(e)
            phs = []
        # pycalphad drops a disordered phase that is the DIS_PART of an ordered one (the ordered phase describes
        # both states), so the SER phase may be represented by its ordered partner
        ref_decl = tdb.elements[el].ref_phase if el in tdb.elements else ""
        cands = [ref_decl] + sorted(o for o, d in _dis_part_of(tdb).items() if d == ref_decl)
        ref = next((c for c in cands if c in phs), ref_decl)
        rec["ref"], rec["ref_declared"] = ref, ref_decl
        for ph in phs:
            if _is_gas(tdb, ph):
                rec["gases"].append(ph)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = Model(db, [el, "VA"], ph)
                pt = _endmember_point(mod, el)
                if pt is None:
                    continue
                g = _calc(db, [el, "VA"], ph, mod, T, pt, "GM")[:, 0]
                cp = _calc(db, [el, "VA"], ph, mod, T, pt, "CPM")[:, 0]
                if ph == ref:
                    g2 = _calc(db, [el, "VA"], ph, mod, [298.15, 298.65], pt, "GM")[:, 0]
                    S = -(g2[1] - g2[0]) / 0.5
                    rec["ref298"] = {"G": float(g2[0]), "S": float(S), "H": float(g2[0] + 298.15 * S)}
            except Exception as e:  # noqa: BLE001
                rec["failed"][ph] = f"{type(e).__name__}: {e}"[:200]
                continue
            if not np.all(np.isfinite(g)):
                rec["failed"][ph] = "non-finite G"
                continue
            if np.allclose(g, 0.0, atol=1e-6) and not _is_liquid(tdb, ph):
                rec.setdefault("zero", []).append(ph)      # no parameter for this end-member (pycalphad: G = 0)
                continue
            rec["G"][ph], rec["Cp"][ph] = g, cp
            (rec["liquids"] if _is_liquid(tdb, ph) else rec["solids"]).append(ph)
        r298 = rec.get("ref298")
        # "absolute": G contains the absolute entropy (S298 > 0), so Cp and melting data are meaningful;
        # "ser": additionally H(298.15) = 0 as the SER convention requires
        rec["absolute"] = (r298["S"] > 1.0) if r298 else None
        rec["ser"] = (rec["absolute"] and abs(r298["H"]) < THRESHOLDS["ser_H298_tol_J"]) if r298 else None
        # melting point & re-stabilisation
        rec["Tm"] = None
        if rec["solids"]:
            # most stable solid; ties (ordered/disordered twins, A1_FCC/FCC_A1...) resolved in favour of the
            # reference phase, then of the phase with fewest sublattices
            pref = sorted(rec["solids"], key=lambda p: (p != ref, tdb.phases[p].n_sublattices
                                                        if p in tdb.phases else 9, p))
            gs = np.vstack([rec["G"][p] for p in pref])
            gsmin = gs.min(axis=0)
            first = np.argmax(gs <= gsmin[None, :] + 0.5, axis=0)
            rec["min_solid"] = [pref[i] for i in first]
            rec["G_solid_min"] = gsmin
        if rec["solids"] and rec["liquids"]:
            gl = np.vstack([rec["G"][p] for p in rec["liquids"]]).min(axis=0)
            d = gl - gsmin                              # > 0: solid more stable
            rec["dG_liq_solid"] = d
            cross = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
            if d[0] <= 0:
                rec["Tm"] = None
                rec["liquid_at_tmin"] = True
            if len(cross):
                i = cross[0]
                rec["Tm"] = float(T[i] + (T[i + 1] - T[i]) * d[i] / (d[i] - d[i + 1]))
        out[el] = rec
    return out


def check_unaries(tdb: TDB, U: Dict[str, Dict], F: Findings) -> None:
    th = THRESHOLDS
    zero = {el: rec["zero"] for el, rec in U.items() if rec.get("zero")}
    if zero:
        F.add("INFO", "unary", "missing_endmember", "pure-element end-members with G identically 0 (no parameter, or "
              "+0: e.g. an ordering-only description without DIS_PART), ignored as far above the stable phase: "
              + "; ".join(f"{el}: {', '.join(v)}" for el, v in zero.items()),
              sorted({p for v in zero.values() for p in v}), endmembers=zero)
    for el, rec in U.items():
        if "skipped" in rec:
            F.add("INFO", "unary", "skipped", f"{el}: unary checks skipped ({rec['skipped']})", [el])
            continue
        for ph, why in rec["failed"].items():
            F.add("WARNING", "unary", "eval_failed", f"{el}: pure {ph} could not be evaluated: {why}", [ph])
        T = rec["T"]
        elem = tdb.elements.get(el)
        ref = rec.get("ref", "")
        allowed = {el, "VA"}
        G = rec["G"]
        # --- SER reference
        ser = None
        r298 = rec.get("ref298")
        if rec.get("ref_declared") and rec["ref_declared"] not in tdb.phases:
            F.add("INFO", "unary", "ref_phase_missing", f"{el}: ELEMENT reference phase {rec['ref_declared']} is not "
                  "a phase of the database", [f"ELEMENT {el}"])
        absolute = rec.get("absolute")
        if r298:
            ser = rec["ser"]
            if absolute and not ser:
                F.add("WARNING", "unary", "ser_offset", f"{el}: the reference phase {ref} has H(298.15) = "
                      f"{r298['H']:.5g} J/mol instead of 0 (S(298.15) = {r298['S']:.4g} J/mol/K): its G is offset "
                      "from the SER convention", _with_functions(tdb, _phase_params(tdb, ref, allowed)),
                      H298=r298["H"], S298=r298["S"])
            elif not absolute:
                F.add("INFO", "unary", "non_ser_reference", f"{el}: G of the reference phase {ref} is not SER-based "
                      f"(H(298.15) = {r298['H']:.4g} J/mol, S(298.15) = {r298['S']:.4g} J/mol/K; SER needs H = 0, "
                      "S > 0); Cp and SGTE melting-point comparisons skipped",
                      _with_functions(tdb, _phase_params(tdb, ref, allowed)), H298=r298["H"], S298=r298["S"])
            elif elem and elem.s298 > 0 and abs(elem.s298 - r298["S"]) > 1.0:
                F.add("INFO", "unary", "ser_entropy_mismatch", f"{el}: ELEMENT line S298 = {elem.s298:g} but "
                      f"-dG/dT of {ref} at 298.15 K = {r298['S']:.4g} J/mol/K", [f"ELEMENT {el}"],
                      S298_element=elem.s298, S298_G=r298["S"])
        # --- reference phase most stable solid at 298.15 K (first T point)
        solids = rec["solids"]
        if ref in G and solids:
            i0 = 0
            gref = G[ref][i0]
            better = [(p, G[p][i0] - gref) for p in solids if G[p][i0] < gref - 1.0]
            if better:
                sev = "ERROR" if absolute else "WARNING"
                p, dg = min(better, key=lambda t: t[1])
                F.add(sev, "unary", "ref_phase_not_stable", f"{el}: {p} is {-dg:.4g} J/mol more stable than the "
                      f"reference phase {ref} at {T[i0]:g} K" + (f" (also: {', '.join(q for q, _ in better if q != p)})"
                                                                 if len(better) > 1 else ""),
                      _with_functions(tdb, _phase_params(tdb, p, allowed) + _phase_params(tdb, ref, allowed)),
                      T=T[i0], dG_J=dg, phase=p, ref_phase=ref)
        # --- melting
        if rec["liquids"] and solids:
            d = rec["dG_liq_solid"]
            liq = rec["liquids"][0]
            liq_params = _phase_params(tdb, liq, allowed)
            tm = rec["Tm"]
            if rec.get("liquid_at_tmin"):
                if not (ref and _is_liquid(tdb, ref)):
                    F.add("ERROR", "unary", "liquid_stable_at_tmin", f"{el}: liquid is more stable than every solid "
                          f"at {T[0]:g} K (by {-d[0]:.4g} J/mol)", _with_functions(tdb, liq_params), dG_J=d[0])
            elif tm is None:
                F.add("ERROR", "unary", "no_melting", f"{el}: the liquid never becomes stable up to {T[-1]:g} K "
                      f"(G(liquid) - G(most stable solid) = {d[-1]:+.4g} J/mol at {T[-1]:g} K, stable solid "
                      f"{rec['min_solid'][-1]})",
                      _with_functions(tdb, liq_params + _phase_params(tdb, rec["min_solid"][-1], allowed)),
                      dG_at_tmax_J=d[-1], solid=rec["min_solid"][-1])
            if tm is not None:
                after = T > tm + 1.0
                bad = after & (d > th["restabilisation_tol_J"])
                if bad.any():
                    t_first = float(T[np.where(bad)[0].min()])
                    phs = sorted({rec["min_solid"][i] for i in np.where(bad)[0]})
                    per = {}
                    for p in phs:
                        dp = G[liq] - G[p] if liq in G else d
                        m = after & (dp > th["restabilisation_tol_J"])
                        if m.any():
                            per[p] = {"T_ranges": _ranges_of(m, T), "max_dG_J": float(np.max(dp[m]))}
                    sev = "ERROR" if t_first <= tm + th["high_T_error_margin_K"] else "WARNING"
                    F.add(sev, "unary", "inverse_melting",
                          f"{el}: solid re-stabilised above Tm = {tm:.1f} K: " + "; ".join(
                              f"{p} below liquid at {_fmt_ranges(v['T_ranges'])} (by up to {v['max_dG_J']:.4g} J/mol)"
                              for p, v in per.items()),
                          _with_functions(tdb, [n for p in phs for n in _phase_params(tdb, p, allowed)] + liq_params),
                          Tm=tm, phases=per)
                ref_tm = SGTE_TM.get(el)
                stable_solid = rec["min_solid"][int(np.searchsorted(T, tm)) - 1]
                if absolute and ref_tm is not None and abs(tm - ref_tm) > th["tm_mismatch_K"]:
                    F.add("WARNING", "unary", "tm_mismatch", f"{el}: computed Tm = {tm:.2f} K ({stable_solid} -> "
                          f"liquid) differs from the SGTE value {ref_tm:g} K by {tm - ref_tm:+.1f} K",
                          _with_functions(tdb, liq_params + _phase_params(tdb, stable_solid, allowed)),
                          Tm=tm, Tm_SGTE=ref_tm, solid=stable_solid)
                else:
                    extra = (f" (SGTE {ref_tm:g} K)" if ref_tm else "")
                    F.add("INFO", "unary", "tm_info", f"{el}: Tm = {tm:.2f} K ({stable_solid} -> liquid){extra}",
                          [], Tm=tm, Tm_SGTE=ref_tm, solid=stable_solid)
        # --- Cp
        if absolute is False:
            continue
        if not G:
            continue
        gall = np.vstack([G[p] for p in G]).min(axis=0)
        for p in G:
            cp = rec["Cp"][p]
            above = G[p] - gall
            for kind, mask in (("negative_cp", cp < th["cp_min_J_K"] - 1e-6), ("huge_cp", cp > th["cp_max_J_K"])):
                if not mask.any():
                    continue
                stable = mask & (above < 1.0)
                near = mask & (above < 20000.0)
                sev = "ERROR" if (kind == "negative_cp" and stable.any()) else \
                      "WARNING" if (stable.any() or (kind == "negative_cp" and near.any())) else "INFO"
                ext = float(cp[mask].min() if kind == "negative_cp" else cp[mask].max())
                where = _fmt_ranges(_ranges_of(mask, T))
                status = ("stable there" if stable.any() else
                          f"metastable, >= {float(above[mask].min()):.3g} J/mol above the stable phase")
                F.add(sev, "unary", kind, f"{el}: Cp of pure {p} is {'negative' if kind == 'negative_cp' else '> ' + str(int(th['cp_max_J_K']))} "
                      f"J/mol/K at {where} ({'min' if kind == 'negative_cp' else 'max'} {ext:.4g} J/mol/K; {status})",
                      _with_functions(tdb, _phase_params(tdb, p, allowed)), phase=p, extreme_Cp=ext,
                      T_ranges=_ranges_of(mask, T), stable=bool(stable.any()))


# ---------------------------------------------------------------------------
# 3a. interaction parameters
# ---------------------------------------------------------------------------
def _atoms_per_formula(tdb: TDB, phase: str) -> float:
    ph = tdb.phases.get(phase)
    if ph is None or not ph.constituents:
        return 1.0
    n = sum(r for r, sl in zip(ph.site_ratios, ph.constituents) if sl != ["VA"])
    return n if n > 0 else 1.0


def check_interactions(tdb: TDB, elements: Sequence[str], tmin: float, tmax: float, F: Findings) -> None:
    th = THRESHOLDS
    ext = _extended_tdb(tdb)
    allowed = set(elements) | {"VA", "*"}
    T = np.linspace(tmin, tmax - 0.01, 300)
    groups: Dict[Tuple, Dict[int, List]] = {}
    for p in tdb.parameters:
        if p.key.ptype != "G" or p.key.phase not in tdb.phases:
            continue
        cons = p.key.constituents
        inter = [i for i, sl in enumerate(cons) if len(sl) > 1]
        if not inter:
            continue
        species = {s for sl in cons for s in sl}
        if not species <= allowed:
            continue
        if any("VA" in cons[i] for i in inter):
            continue       # vacancy parameters (e.g. L(BCC,X,VA:VA)=+100*T) are large by convention
        try:
            v = ext.evaluate(p.expr, T) * p.sign
        except ExpressionError:
            continue
        if not np.all(np.isfinite(v)):
            continue
        atoms = _atoms_per_formula(tdb, p.key.phase)
        va = v / atoms
        imax = int(np.argmax(np.abs(va)))
        if abs(va[imax]) > th["L_max_J_per_mol_atoms"]:
            F.add("WARNING", "interactions", "large_interaction", f"{p.raw_name}: |L| = {abs(v[imax]) / 1000:.4g} "
                  f"kJ/mol of formula ({abs(va[imax]) / 1000:.4g} kJ/mol of atoms) at {T[imax]:.0f} K",
                  [p.raw_name], value_J=v[imax], per_atom_J=va[imax], T=T[imax])
        bps = _breakpoints(tdb, p.expr)
        dv = np.gradient(va, T)
        okm = np.ones_like(T, bool)
        for bp in bps:
            okm &= np.abs(T - bp) > 2 * (T[1] - T[0])
        dv = np.where(okm, dv, np.nan)
        if np.nanmax(np.abs(dv)) > th["dLdT_max_J_K"]:
            j = int(np.nanargmax(np.abs(dv)))
            F.add("WARNING", "interactions", "large_tcoef", f"{p.raw_name}: dL/dT = {dv[j]:+.4g} J/mol/K (per mole of "
                  f"atoms) at {T[j]:.0f} K", [p.raw_name], dLdT=dv[j], T=T[j])
        if len(inter) == 1 and len(cons[inter[0]]) == 2 and all(len(sl) == 1 for i, sl in enumerate(cons)
                                                                 if i != inter[0]):
            gkey = (p.key.phase, inter[0], cons)
            groups.setdefault(gkey, {}).setdefault(p.key.order, []).append((p, va, bps))
    # excess Cp of binary interactions: Cp_xs = y(1-y) sum_k Cp_Lk (1-2y)^k
    y = np.linspace(0.05, 0.95, 19)
    for (phase, _i, cons), orders in groups.items():
        cpx = np.zeros((len(T), len(y)))
        names, bpall = [], set()
        for k, lst in orders.items():
            for p, va, bps in lst:
                d2 = np.gradient(np.gradient(va, T), T)
                cpx += (-T * d2)[:, None] * (y * (1 - y) * (1 - 2 * y) ** k)[None, :]
                names.append(p.raw_name)
                bpall |= set(bps)
        okm = np.ones_like(T, bool)
        for bp in bpall:
            okm &= np.abs(T - bp) > 3 * (T[1] - T[0])
        cpx[~okm] = np.nan
        if not np.any(np.isfinite(cpx)):
            continue
        idx = np.unravel_index(np.nanargmax(np.abs(cpx)), cpx.shape)
        mx = cpx[idx]
        if abs(mx) > th["excess_cp_max_J_K"]:
            F.add("WARNING", "interactions", "excess_cp", f"{phase} {':'.join(','.join(s) for s in cons)}: excess Cp "
                  f"from the interaction parameters reaches {mx:+.4g} J/mol-atoms/K (y = {y[idx[1]]:.2f}, "
                  f"T = {T[idx[0]]:.0f} K)", names, excess_cp=mx, T=T[idx[0]], y=y[idx[1]])


# ---------------------------------------------------------------------------
# 3b. binaries
# ---------------------------------------------------------------------------
def _binary_params(tdb: TDB, phase: str, a: str, b: str) -> List[str]:
    """Interaction parameters of ``phase`` (and its DIS_PART) mixing a and b."""
    phases = {phase}
    dp = _dis_part_of(tdb).get(phase)
    if dp:
        phases.add(dp)
    out = []
    for p in tdb.parameters:
        if p.key.phase in phases and p.key.ptype == "G":
            sp = {s for sl in p.key.constituents for s in sl}
            if {a, b} <= sp and sp <= {a, b, "VA", "*"}:
                out.append(p.raw_name)
    return out


def compute_binary(db, tdb: TDB, a: str, b: str, U: Dict[str, Dict], tmin: float, tmax: float,
                   deadline: float, nT_map: int = 36, pdens: int = 300) -> Dict:
    from pycalphad import Model
    from .diagrams import phases_for
    th = THRESHOLDS
    t0 = time.time()
    res: Dict[str, Any] = {"name": f"{a}-{b}", "a": a, "b": b, "skipped": [], "failed": {}}
    tms = []
    for el in (a, b):
        tm = (U.get(el) or {}).get("Tm")
        tms.append(tm if tm else SGTE_TM.get(el, 2000.0))
    res["Tm"] = tms
    res["ser"] = all((U.get(el) or {}).get("absolute") for el in (a, b))
    t_hot = max(tms) + th["high_T_margin_K"]
    res["T_hot"] = t_hot
    comps = [a, b, "VA"]
    phs = [p for p in phases_for(db, [a, b]) if not _is_gas(tdb, p)]
    liqs = [p for p in phs if _is_liquid(tdb, p)]
    if not liqs:
        res["skipped"].append("no liquid phase")
        return res
    t_lo = max(tmin, 0.5 * min(tms))
    t_top = tmax - 0.01
    Tm_grid = np.linspace(t_lo, t_top, nT_map)
    if t_hot < t_top:
        Tm_grid = np.unique(np.r_[Tm_grid, np.linspace(t_hot, t_top, 6)])
    res["T"] = Tm_grid
    xg = np.linspace(0, 1, 201)
    res["x"] = xg
    models = {}
    for p in phs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models[p] = Model(db, comps, p)
        except Exception as e:  # noqa: BLE001
            res["failed"][p] = f"{type(e).__name__}: {e}"[:200]
    liq = liqs[0]
    if liq not in models:
        res["skipped"].append(f"{liq} model failed")
        return res
    # --- exact liquid G(x) on the map grid
    lpts = _binary_points(models[liq], a, b, xg)
    if lpts is not None:
        gl = _calc(db, comps, liq, models[liq], Tm_grid, lpts, "GM")
        res["G_liq"] = gl
        d2 = np.full_like(gl, np.nan)
        dx = xg[1] - xg[0]
        d2[:, 1:-1] = (gl[:, 2:] - 2 * gl[:, 1:-1] + gl[:, :-2]) / dx ** 2
        res["stab"] = d2 * (xg * (1 - xg))[None, :] / (R_GAS * Tm_grid[:, None])
        liq_x = np.broadcast_to(xg, gl.shape)
        liq_g = gl
    else:
        res["skipped"].append(f"{liq} is not a simple (A,B) substitutional liquid: curvature map skipped, "
                              "sampled liquid used")
        liq_x, liq_g = _calc_sampled(db, comps, liq, models[liq], Tm_grid, pdens)
    # liquid binodal (gap) at each T from the hull of the liquid alone
    gap = np.zeros((len(Tm_grid), len(xg)), bool)
    liq_hull = np.full((len(Tm_grid), len(xg)), np.nan)
    for i in range(len(Tm_grid)):
        hx, hg, _ = _lower_hull(liq_x[i], liq_g[i])
        liq_hull[i] = _hull_on(xg, hx, hg)[0]
        if lpts is not None:
            gap[i] = (liq_g[i] - liq_hull[i]) > th["hull_tol_J"]
    res["liq_gap"] = gap
    # --- solids vs liquid (sampled solids)
    dfor = np.full((len(Tm_grid), len(xg)), np.nan)       # hull_liq - hull_all  (>0: something beats liquid)
    culprit = np.full((len(Tm_grid), len(xg)), "", dtype=object)
    all_x, all_g, all_t = [liq_x], [liq_g], [np.zeros(liq_x.shape, int)]
    names = [liq]
    for p in phs:
        if p == liq or p not in models:
            continue
        if time.time() > deadline:
            res["skipped"].append(f"time limit: phases from {p} on not sampled")
            break
        try:
            sx, sg = _calc_sampled(db, comps, p, models[p], Tm_grid, pdens)
        except Exception as e:  # noqa: BLE001
            res["failed"][p] = f"{type(e).__name__}: {e}"[:200]
            continue
        names.append(p)
        all_x.append(sx)
        all_g.append(sg)
        all_t.append(np.full(sx.shape, len(names) - 1, int))
    res["phase_names"] = names
    X = np.concatenate(all_x, axis=1)
    Gs = np.concatenate(all_g, axis=1)
    Tg = np.concatenate(all_t, axis=1)
    for i in range(len(Tm_grid)):
        hx, hg, ht = _lower_hull(X[i], Gs[i], Tg[i])
        h, (t1, t2) = _hull_on(xg, hx, hg, ht)
        dfor[i] = liq_hull[i] - h
        for j in range(len(xg)):
            if dfor[i, j] > th["hull_tol_J"]:
                culprit[i, j] = "+".join(sorted({names[t] for t in (t1[j], t2[j]) if t != 0}))
    res["dG_solid"] = dfor
    res["culprit"] = culprit
    # --- simple solution phases: high-T gaps
    sol_gaps = {}
    hot = Tm_grid >= t_hot - 1e-6
    if hot.any():
        for p in phs:
            if p == liq or p not in models or _is_liquid(tdb, p):
                continue
            pts = _binary_points(models[p], a, b, xg)
            if pts is None:
                continue
            try:
                gp = _calc(db, comps, p, models[p], Tm_grid[hot], pts, "GM")
            except Exception:  # noqa: BLE001
                continue
            gmask = np.zeros(gp.shape, bool)
            for i in range(gp.shape[0]):
                hx, hg, _ = _lower_hull(xg, gp[i])
                gmask[i] = (gp[i] - _hull_on(xg, hx, hg)[0]) > th["hull_tol_J"]
            if gmask.any():
                sol_gaps[p] = {"T": Tm_grid[hot], "gap": gmask}
    res["solution_gaps"] = sol_gaps
    # --- liquid Cp at high T
    if lpts is not None and time.time() < deadline:
        Tc = np.linspace(max(tms), t_top, 12)
        xc = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        try:
            cp = _calc(db, comps, liq, models[liq], Tc, _binary_points(models[liq], a, b, xc), "CPM")
            ideal = (1 - xc)[None, :] * cp[:, [0]] + xc[None, :] * cp[:, [-1]]
            res["liq_cp"] = {"T": Tc, "x": xc, "cp": cp, "cp_xs": cp - ideal}
        except Exception as e:  # noqa: BLE001
            res["skipped"].append(f"liquid Cp: {e}")
    res["seconds"] = round(time.time() - t0, 2)
    return res


def check_binary(tdb: TDB, res: Dict, tmax: float, F: Findings) -> None:
    th = THRESHOLDS
    a, b, name = res["a"], res["b"], res["name"]
    for s in res.get("skipped", []):
        F.add("INFO", "binary", "skipped", f"{name}: {s}", [])
    for p, why in res.get("failed", {}).items():
        F.add("WARNING", "binary", "eval_failed", f"{name}: {p} could not be evaluated: {why}", [p])
    if "T" not in res or "dG_solid" not in res:
        return
    T, xg, t_hot = res["T"], res["x"], res["T_hot"]
    liq = res["phase_names"][0]
    liq_params = _binary_params(tdb, liq, a, b)
    # (a)/(c) solid phases stable over liquid far above melting
    hot = T >= t_hot - 1e-6
    if hot.any():
        bad = (res["dG_solid"] > th["hull_tol_J"]) & hot[:, None]
        if bad.any():
            per: Dict[str, Dict] = {}
            for i, j in zip(*np.where(bad)):
                for p in res["culprit"][i, j].split("+"):
                    if not p:
                        continue
                    d = per.setdefault(p, {"T": set(), "x": [], "max_dG": 0.0})
                    d["T"].add(float(T[i]))
                    d["x"].append(float(xg[j]))
                    d["max_dG"] = max(d["max_dG"], float(res["dG_solid"][i, j]))
            t_first = float(T[np.where(bad.any(axis=1))[0].min()])
            t_err = max(res["Tm"]) + th["high_T_error_margin_K"]
            sev = "ERROR" if t_first <= t_err else "WARNING"
            msg = "; ".join(f"{p} at T = {min(v['T']):.0f}-{max(v['T']):.0f} K, X({b}) = {min(v['x']):.2f}-"
                            f"{max(v['x']):.2f} (liquid higher by up to {v['max_dG']:.4g} J/mol)" for p, v in per.items())
            resp = []
            for p in per:
                resp += _phase_params(tdb, p, {a, b, "VA"})
            F.add(sev, "binary", "binary_solid_above_liquid",
                  f"{name}: solid phase(s) more stable than the liquid above max(Tm)+{th['high_T_margin_K']:.0f} K "
                  f"= {t_hot:.0f} K: {msg}", _with_functions(tdb, resp[:12]) + liq_params,
                  phases={p: {"T_min": min(v["T"]), "T_max": max(v["T"]), "x_min": min(v["x"]), "x_max": max(v["x"]),
                              "max_dG_J": v["max_dG"]} for p, v in per.items()}, T_start=t_hot)
    # (b) liquid miscibility gap
    gap = res.get("liq_gap")
    if gap is not None and gap.any():
        rows = np.where(gap.any(axis=1))[0]
        tg = T[rows]
        xs = xg[np.where(gap.any(axis=0))[0]]
        tc = float(tg.max())
        has = gap.any(axis=1)
        at_top = bool(has[-1])
        t_err = max(res["Tm"]) + th["high_T_error_margin_K"]
        inverted = False
        if at_top:
            start = len(has) - 1
            while start > 0 and has[start - 1]:
                start -= 1
            inverted = start > 0
            t_open = float(T[start])
            sev = "ERROR" if t_open <= t_err else "WARNING"
            what = (f"inverted liquid miscibility gap (opens on heating at about {t_open:.0f} K and persists to "
                    f"tmax = {T[-1]:.0f} K)" if inverted else
                    f"liquid miscibility gap at all T from {t_open:.0f} K up to tmax = {T[-1]:.0f} K")
            if sev == "WARNING":
                what += f" - only far above the melting points (> max(Tm)+{th['high_T_error_margin_K']:.0f} K)"
        elif tc >= t_hot:
            sev, what = "WARNING", (f"liquid miscibility gap up to about {tc:.0f} K (> max(Tm)+"
                                    f"{th['high_T_margin_K']:.0f} K); fine only for a real monotectic system")
        else:
            sev, what = "INFO", f"liquid miscibility gap below about {tc:.0f} K"
        code = "liquid_miscibility_gap" if sev != "INFO" else "liquid_gap_low_T"
        ext = _extended_tdb(tdb)
        vals = {}
        byname = {p.raw_name: p for p in tdb.parameters}
        for n in liq_params:
            try:
                vals[n] = float(ext.evaluate(byname[n].expr, [tc])[0])
            except (ExpressionError, KeyError):
                pass
        F.add(sev, "binary", code, f"{name}: {what}; X({b}) = {xs.min():.2f}-{xs.max():.2f}. Liquid "
              f"parameters at {tc:.0f} K: " + ", ".join(f"{k} = {v:.4g}" for k, v in vals.items()),
              liq_params, T_gap_min=float(tg.min()), T_gap_max=tc, x_min=float(xs.min()), x_max=float(xs.max()),
              inverted=inverted, L_values=vals)
    # solution phases
    for p, d in res.get("solution_gaps", {}).items():
        rows = np.where(d["gap"].any(axis=1))[0]
        xs = xg[np.where(d["gap"].any(axis=0))[0]]
        F.add("WARNING", "binary", "solution_gap_high_T", f"{name}: {p} has a miscibility gap at "
              f"T = {d['T'][rows].min():.0f}-{d['T'][rows].max():.0f} K (above max(Tm)+"
              f"{th['high_T_margin_K']:.0f} K), X({b}) = {xs.min():.2f}-{xs.max():.2f}",
              _binary_params(tdb, p, a, b), T_max=float(d["T"][rows].max()))
    # (e) liquid Cp
    lc = res.get("liq_cp")
    if lc:
        cp, xs_, Tc = lc["cp"], lc["x"], lc["T"]
        inner = cp[:, 1:-1]
        if res.get("ser") and np.nanmin(inner) < 0:      # absolute Cp only meaningful for SER-based unaries
            i, j = np.unravel_index(np.nanargmin(inner), inner.shape)
            F.add("ERROR", "binary", "liquid_negative_cp", f"{name}: Cp of {liq} is negative ({inner[i, j]:.4g} "
                  f"J/mol/K) at X({b}) = {xs_[j + 1]:.2f}, T = {Tc[i]:.0f} K", liq_params, cp=inner[i, j], T=Tc[i])
        xs_cp = lc["cp_xs"][:, 1:-1]
        if np.nanmax(np.abs(xs_cp)) > th["liquid_cp_xs_max_J_K"]:
            i, j = np.unravel_index(np.nanargmax(np.abs(xs_cp)), xs_cp.shape)
            F.add("WARNING", "binary", "liquid_excess_cp", f"{name}: excess Cp of {liq} reaches {xs_cp[i, j]:+.4g} "
                  f"J/mol/K at X({b}) = {xs_[j + 1]:.2f}, T = {Tc[i]:.0f} K", liq_params, cp_xs=xs_cp[i, j], T=Tc[i])


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def _grid(n: int, w: float = 4.2, h: float = 3.3, ncols: int = 3):
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()[:n]


def plot_unary_G(tdb: TDB, U: Dict, out_png: str) -> Optional[str]:
    els = [e for e, r in U.items() if r.get("G")]
    if not els:
        return None
    fig, axes = _grid(len(els))
    for ax, el in zip(axes, els):
        r = U[el]
        T = r["T"]
        ref = r.get("ref", "")
        base = r["G"].get(ref)
        if base is None:
            base = np.vstack(list(r["G"].values())).min(axis=0)
            ref = "min(G)"
        vals = []
        for p, g in r["G"].items():
            d = (g - base) / 1000.0
            vals.append(d)
            isl = p in r["liquids"]
            ax.plot(T, d, lw=2.2 if isl else 1.2, color="k" if isl else None,
                    label=p, zorder=3 if isl else 2)
        ax.axhline(0, color="0.5", lw=0.8)
        if r.get("Tm"):
            ax.axvline(r["Tm"], color="tab:blue", ls="--", lw=1, label=f"Tm={r['Tm']:.0f} K")
        if el in SGTE_TM:
            ax.axvline(SGTE_TM[el], color="tab:red", ls=":", lw=1, label=f"SGTE {SGTE_TM[el]:g} K")
        allv = np.concatenate(vals)
        lo, hi = np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)
        ax.set_ylim(max(lo, -40) - 2, min(hi, 60) + 2)
        ax.set_title(f"{el}: G(phase) - G({ref})", fontsize=10)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("ΔG (kJ/mol)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(f"{tdb.name}: pure-element Gibbs energies relative to the SER phase", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_unary_cp(tdb: TDB, U: Dict, out_png: str) -> Optional[str]:
    els = [e for e, r in U.items() if r.get("Cp")]
    if not els:
        return None
    fig, axes = _grid(len(els))
    for ax, el in zip(axes, els):
        r = U[el]
        T = r["T"]
        for p, cp in r["Cp"].items():
            isl = p in r["liquids"]
            ax.plot(T, cp, lw=2.2 if isl else 1.2, color="k" if isl else None, label=p)
        ax.axhline(0, color="tab:red", lw=0.8)
        ax.axhline(THRESHOLDS["cp_max_J_K"], color="tab:red", lw=0.8, ls=":")
        ax.set_ylim(-20, 120)
        if r.get("Tm"):
            ax.axvline(r["Tm"], color="tab:blue", ls="--", lw=1)
        ttl = f"{el}: Cp of pure-element end-members"
        if r.get("absolute") is False:
            ttl += " (non-SER!)"
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("Cp (J/mol/K)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(f"{tdb.name}: heat capacities (-T d²G/dT²)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_binary_liquid(B: List[Dict], out_png: str, title: str) -> Optional[str]:
    bs = [r for r in B if "dG_solid" in r]
    if not bs:
        return None
    fig, axes = _grid(len(bs))
    for ax, r in zip(axes, bs):
        T, x = r["T"], r["x"]
        d = np.clip(np.nan_to_num(r["dG_solid"], nan=0.0), 0, None) / 1000.0
        hot = (T >= r["T_hot"] - 1e-6)[:, None]
        low = np.ma.masked_where(hot | (d <= 0), d)
        high = np.ma.masked_where(~hot | (d <= 0), d)
        ax.pcolormesh(x, T, low, cmap="Greys", vmin=0, vmax=max(1e-3, float(np.max(d[~hot[:, 0]], initial=0))),
                      shading="nearest", alpha=0.5)
        vmax = max(1.0, float(np.max(d[hot[:, 0]], initial=0)))
        pc = ax.pcolormesh(x, T, high, cmap="Reds", vmin=0, vmax=vmax, shading="nearest")
        fig.colorbar(pc, ax=ax, label="above margin: G(liq) - G(hull) (kJ/mol)")
        if (d > 0).any() and (d == 0).any():
            ax.contour(x, T, (d > THRESHOLDS["hull_tol_J"] / 1000).astype(float), levels=[0.5], colors="k",
                       linewidths=0.8)
        gap = r.get("liq_gap")
        if gap is not None and gap.any():
            ii, jj = np.where(gap)
            ax.scatter(x[jj], T[ii], s=4, color="tab:blue", label="liquid gap")
            ax.legend(fontsize=7, loc="lower right")
        if r["T_hot"] < T[-1]:
            ax.axhline(r["T_hot"], color="tab:green", ls="--", lw=1.2)
            ax.text(0.01, r["T_hot"], " max(Tm)+margin", color="tab:green", fontsize=7, va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(T[0], T[-1])
        ax.set_title(f"{r['name']}: liquid not stable (grey: normal, red: suspicious)", fontsize=9)
        ax.set_xlabel(f"X({r['b']})")
        ax.set_ylabel("T (K)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_binary_d2g(B: List[Dict], out_png: str, title: str) -> Optional[str]:
    bs = [r for r in B if r.get("stab") is not None]
    if not bs:
        return None
    fig, axes = _grid(len(bs))
    for ax, r in zip(axes, bs):
        T, x, s = r["T"], r["x"], r["stab"]
        pc = ax.pcolormesh(x[1:-1], T, np.clip(s[:, 1:-1], -2, 2), cmap="RdBu", norm=TwoSlopeNorm(0, -2, 2),
                           shading="nearest")
        fig.colorbar(pc, ax=ax, label="x(1-x)/RT · d²G/dx²")
        if np.nanmin(s[:, 1:-1]) < 0 < np.nanmax(s[:, 1:-1]):
            ax.contour(x[1:-1], T, s[:, 1:-1], levels=[0.0], colors="k", linewidths=1.0)
        if r["T_hot"] < T[-1]:
            ax.axhline(r["T_hot"], color="tab:green", ls="--", lw=1.2)
        ax.set_title(f"{r['name']}: GM(LIQUID) curvature (<0: spinodal)", fontsize=9)
        ax.set_xlabel(f"X({r['b']})")
        ax.set_ylabel("T (K)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def default_tmax(tdb: TDB) -> float:
    return float(min(6000.0, max(3000.0, tdb.temp_limits[1] if tdb.temp_limits else 6000.0)))


def format_report(tdb_path: str, F: Sequence[Finding], meta: Dict, max_per_sev: int = 60) -> str:
    lines = [f"TDB check: {tdb_path}",
             f"elements {meta['elements']}, T = {meta['tmin']:g}-{meta['tmax']:g} K, checks {meta['checks']}, "
             f"{meta['seconds']:.1f} s",
             "counts: " + ", ".join(f"{s} {meta['counts'][s]}" for s in SEVERITIES)]
    if meta.get("melting_points"):
        lines.append("melting points: " + ", ".join(
            f"{e} {v['Tm']:.1f} K" + (f" (SGTE {v['SGTE']:g})" if v.get("SGTE") else "") if v.get("Tm") else f"{e} -"
            for e, v in meta["melting_points"].items()))
    shown_fix = set()
    for sev in SEVERITIES:
        items = [f for f in F if f.severity == sev]
        if not items:
            continue
        lines.append("")
        lines.append(f"== {sev} ({len(items)}) ==")
        for f in items[:max_per_sev]:
            lines.append(f"[{sev}] {f.check}/{f.code}: {f.message}")
            if f.responsible:
                lines.append(f"    responsible: {_fmt_list(f.responsible, 8)}")
            if f.fix and (f.code not in shown_fix) and f.code not in ("tm_info",):
                lines.append(f"    -> {f.fix}")
                shown_fix.add(f.code)
        if len(items) > max_per_sev:
            lines.append(f"... {len(items) - max_per_sev} more {sev} findings in the JSON file")
    if meta.get("skipped"):
        lines.append("")
        lines.append("skipped: " + "; ".join(meta["skipped"]))
    return "\n".join(lines)


def run_checks(tdb_path: str, elements: Optional[Sequence[str]] = None, checks: Optional[Sequence[str]] = None,
               tmin: float = 298.15, tmax: Optional[float] = None, output_prefix: Optional[str] = None,
               time_limit: float = 240.0, plots: bool = True, progress=None) -> Dict[str, Any]:
    """Run the checks; returns {"findings", "report", "json", "report_file", "images", "meta"}."""
    t_start = time.time()
    deadline = t_start + time_limit
    say = progress or (lambda m: None)
    checks = [c.lower() for c in (checks or ALL_CHECKS)]
    bad = [c for c in checks if c not in ALL_CHECKS]
    if bad:
        raise ValueError(f"unknown check(s) {bad}; valid: {list(ALL_CHECKS)}")
    text = Path(tdb_path).read_text(encoding="utf-8", errors="replace")
    tdb = parse_tdb(tdb_path)
    tmax = float(tmax) if tmax else default_tmax(tdb)
    tmin = float(tmin)
    all_els = tdb.real_elements
    if elements:
        els = [e.upper() for e in elements]
        unknown = [e for e in els if e not in all_els]
        if unknown:
            raise ValueError(f"element(s) {unknown} not in the database ({all_els})")
    else:
        els = all_els
    F = Findings()
    timings: Dict[str, float] = {}
    skipped: List[str] = []

    def timed(name, fn):
        t0 = time.time()
        say(f"check_tdb: {name}")
        fn()
        timings[name] = round(time.time() - t0, 2)

    if "syntax" in checks:
        timed("syntax", lambda: check_syntax(tdb, text, F))
    if "ranges" in checks:
        timed("ranges", lambda: check_ranges(tdb, tmin, tmax, F))
    if "breakpoints" in checks:
        timed("breakpoints", lambda: check_breakpoints(tdb, F))
    if "load" in checks:
        timed("load", lambda: check_load(tdb_path, F))
    if "interactions" in checks:
        timed("interactions", lambda: check_interactions(tdb, els, tmin, tmax, F))
    U: Dict[str, Dict] = {}
    B: List[Dict] = []
    db = None
    if "unary" in checks or "binary" in checks:
        try:
            from .calc import get_db
            db = get_db(tdb_path)
        except Exception as e:  # noqa: BLE001
            F.add("ERROR", "load", "pycalphad_load", f"pycalphad cannot load the database ({type(e).__name__}: "
                  f"{str(e)[:200]}); unary/binary checks skipped", [])
            skipped.append("unary/binary checks (database does not load)")
    if db is not None:
        t0 = time.time()
        say("check_tdb: unaries")
        U = compute_unaries(db, tdb, els, tmin, tmax, deadline=deadline)
        if "unary" in checks:
            check_unaries(tdb, U, F)
        timings["unary"] = round(time.time() - t0, 2)
        if "binary" in checks:
            t0 = time.time()
            pairs = list(combinations(els, 2))
            per_budget = max(15.0, (deadline - time.time()) / max(1, len(pairs)))
            for a, b in pairs:
                if time.time() > deadline:
                    skipped.append(f"binary {a}-{b} (time limit)")
                    F.add("INFO", "binary", "skipped", f"{a}-{b}: not checked (time limit {time_limit:g} s)", [])
                    continue
                say(f"check_tdb: binary {a}-{b}")
                try:
                    r = compute_binary(db, tdb, a, b, U, tmin, tmax, deadline=min(deadline, time.time() + per_budget))
                except Exception as e:  # noqa: BLE001
                    F.add("WARNING", "binary", "eval_failed", f"{a}-{b}: binary check failed: {type(e).__name__}: "
                          f"{str(e)[:200]}", [])
                    continue
                B.append(r)
                check_binary(tdb, r, tmax, F)
            if len(els) > 2:
                F.add("INFO", "binary", "skipped", "ternary and higher subsystems are not checked (binaries only)", [])
            timings["binary"] = round(time.time() - t0, 2)
    # ---- outputs
    order = {s: i for i, s in enumerate(SEVERITIES)}
    corder = {c: i for i, c in enumerate(ALL_CHECKS)}
    F.sort(key=lambda f: (order[f.severity], corder.get(f.check, 99)))
    counts = {s: sum(1 for f in F if f.severity == s) for s in SEVERITIES}
    mps = {}
    for el, r in U.items():
        if "T" in r:
            mps[el] = {"Tm": r.get("Tm"), "SGTE": SGTE_TM.get(el), "ser": r.get("ser"), "absolute": r.get("absolute")}
    meta = {"tdb": tdb_path, "elements": els, "tmin": tmin, "tmax": tmax, "checks": checks, "counts": counts,
            "seconds": round(time.time() - t_start, 1), "timings": timings, "skipped": skipped,
            "melting_points": mps, "thresholds": THRESHOLDS,
            "binary_seconds": {r["name"]: r.get("seconds") for r in B}}
    images: List[str] = []
    base = output_prefix or str(Path(tdb_path).with_suffix("")) + "_check"
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    if plots:
        stem = Path(tdb_path).name
        for fn, args in ((plot_unary_G, (tdb, U, base + "_unary_G.png")),
                         (plot_unary_cp, (tdb, U, base + "_unary_Cp.png")),
                         (plot_binary_liquid, (B, base + "_binary_liquid_stability.png",
                                               f"{stem}: where solid phases beat the liquid (black line: "
                                               "liquidus; blue dots: liquid miscibility gap)")),
                         (plot_binary_d2g, (B, base + "_binary_liquid_curvature.png",
                                            f"{stem}: liquid stability x(1-x)/RT d²G/dx² (ideal = 1)"))):
            try:
                p = fn(*args)
            except Exception as e:  # noqa: BLE001
                skipped.append(f"plot {fn.__name__}: {e}")
                p = None
            if p:
                images.append(p)
    meta["seconds"] = round(time.time() - t_start, 1)
    report = format_report(tdb_path, F, meta)
    jpath, rpath = base + "_findings.json", base + "_report.txt"
    Path(jpath).write_text(json.dumps({"meta": _clean(meta), "findings": [f.to_dict() for f in F]}, indent=1,
                                      default=str))
    Path(rpath).write_text(report)
    return {"findings": list(F), "report": report, "json": jpath, "report_file": rpath, "images": images,
            "meta": meta, "unaries": U, "binaries": B}


# ---------------------------------------------------------------------------
# tool registration
# ---------------------------------------------------------------------------
def register(tb) -> None:
    S = {"type": "string"}
    N = {"type": "number"}
    STRS = {"type": "array", "items": {"type": "string"}}

    @tb.add("check_tdb",
            "Sanity-check a TDB for CALPHAD assessment pathologies and write a prioritized ERROR/WARNING/INFO report "
            "(evidence + responsible parameters/functions + fix hints), JSON and diagnostic plots. Checks: syntax "
            "(undefined/unused/duplicate functions & parameters, undefined phases/species, TYPE_DEF & DIS_PART "
            "consistency), ranges (T-limit gaps/extrapolation), breakpoints (G and S jumps), load (pycalphad "
            "models), unary (SER phase stable at 298 K, Tm vs SGTE, Cp, inverse melting), interactions (huge L, "
            "T coefficients, excess Cp), binary (solids stable over liquid at high T, liquid/solution miscibility "
            "gaps, liquid Cp).",
            {"tdb": S, "elements": {**STRS, "description": "optional element subset for unary/binary/interaction "
                                                           "checks"},
             "checks": {**STRS, "description": f"subset of {list(ALL_CHECKS)} (default all)"},
             "tmin": {**N, "description": "K, default 298.15"},
             "tmax": {**N, "description": "K, default from the TDB TEMP-LIM clamped to 3000-6000"},
             "output_prefix": {**S, "description": "base name for report/JSON/PNG files"}},
            ["tdb"])
    def check_tdb(tdb, elements=None, checks=None, tmin=298.15, tmax=None, output_prefix=None):
        path = tb._find_tdb(tdb)
        base = tb.outp(output_prefix, f"tdb_check_{Path(path).stem}")
        base = re.sub(r"\.(png|json|txt)$", "", base)
        r = run_checks(path, elements=elements, checks=checks, tmin=tmin, tmax=tmax, output_prefix=base,
                       progress=tb.ui.progress)
        files = [r["report_file"], r["json"]] + r["images"]
        text = r["report"] + "\n\nfiles: " + ", ".join(tb.rel(f) for f in files)
        return text, r["images"]
