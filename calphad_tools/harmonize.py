"""Phase harmonisation: formula-unit normalisation of a phase and extending one database's phase model with the
constituents (and only the parameters that involve them) of another database's version of the phase.

* :func:`rescale_phase` - one phase to new site ratios proportional to the old ones (a formula-unit change by
  ``k = new atoms / old atoms``): every G/L parameter is multiplied by k range by range
  (tdbparse.scale_piecewise), so GM per mole of atoms is unchanged; TC is unchanged.  The IHJ magnetic term
  ``R*T*ln(beta+1)*g(tau)`` is per formula unit and NOT linear in BMAGN, so a magnetic phase cannot be rescaled
  exactly: ``magnetic`` = 'refuse' (error, with the size of the error the other options give), 'keep_parameters'
  (TC/BMAGN as they are, the GM change reported) or 'refit' (BMAGN end-members transformed exactly,
  ``beta' = (1+beta)**k - 1``, interaction BMAGN terms fitted by least squares; the GM residual reported).
* :func:`normalize_phase_formula_unit` - the tool: phases to 'sum_to_one' (written as exact decimals whose sum
  pycalphad reads as 1), explicit ratios, 'integers', or another TDB's phase (``match_tdb``/``match_phase``).
* :func:`extend_phase` - keep ``base_phase`` of the base database authoritative and bring in ``components`` from
  the add database's version of the phase (names may differ): sublattices mapped (tdbstage.sublattice_mapping or
  ``sublattice_map``), the formula-unit basis chosen (below), the new constituents added where add has them, only
  add's parameters that involve a component imported (end-members and interactions; base parameters made of base
  elements are never touched), collisions resolved by policy, FUNCTIONs copied (renamed on clashes), type
  definitions synced, end-members without G listed, and GM checked: the base phase restricted to base elements is
  unchanged (1e-9) and the add phase is reproduced on its subsystems with the components.  Tentative parameters
  (calphad_tools.tentative): on a collision an assessed parameter beats a tentative one whatever ``collision`` says,
  and mixings the extension creates that neither database describes get TENT_MIX / TENT_VA terms (``tentative``).
* Formula-unit basis (``magnetic='auto'``, the default): when exactly one side is magnetic (TC/BMAGN parameters in
  the extended phase), the basis is the MAGNETIC side's - whichever database it is in - and the non-magnetic side
  is rescaled to it (exact), so a magnetic phase is never rescaled and a base phase never overwrites the other
  database's magnetic description.  When both are magnetic with different bases no exact transformation exists:
  'refuse' (default) / 'keep_parameters' / 'refit'.  When neither is magnetic, ``normalization`` decides: 'base'
  (the base's basis) or 'per_atom' (sum to one).
* :func:`staged_extensions` - the same, as staged_merge / combine_databases apply an ``extend`` decision.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
import warnings
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tdbstage as ts
from . import tentative as tent
from .merge import TYPE_CODE_POOL
from .tdbparse import (TDB, Parameter, TypeDef, fmt_ratio, is_tentative, parse_param_name, parse_tdb,
                       scale_piecewise)

GM_RTOL = 1e-9                         # relative GM tolerance of an exact transformation
T_RANGE = (300.0, 2900.0)              # K, where GM is compared (inside the usual 3000 K upper limit)
IMPORT_PTYPES = ("G", "TC", "BMAGN", "NT")
MAGNETIC_MODES = ("auto", "refuse", "keep_parameters", "refit")


class MagneticBasisError(ValueError):
    """A magnetic phase would have to change its formula unit (no exact transformation exists)."""


class HarmonizeError(ValueError):
    pass


# =============================================================================================== ratios
def fmt_k(k: float) -> str:
    """1/3, 19, 2/19 when k is such a fraction, else 6 significant digits."""
    fr = Fraction(k).limit_denominator(1000)
    if abs(float(fr) - k) <= 1e-12 * max(1.0, abs(k)):
        return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
    return f"{k:.6g}"


def _va_only(ph) -> List[int]:
    return [i for i, sl in enumerate(ph.constituents) if sl == ["VA"]]


def _real(ph) -> List[int]:
    va = _va_only(ph)
    return [i for i in range(len(ph.site_ratios)) if i not in va]


def sum_to_one_ratios(ratios: Sequence[float]) -> List[float]:
    """``ratios`` / their sum as exact decimals (15 significant digits) whose decimal sum is exactly 1 and whose float
    sum (what pycalphad adds) is 1 to 1e-15: 2:17 -> 0.105263157894737 / 0.894736842105263."""
    fr = [Fraction(fmt_ratio(r)) for r in ratios]
    tot = sum(fr)
    with localcontext() as ctx:
        ctx.prec = 15
        dec = [+(Decimal(x.numerator) / Decimal(x.denominator)) for x in (f / tot for f in fr)]
    big = max(range(len(dec)), key=lambda i: dec[i])
    dec[big] = Decimal(1) - sum(d for i, d in enumerate(dec) if i != big)
    out = [float(str(d)) for d in dec]
    if abs(sum(out) - 1.0) > 1e-15:          # nudge the largest by one unit in the last place
        for step in (1, -1, 2, -2):
            trial = list(out)
            trial[big] = float(str(dec[big] + Decimal(step) * Decimal(10) ** (dec[big].adjusted() - 14)))
            if abs(sum(trial) - 1.0) <= 1e-15:
                return trial
    return out


def integer_ratios(ratios: Sequence[float]) -> List[float]:
    """Smallest integers proportional to ``ratios`` (0.105263.../0.894736... -> 2/17)."""
    lo = min(ratios)
    q = [Fraction(r / lo).limit_denominator(1000) for r in ratios]
    den = 1
    for x in q:
        den = den * x.denominator // math.gcd(den, x.denominator)
    ints = [int(x * den) for x in q]
    g = 0
    for i in ints:
        g = math.gcd(g, i)
    ints = [i // g for i in ints]
    tot, s = sum(ints), sum(ratios)
    if any(abs(i / tot - r / s) > 1e-6 for i, r in zip(ints, ratios)):
        raise HarmonizeError(f"no small integer ratios proportional to {list(ratios)}")
    return [float(i) for i in ints]


def phase_ratio_target(ph, target) -> List[float]:
    """New site ratios (all sublattices) for 'sum_to_one' | 'integers' | an explicit list (vacancy-only sublattices
    are scaled with the same k)."""
    real = _real(ph)
    old = [ph.site_ratios[i] for i in real]
    if isinstance(target, str):
        t = target.lower()
        if t in ("sum_to_one", "sum1", "per_atom", "one"):
            new_real = sum_to_one_ratios(old)
        elif t in ("integers", "integer", "int"):
            new_real = integer_ratios(old)
        else:
            raise HarmonizeError(f"target {target!r}: use 'sum_to_one', 'integers' or a list of site ratios")
        k = sum(new_real) / sum(old)
        out = [r * k for r in ph.site_ratios]
        for i, r in zip(real, new_real):
            out[i] = r
        return out
    new = [float(x) for x in target]
    if len(new) == len(ph.site_ratios):
        return new
    if len(new) == len(real):
        k = sum(new) / sum(old)
        out = [r * k for r in ph.site_ratios]
        for i, r in zip(real, new):
            out[i] = r
        return out
    raise HarmonizeError(f"{ph.name}: {len(new)} ratios given, the phase has {len(ph.site_ratios)} sublattices")


# =============================================================================================== magnetism
def magnetic_typedef(db: TDB, phase: str) -> Optional[TypeDef]:
    ph = db.phases[phase]
    for td in db.phase_typedefs(ph):
        if td.kind == "MAGNETIC":
            return td
    return None


def magnetic_factors(db: TDB, phase: str) -> Optional[Tuple[float, float]]:
    td = magnetic_typedef(db, phase)
    if td is None:
        return None
    m = re.search(r"MAGNETIC\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)", td.text, re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def is_magnetic(db: TDB, phase: str) -> bool:
    """TC/BMAGN parameters with non-zero values (whether or not a MAGNETIC type definition is attached)."""
    return ts._has_magnetic_params(db, phase)


def beta_transform(beta, k: float, afm: float):
    """BMAGN value(s) of the new formula unit: ln(1+beta_eff') = k*ln(1+beta_eff), beta_eff = beta/afm for beta<=0
    (pycalphad's IHJ convention), k = new atoms per formula unit / old."""
    beta = np.asarray(beta, dtype=float)
    neg = beta <= 0
    eff = np.where(neg, beta / afm, beta)
    new_eff = np.power(1.0 + eff, k) - 1.0
    return np.where(neg, new_eff * afm, new_eff)


# =============================================================================================== pycalphad GM
def pyc_db(db: TDB):
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database.from_string(ts.tdb_to_text(db), fmt="tdb")


def phase_function(dbp, phase: str, comps: Sequence[str], what: str = "GM"):
    """(Model, [(sublattice, species)], f) with f(rows) -> values; rows = [T, P, y...] (site fractions as listed)."""
    import symengine as se
    from pycalphad import Model, variables as v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = Model(dbp, list(comps), phase)
    expr = getattr(mod, what) if isinstance(what, str) else what(mod)
    ys = list(mod.site_fractions)
    f = se.Lambdify([v.T, v.P] + ys, [expr], backend="lambda")
    return mod, [(int(y.sublattice_index), str(y.species.name)) for y in ys], f


def _sublattices(ys) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for s, sp in ys:
        out.setdefault(s, []).append(sp)
    return out


def sample_states(subl: Dict[int, List[str]], n: int, rng, t_range=T_RANGE) -> Tuple[np.ndarray, Dict]:
    """Random T and site fractions (Dirichlet per sublattice) plus every end-member and binary midpoint at a few T."""
    keys = sorted(subl)
    pts: List[Dict[Tuple[int, str], float]] = []
    Ts: List[float] = []
    ems = list(itertools.product(*[subl[s] for s in keys]))[:200]
    for em in ems:
        for T in np.linspace(t_range[0], t_range[1], 5):
            pts.append({(s, sp): 1.0 for s, sp in zip(keys, em)})
            Ts.append(float(T))
    for _ in range(n):
        d = {}
        for s in keys:
            w = rng.dirichlet(np.ones(len(subl[s]))) if len(subl[s]) > 1 else np.ones(1)
            if rng.random() < 0.25 and len(subl[s]) > 1:      # edges: two species only
                w = np.zeros(len(subl[s]))
                i, j = rng.choice(len(subl[s]), 2, replace=False)
                w[i] = rng.random()
                w[j] = 1.0 - w[i]
            for sp, x in zip(subl[s], w):
                d[(s, sp)] = float(x)
        pts.append(d)
        Ts.append(float(rng.uniform(*t_range)))
    return np.asarray(Ts), pts


def _rows(ys, Ts, pts, sub_map=None) -> np.ndarray:
    sub_map = sub_map or {}
    cols = [Ts, np.full(len(Ts), 101325.0)]
    for s, sp in ys:
        src = [k for k, v in sub_map.items() if v == s]
        s0 = src[0] if src else s
        cols.append(np.asarray([p.get((s0, sp), 0.0) for p in pts]))
    return np.column_stack(cols)


def gm_compare(dbA, phA: str, dbB, phB: str, comps: Sequence[str], a_to_b: Optional[Sequence[int]] = None,
               n: int = 400, seed: int = 0, t_range=T_RANGE) -> Dict[str, Any]:
    """Max |GM_A - GM_B| (J/mol of atoms) and relative error at random T / site fractions; A's sublattice i is B's
    a_to_b[i].  States use only species both models have on (mapped) sublattices; the others are listed."""
    try:
        _, ysA, fA = phase_function(dbA, phA, comps)
        _, ysB, fB = phase_function(dbB, phB, comps)
    except Exception as e:  # noqa: BLE001 - reported
        return {"max_abs": math.nan, "max_rel": math.nan, "n": 0, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    sa, sb = _sublattices(ysA), _sublattices(ysB)
    amap = {i: (a_to_b[i] if a_to_b is not None else i) for i in sa}
    common, excluded = {}, []
    for s, sps in sa.items():
        t = amap[s]
        keep = [sp for sp in sps if sp in sb.get(t, [])]
        excluded += [f"{sp} on sublattice {s + 1} of {phA}" for sp in sps if sp not in keep]
        excluded += [f"{sp} on sublattice {t + 1} of {phB}" for sp in sb.get(t, []) if sp not in sps]
        if not keep:
            return {"max_abs": math.nan, "max_rel": math.nan, "n": 0,
                    "error": f"no common species on sublattice {s + 1} of {phA}"}
        common[s] = keep
    rng = np.random.default_rng(seed)
    Ts, pts = sample_states(common, n, rng, t_range)
    ga = np.asarray(fA(_rows(ysA, Ts, pts)), float).reshape(-1)
    gb = np.asarray(fB(_rows(ysB, Ts, pts, {s: amap[s] for s in sa})), float).reshape(-1)
    ok = np.isfinite(ga) & np.isfinite(gb)
    if not ok.any():
        return {"max_abs": math.nan, "max_rel": math.nan, "n": 0, "error": "GM not finite"}
    d = np.abs(ga - gb)
    d[~ok] = 0.0
    rel = d / np.maximum(np.abs(ga), 1.0)
    i = int(np.argmax(d))
    at = ", ".join(f"y({sp}#{s + 1})={x:.3g}" for (s, sp), x in sorted(pts[i].items()) if x > 0)
    return {"max_abs": float(d[i]), "max_rel": float(rel.max()), "n": int(ok.sum()),
            "nonfinite": int((~ok).sum()), "T": [float(t_range[0]), float(t_range[1])],
            "worst": f"T={Ts[i]:.0f} K, {at}", "excluded": sorted(set(excluded))}


def gm_text(res: Dict[str, Any], exact: bool = True) -> str:
    if not math.isfinite(res.get("max_abs", math.nan)):
        return f"not checked ({res.get('error', '?')})"
    s = (f"max |ΔGM| {res['max_abs']:.3g} J/mol of atoms, max relative {res['max_rel']:.1e} over {res['n']} states "
         f"({res['T'][0]:g}-{res['T'][1]:g} K)")
    if exact:
        s += " - exact" if res["max_rel"] <= GM_RTOL else " - NOT exact"
    return s


def phase_comps(db: TDB, phase: str) -> List[str]:
    els = set()
    for sl in db.phases[phase].constituents:
        for s in sl:
            els |= ts._species_els(db, s)
    return sorted(els - {"*"} | {"VA"})


# =============================================================================================== BMAGN refit
def _const_value(db: TDB, p: Parameter) -> Optional[float]:
    vals = db.evaluate(p.expr, [400.0, 1200.0, 2500.0])
    if not np.all(np.isfinite(vals)) or np.ptp(vals) > 1e-12 * max(1.0, float(np.max(np.abs(vals)))):
        return None
    return float(vals[0])          # as written (raw constituent order, as pycalphad reads it)


def _raw_subls(p: Parameter) -> List[List[str]]:
    from .merge import _split_raw_name
    return [[x.strip("%") for x in s] for s in _split_raw_name(p.raw_name)[3]]


def _raw_order(p: Parameter) -> int:
    from .merge import _split_raw_name
    o = _split_raw_name(p.raw_name)[4]
    return int(o) if o not in (None, "") else 0


def _rk_basis(subls: List[List[str]], order: int, Y: Dict[Tuple[int, str], np.ndarray], n: int) -> np.ndarray:
    """A Redlich-Kister term as pycalphad evaluates it: constituents of a sublattice SORTED, odd orders NOT
    sign-flipped (pycalphad reads L(X,B,A;1) as L(X,A,B;1) with the same value)."""
    term = np.ones(n)
    for s, sl in enumerate(subls):
        sl = sorted(x.strip("%") for x in sl)
        if sl == ["*"]:
            continue
        if len(sl) > 2:
            raise HarmonizeError("refit supports binary interactions only")
        for sp in sl:
            term = term * Y.get((s, sp), np.zeros(n))
        if len(sl) == 2 and order:
            term = term * (Y.get((s, sl[0]), np.zeros(n)) - Y.get((s, sl[1]), np.zeros(n))) ** order
    return term


def refit_bmagn(db: TDB, phase: str, k: float, n: int = 600, seed: int = 1) -> Dict[str, Any]:
    """In place: BMAGN of ``phase`` for a formula unit k times larger.  End-members exactly
    (beta' = (1+beta)**k - 1, AFM convention), interaction terms (existing ones + order 0-2 binary terms between
    constituents whose end-members are magnetic) fitted by least squares to the exact target over random site
    fractions.  Returns the fit summary (the GM residual is computed by the caller with pycalphad)."""
    from .merge import _build_raw_name, _split_raw_name
    fac = magnetic_factors(db, phase)
    if fac is None:
        raise HarmonizeError(f"{phase}: no MAGNETIC type definition, nothing to refit")
    afm = fac[0]
    if afm == 0:
        raise HarmonizeError(f"{phase}: AFM factor 0 (Xiong model) - refit not supported")
    ph = db.phases[phase]
    params = [p for p in db.parameters if p.key.phase == phase and p.key.ptype == "BMAGN"]
    vals = {}
    for p in params:
        v = _const_value(db, p)
        if v is None:
            raise HarmonizeError(f"{p.raw_name}: temperature-dependent BMAGN - refit not supported")
        vals[id(p)] = v
    subl = {i: list(sl) for i, sl in enumerate(ph.constituents)}
    rng = np.random.default_rng(seed)
    _, pts = sample_states(subl, n, rng)
    N = len(pts)
    Y = {key: np.asarray([pt.get(key, 0.0) for pt in pts]) for key in {kk for pt in pts for kk in pt}}

    def is_em(p):
        return all(len(sl) == 1 and sl[0] != "*" for sl in _raw_subls(p))
    beta = np.zeros(N)
    for p in params:
        beta += vals[id(p)] * _rk_basis(_raw_subls(p), _raw_order(p), Y, N)
    # pycalphad agreement of the RK evaluation (raw order as written)
    try:
        from tinydb import where
        dbp = pyc_db(db)
        mod, ys, _ = phase_function(dbp, phase, phase_comps(db, phase))
        q = ((where("phase_name") == phase) & (where("parameter_type") == "BMAGN")
             & (where("constituent_array").test(mod._array_validity)))
        mmm = mod.symbol_replace(mod.redlich_kister_sum(dbp.phases[phase], dbp.search, q), mod._symbols)
        import symengine as se
        from pycalphad import variables as v
        f = se.Lambdify([v.T, v.P] + list(mod.site_fractions), [mmm], backend="lambda")
        ref = np.asarray(f(_rows(ys, np.full(N, 1000.0), pts)), float).reshape(-1)
        if np.max(np.abs(ref - beta)) > 1e-9 * max(1.0, float(np.max(np.abs(ref)))):
            raise HarmonizeError(f"{phase}: internal BMAGN evaluation disagrees with pycalphad "
                                 f"(max {np.max(np.abs(ref - beta)):.3g}) - refit not done")
    except HarmonizeError:
        raise
    except Exception:  # noqa: BLE001 - the GM residual check still catches a wrong fit
        pass
    target = beta_transform(beta, k, afm)
    em_p = [p for p in params if is_em(p)]
    other = [p for p in params if not is_em(p)]
    fixed = np.zeros(N)
    em_new = {}
    for p in em_p:
        nv = float(beta_transform(vals[id(p)], k, afm))
        em_new[id(p)] = nv
        fixed += nv * _rk_basis(_raw_subls(p), 0, Y, N)
    # fitted terms: existing non-end-member BMAGN + new binary interactions next to magnetic end-members
    magnetic_ems = {tuple(sl[0] for sl in _raw_subls(p)) for p in em_p if abs(vals[id(p)]) > 0}
    cols, terms = [], []
    have = set()
    for p in other:
        sub, order = _raw_subls(p), _raw_order(p)
        cols.append(_rk_basis(sub, order, Y, N))
        terms.append(("existing", p, sub, order))
        have.add((tuple(tuple(sorted(s)) for s in sub), order))
    for s, sl in subl.items():
        for a, b in itertools.combinations(sl, 2):
            others = [subl[t] for t in sorted(subl) if t != s]
            for combo in itertools.product(*others):
                full_a = list(combo)
                full_a.insert(s, a)
                full_b = list(combo)
                full_b.insert(s, b)
                if tuple(full_a) not in magnetic_ems and tuple(full_b) not in magnetic_ems:
                    continue
                sub = [[x] for x in combo]
                sub.insert(s, sorted([a, b]))
                for order in (0, 1, 2):
                    key = (tuple(tuple(sorted(x)) for x in sub), order)
                    if key in have:
                        continue
                    have.add(key)
                    cols.append(_rk_basis(sub, order, Y, N))
                    terms.append(("new", None, sub, order))
    coef = np.zeros(0)
    if cols:
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, target - fixed, rcond=None)
    fit = fixed + (np.column_stack(cols) @ coef if cols else 0.0)
    resid_beta = float(np.max(np.abs(fit - target))) if N else 0.0
    # write back
    changed, added = [], []
    for p in em_p:
        p.expr = type(p.expr)([(p.expr.tmin, p.expr.tmax, repr(float(f"{em_new[id(p)]:.12g}")))],
                              p.expr.reference)
        changed.append(f"{p.raw_name} {vals[id(p)]:.6g} -> {em_new[id(p)]:.6g}")
    for (kind, p, sub, order), c in zip(terms, coef):
        c = float(f"{c:.10g}")
        if kind == "existing":
            p.expr = type(p.expr)([(p.expr.tmin, p.expr.tmax, repr(c))], p.expr.reference)
            changed.append(f"{p.raw_name} {vals[id(p)]:.6g} -> {c:.6g} (fitted)")
        elif abs(c) > 1e-10:
            raw = _build_raw_name("BMAGN", phase, "", sub, str(order))
            k2, sign, om = parse_param_name(raw)
            tmin, tmax = db.temp_limits
            db.parameters.append(Parameter(raw, k2, sign, om, type(params[0].expr)([(tmin, tmax, repr(c))],
                                                                                   "REFIT")))
            added.append(f"{raw} = {c:.6g} (fitted)")
    return {"k": k, "afm": afm, "end_members": changed[:len(em_p)], "fitted": changed[len(em_p):] + added,
            "beta_residual": resid_beta, "n_samples": N}


# =============================================================================================== rescale
def rescale_phase(db: TDB, phase: str, new_ratios: Sequence[float], magnetic: str = "auto",
                  label: str = "", check: bool = True) -> Dict[str, Any]:
    """In place: ``phase`` of ``db`` to ``new_ratios`` (proportional to the old ones).  See the module docstring for
    the magnetic options.  Returns a report with k, the parameters scaled and the GM check (original vs new)."""
    if magnetic not in MAGNETIC_MODES:
        raise HarmonizeError(f"magnetic={magnetic!r}: one of {', '.join(MAGNETIC_MODES)}")
    ph = db.phases[phase]
    old = list(ph.site_ratios)
    new = [float(x) for x in new_ratios]
    if len(new) != len(old):
        raise HarmonizeError(f"{phase}: {len(new)} ratios for {len(old)} sublattices")
    real = _real(ph)
    k = sum(new[i] for i in real) / sum(old[i] for i in real)
    if any(abs(n - k * o) > 1e-9 * max(new) for i, (n, o) in enumerate(zip(new, old)) if i in real):
        raise HarmonizeError(f"{phase}: {[fmt_ratio(x) for x in new]} is not proportional to "
                             f"{[fmt_ratio(x) for x in old]} - not a formula-unit change (different model)")
    rep: Dict[str, Any] = {"phase": phase, "label": label, "old": old, "new": new, "k": k, "k_text": fmt_k(k),
                           "magnetic": False, "scaled": 0, "unscaled": [], "mode": magnetic}
    if all(a == b for a, b in zip(old, new)):
        rep["gm"] = {"max_abs": 0.0, "max_rel": 0.0, "n": 0, "T": list(T_RANGE)}
        rep["text"] = f"{phase}: already {_ratios_text(new)}"
        return rep
    if ts._order_disorder_related(db, phase):
        raise HarmonizeError(f"{phase}: order-disorder (DIS_PART) phase - its partner would need the same "
                             f"rescale; not supported")
    mag = is_magnetic(db, phase) and abs(k - 1.0) > 1e-15
    rep["magnetic"] = mag
    orig = copy.deepcopy(db)
    if mag and magnetic in ("auto", "refuse"):
        keep = copy.deepcopy(orig)
        r1 = rescale_phase(keep, phase, new, "keep_parameters", label, check=True)
        try:
            ref = copy.deepcopy(orig)
            r2 = rescale_phase(ref, phase, new, "refit", label, check=True)
            refit_txt = f"refit leaves {r2['gm']['max_abs']:.3g} J/mol of atoms"
        except HarmonizeError as e:
            refit_txt = f"refit not possible ({e})"
        raise MagneticBasisError(
            f"{phase}{' (' + label + ')' if label else ''}: magnetic (TC/BMAGN) and its formula unit would change by "
            f"k = {fmt_k(k)} ({_ratios_text(old)} -> {_ratios_text(new)}). The IHJ term R*T*ln(BMAGN+1)*g(T/TC) is "
            f"per formula unit and not linear in BMAGN, so no exact rescale exists. magnetic='keep_parameters' would "
            f"change GM by up to {r1['gm']['max_abs']:.3g} J/mol of atoms; {refit_txt} (max over {r1['gm']['n']} "
            f"states, {T_RANGE[0]:g}-{T_RANGE[1]:g} K). Rescale the other, non-magnetic phase to this basis instead "
            f"(magnetic='auto' does that when only one side is magnetic), or pass magnetic='keep_parameters' / "
            f"'refit'.")
    ph.site_ratios = new
    for p in db.parameters:
        if p.key.phase != phase:
            continue
        if p.key.ptype in ts.SCALED_PTYPES:
            p.expr = scale_piecewise(p.expr, k)
            rep["scaled"] += 1
        elif p.key.ptype not in ts.MAGNETIC_PTYPES:
            rep["unscaled"].append(p.raw_name)
    if mag and magnetic == "refit":
        rep["refit"] = refit_bmagn(db, phase, k)
    if check:
        try:
            rep["gm"] = gm_compare(pyc_db(orig), phase, pyc_db(db), phase, phase_comps(orig, phase))
        except Exception as e:  # noqa: BLE001
            rep["gm"] = {"max_abs": math.nan, "max_rel": math.nan, "n": 0, "error": str(e)[:160]}
    how = ("G/L x" + fmt_k(k) + ", TC unchanged" + (
        "" if not mag else ", BMAGN kept (keep_parameters)" if magnetic == "keep_parameters" else
        ", BMAGN refitted"))
    rep["text"] = (f"{phase}{' (' + label + ')' if label else ''}: {_ratios_text(old)} -> {_ratios_text(new)}, "
                   f"k = {fmt_k(k)}; {rep['scaled']} G/L term(s) scaled, {how}"
                   + (f"; GM {gm_text(rep['gm'], exact=not mag)}" if "gm" in rep else ""))
    return rep


def _ratios_text(r) -> str:
    return ":".join(fmt_ratio(x) for x in r)


# =============================================================================================== tool 1
def normalize_phase_formula_unit(tdb, phases: Sequence[str], target="sum_to_one", output_tdb: Optional[str] = None,
                                 magnetic: str = "auto", match_tdb: Optional[str] = None,
                                 match_phase: Optional[str] = None, sublattice_map=None) -> Dict[str, Any]:
    """Change the formula unit of ``phases`` (GM per mole of atoms unchanged; see rescale_phase).  ``target``:
    'sum_to_one' | 'integers' | [ratios]; or ``match_tdb`` (+ ``match_phase``, default the same name): the other
    database's site ratios, mapped onto this phase's sublattice order.  magnetic 'auto' = 'refuse' for a magnetic
    phase (no exact transformation); rescale the non-magnetic side instead."""
    db = tdb if isinstance(tdb, TDB) else parse_tdb(str(tdb))
    src = db.path
    db = copy.deepcopy(db)
    reports = []
    for name in phases:
        ph_name = name.upper()
        if ph_name not in db.phases:
            raise HarmonizeError(f"{ph_name} is not a phase of {db.name}")
        ph = db.phases[ph_name]
        if match_tdb:
            other = match_tdb if isinstance(match_tdb, TDB) else parse_tdb(str(match_tdb))
            mph = (match_phase or ph_name).upper()
            if mph not in other.phases:
                raise HarmonizeError(f"{mph} is not a phase of {other.name}")
            ov = (sublattice_map or {}).get(ph_name) if isinstance(sublattice_map, dict) else sublattice_map
            m = ts.sublattice_mapping(ph, other.phases[mph], ov)
            if m is None:
                raise HarmonizeError(f"{ph_name} {ts._subl_text(ph)} and {other.name}'s {mph} "
                                     f"{ts._subl_text(other.phases[mph])}: site ratios not proportional")
            perm, _ = m
            o_real = _real(other.phases[mph])
            new = list(ph.site_ratios)
            for t, i in enumerate(_real(ph)):
                new[i] = other.phases[mph].site_ratios[o_real[perm[t]]]
            kk = sum(new[i] for i in _real(ph)) / sum(ph.site_ratios[i] for i in _real(ph))
            for i in _va_only(ph):
                new[i] = ph.site_ratios[i] * kk
        else:
            new = phase_ratio_target(ph, target)
        rep = rescale_phase(db, ph_name, new, magnetic)
        if isinstance(target, str) and target.lower() in ("sum_to_one", "sum1", "per_atom", "one") and not match_tdb:
            rep["sum_text"] = " + ".join(fmt_ratio(db.phases[ph_name].site_ratios[i]) for i in _real(ph))
        reports.append(rep)
    out = {"phases": reports, "input": src}
    if output_tdb:
        hdr = [f"{Path(src).name} with formula units changed (calphad_tools.harmonize.normalize_phase_formula_unit):"]
        hdr += [r["text"] for r in reports]
        ts.write_tdb(db, str(output_tdb), hdr)
        out["output"] = str(output_tdb)
        dbp = pyc_db(parse_tdb(str(output_tdb)))
        for r in reports:
            php = dbp.phases[r["phase"]]
            r["pycalphad_sum"] = float(sum(php.sublattices[i] for i in _real(db.phases[r["phase"]])))
            r["magnetic_hints"] = {k: v for k, v in php.model_hints.items() if k.startswith("ihj")}
        exp = ["only affected phases changed"]
        if not any(r["magnetic"] for r in reports):
            exp.append("energy_per_atom_preserved")
        if "sum_text" in reports[0] if reports else False:
            exp.append("site ratios sum to 1")
        out["validator"] = external_validation(src, str(output_tdb), [r["phase"] for r in reports], exp,
                                               {r["phase"]: r["k"] for r in reports})
    out["tdb"] = db
    out["text"] = "\n".join([r["text"] + (f"; pycalphad reads the site ratios' sum as {r['pycalphad_sum']!r}"
                                          if "pycalphad_sum" in r else "") for r in reports]
                            + [x for x in [validator_line(out.get("validator"), ("site ratios",) if any(
                                abs(sum(r["old"][i] for i in range(len(r["old"]))) - 1) < 1e-9 for r in reports)
                                else ())] if x])
    return out


# =============================================================================================== extend
def _species_hits(db: TDB, sp: str, comps) -> bool:
    return bool(ts._species_els(db, sp) & set(comps))


def _param_elements(db: TDB, p: Parameter) -> set:
    return ts.param_element_set(db, p)


def _free_code(db: TDB, also: str = "") -> str:
    used = set(db.typedefs) | set(also) | {c for ph in db.phases.values() for c in ph.type_codes} | {"%"}
    for c in TYPE_CODE_POOL:
        if c not in used:
            return c
    raise HarmonizeError("no free TYPE_DEFINITION code")


def basis_choice(base_mag: bool, add_mag: bool, base_total: float, add_total: float, normalization: str,
                 magnetic: str, phase: str, base_label: str = "base", add_label: str = "add") -> Dict[str, Any]:
    """Which formula-unit basis the extended phase gets: {'basis': 'base'|'add'|'sum_to_one', 'k_base', 'k_add',
    'exact', 'text', 'magnetic_problem'}.  (No TDB needed: also used to word the plan's question.)"""
    one_side = base_mag != add_mag
    if one_side and magnetic in ("auto", "refuse"):
        basis = "add" if add_mag else "base"
        why = (f"{base_label}'s non-magnetic model rescaled to {add_label}'s magnetic basis" if add_mag else
               f"{add_label}'s non-magnetic parameters rescaled to {base_label}'s magnetic basis")
    elif normalization == "per_atom":
        basis, why = "sum_to_one", "both on one atom per formula unit (normalization 'per_atom')"
    else:
        basis, why = "base", f"{add_label}'s parameters rescaled to {base_label}'s formula unit"
    tot = {"add": add_total, "base": base_total, "sum_to_one": 1.0}[basis]
    kb, ka = tot / base_total, tot / add_total
    problem = []
    if base_mag and abs(kb - 1) > 1e-12:
        problem.append(base_label)
    if add_mag and abs(ka - 1) > 1e-12:
        problem.append(add_label)
    exact = not problem
    k_txt = (f"k = {fmt_k(kb)}" if abs(kb - 1) > 1e-12 and abs(ka - 1) <= 1e-12 else
             f"k = {fmt_k(ka)}" if abs(ka - 1) > 1e-12 and abs(kb - 1) <= 1e-12 else
             f"k({base_label}) = {fmt_k(kb)}, k({add_label}) = {fmt_k(ka)}")
    if abs(kb - 1) <= 1e-12 and abs(ka - 1) <= 1e-12:
        why, k_txt = f"same formula unit ({base_label} {tot:g} sites)", "k = 1"
    text = (f"{phase}: {why} (sum {tot:.6g}), {k_txt}, "
            + ("exact" if exact else f"NOT exact: {' and '.join(problem)} magnetic and rescaled "
                                     f"(magnetic='{magnetic}')"))
    return {"basis": basis, "total": tot, "k_base": kb, "k_add": ka, "exact": exact, "text": text,
            "magnetic_problem": problem, "base_magnetic": base_mag, "add_magnetic": add_mag}


def _imported(add: TDB, phase: str, comps) -> List[Parameter]:
    return [p for p in add.parameters if p.key.phase == phase and _param_elements(add, p) & set(comps)]


def extend_phase_db(base: TDB, add: TDB, base_phase: str, add_phase: Optional[str] = None,
                    components: Optional[Sequence[str]] = None, sublattice_map=None, normalization: str = "base",
                    collision: str = "base", magnetic: str = "auto", include_functions: bool = True,
                    add_constituents: Optional[Dict] = None, base_label: str = "base",
                    add_label: str = "add", tentative=True) -> Tuple[TDB, Dict[str, Any]]:
    """The extension without verification or writing: (new base TDB, report).  See extend_phase."""
    from .merge import _split_raw_name
    if normalization not in ("base", "per_atom"):
        raise HarmonizeError(f"normalization={normalization!r}: 'base' or 'per_atom'")
    if collision not in ("base", "add", "error"):
        raise HarmonizeError(f"collision={collision!r}: 'base', 'add' or 'error'")
    if magnetic not in MAGNETIC_MODES:
        raise HarmonizeError(f"magnetic={magnetic!r}: one of {', '.join(MAGNETIC_MODES)}")
    P = base_phase.upper()
    Q = (add_phase or base_phase).upper()
    if P not in base.phases:
        raise HarmonizeError(f"{P} is not a phase of {base.name}")
    if Q not in add.phases:
        raise HarmonizeError(f"{Q} is not a phase of {add.name}")
    B, D = copy.deepcopy(base), copy.deepcopy(add)
    for db, ph, lab in ((B, P, base_label), (D, Q, add_label)):
        if ts._order_disorder_related(db, ph):
            raise HarmonizeError(f"{ph} in {lab} is an order-disorder (DIS_PART) phase: extend_phase does not "
                                 f"handle partitioned models")
    base_els = set(phase_comps(B, P)) - {"VA"}
    add_els = set(phase_comps(D, Q)) - {"VA"}
    comps = sorted({c.upper() for c in components}) if components else sorted(add_els - base_els)
    if not comps:
        raise HarmonizeError(f"{add_label}'s {Q} has no element that {base_label}'s {P} lacks: nothing to extend "
                             f"(give components)")
    missing = [c for c in comps if c not in add_els]
    if missing:
        raise HarmonizeError(f"{add_label}'s {Q} has no {', '.join(missing)}")
    rep: Dict[str, Any] = {"base_phase": P, "add_phase": Q, "components": comps, "base_label": base_label,
                           "add_label": add_label, "notes": [], "tentative": [], "tentative_replaced": [],
                           "tentative_notes": []}
    cfg = tent.settings(tentative)
    # ---- names: add's phase is called P from here on (a different add phase already called P moves aside)
    if Q != P:
        if P in D.phases:
            ts.rename_phases(D, {P: P + "_ADDSIDE"})
        ts.rename_phases(D, {Q: P})
    pb, pd = B.phases[P], D.phases[P]
    if len(pb.site_ratios) != len(pd.site_ratios) or _va_only(pb) != _va_only(pd):
        raise HarmonizeError(f"{P}: {base_label} {ts._subl_text(pb)} and {add_label} {ts._subl_text(pd)} have "
                             f"different (vacancy-only) sublattices: not one model")
    ov = sublattice_map
    try:
        m = ts.sublattice_mapping(pb, pd, ov)
    except ts.SublatticeMapError as e:
        raise HarmonizeError(str(e).replace("give sublattice_map {'", "give sublattice_map (").replace(
            "': [...]}", ": [...])")) from None
    if m is None:
        raise HarmonizeError(f"{P}: site ratios not proportional ({base_label} {ts._subl_text(pb)}, {add_label} "
                             f"{ts._subl_text(add.phases[Q])}): a different "
                             f"structure, not a formula-unit difference")
    perm, _ = m
    b_idx = _real(pd)
    new_from_old = list(range(len(pd.site_ratios)))
    for t, j in enumerate(perm):
        new_from_old[b_idx[t]] = b_idx[j]
    rep["new_from_old"] = new_from_old
    rep["mapping"] = (f"{add_label}'s {Q} {ts._subl_text(add.phases[Q])} -> {base_label}'s order: {add_label} "
                      f"sublattice " + ", ".join(f"{o + 1}->{n + 1}" for n, o in enumerate(new_from_old)))
    if new_from_old != list(range(len(pd.site_ratios))):
        pd.site_ratios = [pd.site_ratios[o] for o in new_from_old]
        pd.constituents = [pd.constituents[o] for o in new_from_old]
        if pd.majors and len(pd.majors) == len(new_from_old):
            pd.majors = [pd.majors[o] for o in new_from_old]
        for p in D.parameters:
            if p.key.phase == P:
                p.raw_name = ts._permute_raw(p.raw_name, new_from_old)
                p.key, p.sign, p.order_map = parse_param_name(p.raw_name)
    add_tent_pairs = _tent_pairs(D, P)          # add's pairs described only by tentative terms (all its parameters)
    # ---- what is imported: add's parameters of the phase with a component (others dropped from the scratch copy)
    imp_all = _imported(D, P, comps)
    rest = [p for p in D.parameters if p.key.phase == P and p not in imp_all]
    D.parameters = [p for p in D.parameters if p.key.phase != P or p in imp_all]
    base_mag = is_magnetic(B, P)
    add_mag = is_magnetic(D, P)
    tb = sum(pb.site_ratios[i] for i in _real(pb))
    td = sum(pd.site_ratios[i] for i in _real(pd))
    bc = basis_choice(base_mag, add_mag, tb, td, normalization, magnetic, P, base_label, add_label)
    # the add parameters NOT imported (no component) against the base's: where the base's win, add's subsystems
    # with those elements cannot be reproduced
    rep["base_wins"] = _base_wins(B, D, P, rest, bc, base_label, add_label)
    if bc["basis"] == "add":
        basis = list(pd.site_ratios)
    elif bc["basis"] == "sum_to_one":
        real = _real(pb)
        s1 = sum_to_one_ratios([pb.site_ratios[i] for i in real])
        basis = [r / tb for r in pb.site_ratios]
        for i, r in zip(real, s1):
            basis[i] = r
    else:
        basis = list(pb.site_ratios)
    rep["basis"] = bc
    mode = "refuse" if magnetic == "auto" else magnetic
    rep["rescale_base"] = rescale_phase(B, P, basis, mode, base_label, check=False)
    add_basis = list(basis)
    rep["rescale_add"] = rescale_phase(D, P, add_basis, mode, add_label, check=False)
    pb, pd = B.phases[P], D.phases[P]
    base_cons = [list(sl) for sl in pb.constituents]
    # ---- constituents
    added_cons: Dict[int, List[str]] = {}
    want = None
    if add_constituents:
        want = {}
        for j, sps in add_constituents.items():
            j = int(j)
            if j not in new_from_old:
                raise HarmonizeError(f"add_constituents: {add_label}'s {Q} has no sublattice {j}")
            t = new_from_old.index(j)
            sps = [s.upper() for s in ([sps] if isinstance(sps, str) else sps)]
            bad = [s for s in sps if s not in pd.constituents[t]]
            if bad:
                raise HarmonizeError(f"add_constituents: {', '.join(bad)} not on {add_label}'s {Q} sublattice {j} "
                                     f"({','.join(add.phases[Q].constituents[j])}; sublattices numbered from 0 as "
                                     f"in its PHASE line)")
            want[t] = sps
    for t, sl in enumerate(pd.constituents):
        new = [s for s in sl if s not in pb.constituents[t] and s != "VA" and _species_hits(D, s, comps)]
        if want is not None:
            new = [s for s in new if s in want.get(t, [])]
        if new:
            pb.constituents[t] = pb.constituents[t] + new
            added_cons[t] = new
    rep["added_constituents"] = {t: v for t, v in added_cons.items()}
    if not any(_species_hits(B, x, comps) for sl in pb.constituents for x in sl):
        raise HarmonizeError(f"no constituent with {', '.join(comps)} to add to {base_label}'s {P}")
    for c in comps:
        if c not in B.elements and c in D.elements:
            B.elements[c] = copy.deepcopy(D.elements[c])
    for sl in added_cons.values():
        for s in sl:
            if s in D.species and s not in B.species:
                B.species[s] = copy.deepcopy(D.species[s])
                for e in ts._species_els(D, s):
                    if e in D.elements and e not in B.elements:
                        B.elements[e] = copy.deepcopy(D.elements[e])
    # ---- parameters
    base_keys = {p.key.as_tuple(): p for p in B.parameters if p.key.phase == P}
    imported, skipped, collisions = [], [], []
    T = np.linspace(298.15, 3000.0, 60)
    to_add = []
    for p in [x for x in D.parameters if x.key.phase == P]:
        if p.key.ptype not in IMPORT_PTYPES:
            skipped.append(f"{p.raw_name}: parameter type {p.key.ptype} not imported")
            continue
        subls = _split_raw_name(p.raw_name)[3]
        bad = [f"{s} on sublattice {t + 1}" for t, sl in enumerate(subls) for s in sl
               if s.strip("%") != "*" and s.strip("%") not in pb.constituents[t]]
        if bad:
            skipped.append(f"{p.raw_name}: {', '.join(bad)} not in the extended {P} (only species with "
                           f"{', '.join(comps)} are added)")
            continue
        kt = p.key.as_tuple()
        if kt in base_keys:
            bp = base_keys[kt]
            try:
                va, vb = B.evaluate(bp.expr, T) * bp.sign, D.evaluate(p.expr, T) * p.sign
                d = float(np.nanmax(np.abs(va - vb)))
            except Exception:  # noqa: BLE001
                va = vb = None
                d = math.nan
            bt, at = is_tentative(bp, B), is_tentative(p, D)
            if bt or at:
                # tentative precedence: the assessed term wins, whatever ``collision`` says; both tentative: larger
                if bt and at:
                    add_wins = va is not None and float(np.nanmean(vb)) > float(np.nanmean(va))
                    rep["tentative_replaced"].append(tent.Replacement(
                        p.key.label(), tent.ref_key(bp), add_label if not add_wins else base_label,
                        add_label if add_wins else base_label, "both", (p if add_wins else bp).expr.short(40)))
                else:
                    add_wins = bt
                    rep["tentative_replaced"].append(tent.Replacement(
                        p.key.label(), tent.ref_key(bp if bt else p), base_label if bt else add_label,
                        add_label if bt else base_label, "assessed"))
                if not add_wins:
                    continue
                B.parameters = [x for x in B.parameters if x is not bp]
                to_add.append(p)
                continue
            if collision == "error":
                raise HarmonizeError(f"collision: {p.key.label()} is in both ({base_label} and {add_label}, max "
                                     f"|Δ| {d:.4g}); choose collision='base' or 'add'")
            collisions.append({"parameter": p.key.label(), "kept": base_label if collision == "base" else add_label,
                               "max_diff": d})
            if collision == "base":
                continue
            B.parameters = [x for x in B.parameters if x is not bp]
        to_add.append(p)
    # ---- FUNCTIONs the imported parameters need (renamed where the base defines the name differently)
    need = ts._function_closure(D, [s for p in to_add for s in p.expr.symbols()])
    clash = {n for n in need if n in B.functions and B.functions[n].expr.canonical() != D.functions[n].expr.canonical()}
    changed = True
    while changed:                               # a function that uses a renamed one is renamed too
        changed = False
        for n in need:
            if n not in clash and n in B.functions and set(D.functions[n].expr.symbols()) & clash:
                clash.add(n)
                changed = True
    missing_f = [n for n in need if n not in B.functions or n in clash]
    if not include_functions and missing_f:
        raise HarmonizeError(f"the imported parameters need FUNCTION(s) {', '.join(missing_f)} that {base_label} "
                             f"lacks or defines differently (include_functions=True copies / renames them)")
    rename = {}
    for n in sorted(clash):
        i, new = 2, f"{n}_2"
        while new in B.functions or new in D.functions:
            i, new = i + 1, f"{n}_{i}"
        rename[n] = new
    fn_notes = []
    for n in need:
        if n in B.functions and n not in clash:
            continue
        f = copy.deepcopy(D.functions[n])
        f.expr = f.expr.rename_symbols(rename)
        f.name = rename.get(n, n)
        B.functions[f.name] = f
        fn_notes.append(f"FUNCTION {n} copied" + (f" as {f.name} ({base_label} defines {n} differently)"
                                                  if n in rename else ""))
    resorted = []
    for p in to_add:
        q = copy.deepcopy(p)
        q.expr = q.expr.rename_symbols(rename)
        # interaction constituents written in pycalphad's (sorted) order with the expression unchanged: pycalphad
        # sorts them WITHOUT flipping odd Redlich-Kister orders, so this keeps the energy pycalphad evaluates for
        # add (and Thermo-Calc then reads the same energy)
        ptype, phn, diff, subls, order = _split_raw_name(q.raw_name)
        srt = [sorted(sl, key=lambda x: x.strip("%")) for sl in subls]
        if srt != subls:
            from .merge import _build_raw_name
            odd = int(order or 0) % 2 == 1 and any(len(sl) == 2 and sl != o for sl, o in zip(srt, subls))
            q.raw_name = _build_raw_name(ptype, phn, diff, srt, order)
            q.key, q.sign, q.order_map = parse_param_name(q.raw_name)
            if odd:
                resorted.append(f"{p.raw_name} written as {q.raw_name} (same value: pycalphad's reading of "
                                f"{add_label}; Thermo-Calc reads the original with the opposite sign)")
        B.parameters.append(q)
        imported.append(q.key.label())
    rep.update({"imported": imported, "skipped": skipped, "collisions": collisions, "functions": fn_notes})
    rep["notes"] += resorted
    # ---- type definitions
    imag = any(p.key.ptype in ts.MAGNETIC_PTYPES for p in to_add)
    tb_def, td_def = magnetic_typedef(B, P), magnetic_typedef(D, P)
    if imag and td_def is not None:
        if tb_def is None:
            code = _free_code(B)
            B.typedefs[code] = TypeDef(code, td_def.text)
            pb.type_codes = (pb.type_codes or "%") + code
            rep["notes"].append(f"{add_label}'s MAGNETIC type definition ({td_def.text}) attached to {P} as "
                                f"'{code}' (the base phase had none; its own GM is unchanged: TC = BMAGN = 0 there)")
        elif tb_def.canonical(P) != td_def.canonical(P):
            rep["notes"].append(f"{base_label}'s MAGNETIC type definition kept ({tb_def.text}); {add_label}'s "
                                f"({td_def.text}) differs - the imported TC/BMAGN use the base's AFM / structure "
                                f"factors (the {add_label} GM check shows the effect)")
    # ---- tentative parameters: base terms whose pair add describes go; new mixings get TENT_MIX / TENT_VA terms
    _extension_tentative(B, D, P, base_cons, add_tent_pairs, cfg, rep, base_label, add_label)
    # ---- end-members without G
    rep["missing_g"] = missing_endmembers(B, P, comps, D, base_label, add_label, base_cons)
    rep["phase_after"] = ts._subl_text(pb)
    return B, rep


def _tent_pairs(db: TDB, phase: str) -> set:
    return {pr for p in db.parameters if p.key.phase == phase and is_tentative(p, db)
            for pr in [tent.pair_of(p.key.constituents)] if pr is not None}


def _extension_tentative(B: TDB, D: TDB, P: str, base_cons, add_tent_pairs, cfg, rep: Dict[str, Any],
                         base_label: str, add_label: str) -> None:
    """Tentative rules on the extended phase: the base (its constituents before the extension) and add (its own
    constituents, in the base's sublattice order) are the sources."""
    srcs = [tent.Source(base_label, [list(c) for c in base_cons], _tent_pairs(B, P)),
            tent.Source(add_label, [list(c) for c in D.phases[P].constituents], set(add_tent_pairs))]
    desc = tent.described(srcs)
    keep = []
    for p in B.parameters:
        if p.key.phase == P and p.key.ptype == "G" and is_tentative(p, B):
            pr = tent.pair_of(p.key.constituents)
            if pr is not None and pr in desc:
                rep["tentative_replaced"].append(tent.Replacement(p.key.label(), tent.ref_key(p), base_label,
                                                                  ", ".join(desc[pr]), "described"))
                continue
        keep.append(p)
    B.parameters = keep
    if cfg is not None:
        imported = set(rep.get("imported") or [])
        origin = {p.key.as_tuple(): (add_label if p.key.label() in imported else base_label)
                  for p in B.parameters if p.key.phase == P}
        terms, notes = tent.tdb_phase_terms(B, P, srcs, cfg, origin=origin)
        lo, hi = B.temp_limits
        for t in terms:
            t.source_label = f"{base_label} + {add_label}"
            k2, sign, _ = parse_param_name(t.raw_name)
            B.parameters.append(Parameter(t.raw_name, k2, sign, None, t.piecewise(lo, max(hi, 6000.0))))
            rep["tentative"].append(t)
        rep["tentative_notes"] += notes
    if rep["tentative"] or rep["tentative_replaced"]:
        tent.refresh_references(B, [Path(x.path).stem if x.path and x.path != "<string>" else lab
                                    for x, lab in ((B, base_label), (D, add_label))])


def _base_wins(B: TDB, D: TDB, P: str, rest: List[Parameter], bc: Dict[str, Any], base_label: str,
               add_label: str) -> List[Dict[str, Any]]:
    T = np.linspace(298.15, 2900.0, 40)
    addk = {p.key.as_tuple(): p for p in rest}
    out = []
    for bp in [p for p in B.parameters if p.key.phase == P]:
        kt = bp.key.as_tuple()
        els = sorted(_param_elements(B, bp))
        ap = addk.pop(kt, None)
        if ap is None:
            out.append({"parameter": bp.key.label(), "elements": els, "text": f"{bp.key.label()} only in {base_label}"})
            continue
        try:
            fb = bc["k_base"] if bp.key.ptype in ts.SCALED_PTYPES else 1.0
            fa = bc["k_add"] if ap.key.ptype in ts.SCALED_PTYPES else 1.0
            d = float(np.nanmax(np.abs(B.evaluate(bp.expr, T) * bp.sign * fb - D.evaluate(ap.expr, T) * ap.sign * fa)))
        except Exception:  # noqa: BLE001
            d = math.nan
        if not (d <= 1e-6):
            out.append({"parameter": bp.key.label(), "elements": els, "max_diff": d,
                        "text": f"{bp.key.label()} {base_label}'s kept (differs from {add_label}'s by up to "
                                f"{d:.4g} in the common basis)"})
    for kt, ap in addk.items():
        out.append({"parameter": ap.key.label(), "elements": sorted(_param_elements(D, ap)),
                    "text": f"{ap.key.label()} only in {add_label} (no component: not imported)"})
    return out


def missing_endmembers(B: TDB, P: str, comps, D: TDB, base_label="base", add_label="add",
                       base_cons: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
    """End-members of the extended phase with a component and no G parameter (pycalphad uses 0), each with whether
    it matters: one the add (or base) description already had without G changes nothing; a new combination of base
    and add constituents does."""
    from .merge import _split_raw_name
    ph = B.phases[P]
    have = set()
    for p in B.parameters:
        if p.key.phase == P and p.key.ptype == "G":
            sub = [tuple(s.strip("%") for s in sl) for sl in _split_raw_name(p.raw_name)[3]]
            if all(len(s) == 1 for s in sub):
                have.add(tuple(s[0] for s in sub))
    dph = D.phases.get(P)
    out = []
    for em in itertools.product(*ph.constituents):
        if not any(_species_hits(B, s, comps) for s in em) or all(s == "VA" for s in em):
            continue
        if em in have or any(all(x == "*" or x == e for x, e in zip(h, em)) for h in have):
            continue
        in_add = dph is not None and all(s in dph.constituents[t] for t, s in enumerate(em))
        in_base = base_cons is not None and all(s in base_cons[t] for t, s in enumerate(em))
        els = sorted(ts.endmember_elements(ts.StageView(B, "x", elements=sorted(set(B.real_elements))), em))
        out.append({"end_member": ":".join(em), "elements": els, "in_add_space": in_add,
                    "in_base_space": in_base, "matters": not (in_add or in_base),
                    "text": (f"{P} {':'.join(em)}: no G (pycalphad uses 0) - "
                             + (f"{add_label} had no G for it either (its description is unchanged by this)"
                                if in_add else
                                f"{base_label} already had it without G (unchanged by this)" if in_base else
                                f"a new combination of {base_label} and {add_label} constituents: affects the "
                                f"{'-'.join(e.capitalize() for e in els)} subsystem; MATTERS if {P} can become "
                                f"stable there - give G or check the diagram"))})
    return out


def _check_extension(base: TDB, add: TDB, new: TDB, rep: Dict[str, Any], n: int = 400) -> Dict[str, Any]:
    """GM checks: base restricted to base elements unchanged; add reproduced on its subsystems with the
    components."""
    P, Q, comps = rep["base_phase"], rep["add_phase"], rep["components"]
    dbn = pyc_db(new)
    base_comps = phase_comps(base, P)
    out = {"base": gm_compare(pyc_db(base), P, dbn, P, base_comps, n=n)}
    out["base"]["comps"] = [c for c in base_comps if c != "VA"]
    add_els = sorted(set(phase_comps(add, Q)) - {"VA"})
    others = [e for e in add_els if e not in comps]
    subsets = [()] + [(e,) for e in others]
    if len(others) > 1:
        subsets.append(tuple(others))
    a_to_b = [rep["new_from_old"].index(i) for i in range(len(rep["new_from_old"]))]
    dba = pyc_db(add)
    checks = []
    for sub in subsets:
        cs = sorted(set(comps) | set(sub)) + ["VA"]
        r = gm_compare(dba, Q, dbn, P, cs, a_to_b, n=n)
        if r.get("n", 0) == 0:
            continue                      # the phase cannot exist in this subsystem (or no common states)
        r["system"] = "-".join(e.capitalize() for e in cs if e != "VA")
        why = [w["text"] for w in rep.get("base_wins", []) if set(w["elements"]) <= set(cs)]
        r["expected_exact"] = not why and not (rep.get("rescale_add") or {}).get("magnetic")
        r["why_not_exact"] = why
        checks.append(r)
    out["add"] = checks
    return out


def extend_phase(base_tdb, add_tdb, base_phase: str, add_phase: Optional[str] = None,
                 components: Optional[Sequence[str]] = None, sublattice_map=None, normalization: str = "base",
                 collision: str = "base", magnetic: str = "auto", include_functions: bool = True,
                 output_tdb: Optional[str] = None, add_constituents: Optional[Dict] = None,
                 verify: bool = True, tentative=True) -> Dict[str, Any]:
    """Keep ``base_phase`` of ``base_tdb`` authoritative and introduce ``components`` (default: the elements add's
    phase has and the base's lacks) from ``add_tdb``'s ``add_phase`` (default the same name): see the module
    docstring.  Writes ``output_tdb`` (the base database with the extended phase) and returns the report with the
    GM checks (``checks['base']``: the base phase on base elements; ``checks['add']``: add's subsystems).
    ``tentative``: True (default) | False | {'mixing_per_atom': 50000, 'vacancy': 200000} (calphad_tools.tentative)."""
    base = base_tdb if isinstance(base_tdb, TDB) else parse_tdb(str(base_tdb))
    add = add_tdb if isinstance(add_tdb, TDB) else parse_tdb(str(add_tdb))
    new, rep = extend_phase_db(base, add, base_phase, add_phase, components, sublattice_map, normalization,
                               collision, magnetic, include_functions, add_constituents, tentative=tentative)
    if verify:
        rep["checks"] = _check_extension(base, add, new, rep)
    if output_tdb:
        ts.write_tdb(new, str(output_tdb), [f"{base.name} with {rep['base_phase']} extended by "
                                            f"{', '.join(rep['components'])} from {add.name}'s {rep['add_phase']} "
                                            f"(calphad_tools.harmonize.extend_phase):"] + report_lines(rep)[:12])
        rep["output"] = str(output_tdb)
        from .merge import validate
        rep["load"] = validate(str(output_tdb))
        rep["validator"] = external_validation(base, str(output_tdb), [rep["base_phase"]],
                                               ["only affected phases changed", "base subsystems unchanged"],
                                               {rep["base_phase"]: rep["rescale_base"]["k"]})
    rep["tdb"] = new
    rep["text"] = "\n".join(report_lines(rep))
    return rep


def external_validation(before, after, phases: Sequence[str], expect: Sequence[str],
                        scale: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """The read-only TDB edit validator (calphad_tools.tdb_validate.validate_edit), when installed: {'verdict',
    'text', ...} or {'error'}; None without it."""
    try:
        from .tdb_validate import validate_edit
    except ImportError:
        return None
    try:
        r = validate_edit(before, after, affected_phases=list(phases), expect=list(expect),
                          scale={k: v for k, v in (scale or {}).items() if abs(v - 1.0) > 1e-15} or None)
        return {k: r.get(k) for k in ("verdict", "text", "checks", "changed_phases")}
    except Exception as e:  # noqa: BLE001 - never fatal
        return {"verdict": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}", "text": str(e)[:200]}


def validator_line(v: Optional[Dict[str, Any]], expected_fail: Sequence[str] = ()) -> Optional[str]:
    """One line; checks named in ``expected_fail`` (the requested edit does that on purpose) are marked so."""
    if not v:
        return None
    if v.get("error"):
        return f"validate_edit: not run ({v['error']})"
    fails = [c for c in (v.get("checks") or []) if c.get("status") == "fail"]
    unexpected = [c for c in fails if c.get("check") not in expected_fail]
    verdict = v.get("verdict") if unexpected or not fails else "PASS except the intended change"
    return (f"validate_edit: {verdict}"
            + (": " + "; ".join(f"{c['check']}{' (intended)' if c.get('check') in expected_fail else ''}: "
                                f"{str(c.get('detail', ''))[:120]}" for c in fails[:3]) if fails else ""))


def base_label_of(rep: Dict[str, Any]) -> str:
    return rep.get("base_label", "base")


def report_lines(rep: Dict[str, Any]) -> List[str]:
    P, bl, al = rep["base_phase"], rep["base_label"], rep["add_label"]
    L = [f"{P}: {bl}'s phase extended with {', '.join(rep['components'])} from {al}'s {rep['add_phase']} -> "
         f"{rep.get('phase_after', '')}", rep["mapping"], "basis: " + rep["basis"]["text"]]
    for key in ("rescale_base", "rescale_add"):
        r = rep.get(key)
        if r and abs(r["k"] - 1) > 1e-15 and r.get("text"):
            L.append("rescaled: " + r["text"])
    cons = "; ".join(f"sublattice {t + 1} +{','.join(v)}" for t, v in rep["added_constituents"].items())
    L.append(f"constituents added: {cons}")
    L.append(f"imported {len(rep['imported'])} parameter(s): " + ", ".join(rep["imported"][:8])
             + (f", +{len(rep['imported']) - 8} more" if len(rep["imported"]) > 8 else ""))
    if rep["collisions"]:
        L.append("collisions: " + "; ".join(f"{c['parameter']} kept {c['kept']}'s (max |Δ| {c['max_diff']:.4g})"
                                            for c in rep["collisions"]))
    L += tent.summary_lines(rep.get("tentative") or [], rep.get("tentative_replaced") or [],
                            rep.get("tentative_notes") or [])
    if rep["skipped"]:
        L.append("not imported: " + "; ".join(rep["skipped"][:4]))
    L += rep["functions"][:4] + rep["notes"]
    vl = validator_line(rep.get("validator"))
    if vl:
        L.append(vl)
    for m in rep["missing_g"]:
        L.append(m["text"])
    ch = rep.get("checks")
    if ch:
        L.append(f"GM {bl}'s {P} on {'-'.join(e.capitalize() for e in ch['base'].get('comps', []))} (unchanged?): "
                 + gm_text(ch["base"], exact=not (rep.get("rescale_base") or {}).get("magnetic")))
        for r in ch["add"]:
            L.append(f"GM {al}'s {rep['add_phase']} reproduced on {r.get('system', '?')}: "
                     + gm_text(r, exact=r.get("expected_exact", False))
                     + (f" (states without {', '.join(r['excluded'][:3])})" if r.get("excluded") else "")
                     + (f" - expected, {base_label_of(rep)}'s parameters govern: {'; '.join(r['why_not_exact'][:2])}"
                        if r.get("why_not_exact") else ""))
    return L


# =============================================================================================== specs / staged
def _stoich(name: str) -> Optional[Tuple[float, ...]]:
    """Formula of a phase name as sorted proportions: CET2 -> (1, 2), CE2T17 / RE2TM17 -> (2, 17), Laves -> (1, 2)."""
    parts = [x for x in re.split(r"[_#\-]", name.upper()) if x]
    if set(parts) & {"LAVES", "C14", "C15", "C36"}:
        return (1.0 / 3.0, 2.0 / 3.0)
    for part in parts:
        toks = re.findall(r"(TM|RE|[A-Z][A-Z]?)(\d*)", part)
        if len(toks) < 2 or "".join(a + b for a, b in toks) != part:
            continue
        if not all(a in _ELEMENTS or a in ("T", "TM", "RE", "R", "M", "X") for a, _ in toks):
            continue
        n = [float(b) if b else 1.0 for _, b in toks]
        s = sum(n)
        return tuple(sorted(x / s for x in n))
    return None


_ELEMENTS = set("""H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO NI CU ZN GA GE AS SE BR KR RB
SR Y ZR NB MO TC RU RH PD AG CD IN SN SB TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS IR
PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM""".split())


def _ratio_sig(ph) -> Tuple[float, ...]:
    r = [ph.site_ratios[i] for i in _real(ph)]
    s = sum(r)
    return tuple(sorted(x / s for x in r))


def harmonization_candidates(A: TDB, B: TDB) -> List[Dict[str, Any]]:
    """B phases (not in A) that look like an A phase (not in B) with extra constituents: same formula from the
    names (CET2 ~ C15_LAVES ~ CEFE2: 1:2; CE2T17 ~ RE2TM17: 2:17) or shared structure tokens, proportional site
    ratios, a shared element and elements A's phase lacks.  Suggestions only: the user confirms."""
    out = []
    for b in sorted(B.phases):
        if b in A.phases or ts._order_disorder_related(B, b):
            continue
        pb = B.phases[b]
        eb = set(phase_comps(B, b)) - {"VA"}
        for a in sorted(A.phases):
            if a in B.phases or ts._order_disorder_related(A, a):
                continue
            pa = A.phases[a]
            ea = set(phase_comps(A, a)) - {"VA"}
            sa, sb = _stoich(a), _stoich(b)
            same_name = bool(ts.phase_tokens(a) & ts.phase_tokens(b))
            same_formula = sa is not None and sb is not None and all(abs(x - y) < 1e-6 for x, y in zip(sa, sb)) \
                and len(sa) == len(sb)
            if not (same_name or same_formula):
                continue
            if len(_real(pa)) != len(_real(pb)) or not (ea & eb) or not (eb - ea):
                continue
            ra, rb = _ratio_sig(pa), _ratio_sig(pb)
            if any(abs(x - y) > 1e-3 for x, y in zip(ra, rb)):
                continue
            if sa is not None and len(sa) == len(ra) and any(abs(x - y) > 1e-3 for x, y in zip(sa, ra)):
                continue
            try:
                ok = ts.sublattice_mapping(pa, pb) is not None
                why_map = "" if ok else "site ratios not proportional"
            except ts.SublatticeMapError as e:
                ok, why_map = True, f"give sublattice_map ({str(e).split(';')[0]})"
            if not ok:
                continue
            out.append({"a_phase": a, "b_phase": b, "components": sorted(eb - ea),
                        "why": "same formula from the names" if same_formula else "shared structure tokens",
                        "a_model": ts._subl_text(pa), "b_model": ts._subl_text(pb),
                        "a_magnetic": is_magnetic(A, a), "b_magnetic": is_magnetic(B, b), "map_note": why_map})
    return out


_YAML_KEYS = {"source_phase": "phase", "add_phase": "phase", "phase": "phase", "from": "from",
              "target_phase": "target", "sublattice_map": "sublattice_map", "add_constituents": "add_constituents",
              "components": "components", "normalization": "normalization", "collision": "collision",
              "magnetic": "magnetic", "include_functions": "include_functions",
              "site_ratio_normalization": "site_ratio_normalization", "structure_from": "structure_from",
              "scale_parameters_with_formula_unit": "scale_parameters_with_formula_unit",
              "magnetic_model": "magnetic_model", "import_parameters": "import_parameters", "step": "step",
              "source": "source"}


def is_extend(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "extend"
    return isinstance(value, dict) and ("extend" in value or "source_phase" in value)


def extension_spec(target: str, value, A: Optional[TDB] = None, B: Optional[TDB] = None,
                   default_phase: Optional[str] = None) -> Dict[str, Any]:
    """Normalised extension spec from a decision value ('extend', {'extend': {...}}) or a phase_harmonization entry
    (YAML shape: source_phase, sublattice_map, add_constituents, import_parameters.require_components,
    site_ratio_normalization, magnetic_model, ...)."""
    raw = {}
    if isinstance(value, dict):
        raw = value.get("extend") if isinstance(value.get("extend"), dict) else {
            k: v for k, v in value.items() if k != "extend"}
    raw = raw or {}
    unknown = [k for k in raw if k not in _YAML_KEYS]
    if unknown:
        raise HarmonizeError(f"extend {target}: unknown key(s) {', '.join(unknown)}")
    spec: Dict[str, Any] = {"target": target.upper(), "phase": None, "components": None, "sublattice_map": None,
                            "add_constituents": None, "normalization": "base", "collision": "base",
                            "magnetic": "auto", "include_functions": True}
    for k, v in raw.items():
        key = _YAML_KEYS[k]
        if key == "phase":
            spec["phase"] = str(v).upper()
        elif key == "target":
            if str(v).upper() != target.upper():
                raise HarmonizeError(f"extend {target}: target_phase {v} differs from the key {target}")
        elif key == "from":
            if str(v).lower() not in ("db2", "add", "b"):
                raise HarmonizeError(f"extend {target}: from={v!r} - the added database (db2) is the source; the base "
                                     f"phase is always the one extended")
        elif key == "structure_from":
            if str(v).lower() not in ("base", "db1", "a"):
                raise HarmonizeError(f"extend {target}: structure_from={v!r} - only 'base' (the base's sublattice "
                                     f"order, constituents and names govern)")
        elif key == "scale_parameters_with_formula_unit":
            if v is False:
                raise HarmonizeError(f"extend {target}: parameters always scale with the formula unit (otherwise "
                                     f"GM per atom changes)")
        elif key == "site_ratio_normalization":
            s = str(v).lower()
            spec["normalization"] = {"sum_to_one": "per_atom", "per_atom": "per_atom", "base": "base"}.get(s)
            if spec["normalization"] is None:
                raise HarmonizeError(f"extend {target}: site_ratio_normalization {v!r}: 'base' or 'sum_to_one'")
        elif key == "magnetic_model":
            s = str(v).lower()
            spec["magnetic"] = {"source": "auto", "add": "auto", "auto": "auto", "magnetic_side": "auto"}.get(s, s)
        elif key == "import_parameters":
            if isinstance(v, dict) and v.get("require_components"):
                spec["components"] = [str(x).upper() for x in v["require_components"]]
        elif key == "components":
            spec["components"] = [str(x).upper() for x in ([v] if isinstance(v, str) else v)]
        elif key == "add_constituents":
            spec["add_constituents"] = {int(kk): ([vv] if isinstance(vv, str) else list(vv)) for kk, vv in v.items()}
        elif key in ("step", "source"):
            spec[key] = v
        else:
            spec[key] = v
    if spec["normalization"] not in ("base", "per_atom"):
        raise HarmonizeError(f"extend {target}: normalization {spec['normalization']!r}")
    if spec["magnetic"] not in MAGNETIC_MODES:
        raise HarmonizeError(f"extend {target}: magnetic {spec['magnetic']!r}: one of {', '.join(MAGNETIC_MODES)}")
    if spec["add_constituents"] and not spec["components"]:
        spec["components"] = sorted({e for sps in spec["add_constituents"].values() for s in sps
                                     for e in (ts._species_els(B, s) if B is not None else {s.upper()})} - {"VA"})
    if spec["phase"] is None:
        spec["phase"] = (default_phase or target).upper()
    if A is not None and spec["target"] not in A.phases:
        raise HarmonizeError(f"extend: {spec['target']} is not a phase of the base {A.name}")
    if B is not None and spec["phase"] not in B.phases:
        raise HarmonizeError(f"extend {spec['target']}: {spec['phase']} is not a phase of {B.name} (give 'phase': "
                             f"the added database's name)")
    return spec


def _drop_phases(db: TDB, phases: Sequence[str]) -> TDB:
    out = copy.deepcopy(db)
    drop = {p.upper() for p in phases}
    codes = {c for p in drop if p in out.phases for c in out.phases[p].type_codes}
    for p in drop:
        out.phases.pop(p, None)
    out.parameters = [p for p in out.parameters if p.key.phase not in drop]
    still = {c for ph in out.phases.values() for c in ph.type_codes}
    for c in codes - still:
        out.typedefs.pop(c, None)
    return out


def staged_extensions(a_eff: str, b_eff: str, specs: Dict[str, Dict], out_dir: str, shared: Sequence[str],
                      verify: bool = True, a_label: str = "A", b_label: str = "B",
                      progress=lambda s: None, tentative=True) -> Dict[str, Any]:
    """An ``extend`` decision inside staged_merge / combine_databases: A's phase extended from B2 (B with A's SER and
    unaries of the shared elements, as the staged merge builds it - so imported parameters use the base's unary
    functions), written as ``A0x_<A>_extended.tdb``; B without the source phases for the merge
    (``B0x_<B>_for_merge.tdb``).  Returns {'A_ext', 'B_merge', 'reports', 'renames' {B phase: A phase}}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    A, Bn = parse_tdb(str(a_eff)), parse_tdb(str(b_eff))
    VA, VB = ts.StageView(A, "A"), ts.StageView(Bn, "B")
    B1, _ = ts.substitute_ser(Bn, VA, VB, list(shared))
    B2, _ = ts.substitute_unaries(B1, VA, ts.StageView(B1, "B1"), list(shared))
    cur = A
    reports, renames = [], {}
    for target, spec in specs.items():
        progress(f"harmonise: extend {target} from {b_label}'s {spec['phase']}")
        new, rep = extend_phase_db(cur, B2, spec["target"], spec["phase"], spec.get("components"),
                                   spec.get("sublattice_map"), spec.get("normalization", "base"),
                                   spec.get("collision", "base"), spec.get("magnetic", "auto"),
                                   spec.get("include_functions", True), spec.get("add_constituents"),
                                   base_label=a_label, add_label=b_label, tentative=tentative)
        if verify:
            rep["checks"] = _check_extension(cur, B2, new, rep)
            rep["validator"] = external_validation(cur, new, [rep["base_phase"]],
                                                   ["only affected phases changed", "base subsystems unchanged"],
                                                   {rep["base_phase"]: rep["rescale_base"]["k"]})
        rep["text"] = "\n".join(report_lines(rep))
        rep["spec"] = spec
        reports.append(rep)
        if spec["phase"] != spec["target"]:
            renames[spec["phase"]] = spec["target"]
        cur = new
    sa, sb = Path(a_eff).stem, Path(b_eff).stem
    a_ext = ts.write_tdb(cur, str(out / f"A0x_{sa}_extended.tdb"),
                         [f"{Path(a_eff).name} with phase(s) extended from {Path(b_eff).name} (decision 'extend'):"]
                         + [ln for r in reports for ln in report_lines(r)[:6]])
    b_merge = ts.write_tdb(_drop_phases(Bn, [s["phase"] for s in specs.values()]),
                           str(out / f"B0x_{sb}_for_merge.tdb"),
                           [f"{Path(b_eff).name} without {', '.join(s['phase'] for s in specs.values())} (taken into "
                            f"the base's extended phase(s) by the 'extend' decision)"])
    for r in reports:
        r.pop("tdb", None)
    return {"A_ext": a_ext, "B_merge": b_merge, "reports": reports, "renames": renames}


def jsonable_report(rep: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("base_phase", "add_phase", "components", "mapping", "new_from_old", "added_constituents", "imported",
            "skipped", "collisions", "functions", "notes", "missing_g", "phase_after", "text", "spec", "validator")
    out = {k: rep.get(k) for k in keep}
    out["basis"] = {k: v for k, v in rep["basis"].items()}
    out["added_constituents"] = {str(k): v for k, v in (rep.get("added_constituents") or {}).items()}
    ch = rep.get("checks") or {}
    out["checks"] = {"base": ch.get("base"), "add": ch.get("add")}
    for key in ("rescale_base", "rescale_add"):
        r = rep.get(key) or {}
        out[key] = {k: r.get(k) for k in ("k", "k_text", "old", "new", "scaled", "magnetic", "text")}
    out["tentative"] = [t.text() if hasattr(t, "text") else str(t) for t in rep.get("tentative") or []]
    out["tentative_replaced"] = [r.text() if hasattr(r, "text") else str(r) for r in rep.get("tentative_replaced") or []]
    out["tentative_notes"] = list(rep.get("tentative_notes") or [])
    return out


# =============================================================================================== tools
def register(tb) -> None:
    from .staged_merge import TENTATIVE_SCHEMA, tentative_arg
    S = {"type": "string"}

    @tb.add("normalize_phase_formula_unit",
            "Change phases' formula unit, GM per atom kept. Site ratios x k; G/L x k range by range, TC kept, "
            "checked. target 'sum_to_one' (exact decimals) | 'integers' | [ratios] | match_tdb's phase. Magnetic "
            "phases: no exact rescale - magnetic 'auto'/'refuse' errors with the size of the error, "
            "'keep_parameters' or 'refit' (BMAGN) report it; rescale the non-magnetic side instead.",
            {"tdb": S, "phases": {"type": "array", "items": S},
             "target": {"anyOf": [S, {"type": "array", "items": {"type": "number"}}]},
             "output_tdb": S, "magnetic": {"type": "string", "enum": list(MAGNETIC_MODES)},
             "match_tdb": S, "match_phase": S}, ["tdb", "phases"])
    def normalize_tool(tdb, phases, target="sum_to_one", output_tdb=None, magnetic="auto", match_tdb=None,
                       match_phase=None):
        path = tb._find_tdb(tdb)
        out = Path(tb.p(output_tdb)) if output_tdb else tb.out / f"{Path(path).stem}_fu.tdb"
        r = normalize_phase_formula_unit(path, phases, target, str(out), magnetic,
                                         tb._find_tdb(match_tdb) if match_tdb else None, match_phase)
        return r["text"] + f"\nFiles: {tb.rel(r['output'])}", []

    @tb.add("extend_phase",
            "Extend a phase with another TDB's constituents. Keeps base_tdb's base_phase authoritative and adds "
            "components (e.g. ['CO']) from add_tdb's add_phase (names may differ): sublattices mapped (or sublattice_map), formula unit harmonised (the MAGNETIC "
            "side's basis when only one is magnetic, exact), only parameters with a component imported, collisions "
            "by policy, FUNCTIONs copied/renamed, GM checks (base unchanged, add reproduced), end-members without G "
            "listed.",
            {"base_tdb": S, "add_tdb": S, "base_phase": S, "add_phase": S, "components": {"type": "array", "items": S},
             "sublattice_map": {"type": "array", "items": {"type": "integer"}},
             "normalization": {"type": "string", "enum": ["base", "per_atom"]},
             "collision": {"type": "string", "enum": ["base", "add", "error"]},
             "magnetic": {"type": "string", "enum": list(MAGNETIC_MODES)}, "include_functions": {"type": "boolean"},
             "add_constituents": {"type": "object"}, "output_tdb": S, "tentative": TENTATIVE_SCHEMA},
            ["base_tdb", "add_tdb", "base_phase"])
    def extend_tool(base_tdb, add_tdb, base_phase, add_phase=None, components=None, sublattice_map=None,
                    normalization="base", collision="base", magnetic="auto", include_functions=True,
                    add_constituents=None, output_tdb=None, tentative=None):
        a, b = tb._find_tdb(base_tdb), tb._find_tdb(add_tdb)
        out = Path(tb.p(output_tdb)) if output_tdb else tb.out / f"{Path(a).stem}_{base_phase.upper()}_extended.tdb"
        r = extend_phase(a, b, base_phase, add_phase, components, sublattice_map, normalization, collision,
                         magnetic, include_functions, str(out), add_constituents,
                         tentative=tentative_arg(tb, tentative))
        errs = (r.get("load") or {}).get("errors") or []
        return r["text"] + (f"\npycalphad: {'; '.join(errs[:2])}" if errs else "") + \
            f"\nFiles: {tb.rel(r['output'])}", []
