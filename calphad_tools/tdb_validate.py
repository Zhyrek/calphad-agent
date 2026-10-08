"""tdb_validate: deterministic checks that an edited TDB is sound and changed only what it was meant to change.

``validate_edit(before, after, ...)`` (importable; the ``validate_tdb_edit`` tool wraps it) runs, each pass / fail
with numbers:

* pycalphad loads ``after``;
* parse round trip: written with the toolkit's TDB writer (``tdbstage.tdb_to_text``), re-parsed, same parameters,
  functions, phases and type definitions;
* constituents, function references (plus circular references), type definitions and duplicates - the
  ``tdb_checks.check_syntax`` findings of ``after``, split into new ones (fail) and ones ``before`` already had;
* site ratios: a phase whose atoms per formula unit were exactly 1 before still are (or every affected phase, with
  the expectation "site ratios sum to 1");
* phases outside ``affected_phases`` (default: every phase whose model or parameters changed) are identical:
  same model, same parameters, numerically with every FUNCTION inlined;
* GM per mole of atoms of every affected phase, before vs after, at random T and site fractions mapped through the
  sublattice permutation (detected from the models, or ``sublattice_map``) and proportional rescale (``scale``).
  Relative 1e-9 = preserved.  The magnetic (IHJ) term is per formula unit and nonlinear in BMAGN, so a rescaled
  magnetic phase cannot be preserved exactly: its deviation is reported separately ("magnetic contribution changed
  by X J/mol of atoms") instead of failing the phase.

Expectations (``expect``): "only affected phases changed" (nothing added or removed elsewhere either), "base
subsystems unchanged" (everything ``before`` had is unchanged outside the affected phases and their GM is preserved
on ``before``'s constituents), "energy_per_atom_preserved" (GM check must pass), "site ratios sum to 1".
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .tdb_query import (Text, _cut, _empty_view, atoms_per_fu, dis_part_of, inline_parameter, magnetic_of,
                        match_sublattices, model_text, write_tables)
from .tdbparse import TDB, parse_tdb
from .tdbtables import segs_close

EXPECTATIONS = {"only affected phases changed": "only_affected", "base subsystems unchanged": "base_unchanged",
                "energy per atom preserved": "gm_preserved", "site ratios sum to 1": "sum_one"}
REL_TOL = 1e-9
EPS_Y = 1e-12                  # site fraction given to constituents that only ``after`` has
CHECK_GROUPS = {               # tdb_checks.check_syntax codes -> validate_edit check
    "constituents": ("param_constituent_not_in_phase", "param_sublattice_mismatch", "param_undefined_phase",
                     "undeclared_species", "phase_no_constituents", "constituent_undefined_phase"),
    "functions": ("undefined_function",),
    "type definitions": ("typedef_missing_phase", "dis_part_missing_phase", "dis_part_mismatch", "dis_part_unused",
                         "typedef_undefined_code"),
    "duplicates": ("duplicate_function", "duplicate_parameter"),
}


def _expectations(expect) -> set:
    out = set()
    for e in ([expect] if isinstance(expect, str) else list(expect or [])):
        k = " ".join(str(e).lower().replace("_", " ").replace("-", " ").split())
        k = {"energy per atom preserved": "energy per atom preserved", "gm preserved": "energy per atom preserved",
             "site ratios sum to one": "site ratios sum to 1"}.get(k, k)
        if k not in EXPECTATIONS:
            raise ValueError(f"unknown expectation {e!r}; use {list(EXPECTATIONS)}")
        out.add(EXPECTATIONS[k])
    return out


def _as_path(x: Union[str, TDB], tmpdir: str, tag: str) -> str:
    if isinstance(x, TDB):
        from .tdbstage import tdb_to_text
        p = os.path.join(tmpdir, f"{tag}.tdb")
        Path(p).write_text(tdb_to_text(x))
        return p
    return str(x)


def _pyc_db(path: str):
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(path)


# =============================================================================================== structural checks
def check_round_trip(tdb: TDB) -> Dict[str, Any]:
    try:
        from .tdbstage import tdb_to_text
    except Exception as e:  # noqa: BLE001 - the writer module may be mid-edit
        return {"status": "skip", "detail": f"toolkit TDB writer not importable ({type(e).__name__})"}
    back = parse_tdb(tdb_to_text(tdb), is_text=True, name="roundtrip")

    def pmap(t: TDB):
        return {(p.key.ptype, p.key.phase, p.key.constituents, p.key.order, p.key.diffusing):
                (p.sign, p.expr.canonical()) for p in t.parameters}
    a, b = pmap(tdb), pmap(back)
    bad_p = sorted({str(k) for k in set(a) ^ set(b)} | {str(k) for k in set(a) & set(b) if a[k] != b[k]})
    fa = {n: f.expr.canonical() for n, f in tdb.functions.items()}
    fb = {n: f.expr.canonical() for n, f in back.functions.items()}
    bad_f = sorted(n for n in set(fa) | set(fb) if fa.get(n) != fb.get(n))
    ph = {n: (p.site_ratios, p.constituents, p.type_codes) for n, p in tdb.phases.items()}
    phb = {n: (p.site_ratios, p.constituents, p.type_codes) for n, p in back.phases.items()}
    bad_ph = sorted(n for n in set(ph) | set(phb) if ph.get(n) != phb.get(n))
    td = {c: t.canonical() for c, t in tdb.typedefs.items()}
    tdb_ = {c: t.canonical() for c, t in back.typedefs.items()}
    bad_t = sorted(c for c in set(td) | set(tdb_) if td.get(c) != tdb_.get(c))
    n_bad = len(bad_p) + len(bad_f) + len(bad_ph) + len(bad_t)
    detail = (f"{len(a)} parameters, {len(fa)} functions, {len(ph)} phases, {len(td)} type definitions written and "
              f"re-parsed: {n_bad} mismatch(es)")
    if n_bad:
        detail += ": " + "; ".join(x for x in (bad_p and f"parameters {bad_p[:3]}", bad_f and f"functions {bad_f[:5]}",
                                               bad_ph and f"phases {bad_ph[:5]}", bad_t and f"types {bad_t[:5]}") if x)
    return {"status": "fail" if n_bad else "pass", "detail": detail, "mismatches": n_bad}


def _cycles(tdb: TDB) -> List[str]:
    out, state = [], {}

    def visit(n, stack):
        if state.get(n) == 2 or n not in tdb.functions:
            return
        if state.get(n) == 1:
            out.append(" -> ".join(stack[stack.index(n):] + [n]))
            return
        state[n] = 1
        for s in tdb.functions[n].expr.symbols():
            visit(s, stack + [n])
        state[n] = 2
    for n in tdb.functions:
        visit(n, [])
    return out


def syntax_checks(before: TDB, after: TDB, text_b: str, text_a: str) -> List[Dict[str, Any]]:
    from .tdb_checks import Findings, check_syntax
    Fa, Fb = Findings(), Findings()
    check_syntax(after, text_a, Fa)
    check_syntax(before, text_b, Fb)
    old = {(f.code, f.message) for f in Fb}
    rows = []
    for name, codes in CHECK_GROUPS.items():
        hits = [f for f in Fa if f.code in codes]
        new = [f for f in hits if (f.code, f.message) not in old]
        pre = [f for f in hits if (f.code, f.message) in old]
        extra = ""
        if name == "functions":
            cyc = _cycles(after)
            if cyc:
                new += [type("F", (), {"message": f"circular function reference {c}", "severity": "ERROR"})()
                        for c in cyc]
            n_ref = len({s for p in after.parameters for s in p.expr.symbols()}
                        | {s for f in after.functions.values() for s in f.expr.symbols()})
            extra = f"{n_ref} referenced symbols, {len(after.functions)} functions defined"
        elif name == "constituents":
            extra = f"{len(after.parameters)} parameters in {len(after.phases)} phases"
        elif name == "type definitions":
            n_codes = sum(1 for p in after.phases.values() if p.type_codes.strip("%"))
            extra = f"{len(after.typedefs)} type definitions, {n_codes} phases with type codes"
        elif name == "duplicates":
            extra = f"{len(after.parameters)} parameters, {len(after.functions)} functions"
        serious = [f for f in new if f.severity in ("ERROR", "WARNING")]
        status = "fail" if serious else ("warn" if new or pre else "pass")
        msg = "; ".join(_cut(f.message, 140) for f in new[:4])
        detail = extra + (f"; {len(new)} new problem(s): {msg}" if new else "") + (
            f"; {len(pre)} already in before" if pre else "") + ("" if new or pre else ": all OK")
        rows.append({"check": name, "status": status, "detail": detail,
                     "problems": [f.message for f in new], "pre_existing": [f.message for f in pre]})
    return rows


def reversed_odd_order(tdb: TDB) -> List[str]:
    """Interaction parameters written with non-alphabetical constituents where that flips the sign / order
    (odd Redlich-Kister order, ternary order).  Thermo-Calc applies the flip; pycalphad sorts the constituents
    without it, so the two programs evaluate these differently."""
    return [p.raw_name for p in tdb.parameters
            if p.sign != 1.0 or (p.order_map and any(a != b for a, b in p.order_map.items()))]


# =============================================================================================== differences
def _pkey(p, phase: str) -> Tuple:
    return (p.key.ptype, phase, p.key.constituents, p.key.order, p.key.diffusing)


def _klabel(k: Tuple) -> str:
    ptype, phase, cons, order, diff = k
    return f"{ptype}({phase}{'&' + diff if diff else ''},{':'.join(','.join(sl) for sl in cons)};{order})"


def phase_params(tdb: TDB, rename: Dict[str, str]) -> Dict[str, Dict[Tuple, Any]]:
    view = _empty_view(tdb)
    out: Dict[str, Dict[Tuple, Any]] = {}
    for p in tdb.parameters:
        ph = rename.get(p.key.phase, p.key.phase)
        out.setdefault(ph, {})[_pkey(p, ph)] = (p, inline_parameter(view, p))
    return out


def _model_sig(tdb: TDB, name: str, rename: Dict[str, str]):
    ph = tdb.phases[name]
    mag = magnetic_of(tdb, ph)
    dis = dis_part_of(tdb, ph)
    return (tuple(round(r, 12) for r in ph.site_ratios), tuple(frozenset(sl) for sl in ph.constituents),
            mag[:2] if mag else None, rename.get(dis, dis) if dis else None)


def phase_differences(before: TDB, after: TDB, phase_map: Dict[str, str], grid) -> Dict[str, Dict[str, Any]]:
    """after-phase name -> {model: changed?, removed, added, changed: [keys], max_abs}."""
    pb, pa = phase_params(before, phase_map), phase_params(after, {})
    names = sorted({phase_map.get(n, n) for n in before.phases} | set(after.phases) | set(pb) | set(pa))
    inv = {v: k for k, v in phase_map.items()}
    out = {}
    for nm in names:
        bn = inv.get(nm, nm)
        d: Dict[str, Any] = {"in_before": bn in before.phases, "in_after": nm in after.phases, "model": False,
                             "removed": [], "added": [], "changed": [], "max_abs": 0.0, "note": ""}
        if d["in_before"] and d["in_after"]:
            sb, sa = _model_sig(before, bn, phase_map), _model_sig(after, nm, {})
            if sb != sa:
                d["model"] = True
                d["note"] = f"model {model_text(before.phases[bn])} -> {model_text(after.phases[nm])}"
                if sb[2:] != sa[2:]:
                    d["note"] += f"; magnetic/DIS_PART {sb[2:]} -> {sa[2:]}"
        B, A = pb.get(nm, {}), pa.get(nm, {})
        d["removed"] = sorted(set(B) - set(A))
        d["added"] = sorted(set(A) - set(B))
        for k in sorted(set(A) & set(B)):
            (p1, s1), (p2, s2) = B[k], A[k]
            if s1 is None or s2 is None:
                if (p1.sign, p1.expr.canonical()) != (p2.sign, p2.expr.canonical()):
                    d["changed"].append(k)
                continue
            ok, diff = segs_close(s1, s2, grid, REL_TOL, 1e-9)
            if not ok:
                d["changed"].append(k)
                if math.isfinite(diff):
                    d["max_abs"] = max(d["max_abs"], diff)
        d["any"] = bool(d["model"] or d["removed"] or d["added"] or d["changed"] or
                        d["in_before"] != d["in_after"])
        out[nm] = d
    return out


# =============================================================================================== GM per atom
def _sublattice_map(pa, pb, hint=None, k_hint=None) -> Tuple[List[Tuple[Dict[int, int], float]], str]:
    """Candidate maps {before sublattice -> after sublattice} (vacancy-only sublattices left out) with k = after
    sites / before sites, best first, and a note."""
    ia = [i for i in range(pa.n_sublattices) if i not in pa.va_only_sublattices()]
    ib = [i for i in range(pb.n_sublattices) if i not in pb.va_only_sublattices()]
    if hint is not None:
        h = [int(x) for x in hint]
        if len(h) == pa.n_sublattices:
            h = [h[i] for i in ia]
        if len(h) != len(ia) or sorted(h) != sorted(ib):
            raise ValueError(f"sublattice_map {list(hint)}: needs the after-sublattice of each of before's "
                             f"{len(ia)} non-vacancy sublattices {ia} (after's are {ib})")
        ra = [pa.site_ratios[i] for i in ia]
        rb = [pb.site_ratios[j] for j in h]
        k = k_hint if k_hint is not None else sum(rb) / sum(ra)
        bad = [i for i, (x, y) in zip(ia, zip(ra, rb)) if abs(y - k * x) > 1e-5 * abs(k * x)]
        note = "given sublattice_map" + (f"; site ratios not proportional on sublattice(s) {bad}" if bad else "")
        return [(dict(zip(ia, h)), k)], note
    if len(ia) != len(ib):
        return [], f"{len(ia)} vs {len(ib)} non-vacancy sublattices: give sublattice_map"
    ra, rb = [pa.site_ratios[i] for i in ia], [pb.site_ratios[i] for i in ib]
    cands = match_sublattices(ra, rb, [pa.constituents[i] for i in ia], [pb.constituents[i] for i in ib])
    if k_hint is not None:
        cands = [c for c in cands if abs(c[1] - k_hint) <= 1e-6 * abs(k_hint)]
    if not cands:
        return [], (f"site ratios {model_text(pa)} -> {model_text(pb)} not proportional under any permutation with "
                    f"matching constituents: give sublattice_map / scale")
    top = [c for c in cands if c[2] == cands[0][2]]
    note = "" if len(top) == 1 else f"{len(top)} equivalent permutations fit; the best of up to 6 is used"
    return [({ia[t]: ib[p] for t, p in enumerate(perm)}, k) for perm, k, _ in top[:6]], note


def _comps(db, phase: str) -> List[str]:
    ph = db.phases[phase]
    names = {s.name for subl in ph.constituents for s in subl}
    dis = ph.model_hints.get("disordered_phase")
    if dis and dis in db.phases and dis != phase:
        names |= {s.name for subl in db.phases[dis].constituents for s in subl}
    return sorted(names)


def _value(expr, subs) -> float:
    try:
        return float(expr.xreplace(subs))
    except (TypeError, RuntimeError, AttributeError):
        return float(expr) if not hasattr(expr, "free_symbols") or not expr.free_symbols else math.nan


def gm_compare(db_b, db_a, phase_b: str, phase_a: str, smap: Dict[int, int], temps: Sequence[float],
               rng) -> Dict[str, Any]:
    from pycalphad import Model, variables as v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mb, ma = Model(db_b, _comps(db_b, phase_b), phase_b), Model(db_a, _comps(db_a, phase_a), phase_a)
    by_b: Dict[int, List] = {}
    for y in mb.site_fractions:
        by_b.setdefault(int(y.sublattice_index), []).append(y)
    by_a: Dict[int, List] = {}
    for y in ma.site_fractions:
        by_a.setdefault(int(y.sublattice_index), []).append(y)
    for i, ys in by_b.items():
        if i in smap:
            names_a = {y.species.name for y in by_a.get(smap[i], [])}
            lost = [y.species.name for y in ys if y.species.name not in names_a]
            if lost:
                return {"error": f"constituent(s) {lost} of sublattice {i} not in after's sublattice {smap[i]}"}
    if not mb.site_fractions or not ma.site_fractions:
        return {"error": "no site fractions (phase without constituents?)"}
    mag_b, mag_a = mb.models.get("mag"), ma.models.get("mag")
    has_mag = any(m is not None and getattr(m, "free_symbols", None) for m in (mag_b, mag_a))
    worst = {"rel": 0.0, "abs": 0.0, "nonmag_rel": 0.0, "nonmag_abs": 0.0, "mag_abs": 0.0}
    at = None
    for T in temps:
        sb = {v.T: float(T), v.P: 101325.0}
        sa = {v.T: float(T), v.P: 101325.0}
        vals: Dict[Tuple[int, str], float] = {}
        for i, ys in by_b.items():
            w = rng.dirichlet(np.ones(len(ys))) if len(ys) > 1 else np.ones(1)
            for y, wi in zip(ys, w):
                sb[y] = float(wi)
                if i in smap:
                    vals[(smap[i], y.species.name)] = float(wi)
        for j, ys in by_a.items():
            mapped = j in smap.values()
            for y in ys:
                if mapped:
                    sa[y] = vals.get((j, y.species.name), EPS_Y)
                else:                                  # vacancy-only or unmapped sublattice
                    sa[y] = 1.0 if y.species.name == "VA" else EPS_Y
        gb, ga = _value(mb.GM, sb), _value(ma.GM, sa)
        mgb = _value(mag_b, sb) if has_mag and mag_b is not None else 0.0
        mga = _value(mag_a, sa) if has_mag and mag_a is not None else 0.0
        d = abs(ga - gb)
        rel = d / max(abs(gb), 1.0)
        nm = abs((ga - mga) - (gb - mgb))
        if not math.isfinite(rel):
            return {"error": f"GM is not finite at T={T:.6g} K (before {gb:.6g}, after {ga:.6g}): undefined "
                             "function or a value outside the T ranges"}
        if rel >= worst["rel"]:
            at = {"T": round(float(T), 2), "y_before": {f"{y.species.name}@{y.sublattice_index}": round(sb[y], 4)
                                                        for y in mb.site_fractions},
                  "GM_before": gb, "GM_after": ga}
        worst["rel"] = max(worst["rel"], rel)
        worst["abs"] = max(worst["abs"], d)
        worst["nonmag_abs"] = max(worst["nonmag_abs"], nm)
        worst["nonmag_rel"] = max(worst["nonmag_rel"], nm / max(abs(gb), 1.0))
        worst["mag_abs"] = max(worst["mag_abs"], abs(mga - mgb))
    return {**worst, "worst_at": at, "magnetic": has_mag, "n": len(temps)}


# =============================================================================================== main entry
def validate_edit(before: Union[str, TDB], after: Union[str, TDB], affected_phases: Optional[Sequence[str]] = None,
                  expect: Optional[Sequence[str]] = None, temperatures: Optional[Sequence[float]] = None,
                  n_samples: int = 20, sublattice_map: Optional[Dict[str, Sequence[int]]] = None,
                  scale: Optional[Dict[str, float]] = None, phase_map: Optional[Dict[str, str]] = None,
                  output: Optional[str] = None, seed: int = 0, rel=lambda p: p) -> Dict[str, Any]:
    """Validate the edit ``before`` -> ``after`` (TDB paths or parsed tdbparse.TDB objects).

    affected_phases: phases (after's names) allowed to change; default every phase whose model or parameters
    changed.  expect: see EXPECTATIONS.  temperatures: K for the GM samples (cycled; default random 300-3000 K).
    sublattice_map: {phase: [after sublattice of each before sublattice]} (0-based, as on the PHASE line);
    scale: {phase: k} with after's site ratios = k * before's.  phase_map: {before name: after name}.
    Returns {"verdict": "PASS"|"FAIL", "checks": [...], "gm": [...], "changed_phases": [...], "text": ..., "files"}.
    """
    want = _expectations(expect)
    pmap = {k.upper(): v.upper() for k, v in (phase_map or {}).items()}
    smaps = {k.upper(): v for k, v in (sublattice_map or {}).items()}
    scales = {k.upper(): float(v) for k, v in (scale or {}).items()}
    checks: List[Dict[str, Any]] = []
    gm_rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        pb, pa = _as_path(before, tmp, "before"), _as_path(after, tmp, "after")
        tb_, ta = (copy.deepcopy(x) if isinstance(x, TDB) else parse_tdb(x) for x in (before, after))
        text_b, text_a = Path(pb).read_text(errors="replace"), Path(pa).read_text(errors="replace")
        # 1. pycalphad loads after
        db_a = db_b = None
        try:
            db_a = _pyc_db(pa)
            checks.append({"check": "pycalphad loads after", "status": "pass",
                           "detail": f"{len(db_a.phases)} phases, {len(db_a._parameters)} parameters, "
                                     f"{len(db_a.symbols)} symbols"})
        except Exception as e:  # noqa: BLE001
            checks.append({"check": "pycalphad loads after", "status": "fail",
                           "detail": f"{type(e).__name__}: {_cut(str(e), 200)}"})
        new_warn = [w for w in ta.warnings if w not in tb_.warnings]
        if new_warn:
            checks.append({"check": "parse warnings", "status": "warn",
                           "detail": f"{len(new_warn)} new: " + "; ".join(_cut(w, 90) for w in new_warn[:3])})
        # 2. round trip
        checks.append({"check": "parse round trip", **check_round_trip(ta)})
        # 3. constituents / functions / type definitions / duplicates
        checks += [{k: r[k] for k in ("check", "status", "detail")} for r in syntax_checks(tb_, ta, text_b, text_a)]
        rev_b, rev_a = set(reversed_odd_order(tb_)), reversed_odd_order(ta)
        new_rev = [r for r in rev_a if r not in rev_b]
        checks.append({"check": "constituent order (odd RK terms)", "status": "warn" if new_rev else "pass",
                       "detail": (f"{len(new_rev)} new parameter(s) with non-alphabetical constituents in an odd "
                                  f"order, e.g. {', '.join(new_rev[:3])}: Thermo-Calc flips their sign, pycalphad "
                                  "does not - write them in alphabetical order" if new_rev else
                                  f"no new odd-order terms with reversed constituents ({len(rev_a)} in after)")})
        # 4. differences and affected phases
        grid = np.linspace(298.15, 4000.0, 81)
        diffs = phase_differences(tb_, ta, pmap, grid)
        changed = [n for n, d in diffs.items() if d["any"]]
        affected = [p.upper() for p in affected_phases] if affected_phases else list(changed)
        # 5. site ratios
        sr_bad, sr_n = [], 0
        inv = {v: k for k, v in pmap.items()}
        for nm, ph in ta.phases.items():
            bn = inv.get(nm, nm)
            n_a, _ = atoms_per_fu(ph)
            need = ("sum_one" in want and nm in affected)
            if bn in tb_.phases:
                n_b, _ = atoms_per_fu(tb_.phases[bn])
                need = need or abs(n_b - 1.0) <= REL_TOL
            if need:
                sr_n += 1
                if abs(n_a - 1.0) > REL_TOL:
                    sr_bad.append(f"{nm} {model_text(ph)} sums to {n_a!r}")
        checks.append({"check": "site ratios", "status": "fail" if sr_bad else "pass",
                       "detail": f"{sr_n} phase(s) must have 1 atom/formula unit"
                                 + (f"; {len(sr_bad)} do not: " + "; ".join(sr_bad[:4]) if sr_bad else ": all exactly 1")})
        # 6. phases outside affected_phases
        out_bad, out_added = [], []
        for nm, d in diffs.items():
            if nm in affected or not d["any"]:
                continue
            what = []
            if d["in_before"] and not d["in_after"]:
                what.append("phase removed")
            if d["model"]:
                what.append(d["note"])
            if d["removed"]:
                what.append(f"{len(d['removed'])} removed ({_klabel(d['removed'][0])}...)")
            if d["changed"]:
                what.append(f"{len(d['changed'])} changed ({_klabel(d['changed'][0])}, max |d| {d['max_abs']:.4g})")
            if what:
                out_bad.append(f"{nm}: " + ", ".join(what))
            if d["added"] or (d["in_after"] and not d["in_before"]):
                out_added.append(f"{nm}: " + ("new phase" if not d["in_before"] else
                                              f"{len(d['added'])} added ({_klabel(d['added'][0])}...)"))
        strict = "only_affected" in want
        status = "fail" if out_bad or (strict and out_added) else ("warn" if out_added else "pass")
        n_out = sum(1 for n in diffs if n not in affected)
        detail = (f"{n_out} phase(s) outside the affected ones compared (functions inlined, rel 1e-9)"
                  + (": all identical" if not out_bad and not out_added else "")
                  + ("; changed: " + "; ".join(out_bad[:4]) if out_bad else "")
                  + ("; additions: " + "; ".join(out_added[:4]) if out_added else ""))
        checks.append({"check": "unaffected phases identical", "status": status, "detail": detail})
        # 7. GM per mole of atoms of affected phases
        if db_a is not None:
            try:
                db_b = _pyc_db(pb)
            except Exception as e:  # noqa: BLE001
                checks.append({"check": "pycalphad loads before", "status": "fail", "detail": str(e)[:200]})
        rng = np.random.default_rng(seed)
        n = max(1, int(n_samples))
        temps = ([float(temperatures[i % len(temperatures)]) for i in range(n)] if temperatures
                 else list(rng.uniform(300.0, 3000.0, n)))
        need_gm = bool(want & {"gm_preserved", "base_unchanged"})
        for nm in affected:
            bn = inv.get(nm, nm)
            row = {"phase": nm, "before_phase": bn, "status": "", "detail": ""}
            if bn not in tb_.phases or nm not in ta.phases:
                row.update(status="fail" if (bn in tb_.phases and need_gm) else "info",
                           detail="new phase (nothing to compare)" if bn not in tb_.phases else "removed in after")
                gm_rows.append(row)
                continue
            if db_a is None or db_b is None:
                row.update(status="skip", detail="pycalphad could not load both databases")
                gm_rows.append(row)
                continue
            try:
                cands, note = _sublattice_map(tb_.phases[bn], ta.phases[nm], smaps.get(nm, smaps.get(bn)),
                                              scales.get(nm, scales.get(bn)))
            except ValueError as e:
                cands, note = [], str(e)
            if not cands:
                row.update(status="fail" if need_gm else "info", detail=f"no sublattice mapping: {note}")
                gm_rows.append(row)
                continue
            best = None
            for smap, k in cands:
                res = gm_compare(db_b, db_a, bn, nm, smap, temps, np.random.default_rng(seed + 1))
                if "error" in res:
                    if best is None:
                        best = (smap, k, res)
                    continue
                if best is None or "error" in best[2] or res["rel"] < best[2]["rel"]:
                    best = (smap, k, res)
                if res["rel"] <= REL_TOL:
                    break
            smap, k, res = best
            perm = [smap.get(i, None) for i in range(tb_.phases[bn].n_sublattices)]
            row.update(perm=",".join("-" if j is None else str(j) for j in perm), k=k, n=res.get("n", 0))
            moved = any(j is not None and j != i for i, j in enumerate(perm))
            how = ", ".join(x for x in ("permuted" if moved else "", f"rescaled x{k:.6g}" if abs(k - 1) > 1e-12
                                        else "") if x)
            how = f"{how}: " if how else ""
            if "error" in res:
                row.update(status="fail" if need_gm else "info", detail=f"{how}{res['error']}")
            else:
                row.update(max_rel=res["rel"], max_abs_J_per_mol_atoms=res["abs"], mag_abs=res["mag_abs"],
                           nonmag_rel=res["nonmag_rel"], worst_T=res["worst_at"]["T"] if res["worst_at"] else "",
                           worst_y=str(res["worst_at"]["y_before"]) if res["worst_at"] else "")
                where = (f" at T={res['worst_at']['T']:g} K, y={_cut(str(res['worst_at']['y_before']), 90)}"
                         if res["worst_at"] else "")
                if res["rel"] <= REL_TOL:
                    row.update(status="pass", detail=f"{how}GM per atom preserved: max |dGM| {res['abs']:.3g} J/mol "
                                                     f"(rel {res['rel']:.1e}, {res['n']} samples)")
                elif res["magnetic"] and res["nonmag_rel"] <= REL_TOL:
                    row.update(status="warn", detail=f"{how}non-magnetic part preserved (rel {res['nonmag_rel']:.1e}); "
                                                     f"magnetic contribution changed by {res['mag_abs']:.4g} J/mol of "
                                                     f"atoms (IHJ term per formula unit, nonlinear in BMAGN){where}")
                else:
                    extra = (f"; magnetic contribution changed by {res['mag_abs']:.4g} J/mol of atoms"
                             if res["magnetic"] and res["mag_abs"] > 0 else "")
                    row.update(status="fail" if need_gm else "info",
                               detail=f"{how}GM per atom changed: max |dGM| {res['abs']:.6g} J/mol of atoms "
                                      f"(rel {res['rel']:.2e}){where}{extra}")
            rev = [r for r in new_rev if r.upper().split("(", 1)[1].split(",", 1)[0].split("&")[0] == nm]
            if rev and row["status"] != "pass":
                row["detail"] += (f"; {len(rev)} odd-order term(s) with reversed constituents ({rev[0]}): pycalphad "
                                  "sorts them without flipping the sign")
            if note:
                row["detail"] += f" [{note}]"
            gm_rows.append(row)
    if gm_rows:
        bad = [r for r in gm_rows if r["status"] == "fail"]
        warn = [r for r in gm_rows if r["status"] in ("warn", "info")]
        checks.append({"check": "GM per atom (affected phases)", "status": "fail" if bad else
                       ("warn" if warn else "pass"),
                       "detail": f"{len(gm_rows)} phase(s): {sum(1 for r in gm_rows if r['status'] == 'pass')} "
                                 f"preserved, {len(bad)} failed, {len(warn)} changed/magnetic/new"})
    verdict = "FAIL" if any(c["status"] == "fail" for c in checks) else "PASS"
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    name_b = Path(before.path if isinstance(before, TDB) else before).name
    name_a = Path(after.path if isinstance(after, TDB) else after).name
    T = Text(3400)
    T.add(f"{verdict}" + (f" ({n_warn} warning(s))" if n_warn and verdict == "PASS" else "")
          + f": {name_b} -> {name_a}; changed phases: {', '.join(changed) or 'none'}; affected: "
          + (", ".join(affected) or "none") + (f"; expect: {', '.join(sorted(want))}" if want else ""))
    for c in checks:
        T.add(f"  {c['status'].upper():4s}  {c['check']}: {_cut(c['detail'], 260)}")
    if gm_rows:
        T.add("GM per mole of atoms, before vs after:")
        for r in gm_rows:
            T.add(f"  {r['status'].upper():4s}  {r['phase']}"
                  + (f" (before {r['before_phase']})" if r["before_phase"] != r["phase"] else "")
                  + (f" [after sublattice of each before sublattice: {r['perm']}]" if r.get("perm") else "")
                  + f": {_cut(r['detail'], 300)}")
    change_rows = []
    for nm, d in diffs.items():
        for kind in ("removed", "added", "changed"):
            for k in d[kind]:
                change_rows.append({"phase": nm, "change": kind, "parameter": _klabel(k)})
        if d["model"]:
            change_rows.append({"phase": nm, "change": "model", "parameter": d["note"]})
    files = write_tables(output, {"checks": checks, "gm_per_atom": gm_rows, "changes": change_rows}) if output else []
    text = T.text() + ("\nFiles:\n" + "\n".join(rel(f) for f in files) if files else "")
    return {"verdict": verdict, "checks": checks, "gm": gm_rows, "changed_phases": changed, "affected": affected,
            "changes": change_rows, "text": text, "files": files}


def register(tb) -> None:
    S = {"type": "string"}
    STRS = {"type": "array", "items": S}

    @tb.add("validate_tdb_edit",
            "Check an edited TDB against its original. Runs pycalphad load, write/re-parse round trip, constituents, "
            "function references, type definitions, duplicates, site-ratio sums, phases outside affected_phases "
            "unchanged (functions inlined), GM per mole of atoms of affected phases at random T/site fractions "
            "through the detected sublattice permutation and rescale (magnetic deviation reported separately). "
            "PASS/FAIL with numbers.",
            {"before": S, "after": S,
             "affected_phases": {**STRS, "description": "phases allowed to change (default: every changed phase)"},
             "expect": {**STRS, "description": "'only affected phases changed', 'base subsystems unchanged', "
                                               "'energy_per_atom_preserved', 'site ratios sum to 1'"},
             "temperatures": {"type": "array", "items": {"type": "number"}, "description": "K (default random "
                                                                                           "300-3000)"},
             "n_samples": {"type": "integer", "description": "GM samples per phase (default 20)"},
             "sublattice_map": {"type": "object", "description": "{phase: [after sublattice of each before "
                                                                 "sublattice]} 0-based, if not detectable"},
             "scale": {"type": "object", "description": "{phase: k}, after site ratios = k x before's"},
             "phase_map": {"type": "object", "description": "{before name: after name} for renamed phases"},
             "output": {**S, "description": ".xlsx/.csv with all checks, GM rows and changes"}},
            ["before", "after"])
    def validate_tdb_edit(before, after, affected_phases=None, expect=None, temperatures=None, n_samples: int = 20,
                          sublattice_map=None, scale=None, phase_map=None, output=None):
        r = validate_edit(tb._find_tdb(before), tb._find_tdb(after), affected_phases, expect, temperatures,
                          n_samples, sublattice_map, scale, phase_map, tb.outp(output, output) if output else None,
                          rel=tb.rel)
        return r["text"], []
