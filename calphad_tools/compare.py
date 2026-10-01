"""Feature 1: compare two TDB files parameter-by-parameter and export to PowerPoint.

Usage (CLI):  calphad-tools compare A.tdb B.tdb -o comparison.pptx
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import pptx_utils as pu
from .tdbparse import TDB, ExpressionError, Parameter, Piecewise, parse_tdb

STATUSES = ["identical", "equivalent", "different", "only A", "only B"]


@dataclass
class Row:
    section: str             # element | phase | function | parameter
    item: str                # e.g. "GHSERCR" or "L(BCC_A2,CR,FE:VA;0)"
    phase: str
    a: str
    b: str
    status: str
    max_diff: float = math.nan
    note: str = ""
    a_full: str = ""
    b_full: str = ""


@dataclass
class Comparison:
    a: TDB
    b: TDB
    rows: List[Row] = field(default_factory=list)
    t_range: Tuple[float, float] = (298.15, 3000.0)

    def by_section(self, section: str) -> List[Row]:
        return [r for r in self.rows if r.section == section]

    def counts(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for sec in ("element", "phase", "function", "parameter"):
            rows = self.by_section(sec)
            out[sec] = {s: sum(1 for r in rows if r.status == s) for s in STATUSES}
        return out

    def summary_text(self, max_rows: int = 40) -> str:
        c = self.counts()
        lines = [f"Comparison of A={self.a.name} vs B={self.b.name} "
                 f"(numerical checks over {self.t_range[0]:g}-{self.t_range[1]:g} K)"]
        for sec, d in c.items():
            lines.append(f"  {sec+'s':<11s} " + ", ".join(f"{k}: {v}" for k, v in d.items() if v))
        diffs = [r for r in self.rows if r.status not in ("identical",)]
        if diffs:
            lines.append("Non-identical items (first %d):" % min(max_rows, len(diffs)))
            for r in diffs[:max_rows]:
                d = "" if math.isnan(r.max_diff) else f" max|Δ|={r.max_diff:.4g}"
                lines.append(f"  [{r.status}] {r.section} {r.item}: A={r.a!s:.60} | B={r.b!s:.60}{d}"
                             + (f" ({r.note})" if r.note else ""))
        return "\n".join(lines)


def _close(va: np.ndarray, vb: np.ndarray, atol: float, rtol: float) -> Tuple[bool, float]:
    mask = np.isfinite(va) & np.isfinite(vb)
    if not mask.any():
        return False, math.nan
    d = float(np.max(np.abs(va[mask] - vb[mask])))
    scale = float(max(np.max(np.abs(va[mask])), np.max(np.abs(vb[mask])), 1e-30))
    return d <= max(atol, rtol * scale), d


def _eval_param(db: TDB, p: Parameter, T: np.ndarray, lookup) -> Optional[np.ndarray]:
    try:
        return p.sign * p.expr.evaluate(T, lookup)
    except (ExpressionError, ZeroDivisionError, OverflowError, ValueError):
        return None


def _eval_fn(db: TDB, name: str, T: np.ndarray, lookup) -> Optional[np.ndarray]:
    try:
        return lookup(name, T)
    except (ExpressionError, ZeroDivisionError, OverflowError, ValueError):
        return None


def _full(pw: Piecewise) -> str:
    return " ; ".join(f"[{lo:g}-{hi:g}] {e}" for lo, hi, e in pw.ranges)


def compare_tdbs(path_a: str, path_b: str, t_min: float = 298.15, t_max: float = 3000.0,
                 n_points: int = 60, atol: float = 1e-6, rtol: float = 1e-4) -> Comparison:
    """Compare two TDB files.

    * identical  - same text (after whitespace/number-format normalisation) and same values
    * equivalent - written differently, but numerically equal within ``rtol``/``atol``
                   over [t_min, t_max] (e.g. SGTE data with a different number of decimals,
                   re-ordered constituents with the Redlich-Kister sign handled)
    * different  - numerically different (or not evaluable and textually different)
    * only A / only B
    """
    a, b = parse_tdb(path_a), parse_tdb(path_b)
    cmp = Comparison(a, b, t_range=(t_min, t_max))
    T = np.linspace(t_min, t_max, n_points)
    la, lb = a.evaluator(), b.evaluator()

    # ---- elements -------------------------------------------------------
    for el in sorted(set(a.elements) | set(b.elements)):
        ea, eb = a.elements.get(el), b.elements.get(el)
        if ea is None or eb is None:
            e = ea or eb
            cmp.rows.append(Row("element", el, "", _el(ea), _el(eb), "only A" if eb is None else "only B"))
            continue
        same_ref = ea.ref_phase == eb.ref_phase
        nums_eq = np.allclose([ea.mass, ea.h298, ea.s298], [eb.mass, eb.h298, eb.s298], rtol=1e-3, atol=1e-3)
        exact = same_ref and [ea.mass, ea.h298, ea.s298] == [eb.mass, eb.h298, eb.s298]
        st = "identical" if exact else ("equivalent" if same_ref and nums_eq else "different")
        cmp.rows.append(Row("element", el, "", _el(ea), _el(eb), st))

    # ---- phases -----------------------------------------------------------
    drop_a: Dict[str, List[int]] = {}
    drop_b: Dict[str, List[int]] = {}
    for ph in sorted(set(a.phases) | set(b.phases)):
        pa, pb = a.phases.get(ph), b.phases.get(ph)
        if pa is None or pb is None:
            cmp.rows.append(Row("phase", ph, ph, _ph(pa), _ph(pb), "only A" if pb is None else "only B"))
            continue
        notes = []
        if pa.model_signature() != pb.model_signature():
            if pa.reduced_signature() == pb.reduced_signature():
                drop_a[ph], drop_b[ph] = pa.va_only_sublattices(), pb.va_only_sublattices()
                notes.append("same model apart from a vacancy-only sublattice (parameters aligned)")
            else:
                notes.append("different sublattice model")
        ca = [sorted(x) for i, x in enumerate(pa.constituents) if i not in drop_a.get(ph, [])]
        cb = [sorted(x) for i, x in enumerate(pb.constituents) if i not in drop_b.get(ph, [])]
        if ca != cb:
            notes.append("different constituents")
        tda = sorted(t.canonical(ph) for t in a.phase_typedefs(pa) if t.kind != "SEQ")
        tdb_ = sorted(t.canonical(ph) for t in b.phase_typedefs(pb) if t.kind != "SEQ")
        if tda != tdb_:
            notes.append("different type definitions (magnetic/order-disorder)")
        if "different sublattice model" in notes or "different type definitions (magnetic/order-disorder)" in notes:
            st = "different"
        elif notes:
            st = "equivalent"   # same model, different constituent set: compatible
        else:
            st = "identical"
        cmp.rows.append(Row("phase", ph, ph, _ph(pa, a), _ph(pb, b), st, note="; ".join(notes)))

    # ---- functions -----------------------------------------------------------
    for fn in sorted(set(a.functions) | set(b.functions)):
        fa, fb = a.functions.get(fn), b.functions.get(fn)
        if fa is None or fb is None:
            f = fa or fb
            cmp.rows.append(Row("function", fn, "", fa.expr.short() if fa else "",
                                fb.expr.short() if fb else "", "only A" if fb is None else "only B",
                                a_full=_full(fa.expr) if fa else "", b_full=_full(fb.expr) if fb else ""))
            continue
        cmp.rows.append(_compare_expr("function", fn, "", fa.expr, fb.expr,
                                      _eval_fn(a, fn, T, la), _eval_fn(b, fn, T, lb), atol, rtol))

    # ---- parameters -----------------------------------------------------------
    pa_d = {p.key.without_sublattices(drop_a.get(p.key.phase, [])).as_tuple(): p for p in a.parameters}
    pb_d = {p.key.without_sublattices(drop_b.get(p.key.phase, [])).as_tuple(): p for p in b.parameters}
    keys = sorted(set(pa_d) | set(pb_d), key=lambda k: (k[1], sum(len(s) for s in k[2]), k[2], k[0], k[3]))
    for k in keys:
        p1, p2 = pa_d.get(k), pb_d.get(k)
        label = (p1 or p2).key.label()
        if p1 is not None and p2 is not None and p1.key.as_tuple() != p2.key.as_tuple():
            label = p1.key.label()
        phase = k[1]
        if p1 is None or p2 is None:
            p = p1 or p2
            cmp.rows.append(Row("parameter", label, phase, p1.expr.short() if p1 else "",
                                p2.expr.short() if p2 else "", "only A" if p2 is None else "only B",
                                a_full=_full(p1.expr) if p1 else "", b_full=_full(p2.expr) if p2 else ""))
            continue
        va, vb = _eval_param(a, p1, T, la), _eval_param(b, p2, T, lb)
        row = _compare_expr("parameter", label, phase, p1.expr, p2.expr, va, vb, atol, rtol,
                            sign_a=p1.sign, sign_b=p2.sign)
        if p1.raw_name != p2.raw_name:
            row.note = (row.note + "; " if row.note else "") + f"written as {p1.raw_name} vs {p2.raw_name}"
        cmp.rows.append(row)
    return cmp


def _compare_expr(section, name, phase, ea: Piecewise, eb: Piecewise, va, vb, atol, rtol,
                  sign_a=1.0, sign_b=1.0) -> Row:
    same_text = ea.canonical_exprs() == eb.canonical_exprs() and sign_a == sign_b
    note = ""
    if va is not None and vb is not None:
        ok, d = _close(va, vb, atol, rtol)
        exact = not math.isnan(d) and d <= max(1e-9, 1e-12 * float(np.nanmax(np.abs(va)) if np.isfinite(va).any() else 0))
        if same_text and exact:
            st = "identical"
        elif ok:
            st = "equivalent" if not exact else ("identical" if same_text else "equivalent")
        else:
            st = "different"
            if same_text:
                note = "same expression text; referenced functions differ"
    else:
        d = math.nan
        st = "identical" if same_text else "different"
        note = "could not evaluate numerically"
    if (ea.tmin, ea.tmax) != (eb.tmin, eb.tmax):
        note = (note + "; " if note else "") + f"T-range {ea.tmin:g}-{ea.tmax:g} vs {eb.tmin:g}-{eb.tmax:g}"
    return Row(section, name, phase, ea.short(), eb.short(), st, d, note, _full(ea), _full(eb))


def _el(e) -> str:
    if e is None:
        return ""
    return f"{e.ref_phase}  M={e.mass:g}  H298={e.h298:g}  S298={e.s298:g}"


def _ph(p, db: Optional[TDB] = None) -> str:
    if p is None:
        return ""
    cons = " : ".join(",".join(sl) for sl in p.constituents)
    ratios = ":".join(f"{r:g}" for r in p.site_ratios)
    mag = ""
    if db is not None:
        kinds = [t.kind for t in db.phase_typedefs(p) if t.kind != "SEQ"]
        if kinds:
            mag = "  [" + ",".join(sorted(set(k.lower() for k in kinds))) + "]"
    return f"({ratios})  {cons}{mag}"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def write_csv(cmp: Comparison, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "phase", "item", "status", f"max_abs_diff_{cmp.t_range[0]:g}-{cmp.t_range[1]:g}K",
                    f"A ({cmp.a.name})", f"B ({cmp.b.name})", "note"])
        for r in cmp.rows:
            w.writerow([r.section, r.phase, r.item, r.status,
                        "" if math.isnan(r.max_diff) else f"{r.max_diff:.6g}",
                        r.a_full or r.a, r.b_full or r.b, r.note])
    return path


def write_pptx(cmp: Comparison, path: str, include_identical: bool = False,
               rows_per_slide: int = 13, sections=("phase", "function", "parameter")) -> str:
    prs = pu.new_presentation()
    na, nb = cmp.a.name, cmp.b.name
    pu.add_title_slide(prs, "TDB comparison", f"A: {na}\nB: {nb}",
                       note=f"Numerical checks over {cmp.t_range[0]:g}–{cmp.t_range[1]:g} K. "
                            "'equivalent' = written differently but numerically equal within tolerance.")
    # summary table
    c = cmp.counts()
    rows = [[sec.capitalize() + "s"] + [str(c[sec][s]) for s in STATUSES] for sec in c]
    pu.add_table_slides(prs, "Summary", ["Item"] + [s.replace("only A", f"only in A").replace("only B", "only in B")
                                                    for s in STATUSES], rows,
                        [2.4, 1.7, 1.7, 1.7, 1.7, 1.7], font_size=14,
                        subtitle=f"A = {na}    B = {nb}")
    w_diff = 1.2
    for sec in sections:
        rs = [r for r in cmp.by_section(sec) if include_identical or r.status != "identical"]
        if sec == "phase":
            data = [[r.item, r.a, r.b, r.status, r.note] for r in rs]
            pu.add_table_slides(prs, "Phases", ["Phase", f"A: {na}", f"B: {nb}", "Status", "Note"], data,
                                [1.7, 3.9, 3.9, 1.1, 2.1], rows_per_slide=rows_per_slide, font_size=9,
                                status_col=3,
                                subtitle="(site ratios)  constituents  [model add-ons]")
        elif sec == "function":
            data = [[r.item, r.a, r.b, _fmt_d(r.max_diff), r.status] for r in rs]
            pu.add_table_slides(prs, "Functions", ["Function", f"A: {na}", f"B: {nb}", "max |Δ|", "Status"],
                                data, [1.6, 4.6, 4.6, w_diff, 1.1], rows_per_slide=rows_per_slide,
                                font_size=9, status_col=4, mono_cols=(1, 2),
                                subtitle="First T-range shown; full expressions in the CSV")
        elif sec == "parameter":
            data = [[r.item, r.a, r.b, _fmt_d(r.max_diff), r.status] for r in rs]
            pu.add_table_slides(prs, "Parameters", ["Parameter", f"A: {na}", f"B: {nb}", "max |Δ|", "Status"],
                                data, [3.0, 3.85, 3.85, w_diff, 1.1], rows_per_slide=rows_per_slide,
                                font_size=9, status_col=4, mono_cols=(1, 2),
                                subtitle="Constituents sorted alphabetically (odd RK orders sign-corrected); "
                                         "max |Δ| in J/mol (K for TC)")
    return pu.save(prs, path)


def _fmt_d(d: float) -> str:
    if math.isnan(d):
        return "–"
    if d == 0:
        return "0"
    return f"{d:.3g}"


def run(tdb_a: str, tdb_b: str, output: str = "tdb_comparison.pptx", include_identical: bool = False,
        csv_path: Optional[str] = None, t_min: float = 298.15, t_max: float = 3000.0,
        rtol: float = 1e-4) -> Dict:
    cmp = compare_tdbs(tdb_a, tdb_b, t_min=t_min, t_max=t_max, rtol=rtol)
    pptx = write_pptx(cmp, output, include_identical=include_identical)
    csv_path = csv_path or str(Path(output).with_suffix(".csv"))
    write_csv(cmp, csv_path)
    return {"pptx": pptx, "csv": csv_path, "summary": cmp.summary_text(), "counts": cmp.counts()}
