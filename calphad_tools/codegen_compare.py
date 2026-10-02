"""show_code generator for compare_system.

The standalone script builds the same tables as the tool: every parameter as written in each TDB (VV####
handles replaced by their values) with its reference key and citation, the rows that differ, the rows written
differently but numerically equal, and per-database parameter / function lists - using copies of calphad_tools.tdbparse / pwinline / tdbtables (standard library, numpy, symengine).
Values at the temperatures come from an independent route: pycalphad's Database, with its FUNCTIONs (``db.symbols``)
inlined.  Outputs: CSVs, an Excel workbook and a simplified PowerPoint.  What pycalphad's Database cannot tell (the phase_map
applied, the resolved expressions, and parameters written with non-alphabetical constituents - pycalphad sorts
them without changing the sign of odd Redlich-Kister orders) is passed in from the tool.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from .codegen import Ctx, generator


def module_source(ctx: Ctx, module, title: str) -> str:
    """A module's source without its docstring and imports (standard imports requested via ctx.need; imports
    between calphad_tools modules dropped - their sources are copied into the same script)."""
    src = inspect.getsource(module)
    tree = ast.parse(src)
    lines = src.splitlines()
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and (node.module == "__future__" or node.level):
                continue
            for a in node.names:
                if isinstance(node, ast.Import):
                    ctx.need(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
                else:
                    ctx.need(f"from {node.module} import {a.name}")
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        keep.append("\n".join(lines[start - 1:node.end_lineno]))
    return f"# --- {title} ---\n" + "\n\n".join(keep) + "\n\n\n"


def helpers_source(ctx: Ctx) -> str:
    from . import pwinline, tdbparse, tdbtables
    return (module_source(ctx, tdbparse, "TDB text parser (keeps expressions exactly as written)")
            + module_source(ctx, pwinline, "piecewise-in-T algebra: inline FUNCTIONs, merge T ranges, collect terms")
            + module_source(ctx, tdbtables, "subsystem selection, VV#### substitution, written-expression tables"))


_BODY = r'''
def load_database(label, path):
    """Rows {key: {phase, label, segs}} of one database, restricted to ELEMENTS (+VA), functions inlined."""
    db = Database(str(path))
    E = set(ELEMENTS) | {"VA"}
    rename = PHASE_MAP.get(label, {})
    fixes = KEY_FIXES.get(label, {})

    def in_sub(sp):
        return set(sp.constituents or {sp.name: 1}) <= E

    phases = {}
    for name, ph in db.phases.items():
        restricted = [sorted(s.name for s in subl if in_sub(s)) for subl in ph.constituents]
        if not restricted or any(not r for r in restricted) or all(x == "VA" for r in restricted for x in r):
            continue
        phases[name] = (rename.get(name, name), [float(x) for x in ph.sublattices],
                        [i for i, r in enumerate(restricted) if r == ["VA"]])
    cache = {}

    def resolve(name):
        if name not in cache:
            if name not in db.symbols:
                raise MissingRef(name)
            cache[name] = inline(segments_from_piecewise(db.symbols[name]), resolve)
        return cache[name]

    rows = {}
    for p in db._parameters.all():
        if p["phase_name"] not in phases or not all(in_sub(s) for subl in p["constituent_array"] for s in subl):
            continue
        canon, ratios, va_only = phases[p["phase_name"]]
        cons = [[s.name for s in subl] for subl in p["constituent_array"]]
        red = [c for i, c in enumerate(cons) if i not in va_only] if len(cons) == len(ratios) else cons
        ptype = PTYPE_ALIASES.get(p["parameter_type"].upper(), p["parameter_type"].upper())
        diff = getattr(p.get("diffusing_species"), "name", None) or ""
        key = make_key(ptype, canon, red, int(p["parameter_order"]), diff)
        key, sign = fixes.get(key, (key, 1.0))
        order = int(key.rsplit(";", 1)[1].rstrip(")"))
        try:
            segs = inline(scale(segments_from_piecewise(p["parameter"]), sign), resolve)
        except MissingRef as e:
            segs = None
            print(f"{label}: {key} references undefined function {e}")
        rows[key] = {"phase": canon, "label": make_label(ptype, canon, cons, order, diff), "segs": segs}
    return rows, resolve


ROWS, RESOLVE = {}, {}
for label, path in DATABASES.items():
    ROWS[label], RESOLVE[label] = load_database(label, path)

# ---- derived expressions (recipes resolved by the tool: which parameter / function each symbol stands for)
EXPR_SEGS = {}
for ex in EXPRESSIONS:
    for label, recipe in ex["per_db"].items():
        if "na" in recipe:
            EXPR_SEGS[(ex["label"], label)] = None
            continue
        seg_of = {}
        for name, ref in recipe["refs"].items():
            if ref[0] == "param":
                seg_of[name] = scale(ROWS[label][ref[1]]["segs"], ref[2])
            else:
                seg_of[name] = RESOLVE[label](ref[1])
        lo = max([s[0][0] for s in seg_of.values()] or [298.15])
        hi = min([s[-1][1] for s in seg_of.values()] or [6000.0])
        EXPR_SEGS[(ex["label"], label)] = inline(
            [(lo, hi, se.sympify(recipe["expr"]))],
            lambda n, _s=seg_of, _l=label: _s[n] if n in _s else RESOLVE[_l](n))

# ---- long table: values at the temperatures + expanded piecewise expressions
GRID = np.union1d(np.linspace(298.15, 3000.0, 121), TEMPERATURES)


def status(segs, ref_segs):
    if segs is None:
        return "n/a"
    if ref_segs is None:
        return "no ref"
    lo, hi = max(segs[0][0], ref_segs[0][0], GRID[0]), min(segs[-1][1], ref_segs[-1][1], GRID[-1])
    g = GRID[(GRID >= lo) & (GRID <= hi)] if lo < hi else np.asarray(TEMPERATURES)
    same, dmax = close(evaluate(segs, g)[0], evaluate(ref_segs, g)[0], RTOL)
    return "same" if same else f"differs (max |d| {dmax:.4g})"


records = []
keys = sorted({k for rows in ROWS.values() for k in rows})
for key in keys:
    phase = next(r[key]["phase"] for r in ROWS.values() if key in r)
    if PHASES and phase not in PHASES:
        continue
    for label in DATABASES:
        row = ROWS[label].get(key)
        if row is None:
            continue
        segs = row["segs"]
        ref = ROWS[REFERENCE].get(key)
        vals = evaluate(segs, TEMPERATURES)[0] if segs else [float("nan")] * len(TEMPERATURES)
        records.append({"kind": "parameter", "phase": phase, "key": key, "item": row["label"], "database": label,
                        "status_vs_ref": "ref" if label == REFERENCE else status(segs, ref and ref["segs"]),
                        **{f"value_{t:g}K": v for t, v in zip(TEMPERATURES, vals)},
                        "expanded": pw_text(segs) if segs else "undefined function"})
for ex in EXPRESSIONS:
    for label in DATABASES:
        segs = EXPR_SEGS.get((ex["label"], label))
        vals = evaluate(segs, TEMPERATURES)[0] if segs else [float("nan")] * len(TEMPERATURES)
        records.append({"kind": "expression", "phase": "(expressions)", "key": ex["label"], "item": ex["label"],
                        "database": label,
                        "status_vs_ref": "ref" if label == REFERENCE else
                        status(segs, EXPR_SEGS.get((ex["label"], REFERENCE))),
                        **{f"value_{t:g}K": v for t, v in zip(TEMPERATURES, vals)},
                        "expanded": pw_text(segs) if segs else ex["per_db"][label].get("na", "n/a")})
table = pd.DataFrame(records)
table.to_csv(OUT / CSV_NAME, index=False)
print(table[["item", "database"] + [f"value_{t:g}K" for t in TEMPERATURES]].to_string(max_colwidth=36))

# ---- parameters as written in each TDB (VV#### replaced by values), the differences, per-database tables
SUBS = [Subsystem(parse_tdb(str(path)), label, str(path), ELEMENTS, PHASE_MAP.get(label, {}))
        for label, path in DATABASES.items()]
# status: same | equal (written differently, numerically equal within RTOL) | differs (value differs or absent);
# where the databases do not share the same functions the cells show the inlined polynomial in T instead.
# Each parameter's reference key is the token after its last T range ("6000 N 91Din !"); the citation comes from
# the TDB's LIST_OF_REFERENCES (or a $-comment reference list) when it has that key.
WRITTEN = expression_table(SUBS, PHASES, rtol=RTOL)
if not INCLUDE_IDENTICAL:
    WRITTEN = [r for r in WRITTEN if r.status == "differs"]
written = pd.DataFrame([{"phase": r.phase, "key": r.key, "parameter": r.label,
                         **{s.label: (r.shown[s.label] if r.shown[s.label] is not None else "—") for s in SUBS},
                         "status": r.status, "note": r.note,
                         **{f"{s.label} (as written)": (r.cells[s.label].text if r.cells[s.label] is not None
                                                        else "—") for s in SUBS},
                         "max |Δ|": r.max_diff,
                         **{f"{s.label} reference": r.refs.get(s.label, "") for s in SUBS},
                         **{f"{s.label} citation": r.citations.get(s.label, "") for s in SUBS}} for r in WRITTEN])
written.to_csv(OUT / WRITTEN_CSV_NAME, index=False)
diff = written[written["status"] == "differs"]
# equal within the tolerance but written differently: not differences, listed in their own table
equal = written[written["status"] == "equal"]
print(f"\n{(diff['note'].str.contains('absent') == False).sum()} parameters differ numerically, "
      f"{len(equal)} written differently but equal, "
      f"{diff['note'].str.contains('absent').sum()} absent in some database")
print(diff[["parameter"] + list(DATABASES) + ["note"]].to_string(max_colwidth=40))
per_db = {s.label: (pd.DataFrame([{"phase": s.rows[k].phase, "parameter": s.rows[k].label,
                                   "expression": s.written(k).text, "reference": s.rows[k].reference,
                                   "citation": s.citation(k)} for k in database_keys(s, PHASES)]),
                    pd.DataFrame([{"function": n, "expression": w.text} for n, w in database_functions(s, PHASES)]))
          for s in SUBS}

# ---- Excel: differences, numerically equal, all parameters, one sheet per database, values at T
with pd.ExcelWriter(OUT / XLSX_NAME, engine="openpyxl") as xl:
    diff.drop(columns=["key"]).to_excel(xl, sheet_name="Differences", index=False)
    equal.drop(columns=["key"]).to_excel(xl, sheet_name="Numerically equal", index=False)
    written.drop(columns=["key"]).to_excel(xl, sheet_name="All parameters", index=False)
    for label, (params, funcs) in per_db.items():
        name = f"DB {label}"[:31]
        params.to_excel(xl, sheet_name=name, index=False)
        if len(funcs):
            funcs.to_excel(xl, sheet_name=name, index=False, startrow=len(params) + 2)
    table.to_excel(xl, sheet_name="Values", index=False)

# ---- simplified PowerPoint: the same tables as text
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def text_table(title, header, rows, widths, per_slide=12):
    for start in range(0, max(1, len(rows)), per_slide):
        chunk = rows[start:start + per_slide]
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title if len(rows) <= per_slide else f"{title} ({start // per_slide + 1})"
        shape = slide.shapes.add_table(len(chunk) + 1, len(header), Inches(0.3), Inches(1.4), Inches(sum(widths)),
                                       Inches(0.3 * (len(chunk) + 1)))
        tbl = shape.table
        for j, w in enumerate(widths):
            tbl.columns[j].width = Inches(w)
        for j, h in enumerate(header):
            tbl.cell(0, j).text = h
        for i, row in enumerate(chunk, start=1):
            for j, v in enumerate(row):
                tbl.cell(i, j).text = str(v)
        for row in tbl.rows:
            for cell in row.cells:
                for par in cell.text_frame.paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(8)


labels = list(DATABASES)
col_w = [2.6] + [(12.7 - 2.6) / len(labels)] * len(labels)
ROW_OF = {r.key: r for r in WRITTEN}


def cell(r, lab):
    """Cell text with the reference (shorthand · first author + year) on a second line."""
    ref = ROW_OF[r["key"]].ref_text(lab)
    return r[lab] + (f"\n[{ref}]" if ref and r[lab] != "—" else "")


for title, part in (("Differences", diff), ("Numerically equal within tolerance (written differently)", equal),
                    ("All parameters", written)):
    if title.startswith("Numerically") and not len(part):
        continue
    text_table(title, ["parameter"] + labels,
               [[f"{r['phase']}: {r['parameter']}" + (f"\nmax |Δ| {r['max |Δ|']:.3g}" if r["status"] == "equal"
                                                     else "")]
                + [cell(r, lab) for lab in labels] for _, r in part.iterrows()], col_w)
for label, (params, funcs) in per_db.items():
    text_table(f"{label}: parameters", ["phase", "parameter", "expression", "reference"],
               [[r["phase"], r["parameter"], r["expression"],
                 ref_display(r["reference"], r["citation"])] for _, r in params.iterrows()], [1.4, 2.8, 6.6, 1.9])
    if len(funcs):
        text_table(f"{label}: functions used", ["function", "expression"], funcs.values.tolist(), [2.0, 10.7])
prs.save(OUT / PPTX_NAME)
print("wrote", OUT / CSV_NAME, OUT / WRITTEN_CSV_NAME, OUT / XLSX_NAME, OUT / PPTX_NAME)
'''


@generator("compare_system")
def _gen_compare_system(args, ctx: Ctx) -> str:
    from . import compare_system as cs
    from .codegen_calc import _outp

    plan = cs.standalone_plan(ctx.tb, args)
    sys_ = "".join(e.capitalize() for e in plan["elements"])
    default = f"compare_system_{sys_}_{len(plan['labels'])}db.pptx"
    out = args.get("output_pptx")
    if out and not out.endswith(".pptx"):
        out += ".pptx"
    pptx_path = _outp(ctx.tb, out, default)
    stem = Path(pptx_path).stem
    pptx_name = ctx.out_name(pptx_path, default)
    xlsx_name = ctx.out_name(str(Path(pptx_path).with_name(stem + ".xlsx")), stem + ".xlsx")
    csv_name = ctx.out_name(str(Path(pptx_path).with_name(stem + "_long.csv")), stem + "_long.csv")
    wcsv_name = stem + "_written.csv"
    ctx.need("import numpy as np", "import pandas as pd", "import symengine as se", "from pycalphad import Database",
             "from pptx import Presentation", "from pptx.util import Inches, Pt")
    helpers = helpers_source(ctx)
    dbs = "{\n" + "".join(f"    {lab!r}: {ctx.path(p)},\n" for lab, p in zip(plan["labels"], plan["paths"])) + "}"
    if args.get("plot_diagrams"):
        ctx.note("compare_system: the phase-diagram overlay is not reproduced (use show_code on a map_phase_diagram "
                 "call for that)")
    cfg = (f"# compare_system: {'-'.join(plan['elements'])} subsystem of {len(plan['labels'])} databases\n"
           f"DATABASES = {dbs}\n"
           f"REFERENCE = {plan['reference']!r}\n"
           f"ELEMENTS = {Ctx.py(plan['elements'])}\n"
           f"TEMPERATURES = {Ctx.py(plan['temperatures'])}   # K\n"
           f"PHASES = {Ctx.py(plan['phases'])}   # None = every phase of the subsystem\n"
           f"RTOL = {plan['rtol']!r}\n"
           "# phase names merged across databases (native -> canonical), as applied by the tool\n"
           f"PHASE_MAP = {Ctx.py(plan['phase_map'])}\n"
           "# parameters written with non-alphabetical constituents: pycalphad sorts the constituents but keeps the\n"
           "# odd Redlich-Kister sign / ternary order index as written; map to the Thermo-Calc meaning\n"
           f"KEY_FIXES = {Ctx.py(plan['key_fixes'])}\n"
           "# derived expressions: per database, a symengine expression over symbols that stand for parameters\n"
           "# ['param', key, sign] or FUNCTIONs ['function', name] ('na' = not available in that database)\n"
           f"EXPRESSIONS = {Ctx.py(plan['expressions'])}\n"
           f"INCLUDE_IDENTICAL = {bool(plan['include_identical'])!r}\n"
           f"CSV_NAME, WRITTEN_CSV_NAME = {csv_name!r}, {wcsv_name!r}\n"
           f"XLSX_NAME, PPTX_NAME = {xlsx_name!r}, {pptx_name!r}\n\n")
    return cfg + helpers + _BODY.lstrip("\n")
