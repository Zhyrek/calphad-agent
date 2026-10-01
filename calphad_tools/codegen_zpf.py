"""show_code generator for digitized_to_zpf.

The phase assignment (figure regions, legend, optional TDB) ran in the toolkit; the standalone code records its
result per point - the numbers as plotted, the group, the assigned phase and the ZPF row - and rebuilds the ESPEI
JSON with plain Python (unit conversion, rows, datasets), writing the same files.  When a TDB was used, a short
pycalphad cell prints the stable phases on either side of each point as a cross-check.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict

from .codegen import Ctx, generator


@generator("digitized_to_zpf")
def gen_digitized_to_zpf(args: Dict[str, Any], ctx: Ctx) -> str:
    from . import digitize_zpf as Z
    try:
        res = Z.run(ctx.tb, args, preview=True, write_images=False)
    except Exception as e:  # noqa: BLE001 - inputs gone or changed
        ctx.fallback(f"the assignment could not be recovered ({type(e).__name__}: {e})")
    sysm = res["system"]
    pts = [p for p in res["points"] if p.row >= 0]
    rows = {p.row: res["rows"][p.row]["phases"] for p in pts}
    files = {r: res["row_files"][r] for r in sorted(rows)}
    table = [[p.group, {k: round(v, 12) for k, v in p.conv_in.items()}, p.T_raw, p.row, p.phase, bool(p.measured)]
             for p in pts]
    meta = {f: {k: v for k, v in m.items() if v} for f, m in res["meta"].items()}
    prefix = res["prefix"]
    add_K = sysm.T_unit == "C" and sysm.kind == "binary"
    methods = sorted({p.method for p in res["points"] if p.method})
    ctx.need("import json")
    ctx.note("digitized_to_zpf: the phase assignment is recorded per point; the JSON is rebuilt with plain Python")
    src = "\n\n".join(inspect.getsource(f) for f in (Z.to_mole_fractions, Z.rows_from_points, Z.datasets_from_rows))
    L = [f"# digitized_to_zpf: phases of each digitized point were assigned in the toolkit from: {', '.join(methods)}",
         "# (region labels read off the figure and the regions bounded by its drawn lines / the legend / a TDB).",
         "# Per point: group, amounts as plotted (converted below), temperature as plotted, ZPF row, phase whose",
         "# composition the point is, and whether that composition is measured (False: row temperature only).",
         f"ELEMENTS = {Ctx.py(sysm.elements)}",
         f"INDEP = {Ctx.py(sysm.indep)}          # independent components of every phase composition",
         f"UNIT = {sysm.comp_unit!r}             # composition unit of the figure",
         f"MASSES = {Ctx.py(res['masses'])}   # g/mol (weight units only)",
         f"POINTS = {Ctx.py(table, 10)}",
         f"ROW_PHASES = {Ctx.py(rows, 13)}   # phases in equilibrium in each row",
         f"ROW_FILES = {Ctx.py(files, 12)}",
         f"META = {Ctx.py(meta, 7)}",
         "",
         src,
         "",
         "records = []",
         "for group, amounts, t_plotted, row, phase, measured in POINTS:",
         "    X = to_mole_fractions(amounts, UNIT, ELEMENTS, MASSES)",
         "    X = {e: min(1.0, max(0.0, v)) for e, v in X.items()}",
         "    T_K = " + ("float(t_plotted) + 273.15   # °C -> K" if add_K else "float(t_plotted)"),
         "    records.append({'row': row, 'row_phases': ROW_PHASES[row], 'phase': phase, 'measured': measured,",
         "                    'X': {e: X[e] for e in INDEP}, 'T_K': T_K})",
         "rows = rows_from_points(records, INDEP)",
         "datasets = datasets_from_rows(rows, ROW_FILES, ELEMENTS, META)",
         f"out_dir = OUT / 'datasets' / {prefix!r}",
         "out_dir.mkdir(parents=True, exist_ok=True)",
         "for name, ds in datasets.items():",
         "    (out_dir / name).write_text(json.dumps(ds, indent=2))",
         "    print(name, len(ds['values']), 'rows', ds['phases'])",
         "try:                                   # ESPEI's own dataset checker, when ESPEI is installed",
         "    from espei.datasets import load_datasets",
         "    print(len(load_datasets(sorted(str(p) for p in out_dir.glob('*.json'))).all()), 'datasets load in ESPEI')",
         "except ImportError:",
         "    pass",
         ""]
    code = "\n".join(L)
    if args.get("tdb") and sysm.kind == "binary":
        ctx.need("import warnings", "from pycalphad import Database, equilibrium, variables as v")
        code += (
            "\n# cross-check with the database: stable phases just left/right of each point\n"
            f"db = Database(str({ctx.tdb(args['tdb'])}))\n"
            f"phases = {Ctx.py(res['inf'].phases if res.get('inf') else [])}\n"
            f"comps = {Ctx.py(sysm.elements + ['VA'])}\n"
            "for rec in records[:40]:\n"
            f"    x = rec['X'][{sysm.indep[0]!r}]\n"
            "    with warnings.catch_warnings():\n"
            "        warnings.simplefilter('ignore')\n"
            "        eq = equilibrium(db, comps, phases, {v.T: rec['T_K'], v.P: 101325, v.N: 1,\n"
            f"                                             v.X({sysm.indep[0]!r}): [max(1e-5, x - 0.012), min(1 - 1e-5, x + 0.012)]}})\n"
            "    sides = [sorted({str(p) for p in eq.Phase.values.squeeze()[i] if p}) for i in range(2)]\n"
            "    print(round(x, 4), round(rec['T_K'], 1), rec['phase'] or '(T only)', '| left:', sides[0], 'right:', sides[1])\n")
    return code
