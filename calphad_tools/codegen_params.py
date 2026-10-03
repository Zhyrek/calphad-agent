"""show_code generators for the parameter tools (calphad_tools/param_tools.py).

The generated code is standalone (pycalphad / espei / numpy / scipy / pandas / matplotlib):

* TDB edits are emitted as exact text replacements ``(old, new)`` of the edited commands, computed here with the
  same edit engine the tool used, so the edited TDB the code writes is byte-identical to the tool's.
* Free parameters that are coefficients (``L(...)@T``) are fitted, as in the tool, through temporary FUNCTIONs
  ``DVnnnn`` added to the parameter expression; the generated optimizer writes those FUNCTIONs with their fitted
  values (numerically the same database as the tool's plain numeric edit).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from .codegen import Ctx, generator
from .codegen_calc import mapping_check_helpers

APPLY = '''
def apply_edits(text, pairs):
    """Apply exact (old, new) text replacements in order; each old text occurs exactly once."""
    for old, new in pairs:
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text
'''

MAPPING = '''
def phase_namer(db, strategy):
    """Names of composition sets; ordered phases in their disordered state get the disordered-part name."""
    info = {}
    for name, mod in strategy.models.items():
        ph = db.phases.get(name)
        if ph is None or ph.model_hints.get("ordered_phase") != name:
            continue
        dis = ph.model_hints.get("disordered_phase")
        if not dis or dis not in db.phases:
            continue
        n_sub = len(db.phases[name].sublattices) - (len(db.phases[dis].sublattices) - 1)
        if n_sub < 2:
            continue
        m = re.fullmatch(r"([A-Z]\\d+)_([A-Z]+)", dis)
        if m and f"{m.group(2)}_{m.group(1)}" in db.phases:
            dis = f"{m.group(2)}_{m.group(1)}"
        idx = {(int(y.sublattice_index), str(y.species.name)): j for j, y in enumerate(mod.site_fractions)}
        info[name] = (dis, n_sub, idx)

    def one(cs):
        pn = cs.phase_record.phase_name
        if pn not in info:
            return pn
        dis, n_sub, idx = info[pn]
        y = np.asarray(cs.dof)[cs.phase_record.num_statevars:]
        for sp in {sp for (s, sp) in idx if s < n_sub}:
            if np.ptp([y[idx[(s, sp)]] if (s, sp) in idx else 0.0 for s in range(n_sub)]) > 1e-4:
                return pn
        return dis

    def names(compsets):
        out, seen = [], {}
        for cs in compsets:
            n = one(cs)
            seen[n] = seen.get(n, 0) + 1
            out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
        return out
    return names


def map_diagram(db, els, T, xstep=0.02):
    """Binary T-x map (T = (lo, hi, step) in K) or ternary isotherm (T in K): ZPF lines and invariants."""
    comps = sorted(els) + ["VA"]
    phases = sorted(filter_phases(db, unpack_species(db, comps)))
    conds = {v.P: 101325.0, v.N: 1, v.T: T, v.X(els[1]): (0, 1, xstep)}
    if len(els) == 3:
        conds[v.X(els[2])] = (0, 1, xstep)
    strategy = (BinaryStrategy if len(els) == 2 else TernaryStrategy)(db, comps, phases, conds)
    # map, then check / repair the traced lines against equilibrium as the tool does (mapping_check)
    if len(els) == 2:
        recipe = {"kind": "binary", "elements": list(els), "phases": phases, "P": 101325.0, "T_range": [T[0], T[1]],
                  "x_range": [0.0, 1.0], "tmap": [1.0, 0.0], "comp": {"scale": 1.0, "weight": False}}
    else:
        recipe = {"kind": "ternary", "elements": list(els), "phases": phases, "P": 101325.0, "T_K": T,
                  "comp": {"scale": 1.0, "weight": False}}
    map_and_check(strategy, db, recipe, "auto", 600.0, grid_size(recipe["kind"], labels=False))
    names = phase_namer(db, strategy)
    order = [str(e) for e in strategy.elements if str(e) != "VA"]

    def temp(p):
        return float(p.global_conditions[v.T])

    def comp(cs):
        return {e: float(f"{x:.8g}") for e, x in zip(order, np.asarray(cs.X, float))}
    lines, seen = [], set()
    for z in strategy.zpf_lines:
        if not z.points:
            continue
        if all(max(np.max(np.abs(np.asarray(a.X, float) - np.asarray(b.X, float)))
                   for a, b in itertools.combinations(p.stable_composition_sets, 2)) < 1e-5
               for p in z.points if len(p.stable_composition_sets) > 1):
            continue                                   # zero-length tie-lines along an edge
        sig = tuple(sorted((round(temp(p), 3),) + tuple(np.round(np.concatenate(
            [np.asarray(cs.X, float) for cs in p.stable_composition_sets]), 5)) for p in (z.points[0], z.points[-1])))
        if sig in seen:
            continue                                   # the same line traced twice
        seen.add(sig)
        nm = names(z.points[len(z.points) // 2].stable_composition_sets)
        for k, ph in enumerate(nm):
            lines.append({"region": " + ".join(sorted(nm)), "phase": ph,
                          "T": np.array([float(f"{temp(p):.8g}") for p in z.points]),
                          "X": {e: np.array([comp(p.stable_composition_sets[k])[e] for p in z.points]) for e in order}})
    raw = []
    for node in strategy.node_queue.nodes:
        if len(node.stable_composition_sets) != 3:
            continue
        nm = names(node.stable_composition_sets)
        X = {n: {e: x for e, x in zip(order, np.asarray(cs.X, float))} for n, cs in zip(nm, node.stable_composition_sets)}
        key = sorted(tuple(round(c[e], 3) for e in els) for c in X.values())
        if any(abs(r["T"] - temp(node)) < 0.05 and r["_k"] == key for r in raw):
            continue
        raw.append({"T": round(temp(node), 3), "phases": nm, "_k": key,
                    "X": {n: {e: round(c[e], 6) for e in els} for n, c in X.items()}})
    if len(els) == 2:
        raw.sort(key=lambda r: -r["T"])
    invs = []
    for r in raw:
        key = "+".join(sorted(r["phases"]))
        if any(i["phases_key"] == key and abs(i["T"] - r["T"]) < 0.5 for i in invs):
            continue
        n = sum(i["phases_key"] == key for i in invs)
        invs.append({**r, "phases_key": key, "key": key + (f"#{n + 1}" if n else "")})
    return {"lines": lines, "invariants": invs, "els": list(els)}


def boundaries_at(lines, T, el):
    """Phase-boundary compositions where the ZPF lines cross temperature T (binary)."""
    pts = []
    for ln in lines:
        Ts, Xs = ln["T"], ln["X"][el]
        for i in range(len(Ts) - 1):
            a, b = Ts[i], Ts[i + 1]
            if a != b and min(a, b) <= T <= max(a, b):
                f = (T - a) / (b - a)
                pts.append({"region": ln["region"], "phase": ln["phase"], "X": float(Xs[i] + f * (Xs[i + 1] - Xs[i]))})
    pts.sort(key=lambda p: (p["region"], p["phase"], p["X"]))
    out, count = [], {}
    for p in pts:
        if out and (out[-1]["region"], out[-1]["phase"]) == (p["region"], p["phase"]) and abs(out[-1]["X"] - p["X"]) < 1e-4:
            continue
        k = (p["region"], p["phase"])
        count[k] = count.get(k, 0) + 1
        out.append({**p, "key": f"{p['region']}|{p['phase']}" + (f"#{count[k]}" if count[k] > 1 else "")})
    return out
'''

ESPEI = '''
RESIDUAL_TYPES = {"ZPFResidual": "ZPF", "FixedConfigurationPropertyResidual": "thermo_fixed",
                  "EquilibriumPropertyResidual": "thermo_equilibrium", "ActivityResidual": "activity"}


def residual_blocks(ctx, params):
    """Per residual type: z = residual / sigma and sigma (ZPF: sigma = 1000 J/mol / data weight / dataset weight)."""
    out = {}
    for o in ctx["residual_objs"]:
        kind = RESIDUAL_TYPES.get(type(o).__name__, type(o).__name__)
        r, w = o.get_residuals(np.asarray(params, float))
        r, w = np.asarray(r, float).ravel(), np.asarray(w, float).ravel()
        sig = 1000.0 / float(getattr(o, "weight", 1.0)) / w if kind == "ZPF" else w
        z, s = out.setdefault(kind, ([], []))
        z.extend(r / sig)
        s.extend(sig)
    return {k: (np.array(z), np.array(s)) for k, (z, s) in out.items()}


def nll_by_type(blocks):
    """-lnL = sum(z^2)/2 + sum(ln sigma) + n ln(2 pi)/2  (identical to -sum(get_likelihood))."""
    return {k: (0.5 * np.sum(z ** 2) + np.sum(np.log(s)) + 0.5 * len(z) * np.log(2 * np.pi))
            if np.all(np.isfinite(z)) else np.inf for k, (z, s) in blocks.items()}
'''


def _tb_paths(ctx: Ctx, args: Dict):
    from . import param_tools as pt
    t = ctx.tb._find_tdb(args["tdb"])
    return pt, t, Path(t).stem


def _arr(vals) -> str:
    return "[" + ", ".join("np.inf" if v == float("inf") else "-np.inf" if v == -float("inf") else repr(float(v))
                           for v in vals) + "]"


def _pairs_code(pairs) -> str:
    return "[\n" + "".join(f"    ({o!r},\n     {n!r}),\n" for o, n in pairs) + "]"


def _datasets_code(ctx: Ctx, args: Dict) -> str:
    ctx.need("from espei.datasets import load_datasets, recursive_glob")
    pm = f"json.loads(Path({ctx.path(args['phase_models'])}).read_text())" if args.get("phase_models") else "None"
    if args.get("phase_models"):
        ctx.need("import json")
    return (f"datasets = load_datasets(recursive_glob(str({ctx.path(args['datasets_dir'])}), '*.json'))\n"
            f"phase_models = {pm}\n"
            f"data_weights = {Ctx.py(args.get('data_weights'))}\n")


# ---------------------------------------------------------------------------------------------------------
@generator("edit_parameters")
def _gen_edit_parameters(args, ctx: Ctx) -> str:
    pt, t, stem = _tb_paths(ctx, args)
    tb = ctx.tb
    in_place = bool(args.get("in_place"))
    src_path = str(tb.out / f"whatif_{stem}" / f"{stem}_original.tdb") if in_place else t
    if in_place and not Path(src_path).exists():
        ctx.fallback("the original of an in-place edit is not available any more")
    src_text = Path(src_path).read_text(encoding="utf-8", errors="replace")
    pairs, _ = pt.edit_pairs(src_text, args["edits"], update_twins=args.get("update_twins", True) is not False)
    out_path = t if in_place else tb.outp(args.get("output_tdb"), f"{stem}_edited.tdb")
    out_name = ctx.out_name(out_path, f"{stem}_edited.tdb")
    ctx.need("import difflib", "from pycalphad import Database")
    src = (ctx.path(src_path) if in_place else ctx.tdb(args["tdb"]))
    code = [APPLY.lstrip("\n"),
            f"text0 = Path({src}).read_text(encoding='utf-8')" + ("   # original saved before the in-place edit"
                                                                   if in_place else ""),
            "# the edited commands, exactly as edit_parameters rewrote them",
            f"EDITS = {_pairs_code(pairs)}",
            "text1 = apply_edits(text0, EDITS)",
            f"(OUT / {out_name!r}).write_text(text1, encoding='utf-8')",
            "print('\\n'.join(difflib.unified_diff(text0.splitlines(), text1.splitlines(), 'original', 'modified', "
            "n=0, lineterm='')))",
            "db0 = Database.from_string(text0, fmt='tdb')",
            "db1 = Database.from_string(text1, fmt='tdb')", ""]
    if args.get("compare"):
        c = dict(args["compare"])
        els = [e.upper() for e in c["elements"]]
        unit = c.get("temperature_unit", "K")
        off = 273.15 if str(unit).upper().startswith("C") else 0.0
        ctx.need("import itertools", "import re", "import numpy as np", "import pandas as pd",
                 "import matplotlib.pyplot as plt", "from matplotlib.lines import Line2D",
                 "from pycalphad import variables as v", "from pycalphad.core.utils import filter_phases, unpack_species",
                 "from pycalphad.mapping import BinaryStrategy, TernaryStrategy")
        if c.get("phases"):
            ctx.fallback("compare with a phase subset is not reproduced standalone")
        if len(els) == 2:
            tlo, thi = sorted((float(c["tmin"]) + off, float(c["tmax"]) + off))
            Tspec = f"({tlo!r}, {thi!r}, {float(c.get('temperature_step', 10.0))!r})"
        else:
            Tspec = repr(float(c["temperature"]) + off)
        tag = "-".join(els)
        png = ctx.out_name(tb.outp(None, f"whatif_{stem}/compare_{tag}.png"), f"compare_{tag}.png")
        code.append(mapping_check_helpers(ctx) + MAPPING.lstrip("\n"))
        code += [f"els = {els!r}",
                 f"m0, m1 = map_diagram(db0, els, {Tspec}), map_diagram(db1, els, {Tspec})",
                 f"OFF = {off!r}   # K -> display unit",
                 "fig = plt.figure(figsize=(7.4, 5.6))",
                 "if len(els) == 3:\n    from pycalphad.plot import triangular  # noqa: F401\n"
                 "    ax = fig.add_subplot(projection='triangular')\nelse:\n    ax = fig.add_subplot()",
                 "colors = {}\n"
                 "for m, ls in ((m0, (0, (4, 3))), (m1, '-')):\n"
                 "    for ln in m['lines']:\n"
                 "        c = colors.setdefault(ln['phase'], f'C{len(colors) % 10}')\n"
                 "        y = ln['X'][els[2]] if len(els) == 3 else ln['T'] - OFF\n"
                 "        ax.plot(ln['X'][els[1]], y, ls=ls, color=c, lw=1.3)\n"
                 "    for inv in m['invariants']:\n"
                 "        xs = [inv['X'][p][els[1]] for p in inv['phases']]\n"
                 "        if len(els) == 3:\n"
                 "            ys = [inv['X'][p][els[2]] for p in inv['phases']]\n"
                 "            ax.plot(xs + xs[:1], ys + ys[:1], color='#d62728', ls=ls, lw=1.0)\n"
                 "        else:\n"
                 "            ax.plot([min(xs), max(xs)], [inv['T'] - OFF] * 2, color='#d62728', ls=ls, lw=1.0)",
                 "handles = [Line2D([], [], color=c, lw=2, label=p) for p, c in sorted(colors.items())]\n"
                 "handles += [Line2D([], [], color='k', ls=(0, (4, 3)), label='original'), "
                 "Line2D([], [], color='k', label='modified')]\n"
                 "ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)",
                 f"ax.set_title('{tag}: original (dashed) vs modified (solid)', fontsize=10)",
                 f"fig.savefig(OUT / {png!r}, dpi=150, bbox_inches='tight')",
                 "",
                 "# shifts of invariants (and of phase boundaries at the requested temperatures)",
                 "rows = []",
                 "I0 = {i['key']: i for i in m0['invariants']}\nI1 = {i['key']: i for i in m1['invariants']}",
                 "for k in list(I0) + [k for k in I1 if k not in I0]:\n"
                 "    a, b = I0.get(k), I1.get(k)\n"
                 "    rows.append({'invariant': k, 'T0_K': a and a['T'], 'T1_K': b and b['T'],\n"
                 "                 'dT_K': (b['T'] - a['T']) if a and b else None})\n"
                 "    print(k, rows[-1])",
                 f"for T in {[float(x) for x in (c.get('boundary_temperatures') or [])]!r} if len(els) == 2 else []:\n"
                 "    p0 = {p['key']: p for p in boundaries_at(m0['lines'], T + OFF, els[1])}\n"
                 "    p1 = {p['key']: p for p in boundaries_at(m1['lines'], T + OFF, els[1])}\n"
                 "    for k in list(p0) + [k for k in p1 if k not in p0]:\n"
                 "        a, b = p0.get(k), p1.get(k)\n"
                 "        rows.append({'T': T, 'boundary': k, 'X0': a and a['X'], 'X1': b and b['X'],\n"
                 "                     'dX': (b['X'] - a['X']) if a and b else None})\n"
                 "        print(T, k, rows[-1])",
                 f"pd.DataFrame(rows).to_csv(OUT / {Path(png).stem + '_shifts.csv'!r}, index=False)", ""]
    if args.get("datasets_dir"):
        ctx.need("import numpy as np", "from espei.error_functions.context import setup_context")
        code += [ESPEI.lstrip("\n"), _datasets_code(ctx, args),
                 "# ESPEI needs >= 1 fitted symbol: an unused dummy FUNCTION",
                 "def objective(text):\n"
                 "    db = Database.from_string(text + '\\n FUNCTION DV9999 298.15 0; 6000 N !\\n', fmt='tdb')\n"
                 "    ctx = setup_context(db, datasets, ['DV9999'], data_weights=data_weights, phase_models=phase_models)\n"
                 "    return {RESIDUAL_TYPES.get(type(o).__name__, type(o).__name__): -float(o.get_likelihood(np.zeros(1)))\n"
                 "            for o in ctx['residual_objs']}",
                 "nll0, nll1 = objective(text0), objective(text1)",
                 "print(f'ESPEI -lnL: {sum(nll0.values()):.4f} -> {sum(nll1.values()):.4f}')",
                 "for k in nll0:\n    print(f'  {k}: {nll0[k]:.4f} -> {nll1[k]:.4f}')", ""]
    return "\n".join(code)


# ---------------------------------------------------------------------------------------------------------
def _free_params_code(pt, tt, fps, update_twins: bool):
    """Code defining PARAMS, X0, DIRECT, SYMBOLS and the parameterised TDB text (DV symbols)."""
    t2 = pt.TDBText(tt.text, tt.path)
    edits, syms = [], []
    for f in fps:
        if not f.direct:
            edits.append({"target": f.target, "op": "add", "value": f.symbol, "term": f.basis})
            syms.append({"target": f.symbol, "op": "add_parameter", "value": 0})
    if update_twins:
        edits += pt.twin_edits(tt, edits)[0]
    pt.apply_edits(t2, edits + syms)
    return (f"PARAMS = {Ctx.py([f.label for f in fps])}\n"
            f"X0 = np.array({Ctx.py([f.x0 for f in fps])})      # start values (J/mol, J/mol/K, ...)\n"
            f"SCALE = np.array({Ctx.py([f.scale for f in fps])})   # max(|x0|, typical magnitude)\n"
            f"SYMBOLS = {Ctx.py([f.symbol for f in fps])}   # fitted TDB symbols (DVnnnn = added to a coefficient)\n"
            f"DIRECT = {Ctx.py([f.direct for f in fps])}   # True: the symbol IS the value; False: value - X0\n"
            "# parameterised TDB: every coefficient being varied carries its DVnnnn symbol (initially 0)\n"
            f"PARAM_EDITS = {_pairs_code(t2.pairs)}\n"
            "param_text = apply_edits(text0, PARAM_EDITS)\n"
            "ORDER = sorted(SYMBOLS)   # ESPEI's parameter order\n\n"
            "def espei_vector(x):\n"
            "    vec = np.zeros(len(ORDER))\n"
            "    for sym, direct, xi, x0 in zip(SYMBOLS, DIRECT, x, X0):\n"
            "        vec[ORDER.index(sym)] = xi if direct else xi - x0\n"
            "    return vec\n")


@generator("parameter_sensitivity")
def _gen_parameter_sensitivity(args, ctx: Ctx) -> str:
    pt, t, stem = _tb_paths(ctx, args)
    tb = ctx.tb
    tt = pt.TDBText.read(t)
    twins = args.get("update_twins", True) is not False
    fps = pt.resolve_free_params(tt, args.get("parameters"), args.get("phase"))
    rel = float(args.get("rel_step") or 0.01)
    steps = [pt._fd_step(f, rel) for f in fps]
    pre = tb.outp(args.get("output_prefix"), f"sensitivity_{stem}/sens_{stem}")
    csv_name = ctx.out_name(pre + "_sensitivity.csv", f"sens_{stem}_sensitivity.csv")
    ctx.need("import numpy as np", "import pandas as pd", "from pycalphad import Database")
    code = [APPLY.lstrip("\n"), f"text0 = Path({ctx.tdb(args['tdb'])}).read_text(encoding='utf-8')",
            f"STEPS = {Ctx.py(steps)}   # central-difference steps = {rel:g} x max(|p|, typical)", ""]
    code.append(_free_params_code(pt, tt, fps, twins))
    code.append("rows = []")
    if args.get("datasets_dir"):
        ctx.need("from espei.error_functions.context import setup_context")
        code += [ESPEI.lstrip("\n"), _datasets_code(ctx, args),
                 "ctx = setup_context(Database.from_string(param_text, fmt='tdb'), datasets, SYMBOLS, "
                 "data_weights=data_weights, phase_models=phase_models)",
                 "b0 = residual_blocks(ctx, espei_vector(X0))",
                 "types = [k for k, (z, s) in b0.items() if len(z)]",
                 "base = nll_by_type(b0)",
                 "J = []   # d(residual/sigma)/dp: Gauss-Newton Hessian of -lnL = J^T J",
                 "for i, h in enumerate(STEPS):\n"
                 "    res = []\n"
                 "    for sgn in (1, -1):\n"
                 "        x = X0.copy()\n"
                 "        x[i] += sgn * h\n"
                 "        res.append(residual_blocks(ctx, espei_vector(x)))\n"
                 "    nlp, nlm = nll_by_type(res[0]), nll_by_type(res[1])\n"
                 "    rows.append({'quantity': 'NLL_total', 'parameter': PARAMS[i], 'base_value': sum(base.values()),\n"
                 "                 'derivative': (sum(nlp.values()) - sum(nlm.values())) / (2 * h)})\n"
                 "    for k in types:\n"
                 "        rows.append({'quantity': f'NLL_{k}', 'parameter': PARAMS[i], 'base_value': base[k],\n"
                 "                     'derivative': (nlp[k] - nlm[k]) / (2 * h)})\n"
                 "    J.append(np.concatenate([(res[0][k][0] - res[1][k][0]) / (2 * h) for k in types]))",
                 "J = np.array(J).T",
                 "F = J.T @ J",
                 "cov = np.linalg.pinv(F * np.outer(SCALE, SCALE), rcond=1e-12)",
                 "std = np.sqrt(np.clip(np.diag(cov), 0, None)) * SCALE",
                 "d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))",
                 "corr = cov / np.outer(d, d)",
                 "print('approx std:', dict(zip(PARAMS, std.round(6))))",
                 "print('correlation:\\n', pd.DataFrame(corr, index=PARAMS, columns=PARAMS).round(3))", ""]
    use_map = bool(args.get("invariants") or args.get("boundary_temperatures"))
    if use_map or args.get("properties"):
        els = [e.upper() for e in args["elements"]]
        unit = args.get("temperature_unit") or "K"
        off = 273.15 if str(unit).upper().startswith("C") else 0.0
        if args.get("phases"):
            ctx.fallback("a phase subset is not reproduced standalone")
        pert = []
        for f, h in zip(fps, steps):
            for sgn in (1, -1):
                eds = pt.result_edits([f], [f.x0 + sgn * h])
                pairs, _ = pt.edit_pairs(tt.text, eds, update_twins=twins)
                pert.append(pairs)
        code.append(f"# TDB text with parameter i changed by +h / -h (PERTURB[2*i] / PERTURB[2*i+1])\n"
                    f"PERTURB = [\n" + "".join(f"    {_pairs_code(p)},\n" for p in pert) + "]")
        qf = ["def quantities(text):", "    db = Database.from_string(text, fmt='tdb')", "    q = {}"]
        if use_map:
            ctx.need("import itertools", "import re", "from pycalphad import variables as v",
                     "from pycalphad.core.utils import filter_phases, unpack_species",
                     "from pycalphad.mapping import BinaryStrategy, TernaryStrategy")
            code.append(mapping_check_helpers(ctx) + MAPPING.lstrip("\n"))
            tlo, thi = sorted((float(args["tmin"]) + off, float(args["tmax"]) + off))
            bT = [float(x) + off for x in (args.get("boundary_temperatures") or [])]
            qf += [f"    m = map_diagram(db, {els!r}, ({tlo!r}, {thi!r}, 10.0))"]
            if args.get("invariants"):
                qf += ["    for inv in m['invariants']:",
                       "        q[f\"T_inv[{inv['key']}]\"] = float(inv['T'])",
                       "        for ph, c in inv['X'].items():",
                       f"            q[f\"X({els[1]})_inv[{{inv['key']}}]:{{ph}}\"] = float(c[{els[1]!r}])"]
            if bT:
                qf += [f"    for T in {bT!r}:",
                       f"        for p in boundaries_at(m['lines'], T, {els[1]!r}):",
                       f"            q[f\"X({els[1]})@{{T:g}}K[{{p['key']}}]\"] = p['X']"]
        if args.get("properties"):
            import json as _json
            ctx.need("from pycalphad import Workspace, variables as v",
                     "from pycalphad.core.utils import filter_phases, unpack_species")
            conds = []
            for k, val in (args.get("conditions") or {}).items():
                ku = k.upper().replace(" ", "")
                if ku == "T":
                    conds.append(f"v.T: {float(val) + off!r}")
                elif ku == "P":
                    conds.append(f"v.P: {float(val)!r}")
                else:
                    conds.append(f"v.X({ku.replace('X(', '').replace(')', '')!r}): {float(val)!r}")
            if not any(c.startswith("v.P") for c in conds):
                conds.insert(0, "v.P: 101325.0")
            conds.append("v.N: 1.0")
            tag = _json.dumps(args.get("conditions") or {}, sort_keys=True)
            els_s = sorted(els)
            qf += [f"    comps = {els_s!r} + (['VA'] if 'VA' in db.elements else [])",
                   "    phases = sorted(filter_phases(db, unpack_species(db, comps)))",
                   f"    wks = Workspace(db, comps, phases, {{{', '.join(conds)}}})",
                   f"    props = {list(args['properties'])!r}",
                   "    for p, val in zip(props, wks.get(*props)):",
                   f"        q[p + {'@' + tag!r}] = float(np.squeeze(val))"]
        qf.append("    return q")
        code.append("\n".join(qf))
        code += ["q0 = quantities(text0)",
                 "for i, h in enumerate(STEPS):\n"
                 "    qp = quantities(apply_edits(text0, PERTURB[2 * i]))\n"
                 "    qm = quantities(apply_edits(text0, PERTURB[2 * i + 1]))\n"
                 "    for k, b in q0.items():\n"
                 "        if k in qp and k in qm:\n"
                 "            d = (qp[k] - qm[k]) / (2 * h)\n"
                 "        elif k in qp:\n"
                 "            d = (qp[k] - b) / h\n"
                 "        elif k in qm:\n"
                 "            d = (b - qm[k]) / h\n"
                 "        else:\n"
                 "            d = np.nan\n"
                 "        rows.append({'quantity': k, 'parameter': PARAMS[i], 'base_value': b, 'derivative': d})", ""]
    code += ["table = pd.DataFrame(rows)",
             f"table.to_csv(OUT / {csv_name!r}, index=False)",
             "print(table.to_string(max_rows=60))"]
    return "\n".join(code)


# ---------------------------------------------------------------------------------------------------------
@generator("optimize_parameters")
def _gen_optimize_parameters(args, ctx: Ctx) -> str:
    pt, t, stem = _tb_paths(ctx, args)
    tb = ctx.tb
    tt = pt.TDBText.read(t)
    fps = pt.resolve_free_params(tt, args.get("parameters"), args.get("phase"))
    twins = args.get("update_twins", True) is not False
    method = args.get("method") or "least_squares"
    n = len(fps)
    x0 = [f.x0 for f in fps]
    s = [f.scale for f in fps]
    lo, hi = [-float("inf")] * n, [float("inf")] * n
    for k, (a, b) in (args.get("bounds") or {}).items():
        for i, f in enumerate(fps):
            if k in (f.label, f.target, f.symbol):
                if a is not None:
                    lo[i] = (float(a) - x0[i]) / s[i]
                if b is not None:
                    hi[i] = (float(b) - x0[i]) / s[i]
    cap = args.get("max_nfev") or (60 * max(1, n) if method == "least_squares" else 200)
    out_tdb = tb.outp(args.get("output_tdb"), f"optimize_{stem}/{stem}_optimized.tdb")
    tdb_name = ctx.out_name(out_tdb, f"{stem}_optimized.tdb")
    csv_name = ctx.out_name(str(Path(out_tdb).parent / f"{stem}_parameters.csv"), f"{stem}_parameters.csv")
    ctx.need("import re", "import numpy as np", "import pandas as pd", "from pycalphad import Database",
             "from scipy.optimize import least_squares, minimize",
             "from espei.error_functions.context import setup_context")
    prior = args.get("prior_std")
    code = [APPLY.lstrip("\n"), ESPEI.lstrip("\n"), f"text0 = Path({ctx.tdb(args['tdb'])}).read_text(encoding='utf-8')",
            _free_params_code(pt, tt, fps, twins), _datasets_code(ctx, args),
            "ctx = setup_context(Database.from_string(param_text, fmt='tdb'), datasets, SYMBOLS, "
            "data_weights=data_weights, phase_models=phase_models)",
            "b0 = residual_blocks(ctx, espei_vector(X0))",
            "TYPES = [k for k, (z, s) in b0.items() if len(z)]",
            "SIZES = {k: len(b0[k][0]) for k in TYPES}",
            "CONST = sum(np.sum(np.log(b0[k][1])) + 0.5 * SIZES[k] * np.log(2 * np.pi) for k in TYPES)",
            f"PRIOR_STD = {'None' if prior is None else f'{float(prior)!r} * SCALE'}   # Gaussian prior around X0",
            "",
            "def zvec(x):\n"
            "    \"\"\"All residual/sigma (a failed calculation of a data type -> 1000 for each of its points).\"\"\"\n"
            "    bl = residual_blocks(ctx, espei_vector(x))\n"
            "    parts = []\n"
            "    for k in TYPES:\n"
            "        z = bl.get(k, (np.array([]), None))[0]\n"
            "        parts.append(z if len(z) == SIZES[k] and np.all(np.isfinite(z)) else np.full(SIZES[k], 1e3))\n"
            "    if PRIOR_STD is not None:\n"
            "        parts.append((x - X0) / PRIOR_STD)\n"
            "    return np.concatenate(parts)\n",
            "def nll(x):\n"
            "    \"\"\"ESPEI's objective: -sum(get_likelihood) (+ prior).\"\"\"\n"
            "    f = 0.0\n"
            "    for o in ctx['residual_objs']:\n"
            "        r = o.get_likelihood(espei_vector(x))\n"
            "        f -= float(r[0] if isinstance(r, tuple) else r)   # (lnL, gradient) with ESPEI PR #268\n"
            "    if PRIOR_STD is not None:\n"
            "        f += 0.5 * np.sum(((x - X0) / PRIOR_STD) ** 2)\n"
            "    return f if np.isfinite(f) else 1e30\n",
            "# optimise in scaled variables u = (x - X0) / SCALE",
            f"lo, hi = np.array({_arr(lo)}), np.array({_arr(hi)})",
            ]
    ds = float(args.get("diff_step") or 1e-3)
    if method == "least_squares":
        code += [f"res = least_squares(lambda u: zvec(X0 + u * SCALE), np.zeros(len(X0)), jac='3-point', bounds=(lo, hi),\n"
                 f"                    method='trf', diff_step={ds!r}, x_scale=1.0, max_nfev={cap}, xtol=1e-10, "
                 "ftol=1e-10, gtol=1e-10)",
                 "print(res.message)"]
    else:
        code += [f"H = {ds!r}\n"
                 "def grad(u):\n"
                 "    \"\"\"Central differences in u.\"\"\"\n"
                 "    g = np.zeros(len(u))\n"
                 "    for i in range(len(u)):\n"
                 "        e = np.zeros(len(u))\n"
                 "        e[i] = H\n"
                 "        g[i] = (nll(X0 + (u + e) * SCALE) - nll(X0 + (u - e) * SCALE)) / (2 * H)\n"
                 "    return g",
                 "res = minimize(lambda u: nll(X0 + u * SCALE), np.zeros(len(X0)), jac=grad, method='L-BFGS-B',\n"
                 "               bounds=[(a if np.isfinite(a) else None, b if np.isfinite(b) else None) "
                 "for a, b in zip(lo, hi)],\n"
                 f"               options={{'maxiter': {cap}, 'ftol': 1e-12, 'gtol': 1e-8}})",
                 "print(res.message)"]
    code += ["x = X0 + res.x * SCALE",
             "print(f'-lnL: {nll(X0):.4f} -> {nll(x):.4f}')",
             "# approximate std from J^T J at the optimum (J of residual/sigma, central differences)",
             "J = np.zeros((len(zvec(x)), len(x)))",
             f"for i in range(len(x)):\n"
             f"    h = {ds!r} * SCALE[i]\n"
             "    e = np.zeros(len(x))\n"
             "    e[i] = h\n"
             "    J[:, i] = (zvec(x + e) - zvec(x - e)) / (2 * h)",
             "std = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J, rcond=1e-12)), 0, None))",
             "table = pd.DataFrame({'parameter': PARAMS, 'before': X0, 'after': x, 'change': x - X0, 'approx_std': std})",
             "print(table.to_string())",
             f"table.to_csv(OUT / {csv_name!r}, index=False)",
             "",
             "# optimized TDB: the parameterised database with the fitted symbol values (numerically the same as the",
             "# tool's output, which writes the values into the coefficients themselves)",
             "def set_function(text, name, value):\n"
             "    return re.sub(rf'(FUNCTION\\s+{name}\\s+\\S+\\s+)[^;!]*;', lambda m: m.group(1) + repr(float(value)) + ';',\n"
             "                  text, count=1)",
             "text_opt = param_text",
             "for sym, val in zip(ORDER, espei_vector(x)):\n"
             "    text_opt = set_function(text_opt, sym, val)",
             f"(OUT / {tdb_name!r}).write_text(text_opt, encoding='utf-8')"]
    ctx.note("optimize_parameters: the time limit and the before/after comparison plots are not reproduced")
    return "\n".join(code)
