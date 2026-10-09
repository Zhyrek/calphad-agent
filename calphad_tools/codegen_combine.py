"""Reproduction of ``combine_databases``: ``combine.ipynb`` next to the report, and the show_code generator.

Cells:

1. the hash seed the validation diagrams were mapped with (checked against this kernel);
2. standalone: calphad_tools.tdbparse / pwinline / tdbtables / merge / tdbstage copied in (standard library, numpy,
   symengine), then every step of the chain rebuilt with the same order, normalisation maps and decisions - the
   same merge plans staged_merge used for Q - and the result compared with the tool's combined TDB
   (parameter-equivalent: every parameter numerically equal, same phases);
3. standalone (pycalphad only): the equilibrium-grid agreement and the H/S/Cp/G spot check of every source binary
   against the rebuilt database;
4. [calphad_tools]: the mapped overlays, invariant and boundary comparison and verdicts (shaded mapping with the
   mapping check has no standalone generator) via combine.validate_from_state.

Cells 3-4 run unless the environment variable COMBINE_NOTEBOOK_VALIDATION=0 (the test suite rebuilds only).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from .codegen import Ctx, generator

_REBUILD = r'''
def merge_step(k, a, b, step):
    """One step as staged_merge builds Q: B normalised to A's conventions (A rescaled by the magnetic basis rule),
    aliases kept separate, the staged rule's decisions (base wins element data, FUNCTIONs, unaries of shared
    elements) under the logged decisions."""
    A, B = parse_tdb(str(a)), parse_tdb(str(b))
    An, Bn, notes = normalize_pair(A, B, step["phase_map"] or None, step["sublattice_map"] or None, on_error="keep")
    a_eff, b_eff = str(a), str(b)
    if any(n.action in A_CHANGES for n in notes):
        a_eff = write_tdb(An, str(OUT / f"step{k}_A0.tdb"))
    if any(n.action in B_CHANGES for n in notes):
        b_eff = write_tdb(Bn, str(OUT / f"step{k}_B0.tdb"))
    for n in notes:
        print(f"step {k} normalisation: {n.text()}")
    shared = sorted(set(A.real_elements) & set(parse_tdb(b_eff).real_elements))
    if step.get("extensions"):
        # [calphad_tools] 'extend' decisions (phase harmonisation): A's phase extended with B's components,
        # B's source phase left out of the merge - harmonize.staged_extensions has no standalone copy
        from calphad_tools.harmonize import staged_extensions
        ext = staged_extensions(a_eff, b_eff, step["extensions"], str(OUT / f"step{k}_extend"), shared, verify=False,
                                tentative=step.get("tentative", True))
        a_eff, b_eff = ext["A_ext"], ext["B_merge"]
        for x in ext["reports"]:
            print(f"step {k} extend: " + x["text"].split("\n")[0])
    dec = alias_separate([a_eff, b_eff], step["decisions"])
    auto = auto_decisions(plan_merge([a_eff, b_eff], dec, "ask"), shared)
    dec = {**auto, **dec, **step["decisions"]}
    # tentative parameters (TENT_MIX / TENT_VA) as the tool added them: step["tentative"] = its setting
    plan, text = merged_text([a_eff, b_eff], dec, "ask", tentative=step.get("tentative", True))
    for t in plan.tentative:
        print(f"step {k} tentative: {t.text()}")
    if text is None:
        raise SystemExit(f"step {k}: decisions still needed\n" + plan.report())
    q = OUT / f"Q{k}_{Path(str(b)).stem}.tdb"
    q.write_text(text)
    print(f"step {k}: {Path(str(a)).name} + {Path(str(b)).name} -> {q.name}")
    return q


q = SOURCES[ORDER[0]]
for k, (name, step) in enumerate(zip(ORDER[1:], STEPS), 1):
    q = merge_step(k, q, SOURCES[name], step)
F_REBUILT = OUT / "combined_rebuilt.tdb"
F_REBUILT.write_text(Path(q).read_text())

# the rebuilt database against the tool's: every parameter numerically equal (FUNCTIONs inlined), same phases
X, Y = StageView(parse_tdb(str(F_REBUILT)), "rebuilt"), StageView(parse_tdb(str(TOOL_FINAL)), "tool")
els = sorted(X.own_elements | Y.own_elements)
d = diff_versions(X, Y, stage_subsystems(els, (2, 3)))
EQUIVALENT = not d.params and not d.changed and sorted(X.phases) == sorted(Y.phases)
print("parameter-equivalent to the tool's combined TDB:", EQUIVALENT)
for s in d.changed[:10]:
    print("  ", s.name, s.reasons(3))
'''

_GRIDS = r'''
if RUN_VALIDATION:
    import warnings
    import numpy as np
    import pandas as pd
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.core.utils import filter_phases, unpack_species

    from pycalphad import Model

    def disorder_labels(db, comps, phases):
        """Ordered phases whose disordered states are named by their disordered phase (as the diagrams do):
        L12_FCC with equal site fractions on its substitutional sublattices is FCC_A1 (A1_FCC -> FCC_A1)."""
        info = {}
        for name in phases:
            ph = db.phases[name]
            dis = ph.model_hints.get("disordered_phase")
            if ph.model_hints.get("ordered_phase") != name or not dis or dis not in db.phases:
                continue
            n_sub = len(ph.sublattices) - (len(db.phases[dis].sublattices) - 1)
            if n_sub < 2:
                continue
            a, _, b = dis.partition("_")
            if f"{b}_{a}" in db.phases:
                dis = f"{b}_{a}"
            mod = Model(db, comps, name)
            info[name] = (dis, n_sub, {(int(y.sublattice_index), str(y.species.name)): j
                                       for j, y in enumerate(mod.site_fractions)})
        return info

    def label(name, y, info):
        if name not in info:
            return name
        dis, n_sub, idx = info[name]
        for sp in {sp for (s, sp) in idx if s < n_sub}:
            if np.ptp([y[idx[(s, sp)]] if (s, sp) in idx else 0.0 for s in range(n_sub)]) > 1e-4:
                return name
        return dis

    def setup(db, els):
        comps = sorted(els) + ["VA"]
        phases = sorted(filter_phases(db, unpack_species(db, comps)))
        return comps, phases, disorder_labels(db, comps, phases)

    def stable_sets(db, els, xs, Ts):
        comps, phases, info = setup(db, els)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eq = equilibrium(db, comps, phases, {v.X(els[1]): xs, v.T: Ts, v.P: 101325.0, v.N: 1.0})
        dims = ("N", "P", "T", f"X_{els[1]}")
        ph = eq.Phase.transpose(*dims, "vertex").values[0, 0]
        Y = eq.Y.transpose(*dims, "vertex", "internal_dof").values[0, 0]
        return [["+".join(sorted(label(str(p), Y[i, j, k], info) for k, p in enumerate(ph[i, j]) if str(p)))
                 for j in range(len(xs))] for i in range(len(Ts))]

    def spot(db, els, x, T):
        comps, phases, info = setup(db, els)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = equilibrium(db, comps, phases, {v.X(els[1]): x, v.T: T, v.P: 101325.0, v.N: 1.0},
                            output=["HM", "SM", "CPM"])
        out = {p: float(np.squeeze(getattr(q, p).values)) for p in ("GM", "HM", "SM", "CPM")}
        ph = q.Phase.values.reshape(-1)
        Y = q.Y.values.reshape(len(ph), -1)
        out["phases"] = "+".join(sorted(label(str(p), Y[k], info) for k, p in enumerate(ph) if str(p)))
        return out

    dbF = Database(str(F_REBUILT))
    xs = np.round(np.linspace(0.02, 0.98, GRID["nx"]), 6)
    Ts = np.round(np.linspace(TMIN_C, TMAX_C, GRID["nT"]), 4) + 273.15
    grid_rows, prop_rows = [], []
    for c in CHECKS:
        dbS = Database(str(SOURCES[c["source"]]))
        ren = lambda s: "+".join(sorted(c["name_map"].get(p, p) for p in s.split("+") if p))   # noqa: E731
        a, b = stable_sets(dbS, c["elements"], xs, Ts), stable_sets(dbF, c["elements"], xs, Ts)
        pairs = [(ren(sa), sb) for ra, rb in zip(a, b) for sa, sb in zip(ra, rb) if sa and sb]
        agree = sum(x == y for x, y in pairs) / max(1, len(pairs))
        grid_rows.append({"source": c["source"], "subsystem": "-".join(e.capitalize() for e in c["elements"]),
                          "points": len(pairs), "agreement": agree})
        span = TMAX_C - TMIN_C
        for f in (0.25, 0.5, 0.75):
            for x in (0.25, 0.5, 0.75):
                T = TMIN_C + f * span + 273.15
                ps, pf = spot(dbS, c["elements"], x, T), spot(dbF, c["elements"], x, T)
                prop_rows.append({"source": c["source"], "subsystem": grid_rows[-1]["subsystem"], "x": x, "T_K": T,
                                  "S phases": ren(ps["phases"]), "F phases": pf["phases"],
                                  **{f"d{p}": pf[p] - ps[p] for p in ("GM", "HM", "SM", "CPM")}})
    pd.DataFrame(grid_rows).to_csv(OUT / "grid_agreement.csv", index=False)
    pd.DataFrame(prop_rows).to_csv(OUT / "property_check.csv", index=False)
    print(pd.DataFrame(grid_rows).to_string(index=False))
    print(pd.DataFrame(prop_rows).groupby(["source", "subsystem"])[["dGM", "dHM", "dSM", "dCPM"]]
          .agg(lambda s: s.abs().max()).to_string())
'''

_CT_VALIDATION = r'''
if RUN_VALIDATION:
    # NOTE: through calphad_tools - the shaded mapping with the mapping check, the invariant / boundary comparison
    # and the overlays have no standalone generator.  Re-validates the REBUILT database against every source.
    from calphad_tools.combine import validate_from_state
    V = validate_from_state(str(STATE), final_tdb=str(F_REBUILT), out_dir=str(OUT / "validation"), lint=False)
    print(V["text"])
'''


def cells(state: Dict[str, Any], src: Callable[[str], str], ctx: Ctx) -> List[Tuple[str, str, bool]]:
    """[(title, code, standalone)] of the reproduction."""
    from .codegen_staged import _helpers
    ctx.need("import os", "from pathlib import Path")
    helpers = _helpers(ctx)
    order = [Path(s["path"]).stem for s in state["sources"]]
    srcs = "{\n" + "".join(f"    {Path(s['path']).stem!r}: {src(s['path'])},\n" for s in state["sources"]) + "}"
    steps = [{"phase_map": st["phase_map"], "sublattice_map": st["sublattice_map"], "decisions": st["decisions"],
              "tentative": st.get("tentative", True),
              **({"extensions": st["extensions"]} if st.get("extensions") else {})} for st in state["steps"]]
    ext = any(st.get("extensions") for st in state["steps"])
    consts = (f"SOURCES = {srcs}\n"
              f"ORDER = {Ctx.py(order)}   # merge order (base first)\n"
              f"# per step: normalisation maps and the decisions of the log (user / agent / policy)\n"
              f"STEPS = {Ctx.py(steps)}\n"
              f"TOOL_FINAL = Path({state['final']!r})   # the tool's combined TDB, compared below\n"
              f"STATE = Path({str(Path(state['out_dir']) / 'combine_state.json')!r})\n"
              f"RUN_VALIDATION = os.environ.get('COMBINE_NOTEBOOK_VALIDATION', '1') != '0'\n")
    checks = []
    for s in state["sources"]:
        for els, T in s["items"]:
            if T is None:
                checks.append({"source": Path(s["path"]).stem, "elements": list(els), "name_map": s["name_map"]})
    grid = (f"CHECKS = {Ctx.py(checks)}\n"
            f"TMIN_C, TMAX_C = {state['settings']['tmin_c']!r}, {state['settings']['tmax_c']!r}\n"
            f"GRID = {Ctx.py(state['grid'])}\n")
    return [("rebuild the combined database (calphad_tools' TDB parser and merge planner copied in"
             + ("; the 'extend' decisions through calphad_tools.harmonize)" if ext else ")"),
             helpers + consts + _REBUILD, not ext),
            ("equilibrium-grid agreement and property spot check (pycalphad)", grid + _GRIDS, True),
            ("validation overlays, invariants and verdicts", _CT_VALIDATION, False)]


def write_notebook(state: Dict[str, Any], path: str) -> str:
    from .codegen import _SEED_CELL, _SEED_CELL_RANDOM
    ctx = Ctx(None)
    roots = [str(Path(s["path"]).resolve().parent) for s in state["sources"]]
    root = Path(os.path.commonpath(roots)) if roots else Path(".")

    def src(p):
        r = os.path.relpath(str(Path(p).resolve()), str(root))
        return f"WORKDIR / {r!r}"
    cs = cells(state, src, ctx)
    seed = str(state.get("hash_seed") or "random")
    seed_src = (_SEED_CELL.lstrip("\n").replace("<<seed>>", repr(seed)).replace("<<nb>>", repr(Path(path).name))
                if seed.isdigit() else _SEED_CELL_RANDOM.lstrip("\n"))
    header = "\n".join(["# Standalone reproduction of combine_databases", *ctx.imports, "",
                        f"WORKDIR = Path({str(root)!r})   # folder with the input TDBs",
                        "OUT = Path('combine_reproduction').resolve()   # where this notebook writes",
                        "OUT.mkdir(exist_ok=True)"]) + "\n"

    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.splitlines(keepends=True)}

    def md(s):
        return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
    intro = ("# Reproduction of the combined database\n"
             f"Inputs: {', '.join(Path(s['path']).name for s in state['sources'])} (order as listed). "
             "The first code cells rebuild the combined TDB from the inputs with the same order and decisions and "
             "check it is parameter-equivalent to the tool's file; the next recompute the grid / property tables "
             "with pycalphad; the last (marked [calphad_tools]) regenerates the mapped overlays and verdicts. "
             "Set COMBINE_NOTEBOOK_VALIDATION=0 to rebuild only.\n\n"
             "Decisions applied (who: reason):\n\n"
             + "\n".join(f"- `{e['id']}` -> {e['chosen']} ({e['who']}: {e['reason']})" for e in state["log"]
                         if e["who"] != "rule") + "\n")
    nb = [md(intro), code(seed_src), code(header)]
    for title, c, standalone in cs:
        nb.append(md(f"### {title}" + ("" if standalone else "  [calphad_tools]")))
        nb.append(code(c.rstrip() + "\n"))
    out = {"cells": nb, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                    "language": "python"}, "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    Path(path).write_text(json.dumps(out, indent=1))
    return path


@generator("combine_databases")
def _gen_combine(args, ctx: Ctx):
    from .combine import STATE_FILE, default_output_dir
    paths = [ctx.tb._find_tdb(t) for t in args.get("tdbs") or []]
    od = Path(ctx.tb.p(args["output_dir"])) if args.get("output_dir") else ctx.tb.out / default_output_dir(paths)
    sp = od / STATE_FILE
    if not sp.exists():
        ctx.fallback(f"no {STATE_FILE} in {od} (the call stopped at the plan stage or its folder was moved)")
    state = json.loads(sp.read_text())
    ctx.mapped(None)
    ctx.seeds[-1] = str(state.get("hash_seed") or "random")
    out = []
    for title, c, standalone in cells(state, ctx.path, ctx):
        out.append((title + ("" if standalone else " [calphad_tools]"), c))
    if any(not sa for _, _, sa in cells(state, ctx.path, ctx)):
        ctx.note("the last cell (overlays and verdicts) runs through calphad_tools: the shaded mapping with the "
                 "mapping check has no standalone generator")
    return out
