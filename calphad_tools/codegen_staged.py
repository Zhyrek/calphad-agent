"""show_code generators for evaluate_compatibility and staged_merge.

Both scripts copy calphad_tools.tdbparse / pwinline / tdbtables / merge / tdbstage (standard library, numpy,
symengine) so the TDB text handling is exactly the tool's:

* evaluate_compatibility: also copies compatibility.py and runs the same six stages (text + Excel; the short
  PowerPoint is left out, and phase-name candidates from compare_system are only the normalisation ones).
* staged_merge: normalises B (and, for a magnetic B phase, rescales A's phase to it - the magnetic basis rule), writes B1, B2, B3, Q (and A1, A2) with the same text substitutions / merge plans and
  the decisions the tool used, prints the subsystem changes between consecutive versions, loads every version in
  pycalphad and draws the binaries / ternary isotherms that changed with pycalphad's binplot / ternplot (plain
  colour style, with the tool's mapping check between mapping and plotting; the tool's shaded diagrams, invariant
  extraction and the deck are not reproduced).
"""
from __future__ import annotations

from pathlib import Path

from .codegen import Ctx, generator
from .codegen_compare import module_source


def _helpers(ctx: Ctx, with_compat: bool = False) -> str:
    from . import merge, pwinline, tdbparse, tdbstage, tdbtables, tentative
    ctx.need("import types")
    src = (module_source(ctx, tdbparse, "TDB text parser (keeps expressions exactly as written)")
           + module_source(ctx, pwinline, "piecewise-in-T algebra: inline FUNCTIONs, merge T ranges")
           + module_source(ctx, tdbtables, "subsystem selection, written expressions")
           + module_source(ctx, tentative, "tentative parameters (TENT_MIX / TENT_VA) for mixings no source "
                                           "describes")
           + "tent = types.SimpleNamespace(**{k: v for k, v in dict(globals()).items() if not k.startswith('__')})"
             "\n\n\n"
           + module_source(ctx, merge, "merge planning / TDB writer (merge_tdbs)")
           + module_source(ctx, tdbstage, "staged-merge building blocks: views, version diff, substitutions, "
                                          "normalisation"))
    src += "ts = types.SimpleNamespace(**{k: v for k, v in dict(globals()).items() if not k.startswith('__')})\n\n\n"
    if with_compat:
        from . import compatibility
        src += module_source(ctx, compatibility, "evaluate_compatibility stages")
    return src


@generator("evaluate_compatibility")
def _gen_evaluate(args, ctx: Ctx) -> str:
    a, b = ctx.tdb(args["base_tdb"]), ctx.tdb(args["add_tdb"])
    unary = args.get("unary50", "auto")
    body = _helpers(ctx, with_compat=True)
    body += (f"BASE, ADD = {a}, {b}\n"
             f"PHASE_MAP = {Ctx.py(args.get('phase_map') or {})}\n"
             f"SUBLATTICE_MAP = {Ctx.py(args.get('sublattice_map') or {})}\n"
             f"UNARY50 = {unary!r}   # 'auto' = cached copy or download from SGTE, 'off', or a local path\n"
             f"ev = evaluate_pair(str(BASE), str(ADD), phase_map=PHASE_MAP, sublattice_map=SUBLATTICE_MAP, "
             f"unary50=UNARY50, rtol={float(args.get('rtol', 1e-4))!r}, workdir=str(OUT / 'compat_work'))\n"
             f"ev.files['xlsx'] = write_xlsx(ev, str(OUT / 'compat_{Path(str(args['base_tdb'])).stem}+"
             f"{Path(str(args['add_tdb'])).stem}.xlsx'))\n"
             "print(summary_text(ev))\n")
    return body


_STAGED_BODY = r'''
def plot_versions(labels, versions, items, tmin_c, tmax_c, x_step, t_step):
    """Binary T-x diagrams / ternary isotherms with pycalphad's binplot / ternplot, with the tools' mapping check
    between mapping and plotting (one PNG per version and item)."""
    for lab in labels:
        db = Database(str(versions[lab]))
        for els, T_c in items:
            comps = sorted(els) + ["VA"]
            phases = [p for p in db.phases if all(any(set(sp.constituents) <= set(comps) for sp in subl)
                                                  for subl in db.phases[p].constituents)]
            name = "-".join(e.capitalize() for e in els) + ("" if T_c is None else f"_{T_c:g}C")
            if T_c is None:
                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                checked_binplot(db, comps, phases, {v.X(els[1]): (0, 1, x_step),
                                                    v.T: (tmin_c + 273.15, tmax_c + 273.15, t_step), v.P: 101325,
                                                    v.N: 1}, plot_kwargs={"ax": ax})
            else:
                fig = plt.figure(figsize=(6.4, 5.6))
                ax = fig.add_subplot(projection="triangular")
                checked_ternplot(db, comps, phases, {v.X(els[1]): (0, 1, x_step), v.X(els[2]): (0, 1, x_step),
                                                     v.T: T_c + 273.15, v.P: 101325, v.N: 1},
                                 x=v.X(els[1]), y=v.X(els[2]), ax=ax)
            ax.set_title(f"{lab}: {name}")
            fig.savefig(OUT / f"{lab}_{name}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)


def forced(els):
    """force_replot as given to the tool: True, a list of subsystem names, or False."""
    if FORCE_REPLOT is True:
        return True
    want = {tuple(sorted(e.upper() for e in str(n).replace(" ", "").split("-"))) for n in (FORCE_REPLOT or [])}
    return tuple(sorted(e.upper() for e in els)) in want


A, B = str(BASE), str(ADD)
A_ORIG = A
dbA, Bn, notes = normalize_pair(parse_tdb(A), parse_tdb(B), PHASE_MAP, SUBLATTICE_MAP)
for n in notes:
    print("normalisation:", n.text())
if any(n.action in B_CHANGES for n in notes):
    B = write_tdb(Bn, str(OUT / f"B0_{Path(ADD).stem}_normalised.tdb"))
if any(n.action in A_CHANGES for n in notes):   # magnetic basis rule (A takes B's site ratios) or A's read exactly
    A = write_tdb(dbA, str(OUT / f"A0_{Path(BASE).stem}_rescaled.tdb"))
dbA, Bn = parse_tdb(A), parse_tdb(B)
shared = sorted(set(dbA.real_elements) & set(Bn.real_elements))
VA, VB = StageView(dbA, "A"), StageView(Bn, "B")
versions = {"B": B}
B1, log = substitute_ser(Bn, VA, VB, shared)
versions["B1"] = write_tdb(B1, str(OUT / f"B1_{Path(ADD).stem}+{Path(BASE).stem}-SER.tdb"), log)
B2, log = substitute_unaries(parse_tdb(versions["B1"]), VA, StageView(parse_tdb(versions["B1"]), "B1"), shared)
versions["B2"] = write_tdb(B2, str(OUT / f"B2_{Path(ADD).stem}+{Path(BASE).stem}-unaries.tdb"), log)
a_sub = write_tdb(restrict_tdb(dbA, shared, max_elements=2), str(OUT / "A_shared_unary_binary.tdb"))
plan, text = merged_text([a_sub, versions["B2"]], alias_separate([a_sub, versions["B2"]], DECISIONS), "first:predict",
                         tentative=TENTATIVE)
versions["B3"] = str(OUT / f"B3_{Path(ADD).stem}+{Path(BASE).stem}-unaries-binaries.tdb")
Path(versions["B3"]).write_text(text)
plan, text = merged_text([A, B], DECISIONS, POLICY, tentative=TENTATIVE)
for t in plan.tentative:
    print("Q tentative:", t.text())
if text is None:
    raise SystemExit("decisions still needed for Q:\n" + plan.report())
versions["Q"] = str(OUT / f"Q_{Path(BASE).stem}+{Path(ADD).stem}.tdb")
Path(versions["Q"]).write_text(text)
RENAMES = plan.renames_of(1)    # B's phases kept under a new name in Q (decision 'rename'): the same phase
if BUILD_A12:
    for lab, n_el in (("A1", 1), ("A2", 2)):
        part = write_tdb(restrict_tdb(Bn, Bn.real_elements, max_elements=n_el, constituents_from_params=True),
                         str(OUT / f"B_{n_el}_element_terms.tdb"))
        plan, text = merged_text([A, part], alias_separate([A, part], DECISIONS), "first:predict", tentative=TENTATIVE)
        versions[lab] = str(OUT / f"{lab}_{Path(BASE).stem}+{Path(ADD).stem}.tdb")
        Path(versions[lab]).write_text(text)
    versions["A"] = A_ORIG

for chain, els in ((["B", "B1", "B2", "B3", "Q"], sorted(Bn.real_elements)),
                   (["A", "A1", "A2"] if BUILD_A12 else [], sorted(dbA.real_elements))):
    subs = stage_subsystems(els, (2, 3))
    views = {}
    for lab in chain:
        tdb = parse_tdb(versions[lab])
        ren = stage_renames(tdb, RENAMES) if lab in ("B", "B1", "B2", "B3") else {}
        views[lab] = StageView(tdb, lab, elements=els, rename=ren)
    for old, new in zip(chain[:-1], chain[1:]):
        d = diff_versions(views[old], views[new], subs)
        print(f"\n{old} -> {new}: {len(d.params)} parameter change(s)")
        for s in d.subsystems:
            print(f"  {s.name}: {s.reasons(4)}")
        replot = [s for s in d.subsystems if s.changed or forced(s.elements)]
        for s in replot:
            if not s.changed:
                print(f"  forced replot of {s.name} (change exact / nil): compare {old}_ and {new}_ figures")
        changed = [(s.elements, None) for s in replot if len(s.elements) == 2] + \
                  [(s.elements, T) for s in replot if len(s.elements) == 3 for T in TERNARY_T_C]
        if changed:
            plot_versions([old, new], versions, changed, TMIN_C, TMAX_C, X_STEP, T_STEP)
print("\nwrote", ", ".join(str(p) for p in versions.values()))
'''


def _tent_default(ctx: Ctx):
    f = getattr(getattr(ctx, "tb", None), "tentative_default", None)
    return f() if callable(f) else True


@generator("staged_merge")
def _gen_staged(args, ctx: Ctx) -> str:
    if args.get("extend"):
        ctx.fallback("'extend' decisions (phase harmonisation) have no standalone generator")
    a, b = ctx.tdb(args["base_tdb"]), ctx.tdb(args["add_tdb"])
    key = f"staged|{ctx.tb._find_tdb(args['base_tdb'])}|{ctx.tb._find_tdb(args['add_tdb'])}"
    dec = {**(getattr(ctx.tb, "merge_decisions", {}) or {}).get(key, {}), **(args.get("decisions") or {})}
    from . import tdbstage as ts
    from .tdbparse import parse_tdb
    A, Bn, _ = ts.normalize_pair(parse_tdb(ctx.tb._find_tdb(args["base_tdb"])),
                                 parse_tdb(ctx.tb._find_tdb(args["add_tdb"])), args.get("phase_map"),
                                 args.get("sublattice_map"))
    VA, VB = ts.StageView(A, "A"), ts.StageView(Bn, "B")
    build_a12 = any(len(VB.row_elements(k)) <= 2 and k not in VA.rows for k in VB.rows)
    # the decisions the tool applied (auto: what the stages settle goes to A)
    ctx.need("import matplotlib", "import matplotlib.pyplot as plt", "from pycalphad import Database, variables as v",
             "from pycalphad.plot import triangular  # noqa: F401")
    temps = args.get("ternary_temperatures_c")
    from .codegen_calc import mapping_check_helpers
    body = _helpers(ctx) + mapping_check_helpers(ctx)
    body += (f"BASE, ADD = {a}, {b}\n"
             f"PHASE_MAP = {Ctx.py(args.get('phase_map') or {})}\n"
             f"SUBLATTICE_MAP = {Ctx.py(args.get('sublattice_map') or {})}\n"
             f"POLICY = {args.get('policy', 'ask')!r}\n"
             f"TENTATIVE = {Ctx.py(args['tentative'] if args.get('tentative') is not None else _tent_default(ctx))}"
             f"   # tentative L0 terms for new mixings (TENT_MIX / TENT_VA): True | False | values\n"
             f"FORCE_REPLOT = {Ctx.py(args.get('force_replot') or False)}   # replot exact-only subsystems too\n"
             f"BUILD_A12 = {build_a12!r}\n"
             f"TERNARY_T_C = {Ctx.py([1000.0] if temps is None else list(temps))}\n"
             f"TMIN_C, TMAX_C = {float(args.get('tmin_c', 300.0))!r}, {float(args.get('tmax_c', 3000.0))!r}\n"
             f"X_STEP, T_STEP = {float(args.get('composition_step', 0.02))!r}, "
             f"{float(args.get('temperature_step', 10.0))!r}\n"
             "_A, _B = str(BASE), str(ADD)\n"
             "_An, _Bn, _ = normalize_pair(parse_tdb(_A), parse_tdb(_B), PHASE_MAP, SUBLATTICE_MAP)\n"
             "if any(n.action in B_CHANGES for n in _):\n"
             "    _B = write_tdb(_Bn, str(OUT / '_normalised_for_decisions.tdb'))\n"
             "if any(n.action in A_CHANGES for n in _):\n"
             "    _A = write_tdb(_An, str(OUT / '_rescaled_for_decisions.tdb'))\n"
             "_shared = sorted(set(parse_tdb(_A).real_elements) & set(_Bn.real_elements))\n"
             "DECISIONS = alias_separate([_A, _B], " + Ctx.py(dec) + ")\n"
             "DECISIONS = {**auto_decisions(plan_merge([_A, _B], DECISIONS, 'ask'), _shared), **DECISIONS}\n"
             "matplotlib.use('Agg')\n")
    body += _STAGED_BODY
    return body
