"""Live end-to-end ESPEI test of the agent on the ESPEI Cu-Mg tutorial (examples/espei/cu-mg).

    .venv/bin/python -W ignore tests/espei_cumg_live.py --profile claude --scenario files
    .venv/bin/python -W ignore tests/espei_cumg_live.py --profile claude --scenario image
    .venv/bin/python -W ignore tests/espei_cumg_live.py --score <run dir> --scenario image   # re-score only

Scenarios (each in a fresh folder under calphad_outputs/espei_cumg/<scenario>-<time>/, the agent's workdir is
its ``work/``; the tutorial's own result TDBs stay out of its reach in tests/data/espei_cu_mg/):

* ``files``: the tutorial's phase models, datasets and YAMLs.  The agent checks the datasets, generates
  parameters, runs a short MCMC and plots the diagram with the data before and after.
  Scored: parameters identical to the tutorial's Cu-Mg-generated.tdb, the trace's shape, lnprob improving.
* ``image``: the same, but without the ZPF datasets - only ``cumg_zpf_points.png`` (ESPEI's dataplot of the
  tutorial's ZPF data, points only).  The agent digitizes it into ZPF datasets first.
  Scored against the tutorial's ZPF data: valid datasets, points recovered (phase, x within 0.015, T within
  15 K), spurious points, the phases each point is in equilibrium with, then the same ESPEI checks.

Costs one agent session (claude -p: the user's Claude login) plus a few minutes of local MCMC.
"""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EX = ROOT / "examples" / "espei" / "cu-mg"
REF = ROOT / "tests" / "data" / "espei_cu_mg"

ITER, CPP = 10, 2          # MCMC smoke test: 10 iterations, 2 chains per parameter

PROMPTS = {
    "files": (
        "This folder holds the ESPEI Cu-Mg tutorial inputs (phase_models.json, input-data/, run_param_gen.yaml, "
        "run_mcmc.yaml; see SOURCE.md). Run the example end to end with the ESPEI tools: check the datasets, "
        "generate the parameters with the tutorial's settings (SGTE91, linear excess model, the dft / "
        "estimated-entropy tags and the LIQUID AICc penalty factors of run_param_gen.yaml), plot the generated "
        f"phase diagram with the experimental data, then run a short MCMC as a smoke test ({ITER} iterations, "
        f"{CPP} chains per parameter, the tutorial's normal prior) and wait for it to finish. Then analyse the "
        "MCMC (did lnprob improve?), write the optimised TDB and plot its phase diagram with the data. "
        "Report how many parameters were generated and the lnprob before and after."),
    "image": (
        "I want to run the ESPEI Cu-Mg example (phase_models.json, run_param_gen.yaml, run_mcmc.yaml and the "
        "thermochemical datasets in input-data/ are here; see SOURCE.md), but I don't have the phase-boundary "
        "(ZPF) datasets - only the figure cumg_zpf_points.png, which plots them: marker colour is the phase "
        "whose composition was measured, marker shape the reference. Digitize the figure into ESPEI ZPF "
        "datasets in input-data/zpf/ (one per reference; for each point also give the phases it was in "
        "equilibrium with, with unknown compositions, as ESPEI expects), check them, then run the example: "
        "generate the parameters with the tutorial's settings, plot the generated diagram with your digitized "
        f"data, run a short MCMC smoke test ({ITER} iterations, {CPP} chains per parameter, normal prior) on "
        "all the data, wait for it, write the optimised TDB and plot it with the data. Report how many points "
        "you digitized per phase and reference."),
}


def setup(scenario: str) -> Path:
    run = ROOT / "calphad_outputs" / "espei_cumg" / f"{scenario}-{time.strftime('%Y%m%d-%H%M%S')}"
    work = run / "work"
    work.mkdir(parents=True)
    for f in ("phase_models.json", "run_param_gen.yaml", "run_mcmc.yaml", "SOURCE.md"):
        shutil.copy(EX / f, work / f)
    shutil.copytree(EX / "input-data", work / "input-data")
    if scenario == "image":
        shutil.rmtree(work / "input-data" / "zpf")
        (work / "input-data" / "zpf").mkdir()
        shutil.copy(REF / "cumg_zpf_points.png", work / "cumg_zpf_points.png")
    return run


def run_agent(run: Path, scenario: str, profile) -> None:
    from calphad_tools.agent.cli import run_terminal
    from calphad_tools.agent.config import load_config
    cfg = load_config(str(ROOT / "calphad_agent.toml"), profile)
    cfg.agent.workdir, cfg.agent.output_dir = str(run / "work"), "../out"
    cfg.agent.python_tool = "auto_verify"
    cfg.agent.max_tool_rounds = 120
    t = time.time()
    run_terminal(cfg, once=PROMPTS[scenario])
    print(f"\n[agent turn: {time.time() - t:.0f} s]")


# ------------------------------------------------------------------------------------------------- scoring
def _vv(tdb: Path):
    import re
    return {m.group(1): m.group(2).split() for m in re.finditer(r"FUNCTION\s+(VV\d+)\s+(.*?)!", tdb.read_text(),
                                                                re.S)}


def _zpf_points(datasets):
    """[(phase, x_MG, T, frozenset(all phases of the equilibrium), reference)] of every measured composition."""
    pts = []
    for d in datasets:
        Ts = d["conditions"]["T"]
        Ts = Ts if isinstance(Ts, list) else [Ts]
        for T, eq in zip(Ts, d["values"]):
            phases = frozenset(p[0] for p in eq)
            for ph, comps, x in eq:
                if x and x[0] is not None and comps and comps[0].upper() == "MG":
                    pts.append((ph, float(x[0]), float(T), phases, d.get("reference", "")))
                elif x and x[0] is not None and comps:            # given as X(CU)
                    pts.append((ph, 1.0 - float(x[0]), float(T), phases, d.get("reference", "")))
    return pts


def score_zpf(work: Path) -> dict:
    from espei.datasets import load_datasets, recursive_glob
    files = recursive_glob(str(work / "input-data" / "zpf"))
    out = {"zpf_files": len(files)}
    try:
        mine = [d for d in load_datasets(files).all() if d["output"] == "ZPF"]
    except Exception as e:  # noqa: BLE001
        out["load_error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    ref = load_datasets(recursive_glob(str(EX / "input-data" / "zpf"))).all()
    P, Q = _zpf_points(ref), _zpf_points(mine)
    out.update(ref_points=len(P), digitized_points=len(Q))
    dx, dT = 0.015, 15.0

    def near(p, pool):
        c = [q for q in pool if q[0] == p[0] and abs(q[1] - p[1]) <= dx and abs(q[2] - p[2]) <= dT]
        return min(c, key=lambda q: ((q[1] - p[1]) / dx) ** 2 + ((q[2] - p[2]) / dT) ** 2) if c else None
    hits = [(p, near(p, Q)) for p in P]
    found = [(p, q) for p, q in hits if q]
    out["recall"] = round(len(found) / max(len(P), 1), 3)
    out["precision"] = round(sum(1 for q in Q if near(q, P)) / max(len(Q), 1), 3)
    out["partners_correct"] = round(sum(1 for p, q in found if p[3] == q[3]) / max(len(found), 1), 3)
    if found:
        out["median_abs_dx"] = round(sorted(abs(p[1] - q[1]) for p, q in found)[len(found) // 2], 4)
        out["median_abs_dT"] = round(sorted(abs(p[2] - q[2]) for p, q in found)[len(found) // 2], 1)
    per = {}
    for p, q in hits:
        k = p[0]
        per.setdefault(k, [0, 0])
        per[k][0] += 1
        per[k][1] += bool(q)
    out["recall_by_phase"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(per.items())}
    wrong = {}
    for p, q in found:
        if p[3] != q[3]:
            key = f"{'+'.join(sorted(p[3]))} given as {'+'.join(sorted(q[3]))}"
            wrong[key] = wrong.get(key, 0) + 1
    out["partner_errors"] = dict(sorted(wrong.items(), key=lambda kv: -kv[1])[:8])
    return out


def score_espei(run: Path) -> dict:
    import numpy as np
    out = {}
    tdbs = [p for p in run.rglob("*.tdb") if "work" not in p.parts or p.parent.name != "work"]
    gen = [p for p in tdbs if p.name in ("dft.tdb", "Cu-Mg-generated.tdb")]
    out["generated_tdb"] = str(gen[0].relative_to(run)) if gen else None
    if gen:
        a, b = _vv(gen[0]), _vv(REF / "Cu-Mg-generated.tdb")
        out["n_parameters"] = len(a)
        out["parameters_match_tutorial"] = a == b
        if a != b:
            out["parameter_diffs"] = sorted(set(a) ^ set(b) | {k for k in a.keys() & b.keys() if a[k] != b[k]})[:10]
    traces = sorted(run.rglob("trace.npy"))
    if traces:
        tr = np.load(traces[-1])
        lp = np.load(traces[-1].with_name("lnprob.npy"))
        done = int((tr != 0).any(axis=(0, 2)).sum())
        out["trace_shape"] = list(tr.shape)
        out["iterations_done"] = done
        if done:
            out["lnprob_best_first"] = round(float(lp[:, 0].max()), 1)
            out["lnprob_best_last"] = round(float(lp[:, done - 1].max()), 1)
    out["mcmc_tdb"] = [str(p.relative_to(run)) for p in tdbs if "mcmc" in p.name.lower()][:3]
    out["figures"] = len(list((run / "out").rglob("*.png")))
    return out


def main():
    a = sys.argv
    scenario = a[a.index("--scenario") + 1] if "--scenario" in a else "files"
    if "--score" in a:
        run = Path(a[a.index("--score") + 1]).resolve()
    else:
        profile = a[a.index("--profile") + 1] if "--profile" in a else None
        run = setup(scenario)
        print(f"run folder: {run}")
        run_agent(run, scenario, profile)
    res = {"scenario": scenario, "run": str(run)}
    if scenario == "image":
        res["zpf"] = score_zpf(run / "work")
    res["espei"] = score_espei(run)
    (run / "score.json").write_text(json.dumps(res, indent=2))
    print("\n=== score ===\n" + json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
