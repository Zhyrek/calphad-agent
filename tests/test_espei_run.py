"""Background jobs (calphad_tools.jobs) and local ESPEI runs (calphad_tools.espei_run), end to end.

Cr-Ni with LIQUID/FCC_A1/BCC_A2: synthetic HM_MIX/SM_MIX single-phase data (Hallstedt 2006 L0, L1) for
parameter generation plus two small ZPF datasets, then a very short MCMC through the job system.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
R = 8.31451
_L = {"LIQUID": [(-14190.0, 2.3), (-950.0, 4.7)],
      "FCC_A1": [(5320.0, -10.5), (32030.0, -14.4)],
      "BCC_A2": [(22780.0, -14.6), (15650.0, -1.5)]}
_SUBL = {"LIQUID": ([["CR", "NI"]], [1.0]), "FCC_A1": ([["CR", "NI"], ["VA"]], [1.0, 1.0]),
         "BCC_A2": ([["CR", "NI"], ["VA"]], [1.0, 3.0])}
_ZPF = [
    {"components": ["CR", "NI", "VA"], "phases": ["FCC_A1", "LIQUID"],
     "conditions": {"P": 101325.0, "T": [1650.0]}, "broadcast_conditions": False, "output": "ZPF",
     "values": [[["LIQUID", ["CR"], [0.35]], ["FCC_A1", ["CR"], [0.33]]]], "reference": "test", "comment": ""},
    {"components": ["CR", "NI", "VA"], "phases": ["BCC_A2", "LIQUID"],
     "conditions": {"P": 101325.0, "T": [1700.0, 1800.0]}, "broadcast_conditions": False, "output": "ZPF",
     "values": [[["LIQUID", ["CR"], [0.6]], ["BCC_A2", ["CR"], [None]]],
                [["LIQUID", ["CR"], [0.7]], ["BCC_A2", ["CR"], [None]]]], "reference": "test", "comment": ""}]


def write_crni_inputs(d: Path):
    ds = d / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    xs = np.array([0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    for ph, Ls in _L.items():
        sl, ratios = _SUBL[ph]
        xcr, xni = xs, 1 - xs
        dd = xcr - xni
        H = xcr * xni * sum(a * dd ** k for k, (a, b) in enumerate(Ls))
        S = -xcr * xni * sum(b * dd ** k for k, (a, b) in enumerate(Ls)) - R * (xcr * np.log(xcr) + xni * np.log(xni))
        cfg = [[["CR", "NI"]] + (["VA"] if len(sl) > 1 else []) for _ in xs]
        occ = [[[float(a), float(1 - a)]] + ([1.0] if len(sl) > 1 else []) for a in xs]
        for out, vals in (("HM_MIX", H), ("SM_MIX", S)):
            (ds / f"{ph}_{out}.json").write_text(json.dumps({
                "components": ["CR", "NI"] + (["VA"] if len(sl) > 1 else []), "phases": [ph],
                "solver": {"mode": "manual", "sublattice_site_ratios": ratios, "sublattice_configurations": cfg,
                           "sublattice_occupancies": occ},
                "conditions": {"P": 101325.0, "T": 1600.0}, "output": out, "values": [[[float(v) for v in vals]]],
                "excluded_model_contributions": ["mag"], "reference": "synthetic", "comment": "test data"}))
    for i, z in enumerate(_ZPF):
        (ds / f"zpf_{i}.json").write_text(json.dumps(z))
    pm = {"components": ["CR", "NI", "VA"],
          "phases": {p: {"sublattice_model": _SUBL[p][0], "sublattice_site_ratios": _SUBL[p][1]} for p in _L}}
    (d / "phase_models.json").write_text(json.dumps(pm, indent=1))


def _make_tb(workdir, out="out"):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(workdir), output_dir=out, ui=U(), python_tool="off")


@pytest.fixture
def tb(tmp_path):
    write_crni_inputs(tmp_path / "CrNi")
    return _make_tb(tmp_path)


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def _job_id(text):
    return re.search(r"job (\d{4}-\d{6}-[0-9a-f]{4})", text).group(1)


# ---------------------------------------------------------------------------
# generic jobs
# ---------------------------------------------------------------------------
def test_job_progress_status_result(tb):
    from calphad_tools import jobs
    t0 = time.time()
    jid = jobs.submit(tb, "demo", "calphad_tools.jobs:demo",
                      {"steps": 4, "delay": 0.05, "plot": True, "out_dir": str(tb.out / "demo")})
    assert time.time() - t0 < 2                               # detached: submit returns at once
    d = tb.out / "jobs" / jid
    assert (d / "spec.json").exists() and (d / "status.json").exists()
    text, imgs = call(tb, "wait_job", {"job_id": jid, "timeout_s": 60})
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "done" and st["frac"] == 1.0 and st["metrics"]["step"] == 4
    assert st["start"] and st["end"] and st["pid"]
    recs = [json.loads(x) for x in (d / "progress.jsonl").read_text().splitlines()]
    assert [r["metrics"]["step"] for r in recs] == [1, 2, 3, 4]
    res = json.loads((d / "result.json").read_text())
    assert res["text"] == "demo finished 4 steps" and res["steps"] == 4
    assert "state: done" in text and "demo finished 4 steps" in text and "demo_plot.png" in text
    assert imgs and imgs[0].endswith("demo_plot.png")
    # the figure saved inside the job is editable (savefig hook installed in the child)
    from calphad_tools.figures import load_figure
    assert load_figure(imgs[0]) is not None
    assert "log.txt" in text and (d / "log.txt").read_text().count("progress: step") == 4
    lj, _ = call(tb, "list_jobs", {})
    assert jid in lj and "done" in lj


def test_job_failure_reports_traceback(tb):
    from calphad_tools import jobs
    jid = jobs.submit(tb, "boom", "calphad_tools.jobs:demo", {"steps": 4, "delay": 0.01, "fail": True})
    text, _ = call(tb, "wait_job", {"job_id": jid, "timeout_s": 60})
    assert "state: failed" in text and "RuntimeError: demo failure at step 3" in text
    assert "Traceback (most recent call last)" in text and "in demo" in text
    st = jobs.status(tb, jid)
    assert st["state"] == "failed" and "demo failure" in st["error"]


def test_job_bad_target_fails(tb):
    from calphad_tools import jobs
    jid = jobs.submit(tb, "missing", "calphad_tools.jobs:no_such_function", {})
    st = jobs.wait(tb, jid, 60)
    assert st["state"] == "failed" and "AttributeError" in st["message"]


def test_stop_long_job(tb):
    from calphad_tools import jobs
    jid = jobs.submit(tb, "long", "calphad_tools.jobs:demo", {"steps": 1000, "delay": 0.2})
    for _ in range(100):
        if jobs.status(tb, jid).get("n_progress", 0) >= 2:
            break
        time.sleep(0.1)
    pid = jobs.status(tb, jid)["pid"]
    t0 = time.time()
    text, _ = call(tb, "stop_job", {"job_id": jid})
    assert time.time() - t0 < 12
    assert "state: stopped" in text
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    # a stopped job stays stopped
    assert jobs.status(tb, jid)["state"] == "stopped"
    assert "already finished" in call(tb, "stop_job", {"job_id": jid})[0]


def test_killed_job_is_detected(tb):
    import signal
    from calphad_tools import jobs
    jid = jobs.submit(tb, "victim", "calphad_tools.jobs:demo", {"steps": 1000, "delay": 0.2})
    for _ in range(100):
        if jobs.status(tb, jid).get("pid") and jobs.status(tb, jid).get("state") == "running":
            break
        time.sleep(0.1)
    os.kill(jobs.status(tb, jid)["pid"], signal.SIGKILL)
    time.sleep(0.5)
    st = jobs.status(tb, jid)
    assert st["state"] == "failed" and "without reporting" in st["message"]


def test_job_survives_parent_exit(tmp_path):
    """A job submitted by a short-lived agent process keeps running after that process has exited."""
    code = ("import sys; from calphad_tools.agent.tools import ToolBox, UI; from calphad_tools import jobs\n"
            "tb = ToolBox(workdir=sys.argv[1], output_dir='out', ui=UI(), python_tool='off')\n"
            "print(jobs.submit(tb, 'orphan', 'calphad_tools.jobs:demo', {'steps': 6, 'delay': 0.3}))\n")
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    jid = r.stdout.strip().splitlines()[-1]
    tb = _make_tb(tmp_path)
    from calphad_tools import jobs
    assert jobs.status(tb, jid)["state"] in ("queued", "running")
    st = jobs.wait(tb, jid, 60)
    assert st["state"] == "done", st


# ---------------------------------------------------------------------------
# ESPEI
# ---------------------------------------------------------------------------
def test_validate_input(tb, tmp_path):
    text, _ = call(tb, "espei_validate_input", {"phase_models": "CrNi/phase_models.json", "datasets": "CrNi/datasets"})
    assert text.startswith("inputs OK") and "fixed-config HM_MIX: 3" in text and "ZPF: 2" in text
    # the ESPEI 0.9.1 T x X ordering bug is flagged for equilibrium-property datasets
    bad = {"components": ["CR", "NI"], "phases": ["LIQUID"], "output": "HM",
           "conditions": {"P": 101325, "T": [1800, 1900], "X_CR": [0.2, 0.5]}, "values": [[[0, 1], [2, 3]]],
           "reference": "x"}
    (tmp_path / "CrNi/datasets/eq_hm.json").write_text(json.dumps(bad))
    text, _ = call(tb, "espei_validate_input", {"phase_models": "CrNi/phase_models.json", "datasets": "CrNi/datasets"})
    assert "EquilibriumPropertyResidual" in text and "eq_hm.json" in text
    # YAML checked against ESPEI's schema
    (tmp_path / "bad.yaml").write_text("system: {phase_models: CrNi/phase_models.json, datasets: CrNi/datasets}\n"
                                       "mcmc: {iterations: 5, chains_per_parameter: 3}\n")
    text, _ = call(tb, "espei_validate_input", {"yaml": "bad.yaml"})
    assert "PROBLEMS" in text and "schema" in text


def _vv_values(tdb):
    from espei.utils import database_symbols_to_fit, unpack_piecewise
    from pycalphad import Database
    db = Database(str(tdb))
    return {s: float(unpack_piecewise(db.symbols[s])) for s in database_symbols_to_fit(db)}


def test_generate_parameters_and_show_code(tb, tmp_path):
    t0 = time.time()
    text, _ = call(tb, "espei_generate_parameters", {"name": "CrNi", "phase_models": "CrNi/phase_models.json",
                                                     "datasets": "CrNi/datasets", "ref_state": "SGTE91"})
    secs = time.time() - t0
    print(f"\nparameter generation (sync): {secs:.1f} s")
    d = tb.out / "espei_runs" / "CrNi"
    tdb = d / "dft.tdb"
    assert tdb.exists() and (d / "espei_generate.yaml").exists() and (d / "generate_summary.json").exists()
    from pycalphad import Database, calculate
    db = Database(str(tdb))
    assert {"LIQUID", "FCC_A1", "BCC_A2"} <= set(db.phases)
    summ = json.loads((d / "generate_summary.json").read_text())
    assert set(summ["phases"]) == {"LIQUID", "FCC_A1", "BCC_A2"}
    for ph, rows in summ["phases"].items():
        assert any(r["parameter"].startswith(f"L({ph},CR,NI") for r in rows)
    # the fitted L0 of the liquid reproduces the synthetic data (-14190 + 2.3 T)
    liq0 = next(r for r in summ["phases"]["LIQUID"] if r["parameter"].endswith(";0)"))
    a, b = re.fullmatch(r"(-?[\d.]+) \+ ([\d.]+)\*T", liq0["value"]).groups()
    assert float(a) == pytest.approx(-14190, abs=1) and float(b) == pytest.approx(2.3, abs=1e-3)
    assert "L(LIQUID,CR,NI;0)" in text and "fitted symbols" in text and "generate_espei_flux_batch" in text
    calculate(db, ["CR", "NI", "VA"], "LIQUID", T=1800, P=101325, N=1)       # loads + evaluates in pycalphad
    # same YAML structure as the Flux batch generator
    import yaml
    from calphad_tools import espei_flux
    y = yaml.safe_load((d / "espei_generate.yaml").read_text())
    run = espei_flux.Run(name="CrNi", phase_models=str(tmp_path / "CrNi/phase_models.json"),
                         datasets=str(tmp_path / "CrNi/datasets"))
    assert y == espei_flux.espei_yaml(run, None, None, None, "generate", tmp_path)
    # show_code: standalone ESPEI code reproduces the same parameters
    text, _, err = tb.call("show_code", {"target": str(tdb)})
    assert not err and "1 standalone" in text, text
    code = text.split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert "calphad_tools" not in code and "run_espei" in code and "generate_parameters:" in code
    gd = tmp_path / "gen"
    gd.mkdir()
    (gd / "gen.py").write_text(code)
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=gd, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    new = _vv_values(gd / "standalone_outputs" / "CrNi" / "dft.tdb")
    old = _vv_values(tdb)
    assert new.keys() == old.keys() and all(new[k] == pytest.approx(old[k], rel=1e-9, abs=1e-12) for k in old)


def test_espei_input_yaml_and_tags(tmp_path):
    """The ESPEI Cu-Mg tutorial through espei_input=run_param_gen.yaml: its tags reach ESPEI and the 22
    parameters equal the tutorial's Cu-Mg-generated.tdb; MCMC settings inherit the tags and take run_mcmc.yaml's."""
    import shutil
    import yaml
    from calphad_tools import espei_run
    shutil.copytree(ROOT / "examples" / "espei" / "cu-mg", tmp_path / "cu-mg")
    tb = _make_tb(tmp_path)
    text, _ = call(tb, "espei_generate_parameters", {"name": "cumg", "espei_input": "cu-mg/run_param_gen.yaml"})
    d = tb.out / "espei_runs" / "cumg"
    y = yaml.safe_load((d / "espei_generate.yaml").read_text())
    assert y["system"]["tags"]["dft"]["excluded_model_contributions"] == ["idmix", "mag"]
    assert y["generate_parameters"]["aicc_penalty_factor"] == {"LIQUID": {"HM": 1.4, "SM": 1.4}}
    ref = _vv_values(ROOT / "tests" / "data" / "espei_cu_mg" / "Cu-Mg-generated.tdb")
    new = _vv_values(d / "dft.tdb")
    assert len(ref) == 22 and new.keys() == ref.keys()
    assert all(new[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12) for k in ref)
    _, settings, run, _ = espei_run.mcmc_settings(tb, {"name": "cumg"})          # tags kept from the run dir
    assert settings["system"]["tags"] == y["system"]["tags"]
    _, settings, run, _ = espei_run.mcmc_settings(tb, {"name": "cumg", "espei_input": "cu-mg/run_mcmc.yaml",
                                                       "iterations": 10})
    m = settings["mcmc"]
    assert m["iterations"] == 10 and m["chains_per_parameter"] == 4 and m["prior"]["name"] == "normal"
    assert settings["system"]["tags"]["estimated-entropy"]["excluded_model_contributions"] == ["idmix", "mag"]


def test_generate_as_background_job(tb):
    text, _ = call(tb, "espei_generate_parameters", {"name": "bg", "phase_models": "CrNi/phase_models.json",
                                                     "datasets": "CrNi/datasets", "background": True})
    jid = _job_id(text)
    text, _ = call(tb, "wait_job", {"job_id": jid, "timeout_s": 300})
    assert "state: done" in text and "L(BCC_A2,CR,NI:VA;0)" in text
    assert (tb.out / "espei_runs/bg/dft.tdb").exists()


def test_mcmc_job_end_to_end(tb, tmp_path):
    call(tb, "espei_generate_parameters", {"name": "CrNi", "phase_models": "CrNi/phase_models.json",
                                           "datasets": "CrNi/datasets"})
    t0 = time.time()
    # cores=1: serial, so the YAML matches the serial Flux-generator structure exactly (the default is half the
    # cores as a dask cluster; test_hpc covers that)
    text, _ = call(tb, "espei_run_mcmc", {"name": "CrNi", "iterations": 3, "chains_per_parameter": 2,
                                          "save_interval": 1, "cores": 1})
    jid = _job_id(text)
    d = tb.out / "espei_runs" / "CrNi"
    assert "30 chains (2 x 15 parameters)" in text and "analyze_mcmc trace=" in text
    wt, imgs = call(tb, "wait_job", {"job_id": jid, "timeout_s": 300})
    secs = time.time() - t0
    print(f"\nESPEI MCMC job, 3 iterations x 30 chains: {secs:.1f} s")
    assert "state: done" in wt, wt
    assert "3/3 iterations" in wt and "lnprob_progress.png" in wt
    assert any(i.endswith("lnprob_progress.png") for i in imgs)
    for f in ("trace.npy", "lnprob.npy", "mcmc.tdb", "espei_mcmc.yaml", "mcmc.log"):
        assert (d / f).exists(), f
    assert np.load(d / "trace.npy").shape == (30, 3, 15)
    recs = [json.loads(x) for x in (tb.out / "jobs" / jid / "progress.jsonl").read_text().splitlines()]
    its = [r["metrics"]["iteration"] for r in recs if (r.get("metrics") or {}).get("iteration")]
    assert its == [1, 2, 3]
    last = [r for r in recs if (r.get("metrics") or {}).get("iteration") == 3][0]["metrics"]
    assert last["max_lnprob"] >= last["mean_lnprob"]
    # YAML identical in structure to the Flux one (only scheduler differs)
    import yaml
    from calphad_tools import espei_flux
    y = yaml.safe_load((d / "espei_mcmc.yaml").read_text())
    run = espei_flux.Run(name="CrNi", phase_models=str(tmp_path / "CrNi/phase_models.json"),
                         datasets=str(tmp_path / "CrNi/datasets"), iterations=3, save_interval=1)
    assert y == espei_flux.espei_yaml(run, "dft.tdb", None, None, "mcmc", tmp_path)
    # analyze_mcmc works on the outputs as they are
    at, aimgs = call(tb, "analyze_mcmc", {"trace": str(d.relative_to(tmp_path)), "burn_in": "0"})
    assert "VV0000" in at and aimgs
    # job_status (status hook) for a finished ESPEI job
    st, simgs = call(tb, "job_status", {"job_id": jid})
    assert "ESPEI: 3 iterations sampled" in st and simgs
    # show_code maps the job's trace back to the submitting espei_run_mcmc call
    from calphad_tools.codegen import find_producer
    rec = find_producer(tb, str(d / "trace.npy"))
    assert rec["tool"] == "espei_run_mcmc", rec
    code_text, _, err = tb.call("show_code", {"target": str(d / "trace.npy"), "include_parents": False})
    code = code_text.split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert not err and "calphad_tools" not in code and "mcmc:" in code and "run_espei" in code
    gd = tmp_path / "gen_mcmc"
    gd.mkdir()
    (gd / "gen.py").write_text(code)
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=gd, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    # deterministic serial ESPEI: the standalone run reproduces the job's chains
    assert np.allclose(np.load(gd / "standalone_outputs/CrNi/lnprob.npy"), np.load(d / "lnprob.npy"))
    # a second run with the same name keeps the first trace (new run dir)
    text2, _ = call(tb, "espei_run_mcmc", {"name": "CrNi", "iterations": 2, "restart_trace": str(d)})
    assert "CrNi_2" in text2 and "restart from" in text2
    wt2, _ = call(tb, "wait_job", {"job_id": _job_id(text2), "timeout_s": 300})
    assert "state: done" in wt2 and np.load(tb.out / "espei_runs/CrNi_2/trace.npy").shape == (30, 2, 15)


def test_stop_mcmc_mid_run(tb):
    call(tb, "espei_generate_parameters", {"name": "stopme", "phase_models": "CrNi/phase_models.json",
                                           "datasets": "CrNi/datasets"})
    text, _ = call(tb, "espei_run_mcmc", {"name": "stopme", "iterations": 500, "save_interval": 1})
    jid = _job_id(text)
    from calphad_tools import jobs
    t0 = time.time()
    while time.time() - t0 < 180:
        st = jobs.status(tb, jid)
        if (st.get("metrics") or {}).get("iteration", 0) >= 2 or st.get("state") in jobs.FINAL:
            break
        time.sleep(0.3)
    assert st.get("state") == "running", st
    live, limgs = call(tb, "job_status", {"job_id": jid})
    assert "ESPEI:" in live and "ETA" in live and limgs
    t1 = time.time()
    text, _ = call(tb, "stop_job", {"job_id": jid})
    print(f"\nstop_job on a running MCMC: {time.time() - t1:.1f} s")
    assert "state: stopped" in text
    d = tb.out / "espei_runs" / "stopme"
    tr = np.load(d / "trace.npy")
    assert tr.shape[1] == 500 and np.any(tr[:, 1, :] != 0)        # chains so far were saved
    res = jobs.result(tb, jid)
    assert res.get("interrupted") is True or "error" in res
    # partial chains still analyse
    at, _ = call(tb, "analyze_mcmc", {"trace": str(d), "burn_in": "0"})
    assert "VV0000" in at
