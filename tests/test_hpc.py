"""HPC backend for ESPEI optimisations: Flux detection, run-size defaults, batch submission, Flux-backed jobs.

A fake ``flux`` (tests/fake_flux.py) is put first on PATH: it answers the probes (resource info/list, queue list,
batch --help), records every flux batch (argv + #flux: directives), runs the jobs (real ESPEI parameter generation
for gen.flux, an MCMC stub writing mcmc.log / trace.npy / lnprob.npy / mcmc.tdb for mcmc.flux) through
DEPEND/SCHED -> RUN -> COMPLETED, and implements flux jobs / flux cancel.
"""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from test_espei_run import write_crni_inputs  # noqa: E402

from calphad_tools import hpc, jobs  # noqa: E402
from calphad_tools.hpc import HPCConfig  # noqa: E402

FAKE = Path(__file__).parent / "fake_flux.py"


def _shim(bindir: Path) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    f = bindir / "flux"
    f.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    f.chmod(0o755)


@pytest.fixture
def fake_flux(tmp_path, monkeypatch):
    """Fake flux first on PATH; returns its state dir.  set_mode(...) switches the fake's behaviour."""
    _shim(tmp_path / "fakebin")
    state = tmp_path / "flux_state"
    state.mkdir()
    monkeypatch.setenv("PATH", f"{tmp_path / 'fakebin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_FLUX_DIR", str(state))
    monkeypatch.setenv("FAKE_FLUX_MODE", "ok")
    monkeypatch.delenv("FLUX_URI", raising=False)
    hpc.clear_cache()
    yield state
    hpc.clear_cache()


@pytest.fixture
def no_flux(tmp_path, monkeypatch):
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("FLUX_URI", raising=False)
    hpc.clear_cache()
    yield
    hpc.clear_cache()


class RecUI:
    """Records progress lines (with their replace-keys) and answers ask() from a list."""

    def __init__(self, answers=None):
        self.lines, self.asked, self.answers = [], [], list(answers or [])

    def ask(self, q, options=None):
        self.asked.append((q, options))
        return self.answers.pop(0) if self.answers else ""

    def confirm(self, title, body):
        return False

    def progress(self, msg, key=None):
        self.lines.append((key, msg))

    def notice(self, kind, text):
        self.progress(text)


def _tb(workdir, hpc_cfg=None, ui=None, interactive=False):
    from calphad_tools.agent.tools import UI, ToolBox

    class U(RecUI, UI):
        pass
    u = ui or U()
    return ToolBox(workdir=str(workdir), output_dir="out", ui=u, python_tool="off", interactive=interactive,
                   hpc=hpc_cfg)


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def _jid(text):
    return re.search(r"job (\d{4}-\d{6}-[0-9a-f]{4})", text).group(1)


def _submissions(state):
    p = state / "submissions.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _calls(state):
    p = state / "calls.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


OPT = {"name": "CrNi", "phase_models": "CrNi/phase_models.json", "datasets": "CrNi/datasets", "iterations": 6,
       "save_interval": 1}


# ---------------------------------------------------------------------------------------------- detection
def test_detect_no_flux(no_flux):
    info = hpc.detect(refresh=True)
    assert info["scheduler"] == "local" and info["flux"] is None
    assert "no flux executable" in info["reason"]
    assert info["local_cores"] >= 1
    assert hpc.resolve_backend(HPCConfig())[0] == "local"
    with pytest.raises(RuntimeError, match="not usable"):
        hpc.resolve_backend(HPCConfig(), "flux")


def test_detect_flux_found(fake_flux, monkeypatch):
    info = hpc.detect(refresh=True)
    assert info["scheduler"] == "flux" and info["instance"] == "system", info
    assert "flux uptime" in info["reason"]
    assert (info["nodes"], info["cores_per_node"], info["gpus_per_node"]) == (4, 112, 0)
    assert info["resources_from"] == "flux resource info"
    assert info["queues"] == ["pbatch", "pdebug"] and info["default_queue"] == "pbatch"
    assert info["bank_flag"] == "--bank" and info["version"] == "0.68.0"
    # probes only: nothing was submitted
    assert not _submissions(fake_flux)
    assert all(c[0] != "batch" or c[1:] == ["--help"] for c in _calls(fake_flux))
    # cached per process: a second call does not run flux again
    n = len(_calls(fake_flux))
    assert hpc.detect() == info and len(_calls(fake_flux)) == n
    # inside an allocation (FLUX_URI): the instance is asked directly
    monkeypatch.setenv("FLUX_URI", "local:///tmp/flux-xyz/local-0")
    info2 = hpc.detect()
    assert info2["instance"] == "job" and info2["instance_level"] == 1 and "FLUX_URI" in info2["reason"]
    assert info["instance_level"] == 0
    assert ["getattr", "size"] in _calls(fake_flux)
    text = hpc.summary(info2, HPCConfig(bank="b1", queue="pdebug"))
    assert "HPC: Flux" in text and "1 node(s) x 112 cores" in text and "bank b1, queue pdebug" in text
    assert hpc.resolve_backend(HPCConfig())[0] == "flux"
    assert hpc.resolve_backend(HPCConfig(scheduler="local"))[0] == "local"     # config wins over detection
    assert hpc.resolve_backend(HPCConfig(scheduler="local"), "flux")[0] == "flux"   # tool argument wins


def test_detect_flux_not_answering(fake_flux, monkeypatch):
    monkeypatch.setenv("FAKE_FLUX_MODE", "noanswer")
    info = hpc.detect(refresh=True)
    assert info["scheduler"] == "local" and info["flux"]
    assert "no instance answers" in info["reason"] and "Unable to connect" in info["reason"]
    monkeypatch.setenv("FAKE_FLUX_MODE", "hang")
    t0 = time.time()
    info = hpc.detect(refresh=True, timeout=1.0)
    assert time.time() - t0 < 10
    assert info["scheduler"] == "local" and "no answer within 1 s" in info["reason"]


def test_resource_and_queue_parsers():
    assert hpc.parse_resource_info("16 Nodes, 1792 Cores, 64 GPUs") == {"nodes": 16, "cores": 1792, "gpus": 64}
    assert hpc.parse_resource_info("1 Node, 36 Cores, 0 GPUs")["cores"] == 36
    table = ("     STATE QUEUE    NNODES   NCORES    NGPUS NODELIST\n      free pbatch       3      336        0 n[2-4]\n"
             " allocated pbatch       1      112        0 n1\n      down pbatch       2      224        0 n[5-6]\n")
    r = hpc.parse_resource_list(table)
    assert (r["nodes"], r["cores"], r["cores_per_node_max"]) == (4, 448, 112)          # down nodes not counted
    r = hpc.parse_resource_list("free 2 96 4\nallocated 1 48 2\n")                   # -o "{state} {nnodes} ..."
    assert (r["nodes"], r["cores"], r["gpus"], r["cores_per_node_max"]) == (3, 144, 6, 48)
    assert hpc.parse_resource_list("garbage\n") is None
    old = "QUEUE    TDEFAULT TLIMIT NNLIMIT\npbatch*  1h       1d     inf\npdebug   30m      1h     2\n"
    assert hpc.parse_queue_list(old) == (["pbatch", "pdebug"], "pbatch")
    assert hpc.parse_queue_list("") == ([], None)


# ---------------------------------------------------------------------------------------------- defaults
def test_core_defaults():
    assert hpc.local_cores("half", 16) == 8 and hpc.local_cores(None, 16) == 8
    assert hpc.local_cores("half", 1) == 1 and hpc.local_cores("half", 3) == 1
    assert hpc.local_cores("all", 16) == 16 and hpc.local_cores(4, 16) == 4 and hpc.local_cores("64", 16) == 16
    with pytest.raises(ValueError, match="local_cores"):
        hpc.local_cores("lots", 16)
    assert hpc.available_cores() >= 1
    info = {"cores_per_node": 112, "resources_from": "flux resource info"}
    assert hpc.batch_resources(HPCConfig(), info) == (1, 112, "flux resource info")          # 1 node, all cores
    assert hpc.batch_resources(HPCConfig(nodes=2, cores_per_node=56), info) == (2, 56, "[hpc] cores_per_node")
    assert hpc.batch_resources(HPCConfig(), info, nodes=3, cores_per_node=8) == (3, 8, "tool argument")
    with pytest.raises(RuntimeError, match="cores_per_node"):
        hpc.batch_resources(HPCConfig(), {})
    # local MCMC default: half of the available cores, at most one per chain
    from calphad_tools.espei_run import default_cores

    class T:
        hpc = HPCConfig(local_cores="half")
    n = hpc.available_cores()
    assert default_cores(T())[0] == max(1, n // 2)
    assert default_cores(T(), n_chains=1)[0] == 1
    T.hpc = HPCConfig(local_cores=3)
    assert default_cores(T())[0] == min(3, n)


def test_batch_options():
    cfg = HPCConfig(bank="mybank", queue="pbatch", extra_flux_args=["--setattr=system.constraints=x"])
    assert hpc.batch_options(cfg) == ["-q", "pbatch", "--bank=mybank", "--setattr=system.constraints=x"]
    assert hpc.batch_options(cfg, "--setattr=system.bank") [:3] == ["-q", "pbatch", "--setattr=system.bank=mybank"]
    assert hpc.batch_options(HPCConfig(), queue="pdebug", bank="") == ["-q", "pdebug"]
    # inside an allocation (nested instance): no queues / banks there
    assert hpc.batch_options(cfg, nested=True) == ["--setattr=system.constraints=x"]
    hint = hpc.submit_hint("flux-batch: ERROR: bank is required")
    assert "[hpc] bank" in hint
    assert "[hpc] queue" in hpc.submit_hint("invalid queue 'x'")


# ---------------------------------------------------------------------------------------------- config / wizard
def test_config_hpc_table(tmp_path, monkeypatch):
    from calphad_tools.agent.config import EXAMPLE_CONFIG, load_config
    monkeypatch.delenv("CALPHAD_HPC_SCHEDULER", raising=False)
    p = tmp_path / "c.toml"
    p.write_text(EXAMPLE_CONFIG)
    h = load_config(str(p)).hpc
    assert (h.scheduler, h.bank, h.queue, h.nodes, h.cores_per_node, h.time_limit, h.local_cores) == \
        ("auto", "", "", 1, 0, "4h", "half")
    assert h.extra_flux_args == [] and h.fallback_to_local is False and h.wait_max_hours == 24
    p.write_text('[hpc]\nscheduler = "flux"\nbank = "b"\nqueue = "q"\nnodes = 2\nlocal_cores = 6\n'
                 'extra_flux_args = ["--exclusive"]\n')
    h = load_config(str(p)).hpc
    assert (h.scheduler, h.bank, h.queue, h.nodes, h.local_cores, h.extra_flux_args) == \
        ("flux", "b", "q", 2, 6, ["--exclusive"])
    monkeypatch.setenv("CALPHAD_HPC_SCHEDULER", "local")
    assert load_config(str(p)).hpc.scheduler == "local"
    monkeypatch.setenv("CALPHAD_HPC_SCHEDULER", "slurm")
    with pytest.raises(ValueError):
        load_config(str(p))
    # the shipped example file has the same table
    monkeypatch.delenv("CALPHAD_HPC_SCHEDULER")
    assert load_config(str(ROOT / "calphad_agent.example.toml")).hpc.scheduler == "auto"


FLUX_INFO = {"scheduler": "flux", "reason": "Flux instance answers", "queues": ["pbatch", "pdebug"],
             "default_queue": "pbatch", "nodes": 4, "cores_per_node": 112, "local_cores": 8}
LOCAL_INFO = {"scheduler": "local", "reason": "no flux executable on PATH", "local_cores": 8}


def test_wizard_asks_bank_queue_only_with_flux(tmp_path, monkeypatch):
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.setup_wizard import hpc_check, run
    monkeypatch.setenv("MY_KEY", "x")
    monkeypatch.setenv("SOL_API_KEY", "x")
    base = ["https://llm.example.org/v1", "gpt-5.6-sol", "MY_KEY", "responses", "yes", "data", "ask"]
    # Flux found: two extra questions (bank, queue) before the connection test
    answers = iter(base + ["acct42", "pdebug", "n"])
    lines = []
    p = tmp_path / "flux.toml"
    assert run(str(p), inp=lambda prompt="": next(answers), out=lines.append, detect=lambda: FLUX_INFO) == 0
    h = load_config(str(p)).hpc
    assert (h.bank, h.queue, h.scheduler) == ("acct42", "pdebug", "auto")
    assert any("Flux scheduler found" in str(x) for x in lines)
    # no Flux: not asked (the answers end with the connection test)
    answers = iter(base + ["n"])
    p2 = tmp_path / "local.toml"
    assert run(str(p2), inp=lambda prompt="": next(answers), out=lambda *a: None, detect=lambda: LOCAL_INFO) == 0
    assert (load_config(str(p2)).hpc.bank, load_config(str(p2)).hpc.queue) == ("", "")
    # existing config kept: on Flux with no bank/queue it offers to set just those
    answers = iter(["n", "y", "acct7", "pbatch"])
    assert run(str(p2), inp=lambda prompt="": next(answers), out=lambda *a: None, detect=lambda: FLUX_INFO) == 0
    h = load_config(str(p2)).hpc
    assert (h.bank, h.queue) == ("acct7", "pbatch") and load_config(str(p2)).llm.model == "gpt-5.6-sol"
    # --setup --yes --bank --queue
    p3 = tmp_path / "yes.toml"
    assert run(str(p3), yes=True, preset={"bank": "b", "queue": "q"}, out=lambda *a: None,
               detect=lambda: FLUX_INFO) == 0
    assert (load_config(str(p3)).hpc.bank, load_config(str(p3)).hpc.queue) == ("b", "q")
    # calphad-agent --check shows the HPC result (and the reminder when bank/queue are missing)
    out = []
    hpc_check(str(p2), out=out.append, detect=lambda: FLUX_INFO)
    assert "HPC: Flux" in out[0] and "bank acct7, queue pbatch" in out[0] and len(out) == 1
    out = []
    hpc_check(str(tmp_path / "local.toml.missing"), out=out.append, detect=lambda: FLUX_INFO)
    assert any("set bank and queue under [hpc]" in x for x in out)
    out = []
    hpc_check(str(p), out=out.append, detect=lambda: LOCAL_INFO)
    assert out[0].startswith("HPC: local - no flux executable") and "direct runs here" in out[0]


def test_hpc_cli(fake_flux, tmp_path, capsys):
    p = tmp_path / "c.toml"
    p.write_text('[hpc]\nbank = ""\nqueue = ""\n')
    assert hpc.main(["--config", str(p)]) == 0
    out = capsys.readouterr().out
    assert "HPC: Flux" in out and "set bank and queue under [hpc]" in out
    assert hpc.main(["--config", str(p), "--remind-only"]) == 0
    out = capsys.readouterr().out
    assert "HPC: Flux" not in out and "set bank and queue" in out
    assert hpc.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cores_per_node"] == 112


# ---------------------------------------------------------------------------------------------- Flux end to end
@pytest.fixture
def crni(tmp_path):
    write_crni_inputs(tmp_path / "CrNi")
    return tmp_path


def test_flux_optimise_wait_continue(fake_flux, crni, monkeypatch):
    cfg = HPCConfig(bank="mybank", queue="pdebug", extra_flux_args=["--setattr=user.tag=calphad"], poll_s=0.2)
    tb = _tb(crni, cfg)
    text, _ = call(tb, "optimize_system", OPT)
    jid = _jid(text)
    assert "backend: flux" in text and "1 node(s) x 112 cores (flux resource info)" in text
    assert "queue pdebug; bank mybank" in text and "until='done'" in text
    # the submitted commands: bank / queue / extra args on the command line, MCMC chained afterok on generation
    subs = _submissions(fake_flux)
    assert [s["script"] for s in subs] == ["gen.flux", "mcmc.flux"]
    gen, mcmc = subs
    assert gen["argv"] == ["flux", "batch", "-q", "pdebug", "--bank=mybank", "--setattr=user.tag=calphad", "gen.flux"]
    assert mcmc["argv"] == ["flux", "batch", "-q", "pdebug", "--bank=mybank", "--setattr=user.tag=calphad",
                            f"--dependency=afterok:{gen['id']}", "mcmc.flux"]
    # resources are the scripts' directives: generation 1 core, MCMC 1 whole node (default) with 112 dask workers
    assert {"-N 1", "-n 1", "-t 1h"} <= set(gen["directives"])
    assert {"-N 1", "--exclusive", "-t 4h"} <= set(mcmc["directives"])
    d = tb.out / "espei_runs" / "CrNi"
    assert "--nworkers 112" in (d / "mcmc.flux").read_text()
    assert f"--dependency=afterok:$gen" in (d / "submit.sh").read_text()
    sub = json.loads((d / "hpc_submission.json").read_text())
    assert sub["backend"] == "flux" and sub["job_id"] == jid and [s["id"] for s in sub["steps"]] == \
        [gen["id"], mcmc["id"]]
    assert sub["resources"]["cores_per_node"] == 112 and sub["resources"]["bank"] == "mybank"
    spec = json.loads((tb.out / "jobs" / jid / "spec.json").read_text())
    assert spec["backend"] == "flux" and spec["submitted_by"]["tool"] == "optimize_system"
    # the provenance session log records the submission
    tb.provenance._ensure_log()
    log = [json.loads(x) for x in tb.provenance.log_path.read_text().splitlines()]
    ev = [x for x in log if x.get("type") == "hpc_submission"]
    assert ev and ev[0]["backend"] == "flux" and ev[0]["steps"][1]["id"] == mcmc["id"]
    # list_jobs while queued/running
    lj, _ = call(tb, "list_jobs", {})
    assert jid in lj
    # wait until done: status lines with one replace-key, then the report and the result
    t0 = time.time()
    wt, imgs = call(tb, "wait_job", {"job_id": jid, "until": "done"})
    print(f"\nfake Flux generation + MCMC, waited {time.time() - t0:.1f} s")
    assert "state: done" in wt, wt
    assert "ESPEI on Flux finished" in wt and "MCMC: 6/6 iterations" in wt
    assert f"generate {gen['id']} [COMPLETED], mcmc {mcmc['id']} [COMPLETED]" in wt
    assert "plot_diagram_with_data elements=[\"CR\", \"NI\"]" in wt and "mcmc.tdb" in wt
    keys = {k for k, _ in tb.ui.lines}
    assert keys == {f"wait:{jid}"} and "done after waiting" in tb.ui.lines[-1][1]
    assert any(i.endswith("lnprob_progress.png") for i in imgs)
    for f in ("dft.tdb", "trace.npy", "lnprob.npy", "mcmc.tdb", "mcmc.log", "generate_summary.json"):
        assert (d / f).exists(), f
    recs = jobs.progress_records(tb, jid)
    assert recs and recs[-1]["metrics"]["iteration"] == 6
    st = jobs.status(tb, jid)
    assert st["state"] == "done" and st["frac"] == 1.0 and st["start"] and st["end"]
    # ... and the agent continues with the result: the optimised TDB against the data, MCMC post-processing
    pt, pimgs = call(tb, "plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": "CrNi/datasets",
                                                     "tdb": str(d / "mcmc.tdb"), "tmin": 1400, "tmax": 2200})
    assert pimgs
    at, _ = call(tb, "analyze_mcmc", {"trace": str(d), "burn_in": "0"})
    assert "VV0000" in at
    # the record survives an agent restart (a new ToolBox on the same output dir)
    tb2 = _tb(crni, cfg)
    st2, _ = call(tb2, "job_status", {"job_id": jid})
    assert "state: done" in st2 and "backend: Flux batch" in st2
    # show_code for the optimised TDB: the YAMLs + batch scripts + the same flux batch commands
    from calphad_tools import codegen_espei
    monkeypatch.setattr(codegen_espei, "FLUX_POLL_S", 0.2)
    code_text, _, err = tb.call("show_code", {"target": str(d / "mcmc.tdb"), "include_parents": False})
    assert not err, code_text
    code = code_text.split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert "calphad_tools" not in code and "#flux: --exclusive" in code and "dask worker" in code
    assert "flux('batch', '-q', 'pdebug', '--bank=mybank'" in code and "--dependency=afterok:{gen_id}" in code
    compile(code, "show_code.py", "exec")
    # ... and it runs (against the fake flux) and reproduces the run
    gd = crni / "rerun"
    gd.mkdir()
    (gd / "gen.py").write_text(code)
    import subprocess
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=gd, capture_output=True, text=True,
                       timeout=600, env={**os.environ, "MPLBACKEND": "Agg"})
    assert r.returncode == 0, r.stderr[-3000:]
    assert "COMPLETED" in r.stdout
    assert (gd / "standalone_outputs" / "CrNi" / "mcmc.tdb").exists()
    assert len(_submissions(fake_flux)) == 4


def test_flux_stop_job_cancels(fake_flux, crni, monkeypatch):
    monkeypatch.setenv("FAKE_FLUX_MCMC_S", "120")
    tb = _tb(crni, HPCConfig(poll_s=0.2))
    text, _ = call(tb, "optimize_system", {**OPT, "iterations": 400})
    jid = _jid(text)
    t0 = time.time()
    while time.time() - t0 < 300:
        st = jobs.status(tb, jid)
        if (st.get("metrics") or {}).get("iteration", 0) >= 2 or st["state"] in jobs.FINAL:
            break
        time.sleep(0.3)
    assert st["state"] == "running" and "MCMC iteration" in st["message"], st
    live, imgs = call(tb, "job_status", {"job_id": jid})
    assert "state: running" in live and "ETA" in live and "ESPEI:" in live
    out, _ = call(tb, "stop_job", {"job_id": jid})
    assert "state: stopped" in out, out
    mcmc_id = json.loads((tb.out / "espei_runs/CrNi/hpc_submission.json").read_text())["steps"][1]["id"]
    assert ["cancel", mcmc_id] in _calls(fake_flux)
    assert "CANCELED" in out


def test_flux_job_failure_is_surfaced(fake_flux, crni, monkeypatch):
    monkeypatch.setenv("FAKE_FLUX_MODE", "failmcmc")
    tb = _tb(crni, HPCConfig(poll_s=0.2))
    text, _ = call(tb, "optimize_system", OPT)
    wt, _ = call(tb, "wait_job", {"job_id": _jid(text), "until": "done"})
    assert "state: failed" in wt, wt
    assert re.search(r"error: mcmc job \S+: Flux status FAILED, exit code 1", wt)
    assert "ValueError: fake MCMC failure" in wt and "batch output (" in wt and "espei-mcmc-CrNi." in wt


def test_flux_submission_error_never_falls_back_silently(fake_flux, crni, monkeypatch):
    monkeypatch.setenv("FAKE_FLUX_MODE", "reject")
    tb = _tb(crni, HPCConfig(queue="pbatch"))
    text, _ = call(tb, "optimize_system", OPT)
    assert text.startswith("FLUX SUBMISSION FAILED - nothing is running")
    assert "bank is required" in text and "[hpc] bank" in text and "backend='local'" in text
    assert "flux batch -q pbatch gen.flux" in text
    assert not (tb.out / "jobs").exists() or not list((tb.out / "jobs").iterdir())
    # interactive: the user is asked; "don't run" -> nothing runs
    ui = RecUI(["don't run (fix the Flux settings first)"])
    tb = _tb(crni, HPCConfig(), ui=ui, interactive=True)
    text, _ = call(tb, "optimize_system", OPT)
    assert ui.asked and "Run the optimisation locally" in ui.asked[0][0] and "FLUX SUBMISSION FAILED" in text
    assert not (tb.out / "jobs").exists() or not list((tb.out / "jobs").iterdir())
    # fallback_to_local = true: runs here, and says so
    tb = _tb(crni, HPCConfig(fallback_to_local=True))
    text, _ = call(tb, "optimize_system", {**OPT, "cores": 1})
    assert text.startswith("FLUX SUBMISSION FAILED") and "fallback_to_local = true: running it locally" in text
    jid = _jid(text)
    assert "backend: local" in text
    assert json.loads((tb.out / "espei_runs/CrNi/hpc_submission.json").read_text())["fallback_from_flux"]
    jobs.stop(tb, jid)


def test_flux_requested_but_missing(no_flux, crni):
    tb = _tb(crni, HPCConfig())
    text, _, err = tb.call("optimize_system", {**OPT, "backend": "flux"})
    assert err and "Flux was requested but is not usable here" in text


# ---------------------------------------------------------------------------------------------- waiting
def test_wait_until_done_can_be_ended(tmp_path):
    tb = _tb(tmp_path, HPCConfig())
    jid = jobs.submit(tb, "slow demo", "calphad_tools.jobs:demo", {"steps": 200, "delay": 0.1})
    threading.Timer(1.0, tb.stop_waiting.set).start()
    t0 = time.time()
    text, _ = call(tb, "wait_job", {"job_id": jid, "until": "done"})
    assert time.time() - t0 < 10
    assert "stopped waiting" in text and "at the user's request" in text and "still running" in text
    assert tb.ui.lines and all(k == f"wait:{jid}" for k, _ in tb.ui.lines)
    st, why = jobs.wait_until_done(tb, jid, max_s=0.5)
    assert st["state"] == "running" and "wait_max_hours" in why
    (tb.out / "jobs" / jid / "stop_wait").write_text("")
    st, why = jobs.wait_until_done(tb, jid)
    assert "user's request" in why and not (tb.out / "jobs" / jid / "stop_wait").exists()
    jobs.stop(tb, jid)
    # a short job: until done returns the finished report
    jid = jobs.submit(tb, "quick", "calphad_tools.jobs:demo", {"steps": 3, "delay": 0.05})
    text, _ = call(tb, "wait_job", {"job_id": jid, "until": "done"})
    assert "state: done" in text and "demo finished 3 steps" in text


# ---------------------------------------------------------------------------------------------- local path
def test_local_optimise_with_core_default(no_flux, crni):
    tb = _tb(crni, HPCConfig())                      # scheduler auto, no flux here -> local, half the cores
    text, _ = call(tb, "optimize_system", {**OPT, "iterations": 2})
    jid = _jid(text)
    n = hpc.available_cores()
    want = min(max(1, n // 2), 30)                   # 30 chains: 2 x 15 parameters
    assert "backend: local (no flux executable on PATH)" in text
    assert ("serial" in text) if max(1, n // 2) == 1 else (f"dask {max(1, n // 2)} cores" in text)
    t0 = time.time()
    wt, _ = call(tb, "wait_job", {"job_id": jid, "until": "done"})
    print(f"\nlocal optimize_system (generation + 2 MCMC iterations, {want} dask workers): {time.time() - t0:.1f} s")
    assert "state: done" in wt, wt
    assert "parameter generation: 15 fitted symbols" in wt and "2/2 iterations" in wt and "then: plot_diagram" in wt
    import yaml
    d = tb.out / "espei_runs" / "CrNi"
    m = yaml.safe_load((d / "espei_mcmc.yaml").read_text())["mcmc"]
    if want > 1:
        assert m["scheduler"] == "dask" and m["cores"] == want
    else:
        assert m["scheduler"] is None
    assert np.load(d / "trace.npy").shape == (30, 2, 15) and (d / "mcmc.tdb").exists()
    sub = json.loads((d / "hpc_submission.json").read_text())
    assert sub["backend"] == "local" and sub["resources"]["cores"] == max(1, n // 2)
    # show_code: two standalone cells (generation, MCMC)
    code_text, _, err = tb.call("show_code", {"target": str(d / "mcmc.tdb"), "include_parents": False})
    assert not err, code_text
    code = code_text.split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert "calphad_tools" not in code and code.count("run_espei(") == 2 and "generate_parameters:" in code
    compile(code, "show_code.py", "exec")


def test_espei_run_mcmc_uses_core_default(no_flux, crni):
    tb = _tb(crni, HPCConfig(local_cores=2))
    call(tb, "espei_generate_parameters", {"name": "c2", "phase_models": "CrNi/phase_models.json",
                                           "datasets": "CrNi/datasets"})
    text, _ = call(tb, "espei_run_mcmc", {"name": "c2", "iterations": 1})
    want = min(2, hpc.available_cores())
    assert (f"dask {want} cores" if want > 1 else "serial") in text and "local_cores=2" in text
    jobs.stop(tb, _jid(text))
    text, _ = call(tb, "espei_run_mcmc", {"name": "c2", "iterations": 1, "cores": 1})
    assert "serial (cores argument)" in text
    jobs.stop(tb, _jid(text))


def test_inside_allocation_skips_queue_and_bank(fake_flux, crni, monkeypatch):
    monkeypatch.setenv("FLUX_URI", "local:///tmp/flux-abc/local-0")       # agent started inside `flux alloc`
    monkeypatch.setenv("FAKE_FLUX_MCMC_S", "60")
    tb = _tb(crni, HPCConfig(bank="mybank", queue="pbatch", poll_s=0.2))
    text, _ = call(tb, "optimize_system", OPT)
    assert "inside a Flux allocation" in text
    assert [s["argv"][2:-1] for s in _submissions(fake_flux)][0] == []        # no -q / --bank in a sub-instance
    out, _ = call(tb, "stop_job", {"job_id": _jid(text)})
    assert "state: stopped" in out
