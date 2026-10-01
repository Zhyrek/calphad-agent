"""Problem reports: model-initiated, automatic, user-flagged and crash reports (fake OpenAI server)."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from fake_openai_server import serve  # noqa: E402

PORT = [8780]


def make_agent(tmp_path, steps, **agent_kw):
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    from calphad_tools.agent.tools import UI

    class U(UI):
        notices = []

        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False

        def notice(self, kind, text):
            self.notices.append((kind, text))

    PORT[0] += 1
    srv = serve(steps, PORT[0], str(tmp_path / f"req{PORT[0]}.jsonl"))
    cfg = load_config(str(tmp_path / "none.toml"))
    cfg.llm.base_url, cfg.llm.api_key = f"http://127.0.0.1:{PORT[0]}/v1", "sk-secret-test-key-123456"
    cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
    for k, v in agent_kw.items():
        setattr(cfg.agent, k, v)
    ui = U()
    return Agent(cfg, ui, interactive=False), srv, ui


def _report_dir(zip_path):
    z = Path(zip_path)
    assert z.exists() and zipfile.is_zipfile(z)
    return z.with_suffix("")


def test_model_initiated_report(tmp_path):
    steps = [
        {"tool_calls": [{"name": "tdb_info", "arguments": {"tdb": "CrNi-06Hal.tdb"}},
                        {"name": "map_phase_diagram", "arguments": {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                                    "temperature": 1000}}]},
        {"tool_calls": [{"name": "report_problem", "arguments": {
            "summary": "no tool computes a Cr-Ni diffusion couple", "kind": "missing_capability",
            "what_is_missing": "a diffusion simulation (mobility database + DICTRA-like solver)",
            "attempted": "map_phase_diagram (wrong kind of diagram for 2 elements + temperature)",
            "suggested_fix": "add a kawin-based diffusion tool"}}]},
        {"text": "I can't simulate diffusion couples yet; I saved a problem report."},
    ]
    agent, srv, ui = make_agent(tmp_path, steps)
    try:
        out = agent.run_turn("simulate a Cr/Ni diffusion couple at 1300 K for 10 h")
    finally:
        srv.shutdown()
    turn = agent.turns[-1]
    d = _report_dir(turn.report)
    md = (d / "REPORT.md").read_text()
    data = json.loads((d / "report.json").read_text())
    assert "simulate a Cr/Ni diffusion couple" in md and "missing_capability" in md
    assert data["created_by"] == "model" and data["diagnosis"]["suggested_fix"].startswith("add a kawin")
    calls = data["tool_calls"]
    assert [c["tool"] for c in calls] == ["tdb_info", "map_phase_diagram"]
    assert calls[1]["ok"] is False and ("Traceback" in calls[1]["result"] or "Error" in calls[1]["result"])
    assert calls[0]["source"]["file"].endswith("agent/tools.py")
    assert calls[1]["source"]["file"].endswith("mapping_tools.py")
    # a report nobody asked for copies no user files (the database may be proprietary): name + fingerprint only
    held = {f["name"]: f for f in data["files_withheld"]}
    assert data["inputs_copied"] == [] and not (d / "inputs").exists()
    assert "CrNi-06Hal.tdb" in held and len(held["CrNi-06Hal.tdb"]["sha256"]) == 64
    assert (d / "replay.py").exists()
    assert "sk-secret-test-key-123456" not in (d / "report.json").read_text() + md
    assert any(k == "report" for k, _ in ui.notices)
    # no second (automatic) report for the same turn
    assert len(list((agent.toolbox.out / "problem_reports").glob("*.zip"))) == 1
    # the replay script runs (failed steps included, not strict)
    r = subprocess.run([sys.executable, str(d / "replay.py"), "--workdir", str(ROOT / "examples"),
                        "--output-dir", str(tmp_path / "replay_out")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_automatic_report_on_failure(tmp_path):
    steps = [{"tool_calls": [{"name": "calculate_equilibrium", "arguments": {
                 "tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "conditions": {"T": 1000, "X(ZZ)": 0.3},
                 "temperature_unit": "K"}}]},
             {"text": "Sorry, I couldn't compute that equilibrium."}]
    agent, srv, ui = make_agent(tmp_path, steps)
    try:
        out = agent.run_turn("equilibrium of Cr-30Zz at 1000 K")
    finally:
        srv.shutdown()
    assert "[problem report saved:" in out
    data = json.loads((_report_dir(agent.turns[-1].report) / "report.json").read_text())
    assert data["created_by"] == "auto" and "failed tool call" in data["reason"]


def test_user_flagged_report_after_successful_turn(tmp_path):
    steps = [{"tool_calls": [{"name": "convert_composition", "arguments": {
                 "composition": {"CR": 18, "NI": 8}, "balance": "FE", "from_unit": "wt%"}}]},
             {"text": "Fe-18Cr-8Ni is 50 at% Cr."}]              # a wrong answer the user notices
    agent, srv, ui = make_agent(tmp_path, steps)
    try:
        agent.run_turn("what is Fe-18Cr-8Ni wt% in at%?")
    finally:
        srv.shutdown()
    assert agent.turns[-1].report is None                        # looked successful: no auto report
    path = agent.save_report(note="the answer says 50 at% Cr but the tool returned 19.15 at%",
                             share={"files": [], "results_text": True, "consent": "test"})
    d = _report_dir(path)
    md = (d / "REPORT.md").read_text()
    assert "19.15" in md and "50 at% Cr" in md and "user flagged" in md


def test_no_report_when_model_recovers(tmp_path):
    steps = [{"tool_calls": [{"name": "tdb_info", "arguments": {"tdb": "NoSuch.tdb"}}]},
             {"tool_calls": [{"name": "tdb_info", "arguments": {"tdb": "CrNi-06Hal.tdb"}}]},
             {"text": "Cr-Ni-06Hal has 9 phases."}]
    agent, srv, ui = make_agent(tmp_path, steps)
    try:
        agent.run_turn("describe the Cr-Ni database")
    finally:
        srv.shutdown()
    assert agent.turns[-1].report is None
    assert not (agent.toolbox.out / "problem_reports").exists()


def test_crash_report(tmp_path):
    agent, srv, ui = make_agent(tmp_path, [])
    srv.shutdown()                                      # the LLM endpoint is gone -> the turn crashes
    agent.cfg.llm.timeout = 5
    agent.reset()
    with pytest.raises(Exception):
        agent.run_turn("hello")
    rep = agent.turns[-1].report
    md = (_report_dir(rep) / "REPORT.md").read_text()
    assert "The turn crashed" in md and "Traceback" in md and ("APITimeoutError" in md or "Connection" in md)


def test_auto_reports_can_be_disabled(tmp_path):
    steps = [{"tool_calls": [{"name": "tdb_info", "arguments": {"tdb": "NoSuch.tdb"}}]},
             {"text": "I cannot find that database."}]
    agent, srv, ui = make_agent(tmp_path, steps, auto_failure_reports=False)
    try:
        agent.run_turn("describe NoSuch.tdb")
    finally:
        srv.shutdown()
    assert agent.turns[-1].report is None
