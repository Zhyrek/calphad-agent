"""package_feedback / save_report: the user's statement, the conversation, and only the files they agreed to share."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from fake_openai_server import serve  # noqa: E402

PORT = [8840]


class ScriptedUI:
    """Answers the UI questions (consent confirmation, statement) from a list, and records them."""

    def __init__(self, answers):
        self.answers, self.asked, self.notices = list(answers), [], []

    def ask(self, q, options=None):
        self.asked.append(q)
        return self.answers.pop(0) if self.answers else ""

    def confirm(self, t, b):
        return False

    def progress(self, m):
        pass

    def notice(self, kind, text):
        self.notices.append((kind, text))


def make_agent(tmp_path, steps, answers, interactive=True):
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    wd = tmp_path / "wd"
    wd.mkdir()
    shutil.copy(ROOT / "examples" / "tdbs" / "CrFe-82Her.tdb", wd / "Fe-Cr-internal.tdb")
    PORT[0] += 1
    srv = serve(steps, PORT[0], str(tmp_path / "req.jsonl"))
    cfg = load_config(str(tmp_path / "none.toml"))
    cfg.llm.base_url, cfg.llm.api_key = f"http://127.0.0.1:{PORT[0]}/v1", "sk-test-key-000000000000"
    cfg.agent.workdir, cfg.agent.output_dir = str(wd), str(tmp_path / "out")
    cfg.agent.auto_failure_reports = False
    ui = ScriptedUI(answers)
    return Agent(cfg, ui, interactive=interactive), srv, ui


MAP = {"tool_calls": [{"name": "map_phase_diagram", "arguments": {
    "tdb": "Fe-Cr-internal.tdb", "elements": ["FE", "CR"], "tmin": 800, "tmax": 1400, "temperature_unit": "C"}}]}


def _bundle(path):
    z = Path(path)
    assert zipfile.is_zipfile(z)
    d = z.with_suffix("")
    return d, json.loads((d / "report.json").read_text()), (d / "REPORT.md").read_text()


def test_conversation_about_what_to_share(tmp_path):
    """TDB proprietary, the figure made from it is OK: exactly that is packaged, after the user confirms."""
    steps = [MAP, {"text": "Here is the Fe-Cr diagram."},
             {"tool_calls": [{"name": "package_feedback", "arguments": {"action": "list"}}]},
             {"text": "I understand Fe-Cr-internal.tdb may be proprietary - may the diagram image made from it "
                      "be included?"}]
    agent, srv, ui = make_agent(tmp_path, steps, ["yes, write the report"])
    try:
        agent.run_turn("map Fe-Cr from 800 to 1400 C with Fe-Cr-internal.tdb")
        out = agent.run_turn("the sigma field is far too wide, please send this to the developers")
        # what the model saw: the numbered list, with the lineage of the outputs
        listing = agent.toolbox.call_log[-1]["result_full"]
        assert "[input] Fe-Cr-internal.tdb" in listing and "made from Fe-Cr-internal.tdb" in listing
        from calphad_tools.agent.failure import candidate_files, report_turns
        files = list(candidate_files(agent, report_turns(agent)[:1]))
        png = next(i for i, p in enumerate(files, 1) if p.endswith(".png") and "provenance" not in p)
        steps += [{"tool_calls": [{"name": "package_feedback", "arguments": {
            "action": "write", "include": [png], "user_statement": "the sigma field is far too wide",
            "consent_notes": "user: the TDB is proprietary, the figure is fine"}}]},
            {"text": "Saved the report."}]
        agent.run_turn("don't include the tdb, it's proprietary, but the figure is fine")
    finally:
        srv.shutdown()
    assert "may the diagram image" in out
    assert "Is this OK to package" in ui.asked[-1] and "withhold" in ui.asked[-1]
    rep = [t for k, t in ui.notices if k == "report"]
    d, data, md = _bundle(Path(agent.toolbox.out) / rep[-1].split(": ", 1)[1])
    assert data["user_note"] == "the sigma field is far too wide"
    assert [f["name"] for f in data["files_included"]] == [Path(files[png - 1]).name]
    assert not (d / "inputs").exists() and (d / "outputs" / Path(files[png - 1]).name).exists()
    held = {f["name"]: f for f in data["files_withheld"]}
    assert "Fe-Cr-internal.tdb" in held and len(held["Fe-Cr-internal.tdb"]["sha256"]) == 64
    assert "confirmed by the user" in data["sharing"]["consent"]
    assert [t["request"][:14] for t in data["turns"]] == ["map Fe-Cr from", "the sigma fiel"]
    assert "Why the user is not satisfied" in md and "Fe-Cr-internal.tdb" in md
    assert (d / "replay.py").exists()


def test_user_declines_final_list(tmp_path):
    steps = [MAP, {"text": "done"},
             {"tool_calls": [{"name": "package_feedback", "arguments": {
                 "action": "write", "include": [1], "user_statement": "wrong"}}]},
             {"text": "ok, what should change?"}]
    agent, srv, ui = make_agent(tmp_path, steps, ["no, change something"])
    try:
        agent.run_turn("map it")
        agent.run_turn("that's wrong, send it")
    finally:
        srv.shutdown()
    assert "did not confirm" in agent.toolbox.call_log[-1]["result_full"]
    assert not (Path(agent.toolbox.out) / "problem_reports").exists()


def test_statement_required_and_noninteractive_shares_nothing(tmp_path):
    steps = [MAP, {"text": "done"},
             {"tool_calls": [{"name": "package_feedback", "arguments": {"action": "write", "include": [1]}}]},
             {"tool_calls": [{"name": "package_feedback", "arguments": {
                 "action": "write", "include": [1, 2], "user_statement": "labels are wrong"}}]},
             {"text": "saved"}]
    agent, srv, ui = make_agent(tmp_path, steps, [], interactive=False)
    try:
        agent.run_turn("map it")
        agent.run_turn("labels are wrong, report it")
    finally:
        srv.shutdown()
    calls = [c for c in agent.toolbox.call_log if c["tool"] == "package_feedback"]
    assert "user_statement is required" in calls[0]["result_full"]
    assert "0 file(s) included" in calls[1]["result_full"]
    d, data, md = _bundle(sorted((Path(agent.toolbox.out) / "problem_reports").glob("*.zip"))[-1])
    assert data["files_included"] == [] and "non-interactive" in data["sharing"]["consent"]


def test_save_report_menu_consent_and_private_mode(tmp_path):
    """/report and the web button: statement asked if missing; numbered picks; the most private option."""
    agent, srv, ui = make_agent(tmp_path, [MAP, {"text": "done"}], ["the eutectoid is at the wrong T", "2, 1"])
    try:
        agent.run_turn("map it")
    finally:
        srv.shutdown()
    from calphad_tools.agent.failure import candidate_files, report_turns
    files = list(candidate_files(agent, report_turns(agent)))
    path = agent.save_report()                                     # asks: statement, then "2, 1"
    d, data, md = _bundle(path)
    assert data["user_note"] == "the eutectoid is at the wrong T"
    assert [f["path"] for f in data["files_included"]] == [agent.toolbox.rel(files[0])]
    ui.answers = ["no files and no tool-result text (most private)"]
    d2, data2, md2 = _bundle(agent.save_report(note="still wrong", scope="last"))
    assert data2["files_included"] == [] and data2["sharing"]["results_text"] is False
    assert all(c["result"] == "(withheld at the user's request)" for c in data2["tool_calls"])
    # web dialog: the decision comes with the request
    d3, data3, _ = _bundle(agent.save_report(note="x", share={"files": [], "results_text": True,
                                                                "consent": "no files ticked"}))
    assert data3["sharing"]["consent"] == "no files ticked"
