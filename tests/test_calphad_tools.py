"""Fast tests (no LLM, no long mapping runs).  Run:  .venv/bin/python -m pytest -q tests"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"
EXP = ROOT / "examples" / "expdata"
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("tdb", sorted(TDBS.glob("*.tdb")), ids=lambda p: p.name)
def test_parser_matches_pycalphad(tdb):
    from pycalphad import Database, variables as v
    from calphad_tools.tdbparse import parse_tdb
    db = parse_tdb(str(tdb))
    assert not db.warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb = Database(str(tdb))
    assert len(db.parameters) == len(pdb._parameters)
    T = np.array([400.0, 1500.0])
    for name in list(db.functions)[:40]:
        mine = db.evaluate_function(name, T)
        expr = pdb.symbols[name]
        for _ in range(10):
            expr = expr.xreplace({s: pdb.symbols[s.name] for s in expr.free_symbols if s.name in pdb.symbols})
        ref = []
        for t in T:
            try:
                ref.append(float(expr.subs({v.T: t, v.P: 101325})))
            except Exception:
                ref.append(np.nan)
        assert np.allclose(np.nan_to_num(mine), np.nan_to_num(ref), rtol=1e-7, atol=1e-6), name


def test_rk_sign_flip_and_aliases():
    from calphad_tools.tdbparse import parse_param_name
    k1, s1, _ = parse_param_name("L(BCC_A2,FE,CR:VA;1)")
    k2, s2, _ = parse_param_name("G(BCC_A2,CR,FE:VA;1)")
    assert k1.as_tuple() == k2.as_tuple() and s1 == -1.0 and s2 == 1.0
    k3, s3, _ = parse_param_name("L(BCC_A2,FE,CR:VA;2)")
    assert s3 == 1.0
    assert parse_param_name("BM(FCC_A1,NI:VA)")[0].ptype == "BMAGN"


# ---------------------------------------------------------------- compare
def test_compare_aligns_vacancy_sublattice(tmp_path):
    from calphad_tools import compare
    r = compare.run(str(TDBS / "CrFe-82Her.tdb"), str(TDBS / "Cr-Fe_her.tdb"), str(tmp_path / "c.pptx"))
    assert Path(r["pptx"]).stat().st_size > 20000 and Path(r["csv"]).exists()
    assert r["counts"]["phase"]["equivalent"] == 2          # BCC/FCC with vs without VA sublattice
    assert r["counts"]["parameter"]["identical"] >= 10


def test_compare_self_is_identical(tmp_path):
    from calphad_tools import compare
    r = compare.run(str(TDBS / "CrNi-06Hal.tdb"), str(TDBS / "CrNi-06Hal.tdb"), str(tmp_path / "s.pptx"))
    c = r["counts"]
    assert all(c[sec]["different"] == 0 and c[sec]["only A"] == 0 for sec in c)


# ---------------------------------------------------------------- merge
def test_merge_three_binaries(tmp_path):
    from calphad_tools import merge
    from pycalphad import Database, equilibrium, variables as v
    paths = [str(TDBS / f) for f in ("CrFe-82Her.tdb", "CrNi-06Hal.tdb", "FeNi-00Ans-2SL.tdb")]
    plan = merge.plan_merge(paths)
    ids = {c.id for c in plan.pending}
    assert "parameter:G(FCC_A1,CR:VA;0)" in ids                  # pre-SGTE vs SGTE fcc-Cr
    out = tmp_path / "m.tdb"
    r = merge.merge(paths, str(out), decisions={"parameter:G(FCC_A1,CR:VA;0)": "db2"}, policy="first")
    assert r["written"] and r["validation"]["loaded"] and not r["validation"]["errors"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, s = Database(str(out)), Database(paths[1])
        c = {v.X("NI"): 0.4, v.T: 1300, v.P: 101325, v.N: 1}
        gm = [float(equilibrium(d, ["CR", "NI", "VA"], ["LIQUID", "FCC_A1", "BCC_A2"], c).GM.squeeze())
              for d in (m, s)]
    assert abs(gm[0] - gm[1]) < 1e-3                               # merged reproduces source binary


def test_merge_alias_and_function_keep_all(tmp_path):
    from calphad_tools import merge
    paths = [str(TDBS / "CrFe-82Her.tdb"), str(TDBS / "Cr-Fe_her.tdb")]
    dec = {"alias:SIGMA|SIGMA_D8B": "unify:SIGMA_D8B", "function:GLIQCR": "keep_all"}
    plan = merge.plan_merge(paths, dec, policy="first")
    assert "SIGMA" not in plan.phases and "SIGMA_D8B" in plan.phases
    assert "GLIQCR_2" in plan.functions
    assert not plan.pending


# ---------------------------------------------------------------- experimental data
def test_pop_to_zpf():
    from calphad_tools import expdata
    ds, notes = expdata.pop_to_zpf(str(EXP / "CrNi_boundaries.POP"), ["CR", "NI"],
                                   tdb_phases=["LIQUID", "FCC_A1", "BCC_A2"])
    assert len(ds) == 2 and any("ACR" in n for n in notes)
    tie = ds[0]["values"][0]
    assert tie == [["LIQUID", ["CR"], [0.35]], ["FCC_A1", ["CR"], [0.33]]]
    assert ds[1]["conditions"]["T"] == [1700.0, 1800.0, 1900.0]


def test_exp_and_csv_to_zpf():
    from calphad_tools import expdata
    p = expdata.parse_exp(str(EXP / "CrNi_liquidus.exp"))
    assert p["axes_guess"]["y_unit"] == "C" and len(p["datasets"]) == 2
    ds = expdata.exp_to_zpf(p, ["CR", "NI"], {"1": {"phases": ["LIQUID", "FCC_A1"], "measured_phase": "LIQUID"}})
    assert ds[0]["conditions"]["T"][0] == pytest.approx(1693.15)
    c = expdata.csv_to_zpf(str(EXP / "CrNi_digitized.csv"), ["CR", "NI"])
    assert {d["reference"] for d in c} == {"Jones 1975", "Brown 1980"}
    with pytest.raises(expdata.ZPFError):
        expdata.make_zpf_dataset(["CR", "NI"], [1000], [[["FCC_A1", ["CR"], [45.0]], ["BCC_A2", ["CR"], [None]]]])


def test_wt_percent_conversion():
    from calphad_tools.expdata import to_mole_fraction
    assert to_mole_fraction(50, "wt%", "CR", "NI") == pytest.approx(0.53025, rel=1e-4)


# ---------------------------------------------------------------- ESPEI / Flux
def test_espei_flux_generation(tmp_path):
    from calphad_tools import espei_flux as ef, expdata
    pm = ef.phase_models_from_tdb(str(TDBS / "CrNi-06Hal.tdb"), ["CR", "NI"], str(tmp_path / "pm.json"))
    assert "LIQUID" in pm["phases"] and "_validation_error" not in pm
    dsd = tmp_path / "ds"
    ds, _ = expdata.pop_to_zpf(str(EXP / "CrNi_boundaries.POP"), ["CR", "NI"], tdb_phases=list(pm["phases"]))
    for i, d in enumerate(ds):
        expdata.write_dataset(d, str(dsd / f"{i}.json"))
    runs = [ef.Run("a", str(tmp_path / "pm.json"), str(dsd)),
            ef.Run("b", str(tmp_path / "pm.json"), str(dsd), input_db=str(TDBS / "CrNi-06Hal.tdb"),
                   generate_parameters=False)]
    for mode in ("per-job", "bundle"):
        r = ef.generate(runs, ef.FluxSettings(nodes=2, cores_per_node=8, mode=mode), str(tmp_path / mode))
        assert not r["problems"], r["problems"]
    mcmc = (tmp_path / "per-job" / "a" / "mcmc.flux").read_text()
    assert "#flux: -N 2" in mcmc and "dask worker" in mcmc and 'cd "$(dirname' not in mcmc
    assert "--dependency=afterok" in (tmp_path / "per-job" / "submit_all.sh").read_text()


# ---------------------------------------------------------------- agent loop (fake OpenAI server)
def test_agent_loop_with_fake_openai(tmp_path):
    from fake_openai_server import serve
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    from calphad_tools.agent.tools import UI

    class AutoUI(UI):
        def ask(self, q, options=None):
            return (options or ["ok"])[0]

        def confirm(self, title, body):
            return True

    steps = [
        {"tool_calls": [{"name": "tdb_info", "arguments": {"tdb": "CrNi-06Hal.tdb"}},
                        {"name": "phase_property_plot", "arguments": {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                                      "prop": "GM", "temperature": 1500, "temperature_unit": "K"}}]},
        {"tool_calls": [{"name": "run_python", "arguments": {"code": "print(len(load_db('CrNi-06Hal.tdb').phases))"}}]},
        {"text": "done: ", "echo_last_tool": True},
    ]
    log = tmp_path / "req.jsonl"
    srv = serve(steps, 8798, str(log))
    try:
        cfg = load_config(str(tmp_path / "none.toml"))
        cfg.llm.base_url, cfg.llm.api_key = "http://127.0.0.1:8798/v1", "x"
        cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
        out = Agent(cfg, AutoUI()).run_turn("hi")
    finally:
        srv.shutdown()
    assert out.startswith("done: 9")
    reqs = [json.loads(line) for line in log.read_text().splitlines()]
    roles = [m["role"] for m in reqs[1]["messages"]]
    # the plot is for the user: no image message (the model looks with view_image when it needs to)
    assert roles.count("tool") == 2 and roles[-1] == "tool"
    assert any(Path(tmp_path / "out").glob("*.png"))


# ---------------------------------------------------------------- auto-verify python review
def test_static_scan_flags():
    from calphad_tools.agent.review import static_scan, HARD
    hard = lambda c: [f.what for f in static_scan(c) if f.level == HARD]  # noqa: E731
    assert hard("import subprocess\nsubprocess.run(['ls'])")
    assert hard("import os\nos.system('rm -rf /')")
    assert hard("import requests\nrequests.get('http://x')")
    assert hard("exec(base64.b64decode(s))")
    assert hard("open(os.path.expanduser('~/.ssh/id_rsa')).read()")
    assert not hard("import numpy as np\nx = np.linspace(0,1)\nprint(x.sum())")
    assert not hard("db = load_db('a.tdb')\nimport matplotlib.pyplot as plt\nplt.plot([1]); savefig('x')")


def test_auto_verify_gate(tmp_path):
    from fake_openai_server import serve
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    from calphad_tools.agent.tools import UI

    notices, confirms = [], []

    class NoHumanUI(UI):
        def ask(self, q, options=None):
            return "n/a"

        def confirm(self, title, body):
            confirms.append(title)
            return False

        def notice(self, kind, text):
            notices.append((kind, text))

    safe = '{"verdict": "safe", "summary": "prints a number", "concerns": []}'
    steps = [
        {"tool_calls": [{"name": "run_python", "arguments": {"code": "x = 6*7\nprint(x)"}}]},
        {"text": safe},                                            # reviewer for snippet 1
        {"tool_calls": [{"name": "run_python", "arguments": {"code": "import subprocess\nprint(subprocess.run(['echo','hi']))"}}]},
        {"text": safe},                                            # a fooled reviewer - static HARD flag must still block
        {"tool_calls": [{"name": "run_python", "arguments": {"code": "import os\nos.remove(os.path.expanduser('~/notes.txt'))"}}]},
        {"text": '{"verdict": "unsafe", "summary": "deletes a file in home", "concerns": ["line 2 deletes ~/notes.txt"]}'},
        {"text": "finished"},
    ]
    log = tmp_path / "req.jsonl"
    srv = serve(steps, 8797, str(log))
    try:
        cfg = load_config(str(tmp_path / "none.toml"))
        cfg.llm.base_url, cfg.llm.api_key = "http://127.0.0.1:8797/v1", "x"
        cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
        cfg.agent.python_tool = "auto_verify"
        agent = Agent(cfg, NoHumanUI(), interactive=False)
        out = agent.run_turn("go")
    finally:
        srv.shutdown()
    assert out == "finished"
    reqs = [json.loads(line) for line in log.read_text().splitlines()]
    reviewer_reqs = [r for r in reqs if "tools" not in r]
    assert len(reviewer_reqs) == 3
    for r in reviewer_reqs:                       # fresh context: only system + one user message, no tools
        assert [m["role"] for m in r["messages"]] == ["system", "user"]
        assert "go" not in r["messages"][1]["content"].split("CODE TO REVIEW")[0][-50:]
    tool_msgs = [m["content"] for m in reqs[-1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("42")
    assert tool_msgs[1].startswith("Not run") and "shell/subprocess" in tool_msgs[1]
    assert tool_msgs[2].startswith("Not run") and "UNSAFE" in tool_msgs[2]
    assert [k for k, _ in notices] == ["review_ok", "review_flag", "review_flag"]
    assert not confirms                           # nobody asked when non-interactive
    lines = (tmp_path / "out" / "python_reviews.jsonl").read_text().splitlines()
    assert [json.loads(l)["auto_ok"] for l in lines] == [True, False, False]


def test_anthropic_backend_with_fake_server(tmp_path):
    """Claude backend: streaming, tool_use -> tool_result (with image), thinking blocks echoed, reviewer oneshot."""
    from fake_openai_server import serve
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    from calphad_tools.agent.tools import UI

    class NoHumanUI(UI):
        def ask(self, q, options=None):
            return "n/a"

        def confirm(self, title, body):
            return False

    steps = [
        {"text": "Computing.", "tool_calls": [{"name": "phase_property_plot", "arguments": {
            "tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "prop": "GM", "temperature": 1500, "temperature_unit": "K"}}]},
        {"tool_calls": [{"name": "run_python", "arguments": {"code": "print(sum(range(10)))"}}]},
        {"text": '{"verdict": "safe", "summary": "adds numbers", "concerns": []}'},   # reviewer
        {"text": "result: ", "echo_last_tool": True},
    ]
    log = tmp_path / "req.jsonl"
    srv = serve(steps, 8796, str(log))
    try:
        cfg = load_config(str(tmp_path / "none.toml"))
        cfg.llm.provider, cfg.llm.model = "anthropic", "claude-sonnet-5"
        cfg.llm.base_url, cfg.llm.api_key = "http://127.0.0.1:8796", "x"
        cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
        cfg.agent.python_tool = "auto_verify"
        out = Agent(cfg, NoHumanUI(), interactive=False).run_turn("go")
    finally:
        srv.shutdown()
    assert out.startswith("result: 45")
    reqs = [json.loads(line) for line in log.read_text().splitlines()]
    main = [r for r in reqs if "tools" in r]
    review = [r for r in reqs if "tools" not in r]
    assert len(main) == 3 and len(review) == 1
    assert review[0]["messages"][0]["role"] == "user" and len(review[0]["messages"]) == 1
    assert "fallbacks" not in main[0]                                  # not sent for Sonnet
    last = main[-1]["messages"]
    assistant_blocks = [b["type"] for m in last if m["role"] == "assistant" for b in m["content"]]
    assert "thinking" in assistant_blocks and "tool_use" in assistant_blocks   # thinking echoed back
    first_result = last[2]["content"][0]
    assert first_result["type"] == "tool_result" and [c["type"] for c in first_result["content"]] == ["text", "image"]


def test_claude_cli_backend_text_protocol(tmp_path, monkeypatch):
    """`claude -p` backend: tool calls parsed from text, results fed back via --resume, isolated reviewer."""
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    from calphad_tools.agent.tools import UI

    class NoHumanUI(UI):
        def ask(self, q, options=None):
            return "n/a"

        def confirm(self, title, body):
            return False

    script = [
        'I will check the database.\n<tool_call>{"name": "tdb_info", "arguments": {"tdb": "CrNi-06Hal.tdb"}}</tool_call>',
        '<tool_call>\n```json\n{"name": "run_python", "arguments": {"code": "print(2**10)"}}\n```\n</tool_call>',
        '{"verdict": "safe", "summary": "prints a power of two", "concerns": []}',                    # reviewer
        'Cr-Ni has 9 phases and the code printed 1024.',
    ]
    (tmp_path / "script.json").write_text(json.dumps(script))
    log = tmp_path / "cli.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", str(tmp_path / "script.json"))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-session")     # must NOT reach the child
    cfg = load_config(str(tmp_path / "none.toml"))
    cfg.llm.provider, cfg.llm.model = "claude_cli", "sonnet"
    cfg.llm.cli_path = str(Path(__file__).parent / "fake_claude_cli.py")
    cfg.llm.cli_workdir = str(tmp_path / "cwd")
    cfg.llm.cli_env_passthrough = ["FAKE_CLAUDE_SCRIPT", "FAKE_CLAUDE_LOG"]
    cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
    cfg.agent.python_tool = "auto_verify"
    out = Agent(cfg, NoHumanUI(), interactive=False).run_turn("how many phases in CrNi-06Hal.tdb?")
    assert out == "Cr-Ni has 9 phases and the code printed 1024."
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 4
    for c in calls:
        a = c["argv"]
        assert a[a.index("--tools") + 1] == "" and "--strict-mcp-config" in a
        assert "CLAUDE_CODE_SESSION_ID" not in c["env"]
    main = [calls[0], calls[1], calls[3]]
    assert "--resume" not in main[0]["argv"] and all("--resume" in c["argv"] for c in main[1:])
    assert "<tool_call>" in main[0]["argv"][main[0]["argv"].index("--system-prompt") + 1]
    assert '<tool_result name="tdb_info"' in main[1]["stdin"] and "1024" in main[2]["stdin"]
    rev = calls[2]["argv"]
    assert "--resume" not in rev and "--no-session-persistence" in rev
    assert "independent security reviewer" in rev[rev.index("--system-prompt") + 1]


def test_tool_only_mode_offers_everything_but_python(tmp_path):
    """With arbitrary Python off, every plotting/data capability is still offered as a dedicated tool."""
    from fake_openai_server import serve
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    from calphad_tools.agent.tools import UI

    class NoHumanUI(UI):
        def ask(self, q, options=None):
            return "n/a"

        def confirm(self, title, body):
            return False

    steps = [{"tool_calls": [{"name": "step_map", "arguments": {
                 "tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "vary": "T", "start": 1500, "stop": 1800,
                 "fixed": {"X(NI)": 0.4}}}]},
             {"text": "ok"}]
    log = tmp_path / "req.jsonl"
    srv = serve(steps, 8795, str(log))
    try:
        cfg = load_config(str(tmp_path / "none.toml"))
        cfg.llm.base_url, cfg.llm.api_key = "http://127.0.0.1:8795/v1", "x"
        cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
        cfg.agent.python_tool = "off"
        agent = Agent(cfg, NoHumanUI(), interactive=False)
        agent.run_turn("step Cr-40Ni from 1500 to 1800 K")
    finally:
        srv.shutdown()
    req = json.loads(log.read_text().splitlines()[0])
    offered = {t["function"]["name"] for t in req["tools"]}
    assert "run_python" not in offered
    for needed in ("map_phase_diagram", "step_map", "equilibrium_property_plot", "phase_property_plot",
                   "export_equilibrium_data", "scheil_solidification", "write_database", "export_phase_model",
                   "plot_diagram_with_data", "plot_parameter_vs_data", "convert_composition", "make_plot", "edit_figure", "describe_figure",
                   "pycalphad_help", "list_calphad_capabilities"):
        assert needed in offered, needed
    assert "Arbitrary Python is disabled" in req["messages"][0]["content"]
    msgs = json.loads(log.read_text().splitlines()[1])["messages"]
    tool_msg = [m for m in msgs if m["role"] == "tool"][-1]
    assert "1617" in tool_msg["content"]                                         # Cr-Ni eutectic found


def test_text_protocol_parses_native_invoke_blocks():
    from calphad_tools.agent.llm import parse_text_tool_calls
    raw = ('Let me compute.\n<invoke name="run_python">\n<parameter name="code">print(1+1)\n</parameter>\n'
           '</invoke>\nDone.')
    calls, text = parse_text_tool_calls(raw)
    assert calls[0][0] == "run_python" and calls[0][1]["code"].strip() == "print(1+1)"
    assert "<invoke" not in text and "Let me compute." in text
    calls, text = parse_text_tool_calls('x <tool_call>{"name": "tdb_info", "arguments": {"tdb": "a.tdb"}}</tool_call>')
    assert calls == [("tdb_info", {"tdb": "a.tdb"}, '{"name": "tdb_info", "arguments": {"tdb": "a.tdb"}}', "")]
    # a JSON-valued parameter is decoded
    calls, _ = parse_text_tool_calls('<invoke name="t"><parameter name="elements">["CR","NI"]</parameter></invoke>')
    assert calls[0][1]["elements"] == ["CR", "NI"]
