"""Compact tool descriptions (core tools in full, the rest one line + tool_help) and argument checks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from fake_openai_server import serve  # noqa: E402

from calphad_tools.agent.tools import CORE_TOOLS, UI, ToolBox  # noqa: E402


def _size(specs):
    return len(json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters}
                           for t in specs])) // 4


def _tb(tmp_path):
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    cfg = load_config(str(tmp_path / "none.toml"))
    cfg.llm.api_key = "x"
    cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
    return Agent(cfg, UI(), interactive=False).toolbox


def test_compact_list_is_small_and_static(tmp_path):
    tb = _tb(tmp_path)
    full, compact = tb.model_specs("full"), tb.model_specs("compact")
    assert [t.name for t in full] == [t.name for t in compact]          # same tools, same order
    assert _size(compact) < 0.45 * _size(full) and _size(compact) < 10000
    by_name = {t.name: t for t in full}
    for t in compact:
        if t.name in CORE_TOOLS:
            f = by_name[t.name]                                           # full description, schema without the
            assert t.description == f.description                         # per-argument explanations
            assert set(t.parameters["properties"]) == set(f.parameters["properties"])
            assert t.parameters.get("required") == f.parameters.get("required")
            assert "description" not in json.dumps(t.parameters)
        else:
            assert t.description.endswith("[tool_help]") and t.parameters == {"type": "object"}
            assert "Args:" in t.description
    assert tb.model_specs("compact") == compact                          # deterministic (prompt cache)


def test_tool_help_and_argument_errors(tmp_path):
    tb = _tb(tmp_path)
    text, _, err = tb.call("tool_help", {"names": ["convert_composition", "compare_system", "no_such_tol"]})
    assert not err and "## convert_composition" in text and "## compare_system" in text
    assert "no such tool" in text
    assert '"elements"' in text and "required" in text
    text, _, err = tb.call("convert_composition", {"composition": {"CR": 18}, "unit": "wt%"})
    assert err and "unknown argument(s) ['unit']" in text and "## convert_composition" in text
    text, _, err = tb.call("compare_system", {"folder": "tdbs"})
    assert err and "missing required ['elements']" in text


def test_agent_looks_up_then_uses_a_short_tool(tmp_path):
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.session import Agent
    steps = [{"tool_calls": [{"name": "tool_help", "arguments": {"names": ["convert_composition"]}}]},
             {"tool_calls": [{"name": "convert_composition", "arguments": {
                 "composition": {"CR": 18, "NI": 8}, "balance": "FE", "from_unit": "wt%"}}]},
             {"text": "done: ", "echo_last_tool": True}]
    srv = serve(steps, 8851, str(tmp_path / "req.jsonl"))
    try:
        cfg = load_config(str(tmp_path / "none.toml"))
        cfg.llm.base_url, cfg.llm.api_key = "http://127.0.0.1:8851/v1", "x"
        cfg.llm.api = "chat"                # this test checks /chat/completions request formats
        cfg.agent.workdir, cfg.agent.output_dir = str(ROOT / "examples"), str(tmp_path / "out")
        out = Agent(cfg, UI(), interactive=False).run_turn("Fe-18Cr-8Ni wt% to at%?")
    finally:
        srv.shutdown()
    assert out.startswith("done: ") and "19.1" in out
    reqs = [json.loads(line) for line in (tmp_path / "req.jsonl").read_text().splitlines()]
    sent = {t["function"]["name"]: t["function"] for t in reqs[0]["tools"]}
    assert sent["convert_composition"]["parameters"] == {"type": "object"}     # short form on the wire
    assert sent["map_phase_diagram"]["parameters"].get("properties")          # core tool: names and types
    assert len(json.dumps(reqs[0]["tools"])) // 4 < 9000
    assert all(r["tools"] == reqs[0]["tools"] for r in reqs)                  # never changes mid-conversation
    assert "## convert_composition" in json.dumps(reqs[1]["messages"])        # the help came back once
