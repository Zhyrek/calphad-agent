"""Offline demo of the web UI: a scripted fake LLM drives real tools.  python tests/demo_web_offline.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fake_openai_server import serve as fake
from calphad_tools.agent.config import load_config
from calphad_tools.agent.web import serve

steps = [
    {"text": "I'll compute that Co–Cr–Ni isothermal section at 1000 K.",
     "tool_calls": [{"name": "map_phase_diagram", "arguments": {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                                   "temperature": 1000, "temperature_unit": "K"}}]},
    {"tool_calls": [{"name": "ask_user", "arguments": {"question": "Also plot phase fractions vs T for an alloy?",
                                                        "options": ["Co-30Cr-30Ni", "no thanks"]}}]},
    {"tool_calls": [{"name": "run_python", "arguments": {"code": "db = load_db('CoCrFeNiV_Choi2019.tdb')\nprint(len(db.phases), 'phases')"}}]},
    {"text": "Saved **ternary_CO-CR-NI_1000K_CoCrFeNiV_Choi2019.png** in the outputs folder. Result of the check: ", "echo_last_tool": True},
]
fake(steps, 8799, str(Path(__file__).parent / "fake_requests.jsonl"))
cfg = load_config("/nonexistent.toml")
cfg.llm.base_url, cfg.llm.api_key, cfg.llm.model = "http://127.0.0.1:8799/v1", "test", "gpt-5.6-sol (offline fake)"
cfg.agent.workdir, cfg.agent.output_dir = str(Path(__file__).parents[1] / "examples"), "calphad_outputs"
serve(cfg, int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
