"""calphad-agent --setup: answers -> a valid calphad_agent.toml (comments kept); no key ever written by default."""
import os

import pytest

from calphad_tools.agent.config import load_config
from calphad_tools.agent.setup_wizard import run


@pytest.fixture(autouse=True)
def _no_flux(monkeypatch):
    """The answers below assume no Flux bank/queue questions (asked only when Flux is detected; see test_hpc)."""
    monkeypatch.setattr("calphad_tools.hpc.detect", lambda *a, **k: {"scheduler": "local", "reason": "test"})


def test_interactive_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SOL_KEY", "sk-from-env")
    monkeypatch.delenv("CALPHAD_LLM_API_KEY", raising=False)
    answers = iter(["https://llm.example.org/v1", "gpt-5.6-sol", "MY_SOL_KEY", "responses", "no", "data",
                    "ask", "n"])
    lines = []
    p = tmp_path / "calphad_agent.toml"
    assert run(str(p), inp=lambda prompt="": next(answers), out=lines.append) == 0
    cfg = load_config(str(p))
    assert (cfg.llm.base_url, cfg.llm.model, cfg.llm.api, cfg.llm.vision) == ("https://llm.example.org/v1",
                                                                               "gpt-5.6-sol", "responses", False)
    assert cfg.llm.api_key == "sk-from-env" and cfg.agent.python_tool == "ask"
    assert cfg.agent.workdir.endswith("data")
    text = p.read_text()
    assert "sk-from-env" not in text and "# may the model run arbitrary Python?" in text
    if os.name != "nt":
        assert (p.stat().st_mode & 0o777) == 0o600
    assert any("set in this terminal" in str(x) for x in lines)


def test_defaults_and_keep_existing(tmp_path):
    p = tmp_path / "c.toml"
    assert run(str(p), yes=True, preset={"base_url": "https://h/v1", "api_key_env": "K"}, out=lambda *a: None) == 0
    cfg = load_config(str(p))
    assert cfg.llm.base_url == "https://h/v1" and cfg.llm.api_key_env == "K" and cfg.agent.python_tool == "auto_verify"
    before = p.read_text()
    assert run(str(p), inp=lambda prompt="": "n", out=lambda *a: None) == 0      # "replace it?" -> no
    assert p.read_text() == before
