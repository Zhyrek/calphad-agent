"""api_key_env: the key is read from the environment variable the config names."""
import pytest

from calphad_tools.agent.config import load_config


def _cfg(tmp_path, body):
    p = tmp_path / "c.toml"
    p.write_text(body)
    return p


def test_key_from_named_env(tmp_path, monkeypatch):
    for v in ("CALPHAD_LLM_API_KEY", "CALPHAD_LLM_API_KEY_ENV", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MY_SOL_KEY", "sk-from-env")
    cfg = load_config(str(_cfg(tmp_path, '[llm]\napi_key_env = "MY_SOL_KEY"\n')))
    assert cfg.llm.api_key == "sk-from-env"


def test_explicit_key_wins_and_missing_env_warns(tmp_path, monkeypatch):
    for v in ("CALPHAD_LLM_API_KEY", "CALPHAD_LLM_API_KEY_ENV", "OPENAI_API_KEY", "NOPE_KEY"):
        monkeypatch.delenv(v, raising=False)
    cfg = load_config(str(_cfg(tmp_path, '[llm]\napi_key = "sk-file"\napi_key_env = "MY_SOL_KEY"\n')))
    assert cfg.llm.api_key == "sk-file"
    with pytest.warns(UserWarning, match="NOPE_KEY"):
        cfg = load_config(str(_cfg(tmp_path, '[llm]\napi_key_env = "NOPE_KEY"\n')))
    assert cfg.llm.api_key == ""


def test_profile_can_name_its_own_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CALPHAD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("CALPHAD_LLM_API_KEY_ENV", raising=False)
    monkeypatch.setenv("OTHER_KEY", "sk-other")
    cfg = load_config(str(_cfg(tmp_path, '[llm]\napi_key_env = "X"\n[profiles.p]\napi_key_env = "OTHER_KEY"\n')),
                      profile="p")
    assert cfg.llm.api_key == "sk-other"
