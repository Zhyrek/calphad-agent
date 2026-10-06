"""Configuration for the LLM agent.

Looked up in this order (later wins):
  1. built-in defaults
  2. ~/.config/calphad_tools/config.toml
  3. ./calphad_agent.toml   (or the file given with --config)
  4. environment variables: CALPHAD_LLM_PROVIDER, CALPHAD_LLM_BASE_URL, CALPHAD_LLM_MODEL, CALPHAD_LLM_API,
     CALPHAD_LLM_API_KEY.  If no key is set, the key is read from the variable named by
     ``api_key_env``, then from OPENAI_API_KEY / ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover  (python < 3.11)
    import tomli as tomllib  # type: ignore


@dataclass
class LLMConfig:
    provider: str = "openai"          # "openai" (any OpenAI-compatible endpoint) | "anthropic" | "claude_cli"
    base_url: str = ""                # e.g. https://llm.mylab.gov/v1  (empty = provider default)
    api_key: str = ""
    api_key_env: str = ""             # name of an environment variable holding the key (used if api_key is empty)
    model: str = "gpt-5.6-sol"
    api: str = "responses"            # openai only: "responses" (/responses) | "chat" (/chat/completions)
    reasoning_effort: str = "medium"  # "" to omit (non-reasoning models)
    max_output_tokens: int = 16000    # ceiling for one answer
    initial_output_tokens: int = 6000 # what each request reserves (Azure counts it against the TPM limit up
                                      # front); doubled towards max_output_tokens only if an answer is cut off
    temperature: Optional[float] = None
    vision: bool = True               # send plots/PDF pages back to the model as images
    extra_headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 600.0
    # rate limits and caching (append-only conversation; nothing here rewrites earlier messages)
    rate_limit_mode: str = "aggressive"    # 429s: "aggressive" = resend within rate_limit_retry_cap_s, no self-pacing
                                      # (a shared quota goes to whoever asks first) | "polite" = wait as long as the
                                      # server asks, learn a pace and stay below it
    rate_limit_retry_cap_s: float = 5.0    # aggressive: longest wait between 429 retries (also caps Retry-After)
    max_retries: int = 8              # polite: rate-limited (429) requests are resent after the wait the server asks for
    rate_limit_max_wait_s: float = 300.0   # give up after waiting this long in total for one request
    transient_retries: int = 4        # dropped connections / timeouts / 5xx (incl. a stream cut mid-answer)
    tokens_per_minute: Any = "auto"   # pacing: a number (your deployment's TPM) | "auto": learn it from the first
                                      # rate-limit response, then stay below it | 0: off
    prompt_cache_key: str = "auto"    # "auto": one key per conversation (GPT-5.6 cache matching); "" = none
    image_max_px: int = 1600          # plots are downscaled to this long side before they are sent (0 = as is)
    image_detail: str = "auto"        # auto | low | high (low = ~85 tokens per image, less detail)
    attach_images: str = "viewers"    # viewers: the model sees images only from view_image / digitizer previews
                                      # (once; then a note replaces them) | all | none
    verify_ssl: bool = True
    ca_bundle: str = ""               # path to an institutional CA bundle, if needed
    fallbacks: str = "default"        # anthropic only: server-side refusal fallbacks ("" to disable)
    cli_path: str = "claude"          # claude_cli only: path to the Claude Code CLI
    cli_workdir: str = ""             # claude_cli only: neutral cwd for the child (default: a temp dir)
    cli_env_passthrough: list = field(default_factory=list)  # claude_cli: CLAUDE*/ANTHROPIC_* vars to keep
    reviewer_model: str = ""          # model for the independent code review ("" = same as model)
    reviewer_effort: str = ""         # reasoning effort for the review ("" = same as reasoning_effort)


@dataclass
class AgentConfig:
    workdir: str = "."
    output_dir: str = "calphad_outputs"
    python_tool: str = "ask"          # ask | auto_verify | auto | off   (arbitrary Python execution)
    max_tool_rounds: int = 40
    tool_descriptions: str = "compact"    # compact: core tools in full, the rest one line + tool_help on demand
    max_tool_output_chars: int = 8000     # longer replies: the rest goes to <output_dir>/.tool_outputs/
    auto_failure_reports: bool = True     # save a problem report when a turn ends in failure
    report_max_input_mb: float = 5.0      # copy input files up to this size into problem reports
    diagram_style: str = "shaded"         # phase diagrams: shaded (black lines, filled two-phase fields, grey
                                          # ternaries, no legend) | color (colour per phase + legend)
    save_notebooks: bool = True           # save <figure>.ipynb next to every phase diagram: standalone pycalphad
                                          # code that regenerates all of its files (see show_code)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    source_files: list = field(default_factory=list)
    profiles: list = field(default_factory=list)
    profile: str = ""


def _apply(obj, data: Dict):
    for k, v in (data or {}).items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def load_config(path: Optional[str] = None, profile: Optional[str] = None) -> Config:
    """Load settings.  ``profile`` selects a ``[profiles.<name>]`` table, which is a complete
    alternative to ``[llm]`` (fields it leaves out take the built-in defaults, so e.g. an OpenAI key
    in ``[llm]`` never leaks into an Anthropic profile)."""
    cfg = Config()
    profile = profile or os.environ.get("CALPHAD_LLM_PROFILE") or None
    candidates = [Path.home() / ".config" / "calphad_tools" / "config.toml"]
    candidates.append(Path(path) if path else Path.cwd() / "calphad_agent.toml")
    profiles: Dict[str, Dict] = {}
    for p in candidates:
        if p and p.is_file():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            _apply(cfg.llm, data.get("llm", {}))
            _apply(cfg.agent, data.get("agent", {}))
            profiles.update(data.get("profiles", {}))
            cfg.source_files.append(str(p))
    cfg.profiles = sorted(profiles)
    if profile:
        if profile not in profiles:
            raise KeyError(f"profile {profile!r} not found; available: {sorted(profiles) or 'none'}")
        cfg.llm = LLMConfig()
        _apply(cfg.llm, profiles[profile])
        cfg.profile = profile
    env = os.environ
    if env.get("CALPHAD_LLM_PROVIDER"):
        cfg.llm.provider = env["CALPHAD_LLM_PROVIDER"]
    if env.get("CALPHAD_LLM_BASE_URL"):
        cfg.llm.base_url = env["CALPHAD_LLM_BASE_URL"]
    if env.get("CALPHAD_LLM_MODEL"):
        cfg.llm.model = env["CALPHAD_LLM_MODEL"]
    if env.get("CALPHAD_LLM_API"):                  # "chat" | "responses" (OpenAI-compatible endpoints)
        api = env["CALPHAD_LLM_API"].strip().lower()
        if api not in ("chat", "responses"):
            raise ValueError(f"CALPHAD_LLM_API must be 'chat' or 'responses', not {api!r}")
        cfg.llm.api = api
    if env.get("CALPHAD_LLM_API_KEY"):
        cfg.llm.api_key = env["CALPHAD_LLM_API_KEY"]
    if env.get("CALPHAD_LLM_API_KEY_ENV"):
        cfg.llm.api_key_env = env["CALPHAD_LLM_API_KEY_ENV"]
    if not cfg.llm.api_key and cfg.llm.api_key_env:
        cfg.llm.api_key = env.get(cfg.llm.api_key_env.strip().lstrip("$"), "")
        if not cfg.llm.api_key:
            import warnings
            warnings.warn(f"api_key_env = {cfg.llm.api_key_env!r} but that environment variable is not set "
                          f"(or is empty) in this shell")
    if not cfg.llm.api_key:
        fallback = "ANTHROPIC_API_KEY" if cfg.llm.provider == "anthropic" else "OPENAI_API_KEY"
        cfg.llm.api_key = env.get(fallback, "")
    return cfg


EXAMPLE_CONFIG = '''# calphad_agent.toml  -  settings for `calphad-agent`
# Put this file in the directory you run calphad-agent from (or ~/.config/calphad_tools/config.toml).
# Keep it out of version control if it contains your key; you can instead leave api_key empty
# and export CALPHAD_LLM_API_KEY=... in your shell.

[llm]
provider = "openai"                       # any OpenAI-compatible endpoint (vLLM, LiteLLM, Azure-style gateways...)
base_url = "https://YOUR-LLM-HOST/v1"     # <-- custom URL of your hosted model
api_key  = ""                             # <-- your key (or set CALPHAD_LLM_API_KEY)
api_key_env = ""                          # or: the NAME of an env variable holding the key, e.g. "SOL_API_KEY"
model    = "gpt-5.6-sol"                  # model name exactly as your endpoint expects it
api      = "responses"                    # "responses" = /responses ; "chat" = /chat/completions (hosts without /responses)
reasoning_effort = "medium"               # low | medium | high ; "" if the endpoint rejects it
max_output_tokens = 16000                 # ceiling; each request reserves initial_output_tokens (6000)
vision = true                             # let the model look at the plots it makes
rate_limit_mode = "aggressive"            # 429s: resend within 5 s, no self-pacing (shared quotas: first come, first
                                          # served) | "polite": wait as asked, learn a pace and stay below it
# rate_limit_retry_cap_s = 5              # aggressive: longest wait between retries
tokens_per_minute = "auto"                # polite mode: "auto" learns a pace from the first 429 and stays below it;
                                          # or a fixed TPM (paces in both modes, e.g. 150000); 0 = no pacing
# image_detail = "auto"                   # "low" = ~85 tokens per plot image (less detail, far fewer tokens)
# extra_headers = { "X-Custom-Header" = "value" }
# verify_ssl = true
# ca_bundle = "/etc/ssl/certs/my-institution-ca.pem"

# --- alternative models: named profiles, selected with `calphad-agent --profile NAME` ---
# A profile is a complete alternative to [llm] (unset fields take defaults, not [llm]'s values).
# [profiles.claude]
# provider = "anthropic"
# model    = "claude-sonnet-5"
# base_url = "https://api.anthropic.com"  # explicit, so an ANTHROPIC_BASE_URL in your env isn't used
# api_key  = ""                           # or ANTHROPIC_API_KEY

[agent]
workdir = "."                             # relative paths in chat are resolved against this
output_dir = "calphad_outputs"            # plots, decks, datasets are written here
python_tool = "ask"                       # may the model run arbitrary Python?
#   ask         - show you every script and wait for OK
#   auto_verify - a fresh, context-free copy of the model reviews each script; clean "safe" verdicts
#                 (and no hard static-scan flags) run automatically, anything else asks you
#   auto        - run everything without review;  off - no Python tool at all
# reviewer_model = ""                     # (under [llm]) optional different model for the review
max_tool_rounds = 40
diagram_style = "shaded"                  # phase diagrams: shaded (black lines, filled two-phase fields) | color
save_notebooks = true                     # <figure>.ipynb next to every phase diagram: pycalphad code that
                                          # regenerates all of its files (PNG/SVG/PDF, CSVs, JSON)
'''
