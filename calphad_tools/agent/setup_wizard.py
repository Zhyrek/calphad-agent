"""Interactive first-time setup: writes calphad_agent.toml and (optionally) tests the connection.

    calphad-agent --setup                   # asks each question, with defaults in [brackets]
    calphad-agent --setup --yes --base-url https://host/v1 --key-env SOL_API_KEY   # no questions

The installers (install.sh / install.ps1) run this after creating the virtual environment.

When a Flux scheduler is detected (:func:`calphad_tools.hpc.detect`) it also asks for the Flux bank and queue
(``[hpc]``), which batched ESPEI runs need; elsewhere those questions are skipped.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

from .config import EXAMPLE_CONFIG

FIELDS = [
    # key, question, default, help
    ("base_url", "Address (URL) of your model's API", "https://YOUR-LLM-HOST/v1",
     "the OpenAI-compatible endpoint of your hosted model, usually ending in /v1 (empty = api.openai.com)"),
    ("model", "Model name", "gpt-5.6-sol", "exactly as your endpoint expects it"),
    ("api_key_env", "Name of the environment variable that holds your API key", "SOL_API_KEY",
     "recommended: the key stays out of files. Leave empty to paste the key into the config instead"),
    ("api", "API style: responses or chat", "responses",
     "'responses' (/responses; Azure OpenAI v1, OpenAI) is the default; 'chat' (/chat/completions) for hosts without /responses"),
    ("vision", "Can the model look at images? (yes/no)", "yes",
     "lets it check the plots it makes and read figures from papers; say no if your endpoint rejects images"),
    ("workdir", "Folder with your databases and data", "examples",
     "relative paths in chat are resolved here; the included examples folder works to start with"),
    ("python_tool", "May the model run its own Python? ask / auto_verify / off", "auto_verify",
     "ask = you approve each script; auto_verify = an independent review approves safe ones; off = never"),
]


# asked only when Flux is detected (key, question, help); written to [hpc]
HPC_FIELDS = [
    ("bank", "Flux bank (allocation account) for batched ESPEI runs",
     "flux batch --bank=BANK; leave empty if your site needs none"),
    ("queue", "Flux queue for batched ESPEI runs", "flux batch -q QUEUE; leave empty for the site default"),
]


def _detect(detect=None) -> Dict:
    try:
        if detect is not None:
            return detect()
        from ..hpc import detect as _d
        return _d()
    except Exception as e:  # noqa: BLE001 - setup must never fail on the HPC probe
        return {"scheduler": "local", "reason": f"detection failed: {type(e).__name__}: {e}"}


def _ask_hpc(info: Dict, current: Dict[str, str], inp=input, out=print) -> Dict[str, str]:
    q = f"; queues here: {', '.join(info['queues'])}" if info.get("queues") else ""
    out(f"\nFlux scheduler found ({info.get('reason')}): ESPEI optimisations will be submitted as batch jobs{q}.")
    vals = {}
    for key, question, help_ in HPC_FIELDS:
        default = current.get(key) or (info.get("default_queue") or "" if key == "queue" else "")
        vals[key] = _ask(question, default, help_, inp, out)
    return vals


def _ask(question: str, default: str, help_: str, inp=input, out=print) -> str:
    out(f"\n{question}\n  ({help_})")
    ans = inp(f"  [{default}] > ").strip()
    return ans or default


def _set(text: str, key: str, value: str, section: str = "llm") -> str:
    """Replace `key = ...` inside [section] of the example config (keeps the comments)."""
    lit = value if value in ("true", "false") else '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    start = text.index(f"[{section}]")
    nxt = text.find("\n[", start + 1)
    end = len(text) if nxt < 0 else nxt
    block = text[start:end]
    pat = re.compile(rf"(?m)^({re.escape(key)}\s*=\s*)(\"[^\"]*\"|\S+)")
    if pat.search(block):
        block = pat.sub(lambda m: m.group(1) + lit, block, count=1)
    else:
        block = block.rstrip("\n") + f"\n{key} = {lit}\n"
    return text[:start] + block + text[end:]


def build_config(values: Dict[str, str], api_key: str = "", hpc: Optional[Dict[str, str]] = None) -> str:
    t = EXAMPLE_CONFIG
    for k in ("base_url", "model", "api_key_env", "api"):
        t = _set(t, k, values.get(k, ""))
    t = _set(t, "api_key", api_key)
    t = _set(t, "vision", "true" if str(values.get("vision", "yes")).lower().startswith(("y", "t")) else "false")
    for k in ("workdir", "python_tool"):
        t = _set(t, k, values.get(k, ""), section="agent")
    for k, v in (hpc or {}).items():
        t = set_hpc(t, k, v)
    return t


def set_hpc(text: str, key: str, value: str) -> str:
    """Set ``key`` in the [hpc] table of a config text (the table is appended from the example when missing)."""
    if "\n[hpc]" not in "\n" + text:
        text = text.rstrip("\n") + "\n\n" + EXAMPLE_CONFIG[EXAMPLE_CONFIG.index("[hpc]"):]
    return _set(text, key, str(value or ""), section="hpc")


def env_hint(name: str) -> str:
    if os.name == "nt":
        return (f'  PowerShell (this window):  $env:{name} = "your-key"\n'
                f'  permanently (new windows): setx {name} "your-key"')
    shell = Path(os.environ.get("SHELL", "")).name or "sh"
    rc = {"zsh": "~/.zshrc", "bash": "~/.bashrc"}.get(shell, "your shell's startup file")
    return f'  this terminal:  export {name}="your-key"\n  permanently:    add that line to {rc}'


def test_connection(path: Path, out=print) -> bool:
    from .config import load_config
    from .llm import oneshot
    cfg = load_config(str(path))
    if not cfg.llm.api_key:
        out(f"  no API key found yet (set {cfg.llm.api_key_env or 'CALPHAD_LLM_API_KEY'} first) - skipping the test")
        return False
    try:
        reply = oneshot(cfg.llm, "Reply with exactly: pong", "ping", max_tokens=20)
        out(f"  the model answered: {reply.strip()[:60]!r}  -> the connection works")
        return True
    except Exception as e:  # noqa: BLE001
        out(f"  connection test failed: {type(e).__name__}: {str(e)[:300]}")
        out("  check base_url, model and the key, then run:  calphad-agent --check")
        return False


def _offer_hpc_update(p: Path, detect=None, inp=input, out=print) -> None:
    """Existing config kept: on a Flux system with no bank/queue set yet, offer to set just those."""
    info = _detect(detect)
    if info.get("scheduler") != "flux":
        return
    from .config import load_config
    try:
        h = load_config(str(p)).hpc
    except Exception:  # noqa: BLE001
        return
    if h.bank and h.queue:
        return
    a = inp("\nFlux was found but the Flux bank/queue are not set. Set them now? [Y/n] > ").strip().lower()
    if a not in ("", "y", "yes"):
        return
    vals = _ask_hpc(info, {"bank": h.bank, "queue": h.queue}, inp, out)
    t = p.read_text()
    for k, v in vals.items():
        t = set_hpc(t, k, v)
    p.write_text(t)
    out(f"updated [hpc] in {p.resolve()}")


def run(path: str = "calphad_agent.toml", yes: bool = False, preset: Optional[Dict[str, str]] = None,
        inp=input, out=print, detect=None) -> int:
    p = Path(path)
    preset = {k: v for k, v in (preset or {}).items() if v is not None}
    out("CALPHAD agent setup - press Enter to accept the value in [brackets].")
    if p.exists() and not yes:
        a = inp(f"\n{p} already exists. Replace it? [y/N] > ").strip().lower()
        if a not in ("y", "yes"):
            out("kept the existing configuration.")
            _offer_hpc_update(p, detect, inp, out)
            return 0
    vals: Dict[str, str] = {}
    for key, q, default, help_ in FIELDS:
        d = preset.get(key, default)
        vals[key] = d if yes else _ask(q, d, help_, inp, out)
    api_key = preset.get("api_key", "")
    if not vals["api_key_env"] and not api_key and not yes:
        api_key = inp("\nPaste your API key (stored in calphad_agent.toml - keep that file private) > ").strip()
    hpc_vals = {k: preset[k] for k in ("bank", "queue") if preset.get(k)}
    info = _detect(detect)
    if info.get("scheduler") == "flux" and not yes:
        hpc_vals.update(_ask_hpc(info, hpc_vals, inp, out))
    p.write_text(build_config(vals, api_key, hpc=hpc_vals))
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    out(f"\nwrote {p.resolve()}")
    if vals["api_key_env"]:
        have = bool(os.environ.get(vals["api_key_env"]))
        out(f"\nYour key is read from the environment variable {vals['api_key_env']} "
            f"({'set in this terminal' if have else 'NOT set yet'}). To set it:\n{env_hint(vals['api_key_env'])}")
    if not yes:
        a = inp("\nTest the connection to the model now? [Y/n] > ").strip().lower()
        if a in ("", "y", "yes"):
            test_connection(p, out)
    return 0


def hpc_check(path: str = "calphad_agent.toml", out=print, detect=None) -> None:
    """The HPC detection (Flux or local, resources, queues, run-size defaults) for calphad-agent --check."""
    from ..hpc import HPCConfig, reminder, summary
    cfg = HPCConfig()
    if Path(path).is_file():
        try:
            from .config import load_config
            cfg = load_config(path).hpc
        except Exception as e:  # noqa: BLE001
            out(f"  (could not read [hpc] from {path}: {e})")
    info = _detect(detect)
    out(summary(info, cfg))
    rem = reminder(info, cfg, path)
    if rem:
        out(rem)


def check(path: str = "calphad_agent.toml") -> int:
    hpc_check(path)
    return 0 if test_connection(Path(path)) else 1


if __name__ == "__main__":
    sys.exit(run())
