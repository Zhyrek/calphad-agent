"""`calphad-agent` - chat with pycalphad in the terminal (or in a browser with --web)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from .config import EXAMPLE_CONFIG, load_config
from .tools import UI

DIM, BOLD, CYAN, YEL, RED, RST = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    DIM = BOLD = CYAN = YEL = RED = RST = ""


class TerminalUI(UI):
    def __init__(self, interactive: bool = True, allow_python: bool = False):
        self.interactive = interactive
        self.allow_python = allow_python

    def ask(self, question, options=None):
        if not self.interactive:
            return "(user not available - make a reasonable choice, state the assumption, and continue)"
        print(f"\n{YEL}? {question}{RST}")
        for i, o in enumerate(options or [], 1):
            print(f"  {i}) {o}")
        ans = input(f"{YEL}> {RST}").strip()
        if options and ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        return ans

    def confirm(self, title, body):
        if not self.interactive:
            return self.allow_python
        print(f"\n{YEL}{title}{RST}\n{DIM}{textwrap.indent(body, '  | ')}{RST}")
        return input(f"{YEL}run it? [y/N] {RST}").strip().lower() in ("y", "yes")

    _last_key = None

    def progress(self, msg, key=None):
        # a repeated key (e.g. rate-limit retries) rewrites the previous status line instead of adding one
        if key is not None and key == self._last_key and sys.stdout.isatty():
            print(f"\x1b[1A\x1b[2K{DIM}  … {msg}{RST}", flush=True)
        else:
            print(f"{DIM}  … {msg}{RST}", flush=True)
        self._last_key = key

    def notice(self, kind, text):
        if kind == "report":
            print(f"{YEL}  📋 {text}{RST}", flush=True)
            return
        col = CYAN if kind == "review_ok" else YEL
        print(f"{col}  🛡 {textwrap.indent(text, '    ').strip()}{RST}", flush=True)


def _short(d, n=160):
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def run_terminal(cfg, once: Optional[str] = None, allow_python: bool = False) -> int:
    from .session import Agent
    ui = TerminalUI(interactive=once is None, allow_python=allow_python)
    streamed = {"any": False}

    def on_text(t):
        streamed["any"] = True
        print(t, end="", flush=True)

    def on_tool_start(call):
        print(f"\n{CYAN}⚙ {call.name}{RST} {DIM}{_short(call.arguments)}{RST}", flush=True)

    def on_tool_end(call, text, images, err):
        first = text.strip().split("\n")
        head = "\n".join(first[:6]) + ("\n…" if len(first) > 6 else "")
        col = RED if err else DIM
        print(f"{col}{textwrap.indent(head, '  ')}{RST}")
        for im in images:
            print(f"  {BOLD}🖼  {im}{RST}")

    agent = Agent(cfg, ui, on_text=on_text, on_tool_start=on_tool_start, on_tool_end=on_tool_end,
                  interactive=once is None)
    if once is not None:
        out = agent.run_turn(once)
        if not streamed["any"]:
            print(out)
        print()
        rep = agent.turns[-1].report if agent.turns else None
        if rep:
            print(f"problem report: {rep}")
        return 0
    print(f"{BOLD}calphad-agent{RST}  model={cfg.llm.model}  provider={cfg.llm.provider}"
          f"{'  url=' + cfg.llm.base_url if cfg.llm.base_url else ''}")
    print(f"{DIM}workdir={agent.toolbox.workdir}  outputs={agent.toolbox.out}  python_tool={cfg.agent.python_tool}")
    print(f"commands: /reset  /files  /usage  /autorun on|off  /log [n]  /export [script|notebook|both] [steps]  /report [last] [why]"
          f"  /quit{RST}\n")
    while True:
        try:
            msg = input(f"{BOLD}you> {RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not msg:
            continue
        if msg in ("/quit", "/exit", "/q"):
            return 0
        if msg == "/reset":
            agent.reset()
            print("(conversation cleared)")
            continue
        if msg.startswith("/autorun"):
            arg = msg.split()[1] if len(msg.split()) > 1 else ""
            if "run_python" not in agent.toolbox.tools:
                print("(python tool is off in the config)")
            elif arg in ("on", "off"):
                agent.toolbox.python_tool = "auto_verify" if arg == "on" else "ask"
                print(f"(python_tool = {agent.toolbox.python_tool})")
            else:
                print(f"(python_tool = {agent.toolbox.python_tool}; use /autorun on|off)")
            continue
        if msg.startswith("/report"):
            note = msg[len("/report"):].strip()
            scope = "session"
            if note.split()[:1] == ["last"]:
                scope, note = "last", note[4:].strip()
            try:
                path = agent.save_report(note=note, scope=scope)
                print(f"problem report saved: {path}\n(give that zip, or its folder, to Claude Code to fix it)")
            except Exception as e:  # noqa: BLE001
                print(f"could not save report: {e}")
            continue
        if msg.startswith("/log"):
            parts = msg.split()
            args = {"last": int(parts[1])} if len(parts) > 1 and parts[1].isdigit() else {}
            print(agent.toolbox.call("session_log", args)[0])
            continue
        if msg.startswith("/export"):
            parts = msg.split()
            args = {"format": parts[1] if len(parts) > 1 else "both"}
            if len(parts) > 2:
                args["steps"] = parts[2]
            print(agent.toolbox.call("export_session", args)[0])
            continue
        if msg == "/usage":
            print(agent.usage_text())
            continue
        if msg == "/files":
            print(agent.toolbox.call("list_files", {"pattern": "**/*.*"})[0])
            continue
        streamed["any"] = False
        print(f"{BOLD}assistant>{RST} ", end="", flush=True)
        try:
            out = agent.run_turn(msg)
            if not streamed["any"]:
                print(out, end="")
        except KeyboardInterrupt:
            print(f"\n{RED}(interrupted){RST}")
            agent.reset()
            print("(conversation reset after interrupt)")
        except Exception as e:  # noqa: BLE001
            print(f"\n{RED}error: {type(e).__name__}: {e}{RST}")
        print("\n")


def main(argv: Optional[List[str]] = None) -> int:
    from ..hashseed import reexec_with_seed
    reexec_with_seed("calphad_tools.agent.cli", argv)    # PYTHONHASHSEED=0 unless set: reproducible mapping
    ap = argparse.ArgumentParser(description="Chat with pycalphad / CALPHAD tools through an LLM.")
    ap.add_argument("--config", help="config TOML (default ./calphad_agent.toml)")
    ap.add_argument("--init-config", action="store_true", help="write an example calphad_agent.toml and exit")
    ap.add_argument("--setup", action="store_true", help="first-time setup: asks for the settings, writes "
                                                         "calphad_agent.toml, tests the connection")
    ap.add_argument("--check", action="store_true", help="show the HPC detection (Flux or local) and test the "
                                                         "connection to the configured model, then exit")
    ap.add_argument("--link", action="store_true", help="print the access link of your running web UI and exit")
    ap.add_argument("--yes", action="store_true", help="with --setup: no questions, use the given values/defaults")
    ap.add_argument("--key-env", help="with --setup: name of the environment variable holding the API key")
    ap.add_argument("--bank", help="with --setup: Flux bank for batched ESPEI runs ([hpc] bank)")
    ap.add_argument("--queue", help="with --setup: Flux queue for batched ESPEI runs ([hpc] queue)")
    ap.add_argument("--web", action="store_true", help="browser chat UI instead of the terminal")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="with --web: do not open the browser automatically")
    ap.add_argument("--browser", action="store_true", help="with --web: always open the browser (e.g. over ssh -X)")
    ap.add_argument("--once", metavar="PROMPT", help="run a single prompt non-interactively and exit")
    ap.add_argument("--allow-python", action="store_true", help="with --once: allow run_python without asking")
    ap.add_argument("--auto-verify", action="store_true",
                    help="auto-run generated Python when an independent, context-free model review says it is safe")
    ap.add_argument("--profile", help="use [profiles.NAME] from the config instead of [llm]")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--provider", choices=["openai", "anthropic"])
    ap.add_argument("--workdir")
    a = ap.parse_args(argv)
    if a.setup:
        from .setup_wizard import run as setup
        return setup(a.config or "calphad_agent.toml", yes=a.yes,
                     preset={"base_url": a.base_url, "model": a.model, "api_key_env": a.key_env,
                             "workdir": a.workdir, "bank": a.bank, "queue": a.queue})
    if a.link:
        from .web import running_links
        links = running_links()
        if not links:
            print("no CALPHAD agent web UI of yours is running on this machine (start it with ./start-agent.sh)")
            return 1
        print("\n".join(links))
        return 0
    if a.check:
        from .setup_wizard import check
        return check(a.config or "calphad_agent.toml")
    if a.init_config:
        p = Path(a.config or "calphad_agent.toml")
        if p.exists():
            print(f"{p} already exists; not overwriting")
            return 1
        p.write_text(EXAMPLE_CONFIG)
        print(f"wrote {p} - edit base_url / api_key / model")
        return 0
    cfg = load_config(a.config, a.profile)
    if a.model:
        cfg.llm.model = a.model
    if a.base_url:
        cfg.llm.base_url = a.base_url
    if a.provider:
        cfg.llm.provider = a.provider
    if a.workdir:
        cfg.agent.workdir = a.workdir
    if a.auto_verify:
        cfg.agent.python_tool = "auto_verify"
    if not cfg.llm.api_key and cfg.llm.provider == "openai" and not cfg.llm.base_url:
        print("No API key / endpoint configured. Run `calphad-agent --setup` (or edit calphad_agent.toml), "
              "or export CALPHAD_LLM_API_KEY and CALPHAD_LLM_BASE_URL.")
        return 2
    if a.web:
        from .web import serve
        return serve(cfg, a.port, browser=False if a.no_browser else True if a.browser else None)
    return run_terminal(cfg, once=a.once, allow_python=a.allow_python)


if __name__ == "__main__":
    sys.exit(main())
