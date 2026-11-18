"""`calphad-mcp` - the calphad-agent toolbox as an MCP server (stdio), for Claude Code, Codex or any other MCP client.

The client's agent replaces `calphad-agent`'s own chat loop: it reads the tool list, decides what to call, and this
process executes the calls.  The tools are the same ones the built-in agent uses (calphad_tools.agent.tools.ToolBox
plus every plugin module), with the same argument checks, '@last' references, call log and reproduction notebooks.

Why a server and not the client's shell: an MCP stdio server is a child process of the client, not a command run
inside the client's sandbox, so pycalphad / ESPEI / Flux run with the permissions this process was started with -
exactly like `calphad-agent`.  That is why the server offers only the fixed CALPHAD routines: there is no
run_python (python_tool is always off here).  Arbitrary code stays with the client - its own Python runs inside its
own sandbox and approval rules - so the elevated permissions cover pycalphad/ESPEI work, not anything a model writes.

Protocol: newline-delimited JSON-RPC 2.0 on stdin/stdout (MCP stdio transport), no extra dependency.  fd 1 is
pointed at stderr as soon as the server starts and the protocol goes to a private duplicate of the real stdout, so
prints from pycalphad, ESPEI or child processes can never corrupt it.  Tool calls run one at a time on a worker
thread; the reader thread keeps answering pings, cancellations and the replies to our own requests meanwhile.

Questions raised inside a tool (e.g. "run ESPEI locally instead?") go through MCP elicitation when the client
offers it; without it they get the non-interactive answer.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent.tools import UI

# protocol revisions this server speaks (newest first); a client asking for another one gets the newest
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# the client's agent asks its user itself; no arbitrary code in the server (python_tool is forced off as well);
# tool_help is only useful with compact descriptions
HIDDEN_TOOLS = {"ask_user", "run_python"}
GUIDE_TOOL = "calphad_guide"
# progress notification while a long tool runs (when the client sent a token); some clients reset their timeout on it
HEARTBEAT_S = float(os.environ.get("CALPHAD_MCP_HEARTBEAT_S", "15") or 15)

NO_USER = ("(no answer: this MCP client cannot show questions from inside a tool. Make a reasonable choice, state "
           "the assumption in your reply, and continue - or stop and ask your user, then call the tool again with "
           "the decision as an argument.)")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("calphad-tools")
    except Exception:  # noqa: BLE001 - running from a source tree
        return "0"


def _log(msg: str) -> None:
    """Diagnostics go to stderr (clients show or log it; stdout is the protocol)."""
    try:
        sys.stderr.write(f"[calphad-mcp] {msg}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------- guidance text
PYTHON_NOTE = ("- This server has no run_python: use the dedicated tools for every CALPHAD calculation, plot and "
               "data export (calphad_guide / list_calphad_capabilities show which tool does what). For anything they "
               "cannot express, write and run Python with your own tools, under your own permissions; show_code "
               "gives standalone pycalphad code for any output of these tools as a starting point.")


def guide_text(workdir: str, outdir: str, compact: bool, files: Optional[str] = None) -> str:
    """The built-in agent's system prompt, adapted for a client agent that calls these tools over MCP: no
    ask_user (the client asks its own user), no run_python, no problem-report tools, no claim about internet."""
    from .agent.session import SYSTEM_PROMPT
    text = SYSTEM_PROMPT.format(workdir=workdir, outdir=outdir, files=files or "(call list_files / find_databases)",
                                python_note=PYTHON_NOTE)
    head, sep, rest = text.partition("Working directory:")
    intro = ("These tools come from calphad-tools (MCP server `calphad-mcp`): a local Python process with "
             "pycalphad, ESPEI and CALPHAD automation tools loaded. The tools execute in that process, with its "
             "permissions, and write files into the output directory. Relative paths in tool arguments resolve "
             "against the working directory; result files are reported relative to it, and images also with "
             "their absolute paths (open them with your own file/image viewer if you need to look).\n\n")
    text = intro + sep + rest if sep else text
    for old, new in (("(ask_user if available, otherwise ask in your reply and stop)", "(ask your user)"),
                     ("and ask (ask_user) instead", "and ask the user instead"),
                     ("confirm with ask_user if unsure", "confirm with the user if unsure"),
                     ("ask the user (ask_user) how many", "ask the user how many"),
                     ("You have no internet access. ", ""),
                     (" (ask_user)", ""), ("ask_user", "your user")):
        text = text.replace(old, new)
    if not compact:                      # every tool already comes with its full schema
        text = re.sub(r"- Tools whose description ends in \"\[tool_help\]\".*?(?=\n- |\Z)", "", text, flags=re.S)
    # bullets (a "- " line plus its continuation lines) about the built-in agent's own report/feedback tools
    blocks: List[List[str]] = []
    for line in text.split("\n"):
        if line.startswith("- ") or not line.strip() or not blocks or not blocks[-1][0].startswith("- "):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    out = []
    for b in blocks:
        body = "\n".join(b)
        if "package_feedback" in body:
            continue
        if "report_problem" in body:
            body = ("- If you cannot fully satisfy a request (missing capability, a tool error you cannot work "
                    "around, results that look wrong), say so plainly and offer the best alternative. Do not "
                    "pretend a partial result is complete.")
        out.append(body)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


# ---------------------------------------------------------------------------------------------- transport
class Wire:
    """Newline-delimited JSON-RPC messages.  By default: reads stdin, writes a private duplicate of the real
    stdout, and points fd 1 (and sys.stdout) at stderr so nothing else can write into the protocol stream."""

    def __init__(self, inp=None, out=None):
        if out is None:
            sys.stdout.flush()
            fd = os.dup(1)
            os.dup2(2, 1)
            sys.stdout = sys.stderr
            out = os.fdopen(fd, "wb", buffering=0)
        self.out = out
        self.inp = inp if inp is not None else sys.stdin.buffer
        self.lock = threading.Lock()

    def send(self, msg: Dict[str, Any]) -> None:
        data = (json.dumps(msg, separators=(",", ":"), ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self.lock:
            self.out.write(data)
            self.out.flush()

    def messages(self):
        for raw in self.inp:
            raw = raw.strip()
            if raw:
                yield raw


# ---------------------------------------------------------------------------------------------- user interaction
class MCPUI(UI):
    """ToolBox UI on top of the MCP connection: questions become elicitation requests (when the client supports
    them), status lines become progress notifications (when the call carried a progress token)."""

    def __init__(self, server: "MCPServer"):
        self.s = server

    def can_elicit(self) -> bool:
        return isinstance(self.s.client_caps.get("elicitation"), dict)

    def ask(self, question, options=None):
        if not self.can_elicit():
            return NO_USER
        prop: Dict[str, Any] = {"type": "string", "title": "Answer"}
        if options:
            prop["enum"] = [str(o) for o in options]
        r = self.s.request("elicitation/create", {
            "message": str(question),
            "requestedSchema": {"type": "object", "properties": {"answer": prop}, "required": ["answer"]}})
        if not isinstance(r, dict):
            return NO_USER
        if r.get("action") == "accept":
            return str((r.get("content") or {}).get("answer", ""))
        return f"(the user {'declined' if r.get('action') == 'decline' else 'cancelled'} the question)"

    def confirm(self, title, body):
        return False                                  # only run_python confirms, and it is not served

    def progress(self, msg, key=None):
        _log(str(msg))
        self.s.progress(str(msg))

    def notice(self, kind, text):
        self.progress(text)


# ---------------------------------------------------------------------------------------------- server
class MCPServer:
    def __init__(self, cfg, wire: Wire, tool_descriptions: str = "full", attach_images: str = "viewers"):
        self.cfg = cfg
        self.wire = wire
        self.tool_descriptions = "compact" if str(tool_descriptions).lower() == "compact" else "full"
        self.attach_images = str(attach_images or "viewers").lower()
        self.ui = MCPUI(self)
        self.client_caps: Dict[str, Any] = {}
        self.client_info: Dict[str, Any] = {}
        self._ids = itertools.count(1)
        self._pending: Dict[Any, Any] = {}           # our request id -> (event, slot)
        self._lock = threading.Lock()
        self._jobs: "queue.Queue" = queue.Queue()
        self._cancelled: set = set()
        self._current = None                          # id of the tool call being executed
        self._token = None                            # its progress token
        self._progress_n = 0
        self._tb = None
        self._tb_error: Optional[str] = None
        self._tb_ready = threading.Event()
        threading.Thread(target=self._build_toolbox, name="calphad-mcp-build", daemon=True).start()
        threading.Thread(target=self._worker, name="calphad-mcp-worker", daemon=True).start()

    # ------------------------------------------------------------------ toolbox
    def _build_toolbox(self):
        try:
            from .agent.tools import ToolBox
            a = self.cfg.agent
            tb = ToolBox(a.workdir, a.output_dir, self.ui, "off", a.max_tool_output_chars,
                         llm_cfg=None, interactive=False, diagram_style=a.diagram_style,
                         save_notebooks=getattr(a, "save_notebooks", True), hpc=getattr(self.cfg, "hpc", None),
                         merge_cfg=getattr(self.cfg, "merge", None))
            try:
                from .structure_id import configure as configure_structure_id
                configure_structure_id(self.cfg)
            except Exception as e:  # noqa: BLE001 - labels fall back to defaults
                _log(f"structure_id configuration skipped: {e}")
            tb.tools.pop("run_python", None)          # forced off above; never served, whatever happens
            self._tb = tb
            _log(f"{len(self.visible_tools())} tools ready; workdir {tb.workdir}, outputs {tb.out}")
        except Exception:  # noqa: BLE001 - reported on every call instead of killing the server
            self._tb_error = traceback.format_exc(limit=6)
            _log("toolbox failed to load:\n" + self._tb_error)
        finally:
            self._tb_ready.set()

    def toolbox(self):
        self._tb_ready.wait()
        return self._tb

    def visible_tools(self) -> List[str]:
        tb = self._tb
        if tb is None:
            return []
        hidden = set(HIDDEN_TOOLS) | ({"tool_help"} if self.tool_descriptions == "full" else set())
        return [n for n in tb.tools if n not in hidden]

    def guide(self, with_files: bool = True) -> str:
        tb = self.toolbox()
        a = self.cfg.agent
        if tb is None:
            return guide_text(str(Path(a.workdir).resolve()), a.output_dir, self.tool_descriptions == "compact")
        files = None
        if with_files:
            from .agent.session import _list_data_files
            files = _list_data_files(tb.workdir)
        return guide_text(str(tb.workdir), tb.rel(str(tb.out)), self.tool_descriptions == "compact", files)

    def tool_list(self) -> List[Dict[str, Any]]:
        tb = self.toolbox()
        out = [{"name": GUIDE_TOOL,
                "description": "Conventions and step-by-step workflows for the CALPHAD tools of this server "
                               "(units, file resolution, merging, ESPEI, digitizing figures, editing plots), plus "
                               "the working directory and the data files in it. Read it once before the first "
                               "CALPHAD task of a session.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}}]
        if tb is None:
            return out
        specs = tb.specs() if self.tool_descriptions == "full" else tb.model_specs("compact")
        keep = set(self.visible_tools())
        for sp in specs:
            if sp.name in keep:
                schema = dict(sp.parameters or {})
                schema.setdefault("type", "object")
                out.append({"name": sp.name, "description": sp.description, "inputSchema": schema})
        return out

    # ------------------------------------------------------------------ tool calls
    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == GUIDE_TOOL:
            return {"content": [{"type": "text", "text": self.guide()}], "isError": False}
        tb = self.toolbox()
        if tb is None:
            return _error_result(f"calphad-tools failed to load in the MCP server:\n{self._tb_error}")
        if name not in self.visible_tools():
            import difflib
            close = difflib.get_close_matches(str(name), self.visible_tools(), n=3)
            return _error_result(f"unknown tool {name!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
        if not isinstance(args, dict):
            return _error_result("arguments must be a JSON object")
        tb.interactive = self.ui.can_elicit()        # merge questions etc. only when a dialog can be shown
        text, images, err = tb.call(name, args)
        images = [str(Path(i).resolve()) for i in images if i and Path(i).is_file()]
        if images:
            text = text.rstrip("\n") + "\n\nImages (absolute paths):\n" + "\n".join(f"- {i}" for i in images)
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for img in images if self._attach(name) else []:
            block = _image_block(img, getattr(self.cfg.llm, "image_max_px", 1600))
            if block:
                content.append(block)
        return {"content": content, "isError": bool(err)}

    def _attach(self, tool: str) -> bool:
        if self.attach_images == "all":
            return True
        if self.attach_images == "none":
            return False
        from .agent.llm import VIEWER_TOOLS
        return tool in VIEWER_TOOLS

    def _worker(self):
        while True:
            rid, params = self._jobs.get()
            if rid in self._cancelled:
                self._cancelled.discard(rid)
                continue
            name = str(params.get("name", ""))
            meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
            with self._lock:
                self._current, self._token, self._progress_n = rid, meta.get("progressToken"), 0
            stop = threading.Event()
            if self._token is not None:
                threading.Thread(target=self._heartbeat, args=(name, stop), daemon=True).start()
            t0 = time.time()
            _log(f"-> {name}")
            try:
                result = self.call_tool(name, params.get("arguments") or {})
            except Exception:  # noqa: BLE001 - a broken tool must not take the server down
                result = _error_result(traceback.format_exc(limit=4))
            finally:
                stop.set()
                with self._lock:
                    self._current, self._token = None, None
                    cancelled = rid in self._cancelled
                    self._cancelled.discard(rid)
                if self._tb is not None:
                    self._tb.stop_waiting.clear()
            _log(f"<- {name} ({time.time() - t0:.1f} s{', error' if result.get('isError') else ''}"
                 f"{', cancelled' if cancelled else ''})")
            if not cancelled:                          # a cancelled request gets no response (MCP spec)
                self.wire.send({"jsonrpc": "2.0", "id": rid, "result": result})

    def _heartbeat(self, name: str, stop: threading.Event):
        t0 = time.time()
        while not stop.wait(HEARTBEAT_S):
            self.progress(f"{name} still running ({time.time() - t0:.0f} s)")

    def progress(self, message: str) -> None:
        with self._lock:
            token = self._token
            if token is None:
                return
            self._progress_n += 1
            n = self._progress_n
        self.wire.send({"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": token, "progress": n, "message": message[:500]}})

    # ------------------------------------------------------------------ requests to the client
    def request(self, method: str, params: Dict[str, Any], timeout: Optional[float] = None) -> Optional[Dict]:
        """Send a request to the client and wait for its result (None on error, timeout or cancellation)."""
        rid = f"calphad-{next(self._ids)}"
        ev, slot = threading.Event(), {}
        with self._lock:
            self._pending[rid] = (ev, slot)
        self.wire.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        ev.wait(timeout)
        with self._lock:
            self._pending.pop(rid, None)
        if "error" in slot:
            _log(f"{method} failed: {slot['error']}")
        return slot.get("result")

    def _release_pending(self):
        with self._lock:
            waits = list(self._pending.values())
        for ev, _slot in waits:
            ev.set()

    # ------------------------------------------------------------------ dispatch
    def handle(self, msg: Any) -> None:
        if isinstance(msg, list):                      # JSON-RPC batch (2025-03-26)
            for m in msg:
                self.handle(m)
            return
        if not isinstance(msg, dict):
            self._error(None, -32600, "invalid request")
            return
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method is None:                             # a response to one of our requests
            with self._lock:
                ev_slot = self._pending.get(rid)
            if ev_slot:
                ev, slot = ev_slot
                if "result" in msg:
                    slot["result"] = msg["result"]
                else:
                    slot["error"] = msg.get("error")
                ev.set()
            return
        if method == "initialize":
            self.client_caps = params.get("capabilities") or {}
            self.client_info = params.get("clientInfo") or {}
            asked = params.get("protocolVersion")
            _log(f"client {self.client_info.get('name', '?')} {self.client_info.get('version', '')}, protocol {asked}, "
                 f"elicitation {'yes' if self.ui.can_elicit() else 'no'}")
            self._reply(rid, {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                              "capabilities": {"tools": {"listChanged": False}},
                              "serverInfo": {"name": "calphad-tools", "title": "CALPHAD tools (pycalphad, ESPEI)",
                                             "version": _version()},
                              "instructions": self.guide(with_files=False)})
        elif method == "ping":
            self._reply(rid, {})
        elif method == "tools/list":
            self._reply(rid, {"tools": self.tool_list()})
        elif method == "tools/call":
            if rid is None:
                return
            if not isinstance(params.get("name"), str):
                self._error(rid, -32602, "tools/call needs a tool name")
                return
            self._jobs.put((rid, params))
        elif method == "notifications/cancelled":
            target = params.get("requestId")
            with self._lock:
                self._cancelled.add(target)
                running = target == self._current
            if running:
                if self._tb is not None:
                    self._tb.stop_waiting.set()     # ends a wait_job; other tools finish, their result is dropped
                self._release_pending()
        elif rid is not None and not str(method).startswith("notifications/"):
            if method in ("resources/list", "resources/templates/list", "prompts/list"):
                key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates",
                       "prompts/list": "prompts"}[method]
                self._reply(rid, {key: []})
            else:
                self._error(rid, -32601, f"method not found: {method}")

    def _reply(self, rid, result):
        if rid is not None:
            self.wire.send({"jsonrpc": "2.0", "id": rid, "result": result})

    def _error(self, rid, code, message):
        self.wire.send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})

    def serve(self) -> None:
        for raw in self.wire.messages():
            try:
                msg = json.loads(raw)
            except ValueError:
                self._error(None, -32700, "parse error")
                continue
            try:
                self.handle(msg)
            except Exception:  # noqa: BLE001 - keep serving
                _log("handler error:\n" + traceback.format_exc(limit=4))
                if isinstance(msg, dict) and msg.get("id") is not None and msg.get("method"):
                    self._error(msg["id"], -32603, "internal error")
        _log("client closed the connection")


def _error_result(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _image_block(path: str, max_px: int) -> Optional[Dict[str, Any]]:
    try:
        from .agent.llm import _data_url
        url = _data_url(path, max_px)
        head, _, data = url.partition(",")
        return {"type": "image", "data": data, "mimeType": head[5:].split(";")[0] or "image/png"}
    except Exception as e:  # noqa: BLE001 - the path is in the text anyway
        _log(f"image {path} not attached: {e}")
        return None


# ---------------------------------------------------------------------------------------------- entry point
def build_config(args):
    """Settings from the config file, except where files go: the config's workdir / output_dir are written for
    start-agent.sh (relative to the install folder), so here they come from the flags, defaulting to the directory
    the client starts the server in (its project) and calphad_outputs inside it."""
    from .agent.config import load_config
    cfg = load_config(args.config)
    cfg.agent.workdir = os.path.expanduser(args.workdir or os.environ.get("CALPHAD_WORKDIR") or os.getcwd())
    cfg.agent.output_dir = args.output_dir or "calphad_outputs"
    cfg.agent.python_tool = "off"                     # never arbitrary code in the server (see the module doc)
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="calphad-mcp",
        description="Serve the calphad-agent tools (pycalphad, ESPEI, TDB compare/merge, diagram decks, ...) to an "
                    "MCP client such as Claude Code or Codex, over stdio.",
        epilog="Claude Code:  claude mcp add calphad -- /path/to/calphad-mcp --workdir /path/to/data\n"
               "Codex:        codex mcp add calphad -- /path/to/calphad-mcp --workdir /path/to/data",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config TOML (default ./calphad_agent.toml, plus ~/.config/calphad_tools/config.toml)")
    ap.add_argument("--workdir", help="folder with your TDBs and data (default: env CALPHAD_WORKDIR, else the "
                                      "directory the client starts the server in; the config's workdir is not used)")
    ap.add_argument("--output-dir", help="where plots, decks and datasets go, relative to the workdir "
                                         "(default calphad_outputs)")
    ap.add_argument("--tool-descriptions", choices=["full", "compact"], default="full",
                    help="full: every tool with its full schema (default; clients such as Claude Code load MCP tools "
                         "on demand); compact: about 15 core tools in full, the rest one line + tool_help")
    ap.add_argument("--attach-images", choices=["viewers", "all", "none"],
                    help="images sent back as image content: viewers = view_image / digitizer previews (default, "
                         "from [llm] attach_images), all = every plot too, none = paths only")
    ap.add_argument("--version", action="version", version=f"calphad-mcp {_version()}")
    args = ap.parse_args(argv)

    if "PYTHONHASHSEED" not in os.environ and os.name == "posix":
        # fixed hash seed, as in start-agent.sh: pycalphad's mapping traces the same lines on every run.  It only
        # takes effect at interpreter start, so restart once (same process id: the client sees no difference).
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "calphad_tools.mcp_server",
                                  *(sys.argv[1:] if argv is None else argv)])
    os.environ.setdefault("MPLBACKEND", "Agg")      # never a GUI backend in a server
    wire = Wire()                                    # from here on, stdout is the protocol only
    try:
        cfg = build_config(args)
    except Exception as e:  # noqa: BLE001
        _log(f"configuration error: {e}")
        return 2
    server = MCPServer(cfg, wire, tool_descriptions=args.tool_descriptions,
                       attach_images=args.attach_images or getattr(cfg.llm, "attach_images", "viewers"))
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
