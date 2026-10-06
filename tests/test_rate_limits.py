"""Rate limits: resend the same request after the server's wait, learn a pace ("auto"), give up clearly; the
conversation stays append-only and keeps one prompt_cache_key (cache-friendly)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from calphad_tools.agent.config import LLMConfig
from calphad_tools.agent.llm import RateGuard, RateLimitGiveUp, _retry_after


class Fake429(Exception):
    status_code = 429

    def __init__(self, msg="Requests have exceeded token rate limit. Please retry after 7 seconds.", headers=None):
        super().__init__(msg)
        self.response = type("R", (), {"headers": headers or {}})()


def guard(**kw):
    """A guard with a fake clock: sleeping advances it (no real waiting in the tests)."""
    g = RateGuard(LLMConfig(**{"rate_limit_mode": "polite", **kw}))   # these tests: the polite pacing
    g.slept, g.said, g.now = [], [], 1000.0

    def sleep(s):
        g.slept.append(s)
        g.now += s
    g.sleep, g.clock = sleep, (lambda: g.now)
    g.notify = lambda m, key=None: g.said.append(m)
    return g


def test_retry_after_parsing():
    assert _retry_after(Fake429(headers={"retry-after-ms": "2500"})) == 2.5
    assert _retry_after(Fake429(headers={"retry-after": "12"})) == 12
    assert _retry_after(Fake429()) == 7
    assert _retry_after(Fake429("Please retry after 800 milliseconds")) == 0.8


def test_waits_then_resends_and_learns_pace():
    g = guard()                                              # tokens_per_minute = "auto"
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise Fake429()
        return "ok"
    g.sent = [(g.now, 30000)]                                # 30k tokens sent in the last minute
    assert g.run(fn, est=20000) == "ok" and len(calls) == 3
    assert g.slept[0] == 7.0                                 # the server's retry-after
    # first limit: pace at 80 % of the last minute's 30k + 20k; the scripted second limit, even after pacing,
    # tightens it again (80 % of the 20k sent in that minute)
    assert any("pacing requests to about 40000" in s for s in g.said)
    assert g.learned_tpm == 16000 and g.budget() == 16000
    assert any("pacing to 40000 tokens/minute: waiting" in s for s in g.said)   # it did pace before retrying
    assert any("waiting 7 s, then sending the same request again" in s for s in g.said)


def test_pacing_waits_for_the_window():
    g = guard(tokens_per_minute=10000)
    g.sent = [(g.now - 30, 9000)]
    g.run(lambda: "ok", est=5000)                            # 9000 + 5000 > 10000: wait ~30 s for the window
    assert g.slept and 29 <= g.slept[0] <= 31


def test_gives_up_clearly():
    g = guard(max_retries=3, rate_limit_max_wait_s=15)

    def always():
        raise Fake429()
    with pytest.raises(RateLimitGiveUp, match="rate limit"):
        g.run(always, est=100)
    assert sum(g.slept) <= 15
    with pytest.raises(ValueError):                          # other errors are not retried
        guard().run(lambda: (_ for _ in ()).throw(ValueError("x")))


def _serve(handler_state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            handler_state["bodies"].append(body)
            if len(handler_state["bodies"]) <= handler_state["fail"]:
                data = json.dumps({"error": {"code": "429", "message": "Requests to gpt-5.6-sol have exceeded "
                                   "token rate limit. Please retry after 1 second."}}).encode()
                self.send_response(429)
                self.send_header("retry-after", "1")
            else:
                data = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": "pong"}}],
                                   "usage": {"prompt_tokens": 1500, "completion_tokens": 2, "total_tokens": 1502,
                                             "prompt_tokens_details": {"cached_tokens": 1408}}}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_chat_backend_resends_identical_request_with_cache_key():
    from calphad_tools.agent.llm import OpenAIChatBackend
    state = {"bodies": [], "fail": 1}
    srv = _serve(state)
    try:
        cfg = LLMConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", model="m",
                        prompt_cache_key="calphad-agent-abc")
        b = OpenAIChatBackend(cfg, "system prompt", [])
        b.guard.sleep = lambda s: None
        said = []
        b.guard.notify = lambda m, key=None: said.append(m)
        b.add_user("ping")
        res = b.step()
    finally:
        srv.shutdown()
    assert res.text == "pong" and res.usage["cached"] == 1408
    first, second = state["bodies"]
    assert first == second and first["prompt_cache_key"] == "calphad-agent-abc"
    assert any("rate limit (429) - resending within" in s for s in said)    # default: aggressive


def test_images_shown_once_and_plots_not_attached(tmp_path):
    """view_image-style images reach the model once and are then replaced by a note; plots of other tools are not
    attached at all; the output reservation starts small (Azure counts it against the TPM limit)."""
    from PIL import Image
    from calphad_tools.agent.llm import OpenAIChatBackend, ToolCall, ToolOutput
    img = tmp_path / "fig.png"
    Image.new("RGB", (40, 30), "white").save(img)
    state = {"bodies": [], "fail": 0}
    srv = _serve(state)
    try:
        cfg = LLMConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", model="m",
                        prompt_cache_key="")
        b = OpenAIChatBackend(cfg, "system", [])
        b.add_user("look")
        b.step()
        b.add_tool_results([ToolOutput(ToolCall("c1", "view_image", {}), "ok", [str(img)], False),
                            ToolOutput(ToolCall("c2", "map_phase_diagram", {}), "done", [str(img)], False)])
        b.step()
        b.add_user("again")
        b.step()
    finally:
        srv.shutdown()
    def images(body):
        return [p for m in body["messages"] if isinstance(m.get("content"), list) for p in m["content"]
                if p.get("type") == "image_url"]
    assert len(images(state["bodies"][1])) == 1                  # the viewed image, once (not the plot)
    assert images(state["bodies"][2]) == []                      # afterwards: only a note with the path
    assert "image already viewed" in json.dumps(state["bodies"][2]) and "fig.png" in json.dumps(state["bodies"][2])
    assert all(bd["max_completion_tokens"] == 6000 for bd in state["bodies"])


def test_stream_cut_mid_answer_is_requested_again():
    """Like Codex's 'stream closed before response.completed': the stream ends without a finish; the same
    request is sent again and the full answer used."""
    import socket
    from calphad_tools.agent.llm import OpenAIChatBackend
    bodies = []

    def chunk(obj):
        return ("data: " + json.dumps(obj) + "\n\n").encode()

    def delta(content=None, finish=None):
        return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": ({"content": content} if content else {}),
                             "finish_reason": finish}]}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(chunk(delta("par")))
            if len(bodies) == 1:                                 # cut: no finish, no [DONE]
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.wfile.write(chunk(delta("tial answer")) + chunk(delta(finish="stop")) + b"data: [DONE]\n\n")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = LLMConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", model="m",
                        prompt_cache_key="")
        b = OpenAIChatBackend(cfg, "system", [])
        b.guard.sleep = lambda s: None
        said = []
        b.guard.notify = lambda m, key=None: said.append(m)
        b.add_user("hi")
        res = b.step(on_text=lambda t: None)
    finally:
        srv.shutdown()
    assert res.text == "partial answer" and res.stop_reason == "stop"
    assert len(bodies) == 2 and bodies[0]["messages"] == bodies[1]["messages"]      # the same request again
    assert any("connection dropped before the answer was complete" in s for s in said)


def test_transient_errors_retried_but_not_forever():
    class Conn(Exception):
        pass
    Conn.__name__ = "APIConnectionError"
    g = guard(transient_retries=2)
    n = []

    def flaky():
        n.append(1)
        if len(n) < 3:
            raise Conn("connection reset")
        return "ok"
    assert g.run(flaky) == "ok" and g.slept == [2.0, 4.0]
    g2 = guard(transient_retries=2)
    with pytest.raises(Conn):
        g2.run(lambda: (_ for _ in ()).throw(Conn("down")))
    assert len(g2.slept) == 2


def _serve_responses(replies, bodies):
    """A tiny /v1/responses endpoint: returns the queued replies in order and records the request bodies."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            bodies.append(_json.loads(self.rfile.read(n)))
            data = _json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _response(status, output, reason=None):
    r = {"id": "resp_1", "object": "response", "created_at": 0, "model": "m", "status": status,
         "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
         "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                   "input_tokens_details": {"cached_tokens": 64}, "output_tokens_details": {"reasoning_tokens": 5}}}
    if reason:
        r["incomplete_details"] = {"reason": reason}
    return r


def test_responses_backend_asks_again_when_cut_off():
    """api = "responses": an answer stopped by max_output_tokens is dropped and requested again with twice the
    output room (as the chat backend does); the cut-off output is not kept in the conversation."""
    from calphad_tools.agent.llm import OpenAIResponsesBackend
    cut = _response("incomplete", [{"type": "reasoning", "id": "rs_1", "summary": [],
                                     "encrypted_content": "enc-cut"}], reason="max_output_tokens")
    done = _response("completed", [
        {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc-ok"},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "pong", "annotations": []}]}])
    bodies = []
    srv = _serve_responses([cut, done], bodies)
    try:
        cfg = LLMConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", model="m", api="responses",
                        reasoning_effort="medium", initial_output_tokens=6000, max_output_tokens=16000)
        b = OpenAIResponsesBackend(cfg, "system prompt", [])
        said = []
        b.guard.notify = lambda m, key=None: said.append(m)
        b.add_user("ping")
        res = b.step()
    finally:
        srv.shutdown()
    assert res.text == "pong" and res.usage["cached"] == 64
    assert [x["max_output_tokens"] for x in bodies] == [6000, 12000]
    assert bodies[0]["store"] is False and bodies[0]["include"] == ["reasoning.encrypted_content"]
    assert any("answer cut off" in s for s in said)
    kept = [i.get("encrypted_content") for i in b.input if i.get("type") == "reasoning"]
    assert kept == ["enc-ok"]                               # the cut-off answer was not kept
    assert b.cur_max == 6000                                # back to the small reservation for the next step



def test_aggressive_retries_fast_without_self_pacing():
    """Default ("aggressive"): a shared quota goes to whoever asks first - every 429 is resent within
    rate_limit_retry_cap_s (Retry-After capped too), nothing is learned or paced, the retries share one status
    line (same key, a running count), and only the total wait ends the attempts."""
    g = guard(rate_limit_mode="aggressive")
    keys = []
    g.notify = lambda m, key=None: (g.said.append(m), keys.append(key))
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 12:                                  # more than max_retries (8) refusals
            raise Fake429()                                  # with "retry after 7 seconds"
        return "ok"
    g.sent = [(g.now, 30000)]
    g.header_remaining, g.header_reset, g.header_time = 10, 50.0, g.now   # "almost nothing left this minute"
    assert g.run(fn, est=20000) == "ok" and len(calls) == 12
    assert len(g.slept) == 11 and max(g.slept) <= 5.0 and min(g.slept) > 0     # capped, no 50 s reset wait
    assert g.learned_tpm is None and g.budget() == 0                           # no self-pacing
    assert len(set(keys)) == 1 and keys[0]                                     # one replaceable status line
    assert "retry ×11" in g.said[-1] and "resending within 5 s" in g.said[-1]
    # the next request's retries start a new line
    calls.clear()
    g.run(fn, est=100)
    assert len(set(keys)) == 2


def test_aggressive_still_gives_up_after_the_total_wait():
    g = guard(rate_limit_mode="aggressive", rate_limit_max_wait_s=20.0)
    g.notify = lambda m, key=None: g.said.append(m)

    def fn():
        raise Fake429()
    import pytest as _pytest
    from calphad_tools.agent.llm import RateLimitGiveUp
    with _pytest.raises(RateLimitGiveUp):
        g.run(fn, est=100)
    assert sum(g.slept) <= 20.0 and len(g.slept) >= 4


def test_cli_progress_rewrites_repeated_key(capsys, monkeypatch):
    from calphad_tools.agent.cli import TerminalUI
    import sys as _sys
    ui = TerminalUI.__new__(TerminalUI)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True, raising=False)
    ui.progress("rate limit (429) - retry ×1", key="rate_limit:1")
    ui.progress("rate limit (429) - retry ×2", key="rate_limit:1")
    ui.progress("something else")
    out = capsys.readouterr().out
    assert out.count("\x1b[1A\x1b[2K") == 1 and "retry ×2" in out and "something else" in out
