"""A tiny scripted OpenAI-compatible /v1/chat/completions server for offline testing.

It replays tool calls from a JSON script: each time the conversation ends with a
user message (or tool results), the next scripted step is returned.  A step is
either {"tool_calls": [{"name": ..., "arguments": {...}}, ...]} or {"text": "..."}.
Every request body is appended to <log> so tests can inspect message formats.

    python tests/fake_openai_server.py script.json --port 8799 --log requests.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(steps, log_path):
    lock = threading.Lock()
    state = {"i": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            with lock:
                with open(log_path, "a") as f:
                    f.write(json.dumps(body) + "\n")
                step = steps[state["i"]] if state["i"] < len(steps) else {"text": "(script finished)"}
                state["i"] += 1
            if self.path.split("?")[0].endswith("/messages"):
                return self._anthropic(body, step)
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            calls = [{"id": f"call_{state['i']}_{k}", "type": "function",
                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                     for k, c in enumerate(step.get("tool_calls", []))]
            text = step.get("text")
            if step.get("echo_last_tool"):
                last = [m for m in body["messages"] if m.get("role") == "tool"]
                text = (text or "") + (last[-1]["content"][:300] if last else "")
            finish = "tool_calls" if calls else "stop"
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(delta, fr=None):
                    chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": body.get("model"), "choices": [{"index": 0, "delta": delta, "finish_reason": fr}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                send({"role": "assistant"})
                if text:
                    for i in range(0, len(text), 20):
                        send({"content": text[i:i + 20]})
                for k, c in enumerate(calls):
                    args = c["function"]["arguments"]
                    send({"tool_calls": [{"index": k, "id": c["id"], "type": "function",
                                          "function": {"name": c["function"]["name"], "arguments": ""}}]})
                    for i in range(0, len(args), 15):
                        send({"tool_calls": [{"index": k, "function": {"arguments": args[i:i + 15]}}]})
                send({}, finish)
                self.wfile.write(b"data: [DONE]\n\n")
                return
            resp = {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": text, "tool_calls": calls or None}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ---- Anthropic Messages API (for testing the Claude backend) ----
        def _anthropic(self, body, step):
            text = step.get("text")
            if step.get("echo_last_tool"):
                results = [b for m in body["messages"] if m["role"] == "user" and isinstance(m["content"], list)
                           for b in m["content"] if b.get("type") == "tool_result"]
                if results:
                    c = results[-1]["content"]
                    text = (text or "") + (c[0]["text"] if isinstance(c, list) else str(c))[:300]
            blocks = [{"type": "thinking", "thinking": "", "signature": "sig-abc"}]
            if text:
                blocks.append({"type": "text", "text": text})
            for k, c in enumerate(step.get("tool_calls", [])):
                blocks.append({"type": "tool_use", "id": f"toolu_{state['i']}_{k}", "name": c["name"],
                               "input": c["arguments"]})
            stop = "tool_use" if step.get("tool_calls") else "end_turn"
            usage = {"input_tokens": 1, "output_tokens": 1}
            msg = {"id": "msg_x", "type": "message", "role": "assistant", "model": body.get("model"),
                   "content": blocks, "stop_reason": stop, "stop_sequence": None, "usage": usage}
            if not body.get("stream"):
                data = json.dumps(msg).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def ev(name, data):
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            ev("message_start", {"type": "message_start", "message": {**msg, "content": [], "stop_reason": None}})
            for i, b in enumerate(blocks):
                if b["type"] == "thinking":
                    ev("content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "signature_delta", "signature": b["signature"]}})
                elif b["type"] == "text":
                    ev("content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "text", "text": ""}})
                    for j in range(0, len(b["text"]), 20):
                        ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "text_delta", "text": b["text"][j:j + 20]}})
                else:
                    ev("content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"],
                                                                 "input": {}}})
                    js = json.dumps(b["input"])
                    for j in range(0, len(js), 15):
                        ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "input_json_delta", "partial_json": js[j:j + 15]}})
                ev("content_block_stop", {"type": "content_block_stop", "index": i})
            ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                 "usage": {"output_tokens": 1}})
            ev("message_stop", {"type": "message_stop"})
    return H


def serve(steps, port=8799, log_path="fake_requests.jsonl"):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(steps, log_path))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--log", default="fake_requests.jsonl")
    a = ap.parse_args()
    s = serve(json.load(open(a.script)), a.port, a.log)
    print(f"fake OpenAI server on http://127.0.0.1:{a.port}/v1")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.shutdown()
