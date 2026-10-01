"""LLM backends with a common tool-calling interface.

The agent loop runs *inside this Python process*: the model only ever returns
text and tool-call requests; every tool is executed locally by the script, with
the script's own permissions / sandbox.  Backends:

* OpenAIChatBackend       - any OpenAI-compatible /chat/completions endpoint (default)
* OpenAIResponsesBackend  - OpenAI-compatible /responses endpoint (keeps reasoning items)
* AnthropicBackend        - Claude via the Anthropic Messages API
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import LLMConfig


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]       # JSON schema (object)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""
    parse_error: str = ""


@dataclass
class ToolOutput:
    call: ToolCall
    text: str
    images: List[str] = field(default_factory=list)
    is_error: bool = False


@dataclass
class StepResult:
    text: str
    calls: List[ToolCall]
    stop_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


def _data_url(path: str, max_px: int = 0) -> str:
    """Base64 data URL of an image, downscaled to max_px on its long side first (fewer image tokens).  The same
    file always gives the same bytes, so images already in the conversation stay part of the cached prefix."""
    data = Path(path).read_bytes()
    mt = mimetypes.guess_type(path)[0] or "image/png"
    if max_px:
        try:
            import io
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            if max(im.size) > max_px:
                im.thumbnail((max_px, max_px), Image.LANCZOS)
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "PNG", optimize=True)
                data, mt = buf.getvalue(), "image/png"
        except Exception:  # noqa: BLE001 - send the original
            pass
    return f"data:{mt};base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# rate limits: resend the SAME request after the wait the server asks for (keeps the cached prefix), optional
# client-side tokens-per-minute pacing
# ---------------------------------------------------------------------------
class RateLimitGiveUp(RuntimeError):
    """The endpoint kept refusing with 429 for longer than rate_limit_max_wait_s."""


def _is_rate_limit(e: Exception) -> bool:
    return getattr(e, "status_code", None) == 429 or type(e).__name__ == "RateLimitError"


_TRANSIENT_NAMES = {"APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailableError",
                    "RemoteProtocolError", "ReadError", "WriteError", "ConnectError", "ReadTimeout", "ConnectTimeout",
                    "PoolTimeout", "ChunkedEncodingError", "IncompleteRead", "ConnectionResetError",
                    "OverloadedError", "APIStatusError5xx"}


def _is_transient(e: Exception) -> bool:
    """A dropped connection, a timeout or a server-side 5xx/overloaded error: worth sending again."""
    if _is_rate_limit(e):
        return False
    code = getattr(e, "status_code", None)
    if isinstance(code, int) and (code >= 500 or code == 408):
        return True
    for c in type(e).__mro__:
        if c.__name__ in _TRANSIENT_NAMES:
            return True
    cause = e.__cause__ or e.__context__
    return bool(cause is not None and cause is not e and type(cause).__name__ in _TRANSIENT_NAMES)


def _retry_after(e: Exception) -> Optional[float]:
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        if headers.get("retry-after-ms"):
            return float(headers["retry-after-ms"]) / 1000.0
        if headers.get("retry-after"):
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        pass
    m = re.search(r"retry after (\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|seconds?)", str(e), re.I)
    if m:
        v = float(m.group(1))
        return v / 1000.0 if m.group(2).lower().startswith("m") else v
    return None


class RateGuard:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.notify: Callable[[str], None] = lambda msg: None
        self.sent: List = []                     # (time, estimated tokens) of the last minute
        self.waited_total = 0.0
        self.learned_tpm: Optional[int] = None   # "auto": budget learned from the first 429
        self.header_limit: Optional[int] = None   # x-ratelimit-limit-tokens (the deployment's real TPM)
        self.header_remaining: Optional[int] = None
        self.header_reset: Optional[float] = None
        self.header_time = 0.0
        import time as _t
        self.sleep, self.clock = _t.sleep, _t.time          # replaceable in tests

    def observe(self, headers) -> None:
        """Azure/OpenAI rate-limit headers of a response: the real limit and what is left this minute."""
        try:
            h = {k.lower(): v for k, v in dict(headers or {}).items()}
            if h.get("x-ratelimit-limit-tokens"):
                self.header_limit = int(float(h["x-ratelimit-limit-tokens"]))
            if h.get("x-ratelimit-remaining-tokens"):
                self.header_remaining = int(float(h["x-ratelimit-remaining-tokens"]))
                self.header_time = self.clock()
            if h.get("x-ratelimit-reset-tokens"):
                v = str(h["x-ratelimit-reset-tokens"]).strip().lower()
                self.header_reset = float(v[:-2]) / 1000 if v.endswith("ms") else float(v.rstrip("s") or 0)
        except (TypeError, ValueError):
            pass

    def budget(self) -> int:
        v = self.cfg.tokens_per_minute
        if isinstance(v, str):
            if v.strip().lower() == "auto":
                if self.header_limit:                 # the endpoint told us its limit: stay just below it
                    return int(0.9 * min(self.header_limit, self.learned_tpm or self.header_limit))
                return int(self.learned_tpm or 0)
            v = int(v) if v.strip().isdigit() else 0
        return int(v or 0)

    def _learn(self, est: int) -> None:
        """First rate limit in "auto" mode: pace to ~80 % of what was sent in the last minute from now on."""
        if not (isinstance(self.cfg.tokens_per_minute, str) and self.cfg.tokens_per_minute.strip().lower() == "auto"):
            return
        now = self.clock()
        used = sum(n for t, n in self.sent if now - t < 60) + est
        new = max(4000, int(0.8 * used))
        if self.learned_tpm is None or new < self.learned_tpm:
            self.learned_tpm = new
            self.notify(f"pacing requests to about {new} tokens/minute from now on (learned from the rate limit; "
                        f"set tokens_per_minute in calphad_agent.toml to fix a value)")

    def _pace(self, est: int) -> None:
        # the endpoint said how much is left this minute: wait for its reset rather than be refused
        if self.header_remaining is not None and self.header_reset and est > self.header_remaining:
            age = self.clock() - self.header_time
            wait = self.header_reset - age
            if 0 < wait < 90:
                self.notify(f"{self.header_remaining:,} of the deployment's tokens/minute left and this request "
                            f"needs ~{est:,}: waiting {wait:.0f} s for the limit to reset")
                self.sleep(wait + 0.5)
                self.waited_total += wait
                self.header_remaining = None
        tpm = self.budget()
        if tpm <= 0:
            return
        est = min(est, tpm)
        while True:
            now = self.clock()
            self.sent = [(t, n) for t, n in self.sent if now - t < 60]
            if not self.sent or sum(n for _, n in self.sent) + est <= tpm:
                return
            wait = max(0.5, 60 - (now - self.sent[0][0]) + 0.2)
            self.notify(f"pacing to {tpm} tokens/minute: waiting {wait:.0f} s")
            self.sleep(wait)
            self.waited_total += wait

    def transient_wait(self, e: Exception, n: int, what: str = "connection problem") -> bool:
        """Wait before retry number n of a transient failure; False when the retries are used up."""
        if n > int(getattr(self.cfg, "transient_retries", 4)):
            return False
        wait = min(30.0, 2.0 * 2 ** (n - 1))
        self.notify(f"{what} ({type(e).__name__}: {str(e)[:120]}) - sending the same request again in {wait:.0f} s "
                    f"({n}/{self.cfg.transient_retries})")
        self.sleep(wait)
        self.waited_total += wait
        return True

    def run(self, fn: Callable[[], Any], est: int = 0) -> Any:
        waited, attempt, flaky = 0.0, 0, 0
        while True:
            self._pace(est)
            try:
                r = fn()
                self.sent.append((self.clock(), est))
                return r
            except Exception as e:  # noqa: BLE001
                if _is_transient(e):
                    flaky += 1
                    if self.transient_wait(e, flaky):
                        continue
                    raise
                if not _is_rate_limit(e):
                    raise
                self._learn(est)
                attempt += 1
                wait = _retry_after(e)
                wait = max(1.0, min(90.0, wait if wait is not None else 5.0 * 2 ** (attempt - 1)))
                if attempt > int(self.cfg.max_retries) or waited + wait > float(self.cfg.rate_limit_max_wait_s):
                    raise RateLimitGiveUp(
                        f"The model endpoint's rate limit (e.g. tokens per minute) was still exceeded after waiting "
                        f"{waited:.0f} s. Wait a minute and ask again, raise the deployment's TPM quota, or set "
                        f"tokens_per_minute in calphad_agent.toml to pace requests. ({str(e)[:300]})") from e
                self.notify(f"rate limit reached - waiting {wait:.0f} s, then sending the same request again "
                            f"(attempt {attempt}/{self.cfg.max_retries})")
                self.sleep(wait)
                waited += wait
                self.waited_total += wait


def _est_tokens(obj: Any) -> int:
    """Rough request size in tokens (images counted as ~1100 tokens, not by their base64 length)."""
    n_img = 0

    def strip(v):
        nonlocal n_img
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items()}
        if isinstance(v, list):
            return [strip(x) for x in v]
        if isinstance(v, str) and v.startswith("data:image"):
            n_img += 1
            return ""
        return v
    return len(json.dumps(strip(obj), default=str)) // 4 + 1100 * n_img


def _parse_args(raw: str) -> (Dict, str):
    if not raw:
        return {}, ""
    try:
        v = json.loads(raw)
        return (v if isinstance(v, dict) else {"value": v}), ""
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments: {e}"


def _http_client(cfg: LLMConfig, module):
    """Custom HTTP client only when SSL settings need changing (institutional CAs)."""
    if cfg.verify_ssl and not cfg.ca_bundle:
        return None
    verify = cfg.ca_bundle or cfg.verify_ssl
    for name in ("DefaultHttpx2Client", "DefaultHttpxClient"):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls(verify=verify)
    return None


class Backend:
    def __init__(self, cfg: LLMConfig, system: str, tools: List[ToolSpec]):
        self.cfg, self.system, self.tools = cfg, system, tools

    def add_user(self, text: str, images: Optional[List[str]] = None) -> None: ...
    def step(self, on_text: Optional[Callable[[str], None]] = None) -> StepResult: ...
    def add_tool_results(self, results: List[ToolOutput]) -> None: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible: chat completions
# ---------------------------------------------------------------------------
# tools whose images the model is meant to LOOK at (previews to read, overlays to check); images of the other
# tools (plots, diagrams, decks) are for the user - the model sees them only when it calls view_image
VIEWER_TOOLS = {"view_image", "view_pdf_page", "digitize_setup", "digitize_points", "digitize_curve",
                "digitize_fix", "digitized_to_zpf", "describe_figure"}


def shows_images(tool: str, cfg) -> bool:
    mode = str(getattr(cfg, "attach_images", "viewers") or "viewers").lower()
    return mode == "all" or (mode == "viewers" and tool in VIEWER_TOOLS)


def _chat_usage(u) -> Dict[str, int]:
    if not u:
        return {}
    det = getattr(u, "prompt_tokens_details", None)
    return {"input": getattr(u, "prompt_tokens", 0) or 0, "output": getattr(u, "completion_tokens", 0) or 0,
            "cached": (getattr(det, "cached_tokens", 0) or 0) if det else 0}


class OpenAIChatBackend(Backend):
    def __init__(self, cfg, system, tools):
        super().__init__(cfg, system, tools)
        import openai
        kw = dict(api_key=cfg.api_key or "EMPTY", timeout=cfg.timeout, default_headers=cfg.extra_headers or None)
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        hc = _http_client(cfg, openai)
        if hc is not None:
            kw["http_client"] = hc
        self.client = openai.OpenAI(max_retries=0, **kw)        # rate limits: RateGuard (visible waits)
        self.messages: List[Dict] = [{"role": "system", "content": system}]
        self.tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                            "parameters": t.parameters}} for t in tools]
        self._drop_params: set = set()
        self.guard = RateGuard(cfg)
        self._tools_est = _est_tokens(self.tool_defs)
        self.cur_max = self._start_max()             # raised only when an answer is cut off
        self._images: List = []                     # (message, part index, path) not yet seen by the model

    def _image(self, p):
        iu = {"url": _data_url(p, self.cfg.image_max_px)}
        if self.cfg.image_detail and self.cfg.image_detail != "auto":
            iu["detail"] = self.cfg.image_detail
        return {"type": "image_url", "image_url": iu}

    def _start_max(self) -> int:
        return max(256, min(int(self.cfg.max_output_tokens), int(getattr(self.cfg, "initial_output_tokens", 6000))))

    def _track(self, msg, paths):
        for k, part in enumerate(msg["content"]):
            if part.get("type") == "image_url":
                self._images.append((msg, k, paths.pop(0)))

    def _forget_images(self):
        """After the model has looked once, images leave the conversation (replaced by a note with the path), so
        they are not sent again with every later request."""
        for msg, k, path in self._images:
            msg["content"][k] = {"type": "text", "text": f"[image already viewed: {path} - view_image to see it again]"}
        self._images = []

    def add_user(self, text, images=None):
        if images and self.cfg.vision:
            content = [{"type": "text", "text": text}] + [self._image(p) for p in images]
            self.messages.append({"role": "user", "content": content})
            self._track(self.messages[-1], list(images))
        else:
            self.messages.append({"role": "user", "content": text})

    def _request(self, stream: bool):
        kw = dict(model=self.cfg.model, messages=self.messages, tools=self.tool_defs, stream=stream)
        if self.cfg.prompt_cache_key and "prompt_cache_key" not in self._drop_params:
            kw["prompt_cache_key"] = self.cfg.prompt_cache_key
        if stream and "stream_options" not in self._drop_params:
            kw["stream_options"] = {"include_usage": True}
        if "max_completion_tokens" not in self._drop_params:
            kw["max_completion_tokens"] = self.cur_max
        if self.cfg.reasoning_effort and "reasoning_effort" not in self._drop_params:
            kw["reasoning_effort"] = self.cfg.reasoning_effort
        if self.cfg.temperature is not None:
            kw["temperature"] = self.cfg.temperature
        raw = self.client.chat.completions.with_raw_response.create(**kw)
        self.guard.observe(raw.headers)
        return raw.parse()

    def _create(self, stream: bool):
        import openai
        est = self._tools_est + _est_tokens(self.messages) + self.cur_max    # Azure also counts max_tokens
        for _ in range(5):
            try:
                return self.guard.run(lambda: self._request(stream), est)
            except openai.BadRequestError as e:   # some gateways reject newer params: retry without them
                msg = str(e).lower()
                dropped = False
                for p in ("reasoning_effort", "max_completion_tokens", "prompt_cache_key", "stream_options"):
                    if p in msg and p not in self._drop_params:
                        self._drop_params.add(p)
                        dropped = True
                if not dropped:
                    raise
        return self.guard.run(lambda: self._request(stream), est)

    def step(self, on_text=None):
        if on_text is None:
            resp = self._create(False)
            msg = resp.choices[0].message
            text = msg.content or ""
            calls = []
            for tc in msg.tool_calls or []:
                args, err = _parse_args(tc.function.arguments)
                calls.append(ToolCall(tc.id, tc.function.name, args, tc.function.arguments, err))
            stop = resp.choices[0].finish_reason or ""
            usage = _chat_usage(resp.usage)
        else:
            n = 0
            while True:
                try:
                    text, stop, usage, acc = self._consume(on_text)
                    break
                except Exception as e:  # noqa: BLE001 - the stream was cut mid-answer: ask for the same answer again
                    n += 1
                    if not _is_transient(e) or not self.guard.transient_wait(
                            e, n, "connection dropped before the answer was complete"):
                        raise
            calls = []
            for i in sorted(acc):
                a = acc[i]
                args, err = _parse_args(a["args"])
                calls.append(ToolCall(a["id"] or f"call_{i}", a["name"], args, a["args"], err))
        return self._finish(text, calls, stop, usage, on_text)

    def _consume(self, on_text):
        """Create the streamed request and read it to the end: (text, finish reason, usage, tool-call parts)."""
        text, stop, usage = "", "", {}
        acc: Dict[int, Dict] = {}
        for chunk in self._create(True):
            if getattr(chunk, "usage", None):
                usage = _chat_usage(chunk.usage)
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            d = ch.delta
            if d and d.content:
                text += d.content
                on_text(d.content)
            for tc in (d.tool_calls or []) if d else []:
                a = acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    a["id"] = tc.id
                if tc.function and tc.function.name:
                    a["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    a["args"] += tc.function.arguments
            if ch.finish_reason:
                stop = ch.finish_reason
        if not stop:
            # the stream ended without a finish reason: it was cut, not finished
            raise ConnectionResetError("stream closed before the answer was complete")
        return text, stop, usage, acc

    def _finish(self, text, calls, stop, usage, on_text):
        cap = max(int(self.cfg.max_output_tokens), self._start_max())
        if stop == "length" and self.cur_max < cap:
            # the answer was cut off at the (deliberately small) output reservation: ask again with more room
            self.cur_max = min(cap, self.cur_max * 2)
            self.guard.notify(f"answer cut off - asking again with up to {self.cur_max:,} output tokens")
            return self.step(on_text)
        self.cur_max = self._start_max()
        self._forget_images()
        am: Dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            am["tool_calls"] = [{"id": c.id, "type": "function",
                                 "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                                for c in calls]
        self.messages.append(am)
        return StepResult(text, calls, stop, usage)

    def add_tool_results(self, results):
        images, others = [], []
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call.id, "content": r.text})
            for p in r.images:
                (images if shows_images(r.call.name, self.cfg) else others).append((r.call.name, p))
        if images and self.cfg.vision:
            content = [{"type": "text", "text": "Images produced by the tool calls above: "
                        + ", ".join(f"{n}: {Path(p).name}" for n, p in images)}]
            content += [self._image(p) for _, p in images]
            self.messages.append({"role": "user", "content": content})
            self._track(self.messages[-1], [p for _, p in images])


# ---------------------------------------------------------------------------
# OpenAI-compatible: responses API
# ---------------------------------------------------------------------------
class OpenAIResponsesBackend(Backend):
    def __init__(self, cfg, system, tools):
        super().__init__(cfg, system, tools)
        import openai
        kw = dict(api_key=cfg.api_key or "EMPTY", timeout=cfg.timeout, default_headers=cfg.extra_headers or None)
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        hc = _http_client(cfg, openai)
        if hc is not None:
            kw["http_client"] = hc
        self.client = openai.OpenAI(max_retries=0, **kw)
        self.guard = RateGuard(cfg)
        self._drop_params: set = set()
        self.input: List[Any] = []
        self._images: List = []
        self.tool_defs = [{"type": "function", "name": t.name, "description": t.description,
                           "parameters": t.parameters} for t in tools]

    def add_user(self, text, images=None):
        content = [{"type": "input_text", "text": text}]
        if images and self.cfg.vision:
            content += [self._image(p) for p in images]
        self.input.append({"role": "user", "content": content})
        self._track(self.input[-1], list(images or []))

    def _track(self, msg, paths):
        for k, part in enumerate(msg["content"]):
            if part.get("type") == "input_image" and paths:
                self._images.append((msg, k, paths.pop(0)))

    def _forget_images(self):
        for msg, k, path in self._images:
            msg["content"][k] = {"type": "input_text",
                                 "text": f"[image already viewed: {path} - view_image to see it again]"}
        self._images = []

    def _image(self, p):
        d = {"type": "input_image", "image_url": _data_url(p, self.cfg.image_max_px)}
        if self.cfg.image_detail and self.cfg.image_detail != "auto":
            d["detail"] = self.cfg.image_detail
        return d

    def step(self, on_text=None):
        kw = dict(model=self.cfg.model, instructions=self.system, input=self.input, tools=self.tool_defs,
                  max_output_tokens=min(int(self.cfg.max_output_tokens),
                                        int(getattr(self.cfg, "initial_output_tokens", 6000))), store=False)
        if self.cfg.reasoning_effort:
            kw["reasoning"] = {"effort": self.cfg.reasoning_effort}
            kw["include"] = ["reasoning.encrypted_content"]
        if self.cfg.temperature is not None:
            kw["temperature"] = self.cfg.temperature
        if self.cfg.prompt_cache_key and "prompt_cache_key" not in self._drop_params:
            kw["prompt_cache_key"] = self.cfg.prompt_cache_key
        import openai
        est = _est_tokens([self.system, self.input, kw["tools"]]) + int(kw["max_output_tokens"])

        def call():
            raw = self.client.responses.with_raw_response.create(**kw)
            self.guard.observe(raw.headers)
            return raw.parse()
        try:
            resp = self.guard.run(call, est)
        except openai.BadRequestError as e:
            if "prompt_cache_key" in str(e).lower() and "prompt_cache_key" in kw:
                self._drop_params.add("prompt_cache_key")
                kw.pop("prompt_cache_key")
                resp = self.guard.run(call, est)
            else:
                raise
        self._forget_images()
        text, calls = "", []
        for item in resp.output:
            d = item.model_dump(exclude_none=True)
            self.input.append(d)
            if item.type == "message":
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        text += c.text
            elif item.type == "function_call":
                args, err = _parse_args(item.arguments)
                calls.append(ToolCall(item.call_id, item.name, args, item.arguments, err))
        if text and on_text:
            on_text(text)
        det = getattr(resp.usage, "input_tokens_details", None) if resp.usage else None
        usage = {"input": getattr(resp.usage, "input_tokens", 0), "output": getattr(resp.usage, "output_tokens", 0),
                 "cached": (getattr(det, "cached_tokens", 0) or 0) if det else 0} if resp.usage else {}
        return StepResult(text, calls, resp.status or "", usage)

    def add_tool_results(self, results):
        images = []
        for r in results:
            self.input.append({"type": "function_call_output", "call_id": r.call.id, "output": r.text})
            images += [p for p in r.images if shows_images(r.call.name, self.cfg)]
        if images and self.cfg.vision:
            self.input.append({"role": "user", "content": [{"type": "input_text", "text": "Images from the tool calls above:"}]
                               + [self._image(p) for p in images]})
            self._track(self.input[-1], list(images))


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------
class AnthropicBackend(Backend):
    def __init__(self, cfg, system, tools):
        super().__init__(cfg, system, tools)
        import anthropic
        kw = dict(timeout=cfg.timeout, default_headers=cfg.extra_headers or None)
        if cfg.api_key:
            kw["api_key"] = cfg.api_key
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        hc = _http_client(cfg, anthropic)
        if hc is not None:
            kw["http_client"] = hc
        self.client = anthropic.Anthropic(max_retries=0, **kw)
        self.guard = RateGuard(cfg)
        self.messages: List[Dict] = []
        self.tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                          for t in tools]
        # server-side refusal fallbacks exist for the Opus 5 / Fable families; skip elsewhere
        self.use_fallbacks = bool(cfg.fallbacks) and cfg.model.startswith(("claude-opus-5", "claude-fable"))

    def add_user(self, text, images=None):
        content: List[Dict] = []
        for p in (images or []) if self.cfg.vision else []:
            mt = mimetypes.guess_type(p)[0] or "image/png"
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt,
                                                        "data": base64.b64encode(Path(p).read_bytes()).decode()}})
        content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})

    def _stream(self, on_text):
        import anthropic
        kw = dict(model=self.cfg.model, max_tokens=self.cfg.max_output_tokens,
                  system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                  tools=self.tool_defs, messages=self.messages, thinking={"type": "adaptive"})
        if self.cfg.reasoning_effort:
            kw["output_config"] = {"effort": self.cfg.reasoning_effort}
        if self.use_fallbacks:
            api = self.client.beta.messages
            kw["betas"] = ["server-side-fallback-2026-07-01"]
            kw["extra_body"] = {"fallbacks": self.cfg.fallbacks}
        else:
            api = self.client.messages
        try:
            with api.stream(**kw) as stream:
                for ev in stream:
                    if on_text and getattr(ev, "type", "") == "text":
                        on_text(ev.text)
                return stream.get_final_message()
        except anthropic.BadRequestError as e:
            if self.use_fallbacks and "fallback" in str(e).lower():
                self.use_fallbacks = False       # endpoint/model without fallback support
                return self._stream(on_text)
            raise

    def step(self, on_text=None):
        msg = self.guard.run(lambda: self._stream(on_text), _est_tokens([self.system, self.messages, self.tool_defs]))
        blocks = [b for b in msg.content]
        # after a server-side fallback, drop model-internal blocks that precede the last fallback marker
        fb = [i for i, b in enumerate(blocks) if getattr(b, "type", "") == "fallback"]
        if fb:
            cut = fb[-1]
            blocks = [b for i, b in enumerate(blocks)
                      if (i > cut or getattr(b, "type", "") == "text")]
        self.messages.append({"role": "assistant", "content": blocks})
        text = "".join(b.text for b in blocks if getattr(b, "type", "") == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in blocks if getattr(b, "type", "") == "tool_use"]
        if msg.stop_reason == "refusal":
            text += "\n[the model declined this request]"
        usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
                 "cached": getattr(msg.usage, "cache_read_input_tokens", 0) or 0}
        return StepResult(text, calls, msg.stop_reason or "", usage)

    def add_tool_results(self, results):
        content = []
        for r in results:
            parts: List[Dict] = [{"type": "text", "text": r.text}]
            if self.cfg.vision:
                for p in r.images:
                    mt = mimetypes.guess_type(p)[0] or "image/png"
                    parts.append({"type": "image", "source": {"type": "base64", "media_type": mt,
                                                              "data": base64.b64encode(Path(p).read_bytes()).decode()}})
            content.append({"type": "tool_result", "tool_use_id": r.call.id, "content": parts,
                            "is_error": r.is_error})
        self.messages.append({"role": "user", "content": content})


def make_backend(cfg: LLMConfig, system: str, tools: List[ToolSpec]) -> Backend:
    if cfg.provider == "claude_cli":
        return ClaudeCLIBackend(cfg, system, tools)
    if cfg.provider == "anthropic":
        return AnthropicBackend(cfg, system, tools)
    if cfg.api == "responses":
        return OpenAIResponsesBackend(cfg, system, tools)
    return OpenAIChatBackend(cfg, system, tools)


def oneshot(cfg: LLMConfig, system: str, user: str, model: Optional[str] = None,
            max_tokens: int = 6000) -> str:
    """A single, history-free, tool-free completion (used for the independent code review)."""
    model = model or cfg.model
    effort = cfg.reviewer_effort or cfg.reasoning_effort
    if cfg.provider == "claude_cli":       # fresh process, no session persistence, no tools
        return _claude_cli_run(cfg, system, user, model=model).get("result") or ""
    if cfg.provider == "anthropic":
        import anthropic
        kw = dict(timeout=cfg.timeout, default_headers=cfg.extra_headers or None)
        if cfg.api_key:
            kw["api_key"] = cfg.api_key
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        client = anthropic.Anthropic(**kw)
        extra = {"output_config": {"effort": effort}} if effort else {}
        msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}],
                                     thinking={"type": "adaptive"}, **extra)
        if msg.stop_reason == "refusal":
            return '{"verdict": "unsure", "summary": "reviewer declined", "concerns": []}'
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    import openai
    kw = dict(api_key=cfg.api_key or "EMPTY", timeout=cfg.timeout, default_headers=cfg.extra_headers or None)
    if cfg.base_url:
        kw["base_url"] = cfg.base_url
    hc = _http_client(cfg, openai)
    if hc is not None:
        kw["http_client"] = hc
    client = openai.OpenAI(**kw)
    if cfg.api == "responses":
        rk = dict(model=model, instructions=system, input=user, max_output_tokens=max_tokens, store=False)
        if effort:
            rk["reasoning"] = {"effort": effort}
        return client.responses.create(**rk).output_text or ""
    ck = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
              max_completion_tokens=max_tokens, response_format={"type": "json_object"})
    if effort:
        ck["reasoning_effort"] = effort
    for _ in range(4):
        try:
            r = client.chat.completions.create(**ck)
            return r.choices[0].message.content or ""
        except openai.BadRequestError as e:   # gateway doesn't support one of the optional params
            msg = str(e).lower()
            for p in ("response_format", "reasoning_effort", "max_completion_tokens"):
                if p in msg and p in ck:
                    ck.pop(p)
                    if p == "max_completion_tokens":
                        ck["max_tokens"] = max_tokens
                    break
            else:
                raise
    raise RuntimeError("reviewer request kept failing")


# ---------------------------------------------------------------------------
# Claude Code CLI (`claude -p`) - text-protocol tool calling
# ---------------------------------------------------------------------------
TOOL_PROTOCOL = """

# Tool use protocol
You cannot call tools natively in this environment. Instead, to call a tool, write one or more blocks \
exactly like this (valid JSON, one call per block):

<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

After writing your tool_call block(s), STOP and wait: the results come back in the next message as \
<tool_result name="..." id="...">...</tool_result> blocks. Never invent tool results or claim a \
calculation finished before you have seen its tool_result. When you need no more tools, answer normally \
(no tool_call blocks). Available tools (JSON Schema for the arguments):
"""

_CALL_RE = re.compile(r"<tool_call>\s*(?:```(?:json)?\s*)?(\{.*?\})\s*(?:```\s*)?</tool_call>", re.S)
# Claude sometimes falls back to its native call syntax in plain text; accept that too
_INVOKE_RE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.S)
_PARAM_RE = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.S)
_WRAPPER_RE = re.compile(r"</?(?:antml:)?function_calls>", re.S)


def parse_text_tool_calls(raw: str):
    """(calls as [(name, args, raw_json, error)], text without the call blocks) from a text-protocol reply."""
    calls = []
    for m in _CALL_RE.finditer(raw):
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments") or {}
            calls.append((obj.get("name", ""), args if isinstance(args, dict) else {}, m.group(1), ""))
        except json.JSONDecodeError as e:
            calls.append(("?", {}, m.group(1), f"invalid JSON in tool_call: {e}"))
    text = _CALL_RE.sub("", raw)
    if not calls:
        for m in _INVOKE_RE.finditer(text):
            args = {}
            for pm in _PARAM_RE.finditer(m.group(2)):
                v = pm.group(2)
                try:
                    args[pm.group(1)] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    args[pm.group(1)] = v.strip("\n")
            calls.append((m.group(1), args, json.dumps(args), ""))
    text = _WRAPPER_RE.sub("", _INVOKE_RE.sub("", text))
    return calls, text.strip()


def claude_cli_env(passthrough: Sequence[str] = ()) -> Dict[str, str]:
    """Environment for a child `claude -p`: like a fresh terminal - no parent Claude Code session variables."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC_")) or k in passthrough}
    env.setdefault("TERM", "dumb")
    return env


def _claude_cli_run(cfg: LLMConfig, system: str, prompt: str, model: Optional[str] = None,
                    resume: Optional[str] = None, persist: bool = False) -> Dict[str, Any]:
    cwd = Path(cfg.cli_workdir or (Path(tempfile.gettempdir()) / "calphad_claude_cli")).expanduser()
    cwd.mkdir(parents=True, exist_ok=True)       # neutral cwd: no CLAUDE.md / project settings picked up
    cmd = [cfg.cli_path or "claude", "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--disable-slash-commands", "--system-prompt", system]
    if model or cfg.model:
        cmd += ["--model", model or cfg.model]
    if cfg.reasoning_effort:
        cmd += ["--effort", cfg.reasoning_effort]
    if resume:
        cmd += ["--resume", resume]
    elif not persist:
        cmd += ["--no-session-persistence"]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=str(cwd),
                       env=claude_cli_env(cfg.cli_env_passthrough), timeout=cfg.timeout)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {}
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(f"claude -p returned non-JSON output: {p.stdout[:300]} {p.stderr[:300]}")
    if not d or d.get("is_error"):
        raise RuntimeError(f"claude -p error: {d.get('result') or p.stderr[:500] or 'no output'}")
    return d


class ClaudeCLIBackend(Backend):
    """Drives `claude -p` (Claude Code's print mode) with all of its own tools disabled.

    Our tools are described in the system prompt and called through <tool_call> blocks; the conversation
    continues across calls with --resume, so earlier turns stay real model turns.  No image input.
    """

    def __init__(self, cfg, system, tools):
        super().__init__(cfg, system, tools)
        specs = "\n".join(json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters})
                          for t in tools)
        self.full_system = system + TOOL_PROTOCOL + specs
        self.session_id: Optional[str] = None
        self.pending: List[str] = []
        self._n = 0

    def add_user(self, text, images=None):
        note = ("\n[images saved at: " + ", ".join(images) + "]") if images else ""
        self.pending.append(text + note)

    def step(self, on_text=None):
        prompt = "\n\n".join(self.pending) or "Continue."
        self.pending = []
        d = _claude_cli_run(self.cfg, self.full_system, prompt, resume=self.session_id, persist=True)
        self.session_id = d.get("session_id") or self.session_id
        raw = d.get("result") or ""
        calls = []
        parsed, text = parse_text_tool_calls(raw)
        for name, args, rawj, err in parsed:
            self._n += 1
            calls.append(ToolCall(f"call_{self._n}", name, args, rawj, err))
        if text and on_text:
            on_text(text)
        u = d.get("usage") or {}
        return StepResult(text, calls, d.get("stop_reason") or "",
                          {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)})

    def add_tool_results(self, results):
        parts = []
        for r in results:
            imgs = ("\n[image files: " + ", ".join(r.images) + "]") if r.images else ""
            status = ' status="error"' if r.is_error else ""
            parts.append(f'<tool_result name="{r.call.name}" id="{r.call.id}"{status}>\n{r.text}{imgs}\n</tool_result>')
        self.pending.append("\n".join(parts))
