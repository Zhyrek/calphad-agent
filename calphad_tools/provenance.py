"""Reproducibility + provenance for agent sessions.

* Every tool call is appended to ``<output_dir>/.session/<session-id>.jsonl`` (one environment header line,
  then one line per call: tool, redacted args, ok, seconds, input-file hashes, output files + hashes).
* Every PNG a call produces under the output dir gets a ``<file>.provenance.json`` sidecar and an
  ``iTXt`` chunk (keyword ``calphad_tools:provenance``) inserted losslessly into the PNG byte stream; the
  PNG's mtime is restored, so the editable-figure store (:mod:`calphad_tools.figures`) still treats it as fresh.
* Tools: ``session_log`` (list calls), ``provenance`` (who made a file), ``export_session`` (runnable replay
  script and/or Jupyter notebook that re-executes the calls without an LLM).

Public helpers for other code: :func:`file_info`, :func:`environment`, :func:`write_provenance`,
:func:`provenance_for`, :func:`redact`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import struct
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SESSION_DIR = ".session"
SIDECAR_SUFFIX = ".provenance.json"
PNG_KEYWORD = "calphad_tools:provenance"
EMBED_PNG_TEXT = True                   # insert an iTXt chunk into produced PNGs (lossless, mtime preserved)
PACKAGES = ["pycalphad", "espei", "numpy", "scipy", "matplotlib", "pandas", "xarray", "symengine",
            "calphad_tools"]
SECRET_KEY_RE = re.compile(r"key|token|secret|password", re.I)
SECRET_VALUE_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,})")
REDACTED = "[REDACTED]"
OUTPUT_ARG_RE = re.compile(r"^(output.*|out|.*_output|output_name|output_dir|save_as|dest.*)$", re.I)
SKIP_EXPORT = {"session_log", "export_session", "provenance", "ask_user", "recent_results",
               "rerun_call", "package_feedback", "report_problem"}   # rerun_call: its inner call is logged (and replayed) on its own
TEXT_EXTS = {".csv", ".json", ".txt", ".tdb", ".dat", ".yaml", ".yml", ".exp", ".pop", ".dop", ".tsv", ".md"}
HASH_LIMIT = 1 << 30                    # don't hash files > 1 GiB (size + mtime only)
MAX_SCAN = 50000                        # output-dir entries scanned per call for new files
_PATH_TOKEN = re.compile(r"[\w.~/\\+@%=-]*[\w-]\.[A-Za-z][A-Za-z0-9]{0,5}\b")


# --------------------------------------------------------------------------- small utilities
def _now_iso(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else time.time()))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_HASH_CACHE: Dict[Tuple[str, int, int], str] = {}


def file_info(path: str, png_size: bool = True) -> Dict[str, Any]:
    """sha256 (cached by size+mtime), size, mtime and, for PNGs, pixel size."""
    st = os.stat(path)
    info: Dict[str, Any] = {"size": st.st_size, "mtime": _now_iso(st.st_mtime)}
    if st.st_size <= HASH_LIMIT:
        k = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
        if k not in _HASH_CACHE:
            _HASH_CACHE[k] = sha256_file(path)
        info["sha256"] = _HASH_CACHE[k]
    if png_size and str(path).lower().endswith(".png"):
        dims = png_dimensions(path)
        if dims:
            info["png"] = list(dims)
    return info


def png_dimensions(path: str) -> Optional[Tuple[int, int]]:
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return struct.unpack(">II", head[16:24])
    except OSError:
        pass
    return None


def jsonable(obj: Any) -> Any:
    """JSON round trip (non-serializable values become str)."""
    return json.loads(json.dumps(obj, default=str))


def _secret_values(extra: Iterable[str] = ()) -> List[str]:
    vals = [v for k, v in os.environ.items() if SECRET_KEY_RE.search(k) and v and len(v) >= 8]
    return sorted({v for v in list(vals) + [e for e in extra if e and len(e) >= 8]}, key=len, reverse=True)


def redact(obj: Any, secrets_: Iterable[str] = ()) -> Any:
    """Copy of ``obj`` with values of secret-looking keys and secret-looking strings replaced."""
    secrets_ = list(secrets_)

    def _s(s: str) -> str:
        for v in secrets_:
            if v in s:
                s = s.replace(v, REDACTED)
        return SECRET_VALUE_RE.sub(REDACTED, s)

    def rec(o):
        if isinstance(o, dict):
            # secret-named keys: scalar values are redacted, containers are searched (e.g. a quiver "key" dict)
            return {k: (REDACTED if SECRET_KEY_RE.search(str(k)) and not isinstance(v, (dict, list, tuple))
                        and v is not None and not isinstance(v, bool) else rec(v)) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rec(v) for v in o]
        if isinstance(o, str):
            return _s(o)
        return o
    return rec(o=obj)


def versions() -> Dict[str, str]:
    import importlib.metadata as md
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            try:
                out[p] = str(getattr(__import__(p), "__version__", "unknown"))
            except Exception:  # noqa: BLE001
                out[p] = "not installed"
    return out


def environment() -> Dict[str, Any]:
    """Python/platform/package versions (no environment variables, no host/user names)."""
    import calphad_tools
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "versions": versions(),
            "calphad_tools_path": str(Path(calphad_tools.__file__).resolve().parent.parent)}


# --------------------------------------------------------------------------- PNG text chunk
def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def embed_png_text(path: str, keyword: str, text: str) -> bool:
    """Insert (or replace) an iTXt chunk before IEND without re-encoding; keeps the file's mtime."""
    try:
        raw = Path(path).read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        chunks, pos = [], 8
        while pos + 8 <= len(raw):
            n, ctype = struct.unpack(">I", raw[pos:pos + 4])[0], raw[pos + 4:pos + 8]
            end = pos + 12 + n
            if end > len(raw):
                return False                                # truncated / corrupt -> leave alone
            chunks.append((ctype, raw[pos:end]))
            pos = end
            if ctype == b"IEND":
                break
        if not chunks or chunks[-1][0] != b"IEND":
            return False
        kw = keyword.encode("latin-1")
        keep = [c for t, c in chunks if not (t == b"iTXt" and c[8:8 + len(kw) + 1] == kw + b"\0")]
        itxt = _png_chunk(b"iTXt", kw + b"\0\0\0" + b"\0" + b"\0" + text.encode("utf-8"))
        new = raw[:8] + b"".join(keep[:-1]) + itxt + keep[-1] + raw[pos:]
        st = os.stat(path)
        tmp = f"{path}.{secrets.token_hex(4)}.tmp"
        with open(tmp, "wb") as f:
            f.write(new)
        os.replace(tmp, path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # figures.py staleness check uses the mtime
        return True
    except Exception:  # noqa: BLE001 - never break an output file
        return False


def read_png_text(path: str, keyword: str = PNG_KEYWORD) -> Optional[str]:
    raw = Path(path).read_bytes()
    pos, kw = 8, keyword.encode("latin-1") + b"\0"
    while pos + 8 <= len(raw):
        n, ctype = struct.unpack(">I", raw[pos:pos + 4])[0], raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + n]
        if ctype == b"iTXt" and data.startswith(kw):
            rest = data[len(kw) + 2:]                        # compression flag + method
            rest = rest.split(b"\0", 2)                      # language tag, translated keyword, text
            return rest[2].decode("utf-8") if len(rest) == 3 else None
        pos += 12 + n
    return None


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_provenance(path: str, info: Dict[str, Any], embed_png: bool = EMBED_PNG_TEXT) -> Dict[str, Any]:
    """Attach provenance ``info`` to an output file: PNG iTXt chunk (PNG only) + ``<file>.provenance.json``.
    The sidecar records the final file hash.  Returns the sidecar content."""
    p = Path(path)
    if embed_png and p.suffix.lower() == ".png":
        compact = {k: info[k] for k in ("session", "call_index", "time", "tool", "args", "inputs", "versions")
                   if k in info}
        s = json.dumps(compact, default=str)
        if len(s) > 20000:
            compact["args"] = "(large; see sidecar)"
            s = json.dumps(compact, default=str)
        embed_png_text(str(p), PNG_KEYWORD, s)
    side = {"file": p.name, **info, "file_info": file_info(str(p))}
    _atomic_write(p.with_name(p.name + SIDECAR_SUFFIX), json.dumps(side, indent=1, default=str))
    return side


# --------------------------------------------------------------------------- session recorder
class Recorder:
    """Per-ToolBox session state: listener, jsonl log, enriched call records."""

    def __init__(self, tb):
        self.tb = tb
        self.lock = threading.Lock()
        self.records: List[Dict[str, Any]] = []
        self.pending: Optional[Dict[str, Any]] = None
        self.env: Optional[Dict[str, Any]] = None
        self.started = time.time()
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        self.log_path: Optional[Path] = None
        llm = getattr(tb, "llm_cfg", None)
        self._secrets = _secret_values([str(getattr(llm, "api_key", "") or "")])

    # ---------------------------------------------------------------- paths
    @property
    def session_dir(self) -> Path:
        return self.tb.out / SESSION_DIR

    def _open_log(self) -> Path:
        d = self.session_dir
        d.mkdir(parents=True, exist_ok=True)
        for _ in range(20):                               # O_EXCL: concurrent sessions never share a file
            p = d / f"{self.session_id}.jsonl"
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                return p
            except FileExistsError:
                self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        raise RuntimeError("could not create a unique session log")

    def _append(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, default=str) + "\n"
        with open(self.log_path, "a") as f:
            f.write(line)

    def _under_out(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.tb.out)
            return True
        except ValueError:
            return False

    # ---------------------------------------------------------------- inputs
    def _candidate_paths(self, args: Any, key: str = "") -> Iterable[Tuple[str, str]]:
        if isinstance(args, dict):
            for k, v in args.items():
                yield from self._candidate_paths(v, str(k))
        elif isinstance(args, (list, tuple)):
            for v in args[:500]:
                yield from self._candidate_paths(v, key)
        elif isinstance(args, str) and 0 < len(args) < 1024 and not any(ch in args for ch in "\n*?["):
            yield key, args

    def _resolve_input(self, s: str) -> Optional[Path]:
        try:
            p = Path(self.tb.p(s))
            if p.exists():
                return p
            if s.lower().endswith(".tdb"):
                return Path(self.tb._find_tdb(s))
            if len(Path(s).parts) == 1 and (self.tb.out / s).exists():
                return self.tb.out / s
        except Exception:  # noqa: BLE001
            return None
        return None

    def input_hashes(self, args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Hash every existing file (or files in a directory, capped) referenced by the args."""
        out: Dict[str, Dict[str, Any]] = {}
        for key, s in self._candidate_paths(args):
            if OUTPUT_ARG_RE.match(key) or SECRET_KEY_RE.search(key):
                continue
            p = self._resolve_input(s)
            if p is None:
                continue
            files = [p] if p.is_file() else (sorted(q for q in p.rglob("*") if q.is_file()
                                                    and not any(x.startswith(".") for x in q.relative_to(p).parts))[:200]
                                             if p.is_dir() and p.resolve() not in (self.tb.workdir, self.tb.out) else [])
            for f in files:
                try:
                    out[self.tb.rel(str(f))] = {**file_info(str(f), png_size=False),
                                                "generated": self._under_out(f)}
                except OSError:
                    pass
        return out

    # ---------------------------------------------------------------- outputs
    def _snapshot(self) -> Dict[str, Tuple[int, int]]:
        """{path: (mtime_ns, size)} of the files under the output dir (hidden dirs/sidecars excluded)."""
        snap: Dict[str, Tuple[int, int]] = {}
        n = 0
        stack = [self.tb.out]
        while stack and n < MAX_SCAN:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        n += 1
                        if e.name.startswith(".") or e.name.endswith((SIDECAR_SUFFIX, ".tmp")):
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            st = e.stat()
                            snap[str(Path(e.path).resolve())] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
        return snap

    def _paths_in_text(self, text: str) -> List[Path]:
        found = []
        for m in _PATH_TOKEN.finditer(text or ""):
            tok = m.group(0).strip(".,;:'\"()[]")
            for base in (self.tb.workdir, self.tb.out):
                p = Path(tok) if os.path.isabs(tok) else base / tok
                try:
                    if p.is_file() and self._under_out(p) and SESSION_DIR not in p.relative_to(self.tb.out).parts:
                        found.append(p.resolve())
                        break
                except (OSError, ValueError):
                    continue
        return found

    # ---------------------------------------------------------------- hooks
    def before(self, name: str, args: Dict[str, Any]) -> None:
        try:
            self.pending = {"t0": time.time(), "tool": name, "snap": self._snapshot(), "args": jsonable(args),
                            "inputs": self.input_hashes(args)}
        except Exception:  # noqa: BLE001
            self.pending = {"t0": time.time(), "tool": name}

    def after(self, text: str) -> None:
        if self.pending is not None:
            self.pending["text"] = text

    def listener(self, rec: Dict[str, Any]) -> None:
        with self.lock:
            try:
                self._record(rec)
            finally:
                self.pending = None

    def _record(self, rec: Dict[str, Any]) -> None:
        pend = self.pending if (self.pending and self.pending.get("tool") == rec["tool"]) else {}
        t0 = pend.get("t0") or (time.time() - float(rec.get("seconds") or 0) - 1.0)
        args = pend["args"] if "args" in pend else jsonable(rec.get("args", {}))
        inputs = pend["inputs"] if "inputs" in pend else self.input_hashes(rec.get("args", {}))
        text = pend.get("text") or rec.get("result_preview", "")
        index = len(self.tb.call_log)
        # outputs: produced images + files named in the result + files written under out during the call
        cands: Dict[str, Path] = {}
        for im in rec.get("images") or []:
            p = Path(im)
            if p.is_file() and self._under_out(p):
                cands[str(p.resolve())] = p.resolve()
        for p in self._paths_in_text(text):
            cands[str(p)] = p
        after = self._snapshot()
        before = pend.get("snap")
        for k in after:
            cands[k] = Path(k)
        if before is not None:                   # new or changed since the call started
            fresh = {k: p for k, p in cands.items() if k in after and before.get(k) != after[k]}
        else:                                    # no pre-call snapshot: fall back to modification times
            fresh = {k: p for k, p in cands.items() if p.exists() and p.stat().st_mtime >= t0}
        for k in list(inputs):                             # rewritten during the call -> an output, not an input
            if str((self.tb.workdir / k).resolve()) in fresh:
                inputs.pop(k)
        if self.log_path is None:
            self.env = {"type": "header", "session": self.session_id, "started": _now_iso(self.started),
                        "workdir": str(self.tb.workdir), "output_dir": str(self.tb.out),
                        "output_dir_rel": self.tb.rel(str(self.tb.out)),
                        "python_tool": getattr(self.tb, "python_tool", "off"), **environment()}
            self.log_path = self._open_log()
            self._append(self.env)
        red_args = redact(args, self._secrets)
        prov_base = {"session": self.session_id, "call_index": index, "time": rec.get("time"), "tool": rec["tool"],
                     "args": red_args, "inputs": inputs, "versions": self.env["versions"],
                     "python": self.env["python"], "platform": self.env["platform"],
                     "workdir": self.env["workdir"], "log": self.tb.rel(str(self.log_path))}
        outputs = []
        for k, p in sorted(fresh.items()):
            try:
                if p.suffix.lower() == ".png" and rec.get("ok") and rec["tool"] != "rerun_call":
                    write_provenance(str(p), prov_base)
                info = file_info(str(p))
            except OSError:
                continue
            outputs.append({"path": self.tb.rel(str(p)), "out_rel": str(p.relative_to(self.tb.out)), **info})
        entry = {"type": "call", "index": index, "time": rec.get("time"), "tool": rec["tool"], "args": red_args,
                 "ok": bool(rec.get("ok")), "seconds": rec.get("seconds"),
                 "images": [self.tb.rel(i) for i in rec.get("images") or []], "inputs": inputs,
                 "outputs": outputs, "result_preview": redact(rec.get("result_preview", ""), self._secrets)}
        if rec["tool"] == "run_python":
            code = (args or {}).get("code", "")
            entry["executed"] = bool(code) and code in getattr(self.tb, "py_history", [])
        self.records.append(entry)
        self._append(entry)

    # ---------------------------------------------------------------- queries
    def load(self, session: Optional[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """(header, call records) for this session (default) or a logged session id / jsonl path."""
        if not session or session == self.session_id:
            return (self.env or {"session": self.session_id, "workdir": str(self.tb.workdir),
                                 "output_dir": str(self.tb.out), "output_dir_rel": self.tb.rel(str(self.tb.out)),
                                 "python_tool": self.tb.python_tool, **environment()}), list(self.records)
        p = Path(session) if session.endswith(".jsonl") else self.session_dir / f"{session}.jsonl"
        if not p.is_absolute():
            p = p if p.exists() else Path(self.tb.p(str(p)))
        if not p.exists():
            known = sorted(q.stem for q in self.session_dir.glob("*.jsonl"))
            raise FileNotFoundError(f"no session log {session!r}; known sessions: {known[-10:]}")
        header, calls = {}, []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") == "header":
                header = o
            elif o.get("type") == "call":
                calls.append(o)
        return header, calls


# --------------------------------------------------------------------------- formatting helpers
def _compact(v: Any) -> Any:
    """Long lists (anywhere in the structure) -> '[n items]' markers, long strings shortened."""
    if isinstance(v, dict):
        return {k: _compact(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return f"\x00[{len(v)} items]\x00" if len(v) > 6 else [_compact(x) for x in v]
    if isinstance(v, str) and len(v) > 40:
        return v[:37] + "…"
    return v


def _short(v: Any, n: int = 70) -> str:
    c = _compact(v)
    s = c if isinstance(c, str) else json.dumps(c, default=str, separators=(",", ":"), ensure_ascii=False)
    s = re.sub(r'"?\\u0000"?', "", s).replace("\x00", "")
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize_args(args: Dict[str, Any], width: int = 160) -> str:
    s = " ".join(f"{k}={_short(v)}" for k, v in (args or {}).items())
    return s if len(s) <= width else s[: width - 1] + "…"


def parse_steps(spec: Optional[str], since: Optional[int], n: int) -> List[int]:
    idx = set(range(1, n + 1))
    if spec:
        sel = set()
        for part in str(spec).replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                sel |= set(range(int(a or 1), int(b or n) + 1))
            else:
                sel.add(int(part))
        idx &= sel
    if since:
        idx = {i for i in idx if i >= int(since)}
    return sorted(idx)


def provenance_for(tb, file: str, rec: Optional[Recorder] = None) -> Optional[Dict[str, Any]]:
    """Provenance of an output file: its sidecar, else the latest logged call that produced it."""
    p = Path(tb.p(file))
    if not p.exists() and len(Path(file).parts) == 1:
        p = tb.out / file
    p = p.resolve()
    side = p.with_name(p.name + SIDECAR_SUFFIX)
    if side.exists():
        info = json.loads(side.read_text())
        info["source"] = tb.rel(str(side))
        return info
    target = tb.rel(str(p))
    logs = sorted((tb.out / SESSION_DIR).glob("*.jsonl"), key=lambda q: q.stat().st_mtime, reverse=True)
    for log in logs:
        header, best = {}, None
        for line in log.read_text().splitlines():
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("type") == "header":
                header = o
            elif o.get("type") == "call" and any(x.get("path") == target for x in o.get("outputs", [])):
                best = o
        if best:
            fi = next(x for x in best["outputs"] if x.get("path") == target)
            return {"file": p.name, "session": header.get("session", log.stem), "call_index": best["index"],
                    "time": best["time"], "tool": best["tool"], "args": best["args"], "inputs": best["inputs"],
                    "versions": header.get("versions"), "python": header.get("python"),
                    "platform": header.get("platform"), "file_info": fi, "source": tb.rel(str(log))}
    return None


# --------------------------------------------------------------------------- export
HELPERS = r'''
import hashlib, json, os, sys
from pathlib import Path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _png_dims(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return [int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")] if head[12:16] == b"IHDR" else None


TEXT_EXTS = {".csv", ".json", ".txt", ".tdb", ".dat", ".yaml", ".yml", ".exp", ".pop", ".dop", ".tsv", ".md"}


class Replay:
    """Re-executes recorded tool calls in a fresh ToolBox (no LLM involved)."""

    def __init__(self, workdir, output_dir, run_python=False, strict=False, show=None):
        try:
            import calphad_tools  # noqa: F401
        except ImportError:
            sys.path.insert(0, SESSION.get("calphad_tools_path", ""))
        from calphad_tools.agent.tools import UI, ToolBox

        class _ReplayUI(UI):
            def ask(self, question, options=None):
                return ""                      # ask_user is not replayed; answers are already in the args

            def confirm(self, title, body):
                return bool(run_python)

        self.workdir, self.run_python, self.strict, self.show = Path(workdir).resolve(), run_python, strict, show
        self.new_out = str(output_dir)
        self.tb = ToolBox(str(self.workdir), str(output_dir), _ReplayUI(),
                          python_tool="auto" if run_python else "off", interactive=False)
        self.problems, self.notes = [], []
        print(f"replaying session {SESSION['session']} in {self.workdir} -> {self.tb.out}")
        self.check_inputs()

    def check_inputs(self):
        for rel, info in INPUTS.items():
            if info.get("generated"):
                continue                        # produced earlier in the session; recreated by the replay
            p = self.workdir / rel
            if not p.exists():
                print(f"WARNING: input {rel} is missing")
            elif info.get("sha256") and _sha256(p) != info["sha256"]:
                print(f"WARNING: input {rel} changed since the session (sha256 differs)")

    def remap(self, obj):
        """Paths into the original output dir -> the replay output dir."""
        if isinstance(obj, dict):
            return {k: self.remap(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.remap(v) for v in obj]
        if isinstance(obj, str):
            olds = [SESSION["output_dir"]] + ([ORIG_OUT] if self.new_out != ORIG_OUT else [])
            for old in olds:
                if obj == old or obj.startswith(old.rstrip("/") + "/"):
                    return str(self.tb.out) + obj[len(old.rstrip("/")):]
        return obj

    def step(self, i, tool, args, expect_ok=True, python=False):
        if python and not self.run_python:
            print(f"[{i}] {tool}: SKIPPED (arbitrary Python; enable with --run-python / RUN_PYTHON = True)")
            return "", []
        if tool not in self.tb.tools:
            raise RuntimeError(f"step {i}: tool {tool!r} is not available in this installation")
        text, images, err = self.tb.call(tool, self.remap(args))
        print(f"[{i}] {tool}: {'FAILED' if err else 'ok'}")
        print("    " + text[:1500].replace("\n", "\n    "))
        if expect_ok:
            assert not err, f"step {i} ({tool}) failed on replay:\n{text[:3000]}"
        self.check_outputs(i)
        if self.show:
            self.show(images)
        return text, images

    def check_outputs(self, i):
        for rel, info in EXPECTED.get(str(i), {}).items():
            p = self.tb.out / rel
            if not p.exists():
                self.problems.append(f"step {i}: {rel} was not recreated")
                print(f"    MISSING  {rel}")
                continue
            if p.suffix.lower() in TEXT_EXTS and info.get("sha256"):
                same = _sha256(p) == info["sha256"]
                print(f"    {'identical' if same else 'DIFFERS  '} {rel}")
                if not same:
                    self.problems.append(f"step {i}: {rel} differs from the original")
            elif p.suffix.lower() == ".png" and info.get("png"):
                dims = _png_dims(p)
                print(f"    {'same size' if dims == info['png'] else 'SIZE DIFF'} {rel} {dims}")
                if dims != info["png"]:
                    self.notes.append(f"step {i}: {rel} is {dims}, originally {info['png']}")
            else:
                print(f"    recreated {rel}")

    def finish(self):
        print(f"\nreplay finished: {len(self.problems)} problem(s), {len(self.notes)} note(s)")
        for m in self.problems + self.notes:
            print("  - " + m)
        if self.strict and self.problems:
            raise SystemExit(1)
'''


def pretty_json(obj: Any, width: int = 88, level: int = 0) -> str:
    """Indented JSON that keeps short lists/objects on one line."""
    flat = json.dumps(obj, default=str, ensure_ascii=False)
    if not isinstance(obj, (dict, list)) or not obj or len(flat) + level <= width:
        return flat
    pad = " " * (level + 1)
    if isinstance(obj, dict):
        items = [f"{pad}{json.dumps(str(k), ensure_ascii=False)}: {pretty_json(v, width, level + 1)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
    return "[\n" + ",\n".join(pad + pretty_json(v, width, level + 1) for v in obj) + "\n" + " " * level + "]"


def _pyjson(obj: Any) -> str:
    """Python expression that evaluates to the JSON value ``obj``: json.loads(r\"\"\"...\"\"\")."""
    s = pretty_json(obj)
    if "\n" not in s and len(s) < 70 and '"""' not in s.replace('\\"', ""):
        return f'json.loads(r"""{s}""")' if not s.endswith('"') else f"json.loads({s!r})"
    if '"""' in s.replace('\\"', ""):
        return f"json.loads({s!r})"
    return f'json.loads(r"""\n{s}\n""")'


def _pystr(code: str) -> str:
    if "'''" not in code and not code.endswith(("\\", "'")):
        return "r'''" + code + "'''"
    if '"""' not in code and not code.endswith(("\\", '"')):
        return 'r"""' + code + '"""'
    return repr(code)


def _rel_args(obj: Any, workdir: str) -> Any:
    wd = workdir.rstrip("/") + "/"
    if isinstance(obj, dict):
        return {k: _rel_args(v, workdir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rel_args(v, workdir) for v in obj]
    if isinstance(obj, str) and obj.startswith(wd):
        return obj[len(wd):]
    return obj


def _plan(header: Dict[str, Any], calls: List[Dict[str, Any]], include_failed: bool, steps: Optional[str],
          since: Optional[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    wanted = set(parse_steps(steps, since, max([c["index"] for c in calls], default=0)))
    plan, skipped = [], []
    for c in calls:
        if c["index"] not in wanted:
            continue
        why = None
        if c["tool"] in SKIP_EXPORT:
            why = "session/interaction tool"
        elif not c["ok"] and not include_failed:
            why = "failed"
        elif c["tool"] == "run_python" and not c.get("executed", True):
            why = "code was not executed in the session"
        if why:
            skipped.append(f"#{c['index']} {c['tool']} ({why})")
            continue
        plan.append({**c, "args": _rel_args(c["args"], header.get("workdir", ""))})
    return plan, skipped


def _inputs_and_expected(header, plan) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inputs: Dict[str, Any] = {}
    for c in plan:
        for k, v in c.get("inputs", {}).items():
            inputs.setdefault(k, {kk: v.get(kk) for kk in ("sha256", "size", "mtime", "generated")})
    expected = {str(c["index"]): {o["out_rel"]: {k: o.get(k) for k in ("sha256", "size", "png") if o.get(k)}
                                  for o in c.get("outputs", []) if c["ok"]} for c in plan}
    return inputs, {k: v for k, v in expected.items() if v}


def _session_consts(header, inputs, expected) -> str:
    sess = {k: header.get(k) for k in ("session", "started", "workdir", "output_dir", "output_dir_rel",
                                       "python_tool", "python", "platform", "versions", "calphad_tools_path")}
    return (f"SESSION = {_pyjson(sess)}\n"
            f"INPUTS = {_pyjson(inputs)}\n"
            f"EXPECTED = {_pyjson(expected)}   # step -> output file (relative to the output dir) -> original hash\n"
            f"ORIG_OUT = SESSION['output_dir_rel']\n")


def _code_const(c: Dict[str, Any]) -> str:
    return (f"# ---- step {c['index']}: arbitrary Python from the original session (NOT run unless --run-python / "
            f"RUN_PYTHON = True) ----\nCODE_{c['index']} = {_pystr(c['args'].get('code', ''))}\n")


def _step_call(c: Dict[str, Any], var: str = "R", inline_code: bool = True) -> str:
    lines = []
    redacted = REDACTED in json.dumps(c["args"])
    if redacted:
        lines.append("# NOTE: some arguments were redacted in the log ([REDACTED]); fill them in before running.")
    if c["tool"] == "run_python":
        if inline_code:
            lines.append(_code_const(c).rstrip("\n"))
        lines.append(f"{var}.step({c['index']}, 'run_python', {{'code': CODE_{c['index']}}}, python=True)")
    else:
        extra = "" if c["ok"] else ", expect_ok=False"
        lines.append(f"{var}.step({c['index']}, {c['tool']!r}, {_pyjson(c['args'])}{extra})")
    return "\n".join(lines)


def _default_rel(target: str, frm: Path) -> str:
    try:
        return os.path.relpath(target, str(frm))
    except ValueError:
        return target


def build_script(header, plan, skipped, script_path: Path) -> str:
    inputs, expected = _inputs_and_expected(header, plan)
    wd_rel = _default_rel(header["workdir"], script_path.parent)
    orig_out = header.get("output_dir_rel") or header.get("output_dir")
    body = "\n\n".join(f"    # ---- step {c['index']}: {c['tool']}  ({c['time']}, {c['seconds']} s"
                       f"{'' if c['ok'] else ', FAILED in the session'})\n"
                       + "\n".join("    " + ln if ln else "" for ln in _step_call(c, inline_code=False).splitlines())
                       for c in plan) or "    pass"
    env_lines = "\n".join(f"#   {k:<14} {v}" for k, v in (header.get("versions") or {}).items())
    in_lines = "\n".join(f"#   {k}  sha256={(v.get('sha256') or '?')[:16]}…  {v.get('size')} B"
                         + ("  (generated in session)" if v.get("generated") else "")
                         for k, v in inputs.items()) or "#   (none)"
    return f'''#!/usr/bin/env python
"""Replay of calphad_tools agent session {header.get("session")} (started {header.get("started")}).

Re-executes the recorded tool calls deterministically, without an LLM, in a fresh ToolBox.
Generated by calphad_tools.provenance.export_session on {_now_iso()}.

  python {script_path.name} [--workdir DIR] [--output-dir DIR] [--run-python] [--strict]

--workdir     working directory the tools resolve paths against (default: the session's, relative to this file)
--output-dir  where outputs go (relative to workdir; default "{orig_out}_replay")
--run-python  also run the recorded arbitrary-Python (run_python) steps (off by default)
--strict      exit 1 if an output is missing or a text output (CSV/JSON/TDB...) differs
"""
# Environment of the original session: Python {header.get("python")} on {header.get("platform")}
{env_lines}
# Input files (sha256 checked at replay time):
{in_lines}
# Skipped calls: {", ".join(skipped) or "none"}
# Original python_tool setting: {header.get("python_tool")!r}
{HELPERS}

{_session_consts(header, inputs, expected)}DEFAULT_WORKDIR = {wd_rel!r}     # relative to this script

{"".join(_code_const(c) + chr(10) for c in plan if c["tool"] == "run_python")}


def replay(R):
{body}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--workdir", default=str((here / DEFAULT_WORKDIR).resolve())
                    if (here / DEFAULT_WORKDIR).exists() else SESSION["workdir"])
    ap.add_argument("--output-dir", default=ORIG_OUT + "_replay")
    ap.add_argument("--run-python", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    R = Replay(a.workdir, a.output_dir, run_python=a.run_python, strict=a.strict)
    replay(R)
    R.finish()


if __name__ == "__main__":
    main()
'''


def _md(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> Dict[str, Any]:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build_notebook(header, plan, skipped, nb_path: Path) -> Dict[str, Any]:
    inputs, expected = _inputs_and_expected(header, plan)
    wd_rel = _default_rel(header["workdir"], nb_path.parent)
    orig_out = header.get("output_dir_rel") or header.get("output_dir")
    vers = "\n".join(f"| {k} | {v} |" for k, v in (header.get("versions") or {}).items())
    ins = "\n".join(f"| `{k}` | `{(v.get('sha256') or '?')[:16]}…` | {v.get('size')} |"
                    f" {'generated' if v.get('generated') else 'input'} |" for k, v in inputs.items())
    cells = [_md(f"# Replay of session `{header.get('session')}`\n\nStarted {header.get('started')}; "
                 f"Python {header.get('python')} on {header.get('platform')}. Re-executes the recorded tool calls "
                 f"without an LLM. Original python_tool: `{header.get('python_tool')}`.\n\n"
                 f"| package | version |\n|---|---|\n{vers}\n\n"
                 + (f"| input file | sha256 | bytes | kind |\n|---|---|---|---|\n{ins}\n\n" if ins else "")
                 + f"Skipped calls: {', '.join(skipped) or 'none'}"),
             _code("# Settings (the notebook's directory is the current directory in Jupyter)\n"
                   f"WORKDIR = {wd_rel!r}          # relative to this notebook, or an absolute path\n"
                   f"OUTPUT_DIR = {orig_out + '_replay'!r}   # relative to WORKDIR\n"
                   "RUN_PYTHON = False   # set True to also run the recorded arbitrary-Python steps\n"
                   "STRICT = False"),
             _code(HELPERS.strip() + "\n\n" + _session_consts(header, inputs, expected) +
                   "\n\ndef _show(images):\n    try:\n        from IPython.display import Image, display\n"
                   "    except ImportError:\n        return\n    for im in images:\n"
                   "        if str(im).lower().endswith(('.png', '.jpg', '.jpeg')):\n"
                   "            display(Image(filename=im))\n\n\n"
                   "_wd = Path(WORKDIR) if Path(WORKDIR).is_absolute() else (Path.cwd() / WORKDIR)\n"
                   "if not _wd.exists():\n    _wd = Path(SESSION['workdir'])\n"
                   "R = Replay(_wd, OUTPUT_DIR, run_python=RUN_PYTHON, strict=STRICT, show=_show)")]
    for c in plan:
        prev = (c.get("result_preview") or "").strip()
        cells.append(_md(f"## Step {c['index']}: `{c['tool']}`\n\n{c['time']}, {c['seconds']} s"
                         + ("" if c["ok"] else " — **failed in the session**")
                         + (f", outputs: {', '.join('`' + o['path'] + '`' for o in c.get('outputs', []))}"
                            if c.get("outputs") else "")
                         + "\n\nOriginal result (preview):\n\n```\n" + prev.replace("```", "ˋˋˋ") + "\n```"))
        cells.append(_code(_step_call(c)))
    cells.append(_code("R.finish()"))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "calphad_tools": {"session": header.get("session")}},
            "cells": [{**cell, "id": f"cell-{i}"} for i, cell in enumerate(cells)]}


def validate_notebook(nb: Dict[str, Any]) -> None:
    """Structural nbformat-v4 check (plus nbformat's validator when installed)."""
    assert nb.get("nbformat") == 4 and isinstance(nb.get("nbformat_minor"), int)
    assert isinstance(nb.get("metadata"), dict) and isinstance(nb.get("cells"), list)
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code", "raw") and isinstance(c["source"], (list, str))
        assert isinstance(c.get("metadata"), dict)
        if c["cell_type"] == "code":
            assert "outputs" in c and "execution_count" in c
            compile("".join(c["source"]), "<cell>", "exec")
    try:
        import nbformat
    except ImportError:
        return
    nbformat.validate(nbformat.from_dict(nb))


def export_session(tb, rec: Recorder, fmt: str = "script", output: Optional[str] = None,
                   include_failed: bool = False, since: Optional[int] = None, steps: Optional[str] = None,
                   session: Optional[str] = None) -> Dict[str, Any]:
    header, calls = rec.load(session)
    plan, skipped = _plan(header, calls, include_failed, steps, since)
    if not plan:
        raise ValueError("nothing to export (no selected successful tool calls)")
    base = Path(tb.outp(output, f"replay/session_{header.get('session', rec.session_id)}.py"))
    if base.suffix.lower() in (".py", ".ipynb"):
        base = base.with_suffix("")
    files = []
    if fmt in ("script", "both"):
        sp = base.with_suffix(".py")
        src = build_script(header, plan, skipped, sp)
        compile(src, str(sp), "exec")
        sp.write_text(src)
        files.append(sp)
    if fmt in ("notebook", "both"):
        npth = base.with_suffix(".ipynb")
        nb = build_notebook(header, plan, skipped, npth)
        validate_notebook(nb)
        npth.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        json.loads(npth.read_text())
        files.append(npth)
    return {"files": files, "steps": [c["index"] for c in plan], "skipped": skipped,
            "python_steps": [c["index"] for c in plan if c["tool"] == "run_python"],
            "redacted": [c["index"] for c in plan if REDACTED in json.dumps(c["args"])]}


# --------------------------------------------------------------------------- registration
def register(tb) -> None:
    rec = Recorder(tb)
    tb.provenance = rec
    orig_call = tb._call

    def _call(name, args):                    # capture args/input hashes before and full text after
        rec.before(name, args)
        text, images, err = orig_call(name, args)
        rec.after(text)
        return text, images, err
    tb._call = _call
    tb.call_listeners.append(rec.listener)
    S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}

    @tb.add("session_log", "List this session's tool calls (index, tool, key args, ok, seconds, output files), "
            "e.g. to pick steps for export_session.", {"last": {**I, "description": "only the last n calls"},
                                                       "session": {**S, "description": "earlier session id"}}, [])
    def session_log(last: Optional[int] = None, session: Optional[str] = None):
        header, calls = rec.load(session)
        if not calls:
            return "no tool calls recorded yet", []
        shown = calls[-int(last):] if last else calls
        lines = [f"session {header.get('session')}  log: "
                 f"{tb.rel(str(rec.log_path)) if rec.log_path and not session else '(earlier session)'}"]
        for c in shown:
            outs = [o["out_rel"] for o in c.get("outputs", [])]
            o = (" -> " + ", ".join(outs[:5]) + (f" (+{len(outs) - 5})" if len(outs) > 5 else "")) if outs else ""
            lines.append(f"#{c['index']:<3} {c['time'][11:]}  {c['tool']:<24} {'ok  ' if c['ok'] else 'FAIL'} "
                         f"{c['seconds']:>7.2f}s  {summarize_args(c['args'])}{o}")
        if last and len(calls) > len(shown):
            lines.insert(1, f"(last {len(shown)} of {len(calls)} calls)")
        return "\n".join(lines), []

    @tb.add("provenance", "Provenance of an output file: which tool call made it, its args, input files + "
            "SHA-256, package versions, and whether the file changed since.", {"file": S}, ["file"])
    def provenance(file: str):
        info = provenance_for(tb, file, rec)
        if info is None:
            return f"no provenance recorded for {file} (not produced by a logged tool call)", []
        p = Path(tb.p(file)) if Path(tb.p(file)).exists() else tb.out / file
        fi = info.get("file_info") or {}
        if p.exists() and fi.get("sha256"):
            info["unchanged_since_recorded"] = sha256_file(str(p)) == fi["sha256"]
        return tb.j(info), []

    @tb.add("export_session", "Export the session's successful tool calls as a runnable Python replay script "
            "and/or Jupyter notebook (no LLM needed; checks input hashes; run_python steps are included but "
            "disabled by default). Select calls with steps ('3-7,9') or since.",
            {"format": {"type": "string", "enum": ["script", "notebook", "both"]}, "output": S,
             "include_failed": B, "since": {**I, "description": "first call index to include"},
             "steps": {**S, "description": "call indices, e.g. '3-7,9' (see session_log)"},
             "session": {**S, "description": "export an earlier logged session id instead of this one"}}, [])
    def export_session_tool(format: str = "script", output: Optional[str] = None, include_failed: bool = False,
                            since: Optional[int] = None, steps: Optional[str] = None, session: Optional[str] = None):
        r = export_session(tb, rec, format, output, include_failed, since, steps, session)
        msg = [f"exported {len(r['steps'])} step(s) {r['steps']}:"] + [f"  {tb.rel(str(f))}" for f in r["files"]]
        if r["skipped"]:
            msg.append("skipped: " + ", ".join(r["skipped"]))
        if r["python_steps"]:
            msg.append(f"run_python steps {r['python_steps']} run only with --run-python (RUN_PYTHON = True)")
        if r["redacted"]:
            msg.append(f"steps {r['redacted']} contain redacted arguments - edit before running")
        sp = next((f for f in r["files"] if f.suffix == ".py"), None)
        if sp:
            msg.append(f"run: python {tb.rel(str(sp))} --output-dir <dir>  (default: <output>_replay)")
        return "\n".join(msg), []
