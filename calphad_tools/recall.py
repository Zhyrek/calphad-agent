"""Referring back to earlier results: ``@last`` references, ``recent_results`` and ``rerun_call``.

For requests like "take the previous plot and add this data point" or "same diagram, but at 1200 K":

* Any string argument of any tool may be a reference, resolved to a real file before the tool runs (the call
  log records the resolved path, so provenance, show_code and replays see ordinary files):

  - ``@last``                  the main image of the most recent successful call that produced one
  - ``@last.csv``              the most recent output file with that extension (``.png``, ``.csv``, ``.tdb`` ...)
  - ``@last:map_phase_diagram`` (optionally ``.csv``)   the same, restricted to one tool
  - ``@call:12`` (optionally ``.csv``)                  from call 12 of this session (numbers from recent_results)

  When this session has no matching call yet (e.g. after a restart), the most recent earlier session log in
  ``<output_dir>/.session`` is searched.  Files that no longer exist are skipped.
* ``recent_results`` lists the last few results with their files and the reference to use.
* ``rerun_call`` repeats an earlier call with some arguments changed or removed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REF = re.compile(r"^@(?:last(?::(?P<tool>[A-Za-z_][\w]*))?|call:(?P<idx>\d+))(?P<ext>\.[A-Za-z0-9]{1,6})?$")
# tools whose "outputs" are other calls' files (listing, re-running): never the answer to "@last"
NOT_RESULTS = {"recent_results", "rerun_call", "session_log", "provenance", "export_session", "show_code",
               "list_files", "find_databases", "describe_figure", "plot_ops_help", "view_image", "view_pdf_page",
               "inspect_data", "tdb_info", "job_status", "list_jobs", "wait_job", "stop_job", "calc_cache",
               "report_problem", "package_feedback", "pycalphad_help", "list_calphad_capabilities", "ask_user"}


def _records(tb) -> List[Tuple[str, Dict[str, Any]]]:
    """(session label, call record) newest first: this session, then the most recent earlier session."""
    rec = getattr(tb, "provenance", None)
    if rec is None:
        return []
    out = [("this session", c) for c in reversed(rec.records)]
    try:
        logs = sorted(Path(rec.session_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        logs = []
    for p in logs:
        if rec.log_path is not None and Path(p).resolve() == Path(rec.log_path).resolve():
            continue
        try:
            _, calls = rec.load(str(p))
        except Exception:  # noqa: BLE001 - an unreadable old log is just skipped
            continue
        if calls:
            out += [(f"session {p.stem}", c) for c in reversed(calls)]
            break
    return out


def _files(tb, c: Dict[str, Any]) -> List[str]:
    """Existing output files of a call record, main images first (absolute paths)."""
    seen, out = set(), []
    for rel in list(c.get("images") or []) + [o.get("path") for o in c.get("outputs") or []]:
        if not rel:
            continue
        p = Path(tb.p(rel))
        k = str(p.resolve())
        if k not in seen and p.is_file():
            seen.add(k)
            out.append(str(p))
    return out


def resolve(tb, ref: str) -> str:
    m = REF.match(ref.strip())
    if not m:
        raise ValueError(f"not a result reference: {ref!r}")
    tool, idx, ext = m.group("tool"), m.group("idx"), (m.group("ext") or "").lower()
    cands = _records(tb)
    if idx is not None:
        cands = [(s, c) for s, c in cands if s == "this session" and c.get("index") == int(idx)]
        if not cands:
            raise ValueError(f"{ref}: there is no call {idx} in this session (see recent_results)")
    for _, c in cands:
        if not c.get("ok") or (idx is None and c.get("tool") in NOT_RESULTS):
            continue
        if tool and c.get("tool") != tool:
            continue
        files = _files(tb, c)
        if ext:
            files = [f for f in files if f.lower().endswith(ext)]
        else:  # the main image; else the first output file
            files = [f for f in files if f.lower().endswith(".png")] or files
        if files:
            return files[0]
        if idx is not None:
            break
    what = (f"call {idx}" if idx else "any earlier call") + (f" of {tool}" if tool else "")
    raise ValueError(f"{ref}: {what} produced no existing {ext or 'image/output'} file (see recent_results)")


def resolve_refs(tb, args: Any) -> Tuple[Any, Dict[str, str]]:
    """Args with every ``@last``/``@call:N`` string replaced by a path (relative to the workdir)."""
    done: Dict[str, str] = {}

    def walk(v):
        if isinstance(v, str) and v.startswith("@") and REF.match(v.strip()):
            if v not in done:
                done[v] = tb.rel(resolve(tb, v))
            return done[v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v
    return walk(args), done


def _short(v: Any, n: int = 60) -> str:
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def register(tb) -> None:
    @tb.add("recent_results",
            "The last few results (newest first) with their files and how to refer to them. Any tool argument "
            "accepts '@last' (latest image), '@last.csv' (latest file of a type), '@last:TOOL' or '@call:N', e.g. "
            "edit_figure image='@last' to add points to the previous plot. Falls back to the previous session.",
            {"n": {"type": "integer", "description": "how many (default 5)"},
             "tool": {"type": "string", "description": "only results of this tool"}}, [])
    def recent_results(n: int = 5, tool: Optional[str] = None):
        lines, k = [], 0
        for sess, c in _records(tb):
            if not c.get("ok") or c.get("tool") in NOT_RESULTS or (tool and c.get("tool") != tool):
                continue
            files = _files(tb, c)
            if not files:
                continue
            args = ", ".join(f"{a}={_short(v)}" for a, v in (c.get("args") or {}).items()
                             if a not in ("code", "operations", "panels"))[:220]
            ref = f"@call:{c.get('index')}" if sess == "this session" else "(earlier session: use the path)"
            lines.append(f"- {ref} [{sess}] {c.get('tool')}({args})\n    files: "
                         + ", ".join(tb.rel(f) for f in files[:6]) + (f" (+{len(files) - 6})" if len(files) > 6 else ""))
            k += 1
            if k >= int(n):
                break
        if not lines:
            return "no earlier results with existing files" + (f" from {tool}" if tool else ""), []
        return ("Newest first. '@last' = the first image listed here.\n" + "\n".join(lines)), []

    @tb.add("rerun_call",
            "Repeat an earlier call with some arguments changed, e.g. the same diagram at another temperature. "
            "call: 'last' (latest result), 'last:TOOL', or a call number from recent_results; changes: {arg: new "
            "value}; remove: [args to drop].",
            {"call": {"description": "'last', 'last:TOOL' or a call number"},
             "changes": {"type": "object"}, "remove": {"type": "array", "items": {"type": "string"}}}, ["call"])
    def rerun_call(call: Any = "last", changes: Optional[Dict] = None, remove: Optional[List[str]] = None):
        spec = str(call).strip().lstrip("#@")
        tool = spec.split(":", 1)[1] if spec.startswith("last:") else None
        target = None
        for sess, c in _records(tb):
            if spec.isdigit():
                if sess == "this session" and c.get("index") == int(spec):
                    target = c
                    break
                continue
            if c.get("ok") and c.get("tool") not in NOT_RESULTS and (tool is None or c.get("tool") == tool):
                target = c
                break
        if target is None:
            raise ValueError(f"no earlier call matching {call!r} (see recent_results)")
        name = target["tool"]
        if name in NOT_RESULTS or name not in tb.tools:
            raise ValueError(f"call {target.get('index')} ({name}) cannot be re-run")
        args = {k: v for k, v in (target.get("args") or {}).items() if k not in set(remove or [])}
        args.update(changes or {})
        text, images, err = tb.call(name, args)
        head = (f"re-ran call {target.get('index')} ({name})"
                + (f" with changes {changes}" if changes else "") + (f", removed {remove}" if remove else ""))
        if err:
            raise RuntimeError(head + " - it failed:\n" + text)
        return head + ":\n" + text, images
