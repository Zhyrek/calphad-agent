"""Problem reports: self-contained bundles describing a request the agent could not satisfy.

A report is written for a *coding agent* (e.g. Claude Code working on this repository) so it can be handed over
as-is and fixed: the user's request, what the model said, every tool call of the turn with full arguments and
complete tracebacks, the source location of each tool involved, environment/versions (secrets redacted), copies of
small input files, the images produced, a replay script and a zip of everything.

Created three ways:
  * by the model, via the ``report_problem`` tool, when it cannot do what was asked (with its own diagnosis);
  * by the user, ``/report [note]`` in the terminal or the web UI's "Report problem" button;
  * automatically, when a turn ends with failed tool calls and a "cannot / unable" answer, or crashes.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import re
import shutil
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG_ROOT = Path(__file__).resolve().parents[2]         # repository root (contains calphad_tools/)

CANT_RE = re.compile(r"\b(can(?:no|')t|cannot|unable to|not able to|not (?:currently )?possible|isn't possible|"
                     r"no (?:tool|way) to|not supported|doesn't support|does not support|failed to|could not|"
                     r"couldn't)\b", re.I)


@dataclass
class TurnRecord:
    index: int
    user: str
    started: str
    call_start: int                      # index into toolbox.call_log (0-based) at turn start
    call_end: Optional[int] = None
    texts: List[str] = field(default_factory=list)   # every assistant text of the turn, in order
    final: str = ""
    error: str = ""                      # traceback if the turn itself crashed (API errors etc.)
    report: Optional[str] = None         # path of a report written for this turn


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return (s[:n].rstrip("-") or "report")


def _short_json(obj: Any, max_list: int = 12, max_str: int = 4000) -> Any:
    if isinstance(obj, dict):
        return {k: _short_json(v, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_list and all(not isinstance(v, (dict, list)) for v in obj):
            return obj[:max_list] + [f"... ({len(obj)} items total)"]
        return [_short_json(v, max_list, max_str) for v in obj]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + f"... ({len(obj)} chars)"
    return obj


def tool_source(tb, name: str) -> Optional[Dict[str, Any]]:
    """File/line of the function implementing a tool (so a coding agent can jump straight to it)."""
    t = tb.tools.get(name)
    if t is None:
        return None
    fn = t.fn
    try:
        f = Path(inspect.getsourcefile(fn)).resolve()
        line = inspect.getsourcelines(fn)[1]
    except (TypeError, OSError):
        return None
    try:
        rel = str(f.relative_to(PKG_ROOT))
    except ValueError:
        rel = str(f)
    # the tool function is usually a thin wrapper: also point at the module-level function it calls
    return {"file": rel, "line": line, "function": fn.__name__}


def _redacted(tb, obj):
    try:
        from ..provenance import redact
        secrets = []
        cfg = getattr(tb, "llm_cfg", None)
        if cfg is not None and getattr(cfg, "api_key", ""):
            secrets.append(cfg.api_key)
        return redact(obj, secrets)
    except Exception:  # noqa: BLE001 - provenance module optional
        return obj


def _environment() -> Dict[str, Any]:
    try:
        from ..provenance import environment
        return environment()
    except Exception:  # noqa: BLE001
        import platform
        import sys
        return {"python": sys.version.split()[0], "platform": platform.platform()}


def _input_files(tb, args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rec = getattr(tb, "provenance", None)
    if rec is not None and hasattr(rec, "input_hashes"):
        try:
            return rec.input_hashes(args)
        except Exception:  # noqa: BLE001
            pass
    out = {}

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and len(v) < 500:
            p = Path(tb.p(v))
            if p.is_file():
                out[str(p)] = {"size": p.stat().st_size}
            else:
                try:
                    q = Path(tb._find_tdb(v))
                    out[str(q)] = {"size": q.stat().st_size}
                except Exception:  # noqa: BLE001
                    pass
    walk(args)
    return out


def report_turns(agent, scope: str = "session") -> List[TurnRecord]:
    """The turns a report covers: "session" = every turn since the last /reset, "last" = the last turn."""
    turns = list(agent.turns)[getattr(agent, "session_start_turn", 0):]
    if not turns:
        raise ValueError("nothing to report yet - no turns in this conversation")
    return turns[-1:] if scope == "last" else turns


def _calls_of(tb, turns: List[TurnRecord]) -> List[Dict[str, Any]]:
    out = []
    for t in turns:
        end = t.call_end if t.call_end is not None else len(tb.call_log)
        out += [c for c in tb.call_log[t.call_start:end] if c.get("tool") not in ("report_problem", "package_feedback")]
    return out


def candidate_files(agent, turns: List[TurnRecord]) -> Dict[str, Dict[str, Any]]:
    """Files a report could include: {abs path: {kind: input|output, size, sha256, tools}}."""
    tb = agent.toolbox
    calls = _calls_of(tb, turns)
    files: Dict[str, Dict[str, Any]] = {}
    out_root = tb.out.resolve()
    for c in calls:
        for path, info in _input_files(tb, c.get("args") or {}).items():
            p = Path(path)
            p = (p if p.is_absolute() else tb.workdir / p).resolve()
            if not p.is_file() or "calphad_agent" in p.name:
                continue
            kind = "output" if (p.is_relative_to(out_root) or info.get("generated")) else "input"
            files.setdefault(str(p), {"kind": kind, "tools": []})["tools"].append(c.get("tool"))
    # outputs: images of the calls + files the session log recorded as written by them
    rec = getattr(tb, "provenance", None)
    idx0 = {id(c) for c in calls}
    for c in calls:
        for im in c.get("images") or []:
            if Path(im).is_file():
                files.setdefault(str(Path(im).resolve()), {"kind": "output", "tools": []})["tools"].append(c["tool"])
    if rec is not None:
        first = min((t.call_start for t in turns), default=0)
        last = max(((t.call_end if t.call_end is not None else len(tb.call_log)) for t in turns), default=0)
        for r in rec.records:
            if first < int(r.get("index") or 0) <= last and r.get("tool") not in ("report_problem", "package_feedback"):
                for o in r.get("outputs") or []:
                    q = Path(tb.p(o.get("path", "")))
                    if q.is_file() and not q.name.endswith(".provenance.json"):
                        files.setdefault(str(q.resolve()), {"kind": "output", "tools": []})["tools"].append(r["tool"])
    del idx0
    # lineage: which of the listed inputs each output was made from (transitively: an edited figure of a diagram
    # computed from a TDB derives from that TDB)
    made_from: Dict[str, set] = {}
    if rec is not None:
        for r in rec.records:
            srcs = {str(Path(tb.p(k)).resolve()) for k in (r.get("inputs") or {})}
            for o in r.get("outputs") or []:
                made_from.setdefault(str(Path(tb.p(o.get("path", ""))).resolve()), set()).update(srcs)
    for c in calls:
        srcs = {str((Path(q) if Path(q).is_absolute() else tb.workdir / q).resolve())
                for q in _input_files(tb, c.get("args") or {})}
        for im in c.get("images") or []:
            made_from.setdefault(str(Path(im).resolve()), set()).update(srcs)

    def roots(p, seen=()):
        out = set()
        for q in made_from.get(p, ()):
            if q in seen or q == p:
                continue
            if q in made_from and made_from[q]:
                out |= roots(q, seen + (p,)) or {q}
            else:
                out.add(q)
        return out
    for p, info in files.items():
        if info["kind"] == "output":
            info["derived_from"] = sorted(q for q in roots(p) if files.get(q, {}).get("kind") == "input")
    for p, info in files.items():
        st = Path(p).stat()
        info["size"] = st.st_size
        info["tools"] = sorted(set(info["tools"]))
        if st.st_size <= (1 << 30):
            from ..provenance import sha256_file
            info["sha256"] = sha256_file(p)
    return dict(sorted(files.items(), key=lambda kv: (kv[1]["kind"] != "input", kv[0])))


SHARE_CHOICES = ["include all listed files",
                 "include only the files I pick (answer with their numbers, e.g. 1,3)",
                 "no files (names, sizes and SHA-256 fingerprints only)",
                 "no files and no tool-result text (most private)"]


def consent_question(tb, files: Dict[str, Dict[str, Any]]) -> str:
    lines = ["Before packaging: which of these files may go into the report you will send to the developers?",
             "(A database may be proprietary. Tool results in the report text can also quote database contents, "
             "e.g. phase names and parameter values - the last option leaves them out.)"]
    for n, (p, info) in enumerate(files.items(), 1):
        der = info.get("derived_from") or []
        lines.append(f"  {n}. [{info['kind']}] {tb.rel(p)}  ({info['size'] / 1024:.0f} kB; "
                     + (f"made from {', '.join(Path(q).name for q in der)}" if der else
                        f"used by {', '.join(info['tools'][:3])}") + ")")
    if not files:
        lines.append("  (no files were used or produced)")
    return "\n".join(lines)


def parse_consent(answer: str, files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """User's answer -> {"files": [paths], "results_text": bool, "consent": text}.  Unclear answers share nothing."""
    a = (answer or "").strip()
    paths = list(files)
    low = a.lower()
    nums = [int(x) for x in re.findall(r"\d+", a)]
    if a == SHARE_CHOICES[0] or low in ("1", "all", "yes", "y", "include all"):
        return {"files": paths, "results_text": True, "consent": "all listed files"}
    if a == SHARE_CHOICES[3] or low in ("4", "private", "none, no text", "nothing"):
        return {"files": [], "results_text": False, "consent": "no files, no tool-result text"}
    if a == SHARE_CHOICES[2] or low in ("3", "none", "no", "n", "no files"):
        return {"files": [], "results_text": True, "consent": "no files (fingerprints only)"}
    picked = [paths[k - 1] for k in nums if 1 <= k <= len(paths)] if not low.startswith("2") or len(nums) > 1 else []
    if low.startswith("2") and len(nums) > 1:
        picked = [paths[k - 1] for k in nums[1:] if 1 <= k <= len(paths)]
    if picked:
        return {"files": sorted(set(picked), key=paths.index), "results_text": True,
                "consent": "files picked by the user: " + ", ".join(str(paths.index(q) + 1) for q in picked)}
    return {"files": [], "results_text": True, "consent": f"unclear answer ({a[:40]!r}): no files shared"}


def ask_consent(agent, files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Ask the person at the keyboard (never the model) which files may be shared."""
    if not files:
        return {"files": [], "results_text": True, "consent": "no files involved"}
    if not getattr(agent, "interactive", True):
        return {"files": [], "results_text": True, "consent": "nobody asked (non-interactive): no files shared"}
    ans = agent.ui.ask(consent_question(agent.toolbox, files), SHARE_CHOICES)
    return parse_consent(ans, files)


def _reviews(tb, since: str) -> List[Dict[str, Any]]:
    p = tb.out / "python_reviews.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if str(o.get("time", "")) >= since:
                out.append({k: o.get(k) for k in ("time", "verdict", "decision", "reason", "flags", "ran")
                            if k in o})
    return out


def _decisions(tb, turns: List[TurnRecord], text_ok: bool) -> List[Dict[str, Any]]:
    """Decision-log entries tools recorded in the covered turns (combine_databases: who chose what, why).  They
    quote phase / parameter names, so they follow the user's choice about tool-result text."""
    rec = getattr(tb, "provenance", None)
    if rec is None or not hasattr(rec, "decisions_of"):
        return []
    first = min(t.call_start for t in turns)
    last = max((t.call_end if t.call_end is not None else len(tb.call_log)) for t in turns)
    out = []
    for d in rec.decisions_of(first, last):
        if text_ok:
            out.append({k: d.get(k) for k in ("call_index", "tool", "id", "step", "subject", "chosen", "who",
                                              "reason", "evidence", "affected")})
        else:
            out.append({k: d.get(k) for k in ("call_index", "tool", "id", "who")} | {"text": "(withheld)"})
    return out


def write_report(agent, turn, reason: str, created_by: str, note: str = "",
                 diagnosis: Optional[Dict[str, Any]] = None, share: Optional[Dict[str, Any]] = None) -> str:
    """Write the report bundle; returns the zip path (the folder sits next to it).

    ``turn``: a TurnRecord or a list of them (the conversation the report covers).  ``share``: the user's
    consent {"files": [...], "results_text": bool, "consent": str}; None shares no files (reports nobody
    asked for - automatic or model-initiated without a user present - never copy the user's files)."""
    tb = agent.toolbox
    cfg = agent.cfg
    turns = list(turn) if isinstance(turn, (list, tuple)) else [turn]
    share = share or {"files": [], "results_text": True, "consent": f"{created_by} report: files not shared"}
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    title = (diagnosis or {}).get("summary") or note or turns[-1].user
    folder = tb.out / "problem_reports" / f"{stamp}_{_slug(title)}"
    folder.mkdir(parents=True, exist_ok=True)
    calls = _calls_of(tb, turns)

    # ---- files: only what the user agreed to share ------------------------------------------------------
    files = candidate_files(agent, turns)
    allowed = {str(Path(p).resolve()) for p in share.get("files") or []}
    limit = float(getattr(cfg.agent, "report_max_input_mb", 5.0)) * 1024 * 1024 * 4
    copied, withheld, too_big = [], [], []
    for p, info in files.items():
        entry = {"name": Path(p).name, "kind": info["kind"], "size": info["size"],
                 "sha256": info.get("sha256", ""), "path": tb.rel(p)}
        if p not in allowed:
            withheld.append(entry)
            continue
        if info["size"] > limit:
            too_big.append(entry)
            continue
        sub = "inputs" if info["kind"] == "input" else "outputs"
        dst = folder / sub / Path(p).name
        k = 2
        while dst.exists():
            dst = folder / sub / f"{Path(p).stem}_{k}{Path(p).suffix}"
            k += 1
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dst)
        copied.append({**entry, "copy": f"{sub}/{dst.name}"})

    # ---- replay script for the covered calls --------------------------------------------------------------
    replay_note = "no tool calls"
    if calls and "export_session" in tb.tools:
        first = min(t.call_start for t in turns) + 1
        last = max((t.call_end if t.call_end is not None else len(tb.call_log)) for t in turns)
        txt, _, err = tb._call("export_session", {"format": "script", "steps": f"{first}-{last}",
                                                  "include_failed": True,
                                                  "output": str(folder / "replay.py")})
        replay_note = ("replay.py (replays the tool calls without an LLM"
                       + ("; withheld input files must be supplied)" if withheld else ")")) if not err else \
            f"replay export failed: {txt.splitlines()[0] if txt else '?'}"

    # ---- structured record -------------------------------------------------------------------------------
    text_ok = bool(share.get("results_text", True))

    def res_text(c):
        if not text_ok:
            return "(withheld at the user's request)"
        return c.get("result_full") if c.get("result_full") is not None else c.get("result_preview")
    questions = [{"question": (c.get("args") or {}).get("question"), "answer": res_text(c)}
                 for c in calls if c.get("tool") == "ask_user"]
    llm = cfg.llm
    data = {
        "report_version": 2,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "created_by": created_by, "reason": reason, "user_note": note, "diagnosis": diagnosis or {},
        "turns": [{"n": t.index, "started": t.started, "request": t.user, "assistant_texts": t.texts,
                   "final_answer": t.final, "error": t.error} for t in turns],
        "request": turns[-1].user, "final_answer": turns[-1].final, "assistant_texts": turns[-1].texts,
        "turn_error": turns[-1].error,
        "questions_to_user": questions,
        "decisions": _decisions(tb, turns, text_ok),
        "python_reviews": _reviews(tb, turns[0].started),
        "llm": {"provider": llm.provider, "model": llm.model, "api": llm.api, "base_url": llm.base_url or "",
                "profile": getattr(cfg, "profile", "") or ""},
        "agent": {"workdir": str(tb.workdir), "output_dir": str(tb.out), "python_tool": tb.python_tool,
                  "n_tools": len(tb.tools)},
        "tool_calls": [{"n": i + 1, "tool": c.get("tool"), "ok": c.get("ok"), "seconds": c.get("seconds"),
                        "args": _short_json(c.get("args")), "images": c.get("images"), "result": res_text(c),
                        "source": tool_source(tb, c.get("tool"))} for i, c in enumerate(calls)],
        "sharing": {"consent": share.get("consent", ""), "results_text": text_ok},
        "files_included": copied, "files_withheld": withheld, "files_too_big": too_big,
        # compatibility with report_version 1 readers
        "inputs_copied": [f["copy"].split("/", 1)[1] for f in copied if f["kind"] == "input"],
        "outputs_copied": [f["copy"].split("/", 1)[1] for f in copied if f["kind"] == "output"],
        "inputs": {f["path"]: {"size": f["size"], "sha256": f["sha256"]} for f in copied + withheld + too_big
                   if f["kind"] == "input"},
        "environment": _environment(),
    }
    data = _redacted(tb, data)
    (folder / "report.json").write_text(json.dumps(data, indent=1, default=str))
    (folder / "REPORT.md").write_text(_markdown(data, replay_note))
    zip_path = shutil.make_archive(str(folder), "zip", root_dir=folder.parent, base_dir=folder.name)
    for t in turns[-1:]:
        t.report = zip_path
    return zip_path


def _markdown(d: Dict[str, Any], replay_note: str) -> str:
    head = (d["diagnosis"] or {}).get("summary") or d["user_note"] or d["request"][:80]
    L = [f"# Problem report: {head}", "",
         f"- created: {d['created']} by **{d['created_by']}** ({d['reason']})",
         f"- model: {d['llm']['provider']} / {d['llm']['model']}"
         + (f" (profile {d['llm']['profile']})" if d['llm']['profile'] else ""),
         f"- python_tool: {d['agent']['python_tool']}; tools available: {d['agent']['n_tools']}",
         f"- workdir: `{d['agent']['workdir']}`",
         f"- sharing: {d['sharing']['consent']}" + ("" if d["sharing"]["results_text"] else
                                                   " (tool-result text withheld)"), ""]
    if d["user_note"]:
        L += ["## Why the user is not satisfied", "", "> " + d["user_note"].replace("\n", "\n> "), ""]
    dg = d["diagnosis"]
    if dg:
        L += ["## Model's diagnosis", ""]
        for k in ("kind", "summary", "what_is_missing", "attempted", "suggested_fix"):
            if dg.get(k):
                L.append(f"- **{k.replace('_', ' ')}:** {dg[k]}")
        L.append("")
    L += [f"## Conversation ({len(d['turns'])} request(s))", ""]
    for t in d["turns"]:
        L += [f"### Request {t['n']} ({t['started']})", "", "> " + t["request"].replace("\n", "\n> "), ""]
        if t["error"]:
            L += ["The turn crashed:", "", "```", t["error"].rstrip(), "```", ""]
        if len(t["assistant_texts"]) > 1:
            L += ["<details><summary>Intermediate assistant messages (the agent's reasoning / decisions)</summary>",
                  ""]
            for x in t["assistant_texts"][:-1]:
                if x.strip():
                    L += ["```text", x.strip()[:3000], "```"]
            L += ["</details>", ""]
        L += ["Answer:", "", "```text", (t["final_answer"] or "(none)").rstrip()[:6000], "```", ""]
    if d["questions_to_user"]:
        L += ["## Questions the agent asked the user", ""]
        for q in d["questions_to_user"]:
            L += [f"- **Q:** {q['question']}", f"  **A:** {q['answer']}"]
        L.append("")
    if d.get("decisions"):
        L += ["## Decisions recorded by tools (who chose, why)", ""]
        for x in d["decisions"]:
            if "text" in x:
                L.append(f"- call {x['call_index']} `{x['id']}` ({x['who']}): {x['text']}")
            else:
                L.append(f"- call {x['call_index']} `{x['id']}` -> {x.get('chosen')} ({x.get('who')}: "
                         f"{x.get('reason') or 'not given'})")
        L.append("")
    if d["python_reviews"]:
        L += ["## Safety reviews of generated Python", ""]
        L += [f"- {r.get('time')}: {r.get('verdict') or r.get('decision')} - {r.get('reason', '')}"
              for r in d["python_reviews"]]
        L.append("")
    L += [f"## Tool calls ({len(d['tool_calls'])})", ""]
    for c in d["tool_calls"]:
        src = c["source"] or {}
        L += [f"### {c['n']}. `{c['tool']}` - {'ok' if c['ok'] else '**FAILED**'} ({c['seconds']} s)"
              + (f" - source `{src.get('file')}:{src.get('line')}`" if src else ""), "",
              "```json", json.dumps(c["args"], indent=1, default=str)[:6000], "```", ""]
        res = (c["result"] or "").rstrip()
        if res:
            L += [("Result (full):" if not c["ok"] else "Result:"), "", "```text",
                  res if not c["ok"] else res[:2500] + ("\n...[truncated]" if len(res) > 2500 else ""), "```", ""]
        if c["images"]:
            L.append("Images: " + ", ".join(Path(i).name for i in c["images"]))
            L.append("")
    L += ["## Files", "", f"Sharing: **{d['sharing']['consent']}**", ""]
    if d["files_included"]:
        L += ["Included (copies in this bundle):", ""]
        L += [f"- `{f['copy']}` ({f['kind']}, from `{f['path']}`, {f['size']} B, sha256 {f['sha256'][:16]})"
              for f in d["files_included"]]
        L.append("")
    if d["files_withheld"] or d["files_too_big"]:
        L += ["Not included (ask the user for them if needed; the fingerprints identify the exact files):", ""]
        L += [f"- `{f['path']}` ({f['kind']}, {f['size']} B, sha256 {f['sha256'][:16]})"
              + (" - too big" if f in d["files_too_big"] else "") for f in d["files_withheld"] + d["files_too_big"]]
        L.append("")
    L += [f"- {replay_note}", "- report.json: everything above, machine-readable", ""]
    env = d["environment"]
    L += ["## Environment", "", "```json", json.dumps(env, indent=1, default=str)[:3000], "```", "",
          "## For the coding agent fixing this", "",
          "1. Reproduce: from the repository root run `.venv/bin/python <this folder>/replay.py --strict` "
          "(copy `inputs/` files back to their original paths if they are missing), or re-ask the agent with "
          "`.venv/bin/calphad-agent --once \"<request above>\"`.",
          "2. Start with the user's statement of why the answer is not good, then the source locations of the "
          "relevant tools. If the request needs a capability no tool provides, add it (new tool or op) rather "
          "than relying on run_python.",
          "3. Add a regression test under `tests/` built from this report's inputs, and run "
          "`.venv/bin/python -m pytest -q tests`.", ""]
    return "\n".join(L)


def looks_failed(turn: TurnRecord, calls: List[Dict[str, Any]]) -> Optional[str]:
    """Reason string if a finished turn looks like it did not satisfy the request, else None."""
    if turn.error:
        return "the turn crashed"
    failed = [c for c in calls if not c.get("ok")]
    unresolved = failed and not calls[-1].get("ok", True)       # last call failed and was not retried
    says_cant = bool(CANT_RE.search(turn.final or ""))
    if failed and (says_cant or unresolved):
        return f"{len(failed)} failed tool call(s) and the answer " + (
            "says it could not do it" if says_cant else "ended right after a failure")
    return None


def register_feedback_tool(agent) -> None:
    """package_feedback: the user is not satisfied - agree with them, conversationally, on what to share, then
    write the reproducible bundle.  The final list is confirmed by the user through the UI, not by the model."""
    tb = agent.toolbox
    S = {"type": "string"}
    state: Dict[str, Any] = {}

    @tb.add("package_feedback",
            "The user is unhappy with an answer and wants it sent to the developers. Step 1 action=list: the files "
            "the report could include (inputs, outputs, what each output was made from). Then TALK with the user "
            "about each file - databases may be proprietary, and outputs made from them (figures, CSVs, tables) "
            "can reveal their content - until you know exactly what may be shared. Step 2 action=write with "
            "user_statement (their words on why the answer is not good), include (numbers from the list, [] for "
            "none), include_results_text, and consent_notes (what they said about sharing). The user confirms "
            "the final list themselves before anything is written.",
            {"action": {"type": "string", "enum": ["list", "write"]},
             "scope": {"type": "string", "enum": ["session", "last"],
                       "description": "session (default): every request since the last reset; last: last one"},
             "user_statement": S, "include": {"type": "array", "items": {"type": "integer"}},
             "include_results_text": {"type": "boolean", "description": "tool-result text can quote database "
                                                                       "contents (default true)"},
             "consent_notes": S}, ["action"])
    def package_feedback(action, scope="session", user_statement="", include=None, include_results_text=True,
                         consent_notes=""):
        turns = report_turns(agent, scope)
        # the current (in-progress) turn is the complaint itself: report what came before it, plus it
        cur = agent.current_turn
        done = [t for t in turns if t is not cur] or turns
        files = candidate_files(agent, done)
        state["files"] = files
        if action == "list":
            txt = [consent_question(tb, files).split("\n", 2)[-1] if files else "(no files involved)",
                   f"covers {len(done)} request(s): " + " | ".join(t.user[:60] for t in done),
                   "Now agree with the user, file by file where needed, on what may be shared (a proprietary "
                   "database, figures / tables made from it). Then call action=write."]
            return "\n".join(txt), []
        if not user_statement.strip():
            return ("user_statement is required: ask the user why the answer is not good and pass their "
                    "words"), []
        paths = list(files)
        bad = [k for k in (include or []) if not (1 <= int(k) <= len(paths))]
        if bad:
            return f"no file number(s) {bad}: use the numbers of action=list (1..{len(paths)})", []
        chosen = [paths[int(k) - 1] for k in include or []]
        summary = ["Write the report with:", f"  your statement: {user_statement.strip()[:300]}"]
        summary += [f"  INCLUDE {tb.rel(q)}" for q in chosen] or ["  no files"]
        summary += [f"  withhold {tb.rel(q)} (name, size and fingerprint only)" for q in paths if q not in chosen]
        summary.append("  tool-result text: " + ("included" if include_results_text else "WITHHELD"))
        if not getattr(agent, "interactive", True):
            chosen, ok = [], True
            note = "non-interactive: nobody to confirm, so no files were included"
        else:
            ans = agent.ui.ask("\n".join(summary) + "\nIs this OK to package for the developers?",
                               ["yes, write the report", "no, change something"])
            ok = str(ans).strip().lower().startswith(("y", "ok"))
            note = consent_notes
        if not ok:
            return ("the user did not confirm the list; ask what to change, then call action=write again"), []
        share = {"files": chosen, "results_text": bool(include_results_text),
                 "consent": ("confirmed by the user: " + (note or "as listed")) if chosen or note else
                 "confirmed by the user: no files"}
        path = write_report(agent, done, reason="user not satisfied with the answer", created_by="user",
                            note=user_statement.strip(), share=share)
        agent.ui.notice("report", f"feedback report saved: {tb.rel(path)}")
        return (f"feedback report saved: {tb.rel(path)} ({len(chosen)} file(s) included, "
                f"{len(paths) - len(chosen)} withheld). Tell the user they can send this zip to the developers."), []


def register_report_tool(agent) -> None:
    """The model-facing tool (registered by the Agent: it needs the current turn)."""
    tb = agent.toolbox
    S = {"type": "string"}

    @tb.add("report_problem", "Call this (once) when you cannot fully satisfy the user's request - a capability is "
            "missing, a tool fails/has a bug, or results look wrong. It saves a problem report for the developers "
            "(request, your diagnosis, all tool calls with tracebacks, inputs, replay script). Then tell the user "
            "the report path and what you could do instead.",
            {"summary": S, "kind": {"type": "string", "enum": ["missing_capability", "tool_bug", "wrong_result",
                                                                "data_problem", "other"]},
             "what_is_missing": S, "attempted": S, "suggested_fix": S}, ["summary", "kind"])
    def report_problem(summary, kind, what_is_missing="", attempted="", suggested_fix=""):
        turn = agent.current_turn
        if turn is None:
            return "no active turn", []
        diag = {"summary": summary, "kind": kind, "what_is_missing": what_is_missing, "attempted": attempted,
                "suggested_fix": suggested_fix}
        path = write_report(agent, turn, reason=f"model reported: {kind}", created_by="model", diagnosis=diag)
        agent.ui.notice("report", f"problem report saved: {tb.rel(path)}")
        return f"problem report saved: {tb.rel(path)} (folder next to it). Tell the user.", []
