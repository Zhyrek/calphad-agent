"""Background jobs: run a long Python function in a detached subprocess and follow it from the agent.

API (used by the ESPEI tools and other plug-ins)::

    job_id = submit(tb, label, "package.module:function", {"arg": 1})

starts ``python -m calphad_tools.jobs run <jobdir>``, which imports the target and calls
``target(**kwargs, progress=progress)``.  ``progress(msg, frac=None, **metrics)`` appends a line to
``progress.jsonl`` and updates ``status.json``.  The return value (JSON-safe; ideally
``{"text": ..., "files": [...]}``) is written to ``result.json``.

Job directory ``<tb.out>/jobs/<job_id>/``:

* ``spec.json``     - label, target, kwargs, submitting tool call (session, call index, tool, args)
* ``status.json``   - state (queued/running/done/failed/stopped), pid, start/end, last message, frac, metrics
* ``progress.jsonl``- every progress call
* ``log.txt``       - stdout/stderr of the job + Python logging
* ``result.json``   - the target's return value (or the traceback of a failure)
* ``stop``          - marker written by :func:`stop` (the job then reports ``stopped``)

The child runs in its own session / process group (survives the agent exiting, never reads the agent's
stdin; :func:`stop` signals the whole group, e.g. dask workers).  Figures saved by jobs are editable like any
other tool figure (the ``CALPHAD_FIGSTORE`` savefig hook is installed in the child).

Flux-backed jobs (:func:`submit_flux`): batch jobs already submitted with ``flux batch`` (e.g. ESPEI generation ->
MCMC chained with ``--dependency=afterok``) are registered the same way (``spec.json`` with ``backend: flux`` and the
Flux job ids, scripts and commands).  There is no local process: ``status`` asks Flux (``flux jobs``; the job's
eventlog for ids Flux no longer lists), a ``flux_hook`` reads progress from the run directory (ESPEI: iterations in
``mcmc.log``, lnprob), ``stop`` runs ``flux cancel``, and a failed step reports the tail of its batch output file and
Flux's exit status.  Records live on disk, so they survive an agent restart.

Tools: ``list_jobs``, ``job_status``, ``stop_job``, ``wait_job`` (``until="done"``: keep waiting, with a progress
line, until the job finishes - Ctrl-C / the web UI's "stop waiting" ends the wait, not the job).  A job may name a
``status_hook`` (``module:function(tb, jobdir, spec, status) -> (text, images)``) that ``job_status`` calls for extra
job-specific reporting (ESPEI MCMC: lnprob convergence plot).

Provenance: ``spec.json`` records the submitting call; :func:`producer_for` maps any file a job produced
(listed in ``result.json`` ``files`` or written inside the job directory) back to that call, which
``codegen.find_producer`` (show_code) uses.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import secrets
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

JOBS_DIR = "jobs"
STATES = ("queued", "running", "done", "failed", "stopped")
FINAL = {"done", "failed", "stopped"}
WAIT_CAP_S = 600.0
STOP_GRACE_S = 16.0                             # SIGTERM, again after half of it, then SIGKILL
_CHILDREN: Dict[str, subprocess.Popen] = {}     # jobs started by this process (reaped via poll())


# --------------------------------------------------------------------------- small helpers
def _now() -> float:
    return time.time()


def _iso(t: Optional[float]) -> Optional[str]:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) if t else None


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_default))


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    for _ in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, OSError):
            time.sleep(0.05)
    return default


def jobs_root(tb_or_out) -> Path:
    out = getattr(tb_or_out, "out", tb_or_out)
    return Path(out) / JOBS_DIR


def job_dir(tb, job_id: str) -> Path:
    d = jobs_root(tb) / str(job_id).strip()
    if not (d / "spec.json").exists():
        known = [p.name for p in sorted(jobs_root(tb).glob("*/spec.json"))][-8:] if jobs_root(tb).exists() else []
        raise ValueError(f"no job {job_id!r} (known: {[Path(k).parent.name for k in known] or 'none'}; "
                         "see list_jobs)")
    return d


# --------------------------------------------------------------------------- submit
def _new_job_dir(tb) -> Tuple[str, Path]:
    root = jobs_root(tb)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        job_id = time.strftime("%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        try:
            (root / job_id).mkdir()
            return job_id, root / job_id
        except FileExistsError:
            continue
    raise RuntimeError("could not create a job directory")


def _submitted_by(tb, tool: Optional[str], tool_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prov = getattr(tb, "provenance", None)
    pend = getattr(prov, "pending", None) or {}
    call_log = getattr(tb, "call_log", [])
    return {"session": getattr(prov, "session_id", None), "call_index": len(call_log) + 1,
            "tool": tool or pend.get("tool"), "args": tool_args if tool_args is not None else pend.get("args")}


def submit(tb, label: str, target: str, kwargs: Dict[str, Any], *, status_hook: Optional[str] = None,
           tool: Optional[str] = None, tool_args: Optional[Dict[str, Any]] = None,
           cwd: Optional[str] = None) -> str:
    """Start ``target(**kwargs, progress=...)`` ("module:function") in a detached subprocess; return the job id.

    The submitting tool call (name, args, call index, session) is taken from the running tool call when
    ``tool``/``tool_args`` are not given, so show_code / provenance can map the job's files back to it."""
    if ":" not in target:
        raise ValueError("target must be 'module:function'")
    job_id, d = _new_job_dir(tb)
    by = _submitted_by(tb, tool, tool_args)
    spec = {"job_id": job_id, "label": label, "target": target, "kwargs": _jsonable(kwargs),
            "created": _iso(_now()), "workdir": str(getattr(tb, "workdir", os.getcwd())), "out": str(tb.out),
            "cwd": str(cwd or getattr(tb, "workdir", os.getcwd())), "python": sys.executable,
            "figstore": str(Path(tb.out) / ".figures"), "status_hook": status_hook, "submitted_by": by}
    _write_json(d / "spec.json", spec)
    _write_json(d / "status.json", {"job_id": job_id, "label": label, "state": "queued", "pid": None,
                                    "created": spec["created"], "start": None, "end": None, "message": "queued",
                                    "frac": None, "metrics": {}, "n_progress": 0})
    from .hashseed import child_env
    env = child_env()                           # PYTHONHASHSEED=0 unless set (reproducible mapping)
    env["CALPHAD_FIGSTORE"] = spec["figstore"]
    env["PYTHONUNBUFFERED"] = "1"
    root_pkg = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root_pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("MPLBACKEND", None)
    env["MPLBACKEND"] = "Agg"
    with open(d / "log.txt", "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "calphad_tools.jobs", "run", str(d)], cwd=spec["cwd"],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True, close_fds=True)
    _CHILDREN[job_id] = p
    (d / "pid").write_text(str(p.pid))          # status.json belongs to the child from now on
    return job_id


def submit_flux(tb, label: str, steps: List[Dict[str, Any]], *, run_dir: str, kwargs: Optional[Dict] = None,
                flux: Optional[Dict[str, Any]] = None, status_hook: Optional[str] = None,
                flux_hook: Optional[str] = None, tool: Optional[str] = None,
                tool_args: Optional[Dict[str, Any]] = None) -> str:
    """Register Flux batch jobs that were already submitted as one registry job; return its id.

    ``steps``: ``[{"step": "generate", "id": "f...", "script": "gen.flux", "command": [...],
    "output_prefix": "espei-gen-NAME"}, ...]`` in dependency order.  ``flux``: anything else to record (resources,
    detection, the submit script).  ``flux_hook`` ("module:function(tb, jobdir, spec, status, final) -> result or
    None") adds progress from the run directory while the jobs run and builds result.json when they end."""
    job_id, d = _new_job_dir(tb)
    spec = {"job_id": job_id, "label": label, "backend": "flux", "target": "flux batch",
            "kwargs": _jsonable({"run_dir": str(run_dir), **(kwargs or {})}), "created": _iso(_now()),
            "workdir": str(getattr(tb, "workdir", os.getcwd())), "out": str(tb.out), "cwd": str(run_dir),
            "flux": _jsonable({"steps": steps, **(flux or {})}), "status_hook": status_hook, "flux_hook": flux_hook,
            "submitted_by": _submitted_by(tb, tool, tool_args)}
    _write_json(d / "spec.json", spec)
    _write_json(d / "status.json", {
        "job_id": job_id, "label": label, "state": "queued", "backend": "flux", "pid": None,
        "created": spec["created"], "start": None, "end": None, "frac": None, "metrics": {}, "n_progress": 0,
        "message": "submitted to Flux: " + ", ".join(f"{s['step']} {s['id']}" for s in steps),
        "flux": {s["step"]: {"id": s["id"], "status": "SUBMITTED"} for s in steps}})
    return job_id


def flux_output_file(run_dir: Path, prefix: Optional[str], jid: Optional[str] = None) -> Optional[Path]:
    """The batch output file of a step (``--output=<prefix>.{{id}}.out``): the one with this id, else the newest."""
    if not prefix:
        return None
    hits = sorted(Path(run_dir).glob(f"{prefix}.*.out"), key=lambda p: p.stat().st_mtime)
    if jid:
        from .hpc import norm_id
        same = [h for h in hits if norm_id(h.name[len(prefix) + 1:-4]) == norm_id(jid)]
        if same:
            return same[-1]
    return hits[-1] if hits else None


def _flux_status(tb, d: Path, spec: Dict[str, Any], st: Dict[str, Any]) -> Dict[str, Any]:
    if st.get("state") in FINAL:
        return st
    from . import hpc
    steps = (spec.get("flux") or {}).get("steps") or []
    states = hpc.flux_job_states([s["id"] for s in steps])
    if states is None:
        st["message"] = "Flux does not answer right now (flux jobs timed out); state unknown - try again"
        return st
    per = {s["step"]: {"id": s["id"], **states.get(s["id"], {"status": "UNKNOWN"})} for s in steps}
    codes = [per[s["step"]]["status"] for s in steps]
    mapped = [hpc.FLUX_STATE.get(c, "unknown") for c in codes]
    if "failed" in mapped:
        state = "failed"
    elif "stopped" in mapped:
        state = "stopped"
    elif mapped and mapped[-1] == "done" and all(m in ("done", "unknown") for m in mapped):
        state = "done"                              # (an earlier step Flux no longer lists must have succeeded)
    elif "running" in mapped:
        state = "running"
    elif all(m == "unknown" for m in mapped):
        state = st.get("state") or "queued"         # Flux lists none of them: keep the last known state
    else:
        state = "queued"
    st["flux"] = per
    st["state"] = state
    now = _now()
    if state == "running" and not st.get("start"):
        st["start"] = _iso(now)
    active = next((s for s in steps if per[s["step"]]["status"] not in hpc.FLUX_FINAL), steps[-1] if steps else None)
    if active is not None:
        st["message"] = f"{active['step']} job {active['id']}: {per[active['step']]['status']}"
    final = state in FINAL
    res: Optional[Dict[str, Any]] = None
    hook = spec.get("flux_hook")
    if hook:
        try:
            mod, fn = hook.split(":", 1)
            res = getattr(importlib.import_module(mod), fn)(tb, d, spec, st, final)
        except Exception as e:  # noqa: BLE001 - progress extras must never break the state
            st["hook_error"] = f"{type(e).__name__}: {e}"[:300]
    if final:
        st["end"] = st.get("end") or _iso(now)
        res = dict(res or {})
        bad = next((s for s in steps if hpc.FLUX_STATE.get(per[s["step"]]["status"]) in ("failed", "stopped")), None)
        if bad is not None:
            b = per[bad["step"]]
            out = flux_output_file(Path(spec["kwargs"]["run_dir"]), bad.get("output_prefix"), bad["id"])
            why = (f"{bad['step']} job {bad['id']}: Flux status {b['status']}"
                   + (f", exit code {b['returncode']}" if b.get("returncode") is not None else "")
                   + (f", exception: {b['exception']}" if b.get("exception") else ""))
            st["message"] = why
            if state == "failed":
                st["error"] = why
                res["error"] = why
            if out is not None:
                res["output_file"] = str(out)
                res["output_tail"] = _tail(out, 30)
        _write_json(d / "result.json", _jsonable(res))
    _write_json(d / "status.json", st)
    return st


# --------------------------------------------------------------------------- child side
class _Reporter:
    """The ``progress`` callable handed to job targets.

    Stopping: ``stop_requested`` becomes True on the first SIGTERM.  By default the signal also raises
    :class:`JobStopped` (a KeyboardInterrupt) in the target; if that exception gets swallowed somewhere, the next
    ``progress()`` call raises it again.  A target that sets ``progress.cooperative = True`` is not interrupted by
    the first SIGTERM - it should check ``progress.stop_requested`` at safe points (e.g. between iterations) and
    return; a second SIGTERM (sent by :func:`stop` half-way through its grace period) raises in any case."""

    def __init__(self, d: Path):
        self.d = d
        self.status = _read_json(d / "status.json", {})
        self.stop_requested = False
        self.cooperative = False
        self.n_term = 0
        self._reraised = False

    def update(self, **kw) -> None:
        self.status.update(kw)
        _write_json(self.d / "status.json", self.status)

    def __call__(self, msg: str, frac: Optional[float] = None, **metrics) -> None:
        rec = {"t": round(_now(), 3), "msg": str(msg)}
        if frac is not None:
            frac = float(min(1.0, max(0.0, frac)))
            rec["frac"] = frac
        metrics = _jsonable(metrics)
        if metrics:
            rec["metrics"] = metrics
        with open(self.d / "progress.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        st = self.status
        st["message"] = str(msg)
        st["last_progress"] = _iso(rec["t"])
        if frac is not None:
            st["frac"] = frac
        if metrics:
            st.setdefault("metrics", {}).update(metrics)
        st["n_progress"] = int(st.get("n_progress", 0)) + 1
        _write_json(self.d / "status.json", st)
        logging.getLogger("calphad_tools.jobs").info("progress: %s%s", msg,
                                                     f" ({100 * frac:.0f}%)" if frac is not None else "")
        if self.stop_requested and not self.cooperative and not self._reraised:
            self._reraised = True                  # the SIGTERM exception was swallowed by the target: again
            raise JobStopped("stop requested")


class JobStopped(KeyboardInterrupt):
    """Raised in the job by SIGTERM (a KeyboardInterrupt, so e.g. ESPEI saves its chains before exiting)."""


def _run_child(d: Path) -> int:
    spec = _read_json(d / "spec.json")
    rep = _Reporter(d)
    rep.update(state="running", pid=os.getpid(), pgid=os.getpgid(0), start=_iso(_now()), message="starting")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def on_term(signum, frame):
        rep.n_term += 1
        rep.stop_requested = True
        if rep.n_term == 1 and rep.cooperative:
            return                           # the target stops itself at its next safe point
        if rep.n_term <= 2:                  # later signals: don't interrupt the clean-up again
            raise JobStopped(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_term)
    try:
        from .figures import install_from_env
        install_from_env()                    # editable copies of figures the job saves
    except Exception:  # noqa: BLE001
        pass
    print(f"[job {spec['job_id']}] {spec['label']}: {spec['target']} pid {os.getpid()}", flush=True)
    t0 = _now()
    try:
        mod, fn = spec["target"].split(":", 1)
        target: Callable = getattr(importlib.import_module(mod), fn)
        res = target(**spec["kwargs"], progress=rep)
        if res is None:
            res = {}
        elif not isinstance(res, dict):
            res = {"text": str(res)} if isinstance(res, str) else {"value": res}
        was_stopped = rep.stop_requested or (d / "stop").exists()
        _write_json(d / "result.json", _jsonable(res))
        rep.update(state="stopped" if was_stopped else "done", end=_iso(_now()),
                   seconds=round(_now() - t0, 2), message=("stopped (partial result)" if was_stopped
                                                           else rep.status.get("message") or "done"))
        print(f"[job {spec['job_id']}] {'stopped' if was_stopped else 'done'} in {_now() - t0:.1f} s", flush=True)
        return 0
    except BaseException as e:  # noqa: BLE001 - everything is reported to status.json
        tb_text = traceback.format_exc()
        was_stopped = isinstance(e, KeyboardInterrupt) or rep.stop_requested or (d / "stop").exists()
        print(tb_text, file=sys.stderr, flush=True)
        state = "stopped" if was_stopped else "failed"
        _write_json(d / "result.json", {"error": f"{type(e).__name__}: {e}", "traceback": tb_text})
        rep.update(state=state, end=_iso(_now()), seconds=round(_now() - t0, 2),
                   message=("stopped (SIGTERM)" if was_stopped else f"failed: {type(e).__name__}: {e}"[:500]),
                   **({} if was_stopped else {"error": tb_text[-4000:]}))
        return 1 if state == "failed" else 0


# --------------------------------------------------------------------------- parent side
def _alive(job_id: str, pid: Optional[int]) -> bool:
    p = _CHILDREN.get(job_id)
    if p is not None:
        return p.poll() is None
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status(tb, job_id: str) -> Dict[str, Any]:
    """status.json, corrected when the process died without reporting (killed / crashed)."""
    d = job_dir(tb, job_id)
    st = _read_json(d / "status.json", {}) or {}
    if st.get("backend") == "flux":
        return _flux_status(tb, d, _read_json(d / "spec.json", {}) or {}, st)
    if st.get("state") in ("queued", "running"):
        if not st.get("pid") and (d / "pid").exists():
            st["pid"] = int((d / "pid").read_text().strip() or 0) or None
        alive = _alive(job_id, st.get("pid"))
        if not alive:
            time.sleep(0.2)                            # the child may be writing its final status right now
            st = _read_json(d / "status.json", {}) or st
            if st.get("state") in ("queued", "running") and not _alive(job_id, st.get("pid")):
                stopped = (d / "stop").exists()
                st.update(state="stopped" if stopped else "failed", end=st.get("end") or _iso(_now()),
                          message=("stopped (killed)" if stopped else
                                   "process ended without reporting (killed or crashed; see log)"))
                _write_json(d / "status.json", st)
    return st


def result(tb, job_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(job_dir(tb, job_id) / "result.json")


def list_all(tb) -> List[Dict[str, Any]]:
    root = jobs_root(tb)
    if not root.exists():
        return []
    out = []
    for sp in sorted(root.glob("*/spec.json")):
        spec = _read_json(sp, {}) or {}
        try:
            st = status(tb, sp.parent.name)
        except ValueError:
            continue
        out.append({"job_id": sp.parent.name, "label": spec.get("label"), "target": spec.get("target"), **st})
    return out


def stop(tb, job_id: str, grace_s: float = STOP_GRACE_S) -> Dict[str, Any]:
    """SIGTERM the job's process group, SIGKILL whatever is left after ``grace_s``."""
    d = job_dir(tb, job_id)
    st = status(tb, job_id)
    if st.get("state") in FINAL:
        return {"state": st["state"], "note": "already finished"}
    (d / "stop").write_text(_iso(_now()) or "")
    if st.get("backend") == "flux":
        return _stop_flux(tb, job_id, d, st, grace_s)
    pid = st.get("pid")
    for _ in range(20):                           # just started: give the child a moment to report its group
        if st.get("pgid"):
            break
        time.sleep(0.1)
        st = _read_json(d / "status.json", {}) or st
        pid = st.get("pid") or pid
    pgid = st.get("pgid")
    if pgid is None and pid:
        try:
            pgid = os.getpgid(int(pid))
        except ProcessLookupError:
            pgid = None
    sent = []

    def _signal(sig):
        try:
            if pgid and pgid != os.getpgid(0):
                os.killpg(int(pgid), sig)
            elif pid:
                os.kill(int(pid), sig)
            sent.append(sig.name)
        except ProcessLookupError:
            pass
    _signal(signal.SIGTERM)
    t0 = _now()
    again = False
    while _now() - t0 < grace_s and _alive(job_id, pid):
        if not again and _now() - t0 > grace_s / 2:
            _signal(signal.SIGTERM)                # cooperative target did not reach a safe point: interrupt
            again = True
        time.sleep(0.2)
    if _alive(job_id, pid):
        _signal(signal.SIGKILL)
        time.sleep(0.3)
    elif pgid:                                    # leader exited: clean up stragglers (dask workers)
        try:
            os.killpg(int(pgid), signal.SIGKILL)
            sent.append("SIGKILL(group leftovers)")
        except (ProcessLookupError, PermissionError):
            pass
    p = _CHILDREN.get(job_id)
    if p is not None:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    st = status(tb, job_id)
    if st.get("state") in ("queued", "running"):
        st.update(state="stopped", end=_iso(_now()), message="stopped")
        _write_json(d / "status.json", st)
    return {"state": st.get("state"), "signals": sent, "seconds": round(_now() - t0, 2)}


def _stop_flux(tb, job_id: str, d: Path, st: Dict[str, Any], grace_s: float) -> Dict[str, Any]:
    from . import hpc
    spec = _read_json(d / "spec.json", {}) or {}
    steps = (spec.get("flux") or {}).get("steps") or []
    per = st.get("flux") or {}
    ids = [s["id"] for s in steps if (per.get(s["step"]) or {}).get("status") not in hpc.FLUX_FINAL]
    ok, msg = hpc.flux_cancel(ids)
    t0 = _now()
    while _now() - t0 < max(grace_s, 2.0):
        st = status(tb, job_id)
        if st.get("state") in FINAL:
            break
        time.sleep(min(1.0, _poll_s(tb, st)))
    out = {"state": st.get("state"), "flux_cancel": ids, "ok": ok, "seconds": round(_now() - t0, 2)}
    if msg:
        out["flux"] = msg[-500:]
    if st.get("state") not in FINAL:
        out["note"] = "flux cancel sent; Flux is still cleaning up (job_status shows when it has ended)"
    return out


def _poll_s(tb, st: Dict[str, Any], default: float = 0.3) -> float:
    if st.get("backend") != "flux":
        return default
    from .hpc import config_of
    return max(0.05, float(config_of(tb).poll_s or 30.0))


def wait(tb, job_id: str, timeout_s: float = 60.0, poll_s: float = 0.3) -> Dict[str, Any]:
    t0 = _now()
    timeout_s = min(float(timeout_s), WAIT_CAP_S)
    while True:
        st = status(tb, job_id)
        if st.get("state") in FINAL or _now() - t0 >= timeout_s:
            return st
        time.sleep(min(_poll_s(tb, st, poll_s), max(0.05, timeout_s - (_now() - t0))))


def _ui_progress(tb, msg: str, key: str) -> None:
    ui = getattr(tb, "ui", None)
    if ui is None or not hasattr(ui, "progress"):
        return
    try:
        ui.progress(msg, key=key)
    except TypeError:                                  # older / test UIs take only the message
        ui.progress(msg)
    except Exception:  # noqa: BLE001 - a status line must never end the wait
        pass


def wait_until_done(tb, job_id: str, max_s: Optional[float] = None, line_every_s: float = 10.0
                    ) -> Tuple[Dict[str, Any], Optional[str]]:
    """Wait until the job finishes, writing a status line (replace-key ``wait:<id>``) as it goes.

    Returns (status, why the wait ended early or None).  Ends early - leaving the job running - after ``max_s``
    (default [hpc] wait_max_hours), on Ctrl-C, when ``tb.stop_waiting`` is set (web UI button) or when a file
    ``stop_wait`` appears in the job directory."""
    from .hpc import config_of
    d = job_dir(tb, job_id)
    label = (_read_json(d / "spec.json", {}) or {}).get("label") or job_id
    max_s = float(max_s if max_s is not None else float(config_of(tb).wait_max_hours or 24) * 3600)
    ev = getattr(tb, "stop_waiting", None)
    if ev is not None:
        ev.clear()
    t0, last = _now(), 0.0
    key = f"wait:{job_id}"
    try:
        while True:
            st = status(tb, job_id)
            el = _now() - t0
            if st.get("state") in FINAL:
                _ui_progress(tb, f"job {job_id} [{label}]: {st.get('state')} after waiting {_fmt_s(el)}", key)
                return st, None
            if el >= max_s:
                return st, f"stopped waiting after {_fmt_s(el)} (limit [hpc] wait_max_hours); the job keeps running"
            if (ev is not None and ev.is_set()) or (d / "stop_wait").exists():
                try:
                    (d / "stop_wait").unlink()
                except OSError:
                    pass
                return st, f"stopped waiting after {_fmt_s(el)} at the user's request; the job keeps running"
            if _now() - last >= line_every_s or last == 0.0:
                fr = st.get("frac")
                _ui_progress(tb, f"waiting for job {job_id} [{label}]: {st.get('state')}"
                                 + (f" {100 * fr:.0f}%" if fr is not None else "")
                                 + f" - {str(st.get('message') or '')[:120]} (waited {_fmt_s(el)}; Ctrl-C or "
                                   "'stop waiting' ends the wait, not the job)", key)
                last = _now()
            nap = min(_poll_s(tb, st, 1.0), max(0.05, max_s - el))
            t1 = _now()
            while _now() - t1 < nap:
                if ev is not None and ev.is_set():
                    break
                time.sleep(min(0.25, nap))
    except KeyboardInterrupt:
        return status(tb, job_id), "stopped waiting (Ctrl-C); the job keeps running"


def progress_records(tb_or_dir, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    d = Path(tb_or_dir) if job_id is None else job_dir(tb_or_dir, job_id)
    p = d / "progress.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _tail(path: Path, n: int) -> str:
    if not path.exists() or n <= 0:
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200 * n - 4096))
        lines = f.read().decode(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _parse_iso(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def _fmt_s(s: float) -> str:
    s = max(0.0, float(s))
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def job_files(tb, job_id: str) -> List[str]:
    """Files the job reported (result.json 'files') plus files written in its directory (plots etc.)."""
    d = job_dir(tb, job_id)
    res = _read_json(d / "result.json", {}) or {}
    files = []
    for f in res.get("files") or []:
        p = Path(f) if Path(f).is_absolute() else Path(tb.p(f))
        if str(p) not in files:
            files.append(str(p))
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".png", ".csv", ".tdb", ".npy", ".json") and p.name not in (
                "spec.json", "status.json", "result.json"):
            if str(p) not in files:
                files.append(str(p))
    return files


def report(tb, job_id: str, tail: int = 15) -> Tuple[str, List[str]]:
    d = job_dir(tb, job_id)
    spec = _read_json(d / "spec.json", {}) or {}
    st = status(tb, job_id)
    state = st.get("state")
    L = [f"job {job_id} [{spec.get('label')}] state: {state}"]
    if spec.get("backend") == "flux":
        steps = (spec.get("flux") or {}).get("steps") or []
        per = st.get("flux") or {}
        L.append("backend: Flux batch - " + ", ".join(
            f"{s['step']} {s['id']} [{(per.get(s['step']) or {}).get('status', '?')}]" for s in steps))
        L.append(f"run dir {tb.rel(spec['kwargs']['run_dir'])}; from a shell: flux jobs "
                 + " ".join(s["id"] for s in steps))
    t_start = _parse_iso(st.get("start"))
    t_end = _parse_iso(st.get("end"))
    if t_start:
        el = (t_end or _now()) - t_start
        line = f"elapsed {_fmt_s(el)}"
        frac = st.get("frac")
        if frac is not None:
            line += f", progress {100 * frac:.1f}%"
            if state == "running" and 0 < frac < 1:
                line += f", ETA ~{_fmt_s(el * (1 - frac) / frac)}"
        L.append(line)
    if st.get("message"):
        L.append(f"last message: {st['message']}")
    if st.get("metrics"):
        L.append("metrics: " + json.dumps(st["metrics"], default=str)[:600])
    images: List[str] = []
    hook = spec.get("status_hook")
    if hook:
        try:
            mod, fn = hook.split(":", 1)
            txt, imgs = getattr(importlib.import_module(mod), fn)(tb, d, spec, st)
            if txt:
                L.append(txt)
            images += [i for i in imgs if i not in images]
        except Exception as e:  # noqa: BLE001 - extra reporting must never break job_status
            L.append(f"(status hook failed: {type(e).__name__}: {e})")
    if state in FINAL:
        res = _read_json(d / "result.json", {}) or {}
        if res.get("text"):
            L.append("result:\n" + str(res["text"]))
        if res.get("error") and state == "failed":
            L.append("error: " + str(res["error"]))
            if res.get("traceback"):
                L.append("traceback:\n" + str(res["traceback"])[-2500:])
        if res.get("output_tail"):
            L.append(f"batch output ({tb.rel(res.get('output_file', ''))}), last lines:\n{res['output_tail']}")
        files = job_files(tb, job_id)
        if files:
            L.append("files:\n" + "\n".join(f"  {tb.rel(f)}" for f in files))
        images += [f for f in files if f.lower().endswith(".png") and os.path.exists(f) and f not in images]
    t = _tail(d / "log.txt", tail)
    if t:
        L.append(f"log tail ({tb.rel(str(d / 'log.txt'))}):\n{t}")
    return "\n".join(L), images[:4]


# --------------------------------------------------------------------------- provenance
def producer_for(tb, path: str) -> Optional[Dict[str, Any]]:
    """Submitting call (tool, args, call_index, session) of the newest job that produced ``path`` (listed in its
    result.json ``files`` or written in its job directory, and not modified after the job ended), or None."""
    root = jobs_root(tb)
    if not root.exists():
        return None
    full = Path(path).resolve()
    try:
        mtime = full.stat().st_mtime
    except OSError:
        return None
    for sp in sorted(root.glob("*/spec.json"), reverse=True):
        d = sp.parent
        res = _read_json(d / "result.json", {}) or {}
        files = {str((Path(f) if Path(f).is_absolute() else Path(tb.p(f))).resolve()) for f in res.get("files") or []}
        if not (str(full) in files or d.resolve() in full.parents):
            continue
        end = _parse_iso((_read_json(d / "status.json", {}) or {}).get("end"))
        if end and mtime > end + 5 and d.resolve() not in full.parents:
            continue                              # rewritten by something else after the job finished
        by = (_read_json(sp, {}) or {}).get("submitted_by") or {}
        if by.get("tool"):
            return {**by, "job_id": d.name}
    return None


def producer_record(tb, path: str) -> Optional[Dict[str, Any]]:
    """codegen/show_code call record for a file produced by a background job (see :func:`producer_for`)."""
    try:
        by = producer_for(tb, path)
    except Exception:  # noqa: BLE001 - provenance lookups must never break show_code
        return None
    if not by:
        return None
    i = by.get("call_index")
    log = getattr(tb, "call_log", [])
    sess = getattr(getattr(tb, "provenance", None), "session_id", None)
    if i and by.get("session") in (None, sess) and 1 <= i <= len(log) and log[i - 1].get("tool") == by["tool"]:
        r = log[i - 1]
        return {"tool": r["tool"], "args": r["args"], "call_index": i, "images": r.get("images", []),
                "result": r.get("result_full") or r.get("result_preview", ""), "session": "current",
                "job_id": by["job_id"]}
    return {"tool": by["tool"], "args": by.get("args") or {}, "call_index": i, "session": by.get("session"),
            "images": [], "result": "", "job_id": by["job_id"]}


# --------------------------------------------------------------------------- demo target (tests / smoke checks)
def demo(steps: int = 3, delay: float = 0.1, fail: bool = False, plot: bool = False, out_dir: Optional[str] = None,
         progress: Callable = None) -> Dict[str, Any]:
    """Trivial job: ``steps`` progress reports ``delay`` s apart; optionally fails or saves a plot."""
    files = []
    for i in range(steps):
        time.sleep(delay)
        progress(f"step {i + 1}/{steps}", (i + 1) / steps, step=i + 1, value=math.sqrt(i + 1))
        if fail and i == steps // 2:
            raise RuntimeError(f"demo failure at step {i + 1}")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(range(steps), [math.sqrt(i + 1) for i in range(steps)], "o-")
        p = Path(out_dir or ".") / "demo_plot.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=60)
        plt.close(fig)
        files.append(str(p.resolve()))
    return {"text": f"demo finished {steps} steps", "files": files, "steps": steps}


# --------------------------------------------------------------------------- tools
def register(tb) -> None:
    S = {"type": "string"}
    N = {"type": "number"}
    I = {"type": "integer"}

    @tb.add("list_jobs", "List background jobs (id, label, state, progress, last message).", {}, [])
    def list_jobs():
        js = list_all(tb)
        if not js:
            return "no background jobs", []
        L = []
        for j in js:
            fr = f" {100 * j['frac']:.0f}%" if j.get("frac") is not None else ""
            L.append(f"{j['job_id']}  {j.get('state'):<8}{fr:>5}  {j.get('label')}  - {str(j.get('message'))[:90]}")
        return "\n".join(L), []

    @tb.add("job_status", "State, progress/ETA, metrics and log tail of a background job; when finished its "
            "result text, files and plots (ESPEI MCMC: lnprob-so-far convergence plot at any time).",
            {"job_id": S, "tail": {**I, "description": "log lines to show (default 15)"}}, ["job_id"])
    def job_status(job_id: str, tail: int = 15):
        return report(tb, job_id, tail=tail if tail is not None else 15)

    @tb.add("stop_job", "Stop a background job (SIGTERM, SIGKILL after 16 s; Flux: flux cancel). ESPEI MCMC keeps "
            "the chains saved so far.", {"job_id": S}, ["job_id"])
    def stop_job(job_id: str):
        r = stop(tb, job_id)
        text, imgs = report(tb, job_id, tail=5)
        return f"stop: {json.dumps(r)}\n{text}", imgs

    @tb.add("wait_job", f"Wait for a background or Flux job to finish, then report like job_status. "
            f"timeout_s: at most {WAIT_CAP_S:.0f} s; until='done': keep waiting (status line; the user can end "
            "the wait, not the job) up to [hpc] wait_max_hours - use it for ESPEI optimisations, then continue.",
            {"job_id": S, "timeout_s": N,
             "until": {"type": "string", "enum": ["done"], "description": "wait until the job has finished"}},
            ["job_id"])
    def wait_job(job_id: str, timeout_s: float = 120.0, until: Optional[str] = None):
        why = None
        if until == "done":
            st, why = wait_until_done(tb, job_id)
        else:
            st = wait(tb, job_id, timeout_s if timeout_s is not None else 120.0)
        text, imgs = report(tb, job_id)
        if st.get("state") not in FINAL:
            text = (f"still {st.get('state')} ({why or 'timeout reached'}); wait_job until='done' waits until it "
                    "finishes\n") + text
        return text, imgs


# --------------------------------------------------------------------------- entry point
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "run":
        return _run_child(Path(argv[1]))
    print("usage: python -m calphad_tools.jobs run <jobdir>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
