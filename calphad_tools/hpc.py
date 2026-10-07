"""Where long ESPEI runs go: a Flux cluster (batched) or this machine (direct), with run-size defaults.

* :func:`detect` - is this a Flux-capable environment?  ``flux`` on PATH **and** an instance that answers
  (``FLUX_URI`` set: ``flux getattr size``; otherwise the system instance: ``flux uptime``), probed with a short
  timeout and never submitting anything.  Also reads nodes / cores per node / GPUs (``flux resource info``, else
  ``flux resource list``), the queues (``flux queue list``), whether ``flux batch`` knows ``--bank`` and the local
  core count (CPU affinity).  Cached per process; ``refresh=True`` probes again.
* :class:`HPCConfig` - the ``[hpc]`` table of ``calphad_agent.toml`` (scheduler auto|flux|local, bank, queue,
  nodes, cores_per_node, time limits, local_cores, extra_flux_args, fallback_to_local, wait_max_hours).
* Run-size defaults: :func:`local_cores` (direct runs: half of the available cores, at least 1) and
  :func:`batch_resources` (batched runs: 1 node with all of its cores).
* Flux plumbing used by the ESPEI tools and the job registry: :func:`batch_options`, :func:`flux_batch`,
  :func:`flux_job_states`, :func:`flux_cancel`.

``python -m calphad_tools.hpc [--config calphad_agent.toml] [--json] [--remind-only]`` prints the detection (the
installer runs it).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROBE_TIMEOUT_S = 8.0
SUBMIT_TIMEOUT_S = 120.0
SCHEDULERS = ("auto", "flux", "local")
# flux job status -> job-registry state
FLUX_STATE = {"DEPEND": "queued", "PRIORITY": "queued", "SCHED": "queued", "NEW": "queued",
              "RUN": "running", "CLEANUP": "running",
              "COMPLETED": "done", "FAILED": "failed", "TIMEOUT": "failed", "CANCELED": "stopped",
              "CANCELLED": "stopped"}
FLUX_FINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELED", "CANCELLED"}
_CACHE: Dict[str, Any] = {}


@dataclass
class HPCConfig:
    scheduler: str = "auto"           # auto (detect Flux at run time) | flux | local
    bank: str = ""                    # flux batch --bank=BANK (older flux-core: --setattr=system.bank=BANK)
    queue: str = ""                   # flux batch -q QUEUE
    nodes: int = 1                    # batched runs: nodes per ESPEI MCMC job
    cores_per_node: int = 0           # 0 = from flux resource info / list
    time_limit: str = "4h"            # flux batch -t, MCMC job
    gen_time_limit: str = "1h"        # flux batch -t, parameter-generation job (1 core)
    local_cores: Any = "half"         # direct runs: "half", "all" or a number
    extra_flux_args: list = field(default_factory=list)   # added to every flux batch command line
    setup: list = field(default_factory=list)             # batch-script environment lines (default: install.sh's
                                                          # pycalphad-agent-env.sh)
    fallback_to_local: bool = False   # Flux found but submission failed: run here instead (and say so)
    wait_max_hours: float = 24.0      # wait_job until=done stops waiting (not the job) after this long
    poll_s: float = 30.0              # how often a waiting agent asks Flux for the job state


def config_of(tb) -> HPCConfig:
    """The [hpc] settings of a ToolBox.  A ToolBox built without them (library use, tests, session replays) runs
    everything locally - only the agent, which passes the config, detects and submits to Flux."""
    h = getattr(tb, "hpc", None)
    return h if isinstance(h, HPCConfig) else HPCConfig(scheduler="local")


# --------------------------------------------------------------------------- cores
def available_cores() -> int:
    """Cores this process may use (CPU affinity / cgroup cpuset where the OS reports it)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def local_cores(setting: Any = "half", available: Optional[int] = None) -> int:
    """Cores for a direct run: "half" (default) of the available cores (at least 1), "all", or a number (capped at
    the available cores)."""
    n = int(available or available_cores())
    s = "half" if setting is None else str(setting).strip().lower()
    if s in ("", "half", "auto", "default"):
        return max(1, n // 2)
    if s in ("all", "max"):
        return n
    try:
        k = int(float(s))
    except ValueError:
        raise ValueError(f'[hpc] local_cores must be "half", "all" or a number, not {setting!r}') from None
    return max(1, min(k, n))


# --------------------------------------------------------------------------- probing
def _flux_exe() -> Optional[str]:
    return shutil.which("flux")


def _run(args: Sequence[str], timeout: float = PROBE_TIMEOUT_S, cwd: Optional[str] = None,
         exe: Optional[str] = None) -> Tuple[Optional[int], str, str]:
    """(returncode | None on timeout/OS error, stdout, stderr) of ``flux ARGS``."""
    exe = exe or _flux_exe() or "flux"
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout, cwd=cwd,
                           stdin=subprocess.DEVNULL, errors="replace")
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"no answer within {timeout:g} s"
    except OSError as e:
        return None, "", f"{type(e).__name__}: {e}"


def _first_line(s: str) -> str:
    for ln in (s or "").splitlines():
        if ln.strip():
            return ln.strip()[:200]
    return ""


def parse_resource_info(text: str) -> Optional[Dict[str, int]]:
    """``flux resource info``: "4 Nodes, 192 Cores, 8 GPUs" -> {nodes, cores, gpus}."""
    m = re.search(r"(\d+)\s+Nodes?\D+?(\d+)\s+Cores?(?:\D+?(\d+)\s+GPUs?)?", text or "", re.I)
    if not m:
        return None
    return {"nodes": int(m.group(1)), "cores": int(m.group(2)), "gpus": int(m.group(3) or 0)}


def parse_resource_list(text: str) -> Optional[Dict[str, int]]:
    """``flux resource list`` (default table, or ``-o "{state} {nnodes} {ncores} {ngpus}"``): totals over the
    free + allocated rows (down/drained nodes are not counted).  Per row the first three whole numbers are
    NNODES, NCORES, NGPUS - true for the default columns (STATE [QUEUE] [PROPERTIES] NNODES NCORES NGPUS
    NODELIST)."""
    rows = []
    for ln in (text or "").splitlines():
        toks = ln.split()
        if not toks or toks[0].upper() == "STATE":
            continue
        nums = [int(t) for t in toks[1:] if t.isdigit()]
        if len(nums) < 2:
            continue
        rows.append((toks[0].lower(), nums[0], nums[1], nums[2] if len(nums) > 2 else 0))
    if not rows:
        return None
    up = [r for r in rows if r[0] in ("free", "allocated", "alloc", "up")] or rows
    nodes = sum(r[1] for r in up)
    cores = sum(r[2] for r in up)
    if not nodes:
        return None
    per_node = [r[2] // r[1] for r in up if r[1]]
    return {"nodes": nodes, "cores": cores, "gpus": sum(r[3] for r in up),
            "cores_per_node_max": max(per_node) if per_node else 0}


def parse_queue_list(text: str) -> Tuple[List[str], Optional[str]]:
    """``flux queue list`` -> (queue names, default queue).  The default queue is marked with '*'."""
    names, default = [], None
    for ln in (text or "").splitlines():
        toks = ln.split()
        if not toks or toks[0].upper().startswith("QUEUE") or toks[0] in ("-", "*"):
            continue
        name = toks[0]
        if name.endswith("*"):
            name = name.rstrip("*")
            default = default or name
        if re.fullmatch(r"[A-Za-z0-9_.+-]+", name) and name not in names:
            names.append(name)
    return names, default


def detect(refresh: bool = False, timeout: float = PROBE_TIMEOUT_S) -> Dict[str, Any]:
    """Flux or local, and why.  Cached per process (``refresh=True`` probes again).

    Keys: scheduler ("flux" | "local"), reason, flux (path or None), flux_uri, instance ("system" = the
    cluster's instance, "job" = inside an allocation; from ``flux getattr instance-level``), instance_level, nodes,
    cores_per_node, gpus_per_node, resources_from, queues, default_queue, bank_flag ("--bank" or
    "--setattr=system.bank"), version, local_cores (available here), seconds (probe time)."""
    key = "detect"
    if not refresh and key in _CACHE and _CACHE[key].get("_path") == os.environ.get("PATH") \
            and _CACHE[key].get("_uri") == os.environ.get("FLUX_URI"):
        return {k: v for k, v in _CACHE[key].items() if not k.startswith("_")}
    t0 = time.time()
    exe = _flux_exe()
    uri = os.environ.get("FLUX_URI") or None
    info: Dict[str, Any] = {"scheduler": "local", "reason": "", "flux": exe, "flux_uri": uri, "instance": None,
                            "instance_level": None,
                            "nodes": None, "cores_per_node": None, "gpus_per_node": None, "resources_from": None,
                            "queues": [], "default_queue": None, "bank_flag": "--bank", "version": None,
                            "local_cores": available_cores()}
    if not exe:
        info["reason"] = "no flux executable on PATH"
    else:
        if uri:
            rc, out, err = _run(["getattr", "size"], timeout, exe=exe)
            if rc == 0:
                info.update(scheduler="flux", instance="job",
                            reason=f"inside a Flux instance (FLUX_URI set; size {out.strip() or '?'})")
            else:
                info["reason"] = f"FLUX_URI is set but the instance does not answer: {_first_line(err) or rc}"
        if info["scheduler"] != "flux":
            rc, out, err = _run(["uptime"], timeout, exe=exe)
            if rc == 0:
                info.update(scheduler="flux", instance=info["instance"] or "system",
                            reason=f"Flux instance answers (flux uptime: {_first_line(out)})")
            elif not info["reason"]:
                info["reason"] = f"flux is on PATH ({exe}) but no instance answers: {_first_line(err) or rc}"
        if info["scheduler"] == "flux":
            rc, out, _ = _run(["getattr", "instance-level"], timeout, exe=exe)
            if rc == 0 and out.strip().isdigit():
                info["instance_level"] = int(out.strip())
                info["instance"] = "system" if info["instance_level"] == 0 else "job"
            _probe_details(info, exe, timeout)
    info["seconds"] = round(time.time() - t0, 3)
    _CACHE[key] = {**info, "_path": os.environ.get("PATH"), "_uri": os.environ.get("FLUX_URI")}
    return dict(info)


def _probe_details(info: Dict[str, Any], exe: str, timeout: float) -> None:
    rc, out, _ = _run(["resource", "info"], timeout, exe=exe)
    res = parse_resource_info(out) if rc == 0 else None
    src = "flux resource info"
    if not res:
        rc, out, _ = _run(["resource", "list", "-n", "-o", "{state} {nnodes} {ncores} {ngpus}"], timeout, exe=exe)
        res = parse_resource_list(out) if rc == 0 else None
        src = "flux resource list -o"
        if not res:
            rc, out, _ = _run(["resource", "list"], timeout, exe=exe)
            res = parse_resource_list(out) if rc == 0 else None
            src = "flux resource list"
    if res and res.get("nodes"):
        info["nodes"] = res["nodes"]
        info["cores_per_node"] = res.get("cores_per_node_max") or res["cores"] // res["nodes"]
        info["gpus_per_node"] = res.get("gpus", 0) // res["nodes"]
        info["resources_from"] = src
    rc, out, _ = _run(["queue", "list"], timeout, exe=exe)
    if rc == 0:
        info["queues"], info["default_queue"] = parse_queue_list(out)
    rc, out, err = _run(["batch", "--help"], timeout, exe=exe)
    if rc == 0 and "--bank" not in (out + err):
        info["bank_flag"] = "--setattr=system.bank"
    rc, out, _ = _run(["version"], timeout, exe=exe)
    if rc == 0:
        m = re.search(r"commands:\s*(\S+)", out)
        info["version"] = m.group(1) if m else _first_line(out)


def clear_cache() -> None:
    _CACHE.clear()


def summary(info: Dict[str, Any], cfg: Optional[HPCConfig] = None) -> str:
    """Human-readable detection result (install.sh, calphad-agent --check, tool replies)."""
    cfg = cfg or HPCConfig()
    L = []
    if info.get("scheduler") == "flux":
        L.append(f"HPC: Flux - {info.get('reason')}")
        res = (f"{info['nodes']} node(s) x {info['cores_per_node']} cores"
               + (f", {info['gpus_per_node']} GPU(s)/node" if info.get("gpus_per_node") else "")
               + f" ({info.get('resources_from')})") if info.get("nodes") else \
            "resources: could not be read (set [hpc] cores_per_node)"
        L.append(f"  {res}; flux-core {info.get('version') or '?'}")
        if info.get("queues"):
            L.append("  queues: " + ", ".join(q + (" (default)" if q == info.get("default_queue") else "")
                                              for q in info["queues"]))
        nodes, cpn, _ = batch_resources(cfg, info, strict=False)
        L.append(f"  ESPEI optimisations are submitted with flux batch: {nodes} node(s) x {cpn or '?'} cores per "
                 f"MCMC job, time limit {cfg.time_limit}; bank {cfg.bank or '(not set)'}, queue "
                 f"{cfg.queue or '(not set: site default)'}")
        if info.get("instance_level"):
            L.append(f"  inside a Flux allocation (instance-level {info['instance_level']}): jobs are submitted into "
                     "it; [hpc] queue / bank are not used there")
        if info.get("bank_flag") != "--bank":
            L.append("  (this flux batch has no --bank option: the bank is passed as --setattr=system.bank=...)")
    else:
        L.append(f"HPC: local - {info.get('reason')}")
    n = info.get("local_cores") or available_cores()
    try:
        lc = local_cores(cfg.local_cores, n)
    except ValueError as e:
        lc = f"? ({e})"
    L.append(f"  direct runs here: {lc} of {n} cores (local_cores = {cfg.local_cores!r})")
    if cfg.scheduler != "auto":
        L.append(f"  [hpc] scheduler = {cfg.scheduler!r} overrides the detection")
    return "\n".join(L)


def reminder(info: Dict[str, Any], cfg: HPCConfig, config_path: str = "calphad_agent.toml") -> str:
    """What to set in the config when Flux is found but bank/queue are empty ('' when nothing is missing)."""
    if info.get("scheduler") != "flux" or (cfg.bank and cfg.queue):
        return ""
    miss = [k for k in ("bank", "queue") if not getattr(cfg, k)]
    q = f" (queues here: {', '.join(info['queues'])})" if info.get("queues") else ""
    return (f"Flux was found: set {' and '.join(miss)} under [hpc] in {config_path}{q}, e.g.\n"
            "  [hpc]\n  bank = \"your-bank\"      # the allocation account your site gave you\n"
            f"  queue = \"{info.get('default_queue') or 'pbatch'}\"\n"
            "(sites that need neither can leave them empty; flux batch then says what is missing.)")


# --------------------------------------------------------------------------- which backend
def resolve_backend(cfg: HPCConfig, requested: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """("flux" | "local", detection info).  ``requested`` (a tool argument) overrides ``cfg.scheduler``;
    "auto" detects (cached per process)."""
    req = (requested or "auto").strip().lower()
    if req == "auto":
        req = (cfg.scheduler or "auto").strip().lower()
    if req not in SCHEDULERS:
        raise ValueError(f"backend/scheduler must be one of {SCHEDULERS}, not {req!r}")
    if req == "local":
        return "local", {"scheduler": "local", "reason": "local requested ([hpc] scheduler or backend argument)",
                         "local_cores": available_cores()}
    info = detect()
    if req == "flux" and info["scheduler"] != "flux":
        raise RuntimeError(f"Flux was requested but is not usable here: {info['reason']}. Use backend='local' "
                           "or run the agent on the cluster (login node with flux on PATH).")
    return info["scheduler"], info


def batch_resources(cfg: HPCConfig, info: Dict[str, Any], nodes: Optional[int] = None,
                    cores_per_node: Optional[int] = None, strict: bool = True) -> Tuple[int, int, str]:
    """(nodes, cores per node, where the core count came from) for a batched MCMC job: 1 node with all of its
    cores unless the tool arguments or [hpc] nodes / cores_per_node say otherwise."""
    n = int(nodes or cfg.nodes or 1)
    if cores_per_node:
        return n, int(cores_per_node), "tool argument"
    if cfg.cores_per_node:
        return n, int(cfg.cores_per_node), "[hpc] cores_per_node"
    if info.get("cores_per_node"):
        return n, int(info["cores_per_node"]), str(info.get("resources_from") or "flux")
    if strict:
        raise RuntimeError("could not read the cores per node from Flux (flux resource info / list); set [hpc] "
                           "cores_per_node in calphad_agent.toml or pass cores_per_node")
    return n, 0, "unknown"


# --------------------------------------------------------------------------- submission / state
class FluxSubmitError(RuntimeError):
    def __init__(self, message: str, command: List[str], stderr: str = ""):
        super().__init__(message)
        self.command, self.stderr = command, stderr


def batch_options(cfg: HPCConfig, bank_flag: str = "--bank", bank: Optional[str] = None,
                  queue: Optional[str] = None, nested: bool = False) -> List[str]:
    """Options every flux batch command gets: queue, bank, extra_flux_args (nodes / cores / time are directives
    in the script itself).  ``nested`` (the agent runs inside a Flux allocation, instance-level > 0): the jobs go
    into that allocation, which has no queues or banks, so [hpc] queue / bank are left out (tool arguments are
    still passed)."""
    opts: List[str] = []
    q = queue if queue is not None else ("" if nested else cfg.queue)
    b = bank if bank is not None else ("" if nested else cfg.bank)
    if q:
        opts += ["-q", str(q)]
    if b:
        opts.append(f"{bank_flag or '--bank'}={b}")
    opts += [str(x) for x in (cfg.extra_flux_args or [])]
    return opts


def submit_hint(stderr: str) -> str:
    s = (stderr or "").lower()
    tips = []
    if "bank" in s or "account" in s or "association" in s:
        tips.append('bank: set [hpc] bank = "<your bank>" in calphad_agent.toml (or pass bank=...); '
                    "`flux account view-user $USER` or your site docs list your banks")
    if "queue" in s or "partition" in s:
        tips.append('queue: set [hpc] queue = "<name>" (or pass queue=...); `flux queue list` shows the queues')
    if "duration" in s or "time" in s or "limit" in s:
        tips.append('time limit: [hpc] time_limit = "4h" / gen_time_limit = "1h" must fit the queue limits '
                    "(`flux queue list` shows TLIMIT)")
    if "node" in s or "core" in s or "resource" in s or "unsatisfiable" in s:
        tips.append("resources: [hpc] nodes / cores_per_node (or the nodes / cores_per_node arguments) must fit "
                    "the queue")
    if not tips:
        tips.append("check [hpc] bank, queue, time_limit, nodes and extra_flux_args in calphad_agent.toml")
    return "\n".join(f"  - {t}" for t in tips)


def flux_batch(script: str, cwd: str, options: Sequence[str] = (), dependency: Optional[str] = None,
               timeout: float = SUBMIT_TIMEOUT_S) -> Dict[str, Any]:
    """``flux batch [options] [--dependency=afterok:ID] script`` in ``cwd`` -> {"id", "command"}.  Raises
    :class:`FluxSubmitError` with Flux's own message when the submission is refused."""
    cmd = ["flux", "batch", *[str(o) for o in options]]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(script)
    rc, out, err = _run(cmd[1:], timeout, cwd=cwd)
    jid = ""
    for ln in (out or "").splitlines():
        if ln.strip():
            jid = ln.strip()
    if rc != 0 or not jid:
        msg = (err or out or "").strip() or (f"exit status {rc}" if rc is not None else "no answer")
        raise FluxSubmitError(f"flux batch failed: {msg[-1500:]}", cmd, msg)
    return {"id": jid, "command": cmd}


def norm_id(jid: str) -> str:
    """Flux prints f58 ids with 'ƒ' (UTF-8 locale) or 'f'; compare them without that difference."""
    return str(jid).strip().replace("ƒ", "f")


def _eventlog_state(jid: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Fallback for ids ``flux jobs`` no longer lists: the job's eventlog."""
    rc, out, _ = _run(["job", "eventlog", jid], timeout)
    if rc != 0 or not out.strip():
        return None
    names = [ln.split()[1] for ln in out.splitlines() if len(ln.split()) > 1]
    st: Dict[str, Any] = {"status": "SCHED", "returncode": None, "exception": ""}
    if "start" in names:
        st["status"] = "RUN"
    m = re.search(r"\bfinish\b.*?status=(\d+)", out)
    exc = re.search(r"\bexception\b.*?type=(\S+)(.*)", out)
    if exc:
        st["exception"] = (exc.group(1) + " " + exc.group(2).strip()).strip()[:300]
    if "clean" in names or m:
        if exc and exc.group(1) == "cancel":
            st["status"] = "CANCELED"
        elif exc and exc.group(1) == "timeout":
            st["status"] = "TIMEOUT"
        elif m:
            code = int(m.group(1))
            st["returncode"] = code >> 8 if code > 255 else code
            st["status"] = "COMPLETED" if code == 0 else "FAILED"
        else:
            st["status"] = "FAILED"
    return st


def flux_job_states(ids: Sequence[str], timeout: float = 30.0) -> Optional[Dict[str, Dict[str, Any]]]:
    """{id: {"status": DEPEND|SCHED|RUN|...|COMPLETED|FAILED|CANCELED|TIMEOUT|UNKNOWN, "returncode", "exception"}}
    from ``flux jobs -n -o ... IDS`` (``flux job eventlog`` for ids it no longer lists).  None when Flux does not
    answer at all."""
    ids = [i for i in ids if i]
    if not ids:
        return {}
    fmt = "{id}|{status}|{returncode}|{exception.type}|{exception.note}"
    rc, out, err = _run(["jobs", "-n", "-o", fmt, *ids], timeout)
    if rc is None:
        return None
    got: Dict[str, Dict[str, Any]] = {}
    for ln in (out or "").splitlines():
        parts = ln.strip().split("|", 4)
        if len(parts) < 2:
            continue
        rcode = parts[2].strip() if len(parts) > 2 else ""
        got[norm_id(parts[0])] = {"status": parts[1].strip().upper(),
                                  "returncode": int(rcode) if re.fullmatch(r"-?\d+", rcode) else None,
                                  "exception": " ".join(p.strip() for p in parts[3:] if p.strip())[:300]}
    res: Dict[str, Dict[str, Any]] = {}
    for i in ids:
        st = got.get(norm_id(i)) or _eventlog_state(i, timeout)
        res[i] = st or {"status": "UNKNOWN", "returncode": None,
                        "exception": _first_line(err) or "not listed by flux jobs"}
    return res


def flux_cancel(ids: Sequence[str], timeout: float = 30.0) -> Tuple[bool, str]:
    """``flux cancel IDS`` (older flux-core: ``flux job cancel``)."""
    ids = [i for i in ids if i]
    if not ids:
        return True, ""
    rc, out, err = _run(["cancel", *ids], timeout)
    if rc != 0 and re.search(r"unknown|not found|invalid choice|usage", err or "", re.I):
        rc, out, err = _run(["job", "cancel", *ids], timeout)
    return rc == 0, (err or out or "").strip()


# --------------------------------------------------------------------------- CLI (install.sh, --check)
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Is this a Flux-capable environment? (CALPHAD agent ESPEI runs)")
    ap.add_argument("--config", help="calphad_agent.toml with an [hpc] table")
    ap.add_argument("--json", action="store_true", help="print the detection as JSON")
    ap.add_argument("--remind-only", action="store_true",
                    help="print only the bank/queue reminder (when Flux is found and they are not set)")
    a = ap.parse_args(argv)
    cfg = HPCConfig()
    if a.config and os.path.isfile(a.config):
        try:
            from .agent.config import load_config
            cfg = load_config(a.config).hpc
        except Exception as e:  # noqa: BLE001 - detection must still print
            print(f"(could not read {a.config}: {e})", file=sys.stderr)
    info = detect()
    if a.json:
        print(json.dumps({**info, "config": asdict(cfg)}, indent=1, default=str))
        return 0
    rem = reminder(info, cfg, a.config or "calphad_agent.toml")
    if not a.remind_only:
        print(summary(info, cfg))
    if rem:
        print(rem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
