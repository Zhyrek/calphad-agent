"""A stand-in ``flux`` command for the HPC tests (tests/test_hpc.py).

Installed by the tests as an executable ``flux`` shim first on PATH.  State lives in ``$FAKE_FLUX_DIR``:
``jobs/<id>.json`` (status, script, cwd, dependency, ...), ``calls.jsonl`` (every invocation's argv) and
``submissions.jsonl`` (flux batch: argv, options, the script's ``#flux:`` directives, cwd).

Modes (``$FAKE_FLUX_MODE``):
  ok        (default) an instance answers; jobs run
  noanswer  every command that needs an instance fails ("Unable to connect to Flux")
  hang      every command that needs an instance hangs (tests the probe timeout)
  reject    flux batch refuses jobs with a bank error (like a site that needs --bank)
  failmcmc  the MCMC job fails (writes a traceback to its output file, exit code 1)

A submitted job is executed by a detached ``_run`` process: it waits for its ``--dependency=afterok:ID`` (a
failed/cancelled dependency fails it with a "dependency" exception), passes SCHED -> RUN -> COMPLETED, and does the
work of the script: ``gen.flux`` runs real ESPEI parameter generation (``espei --in espei_generate.yaml`` from this
Python's environment); ``mcmc.flux`` is a stub that writes mcmc.log ("(i of N)" lines, slowly), trace.npy,
lnprob.npy and mcmc.tdb (= dft.tdb), like a finished ESPEI run.  ``$FAKE_FLUX_MCMC_S`` sets the stub's duration.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("FAKE_FLUX_DIR", "/tmp/fake_flux"))
MODE = os.environ.get("FAKE_FLUX_MODE", "ok")
FINAL = {"COMPLETED", "FAILED", "CANCELED", "TIMEOUT"}


def _jobs():
    d = STATE / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(jid):
    p = _jobs() / f"{jid}.json"
    for _ in range(20):
        try:
            return json.loads(p.read_text())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            time.sleep(0.02)
    return None


def _save(job):
    p = _jobs() / f"{job['id']}.json"
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(job))
    os.replace(tmp, p)


def _update(jid, **kw):
    job = _load(jid)
    if job is None:
        return None
    if job["status"] == "CANCELED":
        return job                        # cancelled: the runner must not overwrite it
    job.update(kw)
    _save(job)
    return job


def _need_instance():
    if MODE == "noanswer":
        print("flux: ERROR: Unable to connect to Flux: broker socket /run/flux/local was not found", file=sys.stderr)
        sys.exit(1)
    if MODE == "hang":
        time.sleep(60)
        sys.exit(1)


def _directives(script: Path):
    out = []
    for ln in script.read_text().splitlines():
        if ln.startswith("#flux:"):
            out.append(ln[len("#flux:"):].strip())
    return out


def _new_id():
    n = len(list(_jobs().glob("*.json"))) + 1
    return f"ƒFAKE{n:03d}"


def cmd_batch(args):
    if args and args[0] in ("--help", "-h"):
        print("usage: flux batch [OPTIONS...] SCRIPT\n  -q, --queue=NAME   Submit a job to a specific named queue\n"
              "  -B, --bank=BANK    Submit a job to a specific named bank\n  -N, --nodes=N\n  -t, --time-limit=MIN")
        return 0
    _need_instance()
    script = args[-1]
    opts = args[:-1]
    sp = Path.cwd() / script
    if not sp.exists():
        print(f"flux-batch: ERROR: {script}: No such file or directory", file=sys.stderr)
        return 1
    if MODE == "reject":
        print("flux-batch: ERROR: [Errno 22] bank is required: use --bank to choose one of: acct1, acct2",
              file=sys.stderr)
        return 1
    dep = next((o.split(":", 1)[1] for o in opts if o.startswith("--dependency=afterok:")), None)
    jid = _new_id()
    job = {"id": jid, "status": "DEPEND" if dep else "SCHED", "script": script, "cwd": str(Path.cwd()),
           "dependency": dep, "options": opts, "directives": _directives(sp), "returncode": None, "exception": "",
           "submitted": time.time()}
    _save(job)
    with open(STATE / "submissions.jsonl", "a") as f:
        f.write(json.dumps({"id": jid, "argv": ["flux", "batch", *args], "options": opts,
                            "directives": job["directives"], "cwd": job["cwd"], "script": script}) + "\n")
    subprocess.Popen([sys.executable, __file__, "_run", jid], start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    print(jid)
    return 0


def _output_path(job):
    name = next((d.split("=", 1)[1] for d in job["directives"] if d.startswith("--output=")), "flux-{{id}}.out")
    return Path(job["cwd"]) / name.replace("{{id}}", job["id"])


def _cancelled(jid):
    j = _load(jid)
    return j is None or j["status"] == "CANCELED"


def run_job(jid):
    job = _load(jid)
    dep = job.get("dependency")
    while dep:
        dj = _load(dep)
        if dj is None or dj["status"] in ("FAILED", "CANCELED", "TIMEOUT"):
            _update(jid, status="FAILED", exception="dependency dependency job failed or was canceled",
                    returncode=None)
            return
        if dj["status"] == "COMPLETED":
            break
        if _cancelled(jid):
            return
        time.sleep(0.1)
    _update(jid, status="SCHED")
    time.sleep(0.2)
    if _cancelled(jid):
        return
    _update(jid, status="RUN", t_run=time.time())
    out = _output_path(job)
    cwd = Path(job["cwd"])
    rc = 0
    with open(out, "w") as f:
        f.write(f"fake flux job {jid}: {job['script']}\n")
        f.flush()
        if job["script"].endswith("gen.flux"):
            espei = str(Path(sys.executable).parent / "espei")
            env = {**os.environ, "MPLBACKEND": "Agg"}
            p = subprocess.Popen([espei, "--in", "espei_generate.yaml"], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                 env=env)
            while p.poll() is None:
                if _cancelled(jid):
                    p.terminate()
                    return
                time.sleep(0.1)
            rc = p.returncode
        else:
            rc = _stub_mcmc(jid, cwd, f)
    if _cancelled(jid):
        return
    _update(jid, status="COMPLETED" if rc == 0 else "FAILED", returncode=rc)


def _stub_mcmc(jid, cwd: Path, f):
    import numpy as np
    import yaml
    y = yaml.safe_load((cwd / "espei_mcmc.yaml").read_text())
    m = y["mcmc"]
    n_it = int(m["iterations"])
    idb = cwd / m["input_db"]
    from pycalphad import Database
    from espei.utils import database_symbols_to_fit
    n_par = len(database_symbols_to_fit(Database(str(idb))))
    n_ch = int(m.get("chains_per_parameter", 2)) * n_par
    dur = float(os.environ.get("FAKE_FLUX_MCMC_S", "1.5"))
    rng = np.random.default_rng(0)
    trace = np.zeros((n_ch, n_it, n_par))
    lnp = np.zeros((n_ch, n_it))
    with open(cwd / "mcmc.log", "w") as log:
        log.write("INFO:espei.optimizers.opt_mcmc - Running MCMC for %d iterations.\n" % n_it)
        for i in range(n_it):
            if _cancelled(jid):
                return 143
            if MODE == "failmcmc" and i == n_it // 2:
                f.write("Traceback (most recent call last):\n  File \"espei\", line 1\nValueError: fake MCMC failure\n")
                return 1
            trace[:, i, :] = rng.normal(size=(n_ch, n_par))
            lnp[:, i] = -1000.0 / (i + 1) - rng.random(n_ch)
            np.save(cwd / "trace.npy", trace)
            np.save(cwd / "lnprob.npy", lnp)
            n = int(30 * (i + 1) / n_it)
            log.write("INFO:espei.optimizers.opt_mcmc - \r[%s%s] (%d of %d)\n" % ("#" * n, " " * (30 - n), i + 1, n_it))
            log.flush()
            time.sleep(dur / n_it)
    shutil.copy(idb, cwd / y["output"]["output_db"])
    f.write("ESPEI MCMC finished (stub)\n")
    return 0


def cmd_jobs(args):
    _need_instance()
    fmt, ids, i = "{id} {status}", [], 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "--format"):
            fmt, i = args[i + 1], i + 2
            continue
        if a.startswith("--format="):
            fmt = a.split("=", 1)[1]
        elif not a.startswith("-"):
            ids.append(a)
        i += 1
    rc = 0
    for jid in ids:
        j = _load(jid.replace("f", "ƒ", 1) if jid.startswith("f") else jid)
        if j is None:
            print(f"flux-jobs: ERROR: {jid}: unknown job id", file=sys.stderr)
            rc = 1
            continue
        exc = j.get("exception") or ""
        vals = {"id": j["id"], "status": j["status"], "returncode": "" if j["returncode"] is None else j["returncode"],
                "exception.type": exc.split(" ", 1)[0] if exc else "",
                "exception.note": exc.split(" ", 1)[1] if " " in exc else ""}
        print(re.sub(r"\{([\w.]+)\}", lambda m: str(vals.get(m.group(1), "")), fmt))
    return rc


def cmd_cancel(args):
    _need_instance()
    for jid in [a for a in args if not a.startswith("-")]:
        j = _load(jid)
        if j and j["status"] not in FINAL:
            j.update(status="CANCELED", exception="cancel canceled by user")
            _save(j)
    return 0


def main(argv):
    with open(STATE / "calls.jsonl", "a") as f:
        f.write(json.dumps(argv) + "\n")
    if not argv:
        print("usage: flux COMMAND", file=sys.stderr)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "_run":
        run_job(rest[0])
        return 0
    if cmd == "version":
        print("commands:    \t\t0.68.0\nlibflux-core:\t\t0.68.0\nbuild-options:\t\t+systemd")
        return 0
    if cmd == "batch":
        return cmd_batch(rest)
    _need_instance()
    if cmd == "getattr":
        if rest[:1] == ["instance-level"]:
            print("1" if os.environ.get("FLUX_URI") else "0")     # FLUX_URI: inside an allocation
        else:
            print("4")
        return 0
    if cmd == "uptime":
        print(" 10:00:00 run 3.2d,  owner flux,  depth 0,  size 4")
        return 0
    if cmd == "resource" and rest[:1] == ["info"]:
        print("4 Nodes, 448 Cores, 0 GPUs")
        return 0
    if cmd == "resource" and rest[:1] == ["list"]:
        print("     STATE NNODES   NCORES    NGPUS NODELIST\n      free      3      336        0 node[2-4]\n"
              " allocated      1      112        0 node1\n      down      0        0        0 ")
        return 0
    if cmd == "queue" and rest[:1] == ["list"]:
        print("     QUEUE    EN ST TDEFAULT   TLIMIT     NNODES     NCORES      NGPU\n"
              "   pbatch*    ✔  ✔       1h       1d      0-inf      0-inf     0-inf\n"
              "    pdebug    ✔  ✔      30m       1h        0-2      0-inf     0-inf")
        return 0
    if cmd == "jobs":
        return cmd_jobs(rest)
    if cmd == "cancel":
        return cmd_cancel(rest)
    if cmd == "job" and rest[:1] == ["eventlog"]:
        return 1
    print(f"flux: unknown command {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    STATE.mkdir(parents=True, exist_ok=True)
    sys.exit(main(sys.argv[1:]))
