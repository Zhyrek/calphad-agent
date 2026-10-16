"""Feature 6: ESPEI input files + Flux (flux-framework) batch scripts.

Two layouts are generated:

``per-job`` (default)
    Each run gets its own directory with ``espei_generate.yaml`` / ``espei_mcmc.yaml``
    and two Flux batch scripts: a 1-core parameter-generation job and a multi-node
    MCMC job that starts a dask scheduler + one ``dask worker`` per node via
    ``flux submit`` and points ESPEI at the scheduler file.  ``submit_all.sh``
    submits everything, chaining MCMC after generation with ``--dependency=afterok``.

``bundle``
    One ``flux batch`` allocation; inside it every run is dispatched with
    ``flux submit`` using its own slice of cores (ESPEI's local dask cluster),
    then ``flux job wait --all``.  Good for many small runs (several binaries,
    seeds, prior/weight sweeps) sharing one allocation.

CLI:  calphad-tools espei-flux campaign.yaml -o espei_runs/
      calphad-tools phase-models my.tdb --elements CR NI -o phase_models.json
"""
from __future__ import annotations

import json
import os
import shlex
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


@dataclass
class Run:
    name: str
    phase_models: str
    datasets: str
    input_db: Optional[str] = None           # start MCMC from this TDB (skip generation if set and generate=False)
    generate_parameters: bool = True
    ref_state: str = "SGTE91"
    excess_model: str = "linear"
    ridge_alpha: Optional[float] = None
    aicc_penalty_factor: Optional[Dict] = None
    fitting_description: Optional[str] = None  # import path of an ESPEI ModelFittingDescription (default: Gibbs energy)
    iterations: int = 1000
    save_interval: int = 20
    chains_per_parameter: int = 2
    chain_std_deviation: float = 0.1
    deterministic: bool = True
    prior: Optional[Dict] = None
    data_weights: Optional[Dict] = None
    symbols: Optional[List[str]] = None
    restart_trace: Optional[str] = None
    verbosity: int = 2
    cores: Optional[int] = None              # bundle mode: cores for this run
    tags: Optional[Dict] = None              # system.tags: {tag: {excluded_model_contributions, weight, ...}}
    extra_mcmc: Dict = field(default_factory=dict)


def _default_setup() -> List[str]:
    """Environment lines for batch scripts: the installer's pycalphad-agent-env.sh (python module + venv) when
    present, else placeholders to edit."""
    env = Path(__file__).resolve().parents[1] / "pycalphad-agent-env.sh"
    if env.is_file():
        return [f". {env}   # python module + venv, written by install.sh"]
    return ["# module load python   # <- edit for your cluster",
            "# source /path/to/venv/bin/activate"]


@dataclass
class FluxSettings:
    nodes: int = 1
    cores_per_node: int = 36
    time_limit: str = "12h"
    queue: Optional[str] = None               # e.g. pbatch / pdebug
    bank: Optional[str] = None
    gen_time_limit: str = "2h"
    setup: List[str] = field(default_factory=lambda: _default_setup())
    mode: str = "per-job"                     # per-job | bundle
    job_prefix: str = "espei"
    worker_memory_limit: str = "0"            # dask worker --memory-limit
    bank_flag: str = "--bank"                 # older flux-core without --bank: "--setattr=system.bank"


def _abs(p: Optional[str], base: Path) -> Optional[str]:
    if p is None:
        return None
    pp = Path(p).expanduser()
    return str(pp if pp.is_absolute() else (base / pp).resolve())


def espei_yaml(run: Run, mcmc_input_db: Optional[str], scheduler: Optional[str], cores: Optional[int],
               stage: str, base: Path) -> Dict:
    d: Dict = {"system": {"phase_models": _abs(run.phase_models, base), "datasets": _abs(run.datasets, base)},
               "output": {"verbosity": run.verbosity, "logfile": f"{stage}.log"}}
    if run.tags:
        d["system"]["tags"] = run.tags
    if stage == "generate":
        gp = {"excess_model": run.excess_model, "ref_state": run.ref_state}
        if run.ridge_alpha is not None:
            gp["ridge_alpha"] = run.ridge_alpha
        if run.aicc_penalty_factor:
            gp["aicc_penalty_factor"] = run.aicc_penalty_factor
        if run.fitting_description:
            gp["fitting_description"] = run.fitting_description
        if run.input_db:
            gp["input_db"] = _abs(run.input_db, base)
        d["generate_parameters"] = gp
        d["output"]["output_db"] = "dft.tdb"
    else:
        m = {"iterations": run.iterations, "save_interval": run.save_interval,
             "scheduler": scheduler, "input_db": mcmc_input_db, "deterministic": run.deterministic}
        if run.restart_trace:
            m["restart_trace"] = _abs(run.restart_trace, base)
        else:
            m["chains_per_parameter"] = run.chains_per_parameter
            m["chain_std_deviation"] = run.chain_std_deviation
        if cores:
            m["cores"] = cores
        if run.prior:
            m["prior"] = run.prior
        if run.data_weights:
            m["data_weights"] = run.data_weights
        if run.symbols:
            m["symbols"] = run.symbols
        m.update(run.extra_mcmc)
        d["mcmc"] = m
        d["output"].update({"output_db": "mcmc.tdb", "tracefile": "trace.npy", "probfile": "lnprob.npy"})
    return d


def validate_espei_yaml(d: Dict) -> Optional[str]:
    """Return None if ESPEI's schema accepts the settings, else the error text."""
    try:
        from espei.espei_script import get_run_settings
        get_run_settings(json.loads(json.dumps(d)))
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _directives(fs: FluxSettings, name: str, nodes: int, ncores: Optional[int], time_limit: str,
                exclusive: bool = True) -> List[str]:
    lines = [f"#flux: --job-name={name}", f"#flux: -N {nodes}"]
    if ncores:
        lines.append(f"#flux: -n {ncores}")
    if exclusive:
        lines.append("#flux: --exclusive")
    lines.append(f"#flux: -t {time_limit}")
    if fs.queue:
        lines.append(f"#flux: -q {fs.queue}")
    if fs.bank:
        lines.append(f"#flux: {fs.bank_flag or '--bank'}={fs.bank}")
    lines.append(f"#flux: --output={name}.{{{{id}}}}.out")
    return lines


def _header(fs: FluxSettings) -> List[str]:
    # NB: no cd "$(dirname "$0")" - flux batch runs a copy of the script; the job's
    # working directory is the directory it was submitted from (submit_all.sh cds there).
    return ["set -euo pipefail", ""] + fs.setup + [
        "", "command -v espei >/dev/null || { echo 'espei not found - edit the setup lines'; exit 1; }", ""]


def gen_script(fs: FluxSettings, run: Run) -> str:
    L = ["#!/bin/bash"] + _directives(fs, f"{fs.job_prefix}-gen-{run.name}", 1, 1, fs.gen_time_limit,
                                       exclusive=False)
    L += [""] + _header(fs)
    L += ["# Parameter generation (single core)", "espei --in espei_generate.yaml", ""]
    return "\n".join(L) + "\n"


def mcmc_script(fs: FluxSettings, run: Run) -> str:
    n_workers = fs.nodes * fs.cores_per_node
    L = ["#!/bin/bash"] + _directives(fs, f"{fs.job_prefix}-mcmc-{run.name}", fs.nodes, None, fs.time_limit)
    L += [""] + _header(fs)
    L += [
        "SCHED=$PWD/scheduler.json",
        "rm -f \"$SCHED\"",
        "",
        "# 1) dask scheduler: lightweight, runs in this batch shell (not a Flux job, so it",
        "#    does not take a core away from the workers)",
        "dask scheduler --scheduler-file \"$SCHED\" --no-dashboard > dask-scheduler.log 2>&1 &",
        "SCHED_PID=$!",
        "for i in $(seq 120); do [ -s \"$SCHED\" ] && break; sleep 2; done",
        "[ -s \"$SCHED\" ] || { echo 'dask scheduler did not start'; exit 1; }",
        "",
        f"# 2) one dask worker process per core ({n_workers} total) spread over {fs.nodes} node(s)",
        f"flux submit -N {fs.nodes} --tasks-per-node=1 -c {fs.cores_per_node} --output=dask-workers.log \\",
        f"    dask worker --scheduler-file \"$SCHED\" --nworkers {fs.cores_per_node} --nthreads 1 \\",
        f"    --memory-limit {fs.worker_memory_limit} --no-dashboard",
        "",
        "# 3) wait until workers have connected",
        "python - <<'PY'",
        "from distributed import Client",
        "c = Client(scheduler_file='scheduler.json')",
        f"c.wait_for_workers({max(1, n_workers // 2)}, timeout=900)",
        "print('dask workers connected:', len(c.scheduler_info()['workers']))",
        "PY",
        "",
        "# 4) MCMC",
        "status=0",
        "espei --in espei_mcmc.yaml || status=$?",
        "",
        "# 5) tear down dask",
        "flux cancel --all >/dev/null 2>&1 || flux job cancelall -f >/dev/null 2>&1 || true",
        "kill $SCHED_PID 2>/dev/null || true",
        "exit $status",
    ]
    return "\n".join(L) + "\n"


def bundle_script(fs: FluxSettings, runs: List[Run]) -> str:
    L = ["#!/bin/bash"] + _directives(fs, f"{fs.job_prefix}-bundle", fs.nodes, None, fs.time_limit)
    L += [""] + _header(fs)
    L += [f"# {len(runs)} ESPEI runs dispatched inside one allocation "
          f"({fs.nodes} node(s) x {fs.cores_per_node} cores).",
          "# Flux schedules them onto free cores; generation then MCMC run back-to-back per run.", ""]
    for r in runs:
        cores = r.cores or max(1, (fs.nodes * fs.cores_per_node) // max(1, len(runs)))
        cores = min(cores, fs.cores_per_node)
        steps = []
        if r.generate_parameters:
            steps.append("espei --in espei_generate.yaml")
        steps.append("espei --in espei_mcmc.yaml")
        cmd = " && ".join(steps)
        L += [f"# run {r.name}: {cores} cores (ESPEI local dask cluster)",
              f"flux submit -N1 -n1 -c {cores} --job-name={r.name} --output={r.name}/{r.name}.out \\",
              f"    bash -c {shlex.quote(f'cd {r.name} && ' + cmd)}", ""]
    L += ["flux job wait --all", ""]
    return "\n".join(L) + "\n"


def generate(runs: Sequence[Run], flux: FluxSettings, out_dir: str, base_dir: Optional[str] = None) -> Dict:
    """Write ESPEI YAMLs + Flux scripts for every run.  Returns a summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir or os.getcwd())
    written, problems = [], {}
    for r in runs:
        d = out / r.name
        d.mkdir(parents=True, exist_ok=True)
        bundle = flux.mode == "bundle"
        cores = (r.cores or max(1, (flux.nodes * flux.cores_per_node) // max(1, len(runs)))) if bundle else None
        sched = "dask" if bundle else str((d / "scheduler.json").resolve())
        if r.generate_parameters:
            g = espei_yaml(r, None, None, None, "generate", base)
            (d / "espei_generate.yaml").write_text(yaml.safe_dump(g, sort_keys=False))
            written.append(str(d / "espei_generate.yaml"))
            err = validate_espei_yaml(g)
            if err:
                problems[f"{r.name}/espei_generate.yaml"] = err
            mcmc_db = "dft.tdb"
        else:
            if not r.input_db:
                raise ValueError(f"run {r.name}: need input_db when generate_parameters is False")
            mcmc_db = _abs(r.input_db, base)
        m = espei_yaml(r, mcmc_db, sched, min(cores, flux.cores_per_node) if cores else None, "mcmc", base)
        (d / "espei_mcmc.yaml").write_text(yaml.safe_dump(m, sort_keys=False))
        written.append(str(d / "espei_mcmc.yaml"))
        err = validate_espei_yaml(m)
        if err:
            problems[f"{r.name}/espei_mcmc.yaml"] = err
        if not bundle:
            if r.generate_parameters:
                _write_exec(d / "gen.flux", gen_script(flux, r))
                written.append(str(d / "gen.flux"))
            _write_exec(d / "mcmc.flux", mcmc_script(flux, r))
            written.append(str(d / "mcmc.flux"))
    # master submit script
    if flux.mode == "bundle":
        _write_exec(out / "bundle.flux", bundle_script(flux, list(runs)))
        written.append(str(out / "bundle.flux"))
        submit = ["#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"', "flux batch bundle.flux", ""]
    else:
        submit = ["#!/bin/bash", "# Submit every ESPEI run; MCMC waits for its generation step (afterok).",
                  "set -euo pipefail", 'cd "$(dirname "$0")"', ""]
        for r in runs:
            if r.generate_parameters:
                submit += [f"( cd {r.name} && gen=$(flux batch gen.flux) && echo \"{r.name} gen: $gen\" && \\",
                           f"  mcmc=$(flux batch --dependency=afterok:$gen mcmc.flux) && echo \"{r.name} mcmc: $mcmc\" )"]
            else:
                submit += [f"( cd {r.name} && echo \"{r.name} mcmc: $(flux batch mcmc.flux)\" )"]
        submit.append("")
    _write_exec(out / "submit_all.sh", "\n".join(submit))
    written.append(str(out / "submit_all.sh"))
    # syntax-check shell scripts
    import subprocess
    for f in written:
        if f.endswith((".flux", ".sh")):
            r_ = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
            if r_.returncode:
                problems[f] = r_.stderr.strip()
    return {"out_dir": str(out), "files": written, "problems": problems, "mode": flux.mode,
            "submit": f"bash {out / 'submit_all.sh'}"}


def write_per_job_run(run: Run, fs: FluxSettings, run_dir: str, base_dir: Optional[str] = None) -> Dict:
    """One run in the per-job layout, written straight into ``run_dir`` (no campaign-level submit_all.sh):
    espei_generate.yaml + gen.flux (when run.generate_parameters), espei_mcmc.yaml + mcmc.flux.  YAMLs are checked
    against ESPEI's schema and scripts with ``bash -n``.  Returns {files, problems, scripts: {step: script},
    output_prefix: {step: prefix}}."""
    import subprocess
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir or os.getcwd())
    files, problems, scripts, prefix = [], {}, {}, {}
    if run.generate_parameters:
        g = espei_yaml(run, None, None, None, "generate", base)
        (d / "espei_generate.yaml").write_text(yaml.safe_dump(g, sort_keys=False))
        err = validate_espei_yaml(g)
        if err:
            problems["espei_generate.yaml"] = err
        _write_exec(d / "gen.flux", gen_script(fs, run))
        files += [str(d / "espei_generate.yaml"), str(d / "gen.flux")]
        scripts["generate"] = "gen.flux"
        prefix["generate"] = f"{fs.job_prefix}-gen-{run.name}"
        mcmc_db = "dft.tdb"
    else:
        if not run.input_db:
            raise ValueError(f"run {run.name}: need input_db when generate_parameters is False")
        mcmc_db = _abs(run.input_db, base)
    m = espei_yaml(run, mcmc_db, str((d / "scheduler.json").resolve()), None, "mcmc", base)
    (d / "espei_mcmc.yaml").write_text(yaml.safe_dump(m, sort_keys=False))
    err = validate_espei_yaml(m)
    if err:
        problems["espei_mcmc.yaml"] = err
    _write_exec(d / "mcmc.flux", mcmc_script(fs, run))
    files += [str(d / "espei_mcmc.yaml"), str(d / "mcmc.flux")]
    scripts["mcmc"] = "mcmc.flux"
    prefix["mcmc"] = f"{fs.job_prefix}-mcmc-{run.name}"
    for f in files:
        if f.endswith(".flux"):
            r_ = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
            if r_.returncode:
                problems[Path(f).name] = r_.stderr.strip()
    return {"files": files, "problems": problems, "scripts": scripts, "output_prefix": prefix}


def _write_exec(p: Path, text: str) -> None:
    p.write_text(text)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def load_campaign(path: str) -> (List[Run], FluxSettings):
    """Campaign YAML:

    flux: {nodes: 2, cores_per_node: 56, time_limit: 12h, queue: pbatch, bank: mybank,
           mode: per-job, setup: ["module load python/3.11", "source ~/venvs/espei/bin/activate"]}
    defaults: {iterations: 2000, ref_state: SGTE91}
    runs:
      - {name: CrNi, phase_models: CrNi/phase_models.json, datasets: CrNi/datasets}
      - {name: CrFe, phase_models: ..., datasets: ..., input_db: CrFe_start.tdb, generate_parameters: false}
    """
    cfg = yaml.safe_load(Path(path).read_text())
    fs = FluxSettings(**(cfg.get("flux") or {}))
    defaults = cfg.get("defaults") or {}
    runs = [Run(**{**defaults, **r}) for r in cfg["runs"]]
    return runs, fs


# ---------------------------------------------------------------------------
# phase_models.json from a TDB
# ---------------------------------------------------------------------------
def phase_models_from_tdb(tdb: str, elements: Sequence[str], output: Optional[str] = None,
                          phases: Optional[Sequence[str]] = None) -> Dict:
    """ESPEI phase-models spec restricted to ``elements`` (pure-element constituents + VA)."""
    from .tdbparse import parse_tdb
    db = parse_tdb(tdb)
    els = {e.upper() for e in elements}
    allowed = els | {"VA"}
    out = {"components": sorted(els | ({"VA"} if any("VA" in sl for p in db.phases.values()
                                                    for sl in p.constituents) else set())),
           "phases": {}}
    skipped = []
    for name, ph in sorted(db.phases.items()):
        if phases and name not in {p.upper() for p in phases}:
            continue
        subl = [[s for s in sl if s in allowed] for sl in ph.constituents]
        if any(not sl for sl in subl) or not any(s in els for sl in subl for s in sl):
            skipped.append(name)
            continue
        out["phases"][name] = {"sublattice_model": [sorted(sl) for sl in subl],
                               "sublattice_site_ratios": [r for r in ph.site_ratios]}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))
    try:
        from espei.phase_models import PhaseModelSpecification
        PhaseModelSpecification.model_validate(out)
    except Exception as e:  # noqa: BLE001
        out["_validation_error"] = str(e)
    out["_skipped_phases"] = skipped
    return out
