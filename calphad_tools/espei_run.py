"""Local ESPEI runs: parameter generation and MCMC on this machine (MCMC as a background job).

Tools (registered by :func:`register`):

* ``espei_generate_parameters`` - ESPEI parameter generation (AICc model selection over the single-phase
  thermochemical datasets) -> ``dft.tdb`` with ``VV####`` symbols + a per-phase summary of the chosen
  parameters.  Small inputs run synchronously, large ones as a background job.
* ``espei_run_mcmc`` - ESPEI MCMC as a background job (:mod:`calphad_tools.jobs`): progress per iteration
  (mean / max lnprob), ``job_status`` shows an lnprob convergence plot at any time, ``stop_job`` keeps the
  chains sampled so far.
* ``espei_validate_input`` - YAML schema, phase models, every dataset (ESPEI checker), datasets vs phase
  models, and known ESPEI 0.9.1 pitfalls, before anything runs.

Layout: every run lives in ``<output dir>/espei_runs/<name>/`` with ``espei_generate.yaml`` /
``espei_mcmc.yaml`` built by :func:`calphad_tools.espei_flux.espei_yaml` - the *same* YAML structure that
``generate_espei_flux_batch`` writes, so a quick local check and the Flux production run share one config
(only ``scheduler``/``cores`` differ).  ESPEI runs with the run directory as working directory, so its
outputs are ``dft.tdb``, ``trace.npy``, ``lnprob.npy``, ``mcmc.tdb``, ``generate.log``, ``mcmc.log`` there -
exactly what ``analyze_mcmc`` / ``mcmc_to_tdb`` / ``propagate_uncertainty`` expect (``trace=<run dir>``).

Known ESPEI 0.9.1 bug: ``EquilibriumPropertyResidual`` evaluates the points of an equilibrium-property
dataset (no ``solver``; HM/SM/CPM... at equilibrium) in X-major order but flattens ``values`` T/P-major, so a
dataset with several T (or P) *and* several compositions is compared in the wrong order.  The checks here
flag such datasets (split them: at most one of n_P, n_T, n_X > 1; the ``write_equilibrium_dataset`` writers
already do).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from . import espei_flux

RUNS_DIR = "espei_runs"
FITTING_DESCRIPTIONS = {
    "gibbs": "espei.parameter_selection.fitting_descriptions.gibbs_energy_fitting_description",
    "molar_volume": "espei.parameter_selection.fitting_descriptions.molar_volume_fitting_description",
    "molar_volume_gibbs": "espei.parameter_selection.fitting_descriptions."
                          "molar_volume_gibbs_energy_fitting_description",
}
SYNC_MAX_FIT_DATASETS = 60          # espei_generate_parameters runs as a job above this many single-phase datasets
SYNC_MAX_PHASES = 12


# ---------------------------------------------------------------------------
# settings (shared by the tools and the show_code generators)
# ---------------------------------------------------------------------------
def runs_root(tb) -> Path:
    return Path(tb.out) / RUNS_DIR


def _safe_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip()).strip("._")
    if not n:
        raise ValueError("give a run name (letters, digits, _ - .)")
    return n


def _previous_system(d: Path) -> Dict[str, str]:
    """phase_models / datasets / input_db of earlier YAMLs in the run directory (defaults for a follow-up call)."""
    out: Dict[str, str] = {}
    for y in ("espei_mcmc.yaml", "espei_generate.yaml"):
        p = d / y
        if p.exists():
            try:
                cfg = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            for k in ("phase_models", "datasets"):
                if (cfg.get("system") or {}).get(k):
                    out.setdefault(k, cfg["system"][k])
            idb = (cfg.get("mcmc") or {}).get("input_db")
            if idb:
                out.setdefault("input_db", idb if os.path.isabs(idb) else str(d / idb))
    return out


def _fitting_description(fd: Optional[str]) -> Optional[str]:
    if not fd:
        return None
    return FITTING_DESCRIPTIONS.get(fd, fd)


def generate_settings(tb, args: Dict[str, Any]) -> Tuple[Path, Dict[str, Any], espei_flux.Run]:
    """(run dir, ESPEI generate_parameters settings, Run) for espei_generate_parameters arguments."""
    name = _safe_name(args.get("name") or "espei")
    d = runs_root(tb) / name
    prev = _previous_system(d)
    pm = args.get("phase_models") or prev.get("phase_models")
    ds = args.get("datasets") or prev.get("datasets")
    if not pm or not ds:
        raise ValueError("give phase_models (JSON, see make_phase_models) and datasets (directory)")
    run = espei_flux.Run(name=name, phase_models=tb.p(pm), datasets=tb.p(ds),
                         input_db=tb._find_tdb(args["input_db"]) if args.get("input_db") else None,
                         ref_state=args.get("ref_state") or "SGTE91",
                         excess_model=args.get("excess_model") or "linear",
                         ridge_alpha=args.get("ridge_alpha"),
                         aicc_penalty_factor=args.get("aicc_penalty_factor") or None,
                         fitting_description=_fitting_description(args.get("fitting_description")),
                         verbosity=2)
    settings = espei_flux.espei_yaml(run, None, None, None, "generate", Path(tb.workdir))
    return d, settings, run


def mcmc_run_dir(tb, args: Dict[str, Any], restart_trace: Optional[str] = None) -> Tuple[Path, bool]:
    """Run directory for a new MCMC run: <name>, or the next free <name>_2, _3 ... when <name> already holds a
    trace (ESPEI refuses to overwrite one); with overwrite the old trace/lnprob/mcmc.tdb are deleted."""
    name = _safe_name(args.get("name") or "espei")
    d = runs_root(tb) / name
    if not ((d / "trace.npy").exists() or (d / "lnprob.npy").exists()):
        return d, False
    if args.get("overwrite"):
        if restart_trace and Path(restart_trace).resolve() == (d / "trace.npy").resolve():
            raise ValueError("cannot overwrite the trace you restart from; drop overwrite (a new run dir is used)")
        for f in ("trace.npy", "lnprob.npy", "mcmc.tdb", "lnprob_progress.png"):
            if (d / f).exists():
                (d / f).unlink()
        return d, False
    k = 2
    while (runs_root(tb) / f"{name}_{k}").exists():
        k += 1
    return runs_root(tb) / f"{name}_{k}", True


def _restart_path(tb, rt: Optional[str]) -> Optional[str]:
    if not rt:
        return None
    rp = Path(tb.p(rt))
    rt = str(rp / "trace.npy" if rp.is_dir() else rp)
    if not Path(rt).exists():
        raise FileNotFoundError(f"restart trace {rt} not found")
    return rt


def default_cores(tb, n_chains: Optional[int] = None) -> Tuple[int, str]:
    """(cores, why) for a direct (local) MCMC run: [hpc] local_cores (default half of the available cores, at
    least 1), never more than the number of chains (emcee evaluates the chains in parallel)."""
    from .hpc import available_cores, config_of, local_cores
    setting = config_of(tb).local_cores
    n_avail = available_cores()
    c = local_cores(setting, n_avail)
    why = f"local_cores={setting!r} of {n_avail} available"
    if n_chains and c > n_chains:
        c, why = int(n_chains), why + f", capped at {n_chains} chains"
    return max(1, c), why


def _n_chains(run: espei_flux.Run, n_sym: int) -> Optional[int]:
    if run.restart_trace:
        try:
            return int(np.load(run.restart_trace, mmap_mode="r").shape[0])
        except Exception:  # noqa: BLE001
            return None
    n = len(run.symbols) if run.symbols else n_sym
    return int(run.chains_per_parameter) * n if n else None


def mcmc_settings(tb, args: Dict[str, Any], run_dir: Optional[Path] = None, n_chains: Optional[int] = None
                  ) -> Tuple[Path, Dict[str, Any], espei_flux.Run, Dict[str, Any]]:
    """(run dir, ESPEI mcmc settings, Run, info) for espei_run_mcmc arguments (no side effects).  Defaults
    (phase_models, datasets, input_db = dft.tdb) come from the directory of the run *name*; ``run_dir`` is where
    it will actually run (see :func:`mcmc_run_dir`).  ``info``: resolved input_db, cores and where they came from
    (``cores`` argument, else :func:`default_cores` - half of the available cores, capped at ``n_chains``)."""
    name = _safe_name(args.get("name") or "espei")
    d0 = runs_root(tb) / name
    prev = _previous_system(d0)
    pm = args.get("phase_models") or prev.get("phase_models")
    ds = args.get("datasets") or prev.get("datasets")
    if not pm or not ds:
        raise ValueError("give phase_models and datasets (or run espei_generate_parameters with this name first)")
    if args.get("input_db"):
        idb = tb._find_tdb(args["input_db"])
    elif (d0 / "dft.tdb").exists():
        idb = str(d0 / "dft.tdb")
    elif prev.get("input_db") and Path(prev["input_db"]).exists():
        idb = prev["input_db"]
    else:
        raise ValueError(f"no input TDB: give input_db or run espei_generate_parameters name={name} first")
    rt = _restart_path(tb, args.get("restart_trace"))
    d = Path(run_dir) if run_dir else d0
    iters = int(args.get("iterations") or 50)
    save = args.get("save_interval")
    save = int(save) if save else max(1, min(20, iters // 20))
    if args.get("cores"):
        cores, cores_why = int(args["cores"]), "cores argument"
    else:
        cores, cores_why = default_cores(tb, n_chains)
    run = espei_flux.Run(name=d.name, phase_models=tb.p(pm), datasets=tb.p(ds), input_db=idb,
                         generate_parameters=False, iterations=iters, save_interval=save,
                         chains_per_parameter=int(args.get("chains_per_parameter") or 2),
                         chain_std_deviation=float(args.get("chain_std_deviation") or 0.1),
                         deterministic=True if args.get("deterministic") is None else bool(args["deterministic"]),
                         prior=args.get("prior") or None, data_weights=args.get("data_weights") or None,
                         symbols=args.get("symbols") or None, restart_trace=rt, verbosity=2)
    same_dir = Path(idb).resolve().parent == d.resolve()
    settings = espei_flux.espei_yaml(run, Path(idb).name if same_dir else idb, "dask" if cores > 1 else None,
                                     cores if cores > 1 else None, "mcmc", Path(tb.workdir))
    return d, settings, run, {"input_db": idb, "requested_dir": d0, "cores": cores, "cores_why": cores_why}


def flux_entry(run: espei_flux.Run, generate: bool, rel: Callable[[str], str] = str) -> Dict[str, Any]:
    """The generate_espei_flux_batch ``runs`` entry that reproduces this local configuration."""
    e: Dict[str, Any] = {"name": run.name, "phase_models": rel(run.phase_models), "datasets": rel(run.datasets),
                         "generate_parameters": generate, "ref_state": run.ref_state}
    if not generate:
        e["input_db"] = rel(run.input_db)
    for k in ("iterations", "chains_per_parameter", "chain_std_deviation", "symbols", "prior", "data_weights"):
        v = getattr(run, k)
        if v not in (None, [], {}):
            e[k] = v
    return e


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------
def _load_json(p: Path) -> Optional[Dict]:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _n(v) -> int:
    return len(v) if isinstance(v, (list, tuple)) else 1


def check_inputs(phase_models: str, datasets: str, settings: Optional[Dict] = None,
                 run_dir: Optional[str] = None) -> Dict[str, Any]:
    """Problems (ESPEI would fail / silently ignore data) and warnings for a run's inputs."""
    from .expdata import check_datasets_dir
    problems: List[str] = []
    warn: List[str] = []
    counts: Dict[str, int] = {}
    fit_by_phase: Dict[str, int] = {}
    pmp, dsp = Path(phase_models), Path(datasets)
    pm = _load_json(pmp) if pmp.exists() else None
    if pm is None:
        problems.append(f"phase_models {phase_models}: missing or not valid JSON")
        pm = {"components": [], "phases": {}}
    else:
        try:
            from espei.phase_models import PhaseModelSpecification
            PhaseModelSpecification.model_validate(pm)
        except Exception as e:  # noqa: BLE001
            problems.append(f"phase_models: {type(e).__name__}: {str(e)[:300]}")
    comps = {c.upper() for c in pm.get("components", [])}
    phases = set((pm.get("phases") or {}).keys())
    if settings:
        err = espei_flux.validate_espei_yaml(settings)
        if err:
            problems.append(f"YAML rejected by ESPEI's schema: {err[:400]}")
    if not dsp.is_dir():
        problems.append(f"datasets directory {datasets} not found")
        files = []
    else:
        chk = check_datasets_dir(str(dsp))
        for f, e in chk["invalid"].items():
            problems.append(f"dataset {Path(f).name}: {e[:200]}")
        files = sorted(dsp.rglob("*.json"))
        if not files:
            problems.append(f"no *.json datasets under {datasets}")
    for f in files:
        ds = _load_json(f)
        if not isinstance(ds, dict) or "output" not in ds:
            continue
        if ds.get("disabled"):
            counts["disabled"] = counts.get("disabled", 0) + 1
            continue
        out = str(ds["output"])
        kind = ("ZPF" if out == "ZPF" else "ACR" if out.startswith("ACR") else
                "fixed-config " + out if "solver" in ds else "equilibrium " + out)
        counts[kind] = counts.get(kind, 0) + 1
        dcomps = {c.upper() for c in ds.get("components", [])}
        if comps and not dcomps <= comps:
            warn.append(f"{f.name}: components {sorted(dcomps - comps)} not in phase_models -> ignored by ESPEI")
        dph = set(ds.get("phases") or [])
        if phases and dph and not dph & phases:
            warn.append(f"{f.name}: none of its phases {sorted(dph)} are in phase_models -> ignored")
        elif phases and dph - phases and out == "ZPF":
            warn.append(f"{f.name}: phases {sorted(dph - phases)} not modelled")
        if "solver" in ds:
            for ph in dph:
                fit_by_phase[ph] = fit_by_phase.get(ph, 0) + 1
        elif out not in ("ZPF",) and not out.startswith("ACR"):
            c = ds.get("conditions") or {}
            n_x = max([_n(v) for k, v in c.items() if k.upper().startswith("X_")] or [1])
            if (_n(c.get("T")) > 1 or _n(c.get("P")) > 1) and n_x > 1:
                warn.append(f"{f.name}: several T/P AND several compositions - ESPEI 0.9.1 "
                            "EquilibriumPropertyResidual compares these points in the wrong order; split the dataset "
                            "(at most one of n_P, n_T, n_X > 1)")
    no_fit = sorted(phases - set(fit_by_phase))
    if phases and no_fit and not (settings and settings.get("mcmc")):
        warn.append(f"no single-phase (fixed-configuration) data for {no_fit}: parameter generation gives them "
                    "only endmember reference terms (no fitted VV symbols)")
    if settings and settings.get("mcmc"):
        m = settings["mcmc"]
        idb = m.get("input_db")
        if idb:
            ip = Path(idb) if os.path.isabs(idb) else Path(run_dir or ".") / idb
            if not ip.exists():
                problems.append(f"mcmc.input_db {idb} not found")
            else:
                try:
                    from pycalphad import Database
                    from espei.utils import database_symbols_to_fit
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        syms = database_symbols_to_fit(Database(str(ip)))
                    counts["fitted symbols in input_db"] = len(syms)
                    if not syms:
                        problems.append(f"{ip.name} has no VV#### symbols to fit (MCMC needs generated or "
                                        "renamed parameters)")
                    bad = sorted(set(m.get("symbols") or []) - set(syms)) if m.get("symbols") else []
                    if bad:
                        problems.append(f"mcmc.symbols not in {ip.name}: {bad}")
                except Exception as e:  # noqa: BLE001
                    problems.append(f"input_db {ip.name} does not load: {type(e).__name__}: {e}")
        if run_dir:
            for f in ("trace.npy", "lnprob.npy"):
                if (Path(run_dir) / f).exists():
                    problems.append(f"{f} already exists in the run dir (ESPEI refuses to overwrite it)")
    return {"ok": not problems, "problems": problems, "warnings": warn, "dataset_counts": counts,
            "fixed_config_datasets_per_phase": fit_by_phase, "phases": sorted(phases),
            "components": sorted(comps)}


def _check_text(chk: Dict[str, Any], rel: Callable[[str], str] = str) -> str:
    L = ["inputs OK" if chk["ok"] else "PROBLEMS (ESPEI would fail or ignore data):"]
    L += [f"  - {p}" for p in chk["problems"]]
    if chk["warnings"]:
        L.append("warnings:")
        L += [f"  - {w}" for w in chk["warnings"][:25]]
        if len(chk["warnings"]) > 25:
            L.append(f"  ... {len(chk['warnings']) - 25} more")
    if chk["dataset_counts"]:
        L.append("datasets: " + ", ".join(f"{k}: {v}" for k, v in sorted(chk["dataset_counts"].items())))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# parameter generation
# ---------------------------------------------------------------------------
class _EspeiLogFile:
    """Send ESPEI's logging to ``filename`` for the duration of a block, without reconfiguring the root logger
    (ESPEI's own ``config_logger`` would clear the agent's handlers)."""

    def __init__(self, filename: Optional[str], verbosity: int):
        self.filename, self.level = filename, {0: 30, 1: 20, 2: 15, 3: 10}.get(int(verbosity or 0), 20)

    def __enter__(self):
        lg = logging.getLogger("espei")
        self.old = (lg.level, lg.propagate)
        self.h = logging.FileHandler(self.filename) if self.filename else None
        if self.h:
            self.h.setFormatter(logging.Formatter("%(levelname)s:%(name)s - %(message)s"))
            lg.addHandler(self.h)
        lg.setLevel(self.level)
        return self

    def __exit__(self, *exc):
        lg = logging.getLogger("espei")
        if self.h:
            lg.removeHandler(self.h)
            self.h.close()
        lg.setLevel(self.old[0])


def run_generate(run_dir: str, settings: Optional[Dict] = None) -> Dict[str, Any]:
    """ESPEI parameter generation exactly as ``run_espei`` does it (same settings dict), via the Python API
    so the caller's logging is left alone.  Writes output_db (dft.tdb) and generate_summary.json."""
    from espei import generate_parameters
    from espei.datasets import apply_tags, load_datasets, recursive_glob
    from espei.espei_script import get_run_settings
    from pycalphad import Database
    from tinydb import where
    d = Path(run_dir)
    if settings is None:
        settings = yaml.safe_load((d / "espei_generate.yaml").read_text())
    rs = get_run_settings(json.loads(json.dumps(settings)))
    sysd, outd, gp = rs["system"], rs["output"], rs["generate_parameters"]
    t0 = time.time()
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with _EspeiLogFile(outd.get("logfile"), outd.get("verbosity", 0)), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            datasets = load_datasets(sorted(recursive_glob(sysd["datasets"], "*.json")))
            apply_tags(datasets, sysd.get("tags", dict()))
            datasets.remove(where("disabled") == True)  # noqa: E712 - tinydb query
            with open(sysd["phase_models"]) as fp:
                phase_models = json.load(fp)
            input_dbf = Database(gp["input_db"]) if gp.get("input_db") else None
            dbf = generate_parameters(phase_models, datasets, gp["ref_state"], gp["excess_model"],
                                      ridge_alpha=gp["ridge_alpha"], dbf=input_dbf,
                                      aicc_penalty_factor=gp["aicc_penalty_factor"],
                                      fitting_description=gp["fitting_description"])
            dbf.to_file(outd["output_db"], if_exists="overwrite")
    finally:
        os.chdir(cwd)
    tdb = d / outd["output_db"]
    summ = parameter_summary(str(tdb), list(phase_models.get("phases", {})))
    summ["seconds"] = round(time.time() - t0, 2)
    summ["n_datasets"] = len(datasets.all())
    (d / "generate_summary.json").write_text(json.dumps(summ, indent=1))
    return {"tdb": str(tdb), "summary": summ, "files": [str(tdb), str(d / "generate_summary.json")]}


def parameter_summary(tdb: str, phases: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fitted (VV####) parameters per phase with their values."""
    import symengine
    from espei.utils import database_symbols_to_fit, unpack_piecewise
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(tdb)
    syms = database_symbols_to_fit(db)
    vals = {s: float(unpack_piecewise(db.symbols[s])) for s in syms}
    per: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(db._parameters.all(), key=lambda r: (r["phase_name"], r["parameter_type"],
                                                         str(r["constituent_array"]), r["parameter_order"])):
        e = p["parameter"]
        used = sorted(str(s) for s in getattr(e, "free_symbols", ()) if str(s) in vals)
        if not used:
            continue
        e0 = e.args[0] if isinstance(e, symengine.Piecewise) else e      # single T range as ESPEI writes it
        num = e0.xreplace({symengine.Symbol(s): vals[s] for s in used})
        cons = ":".join(",".join(str(getattr(c, "name", c)) for c in sl) for sl in p["constituent_array"])
        per.setdefault(p["phase_name"], []).append({
            "parameter": f"{p['parameter_type']}({p['phase_name']},{cons};{p['parameter_order']})",
            "expression": str(e0), "value": _pretty(str(num)), "symbols": {s: vals[s] for s in used}})
    return {"tdb": tdb, "n_symbols": len(syms), "symbols": vals, "phases": per,
            "phases_without_fitted_parameters": sorted(set(phases or []) - set(per))}


def _pretty(expr: str) -> str:
    return re.sub(r"(\d+\.\d*?)0{3,}\d*(e[+-]?\d+)?", lambda m: m.group(1).rstrip(".") + (m.group(2) or ""), expr)


def summary_text(summ: Dict[str, Any]) -> str:
    L = [f"{summ['n_symbols']} fitted symbols in {len(summ['phases'])} phase(s):"]
    for ph, rows in summ["phases"].items():
        L.append(f"{ph}:")
        for r in rows:
            L.append(f"  {r['parameter']} = {r['expression']} = {r['value']}")
    if summ.get("phases_without_fitted_parameters"):
        L.append("no fitted parameters (no single-phase data): " + ", ".join(summ["phases_without_fitted_parameters"]))
    return "\n".join(L)


def generate_job(run_dir: str, display_dir: str = "", progress: Callable = None) -> Dict[str, Any]:
    """Job target: parameter generation from <run_dir>/espei_generate.yaml."""
    progress("ESPEI parameter generation", 0.0)
    r = run_generate(run_dir)
    progress("done", 1.0, n_symbols=r["summary"]["n_symbols"])
    dd = display_dir or run_dir
    return {"text": f"wrote {dd}/{Path(r['tdb']).name} in {r['summary']['seconds']} s\n"
                    + summary_text(r["summary"]), "files": r["files"]}


# ---------------------------------------------------------------------------
# MCMC (job target)
# ---------------------------------------------------------------------------
def _lnprob_stats(lp: np.ndarray) -> Dict[str, Any]:
    fin = lp[np.isfinite(lp)]
    if fin.size == 0:
        return {"mean_lnprob": None, "max_lnprob": None, "n_nonfinite": int(lp.size)}
    return {"mean_lnprob": round(float(fin.mean()), 4), "max_lnprob": round(float(fin.max()), 4),
            "median_lnprob": round(float(np.median(fin)), 4), "n_nonfinite": int(lp.size - fin.size)}


def mcmc_job(run_dir: str, display_dir: str = "", progress: Callable = None) -> Dict[str, Any]:
    """Job target: ``espei.espei_script.run_espei`` on <run_dir>/espei_mcmc.yaml with per-iteration progress.

    stop_job: the sampling loop leaves after the current iteration (cooperative stop; a second SIGTERM interrupts
    as KeyboardInterrupt), saves the chains so far and ESPEI writes mcmc.tdb from the best parameters found, as its
    own Ctrl-C handling does."""
    import espei.logger
    import espei.optimizers.opt_mcmc as om
    from espei.espei_script import run_espei
    d = Path(run_dir)
    os.chdir(d)
    settings = yaml.safe_load((d / "espei_mcmc.yaml").read_text())
    iters_total = int(settings["mcmc"]["iterations"])
    state: Dict[str, Any] = {"done": 0, "interrupted": False, "t_sampling": None, "first": None, "last": None}

    orig_config = espei.logger.config_logger

    def config_logger(verbosity=0, filename=None, reset_handlers=True):   # ESPEI log file AND the job log
        orig_config(verbosity=verbosity, filename=filename, reset_handlers=reset_handlers)
        if filename is not None:
            h = logging.StreamHandler()            # job log (log.txt): INFO and up, not the per-chain TRACE
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("%(levelname)s:%(name)s - %(message)s"))
            h.addFilter(logging.Filter("espei"))
            logging.getLogger().addHandler(h)
    espei.logger.config_logger = config_logger

    def do_sampling(self, chains, iterations):
        log = logging.getLogger("espei.optimizers.opt_mcmc")
        log.info("Running MCMC for %s iterations.", iterations)
        t0 = time.time()
        state["t_sampling"] = t0
        progress(f"sampling {chains.shape[0]} chains x {chains.shape[1]} parameters", 0.0, iteration=0,
                 chains=int(chains.shape[0]), parameters=int(chains.shape[1]))
        try:
            for i, _ in enumerate(self.sampler.sample(chains, iterations=iterations)):
                state["done"] = i + 1
                if (i + 1) % self.save_interval == 0:
                    self.save_sampler_state()
                st = _lnprob_stats(np.asarray(self.sampler.lnprobability[:, i]))
                if state["first"] is None:
                    state["first"] = st
                state["last"] = st
                el = time.time() - t0
                progress(f"iteration {i + 1}/{iterations}  mean lnprob {st['mean_lnprob']}  max {st['max_lnprob']}",
                         (i + 1) / iterations, iteration=i + 1, seconds=round(el, 1),
                         s_per_iteration=round(el / (i + 1), 2), **st)
                log.info("(%d of %d) %.1f s; lnprob mean %s max %s", i + 1, iterations, el, st["mean_lnprob"],
                         st["max_lnprob"])
                if getattr(progress, "stop_requested", False):       # stop_job: leave after a whole iteration
                    state["interrupted"] = True
                    log.warning("MCMC stopped after %d iterations - saving the chains so far", state["done"])
                    progress(f"stopping after {state['done']} iterations: saving chains", None)
                    break
        except KeyboardInterrupt:
            state["interrupted"] = True
            log.warning("MCMC interrupted after %d iterations - saving the chains so far", state["done"])
            progress(f"stopping after {state['done']} iterations: saving chains", None)
        log.info("MCMC complete.")
        self.save_sampler_state()

    om.EmceeOptimizer.do_sampling = do_sampling
    progress.cooperative = hasattr(progress, "stop_requested")     # stop between iterations, not inside one
    progress("setting up ESPEI (datasets, likelihood functions)", 0.0)
    t0 = time.time()
    try:
        run_espei(settings)
    except KeyboardInterrupt:
        state["interrupted"] = True
    secs = time.time() - t0
    out = settings.get("output", {})
    names = [out.get("tracefile", "trace.npy"), out.get("probfile", "lnprob.npy"), out.get("output_db", "mcmc.tdb")]
    files = [str(d / n) for n in names if n and (d / n).exists()]
    png = None
    try:
        png = plot_lnprob_progress(d, progress_records=None)
    except Exception:  # noqa: BLE001
        png = None
    if png:
        files.append(png)
    dd = display_dir or str(d)
    f, l = state["first"] or {}, state["last"] or {}
    L = [f"ESPEI MCMC {'STOPPED' if state['interrupted'] else 'finished'}: {state['done']}/{iters_total} iterations "
         f"in {secs:.1f} s" + (f" ({(time.time() - state['t_sampling']) / max(1, state['done']):.2f} s/iteration)"
                               if state["t_sampling"] and state["done"] else ""),
         f"lnprob mean {f.get('mean_lnprob')} -> {l.get('mean_lnprob')}, max {f.get('max_lnprob')} -> "
         f"{l.get('max_lnprob')}",
         f"run dir {dd}: " + ", ".join(Path(x).name for x in files),
         f"next: analyze_mcmc trace={dd} (convergence, burn-in, parameter statistics); mcmc_to_tdb trace={dd}"]
    if state["interrupted"] and not (d / names[2]).exists():
        L.append(f"(no {names[2]}: give analyze_mcmc tdb=<the MCMC input TDB>)")
    return {"text": "\n".join(L), "files": files, "iterations_done": state["done"], "interrupted": state["interrupted"]}


def plot_lnprob_progress(run_dir: Path, progress_records: Optional[List[Dict]] = None) -> Optional[str]:
    """lnprob vs iteration: every chain (from lnprob.npy, saved every save_interval) + mean/max from the per-
    iteration progress records.  Written to <run_dir>/lnprob_progress.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    lp = None
    if (run_dir / "lnprob.npy").exists() and (run_dir / "trace.npy").exists():
        try:
            from .espei_post import truncate_zero_filled
            tr, lpa, _ = truncate_zero_filled(np.load(run_dir / "trace.npy"), np.load(run_dir / "lnprob.npy"))
            lp = np.where(np.isfinite(lpa), lpa, np.nan) if lpa.shape[1] else None
        except Exception:  # noqa: BLE001
            lp = None
    recs = [r for r in (progress_records or []) if (r.get("metrics") or {}).get("iteration")]
    if lp is None and not recs:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if lp is not None:
        it = np.arange(1, lp.shape[1] + 1)
        for c in range(lp.shape[0]):
            ax.plot(it, lp[c], color="0.6", lw=0.5, alpha=0.5)
        ax.plot(it, np.nanmean(lp, axis=0), color="C0", lw=1.6, label="mean (saved chains)")
    if recs:
        it = [r["metrics"]["iteration"] for r in recs]
        mx = [r["metrics"].get("max_lnprob") for r in recs]
        mn = [r["metrics"].get("mean_lnprob") for r in recs]
        ax.plot(it, [np.nan if v is None else v for v in mn], "--", color="C1", lw=1.2, label="mean (live)")
        ax.plot(it, [np.nan if v is None else v for v in mx], ":", color="C3", lw=1.2, label="max (live)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("lnprob")
    ax.set_title(f"ESPEI MCMC {run_dir.name}")
    ax.legend(fontsize=8, loc="lower right")
    fin = lp[np.isfinite(lp)] if lp is not None else np.array([])
    if fin.size > 10:                       # keep the early far-off chains from squashing the plot
        lo = np.percentile(fin, 2)
        hi = np.nanmax(fin)
        ax.set_ylim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
    out = run_dir / "lnprob_progress.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def mcmc_status_hook(tb, jobdir: Path, spec: Dict[str, Any], status: Dict[str, Any]) -> Tuple[str, List[str]]:
    """job_status extra for ESPEI MCMC jobs: iterations, lnprob trend, convergence plot, files so far."""
    from .jobs import progress_records
    d = Path(spec["kwargs"]["run_dir"])
    recs = [r for r in progress_records(jobdir) if (r.get("metrics") or {}).get("iteration")]
    L = []
    if recs:
        m = [r["metrics"] for r in recs]
        n = m[-1]["iteration"]
        spi = f" ({m[-1]['s_per_iteration']} s/iteration)" if m[-1].get("s_per_iteration") else ""
        L.append(f"ESPEI: {n} iterations sampled{spi}; lnprob mean "
                 f"{m[0].get('mean_lnprob')} -> {m[-1].get('mean_lnprob')}, max {m[0].get('max_lnprob')} -> "
                 f"{m[-1].get('max_lnprob')}")
        if n >= 10:
            k = max(2, n // 5)
            a = [x.get("mean_lnprob") for x in m[-2 * k:-k] if x.get("mean_lnprob") is not None]
            b = [x.get("mean_lnprob") for x in m[-k:] if x.get("mean_lnprob") is not None]
            if a and b:
                L.append(f"mean lnprob over the last {k} iterations changed by {np.mean(b) - np.mean(a):+.3g} vs the "
                         f"{k} before (near 0 relative to the spread between chains = stationary; use analyze_mcmc "
                         "for burn-in / R-hat)")
    present = [f for f in ("trace.npy", "lnprob.npy", "mcmc.tdb", "mcmc.log") if (d / f).exists()]
    L.append(f"run dir {tb.rel(str(d))}: {', '.join(present) or 'no outputs yet'}")
    imgs = []
    try:
        png = plot_lnprob_progress(d, recs)
        if png:
            imgs.append(png)
    except Exception as e:  # noqa: BLE001
        L.append(f"(plot failed: {e})")
    return "\n".join(L), imgs


# ---------------------------------------------------------------------------
# optimize_system: generation -> MCMC, batched on Flux or run here
# ---------------------------------------------------------------------------
SUBMISSION_FILE = "hpc_submission.json"      # backend, Flux ids, commands, scripts, resources (show_code reads it)
_ITER_RE = re.compile(r"\((\d+) of (\d+)\)")


def optimize_job(run_dir: str, display_dir: str = "", generate: bool = True, cap_cores: bool = True,
                 next_hint: str = "", progress: Callable = None) -> Dict[str, Any]:
    """Job target (local backend): parameter generation (espei_generate.yaml -> dft.tdb), then MCMC
    (espei_mcmc.yaml).  ``cap_cores``: after generation, lower the dask workers to the number of chains."""
    d = Path(run_dir)
    head: List[str] = []
    files: List[str] = []
    if generate:
        progress("ESPEI parameter generation", 0.0)
        r = run_generate(str(d))
        n_sym = r["summary"]["n_symbols"]
        head.append(f"parameter generation: {n_sym} fitted symbols in {r['summary']['seconds']} s -> "
                    f"{display_dir or d}/dft.tdb")
        files += r["files"]
        progress(f"parameter generation done: {n_sym} symbols", 0.0, n_symbols=n_sym)
        if cap_cores:
            y = yaml.safe_load((d / "espei_mcmc.yaml").read_text())
            m = y["mcmc"]
            n_chains = int(m.get("chains_per_parameter") or 2) * (len(m.get("symbols") or []) or n_sym)
            if m.get("cores") and n_chains and int(m["cores"]) > n_chains:
                m["cores"] = n_chains
                if n_chains <= 1:
                    m["scheduler"] = None
                    m.pop("cores", None)
                (d / "espei_mcmc.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
                head.append(f"dask workers lowered to the {n_chains} chains")
    res = mcmc_job(str(d), display_dir, progress=progress)
    res["text"] = "\n".join(head + [res["text"]] + ([f"then: {next_hint}"] if next_hint and
                                                      (d / "mcmc.tdb").exists() else []))
    res["files"] = files + [f for f in res.get("files", []) if f not in files]
    return res


def _log_iteration(log: Path) -> Tuple[int, int]:
    """(iterations done, total) from ESPEI's '(i of N)' progress lines (last 64 kB of mcmc.log)."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            txt = f.read().decode(errors="replace")
    except OSError:
        return 0, 0
    hits = _ITER_RE.findall(txt)
    return (int(hits[-1][0]), int(hits[-1][1])) if hits else (0, 0)


def _lnprob_at(d: Path, it: int) -> Dict[str, Any]:
    p = d / "lnprob.npy"
    if not p.exists() or it < 1:
        return {}
    try:
        lp = np.load(p)
        col = lp[:, it - 1] if lp.ndim == 2 and lp.shape[1] >= it else None
        if col is None or not np.any(col):              # not saved yet (lnprob.npy is written every save_interval)
            return {}
        return _lnprob_stats(np.asarray(col))
    except Exception:  # noqa: BLE001
        return {}


def flux_hook(tb, jobdir: Path, spec: Dict[str, Any], st: Dict[str, Any], final: bool) -> Optional[Dict[str, Any]]:
    """jobs flux_hook for optimize_system on Flux: progress from the run directory while the batch jobs run
    (generation: running/done; MCMC: iteration i/N from mcmc.log, lnprob from lnprob.npy) and, at the end, the
    result text and files."""
    d = Path(spec["kwargs"]["run_dir"])
    per = st.get("flux") or {}
    gen = (per.get("generate") or {}).get("status")
    mc = (per.get("mcmc") or {}).get("status")
    it, n = _log_iteration(d / "mcmc.log")
    if mc in ("RUN", "CLEANUP") or (final and it):
        if it and n:
            st["frac"] = it / n
            stats = _lnprob_at(d, it)
            st["message"] = f"MCMC iteration {it}/{n} (mcmc.log)" + (
                f"  mean lnprob {stats.get('mean_lnprob')}  max {stats.get('max_lnprob')}" if stats else "")
            st.setdefault("metrics", {}).update({"iteration": it, "iterations": n, **stats})
            if it != (st.get("_last_iteration") or 0):
                st["_last_iteration"] = it
                rec = {"t": round(time.time(), 3), "msg": st["message"], "frac": st["frac"],
                       "metrics": {"iteration": it, **stats}}
                with open(jobdir / "progress.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                st["n_progress"] = int(st.get("n_progress", 0)) + 1
                st["last_progress"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        elif not final:
            st["message"] = f"MCMC job {(per.get('mcmc') or {}).get('id')} running: setting up ESPEI / dask"
    elif gen in ("RUN", "CLEANUP"):
        st["message"] = f"parameter generation running (job {(per.get('generate') or {}).get('id')})"
        st["frac"] = 0.0
    elif gen == "COMPLETED" and mc in ("DEPEND", "PRIORITY", "SCHED"):
        st["message"] = f"generation done (dft.tdb); MCMC job {(per.get('mcmc') or {}).get('id')} waiting: {mc}"
    if not final:
        return None
    names = ["dft.tdb", "generate_summary.json", "trace.npy", "lnprob.npy", "mcmc.tdb", "generate.log", "mcmc.log"]
    files = [str(d / x) for x in names if (d / x).exists()]
    if (d / "dft.tdb").exists() and not (d / "generate_summary.json").exists():
        try:                                          # the same per-phase summary the local tool writes
            pm = _load_json(Path(spec["kwargs"].get("phase_models") or "")) or {}
            summ = parameter_summary(str(d / "dft.tdb"), list(pm.get("phases", {})))
            (d / "generate_summary.json").write_text(json.dumps(summ, indent=1))
            files.insert(1, str(d / "generate_summary.json"))
        except Exception:  # noqa: BLE001
            pass
    try:
        png = plot_lnprob_progress(d, None)
        if png:
            files.append(png)
    except Exception:  # noqa: BLE001
        pass
    dd = tb.rel(str(d))
    state = st.get("state")
    L = [f"ESPEI on Flux {'finished' if state == 'done' else state}: "
         + ", ".join(f"{k} {v.get('id')} {v.get('status')}" for k, v in per.items())]
    if it:
        L.append(f"MCMC: {it}/{n} iterations (mcmc.log)")
    L.append(f"run dir {dd}: " + (", ".join(Path(f).name for f in files) or "no outputs"))
    if (d / "mcmc.tdb").exists():
        L.append(f"next: analyze_mcmc trace={dd}; " + _plot_hint(tb, d, spec["kwargs"].get("phase_models"),
                                                                  spec["kwargs"].get("datasets"))
                 + " (the optimised database against the data)")
    return {"text": "\n".join(L), "files": files, "iterations_done": it}


def _plot_hint(tb, d: Path, phase_models: Optional[str], datasets: Optional[str]) -> str:
    pm = _load_json(Path(phase_models)) if phase_models else None
    els = sorted(c for c in (pm or {}).get("components", []) if c.upper() != "VA")
    return (f"plot_diagram_with_data elements={json.dumps(els)} datasets_dir={tb.rel(str(datasets or ''))} "
            f"tdb={tb.rel(str(d / 'mcmc.tdb'))}")


def _submit_script(steps: List[Dict[str, Any]]) -> str:
    """submit.sh: the exact flux batch commands that were run (MCMC chained on the generation job's id)."""
    import shlex
    L = ["#!/bin/bash", "# The flux batch commands optimize_system ran (resources: #flux: lines in the scripts).",
         "set -euo pipefail", 'cd "$(dirname "$0")"']
    for s_ in steps:
        cmd = [("--dependency=afterok:$gen" if c.startswith("--dependency=afterok:") else c) for c in s_["command"]]
        line = " ".join(shlex.quote(c) if not c.startswith("--dependency=afterok:$") else f'"{c}"' for c in cmd)
        var = "gen" if s_["step"] == "generate" else "mcmc"
        L += [f"{var}=$({line})", f'echo "{s_["step"]}: ${var}"']
    return "\n".join(L) + "\n"


def submit_flux_run(tb, run: espei_flux.Run, d: Path, info: Dict[str, Any], *, nodes=None, cores_per_node=None,
                    time_limit=None, queue=None, bank=None, label: str = "", extra_kwargs=None) -> Dict[str, Any]:
    """Write the per-job Flux layout into ``d`` and submit it (generation -> MCMC with afterok).  Returns
    {job_id, steps, resources, files} or raises hpc.FluxSubmitError (already submitted steps are cancelled)."""
    from . import hpc, jobs
    cfg = hpc.config_of(tb)
    n_nodes, cpn, src = hpc.batch_resources(cfg, info, nodes, cores_per_node)
    fs = espei_flux.FluxSettings(nodes=n_nodes, cores_per_node=cpn, time_limit=time_limit or cfg.time_limit,
                                 gen_time_limit=cfg.gen_time_limit, queue=None, bank=None,
                                 setup=list(cfg.setup) if cfg.setup else espei_flux._default_setup(),
                                 mode="per-job", bank_flag=info.get("bank_flag") or "--bank")
    w = espei_flux.write_per_job_run(run, fs, str(d), base_dir=str(tb.workdir))
    if w["problems"]:
        raise ValueError("the generated ESPEI/Flux files have problems (nothing submitted): "
                         + json.dumps(w["problems"])[:1500])
    nested = bool(info.get("instance_level"))       # inside an allocation: no queues / banks there
    opts = hpc.batch_options(cfg, info.get("bank_flag") or "--bank", bank=bank, queue=queue, nested=nested)
    steps: List[Dict[str, Any]] = []
    try:
        dep = None
        for step in ("generate", "mcmc"):
            if step not in w["scripts"]:
                continue
            r = hpc.flux_batch(w["scripts"][step], str(d), opts, dependency=dep)
            steps.append({"step": step, "id": r["id"], "script": w["scripts"][step], "command": r["command"],
                          "output_prefix": w["output_prefix"][step]})
            dep = r["id"]
    except hpc.FluxSubmitError:
        if steps:
            hpc.flux_cancel([s_["id"] for s_ in steps])
        raise
    (d / "submit.sh").write_text(_submit_script(steps))
    resources = {"nodes": n_nodes, "cores_per_node": cpn, "cores_per_node_from": src,
                 "dask_workers": n_nodes * cpn, "time_limit": fs.time_limit,
                 "gen_time_limit": fs.gen_time_limit if run.generate_parameters else None,
                 "queue": queue if queue is not None else ("" if nested else cfg.queue),
                 "bank": bank if bank is not None else ("" if nested else cfg.bank),
                 "inside_allocation": nested,
                 "extra_flux_args": list(cfg.extra_flux_args or []), "setup": fs.setup}
    record = {"backend": "flux", "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": steps,
              "resources": resources, "scripts": [Path(f).name for f in w["files"]] + ["submit.sh"],
              "detected": {k: info.get(k) for k in ("reason", "version", "instance", "instance_level", "nodes",
                                                    "cores_per_node", "queues", "bank_flag")}}
    jid = jobs.submit_flux(tb, label or f"ESPEI optimise {d.name} (Flux)", steps, run_dir=str(d),
                           kwargs={"display_dir": tb.rel(str(d)), **(extra_kwargs or {})},
                           flux={k: v for k, v in record.items() if k != "steps"},
                           status_hook="calphad_tools.espei_run:mcmc_status_hook",
                           flux_hook="calphad_tools.espei_run:flux_hook")
    record["job_id"] = jid
    (d / SUBMISSION_FILE).write_text(json.dumps(record, indent=1, default=str))
    _record_provenance(tb, record)
    return {**record, "files": w["files"] + [str(d / "submit.sh"), str(d / SUBMISSION_FILE)]}


def _record_provenance(tb, record: Dict[str, Any]) -> None:
    prov = getattr(tb, "provenance", None)
    if prov is not None and hasattr(prov, "record_event"):
        try:
            prov.record_event("hpc_submission", record, tool="optimize_system")
        except Exception:  # noqa: BLE001 - provenance must never break a submission
            pass


# ---------------------------------------------------------------------------
# agent tools
# ---------------------------------------------------------------------------
def _n_fit_datasets(datasets: str) -> int:
    n = 0
    for f in Path(datasets).rglob("*.json"):
        ds = _load_json(f)
        n += bool(isinstance(ds, dict) and "solver" in ds)
    return n


def register(tb) -> None:  # noqa: C901 - flat list of tool definitions
    from . import jobs
    S = {"type": "string"}
    N = {"type": "number"}
    I = {"type": "integer"}
    B = {"type": "boolean"}
    STRS = {"type": "array", "items": {"type": "string"}}
    RUN = {"name": S, "phase_models": {**S, "description": "JSON (make_phase_models); default: from the run dir"},
           "datasets": {**S, "description": "directory; default: from the run dir"}}

    @tb.add("espei_generate_parameters",
            "ESPEI parameter generation (AICc model selection over single-phase thermochemical data) -> "
            "<out>/espei_runs/<name>/dft.tdb with VV#### symbols + fitted parameters per phase with values. Same "
            "YAML as generate_espei_flux_batch. Large inputs run as a background job.",
            {**RUN, "ref_state": {"type": "string", "enum": ["SGTE91", "SR2016"]},
             "excess_model": {"type": "string", "enum": ["linear"]},
             "aicc_penalty_factor": {"type": "object", "description": "{phase: {HM|SM|CPM: factor}}; >1 = fewer "
                                                                      "parameters"},
             "ridge_alpha": N,
             "fitting_description": {**S, "description": "gibbs (default), molar_volume, molar_volume_gibbs or an "
                                                         "import path"},
             "input_db": {**S, "description": "TDB with parameters to keep (e.g. unaries)"},
             "background": {**B, "description": "force job (true) or synchronous (false); default by size"}},
            ["name"])
    def espei_generate_parameters(name, phase_models=None, datasets=None, ref_state="SGTE91",
                                  excess_model="linear", aicc_penalty_factor=None, ridge_alpha=None,
                                  fitting_description=None, input_db=None, background=None):
        args = dict(name=name, phase_models=phase_models, datasets=datasets, ref_state=ref_state,
                    excess_model=excess_model, aicc_penalty_factor=aicc_penalty_factor, ridge_alpha=ridge_alpha,
                    fitting_description=fitting_description, input_db=input_db)
        d, settings, run = generate_settings(tb, args)
        chk = check_inputs(run.phase_models, run.datasets, settings)
        if not chk["ok"]:
            return _check_text(chk, tb.rel) + "\nnot run.", []
        d.mkdir(parents=True, exist_ok=True)
        (d / "espei_generate.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        n_fit = _n_fit_datasets(run.datasets)
        big = n_fit > SYNC_MAX_FIT_DATASETS or len(chk["phases"]) > SYNC_MAX_PHASES
        warn = ("\nwarnings:\n" + "\n".join(f"  - {w}" for w in chk["warnings"][:10])) if chk["warnings"] else ""
        flux = json.dumps(flux_entry(run, True, tb.rel))
        if background or (background is None and big):
            jid = jobs.submit(tb, f"ESPEI generate {d.name}", "calphad_tools.espei_run:generate_job",
                              {"run_dir": str(d), "display_dir": tb.rel(str(d))})
            return (f"background job {jid}: ESPEI parameter generation ({n_fit} single-phase datasets, "
                    f"{len(chk['phases'])} phases)\nrun dir {tb.rel(str(d))} (espei_generate.yaml)\n"
                    f"follow with job_status / wait_job job_id={jid}{warn}"), []
        r = run_generate(str(d), settings)
        return (f"wrote {tb.rel(r['tdb'])} in {r['summary']['seconds']} s from {r['summary']['n_datasets']} datasets"
                f" (YAML {tb.rel(str(d / 'espei_generate.yaml'))}, log generate.log)\n"
                + summary_text(r["summary"]) + warn
                + f"\nnext: espei_run_mcmc name={d.name} (local check), then generate_espei_flux_batch runs=[{flux}]"
                ), []

    @tb.add("espei_run_mcmc",
            "Local ESPEI MCMC as a background job (check before a Flux run; same YAML as generate_espei_flux_batch). "
            "Input: input_db or the run's dft.tdb. Writes trace.npy/lnprob.npy/mcmc.tdb to <out>/espei_runs/<name>/ "
            "(-> analyze_mcmc trace=<dir>); an existing trace is kept (new dir <name>_2) unless overwrite.",
            {**RUN, "input_db": S, "iterations": {**I, "description": "default 50"},
             "chains_per_parameter": {**I, "description": "even, default 2"},
             "chain_std_deviation": {**N, "description": "default 0.1"},
             "save_interval": I, "cores": {**I, "description": "dask workers; 1 = serial (default: [hpc] local_cores, "
                                                             "half of this machine's cores, at most 1 per chain)"},
             "restart_trace": {**S, "description": "trace.npy (or run dir) to continue from"},
             "deterministic": {**B, "description": "ESPEI's fixed seed (default true)"},
             "symbols": {**STRS, "description": "fit only these VV symbols"},
             "prior": {"type": "object", "description": "e.g. {\"name\": \"normal\", \"scale\": 0.1}"},
             "data_weights": {"type": "object", "description": "e.g. {\"ZPF\": 20, \"HM\": 1}"},
             "overwrite": B},
            ["name"])
    def espei_run_mcmc(name, phase_models=None, datasets=None, input_db=None, iterations=50,
                       chains_per_parameter=2, chain_std_deviation=0.1, save_interval=None, cores=None,
                       restart_trace=None, deterministic=True, symbols=None, prior=None, data_weights=None,
                       overwrite=False):
        args = dict(name=name, phase_models=phase_models, datasets=datasets, input_db=input_db,
                    iterations=iterations, chains_per_parameter=chains_per_parameter,
                    chain_std_deviation=chain_std_deviation, save_interval=save_interval, cores=cores,
                    restart_trace=restart_trace, deterministic=deterministic, symbols=symbols, prior=prior,
                    data_weights=data_weights, overwrite=overwrite)
        d, renamed = mcmc_run_dir(tb, args, _restart_path(tb, restart_trace))
        d, settings, run, info = mcmc_settings(tb, args, d)
        d.mkdir(parents=True, exist_ok=True)
        chk = check_inputs(run.phase_models, run.datasets, settings, run_dir=str(d))
        if not chk["ok"]:
            return _check_text(chk, tb.rel) + "\nnot started.", []
        n_sym = len(symbols) if symbols else chk["dataset_counts"].get("fitted symbols in input_db", 0)
        if not cores:                                  # default: half the cores, at most one per chain
            d, settings, run, info = mcmc_settings(tb, args, d, n_chains=_n_chains(run, n_sym))
        cores = info["cores"]
        (d / "espei_mcmc.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        jid = jobs.submit(tb, f"ESPEI MCMC {d.name}", "calphad_tools.espei_run:mcmc_job",
                          {"run_dir": str(d), "display_dir": tb.rel(str(d))},
                          status_hook="calphad_tools.espei_run:mcmc_status_hook")
        rd = tb.rel(str(d))
        L = [f"background job {jid}: ESPEI MCMC {run.iterations} iterations, "
             + (f"restart from {tb.rel(run.restart_trace)}" if run.restart_trace else
                f"{run.chains_per_parameter * n_sym} chains ({run.chains_per_parameter} x {n_sym} parameters)")
             + (f", dask {cores} cores" if cores and cores > 1 else ", serial") + f" ({info['cores_why']})",
             f"run dir {rd}" + (f" (new: {tb.rel(str(info['requested_dir']))} already has a trace)"
                                if renamed else "") + f"; input TDB {tb.rel(info['input_db'])}",
             f"outputs there: trace.npy, lnprob.npy (every {run.save_interval} iterations), mcmc.tdb, mcmc.log",
             f"follow: job_status job_id={jid} (lnprob plot); stop_job keeps the chains so far; when done: "
             f"analyze_mcmc trace={rd}",
             f"production: generate_espei_flux_batch runs=[{json.dumps(flux_entry(run, False, tb.rel))}]"]
        if chk["warnings"]:
            L.append("warnings:\n" + "\n".join(f"  - {w}" for w in chk["warnings"][:10]))
        return "\n".join(L), []

    @tb.add("optimize_system",
            "ESPEI optimisation (generation -> MCMC): Flux batch jobs if detected, else a local job. "
            "Inputs: prepared datasets (dir) + phase models. Flux: generation -> MCMC chained (afterok), [hpc] "
            "bank/queue/nodes/time_limit from calphad_agent.toml, 1 node with all its cores by default. Local: a "
            "background job with half the cores. Then wait_job until='done' and continue (analyze_mcmc, "
            "plot_diagram_with_data with the run's mcmc.tdb).",
            {**RUN, "input_db": {**S, "description": "generate=true: TDB whose parameters generation keeps (e.g. "
                                                     "unaries); generate=false: MCMC starting TDB (VV symbols)"},
             "generate": {**B, "description": "run parameter generation first (default true)"},
             "iterations": {**I, "description": "MCMC iterations (default 1000)"},
             "chains_per_parameter": {**I, "description": "even, default 2"}, "save_interval": I,
             "symbols": {**STRS, "description": "fit only these VV symbols"},
             "prior": {"type": "object"}, "data_weights": {"type": "object", "description": "e.g. {\"ZPF\": 20}"},
             "ref_state": {"type": "string", "enum": ["SGTE91", "SR2016"]},
             "aicc_penalty_factor": {"type": "object"},
             "backend": {"type": "string", "enum": ["auto", "flux", "local"],
                         "description": "auto (default): Flux if detected / [hpc] scheduler"},
             "cores": {**I, "description": "local: dask workers (default [hpc] local_cores = half the cores)"},
             "nodes": {**I, "description": "Flux: nodes for the MCMC job (default [hpc] nodes = 1)"},
             "cores_per_node": {**I, "description": "Flux: default from flux resource info"},
             "time_limit": {**S, "description": "Flux MCMC job, e.g. 8h (default [hpc] time_limit)"},
             "queue": S, "bank": S, "overwrite": B},
            ["name"])
    def optimize_system(name, phase_models=None, datasets=None, input_db=None, generate=True, iterations=1000,
                        chains_per_parameter=2, save_interval=None, symbols=None, prior=None, data_weights=None,
                        ref_state="SGTE91", aicc_penalty_factor=None, backend="auto", cores=None, nodes=None,
                        cores_per_node=None, time_limit=None, queue=None, bank=None, overwrite=False):
        from . import hpc
        generate = True if generate is None else bool(generate)
        nm = _safe_name(name)
        prev = _previous_system(runs_root(tb) / nm)
        pm = phase_models or prev.get("phase_models")
        ds = datasets or prev.get("datasets")
        if not pm or not ds:
            raise ValueError("give phase_models (JSON, make_phase_models) and datasets (directory of ESPEI datasets)")
        if not generate and not input_db:
            raise ValueError("generate=false needs input_db (the TDB with VV#### symbols to start the MCMC from)")
        d, renamed = mcmc_run_dir(tb, {"name": nm, "overwrite": overwrite})
        iters = int(iterations or 1000)
        run = espei_flux.Run(name=d.name, phase_models=tb.p(pm), datasets=tb.p(ds),
                             input_db=tb._find_tdb(input_db) if input_db else None, generate_parameters=generate,
                             ref_state=ref_state or "SGTE91", aicc_penalty_factor=aicc_penalty_factor or None,
                             iterations=iters, save_interval=int(save_interval or max(1, min(20, iters // 20))),
                             chains_per_parameter=int(chains_per_parameter or 2), prior=prior or None,
                             data_weights=data_weights or None, symbols=symbols or None, verbosity=2)
        base = Path(tb.workdir)
        gen_settings = espei_flux.espei_yaml(run, None, None, None, "generate", base) if generate else None
        mcmc_db = "dft.tdb" if generate else run.input_db
        probe = espei_flux.espei_yaml(run, mcmc_db, None, None, "mcmc", base)
        if generate:
            chk = check_inputs(run.phase_models, run.datasets, gen_settings)
            err = espei_flux.validate_espei_yaml(probe)
            if err:
                chk["problems"].append(f"MCMC YAML rejected by ESPEI's schema: {err[:400]}")
                chk["ok"] = False
        else:
            chk = check_inputs(run.phase_models, run.datasets, probe, run_dir=str(d))
        if not chk["ok"]:
            return _check_text(chk, tb.rel) + "\nnot started.", []
        n_sym = chk["dataset_counts"].get("fitted symbols in input_db", 0)
        warn = ("\nwarnings:\n" + "\n".join(f"  - {w}" for w in chk["warnings"][:10])) if chk["warnings"] else ""
        rd = tb.rel(str(d))
        where = f"run dir {rd}" + (f" (new: {nm} already has a trace)" if renamed else "")
        chains = (f"{run.chains_per_parameter} chains per fitted parameter" if generate else
                  f"{_n_chains(run, n_sym)} chains ({run.chains_per_parameter} x {len(symbols) if symbols else n_sym})")
        cfg = hpc.config_of(tb)
        kind, info = hpc.resolve_backend(cfg, backend)
        d.mkdir(parents=True, exist_ok=True)
        note = ""
        if kind == "flux":
            try:
                sub = submit_flux_run(tb, run, d, info, nodes=nodes, cores_per_node=cores_per_node,
                                      time_limit=time_limit, queue=queue, bank=bank,
                                      extra_kwargs={"phase_models": run.phase_models, "datasets": run.datasets})
            except hpc.FluxSubmitError as e:
                import shlex
                msg = (f"FLUX SUBMISSION FAILED - nothing is running.\n{e}\ncommand: {shlex.join(e.command)} "
                       f"(in {rd})\nhow to fix:\n{hpc.submit_hint(e.stderr)}")
                ans = ""
                if cfg.fallback_to_local:
                    note = msg + "\n[hpc] fallback_to_local = true: running it locally instead.\n"
                elif getattr(tb, "interactive", False) and getattr(tb, "ui", None) is not None:
                    ans = str(tb.ui.ask(msg + "\nRun the optimisation locally on this machine instead?",
                                        ["run locally", "don't run (fix the Flux settings first)"]) or "")
                    if ans.strip().lower().startswith("run locally"):
                        note = msg + "\nrunning it locally instead, as the user chose.\n"
                if not note:
                    return (msg + "\nNot run locally (no silent fallback): fix [hpc] bank / queue / time_limit in "
                            "calphad_agent.toml (or pass bank= / queue=) and call optimize_system again, or call it "
                            "with backend='local' if the user wants it run on this machine."), []
            else:
                res = sub["resources"]
                ids = [s_["id"] for s_ in sub["steps"]]
                L = [f"Flux job {sub['job_id']}: ESPEI optimisation of {d.name} submitted with flux batch ("
                     + " -> ".join(f"{s_['step']} {s_['id']}" for s_ in sub["steps"]) + ")",
                     f"backend: flux ({info.get('reason')})",
                     f"resources: MCMC {res['nodes']} node(s) x {res['cores_per_node']} cores "
                     f"({res['cores_per_node_from']}), {res['dask_workers']} dask workers, -t {res['time_limit']}"
                     + (f"; generation 1 core, -t {res['gen_time_limit']}" if generate else "")
                     + f"; queue {res['queue'] or '(site default)'}; bank {res['bank'] or '(none)'}"
                     + (" (inside a Flux allocation: its own resources, [hpc] queue/bank not used)"
                        if res.get("inside_allocation") else ""),
                     "commands: " + " ; ".join(" ".join(s_["command"]) for s_ in sub["steps"]),
                     f"{where} (espei_generate.yaml, espei_mcmc.yaml, gen.flux, mcmc.flux, submit.sh); "
                     f"MCMC {iters} iterations, {chains}",
                     f"wait: wait_job job_id={sub['job_id']} until='done', then analyze_mcmc trace={rd}; "
                     + _plot_hint(tb, d, run.phase_models, run.datasets),
                     f"from a shell: flux jobs {' '.join(ids)}"]
                return "\n".join(L) + warn, []
        # ------------------------------------------------ local (direct) run
        if gen_settings is not None:
            (d / "espei_generate.yaml").write_text(yaml.safe_dump(gen_settings, sort_keys=False))
        if cores:
            n_cores, why = int(cores), "cores argument"
        else:
            n_cores, why = default_cores(tb, None if generate else _n_chains(run, n_sym))
        mdb = mcmc_db if generate or Path(mcmc_db).resolve().parent != d.resolve() else Path(mcmc_db).name
        mset = espei_flux.espei_yaml(run, mdb, "dask" if n_cores > 1 else None, n_cores if n_cores > 1 else None,
                                     "mcmc", base)
        (d / "espei_mcmc.yaml").write_text(yaml.safe_dump(mset, sort_keys=False))
        jid = jobs.submit(tb, f"ESPEI optimise {d.name}", "calphad_tools.espei_run:optimize_job",
                          {"run_dir": str(d), "display_dir": rd, "generate": generate, "cap_cores": not cores,
                           "next_hint": _plot_hint(tb, d, run.phase_models, run.datasets)},
                          status_hook="calphad_tools.espei_run:mcmc_status_hook")
        from .hpc import available_cores
        record = {"backend": "local", "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": jid,
                  "resources": {"cores": n_cores, "cores_from": why, "available_cores": available_cores()},
                  "reason": info.get("reason"), "fallback_from_flux": bool(note)}
        (d / SUBMISSION_FILE).write_text(json.dumps(record, indent=1, default=str))
        _record_provenance(tb, record)
        L = [f"background job {jid}: ESPEI optimisation of {d.name} on this machine ("
             + ("parameter generation -> " if generate else "") + f"MCMC {iters} iterations, {chains})",
             f"backend: local ({info.get('reason')}); "
             + (f"dask {n_cores} cores" if n_cores > 1 else "serial") + f" ({why})",
             f"{where}" + (f"; input TDB {tb.rel(run.input_db)}" if run.input_db else ""),
             f"wait: wait_job job_id={jid} until='done', then analyze_mcmc trace={rd}; "
             + _plot_hint(tb, d, run.phase_models, run.datasets)]
        return note + "\n".join(L) + warn, []

    @tb.add("espei_validate_input",
            "Check ESPEI inputs before running: YAML against ESPEI's schema, phase_models, every dataset (ESPEI "
            "checker), datasets vs phase_models, input TDB symbols, known ESPEI 0.9.1 pitfalls (e.g. the "
            "equilibrium-property T x X ordering bug).",
            {"yaml": {**S, "description": "ESPEI input YAML (paths inside it are used)"},
             "phase_models": S, "datasets": S}, [])
    def espei_validate_input(yaml=None, phase_models=None, datasets=None):
        import yaml as _yaml
        settings, run_dir = None, None
        if yaml:
            yp = Path(tb.p(yaml))
            settings = _yaml.safe_load(yp.read_text())
            run_dir = str(yp.parent)
            sysd = settings.get("system") or {}

            def _r(p):
                return p if not p or os.path.isabs(p) else str(yp.parent / p)
            phase_models = phase_models or _r(sysd.get("phase_models"))
            datasets = datasets or _r(sysd.get("datasets"))
        if not phase_models or not datasets:
            raise ValueError("give yaml, or phase_models and datasets")
        chk = check_inputs(tb.p(phase_models), tb.p(datasets), settings, run_dir=run_dir)
        per = chk["fixed_config_datasets_per_phase"]
        extra = (f"\nphases: {', '.join(chk['phases'])}; single-phase datasets per phase: {json.dumps(per)}"
                 if chk["phases"] else "")
        return _check_text(chk, tb.rel) + extra, []
