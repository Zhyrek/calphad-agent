"""show_code generators for the ESPEI tools (espei_generate_parameters, espei_run_mcmc, optimize_system).

The standalone code writes the ESPEI input YAML inline (the same settings the tool used, i.e. the same
structure as the Flux batch YAMLs), points its input paths at WORKDIR / OUT, and runs
``espei.espei_script.run_espei`` in ``OUT / <run name>`` (ESPEI writes its outputs relative to the working
directory).  Imports: espei, yaml, numpy, matplotlib only.  An optimize_system run on Flux is reproduced by
writing the same YAMLs and batch scripts and resubmitting them with ``flux batch`` (MCMC chained afterok).
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .codegen import Ctx, generator


def _yaml_text(settings: Dict[str, Any]) -> str:
    return yaml.safe_dump(settings, sort_keys=False, default_flow_style=False).rstrip() + "\n"


def _rel_for_yaml(ctx: Ctx, p: str) -> str:
    """Path as shown inside the inline YAML (relative to WORKDIR where possible; overridden in code)."""
    return ctx._rel(Path(p))


def _run_block(settings_var: str, yaml_name: str) -> str:
    return (f"(run_dir / {yaml_name!r}).write_text(yaml.safe_dump({settings_var}, sort_keys=False))\n"
            "cwd = os.getcwd()\n"
            "os.chdir(run_dir)          # ESPEI writes output_db / trace / logfile relative to the working directory\n"
            "try:\n"
            f"    result = run_espei({settings_var})\n"
            "finally:\n"
            "    os.chdir(cwd)\n")


def _main_guard(body: str) -> str:
    """dask's LocalCluster starts worker processes that re-import a script: run only in the main process."""
    return ("# dask LocalCluster starts worker processes (which re-import this script): run under a __main__ guard\n"
            "if __name__ == '__main__':\n" + "".join("    " + ln + "\n" if ln.strip() else "\n"
                                                     for ln in body.rstrip("\n").split("\n")))


@generator("espei_generate_parameters")
def _gen_generate(args: Dict[str, Any], ctx: Ctx) -> str:
    from .espei_run import generate_settings
    d, settings, run = generate_settings(ctx.tb, args)
    return _generate_code(ctx, d, settings)


def _generate_code(ctx: Ctx, d: Path, settings: Dict[str, Any]) -> str:
    """Standalone parameter generation for the settings of run directory ``d`` (paths in it are absolute)."""
    ctx.need("import os", "import yaml", "from espei.espei_script import run_espei",
             "from espei.utils import database_symbols_to_fit, unpack_piecewise", "from symengine import Piecewise")
    sysd = settings["system"]
    shown = copy.deepcopy(settings)
    shown["system"]["phase_models"] = _rel_for_yaml(ctx, sysd["phase_models"])
    shown["system"]["datasets"] = _rel_for_yaml(ctx, sysd["datasets"])
    lines = [f"settings['system']['phase_models'] = str({ctx.path(sysd['phase_models'])})",
             f"settings['system']['datasets'] = str({ctx.path(sysd['datasets'])})"]
    idb = settings["generate_parameters"].get("input_db")
    if idb:
        shown["generate_parameters"]["input_db"] = _rel_for_yaml(ctx, idb)
        lines.append(f"settings['generate_parameters']['input_db'] = str({ctx.path(idb)})")
    out_db = settings["output"]["output_db"]
    ctx.produced[str((d / out_db).resolve())] = f"{d.name}/{out_db}"
    return (f"# ESPEI parameter generation (run {d.name!r}); same YAML structure as the Flux batch inputs\n"
            f"run_dir = OUT / {d.name!r}\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            f'ESPEI_YAML = """\\\n{_yaml_text(shown)}"""\n'
            "settings = yaml.safe_load(ESPEI_YAML)\n"
            "# input paths in the YAML are relative to WORKDIR; make them absolute (ESPEI runs inside run_dir)\n"
            + "\n".join(lines) + "\n"
            + _run_block("settings", "espei_generate.yaml")
            + "dbf = result\n\n"
            "# fitted parameters (VV#### symbols) and their values\n"
            "for name in database_symbols_to_fit(dbf):\n"
            "    print(name, float(unpack_piecewise(dbf.symbols[name])))\n"
            "for param in dbf._parameters.all():\n"
            "    expr = param['parameter']\n"
            "    if any(str(s).startswith('VV') for s in expr.free_symbols):\n"
            "        cons = ':'.join(','.join(c.name for c in subl) for subl in param['constituent_array'])\n"
            "        print(f\"{param['parameter_type']}({param['phase_name']},{cons};{param['parameter_order']}) =\","
            " expr.args[0] if isinstance(expr, Piecewise) else expr)\n")


def _mcmc_run_dir(ctx: Ctx) -> Optional[Path]:
    m = re.search(r"run dir (\S+)", ctx.record.get("result") or "")
    if not m:
        return None
    p = Path(ctx.tb.p(m.group(1)))
    return p if (p / "espei_mcmc.yaml").exists() else None


@generator("espei_run_mcmc")
def _gen_mcmc(args: Dict[str, Any], ctx: Ctx) -> str:
    from .espei_run import mcmc_settings
    d = _mcmc_run_dir(ctx)
    if d is not None:                         # the YAML the job actually ran (resolved run dir / input TDB)
        settings = yaml.safe_load((d / "espei_mcmc.yaml").read_text())
    else:
        d, settings, _run, _info = mcmc_settings(ctx.tb, args)
    return _mcmc_code(ctx, d, settings)


def _mcmc_code(ctx: Ctx, d: Path, settings: Dict[str, Any]) -> str:
    """Standalone local MCMC for the espei_mcmc.yaml settings of run directory ``d``."""
    ctx.need("import os", "import yaml", "import numpy as np", "import matplotlib.pyplot as plt",
             "from espei.espei_script import run_espei")
    sysd, m = settings["system"], settings["mcmc"]

    def absolute(p: str) -> str:
        return p if Path(p).is_absolute() else str(d / p)
    shown = copy.deepcopy(settings)
    shown["system"]["phase_models"] = _rel_for_yaml(ctx, sysd["phase_models"])
    shown["system"]["datasets"] = _rel_for_yaml(ctx, sysd["datasets"])
    lines = [f"settings['system']['phase_models'] = str({ctx.path(sysd['phase_models'])})",
             f"settings['system']['datasets'] = str({ctx.path(sysd['datasets'])})"]
    idb = absolute(m["input_db"])
    shown["mcmc"]["input_db"] = _rel_for_yaml(ctx, idb)
    lines.append(f"settings['mcmc']['input_db'] = str({ctx.path(idb)})")
    if m.get("restart_trace"):
        shown["mcmc"]["restart_trace"] = _rel_for_yaml(ctx, absolute(m["restart_trace"]))
        lines.append(f"settings['mcmc']['restart_trace'] = str({ctx.path(absolute(m['restart_trace']))})")
    dask = m.get("scheduler") == "dask"
    out = settings.get("output", {})
    for k, default in (("tracefile", "trace.npy"), ("probfile", "lnprob.npy"), ("output_db", "mcmc.tdb")):
        ctx.produced[str((d / out.get(k, default)).resolve())] = f"{d.name}/{out.get(k, default)}"
    ctx.note("ESPEI MCMC reruns the full sampling; deterministic=True reproduces the chains exactly (serial "
             "scheduler)")
    head = (f"# ESPEI MCMC (run {d.name!r}); same YAML structure as the Flux batch inputs"
            + (" (scheduler: dask local cluster)" if dask else " (serial)") + "\n"
            f"run_dir = OUT / {d.name!r}\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            f'ESPEI_YAML = """\\\n{_yaml_text(shown)}"""\n'
            "settings = yaml.safe_load(ESPEI_YAML)\n"
            "# input paths in the YAML are relative to WORKDIR; make them absolute (ESPEI runs inside run_dir)\n"
            + "\n".join(lines) + "\n")
    body = ("# ESPEI refuses to overwrite an existing trace: clear the outputs of a previous run of this cell\n"
            f"for name in ({out.get('tracefile', 'trace.npy')!r}, {out.get('probfile', 'lnprob.npy')!r}):\n"
            "    (run_dir / name).unlink(missing_ok=True)\n"
            + _run_block("settings", "espei_mcmc.yaml")
            + "\n# lnprob of every chain vs iteration\n"
            f"lnprob = np.load(run_dir / {out.get('probfile', 'lnprob.npy')!r})\n"
            "fig, ax = plt.subplots(figsize=(6.4, 3.8))\n"
            "ax.plot(np.arange(1, lnprob.shape[1] + 1), lnprob.T, color='0.6', lw=0.5, alpha=0.5)\n"
            "ax.plot(np.arange(1, lnprob.shape[1] + 1), lnprob.mean(axis=0), color='C0', lw=1.6, label='mean')\n"
            "ax.set_xlabel('iteration')\nax.set_ylabel('lnprob')\n"
            f"ax.set_title('ESPEI MCMC {d.name}')\nax.legend(fontsize=8)\n"
            "fig.savefig(run_dir / 'lnprob_progress.png', dpi=110, bbox_inches='tight')\n")
    return head + (_main_guard(body) if dask else body)


# ---------------------------------------------------------------------------------------------------------
# optimize_system: generation -> MCMC, run locally (a background job) or as Flux batch jobs
# ---------------------------------------------------------------------------------------------------------
FLUX_POLL_S = 30


@generator("optimize_system")
def _gen_optimize(args: Dict[str, Any], ctx: Ctx):
    import json
    d = _mcmc_run_dir(ctx)
    if d is None or not (d / "hpc_submission.json").exists():
        ctx.fallback("the run directory of this optimize_system call (hpc_submission.json) is gone")
    sub = json.loads((d / "hpc_submission.json").read_text())
    if sub.get("backend") == "flux":
        return _flux_code(ctx, d, sub)
    cells = []
    if (d / "espei_generate.yaml").exists() and (d / "dft.tdb").exists():
        cells.append(("parameter generation", _generate_code(ctx, d, yaml.safe_load(
            (d / "espei_generate.yaml").read_text()))))
    cells.append(("MCMC", _mcmc_code(ctx, d, yaml.safe_load((d / "espei_mcmc.yaml").read_text()))))
    return cells


def _text_literal(text: str) -> str:
    """A readable Python literal for a script's text (raw triple-quoted when possible)."""
    if "'''" not in text and not text.endswith("\\"):
        return "r'''" + text + "'''"
    return repr(text)


def _json_short(obj: Any, n: int = 160) -> str:
    import json
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n - 3] + "..."


def _flux_code(ctx: Ctx, d: Path, sub: Dict[str, Any]) -> str:
    """Standalone reproduction of a Flux run: the ESPEI YAMLs and the batch scripts exactly as submitted, the
    same flux batch commands (MCMC chained on the generation job), waiting with flux jobs, then the lnprob plot."""
    ctx.need("import subprocess", "import time", "import yaml", "import numpy as np", "import matplotlib.pyplot as plt")
    steps = sub.get("steps") or []
    code = [f"# ESPEI optimisation of {d.name!r} as Flux batch jobs, exactly as optimize_system submitted it",
            f"# resources: {_json_short(sub.get('resources'), 300)}",
            f"run_dir = OUT / {d.name!r}", "run_dir.mkdir(parents=True, exist_ok=True)", ""]
    for step in steps:
        yname = "espei_generate.yaml" if step["step"] == "generate" else "espei_mcmc.yaml"
        settings = yaml.safe_load((d / yname).read_text())
        var = "gen_settings" if step["step"] == "generate" else "mcmc_settings"
        shown = copy.deepcopy(settings)
        sysd = settings["system"]
        shown["system"]["phase_models"] = _rel_for_yaml(ctx, sysd["phase_models"])
        shown["system"]["datasets"] = _rel_for_yaml(ctx, sysd["datasets"])
        fix = [f"{var}['system']['phase_models'] = str({ctx.path(sysd['phase_models'])})",
               f"{var}['system']['datasets'] = str({ctx.path(sysd['datasets'])})"]
        gp = settings.get("generate_parameters") or {}
        if gp.get("input_db"):
            shown["generate_parameters"]["input_db"] = _rel_for_yaml(ctx, gp["input_db"])
            fix.append(f"{var}['generate_parameters']['input_db'] = str({ctx.path(gp['input_db'])})")
        m = settings.get("mcmc") or {}
        if m:
            shown["mcmc"]["scheduler"] = "scheduler.json"
            fix.append(f"{var}['mcmc']['scheduler'] = str(run_dir / 'scheduler.json')   # the batch job's dask "
                       "scheduler file")
            if m.get("input_db") and Path(m["input_db"]).is_absolute():
                shown["mcmc"]["input_db"] = _rel_for_yaml(ctx, m["input_db"])
                fix.append(f"{var}['mcmc']['input_db'] = str({ctx.path(m['input_db'])})")
            out = settings.get("output", {})
            for k, default in (("tracefile", "trace.npy"), ("probfile", "lnprob.npy"), ("output_db", "mcmc.tdb")):
                ctx.produced[str((d / out.get(k, default)).resolve())] = f"{d.name}/{out.get(k, default)}"
        else:
            ctx.produced[str((d / "dft.tdb").resolve())] = f"{d.name}/dft.tdb"
        script = (d / step["script"]).read_text()
        code += [f"# {step['step']}: {yname} and the batch script {step['script']} (#flux: lines = resources)",
                 f'{var} = yaml.safe_load("""\\\n{_yaml_text(shown)}""")', *fix,
                 f"(run_dir / {yname!r}).write_text(yaml.safe_dump({var}, sort_keys=False))",
                 f"(run_dir / {step['script']!r}).write_text({_text_literal(script)})", ""]
    code += ["def flux(*args):", "    \"\"\"Run a flux command in run_dir and return its output.\"\"\"",
             "    r = subprocess.run(['flux', *args], cwd=run_dir, capture_output=True, text=True)",
             "    if r.returncode:",
             "        raise RuntimeError(f\"flux {' '.join(args)}: {r.stderr.strip()}\")",
             "    return r.stdout.strip()", "",
             "# the same flux batch commands as the original run (queue / bank / extra options); the MCMC job starts",
             "# only after the generation job succeeded (afterok)"]
    ids = []
    for step in steps:
        args = [c for c in step["command"][2:] if not c.startswith("--dependency=")]
        script = args.pop()
        dep = "f'--dependency=afterok:{gen_id}', " if any(c.startswith("--dependency=") for c in step["command"]) \
            else ""
        var = "gen_id" if step["step"] == "generate" else "mcmc_id"
        ids.append(var)
        code.append(f"{var} = flux('batch', " + "".join(f"{a!r}, " for a in args) + dep + f"{script!r})")
    code += [f"print('submitted:', {', '.join(ids)})", "",
             "# wait until the jobs have finished (from a shell: flux jobs)",
             "FINAL = ('COMPLETED', 'FAILED', 'CANCELED', 'TIMEOUT')", "while True:",
             f"    states = flux('jobs', '-n', '-o', '{{status}}', {', '.join(ids)}).split()",
             "    if states and all(s in FINAL for s in states):", "        break",
             f"    time.sleep({FLUX_POLL_S})", "print('flux job states:', states)", "",
             "# lnprob of every chain vs iteration (ESPEI zero-fills iterations that were not sampled)",
             "lnprob = np.load(run_dir / 'lnprob.npy')",
             "lnprob = lnprob[:, :int(np.count_nonzero(lnprob.any(axis=0)))]",
             "fig, ax = plt.subplots(figsize=(6.4, 3.8))",
             "ax.plot(np.arange(1, lnprob.shape[1] + 1), lnprob.T, color='0.6', lw=0.5, alpha=0.5)",
             "ax.plot(np.arange(1, lnprob.shape[1] + 1), lnprob.mean(axis=0), color='C0', lw=1.6, label='mean')",
             "ax.set_xlabel('iteration')", "ax.set_ylabel('lnprob')", f"ax.set_title('ESPEI MCMC {d.name}')",
             "ax.legend(fontsize=8)", "fig.savefig(run_dir / 'lnprob_progress.png', dpi=110, bbox_inches='tight')"]
    ctx.note("Flux run: the code writes the same YAMLs and batch scripts, resubmits them with flux batch and waits")
    return "\n".join(code) + "\n"
