"""show_code generators for the local ESPEI tools (espei_generate_parameters, espei_run_mcmc).

The standalone code writes the ESPEI input YAML inline (the same settings the tool used, i.e. the same
structure as the Flux batch YAMLs), points its input paths at WORKDIR / OUT, and runs
``espei.espei_script.run_espei`` in ``OUT / <run name>`` (ESPEI writes its outputs relative to the working
directory).  Imports: espei, yaml, numpy, matplotlib only.
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
    ctx.need("import os", "import yaml", "from espei.espei_script import run_espei",
             "from espei.utils import database_symbols_to_fit, unpack_piecewise", "from symengine import Piecewise")
    shown = copy.deepcopy(settings)
    shown["system"]["phase_models"] = _rel_for_yaml(ctx, run.phase_models)
    shown["system"]["datasets"] = _rel_for_yaml(ctx, run.datasets)
    lines = [f"settings['system']['phase_models'] = str({ctx.path(run.phase_models)})",
             f"settings['system']['datasets'] = str({ctx.path(run.datasets)})"]
    if settings["generate_parameters"].get("input_db"):
        shown["generate_parameters"]["input_db"] = _rel_for_yaml(ctx, run.input_db)
        lines.append(f"settings['generate_parameters']['input_db'] = str({ctx.path(run.input_db)})")
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
