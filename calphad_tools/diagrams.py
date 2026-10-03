"""Features 2 & 3: phase diagrams for every binary / ternary subsystem of a TDB.

Each subsystem is computed in its own process (pycalphad mapping can be slow
or occasionally hang), with a timeout.  Individual PNGs are then stitched into
a grid image (one image per slide) and written to a PowerPoint deck.

CLI:
  calphad-tools binaries  my.tdb -o binaries.pptx --tmin 100 --tmax 3000
  calphad-tools ternaries my.tdb -o ternaries.pptx --temps 800 1000 1200
"""
from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
import time
import traceback
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import pcache  # noqa: E402
from . import pptx_utils as pu  # noqa: E402

KELVIN = 273.15
TIELINE_COLOR = (0.55, 0.55, 0.55, 0.35)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_db(tdb: str):
    """Shared, cached Database (read-only; see pcache)."""
    return pcache.get_db(tdb)


def db_elements(db) -> List[str]:
    return sorted(str(e) for e in db.elements if str(e) not in ("VA", "/-"))


def phases_for(db, elements: Sequence[str], include=None, exclude=None) -> List[str]:
    """Phases that can form in the given subsystem (pycalphad filter)."""
    from pycalphad.core.utils import filter_phases, unpack_species
    comps = [e.upper() for e in elements] + ["VA"]
    ph = filter_phases(db, unpack_species(db, comps))
    if include:
        ph = [p for p in ph if p in {i.upper() for i in include}]
    if exclude:
        ph = [p for p in ph if p not in {e.upper() for e in exclude}]
    return sorted(ph)


def has_interaction(db, elements: Sequence[str]) -> bool:
    """True if the database has any parameter mixing *all* of ``elements``."""
    want = {e.upper() for e in elements}
    for p in db._parameters.all():
        sp = {str(s.name if hasattr(s, "name") else s) for sl in p["constituent_array"] for s in sl}
        if want <= sp:
            return True
    return False


def to_celsius_axis(ax):
    """Relabel a Kelvin y-axis in °C with ticks on round Celsius values."""
    from matplotlib.ticker import FixedLocator, MaxNLocator
    from .figures import ScaledFormatter
    lo, hi = ax.get_ylim()
    ticks_c = [t for t in MaxNLocator(8).tick_values(lo - KELVIN, hi - KELVIN)
               if lo - 1e-6 <= t + KELVIN <= hi + 1e-6]
    ax.yaxis.set_major_locator(FixedLocator([t + KELVIN for t in ticks_c]))
    ax.yaxis.set_major_formatter(ScaledFormatter(offset=-KELVIN, fmt="{:.0f}"))   # picklable, records K->°C
    ax.set_ylabel("Temperature (°C)")


def _label_el(e: str) -> str:
    return e.capitalize()


# ---------------------------------------------------------------------------
# Single-diagram calculators (usable directly, or from worker processes)
# ---------------------------------------------------------------------------
@pcache.cached
def plot_binary(tdb: str, elements: Sequence[str], out_png: str, tmin_c: float = 100.0,
                tmax_c: float = 3000.0, x_step: float = 0.02, t_step: float = 10.0,
                phases: Optional[Sequence[str]] = None, exclude_phases=None,
                pressure: float = 101325.0, tielines: int | bool = 4, dpi: int = 150,
                title: Optional[str] = None, units: str = "C", keep_order: bool = False,
                check_mapping: str = "auto") -> Dict:
    """Binary T-x phase diagram.  Temperatures in °C unless units='K'.  x axis = second element: alphabetical, or
    the given order with keep_order ([left, right])."""
    from pycalphad import variables as v
    from pycalphad.mapping import BinaryStrategy, plot_binary as pc_plot_binary
    from .mapping_check import grid_size, map_and_check

    db = load_db(tdb)
    a, b = [e.upper() for e in elements] if keep_order else sorted(e.upper() for e in elements)
    ph = list(phases) if phases else phases_for(db, [a, b], exclude=exclude_phases)
    tlo, thi = (tmin_c + KELVIN, tmax_c + KELVIN) if units.upper() == "C" else (tmin_c, tmax_c)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    t0 = time.time()
    recipe = {"kind": "binary", "elements": [a, b], "phases": ph, "P": float(pressure), "T_range": [tlo, thi],
              "x_range": [0.0, 1.0], "tmap": [1.0, 0.0], "T_unit": "C" if units.upper() == "C" else "K",
              "comp": {"scale": 1.0, "weight": False}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # binplot, with the mapping check (mapping_check) between map and plot
        strategy = BinaryStrategy(db, [a, b, "VA"], ph,
                                  {v.X(b): (0, 1, x_step), v.T: (tlo, thi, t_step), v.P: pressure, v.N: 1})
        check = map_and_check(strategy, db, recipe, check_mapping, grid_n=grid_size("binary", labels=False))
        pc_plot_binary(strategy, ax=ax, tielines=tielines, tieline_color=TIELINE_COLOR)
        ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(tlo, thi)
    if units.upper() == "C":
        to_celsius_axis(ax)
    ax.set_xlabel(f"Mole fraction {_label_el(b)}")
    note = "" if has_interaction(db, [a, b]) else "  (no binary parameters in TDB)"
    ax.set_title(title or f"{_label_el(a)}–{_label_el(b)}{note}", fontsize=12)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"png": out_png, "phases": ph, "seconds": round(time.time() - t0, 1), "note": _check_note(check)}


@pcache.cached
def plot_ternary(tdb: str, elements: Sequence[str], temperature: float, out_png: str,
                 units: str = "C", step: float = 0.02, phases: Optional[Sequence[str]] = None,
                 exclude_phases=None, pressure: float = 101325.0, tielines: int | bool = 4,
                 dpi: int = 150, title: Optional[str] = None, keep_order: bool = False,
                 check_mapping: str = "auto") -> Dict:
    """Ternary isothermal section.  Corners bottom-left, bottom-right (x axis), top (y axis) = the elements in
    alphabetical order, or in the given order with keep_order."""
    from pycalphad import variables as v
    from pycalphad.mapping import TernaryStrategy, plot_ternary as pc_plot_ternary
    from .mapping_check import grid_size, map_and_check
    from .orientation import corner_labels

    db = load_db(tdb)
    a, b, c = [e.upper() for e in elements] if keep_order else sorted(e.upper() for e in elements)
    ph = list(phases) if phases else phases_for(db, [a, b, c], exclude=exclude_phases)
    T = temperature + KELVIN if units.upper() == "C" else temperature
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(projection="triangular")
    t0 = time.time()
    recipe = {"kind": "ternary", "elements": [a, b, c], "phases": ph, "P": float(pressure), "T_K": float(T),
              "T_unit": "C" if units.upper() == "C" else "K", "comp": {"scale": 1.0, "weight": False}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # ternplot, with the mapping check (mapping_check) between map and plot
        strategy = TernaryStrategy(db, [a, b, c, "VA"], ph,
                                   {v.X(b): (0, 1, step), v.X(c): (0, 1, step), v.T: T, v.P: pressure, v.N: 1})
        check = map_and_check(strategy, db, recipe, check_mapping, grid_n=grid_size("ternary", labels=False))
        pc_plot_ternary(strategy, x=v.X(b), y=v.X(c), ax=ax, tielines=tielines, tieline_color=TIELINE_COLOR)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    corner_labels(ax, [a, b, c])
    tc = T - KELVIN
    ax.set_title(title or f"{_label_el(a)}–{_label_el(b)}–{_label_el(c)}   {tc:.0f} °C ({T:.0f} K)",
                 fontsize=12)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"png": out_png, "phases": ph, "seconds": round(time.time() - t0, 1), "note": _check_note(check)}


# ---------------------------------------------------------------------------
# Parallel execution with per-task timeout
# ---------------------------------------------------------------------------
@dataclass
class TaskResult:
    name: str
    ok: bool
    png: Optional[str] = None
    error: str = ""
    seconds: float = 0.0
    note: str = ""              # e.g. the mapping check's finding (mapping_check)


def map_diagram(**kwargs) -> Dict:
    """mapping_tools.map_phase_diagram in a worker (PNG + boundary CSV/JSON next to it)."""
    from .mapping_tools import map_phase_diagram
    r = map_phase_diagram(**kwargs)
    return {"png": kwargs["out_png"], "files": r.get("files", {}), "note": _check_note(r.get("mapping_check"))}


def _check_note(check: Optional[Dict]) -> str:
    """The mapping check's line for an overview, empty when it found nothing (or did not run)."""
    if not check or not check.get("text") or check.get("mode") == "off":
        return ""
    t = check["text"]
    return "" if "no problems found" in t else t.replace("mapping check: ", "")


_WORKERS: Dict[str, Callable] = {"binary": plot_binary, "ternary": plot_ternary, "map": map_diagram}


def _run_one(kind: str, kwargs: Dict) -> Dict:
    try:
        return {"ok": True, **_WORKERS[kind](**kwargs)}
    except Exception as e:  # noqa: BLE001 - report everything back to the parent
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}


def _worker_loop(conn):
    """Persistent worker: runs tasks until told to stop, keeping its pcache (parsed database, models,
    compiled code) warm between tasks - e.g. the same ternary at several temperatures compiles once."""
    from .figures import install_from_env
    install_from_env()          # keep editable copies of worker-made figures too
    try:                        # the slow imports happen before "ready", so they don't count against a task's timeout
        import pycalphad  # noqa: F401
        import pycalphad.mapping  # noqa: F401
        conn.send({"ready": True})
    except (BrokenPipeError, OSError):
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        kind, kwargs = msg
        res = _run_one(kind, kwargs)
        try:
            conn.send(json.loads(json.dumps(res, default=str)))
        except (BrokenPipeError, OSError):
            return


def _affinity(t: Dict) -> tuple:
    kw = t["kwargs"]
    return (t["kind"], str(kw.get("tdb")), tuple(sorted(str(e).upper() for e in kw.get("elements", ()))))


def run_tasks(tasks: List[Dict], n_workers: Optional[int] = None, timeout: float = 900.0,
              progress: Optional[Callable[[str], None]] = None) -> List[TaskResult]:
    """Run [{'name','kind','kwargs'}] in ``n_workers`` worker processes with a per-task timeout.

    Workers are reused for several tasks (a timed-out or crashed worker is replaced), and a task for a
    subsystem a worker has already computed is preferably given to that worker, so its cache is reused."""
    n_workers = n_workers or max(1, min(len(tasks), (os.cpu_count() or 2) - 1))
    ctx = mp.get_context("spawn")
    pending = list(tasks)
    workers: List[Dict] = []          # {"proc", "conn", "task", "t0", "seen"}
    results: Dict[str, TaskResult] = {}
    say = progress or (lambda s: None)

    def spawn() -> Dict:
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker_loop, args=(child,), daemon=True)
        proc.start()
        child.close()
        return {"proc": proc, "conn": parent, "task": None, "t0": 0.0, "seen": set(), "ready": False,
                "born": time.time()}

    def retire(w: Dict, kill: bool) -> None:
        if kill:
            w["proc"].terminate()
        w["proc"].join(5)
        w["conn"].close()
        workers.remove(w)

    try:
        while pending or any(w["task"] for w in workers):
            idle = [w for w in workers if w["task"] is None]
            while pending and (idle or len(workers) < n_workers):
                w = idle.pop(0) if idle else spawn()
                if w not in workers:
                    workers.append(w)
                i = next((k for k, t in enumerate(pending) if _affinity(t) in w["seen"]), 0)
                t = pending.pop(i)
                try:
                    w["conn"].send((t["kind"], t["kwargs"]))
                except (BrokenPipeError, OSError):
                    pending.insert(0, t)
                    retire(w, kill=True)
                    continue
                w["task"] = t
                if w["ready"]:
                    w["t0"] = time.time()
                w["seen"].add(_affinity(t))
                say(f"started {t['name']}")
            time.sleep(0.1)
            for w in list(workers):
                t = w["task"]
                if t is None:
                    continue
                if not w["ready"]:                 # still starting up: the task's clock has not started
                    try:
                        if w["conn"].poll():
                            w["conn"].recv()
                            w["ready"], w["t0"] = True, time.time()
                    except (EOFError, OSError):
                        pass
                    if not w["ready"]:
                        if w["proc"].is_alive() and time.time() - w["born"] <= max(120.0, timeout):
                            continue
                        results[t["name"]] = TaskResult(t["name"], False, None, "worker failed to start",
                                                        round(time.time() - w["born"], 1))
                        w["task"] = None
                        retire(w, kill=True)
                        continue
                elapsed = time.time() - w["t0"]
                r = None
                try:
                    if w["conn"].poll():
                        r = w["conn"].recv()
                except (EOFError, OSError):
                    r = {"ok": False, "error": f"worker exited with code {w['proc'].exitcode}"}
                if r is None and not w["proc"].is_alive():
                    r = {"ok": False, "error": f"worker exited with code {w['proc'].exitcode}"}
                if r is not None:
                    results[t["name"]] = TaskResult(t["name"], r.get("ok", False), r.get("png"),
                                                    r.get("error", ""), round(elapsed, 1), r.get("note", "") or "")
                    w["task"] = None
                    say(f"{'finished' if r.get('ok') else 'FAILED'} {t['name']} ({elapsed:.0f}s)"
                        + ("" if r.get("ok") else f": {r.get('error')}"))
                    if not w["proc"].is_alive():
                        retire(w, kill=False)
                elif elapsed > timeout:
                    results[t["name"]] = TaskResult(t["name"], False, None, f"timed out after {timeout:.0f}s",
                                                    round(elapsed, 1))
                    retire(w, kill=True)
                    say(f"TIMEOUT {t['name']}")
    finally:
        for w in list(workers):
            try:
                w["conn"].send(None)
            except (BrokenPipeError, OSError):
                pass
        for w in list(workers):
            w["proc"].join(5)
            if w["proc"].is_alive():
                w["proc"].terminate()
                w["proc"].join(2)
            w["conn"].close()
    return [results[t["name"]] for t in tasks]


# ---------------------------------------------------------------------------
# Grid composition
# ---------------------------------------------------------------------------
def make_grid(results: List[TaskResult], out_png: str, ncols: Optional[int] = None,
              suptitle: Optional[str] = None, dpi: int = 200) -> str:
    """Stitch individual diagram PNGs into one image (matplotlib subplots)."""
    from PIL import Image

    n = len(results)
    ncols = ncols or (1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4)
    nrows = math.ceil(n / ncols)
    aspect = 0.8
    for r in results:
        if r.ok and r.png and os.path.exists(r.png):
            with Image.open(r.png) as im:
                aspect = im.size[1] / im.size[0]
            break
    cell_w = 16.0 / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, cell_w * aspect * nrows + (0.5 if suptitle else 0)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, results):
        if r.ok and r.png and os.path.exists(r.png):
            ax.imshow(plt.imread(r.png))
        else:
            ax.text(0.5, 0.5, f"{r.name}\n\ncalculation failed:\n{r.error[:120]}", ha="center",
                    va="center", fontsize=9, wrap=True, transform=ax.transAxes, color="#9b1c1c")
    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


# ---------------------------------------------------------------------------
# Top-level feature functions
# ---------------------------------------------------------------------------
def _shaded_binary_task(tdb, a, b, out_png, tmin_c, tmax_c, x_step, t_step, exclude_phases, title, formats,
                        timeout, check_mapping="auto") -> Dict:
    """map_phase_diagram (worker "map") arguments of one shaded overview binary (°C axis, phase-region labels; see
    diagram_style)."""
    return dict(tdb=str(tdb), elements=[a, b], right=b, out_png=out_png, tmin=tmin_c, tmax=tmax_c,
                temperature_unit="C", composition_step=x_step, temperature_step=t_step, exclude_phases=exclude_phases,
                tielines=0, style="shaded", title=title, formats=formats, time_limit_s=0.9 * timeout,
                check_mapping=check_mapping)


def all_binaries(tdb: str, output_pptx: str = "binaries.pptx", tmin_c: float = 100.0,
                 tmax_c: float = 3000.0, out_dir: Optional[str] = None, n_workers: Optional[int] = None,
                 timeout: float = 900.0, per_slide: int = 6, x_step: float = 0.02, t_step: float = 10.0,
                 individual_slides: bool = False, exclude_phases=None, elements=None,
                 progress: Optional[Callable[[str], None]] = None, element_order=None, orientation=None,
                 style: str = "shaded", formats: Optional[Sequence[str]] = None,
                 check_mapping: str = "auto") -> Dict:
    """element_order: earlier elements on the left of every binary; orientation: per-system overrides
    {"CR-FE": ["FE", "CR"] | "CR" (right element)} (see orientation.subsystem_order); default alphabetical.
    style: "shaded" (map_phase_diagram in the shaded style with phase-region labels, its CSV/JSON data and SVG+PDF
    next to each PNG unless formats says otherwise) or "color" (pycalphad binplot, one colour per phase)."""
    from .diagram_style import check_style
    from .orientation import check_overrides, subsystem_order
    style = check_style(style)
    db = load_db(tdb)
    els = [e.upper() for e in (elements or db_elements(db))]
    pairs = list(itertools.combinations(sorted(els), 2))
    check_overrides(orientation, pairs)
    pairs = [tuple(subsystem_order(p, element_order, orientation)) for p in pairs]
    stem = Path(tdb).stem
    out_dir = Path(out_dir or Path(output_pptx).with_suffix("").as_posix() + "_png")
    out_dir.mkdir(parents=True, exist_ok=True)
    if style == "shaded":
        tasks = [{"name": f"{a}-{b}", "kind": "map",
                  "kwargs": _shaded_binary_task(tdb, a, b, str(out_dir / f"{stem}_{a}-{b}.png"), tmin_c, tmax_c,
                                                x_step, t_step, exclude_phases,
                                                f"{_label_el(a)}–{_label_el(b)}" + ("" if has_interaction(db, [a, b])
                                                                                    else "  (no binary parameters in TDB)"),
                                                formats, timeout, check_mapping)} for a, b in pairs]
    else:
        tasks = [{"name": f"{a}-{b}", "kind": "binary",
                  "kwargs": dict(tdb=str(tdb), elements=[a, b], out_png=str(out_dir / f"{stem}_{a}-{b}.png"),
                                 tmin_c=tmin_c, tmax_c=tmax_c, x_step=x_step, t_step=t_step,
                                 exclude_phases=exclude_phases, keep_order=True, check_mapping=check_mapping)}
                 for a, b in pairs]
    results = run_tasks(tasks, n_workers=n_workers, timeout=timeout, progress=progress)
    prs = pu.new_presentation()
    pu.add_title_slide(prs, f"Binary phase diagrams: {Path(tdb).name}",
                       f"{len(pairs)} binaries of {'-'.join(e.capitalize() for e in els)}   •   "
                       f"{tmin_c:g}–{tmax_c:g} °C, 1 atm",
                       note="Calculated with pycalphad mapping (BinaryStrategy).")
    grids = []
    for i, chunk in enumerate(_chunks(results, per_slide)):
        g = make_grid(chunk, str(out_dir / f"{stem}_binaries_grid{i + 1}.png"))
        grids.append(g)
        pu.add_image_slide(prs, "Binary phase diagrams" + (f" ({i + 1})" if len(results) > per_slide else ""),
                           g, subtitle=f"{Path(tdb).name}   •   {tmin_c:g}–{tmax_c:g} °C")
    if individual_slides:
        for r in results:
            if r.ok:
                pu.add_image_slide(prs, r.name.replace("-", "–").title(), r.png, subtitle=Path(tdb).name)
    pu.save(prs, output_pptx)
    return {"pptx": output_pptx, "grid_images": grids, "systems": [asdict(r) for r in results]}


def all_ternaries(tdb: str, temperatures_c: Sequence[float], output_pptx: str = "ternaries.pptx",
                  out_dir: Optional[str] = None, n_workers: Optional[int] = None, timeout: float = 1800.0,
                  per_slide: int = 6, step: float = 0.02, individual_slides: bool = False,
                  exclude_phases=None, elements=None, units: str = "C",
                  progress: Optional[Callable[[str], None]] = None, element_order=None, orientation=None,
                  style: str = "shaded", formats: Optional[Sequence[str]] = None,
                  check_mapping: str = "auto") -> Dict:
    """element_order: earlier elements go bottom-left, then bottom-right, then top; orientation: per-system
    overrides {"CR-FE-NI": {"left": .., "right": .., "top": ..} | [bottom-left, bottom-right, top]}.
    style: "shaded" (map_phase_diagram: grey boundaries and tie-triangles, no tie-lines, phase-region labels
    instead of a legend; SVG+PDF next to each PNG unless formats says otherwise) or "color" (pycalphad ternplot,
    one colour per phase)."""
    from .diagram_style import check_style
    from .orientation import check_overrides, subsystem_order
    style = check_style(style)
    db = load_db(tdb)
    els = [e.upper() for e in (elements or db_elements(db))]
    if len(els) < 3:
        raise ValueError(f"{tdb} has only {len(els)} elements; need at least 3 for ternaries")
    triples = list(itertools.combinations(sorted(els), 3))
    check_overrides(orientation, triples)
    triples = [tuple(subsystem_order(t, element_order, orientation)) for t in triples]
    stem = Path(tdb).stem
    out_dir = Path(out_dir or Path(output_pptx).with_suffix("").as_posix() + "_png")
    out_dir.mkdir(parents=True, exist_ok=True)
    u = "°C" if units.upper() == "C" else "K"
    tasks = []
    for T in temperatures_c:
        for tri in triples:
            name = f"{'-'.join(tri)} @ {T:g}{u}"
            png = str(out_dir / f"{stem}_{'-'.join(tri)}_{T:g}{units.upper()}.png")
            if style == "shaded":
                T_K = T + KELVIN if units.upper() == "C" else T
                kw = dict(tdb=str(tdb), elements=list(tri), corners=dict(zip(("left", "right", "top"), tri)),
                          temperature=T, temperature_unit="C" if units.upper() == "C" else "K", out_png=png,
                          composition_step=step, exclude_phases=exclude_phases, tielines=0, style="shaded",
                          formats=formats, time_limit_s=0.9 * timeout, check_mapping=check_mapping,
                          title="–".join(_label_el(e) for e in tri) + f"   {T_K - KELVIN:.0f} °C ({T_K:.0f} K)")
                tasks.append({"name": name, "kind": "map", "T": T, "kwargs": kw})
            else:
                tasks.append({"name": name, "kind": "ternary", "T": T,
                              "kwargs": dict(tdb=str(tdb), elements=list(tri), temperature=T, units=units,
                                             out_png=png, step=step, exclude_phases=exclude_phases,
                                             keep_order=True, check_mapping=check_mapping)})
    results = run_tasks(tasks, n_workers=n_workers, timeout=timeout, progress=progress)
    prs = pu.new_presentation()
    pu.add_title_slide(prs, f"Ternary isothermal sections: {Path(tdb).name}",
                       f"{len(triples)} ternaries of {'-'.join(e.capitalize() for e in els)}   •   "
                       f"T = {', '.join(f'{t:g}' for t in temperatures_c)} {u}",
                       note="Calculated with pycalphad mapping (TernaryStrategy).")
    grids = []
    for T in temperatures_c:
        res_T = [r for r, t in zip(results, tasks) if t["T"] == T]
        for i, chunk in enumerate(_chunks(res_T, per_slide)):
            g = make_grid(chunk, str(out_dir / f"{stem}_ternaries_{T:g}{units.upper()}_grid{i + 1}.png"))
            grids.append(g)
            pu.add_image_slide(prs, f"Ternary sections at {T:g} {u}" +
                               (f" ({i + 1})" if len(res_T) > per_slide else ""), g,
                               subtitle=Path(tdb).name)
    if individual_slides:
        for r in results:
            if r.ok:
                pu.add_image_slide(prs, r.name, r.png, subtitle=Path(tdb).name)
    pu.save(prs, output_pptx)
    return {"pptx": output_pptx, "grid_images": grids, "systems": [asdict(r) for r in results]}
