"""Phase-region labels for the diagram tools (toolkit side).

* :func:`label_axes` - labels a binary T-x or ternary isotherm axes from a *recipe* (database, elements, phases,
  conditions, plot units; see :mod:`phase_regions`), reusing mapped ZPF data when the caller has it.  The recipe
  and the region data are stored on the axes (``ax._calphad_diagram``), so a stored figure can be labelled
  (again) later by the ``label_phase_regions`` figure operation.
* figure operations: ``label_phase_regions``, ``move_label``, ``delete_label``, ``relabel``.

The geometry / placement code is in :mod:`label_place`, the equilibrium side in :mod:`phase_regions`, the names
in :mod:`phase_names` (those three are also copied into show_code scripts).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .figures import OpContext, register_op

GRID_N = {"binary": (80, 60), "ternary": (40, 40)}
STYLE_KEYS = {"fontsize", "min_fontsize", "color", "background", "background_alpha", "rotate", "leader",
              "invariant_fontsize", "invariant_color", "invariant_type", "fontweight"}


def _db_elements(db) -> List[str]:
    return sorted(str(e) for e in db.elements if str(e) not in ("VA", "/-"))


def liquids_of(db, phases) -> List[str]:
    out = []
    for p in phases:
        ph = db.phases.get(p)
        if (ph is not None and ph.model_hints.get("liquid")) or p.upper().startswith(("LIQ", "IONIC_LIQ")):
            out.append(p)
    return out


def check_style(style: Optional[Dict]) -> Dict:
    style = dict(style or {})
    bad = [k for k in style if k not in STYLE_KEYS]
    if bad:
        raise ValueError(f"label_style keys {bad} unknown; allowed: {sorted(STYLE_KEYS)}")
    return style


def label_axes(ax, recipe: Dict, *, lines=None, invariants=None, invariant_types: Optional[Dict] = None,
               label_invariants: bool = True, phase_labels: Optional[Dict[str, str]] = None,
               style: Optional[Dict] = None, reuse: bool = True) -> Dict:
    """Label the phase regions of a binary / ternary diagram axes.  Returns the placement report + "text"."""
    from . import pcache
    from .calc import get_db
    from .label_place import label_diagram
    from .phase_regions import (equilibrium_grid, key_at, map_lines, ordered_info, regions_from_lines)
    t0 = time.time()
    style = check_style(style)
    kind = recipe["kind"]
    if kind not in ("binary", "ternary"):
        raise ValueError("phase-region labels are available for binary T-x diagrams and ternary isotherms")
    db = get_db(recipe["tdb"])
    comps = sorted(recipe["elements"]) + ["VA"]
    stored = getattr(ax, "_calphad_diagram", None) or {}
    with pcache.use():
        info = ordered_info(db, comps, recipe["phases"], pcache.cached_models(db, comps, recipe["phases"])) \
            if recipe.get("rename_disordered", True) else {}
        regions = stored.get("regions") if (reuse and lines is None) else None
        if regions is None:
            if lines is None:
                lines, invariants = map_lines(db, recipe, info, recipe.get("composition_step", 0.02),
                                              recipe.get("temperature_step", 10.0))
            regions = regions_from_lines(lines, invariants or [], recipe)
        invariant_types = invariant_types or stored.get("invariant_types")
        for inv in regions.get("invariants", []):
            if invariant_types:
                typ = next((t for T, t in invariant_types.items() if abs(float(T) - inv["T_K"]) < 0.5), None)
                if typ:
                    inv["type"] = typ
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        grid = stored.get("grid") if reuse and stored.get("grid_lim") == [list(xlim), list(ylim)] else None
        if grid is None:
            grid = equilibrium_grid(db, recipe, info, xlim, ylim, GRID_N[kind])
        n_checks = [0]

        def probe(x, y):
            n_checks[0] += 1
            with pcache.use():
                return key_at(db, recipe, info, x, y)
        ax._calphad_diagram = {**stored, "recipe": dict(recipe), "regions": regions, "grid": grid,
                               "grid_lim": [list(xlim), list(ylim)]}
        rep = label_diagram(ax, regions, grid, probe, kind=kind, elements=_db_elements(db),
                            phase_labels=phase_labels, liquids=liquids_of(db, recipe["phases"]),
                            invariants=label_invariants, T_unit=recipe.get("T_unit", "K"), style=style)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["equilibrium_checks"] = n_checks[0]
    rep["text"] = report_text(rep)
    return rep


def report_text(rep: Dict) -> str:
    labs = rep["labels"]
    reg = [lab["text"] + (" (leader)" if lab["leader"] else "") for lab in labs if lab["kind"] != "invariant"]
    inv = [lab["text"] for lab in labs if lab["kind"] == "invariant"]
    out = [f"phase labels ({len(reg)}): " + ", ".join(reg)]
    if inv:
        out.append("invariant temperatures: " + ", ".join(inv))
    unconf = [lab["text"] for lab in labs if lab["verified"] is None and lab["kind"] != "invariant"]
    if unconf:
        out.append("not checked by equilibrium: " + ", ".join(unconf))
    if rep["skipped"]:
        out.append("not labelled: " + "; ".join(f"{s['label']} ({s['reason']})" for s in rep["skipped"]))
    out.append("labels are separate text objects (SVG ids phase_label:<PHASES> / invariant_label:<T>); move, "
               "delete or reword them with edit_figure move_label / delete_label / relabel")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------
# figure operations
# ---------------------------------------------------------------------------------------------------------
@register_op("label_phase_regions",
             "label_phase_regions {invariants: true, phase_labels: {NAME: text}, fontsize, min_fontsize, color, "
             "background, invariant_type, leader, rotate}: phase names in each region / tie-triangle and invariant "
             "temperatures, placed clear of data points (diagrams from map_phase_diagram or "
             "plot_diagram_with_data); replaces earlier labels", group="annotate")
def _op_label(ctx: OpContext):
    ax, op = ctx.ax, ctx.op
    d = getattr(ax, "_calphad_diagram", None)
    if not d or "recipe" not in d:
        raise ValueError("this axes has no phase-diagram recipe: make the diagram with map_phase_diagram "
                         "(binary or ternary isotherm) or plot_diagram_with_data first")
    style = {k: op[k] for k in STYLE_KEYS if k in op}
    rep = label_axes(ax, d["recipe"], label_invariants=op.get("invariants", True),
                     phase_labels=op.get("phase_labels"), style=style)
    ctx.note(rep["text"].replace("\n", "; "))


def _targets(ctx: OpContext):
    from .label_place import find_labels
    op = ctx.op
    if not op.get("id") and op.get("text") is None:
        raise ValueError("give id (e.g. 'phase_label:LIQUID.FCC_A1' or 'LIQUID.FCC_A1') or text of the label")
    hits = find_labels(ctx.ax, op.get("id"), op.get("text"))
    if not hits:
        have = [f"{t.get_gid()} ({t.get_text()})" for t in find_labels(ctx.ax)]
        raise ValueError(f"no label matches {op.get('id') or op.get('text')!r}; labels: {have}")
    return hits


@register_op("move_label", "move_label {id | text, dx, dy (points) | x, y (as displayed)}: move a phase/invariant "
             "label", group="annotate")
def _op_move(ctx: OpContext):
    from .label_place import move_label
    for t in _targets(ctx):
        if ctx.op.get("x") is not None and ctx.op.get("y") is not None:
            move_label(t, xy=(float(ctx.X()[0]), float(ctx.Y()[0])))
        else:
            move_label(t, float(ctx.op.get("dx", 0.0)), float(ctx.op.get("dy", 0.0)))


@register_op("delete_label", "delete_label {id | text}: remove a phase/invariant label (id 'invariant_label' = all "
             "temperature labels)", group="annotate")
def _op_delete(ctx: OpContext):
    hits = _targets(ctx)
    for t in hits:
        t.remove()
    ctx.note(f"removed {len(hits)} label(s)")


@register_op("relabel", "relabel {id | text, new_text, fontsize, color, rotation, fontweight}: reword/restyle a "
             "phase/invariant label", group="annotate")
def _op_relabel(ctx: OpContext):
    op = ctx.op
    for t in _targets(ctx):
        if op.get("new_text") is not None:
            t.set_text(str(op["new_text"]))
        for k in ("fontsize", "color", "rotation", "fontweight"):
            if k in op:
                getattr(t, f"set_{k}")(op[k])
