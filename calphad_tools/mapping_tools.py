"""Agent tools for pycalphad mapping data export, Scheil solidification and database/model output.

Registered by :func:`register` (called from ``calphad_tools.agent.tools``):

* ``map_phase_diagram``  - binary T-x, ternary isotherm or isopleth via ``pycalphad.mapping``
  strategies: configurable PNG + machine-readable phase boundaries, tie-lines, invariants.
* ``step_map``           - ``StepStrategy`` (one varying condition): phase amounts, phase
  compositions, chemical potentials and system properties vs the stepped variable.
* ``scheil_solidification`` - Scheil-Gulliver + equilibrium solidification (``scheil`` package).
* ``write_database``     - write a database (optionally an element/phase subset) with pycalphad's
  TDB writer, verified by reloading.
* ``export_phase_model`` - symbolic Gibbs energy contributions of one phase, contributing
  parameters (CSV), optional LaTeX and numeric evaluation.

Everything is plain functions so it can also be used without the agent.
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import re
import time
import warnings
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import pcache  # noqa: E402  (process-wide model / compiled-code cache)

KELVIN = 273.15
STYLE_SCHEMA = {"type": "string", "enum": ["shaded", "color"],
                "description": "shaded: black lines, two-phase fields filled, grey ternary, phases named by labels "
                               "in the figure (no legend); color: colour per phase + tie-lines + legend. Default: "
                               "config diagram_style (shaded)"}
TIELINE_COLOR = (0.45, 0.45, 0.45, 0.45)
INVARIANT_COLOR = "#d62728"

# ---------------------------------------------------------------------------
# Consistent phase colours (same phase -> same colour in every plot)
# ---------------------------------------------------------------------------
_FIXED_COLORS = {
    "LIQUID": "#1f77b4", "IONIC_LIQ": "#1f77b4", "GAS": "#17becf",
    "FCC_A1": "#ff7f0e", "A1_FCC": "#ff7f0e", "FCC_L12": "#b35806", "L12_FCC": "#b35806",
    "BCC_A2": "#9467bd", "A2_BCC": "#9467bd", "BCC_B2": "#4b2a7b", "B2_BCC": "#4b2a7b",
    "HCP_A3": "#8c564b", "A3_HCP": "#8c564b", "SIGMA": "#e377c2", "SIGMA_D8B": "#e377c2",
    "LAVES_C14": "#bcbd22", "LAVES_C15": "#7f7f00", "LAVES_C36": "#dbdb8d", "MU_PHASE": "#393b79",
    "CEMENTITE": "#843c39", "M23C6": "#ad494a", "M7C3": "#d6616b", "GRAPHITE": "#252525",
    "DIAMOND_A4": "#636363",
}
_PALETTE = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#393b79",
            "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1",
            "#636363", "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#969696", "#5254a3", "#8ca252",
            "#bd9e39", "#ad494a", "#a55194", "#b35806", "#542788", "#01665e", "#8c510a", "#c51b7d"]


def phase_color(name: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """Deterministic colour for a phase name (miscibility-gap suffix '#2' ignored)."""
    base = str(name).split("#")[0].upper()
    if overrides:
        ov = {k.upper(): val for k, val in overrides.items()}
        if base in ov:
            return ov[base]
    if base in _FIXED_COLORS:
        return _FIXED_COLORS[base]
    return _PALETTE[zlib.crc32(base.encode()) % len(_PALETTE)]


def _legend(ax, names: Sequence[str], overrides=None, title=None, loc="center left"):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=phase_color(n, overrides), label=n) for n in names]
    if handles:
        ax.legend(handles=handles, loc=loc, bbox_to_anchor=(1.02, 0.5), fontsize=8, title=title,
                  title_fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _el(s: str) -> str:
    s = str(s).strip().upper()
    m = re.fullmatch(r"[XW]\(([A-Z]{1,2})\)", s)
    return m.group(1) if m else s


def _cap(e: str) -> str:
    return e.capitalize()


def _lc(label: str) -> str:
    """'Mole fraction Cr' -> 'mole fraction Cr'."""
    return label[:1].lower() + label[1:]


def _f(x) -> float:
    return float(np.squeeze(np.asarray(x, dtype=float)))


def _toK(t: float, unit: str) -> float:
    return float(t) + (KELVIN if str(unit).upper().startswith("C") else 0.0)


def _is_liquid(db, name: str) -> bool:
    base = name.split("#")[0]
    ph = db.phases.get(base)
    if ph is not None and ph.model_hints.get("liquid"):
        return True
    return base.upper().startswith(("LIQ", "IONIC_LIQ"))


def _check_elements(db, elements: Sequence[str]) -> List[str]:
    els = [_el(e) for e in elements]
    have = sorted(str(e) for e in db.elements if str(e) not in ("VA", "/-"))
    bad = [e for e in els if e not in have]
    if bad:
        raise ValueError(f"elements {bad} not in database; available: {have}")
    if len(set(els)) != len(els):
        raise ValueError(f"duplicate elements in {els}")
    return els


def _select_phases(db, els: Sequence[str], phases=None, exclude_phases=None) -> List[str]:
    from .diagrams import phases_for
    avail = phases_for(db, els)
    if phases:
        want = [p.upper() for p in phases]
        bad = [p for p in want if p not in avail]
        if bad:
            raise ValueError(f"phases {bad} not available for {'-'.join(els)}; available: {avail}")
        avail = want
    if exclude_phases:
        ex = {p.upper() for p in exclude_phases}
        avail = [p for p in avail if p not in ex]
    if not avail:
        raise ValueError("no phases left to calculate")
    return sorted(avail)


def _write_csv(path: str, rows: List[Dict], columns: Optional[List[str]] = None) -> str:
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})
    return path


def _fmt(x):
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else f"{float(x):.8g}"
    return x


def _siblings(png: str) -> Tuple[Path, str]:
    p = Path(png)
    return p.parent, p.stem


def _masses(db, els: Sequence[str]) -> Dict[str, float]:
    out = {}
    for e in els:
        m = float(db.refstates.get(e, {}).get("mass", 0.0) or 0.0)
        if m <= 0:
            raise ValueError(f"no atomic mass for {e} in the database ELEMENT line; use a mole-fraction unit")
        out[e] = m
    return out


def _to_weight(xdict: Dict[str, float], masses: Dict[str, float]) -> Dict[str, float]:
    tot = sum(xdict[e] * masses[e] for e in masses)
    return {e: xdict[e] * masses[e] / tot for e in masses}


# ---------------------------------------------------------------------------
# Ordered/disordered phase naming
# ---------------------------------------------------------------------------
class _Labeler:
    """Names composition sets; an ordered phase (partitioned model) whose equivalent substitutional
    sublattices have equal site fractions is reported by its disordered-part name (e.g. B2_BCC -> A2_BCC)."""

    def __init__(self, db, models: Dict, rename: bool = True, tol: float = 1e-4):
        self.info = {}
        self.renamed: Dict[str, str] = {}
        if not rename:
            return
        for name, mod in models.items():
            ph = db.phases.get(name)
            if ph is None or ph.model_hints.get("ordered_phase") != name:
                continue
            dis = ph.model_hints.get("disordered_phase")
            if not dis or dis not in db.phases:
                continue
            n_sub = len(db.phases[name].sublattices) - (len(db.phases[dis].sublattices) - 1)
            if n_sub < 2:
                continue
            m = re.fullmatch(r"([A-Z]\d+)_([A-Z]+)", dis)  # A1_FCC -> FCC_A1 if that phase also exists
            if m and f"{m.group(2)}_{m.group(1)}" in db.phases:
                dis = f"{m.group(2)}_{m.group(1)}"
            idx = {}
            for j, y in enumerate(mod.site_fractions):
                idx[(int(y.sublattice_index), str(y.species.name))] = j
            self.info[name] = (dis, n_sub, idx)
        self.tol = tol

    def name(self, cs) -> str:
        pn = cs.phase_record.phase_name
        inf = self.info.get(pn)
        if inf is None:
            return pn
        dis, n_sub, idx = inf
        nsv = cs.phase_record.num_statevars
        y = np.asarray(cs.dof)[nsv:]
        species = {sp for (s, sp) in idx if s < n_sub}
        for sp in species:
            vals = [y[idx[(s, sp)]] if (s, sp) in idx else 0.0 for s in range(n_sub)]
            if np.ptp(vals) > self.tol:
                return pn
        self.renamed[pn] = dis
        return dis

    def names(self, compsets) -> List[str]:
        out, seen = [], {}
        for cs in compsets:
            n = self.name(cs)
            seen[n] = seen.get(n, 0) + 1
            out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
        return out


# ---------------------------------------------------------------------------
# Mapping driver
# ---------------------------------------------------------------------------
def _run_strategy(strategy, time_limit: float) -> Dict:
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(strategy.node_queue.nodes) == 0 and len(strategy.zpf_lines) == 0:
            strategy.generate_automatic_starting_points()
        finished, n = False, 0
        while not finished:
            finished = strategy.iterate()
            n += 1
            if not finished and time.time() - t0 > time_limit:
                break
    return {"finished": bool(finished), "iterations": n, "seconds": round(time.time() - t0, 1)}


def _point_T(p) -> float:
    from pycalphad import variables as v
    return _f(p.global_conditions[v.T])


def _cs_X(cs, elements) -> Dict[str, float]:
    x = np.asarray(cs.X, dtype=float)
    return {e: float(x[i]) for i, e in enumerate(elements)}


class _Units:
    """Conversion of temperatures / compositions to display units."""

    def __init__(self, temperature_unit="K", composition_unit="mole_fraction", masses=None):
        self.tC = str(temperature_unit).upper().startswith("C")
        cu = (composition_unit or "mole_fraction").lower()
        if cu not in ("mole_fraction", "mole_percent", "weight_fraction", "weight_percent"):
            raise ValueError("composition_unit must be mole_fraction, mole_percent, weight_fraction or weight_percent")
        self.weight = cu.startswith("weight")
        self.scale = 100.0 if cu.endswith("percent") else 1.0
        self.masses = masses
        self.cu = cu

    def T(self, tk: float) -> float:
        return tk - KELVIN if self.tC else tk

    def comp(self, xdict: Dict[str, float], el: str) -> float:
        if self.weight:
            xdict = _to_weight(xdict, self.masses)
        return xdict[el] * self.scale

    def T_label(self) -> str:
        return "Temperature (°C)" if self.tC else "Temperature (K)"

    def comp_label(self, el: str) -> str:
        kind = "Mass" if self.weight else "Mole"
        return f"{kind} {'percent' if self.scale == 100 else 'fraction'} {_cap(el)}"


def _end_pair_side(lines, inv: Dict, b: str, xs3: Sequence[float], tolT: float = 0.5, tolX: float = 5e-3):
    """+1 if the two-phase region of the end phases (lowest/highest composition xs3[0], xs3[2]) of a binary
    invariant lies above the invariant temperature, -1 if below, 0 if unknown.  Uses the mapped ZPF lines
    (name independent: line end points are matched to the nearest invariant phase composition)."""
    T0 = inv["T_K"]
    votes = 0
    for ln in lines:
        pts = ln["points"]
        if len(pts) < 2 or len(ln["phases"]) != 2:
            continue
        for end in (pts[0], pts[-1]):
            if abs(end["T"] - T0) > tolT:
                continue
            idx = []
            for _, X in end["phases"]:
                d = [abs(X[b] - x) for x in xs3]
                k = int(np.argmin(d))
                idx.append(k if d[k] < tolX else -1)
            if sorted(idx) == [0, 2]:
                meanT = float(np.mean([p["T"] for p in pts]))
                votes += 1 if meanT > T0 else -1
    return int(np.sign(votes))


def _classify_binary(db, comps, phases, els, inv: Dict, lines, is_liquid: Callable[[str], bool]) -> Dict:
    """Eutectic-type (middle phase decomposes on cooling) or peritectic-type (end phases react to form the
    middle phase) from the side on which the end-phase two-phase region lies; falls back to an equilibrium
    calculation 1 K below the invariant at the middle-phase composition."""
    b = els[1]
    ph = sorted(inv["phases"], key=lambda p: inv["compositions"][p][b])
    lo, mid, hi = ph
    x = {p: inv["compositions"][p][b] for p in ph}
    side = _end_pair_side(lines, inv, b, [x[lo], x[mid], x[hi]])
    how = "zpf-lines"
    if side == 0:
        how = "equilibrium check"
        try:
            from pycalphad import equilibrium, variables as v
            xm = min(max(x[mid], 1e-6), 1 - 1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eq = equilibrium(db, comps, phases, {v.T: inv["T_K"] - 1.0, v.P: inv.get("P", 101325.0), v.N: 1,
                                                     v.X(b): xm})
            xs = np.atleast_1d(np.squeeze(eq.X.sel(component=b).values))
            nps = np.atleast_1d(np.squeeze(eq.NP.values))
            ok = np.isfinite(nps) & (nps > 0)
            near = np.abs(xs - x[mid]) < 5e-3
            side = 1 if np.nansum(nps[ok & near]) > 0.9 else -1
        except Exception:  # noqa: BLE001 - classification is best effort
            return {"type": "unclassified", "reaction_on_cooling": " + ".join(ph)}
    liq = {p: is_liquid(p) for p in ph}
    if side > 0:  # lo + hi -> mid on cooling
        if liq[mid]:
            kind = "unusual (liquid forms on cooling)"
        elif liq[lo] and liq[hi]:
            kind = "syntectic"
        elif liq[lo] or liq[hi]:
            kind = "peritectic"
        else:
            kind = "peritectoid"
        reaction = f"{lo} + {hi} -> {mid}"
    else:  # mid -> lo + hi on cooling
        if liq[mid]:
            kind = "monotectic" if (liq[lo] or liq[hi]) else "eutectic"
        else:
            kind = "metatectic" if (liq[lo] or liq[hi]) else "eutectoid"
        reaction = f"{mid} -> {lo} + {hi}"
    return {"type": kind, "reaction_on_cooling": reaction, "classified_by": how}


# ---------------------------------------------------------------------------
# map_phase_diagram
# ---------------------------------------------------------------------------
@pcache.cached
def map_phase_diagram(tdb: str, elements: Sequence[str], out_png: str, *, tmin: Optional[float] = None,
                      tmax: Optional[float] = None, temperature: Optional[float] = None,
                      temperature_unit: str = "K", vary: Optional[str] = None,
                      fixed: Optional[Dict[str, float]] = None, x_range: Optional[Sequence[float]] = None,
                      composition_step: Optional[float] = None, temperature_step: float = 10.0,
                      composition_unit: str = "mole_fraction", potential_axis: Optional[str] = None,
                      tielines: int = 3, label_nodes: bool = False, xlim=None, ylim=None,
                      phases=None, exclude_phases=None, phase_colors: Optional[Dict[str, str]] = None,
                      title: Optional[str] = None, rename_disordered: bool = True,
                      pressure: float = 101325.0, time_limit_s: float = 600.0, dpi: int = 150,
                      label_regions: Optional[bool] = None, label_invariants: bool = True,
                      phase_labels: Optional[Dict[str, str]] = None, label_style: Optional[Dict] = None,
                      formats: Optional[Sequence[str]] = None, right: Optional[str] = None,
                      corners: Optional[Dict[str, str]] = None, style: Optional[str] = "shaded") -> Dict:
    """Run a pycalphad mapping strategy and write PNG + CSV/JSON data.  Returns a result dict.

    label_regions: write the phases of every region / tie-triangle (and binary invariant temperatures) on the
    diagram (see region_labels); None = on in the shaded style (binary T-x and ternary isotherms; isopleth and
    T-MU lines are named on the line instead), off in the color style.  formats: extra outputs next to the PNG
    (default svg + pdf in the shaded style or when labelling).  style: "shaded" (black boundaries, aliceblue
    two-phase fields, grey ternary lines, no legend; see diagram_style) or "color" (one colour per phase, tie-lines,
    phase legend).
    Orientation (see orientation.py): right = element on the right of a binary (x axis = its fraction; isopleth:
    the varying element); corners = {left, right, top} of a ternary; otherwise the element order decides
    (binary [left, right], ternary [bottom-left, bottom-right, top])."""
    from pycalphad import variables as v
    from pycalphad.mapping import BinaryStrategy, IsoplethStrategy, TernaryStrategy
    from .calc import get_db
    from . import diagram_style as ds
    style = ds.check_style(style)
    shaded = style == "shaded"

    db = get_db(tdb)
    els = _check_elements(db, elements)
    fixed = {_el(k): float(val) for k, val in (fixed or {}).items()}
    if right is not None and fixed:               # isopleth: the x axis is the varying element
        if vary and _el(vary) != _el(right):
            raise ValueError(f"isopleth: the x axis is the varying element {_el(vary)}; right={right!r} "
                             f"contradicts it (give one of them)")
        vary = vary or right
    if fixed or vary:
        kind = "isopleth"
    elif len(els) == 2:
        kind = "binary"
    elif len(els) == 3 and temperature is not None:
        kind = "ternary"
    else:
        raise ValueError("give 2 elements (binary T-x, needs tmin/tmax), 3 elements + temperature (ternary "
                         "isotherm), or elements + vary + fixed + tmin/tmax (isopleth)")
    from .orientation import binary_order, corner_labels, orientation_info, orientation_text, ternary_order
    if corners and kind != "ternary":
        raise ValueError("corners is for ternary isotherms; a binary (or isopleth) uses right")
    if kind == "binary":
        els = binary_order(els, right)
    elif kind == "ternary":
        els = ternary_order(els, corners, right)
    ph = _select_phases(db, els, phases, exclude_phases)
    comps = sorted(els) + ["VA"]
    masses = _masses(db, sorted(els)) if str(composition_unit).lower().startswith("weight") else None
    U = _Units(temperature_unit, composition_unit, masses)
    conds = {v.P: float(pressure), v.N: 1}
    xr = [float(x_range[0]), float(x_range[1])] if x_range else [0.0, 1.0]

    if kind in ("binary", "isopleth"):
        if tmin is None or tmax is None:
            raise ValueError(f"{kind} needs tmin and tmax")
        tlo, thi = sorted((_toK(tmin, temperature_unit), _toK(tmax, temperature_unit)))
        conds[v.T] = (tlo, thi, float(temperature_step))
    if kind == "binary":
        axis_el = els[1]
        step = composition_step or 0.02
        conds[v.X(axis_el)] = (xr[0], xr[1], step)
        Strat = BinaryStrategy
    elif kind == "ternary":
        step = composition_step or 0.02
        conds[v.T] = _toK(temperature, temperature_unit)
        conds[v.X(els[1])] = (0, 1, step)
        conds[v.X(els[2])] = (0, 1, step)
        Strat = TernaryStrategy
    else:
        if not vary or not fixed:
            raise ValueError("isopleth needs `vary` (element on the x axis) and `fixed` mole fractions")
        axis_el = _el(vary)
        if axis_el not in els or any(k not in els for k in fixed) or axis_el in fixed:
            raise ValueError(f"vary/fixed elements must be distinct members of {els}")
        if len(fixed) != len(els) - 2:
            raise ValueError(f"isopleth: fix the mole fraction of all elements except `vary` and one balance "
                             f"element (elements={els}, fixed={list(fixed)})")
        step = composition_step or 0.02
        xmax = min(xr[1], 1.0 - sum(fixed.values()))
        conds[v.X(axis_el)] = (xr[0], xmax, step)
        for k, val in fixed.items():
            conds[v.X(k)] = val
        Strat = IsoplethStrategy
    if label_regions is None:   # shaded: no legend, so the phases are named inside the figure
        label_regions = shaded and kind in ("binary", "ternary") and not potential_axis
    if label_regions and kind == "isopleth":
        raise ValueError("label_regions is available for binary T-x diagrams and ternary isotherms, not isopleths")
    if label_regions and potential_axis:
        raise ValueError("label_regions is not available on a T-MU (potential_axis) diagram")
    if label_style:
        from .region_labels import check_style
        check_style(label_style)
    fmts = [str(f).lower().lstrip(".") for f in (formats if formats is not None else
                                                  (["svg", "pdf"] if label_regions or shaded else []))]
    bad = [f for f in fmts if f not in ("png", "svg", "pdf", "eps")]
    if bad:
        raise ValueError(f"unsupported formats {bad} (png, svg, pdf, eps)")
    if potential_axis and kind != "binary":
        raise ValueError("potential_axis is only supported for binary diagrams")
    mu_el = _el(potential_axis) if potential_axis else None
    if mu_el and mu_el not in els:
        raise ValueError(f"potential_axis element must be one of {els}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strategy = Strat(db, comps, ph, conds)
    run = _run_strategy(strategy, time_limit_s)
    lab = _Labeler(db, strategy.models, rename_disordered)
    s_els = list(strategy.elements)  # order of cs.X / chemical potentials
    s_els = [e for e in s_els if e != "VA"]
    outdir, stem = _siblings(out_png)
    files = {"png": out_png}
    system = "-".join(_cap(e) for e in els)

    if kind == "isopleth":
        res = _isopleth_data(strategy, db, els, axis_el, fixed, s_els, lab, U, outdir, stem, files)
    else:
        res = _tieline_data(strategy, db, comps, ph, els, kind, s_els, lab, U, outdir, stem, files)
        if kind == "ternary":  # the triangular projection needs fractions; percent is shown via tick labels
            res["U"] = _Units(temperature_unit, "weight_fraction" if U.weight else "mole_fraction", masses)

    # ------------------------------------------------------------ plot
    fig_kw = {"figsize": (7.2, 5.4)} if kind != "ternary" else {"figsize": (7.0, 6.0)}
    fig = plt.figure(**fig_kw)
    if kind == "ternary":
        from pycalphad.plot import triangular  # noqa: F401 - registers the projection
        ax = fig.add_subplot(projection="triangular")
    else:
        ax = fig.add_subplot()
    if kind == "isopleth":
        _plot_isopleth(ax, res, phase_colors, style)
        ax.set_xlabel(U.comp_label(axis_el))
        ax.set_ylabel(U.T_label())
        legend_title = "ZPF line of"
    else:
        _plot_tielines(ax, res, kind, tielines, label_nodes, phase_colors, mu_el, style)
        legend_title = None
        if kind == "binary":
            ax.set_xlabel(f"MU({_cap(mu_el)}) (kJ/mol)" if mu_el else U.comp_label(els[1]))
            ax.set_ylabel(U.T_label())
        else:
            ax.set_xlabel(U.comp_label(els[1]))
            ax.set_ylabel(U.comp_label(els[2]))
    if kind == "ternary":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if U.scale == 100:
            from .figures import ScaledFormatter
            ax.xaxis.set_major_formatter(ScaledFormatter(scale=100))
            ax.yaxis.set_major_formatter(ScaledFormatter(scale=100))
        ax.yaxis.label.set_rotation(60)
        ax.yaxis.set_label_coords(x=0.12, y=0.5)
        corner_labels(ax, els)
    else:
        if not mu_el:
            if kind == "binary":
                ax.set_xlim(U.comp({els[0]: 1 - xr[0], els[1]: xr[0]}, els[1]),
                            U.comp({els[0]: 1 - xr[1], els[1]: xr[1]}, els[1]))
            elif res.get("xlim"):
                ax.set_xlim(*res["xlim"])
        ax.set_ylim(U.T(conds[v.T][0]), U.T(conds[v.T][1]))
        if shaded:
            ds.ticks_in(ax)
        else:
            ax.grid(alpha=0.25)
    if xlim:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if not shaded:
        _legend(ax, sorted(res["phases_found"]), phase_colors, title=legend_title)
    elif kind == "isopleth" or mu_el:   # no region labels there: name each line on the line
        _shaded_line_labels(ax, res, kind, els, mu_el, phase_labels)
    if kind == "ternary":
        Tk = conds[v.T]
        default_title = f"{system}   {Tk - KELVIN:.0f} °C ({Tk:.0f} K)"
    elif kind == "isopleth":
        default_title = f"{system} isopleth, " + ", ".join(
            f"{'w' if U.weight else 'x'}({_cap(k)})={val:g}" for k, val in fixed.items()) + \
            (" (mole fractions)" if U.weight else "")
    else:
        default_title = system
    if not run["finished"]:
        default_title += "  [PARTIAL: time limit]"
    ax.set_title(title or default_title, fontsize=11)
    recipe = _recipe(tdb, kind, els, ph, pressure, conds, U, temperature_unit, rename_disordered, xr,
                     composition_step or 0.02, temperature_step) if not mu_el else None
    if kind == "isopleth":      # orientation / units for composition points (no region labels on isopleths)
        balance = [e for e in els if e != axis_el and e not in fixed][0]
        recipe.update(axis_element=axis_el, balance=balance, fixed=dict(fixed))
        ax._calphad_diagram = {"recipe": recipe, "style": style}
        recipe = None
    labels = None
    if recipe is not None:      # what edit_figure label_phase_regions / composition_points need later
        from .phase_regions import regions_from_lines
        ax._calphad_diagram = {"recipe": recipe, "regions": regions_from_lines(res["lines"], res["invariants"],
                                                                               recipe, db),
                               "invariant_types": {inv["T_K"]: inv.get("type", "") for inv in res["invariants"]},
                               "style": style}
    if label_regions:
        from .region_labels import label_axes
        labels = label_axes(ax, recipe, label_invariants=label_invariants, phase_labels=phase_labels,
                            style=label_style)
        from .phase_names import prettify_legend
        prettify_legend(ax, els, phase_labels)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    for ext in fmts:
        if ext != "png":
            from .label_place import save_vector
            files[ext] = str(Path(out_png).with_suffix("." + ext))
            save_vector(fig, files[ext], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "kind": kind, "system": system, "tdb": str(tdb), "elements": els, "phases_considered": ph,
        "phases_found": sorted(res["phases_found"]),
        "conditions": {str(k): (list(val) if isinstance(val, tuple) else val) for k, val in conds.items()},
        "temperature_unit_in_files": "K and C columns", "composition_unit_in_plot": U.cu,
        "mapping": {**run, "zpf_lines": len(strategy.zpf_lines),
                    "degenerate_edge_lines_dropped": res.get("degenerate_lines_dropped", 0),
                    "duplicate_lines_dropped": res.get("duplicate_lines_dropped", 0)},
        "regions": res["regions"], "invariants": res["invariants"], "files": files,
        "renamed_when_disordered": dict(lab.renamed), "style": style,
    }
    if kind == "isopleth":
        bal = [e for e in els if e != axis_el and e not in fixed][0]
        summary["orientation"] = orientation_info(kind, els, axis_el, bal)
        o_text = orientation_text(kind, els, _lc(U.comp_label(axis_el)), axis_el, bal)
    else:
        summary["orientation"] = orientation_info(kind, els)
        o_text = orientation_text(kind, els, (f"MU({_cap(mu_el)})" if mu_el else _lc(U.comp_label(els[1])))
                                  if kind == "binary" else None)
    if labels is not None:
        summary["labels"] = {k: labels[k] for k in ("labels", "skipped", "seconds", "equilibrium_checks")}
    jpath = str(outdir / f"{stem}.json")
    Path(jpath).write_text(json.dumps(summary, indent=1, default=str))
    files["json"] = jpath
    head, _, rest = _map_text(summary, U).partition("\n")
    summary["text"] = head + "\n" + o_text + "\n" + rest + (("\n" + labels["text"]) if labels is not None else "")
    return summary


def _recipe(tdb, kind, els, ph, pressure, conds, U, temperature_unit, rename, xr, step, T_step) -> Dict:
    """What region labelling needs to recompute this diagram (stored on the axes; see phase_regions)."""
    from pycalphad import variables as v
    r = {"kind": kind, "tdb": str(Path(tdb).resolve()), "elements": list(els), "phases": list(ph),
         "P": float(pressure), "rename_disordered": bool(rename), "composition_step": float(step),
         "temperature_step": float(T_step), "T_unit": "C" if U.tC else "K",
         "comp": {"scale": 1.0 if kind == "ternary" else float(U.scale), "weight": bool(U.weight),
                  "masses": dict(U.masses or {})}}
    if kind in ("binary", "isopleth"):
        r["tmap"] = [1.0, KELVIN if U.tC else 0.0]
        r["T_range"] = [float(conds[v.T][0]), float(conds[v.T][1])]
        r["x_range"] = list(xr)
    else:
        r["T_K"] = float(conds[v.T])
    return r


def _tieline_data(strategy, db, comps, ph, els, kind, s_els, lab, U, outdir, stem, files) -> Dict:
    from pycalphad import variables as v
    lines, bound_rows, tie_rows = [], [], []
    degenerate, duplicates, seen_sigs = 0, 0, set()
    for li, z in enumerate(strategy.zpf_lines):
        if not z.points:
            continue
        # zero-length tie-lines everywhere: artefact running along a pure-element / binary edge
        if all(max(np.max(np.abs(np.asarray(a.X, float) - np.asarray(b.X, float)))
                   for a, b in itertools.combinations(p.stable_composition_sets, 2)) < 1e-5
               for p in z.points if len(p.stable_composition_sets) > 1):
            degenerate += 1
            continue
        sig = tuple(sorted((round(_point_T(p), 3),) + tuple(np.round(np.concatenate(
            [np.asarray(cs.X, float) for cs in p.stable_composition_sets]), 5)) for p in (z.points[0], z.points[-1])))
        if sig in seen_sigs:  # same line traced twice (e.g. both directions from a starting point)
            duplicates += 1
            continue
        seen_sigs.add(sig)
        mid = z.points[len(z.points) // 2]
        names = lab.names(mid.stable_composition_sets)
        region = " + ".join(sorted(names))
        pts = []
        for pi, p in enumerate(z.points):
            T = _point_T(p)
            mu = {e: float(p.chemical_potentials[i]) for i, e in enumerate(s_els)}
            per = []
            tie = {"line": li, "point": pi, "region": region, "T_K": T, "T_C": T - KELVIN}
            for k, (n, cs) in enumerate(zip(names, p.stable_composition_sets)):
                X = _cs_X(cs, s_els)
                per.append((n, X))
                row = {"line": li, "region": region, "point": pi, "phase": n, "T_K": T, "T_C": T - KELVIN}
                row.update({f"X({e})": X[e] for e in els})
                if U.weight:
                    row.update({f"W({e})": w for e, w in _to_weight(X, U.masses).items()})
                row.update({f"MU({e})": mu[e] for e in els})
                bound_rows.append(row)
                tie[f"phase_{k + 1}"] = n
                tie.update({f"X{k + 1}({e})": X[e] for e in els})
            tie.update({f"MU({e})": mu[e] for e in els})
            tie_rows.append(tie)
            pts.append({"T": T, "mu": mu, "phases": per})
        lines.append({"id": li, "phases": names, "region": region, "points": pts})

    # invariants / tie-triangles (3-phase nodes), de-duplicated
    invs = []
    for node in strategy.node_queue.nodes:
        if len(node.stable_composition_sets) != 3:
            continue
        names = lab.names(node.stable_composition_sets)
        T = _point_T(node)
        compsx = {n: _cs_X(cs, s_els) for n, cs in zip(names, node.stable_composition_sets)}
        key = sorted(tuple(round(c[e], 3) for e in els) for c in compsx.values())
        dup = next((i for i in invs if abs(i["T_K"] - T) < 0.05 and i["_key"] == key), None)
        if dup is not None:
            alias = " + ".join(sorted(names))
            if alias != " + ".join(sorted(dup["phases"])):
                dup.setdefault("also_found_as", []).append(alias)
            continue
        invs.append({"T_K": round(T, 3), "T_C": round(T - KELVIN, 3), "phases": names,
                     "compositions": {n: {e: round(c[e], 6) for e in els} for n, c in compsx.items()},
                     "MU": {e: round(float(node.chemical_potentials[i]), 3) for i, e in enumerate(s_els)},
                     "P": _f(node.global_conditions.get(v.P, 101325.0)), "_key": key})
    for inv in invs:
        inv.pop("_key")
    if kind == "binary":
        for inv in invs:
            inv.update(_classify_binary(db, comps, ph, els, inv, lines, lambda p: _is_liquid(db, p)))
        invs.sort(key=lambda i: -i["T_K"])
    inv_rows = []
    for k, inv in enumerate(invs):
        for n in inv["phases"]:
            r = {"invariant": k, "T_K": inv["T_K"], "T_C": inv["T_C"], "type": inv.get("type", "three-phase region"),
                 "reaction_on_cooling": inv.get("reaction_on_cooling", ""), "phases": " + ".join(inv["phases"]),
                 "phase": n}
            r.update({f"X({e})": inv["compositions"][n][e] for e in els})
            inv_rows.append(r)

    # region summary
    regions = {}
    for ln in lines:
        rg = regions.setdefault(ln["region"], {"phases": sorted(ln["phases"]), "lines": 0, "T_K": [np.inf, -np.inf],
                                               "composition_ranges": {}})
        rg["lines"] += 1
        for p in ln["points"]:
            rg["T_K"] = [min(rg["T_K"][0], p["T"]), max(rg["T_K"][1], p["T"])]
            for n, X in p["phases"]:
                cr = rg["composition_ranges"].setdefault(n, {e: [1.0, 0.0] for e in els[1:]})
                for e in els[1:]:
                    cr[e] = [min(cr[e][0], X[e]), max(cr[e][1], X[e])]
    reg_list = []
    for name, rg in sorted(regions.items()):
        reg_list.append({"region": name, "lines": rg["lines"],
                         "T_range_K": [round(rg["T_K"][0], 2), round(rg["T_K"][1], 2)],
                         "X_ranges": {n: {f"X({e})": [round(a, 5), round(b, 5)] for e, (a, b) in cr.items()}
                                      for n, cr in rg["composition_ranges"].items()}})
    if kind == "ternary":  # T is constant: drop T range
        for r in reg_list:
            r.pop("T_range_K")

    files["boundaries_csv"] = _write_csv(str(outdir / f"{stem}_boundaries.csv"), bound_rows)
    files["tielines_csv"] = _write_csv(str(outdir / f"{stem}_tielines.csv"), tie_rows)
    files["invariants_csv"] = _write_csv(
        str(outdir / f"{stem}_invariants.csv"), inv_rows,
        ["invariant", "T_K", "T_C", "type", "reaction_on_cooling", "phases", "phase"] + [f"X({e})" for e in els])
    found = set()
    for ln in lines:
        found |= set(ln["phases"])
    for inv in invs:
        found |= set(inv["phases"])
    return {"lines": lines, "invariants": invs, "regions": reg_list, "phases_found": found, "els": els, "U": U,
            "degenerate_lines_dropped": degenerate, "duplicate_lines_dropped": duplicates}


def _isopleth_data(strategy, db, els, axis_el, fixed, s_els, lab, U, outdir, stem, files) -> Dict:
    from pycalphad import variables as v
    import pycalphad.mapping.utils as map_utils
    balance = [e for e in els if e != axis_el and e not in fixed][0]

    def glob_x(xv):
        d = {**fixed, axis_el: xv}
        d[balance] = 1.0 - sum(d.values())
        return d

    lines, rows = [], []
    for li, z in enumerate(strategy.zpf_lines):
        if not z.points:
            continue
        mid = z.points[len(z.points) // 2]
        names_mid = lab.names(mid.stable_composition_sets)
        zero = names_mid[0]  # fixed (zero-fraction) composition set comes first
        region = " + ".join(sorted(names_mid))
        xs, ts, tks = [], [], []
        for pi, p in enumerate(z.points):
            T = _point_T(p)
            xv = _f(p.global_conditions[v.X(axis_el)])
            xs.append(U.comp(glob_x(xv), axis_el))
            ts.append(U.T(T))
            tks.append(T)
            row = {"line": li, "point": pi, "zero_phase": zero, "phases": region, "T_K": T, "T_C": T - KELVIN,
                   f"X({axis_el})": xv}
            for k, (n, cs) in enumerate(zip(lab.names(p.stable_composition_sets), p.stable_composition_sets)):
                X = _cs_X(cs, s_els)
                row[f"phase_{k + 1}"] = n
                row[f"NP_{k + 1}"] = float(cs.NP)
                row.update({f"X{k + 1}({e})": X[e] for e in els})
            rows.append(row)
        lines.append({"id": li, "zero_phase": zero, "region": region, "x": xs, "T": ts, "T_K": tks})

    invs = []
    inv_nodes = [n for n in strategy.node_queue.nodes
                 if map_utils.degrees_of_freedom(n, strategy.components, strategy.num_potential_condition) == 0]
    try:
        inv_xy = strategy.get_invariant_data(v.X(axis_el), v.T)
    except Exception:  # noqa: BLE001
        inv_xy = [None] * len(inv_nodes)
    for node, xy in zip(inv_nodes, inv_xy):
        names = lab.names(node.stable_composition_sets)
        T = _point_T(node)
        if any(abs(i["T_K"] - T) < 0.05 and sorted(i["phases"]) == sorted(names) for i in invs):
            continue
        seg = []
        if xy is not None:
            seg = sorted(float(np.squeeze(d.x)) for d in xy.data)
        invs.append({"T_K": round(T, 3), "T_C": round(T - KELVIN, 3), "phases": names,
                     "compositions": {n: {e: round(c, 6) for e, c in _cs_X(cs, s_els).items()}
                                      for n, cs in zip(names, node.stable_composition_sets)},
                     f"X({axis_el})_segment": [round(s, 6) for s in seg],
                     "_seg_plot": [U.comp(glob_x(s), axis_el) for s in seg]})
    invs.sort(key=lambda i: -i["T_K"])
    inv_rows = []
    for k, inv in enumerate(invs):
        for n in inv["phases"]:
            r = {"invariant": k, "T_K": inv["T_K"], "T_C": inv["T_C"], "phases": " + ".join(inv["phases"]),
                 "phase": n, f"X({axis_el})_segment": " ".join(f"{s:g}" for s in inv[f"X({axis_el})_segment"])}
            r.update({f"X({e})": inv["compositions"][n][e] for e in els})
            inv_rows.append(r)
    regions = {}
    for ln in lines:
        rg = regions.setdefault(ln["zero_phase"], {"lines": 0, "T_K": [np.inf, -np.inf]})
        rg["lines"] += 1
        tk = ln["T_K"]
        rg["T_K"] = [min(rg["T_K"][0], min(tk)), max(rg["T_K"][1], max(tk))]
    reg_list = [{"zero_phase_line": k, "lines": r["lines"],
                 "T_range_K": [round(r["T_K"][0], 2), round(r["T_K"][1], 2)]} for k, r in sorted(regions.items())]
    files["boundaries_csv"] = _write_csv(str(outdir / f"{stem}_zpf_lines.csv"), rows)
    files["invariants_csv"] = _write_csv(str(outdir / f"{stem}_invariants.csv"), inv_rows)
    found = {ln["zero_phase"] for ln in lines}
    for inv in invs:
        found |= set(inv["phases"])
    allx = [x for ln in lines for x in ln["x"]]
    xlim = [min(allx), max(allx)] if allx else None
    out_invs = [{k: val for k, val in i.items() if not k.startswith("_")} for i in invs]
    return {"lines": lines, "invariants": out_invs, "_invs_plot": invs, "regions": reg_list,
            "phases_found": found, "xlim": xlim, "U": U}


def _plot_tielines(ax, res, kind, tielines, label_nodes, colors, mu_el, style="color"):
    from matplotlib.collections import LineCollection
    U, els = res["U"], res["els"]

    def xy(X, T, mu):
        if kind == "binary":
            x = mu[mu_el] / 1000.0 if mu_el else U.comp(X, els[1])
            return x, U.T(T)
        return U.comp(X, els[1]), U.comp(X, els[2])

    allxy = []
    for ln in res["lines"]:
        for p in ln["points"]:
            for _, X in p["phases"]:
                allxy.append(xy(X, p["T"], p["mu"]))
    if not allxy:
        return
    allxy = np.array(allxy)
    rng = np.ptp(allxy, axis=0) * 1e-3
    if style == "shaded":       # aliceblue two-phase fields; black (binary) / grey (ternary) lines; diagram_style
        from .diagram_style import draw_shaded
        bounds = [[np.array([xy(p["phases"][k][1], p["T"], p["mu"]) for p in ln["points"]])
                   for k in range(len(ln["phases"]))] for ln in res["lines"]]
        invs = [np.array([xy(inv["compositions"][n], inv["T_K"], inv["MU"]) for n in inv["phases"]])
                for inv in res["invariants"]]
        draw_shaded(ax, bounds, invs, "potential" if mu_el else kind)
    for ln in (res["lines"] if style != "shaded" else []):
        for k, n in enumerate(ln["phases"]):
            pts = np.array([xy(p["phases"][k][1], p["T"], p["mu"]) for p in ln["points"]])
            c = phase_color(n, colors)
            if np.ptp(pts[:, 0]) < rng[0] and np.ptp(pts[:, 1]) < rng[1]:
                ax.scatter([pts[:, 0].mean()], [pts[:, 1].mean()], color=c, s=8, zorder=3)
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.3, solid_capstyle="butt", zorder=2,
                        ls=(0, (4, 4)) if (mu_el and k == 1) else "-")  # coinciding curves in MU space
        if tielines and int(tielines) > 0 and len(ln["phases"]) == 2:
            segs = [[xy(p["phases"][0][1], p["T"], p["mu"]), xy(p["phases"][1][1], p["T"], p["mu"])]
                    for p in ln["points"][::int(tielines)]]
            ax.add_collection(LineCollection(segs, colors=[TIELINE_COLOR], linewidths=0.5, zorder=1))
    for inv in res["invariants"]:
        pts = [xy(inv["compositions"][n], inv["T_K"], inv["MU"]) for n in inv["phases"]]
        loop = pts + [pts[0]]
        if style != "shaded":
            ax.plot([p[0] for p in loop], [p[1] for p in loop], color=INVARIANT_COLOR, lw=1.1, zorder=2.5)
        if label_nodes:
            for n, p in zip(inv["phases"], pts):
                ax.scatter([p[0]], [p[1]], color=phase_color(n, colors), s=14, zorder=3, edgecolors="k",
                           linewidths=0.3)
    if kind == "binary" and mu_el:  # MU -> -inf at infinite dilution: auto-range on non-dilute points
        sel = [xy(X, p["T"], p["mu"])[0] for ln in res["lines"] for p in ln["points"] for _, X in p["phases"]
               if 1e-4 < X[mu_el] and all(1e-4 < X2[mu_el] for _, X2 in p["phases"])]
        sel = np.array(sel if sel else allxy[:, 0])
        lo, hi = np.nanmin(sel), np.nanmax(sel)
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        ax.set_xlim(lo - pad, hi + pad)


def _shaded_line_labels(ax, res, kind, els, mu_el, phase_labels=None):
    """Isopleth: each ZPF line named by its zero-fraction phase; T-MU: each line by its two-phase equilibrium."""
    from .diagram_style import line_labels
    from .phase_names import format_phase, region_label
    U = res["U"]
    if kind == "isopleth":
        items = [(format_phase(ln["zero_phase"], els, phase_labels), np.column_stack([ln["x"], ln["T"]]))
                 for ln in res["lines"]]
    else:
        items = [(region_label(ln["phases"], els, phase_labels),
                  np.array([(p["mu"][mu_el] / 1000.0, U.T(p["T"])) for p in ln["points"]])) for ln in res["lines"]]
    line_labels(ax, items)


def _plot_isopleth(ax, res, colors, style="color"):
    U = res["U"]
    if style == "shaded":       # black ZPF lines and invariant segments (see diagram_style)
        from .diagram_style import draw_shaded
        draw_shaded(ax, [[np.column_stack([ln["x"], ln["T"]])] for ln in res["lines"]],
                    [np.array([[min(s), U.T(inv["T_K"])], [max(s), U.T(inv["T_K"])]])
                     for inv, s in ((i, i["_seg_plot"]) for i in res["_invs_plot"]) if len(s) >= 2], "isopleth")
        return
    for ln in res["lines"]:
        ax.plot(ln["x"], ln["T"], color=phase_color(ln["zero_phase"], colors), lw=1.3, solid_capstyle="butt")
    for inv in res["_invs_plot"]:
        s = inv["_seg_plot"]
        if len(s) >= 2:
            ax.plot([min(s), max(s)], [U.T(inv["T_K"])] * 2, color=INVARIANT_COLOR, lw=1.1, zorder=2.5)


def _map_text(summary: Dict, U: _Units) -> str:
    out = [f"{summary['kind']} {summary['system']}: {len(summary['phases_found'])} phases found "
           f"({', '.join(summary['phases_found'])}); mapping {summary['mapping']['seconds']} s, "
           f"{summary['mapping']['zpf_lines']} ZPF lines" +
           ("" if summary["mapping"]["finished"] else "  ** PARTIAL: time limit reached **")]
    invs = summary["invariants"]
    if summary["kind"] == "binary":
        els = summary["elements"]
        out.append(f"Invariant three-phase equilibria ({len(invs)}):")
        for inv in invs:
            comp = ", ".join(f"{n} X({els[1]})={inv['compositions'][n][els[1]]:.4f}"
                             for n in sorted(inv["phases"], key=lambda n: inv["compositions"][n][els[1]]))
            out.append(f"  {inv.get('type', '?')}: {inv.get('reaction_on_cooling', '')} at {inv['T_K']:.2f} K "
                       f"({inv['T_C']:.2f} °C); {comp}")
    elif summary["kind"] == "ternary":
        els = summary["elements"]
        out.append(f"Three-phase tie-triangles ({len(invs)}):")
        for inv in invs:
            comp = "; ".join(f"{n} (" + ", ".join(f"X({e})={inv['compositions'][n][e]:.4f}" for e in els[1:]) + ")"
                             for n in inv["phases"])
            out.append(f"  {' + '.join(inv['phases'])}: {comp}")
    else:
        out.append(f"Invariant (zero degrees of freedom) equilibria crossed by the section ({len(invs)}):")
        for inv in invs:
            seg = [k for k in inv if k.endswith("_segment")][0]
            out.append(f"  {' + '.join(inv['phases'])} at {inv['T_K']:.2f} K ({inv['T_C']:.2f} °C), "
                       f"{seg}={inv[seg]}")
    if summary.get("renamed_when_disordered"):
        out.append("Note: disordered states of ordered phases are reported as " + ", ".join(
            f"{k}->{val}" for k, val in summary["renamed_when_disordered"].items()) + " (rename_disordered=false to keep)")
    out.append("Phase regions / ZPF lines:")
    for r in summary["regions"][:40]:
        name = r.get("region") or f"ZPF line of {r.get('zero_phase_line')}"
        tr = f" T {r['T_range_K'][0]:.1f}-{r['T_range_K'][1]:.1f} K" if "T_range_K" in r else ""
        out.append(f"  {name}:{tr} ({r['lines']} line(s))")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# step_map
# ---------------------------------------------------------------------------
@pcache.cached
def step_map(tdb: str, elements: Sequence[str], vary: str, start: float, stop: float, out_png: str, *,
             step: Optional[float] = None, fixed: Optional[Dict[str, float]] = None, temperature_unit: str = "K",
             properties: Optional[Sequence[str]] = None, plot: Optional[Sequence[str]] = None,
             phases=None, exclude_phases=None, phase_colors=None, rename_disordered: bool = True,
             pressure: float = 101325.0, title: Optional[str] = None, time_limit_s: float = 300.0,
             dpi: int = 150) -> Dict:
    from pycalphad import variables as v
    from pycalphad.mapping import StepStrategy
    from pycalphad.property_framework import as_property
    from .calc import get_db

    db = get_db(tdb)
    els = _check_elements(db, elements)
    ph = _select_phases(db, els, phases, exclude_phases)
    tC = str(temperature_unit).upper().startswith("C")
    vk = vary.strip().upper()
    fixed = dict(fixed or {})
    conds = {v.P: float(pressure), v.N: 1}
    fx = {}
    for k, val in fixed.items():
        ku = k.strip().upper()
        if ku in ("T", "T_K"):
            conds[v.T] = _toK(val, temperature_unit)
            fx["T"] = conds[v.T]
        elif ku == "P":
            conds[v.P] = float(val)
        else:
            e = _el(ku)
            if e not in els:
                raise ValueError(f"fixed condition {k!r}: {e} not in {els}")
            conds[v.X(e)] = float(val)
            fx[f"X({e})"] = float(val)
    if vk in ("T", "T_K"):
        lo, hi = _toK(start, temperature_unit), _toK(stop, temperature_unit)
        conds[v.T] = (min(lo, hi), max(lo, hi), float(step or 5.0))
        axis, axis_name = v.T, "T_K"
    elif vk == "P":
        conds[v.P] = (min(start, stop), max(start, stop), float(step or (abs(stop - start) / 100)))
        axis, axis_name = v.P, "P_Pa"
    else:
        e = _el(vk)
        if e not in els:
            raise ValueError(f"vary must be T, P or X(EL) with EL in {els}")
        conds[v.X(e)] = (min(start, stop), max(start, stop), float(step or 0.005))
        axis, axis_name = v.X(e), f"X({e})"
    if v.T not in conds:
        raise ValueError("give a fixed temperature, e.g. fixed={'T': 1273}")
    nX = len([k for k in conds if isinstance(k, v.X)])
    if nX != len(els) - 1:
        raise ValueError(f"need mole fractions of all elements but one (balance) as fixed/varied conditions; "
                         f"elements={els}, got {[str(k) for k in conds if isinstance(k, v.X)]}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = StepStrategy(db, sorted(els) + ["VA"], ph, conds)
    run = _run_strategy(st, time_limit_s)
    lab = _Labeler(db, st.models, rename_disordered)
    s_els = [e for e in st.elements if e != "VA"]
    props = [p for p in (properties if properties is not None else ["GM", "HM", "SM", "CPM"])]
    prop_objs = {}
    for p in props:
        try:
            prop_objs[p] = as_property(p)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unknown property {p!r} ({e}); examples: GM, HM, SM, CPM, HM.T, MU(CR), "
                             f"X(FCC_A1,CR), NP(LIQUID), DOO") from e

    pts = []
    for z in st.zpf_lines:
        for p in z.points:
            pts.append(p)
    rows = []
    phases_seen = []
    for p in pts:
        names = lab.names(p.stable_composition_sets)
        axv = _f(p.global_conditions[axis])
        r = {axis_name: axv}
        if axis is not v.T:
            r["T_K"] = _point_T(p)
        r["T_C"] = _point_T(p) - KELVIN
        r["stable_phases"] = " + ".join(names)
        for n, cs in zip(names, p.stable_composition_sets):
            if n not in phases_seen:
                phases_seen.append(n)
            r[f"NP({n})"] = r.get(f"NP({n})", 0.0) + float(cs.NP)
            for e, x in _cs_X(cs, s_els).items():
                r[f"X({n},{e})"] = x
        for i, e in enumerate(s_els):
            r[f"MU({e})"] = float(p.chemical_potentials[i])
        for pn, po in prop_objs.items():
            try:
                r[pn] = _f(p.get_property(po))
            except Exception:  # noqa: BLE001
                r[pn] = float("nan")
        rows.append(r)
    rows.sort(key=lambda r: r[axis_name])
    phases_seen = sorted(phases_seen)
    for r in rows:
        for n in phases_seen:
            r.setdefault(f"NP({n})", 0.0)
    cols = [axis_name] + (["T_K"] if axis is not v.T else []) + ["T_C", "stable_phases"] + \
        [f"NP({n})" for n in phases_seen] + [f"X({n},{e})" for n in phases_seen for e in s_els] + \
        [f"MU({e})" for e in s_els] + list(prop_objs)
    outdir, stem = _siblings(out_png)
    csv_path = _write_csv(str(outdir / f"{stem}.csv"), rows, cols)

    # phase regions along the axis & transitions
    segs = []
    for z in st.zpf_lines:
        if not z.points:
            continue
        a = [_f(p.global_conditions[axis]) for p in z.points]
        names = tuple(sorted(lab.names(z.points[len(z.points) // 2].stable_composition_sets)))
        segs.append([min(a), max(a), names])
    segs.sort(key=lambda s: (s[0], s[1]))
    merged = []
    for s in segs:
        if merged and merged[-1][2] == s[2] and s[0] <= merged[-1][1] + 1e-6 * max(1.0, abs(s[0])):
            merged[-1][1] = max(merged[-1][1], s[1])
        elif s[1] - s[0] > 0 or not merged:
            merged.append(list(s))
    fmt_ax = (lambda a: f"{a:.2f} K ({a - KELVIN:.2f} °C)") if axis is v.T else (lambda a: f"{axis_name}={a:.5g}")
    regions = [{"from": round(a, 6), "to": round(b, 6), "phases": list(n)} for a, b, n in merged]
    transitions = []
    for r0, r1 in zip(merged, merged[1:]):
        gained = sorted(set(r1[2]) - set(r0[2]))
        lost = sorted(set(r0[2]) - set(r1[2]))
        transitions.append({"at": round(r1[0], 6), "below_or_left": list(r0[2]), "above_or_right": list(r1[2]),
                            "appears_on_increase": gained, "disappears_on_increase": lost})

    # plot
    panels = list(plot) if plot else ["NP"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 3.2 * len(panels) + 0.6), sharex=True, squeeze=False)
    xs = np.array([r[axis_name] for r in rows])
    xplot = xs - KELVIN if (axis is v.T and tC) else xs
    for ax, pan in zip(axes[:, 0], panels):
        pu = pan.strip()
        m = re.fullmatch(r"X\(\*\s*,\s*([A-Za-z]{1,2})\)", pu, re.I)
        if pu.upper() == "NP":
            for n in phases_seen:
                ax.plot(xplot, [r[f"NP({n})"] for r in rows], color=phase_color(n, phase_colors), lw=1.6, label=n)
            ax.set_ylabel("Phase fraction")
            ax.set_ylim(-0.02, 1.02)
            _legend(ax, phases_seen, phase_colors)
        elif m:
            e = m.group(1).upper()
            for n in phases_seen:
                yv = np.array([r.get(f"X({n},{e})", np.nan) for r in rows], dtype=float)
                yv[np.array([r[f"NP({n})"] for r in rows]) <= 0] = np.nan
                ax.plot(xplot, yv, color=phase_color(n, phase_colors), lw=1.6)
            ax.set_ylabel(f"X({_cap(e)}) in phase")
            _legend(ax, phases_seen, phase_colors)
        else:
            key = pu if pu in rows[0] else next((k for k in rows[0] if k.upper() == pu.upper()), None)
            if key is None:
                raise ValueError(f"plot panel {pan!r}: use NP, X(*,EL), a MU(EL) or one of properties {props}")
            ax.plot(xplot, [r[key] for r in rows], color="k", lw=1.5)
            ax.set_ylabel(key)
        ax.grid(alpha=0.25)
        for t in transitions:
            a = t["at"] - KELVIN if (axis is v.T and tC) else t["at"]
            ax.axvline(a, color="0.6", lw=0.6, ls=":")
    axes[-1, 0].set_xlabel(("Temperature (°C)" if tC else "Temperature (K)") if axis is v.T else
                           ("Pressure (Pa)" if axis is v.P else f"Mole fraction {_cap(axis_name[2:-1])}"))
    comp_s = ", ".join(f"{k}={val:g}" for k, val in fx.items() if k != "T")
    if "T" in fx:
        comp_s += (", " if comp_s else "") + (f"T={fx['T'] - KELVIN:g} °C" if tC else f"T={fx['T']:g} K")
    axes[0, 0].set_title(title or f"{'-'.join(_cap(e) for e in els)}  {comp_s}" +
                         ("" if run["finished"] else "  [PARTIAL]"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    lines = [f"step {axis_name} {start:g}->{stop:g} ({'-'.join(els)}, {comp_s}); {len(rows)} points, "
             f"{run['seconds']} s" + ("" if run["finished"] else " ** PARTIAL (time limit) **")]
    lines.append("Phase regions:")
    for r in regions:
        lines.append(f"  {fmt_ax(r['from'])} .. {fmt_ax(r['to'])}: {' + '.join(r['phases'])}")
    lines.append("Transitions:")
    for t in transitions:
        ch = []
        if t["appears_on_increase"]:
            ch.append(f"+{','.join(t['appears_on_increase'])}")
        if t["disappears_on_increase"]:
            ch.append(f"-{','.join(t['disappears_on_increase'])}")
        lines.append(f"  at {fmt_ax(t['at'])}: {' + '.join(t['below_or_left'])} -> {' + '.join(t['above_or_right'])}"
                     f" ({'on heating' if axis is v.T else 'on increasing ' + axis_name}: {' '.join(ch)})")
    return {"png": out_png, "csv": csv_path, "regions": regions, "transitions": transitions,
            "phases": phases_seen, "columns": cols, "mapping": run, "text": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Scheil
# ---------------------------------------------------------------------------
def _liquidus_start(db, comps, phases, comp_conds, liquid, tmax=4000.0) -> float:
    """Lowest temperature on a coarse grid above which the alloy is fully liquid (+ margin)."""
    from pycalphad import equilibrium, variables as v
    T = np.arange(600.0, tmax + 1, 25.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, comps, phases, {v.T: T, v.P: 101325, v.N: 1, **comp_conds})
    P = np.squeeze(eq.Phase.values)
    NP = np.squeeze(eq.NP.values)
    liq = np.nansum(np.where(P == liquid, NP, 0.0), axis=-1)
    full = liq > 1 - 1e-8
    if not full[-1]:
        raise ValueError(f"alloy is not fully {liquid} even at {tmax} K; give start_temperature")
    idx = len(full) - 1
    while idx > 0 and full[idx - 1]:
        idx -= 1
    return float(T[idx] + 30.0)


@pcache.cached
def scheil_solidification(tdb: str, elements: Sequence[str], composition: Dict[str, float], out_png: str, *,
                          start_temperature: Optional[float] = None, temperature_unit: str = "K",
                          step_temperature: float = 1.0, include_equilibrium: bool = True,
                          liquid_phase: Optional[str] = None, phases=None, exclude_phases=None,
                          stop: float = 1e-4, phase_colors=None, title: Optional[str] = None,
                          dpi: int = 150) -> Dict:
    try:
        from scheil import simulate_equilibrium_solidification, simulate_scheil_solidification
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("the 'scheil' package is not installed (pip install scheil)") from e
    from pycalphad import variables as v
    from .calc import get_db

    db = get_db(tdb)
    comp = {_el(k): float(x) for k, x in composition.items()}
    els = _check_elements(db, sorted(set(_el(e) for e in elements) | set(comp)))
    if len(comp) != len(els) - 1:
        raise ValueError(f"give mole fractions of all elements except the balance one; elements={els}, got {comp}")
    ph = _select_phases(db, els, phases, exclude_phases)
    liq = (liquid_phase or next((p for p in ph if _is_liquid(db, p)), "")).upper()
    if liq not in ph:
        raise ValueError(f"no liquid phase among {ph}; give liquid_phase")
    comps = sorted(els) + ["VA"]
    cc = {v.X(k): val for k, val in comp.items()}
    t0 = time.time()
    T0 = _toK(start_temperature, temperature_unit) if start_temperature is not None else \
        _liquidus_start(db, comps, ph, cc, liq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sch = simulate_scheil_solidification(db, comps, ph, dict(cc), T0, step_temperature=step_temperature,
                                             liquid_phase_name=liq, stop=stop)
        eqs = simulate_equilibrium_solidification(db, comps, ph, dict(cc), T0, step_temperature=step_temperature,
                                                  liquid_phase_name=liq) if include_equilibrium else None
    tC = str(temperature_unit).upper().startswith("C")
    outdir, stem = _siblings(out_png)
    files = {"png": out_png}
    for res in (sch, eqs):  # scheil names disordered states A2_BCC/A1_FCC; use BCC_A2/FCC_A1 if those exist
        if res is None:
            continue
        present = {p for p, a in res.cum_phase_amounts.items() if a and a[-1] > 0}
        ren = {}
        for pname in list(res.phase_amounts):
            m = re.fullmatch(r"([A-Z]\d+)_([A-Z]+)", pname)
            alt = f"{m.group(2)}_{m.group(1)}" if m else None
            if alt and alt in db.phases and alt not in present and alt not in ren.values():
                ren[pname] = alt
        for old_n, new_n in ren.items():
            for d in (res.phase_amounts, res.cum_phase_amounts, res.phase_compositions):
                if old_n in d:
                    d[new_n] = d.pop(old_n)

    def df_out(res, name):
        df = res.to_dataframe(include_zero_phases=False)
        df.insert(1, "Temperature (C)", df["Temperature (K)"] - KELVIN)
        df.insert(2, "fraction_solid", res.fraction_solid)
        path = str(outdir / f"{stem}_{name}.csv")
        df.to_csv(path, index=False, float_format="%.8g")
        files[f"{name}_csv"] = path

    df_out(sch, "scheil")
    if eqs is not None:
        df_out(eqs, "equilibrium")

    def summarize(res) -> Dict:
        T = np.asarray(res.temperatures, dtype=float)
        fs = np.asarray(res.fraction_solid, dtype=float)
        first = int(np.argmax(fs > 1e-10)) if np.any(fs > 1e-10) else None
        d = {"converged": bool(res.converged), "start_T_K": float(T[0]), "end_T_K": float(T[-1]),
             "first_solid_T_K": float(T[first]) if first is not None else None,
             "final_fraction_solid_before_last_step": float(fs[-2]) if len(fs) > 1 else float(fs[-1])}
        for q in (0.5, 0.9, 0.99):
            i = np.argmax(fs >= q) if np.any(fs >= q) else None
            d[f"T_at_fs_{q}"] = float(T[i]) if i is not None else None
        order = []
        for pname, amts in res.phase_amounts.items():
            a = np.asarray(amts, dtype=float)
            if np.any(a > 1e-8):
                order.append((float(T[int(np.argmax(a > 1e-8))]), pname, float(np.sum(a))))
        order.sort(key=lambda t: -t[0])
        d["phases_forming"] = [{"phase": p, "first_T_K": round(t, 2), "total_amount": round(a, 5)}
                               for t, p, a in order]
        return d

    s_sum = summarize(sch)
    e_sum = summarize(eqs) if eqs is not None else None

    # plot: T vs fraction solid + cumulative Scheil phase amounts
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    tconv = (lambda arr: np.asarray(arr) - KELVIN) if tC else (lambda arr: np.asarray(arr))
    a1.plot(sch.fraction_solid, tconv(sch.temperatures), color="#1f77b4", lw=1.8, label="Scheil-Gulliver")
    if eqs is not None:
        a1.plot(eqs.fraction_solid, tconv(eqs.temperatures), color="#ff7f0e", lw=1.8, ls="--", label="Equilibrium")
    a1.set_xlabel("Fraction solid")
    a1.set_ylabel("Temperature (°C)" if tC else "Temperature (K)")
    a1.set_xlim(0, 1)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=9)
    shown = []
    for pname, cum in sorted(sch.cum_phase_amounts.items()):
        if cum[-1] > 1e-6:
            a2.plot(tconv(sch.temperatures), cum, color=phase_color(pname, phase_colors), lw=1.6, label=pname)
            shown.append(pname)
    a2.plot(tconv(sch.temperatures), sch.fraction_liquid, color=phase_color(liq, phase_colors), lw=1.6, ls=":",
            label=f"{liq} (remaining)")
    a2.set_xlabel("Temperature (°C)" if tC else "Temperature (K)")
    a2.set_ylabel("Cumulative phase amount (Scheil)")
    a2.set_ylim(-0.02, 1.02)
    a2.invert_xaxis()
    a2.grid(alpha=0.25)
    a2.legend(fontsize=8)
    comp_s = ", ".join(f"X({k})={val:g}" for k, val in comp.items())
    fig.suptitle(title or f"{'-'.join(_cap(e) for e in els)} solidification, {comp_s}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    def fmtT(t):
        return "n/a" if t is None else f"{t:.1f} K ({t - KELVIN:.1f} °C)"

    txt = [f"Solidification of {'-'.join(els)} {comp_s} from {fmtT(T0)} (step {step_temperature:g} K), "
           f"{time.time() - t0:.1f} s"]
    txt.append(f"Scheil: first solid at {fmtT(s_sum['first_solid_T_K'])}; fs=0.9 at {fmtT(s_sum['T_at_fs_0.9'])}; "
               f"ends at {fmtT(s_sum['end_T_K'])}" +
               ("" if s_sum["converged"] else
                f" (stopping criterion not reached: remaining liquid "
                f"{1 - s_sum['final_fraction_solid_before_last_step']:.3f} solidified at the last step, "
                f"typically an invariant/eutectic reaction)"))
    txt.append("  phases (order of formation): " + ", ".join(
        f"{p['phase']} from {p['first_T_K']:.1f} K (total {p['total_amount']:.3f})" for p in s_sum["phases_forming"]))
    if e_sum:
        txt.append(f"Equilibrium: liquidus {fmtT(e_sum['first_solid_T_K'])}, solidus {fmtT(e_sum['end_T_K'])}, "
                   f"freezing range {(e_sum['first_solid_T_K'] or 0) - e_sum['end_T_K']:.1f} K" +
                   ("" if e_sum["converged"] else " (did not fully solidify)"))
        txt.append("  phases: " + ", ".join(f"{p['phase']} from {p['first_T_K']:.1f} K"
                                            for p in e_sum["phases_forming"]))
        if s_sum["end_T_K"] is not None:
            txt.append(f"Scheil freezing range {(s_sum['first_solid_T_K'] or 0) - s_sum['end_T_K']:.1f} K "
                       f"vs equilibrium {(e_sum['first_solid_T_K'] or 0) - e_sum['end_T_K']:.1f} K")
    return {"files": files, "scheil": s_sum, "equilibrium": e_sum, "liquid_phase": liq, "start_T_K": T0,
            "text": "\n".join(txt)}


# ---------------------------------------------------------------------------
# Database subset / rewrite
# ---------------------------------------------------------------------------
def subset_database(db, elements: Optional[Sequence[str]] = None, phases: Optional[Sequence[str]] = None,
                    exclude_phases: Optional[Sequence[str]] = None):
    """Return (new Database restricted to elements/phases, info dict).  The input is not modified."""
    from pycalphad import Database
    from pycalphad.io.database import Phase

    all_el = {str(e) for e in db.elements}
    keep_el = all_el if not elements else ({_el(e) for e in elements} | ({"VA", "/-"} & all_el))
    bad = sorted({_el(e) for e in (elements or [])} - all_el)
    if bad:
        raise ValueError(f"elements {bad} not in database; available: {sorted(all_el - {'VA', '/-'})}")
    keep_sp = {sp for sp in db.species if set(sp.constituents) <= keep_el}
    keep_names = {sp.name for sp in keep_sp}
    want = {p.upper() for p in phases} if phases else None
    if want:
        badp = sorted(want - set(db.phases))
        if badp:
            raise ValueError(f"phases {badp} not in database; available: {sorted(db.phases)}")
    excl = {p.upper() for p in (exclude_phases or [])}

    new_phases, dropped = {}, {}
    for name, ph in db.phases.items():
        if (want and name not in want) or name in excl:
            dropped[name] = "not selected"
            continue
        cons = tuple(frozenset(s for s in subl if s.name in keep_names) for s_i, subl in enumerate(ph.constituents))
        if any(len(c) == 0 for c in cons):
            dropped[name] = "a sublattice has no constituent in the element subset"
            continue
        if all({s.name for s in c} <= {"VA"} for c in cons):
            dropped[name] = "only vacancies left"
            continue
        np_ = Phase()
        np_.name = ph.name
        np_.sublattices = tuple(ph.sublattices)
        np_.model_hints = copy.deepcopy(ph.model_hints)
        np_.constituents = cons
        new_phases[name] = np_
    # order/disorder partners must both be present
    for name in list(new_phases):
        h = new_phases[name].model_hints
        o, d = h.get("ordered_phase"), h.get("disordered_phase")
        if o and d:
            if o == name and d not in new_phases:
                if d in db.phases and d not in excl:
                    dp = db.phases[d]
                    cons = tuple(frozenset(s for s in subl if s.name in keep_names) for subl in dp.constituents)
                    if all(len(c) for c in cons):
                        np_ = Phase()
                        np_.name, np_.sublattices = dp.name, tuple(dp.sublattices)
                        np_.model_hints, np_.constituents = copy.deepcopy(dp.model_hints), cons
                        new_phases[d] = np_
                        dropped.pop(d, None)
                        continue
                del new_phases[name]
                dropped[name] = f"its disordered part {d} is unavailable"
            elif d == name and o not in new_phases:
                h.pop("ordered_phase", None)
                h.pop("disordered_phase", None)

    def sp_ok(s):
        return s.name in keep_names or s.name == "*"

    params = []
    for p in db._parameters.all():
        if p["phase_name"] not in new_phases:
            continue
        if not all(sp_ok(s) for subl in p["constituent_array"] for s in subl):
            continue
        ds = p.get("diffusing_species")
        if ds is not None and getattr(ds, "name", None) not in (None, "") and ds.name not in keep_names:
            continue
        params.append(dict(p))

    # functions needed (transitively)
    needed, stack = set(), []
    for p in params:
        stack += [str(s) for s in getattr(p["parameter"], "free_symbols", ())]
    while stack:
        s = stack.pop()
        if s in db.symbols and s not in needed:
            needed.add(s)
            stack += [str(x) for x in getattr(db.symbols[s], "free_symbols", ())]

    new = Database()
    new.elements = set(keep_el)
    new.species = set(keep_sp)
    new.phases = new_phases
    new.refstates = {k: copy.deepcopy(val) for k, val in db.refstates.items() if k in keep_el}
    new.symbols = {k: val for k, val in db.symbols.items() if k in needed}
    new.references = copy.deepcopy(getattr(db, "references", {}))
    new._structure_dict = {k: val for k, val in getattr(db, "_structure_dict", {}).items() if k in new_phases}
    if params:
        new._parameters.insert_multiple(params)
    info = {"elements": sorted(keep_el - {"VA", "/-"}), "phases": sorted(new_phases), "dropped_phases": dropped,
            "parameters": len(params), "parameters_original": len(db._parameters),
            "functions": len(new.symbols), "functions_original": len(db.symbols)}
    return new, info


def _model_check(db_a, db_b, phases, els, n=3, seed=0) -> Dict[str, float]:
    """Max |GM_a - GM_b| (J/mol) at random states for each phase (symbolic models, no compilation)."""
    from pycalphad import Model, variables as v
    rng = np.random.default_rng(seed)
    comps = sorted(els) + ["VA"]
    out = {}
    for ph in phases:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ma, mb = Model(db_a, comps, ph), Model(db_b, comps, ph)
        except Exception:  # noqa: BLE001 - phase not buildable in this subsystem
            continue
        worst = 0.0
        for _ in range(n):
            sub = {v.T: float(rng.uniform(400, 2500)), v.P: 101325.0}
            by_sub = {}
            for y in ma.site_fractions:
                by_sub.setdefault(y.sublattice_index, []).append(y)
            for ys in by_sub.values():
                w = rng.dirichlet(np.ones(len(ys)))
                sub.update({y: float(wi) for y, wi in zip(ys, w)})
            ga, gb = float(ma.GM.xreplace(sub)), float(mb.GM.xreplace(sub))
            worst = max(worst, abs(ga - gb))
        out[ph] = worst
    return out


def write_database(tdb: str, out_path: str, *, elements=None, phases=None, exclude_phases=None,
                   groupby: str = "subsystem", fix_incompatible: bool = True, dialect: Optional[str] = None) -> Dict:
    from pycalphad import Database
    from .calc import get_db
    db = get_db(tdb)
    if elements or phases or exclude_phases:
        new, info = subset_database(db, elements, phases, exclude_phases)
    else:
        new, info = db, {"elements": sorted(str(e) for e in db.elements if str(e) not in ("VA", "/-")),
                         "phases": sorted(db.phases), "dropped_phases": {}, "parameters": len(db._parameters),
                         "parameters_original": len(db._parameters), "functions": len(db.symbols),
                         "functions_original": len(db.symbols)}
    if dialect:     # thermocalc / opencalphad / pycalphad rules, consistent renames, verified (dbformats)
        from .dbformats import export_verified
        info.update(export_verified(db, new, out_path, dialect, groupby), writer_warnings=[])
        return info
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        new.to_file(out_path, fmt="tdb", if_exists="overwrite", groupby=groupby,
                    if_incompatible="fix" if fix_incompatible else "warn")
    wmsgs = sorted({str(w.message)[:300] for w in wlist})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = Database(out_path)
    info["reloaded"] = {"elements": sorted(str(e) for e in back.elements), "phases": len(back.phases),
                        "parameters": len(back._parameters), "functions": len(back.symbols)}
    els = info["elements"]
    check_phases = sorted(back.phases)
    try:
        diffs = _model_check(db, back, check_phases, els)
        info["model_check_max_abs_dGM_J_per_mol"] = {k: round(val, 6) for k, val in diffs.items()}
    except Exception as e:  # noqa: BLE001
        info["model_check_error"] = f"{type(e).__name__}: {e}"
    info["writer_warnings"] = wmsgs
    info["path"] = out_path
    return info


# ---------------------------------------------------------------------------
# Model expression export
# ---------------------------------------------------------------------------
_CONTRIB_NAMES = {"ref": "reference (surface of reference / endmember) energy", "idmix": "ideal mixing energy",
                  "xsmix": "excess mixing energy (Redlich-Kister / Muggianu)", "mag": "magnetic energy",
                  "2st": "two-state energy", "ein": "Einstein energy", "vol": "volume (pressure) energy",
                  "ord": "atomic ordering energy (partitioned order/disorder model)"}


def _readable(expr, site_fractions):
    """Replace Y(PHASE,s,EL) by symbols named y<s>_<EL> for readability."""
    import symengine
    rep = {y: symengine.Symbol(f"y{y.sublattice_index}_{y.species.name}") for y in site_fractions}
    return expr.xreplace(rep) if rep else expr


def _latex_ready(expr, site_fractions):
    """Site fractions -> y^{(s)}_{EL}, other named symbols (functions) -> \\mathrm{NAME} for LaTeX output."""
    import symengine
    rep = {y: symengine.Symbol(f"y^{{({y.sublattice_index})}}_{{\\mathrm{{{y.species.name}}}}}") for y in site_fractions}
    for sym in expr.free_symbols:
        if sym not in rep and type(sym) is symengine.Symbol and len(str(sym)) > 1:
            rep[sym] = symengine.Symbol("\\mathrm{" + str(sym).replace("_", "\\_") + "}")
    return expr.xreplace(rep)


def _param_str(p) -> str:
    cons = ":".join(",".join(sorted(s.name for s in subl)) for subl in p["constituent_array"])
    return f"{p['parameter_type']}({p['phase_name']},{cons};{p['parameter_order']})"


def export_phase_model(tdb: str, phase: str, out_prefix: str, *, elements=None, expand_functions: bool = False,
                       evaluate_at: Optional[Dict[str, float]] = None, latex: bool = False,
                       max_chars: int = 1500) -> Dict:
    import symengine
    from pycalphad import Model, variables as v
    from .calc import get_db

    db = get_db(tdb)
    phase = phase.upper()
    if phase not in db.phases:
        raise ValueError(f"phase {phase} not in database; available: {sorted(db.phases)}")
    ph = db.phases[phase]
    phase_els = sorted({el for subl in ph.constituents for s in subl for el in s.constituents} - {"VA", "/-"})
    els = _check_elements(db, elements) if elements else phase_els
    comps = sorted(set(els)) + ["VA"]
    keep_symbolic = [] if expand_functions else list(db.symbols.keys())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = Model(db, comps, phase, parameters=keep_symbolic)
    sfs = mod.site_fractions
    contribs = {k: e for k, e in mod.models.items() if str(e) not in ("0", "0.0")}
    props = {}
    for pn in ("TC", "BMAG", "NT"):
        try:
            e = getattr(mod, pn)
            if str(e) not in ("0", "0.0"):
                props[pn] = e
        except Exception:  # noqa: BLE001
            pass

    # functions referenced
    used, stack = [], [str(s) for s in mod.ast.free_symbols] + [str(s) for e in props.values() for s in e.free_symbols]
    while stack and not expand_functions:
        s = stack.pop()
        if s in db.symbols and s not in used:
            used.append(s)
            stack += [str(x) for x in db.symbols[s].free_symbols]
    used.sort()

    # parameters of this phase (and its disordered part) within the chosen elements
    names = {phase}
    if ph.model_hints.get("ordered_phase") == phase and ph.model_hints.get("disordered_phase"):
        names.add(ph.model_hints["disordered_phase"])
    okset = set(comps) | {"*"}
    prow = []
    for p in db._parameters.all():
        if p["phase_name"] not in names:
            continue
        if not all(s.name in okset or set(s.constituents) <= set(comps) for subl in p["constituent_array"] for s in subl):
            continue
        prow.append({"parameter": _param_str(p), "phase": p["phase_name"], "type": p["parameter_type"],
                     "constituents": ":".join(",".join(sorted(s.name for s in subl)) for subl in p["constituent_array"]),
                     "order": p["parameter_order"],
                     "part": "disordered" if p["phase_name"] != phase else "",
                     "expression": str(p["parameter"]), "reference": p.get("reference") or ""})
    prow.sort(key=lambda r: (r["phase"], r["type"], r["constituents"].count(",") + r["constituents"].count(":"),
                             r["constituents"], r["order"]))

    # evaluation
    evaluation = None
    if evaluate_at:
        sub = {v.P: 101325.0}
        ev = {str(k).strip().upper(): float(val) for k, val in evaluate_at.items()}
        if "T" not in ev:
            raise ValueError("evaluate_at needs T (K)")
        sub[v.T] = ev.pop("T")
        if "P" in ev:
            sub[v.P] = ev.pop("P")
        xs = {}
        ymap = {str(y).upper(): y for y in sfs}
        ymap.update({f"Y{y.sublattice_index}_{y.species.name}".upper(): y for y in sfs})
        for k, val in ev.items():
            if k in ymap:
                sub[ymap[k]] = val
            elif re.fullmatch(r"X\([A-Z]{1,2}\)", k):
                xs[_el(k)] = val
            else:
                raise ValueError(f"evaluate_at key {k!r}: use T, P, X(EL) or site fractions {[str(y) for y in sfs]}")
        if xs:
            by_sub = {}
            for y in sfs:
                by_sub.setdefault(y.sublattice_index, []).append(y)
            for s, ys in by_sub.items():
                tot = sum(xs.get(y.species.name, 0.0) for y in ys if y.species.name != "VA")
                for y in ys:
                    if y in sub:
                        continue
                    if y.species.name == "VA":
                        sub[y] = 0.0 if tot > 0 else 1.0
                    else:
                        sub[y] = xs.get(y.species.name, 0.0) / tot if tot > 0 else 0.0
        missing = [str(y) for y in sfs if y not in sub]
        if missing:
            raise ValueError(f"missing site fractions {missing} (give X(EL) values or the site fractions)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = Model(db, comps, phase)
        evaluation = {"state": {str(k): val for k, val in sub.items()}}
        for k, e in list(full.models.items()) + [("GM", full.GM)] + \
                [(pn, getattr(full, pn)) for pn in ("TC", "BMAG") if pn in props]:
            try:
                evaluation[k] = float(e.xreplace(sub))
            except Exception as ex:  # noqa: BLE001
                evaluation[k] = f"could not evaluate ({type(ex).__name__})"

    # write files
    out_txt = out_prefix + ".txt"
    L = [f"Phase {phase} from {Path(tdb).name}; components {', '.join(comps)}",
         f"Sublattice site ratios: {list(ph.sublattices)}",
         "Constituents (active): " + " : ".join(",".join(sorted(s.name for s in c)) for c in mod.constituents),
         f"Model hints: {ph.model_hints}",
         "Site fractions: " + ", ".join(f"y{y.sublattice_index}_{y.species.name} = {y}" for y in sfs),
         "Energies are per mole of formula unit normalised to one mole of atoms (J/mol); GM = sum of contributions.",
         "Functions are kept symbolic (see FUNCTIONS below)" if not expand_functions else "Functions expanded.", ""]
    for k, e in contribs.items():
        L += [f"=== {k}: {_CONTRIB_NAMES.get(k, k)}", str(_readable(e, sfs)), ""]
    for k, e in props.items():
        L += [f"=== physical property {k}", str(_readable(e, sfs)), ""]
    if used:
        L.append("=== FUNCTIONS")
        for s in used:
            L.append(f"{s} = {db.symbols[s]}")
        L.append("")
    if evaluation:
        L.append("=== evaluation (all functions expanded)")
        L += [f"{k} = {val}" for k, val in evaluation.items()]
    Path(out_txt).write_text("\n".join(L))
    files = {"txt": out_txt}
    csv_path = _write_csv(out_prefix + "_parameters.csv", prow,
                          ["parameter", "phase", "part", "type", "constituents", "order", "expression", "reference"])
    files["parameters_csv"] = csv_path
    if latex:
        tex = [f"% Gibbs energy contributions of {phase} ({', '.join(comps)}); y<s>_<EL> = site fraction"]
        for k, e in contribs.items():
            try:
                tex.append(f"% {k}: {_CONTRIB_NAMES.get(k, k)}\n\\begin{{dmath}}\nG^{{\\mathrm{{{k}}}}} = "
                           f"{symengine.latex(_latex_ready(e, sfs))}\n\\end{{dmath}}")
            except Exception as ex:  # noqa: BLE001
                tex.append(f"% {k}: LaTeX printing failed ({ex})")
        Path(out_prefix + ".tex").write_text("\n\n".join(tex) + "\n")
        files["tex"] = out_prefix + ".tex"

    def short(e):
        s = str(_readable(e, sfs))
        return s if len(s) <= max_chars else s[:max_chars] + f" ...[{len(s)} chars, full text in file]"

    txt = [f"{phase} ({', '.join(comps)}): site ratios {list(ph.sublattices)}; constituents "
           + " : ".join(",".join(sorted(s.name for s in c)) for c in mod.constituents),
           "site fractions: " + ", ".join(f"y{y.sublattice_index}_{y.species.name}" for y in sfs),
           f"non-zero contributions: {', '.join(contribs)}"]
    for k, e in contribs.items():
        txt.append(f"[{k}] {_CONTRIB_NAMES.get(k, k)}:\n  {short(e)}")
    for k, e in props.items():
        txt.append(f"[{k}]:\n  {short(e)}")
    txt.append(f"{len(prow)} parameters (see CSV):")
    txt += [f"  {r['parameter']} = {r['expression'][:160]}" for r in prow[:40]]
    if used:
        txt.append(f"functions used: {', '.join(used)}")
    if evaluation:
        txt.append("evaluation: " + json.dumps({k: (round(val, 4) if isinstance(val, float) else val)
                                                 for k, val in evaluation.items()}))
    return {"files": files, "contributions": list(contribs), "n_parameters": len(prow), "functions": used,
            "evaluation": evaluation, "text": "\n".join(txt)}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
def register(tb) -> None:  # noqa: C901 - flat list of tool definitions
    S = {"type": "string"}
    N = {"type": "number"}
    B = {"type": "boolean"}
    STRS = {"type": "array", "items": {"type": "string"}}
    NUMS2 = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    UNIT = {"type": "string", "enum": ["K", "C"]}
    FRACS = {"type": "object", "additionalProperties": {"type": "number"}}
    COLORS = {"type": "object", "additionalProperties": {"type": "string"},
              "description": "optional phase -> colour overrides; colours are otherwise fixed per phase name "
                             "(consistent across all plots)"}

    STYLE = STYLE_SCHEMA

    def rel_files(files: Dict[str, str]) -> str:
        return "\n".join(f"  {k}: {tb.rel(p)}" for k, p in files.items())

    @tb.add("map_phase_diagram",
            "Phase diagram via pycalphad mapping with DATA export. Kind is inferred: 2 elements + tmin/tmax -> binary "
            "T-x; 3 elements + temperature -> ternary isotherm; elements + vary + fixed + tmin/tmax -> isopleth. "
            "Writes PNG + CSVs (phase boundaries per region, tie-line endpoints, invariants) + JSON and returns a "
            "summary of invariant reactions (eutectic/peritectic/... with T and phase compositions) and phase "
            "regions. Orientation: right (binary: element on the right, x = its fraction) or corners (ternary); "
            "else element order: binary [left, right], ternary [bottom-left, bottom-right, top].",
            {"tdb": S, "elements": STRS, "tmin": N, "tmax": N,
             "right": {**S, "description": "binary: element on the right (x axis = its fraction); isopleth: = vary"},
             "corners": {"type": "object", "properties": {"left": S, "right": S, "top": S},
                         "description": "ternary: element at bottom-left, bottom-right, top"},
             "temperature": {**N, "description": "ternary isotherm temperature"},
             "temperature_unit": UNIT,
             "vary": {**S, "description": "isopleth x-axis element"},
             "fixed": {**FRACS, "description": "isopleth fixed mole fractions, e.g. {\"NI\": 0.1}"},
             "x_range": {**NUMS2, "description": "composition range of the x axis (mole fraction), default [0,1]"},
             "composition_step": {**N, "description": "mapping step in mole fraction (default 0.02)"},
             "temperature_step": {**N, "description": "mapping step in K (default 10)"},
             "composition_unit": {"type": "string",
                                  "enum": ["mole_fraction", "mole_percent", "weight_fraction", "weight_percent"]},
             "potential_axis": {**S, "description": "binary only: element whose chemical potential MU is the "
                                                    "x axis (T-MU diagram)"},
             "tielines": {"type": "integer", "description": "draw every n-th tie-line (default 3; 0 = none)"},
             "label_nodes": {**B, "description": "mark phase compositions of invariants"},
             "xlim": NUMS2, "ylim": {**NUMS2, "description": "plot limits in display units"},
             "phases": STRS, "exclude_phases": STRS, "phase_colors": COLORS, "title": S,
             "rename_disordered": {**B, "description": "report ordered phases that are disordered under the "
                                                       "disordered-part name, e.g. B2_BCC->A2_BCC (default true)"},
             "pressure": {**N, "description": "Pa, default 101325"},
             "time_limit_s": {**N, "description": "stop mapping after this time (partial result), default 600"},
             "label_regions": {**B, "description": "write the phases of each region / tie-triangle (e.g. L+FCC, "
                                                   "BCC+σ) inside it, stoichiometric phases (Al₃Ni) outside the "
                                                   "axes and binary invariant temperatures by their line, clear of "
                                                   "data; binary or ternary isotherm; default on when shaded"},
             "label_invariants": {**B, "description": "with label_regions (default true)"},
             "phase_labels": {"type": "object", "additionalProperties": {"type": "string"},
                              "description": "label text overrides, e.g. {\"SIGMA\": \"σ-phase\"}"},
             "label_style": {"type": "object", "description": "fontsize, min_fontsize (default 8; labels that "
                                                              "do not fit are left out), color, background, rotate"},
             "formats": {**STRS, "description": "extra files svg/pdf/eps with editable text (default svg+pdf "
                                                 "when shaded or labelling)"},
             "style": STYLE,
             "output_prefix": {**S, "description": "file name stem for outputs"}},
            ["tdb", "elements"])
    def _map(tdb, elements, tmin=None, tmax=None, temperature=None, temperature_unit="K", vary=None, fixed=None,
             x_range=None, composition_step=None, temperature_step=10.0, composition_unit="mole_fraction",
             potential_axis=None, tielines=3, label_nodes=False, xlim=None, ylim=None, phases=None,
             exclude_phases=None, phase_colors=None, title=None, rename_disordered=True, pressure=101325.0,
             time_limit_s=600.0, output_prefix=None, label_regions=None, label_invariants=True, phase_labels=None,
             label_style=None, formats=None, right=None, corners=None, style=None):
        from .orientation import binary_order, ternary_order
        t = tb._find_tdb(tdb)
        els = [_el(e) for e in elements]
        if not (fixed or vary):             # file names follow the orientation (Cr-Fe and Fe-Cr do not collide)
            if len(els) == 2 and not corners:
                els = binary_order(els, right)
            elif len(els) == 3 and temperature is not None:
                els = ternary_order(els, corners, right)
        tag = "-".join(els)
        if fixed or vary:
            tag += f"_isopleth_{_el(vary or right or '')}_" + "_".join(f"{_el(k)}{val:g}"
                                                                     for k, val in (fixed or {}).items())
        elif temperature is not None and len(els) == 3:
            tag += f"_{temperature:g}{temperature_unit}"
        out = tb.outp((output_prefix + ".png") if output_prefix else None, f"map_{tag}_{Path(t).stem}.png")
        r = map_phase_diagram(t, elements, out, tmin=tmin, tmax=tmax, temperature=temperature,
                              temperature_unit=temperature_unit, vary=vary, fixed=fixed, x_range=x_range,
                              composition_step=composition_step, temperature_step=temperature_step,
                              composition_unit=composition_unit, potential_axis=potential_axis, tielines=tielines,
                              label_nodes=label_nodes, xlim=xlim, ylim=ylim, phases=phases,
                              exclude_phases=exclude_phases, phase_colors=phase_colors, title=title,
                              rename_disordered=rename_disordered, pressure=pressure, time_limit_s=time_limit_s,
                              label_regions=label_regions, label_invariants=label_invariants,
                              phase_labels=phase_labels, label_style=label_style, formats=formats, right=right,
                              corners=corners, style=style or tb.diagram_style)
        return r["text"] + "\nFiles:\n" + rel_files(r["files"]), [out]

    @tb.add("step_map",
            "Step calculation with pycalphad StepStrategy: vary T (or X(EL) or P) at fixed other conditions. Exact "
            "phase-transition points (phase appears/disappears) + CSV of phase fractions NP(phase), phase "
            "compositions X(phase,EL), chemical potentials MU(EL) and system properties vs the stepped variable, "
            "and a PNG with one panel per `plot` entry.",
            {"tdb": S, "elements": STRS,
             "vary": {**S, "description": "T, P or X(EL)"}, "start": N, "stop": N,
             "step": {**N, "description": "default 5 K / 0.005 mole fraction"},
             "fixed": {**FRACS, "description": "other conditions, e.g. {\"X(NI)\": 0.3} or {\"T\": 1273, \"X(CR)\": 0.2}"},
             "temperature_unit": UNIT,
             "properties": {**STRS, "description": "system properties per point (pycalphad names), default "
                                                   "[GM, HM, SM, CPM]; e.g. HM.T (apparent Cp incl. latent heat), "
                                                   "DOO, X(FCC_A1,CR)"},
             "plot": {**STRS, "description": "panels: NP (default), X(*,EL) phase compositions, or any "
                                             "property/MU(EL) column"},
             "phases": STRS, "exclude_phases": STRS, "phase_colors": COLORS, "rename_disordered": B,
             "title": S, "output_prefix": S},
            ["tdb", "elements", "vary", "start", "stop"])
    def _step(tdb, elements, vary, start, stop, step=None, fixed=None, temperature_unit="K", properties=None,
              plot=None, phases=None, exclude_phases=None, phase_colors=None, rename_disordered=True, title=None,
              output_prefix=None):
        t = tb._find_tdb(tdb)
        tag = "-".join(_el(e) for e in elements) + "_" + re.sub(r"[^A-Za-z0-9.]+", "", str(vary)) + "_" + \
            "_".join(f"{re.sub(r'[^A-Za-z0-9.]+', '', k)}{val:g}" for k, val in (fixed or {}).items())
        out = tb.outp((output_prefix + ".png") if output_prefix else None, f"step_{tag}_{Path(t).stem}.png")
        r = step_map(t, elements, vary, start, stop, out, step=step, fixed=fixed, temperature_unit=temperature_unit,
                     properties=properties, plot=plot, phases=phases, exclude_phases=exclude_phases,
                     phase_colors=phase_colors, rename_disordered=rename_disordered, title=title)
        return r["text"] + f"\nFiles:\n  png: {tb.rel(r['png'])}\n  csv: {tb.rel(r['csv'])} " \
                           f"({len(r['columns'])} columns)", [out]

    @tb.add("scheil_solidification",
            "Scheil-Gulliver (no diffusion in solid) and equilibrium solidification of an alloy with the `scheil` "
            "package. composition = mole fractions of all elements except the balance element. Writes a PNG "
            "(T vs fraction solid for both models + cumulative Scheil phase amounts), CSVs of phase amounts and "
            "phase compositions vs T, and returns liquidus, solidus, Scheil end temperature and phase sequence.",
            {"tdb": S, "elements": STRS, "composition": FRACS,
             "start_temperature": {**N, "description": "must be fully liquid; default: automatic (liquidus+~30 K)"},
             "temperature_unit": UNIT,
             "step_temperature": {**N, "description": "K, default 1"},
             "include_equilibrium": {**B, "description": "default true"},
             "liquid_phase": S, "phases": STRS, "exclude_phases": STRS, "phase_colors": COLORS, "title": S,
             "output_prefix": S},
            ["tdb", "elements", "composition"])
    def _scheil(tdb, elements, composition, start_temperature=None, temperature_unit="K", step_temperature=1.0,
                include_equilibrium=True, liquid_phase=None, phases=None, exclude_phases=None, phase_colors=None,
                title=None, output_prefix=None):
        t = tb._find_tdb(tdb)
        tag = "-".join(_el(e) for e in elements) + "_" + "_".join(f"{_el(k)}{val:g}" for k, val in composition.items())
        out = tb.outp((output_prefix + ".png") if output_prefix else None, f"scheil_{tag}_{Path(t).stem}.png")
        r = scheil_solidification(t, elements, composition, out, start_temperature=start_temperature,
                                  temperature_unit=temperature_unit, step_temperature=step_temperature,
                                  include_equilibrium=include_equilibrium, liquid_phase=liquid_phase, phases=phases,
                                  exclude_phases=exclude_phases, phase_colors=phase_colors, title=title)
        return r["text"] + "\nFiles:\n" + rel_files(r["files"]), [out]

    @tb.add("write_database",
            "Write a database with pycalphad's TDB writer: normalise formatting, or extract a subsystem by "
            "restricting elements and/or phases (unused functions and parameters are dropped; order/disorder "
            "partners kept). The file is reloaded and every phase's Gibbs energy is checked against the source. "
            "dialect thermocalc|opencalphad|pycalphad enforces that program's limits (name lengths, line length, "
            "type codes, ...) with readable, listed renames. Note: comments are not preserved.",
            {"tdb": S, "output_tdb": S, "elements": STRS, "phases": STRS, "exclude_phases": STRS,
             "groupby": {"type": "string", "enum": ["subsystem", "phase"]},
             "dialect": {"type": "string", "enum": ["thermocalc", "opencalphad", "pycalphad"]},
             "fix_incompatible": {**B, "description": "no dialect: shorten >8-char function names (hashed) for "
                                                      "Thermo-Calc (default true)"}},
            ["tdb", "output_tdb"])
    def _write(tdb, output_tdb, elements=None, phases=None, exclude_phases=None, groupby="subsystem",
               fix_incompatible=True, dialect=None):
        t = tb._find_tdb(tdb)
        out = tb.outp(output_tdb, f"{Path(t).stem}_subset.tdb")
        if Path(out).resolve() == Path(t).resolve():
            raise ValueError("refusing to overwrite the source database; choose another output_tdb")
        r = write_database(t, out, elements=elements, phases=phases, exclude_phases=exclude_phases,
                           groupby=groupby, fix_incompatible=fix_incompatible, dialect=dialect)
        chk = r.get("model_check_max_abs_dGM_J_per_mol", {})
        worst = max(chk.values()) if chk else None
        txt = [f"wrote {tb.rel(out)}: elements {r['elements']}, {len(r['phases'])} phases, "
               f"{r['parameters']}/{r['parameters_original']} parameters, "
               f"{r['functions']}/{r['functions_original']} functions",
               f"reloaded OK: {r['reloaded']}",
               f"GM check vs source ({len(chk)} phases, random states): max |dGM| = "
               f"{'n/a' if worst is None else f'{worst:.3g} J/mol'}"]
        if r.get("model_check_error"):
            txt.append(f"model check error: {r['model_check_error']}")
        if r["dropped_phases"]:
            txt.append("dropped phases: " + "; ".join(f"{k} ({val})" for k, val in sorted(r["dropped_phases"].items())))
        if dialect:
            for kind, m in r.get("renames", {}).items():
                txt.append(f"renamed {kind} ({len(m)}): " + ", ".join(f"{a} -> {b}" for a, b in sorted(m.items())))
            v = r.get("violations") or []
            txt.append(f"{dialect} limits: " + ("all met" if not v else f"{len(v)} problem(s): " + "; ".join(v[:5])))
            if r.get("syntax_findings"):
                txt.append("TDB syntax check: " + " | ".join(x[:150] for x in r["syntax_findings"][:4]))
        if r["writer_warnings"]:
            ww = r["writer_warnings"]
            txt.append(f"writer warnings ({len(ww)}; harmless if the GM check is ~0): " + " | ".join(
                w[:160] for w in ww[:3]) + (" | ..." if len(ww) > 3 else ""))
        return "\n".join(txt), []

    @tb.add("export_phase_model",
            "Symbolic Gibbs energy model of one phase (pycalphad Model): reference, ideal mixing, excess, magnetic, "
            "ordering contributions and physical properties (TC, BMAG), written to a text file (optionally LaTeX), "
            "plus a CSV of all contributing parameters. Optionally evaluates each contribution at T and X(EL) "
            "(random mixing on each sublattice) or explicit site fractions. E.g. 'excess Gibbs energy of FCC_A1 "
            "in Cr-Ni'.",
            {"tdb": S, "phase": S, "elements": {**STRS, "description": "subsystem (default: all phase elements)"},
             "expand_functions": {**B, "description": "substitute FUNCTIONs (default false: kept symbolic and "
                                                      "listed)"},
             "evaluate_at": {**FRACS, "description": "e.g. {\"T\": 1000, \"X(CR)\": 0.3, \"X(NI)\": 0.7} or "
                                                    "site fractions {\"Y(FCC_A1,0,CR)\": 0.3, ...}"},
             "latex": B, "output_prefix": S},
            ["tdb", "phase"])
    def _model(tdb, phase, elements=None, expand_functions=False, evaluate_at=None, latex=False, output_prefix=None):
        t = tb._find_tdb(tdb)
        tag = f"{phase.upper()}_{'-'.join(_el(e) for e in elements) if elements else 'all'}"
        prefix = tb.outp(output_prefix, f"model_{tag}_{Path(t).stem}")
        r = export_phase_model(t, phase, prefix, elements=elements, expand_functions=expand_functions,
                               evaluate_at=evaluate_at, latex=latex)
        return r["text"] + "\nFiles:\n" + rel_files(r["files"]), []
