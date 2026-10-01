"""Editable figures: matplotlib control for the chat agent without arbitrary Python.

Every figure saved by the tools (``Figure.savefig`` is hooked) also stores a pickled copy of the
matplotlib Figure in ``<output_dir>/.figures``.  The agent can then

* ``describe_figure``  - list the axes, their limits/labels/scales and the plotted artists,
* ``edit_figure``      - draw on the *real axes in data coordinates* (points, lines, error bars, text,
                         arrows, spans, shapes...), restyle/remove existing artists, change limits/labels/
                         scales/legend, and export PNG/PDF/SVG,
* ``make_plot``        - build a new (multi-panel) figure from inline data or CSV columns.

Images that were not produced here (no stored figure) are edited as a background image: coordinates are
fractions of the image, or data coordinates after a pixel calibration.

Axis display transforms (e.g. data in kelvin shown in °C, fractions shown as percent) are recorded by
:class:`ScaledFormatter`, so values given "as displayed" land in the right place.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import Formatter  # noqa: E402

MAX_PICKLE_BYTES = 80 * 1024 * 1024
_STATE: Dict[str, Any] = {"dir": None, "orig": None}


# ---------------------------------------------------------------------------
# picklable display transform
# ---------------------------------------------------------------------------
class ScaledFormatter(Formatter):
    """Tick labels show ``data * scale + offset``.  Picklable (unlike FuncFormatter(lambda ...))."""

    def __init__(self, scale: float = 1.0, offset: float = 0.0, fmt: str = "{:g}"):
        self.scale, self.offset, self.fmt = scale, offset, fmt

    def __call__(self, x, pos=None):
        return self.fmt.format(x * self.scale + self.offset)

    def to_data(self, displayed):
        return (np.asarray(displayed, dtype=float) - self.offset) / self.scale

    def to_display(self, data):
        return np.asarray(data, dtype=float) * self.scale + self.offset


def _to_data(axis, values, mode: str):
    if values is None:
        return None
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    fmt = axis.get_major_formatter()
    if mode == "displayed" and isinstance(fmt, ScaledFormatter):
        return fmt.to_data(arr)
    return arr


def _to_display(axis, values):
    fmt = axis.get_major_formatter()
    arr = np.asarray(values, dtype=float)
    return fmt.to_display(arr) if isinstance(fmt, ScaledFormatter) else arr


# ---------------------------------------------------------------------------
# figure store (savefig hook)
# ---------------------------------------------------------------------------
def _key(path: str) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:20]


def _store_path(img_path: str) -> Optional[Path]:
    d = _STATE["dir"] or os.environ.get("CALPHAD_FIGSTORE")
    return Path(d) / f"{_key(img_path)}.pickle" if d else None


def install(store_dir: str) -> None:
    """Hook Figure.savefig so every saved figure also stores an editable pickle."""
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    _STATE["dir"] = str(store_dir)
    os.environ["CALPHAD_FIGSTORE"] = str(store_dir)       # inherited by spawned diagram workers
    if _STATE["orig"] is not None:
        return
    orig = Figure.savefig
    _STATE["orig"] = orig

    def savefig(self, fname, *args, **kwargs):
        res = orig(self, fname, *args, **kwargs)
        try:
            if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith((".png", ".jpg", ".jpeg")):
                sp = _store_path(str(fname))
                if sp is not None:
                    data = pickle.dumps(self)
                    if len(data) <= MAX_PICKLE_BYTES:
                        sp.write_bytes(data)
        except Exception:  # noqa: BLE001 - storing the editable copy must never break a plot
            pass
        return res

    Figure.savefig = savefig


def install_from_env() -> None:
    d = os.environ.get("CALPHAD_FIGSTORE")
    if d:
        install(d)


def load_figure(img_path: str) -> Optional[Figure]:
    """The stored Figure for an image written by the tools, or None (stale / not stored)."""
    sp = _store_path(img_path)
    if sp is None or not sp.exists() or not Path(img_path).exists():
        return None
    if Path(img_path).stat().st_mtime > sp.stat().st_mtime + 5:   # image replaced by something else
        return None
    import pycalphad.plot.triangular  # noqa: F401 - registers the 'triangular' projection for unpickling
    with open(sp, "rb") as f:
        fig = pickle.load(f)   # only files this module wrote into the private store are ever loaded
    try:
        fig.set_canvas(matplotlib.backends.backend_agg.FigureCanvasAgg(fig))
    except Exception:  # noqa: BLE001
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    return fig


def raster_figure(img_path: str, calibration: Optional[Dict] = None):
    """A figure showing an image, with an overlay axes for drawing.

    Without calibration the overlay spans the whole image with limits 0..1 (x right, y up).
    calibration = {"pixel_box": [left, top, right, bottom] (pixels of the plot frame),
                   "xlim": [x at left, x at right], "ylim": [y at bottom, y at top]}.
    """
    img = plt.imread(img_path)
    h, w = img.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(img)
    bg.set_axis_off()
    bg.set_label("background image")
    if calibration:
        l, t, r, b = [float(v) for v in calibration["pixel_box"]]
        ov = fig.add_axes([l / w, 1 - b / h, (r - l) / w, (b - t) / h])
        ov.set_xlim(*calibration["xlim"])
        ov.set_ylim(*calibration["ylim"])
        if calibration.get("xscale"):
            ov.set_xscale(calibration["xscale"])
        if calibration.get("yscale"):
            ov.set_yscale(calibration["yscale"])
    else:
        ov = fig.add_axes([0, 0, 1, 1])
        ov.set_xlim(0, 1)
        ov.set_ylim(0, 1)
    ov.patch.set_alpha(0)
    ov.set_axis_off()
    ov.set_label("overlay")
    return fig


def _is_secondary(a) -> bool:
    return type(a).__name__ == "SecondaryAxis"


def main_axes(fig) -> List:
    """Axes the user thinks of as plots (colour bars and background images excluded).

    Top-level axes first (stable indices), then inset child axes.  Secondary axes (secondary_xaxis/yaxis) are
    reported by describe() under their parent, not listed here.
    """
    top = [a for a in fig.axes if a.get_label() not in ("<colorbar>", "background image")]
    kids = [c for a in top for c in getattr(a, "child_axes", [])
            if not _is_secondary(c) and c.get_label() != "<colorbar>" and c not in top]
    return top + kids


# ---------------------------------------------------------------------------
# description
# ---------------------------------------------------------------------------
def describe(fig) -> Dict:
    out = {"figsize_in": [round(float(v), 2) for v in fig.get_size_inches()], "axes": []}
    st = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else ""
    if st:
        out["suptitle"] = st
    for i, ax in enumerate(main_axes(fig)):
        xl = _to_display(ax.xaxis, ax.get_xlim()).round(4).tolist()
        yl = _to_display(ax.yaxis, ax.get_ylim()).round(4).tolist()
        d = {"index": i, "projection": ax.name, "title": ax.get_title(), "xlabel": ax.get_xlabel(),
             "ylabel": ax.get_ylabel(), "xlim_displayed": xl, "ylim_displayed": yl,
             "xscale": ax.get_xscale(), "yscale": ax.get_yscale()}
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            f = axis.get_major_formatter()
            if isinstance(f, ScaledFormatter):
                d[f"{name}_display_transform"] = f"displayed = data*{f.scale:g} + {f.offset:g}"
        lines = []
        for ln in ax.get_lines():
            lab = ln.get_label()
            lines.append({"label": lab if not lab.startswith("_") else "", "color": matplotlib.colors.to_hex(ln.get_color()),
                          "n": len(ln.get_xdata()), "style": ln.get_linestyle(), "marker": str(ln.get_marker())})
        # summarise unlabeled lines by colour to keep this short
        labeled = [l for l in lines if l["label"]]
        unlabeled: Dict[str, int] = {}
        for l in lines:
            if not l["label"]:
                unlabeled[l["color"]] = unlabeled.get(l["color"], 0) + 1
        d["labeled_lines"] = labeled[:40]
        d["unlabeled_lines_by_color"] = unlabeled
        d["collections"] = [{"type": type(c).__name__, "label": c.get_label() if not c.get_label().startswith("_") else "",
                             "n": len(c.get_offsets()) if hasattr(c, "get_offsets") else None}
                            for c in ax.collections][:30]
        d["n_texts"] = len(ax.texts)
        dg = getattr(ax, "_calphad_diagram", None) or {}
        if dg.get("recipe") or dg.get("orientation"):
            try:
                from .orientation import recipe_axes_text
                d["phase_diagram"] = recipe_axes_text(dg.get("recipe") or dg["orientation"]) + \
                    "; add data by composition with composition_points"
            except Exception:  # noqa: BLE001 - description only
                pass
        legend = _legend_colors(ax)
        if legend:
            d["legend"] = legend
        sec = [c for c in getattr(ax, "child_axes", []) if _is_secondary(c)]
        if sec:
            d["secondary_axes"] = [{"xlabel": c.get_xlabel(), "ylabel": c.get_ylabel(),
                                    "orientation": getattr(c, "_orientation", "")} for c in sec]
        if ax in getattr(ax.figure, "axes", []) is False:
            d["inset"] = True
        out["axes"].append(d)
    return out


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------
COMMON_HELP = """Operations are objects {"op": NAME, ...}. Common keys:
- "axes": index from describe_figure (default 0; in make_plot the panel's own axes) or an axes title.
- "coords": "data" (default) | "axes" (0-1 fractions of the plot frame) | "figure".
- "units": "displayed" (default: values as shown on tick labels, e.g. °C on a kelvin axis) | "data".
- Data: inline (x, y, ... lists or numbers) OR a source "csv"/"data": path (csv, xlsx, json, nc) or inline records,
  with "<key>_col" to pick columns (x_col, y_col, z_col, yerr_col, c_col, s_col, ...; unambiguous prefixes
  work), "where": {col: value|[values]|{">": v, "between": [a, b], "contains": s, ...}} and "transform": [steps]
  (derive, filter, groupby, pivot, join, interpolate, smooth, derivative, integrate, normalize, bin, convert ...).
  Details: plot_ops_help(["data"]) for sources/where, plot_ops_help(["transform"]) for the pipeline.
- Style keys are matplotlib names: color, marker, markersize, s, linestyle, linewidth, alpha, label, zorder,
  edgecolor, facecolor, fontsize, fontweight, ha, va, rotation, cmap, vmin, vmax, norm, hatch.
- Colour-mapped ops accept "colorbar": true | {label, ticks, extend, orientation, shrink} and
  "norm": {"type": "log"|"symlog"|"diverging"|"bounds"|"power", ...}."""

BUILTIN_HELP = {
    "scatter": "scatter {x, y, [c (values -> cmap), s (sizes), cmap, norm, colorbar], color (one or per point), marker}",
    "line": "line {x, y, color, linestyle, linewidth, marker, label}  (alias: plot)",
    "step": "step {x, y, where_step: pre|mid|post}",
    "errorbar": "errorbar {x, y, yerr, xerr, capsize, marker, color, label}",
    "fill_between": "fill_between {x, y1, y2, color, alpha, label}",
    "bar": "bar {x, height, width, color, label}",
    "barh": "barh {y, width, height, color, label}",
    "hist": "hist {values, bins, color, alpha, label}",
    "text": "text {x, y, text, color, fontsize, ha, va, rotation, bbox: true}",
    "annotate": "annotate {text, xy: [x, y], xytext: [x, y], arrow: true, arrowstyle, color, fontsize}",
    "arrow": "arrow {x, y, dx, dy, color, linewidth, arrowstyle}",
    "hline": "hline {y, xmin, xmax (axes fractions), color, linestyle, label}",
    "vline": "vline {x, ymin, ymax (axes fractions), color, linestyle, label}",
    "hspan": "hspan {y1, y2, color, alpha, label}",
    "vspan": "vspan {x1, x2, color, alpha, label}",
    "rectangle": "rectangle {x, y, width, height, edgecolor, facecolor, alpha}",
    "circle": "circle {x, y, radius, edgecolor, facecolor}",
    "ellipse": "ellipse {x, y, width, height}",
    "polygon": "polygon {points: [[x, y], ...], edgecolor, facecolor}",
    "contour": "contour {x, y (1-D grid axes), z (2-D list [len(y)][len(x)]), levels, cmap, clabel}",
    "contourf": "contourf {x, y, z, levels, cmap, colorbar}",
    "set": ("set {xlim, ylim, xlabel, ylabel, title, xscale, yscale: linear|log|symlog, grid, aspect, invert_x, "
            "invert_y, xticks, yticks, xticklabels, yticklabels, tick_fontsize, label_fontsize, title_fontsize, "
            "facecolor}"),
    "legend": "legend {show: true|false, loc, ncol, fontsize, title, outside: true, frameon} (existing entries kept)",
    "twinx": "twinx {ylabel, ylim}: new y-axis sharing x (new axes index reported)",
    "inset": "inset {bounds: [x0, y0, w, h] axes fractions, xlim, ylim}: zoomed copy of the data",
    "style": ("style {match: label | phase name from the legend | 'color:#hex' | regex, color, linewidth, "
              "linestyle, alpha, marker, markersize, visible, new_label}"),
    "remove": "remove {match}",
    "figure": "figure {size: [w_in, h_in], suptitle, dpi, tight_layout}",
}
BUILTIN_GROUP = {k: g for g, ks in {
    "draw": ["scatter", "line", "step", "errorbar", "fill_between", "bar", "barh", "hist", "contour", "contourf"],
    "annotate": ["text", "annotate", "arrow", "hline", "vline", "hspan", "vspan", "rectangle", "circle", "ellipse",
                 "polygon"],
    "axes & layout": ["set", "legend", "twinx", "inset", "figure"],
    "existing artists": ["style", "remove"]}.items() for k in ks}


def all_op_names() -> List[str]:
    ensure_extensions()
    return sorted(set(BUILTIN_HELP) | set(OPS))


def op_help(name: str) -> str:
    return OP_HELP.get(name) or BUILTIN_HELP.get(name, "")


def ops_summary() -> str:
    """Compact op list by group (for tool descriptions)."""
    ensure_extensions()
    groups: Dict[str, List[str]] = {}
    for n in all_op_names():
        groups.setdefault(OP_GROUP.get(n) or BUILTIN_GROUP.get(n, "other"), []).append(n)
    return "; ".join(f"{g}: {', '.join(v)}" for g, v in groups.items())


def make_norm(spec):
    """Colour normalisation from a JSON spec (picklable matplotlib Normalize objects)."""
    from matplotlib import colors as mc
    if spec is None or isinstance(spec, mc.Normalize):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    t = spec.get("type", "linear").lower()
    vmin, vmax = spec.get("vmin"), spec.get("vmax")
    if t == "linear":
        return mc.Normalize(vmin, vmax)
    if t == "log":
        return mc.LogNorm(vmin, vmax)
    if t == "symlog":
        return mc.SymLogNorm(linthresh=spec.get("linthresh", 1.0), vmin=vmin, vmax=vmax)
    if t == "power":
        return mc.PowerNorm(spec.get("gamma", 0.5), vmin, vmax)
    if t in ("diverging", "twoslope", "center"):
        return mc.TwoSlopeNorm(vcenter=spec.get("center", 0.0), vmin=vmin, vmax=vmax)
    if t in ("bounds", "discrete", "boundary"):
        b = spec["boundaries"]
        return mc.BoundaryNorm(b, ncolors=spec.get("ncolors", 256), extend=spec.get("extend", "neither"))
    raise ValueError(f"unknown norm type {t!r} (linear, log, symlog, power, diverging, bounds)")


_DRAW_STYLE = {"color", "marker", "markersize", "linestyle", "linewidth", "alpha", "label", "zorder", "markeredgecolor",
               "markerfacecolor", "markeredgewidth", "drawstyle", "fillstyle", "dashes"}
_SCATTER_STYLE = {"color", "marker", "s", "alpha", "label", "zorder", "edgecolor", "edgecolors", "facecolor",
                  "facecolors", "linewidths", "cmap", "vmin", "vmax"}
_TEXT_STYLE = {"color", "fontsize", "fontweight", "ha", "va", "rotation", "alpha", "zorder", "fontstyle", "family"}
_PATCH_STYLE = {"color", "edgecolor", "facecolor", "linewidth", "linestyle", "alpha", "label", "zorder", "fill",
                "hatch"}


def _pick(op: Dict, allowed: set) -> Dict:
    kw = {k: v for k, v in op.items() if k in allowed}
    for k in ("color", "edgecolor", "facecolor", "markerfacecolor", "markeredgecolor"):
        if k in kw and isinstance(kw[k], str) and not matplotlib.colors.is_color_like(kw[k]):
            raise ValueError(f"{k}={kw[k]!r} is not a matplotlib colour (use a name like 'red', hex '#1f77b4', "
                             f"or 'C0'..'C9')")
    return kw


# ---------------------------------------------------------------------------
# data sources (extensible: calphad_tools.figdata adds file types + transform pipelines)
# ---------------------------------------------------------------------------
SOURCE_KEYS = ("csv", "data")
COL_KEYS = ("x", "y", "z", "yerr", "xerr", "c", "s", "y1", "y2", "values", "height", "width", "labels", "weights")
DATA_LOADERS: List[Callable] = []     # fn(op, resolve) -> DataFrame | None; the most recently added wins


def add_data_loader(fn: Callable) -> Callable:
    DATA_LOADERS.append(fn)
    return fn


def _default_frame(op: Dict, resolve):
    import pandas as pd
    src = op.get("csv") or op.get("data")
    if isinstance(src, (list, dict)):          # inline records / columns
        return pd.DataFrame(src)
    df = pd.read_csv(resolve(src))
    for col, val in (op.get("where") or {}).items():
        col = find_col(df, col, src)
        df = df[df[col].isin(val if isinstance(val, list) else [val])]
    return df


def load_frame(op: Dict, resolve):
    """DataFrame for an operation's data source (after any loader-specific filtering/transforms)."""
    for fn in reversed(DATA_LOADERS):
        df = fn(op, resolve)
        if df is not None:
            return df
    return _default_frame(op, resolve)


def find_col(df, col: str, src="data") -> str:
    """Exact column name, else an unambiguous prefix or substring (e.g. "ACR(CR" or "T [")."""
    if col in df.columns:
        return col
    cand = [c for c in df.columns if str(c).startswith(col)] or [c for c in df.columns if col in str(c)]
    if len(cand) != 1:
        raise ValueError(f"column {col!r} not in {src if isinstance(src, str) else 'data'}; "
                         f"columns: {list(df.columns)}")
    return cand[0]


# ---------------------------------------------------------------------------
# operation registry (extension modules register more ops; registered ops override built-ins)
# ---------------------------------------------------------------------------
OPS: Dict[str, Callable] = {}
OP_HELP: Dict[str, str] = {}
OP_NEEDS_AXES: Dict[str, bool] = {}
OP_GROUP: Dict[str, str] = {}


def register_op(name: str, help: str, needs_axes: bool = True, group: str = "other"):
    """Decorator: fn(ctx: OpContext) -> None.  ``help`` is the one-op syntax shown by plot_ops_help."""
    def deco(fn):
        OPS[name] = fn
        OP_HELP[name] = help
        OP_NEEDS_AXES[name] = needs_axes
        OP_GROUP[name] = group
        return fn
    return deco


EXTENSION_MODULES = ["calphad_tools.figdata", "calphad_tools.figfit", "calphad_tools.figops_extra",
                     "calphad_tools.figcompose", "calphad_tools.region_labels", "calphad_tools.orientation"]
VECTOR_RC = {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}   # text stays text in SVG/PDF/EPS
LABEL_OPS = ("label_phase_regions", "move_label", "delete_label", "relabel")


def ensure_extensions() -> None:
    import importlib
    for m in EXTENSION_MODULES:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            if e.name != m:
                raise


class OpContext:
    """Everything an operation needs.  Values given 'as displayed' are converted by X/Y (e.g. °C -> K)."""

    def __init__(self, fig, ax, op: Dict, n: int, resolve, notes: List[str]):
        self.fig, self.ax, self.op, self.n, self.resolve, self.notes = fig, ax, op, n, resolve, notes
        self.coords = op.get("coords", "data")
        self.mode = op.get("units", "displayed") if self.coords == "data" else "data"
        self.tr = _transform(ax, fig, self.coords) if ax is not None else None
        self._frame = None

    # -- data --------------------------------------------------------------
    @property
    def has_source(self) -> bool:
        return any(k in self.op for k in SOURCE_KEYS)

    def frame(self):
        """pandas DataFrame of the op's data source (raises if the op has none)."""
        if self._frame is None:
            if not self.has_source:
                raise ValueError("this operation needs a data source: csv/data (+ *_col keys)")
            self._frame = load_frame(self.op, self.resolve)
        return self._frame

    def colname(self, key: str) -> Optional[str]:
        col = self.op.get(f"{key}_col")
        return find_col(self.frame(), col, self.op.get("csv") or self.op.get("data")) if col else None

    def raw(self, key: str, dtype=float):
        """Values for ``key`` from the '<key>_col' column of the data source, else the inline value."""
        col = self.op.get(f"{key}_col")
        if col and self.has_source:
            v = self.frame()[self.colname(key)].to_numpy()
            return v.astype(dtype) if dtype is not None else v
        v = self.op.get(key)
        if v is None:
            return None
        return np.atleast_1d(np.asarray(v, dtype=dtype)) if dtype is not None else v

    def cols(self) -> Dict[str, np.ndarray]:
        return {k: self.raw(k) for k in COL_KEYS if self.op.get(f"{k}_col") and self.has_source}

    # -- coordinates -------------------------------------------------------
    def X(self, key="x"):
        v = self.raw(key)
        return _to_data(self.ax.xaxis, v, self.mode) if self.coords == "data" else v

    def Y(self, key="y"):
        v = self.raw(key)
        return _to_data(self.ax.yaxis, v, self.mode) if self.coords == "data" else v

    def _delta(self, axis, key):
        v = self.raw(key)
        if v is None:
            return None
        f = axis.get_major_formatter()
        v = np.ravel(np.asarray(v, dtype=float))
        v = v[0] if v.size == 1 else v                  # a width/height is a scalar (numpy 2: float() needs 0-d)
        return v / f.scale if (self.mode == "displayed" and isinstance(f, ScaledFormatter)) else v

    def dX(self, key):
        return self._delta(self.ax.xaxis, key)

    def dY(self, key):
        return self._delta(self.ax.yaxis, key)

    # -- helpers -----------------------------------------------------------
    def pick(self, allowed: set) -> Dict:
        return _pick(self.op, allowed)

    def colorbar(self, mappable, ax=None):
        """Honour op['colorbar'] (true or {label, ticks, extend, orientation, shrink, location}) / 'colorbar_label'."""
        spec = self.op.get("colorbar")
        if not spec:
            return None
        spec = spec if isinstance(spec, dict) else {}
        kw = {k: spec[k] for k in ("extend", "orientation", "shrink", "location", "pad", "aspect") if k in spec}
        cb = self.fig.colorbar(mappable, ax=ax or self.ax, **kw)
        label = spec.get("label", self.op.get("colorbar_label"))
        if label:
            cb.set_label(label)
        if spec.get("ticks") is not None:
            cb.set_ticks(spec["ticks"])
        return cb

    def note(self, text: str):
        self.notes.append(f"op {self.n}: {text}")

    def new_axes(self, ax2, what: str):
        self.note(f"{what} created as axes {main_axes(self.fig).index(ax2)}")


def _arr(v):
    if v is None:
        return None
    return np.atleast_1d(np.asarray(v, dtype=float))


def _transform(ax, fig, coords: str):
    return {"axes": ax.transAxes, "figure": fig.transFigure}.get(coords, ax.transData)


def apply_ops(fig, ops: Sequence[Dict], resolve=lambda p: p) -> List[str]:
    """Apply operations to a figure; returns notes (e.g. indices of newly created axes)."""
    from matplotlib import patches as mp
    ensure_extensions()
    notes = []
    for n, op in enumerate(ops, 1):
        kind = str(op.get("op", "")).lower()
        try:
            axes_list = main_axes(fig)
            needs_axes = OP_NEEDS_AXES.get(kind, kind != "figure")
            idx = op.get("axes", 0)
            ax = None
            if isinstance(idx, str):        # allow an axes to be picked by its title
                hits = [a for a in axes_list if a.get_title().lower() == idx.lower()]
                if not hits:
                    raise ValueError(f"no axes titled {idx!r}; titles: {[a.get_title() for a in axes_list]}")
                ax = hits[0]
            elif axes_list:
                idx = int(idx)
                if needs_axes and not (0 <= idx < len(axes_list)):
                    raise ValueError(f"axes index {idx} out of range (0..{len(axes_list) - 1})")
                ax = axes_list[idx] if 0 <= idx < len(axes_list) else None
            if needs_axes and ax is None:
                raise ValueError("figure has no axes")
            ctx = OpContext(fig, ax, op, n, resolve, notes)
            if kind in OPS:
                OPS[kind](ctx)
                continue
            coords, mode, tr = ctx.coords, ctx.mode, ctx.tr
            d = ctx.cols()
            X, Y, dX, dY = ctx.X, ctx.Y, ctx.dX, ctx.dY

            if kind == "scatter":
                kw = _pick(op, _SCATTER_STYLE)
                c = d.get("c", op.get("c"))
                if c is not None:
                    kw["c"] = c
                    kw.pop("color", None)
                if ctx.raw("s") is not None and "s_col" in op:
                    kw["s"] = ctx.raw("s")
                for k in ("norm",):
                    if k in op:
                        kw["norm"] = make_norm(op["norm"])
                        kw.pop("vmin", None)
                        kw.pop("vmax", None)
                sc = ax.scatter(X(), Y(), transform=tr, **kw)
                if c is not None:
                    ctx.colorbar(sc)
            elif kind in ("line", "plot"):
                ax.plot(X(), Y(), transform=tr, **_pick(op, _DRAW_STYLE))
            elif kind == "step":
                ax.step(X(), Y(), where=op.get("where_step", "mid"), transform=tr, **_pick(op, _DRAW_STYLE))
            elif kind == "errorbar":
                kw = _pick(op, _DRAW_STYLE)
                kw.setdefault("linestyle", "none")
                kw.setdefault("marker", "o")
                ax.errorbar(X(), Y(), yerr=dY("yerr"), xerr=dX("xerr"), capsize=op.get("capsize", 3),
                            transform=tr, **kw)
            elif kind == "fill_between":
                ax.fill_between(X(), Y("y1"), Y("y2"), transform=tr, **_pick(op, _PATCH_STYLE - {"fill"}))
            elif kind == "bar":
                ax.bar(X(), dY("height"), width=op.get("width", 0.8), transform=tr, **_pick(op, _PATCH_STYLE - {"fill"}))
            elif kind == "barh":
                ax.barh(Y(), dX("width"), height=op.get("height", 0.8), transform=tr,
                        **_pick(op, _PATCH_STYLE - {"fill"}))
            elif kind == "hist":
                ax.hist(d.get("values", op.get("values")), bins=op.get("bins", 20),
                        **_pick(op, _PATCH_STYLE - {"fill"}))
            elif kind == "text":
                kw = _pick(op, _TEXT_STYLE)
                if op.get("bbox"):
                    kw["bbox"] = {"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"}
                ax.text(float(X()[0]), float(Y()[0]), str(op["text"]), transform=tr, **kw)
            elif kind == "annotate":
                xy = op["xy"]
                xt = op.get("xytext", xy)
                p = [float(_to_data(ax.xaxis, [xy[0]], mode)[0]), float(_to_data(ax.yaxis, [xy[1]], mode)[0])] \
                    if coords == "data" else xy
                pt = [float(_to_data(ax.xaxis, [xt[0]], mode)[0]), float(_to_data(ax.yaxis, [xt[1]], mode)[0])] \
                    if coords == "data" else xt
                kw = _pick(op, _TEXT_STYLE)
                if op.get("arrow", True):
                    kw["arrowprops"] = {"arrowstyle": op.get("arrowstyle", "->"), "color": op.get("color", "black"),
                                        "lw": op.get("linewidth", 1.2)}
                ax.annotate(str(op["text"]), xy=p, xytext=pt, xycoords=tr, textcoords=tr, **kw)
            elif kind == "arrow":
                x0, y0 = float(X()[0]), float(Y()[0])
                x1 = float(_to_data(ax.xaxis, [op["x"] + op["dx"]], mode)[0]) if coords == "data" else op["x"] + op["dx"]
                y1 = float(_to_data(ax.yaxis, [op["y"] + op["dy"]], mode)[0]) if coords == "data" else op["y"] + op["dy"]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=tr, textcoords=tr,
                            arrowprops={"arrowstyle": op.get("arrowstyle", "->"), "color": op.get("color", "black"),
                                        "lw": op.get("linewidth", 1.5)})
            elif kind == "hline":
                ax.axhline(float(Y()[0]), op.get("xmin", 0), op.get("xmax", 1), **_pick(op, _DRAW_STYLE))
            elif kind == "vline":
                ax.axvline(float(X()[0]), op.get("ymin", 0), op.get("ymax", 1), **_pick(op, _DRAW_STYLE))
            elif kind == "hspan":
                ax.axhspan(float(Y("y1")[0]), float(Y("y2")[0]), **_pick(op, _PATCH_STYLE - {"fill"}))
            elif kind == "vspan":
                ax.axvspan(float(X("x1")[0]), float(X("x2")[0]), **_pick(op, _PATCH_STYLE - {"fill"}))
            elif kind in ("rectangle", "circle", "ellipse", "polygon"):
                kw = _pick(op, _PATCH_STYLE)
                kw.setdefault("fill", "facecolor" in op)
                if kind == "rectangle":
                    patch = mp.Rectangle((float(X()[0]), float(Y()[0])), float(dX("width")), float(dY("height")),
                                         transform=tr, **kw)
                elif kind == "circle":
                    patch = mp.Circle((float(X()[0]), float(Y()[0])), float(op["radius"]), transform=tr, **kw)
                elif kind == "ellipse":
                    patch = mp.Ellipse((float(X()[0]), float(Y()[0])), float(dX("width")), float(dY("height")),
                                       transform=tr, **kw)
                else:
                    pts = np.asarray(op["points"], dtype=float)
                    if coords == "data":
                        pts = np.column_stack([_to_data(ax.xaxis, pts[:, 0], mode), _to_data(ax.yaxis, pts[:, 1], mode)])
                    patch = mp.Polygon(pts, closed=True, transform=tr, **kw)
                ax.add_patch(patch)
            elif kind in ("contour", "contourf"):
                z = np.asarray(op["z"], dtype=float)
                fn = ax.contourf if kind == "contourf" else ax.contour
                cs = fn(X(), Y(), z, levels=op.get("levels", 10), cmap=op.get("cmap", "viridis"),
                        alpha=op.get("alpha", 1.0))
                ctx.colorbar(cs)
                if kind == "contour" and op.get("clabel", True):
                    ax.clabel(cs, fontsize=op.get("fontsize", 8))
            elif kind == "set":
                _set_axes(ax, op, mode)
            elif kind == "legend":
                if op.get("show", True) is False:
                    if ax.get_legend():
                        ax.get_legend().remove()
                else:
                    kw = {k: op[k] for k in ("loc", "ncol", "fontsize", "title", "frameon", "framealpha") if k in op}
                    if op.get("outside"):
                        kw.update(loc="center left", bbox_to_anchor=(1.02, 0.5))
                    rebuild_legend(ax, **kw)
                ax._calphad_legend_set = True
            elif kind == "twinx":
                t = ax.twinx()
                if op.get("ylabel"):
                    t.set_ylabel(op["ylabel"])
                if op.get("ylim"):
                    t.set_ylim(*op["ylim"])
                notes.append(f"op {n}: twin y-axis created as axes {main_axes(fig).index(t)}")
            elif kind == "inset":
                ins = ax.inset_axes(op.get("bounds", [0.55, 0.55, 0.4, 0.4]))
                for ln in ax.get_lines():
                    ins.plot(ln.get_xdata(), ln.get_ydata(), color=ln.get_color(), linestyle=ln.get_linestyle(),
                             linewidth=ln.get_linewidth(), marker=ln.get_marker(), markersize=ln.get_markersize())
                for c in ax.collections:
                    if hasattr(c, "get_offsets") and len(c.get_offsets()):
                        off = c.get_offsets()
                        ins.scatter(off[:, 0], off[:, 1], s=12, color=c.get_facecolor()[:1] if len(c.get_facecolor()) else None)
                if op.get("xlim"):
                    ins.set_xlim(*_to_data(ax.xaxis, op["xlim"], mode))
                if op.get("ylim"):
                    ins.set_ylim(*_to_data(ax.yaxis, op["ylim"], mode))
                ins.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())
                ins.yaxis.set_major_formatter(ax.yaxis.get_major_formatter())
                ins.tick_params(labelsize=7)
                ax.indicate_inset_zoom(ins, edgecolor="0.4")
                notes.append(f"op {n}: inset created as axes {main_axes(fig).index(ins)}")
            elif kind == "style":
                hits = _match_artists(ax, op["match"])
                if not hits:
                    raise ValueError(f"no artist matches {op['match']!r}; see describe_figure for labels/colours")
                for a in hits:
                    for k in ("color", "linewidth", "linestyle", "alpha", "marker", "markersize", "zorder",
                              "visible", "edgecolor", "facecolor"):
                        if k in op and hasattr(a, f"set_{k}"):
                            if k in ("color", "edgecolor", "facecolor"):
                                _pick({k: op[k]}, {k})
                            getattr(a, f"set_{k}")(op[k])
                    if "new_label" in op:
                        a.set_label(op["new_label"])
                notes.append(f"op {n}: styled {len(hits)} artist(s)")
            elif kind == "remove":
                hits = _match_artists(ax, op["match"])
                for a in hits:
                    a.remove()
                notes.append(f"op {n}: removed {len(hits)} artist(s)")
            elif kind == "figure":
                if op.get("size"):
                    fig.set_size_inches(*op["size"])
                if op.get("suptitle") is not None:
                    fig.suptitle(op["suptitle"], fontsize=op.get("fontsize", 14))
                if op.get("dpi"):
                    fig.set_dpi(op["dpi"])
                if op.get("tight_layout"):
                    fig.tight_layout()
            else:
                import difflib
                close = difflib.get_close_matches(kind, all_op_names(), n=4, cutoff=0.6) or \
                    [k for k in all_op_names() if kind and (kind in k or k in kind)][:4]
                raise ValueError(f"unknown op {kind!r}" + (f"; similar: {close}" if close else "")
                                 + f". Known ops: {', '.join(all_op_names())}")
        except Exception as e:  # noqa: BLE001 - report which operation failed, with that op's syntax
            syntax = op_help(kind) if kind in all_op_names() else ""
            raise ValueError(f"operation {n} ({kind or '?'}) failed: {type(e).__name__}: {e}"
                             + (f"\nSyntax: {syntax}" if syntax else "")
                             + "\n(plot_ops_help gives the syntax of any op)") from None
    # new labelled artists but no explicit legend op: refresh legends so they show up (old entries kept)
    for ax in main_axes(fig):
        if getattr(ax, "_calphad_legend_set", False):
            continue
        shown = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() is not None else []
        if any(lab not in shown for lab in ax.get_legend_handles_labels()[1]):
            rebuild_legend(ax)
    return notes


def rebuild_legend(ax, **kw):
    """Legend = existing entries (incl. proxy handles, e.g. phase colours) + newly labelled artists."""
    old = ax.get_legend()
    handles, labels = [], []
    if old is not None:
        for h, t in zip(old.legend_handles, old.get_texts()):
            handles.append(h)
            labels.append(t.get_text())
        if not kw:        # keep the previous placement and font size
            kw["loc"] = old._loc
            if old._bbox_to_anchor is not None:
                # re-express the anchor in axes fractions (the stored one is a live display-space box)
                pts = ax.transAxes.inverted().transform(old.get_bbox_to_anchor().get_points())
                (x0, y0), (x1, y1) = pts
                kw["bbox_to_anchor"] = (x0, y0) if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9 \
                    else (x0, y0, x1 - x0, y1 - y0)
            if old.get_texts():
                kw["fontsize"] = old.get_texts()[0].get_fontsize()
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    if handles:
        ax.legend(handles, labels, **kw)


def _legend_colors(ax) -> Dict[str, str]:
    """{legend label: colour} - phase diagrams use proxy legend handles, so this is how phases map to colours."""
    leg = ax.get_legend()
    out = {}
    if leg is None:
        return out
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        try:
            c = h.get_color() if hasattr(h, "get_color") else h.get_facecolor()
            c = c[0] if isinstance(c, np.ndarray) and c.ndim == 2 and len(c) else c
            out[t.get_text()] = matplotlib.colors.to_hex(c)
        except (ValueError, TypeError, IndexError, AttributeError):
            out[t.get_text()] = ""
    return out


def _match_artists(ax, match: str):
    arts = list(ax.get_lines()) + list(ax.collections) + list(ax.patches) + list(ax.texts)
    labeled = _match_labeled(ax, arts, match)
    if labeled or match.lower().startswith("color:"):
        return labeled
    # phase names that only exist in the legend (proxy handles): select artists drawn in that colour
    leg = _legend_colors(ax)
    exact = [lab for lab in leg if lab.lower() == match.lower()]
    hits = exact or [lab for lab in leg if match.lower() in lab.lower()]
    out = []
    for lab in hits:
        if leg[lab]:
            out += [a for a in _match_labeled(ax, arts, "color:" + leg[lab]) if a not in out]
    leg_obj = ax.get_legend()
    if out and leg_obj is not None:        # keep the legend swatch in sync with the restyled artists
        for h, t in zip(leg_obj.legend_handles, leg_obj.get_texts()):
            if t.get_text() in hits:
                out.append(h)
    return out


def _match_labeled(ax, arts, match: str):
    if match.lower().startswith("color:"):
        want = matplotlib.colors.to_hex(match.split(":", 1)[1].strip())
        out = []
        for a in arts:
            try:
                c = a.get_color() if hasattr(a, "get_color") else a.get_facecolor()
                c = c[0] if isinstance(c, np.ndarray) and c.ndim == 2 and len(c) else c
                if matplotlib.colors.to_hex(c) == want:
                    out.append(a)
            except (ValueError, TypeError, IndexError):
                continue
        return out
    rx = re.compile(match, re.I) if any(ch in match for ch in "^$[]()*+?|\\") else None
    out = []
    for a in arts:
        lab = a.get_text() if hasattr(a, "get_text") and not a.get_label() else a.get_label()
        if not lab or lab.startswith("_"):
            continue
        if (rx and rx.search(lab)) or (not rx and match.lower() in lab.lower()):
            out.append(a)
    return out


def _set_axes(ax, op: Dict, mode: str):
    if "xlim" in op:
        ax.set_xlim(*_to_data(ax.xaxis, op["xlim"], mode))
    if "ylim" in op:
        ax.set_ylim(*_to_data(ax.yaxis, op["ylim"], mode))
    for k in ("xlabel", "ylabel"):
        if k in op:
            getattr(ax, f"set_{k}")(op[k], fontsize=op.get("label_fontsize"))
    if "title" in op:
        ax.set_title(op["title"], fontsize=op.get("title_fontsize"))
    for k in ("xscale", "yscale"):
        if k in op:
            getattr(ax, f"set_{k}")(op[k])
    if "grid" in op:
        ax.grid(bool(op["grid"]), alpha=0.35)
    if "aspect" in op:
        ax.set_aspect(op["aspect"])
    if op.get("invert_x"):
        ax.invert_xaxis()
    if op.get("invert_y"):
        ax.invert_yaxis()
    if "xticks" in op:
        ax.set_xticks(_to_data(ax.xaxis, op["xticks"], mode))
    if "yticks" in op:
        ax.set_yticks(_to_data(ax.yaxis, op["yticks"], mode))
    if "xticklabels" in op:
        ax.set_xticklabels(op["xticklabels"])
    if "yticklabels" in op:
        ax.set_yticklabels(op["yticklabels"])
    if "tick_fontsize" in op:
        ax.tick_params(labelsize=op["tick_fontsize"])
    if "facecolor" in op:
        ax.set_facecolor(op["facecolor"])


def _check_extent(fig, limit_in: float = 60.0) -> None:
    """Refuse to save if something sits far outside the figure (bbox_inches='tight' would otherwise try to
    allocate a gigantic canvas) - almost always a coordinate given in the wrong units or coords mode."""
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    if bb.width <= limit_in and bb.height <= limit_in:
        return
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    culprits = []
    for ax in fig.axes:
        for a in ax.get_children():
            if not isinstance(a, (Text, Line2D, PathCollection, Patch)) or a is ax.patch:
                continue
            try:
                e = a.get_window_extent(r)
            except Exception:  # noqa: BLE001
                continue
            if e.width / fig.dpi > limit_in or e.height / fig.dpi > limit_in or \
                    abs(e.x0) / fig.dpi > limit_in or abs(e.y0) / fig.dpi > limit_in:
                lab = a.get_label() if not str(a.get_label()).startswith("_") else ""
                txt = a.get_text() if hasattr(a, "get_text") else ""
                culprits.append(f"{type(a).__name__} {lab or txt!r}".strip())
    raise ValueError(f"an element lies far outside the plot (figure would be {bb.width:.0f}x{bb.height:.0f} in): "
                     f"{', '.join(culprits[:5]) or 'unknown'}. Check that coordinates are in the axis units shown "
                     "by describe_figure (e.g. °C vs K, fraction vs percent) and the coords mode.")


# ---------------------------------------------------------------------------
# agent tools
# ---------------------------------------------------------------------------
PRESETS = {   # figure size (in) + base font size
    "journal-single": ((3.5, 2.6), 8), "journal-double": ((7.2, 4.4), 9), "nature-single": ((3.5, 2.6), 7),
    "nature-double": ((7.2, 4.4), 7), "acs-single": ((3.25, 2.5), 8), "acs-double": ((7.0, 4.3), 8),
    "slide": ((10.0, 5.6), 14), "slide-half": ((6.0, 4.5), 14), "poster": ((12.0, 9.0), 20), "default": (None, None)}
PALETTES = {
    "okabe-ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "tol-bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    "tol-muted": ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499"],
    "tableau-colorblind10": ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200", "#898989",
                             "#A2C8EC", "#FFBC79", "#CFCFCF"],
    "tab10": [matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab10").colors],
}
RC_ALLOWED = {"font.family", "font.size", "font.sans-serif", "font.serif", "axes.labelsize", "axes.titlesize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "lines.linewidth", "lines.markersize",
              "axes.linewidth", "axes.grid", "grid.alpha", "grid.linestyle", "mathtext.fontset", "axes.spines.top",
              "axes.spines.right", "xtick.direction", "ytick.direction", "xtick.minor.visible",
              "ytick.minor.visible", "legend.frameon", "savefig.transparent", "axes.titleweight",
              "axes.labelweight", "figure.facecolor", "axes.facecolor"}


def style_rc(preset: Optional[str] = None, palette=None, font: Optional[Dict] = None, rc: Optional[Dict] = None):
    """rcParams dict for a preset / palette / font / raw rc overrides (allow-listed)."""
    from cycler import cycler
    out: Dict[str, Any] = {}
    size = None
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; presets: {list(PRESETS)}")
        figsize, size = PRESETS[preset]
        if figsize:
            out["figure.figsize"] = figsize
    if size:
        out.update({"font.size": size, "axes.labelsize": size, "axes.titlesize": size + 1,
                    "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 1})
    if palette:
        if isinstance(palette, str):
            if palette in PALETTES:
                cols = PALETTES[palette]
            else:            # "viridis:6" -> 6 colours sampled from a colormap
                name, _, k = palette.partition(":")
                try:
                    cmap = plt.get_cmap(name)
                except ValueError:
                    raise ValueError(f"unknown palette {palette!r}; palettes: {list(PALETTES)} or 'cmap:N'") from None
                k = int(k or 8)
                cols = [matplotlib.colors.to_hex(cmap(i / max(1, k - 1))) for i in range(k)]
        else:
            cols = list(palette)
        out["axes.prop_cycle"] = cycler(color=cols)
    if font:
        if font.get("family"):
            out["font.family"] = font["family"]
        if font.get("size"):
            out["font.size"] = font["size"]
    for k, v in (rc or {}).items():
        if k not in RC_ALLOWED:
            raise ValueError(f"rc key {k!r} not allowed; allowed: {sorted(RC_ALLOWED)}")
        out[k] = v
    return out


def register(tb) -> None:
    install(str(tb.out / ".figures"))
    ensure_extensions()
    OPS = {"type": "array", "items": {"type": "object"},
           "description": "operations (plot_ops_help gives the syntax). " + ops_summary()}
    S = {"type": "string"}

    @tb.add("plot_ops_help", "Syntax of edit_figure/make_plot operations (and data transform steps). Give op names, "
            "or none for the common keys + list of all ops.",
            {"ops": {"type": "array", "items": {"type": "string"}}}, [])
    def plot_ops_help(ops=None):
        names = all_op_names()
        if not ops:
            topics = sorted(k for k in OP_HELP if k not in names)
            return (COMMON_HELP + "\n\nOperations by group: " + ops_summary()
                    + (f"\nOther help topics: {', '.join(topics)}" if topics else "")), []
        out = []
        for o in ops:
            o = o.lower()
            if o in names:
                out.append(f"- {op_help(o)}")
            elif o in OP_HELP:          # pseudo-topics registered by extensions (e.g. "transform")
                out.append(OP_HELP[o])
            else:
                import difflib
                close = difflib.get_close_matches(o, names, n=4, cutoff=0.6) or [k for k in names if o in k or k in o]
                out.append(f"- {o}: unknown" + (f" (similar: {close})" if close else ""))
        return "\n".join(out), []

    def _save(fig, out_png: str, formats, dpi, raster: bool = False, transparent: bool = False) -> List[str]:
        _check_extent(fig)
        files = [out_png]
        # raster overlays keep the source image's exact pixel size (no re-scaling / cropping)
        opts = {"dpi": fig.dpi, "bbox_inches": None} if raster else {"dpi": dpi or 150, "bbox_inches": "tight"}
        if transparent:
            opts["transparent"] = True
        fig.savefig(out_png, **opts)
        for ext in formats or []:
            ext = ext.lower().lstrip(".")
            if ext not in ("pdf", "svg", "eps", "png", "jpg", "tiff"):
                raise ValueError(f"unsupported format {ext}")
            p = str(Path(out_png).with_suffix("." + ext))
            if p != out_png:
                with matplotlib.rc_context(VECTOR_RC if ext in ("pdf", "svg", "eps") else {}):
                    fig.savefig(p, **opts)
                files.append(p)
        return files

    @tb.add("describe_figure", "Describe a plot image made by the tools: its axes (index, labels, limits AS DISPLAYED, "
            "scales, display transforms like K shown as °C), labelled lines/colours, legend. Use before edit_figure.",
            {"image": S}, ["image"])
    def describe_figure(image: str):
        p = tb.p(image)
        fig = load_figure(p)
        if fig is None:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size
            return (f"{image}: no editable copy stored (not made by these tools in this output directory, or "
                    f"replaced since). {w}x{h} px. edit_figure can still draw on it as an image: coordinates are "
                    "fractions of the image (0-1, origin bottom-left), or give calibration={pixel_box:[left,top,right,"
                    "bottom], xlim:[..], ylim:[..]} (estimate the plot frame with view_image) to use data values."), []
        info = describe(fig)
        plt.close(fig)
        return tb.j(info), []

    @tb.add("edit_figure", "Draw on / restyle an existing plot and re-save it (e.g. add your data points in given "
            "colours, lines, labels, arrows, shaded regions; change limits, scales, labels, legend; export PDF/SVG). "
            "Figures made by the tools are edited on their real axes in data coordinates (values as displayed, "
            "e.g. °C). Op syntax: plot_ops_help.",
            {"image": S, "operations": OPS, "output": {"type": "string", "description": "new file name (default: "
             "<name>_edited.png); same name as image to overwrite"},
             "formats": {"type": "array", "items": {"type": "string"}, "description": "extra formats e.g. pdf, svg"},
             "dpi": {"type": "integer"}, "transparent": {"type": "boolean"},
             "calibration": {"type": "object", "description": "only for images without an editable copy"}},
            ["image", "operations"])
    def edit_figure(image: str, operations, output: Optional[str] = None, formats=None, dpi: Optional[int] = None,
                    calibration: Optional[Dict] = None, transparent: bool = False):
        p = tb.p(image)
        fig = load_figure(p)
        mode = "editable figure"
        if fig is None:
            fig = raster_figure(p, calibration)
            mode = "image overlay (" + ("calibrated data coordinates" if calibration else "0-1 image fractions") + ")"
        if formats is None and any(str(o.get("op", "")).lower() in LABEL_OPS for o in operations):
            formats = ["svg", "pdf"]            # phase labels: editable-text vector copies by default
        try:
            notes = apply_ops(fig, operations, resolve=tb.p)
            default = Path(p).stem + "_edited.png"
            out = tb.outp(output, default) if output else str(Path(p).with_name(default))
            if not out.lower().endswith((".png", ".jpg", ".jpeg")):
                out = str(Path(out).with_suffix(".png"))
            files = _save(fig, out, formats, dpi, raster=mode.startswith("image overlay"), transparent=transparent)
        finally:
            plt.close(fig)
        return (f"{mode}: applied {len(operations)} operation(s); saved " + ", ".join(tb.rel(f) for f in files)
                + ("\n" + "\n".join(notes) if notes else "")), [files[0]]

    panel_schema = {"type": "object", "properties": {
        "operations": OPS, "row": {"type": "integer"}, "col": {"type": "integer"}, "rowspan": {"type": "integer"},
        "colspan": {"type": "integer"},
        "projection": {"type": "string", "enum": ["rectilinear", "polar", "3d", "triangular"]}}}

    @tb.add("make_plot", "Create a new figure (one or more panels) from inline data or data files (CSV/Excel/JSON/"
            "netCDF, with transform pipelines), with full control of plot types, colours, annotations, axes, colour "
            "bars, layout (grid spans, ratios, polar/3d/triangular panels), journal presets, colour-blind palettes and "
            "export formats. Panel operations use edit_figure syntax (plain data units). Op syntax: plot_ops_help.",
            {"panels": {"type": "array", "items": panel_schema,
                        "description": "list of {operations: [...], optional row/col/rowspan/colspan, projection}"},
             "layout": {"type": "array", "items": {"type": "integer"}, "description": "[rows, cols]"},
             "width_ratios": {"type": "array", "items": {"type": "number"}},
             "height_ratios": {"type": "array", "items": {"type": "number"}},
             "figsize": {"type": "array", "items": {"type": "number"}}, "sharex": {"type": "boolean"},
             "sharey": {"type": "boolean"}, "suptitle": S,
             "style": {"type": "string", "description": "matplotlib style, e.g. seaborn-v0_8-whitegrid, ggplot, "
                       "bmh, grayscale, dark_background"},
             "preset": {"type": "string", "enum": list(PRESETS), "description": "figure size + font sizes"},
             "palette": {"description": "okabe-ito | tol-bright | tol-muted | tableau-colorblind10 | tab10 | "
                         "'<cmap>:N' | list of colours (default colour cycle)"},
             "font": {"type": "object", "description": "{family, size}"},
             "rc": {"type": "object", "description": "extra allow-listed rcParams"},
             "transparent": {"type": "boolean"},
             "output": S, "formats": {"type": "array", "items": {"type": "string"}}, "dpi": {"type": "integer"}},
            ["panels", "output"])
    def make_plot(panels, output: str, layout=None, figsize=None, sharex: bool = False, sharey: bool = False,
                  suptitle: Optional[str] = None, style: Optional[str] = None, formats=None, dpi: Optional[int] = None,
                  width_ratios=None, height_ratios=None, preset: Optional[str] = None, palette=None, font=None,
                  rc=None, transparent: bool = False):
        from matplotlib.gridspec import GridSpec
        n = len(panels)
        positioned = any(("row" in p or "col" in p) for p in panels)
        rows, cols = (layout or ([n, 1] if not positioned else [
            max(p.get("row", 0) + p.get("rowspan", 1) for p in panels),
            max(p.get("col", 0) + p.get("colspan", 1) for p in panels)]))[:2]
        if not positioned and rows * cols < n:
            raise ValueError(f"layout {rows}x{cols} has fewer cells than {n} panels")
        if style and style not in plt.style.available and style != "default":
            raise ValueError(f"unknown style {style!r}; available: {plt.style.available}")
        rcp = style_rc(preset, palette, font, rc)
        size = figsize or rcp.pop("figure.figsize", None) or (6.4 * cols, 4.4 * rows)
        if preset and not figsize and rows > 1:
            # journal presets fix the WIDTH; grow the height moderately with the number of rows
            size = (size[0], size[1] * (0.55 * rows + 0.45))
        with plt.style.context(style or "default"), plt.rc_context(rcp):
            fig = plt.figure(figsize=size)
            gs = GridSpec(rows, cols, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)
            panel_axes = []
            for i, p in enumerate(panels):
                if positioned:
                    r, c = p.get("row", 0), p.get("col", 0)
                    cell = gs[r:r + p.get("rowspan", 1), c:c + p.get("colspan", 1)]
                else:
                    cell = gs[i // cols, i % cols]
                proj = p.get("projection")
                proj = None if proj in (None, "rectilinear") else proj
                if proj == "triangular":
                    import pycalphad.plot.triangular  # noqa: F401 - registers the projection
                share = {}
                if panel_axes and proj is None:
                    if sharex:
                        share["sharex"] = panel_axes[0]
                    if sharey:
                        share["sharey"] = panel_axes[0]
                panel_axes.append(fig.add_subplot(cell, projection=proj, **share))
            try:
                notes = []
                for i, panel in enumerate(panels):
                    ops = []
                    for o in panel.get("operations", []):
                        o = dict(o, units="data")
                        if "axes" not in o:
                            o["axes"] = main_axes(fig).index(panel_axes[i])
                        ops.append(o)
                    # panel axes index may shift when earlier panels add twin/inset axes: resolve per op
                    notes += apply_ops(fig, ops, resolve=tb.p)
                    ax = panel_axes[i]
                    if not getattr(ax, "_calphad_legend_set", False) and ax.get_legend() is None and \
                            any(not a.get_label().startswith("_")
                                                       for a in ax.get_lines() + ax.collections):
                        ax.legend(fontsize=rcp.get("legend.fontsize", 9))
                if suptitle:
                    fig.suptitle(suptitle)
                try:
                    fig.tight_layout()
                except Exception:  # noqa: BLE001 - 3d/triangular axes may not support it fully
                    pass
                out = tb.outp(output if output.lower().endswith((".png", ".jpg")) else output + ".png", "plot.png")
                files = _save(fig, out, formats, dpi, transparent=transparent)
            finally:
                plt.close(fig)
        return "saved " + ", ".join(tb.rel(f) for f in files) + ("\n" + "\n".join(notes) if notes else ""), [files[0]]
