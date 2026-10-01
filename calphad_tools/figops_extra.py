"""Extra figure operations for edit_figure / make_plot (see docs/figure_extension_api.md).

Groups:
* grid & fields  - heatmap, pcolormesh, contour/contourf (extend the built-ins), tricontour/tricontourf/tripcolor
                   (scattered data, also on pycalphad triangular axes), quiver
* draw           - stackplot, bars (grouped/stacked), pie/donut, radar (polar), 3-D surface/wireframe/scatter3d/line3d
* axes & layout  - secondary_xaxis/secondary_yaxis (picklable conversions incl. mole<->mass fraction), twiny, ticks,
                   spines, view (3-D), broken_yaxis, colorbar (shared), panel_labels
* annotate       - annotate_points, label_lines

Everything attached to a figure (formatters, locators, secondary-axis functions) is a module-level class, so figures
stay picklable.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
from matplotlib import ticker as mticker

from .figures import (OpContext, ScaledFormatter, _to_data, find_col, main_axes, make_norm,  # noqa: F401
                      register_op)

G_GRID, G_DRAW, G_AXES, G_ANN = "grid & fields", "draw", "axes & layout", "annotate"
_LINE_KW = {"color", "linestyle", "linewidth", "alpha", "zorder", "marker", "markersize", "label"}
_PATCH_KW = {"edgecolor", "linewidth", "alpha", "zorder", "hatch", "linestyle"}


# ---------------------------------------------------------------------------
# picklable helpers attached to figures
# ---------------------------------------------------------------------------
class AxisMap:
    """Picklable monotonic conversion for secondary axes: value -> kind(value*pre_scale + pre_offset).

    kinds: linear (scale, offset), reciprocal (k/x), log10, pow10, mole_to_mass / mass_to_mole (binary, with
    masses m_el of the axis element and m_other; in_scale/out_scale 1 = fraction, 100 = percent)."""

    def __init__(self, kind: str, p: Dict, pre_scale: float = 1.0, pre_offset: float = 0.0, inverse: bool = False):
        self.kind, self.p, self.s, self.o, self.inverse = kind, dict(p), float(pre_scale), float(pre_offset), inverse

    # ---- elementary maps (u = value on the parent axis as displayed) ----
    def _f(self, u, inv: bool):
        k, p = self.kind, self.p
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if k == "linear":
                a, b = p.get("scale", 1.0), p.get("offset", 0.0)
                return (u - b) / a if inv else u * a + b
            if k == "reciprocal":
                u = np.array(u, dtype=float, copy=True)
                near = np.isclose(u, 0.0)
                u[near] = np.inf
                u[~near] = p.get("k", 1.0) / u[~near]
                return u
            if k == "log10":
                return np.power(10.0, u) if inv else np.log10(np.clip(u, 1e-300, None))
            if k == "pow10":
                return np.log10(np.clip(u, 1e-300, None)) if inv else np.power(10.0, u)
            if k in ("mole_to_mass", "mass_to_mole"):
                to_mass = (k == "mole_to_mass") != inv
                i_s, o_s = (p["in_scale"], p["out_scale"]) if not inv else (p["out_scale"], p["in_scale"])
                x = u / i_s
                return _comp_convert(x, p["m_el"], p["m_other"], to_mass) * o_s
        raise ValueError(f"unknown conversion {k!r}")

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if not self.inverse:
            return self._f(x * self.s + self.o, False)
        return (self._f(x, True) - self.o) / self.s


def _comp_convert(x, m_el, m_other, to_mass: bool):
    """Binary mole fraction <-> mass fraction of the axis element, linearly extended outside 0..1 (monotonic)."""
    x = np.asarray(x, dtype=float)
    a, b = (m_el, m_other) if to_mass else (1.0 / m_el, 1.0 / m_other)
    xc = np.clip(x, 0.0, 1.0)
    y = xc * a / (xc * a + (1 - xc) * b)
    y = np.where(x < 0, x * a / b, y)
    return np.where(x > 1, 1 + (x - 1) * b / a, y)


class DisplayedLocator(mticker.Locator):
    """Ticks on round *displayed* values of an axis with a ScaledFormatter (e.g. K data shown in °C)."""

    def __init__(self, scale: float = 1.0, offset: float = 0.0, nbins=None, step=None):
        self.scale, self.offset, self.nbins, self.step = scale, offset, nbins, step

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        a, b = sorted([vmin * self.scale + self.offset, vmax * self.scale + self.offset])
        base = mticker.MultipleLocator(self.step) if self.step else mticker.MaxNLocator(self.nbins or "auto")
        t = np.asarray(base.tick_values(a, b), dtype=float)
        return (t - self.offset) / self.scale


class PctFmt:
    """Picklable autopct for pie charts ('{:.1f}%' or '%.1f%%' style)."""

    def __init__(self, fmt: str):
        self.fmt = fmt

    def __call__(self, pct):
        return _fmt(self.fmt, pct)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _fmt(fmt: Optional[str], v) -> str:
    """Format a value with '{:.2f}' / '{x:.2f}' / '%.2f' / '.2f' styles."""
    if fmt is None:
        fmt = "{:.3g}"
    if "{" in fmt:
        return re.sub(r"\{x(?=[:}!])", "{", fmt).format(v)
    if "%" in fmt:
        return fmt % v
    return format(v, fmt)


def _pyfmt(fmt: str) -> str:
    """'{x:.2f}' -> '{:.2f}' (for ScaledFormatter / bar_label)."""
    return re.sub(r"\{x(?=[:}!])", "{", fmt)


def _lab(v) -> str:
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{v:g}"
    return str(v)


def _is_num(vals) -> bool:
    try:
        np.asarray(list(vals), dtype=float)
        return True
    except (TypeError, ValueError):
        return False


def _need(ctx: OpContext, names: Sequence[str], what: str):
    ax = ctx.ax
    if ax.name not in names:
        hint = {"3d": "a 3-D axes: in make_plot give the panel \"projection\": \"3d\"",
                "polar": "a polar axes: in make_plot give the panel \"projection\": \"polar\""}
        need = " or ".join(hint.get(n, f"a {n} axes") for n in names)
        raise ValueError(f"{what} needs {need} (this axes is {ax.name!r})")


def _xd(ctx: OpContext, v):
    """Displayed x positions -> data (honours coords/units)."""
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    return _to_data(ctx.ax.xaxis, v, ctx.mode).reshape(v.shape) if ctx.coords == "data" else v


def _yd(ctx: OpContext, v):
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    return _to_data(ctx.ax.yaxis, v, ctx.mode).reshape(v.shape) if ctx.coords == "data" else v


def _dscale(ctx: OpContext, axis, v):
    """Scale-only conversion (for heights / widths)."""
    f = axis.get_major_formatter()
    v = np.asarray(v, dtype=float)
    return v / f.scale if (ctx.mode == "displayed" and isinstance(f, ScaledFormatter)) else v


def _check_color(c, key="color"):
    if isinstance(c, str) and not matplotlib.colors.is_color_like(c):
        raise ValueError(f"{key}={c!r} is not a matplotlib colour (name like 'red', hex '#1f77b4', or 'C0'..'C9')")
    return c


def _colors(spec, labels: Sequence) -> Optional[List]:
    """Colours per label from None | list | {label: colour} | colormap name ('viridis' or 'viridis:N')."""
    n = len(labels)
    if spec is None:
        return None
    if isinstance(spec, dict):
        low = {str(k).lower(): v for k, v in spec.items()}
        return [_check_color(low.get(str(lab).lower(), f"C{i % 10}")) for i, lab in enumerate(labels)]
    if isinstance(spec, str):
        if matplotlib.colors.is_color_like(spec):
            return [spec] * n
        import matplotlib.pyplot as plt
        name = spec.split(":")[0]
        try:
            cmap = plt.get_cmap(name)
        except ValueError:
            raise ValueError(f"colors={spec!r} is neither a colour nor a colormap name") from None
        return [matplotlib.colors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]
    cols = list(spec)
    if not cols:
        return None
    for c in cols:
        _check_color(c)
    return [cols[i % len(cols)] for i in range(n)]


def _cmap_kw(ctx: OpContext, allow_colors: bool = False) -> Dict:
    op = ctx.op
    kw: Dict = {}
    if allow_colors and op.get("colors") is not None:
        cols = op["colors"]
        kw["colors"] = _check_color(cols, "colors") if isinstance(cols, str) else [_check_color(c, "colors") for c in cols]
        return kw
    kw["cmap"] = op.get("cmap", "viridis")
    if op.get("norm") is not None:
        spec = op["norm"]
        if isinstance(spec, dict):
            spec = {"vmin": op.get("vmin"), "vmax": op.get("vmax"), **spec}
        elif isinstance(spec, str) and (op.get("vmin") is not None or op.get("vmax") is not None):
            spec = {"type": spec, "vmin": op.get("vmin"), "vmax": op.get("vmax")}
        kw["norm"] = make_norm(spec)
    else:
        for k in ("vmin", "vmax"):
            if op.get(k) is not None:
                kw[k] = op[k]
    return kw


def _order(vals) -> list:
    """Unique values: sorted if numeric, else in order of appearance."""
    import pandas as pd
    u = list(pd.unique(np.asarray(vals, dtype=object)))
    if _is_num(u):
        return sorted(u, key=float)
    return u


def _pivot(ctx: OpContext, rkey: str, ckey: str, vkey: str, fill=np.nan):
    """Long data -> (row values, column values, 2-D array [rows, cols]) (mean of duplicates)."""
    df = ctx.frame()
    rc, cc, vc = ctx.colname(rkey), ctx.colname(ckey), ctx.colname(vkey)
    if not (rc and cc and vc):
        raise ValueError(f"long data needs {rkey}_col, {ckey}_col and {vkey}_col")
    p = df.pivot_table(index=rc, columns=cc, values=vc, aggfunc="mean", sort=False)
    p = p.reindex(index=_order(df[rc]), columns=_order(df[cc]))
    if not (isinstance(fill, float) and math.isnan(fill)):
        p = p.fillna(fill)
    return p.index.to_numpy(), p.columns.to_numpy(), p.to_numpy(dtype=float)


def _grid(ctx: OpContext):
    """(x, y, Z, scattered) for grid ops: inline x, y (1-D or 2-D) + z 2-D, or long data x_col/y_col/z_col.

    scattered=True (x, y, z 1-D point lists) when z is 1-D or the long data do not form a grid."""
    op = ctx.op
    if ctx.has_source and op.get("z_col"):
        df = ctx.frame()
        ys, xs, Z = _pivot(ctx, "y", "x", "z")
        if not (_is_num(xs) and _is_num(ys)):
            raise ValueError("x/y columns must be numeric (use heatmap for categories)")
        if np.isnan(Z).mean() > 0.3 and len(df) >= 3:        # not a grid: scattered points
            return ctx.X(), ctx.Y(), ctx.raw("z"), True
        return _xd(ctx, xs.astype(float)), _yd(ctx, ys.astype(float)), Z, False
    if op.get("z") is None:
        raise ValueError("give z (2-D list [len(y)][len(x)]) with x and y, or data + x_col, y_col, z_col")
    Z = np.asarray(op["z"], dtype=float)
    x, y = ctx.X(), ctx.Y()
    if Z.ndim == 1:
        if x is None or y is None or not (len(x) == len(y) == len(Z)):
            raise ValueError("1-D z needs x and y of the same length (scattered points)")
        return x, y, Z, True
    if Z.ndim != 2:
        raise ValueError(f"z must be 2-D, got shape {Z.shape}")
    if x is None:
        x = np.arange(Z.shape[1], dtype=float)
    if y is None:
        y = np.arange(Z.shape[0], dtype=float)
    if x.ndim == 1 and y.ndim == 1 and Z.shape != (len(y), len(x)):
        if Z.shape == (len(x), len(y)):
            Z = Z.T
            ctx.note("z was [len(x)][len(y)]; transposed to [len(y)][len(x)]")
        elif Z.shape not in ((len(y) - 1, len(x) - 1),):
            raise ValueError(f"z shape {Z.shape} does not match len(y)={len(y)} x len(x)={len(x)}")
    return x, y, Z, False


# ---------------------------------------------------------------------------
# grid & fields
# ---------------------------------------------------------------------------
@register_op("heatmap", "heatmap {z: 2-D [rows][cols], x, y (column/row labels) | data + x_col, y_col, z_col; "
             "cmap, norm, vmin, vmax, aspect (auto), origin (lower if y numeric else upper), annotate: true, fmt "
             "('{:.2f}'), fontsize, xrotation, colorbar}", group=G_GRID)
def _heatmap(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    _need(ctx, ["rectilinear"], "heatmap")
    if ctx.has_source and op.get("z_col"):
        ylab, xlab, Z = _pivot(ctx, "y", "x", "z")
    else:
        if op.get("z") is None:
            raise ValueError("give z (2-D list) or data + x_col, y_col, z_col")
        Z = np.asarray(op["z"], dtype=float)
        if Z.ndim != 2:
            raise ValueError(f"z must be 2-D, got shape {Z.shape}")
        xlab = op.get("x") or op.get("xlabels") or list(range(Z.shape[1]))
        ylab = op.get("y") or op.get("ylabels") or list(range(Z.shape[0]))
        if len(xlab) != Z.shape[1] or len(ylab) != Z.shape[0]:
            raise ValueError(f"z is {Z.shape[0]}x{Z.shape[1]} but got {len(ylab)} y and {len(xlab)} x labels")
    origin = op.get("origin", "lower" if _is_num(ylab) else "upper")
    kw = _cmap_kw(ctx)
    im = ax.imshow(np.ma.masked_invalid(Z), aspect=op.get("aspect", "auto"), origin=origin,
                   interpolation=op.get("interpolation", "nearest"), **kw, **ctx.pick({"alpha", "zorder"}))
    nx, ny = Z.shape[1], Z.shape[0]
    sx, sy = max(1, math.ceil(nx / 25)), max(1, math.ceil(ny / 25))
    ax.set_xticks(np.arange(0, nx, sx), [_lab(v) for v in list(xlab)[::sx]])
    ax.set_yticks(np.arange(0, ny, sy), [_lab(v) for v in list(ylab)[::sy]])
    rot = op.get("xrotation", 45 if max((len(_lab(v)) for v in xlab), default=0) > 4 else 0)
    if rot:
        for t in ax.get_xticklabels():
            t.set_rotation(rot)
            t.set_ha("right")
    if op.get("annotate"):
        fs = op.get("fontsize", 8 if nx * ny <= 100 else 6)
        for i in range(ny):
            for j in range(nx):
                v = Z[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = im.cmap(im.norm(v))
                col = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
                ax.text(j, i, _fmt(op.get("fmt"), v), ha="center", va="center", fontsize=fs, color=col)
    ctx.colorbar(im)


@register_op("pcolormesh", "pcolormesh {x, y (centres or edges; 1-D or 2-D), z 2-D | data + x_col, y_col, z_col; "
             "shading auto|nearest|flat|gouraud, cmap, norm, vmin, vmax, edgecolor, linewidth, colorbar}",
             group=G_GRID)
def _pcolormesh(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    x, y, Z, scattered = _grid(ctx)
    if scattered:
        raise ValueError("pcolormesh needs gridded data; use tripcolor for scattered points")
    kw = _cmap_kw(ctx)
    kw.update(ctx.pick({"edgecolor", "linewidth", "alpha", "zorder"}))
    if "edgecolor" in kw:
        kw["edgecolors"] = kw.pop("edgecolor")
    if "linewidth" in kw:
        kw["linewidths"] = kw.pop("linewidth")
    qm = ax.pcolormesh(x, y, np.ma.masked_invalid(Z), shading=op.get("shading", "auto"), **kw)
    ctx.colorbar(qm)


def _contour_common(ctx: OpContext, filled: bool, x=None, y=None, Z=None, tri=None):
    op, ax = ctx.op, ctx.ax
    kw = _cmap_kw(ctx, allow_colors=True)
    kw.update(ctx.pick({"alpha", "zorder"}))
    lv = op.get("levels", 10)
    levels = int(lv) if isinstance(lv, (int, float)) else [float(v) for v in lv]
    if filled:
        if op.get("extend"):
            kw["extend"] = op["extend"]
        if op.get("hatches"):
            kw["hatches"] = op["hatches"]
    else:
        for k in ("linestyles", "linewidths"):
            if op.get(k) is not None:
                kw[k] = op[k]
    if tri is not None:
        fn = ax.tricontourf if filled else ax.tricontour
        cs = fn(tri, Z, levels=levels, **kw)
    else:
        fn = ax.contourf if filled else ax.contour
        cs = fn(x, y, np.ma.masked_invalid(Z), levels=levels, **kw)
    lines = cs if not filled else None
    if filled and op.get("lines"):
        lkw = {"colors": op.get("line_colors", "k"), "linewidths": op.get("linewidths", 0.6)}
        if op.get("linestyles") is not None:
            lkw["linestyles"] = op["linestyles"]
        lines = (ax.tricontour(tri, Z, levels=cs.levels, **lkw) if tri is not None
                 else ax.contour(x, y, np.ma.masked_invalid(Z), levels=cs.levels, **lkw))
    if lines is not None and op.get("clabel", not filled):
        lf = op.get("clabel_fmt")
        ax.clabel(lines, fontsize=op.get("fontsize", 8), fmt=_pyfmt(lf) if lf else None)
    if op.get("label"):
        if filled:
            cs.set_label(op["label"])
        else:
            (lines or cs).set_label(op["label"])
    ctx.colorbar(cs)
    return cs


def _contour_op(filled: bool):
    def fn(ctx: OpContext):
        x, y, Z, scattered = _grid(ctx)
        if scattered:
            ctx.note("scattered x, y, z: drawn with a triangulation (tricontour)")
            return _tri_draw(ctx, "tricontourf" if filled else "tricontour", x, y, Z)
        _contour_common(ctx, filled, x, y, Z)
    return fn


_CONTOUR_KEYS = ("x, y (1-D grid axes or 2-D), z (2-D [len(y)][len(x)]; 1-D = scattered points) | data + x_col, "
                 "y_col, z_col (long format, pivoted); levels (int or list), cmap, norm, vmin, vmax, colors, alpha")
register_op("contour", "contour {" + _CONTOUR_KEYS + ", linestyles, linewidths, clabel (default true), clabel_fmt "
            "('{:.0f}'), label, colorbar}", group=G_GRID)(_contour_op(False))
register_op("contourf", "contourf {" + _CONTOUR_KEYS + ", extend, hatches, lines: true (+ line_colors, linewidths, "
            "linestyles, clabel), colorbar}", group=G_GRID)(_contour_op(True))


def _tri_draw(ctx: OpContext, kind: str, x, y, z):
    from matplotlib.tri import Triangulation
    op, ax = ctx.op, ctx.ax
    x, y, z = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"x, y, z lengths differ ({len(x)}, {len(y)}, {len(z)})")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if op.get("triangle", ax.name == "triangular"):
        tol = 1e-9
        inside = (x >= -tol) & (y >= -tol) & (x + y <= 1 + tol)
        if (ok & ~inside).any():
            ctx.note(f"dropped {int((ok & ~inside).sum())} point(s) outside the composition triangle")
        ok &= inside
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 3:
        raise ValueError("need at least 3 finite points to triangulate")
    try:
        tri = Triangulation(x, y)
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"cannot triangulate the points (collinear or duplicated?): {e}") from None
    if op.get("max_edge"):
        t = tri.triangles
        px, py = x[t], y[t]
        edges = np.hypot(px - np.roll(px, 1, axis=1), py - np.roll(py, 1, axis=1)).max(axis=1)
        tri.set_mask(edges > float(op["max_edge"]))
    if kind == "tripcolor":
        kw = _cmap_kw(ctx)
        kw.update(ctx.pick({"alpha", "zorder", "edgecolors"}))
        m = ax.tripcolor(tri, z, shading=op.get("shading", "flat"), **kw)
        ctx.colorbar(m)
        return m
    return _contour_common(ctx, kind == "tricontourf", Z=z, tri=tri)


def _tri_op(kind: str):
    def fn(ctx: OpContext):
        _tri_draw(ctx, kind, ctx.X(), ctx.Y(), ctx.raw("z"))
    return fn


_TRI_KEYS = ("x, y, z (scattered points, 1-D) | data + x_col, y_col, z_col; triangle: true (drop points outside "
             "the composition triangle; default on triangular axes), max_edge (mask long triangles), cmap, norm, "
             "vmin, vmax, alpha, colorbar")
register_op("tricontour", "tricontour {" + _TRI_KEYS + ", levels, colors, linestyles, linewidths, clabel}",
            group=G_GRID)(_tri_op("tricontour"))
register_op("tricontourf", "tricontourf {" + _TRI_KEYS + ", levels, colors, extend, lines: true, clabel}",
            group=G_GRID)(_tri_op("tricontourf"))
register_op("tripcolor", "tripcolor {" + _TRI_KEYS + ", shading flat|gouraud}", group=G_GRID)(_tri_op("tripcolor"))


@register_op("quiver", "quiver {x, y (1-D grid axes or point lists), u, v (2-D grid or per point) | data + x_col, "
             "y_col, u_col, v_col; c (values -> cmap), color, scale, scale_units, angles, width, pivot, "
             "key: {U, label, x, y (axes fractions)}, colorbar}", group=G_GRID)
def _quiver(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    x, y, u, v = ctx.X(), ctx.Y(), ctx.raw("u"), ctx.raw("v")
    if any(a is None for a in (x, y, u, v)):
        raise ValueError("quiver needs x, y, u, v")
    if u.ndim == 2 and x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)
    kw = {k: op[k] for k in ("scale", "scale_units", "angles", "width", "pivot", "headwidth", "headlength") if k in op}
    kw.update(ctx.pick({"color", "alpha", "zorder", "label"}))
    c = ctx.raw("c")
    if c is not None:
        kw.pop("color", None)
        kw.update(_cmap_kw(ctx))
        q = ax.quiver(x, y, u, v, c, **kw)
        ctx.colorbar(q)
    else:
        q = ax.quiver(x, y, u, v, **kw)
    key = op.get("key")
    if key:
        ax.quiverkey(q, key.get("x", 0.9), key.get("y", 1.03), key.get("U", 1.0), key.get("label", str(key.get("U", 1))),
                     labelpos=key.get("labelpos", "E"), coordinates="axes")


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------
@register_op("stackplot", "stackplot {x, ys: [[..], ..] (one list per layer), labels | data + x_col + y_cols: [..] | "
             "data + x_col, group_col, value_col (long, e.g. T, phase, NP); normalize: true (layers sum to 1), colors "
             "(list | {label: colour} | cmap name), baseline zero|sym|wiggle, alpha, edgecolor, linewidth}",
             group=G_DRAW)
def _stackplot(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    _need(ctx, ["rectilinear"], "stackplot")
    if ctx.has_source and op.get("group_col"):
        vkey = "value" if op.get("value_col") else "y"
        xs, labels, M = _pivot(ctx, "x", "group", vkey, fill=0.0)
        x = np.asarray(xs, dtype=float)
        order = np.argsort(x)
        x, ys = x[order], M[order].T
        labels = [str(v) for v in labels]
    elif ctx.has_source and op.get("y_cols"):
        df = ctx.frame()
        src = op.get("csv") or op.get("data")
        cols = [find_col(df, c, src) for c in op["y_cols"]]
        x = ctx.raw("x")
        ys = np.vstack([df[c].to_numpy(dtype=float) for c in cols])
        labels = op.get("labels") or [str(c) for c in cols]
    else:
        x = ctx.raw("x")
        ys = op.get("ys", op.get("y"))
        if x is None or ys is None:
            raise ValueError("give x + ys (list of layers), or data + x_col + y_cols, or long data + group_col")
        ys = np.atleast_2d(np.asarray(ys, dtype=float))
        labels = op.get("labels") or [f"layer {i + 1}" for i in range(len(ys))]
    ys = np.nan_to_num(np.asarray(ys, dtype=float))
    if ys.shape[1] != len(x):
        raise ValueError(f"each layer needs {len(x)} values (len(x)); got {ys.shape[1]}")
    if len(labels) != len(ys):
        raise ValueError(f"{len(ys)} layers but {len(labels)} labels")
    if op.get("normalize"):
        tot = ys.sum(axis=0)
        ys = np.divide(ys, tot, out=np.zeros_like(ys), where=tot != 0)
    if op.get("drop_zero", True):
        keep = np.abs(ys).max(axis=1) > 0
        ys, labels = ys[keep], [lab for lab, k in zip(labels, keep) if k]
    kw = ctx.pick({"alpha", "edgecolor", "linewidth", "zorder"})
    cols = _colors(op.get("colors"), labels)
    if cols:
        kw["colors"] = cols
    ax.stackplot(_xd(ctx, x), _dscale(ctx, ax.yaxis, ys), labels=labels, baseline=op.get("baseline", "zero"), **kw)


def _series(ctx: OpContext, cat_key="category"):
    """(categories, {series name: values}) from {categories, series} or long data category_col/group_col/value_col."""
    op = ctx.op
    if ctx.has_source and op.get(f"{cat_key}_col"):
        vkey = next((k for k in ("value", "y", "height") if op.get(f"{k}_col")), None)
        if vkey is None:
            raise ValueError("long data needs value_col (with category_col and optional group_col)")
        if op.get("group_col"):
            cats, groups, M = _pivot(ctx, cat_key, "group", vkey)
            series = {str(g): M[:, j] for j, g in enumerate(groups)}
        else:
            df = ctx.frame()
            cc, vc = ctx.colname(cat_key), ctx.colname(vkey)
            s = df.groupby(cc, sort=False)[vc].mean()
            cats = list(s.index)
            series = {op.get("label", str(vc)): s.to_numpy(dtype=float)}
        return [_lab(c) for c in cats], series
    cats = op.get("categories")
    series = op.get("series")
    if series is None and op.get("values") is not None:
        series = {op.get("label", "values"): op["values"]}
    if cats is None or series is None:
        raise ValueError("give categories + series {name: [values]}, or data + category_col, value_col (+ group_col)")
    if not isinstance(series, dict):
        series = {f"series {i + 1}": v for i, v in enumerate(series)}
    out = {}
    for k, v in series.items():
        v = np.asarray(v, dtype=float)
        if len(v) != len(cats):
            raise ValueError(f"series {k!r} has {len(v)} values for {len(cats)} categories")
        out[str(k)] = v
    return [str(c) for c in cats], out


@register_op("bars", "bars {categories, series: {name: [values]} | data + category_col, value_col, group_col; "
             "mode grouped|stacked, horizontal, width (0.8 per group), colors (list | {name: colour} | cmap), errors: "
             "{name: [..]}, value_labels: true | fmt ('{:.1f}'), label_fontsize, rotation, edgecolor, alpha, hatch}",
             group=G_DRAW)
def _bars(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    _need(ctx, ["rectilinear"], "bars")
    cats, series = _series(ctx)
    names = list(series)
    mode = op.get("mode", "grouped")
    if mode not in ("grouped", "stacked"):
        raise ValueError("mode must be grouped or stacked")
    horiz = bool(op.get("horizontal"))
    num_axis = ax.xaxis if horiz else ax.yaxis
    pos = np.arange(len(cats), dtype=float)
    width = float(op.get("width", 0.8))
    cols = _colors(op.get("colors"), names) or [None] * len(names)
    errs = op.get("errors") or {}
    kw = ctx.pick(_PATCH_KW)
    pos_base, neg_base = np.zeros(len(cats)), np.zeros(len(cats))
    vl = op.get("value_labels")
    for i, name in enumerate(names):
        vals = np.nan_to_num(_dscale(ctx, num_axis, series[name]))
        if mode == "grouped":
            w = width / len(names)
            p, base = pos - width / 2 + w * (i + 0.5), None
        else:
            w, p = width, pos
            base = np.where(vals >= 0, pos_base, neg_base)
        e = errs.get(name)
        e = _dscale(ctx, num_axis, e) if e is not None else None
        ekw = {"capsize": op.get("capsize", 3)} if e is not None else {}
        if horiz:
            cont = ax.barh(p, vals, height=w, left=base, xerr=e, color=cols[i], label=name, **ekw, **kw)
        else:
            cont = ax.bar(p, vals, width=w, bottom=base, yerr=e, color=cols[i], label=name, **ekw, **kw)
        if mode == "stacked":
            pos_base = pos_base + np.where(vals >= 0, vals, 0)
            neg_base = neg_base + np.where(vals < 0, vals, 0)
        if vl:
            ax.bar_label(cont, fmt=_pyfmt(vl) if isinstance(vl, str) else "%g",
                         label_type="center" if mode == "stacked" else "edge",
                         fontsize=op.get("label_fontsize", 8), padding=2)
    if horiz:
        ax.set_yticks(pos, cats)
        ax.invert_yaxis() if op.get("top_down", True) else None
    else:
        ax.set_xticks(pos, cats)
        rot = op.get("rotation", 30 if max(len(c) for c in cats) > 6 else 0)
        if rot:
            for t in ax.get_xticklabels():
                t.set_rotation(rot)
                t.set_ha("right")
    if len(names) == 1 and not op.get("legend", False):
        _hide_legend(ax)


def _hide_legend(ax):
    """Hide the legend and stop apply_ops/make_plot from re-adding one."""
    leg = ax.get_legend()
    if leg is None and ax.get_legend_handles_labels()[0]:
        leg = ax.legend()
    if leg is not None:
        leg.set_visible(False)
    ax._calphad_legend_set = True


@register_op("pie", "pie {values, labels | data + values_col, labels_col; colors, autopct: true | '{:.1f}%', "
             "width (donut ring width 0-1), startangle (90), counterclock (false), explode: [..], pctdistance, "
             "labeldistance, fontsize, legend: true (labels in a legend), center_text}", group=G_DRAW)
def _pie(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    _need(ctx, ["rectilinear"], "pie")
    vals = ctx.raw("values")
    if vals is None:
        raise ValueError("pie needs values (or data + values_col)")
    labels = ctx.raw("labels", dtype=None)
    labels = [str(v) for v in labels] if labels is not None else None
    if labels is not None and len(labels) != len(vals):
        raise ValueError(f"{len(vals)} values but {len(labels)} labels")
    kw = {"startangle": op.get("startangle", 90), "counterclock": op.get("counterclock", False)}
    for k in ("explode", "pctdistance", "labeldistance"):
        if k in op:
            kw[k] = op[k]
    cols = _colors(op.get("colors"), labels or list(range(len(vals))))
    if cols:
        kw["colors"] = cols
    ap = op.get("autopct")
    if ap:
        kw["autopct"] = PctFmt(ap if isinstance(ap, str) else "{:.1f}%")
    wp = {"edgecolor": op.get("edgecolor", "white"), "linewidth": op.get("linewidth", 1)}
    if op.get("width"):
        wp["width"] = float(op["width"])
        kw.setdefault("pctdistance", 1 - float(op["width"]) / 2)
    as_legend = bool(op.get("legend"))
    res = ax.pie(vals, labels=None if as_legend else labels, wedgeprops=wp,
                 textprops={"fontsize": op.get("fontsize", 9)}, **kw)
    wedges = res[0]
    for w, lab in zip(wedges, labels or []):
        w.set_label(lab if as_legend else "_" + lab)
    if as_legend:
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=op.get("fontsize", 9))
        ax._calphad_legend_set = True
    else:
        _hide_legend(ax)
    if op.get("center_text"):
        ax.text(0, 0, str(op["center_text"]), ha="center", va="center", fontsize=op.get("fontsize", 9) + 2)
    ax.set_aspect("equal")


@register_op("radar", "radar {categories, series: {name: [values]} | data + category_col, value_col, group_col; "
             "colors, fill (true), alpha (0.15), linewidth, marker, rlim: [lo, hi], rticks, clockwise (true), legend} "
             "(needs a polar panel)", group=G_DRAW)
def _radar(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    _need(ctx, ["polar"], "radar")
    cats, series = _series(ctx)
    n = len(cats)
    if n < 3:
        raise ValueError("radar needs at least 3 categories")
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1 if op.get("clockwise", True) else 1)
    names = list(series)
    cols = _colors(op.get("colors"), names) or [f"C{i % 10}" for i in range(len(names))]
    for name, col in zip(names, cols):
        v = np.asarray(series[name], dtype=float)
        a2, v2 = np.append(ang, ang[0]), np.append(v, v[0])
        ax.plot(a2, v2, color=col, linewidth=op.get("linewidth", 1.5), marker=op.get("marker", "o"),
                markersize=op.get("markersize", 4), label=name)
        if op.get("fill", True):
            ax.fill(a2, v2, color=col, alpha=op.get("alpha", 0.15), label="_" + name)
    ax.set_xticks(ang, cats)
    if op.get("rlim"):
        ax.set_ylim(*op["rlim"])
    if op.get("rticks") is not None:
        ax.set_yticks(op["rticks"])
    ax.tick_params(axis="y", labelsize=op.get("fontsize", 8))
    if op.get("legend", True):            # outside: the radial labels fill the polar frame
        ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.05), fontsize=op.get("fontsize", 8) + 1)
        ax._calphad_legend_set = True


def _grid3d(ctx: OpContext):
    x, y, Z, scattered = _grid(ctx)
    return x, y, Z, scattered


def _colorbar3d(ctx: OpContext, mappable):
    """Colour bar with extra padding (3-D axes put the z label where a default bar would sit)."""
    spec = ctx.op.get("colorbar")
    if spec:
        ctx.op = dict(ctx.op, colorbar={"pad": 0.12, "shrink": 0.7, **(spec if isinstance(spec, dict) else {})})
    ctx.colorbar(mappable)


@register_op("surface", "surface {x, y (1-D), z 2-D [len(y)][len(x)] | data + x_col, y_col, z_col (grid or "
             "scattered); cmap, norm, vmin, vmax, color, alpha, edgecolor, linewidth, colorbar} (3-D panel)",
             group=G_DRAW)
def _surface(ctx: OpContext):
    _need(ctx, ["3d"], "surface")
    op, ax = ctx.op, ctx.ax
    x, y, Z, scattered = _grid3d(ctx)
    kw = ctx.pick({"alpha", "edgecolor", "linewidth", "zorder"})
    if op.get("color") and not op.get("cmap"):
        kw["color"] = _check_color(op["color"])
    else:
        kw.update(_cmap_kw(ctx))
    if scattered:
        s = ax.plot_trisurf(np.ravel(x), np.ravel(y), np.ravel(Z), **kw)
    else:
        X, Y = (x, y) if np.ndim(x) == 2 else np.meshgrid(x, y)
        s = ax.plot_surface(X, Y, np.ma.masked_invalid(Z), rstride=op.get("rstride", 1), cstride=op.get("cstride", 1),
                            **kw)
    if op.get("label"):
        s.set_label(op["label"])
    _colorbar3d(ctx, s)


@register_op("wireframe", "wireframe {x, y, z grid | data + x_col, y_col, z_col; color, linewidth, rstride, cstride, "
             "label} (3-D panel)", group=G_DRAW)
def _wireframe(ctx: OpContext):
    _need(ctx, ["3d"], "wireframe")
    op, ax = ctx.op, ctx.ax
    x, y, Z, scattered = _grid3d(ctx)
    if scattered:
        raise ValueError("wireframe needs gridded data (use surface for scattered points)")
    X, Y = (x, y) if np.ndim(x) == 2 else np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, Z, rstride=op.get("rstride", 1), cstride=op.get("cstride", 1),
                      **ctx.pick({"color", "linewidth", "alpha", "label", "zorder", "linestyle"}))


@register_op("scatter3d", "scatter3d {x, y, z | data + x_col, y_col, z_col; c (values -> cmap), s, color, marker, "
             "cmap, norm, alpha, label, depthshade, colorbar} (3-D panel)", group=G_DRAW)
def _scatter3d(ctx: OpContext):
    _need(ctx, ["3d"], "scatter3d")
    op, ax = ctx.op, ctx.ax
    x, y, z = ctx.X(), ctx.Y(), ctx.raw("z")
    if x is None or y is None or z is None:
        raise ValueError("scatter3d needs x, y, z")
    kw = ctx.pick({"color", "marker", "alpha", "label", "zorder", "edgecolor"})
    s = ctx.raw("s")
    if s is not None:
        kw["s"] = s
    c = ctx.raw("c")
    if c is not None:
        kw.pop("color", None)
        kw["c"] = c
        kw.update(_cmap_kw(ctx))
    sc = ax.scatter(x, y, z, depthshade=op.get("depthshade", True), **kw)
    if c is not None:
        _colorbar3d(ctx, sc)


@register_op("line3d", "line3d {x, y, z | data + x_col, y_col, z_col; color, linestyle, linewidth, marker, label} "
             "(3-D panel)", group=G_DRAW)
def _line3d(ctx: OpContext):
    _need(ctx, ["3d"], "line3d")
    x, y, z = ctx.X(), ctx.Y(), ctx.raw("z")
    if x is None or y is None or z is None:
        raise ValueError("line3d needs x, y, z")
    ctx.ax.plot(x, y, z, **ctx.pick(_LINE_KW))


@register_op("view", "view {elev, azim, roll, zlabel, zlim, xlabel, ylabel, box_aspect: [x, y, z], proj_type "
             "persp|ortho, zscale} (3-D panel)", group=G_AXES)
def _view(ctx: OpContext):
    _need(ctx, ["3d"], "view")
    op, ax = ctx.op, ctx.ax
    if any(k in op for k in ("elev", "azim", "roll")):
        ax.view_init(elev=op.get("elev", ax.elev), azim=op.get("azim", ax.azim), roll=op.get("roll", ax.roll))
    for k in ("xlabel", "ylabel", "zlabel"):
        if k in op:
            getattr(ax, f"set_{k}")(op[k])
    if op.get("zlim"):
        ax.set_zlim(*op["zlim"])
    if op.get("box_aspect"):
        ax.set_box_aspect(op["box_aspect"])
    if op.get("proj_type"):
        ax.set_proj_type(op["proj_type"])
    if op.get("zscale"):
        ax.set_zscale(op["zscale"])


# ---------------------------------------------------------------------------
# axes & layout
# ---------------------------------------------------------------------------
_EL_RX = re.compile(r"(?:X\(|W\(|fraction\s+|percent\s+|at\.?\s*%\s*|wt\.?\s*%\s*)([A-Z][a-z]?)\b", re.I)


def _axis_map(ctx: OpContext, axis, which: str):
    """(forward, inverse) AxisMap pair for op['convert'] on the given parent axis."""
    from .expdata import ATOMIC_MASS
    op = ctx.op
    conv = op.get("convert", "linear")
    if isinstance(conv, dict):          # also accept {"type": "linear", "scale": 100, ...}
        op = {**op, **{k: v for k, v in conv.items() if k != "type"}}
        conv = conv.get("type", conv.get("kind", "linear"))
    kind = str(conv).lower()
    f = axis.get_major_formatter()
    s, o = (f.scale, f.offset) if (op.get("of", "displayed") == "displayed" and isinstance(f, ScaledFormatter)) \
        else (1.0, 0.0)
    p: Dict = {}
    if kind == "linear":
        p = {"scale": float(op.get("scale", 1.0)), "offset": float(op.get("offset", 0.0))}
        if p["scale"] == 0:
            raise ValueError("scale must be non-zero")
    elif kind == "reciprocal":
        p = {"k": float(op.get("k", 1000.0))}
    elif kind in ("mole_to_mass", "mass_to_mole"):
        el = op.get("element")
        if not el:
            lab = ctx.ax.get_xlabel() if which == "x" else ctx.ax.get_ylabel()
            m = _EL_RX.search(lab or "")
            if not m:
                raise ValueError("give element (the element on this axis) and other (the second component)")
            el = m.group(1)
        other = op.get("other")
        if not other:
            # binary diagrams are titled like "Cr-Ni" / "Cr–Ni (...)": take the element that is not on the axis
            titles = [ctx.ax.get_title()] + ([ctx.fig._suptitle.get_text()] if getattr(ctx.fig, "_suptitle", None) else [])
            for t in titles:
                syms = [s.upper() for s in re.findall(r"\b([A-Z][a-z]?)\b", re.split(r"\s{2,}|\(", t or "")[0])
                        if s.upper() in ATOMIC_MASS]
                cands = [s for s in dict.fromkeys(syms) if s != str(el).upper()]
                if len(syms) == 2 and len(cands) == 1:
                    other = cands[0]
                    ctx.note(f"second element taken from the title: {other}")
                    break
        if not other:
            raise ValueError("give other: the second element of the binary (e.g. 'CR' for a Cr-Ni diagram in X(NI))")
        try:
            m_el, m_other = ATOMIC_MASS[str(el).upper()], ATOMIC_MASS[str(other).upper()]
        except KeyError as e:
            raise ValueError(f"unknown element {e.args[0]!r}") from None
        lo, hi = axis.get_view_interval()
        disp_hi = max(abs(lo * s + o), abs(hi * s + o))
        in_scale = float(op.get("in_scale", 100.0 if disp_hi > 1.5 else 1.0))
        pct = op.get("percent")
        out_scale = float(op.get("out_scale", (100.0 if pct else 1.0) if pct is not None else in_scale))
        p = {"m_el": m_el, "m_other": m_other, "in_scale": in_scale, "out_scale": out_scale, "element": str(el)}
    elif kind not in ("log10", "pow10"):
        raise ValueError(f"convert must be linear, reciprocal, log10, pow10, mole_to_mass or mass_to_mole, not {kind!r}")
    return AxisMap(kind, p, s, o), AxisMap(kind, p, s, o, inverse=True)


def _secondary(which: str):
    def fn(ctx: OpContext):
        op, ax = ctx.op, ctx.ax
        _need(ctx, ["rectilinear"], f"secondary_{which}axis")
        axis = ax.xaxis if which == "x" else ax.yaxis
        fwd, inv = _axis_map(ctx, axis, which)
        loc = op.get("location", "top" if which == "x" else "right")
        sec = (ax.secondary_xaxis if which == "x" else ax.secondary_yaxis)(loc, functions=(fwd, inv))
        sax = sec.xaxis if which == "x" else sec.yaxis
        if op.get("label"):
            (sec.set_xlabel if which == "x" else sec.set_ylabel)(op["label"])
        if op.get("format"):
            sax.set_major_formatter(_formatter(op["format"], op))
        if op.get("ticks") is not None:
            sax.set_ticks(op["ticks"])
        if op.get("color"):
            c = _check_color(op["color"])
            sec.tick_params(axis=which, colors=c)
            (sec.xaxis if which == "x" else sec.yaxis).label.set_color(c)
        if op.get("labelsize"):
            sec.tick_params(axis=which, labelsize=op["labelsize"])
        leg = ax.get_legend()
        if which == "y" and loc == "right" and leg is not None and leg.get_visible():
            try:
                bb = leg.get_window_extent(ctx.fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
                if bb.x0 > 0.98:
                    ctx.note("the legend sits outside on the right, over the new axis: move it, e.g. "
                             "legend {loc: 'upper left', outside: false} or legend {loc: 'lower right'}")
            except Exception:  # noqa: BLE001 - advisory only
                pass
        if fwd.kind in ("mole_to_mass", "mass_to_mole"):
            ctx.note(f"secondary {which}-axis: {fwd.kind} for {fwd.p['element']} (x{fwd.p['in_scale']:g} -> "
                     f"x{fwd.p['out_scale']:g})")
        else:
            ctx.note(f"secondary {which}-axis added ({fwd.kind})")
    return fn


_SEC_KEYS = ("convert: linear {scale, offset} (e.g. K->°C offset -273.15; fraction->% scale 100) | reciprocal {k: "
             "1000} | log10 | pow10 | mole_to_mass | mass_to_mole {element (default from the axis label), other, "
             "percent: true}; of: displayed|data (input values, default displayed), label, format, ticks, color, "
             "labelsize")
register_op("secondary_xaxis", "secondary_xaxis {location: top|bottom|y-fraction, " + _SEC_KEYS + "}",
            group=G_AXES)(_secondary("x"))
register_op("secondary_yaxis", "secondary_yaxis {location: right|left|x-fraction, " + _SEC_KEYS + "}",
            group=G_AXES)(_secondary("y"))


@register_op("twiny", "twiny {xlabel, xlim}: new x-axis (top) sharing y (new axes index reported)", group=G_AXES)
def _twiny(ctx: OpContext):
    _need(ctx, ["rectilinear"], "twiny")
    t = ctx.ax.twiny()
    if ctx.op.get("xlabel"):
        t.set_xlabel(ctx.op["xlabel"])
    if ctx.op.get("xlim"):
        t.set_xlim(*ctx.op["xlim"])
    t._calphad_legend_set = True
    ctx.new_axes(t, "twin x-axis")


def _formatter(spec: str, op: Dict, scaled: Optional[ScaledFormatter] = None):
    s = str(spec)
    low = s.lower()
    if scaled is not None:          # keep the display transform (e.g. K shown as °C)
        if "{" in s:
            return ScaledFormatter(scaled.scale, scaled.offset, _pyfmt(s))
        if low in ("plain", "g"):
            return ScaledFormatter(scaled.scale, scaled.offset, "{:g}")
        if low == "sci":
            return ScaledFormatter(scaled.scale, scaled.offset, "{:.2e}")
        raise ValueError(f"this axis shows data*{scaled.scale:g}{scaled.offset:+g}; use a format string like "
                         "'{x:.0f}' (or plain/sci) so the display transform is kept")
    if "{" in s:
        return mticker.StrMethodFormatter(s if "{x" in s else _pyfmt(s).replace("{", "{x", 1))
    if "%" in s and low not in ("percent",):
        return mticker.FormatStrFormatter(s)
    if low == "sci":
        f = mticker.ScalarFormatter(useMathText=True)
        f.set_powerlimits((0, 0))
        return f
    if low == "plain":
        f = mticker.ScalarFormatter(useOffset=False)
        f.set_scientific(False)
        return f
    if low == "percent":
        return mticker.PercentFormatter(xmax=op.get("xmax", 1.0), decimals=op.get("decimals"))
    if low == "eng":
        return mticker.EngFormatter(unit=op.get("unit", ""), places=op.get("decimals"))
    if low == "log":
        return mticker.LogFormatterSciNotation()
    raise ValueError(f"format {spec!r}: use a format string like '{{x:.2f}}', or sci, plain, percent, eng, log")


@register_op("ticks", "ticks {axis: x|y|both, format: '{x:.2f}'|sci|plain|percent (xmax)|eng (unit)|log, nbins, "
             "step, minor: true|n, rotation, ha, direction in|out|inout, length, width, labelsize, color, pad, "
             "mirror: true (ticks on top/right too), top, right, grid: major|minor|both|none}", group=G_AXES)
def _ticks(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    which_axes = {"x": ["x"], "y": ["y"], "both": ["x", "y"], "z": ["z"]}.get(str(op.get("axis", "both")).lower())
    if which_axes is None:
        raise ValueError("axis must be x, y or both")
    for a in which_axes:
        axis = getattr(ax, f"{a}axis", None)
        if axis is None:
            raise ValueError(f"this axes has no {a} axis")
        cur = axis.get_major_formatter()
        scaled = cur if isinstance(cur, ScaledFormatter) else None
        if op.get("format"):
            axis.set_major_formatter(_formatter(op["format"], op, scaled))
        if op.get("nbins") or op.get("step"):
            if axis.get_scale() != "linear":
                raise ValueError(f"nbins/step need a linear {a} axis (this one is {axis.get_scale()})")
            if scaled is not None:
                axis.set_major_locator(DisplayedLocator(scaled.scale, scaled.offset, op.get("nbins"), op.get("step")))
            elif op.get("step"):
                axis.set_major_locator(mticker.MultipleLocator(float(op["step"])))
            else:
                axis.set_major_locator(mticker.MaxNLocator(int(op["nbins"])))
        if "minor" in op:
            m = op["minor"]
            if m is False:
                axis.set_minor_locator(mticker.NullLocator())
            elif axis.get_scale() == "linear":
                axis.set_minor_locator(mticker.AutoMinorLocator(None if m is True else int(m)))
            else:
                axis.set_minor_locator(mticker.LogLocator(subs="auto"))
        tp = {k: op[k] for k in ("direction", "length", "width", "labelsize", "color", "pad", "labelcolor") if k in op}
        if tp:
            ax.tick_params(axis=a, which="major", **tp)
            mtp = dict(tp)
            if "length" in mtp:
                mtp["length"] = 0.6 * float(mtp["length"])
            mtp.pop("labelsize", None)
            ax.tick_params(axis=a, which="minor", **mtp)
        if op.get("rotation") is not None:
            ax.tick_params(axis=a, labelrotation=op["rotation"])
            if a == "x" and op.get("rotation") not in (0, 90):
                for t in ax.get_xticklabels():
                    t.set_ha(op.get("ha", "right"))
        mirror = op.get("mirror")
        if a == "x" and (mirror or "top" in op):
            ax.tick_params(axis="x", which="both", top=bool(op.get("top", mirror)))
        if a == "y" and (mirror or "right" in op):
            ax.tick_params(axis="y", which="both", right=bool(op.get("right", mirror)))
        g = op.get("grid")
        if g is not None:
            if g in (False, "none", "off"):
                ax.grid(False, axis=a, which="both")
            else:
                g = "major" if g is True else g
                if g in ("minor", "both") and axis.get_minor_locator().__class__ is mticker.NullLocator:
                    axis.set_minor_locator(mticker.AutoMinorLocator())
                if g in ("major", "both"):
                    ax.grid(True, axis=a, which="major", alpha=0.35)
                if g in ("minor", "both"):
                    ax.grid(True, axis=a, which="minor", alpha=0.18, linestyle=":")


@register_op("spines", "spines {hide: [top, right] | 'all', show: [..], offset: pts | {left: 5}, linewidth, color, "
             "zero: [left, bottom] (through data 0)}", group=G_AXES)
def _spines(ctx: OpContext):
    op, ax = ctx.op, ctx.ax
    names = list(ax.spines.keys())

    def _names(v):
        v = names if v == "all" else ([v] if isinstance(v, str) else list(v))
        bad = [n for n in v if n not in names]
        if bad:
            raise ValueError(f"unknown spine(s) {bad}; this axes has {names}")
        return v
    for n in _names(op.get("hide", [])):
        ax.spines[n].set_visible(False)
        if n == "top":
            ax.xaxis.set_ticks_position("bottom")
        elif n == "right":
            ax.yaxis.set_ticks_position("left")
    for n in _names(op.get("show", [])):
        ax.spines[n].set_visible(True)
    off = op.get("offset")
    if off is not None:
        items = off.items() if isinstance(off, dict) else [(n, off) for n in names if ax.spines[n].get_visible()]
        for n, v in items:
            _names([n])
            ax.spines[n].set_position(("outward", float(v)))
    for n in _names(op.get("zero", [])):
        ax.spines[n].set_position("zero")
    for n in names:
        if "linewidth" in op:
            ax.spines[n].set_linewidth(op["linewidth"])
        if "color" in op:
            ax.spines[n].set_edgecolor(_check_color(op["color"]))


def _mappables(ax):
    from matplotlib.cm import ScalarMappable
    out = [a for a in list(ax.images) + list(ax.collections)
           if isinstance(a, ScalarMappable) and a.get_array() is not None]
    return out


def _axes_from(ctx: OpContext, key="axes_list"):
    axs = main_axes(ctx.fig)
    lst = ctx.op.get(key)
    if lst is None:
        return None
    out = []
    for i in lst:
        if not (0 <= int(i) < len(axs)):
            raise ValueError(f"axes index {i} out of range (0..{len(axs) - 1})")
        out.append(axs[int(i)])
    return out


def _same_cell(a, b) -> bool:
    sa, sb = a.get_subplotspec(), b.get_subplotspec()
    return (sa is not None and sb is not None and sa.get_gridspec() is sb.get_gridspec()
            and sa.rowspan == sb.rowspan and sa.colspan == sb.colspan)


def _shared_cax(fig, axes: List, location: str, fraction: float, pad: float, wspace, hspace):
    """Colour-bar axes next to a rectangular block of gridspec panels, managed by the gridspec (so tight_layout
    keeps panels and colour bar apart).  None if the panels do not form such a block."""
    sss = [a.get_subplotspec() for a in axes]
    if any(s is None for s in sss):
        return None
    gs = sss[0].get_gridspec()
    if any(s.get_gridspec() is not gs for s in sss):
        return None
    r0, r1 = min(s.rowspan.start for s in sss), max(s.rowspan.stop for s in sss)
    c0, c1 = min(s.colspan.start for s in sss), max(s.colspan.stop for s in sss)
    members = []
    for a in fig.axes:
        s = a.get_subplotspec()
        if s is None or s.get_gridspec() is not gs or a.get_label() == "<colorbar>":
            continue
        inside = s.rowspan.start < r1 and s.rowspan.stop > r0 and s.colspan.start < c1 and s.colspan.stop > c0
        if inside:
            if not any(_same_cell(a, b) for b in axes):
                raise ValueError("axes_list must form a rectangular block of panels (another panel lies inside it)")
            members.append(a)
    block = gs[r0:r1, c0:c1]
    vertical = location in ("right", "left")
    ratios = [1.0, fraction] if location in ("right", "bottom") else [fraction, 1.0]
    outer = (block.subgridspec(1, 2, width_ratios=ratios, wspace=pad) if vertical
             else block.subgridspec(2, 1, height_ratios=ratios, hspace=pad))
    main_cell, cb_cell = (outer[0], outer[1]) if location in ("right", "bottom") else (outer[1], outer[0])
    inner = main_cell.subgridspec(r1 - r0, c1 - c0, wspace=wspace, hspace=hspace)
    for a in members:
        s = a.get_subplotspec()
        a.set_subplotspec(inner[s.rowspan.start - r0:s.rowspan.stop - r0, s.colspan.start - c0:s.colspan.stop - c0])
    return fig.add_subplot(cb_cell, label="<colorbar>")


@register_op("colorbar", "colorbar {axes_list: [0, 1] (panels sharing one bar; default this axes), mappable_axes "
             "(index holding the image/contour, default first), sync: true (same colour limits for all), label, "
             "ticks, extend, location right|left|top|bottom, fraction (0.05), pad, format} (put it after the "
             "coloured plots; in make_plot in the last panel)", needs_axes=False, group=G_AXES)
def _colorbar(ctx: OpContext):
    op, fig = ctx.op, ctx.fig
    axes = _axes_from(ctx) or ([ctx.ax] if ctx.ax is not None else [])
    if not axes:
        raise ValueError("figure has no axes")
    src_ax = main_axes(fig)[int(op["mappable_axes"])] if op.get("mappable_axes") is not None else None
    maps = [m for a in axes for m in _mappables(a)]
    cand = _mappables(src_ax) if src_ax is not None else [m for a in axes for m in _mappables(a)]
    if not cand:
        raise ValueError("no colour-mapped plot (image, contourf, pcolormesh, scatter with c) on the listed axes; "
                         "draw it first (in make_plot put this op in the last panel)")
    m = cand[-1]
    if op.get("sync", True) and len(maps) > 1:
        lims = np.array([mm.get_clim() for mm in maps], dtype=float)
        lo, hi = np.nanmin(lims[:, 0]), np.nanmax(lims[:, 1])
        for mm in maps:
            if mm.norm is m.norm or type(mm.norm) is type(m.norm):
                mm.set_clim(lo, hi)
    location = op.get("location", "right")
    if location not in ("right", "left", "top", "bottom"):
        raise ValueError("location must be right, left, top or bottom")
    kw = {k: op[k] for k in ("extend",) if k in op}
    if op.get("format"):
        kw["format"] = _formatter(op["format"], op)
    cax = None
    if len(axes) > 1:
        cax = _shared_cax(fig, axes, location, float(op.get("fraction", 0.05)), float(op.get("pad", 0.08)),
                          op.get("wspace", 0.3), op.get("hspace", 0.35))
    if cax is not None:
        cb = fig.colorbar(m, cax=cax, orientation="vertical" if location in ("right", "left") else "horizontal",
                          location=location, **kw)
    else:
        cb = fig.colorbar(m, ax=axes if len(axes) > 1 else axes[0], location=location,
                          fraction=float(op.get("fraction", 0.05 if len(axes) > 1 else 0.15)),
                          pad=float(op.get("pad", 0.05)), **kw)
    if op.get("label"):
        cb.set_label(op["label"])
    if op.get("ticks") is not None:
        cb.set_ticks(op["ticks"])


_POS = {"top left": ((0.03, 0.97), "left", "top"), "top right": ((0.97, 0.97), "right", "top"),
        "bottom left": ((0.03, 0.03), "left", "bottom"), "bottom right": ((0.97, 0.03), "right", "bottom"),
        "outside top left": ((-0.02, 1.02), "right", "bottom"), "outside top right": ((1.0, 1.02), "right", "bottom")}


@register_op("panel_labels", "panel_labels {axes_list (default all panels; twins skipped), labels: [..] (custom), "
             "start: 'a', parentheses: true, uppercase: false, bold: true, position: 'outside top left' | 'top left' "
             "| 'top right' | 'bottom left' | 'bottom right' | [x, y] axes fractions, fontsize (12), color}",
             needs_axes=False, group=G_AXES)
def _panel_labels(ctx: OpContext):
    op = ctx.op
    axes = _axes_from(ctx)
    if axes is None:
        axes = []
        for a in main_axes(ctx.fig):
            if getattr(a, "_calphad_broken_of", None) is not None:
                continue
            if any(_same_cell(a, b) or np.allclose(a.get_position().bounds, b.get_position().bounds) for b in axes):
                continue          # twin axes share the panel
            axes.append(a)
    labels = op.get("labels")
    if labels is None:
        start = ord(str(op.get("start", "a")).lower()[0]) - ord("a")
        labels = []
        for i in range(len(axes)):
            k = start + i
            s = chr(ord("a") + k % 26) * (k // 26 + 1)
            labels.append(s.upper() if op.get("uppercase") else s)
        if op.get("parentheses", True):
            labels = [f"({s})" for s in labels]
    if len(labels) < len(axes):
        raise ValueError(f"{len(axes)} panels but {len(labels)} labels")
    pos = op.get("position", "outside top left")
    if isinstance(pos, str):
        if pos not in _POS:
            raise ValueError(f"position must be one of {list(_POS)} or [x, y]")
        xy, ha, va = _POS[pos]
    else:
        xy, ha, va = tuple(pos), op.get("ha", "left"), op.get("va", "top")
    for a, lab in zip(axes, labels):
        a.annotate(lab, xy=xy, xycoords="axes fraction", ha=ha, va=va, fontsize=op.get("fontsize", 12),
                   fontweight="bold" if op.get("bold", True) else "normal", color=_check_color(op.get("color", "k")),
                   annotation_clip=False)
    ctx.note(f"labelled {len(axes)} panel(s)")


# ---------------------------------------------------------------------------
# broken y-axis
# ---------------------------------------------------------------------------
def _copy_artists(src, dst) -> List[str]:
    """Copy lines, scatter/line/poly collections, rectangles/polygons and texts drawn in data coordinates."""
    from matplotlib import collections as mc
    from matplotlib import patches as mp
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.transforms import IdentityTransform
    skipped = []
    td = src.transData
    for ln in src.get_lines():
        if ln.get_label() == "_break" or ln.get_transform() != td:
            continue
        nl = Line2D(ln.get_xdata(), ln.get_ydata(), color=ln.get_color(), linestyle=ln.get_linestyle(),
                    linewidth=ln.get_linewidth(), marker=ln.get_marker(), markersize=ln.get_markersize(),
                    markerfacecolor=ln.get_markerfacecolor(), markeredgecolor=ln.get_markeredgecolor(),
                    markeredgewidth=ln.get_markeredgewidth(), alpha=ln.get_alpha(), zorder=ln.get_zorder(),
                    drawstyle=ln.get_drawstyle(), label=ln.get_label())
        dst.add_line(nl)
    for c in src.collections:
        common = dict(alpha=c.get_alpha(), zorder=c.get_zorder(), label=c.get_label())
        try:
            c.update_scalarmappable()
        except Exception:  # noqa: BLE001
            pass
        if isinstance(c, mc.PathCollection):
            if c.get_offset_transform() != td:
                skipped.append(type(c).__name__)
                continue
            nc = mc.PathCollection(c.get_paths(), sizes=c.get_sizes(), offsets=c.get_offsets(),
                                   offset_transform=dst.transData, facecolors=c.get_facecolors(),
                                   edgecolors=c.get_edgecolors(), linewidths=c.get_linewidths(), **common)
            nc.set_transform(IdentityTransform())
            dst.add_collection(nc, autolim=False)
        elif isinstance(c, (mc.LineCollection, mc.PolyCollection)) and c.get_transform() == td:
            segs = [p.vertices for p in c.get_paths()]
            if isinstance(c, mc.LineCollection):
                nc = mc.LineCollection(segs, colors=c.get_colors(), linewidths=c.get_linewidths(),
                                       linestyles=c.get_linestyles(), **common)
            else:
                nc = mc.PolyCollection(segs, facecolors=c.get_facecolors(), edgecolors=c.get_edgecolors(),
                                       linewidths=c.get_linewidths(), **common)
            dst.add_collection(nc, autolim=False)
        else:
            skipped.append(type(c).__name__)
    for p in src.patches:
        if p.get_transform() != td and not isinstance(p, mp.Rectangle):
            skipped.append(type(p).__name__)
            continue
        kw = dict(facecolor=p.get_facecolor(), edgecolor=p.get_edgecolor(), linewidth=p.get_linewidth(),
                  alpha=p.get_alpha(), zorder=p.get_zorder(), label=p.get_label(), hatch=p.get_hatch(),
                  fill=p.get_fill())
        if isinstance(p, mp.Rectangle):
            dst.add_patch(mp.Rectangle(p.get_xy(), p.get_width(), p.get_height(), **kw))
        elif isinstance(p, mp.Polygon):
            dst.add_patch(mp.Polygon(p.get_xy(), closed=True, **kw))
        else:
            skipped.append(type(p).__name__)
    for t in src.texts:
        if type(t) is Text and t.get_transform() == td:
            dst.text(*t.get_position(), t.get_text(), color=t.get_color(), fontsize=t.get_fontsize(),
                     ha=t.get_ha(), va=t.get_va(), rotation=t.get_rotation(), zorder=t.get_zorder())
        else:
            skipped.append(type(t).__name__)
    return sorted(set(skipped))


@register_op("broken_yaxis", "broken_yaxis {ylims: [[lo, hi] (lower part), [lo2, hi2] (upper part)], ratio: "
             "[lower, upper] heights (1, 1), gap (0.08), legend: lower|upper, marks: true}: splits the panel after "
             "drawing (lines, scatter, bars, error bars, fills are copied); the upper part is a new axes index",
             group=G_AXES)
def _broken_yaxis(ctx: OpContext):
    op, ax, fig = ctx.op, ctx.ax, ctx.fig
    _need(ctx, ["rectilinear"], "broken_yaxis")
    yl = op.get("ylims")
    if not (isinstance(yl, list) and len(yl) == 2 and all(len(v) == 2 for v in yl)):
        raise ValueError("ylims must be [[lo, hi], [lo2, hi2]] (lower part, upper part, displayed units)")
    (lo1, hi1), (lo2, hi2) = [sorted(_to_data(ax.yaxis, v, ctx.mode)) for v in yl]
    if hi1 > lo2:
        raise ValueError("the lower range must lie below the upper range")
    rb, rt = [float(v) for v in op.get("ratio", [1, 1])]
    gap = float(op.get("gap", 0.08))
    ss = ax.get_subplotspec()
    if ss is not None:
        sub = ss.subgridspec(2, 1, height_ratios=[rt, rb], hspace=gap)
        ax.set_subplotspec(sub[1])
        top = fig.add_subplot(sub[0], sharex=ax)
    else:
        x0, y0, w, h = ax.get_position().bounds
        tot = rb + rt + gap * (rb + rt) / 2
        hb, ht, g = h * rb / tot, h * rt / tot, h * gap * (rb + rt) / 2 / tot
        ax.set_position([x0, y0, w, hb])
        top = fig.add_axes([x0, y0 + hb + g, w, ht], sharex=ax)
    skipped = _copy_artists(ax, top)
    top._calphad_broken_of = True
    top._calphad_legend_set = True
    top.set_yscale(ax.get_yscale())
    f = ax.yaxis.get_major_formatter()
    if isinstance(f, ScaledFormatter):
        top.yaxis.set_major_formatter(ScaledFormatter(f.scale, f.offset, f.fmt))
    ax.set_ylim(lo1, hi1)
    top.set_ylim(lo2, hi2)
    ax.spines["top"].set_visible(False)
    top.spines["bottom"].set_visible(False)
    ax.xaxis.tick_bottom()
    top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    if ax.get_title():
        top.set_title(ax.get_title())
        ax.set_title("")
    if ax.yaxis.get_gridlines() and ax.yaxis._major_tick_kw.get("gridOn"):
        top.grid(True, alpha=0.35)
    if ax.get_ylabel():       # one label centred on both parts (x stays automatic)
        ax.yaxis.label.set_y((rb + rt + gap * (rb + rt) / 2) / (2 * rb))
    if op.get("marks", True):
        d = 0.5
        kw = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none", color="k", mec="k", mew=1,
                  clip_on=False, label="_break")
        top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
        ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kw)
    leg = ax.get_legend()
    if op.get("legend", "upper") == "upper" and leg is not None:
        handles = list(leg.legend_handles)
        texts = [t.get_text() for t in leg.get_texts()]
        top.legend(handles, texts, loc=op.get("loc", "best"))
        leg.set_visible(False)
        ax._calphad_legend_set = True
    elif op.get("legend", "upper") == "upper" and ax.get_legend_handles_labels()[0]:
        top.legend(*top.get_legend_handles_labels(), loc=op.get("loc", "best"))
        _hide_legend(ax)
    idx = main_axes(fig).index(top)
    ctx.note(f"broken y-axis: lower part is axes {main_axes(fig).index(ax)}, upper part is axes {idx}"
             + (f"; not copied to the upper part: {', '.join(skipped)}" if skipped else ""))


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------
@register_op("annotate_points", "annotate_points {x, y, labels | data + x_col, y_col, labels_col; fontsize (8), "
             "color, offset: [dx, dy] points (5, 5), avoid: true (try other offsets to avoid overlaps), arrow, "
             "bbox: true, scatter: true (also draw the points: marker, s, color)}", group=G_ANN)
def _annotate_points(ctx: OpContext):
    op, ax, fig = ctx.op, ctx.ax, ctx.fig
    x, y = ctx.X(), ctx.Y()
    ax.get_xlim(), ax.get_ylim()          # apply pending autoscaling before measuring
    labels = ctx.raw("labels", dtype=None)
    if x is None or y is None or labels is None:
        raise ValueError("annotate_points needs x, y and labels (or data + x_col, y_col, labels_col)")
    labels = [("" if v is None else (_lab(v) if not isinstance(v, str) else v)) for v in np.atleast_1d(labels)]
    if not (len(x) == len(y) == len(labels)):
        raise ValueError(f"x, y, labels lengths differ ({len(x)}, {len(y)}, {len(labels)})")
    if op.get("scatter"):
        ax.scatter(x, y, transform=ctx.tr, s=op.get("s", 20), marker=op.get("marker", "o"),
                   color=_check_color(op.get("color", "k")), zorder=3)
    dx, dy = [float(v) for v in op.get("offset", [5, 5])]
    cands = [(dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy), (0, 1.8 * dy), (0, -1.8 * dy), (2 * dx, 0), (-2 * dx, 0),
             (2 * dx, 2 * dy), (-2 * dx, -2 * dy)]
    avoid = op.get("avoid", True) and len(labels) <= 400
    renderer = fig.canvas.get_renderer() if avoid else None
    placed = []
    style = {"fontsize": op.get("fontsize", 8), "color": _check_color(op.get("text_color", op.get("color", "k")))}
    if op.get("bbox"):
        style["bbox"] = {"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"}
    if op.get("arrow"):
        style["arrowprops"] = {"arrowstyle": "-", "color": "0.4", "lw": 0.6}
    n_moved = 0
    for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
        if not lab or not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        order = cands[i % 2:] + cands[:i % 2] if not avoid else cands
        best = None
        for j, (ox, oy) in enumerate(order):
            ha = "left" if ox > 0 else ("right" if ox < 0 else "center")
            va = "bottom" if oy > 0 else ("top" if oy < 0 else "center")
            t = ax.annotate(lab, xy=(xi, yi), xycoords=ctx.tr, xytext=(ox, oy), textcoords="offset points",
                            ha=ha, va=va, **style)
            if not avoid:
                best = t
                break
            bb = t.get_window_extent(renderer).expanded(1.05, 1.1)
            if not any(bb.overlaps(p) for p in placed) or j == len(order) - 1:
                if j:
                    n_moved += 1
                placed.append(bb)
                best = t
                break
            t.remove()
    if n_moved:
        ctx.note(f"moved {n_moved} label(s) to avoid overlaps")


@register_op("label_lines", "label_lines {match: label substring | [..] (default all labelled lines), x (position, "
             "default line end), where: end|start|max|min, fontsize, offset: [dx, dy] points, bbox: true, avoid: true, "
             "legend: false (hide the legend)}: writes each line's label next to it in its colour", group=G_ANN)
def _label_lines(ctx: OpContext):
    op, ax, fig = ctx.op, ctx.ax, ctx.fig
    m = op.get("match")
    ms = [m] if isinstance(m, str) else (list(m) if m else None)
    lines = []
    for ln in ax.get_lines():
        lab = ln.get_label()
        if not lab or lab.startswith("_"):
            continue
        if ms and not any(s.lower() in lab.lower() for s in ms):
            continue
        lines.append(ln)
    if not lines:
        raise ValueError("no labelled lines match" + (f" {m!r}" if m else "") + "; see describe_figure")
    ax.get_xlim(), ax.get_ylim()          # apply pending autoscaling before measuring
    where = op.get("where", "end")
    xpos = _xd(ctx, [op["x"]])[0] if op.get("x") is not None else None
    items = []
    for ln in lines:
        xd = np.asarray(ln.get_xdata(), dtype=float)
        yd = np.asarray(ln.get_ydata(), dtype=float)
        ok = np.isfinite(xd) & np.isfinite(yd)
        xd, yd = xd[ok], yd[ok]
        if not len(xd):
            continue
        if xpos is not None:
            o = np.argsort(xd)
            if not (xd.min() <= xpos <= xd.max()):
                ctx.note(f"x outside the range of {ln.get_label()!r}: label placed at the nearest end")
            px, py = xpos, float(np.interp(xpos, xd[o], yd[o]))
            ha = "center"
        elif where in ("max", "min"):
            k = int(np.argmax(yd) if where == "max" else np.argmin(yd))
            px, py, ha = xd[k], yd[k], "center"
        else:
            k = -1 if where == "end" else 0
            px, py, ha = xd[k], yd[k], "left" if where == "end" else "right"
        items.append([ln, px, py, ha])
    dx, dy = [float(v) for v in op.get("offset", [4 if where == "end" else (-4 if where == "start" else 0),
                                                  0 if where in ("end", "start") and op.get("x") is None else 4])]
    fs = op.get("fontsize", 9)
    # vertical de-overlap in points
    shifts = [0.0] * len(items)
    if op.get("avoid", True) and len(items) > 1:
        disp = [ax.transData.transform((it[1], it[2]))[1] * 72.0 / fig.dpi for it in items]
        order = np.argsort(disp)
        min_sep = fs * 1.15
        last = -np.inf
        for k in order:
            want = disp[k]
            if want < last + min_sep:
                shifts[k] = last + min_sep - want
                want = last + min_sep
            last = want
    for (ln, px, py, ha), sh in zip(items, shifts):
        kw = {}
        if op.get("bbox", False):
            kw["bbox"] = {"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
        ax.annotate(ln.get_label(), xy=(px, py), xycoords="data", xytext=(dx if ha != "center" else 0, dy + sh),
                    textcoords="offset points", ha=ha, va="center" if ha != "center" else "bottom", fontsize=fs,
                    color=ln.get_color(), annotation_clip=False, **kw)
    if op.get("legend") is False:
        _hide_legend(ax)


def register(tb) -> None:
    """Plug-in hook of the ToolBox loader: this module adds figure operations only (no extra agent tools)."""
    return None
