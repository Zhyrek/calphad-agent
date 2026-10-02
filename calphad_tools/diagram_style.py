"""Drawing style of the phase-diagram figures (map_phase_diagram, plot_all_binaries / plot_all_ternaries,
plot_diagram_with_data).

Two styles:

* ``"shaded"`` (default): binary T-x diagrams with black phase boundaries and invariant lines (lw 1.25, solid),
  two-phase fields filled ``aliceblue`` between their two boundaries, no tie-lines, minor ticks on and all ticks
  pointing in on all four sides; ternary isotherms with grey (#999999) phase boundaries and tie-triangle edges, no
  tie-lines.  No phase legend: the phases are named by labels inside the figure (region labels; on isopleths and
  T-MU diagrams, where regions are not labelled, each line is labelled with :func:`line_labels`).
* ``"color"``: one colour per phase, grey tie-lines, red invariants and a phase legend.

This module only needs numpy and a matplotlib axes, so show_code copies it into the standalone scripts: the tools
and the generated code draw a shaded diagram with the same function, :func:`draw_shaded`.
"""
from __future__ import annotations

import numpy as np

STYLES = ("shaded", "color")
LINE_COLOR = "k"                 # binary / isopleth / T-MU boundaries and invariant lines (shaded)
LINE_WIDTH = 1.25
FILL_COLOR = "aliceblue"         # two-phase fields of a binary T-x diagram
TERNARY_COLOR = "#999999"        # ternary boundaries and tie-triangle edges
TERNARY_WIDTH = 1.0
LABEL_COLOR = "black"            # phase labels and invariant temperatures on a shaded diagram


def check_style(style, default="shaded") -> str:
    """'shaded' or 'color' (None -> default)."""
    s = str(style or default).strip().lower()
    if s in ("colour", "colored", "coloured"):
        s = "color"
    if s not in STYLES:
        raise ValueError(f"style must be one of {list(STYLES)}, not {style!r}")
    return s


def ticks_in(ax):
    """Minor ticks on; major and minor ticks pointing in on all four sides (pad 7)."""
    ax.minorticks_on()
    for which in ("minor", "major"):
        ax.tick_params(axis="both", which=which, direction="in", bottom=True, top=True, left=True, right=True,
                       pad=7)


def draw_shaded(ax, boundaries, invariants, kind="binary", tiny=1e-3):
    """Draw a phase diagram in the shaded style.

    boundaries: one entry per two-phase ZPF line: [xy of phase 1, xy of phase 2], each an (n, 2) array of plotted
                (x, y) points (a binary: same y = temperature at each point, i.e. the ends of a tie-line).
    invariants: (3, 2) arrays: the three phase compositions of each invariant / tie-triangle.
    kind: "binary" (black, two-phase fields filled), "ternary" (grey, nothing filled), "potential" (T-MU: black,
          nothing filled - both boundaries of a two-phase equilibrium coincide there).
    A boundary that collapses onto one point (relative extent < tiny) is drawn as a dot."""
    color, lw = (TERNARY_COLOR, TERNARY_WIDTH) if kind == "ternary" else (LINE_COLOR, LINE_WIDTH)
    pts_all = [np.asarray(p, float).reshape(-1, 2) for b in boundaries for p in b]
    allxy = np.concatenate(pts_all) if pts_all else np.zeros((0, 2))
    allxy = allxy[np.isfinite(allxy).all(1)]
    span = np.ptp(allxy, axis=0) * tiny if len(allxy) else np.zeros(2)
    for b in boundaries:
        pair = [np.asarray(p, float).reshape(-1, 2) for p in b]
        if kind == "binary" and len(pair) == 2 and len(pair[0]) == len(pair[1]) > 1:
            ax.fill_betweenx(pair[0][:, 1], pair[0][:, 0], pair[1][:, 0], color=FILL_COLOR, zorder=1)
        for p in pair:
            if not len(p):
                continue
            if np.ptp(p[:, 0]) <= span[0] and np.ptp(p[:, 1]) <= span[1]:
                ax.scatter([np.nanmean(p[:, 0])], [np.nanmean(p[:, 1])], color=color, s=8, zorder=3)
            else:
                ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls="-", solid_capstyle="butt", zorder=2)
    for inv in invariants:
        p = np.asarray(inv, float).reshape(-1, 2)
        if kind == "ternary":
            p = np.vstack([p, p[:1]])                       # tie-triangle edges
        else:
            p = p[np.argsort(p[:, 0])]                      # invariant line, lowest to highest x
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls="-", solid_capstyle="butt", zorder=2.5)


def strategy_boundaries(strategy, x, y, y_offset=0.0):
    """(boundaries, invariants) for draw_shaded from a pycalphad BinaryStrategy / TernaryStrategy
    (get_tieline_data / get_invariant_data); y_offset shifts y, e.g. -273.15 to plot a Kelvin axis in °C.  Lines
    whose two boundaries coincide everywhere (zero-length tie-lines along an edge) are left out."""
    def xy(d):
        return np.column_stack([np.atleast_1d(np.asarray(d.x, float)),
                                np.atleast_1d(np.asarray(d.y, float)) + y_offset])
    boundaries = []
    for tie in strategy.get_tieline_data(x, y):
        pair = [xy(d) for d in tie.data]
        if len(pair) == 2 and pair[0].shape == pair[1].shape and np.allclose(pair[0], pair[1], atol=1e-5):
            continue                # zero-length tie-lines everywhere: artefact along a pure-element / binary edge
        boundaries.append(pair)
    invariants = [np.vstack([xy(d) for d in inv.data]) for inv in strategy.get_invariant_data(x, y)]
    return boundaries, invariants


def line_labels(ax, items, fontsize=7.0, color=LABEL_COLOR):
    """Name each line inside the plot (no legend): items = [(text, (n, 2) xy array)].  The text goes on the line,
    at the first of mid / 35 % / 65 % / 20 % / 80 % of its length (in the axes) that does not overlap a label
    already placed; lines too short for their text get none, and a line next to one already named the same (e.g.
    a line traced twice) is not named again.  Call after the axis limits are set."""
    trans = ax.transData
    box = ax.get_window_extent()
    pt = ax.figure.dpi / 72.0
    placed = []
    for k, (text, xy) in enumerate(items):
        xy = np.asarray(xy, float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(1)]
        if len(xy) < 2:
            continue
        px = trans.transform(xy)
        inside = (px[:, 0] >= box.x0) & (px[:, 0] <= box.x1) & (px[:, 1] >= box.y0) & (px[:, 1] <= box.y1)
        px, xy = px[inside], xy[inside]
        if len(px) < 2:
            continue
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(px, axis=0).T))])
        w, h = 0.62 * fontsize * len(text) * pt, 1.3 * fontsize * pt
        if s[-1] < 0.8 * w:
            continue
        for f in (0.5, 0.35, 0.65, 0.2, 0.8):
            i = int(np.argmin(np.abs(s - f * s[-1])))
            r = (px[i, 0] - w / 2, px[i, 1] - h / 2, px[i, 0] + w / 2, px[i, 1] + h / 2)
            if r[0] < box.x0 or r[2] > box.x1 or r[1] < box.y0 or r[3] > box.y1:
                continue
            if any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q, _ in placed):
                continue
            if any(t == text and np.hypot((r[0] + r[2] - q[0] - q[2]) / 2, (r[1] + r[3] - q[1] - q[3]) / 2) < 4 * w
                   for q, t in placed):
                break                                       # same name right next to it: once is enough
            placed.append((r, text))
            ax.text(xy[i, 0], xy[i, 1], text, fontsize=fontsize, color=color, ha="center", va="center", zorder=6,
                    bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    gid=f"line_label:{k}")
            break
