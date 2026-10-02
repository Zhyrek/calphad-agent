"""Phase labels on a phase diagram: region masks, obstacle avoidance and text placement.

Needs numpy, scipy and matplotlib.  Everything inside the plot works on a raster of the axes in points (``res`` pt
per cell):

1. regions: exact polygons (two-phase fields from the ZPF lines, tie-triangles) are painted first; the rest of
   the diagram is split by an equilibrium grid (nearest grid node) into connected components; pieces of one
   region are merged; grid slivers along a polygon edge are dropped;
2. obstacles: markers of data points (scatter collections, lines with markers), other texts, the legend, and
   every label already placed;
3. each region label goes to the point of its region farthest from the region edge and the obstacles where the
   text box, grown by ``margin`` (2.5 pt) all round, fits, from ``fontsize`` down to ``min_fontsize`` (default 8
   pt), horizontal or turned (along a skinny region, or any multiple of 15 degrees when that gives a clearly larger
   label).  Two- and three-phase fields that do not fit are left out (reported).  A single-phase field that does
   not fit (or is too thin to be a region at all) is named outside the plot like a stoichiometric phase (5.) when
   it reaches an edge - in a binary only intermediate phases, never the liquid or an end-member phase; a
   ternary-only field gets a label in the nearest free space with a thin leader line (never left out);
4. binary invariant temperatures (no reaction types) go right above or right below their line, inside the
   adjacent region, from ``invariant_fontsize`` down to ``min_fontsize``, or are left out;
5. stoichiometric phases (binary line compounds, ternary point / line compounds) are never named inside their
   field: on an edge, their name goes outside, next to the composition (binary: below the x axis, text at 60
   degrees; ternary: beyond the nearest edge, 60 degrees at the bottom edge, -30 / +30 degrees - square to the
   edge - at the left / right edge); tick labels they cover are hidden (:class:`TickLabelHider`), and axis titles
   moved clear.  A ternary compound inside the triangle gets a leader label (and a dot at a point compound);
6. every position is confirmed with ``key_at`` (an equilibrium calculation) before the label is drawn.

Each label is one Text artist (``gid`` "phase_label:<PHASES>" / "invariant_label:<T>"), so it stays one editable
text object in SVG (``svg.fonttype = 'none'``) and PDF (``pdf.fonttype = 42``): see :func:`save_vector`.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.artist
import numpy as np
import matplotlib.path as mpath

from .phase_names import is_liquid_name, region_label

FONT_FAMILY = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
LABEL_PREFIXES = ("phase_label:", "invariant_label:")
DEFAULT_STYLE = {
    "fontsize": 9.5, "min_fontsize": 8.0, "color": "black", "family": FONT_FAMILY, "fontweight": "normal",
    "background": False, "background_alpha": 0.75, "rotate": True, "margin": 2.5,
    "invariant_fontsize": 9.0, "invariant_color": "#b2182b",
    "res": 0.5,
}
# no longer used: reaction types are never written, and labels never get leader lines (accepted and ignored)
IGNORED_STYLE = ("invariant_type", "leader")
# angles tried when a region label does not fit at full size horizontally or along the region
EXTRA_ANGLES = (90.0, -60.0, 60.0, -30.0, 30.0, -75.0, 75.0, -45.0, 45.0, -15.0, 15.0)
LEADER_COLOR = "0.2"           # leader lines (ternary-only phases inside the triangle)
DOT_PREFIX = "phase_label_dot:"  # gid of the dot marking a ternary point compound (with its label's gid)
# names of stoichiometric phases outside the plot: edge -> (text angle, side of the text at the edge)
EDGE_TEXT = {"bottom": (60.0, "right"), "left": (-30.0, "right"), "right": (30.0, "left")}


def vector_rc() -> Dict:
    """rcParams that keep text as text in SVG / PDF / EPS output (editable in Inkscape / Illustrator)."""
    return {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}


def save_vector(fig, path, **kw):
    """fig.savefig with text kept as text objects."""
    with matplotlib.rc_context(vector_rc()):
        return fig.savefig(path, **kw)


def _token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "", str(s))


# ---------------------------------------------------------------------------------------------------------
# raster of the axes
# ---------------------------------------------------------------------------------------------------------
class Raster:
    """Cells of ``res`` points over the axes box; row 0 at the bottom."""

    def __init__(self, ax, res: float = 0.5):
        self.ax = ax
        self.res = float(res)
        self.ppp = ax.figure.dpi / 72.0                 # pixels per point
        bb = ax.bbox
        self.x0, self.y0 = bb.x0, bb.y0
        self.s = self.res * self.ppp                    # pixels per cell
        self.W = max(4, int(math.ceil(bb.width / self.s)))
        self.H = max(4, int(math.ceil(bb.height / self.s)))
        cols, rows = np.meshgrid(np.arange(self.W), np.arange(self.H))
        self.px = np.column_stack([self.x0 + (cols.ravel() + 0.5) * self.s, self.y0 + (rows.ravel() + 0.5) * self.s])
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self.ax.transData.inverted().transform(self.px)
        return self._data

    def zeros(self, dtype=bool):
        return np.zeros((self.H, self.W), dtype)

    def cell(self, px, py):
        """(col, row) float position of a display point."""
        return (np.asarray(px) - self.x0) / self.s - 0.5, (np.asarray(py) - self.y0) / self.s - 0.5

    def to_px(self, col, row):
        return self.x0 + (np.asarray(col) + 0.5) * self.s, self.y0 + (np.asarray(row) + 0.5) * self.s

    def poly(self, pts_px) -> np.ndarray:
        """Cells whose centre lies inside a polygon (display coordinates)."""
        m = self.zeros()
        pts = np.asarray(pts_px, float)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 3:
            return m
        c, r = self.cell(pts[:, 0], pts[:, 1])
        c0, c1 = max(0, int(np.floor(c.min()))), min(self.W - 1, int(np.ceil(c.max())))
        r0, r1 = max(0, int(np.floor(r.min()))), min(self.H - 1, int(np.ceil(r.max())))
        if c0 > c1 or r0 > r1:
            return m
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        q = np.column_stack([self.x0 + (cc.ravel() + 0.5) * self.s, self.y0 + (rr.ravel() + 0.5) * self.s])
        inside = mpath.Path(pts).contains_points(q, radius=1e-9)
        m[r0:r1 + 1, c0:c1 + 1] = inside.reshape(cc.shape)
        return m

    def polyline(self, m, pts_px):
        """Mark the cells along a polyline (display coordinates)."""
        pts = np.asarray(pts_px, float)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = int(np.ceil(np.hypot(x1 - x0, y1 - y0) / (0.5 * self.s))) + 1
            c, r = self.cell(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
            c, r = np.round(c).astype(int), np.round(r).astype(int)
            ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
            m[r[ok], c[ok]] = True

    def disk(self, m, px, py, r_pt):
        c, r = self.cell(px, py)
        R = r_pt / self.res
        c0, c1 = max(0, int(np.floor(c - R))), min(self.W - 1, int(np.ceil(c + R)))
        r0, r1 = max(0, int(np.floor(r - R))), min(self.H - 1, int(np.ceil(r + R)))
        if c0 > c1 or r0 > r1:
            return
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        m[r0:r1 + 1, c0:c1 + 1] |= (cc - c) ** 2 + (rr - r) ** 2 <= R * R

    def rect(self, m, x0, y0, x1, y1):
        """Mark an axis-aligned display rectangle."""
        c0, r0 = self.cell(x0, y0)
        c1, r1 = self.cell(x1, y1)
        c0, c1 = max(0, int(np.floor(min(c0, c1)))), min(self.W - 1, int(np.ceil(max(c0, c1))))
        r0, r1 = max(0, int(np.floor(min(r0, r1)))), min(self.H - 1, int(np.ceil(max(r0, r1))))
        if c0 <= c1 and r0 <= r1:
            m[r0:r1 + 1, c0:c1 + 1] = True


def _rect_kernel(hw: float, hh: float, ang_deg: float, res: float) -> np.ndarray:
    R = int(math.ceil(math.hypot(hw, hh) / res)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1] * res
    a = math.radians(ang_deg)
    u = xx * math.cos(a) + yy * math.sin(a)
    v = -xx * math.sin(a) + yy * math.cos(a)
    return (np.abs(u) <= hw) & (np.abs(v) <= hh)


def _fits(blocked: np.ndarray, kernel: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, tuple]:
    """Centres (within box = r0, r1, c0, c1) where the kernel does not touch a blocked cell.  Returns
    (bool array over the box, box)."""
    from scipy.signal import fftconvolve
    K = kernel.shape[0] // 2
    r0, r1, c0, c1 = box
    H, W = blocked.shape
    R0, R1, C0, C1 = r0 - K, r1 + K + 1, c0 - K, c1 + K + 1
    crop = np.ones((R1 - R0, C1 - C0), float)
    sr0, sr1, sc0, sc1 = max(0, R0), min(H, R1), max(0, C0), min(W, C1)
    crop[sr0 - R0:sr1 - R0, sc0 - C0:sc1 - C0] = blocked[sr0:sr1, sc0:sc1]
    conv = fftconvolve(crop, kernel.astype(float), mode="same")
    ok = conv[K:K + (r1 - r0 + 1), K:K + (c1 - c0 + 1)] < 0.5
    return ok, box


def _box_cells(R: "Raster", cx: float, cy: float, w_pt: float, h_pt: float):
    """(r0, r1, c0, c1) of the cells whose centres lie in an axis-aligned box (display centre, size in pt)."""
    c0, r0 = R.cell(cx - w_pt * R.ppp / 2, cy - h_pt * R.ppp / 2)
    c1, r1 = R.cell(cx + w_pt * R.ppp / 2, cy + h_pt * R.ppp / 2)
    return int(np.ceil(r0)), int(np.floor(r1)), int(np.ceil(c0)), int(np.floor(c1))


def _bbox_of(mask: np.ndarray, pad: int = 0):
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return None
    H, W = mask.shape
    return (max(0, rows[0] - pad), min(H - 1, rows[-1] + pad), max(0, cols[0] - pad), min(W - 1, cols[-1] + pad))


def _edt(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    p = np.pad(mask, 1, constant_values=False)
    return distance_transform_edt(p)[1:-1, 1:-1]


# ---------------------------------------------------------------------------------------------------------
# obstacles
# ---------------------------------------------------------------------------------------------------------
def _is_ours(a) -> bool:
    gid = a.get_gid() or ""
    return gid.startswith(LABEL_PREFIXES)


def obstacles(ax, R: Raster, renderer, margin: float = 1.5) -> Tuple[np.ndarray, int]:
    """Cells covered by data markers, texts that are not labels, and the legend.  Returns (mask, n_points)."""
    from matplotlib.collections import PathCollection
    m = R.zeros()
    n = 0
    for c in ax.collections:
        if not isinstance(c, PathCollection) or not c.get_visible():
            continue
        offs = np.asarray(c.get_offsets(), float)
        if not len(offs):
            continue
        px = c.get_offset_transform().transform(offs)
        sizes = np.asarray(c.get_sizes(), float)
        sizes = sizes if len(sizes) else np.array([36.0])
        for i, (x, y) in enumerate(px):
            if np.isfinite(x) and np.isfinite(y):
                R.disk(m, x, y, math.sqrt(sizes[i % len(sizes)]) / 2 + margin)
                n += 1
    for ln in ax.get_lines():
        mk = ln.get_marker()
        if not ln.get_visible() or mk in (None, "None", "none", "", " ") or ln.get_markersize() <= 0:
            continue
        xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
        if not len(xy):
            continue
        px = ln.get_transform().transform(xy)
        r = ln.get_markersize() / 2 + margin
        for x, y in px:
            if np.isfinite(x) and np.isfinite(y):
                R.disk(m, x, y, r)
                n += 1
    for t in ax.texts:
        if _is_ours(t) or not t.get_visible() or not t.get_text().strip():
            continue
        try:
            e = t.get_window_extent(renderer)
        except Exception:  # noqa: BLE001
            continue
        R.rect(m, e.x0 - margin * R.ppp, e.y0 - margin * R.ppp, e.x1 + margin * R.ppp, e.y1 + margin * R.ppp)
    axis_texts = list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + [ax.xaxis.label, ax.yaxis.label, ax.title]
    for t in axis_texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            e = t.get_window_extent(renderer)
        except Exception:  # noqa: BLE001
            continue
        R.rect(m, e.x0 - margin * R.ppp, e.y0 - margin * R.ppp, e.x1 + margin * R.ppp, e.y1 + margin * R.ppp)
    leg = ax.get_legend()
    if leg is not None and leg.get_visible():
        e = leg.get_window_extent(renderer)
        R.rect(m, e.x0, e.y0, e.x1, e.y1)
    return m, n


# ---------------------------------------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------------------------------------
def build_regions(ax, R: Raster, inside: np.ndarray, regions: Dict, grid: Optional[Dict] = None,
                  min_area_pt2: float = 16.0, probe: Optional[Callable] = None):
    """(regions [{key, mask, src, thin, pole_px}], region-id raster (-1 none), keys of small multi-phase areas
    found only by the grid)."""
    from scipy import ndimage
    rid = np.full((R.H, R.W), -1, int)
    regs: List[Dict] = []
    notes: List[str] = []
    def area(p):
        q = ax.transData.transform(np.asarray(p["xy"], float))
        return 0.5 * abs(float(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1))))
    # tie-triangles first, then the larger fields (a sliver traced twice then joins the full field)
    polys = sorted(regions.get("polygons") or [], key=lambda p: (not p.get("tri", False), -area(p)))
    for p in polys:
        key = tuple(p["key"])
        pts = ax.transData.transform(np.asarray(p["xy"], float))
        raw = R.poly(pts) & inside
        m = raw & (rid == -1)
        same = [int(i) for i in np.unique(rid[raw & (rid >= 0)]) if regs[int(i)]["key"] == key]
        if same and raw.sum() >= 3 and np.isin(rid[raw], same).sum() >= 0.5 * raw.sum():
            regs[same[0]]["mask"] |= m          # the same field traced twice: one region
            rid[m] = same[0]
            continue
        reg = {"key": key, "mask": m, "src": "tri" if p.get("tri") else "poly", "thin": False}
        if m.sum() < 3:
            # thinner than the raster: remember the middle of its widest tie-line (or the centroid)
            ties = p.get("ties") or []
            if ties:
                seg = [ax.transData.transform(np.asarray(t, float)) for t in ties]
                k = int(np.argmax([np.hypot(*(s_[1] - s_[0])) for s_ in seg]))
                pole = seg[k].mean(axis=0)
            else:
                pole = pts.mean(axis=0)
            c, r = R.cell(*pole)
            ci, ri = int(round(float(c))), int(round(float(r)))
            if not (0 <= ci < R.W and 0 <= ri < R.H) or not inside[ri, ci]:
                continue
            near = rid[max(0, ri - 6):ri + 7, max(0, ci - 6):ci + 7]
            if any(regs[int(i)]["key"] == key for i in np.unique(near[near >= 0])):
                continue                        # a sliver of a region that is already there
            reg.update(thin=True, pole_px=tuple(pole))
        rid[m] = len(regs)
        regs.append(reg)
    # the rest of the diagram: connected pieces, named by the equilibrium grid nodes inside them
    if grid is not None and len(grid.get("xy", [])):
        from scipy.spatial import cKDTree
        gxy = ax.transData.transform(np.asarray(grid["xy"], float))
        ok = np.all(np.isfinite(gxy), axis=1)
        gxy, gk = gxy[ok], np.asarray(grid["keys"], int)[ok]
        table = [tuple(k) for k in grid["table"]]
        rem = inside & (rid == -1)
        if rem.any() and len(gxy):
            tree = cKDTree(gxy)
            spacing = float(np.median(tree.query(gxy, k=2)[0][:, 1])) if len(gxy) > 1 else 10.0
            gc, gr = R.cell(gxy[:, 0], gxy[:, 1])
            gc, gr = np.round(gc).astype(int), np.round(gr).astype(int)
            on = (gc >= 0) & (gc < R.W) & (gr >= 0) & (gr < R.H)
            lab, n = ndimage.label(rem)
            node_comp = np.zeros(len(gxy), int)
            node_comp[on] = lab[gr[on], gc[on]]
            for c in range(1, n + 1):
                comp = lab == c
                votes = gk[(node_comp == c) & (gk >= 0)]
                if len(votes):
                    ids, cnt = np.unique(votes, return_counts=True)
                    top = int(ids[np.argmax(cnt)])
                    uniform = len(votes) - cnt.max() < 3      # a second phase set on 3+ nodes: split
                else:
                    top, uniform = -1, True
                if uniform:
                    if top < 0:             # no grid node inside (a small region): ask at its thickest point
                        if probe is None or comp.sum() * R.res ** 2 < min_area_pt2:
                            continue
                        d = _edt(comp)
                        row, col = np.unravel_index(int(np.argmax(d)), d.shape)
                        if d[row, col] * R.res <= 1.0:
                            continue
                        try:
                            got = probe(*ax.transData.inverted().transform(R.to_px(col, row)))
                        except Exception:  # noqa: BLE001
                            got = None
                        if not got:
                            continue
                        got = tuple(got)
                        if got not in table:
                            table.append(got)
                        top = table.index(got)
                    pieces = [(table[top], comp)]
                else:               # several phase sets inside: split by the nearest grid node inside it
                    sel = np.flatnonzero((node_comp == c) & (gk >= 0))
                    sub_tree = cKDTree(gxy[sel])
                    cells = np.flatnonzero(comp.ravel())
                    d, j = sub_tree.query(R.px[cells])
                    kid = np.full(R.H * R.W, -1, int)
                    kid[cells] = np.where(d <= 2.5 * spacing, gk[sel][j], -1)
                    kid = kid.reshape(R.H, R.W)
                    pieces = []
                    for t_id in np.unique(kid[comp]):
                        if t_id < 0:
                            continue
                        sub, ns = ndimage.label(kid == t_id)
                        pieces += [(table[int(t_id)], sub == q) for q in range(1, ns + 1)]
                for key, m in pieces:
                    if m.sum() * R.res ** 2 < min_area_pt2 or _edt(m).max() * R.res <= 1.0:
                        continue
                    near = ndimage.binary_dilation(m, iterations=2) & (rid >= 0)
                    same = [int(i) for i in np.unique(rid[near]) if regs[int(i)]["key"] == key]
                    if same:                # an extension of a mapped region (mapping incomplete)
                        regs[same[0]]["mask"] |= m
                        rid[m] = same[0]
                        continue
                    if len(key) > 1 and (m.sum() * R.res ** 2 < 150.0 or _edt(m).max() * R.res < 3.0):
                        notes.append("+".join(key))     # small multi-phase area the mapping did not draw
                        continue
                    rid[m] = len(regs)
                    regs.append({"key": key, "mask": m, "src": "grid", "thin": False})
    # merge pieces of one region (a field traced as two lines, a line traced twice ...), but not two fields of the
    # same phases on either side of a phase boundary (e.g. FCC+M3V on both sides of a line compound)
    bound = R.zeros()
    for p in polys:
        for k in (0, 1):
            if p.get("ties"):
                R.polyline(bound, ax.transData.transform(np.asarray([t[k] for t in p["ties"]], float)))
    bound = ndimage.binary_dilation(bound, iterations=2)
    parent = list(range(len(regs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(regs)):
        if regs[i]["thin"]:
            continue
        gap = int(round(3.0 / R.res))           # pieces closer than 3 pt: one region (split again at necks below)
        bi = _bbox_of(regs[i]["mask"], gap + 1)
        if bi is None:
            continue
        di = ndimage.binary_dilation(regs[i]["mask"][bi[0]:bi[1] + 1, bi[2]:bi[3] + 1], iterations=gap)
        for j in range(i + 1, len(regs)):
            if regs[j]["thin"] or regs[j]["key"] != regs[i]["key"] or find(i) == find(j):
                continue
            mj = regs[j]["mask"][bi[0]:bi[1] + 1, bi[2]:bi[3] + 1]
            if not (di & mj).any():
                continue
            touch = ndimage.binary_dilation(regs[i]["mask"][bi[0]:bi[1] + 1, bi[2]:bi[3] + 1]) & mj
            if touch.sum() >= 6 and bound[bi[0]:bi[1] + 1, bi[2]:bi[3] + 1][touch].mean() > 0.6:
                continue                        # they meet along a drawn phase boundary
            parent[find(j)] = find(i)
    out: Dict[int, Dict] = {}
    for i, r in enumerate(regs):
        root = find(i)
        if root not in out:
            out[root] = dict(r)
        else:
            out[root]["mask"] = out[root]["mask"] | r["mask"]
            out[root]["src"] = out[root]["src"] if out[root]["src"] == r["src"] else "mixed"
    final = []
    for r in out.values():
        final += _split_necks(r, R) if not r["thin"] else [r]
    rid2 = np.full((R.H, R.W), -1, int)
    for i, r in enumerate(final):
        rid2[r["mask"]] = i
    return final, rid2, notes


def _split_necks(r: Dict, R: Raster, neck_pt: float = 3.0) -> List[Dict]:
    """A region pinched to a point (e.g. the two sides of a congruent maximum) -> one region per lobe."""
    from scipy import ndimage
    m = r["mask"]
    area = m.sum() * R.res ** 2
    if area < 400.0:
        return [r]
    it = int(round(neck_pt / R.res))
    core = ndimage.binary_erosion(m, iterations=it)
    lab, n = ndimage.label(core)
    if n < 2:
        return [r]
    sizes = ndimage.sum(core, lab, range(1, n + 1)) * R.res ** 2
    big = [i + 1 for i, a in enumerate(sizes) if a >= max(60.0, 0.08 * area)]
    if len(big) < 2:
        return [r]
    lobes = np.isin(lab, big)
    _, (ri, ci) = ndimage.distance_transform_edt(~lobes, return_indices=True)
    owner = lab[ri, ci]
    return [{**r, "mask": m & (owner == b)} for b in big]


def walls_of(rid: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Region edges (drawn phase boundaries), one cell each side."""
    from scipy import ndimage
    w = np.zeros_like(inside)
    a = np.where(inside, rid, -2)
    dv = a[1:, :] != a[:-1, :]
    dh = a[:, 1:] != a[:, :-1]
    w[1:, :] |= dv
    w[:-1, :] |= dv
    w[:, 1:] |= dh
    w[:, :-1] |= dh
    return ndimage.binary_dilation(w & inside, iterations=2)


# ---------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------
class _Measure:
    def __init__(self, ax, renderer, style):
        self.ax, self.r, self.st, self.cache = ax, renderer, style, {}

    def size(self, s: str, fs: float) -> Tuple[float, float]:
        """(width, height) in points of unrotated text at font size fs, including the background padding."""
        if s not in self.cache:
            t = self.ax.text(0, 0, s, fontsize=10.0, family=self.st["family"], fontweight=self.st["fontweight"],
                             transform=None)
            e = t.get_window_extent(self.r)
            t.remove()
            self.cache[s] = (e.width / self.ax.figure.dpi * 72 / 10.0, e.height / self.ax.figure.dpi * 72 / 10.0)
        w, h = self.cache[s]
        pad = 0.15 * fs if self.st["background"] else 0.0
        return w * fs + 2 * pad, h * fs + 2 * pad


def _draw(ax, s, xy, fs, rot, style, gid, info, color=None):
    kw = dict(fontsize=fs, color=color or style["color"], ha="center", va="center", rotation=rot,
              rotation_mode="anchor", family=style["family"], fontweight=style["fontweight"], zorder=6)
    if style["background"]:
        kw["bbox"] = {"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none",
                      "alpha": style["background_alpha"]}
    t = ax.text(xy[0], xy[1], s, **kw)
    t.set_gid(gid)
    t._calphad_label = dict(info)
    return t


def _mark_text(R: Raster, placed: np.ndarray, t, renderer, margin: float = 1.0):
    """Mark the (rotated) box of a drawn text as occupied."""
    if t.get_bbox_patch() is not None:
        margin += 0.15 * t.get_fontsize()
    p = text_corners(t, renderer, margin * R.ppp)
    rot = t.get_rotation() % 90
    if min(rot, 90 - rot) < 0.5:
        R.rect(placed, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
    else:
        placed |= R.poly(p)


def text_corners(t, renderer, pad: float = 0.0) -> np.ndarray:
    """Corners (display px, counter-clockwise) of the rotated box of a text's glyphs (not its background box or
    leader), grown by ``pad`` px.  Exact for any ha / va with rotation_mode 'anchor' (and 'default' when centred)."""
    from matplotlib.text import Text
    if hasattr(t, "update_positions"):
        t.update_positions(renderer)                # an annotation: place its text first
    rot = t.get_rotation()
    t.set_rotation(0)
    e = Text.get_window_extent(t, renderer)
    t.set_rotation(rot)
    ax_, ay_ = t.get_transform().transform(t.get_unitless_position())
    if t.get_rotation_mode() != "anchor":           # rotated about the centre of the box
        ax_, ay_ = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    out = []
    for u, v in ((e.x0 - pad, e.y0 - pad), (e.x1 + pad, e.y0 - pad), (e.x1 + pad, e.y1 + pad),
                 (e.x0 - pad, e.y1 + pad)):
        du, dv = u - ax_, v - ay_
        out.append((ax_ + du * c - dv * s, ay_ + du * s + dv * c))
    return np.array(out)


def quads_overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals (display px)."""
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() <= b.min() + 1e-6 or b.max() <= a.min() + 1e-6:
                return False
    return True


def _sizes(top: float, bottom: float, step: float = 0.5) -> List[float]:
    """Font sizes from ``top`` down to ``bottom`` (the smallest size a label may have)."""
    out, f = [], float(top)
    while f >= min(bottom, top) - 1e-9:
        out.append(round(f, 2))
        f -= step
    return out


def _principal_angle(m: np.ndarray) -> Optional[float]:
    """Direction of a skinny region (degrees, -90..90], or None if it is not elongated."""
    rows, cols = np.nonzero(m)
    if len(rows) <= 10:
        return None
    ev, vec = np.linalg.eigh(np.cov(np.vstack([cols, rows]).astype(float)))
    if not (ev[0] > 0 and math.sqrt(ev[1] / ev[0]) > 1.8):
        return None
    ang = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
    ang = (ang + 90) % 180 - 90
    if ang <= -89.5:
        ang = 90.0
    return round(ang, 1) if abs(ang) > 8 else None


def _fit(meas, s, sizes, blocked, avail, bb, angles, res, margin: float = 0.0) -> Dict[float, tuple]:
    """{angle: (largest font size that fits, centres where it fits)} for each angle where the text box, grown by
    ``margin`` pt on every side, fits at all."""
    choice = {}
    for ang in angles:
        lo, hi, found = 0, len(sizes) - 1, None
        while lo <= hi:                             # fitting is monotonic in size: binary search
            mid = (lo + hi) // 2
            w, h = meas.size(s, sizes[mid])
            ok, _ = _fits(blocked, _rect_kernel(w / 2 + margin, h / 2 + margin, ang, res), bb)
            ok &= avail[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1]
            if ok.any():
                found = (sizes[mid], ok)
                hi = mid - 1
            else:
                lo = mid + 1
        if found:
            choice[ang] = found
    return choice


def _pick_angle(choice: Dict[float, tuple], principal: Optional[float], fontsize: float) -> float:
    """Horizontal unless a turned label is clearly larger (> 1 pt); then along the region, else the least turned."""
    best = max(fs for fs, _ in choice.values())
    if 0.0 in choice and choice[0.0][0] >= min(best, fontsize) - 1.0:
        return 0.0
    top = [a for a in choice if choice[a][0] >= best - 1e-9]
    return min(top, key=lambda a: (principal is None or abs(a - principal) > 0.5, abs(a), -a))


# ---------------------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------------------
def clear_labels(ax) -> int:
    """Remove the labels (and the tick-label hider, and restore axis labels moved for outside labels)."""
    n = 0
    for t in list(ax.texts):
        if _is_ours(t):
            t.remove()
            n += 1
    for a in list(ax.artists):
        if isinstance(a, TickLabelHider):
            a.remove()
    for ln in list(ax.get_lines()):
        if str(ln.get_gid() or "").startswith(DOT_PREFIX):
            ln.remove()
    saved = getattr(ax, "_calphad_axis_labels", None)
    if saved:
        for name, (pad, auto, pos) in saved.items():
            axis = getattr(ax, name + "axis")
            axis.labelpad = pad
            if not auto:
                axis.set_label_coords(*pos)
        ax._calphad_axis_labels = {}
    return n


def label_diagram(ax, regions: Dict, grid: Optional[Dict] = None, key_at: Optional[Callable] = None, *,
                  kind: str = "binary", elements: Optional[Sequence[str]] = None, phase_labels=None,
                  liquids: Sequence[str] = (), invariants: bool = True, T_unit: str = "C",
                  style: Optional[Dict] = None) -> Dict:
    """Draw phase-region labels, outside names of stoichiometric phases and binary invariant temperatures.

    regions: {"polygons": [{"key", "xy" (plotted coordinates), "tri"?, "ties"?}], "compounds": [...],
              "invariants": [{"T_K", "y", "x0", "x1"}]} (see phase_regions.regions_from_lines).
    grid: equilibrium grid {"xy", "keys", "table"}; key_at(x, y) -> region key at a plotted point.
    Rules: a region label lies inside its region (turned if that lets it be larger), between ``fontsize`` and
    ``min_fontsize``, or is left out; an invariant temperature lies right above or below its line, in the
    adjacent region, between ``invariant_fontsize`` and ``min_fontsize``, or is left out; a stoichiometric phase
    (line compound / ternary point or line compound) is named outside the plot next to its composition.
    Returns {"labels": [...], "skipped": [...]}."""
    st = {**DEFAULT_STYLE, **{k: v for k, v in (style or {}).items() if k not in IGNORED_STYLE}}
    fig = ax.figure
    clear_labels(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    R = Raster(ax, st["res"])
    inside = np.ones((R.H, R.W), bool)
    if kind == "ternary":
        d = R.data
        inside = ((d[:, 0] >= -1e-9) & (d[:, 1] >= -1e-9) & (d[:, 0] + d[:, 1] <= 1 + 1e-9)).reshape(R.H, R.W)
    inner = inside & (_edt(inside) * R.res > 1.0)       # keep 1 pt from the frame / triangle edges
    regs, rid, grid_only = build_regions(ax, R, inside, regions, grid, probe=key_at)
    walls = walls_of(rid, inside)
    obst, n_pts = obstacles(ax, R, renderer)
    placed = R.zeros()
    meas = _Measure(ax, renderer, st)
    liquids = list(liquids) + [k for r in regs for k in r["key"] if is_liquid_name(k)]
    report: Dict[str, List] = {"labels": [], "skipped": []}
    used_gids: Dict[str, int] = {}
    inv_px = ax.transData.inverted()
    small = f"no room for {st['min_fontsize']:g} pt"

    def gid_for(prefix, token):
        g = prefix + _token(token)
        used_gids[g] = used_gids.get(g, 0) + 1
        return g if used_gids[g] == 1 else f"{g}-{used_gids[g]}"

    def data_of(col, row):
        px, py = R.to_px(col, row)
        return tuple(float(v) for v in inv_px.transform((float(px), float(py))))

    def check(key, xy, contains=False):
        if key_at is None:
            return None
        try:
            got = key_at(*xy)
        except Exception:  # noqa: BLE001 - a failed calculation: unconfirmed
            return None
        if got is None:
            return False
        return (key[0] in got) if contains else (tuple(got) == tuple(key))

    def record(t, kind_, key, xy, fs, rot, verified, **extra):
        report["labels"].append({"text": t.get_text(), "gid": t.get_gid(), "kind": kind_, "key": list(key),
                                 "x": xy[0], "y": xy[1], "fontsize": fs, "rotation": rot, "verified": verified,
                                 **extra})

    # -------------------------------------------------------------- regions (most constrained first)
    for r in regs:
        m = r["mask"] & inner
        r["cap"] = float(_edt(m).max()) * R.res if m.any() else 0.0
    sizes = _sizes(st["fontsize"], st["min_fontsize"])
    for r in sorted(regs, key=lambda r: r["cap"]):
        key = r["key"]
        s = region_label(key, elements, phase_labels, liquids)
        m = r["mask"] & inner
        avail = m & ~obst & ~placed & ~walls            # keep clear of the drawn boundary lines
        if r["thin"] or m.sum() * R.res ** 2 < 4.0 or r["cap"] < 0.5 or not avail.any():
            report["skipped"].append({"label": s, "reason": small, "_region": r})
            continue
        bb = _bbox_of(m)
        score = _edt(avail) * R.res
        principal = _principal_angle(m) if st["rotate"] else None
        angles = [0.0] + ([principal] if principal is not None else [])
        more = [a for a in EXTRA_ANGLES if all(abs(a - b) > 0.5 for b in angles)] if st["rotate"] else []
        tried = R.zeros()
        placed_ok, unconfirmed = False, False
        for attempt in range(3):
            blocked = ~avail | tried
            choice = _fit(meas, s, sizes, blocked, avail, bb, angles, R.res, st["margin"])
            if more and (not choice or max(fs for fs, _ in choice.values()) < sizes[0] - 1e-9):
                choice.update(_fit(meas, s, sizes, blocked, avail, bb, more, R.res, st["margin"]))   # turned
            if not choice:
                break
            ang = _pick_angle(choice, principal, st["fontsize"])
            fs, ok = choice[ang]
            sc = np.where(ok, score[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1], -1.0)
            top = sc >= sc.max() - 1.0              # nearly as clear as the best: take the most central one
            tr_, tc_ = np.nonzero(top)
            k = int(np.argmin((tr_ - tr_.mean()) ** 2 + (tc_ - tc_.mean()) ** 2))
            row, col = tr_[k] + bb[0], tc_[k] + bb[2]
            xy = data_of(col, row)
            v = check(key, xy)
            if v is False:
                unconfirmed = True
                w, h = meas.size(s, fs)
                R.disk(tried, *R.to_px(col, row), max(w, h) / 2)
                continue
            t = _draw(ax, s, xy, fs, ang, st, gid_for("phase_label:", ".".join(key)),
                      {"kind": "region", "key": list(key)})
            _mark_text(R, placed, t, renderer)
            record(t, "region", key, xy, fs, ang, v)
            placed_ok = True
            break
        if not placed_ok:
            report["skipped"].append({"label": s, "reason": "not confirmed by equilibrium" if unconfirmed
                                      else small, "_region": r})

    # -------------------------------------------------------------- binary invariant temperatures
    if kind == "binary" and invariants:
        inv_sizes = _sizes(st["invariant_fontsize"], st["min_fontsize"])
        gap = 1.8                                       # pt between the line and the text box
        for inv in invariant_lines(regions.get("invariants") or []):
            celsius = str(T_unit).upper().startswith("C")
            T = inv["T_K"] - 273.15 if celsius else inv["T_K"]
            s = f"{T:.0f} °C" if celsius else f"{T:.0f} K"
            (xa, y), (xb, _) = ax.transData.transform([(inv["x0"], inv["y"]), (inv["x1"], inv["y"])])
            xa, xb = sorted((xa, xb))
            if not (R.y0 <= y <= R.y0 + R.H * R.s):
                continue
            xa, xb = max(xa, R.x0), min(xb, R.x0 + R.W * R.s)
            blocked = ~inner | walls | obst | placed
            clear = _edt(~blocked) * R.res
            own = R.zeros()                             # the invariant line itself (and its wall)
            r_lo, r_hi = (int(v) for v in np.clip(np.round(R.cell(0, [y - 2.5 * R.ppp, y + 2.5 * R.ppp])[1]),
                                                  0, R.H - 1))
            c_lo, c_hi = (int(v) for v in np.clip(np.round(R.cell([xa, xb], 0)[0]), 0, R.W - 1))
            own[r_lo:r_hi + 1, c_lo:c_hi + 1] = True
            others = blocked & ~own
            mg = st["margin"]
            best = None
            for fs in inv_sizes:
                w, h = meas.size(s, fs)
                lo, hi = xa + w * R.ppp / 2, xb - w * R.ppp / 2     # the text stays over its line
                if hi < lo:
                    continue
                for sgn in (1, -1):                     # right above, then right below the line
                    cy = y + sgn * (gap + h / 2) * R.ppp
                    xs = np.unique(np.concatenate([[lo, hi], np.arange(lo, hi, 2 * R.ppp)]))
                    for cx in xs:
                        r0, r1, c0, c1 = _box_cells(R, cx, cy, w, h)
                        if c0 < 0 or r0 < 0 or c1 >= R.W or r1 >= R.H or blocked[r0:r1 + 1, c0:c1 + 1].any():
                            continue
                        g0, g1, g2, g3 = _box_cells(R, cx, cy, w + 2 * mg, h + 2 * mg)    # clear of other lines
                        if others[max(0, g0):g1 + 1, max(0, g2):g3 + 1].any():
                            continue
                        end = 2.0 if cx in (lo, hi) else 0.0
                        score = min(float(clear[r0:r1 + 1, c0:c1 + 1].min()), 4.0) + end + (1.0 if sgn > 0 else 0)
                        if best is None or score > best[0]:
                            best = (score, cx, cy, fs)
                if best is not None:
                    break
            if best is None:
                report["skipped"].append({"label": s, "reason": small})
                continue
            xy = tuple(float(v) for v in inv_px.transform((best[1], best[2])))
            t = _draw(ax, s, xy, best[3], 0.0, st, gid_for("invariant_label:", s.replace(" °", "")),
                      {"kind": "invariant", "T_K": inv["T_K"], "key": list(inv.get("key", ()))},
                      color=st["invariant_color"])
            _mark_text(R, placed, t, renderer)
            record(t, "invariant", inv.get("key", ()), xy, best[3], 0.0, None)

    # -------------------------------------------------------------- outside the plot: stoichiometric phases, and
    # single-phase regions too small for a label inside that reach an edge (not terminal phases of a binary)
    # (ternary: if it touches no edge, or is a compound inside the triangle, with a leader line instead)
    name_of = lambda key: region_label(key, elements, phase_labels, liquids)      # noqa: E731
    edges, ticks = _edges(ax, kind)
    items, leaders = _compound_items(ax, kind, regions.get("compounds") or [], edges, check, report, name_of)
    more_items, more_leaders = _single_phase_items(ax, kind, R, edges, report, check, liquids, regions, grid,
                                                   {it["key"] for it in items + leaders}, name_of)
    _label_outside(ax, kind, items + more_items, edges, ticks, st, meas, renderer, gid_for, record)
    for sk in report["skipped"]:
        sk.pop("_region", None)
    if leaders or more_leaders:
        for t in ax.texts:                              # the names outside are obstacles for leader labels
            if _is_ours(t) and (getattr(t, "_calphad_label", None) or {}).get("outside"):
                _mark_text(R, placed, t, renderer)
        _place_leaders(ax, R, leaders + more_leaders, st, meas, renderer, inside, inner, walls, obst, placed,
                       gid_for, record)
    for k in sorted(set(grid_only)):
        report["skipped"].append({"label": region_label(k.split("+"), elements, phase_labels, liquids),
                                  "reason": "small area found by the equilibrium grid but not drawn by the mapping"})
    report["n_obstacle_points"] = n_pts
    return report


def invariant_lines(invariants: Sequence[Dict], dT: float = 0.5) -> List[Dict]:
    """Invariant lines, hottest first, each once: entries within ``dT`` K whose composition ranges overlap are one
    line (an invariant found twice by the mapping, or the same line drawn from two sides) and get one temperature.
    Distinct invariants a few K apart stay separate: each label then sits by its own line (the other line blocks the
    side between them), so they stay unambiguous."""
    out: List[Dict] = []
    for inv in sorted(invariants, key=lambda i: -i["T_K"]):
        x0, x1 = sorted((float(inv["x0"]), float(inv["x1"])))
        same = next((j for j in out if abs(j["T_K"] - inv["T_K"]) < dT and x0 <= j["x1"] + 1e-9
                     and j["x0"] <= x1 + 1e-9), None)
        if same is not None:
            same["x0"], same["x1"] = min(same["x0"], x0), max(same["x1"], x1)
            continue
        out.append({**inv, "x0": x0, "x1": x1})
    return out


# ---------------------------------------------------------------------------------------------------------
# names outside the plot: stoichiometric phases, and single-phase regions too small inside
# ---------------------------------------------------------------------------------------------------------
def _tick_out(axis) -> float:
    """How far (pt) the major ticks of an axis stick out of the plot."""
    try:
        t = axis.get_major_ticks()[0]
        d, size = t._tickdir, float(t._size)
    except Exception:  # noqa: BLE001
        d, size = matplotlib.rcParams["xtick.direction"], float(matplotlib.rcParams["xtick.major.size"])
    return size if d == "out" else size / 2 if d == "inout" else 0.0


def _spread(want: Sequence[float], foot: Sequence[float], lo: float, hi: float, gap: float) -> np.ndarray:
    """Positions along an edge closest to ``want`` (least squares) with neighbours at least (foot_i + foot_j) / 2
    + gap apart, within [lo, hi] (pool-adjacent-violators on the shifted positions)."""
    order = np.argsort(want)
    w = np.asarray(want, float)[order]
    f = np.asarray(foot, float)[order]
    off = np.concatenate([[0.0], np.cumsum((f[:-1] + f[1:]) / 2 + gap)])
    q = w - off
    blocks = []                                         # [sum, count]
    for v in q:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s_, c_ = blocks.pop()
            blocks[-1][0] += s_
            blocks[-1][1] += c_
    fit = np.concatenate([[b[0] / b[1]] * b[1] for b in blocks])
    fit = np.clip(fit, lo, max(lo, hi - off[-1]))
    out = np.empty(len(w))
    out[order] = fit + off
    return out


def _edges(ax, kind):
    """{edge: (start, end, outward normal)} in display px, and how far ticks stick out at each edge (pt)."""
    if kind == "binary":
        a, b = ax.transAxes.transform([(0.0, 0.0), (1.0, 0.0)])
        return {"bottom": (a, b, np.array([0.0, -1.0]))}, {"bottom": _tick_out(ax.xaxis)}
    c = ax.transData.transform([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    mid = c.mean(axis=0)
    edges = {}
    for name, (p, q) in (("bottom", (c[0], c[1])), ("left", (c[0], c[2])), ("right", (c[1], c[2]))):
        u = (q - p) / np.linalg.norm(q - p)
        n = np.array([-u[1], u[0]])
        edges[name] = (p, q, n if n @ ((p + q) / 2 - mid) > 0 else -n)
    return edges, {"bottom": _tick_out(ax.xaxis), "left": _tick_out(ax.yaxis), "right": 0.0}


def _on_edge(ax, edge, P):
    """Data point exactly on a ternary edge (or the binary x axis) for a display point on it."""
    xy = tuple(float(v) for v in ax.transData.inverted().transform(P))
    return (xy[0], 0.0) if edge == "bottom" else (0.0, xy[1]) if edge == "left" else (xy[0], 1.0 - xy[0])


def _compound_items(ax, kind, compounds, edges, check, report, name_of, edge_tol: float = 5e-3):
    """Where each stoichiometric phase is named: binary - below the x axis at its composition; ternary - on (or
    reaching) an edge: beyond that edge (bottom: along the line of constant x; slanted edges: square to the edge);
    inside the triangle: with a leader line (and a dot at a point compound).  Returns (edge items, leader items)."""
    xlo, xhi = sorted(ax.get_xlim())
    items, leaders = [], []
    for comp in compounds:
        key = tuple(comp["key"])
        s = name_of(key)
        if kind == "binary":
            if not (xlo <= comp["x"] <= xhi):
                continue
            P = np.array([ax.transData.transform((comp["x"], ax.get_ylim()[0]))[0], edges["bottom"][0][1]])
            edge, probe, xy = "bottom", (comp["x"], 0.5 * (comp["y0"] + comp["y1"])), (float(comp["x"]), 0.0)
        else:
            pts = [np.asarray(p, float) for p in comp["line"]] if "line" in comp else [np.array([comp["x"], comp["y"]])]
            probe = np.mean(pts, axis=0)
            if len(pts) == 1 and max(probe[0], probe[1], 1 - probe[0] - probe[1]) > 1 - 2e-3:
                continue                                # a pure element (corner): named by the corner label
            to_mid = np.array([1 / 3, 1 / 3]) - probe   # checked a hair inside (an edge composition is degenerate)
            probe = tuple(float(v) for v in probe + 2e-3 * to_mid / max(np.linalg.norm(to_mid), 1e-12))
            best = None
            for p in pts:
                for rank, (edge_, dist) in enumerate((("right", 1 - p[0] - p[1]), ("bottom", p[1]), ("left", p[0]))):
                    cand = (round(float(dist), 3), rank, edge_, p)
                    if best is None or cand[:2] < best[:2]:
                        best = cand
            dist, _, edge, p = best
            if dist > edge_tol:                         # a ternary compound inside the triangle: leader line
                ok = check(key, probe, contains=True)
                if ok is False:
                    report["skipped"].append({"label": s, "reason": "not confirmed by equilibrium"})
                    continue
                leaders.append({"key": key, "s": s, "target": tuple(float(v) for v in np.mean(pts, axis=0)),
                                "dot": len(pts) == 1, "ok": ok, "kind": "compound", "probe": probe})
                continue
            if edge == "bottom":                        # along the line of constant x (the bottom axis reading)
                xy = (float(p[0]), 0.0)
                P = ax.transData.transform(xy)
            else:                                       # square to the edge
                e0, e1, _ = edges[edge]
                u = (e1 - e0) / np.linalg.norm(e1 - e0)
                P = e0 + ((ax.transData.transform(p) - e0) @ u) * u
                xy = _on_edge(ax, edge, P)
        ok = check(key, probe, contains=True)
        if ok is False:
            report["skipped"].append({"label": s, "reason": "not confirmed by equilibrium"})
            continue
        items.append({"key": key, "s": s, "edge": edge, "P": np.asarray(P, float), "xy": xy, "probe": probe,
                      "ok": ok, "kind": "compound"})
    return items, leaders


def _is_terminal(key, grid, pts_data, xlim, tol: float = 2e-3) -> bool:
    """A binary end-member phase: stable next to a pure element (the first / last column of the equilibrium grid),
    or its field reaches the left / right side of the plot."""
    lo, hi = sorted(xlim)
    span = hi - lo
    if len(pts_data) and (pts_data[:, 0].min() - lo < tol * span or hi - pts_data[:, 0].max() < tol * span):
        return True
    if grid is not None and len(grid.get("xy", [])):
        xy = np.asarray(grid["xy"], float)
        table = [tuple(k) for k in grid["table"]]
        keys = np.asarray(grid["keys"], int)
        xs = np.unique(np.round(xy[:, 0], 12))
        for x_end in (xs.min(), xs.max()):
            sel = np.isclose(xy[:, 0], x_end) & (keys >= 0)
            if any(key[0] in table[k] for k in np.unique(keys[sel])):
                return True
    return False


def _single_phase_items(ax, kind, R, edges, report, check, liquids, regions, grid, taken, name_of,
                        reach_pt: float = 2.0):
    """Single-phase fields without a label inside - too small for one, or too thin to be a region of the plot at
    all (only their boundary points, ``regions["phase_points"]``, are known).  Each region is handled on its own:
    a second region of a phase that is named elsewhere still gets its own name.

    Binary: an intermediate phase is named below the x axis at its mid composition; the liquid and terminal
    (end-member) phases never are.  Ternary: a field touching an edge (the phase exists in that binary) is named
    beyond that edge, at the middle of its contact (the longer one); a field inside the triangle gets a leader
    label.  Returns (edge items, leader items); the report's omitted list keeps only what got nowhere."""
    ppp = R.ppp
    inv = ax.transData.inverted()
    named = {tuple(t._calphad_label["key"]) for t in ax.texts if _is_ours(t) and hasattr(t, "_calphad_label")
             and t._calphad_label.get("kind") == "region"}
    compounds = {tuple(c["key"]) for c in regions.get("compounds") or []}
    cands = []                                          # (key, label, display points, probe (data), skip entry)
    for sk in report["skipped"]:
        r = sk.get("_region")
        if r is None or len(r["key"]) != 1:
            continue
        m = r["mask"]
        if m.sum() >= 3:
            rows, cols = np.nonzero(m)
            pts = R.px.reshape(R.H, R.W, 2)[rows, cols]
            d = _edt(m)                                 # checked at the thickest point of the region
            row, col = np.unravel_index(int(np.argmax(d)), d.shape)
            pole = R.to_px(col, row)
        elif r.get("pole_px") is not None:
            pts = np.array([r["pole_px"]], float)
            pole = r["pole_px"]
        else:
            continue
        cands.append((tuple(r["key"]), sk["label"], pts, tuple(float(v) for v in inv.transform(pole)), sk))
    seen = {c[0] for c in cands}
    for n, pp in sorted((regions.get("phase_points") or {}).items()):
        key = (n,)
        if key in seen or key in named or key in taken or key in compounds:
            continue
        data = np.asarray(pp, float).reshape(-1, 2)
        pts = ax.transData.transform(data)
        inb = (pts[:, 0] >= R.x0 - 1) & (pts[:, 0] <= R.x0 + R.W * R.s + 1) & \
            (pts[:, 1] >= R.y0 - 1) & (pts[:, 1] <= R.y0 + R.H * R.s + 1)
        if not inb.any():
            continue
        pts, data = pts[inb], data[inb]
        c = data[int(np.argmin(np.hypot(*(data - data.mean(axis=0)).T)))]
        if kind == "ternary":                           # checked a hair inside (an edge composition is degenerate)
            to_mid = np.array([1 / 3, 1 / 3]) - c
            c = c + 2e-3 * to_mid / max(np.linalg.norm(to_mid), 1e-12)
        cands.append((key, name_of(key), pts, (float(c[0]), float(c[1])), None))
    edge_items, leaders = [], []
    for key, s, pts, probe, sk in cands:
        if sk is None and (key in named or key in taken):
            continue                                    # boundary points only, and the phase is named already
        best = None
        if kind == "binary":
            if key[0] in liquids or is_liquid_name(key[0]) or \
                    _is_terminal(key, grid, inv.transform(pts), ax.get_xlim()):
                continue                                # never the liquid or an end-member phase
            x0, x1 = pts[:, 0].min(), pts[:, 0].max()
            best = (x1 - x0, "bottom", np.array([(x0 + x1) / 2, edges["bottom"][0][1]]))
        else:
            for edge, (e0, e1, n) in edges.items():
                u = (e1 - e0) / np.linalg.norm(e1 - e0)
                near = np.abs((pts - e0) @ n) / ppp < reach_pt + R.res
                if not near.any():
                    continue
                t_ = (pts[near] - e0) @ u
                cand = (float(t_.max() - t_.min()), edge, e0 + 0.5 * (t_.max() + t_.min()) * u)
                if best is None or cand[0] > best[0]:
                    best = cand
        ok = check(key, probe, contains=True)
        if ok is False:
            continue
        if sk is not None:                              # each region on its own (a second field of a phase too)
            report["skipped"].remove(sk)
        taken = taken | {key}
        if best is None:                                # a ternary-only field inside the triangle: leader line
            leaders.append({"key": key, "s": s, "target": probe, "dot": False, "ok": ok, "kind": "region",
                            "probe": probe})
            continue
        _, edge, P = best
        xy = (float(inv.transform(P)[0]), 0.0) if kind == "binary" else _on_edge(ax, edge, P)
        edge_items.append({"key": key, "s": s, "edge": edge, "P": np.asarray(P, float), "xy": xy, "probe": probe,
                           "ok": ok, "kind": "region"})
    return edge_items, leaders


def _free_spot(R: Raster, blocked: np.ndarray, hard: np.ndarray, w: float, h: float, target_px,
               reach_pt: float = 90.0, min_pt: float = 4.0):
    """Nearest centre (display px) for a w x h pt text box near target_px (not on a blocked cell), whose straight
    line to the target crosses as few ``hard`` cells (other labels) as possible - ideally none.  Returns
    ((x, y), cells of the line, crossings) or None."""
    pc, pr = (float(q) for q in R.cell(*target_px))
    kern = _rect_kernel(w / 2, h / 2, 0.0, R.res)
    reach = int((reach_pt + max(w, h)) / R.res)
    box = (max(0, int(pr) - reach), min(R.H - 1, int(pr) + reach), max(0, int(pc) - reach),
           min(R.W - 1, int(pc) + reach))
    if box[0] > box[1] or box[2] > box[3]:
        return None
    ok, _ = _fits(blocked, kern, box)
    rows, cols = np.nonzero(ok)
    if not len(rows):
        return None
    rows, cols = rows + box[0], cols + box[2]
    dist = np.hypot(cols - pc, rows - pr) * R.res
    good = dist >= math.hypot(w / 2, h / 2) + min_pt
    rows, cols, dist = rows[good], cols[good], dist[good]
    if not len(rows):
        return None
    order = np.argsort(dist)[:3000]
    rows, cols = rows[order], cols[order]
    f = np.linspace(0.0, 1.0, int(min(500, reach_pt * 2 + 40)))[None, :]
    sr = rows[:, None] + (pr - rows[:, None]) * f
    sc = cols[:, None] + (pc - cols[:, None]) * f
    in_box = (np.abs(sc - cols[:, None]) * R.res <= w / 2 + 0.5) & (np.abs(sr - rows[:, None]) * R.res <= h / 2 + 0.5)
    near_t = np.hypot(sc - pc, sr - pr) * R.res < 1.5
    ri = np.clip(np.round(sr).astype(int), 0, R.H - 1)
    ci = np.clip(np.round(sc).astype(int), 0, R.W - 1)
    on_line = ~in_box & ~near_t
    crosses = (hard[ri, ci] & on_line).sum(axis=1)
    k = int(np.argmin(crosses))                         # the nearest with the fewest crossings
    line = np.zeros_like(hard)
    for a, b in zip(ri[k][on_line[k]], ci[k][on_line[k]]):
        line[max(0, a - 1):a + 2, max(0, b - 1):b + 2] = True
    return tuple(float(q) for q in R.to_px(cols[k], rows[k])), line, int(crosses[k])


def _place_leaders(ax, R, items, st, meas, renderer, inside, inner, walls, obst, placed, gid_for, record):
    """Ternary-only single-phase fields and compounds inside the triangle: the label goes to the nearest free
    space - inside the triangle on a free area (no line, no other label), else just outside it - with a thin
    leader line to the field (a dot at a point compound).  Never left out."""
    inv = ax.transData.inverted()
    lines = R.zeros()
    sizes = _sizes(st["fontsize"], st["min_fontsize"])
    outside = ~inside & (_edt(~inside) * R.res > 2.0)
    for it in items:
        tpx = ax.transData.transform(it["target"])
        spot = None
        for blocked in (~inner | walls | obst | placed | lines, ~outside | obst | placed | lines):
            for fs in sizes:
                w, h = meas.size(it["s"], fs)
                got = _free_spot(R, blocked, obst | placed, w + 2 * st["margin"], h + 2 * st["margin"], tpx)
                if got is not None and (spot is None or got[2] < spot[0][2]):
                    spot = (got, fs)
                if got is not None and got[2] == 0:
                    break
            if spot is not None and spot[0][2] == 0:
                break
        if spot is None:                                # nowhere free: right beside the field anyway
            fs = sizes[-1]
            spot = (((tpx[0] + 12 * R.ppp, tpx[1] + 12 * R.ppp), R.zeros(), 0), fs)
        (cx, cy), line, _ = spot[0]
        fs = spot[1]
        xy = tuple(float(v) for v in inv.transform((cx, cy)))
        gid = gid_for("phase_label:", ".".join(it["key"]))
        t = ax.annotate(it["s"], xy=it["target"], xycoords="data", xytext=xy, textcoords="data", fontsize=fs,
                        color=st["color"], ha="center", va="center", family=st["family"],
                        fontweight=st["fontweight"], zorder=6, annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": LEADER_COLOR, "shrinkA": 1.5,
                                    "shrinkB": 2.0 if it["dot"] else 0.0})
        t.set_gid(gid)
        t._calphad_label = {"kind": it["kind"], "key": list(it["key"]), "leader": True,
                            "at": [float(v) for v in it["probe"]]}
        if it["dot"]:
            ax.plot([it["target"][0]], [it["target"][1]], "o", ms=2.5, color=st["color"], zorder=6,
                    gid=DOT_PREFIX + gid)
        _mark_text(R, placed, t, renderer)
        placed |= line                                  # later labels keep off this leader line
        lines |= line
        record(t, it["kind"], it["key"], it["probe"], fs, 0.0, it["ok"], leader=True, text_xy=list(xy))


def _label_outside(ax, kind, items, edges, ticks, st, meas, renderer, gid_for, record):
    """Write the names outside the plot: binary - below the x axis (text at 60°); ternary - beyond the edge
    (bottom: 60°; slanted edges: ±30°, square to the edge).  Colliding neighbours are spread along the edge (with
    a thin guide line when a label had to move); the tick labels they cover are hidden by a
    :class:`TickLabelHider`."""
    if not items:
        return
    ppp = ax.figure.dpi / 72.0
    fs = st["fontsize"]
    for edge, (e0, e1, n) in edges.items():
        group = [it for it in items if it["edge"] == edge]
        if not group:
            continue
        rot, ha = EDGE_TEXT[edge]
        d = np.array([math.cos(math.radians(rot + (180.0 if ha == "right" else 0.0))),
                      math.sin(math.radians(rot + (180.0 if ha == "right" else 0.0)))])
        e = np.array([-d[1], d[0]])
        nd = max(float(n @ d), 0.2)
        u = (e1 - e0) / np.linalg.norm(e1 - e0)
        L = float(np.linalg.norm(e1 - e0)) / ppp      # edge length, pt
        heights = [meas.size(it["s"], fs)[1] for it in group]
        foot = [h / nd for h in heights]                # width of a label strip along the edge
        want = [float((it["P"] - e0) @ u) / ppp for it in group]
        pos = _spread(want, foot, 2.0, L - 2.0, 1.0)
        for it, h, wnt, p in zip(group, heights, want, pos):
            t_out = (2.0 + ticks[edge] + h / 2 * abs(float(n @ e))) / nd    # from the edge to the anchor, pt
            off = (p - wnt) * u + t_out * d
            xy = it["xy"]
            kw = dict(fontsize=fs, color=st["color"], ha=ha, va="center", rotation=rot, rotation_mode="anchor",
                      family=st["family"], fontweight=st["fontweight"], zorder=6, annotation_clip=False,
                      textcoords="offset points", xytext=(float(off[0]), float(off[1])))
            if abs(p - wnt) > 2.5:                       # moved along the edge: a thin guide to the composition
                kw["arrowprops"] = {"arrowstyle": "-", "lw": 0.5, "color": st["color"], "shrinkA": 1.0,
                                    "shrinkB": 0.0}
            if kind == "binary":
                t = ax.annotate(it["s"], xy=xy, xycoords=("data", "axes fraction"), **kw)
            else:
                t = ax.annotate(it["s"], xy=xy, xycoords="data", **kw)
            t.set_gid(gid_for("phase_label:", ".".join(it["key"])))
            t._calphad_label = {"kind": it["kind"], "key": list(it["key"]), "outside": edge,
                                "at": [float(v) for v in it["probe"]]}
            record(t, it["kind"], it["key"], it["probe"], fs, rot, it["ok"], outside=edge, edge_xy=list(xy),
                   guide="arrowprops" in kw)
    if any((getattr(t, "_calphad_label", None) or {}).get("outside") for t in ax.texts):
        ax.add_artist(TickLabelHider())
        _clear_axis_labels(ax, renderer)


class TickLabelHider(matplotlib.artist.Artist):
    """Hides the tick labels (the composition numbers) covered by a phase label written outside the plot - only
    those.  It acts at every draw, before the axes draw their ticks, so a moved or deleted label gives its tick
    label back.  Invisible itself and not part of the layout."""

    def __init__(self, margin_pt: float = 1.0):
        super().__init__()
        self.margin = float(margin_pt)
        self.set_zorder(-1e6)
        self.set_gid("label_tick_hider")
        self.set_in_layout(False)

    def get_window_extent(self, renderer=None):
        from matplotlib.transforms import Bbox
        return Bbox.null()

    def get_tightbbox(self, renderer=None):
        return None

    def draw(self, renderer):
        ax = self.axes
        if ax is None:
            return
        quads = [text_corners(t, renderer) for t in ax.texts
                 if t.get_visible() and _is_ours(t) and (getattr(t, "_calphad_label", None) or {}).get("outside")]
        pad = self.margin * ax.figure.dpi / 72.0
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()           # the ticks (and texts) this draw will show
            except Exception:  # noqa: BLE001
                continue
            for tick in ticks:
                lab = tick.label1
                if getattr(lab, "_calphad_hidden", False):
                    lab.set_visible(True)
                    lab._calphad_hidden = False
                if not self.get_visible() or not lab.get_visible() or not lab.get_text().strip():
                    continue
                q = text_corners(lab, renderer, pad)
                if any(quads_overlap(q, p) for p in quads):
                    lab.set_visible(False)
                    lab._calphad_hidden = True


def _clear_axis_labels(ax, renderer, step_pt: float = 2.0, max_steps: int = 60):
    """Move the axis titles (Mole fraction ..) away from the plot until no outside phase label touches them.  The
    original positions are kept on the axes (``_calphad_axis_labels``) and restored by :func:`clear_labels`."""
    fig = ax.figure
    fig.canvas.draw()                                   # tick labels hidden, axis titles placed
    quads = [text_corners(t, renderer, 1.0) for t in ax.texts
             if _is_ours(t) and (getattr(t, "_calphad_label", None) or {}).get("outside")]
    saved = getattr(ax, "_calphad_axis_labels", None) or {}
    ppp = fig.dpi / 72.0
    for name in ("x", "y"):
        axis = getattr(ax, name + "axis")
        lab = axis.label
        if not lab.get_visible() or not lab.get_text().strip():
            continue
        auto = bool(getattr(axis, "_autolabelpos", True))
        orig = (axis.labelpad, auto, tuple(lab.get_position()))
        moved = False
        for _ in range(max_steps):
            if auto:
                try:
                    axis._update_label_position(renderer)
                except Exception:  # noqa: BLE001
                    break
            if not any(quads_overlap(text_corners(lab, renderer), q) for q in quads):
                break
            moved = True
            if auto:
                axis.labelpad += step_pt
            else:                                       # placed by coordinates (ternary y title): move outwards
                bb = ax.bbox
                x, y = lab.get_position()
                c = ax.transData.transform([(0.0, 0.0), (0.0, 1.0)])     # the left edge
                v = c[1] - c[0]
                n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
                axis.set_label_coords(x + n[0] * step_pt * ppp / bb.width, y + n[1] * step_pt * ppp / bb.height)
        if moved and name not in saved:
            saved[name] = orig
    ax._calphad_axis_labels = saved


# ---------------------------------------------------------------------------------------------------------
# editing labels afterwards
# ---------------------------------------------------------------------------------------------------------
def find_labels(ax, id: Optional[str] = None, text: Optional[str] = None) -> List:
    """Labels by gid ('phase_label:LIQUID.FCC_A1', or the part after the colon, or a prefix such as
    'invariant_label') or by their text."""
    labs = [t for t in ax.texts if _is_ours(t)]
    if id:
        want = str(id).strip()
        hits = [t for t in labs if t.get_gid() == want or t.get_gid().split(":", 1)[1] == want]
        if not hits and want.rstrip(":") in ("phase_label", "invariant_label"):
            hits = [t for t in labs if t.get_gid().startswith(want.rstrip(":") + ":")]
        if not hits:
            hits = [t for t in labs if t.get_gid().lower() == want.lower()
                    or t.get_gid().split(":", 1)[1].lower() == want.lower()]
        return hits
    if text is not None:
        want = str(text).strip()
        hits = [t for t in labs if t.get_text() == want]
        return hits or [t for t in labs if t.get_text().replace(" ", "").lower() == want.replace(" ", "").lower()]
    return labs


def remove_label(t):
    """Delete a label with what belongs to it (its leader line is part of it; the dot of a ternary point compound
    is a separate marker)."""
    ax, gid = t.axes, t.get_gid()
    t.remove()
    if ax is not None:
        for ln in list(ax.get_lines()):
            if ln.get_gid() == DOT_PREFIX + str(gid):
                ln.remove()


def move_label(t, dx_pt: float = 0.0, dy_pt: float = 0.0, xy=None):
    """Move a label by (dx, dy) points, or to data position xy (a leader line follows: it is drawn from the text
    to the fixed target)."""
    from matplotlib.text import Annotation
    ax = t.axes
    f = ax.figure.dpi / 72.0
    if isinstance(t, Annotation) and t.anncoords == "offset points":     # a name outside the plot
        ox, oy = t.xyann
        if xy is not None:
            ax.figure.canvas.draw()
            t.update_positions(ax.figure.canvas.get_renderer())
            px, py = t._get_xy_display()
            qx, qy = ax.transData.transform((float(xy[0]), float(xy[1])))
            t.xyann = ((qx - px) / f, (qy - py) / f)
        else:
            t.xyann = (ox + dx_pt, oy + dy_pt)
        return
    if xy is not None:
        t.set_position((float(xy[0]), float(xy[1])))
        return
    px, py = ax.transData.transform(t.get_position())
    t.set_position(tuple(ax.transData.inverted().transform((px + dx_pt * f, py + dy_pt * f))))
