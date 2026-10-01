"""Phase-region labels on a phase diagram: region masks, obstacle avoidance and text placement.

Needs numpy, scipy and matplotlib.  Everything works on a raster of the axes in points (``res`` pt per cell):

1. regions: exact polygons (two-phase fields from the ZPF lines, tie-triangles) are painted first; the rest of
   the diagram is split by an equilibrium grid (nearest grid node) into connected components; pieces of one
   region are merged; grid slivers along a polygon edge are dropped;
2. obstacles: markers of data points (scatter collections, lines with markers), other texts, the legend, and
   every label already placed;
3. each label goes to the point of its region farthest from the region edge and the obstacles where the text
   box (optionally turned along a skinny region) fits, starting at ``fontsize`` and shrinking to
   ``min_fontsize``; otherwise it is written outside with a thin leader line to that point;
4. every position is confirmed with ``key_at`` (an equilibrium calculation) before the label is drawn.

Each label is one Text artist (``gid`` "phase_label:<PHASES>" / "invariant_label:<T>"), so it stays one editable
text object in SVG (``svg.fonttype = 'none'``) and PDF (``pdf.fonttype = 42``): see :func:`save_vector`.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import matplotlib.path as mpath

from .phase_names import is_liquid_name, region_label

FONT_FAMILY = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
LABEL_PREFIXES = ("phase_label:", "invariant_label:")
DEFAULT_STYLE = {
    "fontsize": 9.5, "min_fontsize": 6.0, "color": "black", "family": FONT_FAMILY, "fontweight": "normal",
    "background": True, "background_alpha": 0.75, "rotate": True, "leader": True,
    "invariant_fontsize": 8.0, "invariant_color": "#b2182b", "invariant_type": False,
    "res": 0.5,
}


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


def _rot_cells(R: "Raster", cx: float, cy: float, w_pt: float, h_pt: float, ang: float):
    """Index arrays of the cells under a rotated w x h pt box centred at display (cx, cy), or None if it leaves
    the raster."""
    k = _rect_kernel(w_pt / 2, h_pt / 2, ang, R.res)
    K = k.shape[0] // 2
    c, r = R.cell(cx, cy)
    c, r = int(round(float(c))), int(round(float(r)))
    if r - K < 0 or c - K < 0 or r + K >= R.H or c + K >= R.W:
        return None
    rr, cc = np.nonzero(k)
    return rr + r - K, cc + c - K


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


def _draw(ax, s, xy, fs, rot, style, gid, info, color=None, leader_to=None):
    kw = dict(fontsize=fs, color=color or style["color"], ha="center", va="center", rotation=rot,
              rotation_mode="anchor", family=style["family"], fontweight=style["fontweight"], zorder=6)
    if style["background"]:
        kw["bbox"] = {"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none",
                      "alpha": style["background_alpha"]}
    if leader_to is not None:
        t = ax.annotate(s, xy=leader_to, xytext=xy, xycoords="data", textcoords="data", annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": color or style["color"], "shrinkA": 0,
                                    "shrinkB": 0}, **kw)
    else:
        t = ax.text(xy[0], xy[1], s, **kw)
    t.set_gid(gid)
    t._calphad_label = dict(info)
    return t


def _mark_text(R: Raster, placed: np.ndarray, t, renderer, margin: float = 1.0):
    """Mark the (rotated) box of a drawn text (not its leader line) as occupied."""
    from matplotlib.text import Text
    if hasattr(t, "update_positions"):
        t.update_positions(renderer)                # an annotation: place its text first
    if t.get_bbox_patch() is not None:
        margin += 0.15 * t.get_fontsize()
    e = Text.get_window_extent(t, renderer)
    rot = t.get_rotation() % 180
    if abs(rot) < 0.5 or abs(rot - 90) < 0.5:
        R.rect(placed, e.x0 - margin * R.ppp, e.y0 - margin * R.ppp, e.x1 + margin * R.ppp, e.y1 + margin * R.ppp)
        return
    # rotated: the true rectangle
    cx, cy = t.get_transform().transform(t.get_unitless_position())
    w, h = _unrot_size(t, renderer)
    a = math.radians(t.get_rotation())
    hw, hh = w / 2 + margin * R.ppp, h / 2 + margin * R.ppp
    corners = [(cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a))
               for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    placed |= R.poly(corners)


def _unrot_size(t, renderer):
    from matplotlib.text import Text
    rot = t.get_rotation()
    t.set_rotation(0)
    e = Text.get_window_extent(t, renderer)
    t.set_rotation(rot)
    return e.width, e.height


# ---------------------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------------------
def clear_labels(ax) -> int:
    n = 0
    for t in list(ax.texts):
        if _is_ours(t):
            t.remove()
            n += 1
    return n


def label_diagram(ax, regions: Dict, grid: Optional[Dict] = None, key_at: Optional[Callable] = None, *,
                  kind: str = "binary", elements: Optional[Sequence[str]] = None, phase_labels=None,
                  liquids: Sequence[str] = (), invariants: bool = True, T_unit: str = "C",
                  style: Optional[Dict] = None) -> Dict:
    """Draw phase-region labels (and binary invariant temperatures) on a phase diagram axes.

    regions: {"polygons": [{"key", "xy" (plotted coordinates), "tri"?, "ties"?}], "compounds": [...],
              "invariants": [{"T_K", "y", "x0", "x1", "type"}]} (see phase_regions.regions_from_lines).
    grid: equilibrium grid {"xy", "keys", "table"}; key_at(x, y) -> region key at a plotted point.
    Returns {"labels": [...], "skipped": [...]}."""
    st = {**DEFAULT_STYLE, **(style or {})}
    fig = ax.figure
    clear_labels(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    R = Raster(ax, st["res"])
    inside = np.ones((R.H, R.W), bool)
    if kind == "ternary":
        d = R.data
        inside = ((d[:, 0] >= -1e-9) & (d[:, 1] >= -1e-9) & (d[:, 0] + d[:, 1] <= 1 + 1e-9)).reshape(R.H, R.W)
    edge = _edt(inside) * R.res
    inner = inside & (edge > 1.0)                       # keep 1 pt from the frame / triangle edges
    frame = np.ones_like(inside)
    zone = inner | (frame & ~inside & (_edt(frame) * R.res > 1.0))    # leader texts may also sit beside a triangle
    regs, rid, grid_only = build_regions(ax, R, inside, regions, grid, probe=key_at)
    walls = walls_of(rid, inside)
    obst, n_pts = obstacles(ax, R, renderer)
    placed = R.zeros()
    R.lines = R.zeros()                                 # leader lines drawn so far
    meas = _Measure(ax, renderer, st)
    liquids = list(liquids) + [k for r in regs for k in r["key"] if is_liquid_name(k)]
    report: Dict[str, List] = {"labels": [], "skipped": []}
    used_gids: Dict[str, int] = {}
    inv_px = ax.transData.inverted()

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

    def record(t, kind_, key, xy, fs, rot, leader, verified):
        report["labels"].append({"text": t.get_text(), "gid": t.get_gid(), "kind": kind_, "key": list(key),
                                 "x": xy[0], "y": xy[1], "fontsize": fs, "rotation": rot, "leader": leader,
                                 "verified": verified})

    # -------------------------------------------------------------- binary invariant temperatures
    if kind == "binary" and invariants:
        free_base = ~inner | walls | obst
        for inv in sorted(regions.get("invariants") or [], key=lambda i: -i["T_K"]):
            T = inv["T_K"] - 273.15 if str(T_unit).upper().startswith("C") else inv["T_K"]
            s = f"{T:.0f} °C" if str(T_unit).upper().startswith("C") else f"{T:.0f} K"
            if st["invariant_type"] and inv.get("type"):
                s += f" ({inv['type']})"
            fs = st["invariant_fontsize"]
            w, h = meas.size(s, fs)
            (xa, y), (xb, _) = ax.transData.transform([(inv["x0"], inv["y"]), (inv["x1"], inv["y"])])
            if not (R.y0 <= y <= R.y0 + R.H * R.s):
                continue
            xa, xb = max(xa, R.x0), min(xb, R.x0 + R.W * R.s)
            blocked = free_base | placed
            clear = _edt(~blocked) * R.res
            gap = 2.2 * R.ppp
            cands = []
            for side, sgn in (("above", 1), ("below", -1)):
                cy = y + sgn * (gap + h * R.ppp / 2)
                cands.append((xa + 3 * R.ppp + w * R.ppp / 2, cy, 3.0 if side == "above" else 1.5))
                cands.append((xb - 3 * R.ppp - w * R.ppp / 2, cy, 3.0 if side == "above" else 1.5))
                for f in np.linspace(0.1, 0.9, 9):
                    cands.append((xa + f * (xb - xa), cy, 0.0))
            best = None
            for cx, cy, bonus in cands:
                r0, r1, c0, c1 = _box_cells(R, cx, cy, w, h)
                if c0 < 0 or r0 < 0 or c1 >= R.W or r1 >= R.H:
                    continue
                if blocked[r0:r1 + 1, c0:c1 + 1].any():
                    continue
                score = float(clear[r0:r1 + 1, c0:c1 + 1].min()) + bonus
                if best is None or score > best[0]:
                    best = (score, cx, cy)
            to, line = None, None
            if best is None and st["leader"]:      # crowded: nearby, with a leader line to the invariant line
                mid = (0.5 * (xa + xb), y)
                spot = _free_spot(R, (blocked | ~zone,), placed | obst, w, h, mid, 45.0)
                if spot is not None:
                    (cx, cy), line = spot
                    best, to = (0.0, cx, cy), tuple(float(v) for v in inv_px.transform(mid))
            if best is None:
                report["skipped"].append({"label": s, "reason": "no free space next to the invariant line"})
                continue
            xy = tuple(float(v) for v in inv_px.transform((best[1], best[2])))
            t = _draw(ax, s, xy, fs, 0.0, st, gid_for("invariant_label:", s.split(" (")[0].replace(" °", "")),
                      {"kind": "invariant", "T_K": inv["T_K"], "key": list(inv.get("key", ()))},
                      color=st["invariant_color"], leader_to=to)
            _mark_text(R, placed, t, renderer)
            if line is not None:
                placed |= line
                R.lines |= line
            record(t, "invariant", inv.get("key", ()), xy, fs, 0.0, to is not None, None)

    # -------------------------------------------------------------- line compounds / compound points
    for comp in regions.get("compounds") or []:
        key = tuple(comp["key"])
        s = region_label(key, elements, phase_labels, liquids)
        fs = st["fontsize"]
        blocked = ~inner | walls | obst | placed
        clear = _edt(~blocked) * R.res
        best = None
        while best is None and fs >= st["min_fontsize"] - 1e-9:
            w, h = meas.size(s, fs)
            cands = []
            if kind == "binary":
                (xc, ya), (_, yb) = ax.transData.transform([(comp["x"], comp["y0"]), (comp["x"], comp["y1"])])
                for sgn in (1, -1):
                    for f in np.linspace(0.15, 0.85, 8):
                        cands.append((xc + sgn * (1.5 * R.ppp + h * R.ppp / 2), ya + f * (yb - ya), 90.0, w, h,
                                      1.0 - abs(f - 0.5)))
                cands.append((xc, yb + (1.5 * R.ppp + h * R.ppp / 2), 0.0, w, h, 1.0))
            elif "line" in comp:        # a line phase in a ternary: text along the line, beside it
                (xa, ya), (xb, yb) = ax.transData.transform(np.asarray(comp["line"], float))
                ang = math.degrees(math.atan2(yb - ya, xb - xa))
                ang = (ang + 90) % 180 - 90
                nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
                for sgn in (1, -1):
                    for f in np.linspace(0.1, 0.9, 17):
                        off = (1.5 + h / 2) * R.ppp * sgn
                        cands.append((xa + f * (xb - xa) + nx * off, ya + f * (yb - ya) + ny * off, ang, w, h,
                                      1.0 - abs(f - 0.5)))
            else:
                xc, yc = ax.transData.transform((comp["x"], comp["y"]))
                for rr in (3.0, 7.0):
                    for a in range(0, 360, 30):
                        dx = math.cos(math.radians(a)) * (rr + w / 2)
                        dy = math.sin(math.radians(a)) * (rr + h / 2)
                        cands.append((xc + dx * R.ppp, yc + dy * R.ppp, 0.0, w, h, 1.0 if rr == 3.0 else 0.0))
            for cx, cy, rot, ww, hh, bonus in cands:
                if rot not in (0.0, 90.0):
                    cells = _rot_cells(R, cx, cy, ww, hh, rot)
                    if cells is None or blocked[cells].any():
                        continue
                    score = float(clear[cells].min()) + bonus
                    if best is None or score > best[0]:
                        best = (score, cx, cy, rot)
                    continue
                bw, bh = (hh, ww) if rot == 90.0 else (ww, hh)
                r0, r1, c0, c1 = _box_cells(R, cx, cy, bw, bh)
                if c0 < 0 or r0 < 0 or c1 >= R.W or r1 >= R.H or blocked[r0:r1 + 1, c0:c1 + 1].any():
                    continue
                score = float(clear[r0:r1 + 1, c0:c1 + 1].min()) + bonus
                if best is None or score > best[0]:
                    best = (score, cx, cy, rot)
            if best is None:
                fs -= 1.0
        if best is None:
            report["skipped"].append({"label": s, "reason": "no free space beside the compound"})
            continue
        if "line" in comp:
            probe = tuple(np.mean(np.asarray(comp["line"], float), axis=0))
        else:
            probe = (comp["x"], 0.5 * (comp["y0"] + comp["y1"])) if kind == "binary" else (comp["x"], comp["y"])
        ok = check(key, probe, contains=True)
        if ok is False:
            report["skipped"].append({"label": s, "reason": "not confirmed by equilibrium"})
            continue
        xy = tuple(float(v) for v in inv_px.transform((best[1], best[2])))
        t = _draw(ax, s, xy, fs, best[3], st, gid_for("phase_label:", ".".join(key)),
                  {"kind": "compound", "key": list(key)})
        _mark_text(R, placed, t, renderer)
        record(t, "compound", key, xy, fs, best[3], False, ok)

    # -------------------------------------------------------------- regions (most constrained first)
    for r in regs:
        m = r["mask"] & inner
        r["cap"] = float(_edt(m).max()) * R.res if m.any() else 0.0
    todo = sorted(regs, key=lambda r: r["cap"])
    sizes = []
    fsz = st["fontsize"]
    while fsz >= st["min_fontsize"] - 1e-9:
        sizes.append(round(fsz, 2))
        fsz -= 0.5
    for r in todo:
        key = r["key"]
        s = region_label(key, elements, phase_labels, liquids)
        m = r["mask"] & inner
        if r["thin"] or m.sum() * R.res ** 2 < 4.0 or r["cap"] < 0.5:
            pole = r.get("pole_px")
            if pole is None and m.any():
                d = _edt(m)
                row, col = np.unravel_index(int(np.argmax(d)), d.shape)
                pole = R.to_px(col, row)
            if pole is None:
                report["skipped"].append({"label": s, "reason": "region too small"})
                continue
            _leader(ax, R, r, s, pole, st, meas, zone, walls, obst, placed, renderer, report, check, gid_for,
                    inv_px, record)
            continue
        bb = _bbox_of(m)
        avail = m & ~obst & ~placed & ~walls         # keep clear of the drawn boundary lines
        score = _edt(avail) * R.res
        angles = [0.0]
        if st["rotate"]:
            rows, cols = np.nonzero(m)
            if len(rows) > 10:
                cov = np.cov(np.vstack([cols, rows]).astype(float))
                ev, vec = np.linalg.eigh(cov)
                if ev[0] > 0 and math.sqrt(ev[1] / ev[0]) > 1.8:
                    ang = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
                    ang = (ang + 90) % 180 - 90
                    if ang <= -89.5:
                        ang = 90.0
                    if abs(ang) > 8:
                        angles.append(round(ang, 1))
        tried = R.zeros()
        placed_ok = False
        for attempt in range(3):
            blocked = ~avail | tried
            choice = {}
            for ang in angles:
                lo, hi, found = 0, len(sizes) - 1, None
                # largest font size that fits (fitting is monotonic in size): binary search
                while lo <= hi:
                    mid = (lo + hi) // 2
                    w, h = meas.size(s, sizes[mid])
                    ok, box = _fits(blocked, _rect_kernel(w / 2, h / 2, ang, R.res), bb)
                    ok &= avail[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1]
                    if ok.any():
                        found = (sizes[mid], ok)
                        hi = mid - 1
                    else:
                        lo = mid + 1
                if found:
                    choice[ang] = found
            if not choice:
                break
            h_fs = choice.get(0.0, (None,))[0]
            rot_ang = next((a for a in angles if a != 0.0 and a in choice), None)
            r_fs = choice[rot_ang][0] if rot_ang is not None else None
            if h_fs is not None and (r_fs is None or h_fs >= min(st["fontsize"], r_fs) - 1.0):
                ang = 0.0
            else:
                ang = rot_ang
            fs, ok = choice[ang]
            sc = np.where(ok, score[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1], -1.0)
            top = sc >= sc.max() - 1.0              # nearly as clear as the best: take the most central one
            tr_, tc_ = np.nonzero(top)
            k = int(np.argmin((tr_ - tr_.mean()) ** 2 + (tc_ - tc_.mean()) ** 2))
            rr, cc = tr_[k], tc_[k]
            row, col = rr + bb[0], cc + bb[2]
            xy = data_of(col, row)
            v = check(key, xy)
            if v is False:
                w, h = meas.size(s, fs)
                R.disk(tried, *R.to_px(col, row), max(w, h) / 2)
                continue
            t = _draw(ax, s, xy, fs, ang, st, gid_for("phase_label:", ".".join(key)),
                      {"kind": "region", "key": list(key)})
            _mark_text(R, placed, t, renderer)
            record(t, "region", key, xy, fs, ang, False, v)
            placed_ok = True
            break
        if placed_ok:
            continue
        d = _edt(m & ~tried)
        if not d.any():
            report["skipped"].append({"label": s, "reason": "not confirmed by equilibrium"})
            continue
        row, col = np.unravel_index(int(np.argmax(d)), d.shape)
        _leader(ax, R, r, s, R.to_px(col, row), st, meas, zone, walls, obst, placed, renderer, report, check,
                gid_for, inv_px, record)
    for k in sorted(set(grid_only)):
        report["skipped"].append({"label": region_label(k.split("+"), elements, phase_labels, liquids),
                                  "reason": "small area found by the equilibrium grid but not drawn by the mapping"})
    report["n_obstacle_points"] = n_pts
    return report


def _free_spot(R: Raster, blocked_options, hard: np.ndarray, w: float, h: float, target_px, reach_pt: float = 70.0):
    """Nearest centre (display px) for a w x h pt text box near target_px, trying each blocked mask in turn, whose
    straight line to the target crosses no ``hard`` cell.  Returns ((x, y), line cells) or None."""
    pc, pr = (float(q) for q in R.cell(*target_px))
    kern = _rect_kernel(w / 2, h / 2, 0.0, R.res)
    reach = int((reach_pt + max(w, h)) / R.res)
    box = (max(0, int(pr) - reach), min(R.H - 1, int(pr) + reach), max(0, int(pc) - reach),
           min(R.W - 1, int(pc) + reach))
    if box[0] > box[1] or box[2] > box[3]:
        return None
    for blocked in blocked_options:
        ok, _ = _fits(blocked, kern, box)
        rows, cols = np.nonzero(ok)
        if not len(rows):
            continue
        rows, cols = rows + box[0], cols + box[2]
        dist = np.hypot(cols - pc, rows - pr) * R.res
        good = dist >= math.hypot(w / 2, h / 2) + 4.0
        rows, cols, dist = rows[good], cols[good], dist[good]
        if not len(rows):
            continue
        order = np.argsort(dist)[:3000]
        rows, cols = rows[order], cols[order]
        f = np.linspace(0.0, 1.0, int(min(400, reach_pt * 2 + 40)))[None, :]   # about every 0.5 pt
        sr = rows[:, None] + (pr - rows[:, None]) * f
        sc = cols[:, None] + (pc - cols[:, None]) * f
        in_box = (np.abs(sc - cols[:, None]) * R.res <= w / 2 + 0.5) & \
            (np.abs(sr - rows[:, None]) * R.res <= h / 2 + 0.5)
        near_t = np.hypot(sc - pc, sr - pr) * R.res < 1.5
        ri = np.clip(np.round(sr).astype(int), 0, R.H - 1)
        ci = np.clip(np.round(sc).astype(int), 0, R.W - 1)
        on_line = ~in_box & ~near_t
        lines = getattr(R, "lines", None)
        hit = hard[ri, ci] & on_line
        soft = (hit & lines[ri, ci]).sum(axis=1) if lines is not None else 0
        crosses = 10 * hit.sum(axis=1) - 9 * soft        # crossing another leader line is better than a text
        k = int(np.argmin(crosses))                     # first (nearest) with the fewest crossings, ideally none
        line = np.zeros_like(hard)
        on = ~in_box[k] & ~near_t[k]
        for a, b in zip(ri[k][on], ci[k][on]):
            line[max(0, a - 1):a + 2, max(0, b - 1):b + 2] = True
        return tuple(float(q) for q in R.to_px(cols[k], rows[k])), line
    return None


def _leader(ax, R, r, s, pole_px, st, meas, zone, walls, obst, placed, renderer, report, check, gid_for, inv_px,
            record):
    """Label outside its (small) region with a thin line to the region's thickest point.  The text goes to the
    nearest free spot whose line crosses no other label, marker or leader line."""
    key = r["key"]
    target = tuple(float(v) for v in inv_px.transform((float(pole_px[0]), float(pole_px[1]))))
    if not st["leader"]:
        report["skipped"].append({"label": s, "reason": "too small for the text (leader lines off)"})
        return
    v = check(key, target)
    if v is False:
        report["skipped"].append({"label": s, "reason": "not confirmed by equilibrium"})
        return
    fs = max(st["min_fontsize"], st["fontsize"] - 1.5)
    w, h = meas.size(s, fs)
    spot = _free_spot(R, (~zone | walls | obst | placed, ~zone | obst | placed), obst | placed, w, h, pole_px)
    if spot is not None:
        (cx, cy), line = spot
        xy = tuple(float(v2) for v2 in inv_px.transform((cx, cy)))
        t = _draw(ax, s, xy, fs, 0.0, st, gid_for("phase_label:", ".".join(key)),
                  {"kind": "region", "key": list(key), "leader": True}, leader_to=target)
        _mark_text(R, placed, t, renderer)
        placed |= line                                  # later labels keep off the leader line
        R.lines |= line
        record(t, "region", key, xy, fs, 0.0, True, v)
        return
    report["skipped"].append({"label": s, "reason": "no free space for a leader label"})


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


def move_label(t, dx_pt: float = 0.0, dy_pt: float = 0.0, xy=None):
    """Move a label by (dx, dy) points, or to data position xy."""
    ax = t.axes
    if xy is not None:
        t.set_position((float(xy[0]), float(xy[1])))
        return
    px, py = ax.transData.transform(t.get_position())
    f = ax.figure.dpi / 72.0
    t.set_position(tuple(ax.transData.inverted().transform((px + dx_pt * f, py + dy_pt * f))))
