"""Phase regions of an isopleth (vertical section), for the shaded fill and the region labels.

In an isopleth the phase regions are not bounded by tie-lines: a ZPF line (one phase at zero fraction) separates
a region with that phase from one without it.  The regions are therefore found from the drawn lines, named from
the lines and checked by equilibrium:

1. **walls** - a raster of the axes (``RES`` pt cells); the traced ZPF lines and the invariant segments are walls.
   A line end within ``SNAP`` pt of another line (or of the frame) is joined to it, so a trace that stops just
   short of the line it meets does not leave a gap;
2. **areas** - the connected areas between the walls.  Each is named by the ZPF lines along it: a line with phase
   set S (at the line; its zero-fraction phase z included) has S on one side and S without z on the other, so the
   set every bounding line allows is the area's.  Where the lines leave two candidates (an area bounded by one
   line, or by lines of one phase set) the nodes of an equilibrium grid inside it decide
   (phase_regions.equilibrium_grid, sampling ``PDENS``; weighted by their distance from the lines); with no node
   inside, the lower Gibbs energy of the candidates at its thickest point (phase_regions.lowest_key); an area no
   line names (inconsistent lines) takes its grid nodes' majority, or one equilibrium at its thickest point.
   Ordered/disordered states are named as everywhere in the toolkit (B2_BCC disordered = BCC_A2);
3. **check** - the grid nodes well inside an area (at least one grid spacing from every line) must agree with its
   name.  ``MIN_NODES`` or more nodes of another phase set count only if that set is lower in Gibbs energy there
   (a full equilibrium's global minimisation misses e.g. γ' next to γ, or the better of two σ descriptions: those
   nodes are reported as corrected).  If they hold most of the area, the area takes their name (the lines around
   it are not the equilibrium ones); otherwise it is two regions no traced line separates: split by the nearest
   grid node, and reported.  Along every ZPF line (every ``CHECK_STEP`` pt) the regions ``SIDE`` pt on either
   side must differ by its zero-fraction phase; a line where they do not on most of its length is checked by two
   point equilibria and, if confirmed, reported as not a phase boundary;
4. **outlines** - the wall cells go to the nearest area; the outline of each region (a contour of its cells) has
   every vertex within 1.5 cells of a drawn line moved onto that line, and the vertices beyond the frame onto the
   frame, then it is simplified.  So the fill meets the black lines exactly (it neither crosses nor falls short of
   them).

Even phase counts (two-phase, four-phase regions) are shaded like the two-phase fields of a binary
(diagram_style.fill_regions).  Needs numpy, scipy and matplotlib (contourpy) only - the equilibrium side comes in
as the grid and probe / resolve functions - so show_code copies it into the standalone scripts.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

RES = 0.25          # pt per raster cell
SNAP = 2.0          # pt: a line end this close to another line or the frame is joined to it
MIN_NODES = 4       # grid nodes of a second phase set in one area: two regions no traced line separates
MIN_AREA = 1.0      # pt^2: smaller areas between lines are not regions (they go to their neighbours)
MAX_PROBES = 80     # equilibria for areas with no grid node inside
SIDE = 2.5          # pt: the line check looks this far on either side of a ZPF line
CHECK_STEP = 15.0   # pt between checked points along a ZPF line
MAX_CONFIRM = 24    # point equilibria to confirm lines that fail the check


def _iso_base(name: str) -> str:
    s = str(name).strip().upper()
    return s.split("#")[0]


class _IsoCells:
    """Raster geometry: cells of about ``res`` pt over the axes box; cell (col, row) centre at integer (u, v)."""

    def __init__(self, ax, res: float = RES):
        ppp = ax.figure.dpi / 72.0
        bb = ax.bbox
        self.wpt, self.hpt = bb.width / ppp, bb.height / ppp
        self.W = max(8, int(math.ceil(self.wpt / res)))
        self.H = max(8, int(math.ceil(self.hpt / res)))
        self.sx, self.sy = self.wpt / self.W, self.hpt / self.H        # pt per cell
        self.xlim = [float(v) for v in ax.get_xlim()]
        self.ylim = [float(v) for v in ax.get_ylim()]

    def uv(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        u = (xy[:, 0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.W - 0.5
        v = (xy[:, 1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.H - 0.5
        return np.column_stack([u, v])

    def xy(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        x = self.xlim[0] + (uv[:, 0] + 0.5) / self.W * (self.xlim[1] - self.xlim[0])
        y = self.ylim[0] + (uv[:, 1] + 0.5) / self.H * (self.ylim[1] - self.ylim[0])
        return np.column_stack([x, y])

    def pt(self, d_uv) -> np.ndarray:
        """Cell offsets -> points."""
        d = np.asarray(d_uv, float).reshape(-1, 2)
        return d * [self.sx, self.sy]


def _densify(uv: np.ndarray, step: float) -> np.ndarray:
    """Points along a polyline (cell coordinates) at most ``step`` cells apart (the vertices included)."""
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) < 2:
        return uv
    d = np.hypot(*np.diff(uv, axis=0).T)
    n = np.maximum(1, np.ceil(d / step).astype(int))
    out = [uv[:1]]
    for a, b, k in zip(uv[:-1], uv[1:], n):
        f = (np.arange(1, k + 1) / k)[:, None]
        out.append(a + (b - a) * f)
    return np.vstack(out)


def _mark(wall, owner, pts_uv, idx):
    """Mark the cells of densified polyline points (8-connected) as wall, owned by line idx."""
    H, W = wall.shape
    c, r = np.round(pts_uv[:, 0]).astype(int), np.round(pts_uv[:, 1]).astype(int)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    wall[r[ok], c[ok]] = True
    owner[r[ok], c[ok]] = idx


def _rdp(P: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an open polyline."""
    n = len(P)
    if n < 3:
        return P
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = math.hypot(*ab)
        seg = P[i + 1:j]
        if L < 1e-12:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return P[keep]


def _ring(P: np.ndarray, eps: float) -> np.ndarray:
    """Simplify a closed ring (first point not repeated): split at the point farthest from the first."""
    if len(P) < 5:
        return P
    k = int(np.argmax(np.hypot(*(P - P[0]).T)))
    a = _rdp(P[:k + 1], eps)
    b = _rdp(np.vstack([P[k:], P[:1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


def _ring_area(P: np.ndarray) -> float:
    return 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


def _seeds(gxy: np.ndarray, nodes: np.ndarray, node_key: np.ndarray, n_max: int = 3) -> List[Dict]:
    """Where a boundary no traced line follows crosses the grid: up to n_max short steps through it - between two
    grid nodes one above the other with different phase sets ({"x", "y": [y0, y1]}, a step in T), else side by
    side ({"y", "x": [x0, x1]}) - one grid spacing beyond each node (mapping_check.retrace_isopleth)."""
    xs, ys = np.unique(np.round(gxy[:, 0], 12)), np.unique(np.round(gxy[:, 1], 12))
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.0
    at = {(int(np.searchsorted(xs, round(float(gxy[q, 0]), 12))), int(np.searchsorted(ys, round(float(gxy[q, 1]),
                                                                                                12)))): int(k)
          for q, k in zip(nodes, node_key)}
    vert = sorted((i, j) for (i, j), k in at.items() if (i, j + 1) in at and at[(i, j + 1)] != k)
    horiz = sorted((j, i) for (i, j), k in at.items() if (i + 1, j) in at and at[(i + 1, j)] != k)
    if vert:
        pick = [vert[int(t)] for t in np.unique(np.linspace(0, len(vert) - 1, min(n_max, len(vert))).round())]
        return [{"x": float(xs[i]), "y": [float(ys[j] - dy), float(ys[j + 1] + dy)]} for i, j in pick]
    pick = [horiz[int(t)] for t in np.unique(np.linspace(0, len(horiz) - 1, min(n_max, len(horiz))).round())] \
        if horiz else []
    return [{"y": float(ys[j]), "x": [float(xs[i] - dx), float(xs[i + 1] + dx)]} for j, i in pick]


def isopleth_regions(ax, lines: Sequence[Dict], invariants: Sequence[Dict], grid: Optional[Dict],
                     probe: Optional[Callable] = None, *, resolve: Optional[Callable] = None,
                     info: Optional[Dict] = None, check: bool = True, res: float = RES,
                     max_probes: int = MAX_PROBES, deadline: Optional[float] = None) -> Dict:
    """The phase regions of an isopleth axes (call after the axis limits are set).

    lines: the ZPF lines as drawn: [{"xy": (n, 2) plotted points, "zero_phase": name, "names": [phase names at
    each point, the zero-fraction phase first] (optional; else "phases" or nothing: such a line names no area)}];
    invariants: [{"T_K", "y" (plotted), "x0", "x1", "phases"}] (horizontal invariant lines);
    grid: phase_regions.equilibrium_grid over the axes ({"xy", "keys", "table"}); probe(x, y) -> phase-set key
    at a plotted point (an equilibrium) or None; resolve(x, y, [key, ...]) -> the key of lowest Gibbs energy there
    (phase_regions.lowest_key): grid nodes of a second phase set inside an area are a missing boundary only if
    that set is really lower in energy there (a full equilibrium's global minimisation can miss the stable set);
    info: ordered phase -> (disordered name, ...) (an ordered phase and its disordered state count as one phase in
    the line check).
    Returns {"regions": {"polygons": [{"key", "xy", "holes"}], "invariants", "compounds": [], "phase_points": {}}
    (what label_place.label_diagram reads), "summary": [one entry per region], "check": {..., "text"},
    "probes": number of point equilibria}."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    t0 = time.time()
    C = _IsoCells(ax, res)
    H, W = C.H, C.W
    snap_pt = SNAP
    # ------------------------------------------------------------------ 1. walls
    polys = []                                                  # (uv points, index, is ZPF line)
    for i, ln in enumerate(lines):
        uv = C.uv(ln["xy"])
        uv = uv[np.isfinite(uv).all(axis=1)]
        if len(uv) >= 2:
            polys.append((uv, i, True))
    n_lines = len(lines)
    for j, inv in enumerate(invariants):
        if float(inv["x1"]) - float(inv["x0"]) != 0:
            polys.append((C.uv([[inv["x0"], inv["y"]], [inv["x1"], inv["y"]]]), n_lines + j, False))
    wall = np.zeros((H, W), bool)
    owner = np.full((H, W), -1, int)
    samples, sample_id = [], []
    for uv, i, _ in polys:
        d = _densify(uv, 0.25)
        _mark(wall, owner, d, i)
        samples.append(d)
        sample_id.append(np.full(len(d), i))
    S = np.vstack(samples) if samples else np.zeros((0, 2))
    S_id = np.concatenate(sample_id) if sample_id else np.zeros(0, int)
    S_pt = S * [C.sx, C.sy]                                     # sample points in pt (isotropic distances)
    tree = cKDTree(S_pt) if len(S) else None
    joined = 0
    for uv, i, _ in polys:                                      # join line ends that stop just short
        for E in (uv[0], uv[-1]):
            if not (-0.5 <= E[0] <= W - 0.5 and -0.5 <= E[1] <= H - 0.5):
                continue
            Ept = E * [C.sx, C.sy]
            best, P = None, None
            if tree is not None:
                for k in tree.query_ball_point(Ept, snap_pt):
                    if S_id[k] == i:
                        continue
                    dist = float(np.hypot(*(S_pt[k] - Ept)))
                    if best is None or dist < best:
                        best, P = dist, S[k]
            frame = [(abs(E[0] + 0.5) * C.sx, (-0.5, E[1])), (abs(W - 0.5 - E[0]) * C.sx, (W - 0.5, E[1])),
                     (abs(E[1] + 0.5) * C.sy, (E[0], -0.5)), (abs(H - 0.5 - E[1]) * C.sy, (E[0], H - 0.5))]
            fd, fP = min(frame, key=lambda f: f[0])
            if fd < snap_pt and (best is None or fd < best):
                best, P = fd, np.asarray(fP, float)
            if best is not None and best > 0.5 * min(C.sx, C.sy):
                _mark(wall, owner, _densify(np.vstack([E, P]), 0.25), i)
                joined += 1
    # ------------------------------------------------------------------ 2. areas, named by the lines
    free = ~wall
    lab, n = ndimage.label(free)
    idx = np.arange(1, n + 1)
    size = np.asarray(ndimage.sum(free, lab, idx), float) * C.sx * C.sy if n else np.zeros(0)
    depth = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]
    thickest = ndimage.maximum_position(depth, lab, idx) if n else []
    table: List[tuple] = [tuple(k) for k in (grid or {}).get("table", [])]
    votes: Dict[int, Dict[int, List[int]]] = {}         # area -> key -> grid nodes inside
    weight: Dict[int, Dict[int, float]] = {}            # area -> key -> summed depth of its nodes
    deep_nodes: Dict[int, Dict[int, List[int]]] = {}    # nodes at least one grid spacing from every line
    if grid is not None and len(grid.get("xy", [])):
        g_uv = C.uv(grid["xy"])
        gk = np.asarray(grid["keys"], int)
        gc, gr = np.round(g_uv[:, 0]).astype(int), np.round(g_uv[:, 1]).astype(int)
        on = (gc >= 0) & (gc < W) & (gr >= 0) & (gr < H) & (gk >= 0)
        g_pt = g_uv * [C.sx, C.sy]
        spacing = float(np.median(cKDTree(g_pt).query(g_pt, k=2)[0][:, 1])) if len(g_pt) > 1 else 5.0
        for q in np.flatnonzero(on):
            comp = int(lab[gr[q], gc[q]])
            if comp <= 0:
                continue
            dq = float(depth[gr[q], gc[q]]) * min(C.sx, C.sy)          # pt to the nearest line
            votes.setdefault(comp, {}).setdefault(int(gk[q]), []).append(int(q))
            w_ = weight.setdefault(comp, {})
            w_[int(gk[q])] = w_.get(int(gk[q]), 0.0) + dq
            if dq >= spacing:
                deep_nodes.setdefault(comp, {}).setdefault(int(gk[q]), []).append(int(q))
    rid = np.full((H, W), -1, int)              # region id per cell (-1: wall / unknown)
    regions: List[Dict] = []
    unbounded: List[Dict] = []
    corrected: List[Dict] = []
    renamed: List[Dict] = []
    stats = {"lines": 0, "grid": 0, "probe": 0}
    n_probes = [0]
    unknown = 0

    def tid(key) -> int:
        key = tuple(key)
        if key not in table:
            table.append(key)
        return table.index(key)

    def ask(fn, *a):
        if deadline is not None and time.time() > deadline:
            return None
        n_probes[0] += 1
        try:
            got = fn(*a)
        except Exception:  # noqa: BLE001 - no answer
            return None
        return tuple(got) if got else None

    line_uv = {i: uv for uv, i, is_line in polys if is_line}
    min_cells = max(4, int(round(2.0 / min(C.sx, C.sy))))     # a line along >= 2 pt of an area bounds it

    def line_pair(i, cells):
        """{S, S - z} of line i next to the given cells (rows, cols): the phases on its two sides."""
        uv = C.uv(lines[i]["xy"])                               # (all points: aligned with "names")
        names = lines[i].get("names")
        pick = cells[np.linspace(0, len(cells) - 1, min(5, len(cells))).astype(int)]
        seen: Dict[tuple, int] = {}
        for r_, c_ in pick:
            k = int(np.nanargmin(np.hypot(uv[:, 0] - c_, uv[:, 1] - r_)))
            nm = list(names[min(k, len(names) - 1)]) if names else list(lines[i].get("phases") or [])
            if not nm:
                continue
            S = tuple(sorted(_iso_base(x) for x in nm))
            z = _iso_base(nm[0]) if names else _iso_base(lines[i]["zero_phase"])
            rest = list(S)
            if z in rest:
                rest.remove(z)
            pair = (S, tuple(rest))
            seen[pair] = seen.get(pair, 0) + 1
        return max(seen, key=seen.get) if seen else None

    for c in range(1, n + 1):
        if size[c - 1] < MIN_AREA:
            continue                                            # a sliver between lines: goes to its neighbours
        comp = lab == c
        rows_c, cols_c = np.nonzero(comp)
        r0, r1, c0, c1 = rows_c.min(), rows_c.max(), cols_c.min(), cols_c.max()
        sl = (slice(max(0, r0 - 3), r1 + 4), slice(max(0, c0 - 3), c1 + 4))
        ring = ndimage.binary_dilation(comp[sl], iterations=2) & wall[sl]
        ids, cnt = np.unique(owner[sl][ring], return_counts=True)
        support: Dict[tuple, int] = {}
        n_adj = 0
        for i, k in zip(ids, cnt):
            if not (0 <= i < n_lines) or k < min_cells or i not in line_uv:
                continue
            rr, cc = np.nonzero(ring & (owner[sl] == i))
            pair = line_pair(int(i), np.column_stack([rr + sl[0].start, cc + sl[1].start]))
            if pair is None:
                continue
            n_adj += 1
            for key in set(pair):
                if key:
                    support[key] = support.get(key, 0) + 1
        cands = [k for k, m in support.items() if n_adj and m == n_adj]
        v = votes.get(c, {})
        w_ = weight.get(c, {})
        key = None
        if len(cands) == 1:                                     # the lines around it name it
            key, src = cands[0], "lines"
        elif len(cands) > 1:                                    # S or S - z: the equilibrium decides
            scored = [(w_.get(table.index(k), 0.0) if k in table else 0.0, k) for k in cands]
            best = max(scored)
            if best[0] > 0:
                key, src = best[1], "lines+grid"
            elif resolve is not None:
                r_, c_ = thickest[c - 1]
                key, src = ask(resolve, *(float(t) for t in C.xy([[c_, r_]])[0]), cands), "lines+energy"
        if key is None and v:                                   # no (consistent) lines: the grid nodes
            key, src = table[max(v, key=lambda k_: (w_.get(k_, 0.0), len(v[k_])))], "grid"
        if key is None and probe is not None and n_probes[0] < max_probes:
            r_, c_ = thickest[c - 1]
            key, src = ask(probe, *(float(t) for t in C.xy([[c_, r_]])[0])), "probe"
        if key is None:
            unknown += 1
            continue
        stats[src.split("+")[0]] = stats.get(src.split("+")[0], 0) + 1
        k0 = tid(key)
        # check against the point equilibria of the grid nodes well inside the area (not right next to a line,
        # where the traced line and the point equilibria can differ by a fraction of a grid spacing): a second
        # phase set there is a region the lines do not bound - if it is lower in Gibbs energy than the area's
        # (pycalphad's global minimisation misses e.g. γ' next to γ, or the better of two σ descriptions)
        dn = deep_nodes.get(c, {})
        n_deep = sum(len(qs) for qs in dn.values())
        real = []
        for k_, qs in sorted(dn.items(), key=lambda kv: -len(kv[1])):
            if k_ == k0 or len(qs) < MIN_NODES:
                continue
            pick = qs[::max(1, len(qs) // 5)][:5]
            wins = 0
            for q in pick:
                got = ask(resolve, *(float(t) for t in grid["xy"][q]), [table[k0], table[k_]]) \
                    if resolve is not None else table[k_]
                wins += got is not None and tuple(got) == tuple(table[k_])
            if 2 * wins > len(pick):
                real.append((k_, qs))
            else:
                corrected.append({"phases": " + ".join(table[k_]), "nodes": len(qs), "region": " + ".join(key)})
        if real and sum(len(qs) for _, qs in real) >= 0.75 * n_deep and len(real) == 1:
            renamed.append({"from": " + ".join(key), "to": " + ".join(table[real[0][0]]), "named_by": src})
            key, k0, real, src = table[real[0][0]], real[0][0], [], "grid (energy)"
        if not real:
            rid[comp] = len(regions)
            regions.append({"key": tuple(key), "src": src})
            continue
        # two (or more) phase sets in one area: no traced line between them - split by the nearest grid node
        used = [(k0, list(v.get(k0, [])))] + real
        used = [(k_, qs) for k_, qs in used if qs]
        nodes = np.array([q for _, qs in used for q in qs], int)
        node_key = np.array([k_ for k_, qs in used for _ in qs], int)
        sub = cKDTree(g_uv[nodes] * [C.sx, C.sy])
        _, j = sub.query(np.column_stack([cols_c, rows_c]) * [C.sx, C.sy])
        kid = np.full((H, W), -1, int)
        kid[rows_c, cols_c] = node_key[j]
        for k_, _ in used:
            pieces, m = ndimage.label(kid == k_)
            for q in range(1, m + 1):
                piece = pieces == q
                if piece.sum() * C.sx * C.sy < MIN_AREA:
                    continue
                rid[piece] = len(regions)
                regions.append({"key": table[k_], "src": "grid-split"})
        xy = C.xy(np.column_stack([cols_c, rows_c]))
        unbounded.append({"phases": [" + ".join(table[k_]) for k_, _ in used],
                          "x": [float(xy[:, 0].min()), float(xy[:, 0].max())],
                          "y": [float(xy[:, 1].min()), float(xy[:, 1].max())],
                          "seeds": _seeds(np.asarray(grid["xy"], float), nodes, node_key)})
    n_probes = n_probes[0]
    # ------------------------------------------------------------------ 4. outlines
    known = rid >= 0
    blank = np.zeros((H, W), bool)                 # areas no equilibrium named (left unfilled, unlabelled)
    for c in range(1, n + 1):
        if size[c - 1] >= MIN_AREA:
            comp = lab == c
            if not known[comp].any():
                blank |= comp
    grow = ~known & ~blank                         # walls and slivers: to the nearest region
    if known.any() and grow.any():
        _, (ri, ci) = ndimage.distance_transform_edt(~known, return_indices=True)
        full = rid[ri, ci]
        full[~grow] = rid[~grow]
    else:
        full = rid.copy()
    ext = np.pad(full, 1, mode="edge")             # one cell beyond the frame (snapped onto it below)
    snap_tree = cKDTree(S_pt) if len(S) else None
    polygons, summary = [], []
    import contourpy
    for r, reg in enumerate(regions):
        m = ext == r
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        if not len(rows):
            continue
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        crop = np.pad(m[r0:r1 + 1, c0:c1 + 1], 1).astype(float)
        gen = contourpy.contour_generator(z=crop, fill_type=contourpy.FillType.OuterOffset)
        pts_list, offs_list = gen.filled(0.5, 1.5)
        for pts, offs in zip(pts_list, offs_list):
            rings = []
            for a, b in zip(offs[:-1], offs[1:]):
                ring = np.asarray(pts[a:b], float)
                if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
                    ring = ring[:-1]
                ring = ring + [c0 - 2, r0 - 2]             # crop / ext padding -> cell coordinates
                ring[:, 0] = np.clip(ring[:, 0], -0.5, W - 0.5)
                ring[:, 1] = np.clip(ring[:, 1], -0.5, H - 0.5)
                inner = (ring[:, 0] > -0.5) & (ring[:, 0] < W - 0.5) & (ring[:, 1] > -0.5) & (ring[:, 1] < H - 0.5)
                if snap_tree is not None and inner.any():
                    d, k = snap_tree.query(ring[inner] * [C.sx, C.sy])
                    near = d <= 1.5 * max(C.sx, C.sy)
                    q = ring[inner]
                    q[near] = S[k[near]]
                    ring[inner] = q
                ring = _ring(ring, 0.2)
                if len(ring) >= 3 and abs(_ring_area(ring)) > 1e-9:
                    rings.append(ring)
            if not rings:
                continue
            outer, holes = rings[0], rings[1:]
            if _ring_area(outer) < 0:
                outer = outer[::-1]
            holes = [h[::-1] if _ring_area(h) > 0 else h for h in holes]
            xy_o = C.xy(outer)
            polygons.append({"key": tuple(reg["key"]), "xy": [list(map(float, p)) for p in xy_o],
                             "holes": [[list(map(float, p)) for p in C.xy(h)] for h in holes]})
        cells = np.argwhere(full == r)
        if len(cells):
            xy = C.xy(cells[:, ::-1])
            summary.append({"phases": " + ".join(reg["key"]), "n_phases": len(reg["key"]),
                            "shaded": len(reg["key"]) % 2 == 0, "named_by": reg["src"],
                            "x": [round(float(xy[:, 0].min()), 5), round(float(xy[:, 0].max()), 5)],
                            "y": [round(float(xy[:, 1].min()), 3), round(float(xy[:, 1].max()), 3)],
                            "area_fraction": round(len(cells) / float(H * W), 4)})
    inv_lines = [{"key": tuple(sorted(_iso_base(p) for p in inv.get("phases", ()))), "T_K": float(inv["T_K"]),
                  "y": float(inv["y"]), "x0": float(min(inv["x0"], inv["x1"])), "x1": float(max(inv["x0"], inv["x1"])),
                  "type": ""} for inv in invariants if float(inv["x1"]) != float(inv["x0"])]
    out = {"regions": {"polygons": polygons, "invariants": inv_lines, "compounds": [], "phase_points": {}},
           "summary": summary, "probes": n_probes}
    # ------------------------------------------------------------------ 3. (cont.) the lines against their sides
    rep = {"unbounded": unbounded, "renamed_by_equilibrium": renamed, "grid_nodes_corrected": corrected,
           "areas_named_by": stats, "lines_checked": 0, "not_boundaries": [], "line_ends_joined": joined,
           "areas_without_equilibrium": unknown}
    if check:
        alias = {}
        for o, inf in (info or {}).items():
            alias[_iso_base(o)] = _iso_base(inf[0])

        def differ(a, b, z):        # the two sides differ by one z (an ordered phase: by its state, or at all)
            exact = abs(sum(_iso_base(k) == z for k in a) - sum(_iso_base(k) == z for k in b)) == 1
            za = alias.get(z, z)
            loose = abs(sum(alias.get(_iso_base(k), _iso_base(k)) == za for k in a)
                        - sum(alias.get(_iso_base(k), _iso_base(k)) == za for k in b)) == 1
            return exact or loose
        confirms = 0
        for uv, i, is_line in polys:
            if not is_line:
                continue
            z = _iso_base(lines[i]["zero_phase"])
            d = _densify(uv, 0.5)
            d = d[(d[:, 0] >= 0) & (d[:, 0] <= W - 1) & (d[:, 1] >= 0) & (d[:, 1] <= H - 1)]
            if len(d) < 3:
                continue
            s_pt = np.concatenate([[0.0], np.cumsum(np.hypot(*((np.diff(d, axis=0)) * [C.sx, C.sy]).T))])
            if s_pt[-1] < 2 * SIDE:
                continue
            at = np.arange(SIDE, s_pt[-1] - SIDE + 1e-9, CHECK_STEP)
            if not len(at):
                at = np.array([s_pt[-1] / 2])
            good = bad = 0
            bad_at = None
            for s in at:
                k = int(np.clip(np.searchsorted(s_pt, s), 1, len(d) - 2))
                P = d[k]
                t = (d[k + 1] - d[k - 1]) * [C.sx, C.sy]
                if np.hypot(*t) < 1e-12:
                    continue
                nrm = np.array([-t[1], t[0]]) / np.hypot(*t)
                r_ = int(math.ceil((SIDE + 1.0) / min(C.sx, C.sy)))
                pc, pr = int(round(P[0])), int(round(P[1]))
                win = owner[max(0, pr - r_):pr + r_ + 1, max(0, pc - r_):pc + r_ + 1]
                if ((win >= 0) & (win != i)).any():
                    continue                                    # another line close by: not a clean test
                sides = []
                for sg in (1, -1):
                    Q = P + sg * SIDE * nrm / [C.sx, C.sy]
                    qc, qr = int(round(Q[0])), int(round(Q[1]))
                    if not (0 <= qc < W and 0 <= qr < H) or full[qr, qc] < 0:
                        break
                    sides.append((regions[full[qr, qc]]["key"], Q))
                if len(sides) < 2:
                    continue
                if differ(sides[0][0], sides[1][0], z):
                    good += 1
                else:
                    bad += 1
                    bad_at = sides
            if good + bad:
                rep["lines_checked"] += 1
            if bad and bad >= good and probe is not None and confirms < MAX_CONFIRM and \
                    (deadline is None or time.time() < deadline):
                confirms += 1
                got = []
                for key_side, Q in bad_at:
                    x, y = C.xy([Q])[0]
                    n_probes += 1
                    try:
                        g_ = probe(float(x), float(y))
                        if g_ and resolve is not None and tuple(g_) != tuple(key_side):
                            g_ = resolve(float(x), float(y), [key_side, g_]) or g_
                    except Exception:  # noqa: BLE001
                        g_ = None
                    got.append(g_)
                if all(g for g in got) and not differ(got[0], got[1], z):
                    xy = np.asarray(lines[i]["xy"], float)
                    rep["not_boundaries"].append({"zero_phase": lines[i]["zero_phase"],
                                                  "x": [float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))],
                                                  "y": [float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1]))],
                                                  "equilibrium_sides": [" + ".join(g) for g in got]})
    rep["seconds"] = round(time.time() - t0, 2)
    out["probes"] = n_probes
    out["check"] = rep
    return out


def iso_check_text(rep: Dict, fmt_x: Callable = lambda v: f"{v:.3g}", fmt_y: Callable = lambda v: f"{v:.0f}") -> str:
    """One line for the tool result: regions no traced line bounds, ZPF lines that are not phase boundaries."""
    parts = []
    rt = rep.get("retraced") or {}
    if rt.get("lines_added"):
        was = "; ".join(" / ".join(u["phases"]) for u in rt.get("regions_before", []))
        parts.append(f"boundaries the mapping missed traced ({was}): {rt['lines_added']} line(s) added")
    elif rt:
        parts.append("tracing the missing boundaries found no new line")
    for u in rep.get("unbounded", []):
        parts.append(f"region {' / '.join(u['phases'])} not bounded by traced lines (x {fmt_x(u['x'][0])}–"
                     f"{fmt_x(u['x'][1])}, T {fmt_y(u['y'][0])}–{fmt_y(u['y'][1])}; split by the equilibrium grid)")
    for b in rep.get("not_boundaries", []):
        parts.append(f"ZPF line of {b['zero_phase']} (x {fmt_x(b['x'][0])}–{fmt_x(b['x'][1])}, "
                     f"T {fmt_y(b['y'][0])}–{fmt_y(b['y'][1])}) is not a phase boundary by equilibrium "
                     f"({b['equilibrium_sides'][0]} | {b['equilibrium_sides'][1]})")
    for c in rep.get("grid_nodes_corrected", []):
        parts.append(f"{c['nodes']} grid node(s) inside {c['region']} gave {c['phases']}, higher in Gibbs energy "
                     f"than {c['region']} there (missed minimum): not a region")
    if rep.get("areas_without_equilibrium"):
        parts.append(f"{rep['areas_without_equilibrium']} area(s) between lines not named (no equilibrium result), "
                     "left blank")
    if not parts:
        return (f"isopleth check: every region bounded by traced lines; {rep.get('lines_checked', 0)} ZPF lines "
                "separate regions that differ by their zero-fraction phase")
    return "isopleth check: " + "; ".join(parts)
