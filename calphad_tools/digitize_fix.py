"""Human-in-the-loop corrections for extracted markers.

* ``check_sheet``: one image with an enlarged crop of every doubtful detection (flagged overlaps, split blobs,
  low scores), each labelled with its big ``#N`` and the current guesses marked, so the user can answer in one go:
  "#7 is 3 points, #13 is 2, #23 is not a point".
* ``refit_blob``: re-solve one blob with exactly k markers (template matching pursuit with the count fixed, then
  sub-pixel refinement).  With the count known the blob's outline pins the centres down well even for heavy
  overlap (synthetic identical circles at 0.15-0.6 diameters: ~0.05 diameter error).
* ``snap_point``: a point the user adds by hand, moved to the best template fit nearby (or kept where given).

The tool ``digitize_fix`` (registered in digitize.py, next to digitize_points) uses these.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------------------------- masks and templates
def fill_mask(fg: np.ndarray) -> np.ndarray:
    """Foreground with holes filled (open markers become discs, like the detector sees them)."""
    return ndi.binary_fill_holes(fg)


def component_at(mask: np.ndarray, cx: float, cy: float, r: float) -> Optional[np.ndarray]:
    """Boolean image of the connected blob at (cx, cy) (or the nearest blob pixel within r)."""
    H, W = mask.shape
    y, x = int(round(cy)), int(round(cx))
    y0, y1, x0, x1 = max(0, y - int(r)), min(H, y + int(r) + 1), max(0, x - int(r)), min(W, x + int(r) + 1)
    sub = mask[y0:y1, x0:x1]
    if not sub.any():
        return None
    if not (0 <= y < H and 0 <= x < W and mask[y, x]):
        yy, xx = np.nonzero(sub)
        j = int(np.argmin((yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2))
        y, x = int(yy[j] + y0), int(xx[j] + x0)
    lab, _ = ndi.label(mask, structure=np.ones((3, 3), bool))
    return lab == lab[y, x]


def template_from(mask: np.ndarray, cx: float, cy: float, size: float) -> Optional[np.ndarray]:
    """Odd square template of the blob at (cx, cy), centred on its bounding-box centre (the plotting anchor)."""
    comp = component_at(mask, cx, cy, 0.6 * size)
    if comp is None:
        return None
    ys, xs = np.nonzero(comp)
    if len(ys) < 4 or (ys.max() - ys.min()) > 2.2 * size or (xs.max() - xs.min()) > 2.2 * size:
        return None                                   # touching something: not a clean single marker
    yc, xc = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    R = int(math.ceil(max(ys.max() - yc, yc - ys.min(), xs.max() - xc, xc - xs.min()))) + 1
    T = np.zeros((2 * R + 1, 2 * R + 1), bool)
    oy, ox = int(round(yc)) - R, int(round(xc)) - R
    for y, x in zip(ys, xs):
        if 0 <= y - oy < T.shape[0] and 0 <= x - ox < T.shape[1]:
            T[y - oy, x - ox] = True
    return T


def learn_template(mask: np.ndarray, clean: Sequence[Tuple[float, float]], size: float) -> Tuple[np.ndarray, str]:
    """Median-area template from clean detections; a disc of the marker size when there is none."""
    cands = [T for T in (template_from(mask, cx, cy, size) for cx, cy in list(clean)[:15]) if T is not None]
    if cands:
        areas = [int(T.sum()) for T in cands]
        return cands[int(np.argsort(areas)[len(areas) // 2])], f"learned from {len(cands)} clean markers"
    r = max(1.5, size / 2.0)
    n = int(math.ceil(r)) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (xx ** 2 + yy ** 2) <= r * r, "disc of the marker size (no clean marker to learn from)"


# --------------------------------------------------------------------------------------------- fitting
def refit_blob(blob: np.ndarray, T: np.ndarray, k: int) -> Tuple[List[Tuple[float, float]], Dict]:
    """Exactly k marker centres (x, y) in image pixels for a blob (bool image).  k = 0 -> []."""
    from .digitize import _pursuit, _subpixel_refine
    if k <= 0:
        return [], {"k": 0}
    ys, xs = np.nonzero(blob)
    R = T.shape[0] // 2 + 2
    y0, x0 = max(0, ys.min() - R), max(0, xs.min() - R)
    y1, x1 = min(blob.shape[0], ys.max() + R + 1), min(blob.shape[1], xs.max() + R + 1)
    B = blob[y0:y1, x0:x1]
    picks, info = _pursuit(B, [T], max_k=k, min_gain=0.0)
    while len(picks) < k and picks:                   # the blob is too small for k whole templates: stack extras
        picks.append(picks[len(picks) % max(1, len(picks))])
    if not picks:                                     # no position fits inside: centre of the blob
        return [(float(xs.mean()), float(ys.mean()))] * k, {"k": k, "fallback": True}
    try:
        cen = _subpixel_refine(B, [T.astype(float)], picks)
    except Exception:  # noqa: BLE001 - the integer fit is still good
        cen = [(float(q[0]), float(q[1])) for q in picks]
    return [(x + x0, y + y0) for y, x in cen], {"k": k, "residual": info.get("residual")}


def snap_point(mask: np.ndarray, T: np.ndarray, px: float, py: float, radius: float) -> Tuple[float, float, bool]:
    """Best template position within ``radius`` of (px, py); (px, py, False) when nothing there looks like one."""
    from .digitize import _corr
    H, W = mask.shape
    R = T.shape[0] // 2
    pad = int(radius) + R + 1
    y0, y1 = max(0, int(py) - pad), min(H, int(py) + pad + 1)
    x0, x1 = max(0, int(px) - pad), min(W, int(px) + pad + 1)
    sub = mask[y0:y1, x0:x1].astype(float)
    if sub.sum() < 0.3 * T.sum():
        return px, py, False
    c = _corr(sub, T)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    c = np.where(np.hypot(yy - py, xx - px) <= radius, c, -1.0)
    j = np.unravel_index(int(np.argmax(c)), c.shape)
    if c[j] < 0.55 * T.sum():
        return px, py, False
    return float(xx[j]), float(yy[j]), True


# --------------------------------------------------------------------------------------------- check sheet
_SEVERITY = ("may be", "low score", "split", "")


def group_blobs(items: Sequence[Dict]) -> List[Dict]:
    """Doubtful detections that sit on the same blob (centres within 1.6 marker sizes, transitively) -> one tile:
    {index (lowest), members, cx, cy, size, why (most serious reason)}."""
    items = list(items)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = math.hypot(items[i]["cx"] - items[j]["cx"], items[i]["cy"] - items[j]["cy"])
            if d < 1.6 * max(items[i]["size"], items[j]["size"]):
                parent[find(i)] = find(j)
    groups: Dict[int, List[Dict]] = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    out = []
    for g in groups.values():
        g.sort(key=lambda it: it["index"])
        why = min((it.get("why", "") for it in g),
                  key=lambda w: next(k for k, s in enumerate(_SEVERITY) if s in w))
        out.append({"index": g[0]["index"], "members": [it["index"] for it in g],
                    "cx": float(np.mean([it["cx"] for it in g])), "cy": float(np.mean([it["cy"] for it in g])),
                    "size": max(it["size"] for it in g), "why": why})
    out.sort(key=lambda t: (next(k for k, s in enumerate(_SEVERITY) if s in t["why"]), t["index"]))
    return out


def check_sheet(rgb: np.ndarray, items: Sequence[Dict], others: Sequence[Dict], out_png: str,
                title: str = "", question: bool = True) -> Optional[str]:
    """Grid of enlarged crops, ONE PER BLOB (doubtful detections on the same blob share a tile; the tile is named
    after its lowest detection number and lists the others).  Every detection inside a crop is marked with its
    number.  ``question``: add the how-many-points instructions (off for the after-correction sheet)."""
    if not items:
        return None
    items = group_blobs(items)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    ncol = min(5, n)
    nrow = int(math.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.2 * nrow + 0.9), squeeze=False)
    H, W = rgb.shape[:2]
    for ax in axs.flat:
        ax.axis("off")
    for ax, it in zip(axs.flat, items):
        half = max(18.0, 2.2 * float(it["size"]))
        x0, x1 = int(max(0, it["cx"] - half)), int(min(W, it["cx"] + half))
        y0, y1 = int(max(0, it["cy"] - half)), int(min(H, it["cy"] + half))
        ax.imshow(rgb[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="lanczos")
        for o in others:
            if x0 <= o["cx"] <= x1 and y0 <= o["cy"] <= y1:
                me = o["index"] in it["members"]
                ax.plot([o["cx"]], [o["cy"]], marker="+", ms=11 if me else 8, mew=1.6,
                        color="red" if me else "#1565c0")
                if not me:
                    ax.annotate(f"#{o['index']}", (o["cx"], o["cy"]), textcoords="offset points", xytext=(5, -9),
                                fontsize=7, color="#1565c0")
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y1 - 0.5, y0 - 0.5)
        ax.set_title(f"#{it['index']}", fontsize=15, fontweight="bold", color="red", loc="left", pad=3)
        found = (f"found {len(it['members'])}: " + ", ".join(f"#{m}" for m in it["members"])
                 if len(it["members"]) > 1 else "found 1")
        ax.text(0.99, 1.01, f"{it.get('why', '')}\n{found}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.3")
        for s in ("top", "bottom", "left", "right"):
            ax.spines[s].set_visible(True)
    ask = ("How many data points is each numbered blob?  Reply e.g. \"#7: 3 points, #13: 2, #23: not a point\"\n"
           if question else "")
    title = f"{title} - {n} blob(s)" if title else f"{n} blob(s)"
    fig.suptitle(title + "\n" + ask +
                 "red + = the detections of this blob, blue + = other detections nearby", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if nrow == 1 else 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def doubtful(rows: Sequence[Dict], low_score: float = 0.5) -> List[Dict]:
    """CSV rows that belong on the check sheet, with the reason."""
    out = []
    for r in rows:
        note = r.get("note") or ""
        try:
            sc = float(r.get("score") or 1)
        except ValueError:
            sc = 1.0
        why = ("may be several markers" if "may be" in note else "split from a blob" if "split" in note
               else f"low score {sc:.2f}" if sc < low_score else "")
        if why:
            out.append({**r, "why": why})
    return out
