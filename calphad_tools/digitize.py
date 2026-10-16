"""Plot digitiser (WebPlotDigitizer-style) for extracting data from figures in papers.

Workflow for the agent (the model reads tick labels with vision; the tool does all pixel-precise work):

1. ``digitize_setup``  - load an image or a PDF page (optionally cropped to the figure), detect the plot frame
   and tick marks, and write a preview where every detected tick is numbered (x0, x1, ... / y0, y1, ...).
   Also reports the colour clusters and typical marker size of the plotted symbols.
2. The model looks at the preview and maps a few tick indices to the values printed next to them
   (e.g. {"x": {"0": 0, "5": 1.0}, "y": {"1": 800, "4": 1400}}), or gives the axis range at the frame edges.
3. ``digitize_points`` - detect markers (component analysis with line removal, hole filling and splitting of
   touching markers, or template matching from an example marker), exclude the legend, classify shape / fill /
   colour, convert sub-pixel centres to data values, optionally measure error bars; writes a CSV and an overlay
   with numbered detections for visual verification.
   ``digitize_curve`` - trace a line of a given colour and sample it.
4. Both keep a combined *provenance figure* per source image (``<image>_provenance.png``): every extracted value
   mapped back through its calibration and drawn over the original, with calibration check marks, for a person
   to verify the extraction at a glance.

Overlapping markers: a blob bigger than one marker is re-solved by template matching pursuit (``_pursuit``) with
one-marker templates learned from clean isolated markers; what cannot be resolved with confidence is flagged
(``Marker.note``, CSV column ``note``), never merged silently.

All steps are deterministic numpy / scipy / scikit-image code (no OCR, no network).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# image loading
# ---------------------------------------------------------------------------
def load_rgb(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0


def estimate_skew(rgb: np.ndarray, max_deg: float = 4.0) -> float:
    """Rotation (degrees) of the plot axes, measured ONLY from long straight segments (>= 45 % of the image),
    i.e. the frame - sloped data curves or phase boundaries must not drive the deskew. 0 if none found."""
    from skimage.transform import probabilistic_hough_line
    gray = rgb.mean(axis=2)
    dark = gray < dark_threshold(gray)
    H, W = dark.shape
    angs = []
    for base, length in ((0.0, int(0.45 * H)), (90.0, int(0.45 * W))):
        theta = np.deg2rad(np.linspace(base - max_deg, base + max_deg, int(max_deg * 40) + 1))
        segs = probabilistic_hough_line(dark, threshold=10, line_length=length, line_gap=3, theta=theta, rng=0)
        ys_all, xs_all = np.nonzero(dark)
        for (x0, y0), (x1, y1) in segs:
            # refine with a least-squares fit to the dark pixels within 2 px of the Hough segment
            dx_, dy_ = x1 - x0, y1 - y0
            L = math.hypot(dx_, dy_)
            if L == 0:
                continue
            dist = np.abs(dy_ * (xs_all - x0) - dx_ * (ys_all - y0)) / L
            t = ((xs_all - x0) * dx_ + (ys_all - y0) * dy_) / L
            sel = (dist <= 2.0) & (t >= 0) & (t <= L)
            if sel.sum() < 10:
                continue
            if base == 0.0:           # near-vertical: x as a function of y
                slope = np.polyfit(ys_all[sel], xs_all[sel], 1)[0]
                angs.append(-math.degrees(math.atan(slope)))
            else:                     # near-horizontal: y as a function of x
                slope = np.polyfit(xs_all[sel], ys_all[sel], 1)[0]
                angs.append(math.degrees(math.atan(slope)))
    angs = [a for a in angs if abs(a) <= max_deg]
    if len(angs) < 2:
        return 0.0
    return float(np.median(angs))


def prepare_image(source: str, out_dir: Path, page: Optional[int] = None, crop=None, dpi: int = 300,
                  deskew: bool = True) -> str:
    """Render a PDF page / copy an image, crop, and return the working PNG path.

    crop = [left, top, right, bottom] as fractions (all <= 1) or pixels of the (rendered) image.
    """
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    if src.suffix.lower() == ".pdf":
        if not page:
            raise ValueError("give page (1-based) for a PDF")
        from .expdata import render_pdf_page
        base = render_pdf_page(str(src), int(page), str(out_dir / f"{src.stem}_p{page}_{dpi}dpi.png"), dpi=dpi)
    else:
        base = str(src)
    im = Image.open(base).convert("RGB")
    tag = f"{src.stem}" + (f"_p{page}" if page else "")
    if crop:
        w, h = im.size
        c = [float(v) for v in crop]
        if all(0 <= v <= 1 for v in c):
            c = [c[0] * w, c[1] * h, c[2] * w, c[3] * h]
        l, t, r, b = [int(round(v)) for v in c]
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise ValueError(f"crop {crop} outside the image ({w}x{h} px)")
        im = im.crop((l, t, r, b))
        tag += f"_crop{l}-{t}-{r}-{b}"
    if deskew:
        arr = np.asarray(im, dtype=np.float64) / 255.0
        small = arr[::2, ::2] if max(arr.shape[:2]) > 1600 else arr
        ang = estimate_skew(small)
        if abs(ang) > 0.08:
            from PIL import Image as _I
            bgc = tuple(int(v * 255) for v in background_color(arr))
            im = im.rotate(ang, resample=_I.BICUBIC, expand=False, fillcolor=bgc)   # undo the measured tilt
            tag += f"_deskew{ang:+.2f}"
    out = out_dir / f"{tag}.png"
    im.save(out)
    return str(out)


# ---------------------------------------------------------------------------
# frame and tick detection
# ---------------------------------------------------------------------------
@dataclass
class Line:
    pos: float          # centre row (horizontal) or column (vertical), sub-pixel
    start: int
    end: int
    thickness: int

    @property
    def length(self):
        return self.end - self.start + 1


@dataclass
class Frame:
    left: float
    right: float
    top: float
    bottom: float
    thickness: float
    boxed: bool

    def as_dict(self):
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def _runs(row: np.ndarray):
    """(start, end) of True runs in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1
    return s, e


def _long_lines(mask: np.ndarray, min_len: int, axis: int) -> List[Line]:
    """Straight horizontal (axis=0: scan rows) or vertical (axis=1) dark lines at least min_len long."""
    m = mask if axis == 0 else mask.T
    hits = []
    for i in range(m.shape[0]):
        s, e = _runs(m[i])
        if len(s):
            k = int(np.argmax(e - s))
            if e[k] - s[k] + 1 >= min_len:
                hits.append((i, int(s[k]), int(e[k])))
    lines: List[Line] = []
    group: List[tuple] = []
    for h in hits:
        if group and h[0] == group[-1][0] + 1 and abs(h[1] - group[-1][1]) < 6 and abs(h[2] - group[-1][2]) < 6:
            group.append(h)
        else:
            if group:
                lines.append(_mk_line(group))
            group = [h]
    if group:
        lines.append(_mk_line(group))
    return lines


def _mk_line(group) -> Line:
    rows = [g[0] for g in group]
    return Line(pos=float(np.mean(rows)), start=int(np.median([g[1] for g in group])),
                end=int(np.median([g[2] for g in group])), thickness=len(rows))


def dark_threshold(gray: np.ndarray) -> float:
    """Halfway between the darkest ink and the paper colour (robust to grey paper, noise, JPEG)."""
    bg = float(np.median(gray))
    ink = float(np.percentile(gray, 0.5))
    return ink + 0.5 * (bg - ink)


def detect_frame(rgb: np.ndarray, dark_thr: Optional[float] = None) -> Frame:
    gray = rgb.mean(axis=2)
    dark = gray < (dark_thr if dark_thr is not None else dark_threshold(gray))
    # bridge 1-3 px gaps in lines caused by noise / compression before looking for long straight runs
    dark = ndi.binary_closing(dark, structure=np.ones((1, 5), bool)) | ndi.binary_closing(dark, structure=np.ones((5, 1), bool))
    H, W = dark.shape
    hl = [l for l in _long_lines(dark, int(0.3 * W), 0)]
    vl = [l for l in _long_lines(dark, int(0.3 * H), 1)]
    if not hl or not vl:
        raise ValueError("could not find the plot axes (no long dark horizontal/vertical lines). Crop tighter "
                         "around the figure or check the image.")
    bottom = max(hl, key=lambda l: (l.pos, l.length))
    left = min(vl, key=lambda l: (l.pos, -l.length))
    # a matching top/right line makes a closed box
    top = [l for l in hl if l.pos < bottom.pos - 0.25 * H and abs(l.start - bottom.start) < 0.05 * W
           and abs(l.end - bottom.end) < 0.05 * W]
    right = [l for l in vl if l.pos > left.pos + 0.25 * W and abs(l.start - left.start) < 0.05 * H
             and abs(l.end - left.end) < 0.05 * H]
    fr = Frame(left=left.pos, right=(max(right, key=lambda l: l.pos).pos if right else float(bottom.end)),
               top=(min(top, key=lambda l: l.pos).pos if top else float(left.start)), bottom=bottom.pos,
               thickness=float(np.median([bottom.thickness, left.thickness])), boxed=bool(top and right))
    if fr.right - fr.left < 0.2 * W or fr.bottom - fr.top < 0.2 * H:
        raise ValueError("detected plot frame is implausibly small; crop the figure more tightly")
    return fr


def _tick_scan(dark: np.ndarray, axis_pos: float, lo: float, hi: float, thick: float, horizontal_axis: bool,
               max_len: int) -> Dict[str, List[Tuple[float, int]]]:
    """Find short tick lines perpendicular to an axis line, on both sides. Returns side -> [(pos, length)]."""
    out = {}
    half = thick / 2.0
    for side, sign in (("out", 1), ("in", -1)):
        found = []
        a0 = int(round(axis_pos + sign * (half + 1)))
        for t in range(int(math.floor(lo)) - 2, int(math.ceil(hi)) + 3):
            n = 0
            for k in range(max_len):
                r = a0 + sign * k
                if horizontal_axis:
                    if not (0 <= r < dark.shape[0] and 0 <= t < dark.shape[1]) or not dark[r, t]:
                        break
                else:
                    if not (0 <= r < dark.shape[1] and 0 <= t < dark.shape[0]) or not dark[t, r]:
                        break
                n += 1
            found.append(n)
        found = np.array(found)
        ts = np.arange(int(math.floor(lo)) - 2, int(math.ceil(hi)) + 3)
        is_tick = found >= 2
        s, e = _runs(is_tick)
        ticks = []
        for a, b in zip(s, e):
            width = b - a + 1
            if width > max(4, 2 * thick + 2):       # too wide: a line/marker touching the axis, not a tick
                continue
            seg = found[a:b + 1]
            pos = float(np.average(ts[a:b + 1], weights=seg))
            ticks.append((pos, int(seg.max())))
        out[side] = ticks
    return out


def _regularity(pos: Sequence[float]) -> float:
    if len(pos) < 3:
        return 0.0
    d = np.diff(sorted(pos))
    return float(np.median(d) / (np.std(d) + 1e-9))


def _regular_ticks(ticks: List[Tuple[float, int]], lo: float, hi: float, max_len: int) -> List[Tuple[float, int]]:
    """Keep the tick candidates that form the longest evenly spaced progression.

    Drops lines that merely touch the axis (data curves, phase boundaries) and, on a boxed frame with inward ticks,
    replaces the perpendicular frame edges at the ends by the ticks hidden in them (when they fall on the
    progression).  Returns the input unchanged when no progression of >= 3 ticks exists."""
    if len(ticks) < 3:
        return ticks
    pos = np.array([p for p, _ in ticks])
    lens = np.array([l for _, l in ticks])
    inner = [(p, l) for p, l in ticks if min(abs(p - lo), abs(p - hi)) > 2.5]
    best = None
    ip = np.array([p for p, _ in inner]) if inner else pos
    for i in range(len(ip)):
        for j in range(i + 1, len(ip)):
            d = ip[j] - ip[i]
            if d < 3:
                continue
            k = np.round((pos - ip[i]) / d)
            on = np.abs(pos - (ip[i] + k * d)) <= max(1.5, 0.02 * d)
            score = int(on.sum())
            if best is None or score > best[0] or (score == best[0] and d > best[1]):
                best = (score, d, on)
    if best is None or best[0] < 3:
        return ticks
    on = best[2]
    typical = float(np.median(lens[on & (np.minimum(np.abs(pos - lo), np.abs(pos - hi)) > 2.5)])) \
        if np.any(on & (np.minimum(np.abs(pos - lo), np.abs(pos - hi)) > 2.5)) else float(np.median(lens[on]))
    out = []
    for p, l, o in zip(pos, lens, on):
        if not o:
            continue
        if min(abs(p - lo), abs(p - hi)) <= 2.5 and (l >= max_len or l > 2.5 * typical):
            l = int(round(typical)) if l > 2.5 * typical else l      # a tick hidden in the frame edge
            if abs(l - typical) > 0.5 * typical and l >= max_len:
                continue
        out.append((float(p), int(l)))
    return out


def detect_ticks(rgb: np.ndarray, fr: Frame, dark_thr: Optional[float] = None) -> Dict[str, List[Dict]]:
    """Major (and minor) tick positions along the bottom and left axes (falls back to top/right)."""
    gray = rgb.mean(axis=2)
    if dark_thr is None:      # ticks are thin and anti-aliased: be a little more permissive than for the frame
        bg = float(np.median(gray))
        dark_thr = dark_threshold(gray) + 0.3 * (bg - dark_threshold(gray))
    dark = gray < dark_thr
    H, W = dark.shape
    max_len = int(max(6, min(30, 0.035 * min(H, W))))
    res = {}
    for name, axis_pos, lo, hi, horiz in (("x", fr.bottom, fr.left, fr.right, True),
                                           ("y", fr.left, fr.top, fr.bottom, False)):
        cand = _tick_scan(dark, axis_pos, lo, hi, fr.thickness, horiz, max_len)
        cand = {k: _regular_ticks(v, lo, hi, max_len) for k, v in cand.items()}
        # choose the side with more / more regular ticks
        best = max(cand.values(), key=lambda t: (len(t) >= 2, _regularity([p for p, _ in t]) + len(t) * 0.1))
        if len(best) < 2 and fr.boxed:          # try the opposite axis of a boxed frame
            opp = fr.top if horiz else fr.right
            cand2 = _tick_scan(dark, opp, lo, hi, fr.thickness, horiz, max_len)
            best = max(list(cand2.values()) + [best], key=len)
        ticks = sorted(best)
        lens = np.array([l for _, l in ticks]) if ticks else np.array([])
        major_len = lens.max() if lens.size else 0
        out = []
        for p, l in ticks:
            out.append({"px": round(p, 2), "len": int(l), "major": bool(l >= 0.7 * major_len)})
        majors = [t for t in out if t["major"]]
        for i, t in enumerate(majors):
            t["index"] = i
        res[name] = out
    return res


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@dataclass
class AxisCal:
    a: float
    b: float
    scale: str           # linear | log
    residual_px: float
    pairs: List[Tuple[float, float]] = field(default_factory=list)

    def to_value(self, px):
        v = self.a + self.b * np.asarray(px, dtype=float)
        return 10 ** v if self.scale == "log" else v

    def to_px(self, value):
        v = np.log10(np.asarray(value, dtype=float)) if self.scale == "log" else np.asarray(value, dtype=float)
        return (v - self.a) / self.b


def calibrate_axis(spec: Dict, ticks: List[Dict], lo_edge: float, hi_edge: float, name: str) -> AxisCal:
    """spec: {"ticks": {index: value, ...}} | {"range": [value at low-px edge, value at high-px edge]} |
    {"pixels": [[px, value], ...]}, plus "scale": linear|log.  For y, the low-px edge is the TOP of the frame."""
    scale = spec.get("scale", "linear")
    pairs: List[Tuple[float, float]] = []
    majors = {t["index"]: t["px"] for t in ticks if "index" in t}
    if spec.get("ticks"):
        for k, val in spec["ticks"].items():
            k = int(str(k).lstrip(name).lstrip("_"))
            if k not in majors:
                raise ValueError(f"{name} tick index {k} not detected (detected {name}0..{name}{len(majors) - 1})")
            pairs.append((majors[k], float(val)))
    if spec.get("pixels"):
        pairs += [(float(p), float(v)) for p, v in spec["pixels"]]
    if spec.get("range"):
        a, b = spec["range"]
        pairs += [(lo_edge, float(a)), (hi_edge, float(b))]
    if len(pairs) < 2:
        raise ValueError(f"{name} axis needs at least two calibration values (ticks: {{index: value}}, "
                         f"range: [low, high], or pixels: [[px, value], ...])")
    px = np.array([p for p, _ in pairs])
    vals = np.array([v for _, v in pairs])
    if scale == "log":
        if np.any(vals <= 0):
            raise ValueError(f"{name} axis is log scale but a calibration value is <= 0")
        vals = np.log10(vals)
    if np.ptp(px) < 5:
        raise ValueError(f"{name} calibration points are too close together in pixels")
    b, a = np.polyfit(px, vals, 1)
    resid = (vals - (a + b * px)) / b if b else np.zeros_like(px)
    # sanity: majors should be evenly spaced in value for a linear axis -> check predicted tick values
    return AxisCal(a=float(a), b=float(b), scale=scale,
                   residual_px=float(np.max(np.abs(resid))) if len(px) > 2 else 0.0, pairs=pairs)


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------
_NAMED = None


def _named_colors():
    global _NAMED
    if _NAMED is None:
        import matplotlib.colors as mc
        base = {n: mc.to_rgb(c) for n, c in mc.BASE_COLORS.items()}
        css = {n: mc.to_rgb(c) for n, c in mc.CSS4_COLORS.items()}
        tab = {n.replace("tab:", ""): mc.to_rgb(c) for n, c in mc.TABLEAU_COLORS.items()}
        _NAMED = {**css, **tab, **{"k": base["k"], "r": base["r"], "g": base["g"], "b": base["b"]}}
    return _NAMED


def parse_color(c) -> np.ndarray:
    import matplotlib.colors as mc
    if isinstance(c, (list, tuple)) and len(c) == 3:
        v = np.asarray(c, dtype=float)
        return v / 255.0 if v.max() > 1 else v
    try:
        return np.asarray(mc.to_rgb(c))
    except ValueError:
        raise ValueError(f"unknown colour {c!r} (use a name like 'red', '#1f77b4' or [r, g, b])") from None


def to_lab(rgb):
    from skimage.color import rgb2lab
    arr = np.asarray(rgb, dtype=float)
    shp = arr.shape
    return rgb2lab(arr.reshape(-1, 1, 3)).reshape(shp)


def nearest_name(rgb) -> str:
    lab = to_lab(np.asarray(rgb).reshape(1, 3))[0]
    best, bd = "", 1e9
    for n, c in _named_colors().items():
        if len(n) == 1:
            continue
        d = np.linalg.norm(to_lab(np.asarray(c).reshape(1, 3))[0] - lab)
        if d < bd:
            best, bd = n, d
    return best


def background_color(rgb: np.ndarray, fr: Optional[Frame] = None) -> np.ndarray:
    if fr is not None:
        sub = rgb[int(fr.top) + 3:int(fr.bottom) - 2, int(fr.left) + 3:int(fr.right) - 2]
    else:
        sub = rgb
    q = np.round(sub.reshape(-1, 3) * 31).astype(int)
    keys, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
    k = keys[np.argmax(counts)]
    return np.array([k // 1024, (k // 32) % 32, k % 32]) / 31.0


def color_clusters(rgb: np.ndarray, mask: np.ndarray, k_max: int = 8) -> List[Dict]:
    """Distinct colours among foreground pixels (k-means in Lab), largest first."""
    from sklearn.cluster import KMeans
    px = rgb[mask]
    if len(px) < 20:
        return []
    if len(px) > 40000:
        px = px[np.random.default_rng(0).choice(len(px), 40000, replace=False)]
    lab = to_lab(px)
    best = None
    for k in range(2, min(k_max, len(px) // 10) + 1):
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(lab)
        cents = km.cluster_centers_
        dmin = min(np.linalg.norm(cents[i] - cents[j]) for i in range(k) for j in range(i + 1, k))
        if dmin < 18:            # clusters no longer distinct
            break
        best = km
    if best is None:
        best = KMeans(n_clusters=1, n_init=1, random_state=0).fit(lab)
    out = []
    from skimage.color import lab2rgb
    for i, c in enumerate(best.cluster_centers_):
        n = int(np.sum(best.labels_ == i))
        rgbc = np.clip(lab2rgb(c.reshape(1, 1, 3)).reshape(3), 0, 1)
        out.append({"color": "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgbc), "name": nearest_name(rgbc),
                    "pixels": n})
    return sorted(out, key=lambda d: -d["pixels"])


def _inner_region(shape_hw, fr: Frame, pad: Optional[float] = None) -> np.ndarray:
    """Pixels strictly inside the plot frame (the frame lines themselves excluded)."""
    H, W = shape_hw
    t = fr.thickness / 2 if pad is None else pad
    out = np.zeros((H, W), bool)
    out[max(0, int(fr.top + t) + 1):min(H, int(math.ceil(fr.bottom - t))),
        max(0, int(fr.left + t) + 1):min(W, int(math.ceil(fr.right - t)))] = True
    return out


def ink_colours(rgb: np.ndarray, region: np.ndarray, bg, min_frac: float = 0.001) -> List[Dict]:
    """The distinct INK colours (not anti-aliased blends with the background) among pixels of region.

    Only 'core' pixels vote: the pixel of a stroke that differs most from the background within its 3x3
    neighbourhood (the edge pixels of a thin stroke are blends with the background and would form spurious
    lighter clusters).  Their Lab values are grouped greedily (most common bins first, 14 Lab units), so a
    colour used by only a few markers (e.g. three orange-red markers next to hundreds of crimson ones) still
    forms its own ink.  Returns [{rgb, lab, pixels}] (most pixels first)."""
    lab = to_lab(rgb)
    bl = to_lab(np.asarray(bg, float).reshape(1, 3))[0]
    dbg = np.linalg.norm(lab - bl, axis=2)
    core = (dbg > 25) & (dbg >= ndi.maximum_filter(dbg, 3) - 1e-6) & region
    L, P = lab[core], rgb[core]
    if len(L) < 5:
        return []
    keys, inv, cnt = np.unique(np.round(L / 6).astype(int), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    sums = np.zeros((len(keys), 3))
    psum = np.zeros((len(keys), 3))
    np.add.at(sums, inv, L)
    np.add.at(psum, inv, P)
    modes: List[List] = []
    for k in np.argsort(-cnt):
        c = sums[k] / cnt[k]
        for md in modes:
            if np.linalg.norm(md[0] / md[1] - c) < 14:
                md[0] = md[0] + sums[k]
                md[1] += int(cnt[k])
                md[2] = md[2] + psum[k]
                break
        else:
            modes.append([sums[k].copy(), int(cnt[k]), psum[k].copy()])
    n_min = max(8, min_frac * len(L))
    out = [{"rgb": md[2] / md[1], "lab": md[0] / md[1], "pixels": md[1]} for md in modes if md[1] >= n_min]
    bgv = np.asarray(bg, float)
    # a lighter mode on the line between the background and a more saturated ink is that ink's blend
    keep = []
    for i, o in enumerate(out):
        blend = False
        for j, q in enumerate(out):
            if i == j or np.linalg.norm(q["lab"] - bl) <= np.linalg.norm(o["lab"] - bl):
                continue
            d, a = _seg_dist(o["rgb"].reshape(1, 3), o["lab"].reshape(1, 3), q["rgb"], bgv)
            if d[0] < 8 and a[0] < 0.9:
                blend = True
                break
        if not blend:
            keep.append(o)
    return sorted(keep, key=lambda o: -o["pixels"])


def _seg_dist(px_rgb: np.ndarray, px_lab: np.ndarray, ink_rgb, bg_rgb) -> Tuple[np.ndarray, np.ndarray]:
    """Lab distance of pixels (N x 3) to the anti-aliasing line background -> ink (blends are linear in RGB),
    and the blend fraction (0 = background, 1 = ink) of the nearest point."""
    ink = np.asarray(ink_rgb, float)
    bgv = np.asarray(bg_rgb, float)
    seg = ink - bgv
    a = np.clip(((px_rgb - bgv) @ seg) / max(1e-9, float(seg @ seg)), 0, 1)
    return np.linalg.norm(to_lab(bgv + a[:, None] * seg) - px_lab, axis=1), a


def rival_colors(rgb: np.ndarray, fr: Frame, bg=None) -> np.ndarray:
    """Lab centres of the distinct ink colours inside the plot frame (for nearest-colour separation)."""
    bg = background_color(rgb, fr) if bg is None else bg
    inks = ink_colours(rgb, _inner_region(rgb.shape[:2], fr, pad=fr.thickness + 1), bg)
    if inks:
        return np.array([o["lab"] for o in inks])
    fg, _ = _foreground(rgb, bg)
    inner = np.zeros_like(fg)
    pad = int(fr.thickness) + 2
    inner[int(fr.top) + pad:int(fr.bottom) - pad, int(fr.left) + pad:int(fr.right) - pad] = True
    cl = color_clusters(rgb, fg & inner)
    return np.array([to_lab(parse_color(c["color"]).reshape(1, 3))[0] for c in cl]) if cl else np.zeros((0, 3))


# ---------------------------------------------------------------------------
# marker detection
# ---------------------------------------------------------------------------
@dataclass
class Marker:
    cx: float
    cy: float
    area: float
    size_px: float
    shape: str
    filled: bool
    color: str
    score: float
    in_legend: bool = False
    err_up_px: Optional[float] = None
    err_down_px: Optional[float] = None
    group: int = 0
    features: Optional[List[float]] = None
    note: str = ""               # "split from a blob of N" | "may be N+ overlapping markers" (check the overlay)


def _lab2rgb(lab) -> np.ndarray:
    from skimage.color import lab2rgb
    arr = np.asarray(lab, float)
    return np.clip(lab2rgb(arr.reshape(-1, 1, 3)).reshape(arr.shape), 0, 1)


def _foreground(rgb, bg, thr=18.0, color=None, tol=35.0, rivals: Optional[np.ndarray] = None):
    """Foreground mask (+ distance map).  With a target colour: pixels within tol of it whose nearest ink is
    the target's - nearest by the distance to each ink's anti-aliasing line (background -> ink), so a light
    edge pixel of a crimson marker is not taken for an orange-red one and vice versa (similar series colours,
    e.g. blue vs purple or #c10020 vs #f13a13, separate)."""
    lab = to_lab(rgb)
    if color is not None:
        tl = to_lab(np.asarray(color).reshape(1, 3))[0]
        d = np.linalg.norm(lab - tl, axis=2)
        mask = d < tol
        if rivals is not None and len(rivals) and mask.any():
            dists = [np.linalg.norm(rv - tl) for rv in rivals]
            own = int(np.argmin(dists)) if min(dists) < max(tol, 40) else -1   # the target's own cluster
            others = [rv for i, rv in enumerate(rivals) if i != own and dists[i] >= 12]
            if others:
                ys, xs = np.nonzero(mask)
                px, pl = rgb[ys, xs], lab[ys, xs]
                own_rgb = _lab2rgb(rivals[own]) if own >= 0 else np.asarray(color, float)
                d_own, _ = _seg_dist(px, pl, own_rgb, bg)
                ok = np.ones(len(ys), bool)
                for rv in others:
                    ok &= d_own <= _seg_dist(px, pl, _lab2rgb(rv), bg)[0]
                mask[ys[~ok], xs[~ok]] = False
        return mask, d
    bl = to_lab(np.asarray(bg).reshape(1, 3))[0]
    d = np.linalg.norm(lab - bl, axis=2)
    return d > thr, d


def _remove_straight_lines(mask: np.ndarray, min_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Remove horizontal/vertical line pixels longer than min_len (gridlines, tie-lines, error-bar stems...)."""
    h = ndi.binary_opening(mask, structure=np.ones((1, min_len), bool))
    v = ndi.binary_opening(mask, structure=np.ones((min_len, 1), bool))
    lines = h | v
    return mask & ~lines, lines


def _disk(r: float) -> np.ndarray:
    r = max(0.5, float(r))
    n = int(math.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (xx ** 2 + yy ** 2) <= r * r + 0.25


def _legend_boxes(fg: np.ndarray, fr: Frame) -> List[Tuple[int, int, int, int]]:
    """Rectangles drawn inside the plot frame (legend frames): pairs of horizontal segments with matching extents
    joined by vertical segments at both ends."""
    H, W = fg.shape
    inner = np.zeros_like(fg)
    t, b, l, r = int(fr.top + fr.thickness + 2), int(fr.bottom - fr.thickness - 2), int(fr.left + fr.thickness + 2), \
        int(fr.right - fr.thickness - 2)
    inner[t:b, l:r] = fg[t:b, l:r]
    fw = r - l
    hl = [x for x in _long_lines(inner, max(12, int(0.06 * fw)), 0) if x.length < 0.95 * fw]
    boxes = []
    for i, h1 in enumerate(hl):
        for h2 in hl[i + 1:]:
            if h2.pos - h1.pos < 8 or abs(h1.start - h2.start) > 6 or abs(h1.end - h2.end) > 6:
                continue
            x0, x1 = min(h1.start, h2.start), max(h1.end, h2.end)
            y0, y1 = int(h1.pos), int(h2.pos)
            span = slice(y0 + max(3, (y1 - y0) // 6), y1 - max(3, (y1 - y0) // 6))
            if y1 - y0 <= 8:
                continue
            # rounded (fancybox) corners: the vertical edges sit up to ~8 px outside the straight top/bottom runs
            lc = max(inner[span, c].mean() for c in range(max(0, x0 - 9), x0 + 3))
            rc = max(inner[span, c].mean() for c in range(x1 - 2, min(W, x1 + 10)))
            if lc > 0.8 and rc > 0.8:
                boxes.append((max(0, x0 - 9), y0, min(W - 1, x1 + 9), y1))
    # drop boxes contained in others
    boxes = [bx for bx in boxes if not any(o != bx and o[0] <= bx[0] and o[1] <= bx[1] and o[2] >= bx[2]
                                           and o[3] >= bx[3] for o in boxes)]
    return boxes


def _classify(patch: np.ndarray, filled_patch: np.ndarray) -> Tuple[str, bool]:
    """Shape + filled/open from the marker's own mask and its hole-filled version.

    Asymmetry of the area centroid within the bounding box is the most robust cue at small sizes (stroke width
    rounds off corners): triangles are strongly off-centre, circles/squares/diamonds are not.
    """
    area_f = filled_patch.sum()
    if area_f == 0:
        return "?", True
    ink = patch.sum() / area_f
    filled = ink > 0.75
    ys, xs = np.nonzero(filled_patch)
    h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
    extent = area_f / float(h * w)
    dy = (ys.mean() - ys.min() - (h - 1) / 2) / h
    dx = (xs.mean() - xs.min() - (w - 1) / 2) / w
    if ink > 0.9 and extent < 0.45:          # thin strokes without an enclosed area
        mid_r = filled_patch[(h - 1) // 2].mean() if h > 2 else 0
        return ("plus" if mid_r > 0.7 else "cross"), True
    if abs(dy) > 0.045 and abs(dy) >= 1.5 * abs(dx):
        return ("triangle_up" if dy > 0 else "triangle_down"), filled
    if abs(dx) > 0.045 and abs(dx) >= 1.5 * abs(dy):
        return ("triangle_left" if dx > 0 else "triangle_right"), filled
    if extent > 0.86:
        return "square", filled
    if extent < 0.64:
        return "diamond", filled
    return "circle", filled


# ---------------------------------------------------------------------------
# overlapping markers: reference size, shape template, matching pursuit
# ---------------------------------------------------------------------------
OVERLAP_NOTE = "may be {n}+ overlapping markers"
SPLIT_NOTE = "split from a blob of {n}"
HIDDEN_NOTE = "may be partly hidden under another marker (centre estimated)"


def is_flagged(m) -> bool:
    """A detection that needs a human look (several merged markers, a partly hidden marker).  Accepts a Marker
    or a note string; a plain 'split from a blob of N' is a resolved split, not a flag."""
    note = m if isinstance(m, str) else (m.note or "")
    return "may be" in note


def _size_mode(areas: np.ndarray, width: float = 1.3) -> Tuple[float, int]:
    """Most common size (count mode on a log scale, ties -> larger); returns (median area in the window, count)."""
    la = np.log(np.asarray(areas, float))
    w = math.log(width)
    counts = np.array([(np.abs(la - v) < w).sum() for v in la])
    best = counts.max()
    i = max(np.flatnonzero(counts == best), key=lambda j: la[j])
    sel = np.abs(la - la[i]) < w
    return float(np.median(np.asarray(areas, float)[sel])), int(best)


def _clean_single(p) -> bool:
    """Clearly one round-ish marker: convex and not elongated (a merged pair is elongated or dented)."""
    return p.solidity > 0.88 and p.eccentricity < 0.6


def _stamp(shape_hw, T: np.ndarray, pos) -> np.ndarray:
    """Boolean image of template T (odd square, centre = middle pixel) placed at integer pos (y, x)."""
    H, W = shape_hw
    R = T.shape[0] // 2
    out = np.zeros(shape_hw, bool)
    y, x = int(pos[0]), int(pos[1])
    y0, y1, x0, x1 = y - R, y + R + 1, x - R, x + R + 1
    ty0, tx0 = max(0, -y0), max(0, -x0)
    ty1, tx1 = T.shape[0] - max(0, y1 - H), T.shape[1] - max(0, x1 - W)
    if ty1 > ty0 and tx1 > tx0:
        out[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = T[ty0:ty1, tx0:tx1]
    return out


def _corr(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """ndi.correlate(img, k, mode='constant') for an odd kernel, via FFT (fast for marker-sized kernels)."""
    from scipy.signal import fftconvolve
    if k.size <= 49:
        return ndi.correlate(img.astype(float), k.astype(float), mode="constant")
    return np.round(fftconvolve(img.astype(float), k[::-1, ::-1].astype(float), mode="same"), 6)


def _crop_centred(a: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Smallest centred odd square window of a (same shape as T) that holds all of T's pixels (+1 px)."""
    R = T.shape[0] // 2
    ys, xs = np.nonzero(T)
    rr = int(max(np.abs(ys - R).max(), np.abs(xs - R).max())) + 2 if len(ys) else R
    rr = min(rr, R)
    return a[R - rr:R + rr + 1, R - rr:R + rr + 1]


def _iou_centred(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two odd-sized centred binary templates (padded to a common size)."""
    n = max(a.shape[0], b.shape[0])
    A = np.zeros((n, n), bool)
    Bm = np.zeros((n, n), bool)
    pa, pb = (n - a.shape[0]) // 2, (n - b.shape[0]) // 2
    A[pa:pa + a.shape[0], pa:pa + a.shape[1]] = a
    Bm[pb:pb + b.shape[0], pb:pb + b.shape[1]] = b
    return float((A & Bm).sum() / max(1, (A | Bm).sum()))


def _patch_at(mask: np.ndarray, cy: float, cx: float, R: int) -> Optional[np.ndarray]:
    """(2R+1)^2 float patch of mask with the point (cy, cx) moved to the centre (sub-pixel, bilinear)."""
    H, W = mask.shape
    y, x = int(round(cy)), int(round(cx))
    if not (R + 1 <= y < H - R - 1 and R + 1 <= x < W - R - 1):
        return None
    pt = mask[y - R - 1:y + R + 2, x - R - 1:x + R + 2].astype(float)
    pt = ndi.shift(pt, (y - cy, x - cx), order=1, mode="constant")
    return pt[1:-1, 1:-1]


def _pursuit(B: np.ndarray, tpls: Sequence[np.ndarray], max_k: int, min_gain: float = 0.4
             ) -> Tuple[List[Tuple[int, int, int]], Dict]:
    """Matching pursuit of binary one-marker templates inside a binary blob B (bool, padded patch).

    Returns picks (y, x, template index) and fit info.  Positions are restricted to where a template (eroded by
    1 px, for edge tolerance) lies inside the blob: for overlapping identical markers that pins it to the true
    centres, not to the pair's midpoint.  Each step takes the position explaining most of the still unexplained
    blob (ties -> the most extreme position), removes its footprint and repeats while a new position explains
    > min_gain of its template.  Then all positions are refined jointly (integer coordinate descent on the
    symmetric difference between the blob and the union of placed templates)."""
    Bf = B.astype(float)
    nT, oks = [], []
    for T in tpls:
        core = ndi.binary_erosion(T, structure=np.ones((3, 3), bool))
        if core.sum() < 0.3 * T.sum():
            core = T
        inside = _corr(Bf, core)
        oks.append((inside >= 0.9 * core.sum() - 1e-6) & B)
        nT.append(float(T.sum()))
    info: Dict = {"residual": 1.0, "overhang": 0.0, "unique": [], "cost": float(B.sum()),
                  "n_fit": int(sum(int(ok.sum()) for ok in oks))}
    if not any(ok.any() for ok in oks):
        return [], info
    resid = B.copy()
    picks: List[Tuple[int, int, int]] = []
    cy, cx = np.argwhere(B).mean(axis=0)
    while len(picks) < max_k:
        best = None
        for t, (T, ok) in enumerate(zip(tpls, oks)):
            if not ok.any():
                continue
            g = np.where(ok, _corr(resid, T), -1.0)
            gmax = float(g.max())
            if gmax <= 0 or (picks and gmax < min_gain * nT[t]):
                continue
            cand = np.argwhere(g >= gmax - 0.03 * nT[t])
            if not picks:
                d = np.hypot(cand[:, 0] - cy, cand[:, 1] - cx)
            else:
                d = np.min([np.hypot(cand[:, 0] - q[0], cand[:, 1] - q[1]) for q in picks], axis=0)
            j = int(np.argmax(d))
            key = (gmax, d[j])
            if best is None or key > best[0]:
                best = (key, (int(cand[j][0]), int(cand[j][1]), t))
        if best is None:
            break
        picks.append(best[1])
        resid &= ~_stamp(B.shape, tpls[best[1][2]], best[1][:2])

    def union(ps):
        U = np.zeros(B.shape, bool)
        for q in ps:
            U |= _stamp(B.shape, tpls[q[2]], q[:2])
        return U
    best_c = int((union(picks) ^ B).sum())
    for _ in range(6):                     # joint local refinement (each centre +-1 px)
        moved = False
        for i in range(len(picks)):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                q = (picks[i][0] + dy, picks[i][1] + dx, picks[i][2])
                if not (0 <= q[0] < B.shape[0] and 0 <= q[1] < B.shape[1]) or not oks[q[2]][q[0], q[1]]:
                    continue
                trial = picks[:i] + [q] + picks[i + 1:]
                c = int((union(trial) ^ B).sum())
                if c < best_c:
                    picks, best_c, moved = trial, c, True
        if not moved:
            break
    stamps = [_stamp(B.shape, tpls[q[2]], q[:2]) for q in picks]
    U = union(picks)
    info["cost"] = float(best_c)
    # the best further position that was NOT taken: marker-shaped ink left over hints at a hidden marker
    left = B & ~U
    info["next_gain"] = max((float(np.where(ok, _corr(left, T), 0).max()) / float(T.sum())
                             for T, ok in zip(tpls, oks)), default=0.0)
    info["residual"] = float((B & ~ndi.binary_dilation(U)).sum() / max(nT[q[2]] for q in picks)) if picks else 1.0
    info["overhang"] = float((U & ~ndi.binary_dilation(B)).sum() / max(1.0, U.sum()))
    others = [np.any([t for j, t in enumerate(stamps) if j != i] or [np.zeros_like(B)], axis=0)
              for i in range(len(stamps))]
    info["unique"] = [float((s & B & ~o).sum() / nT[picks[i][2]]) for i, (s, o) in enumerate(zip(stamps, others))]
    # a template sitting INSIDE a bigger shape has ink all around it: the ring just outside its footprint is
    # mostly blob that no other placed template explains
    emb = []
    for s, o in zip(stamps, others):
        ring = ndi.binary_dilation(s, iterations=2) & ~s & ~o
        emb.append(float((ring & B).sum() / max(1, ring.sum())))
    info["embedded"] = emb
    return picks, info


def _explain(B: np.ndarray, tpls: Sequence[np.ndarray], max_k: int, penalty: float = 0.12
             ) -> Tuple[List[Tuple[int, int, int]], Dict]:
    """Best explanation of a blob: pursuit with each template alone and with all of them mixed; the lowest
    symmetric difference + penalty * template area per placed marker wins (a big template covering two small
    markers does not win just by covering more pixels)."""
    runs = [[i] for i in range(len(tpls))] + ([list(range(len(tpls)))] if len(tpls) > 1 else [])
    best = None
    for sel in runs:
        sub = [tpls[i] for i in sel]
        picks, pi = _pursuit(B, sub, max_k)
        # drop picks that explain almost nothing on their own or sit inside a bigger shape
        while len(picks) >= 2 and (min(pi["unique"]) < 0.22 or max(pi["embedded"]) > 0.45):
            picks, pi = _pursuit(B, sub, len(picks) - 1)
        if not picks or max(pi["embedded"]) > 0.45:
            continue
        picks = [(y, x, sel[t]) for y, x, t in picks]
        c = pi["cost"] + penalty * sum(float(tpls[q[2]].sum()) for q in picks)
        if best is None or c < best[0] - 1e-9:
            best = (c, picks, pi)
    if best is None:
        n_fit = sum(_pursuit(B, [t], 0)[1].get("n_fit", 0) for t in tpls)
        return [], {"residual": 1.0, "overhang": 0.0, "unique": [], "cost": float(B.sum()), "n_fit": n_fit}
    return best[1], best[2]


def _subpixel_refine(B: np.ndarray, softs: Sequence[np.ndarray], picks: List[Tuple[int, int, int]]
                     ) -> List[Tuple[float, float]]:
    """Sub-pixel centres: least squares between the smoothed blob and the union (max) of shifted soft templates."""
    from scipy.optimize import minimize
    Bs = ndi.gaussian_filter(B.astype(float), 0.7)
    sm = [ndi.gaussian_filter(t, 0.7) for t in softs]
    H, W = B.shape

    def render(par):
        img = np.zeros((H, W))
        for k, q in enumerate(picks):
            Ts = sm[q[2]]
            R = Ts.shape[0] // 2
            y, x = par[2 * k], par[2 * k + 1]
            iy, ix = int(math.floor(y)), int(math.floor(x))
            sh = ndi.shift(Ts, (y - iy, x - ix), order=1, mode="constant")
            y0, x0 = iy - R, ix - R
            sy0, sx0 = max(0, -y0), max(0, -x0)
            sy1, sx1 = Ts.shape[0] - max(0, y0 + Ts.shape[0] - H), Ts.shape[1] - max(0, x0 + Ts.shape[1] - W)
            if sy1 > sy0 and sx1 > sx0:
                sl = (slice(max(0, y0), max(0, y0) + sy1 - sy0), slice(max(0, x0), max(0, x0) + sx1 - sx0))
                img[sl] = np.maximum(img[sl], sh[sy0:sy1, sx0:sx1])
        return img
    x0 = np.array([v for q in picks for v in q[:2]], float)
    f0 = float(((render(x0) - Bs) ** 2).sum())
    res = minimize(lambda par: float(((render(par) - Bs) ** 2).sum()), x0, method="Powell",
                   options={"xtol": 0.05, "ftol": 1e-4, "maxfev": 60 * len(x0)})
    par = res.x if res.fun < f0 and np.max(np.abs(res.x - x0)) <= 1.5 else x0     # no runaway fits
    return [(float(par[2 * k]), float(par[2 * k + 1])) for k in range(len(picks))]



def detect_markers(rgb: np.ndarray, fr: Frame, color=None, tol: float = 35.0, marker_size: Optional[float] = None,
                   exclude: Sequence = (), min_area: int = 6, keep_legend: bool = False,
                   errorbars: bool = False) -> Tuple[List[Marker], Dict]:
    """Component-based marker detection inside the frame.

    Blobs larger than one marker (touching / overlapping markers) are split by template matching pursuit
    (``_pursuit``) with a template learned from clean isolated markers; a blob that cannot be split with
    confidence is kept as ONE detection flagged ``may be N+ overlapping markers`` (Marker.note), never merged
    silently."""
    from skimage.measure import label as sklabel, regionprops
    H, W = rgb.shape[:2]
    bg = background_color(rgb, fr)
    rivals = rival_colors(rgb, fr, bg) if color is not None else None
    fg, dist = _foreground(rgb, bg, color=parse_color(color) if color is not None else None, tol=tol, rivals=rivals)
    info: Dict = {"background": "#%02x%02x%02x" % tuple(int(v * 255) for v in bg)}
    pad = int(math.ceil(fr.thickness / 2 + 1))
    region = np.zeros((H, W), bool)
    region[int(fr.top) + pad:int(math.ceil(fr.bottom)) - pad + 1, int(fr.left) + pad:int(math.ceil(fr.right)) - pad + 1] = True
    raw = fg & region
    for box in exclude:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        raw[max(0, y0):y1, max(0, x0):x1] = False
    frame_w = fr.right - fr.left
    # 1st pass: straight lines out, holes filled (open markers -> discs), thin stuff opened away
    line_len = max(15, int(marker_size * 2.2) if marker_size else int(0.05 * frame_w))
    nolines, lines = _remove_straight_lines(raw, line_len)
    # a line through a marker (error-bar stem, gridline) leaves a gap that splits it: close small gaps, but only
    # with real foreground pixels, so stems outside markers are not restored
    nolines = nolines | (ndi.binary_closing(nolines, structure=_disk(2.0)) & raw)
    filled = ndi.binary_fill_holes(nolines)
    # lines that were removed may cut markers: add back line pixels inside filled marker bodies later
    r_open = max(1.0, (marker_size or 0) * 0.18) if marker_size else 1.0
    body = ndi.binary_opening(filled, structure=_disk(r_open))
    lab = sklabel(body, connectivity=2)
    props = [p for p in regionprops(lab) if p.area >= min_area]
    if not props:
        return [], {**info, "note": "no marker-like blobs found (try color=..., a template, or a smaller min_area)"}
    fg_faint, _ = _foreground(rgb, bg, thr=7.0)          # legend frames are often light grey
    legend = _legend_boxes(fg_faint, fr)
    info["legend_boxes"] = [list(b) for b in legend]

    def in_leg(cx, cy):
        return any(bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] for bx in legend)
    # typical single-marker area: the most common size among clearly single, round-ish components (a merged pair
    # is elongated or dented, so it does not vote); the legend marker of the requested colour when it agrees
    clean = [p for p in props if _clean_single(p)]
    plot_clean = [p for p in clean if not in_leg(p.centroid[1], p.centroid[0])]
    size_src = "given"
    if marker_size:
        a0 = math.pi * (marker_size / 2) ** 2 * 0.85
    elif plot_clean:
        a0, _ = _size_mode([p.area for p in plot_clean])
        size_src = "most common isolated marker"
        leg = [p.area for p in clean if in_leg(p.centroid[1], p.centroid[0])]
        if color is not None and 1 <= len(leg) <= 3:
            a_leg = float(np.median(leg))
            if sum(abs(math.log(p.area / a_leg)) < math.log(1.25) for p in plot_clean) >= 2:
                a0, size_src = a_leg, "legend marker"
    else:
        areas = np.array([p.area for p in props])
        a0 = float(np.median(areas))
        big = areas[areas > 1.6 * a0]
        if len(big) > len(areas) / 2:
            a0 = float(np.median(big))
        size_src = "median blob"
    r0 = math.sqrt(a0 / math.pi)
    info.update({"marker_area_px": round(a0, 1), "marker_diameter_px": round(2 * r0, 1), "marker_size_from": size_src})
    # 2nd pass with a size-aware opening that strips error-bar stems and curve stubs from marker bodies
    r2 = max(1.0, 0.3 * r0)
    body2 = ndi.binary_opening(filled, structure=_disk(r2))
    lab = sklabel(body2, connectivity=2)
    props = [p for p in regionprops(lab) if p.area >= max(min_area, 0.25 * a0)]
    weights = np.clip(dist / (dist.max() + 1e-9), 0, 1) if color is None else np.clip(1 - dist / tol, 0, 1)
    # the same reference on the opened mask (the opening trims a little area)
    clean2 = [p for p in props if _clean_single(p) and 0.6 * a0 <= p.area <= 1.6 * a0
              and not in_leg(p.centroid[1], p.centroid[0])]
    a1 = _size_mode([p.area for p in clean2])[0] if clean2 else a0

    def patches_of(g):
        # the size-aware opening rounds corners: classify on the un-opened, hole-filled pixels around the blob
        e = int(math.ceil(r2)) + 1
        gy0, gx0 = np.maximum(g.min(axis=0) - e, 0)
        gy1, gx1 = np.minimum(g.max(axis=0) + e, [H - 1, W - 1])
        core = np.zeros((gy1 - gy0 + 1, gx1 - gx0 + 1), bool)
        core[g[:, 0] - gy0, g[:, 1] - gx0] = True
        near = ndi.binary_dilation(core, structure=_disk(r2 + 1))
        patch_fill = filled[gy0:gy1 + 1, gx0:gx1 + 1] & near
        if patch_fill.sum() < 3:
            patch_fill = core
        ys_, xs_ = np.nonzero(patch_fill)
        gy0, gy1, gx0, gx1 = gy0 + ys_.min(), gy0 + ys_.max(), gx0 + xs_.min(), gx0 + xs_.max()
        patch_fill = patch_fill[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
        patch_raw = patch_fill & nolines[gy0:gy1 + 1, gx0:gx1 + 1]
        return patch_fill, patch_raw, gy0, gy1, gx0, gx1

    # one-marker templates, one per common look (shape, fill, size class), from clean isolated examples
    R = int(math.ceil(1.8 * math.sqrt(3 * a1 / math.pi))) + 2
    cents = np.array([p.centroid for p in props])
    classes: Dict = {}
    for p in props:
        if not (_clean_single(p) and 0.4 * a1 <= p.area <= 3.0 * a1) or in_leg(p.centroid[1], p.centroid[0]):
            continue
        dd = np.sort(np.hypot(cents[:, 0] - p.centroid[0], cents[:, 1] - p.centroid[1]))
        if len(dd) > 1 and dd[1] < 2.6 * math.sqrt(p.area / math.pi):
            continue                                        # a neighbour close by: not a clean example
        pf, pr = patches_of(p.coords)[:2]
        classes.setdefault(_classify(pr, pf), []).append(p)
    tpls: List[Dict] = []
    for look, ex in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        ex = sorted(ex, key=lambda p: p.area)
        while ex:                                          # split a look into size classes (ratio < 1.25)
            cls = [p for p in ex if p.area < 1.25 * ex[0].area]
            ex = ex[len(cls):]
            if len(cls) < 2:
                continue
            am = float(np.median([p.area for p in cls]))
            cls = sorted(cls, key=lambda p: abs(math.log(p.area / am)))[:15]
            pts = []
            for p in cls:
                y0_, x0_, y1_, x1_ = p.bbox
                loc = np.zeros((y1_ - y0_ + 2 * R + 4, x1_ - x0_ + 2 * R + 4), bool)
                loc[R + 2:R + 2 + y1_ - y0_, R + 2:R + 2 + x1_ - x0_] = p.image
                pt = _patch_at(loc, p.centroid[0] - y0_ + R + 2, p.centroid[1] - x0_ + R + 2, R)
                if pt is not None:
                    pts.append(pt)
            if len(pts) >= 2:
                soft = np.mean(pts, axis=0)
                if (soft > 0.5).sum() >= 5:
                    soft = _crop_centred(soft, soft > 0.5)
                    tpls.append({"soft": soft, "T": soft > 0.5, "look": look, "n": len(pts), "area": am})
    # the most common look always; others need a footprint of their own (not a near-copy of a more common
    # template, e.g. a few squares misclassified as circles)
    kept: List[Dict] = []
    for t in sorted(tpls, key=lambda t: -t["n"]):
        # (a look with only 2 examples must differ clearly from the kept ones: a couple of misclassified
        # markers of a kept look would otherwise relabel that look)
        if any(_iou_centred(t["T"], k["T"]) > (0.9 if t["n"] >= 3 else 0.75) for k in kept):
            continue
        if any(t["look"] == k["look"] and 1.15 < t["area"] / k["area"] < (1.8 if t["n"] >= 3 else 4.0) for k in kept):
            continue                  # the same look, somewhat bigger: heavily overlapped pairs, not a marker size
        kept.append(t)
    tpls = kept[:4]
    if not tpls:                            # no clean examples: an ideal disc of the reference size
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        soft = np.clip(math.sqrt(a1 / math.pi) + 0.5 - np.hypot(yy, xx), 0, 1)
        soft = _crop_centred(soft, soft > 0.5)
        tpls = [{"soft": soft, "T": soft > 0.5, "look": None, "n": 0, "area": a1}]
    for t in tpls:
        tys, txs = np.nonzero(t["T"])
        c_ = t["T"].shape[0] // 2
        t["anchor"] = (0.0, 0.0)            # triangles are anchored at the bounding-box centre
        shp = (t["look"] or ("circle", True))[0]
        if shp in ("triangle_up", "triangle_down"):
            t["anchor"] = ((tys.min() + tys.max()) / 2.0 - c_, 0.0)
        elif shp in ("triangle_left", "triangle_right"):
            t["anchor"] = (0.0, (txs.min() + txs.max()) / 2.0 - c_)
    info["overlap_templates"] = [(f"{'filled' if t['look'][1] else 'open'} {t['look'][0]}" if t["look"] else "disc")
                                 + f" ({t['n']} examples, {t['area']:.0f} px)" for t in tpls]
    all_filled = all(t["look"] is None or t["look"][1] for t in tpls)
    if all_filled:
        holes = ndi.binary_fill_holes(nolines) & ~nolines          # fill only pinholes (<= 4 px) in the ink
        hl, nh = ndi.label(holes)
        tiny = np.isin(hl, 1 + np.flatnonzero(ndi.sum(holes, hl, range(1, nh + 1)) <= 4)) if nh else holes
        ink2 = ndi.binary_opening(nolines | tiny, structure=_disk(r2))
    markers: List[Marker] = []
    rejected = 0
    handled = []
    n_split = n_flag = 0

    def emit(g, p, centre=None, note="", score=None, look=None, area=None):
        patch_fill, patch_raw, gy0, gy1, gx0, gx1 = patches_of(g)
        shape, is_filled = look or _classify(patch_raw, patch_fill)
        fy, fx = np.nonzero(patch_fill)
        if centre is None:
            cy, cx = float(fy.mean() + gy0), float(fx.mean() + gx0)
            if shape.startswith("triangle"):
                # plotting libraries anchor triangles at the bounding-box centre, not the area centroid
                cy = float((gy0 + gy1) / 2.0)
        else:
            cy, cx = centre
        cols = rgb[gy0:gy1 + 1, gx0:gx1 + 1][patch_raw] if patch_raw.any() else rgb[g[:, 0], g[:, 1]]
        if len(cols):            # outline colour for open markers: the most saturated / darkest pixels
            sat = cols.max(axis=1) - cols.min(axis=1) + (1 - cols.mean(axis=1)) * 0.5
            cols = cols[sat >= np.percentile(sat, 50)]
        mc = np.median(cols, axis=0) if len(cols) else np.array([0, 0, 0])
        area = float(len(g)) if area is None else area
        if score is None:
            size_f = float(np.clip(1 - abs(math.log(area / a0)) / math.log(4), 0.2, 1))
            score = round(float(size_f * min(1.0, p.solidity + 0.1)), 3)
        m = Marker(cx=cx, cy=cy, area=area, size_px=float(2 * math.sqrt(area / math.pi)),
                   shape=shape, filled=bool(is_filled), color="#%02x%02x%02x" % tuple(int(v * 255) for v in mc),
                   score=score, note=note)
        h, w = patch_fill.shape
        m.features = [math.log(area), patch_fill.sum() / float(h * w), patch_raw.sum() / max(1.0, patch_fill.sum()),
                      h / float(w), (fy.mean() - (h - 1) / 2) / h]
        m.in_legend = in_leg(cx, cy)
        if errorbars:
            m.err_up_px, m.err_down_px = _errorbar(raw, lines, cx, cy, r0)
        markers.append(m)
        return m

    # ink of OTHER colours (not the target, not background, not an anti-aliased blend of the two): a marker of
    # the target colour drawn under another marker shows only a fragment
    other = None
    if color is not None:
        lab_img = to_lab(rgb)
        tl = to_lab(parse_color(color).reshape(1, 3))[0]
        bl = to_lab(np.asarray(bg).reshape(1, 3))[0]
        seg = bl - tl
        tt = np.clip(((lab_img - tl) @ seg) / max(1e-9, float(seg @ seg)), 0, 1)
        d_seg = np.linalg.norm(lab_img - (tl + tt[..., None] * seg), axis=2)
        other = (np.linalg.norm(lab_img - bl, axis=2) > 18) & ~fg & (d_seg > 12) & region
        other = ndi.binary_fill_holes(ndi.binary_opening(other, structure=np.ones((2, 2), bool)))

    def touches_other(g):
        y0, x0 = np.maximum(g.min(axis=0) - 2, 0)
        y1, x1 = np.minimum(g.max(axis=0) + 3, [H, W])
        loc = np.zeros((y1 - y0, x1 - x0), bool)
        loc[g[:, 0] - y0, g[:, 1] - x0] = True
        return int((ndi.binary_dilation(loc, iterations=2) & other[y0:y1, x0:x1]).sum()) >= 4

    for p in props:
        if p.area > 12 * a0 or (p.eccentricity > 0.97 and p.area > 3 * a0):
            rejected += 1                           # big blob / thick curve, not markers
            continue
        coords = p.coords
        # one marker: about one marker's area and not elongated.  Anything bigger goes through the pursuit,
        # which also recognises single markers of another look / size (series can use different markers)
        single = p.area <= 1.25 * a1 and not (p.eccentricity > 0.75 and p.area > 1.05 * a1)
        if other is not None and p.area < 0.8 * a1 and touches_other(coords):
            continue                                # a partly hidden marker: see the occlusion pass below
        if single or in_leg(p.centroid[1], p.centroid[0]):
            emit(coords, p)
            continue
        handled.append(p.label)
        # several markers?  matching pursuit with the one-marker templates inside this blob
        y0_, x0_, y1_, x1_ = p.bbox
        pd_ = R + 2
        Bp = np.zeros((y1_ - y0_ + 2 * pd_, x1_ - x0_ + 2 * pd_), bool)
        Bp[pd_:pd_ + y1_ - y0_, pd_:pd_ + x1_ - x0_] = p.image
        if all_filled:        # filled markers: gaps enclosed by touching markers are not ink
            Bp[pd_:pd_ + y1_ - y0_, pd_:pd_ + x1_ - x0_] &= ink2[y0_:y1_, x0_:x1_]
        oy, ox = y0_ - pd_, x0_ - pd_
        cap = int(min(8, max(2, p.area / min(t["area"] for t in tpls) + 1.5)))
        picks, pi = _explain(Bp, [t["T"] for t in tpls], cap)
        if len(picks) >= 2:
            cen = _subpixel_refine(Bp, [t["soft"] for t in tpls], picks)
            n_t = max(float(tpls[q[2]]["T"].sum()) for q in picks)
            # unexplained ink left (a hidden further marker?) or templates that fit the blob badly
            unsure = (pi["residual"] >= 0.08 or pi["overhang"] > 0.2 or pi.get("next_gain", 0) > 0.25
                      or pi["cost"] > 0.35 * n_t)
            k = len(picks)
            n_split += k
            for (py, px), q in zip(cen, picks):
                t = tpls[q[2]]
                st = _stamp(Bp.shape, t["T"], q[:2]) & Bp
                g = np.argwhere(st) + [oy, ox]
                if len(g) < 3:
                    g = coords
                note = SPLIT_NOTE.format(n=k)
                sc = round(float(max(0.3, min(0.7, 1 - pi["overhang"] - 0.5 * pi["residual"]))), 3)
                if unsure:
                    note = OVERLAP_NOTE.format(n=k + 1) + "; " + note
                    sc = min(sc, 0.45)
                    n_flag += 1
                emit(g, p, centre=(oy + py + t["anchor"][0], ox + px + t["anchor"][1]), note=note, score=sc,
                     look=t["look"], area=float(t["T"].sum()))
            continue
        # one template explains it, or nothing does: keep ONE detection, but a blob clearly bigger than one
        # marker is flagged (a merged pair must never pass silently)
        ref = tpls[picks[0][2]]["area"] if picks else a1
        big = p.area > 1.3 * ref or (bool(picks) and pi["residual"] > 0.3)
        if picks and not big and tpls[picks[0][2]]["look"]:
            # one known marker plus stubs (a line through it): its look and fitted centre, not the blob's
            t = tpls[picks[0][2]]
            (py, px), = _subpixel_refine(Bp, [t["soft"]], [(picks[0][0], picks[0][1], 0)])
            emit(coords, p, centre=(oy + py + t["anchor"][0], ox + px + t["anchor"][1]), look=t["look"])
            continue
        # (a flagged blob keeps the look of the marker that explains part of it, so shape filters keep it)
        m = emit(coords, p, note=OVERLAP_NOTE.format(n=max(2, int(round(p.area / ref)))) if big else "",
                 look=tpls[picks[0][2]]["look"] if (big and picks) else None)
        if big:
            m.score = min(m.score, 0.45)
            n_flag += 1
    if other is not None:
        # occlusion pass: small target-colour fragments that touch other ink = markers partly hidden under
        # another marker.  Place the one-marker template so it covers the visible fragment and the rest of it
        # lies under the other ink; report it flagged (the centre is an estimate)
        t0 = tpls[0]
        T = t0["T"]
        nT = float(T.sum())
        core = ndi.binary_erosion(T, structure=np.ones((3, 3), bool))
        core = core if core.sum() >= 0.3 * nT else T
        labf, _ = ndi.label(filled & region, structure=np.ones((3, 3), bool))
        taken = [(m.cy, m.cx) for m in markers]
        for sl_i, sl in enumerate(ndi.find_objects(labf), start=1):
            if sl is None:
                continue
            Bc = labf[sl] == sl_i
            area = float(Bc.sum())
            if not (max(min_area, 0.06 * a1) <= area < 0.8 * a1):
                continue
            g = np.argwhere(Bc) + [sl[0].start, sl[1].start]
            cyc, cxc = g.mean(axis=0)
            if any(math.hypot(cyc - y, cxc - x) < 0.8 * r0 for y, x in taken) or not touches_other(g):
                continue
            if in_leg(cxc, cyc):
                continue
            pd_ = R + 2
            y0, x0 = sl[0].start - pd_, sl[1].start - pd_
            ys = slice(max(0, y0), min(H, sl[0].stop + pd_))
            xs = slice(max(0, x0), min(W, sl[1].stop + pd_))
            Bl = np.zeros((ys.stop - ys.start, xs.stop - xs.start), bool)
            Bl[g[:, 0] - ys.start, g[:, 1] - xs.start] = True
            A = Bl | other[ys, xs]
            inside = _corr(A, core)
            cover = _corr(Bl, T)
            ok = (inside >= 0.9 * core.sum()) & (cover >= 0.85 * area)
            if not ok.any():
                continue
            cand = np.argwhere(ok & (cover >= cover[ok].max() - 0.5))
            d = np.hypot(cand[:, 0] - (cyc - ys.start), cand[:, 1] - (cxc - xs.start))
            py, px = cand[int(np.argmin(d))]
            cy, cx = py + ys.start + t0["anchor"][0], px + xs.start + t0["anchor"][1]
            emit(g, None, centre=(float(cy), float(cx)), note=HIDDEN_NOTE, score=0.4, look=t0["look"], area=nT)
            n_flag += 1
            taken.append((cy, cx))
    info["rejected_blobs"] = rejected
    info["split_markers"] = n_split
    info["flagged_overlaps"] = n_flag
    # second pass: learn the marker's look from clean detections and find the ones the component analysis
    # missed (markers on lines of the same colour); blobs already resolved above are left alone
    skip = np.isin(lab, handled) if handled else None
    added = _template_pass(rgb, fr, markers, weights, r0, exclude, legend, filled, nolines, raw, skip=skip)
    info["found_by_template"] = len(added)
    markers.extend(added)
    info["groups"] = assign_groups(markers, use_color=color is None)
    if not keep_legend:
        info["legend_markers_excluded"] = sum(m.in_legend for m in markers)
    markers.sort(key=lambda m: (m.cx, m.cy))
    return markers, info



def clutter(fg: np.ndarray, m: "Marker") -> float:
    """Foreground fraction in a ring just outside a marker (lines/neighbours touching it)."""
    r = m.size_px / 2
    H, W = fg.shape
    R = int(math.ceil(2.2 * r)) + 1
    y0, y1, x0, x1 = int(max(0, m.cy - R)), int(min(H, m.cy + R + 1)), int(max(0, m.cx - R)), int(min(W, m.cx + R + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(yy - m.cy, xx - m.cx)
    ring = (d > 1.3 * r) & (d <= 2.2 * r)
    return float(fg[y0:y1, x0:x1][ring].mean()) if ring.any() else 1.0


def pick_example(rgb: np.ndarray, fr: Frame, markers: List["Marker"], candidates: Sequence[int]) -> Optional[int]:
    """Index of the cleanest candidate marker (no lines or neighbours touching it) to use as a template."""
    if not candidates:
        return None
    fg, _ = _foreground(rgb, background_color(rgb, fr))
    return min(candidates, key=lambda i: (clutter(fg, markers[i]), -markers[i].score))


def _template_pass(rgb, fr, markers, feat, r0, exclude, legend, filled, nolines, raw, min_score=0.62,
                   max_tpl=12, skip: Optional[np.ndarray] = None):
    from skimage.feature import match_template, peak_local_max
    H, W = feat.shape
    R = int(math.ceil(1.3 * r0)) + 1
    good = [m for m in markers if m.score >= 0.8 and not m.in_legend
            and R <= m.cx < W - R and R <= m.cy < H - R
            and all(math.hypot(m.cx - o.cx, m.cy - o.cy) > 3 * r0 for o in markers if o is not m)]
    if len(good) < 3:
        return []
    good = sorted(good, key=lambda m: -m.score)[:max_tpl]
    patches = []
    for m in good:
        y, x = int(round(m.cy)), int(round(m.cx))
        patches.append(feat[y - R:y + R + 1, x - R:x + R + 1])
    tpl = np.mean(patches, axis=0)
    if tpl.std() == 0:
        return []

    def ink(cx, cy):          # fraction of the marker disc covered by foreground pixels
        y0, y1, x0, x1 = int(max(0, cy - r0)), int(min(H, cy + r0 + 1)), int(max(0, cx - r0)), int(min(W, cx + r0 + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        win = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r0 * r0
        return (raw[y0:y1, x0:x1] & win).sum() / max(1, win.sum()), (y0, y1, x0, x1, win)
    ref_ink = float(np.median([ink(m.cx, m.cy)[0] for m in good]))
    pad = int(math.ceil(fr.thickness / 2 + 1))
    t, b, l, r = int(fr.top) + pad, int(math.ceil(fr.bottom)) - pad, int(fr.left) + pad, int(math.ceil(fr.right)) - pad
    sub = feat[t:b + 1, l:r + 1].copy()
    for box in list(exclude) + [tuple(bx) for bx in legend]:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        sub[max(0, y0 - t):max(0, y1 - t), max(0, x0 - l):max(0, x1 - l)] = 0
    corr = match_template(sub, tpl, pad_input=True)
    peaks = peak_local_max(corr, min_distance=max(2, int(0.8 * r0)), threshold_abs=min_score, exclude_border=False)
    out = []
    for py, px in peaks:
        cy, cx = py + t, px + l
        if skip is not None and skip[cy, cx]:
            continue
        if any(math.hypot(cx - m.cx, cy - m.cy) < 0.9 * r0 for m in markers + out):
            continue
        # must contain about as much ink as a real marker (a bare line crossing has much less)
        frac, (y0, y1, x0, x1, win) = ink(cx, cy)
        if frac < 0.6 * ref_ink:
            continue
        mc = np.median(rgb[y0:y1, x0:x1][nolines[y0:y1, x0:x1] & win], axis=0) if (nolines[y0:y1, x0:x1] & win).any() \
            else rgb[int(cy), int(cx)]
        out.append(Marker(cx=float(cx), cy=float(cy), area=math.pi * r0 * r0, size_px=2 * r0, shape=good[0].shape,
                          filled=good[0].filled, color="#%02x%02x%02x" % tuple(int(v * 255) for v in mc),
                          score=round(float(corr[py, px]), 3),
                          in_legend=any(bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] for bx in legend)))
    return out


def assign_groups(markers: List[Marker], use_color: bool = True) -> List[Dict]:
    """Group markers by appearance: (shape, filled) label, split by colour when colours differ -> Marker.group.

    Groups are numbered largest first; legend markers take part, so each legend entry shows its group on the
    overlay (use digitize_points group=N to extract one series).
    """
    keys = []
    for m in markers:
        k = (m.shape, m.filled)
        if use_color:
            lab = to_lab(parse_color(m.color).reshape(1, 3))[0]
            # coarse colour bin: black/grey by lightness, colours by hue
            chroma = math.hypot(lab[1], lab[2])
            k += (("grey", int(lab[0] // 34)) if chroma < 18 else ("hue", int(((math.degrees(math.atan2(lab[2], lab[1]))
                                                                               + 360) % 360) // 30)),)
        keys.append(k)
    counts: Dict = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    order = sorted(counts, key=lambda k: -counts[k])
    remap = {k: i for i, k in enumerate(order)}
    for m, k in zip(markers, keys):
        m.group = remap[k]
    out = []
    for k in order:
        ms = [m for m, kk in zip(markers, keys) if kk == k]
        out.append({"group": remap[k], "n": len(ms), "n_legend": sum(m.in_legend for m in ms),
                    "shape": f"{'filled' if k[1] else 'open'} {k[0]}", "color": ms[0].color,
                    "diameter_px": round(float(np.median([m.size_px for m in ms])), 1)})
    return out


def _errorbar(raw, lines, cx, cy, r0):
    """Vertical error-bar extent above/below a marker (pixels beyond the marker centre), or None."""
    H = raw.shape[0]
    x = int(round(cx))
    col = raw[:, max(0, x - 1):x + 2].any(axis=1)
    up = down = None
    for sign in (-1, 1):
        k = int(r0) + 1
        y = int(round(cy)) + sign * k
        if not (0 <= y < H) or not col[y]:
            continue
        while 0 <= y + sign < H and col[y + sign]:
            y += sign
        ext = abs(y - cy)
        if ext > r0 * 1.3:
            if sign < 0:
                up = ext
            else:
                down = ext
    return up, down


def template_markers(rgb: np.ndarray, fr: Frame, center_px: Tuple[float, float], radius: float,
                     min_score: float = 0.65, exclude: Sequence = (), color=None, tol: float = 35.0,
                     shape: str = "") -> List[Marker]:
    """Normalised cross-correlation with an example marker (any shape, open or filled, on lines)."""
    from skimage.feature import match_template, peak_local_max
    bg = background_color(rgb, fr)
    fg, dist = _foreground(rgb, bg, color=parse_color(color) if color is not None else None, tol=tol,
                           rivals=rival_colors(rgb, fr, bg) if color is not None else None)
    feat = (np.clip(dist / 60.0, 0, 1) if color is None else np.clip(1 - dist / tol, 0, 1) * fg).astype(float)
    cx, cy = center_px
    r = int(math.ceil(radius * 1.25)) + 1
    tpl = feat[int(cy) - r:int(cy) + r + 1, int(cx) - r:int(cx) + r + 1]
    if tpl.shape != (2 * r + 1, 2 * r + 1) or tpl.std() == 0:
        raise ValueError("template position too close to the image edge or empty")
    pad = int(math.ceil(fr.thickness / 2 + 1))
    t, b, l, rr = int(fr.top) + pad, int(math.ceil(fr.bottom)) - pad, int(fr.left) + pad, int(math.ceil(fr.right)) - pad
    sub = feat[t:b + 1, l:rr + 1].copy()
    for box in exclude:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        sub[max(0, y0 - t):max(0, y1 - t), max(0, x0 - l):max(0, x1 - l)] = 0
    # anchor = bounding-box centre of the template's own blob (removes any offset in the example's centre;
    # also the right anchor for triangles)
    # anchor: mass-weighted centroid of the marker blob; for triangles the bounding-box centre along the
    # direction they point (that is how plotting libraries place them)
    blob = ndi.binary_fill_holes(tpl > 0.5 * tpl.max())
    lbl, nlab = ndi.label(blob)
    oy = ox = 0.0
    if nlab:
        k = lbl[r, r] if lbl[r, r] else 1 + int(np.argmax(ndi.sum(blob, lbl, range(1, nlab + 1))))
        comp = lbl == k
        ys, xs = np.nonzero(comp)
        w = tpl[comp]
        oy = float(np.average(ys, weights=w)) - r
        ox = float(np.average(xs, weights=w)) - r
        if shape in ("triangle_up", "triangle_down"):
            oy = (ys.min() + ys.max()) / 2.0 - r
        elif shape in ("triangle_left", "triangle_right"):
            ox = (xs.min() + xs.max()) / 2.0 - r
    corr = match_template(sub, tpl, pad_input=True)
    peaks = peak_local_max(corr, min_distance=max(2, int(0.7 * radius)), threshold_abs=min_score, exclude_border=False)
    out = []
    for py, px in peaks:
        # quadratic sub-pixel refinement of the correlation peak
        dy = dx = 0.0
        if 0 < py < corr.shape[0] - 1:
            c0, cm, cp = corr[py, px], corr[py - 1, px], corr[py + 1, px]
            den = cm - 2 * c0 + cp
            dy = 0.5 * (cm - cp) / den if den < 0 else 0.0
        if 0 < px < corr.shape[1] - 1:
            c0, cm, cp = corr[py, px], corr[py, px - 1], corr[py, px + 1]
            den = cm - 2 * c0 + cp
            dx = 0.5 * (cm - cp) / den if den < 0 else 0.0
        mc = rgb[int(py) + t, int(px) + l]
        out.append(Marker(cx=px + l + dx + ox, cy=py + t + dy + oy, area=math.pi * radius ** 2, size_px=2 * radius,
                          shape="template", filled=True, color="#%02x%02x%02x" % tuple(int(v * 255) for v in mc),
                          score=round(float(corr[py, px]), 3)))
    cmap = np.full(fg.shape, -1.0)
    cmap[t:t + corr.shape[0], l:l + corr.shape[1]] = corr
    out = _template_overlaps(rgb, out, fg, fr, (cx, cy), radius, (oy, ox), exclude, corr=cmap)
    out.sort(key=lambda m: (m.cx, m.cy))
    return out


def _template_overlaps(rgb, found: List[Marker], fg: np.ndarray, fr: Frame, example: Tuple[float, float],
                       radius: float, off: Tuple[float, float], exclude: Sequence = (),
                       corr: Optional[np.ndarray] = None, min_corr: float = 0.6) -> List[Marker]:
    """Blob accounting after template matching: correlation peaks of overlapping identical markers merge into
    one peak between them.  Every blob (same foreground mask, lines removed, holes filled) that holds more ink
    than its detections explain is re-solved by matching pursuit with the example's footprint; what cannot be
    resolved is flagged ``may be N+ overlapping markers``."""
    H, W = fg.shape
    pad = int(math.ceil(fr.thickness / 2 + 1))
    region = np.zeros((H, W), bool)
    region[int(fr.top) + pad:int(math.ceil(fr.bottom)) - pad + 1,
           int(fr.left) + pad:int(math.ceil(fr.right)) - pad + 1] = True
    raw = fg & region
    for box in exclude:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        raw[max(0, y0):y1, max(0, x0):x1] = False
    nolines, _ = _remove_straight_lines(raw, max(15, int(0.05 * (fr.right - fr.left)), int(5 * radius)))
    nolines = nolines | (ndi.binary_closing(nolines, structure=_disk(2.0)) & raw)
    r2 = max(1.0, 0.3 * radius)
    body = ndi.binary_opening(ndi.binary_fill_holes(nolines), structure=_disk(r2))
    lab, _ = ndi.label(body, structure=np.ones((3, 3), bool))
    # the example's own footprint in the same mask (same threshold, same opening) = the one-marker template
    R = int(math.ceil(1.8 * radius)) + 2
    ex, ey = int(round(example[0])), int(round(example[1]))
    if not (R <= ey < H - R and R <= ex < W - R):
        return found
    k = lab[ey, ex]
    if k == 0:
        return found
    T = lab[ey - R:ey + R + 1, ex - R:ex + R + 1] == k
    if T.sum() < 5:
        return found
    ys, xs = np.nonzero(T)
    cy_t, cx_t = ys.mean() - R, xs.mean() - R            # footprint centroid relative to the example centre
    # recentre the footprint on its centroid (the pursuit places template centroids)
    T = _stamp((2 * R + 1, 2 * R + 1), T, (R - int(round(cy_t)), R - int(round(cx_t))))
    T = _crop_centred(T, T)
    R = T.shape[0] // 2
    ys, xs = np.nonzero(T)
    cen_off = (ys.mean() - R, xs.mean() - R)
    Tsoft = T.astype(float)
    nT = float(T.sum())
    perim = float((T & ~ndi.binary_erosion(T)).sum())      # footprint edge: ~ the pixel noise of a real match
    from skimage.measure import regionprops
    ecc_t = float(regionprops(T.astype(np.uint8))[0].eccentricity)
    # where the detections sit: peak position = the example centre convention; offset to the footprint centroid
    slices = ndi.find_objects(lab)
    by: Dict[int, List[int]] = {}
    free = []
    for i, m in enumerate(found):
        y, x = int(round(m.cy)), int(round(m.cx))
        win = lab[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        ids = win[win > 0]
        if not len(ids):
            free.append(m)
            continue
        by.setdefault(int(np.bincount(ids).argmax()), []).append(i)
    if corr is not None:
        # blobs bigger than one marker without any peak: a neighbour can spoil the correlation of a marker that
        # touches it; examine those with a moderate peak inside
        for L, sl in enumerate(slices, start=1):
            if sl is None or L in by or L == k:
                continue
            B = lab[sl] == L
            if B.sum() > 1.25 * nT and B.sum() < 12 * nT and float(np.where(B, corr[sl], -1).max()) >= 0.45:
                by[L] = []
    # the example's reported centre relative to its footprint centroid (reported = centroid + this)
    rep_off = (example[1] - (ey + cy_t), example[0] - (ex + cx_t))
    out = list(free)
    for L, idx in by.items():
        sl = slices[L - 1]
        B = lab[sl] == L
        area = float(B.sum())
        if len(idx) == 1 and area <= 2.2 * nT:
            # a clean single blob whose footprint does not match the example is a different marker (a filled
            # circle vs a filled square correlate well, their footprints do not)
            m = found[idx[0]]
            py0 = int(round(m.cy - rep_off[0] - cen_off[0])) - sl[0].start
            px0 = int(round(m.cx - rep_off[1] - cen_off[1])) - sl[1].start
            fits = [(float((B ^ (st := _stamp(B.shape, T, (py0 + a, px0 + b)))).sum()), float((st & ~B).sum()))
                    for a in range(-2, 3) for b in range(-2, 3)]
            mis, over = min(fits)
            rp = regionprops(B.astype(np.uint8))[0]
            if mis > max(0.6 * perim, 0.15 * nT) and rp.solidity > 0.9 and rp.eccentricity < 0.3:
                continue
            if area >= 0.85 * nT and over > 0.15 * nT:
                continue            # the footprint sticks out of the blob: a different marker (circles for squares)
            if mis > max(0.8 * perim, 0.25 * nT) or (rp.eccentricity - ecc_t > 0.3 and area > 1.04 * nT):
                # the blob is not this marker's shape: something else overlaps it (a hidden marker?)
                m.note = OVERLAP_NOTE.format(n=2)
                m.score = min(m.score, 0.45)
        if area <= 1.25 * nT and len(idx) <= 1:
            out.extend(found[i] for i in idx)
            continue
        pd_ = R + 2
        Bp = np.zeros((B.shape[0] + 2 * pd_, B.shape[1] + 2 * pd_), bool)
        Bp[pd_:-pd_, pd_:-pd_] = B
        oy, ox = sl[0].start - pd_, sl[1].start - pd_
        cap = int(min(8, max(2, area / nT + 1.5)))
        picks, pi = _explain(Bp, [T], cap)
        if corr is not None and (len(picks) >= 2 or not idx):
            # every placed marker must also look like the example there (a triangle outline fitted inside a
            # square outline fills the square's footprint but does not correlate)
            def cval(q):
                y = int(round(oy + q[0] + cen_off[0] + rep_off[0] - off[0]))
                x = int(round(ox + q[1] + cen_off[1] + rep_off[1] - off[1]))
                win = corr[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                return float(win.max()) if win.size else -1.0
            good = [q for q in picks if cval(q) >= (min_corr if idx else 0.45)
                    and float((_stamp(Bp.shape, T, q[:2]) & ~Bp).sum()) <= 0.15 * nT]
            if len(good) < len(picks):
                picks = good
                if len(picks) >= 2:
                    pi = dict(pi, residual=max(pi["residual"], 0.31))      # something in the blob is unexplained
        best_score = max([found[i].score for i in idx] or [min_corr])
        if not idx:
            # no correlation peak here before: keep only picks that correlate AND sit on a real marker
            if len(picks) == 1:
                q = picks[0]
                cy = oy + q[0] + cen_off[0] + rep_off[0]
                cx = ox + q[1] + cen_off[1] + rep_off[1]
                c = rgb[int(min(H - 1, max(0, round(cy)))), int(min(W - 1, max(0, round(cx))))]
                out.append(Marker(cx=float(cx), cy=float(cy), area=math.pi * radius ** 2, size_px=2 * radius,
                                  shape="template", filled=True, color="#%02x%02x%02x" % tuple(int(v * 255) for v in c),
                                  score=0.45, note=OVERLAP_NOTE.format(n=2) + "; touches another marker"))
                continue
            if len(picks) < 2:
                continue
        if len(picks) >= 2 and len(picks) >= len(idx):
            cen = _subpixel_refine(Bp, [Tsoft], picks)
            unsure = (pi["residual"] >= 0.08 or pi["overhang"] > 0.2 or pi.get("next_gain", 0) > 0.25
                      or pi["cost"] > 0.35 * nT)
            for (py, px) in cen:
                cy = oy + py + cen_off[0] + rep_off[0]
                cx = ox + px + cen_off[1] + rep_off[1]
                note = SPLIT_NOTE.format(n=len(picks))
                sc = round(min(best_score, 0.7), 3)
                if unsure:
                    note = OVERLAP_NOTE.format(n=len(picks) + 1) + "; " + note
                    sc = min(sc, 0.45)
                c = rgb[int(min(H - 1, max(0, round(cy)))), int(min(W - 1, max(0, round(cx))))]
                out.append(Marker(cx=float(cx), cy=float(cy), area=math.pi * radius ** 2, size_px=2 * radius,
                                  shape="template", filled=True, color="#%02x%02x%02x" % tuple(int(v * 255) for v in c),
                                  score=sc, note=note))
            continue
        if idx and not picks:
            # the example's footprint fits nowhere in this blob, or only inside a bigger shape: the correlation
            # peak sits on other markers (a square template on a merged pair of circles, a circle in a square)
            continue
        # keep the correlation peaks; flag a blob with clearly more ink than they explain
        U = np.zeros_like(B)
        for i in idx:
            m = found[i]
            U |= _stamp(B.shape, T, (int(round(m.cy - rep_off[0] - cen_off[0])) - sl[0].start,
                                     int(round(m.cx - rep_off[1] - cen_off[1])) - sl[1].start))
        left = float((B & ~ndi.binary_dilation(U)).sum()) / nT
        for i in idx:
            m = found[i]
            if left > 0.3 or (len(idx) == 1 and area > 1.25 * nT):
                m.note = OVERLAP_NOTE.format(n=max(len(idx) + 1, int(round(area / nT))))
                m.score = min(m.score, 0.45)
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# open (hollow) markers: outline templates of several shapes, fitted jointly to the ink
# ---------------------------------------------------------------------------
# shape name -> matplotlib marker (the vertex geometry of the outline); thin_diamond is reported as "diamond"
OUTLINE_SHAPES = {"circle": "o", "square": "s", "diamond": "D", "thin_diamond": "d", "triangle_up": "^",
                  "triangle_down": "v", "triangle_left": "<", "triangle_right": ">"}
_HALF_OFFSETS = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5))
OUTLINE_NOTE = "outline fit among overlapping markers"


def shape_family(shape: str) -> str:
    return "diamond" if shape in ("diamond", "thin_diamond") else shape


def _family_members(shape: str) -> List[str]:
    return ["diamond", "thin_diamond"] if shape_family(shape) == "diamond" else [shape]


_TPL_CACHE: Dict[tuple, np.ndarray] = {}


def outline_template(shape: str, m: float, w: float, off: Tuple[float, float] = (0.0, 0.0), ss: int = 5
                     ) -> np.ndarray:
    """Anti-aliased stroke of an open marker (odd square array, values 0..1): markersize m px (the path size,
    as matplotlib's markersize), edge width w px, miter joins; the shape centre sits at the middle pixel + off
    (y, x).  Polygons are exact (outer and inner offset half-planes), the circle is a ring."""
    key = (shape, round(m, 2), round(w, 2), off, ss)
    T = _TPL_CACHE.get(key)
    if T is not None:
        return T
    R = int(math.ceil(0.75 * m + w)) + 1
    n = 2 * R + 1
    g = (np.arange(n * ss) + 0.5) / ss - R - 0.5
    yy, xx = np.meshgrid(g - off[0], g - off[1], indexing="ij")
    if shape == "circle":
        st = np.abs(np.hypot(xx, yy) - m / 2) <= w / 2
    else:
        from matplotlib.markers import MarkerStyle
        ms = MarkerStyle(OUTLINE_SHAPES[shape])
        v = ms.get_path().transformed(ms.get_transform()).vertices[:-1] * m * [1, -1]     # image y is down
        cen = v.mean(axis=0)
        outer = np.ones(xx.shape, bool)
        inner = np.ones(xx.shape, bool)
        for i in range(len(v)):
            e = v[(i + 1) % len(v)] - v[i]
            nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)
            if nrm @ (cen - v[i]) > 0:
                nrm = -nrm
            c = float(nrm @ v[i])
            val = xx * nrm[0] + yy * nrm[1]
            outer &= val <= c + w / 2
            inner &= val <= c - w / 2
        st = outer & ~inner
    T = st.reshape(n, ss, n, ss).mean(axis=(1, 3))
    if len(_TPL_CACHE) > 2000:
        _TPL_CACHE.clear()
    _TPL_CACHE[key] = T
    return T


def ink_weight(rgb: np.ndarray, region: np.ndarray, bg, color=None, tol: float = 35.0,
               inks: Optional[List[Dict]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Soft ink map (0..1 = blend fraction from the background to the ink) of one colour, or of all ink.

    Every foreground pixel is given to the ink whose anti-aliasing line (background -> ink) passes nearest
    (Lab); with a colour only the pixels of the ink nearest to that colour count.  A fixed colour tolerance
    alone cannot separate close colours (crimson #c10020 / orange-red #f13a13 are 24 Lab apart), the nearest
    ink can.  Returns (W, the ink's RGB or None)."""
    bgv = np.asarray(bg, float)
    lab = to_lab(rgb)
    bl = to_lab(bgv.reshape(1, 3))[0]
    fg = (np.linalg.norm(lab - bl, axis=2) > 8) & region
    if inks is None:
        inks = ink_colours(rgb, region, bgv)
    cols = [np.asarray(o["rgb"], float) for o in inks]
    W = np.zeros(rgb.shape[:2])
    ys, xs = np.nonzero(fg)
    if not len(ys):
        return W, None
    px, pl = rgb[ys, xs], lab[ys, xs]
    own = -1
    if color is not None:
        target = parse_color(color) if not isinstance(color, np.ndarray) else color
        tl = to_lab(np.asarray(target, float).reshape(1, 3))[0]
        dt = [float(np.linalg.norm(o["lab"] - tl)) for o in inks]
        if dt and min(dt) < max(tol, 40):
            own = int(np.argmin(dt))
        else:
            cols.append(np.asarray(target, float))
            own = len(cols) - 1
    if not cols:                    # no distinct ink found: plain distance from the background
        W[ys, xs] = np.clip((np.linalg.norm(pl - bl, axis=1) - 8) / 40.0, 0, 1)
        return W, None
    res = [_seg_dist(px, pl, c, bgv) for c in cols]
    ds = np.array([r[0] for r in res])
    al = np.array([r[1] for r in res])
    win = np.argmin(ds, axis=0)
    dmin = ds[win, np.arange(len(win))]
    if own >= 0:
        W[ys, xs] = np.where((win == own) & (dmin < tol), al[own], 0.0)
        return W, cols[own]
    W[ys, xs] = np.where(dmin < tol, al[win, np.arange(len(win))], 0.0)
    return W, None


def _stroke_measure(Wm: np.ndarray) -> float:
    """Ink per skeleton pixel (~ the stroke width; compared with the same measure of a template)."""
    from skimage.morphology import skeletonize
    sk = skeletonize(Wm > 0.5)
    return float(Wm.sum() / max(1, sk.sum()))


def _stroke_width(crops: Sequence[np.ndarray], D: float) -> float:
    meas = float(np.median([_stroke_measure(c) for c in crops]))
    ws = np.arange(0.8, 4.01, 0.2)
    tm = np.array([_stroke_measure(outline_template("circle", max(4.0, 0.8 * D), w)) for w in ws])
    if np.all(np.diff(tm) > 0):
        return float(np.interp(meas, tm, ws))
    return float(ws[int(np.argmin(np.abs(tm - meas)))])


def _corr_full(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    return fftconvolve(img, k[::-1, ::-1], mode="same")


def estimate_outline_geometry(W: np.ndarray, region: np.ndarray, shapes: Sequence[str] = tuple(OUTLINE_SHAPES),
                              size: Optional[float] = None, max_crops: int = 24) -> Dict:
    """Marker size m (path size, like matplotlib's markersize) and edge width w of the open markers in the ink
    map W, fitted on blobs that look like single markers (convex once filled, round-ish).

    Each candidate m is scored by how many blobs ONE outline template of some shape explains well (sum of Dice
    overlap above 0.7): the right size explains the isolated markers almost exactly, a larger one only fits
    merged pairs loosely, a smaller one nothing.  size (outer size px) narrows the search.  Returns {} without
    ink, else {m, w, dice, n_examples, present: {shape: blobs it fits best}}."""
    from skimage.measure import regionprops
    lab, _ = ndi.label((W > 0.3) & region, structure=np.ones((3, 3), bool))
    singles = []
    for rp in regionprops(lab):
        h, w_ = rp.bbox[2] - rp.bbox[0], rp.bbox[3] - rp.bbox[1]
        if max(h, w_) >= 5 and min(h, w_) >= 0.45 * max(h, w_) and rp.area_filled > 0.85 * rp.area_convex:
            singles.append((rp.label, rp.slice, max(h, w_)))
    if not singles:
        if not size:
            return {}
        return {"m": max(3.0, 0.85 * size), "w": max(1.0, 0.12 * size), "dice": None, "n_examples": 0,
                "present": {}}
    singles.sort(key=lambda c: c[2])
    if size:
        singles = [c for c in singles if 0.7 * size <= c[2] <= 1.4 * size] or singles
    else:                          # big convex blobs are merged markers or other ink, not one marker
        q1 = float(np.percentile([c[2] for c in singles], 25))
        singles = [c for c in singles if c[2] <= 2.5 * q1]
    if len(singles) > max_crops:
        singles = [singles[i] for i in np.linspace(0, len(singles) - 1, max_crops).astype(int)]
    H, Wd = W.shape
    crops = []
    for i, sl, sz in singles:
        pad = sz // 2 + 3
        y0, y1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(Wd, sl[1].stop + pad)
        crops.append(np.where(lab[y0:y1, x0:x1] == i, W[y0:y1, x0:x1], 0.0))
    sizes = np.array([c[2] for c in singles], float)
    w = _stroke_width(crops, float(np.median(sizes)))
    # each blob is one marker: its best alignment lies within a few px of the blob's centre (direct window
    # products there, much cheaper than full correlations)
    Rmax = int(math.ceil(0.75 * 1.1 * max(float(sizes.max()), size or 0) + 1.3 * w)) + 3
    pads = []
    for c in crops:
        cp = np.zeros((c.shape[0] + 2 * Rmax, c.shape[1] + 2 * Rmax))
        cp[Rmax:Rmax + c.shape[0], Rmax:Rmax + c.shape[1]] = c
        ys_, xs_ = np.nonzero(c > 0.3)
        cy_, cx_ = (ys_.min() + ys_.max()) // 2 + Rmax, (xs_.min() + xs_.max()) // 2 + Rmax
        r_ = int(0.15 * max(c.shape)) + 2
        pads.append((cp, cy_, cx_, r_, float(c.sum())))

    def fits(m, w, offs):
        from numpy.lib.stride_tricks import sliding_window_view
        names = [s for s in shapes for _ in offs]
        Ts = np.stack([outline_template(s, m, w, off) for s in shapes for off in offs], axis=-1)
        nT = Ts.sum(axis=(0, 1))
        R = Ts.shape[0] // 2
        best = [(0.0, None)] * len(crops)
        if R > Rmax:
            return best
        for j, (cp, cy_, cx_, r_, sc) in enumerate(pads):
            win = sliding_window_view(cp, Ts.shape[:2])[cy_ - r_ - R:cy_ + r_ - R + 1, cx_ - r_ - R:cx_ + r_ - R + 1]
            d = 2 * np.tensordot(win, Ts, axes=([2, 3], [0, 1])).max(axis=(0, 1)) / (sc + nT)
            k = int(np.argmax(d))
            best[j] = (float(d[k]), names[k])
        return best

    def score(r):
        return float(sum(max(0.0, d - 0.7) for d, _ in r))
    if size:
        grid = np.arange(max(3.0, 0.6 * size), 1.05 * size + 1e-9, 0.5)
    else:
        lo, hi = float(np.percentile(sizes, 5)), float(np.percentile(sizes, 95))
        grid = np.arange(max(3.0, 0.5 * lo), 1.05 * hi + 1e-9, 1.0 if hi > 12 else 0.5)
    sc_m = [(score(fits(m, w, _HALF_OFFSETS[::3])), float(m)) for m in grid]
    q, m0 = max(sc_m)
    m = m0
    for m2 in np.arange(m0 - 0.75, m0 + 0.76, 0.25):
        if m2 >= 3 and abs(m2 - m0) > 1e-9:
            q2 = score(fits(m2, w, _HALF_OFFSETS[::3]))
            if q2 > q + 1e-9:
                q, m = q2, float(m2)
    for w2 in (0.8 * w, 0.9 * w, 1.1 * w, 1.2 * w):
        q2 = score(fits(m, w2, _HALF_OFFSETS[::3]))
        if q2 > q + 1e-9:
            q, w = q2, w2
    r = fits(m, w, _HALF_OFFSETS)
    good = [(d, s) for d, s in r if d >= 0.75]
    present: Dict[str, int] = {}
    for d, s in good:
        present[s] = present.get(s, 0) + 1
    return {"m": round(m, 2), "w": round(w, 2), "dice": round(float(np.median([d for d, _ in good])), 3) if good
            else None, "n_examples": len(good), "present": present}


def _outline_candidates(Wc: np.ndarray, Rf: np.ndarray, shapes: Sequence[str], m: float, w: float,
                        cov_min: float = 0.7, edge_min: float = 0.3, centre_ok: Optional[np.ndarray] = None
                        ) -> Dict[str, List[tuple]]:
    """Per shape: local maxima of the stroke coverage (best of four half-pixel template offsets), with
    coverage = ink under the stroke / stroke inside the region (markers clipped by the frame count; their
    centres must still lie where centre_ok, i.e. inside the frame)."""
    H, Wd = Wc.shape
    out: Dict[str, List[tuple]] = {}
    for s in shapes:
        T0 = outline_template(s, m, w)
        inside = _corr_full(Rf, T0) / float(T0.sum())
        best_cov = np.zeros((H, Wd))
        best_off = np.zeros((H, Wd), int)
        for k, off in enumerate(_HALF_OFFSETS):
            T = outline_template(s, m, w, off)
            cov = _corr_full(Wc, T) / (np.maximum(inside, 1e-6) * float(T.sum()))
            upd = cov > best_cov
            best_cov[upd] = cov[upd]
            best_off[upd] = k
        best_cov[inside < edge_min] = 0
        if centre_ok is not None:
            best_cov[~centre_ok] = 0
        mx = ndi.maximum_filter(best_cov, size=3)
        out[s] = [(float(best_cov[y, x]), s, int(best_off[y, x]), int(y), int(x))
                  for y, x in np.argwhere((best_cov >= mx - 1e-9) & (best_cov >= cov_min))]
    return out


def outline_pursuit(W: np.ndarray, region: np.ndarray, shapes: Sequence[str], m: float, w: float,
                    gain_min: float = 0.25, uniq_min: float = 0.15, cands: Optional[Dict[str, List]] = None
                    ) -> Tuple[List[Dict], Dict]:
    """Greedy joint fit of open-marker outlines of several shapes to the ink map W.

    Candidates (``_outline_candidates``) are taken best coverage first when at least gain_min of their stroke
    covers ink that no earlier pick explains; afterwards picks whose ink the others explain (unique share <
    uniq_min) are dropped, weakest first.  Overlapping hollow markers keep most of their own stroke visible, so
    touching rows of circles or a dense band of mixed shapes resolve into individual markers, each with its
    shape.  Returns (picks [{shape, cy, cx, cov, novel, unique, inside}], info)."""
    Wc = np.clip(W, 0, 1)
    Rf = region.astype(float)
    H, Wd = W.shape
    if cands is None:
        cands = _outline_candidates(Wc, Rf, shapes, m, w)
    order = sorted((c for s in shapes for c in cands.get(s, [])), key=lambda c: -c[0])

    def window(T, y, x):
        R = T.shape[0] // 2
        y0, y1, x0, x1 = y - R, y + R + 1, x - R, x + R + 1
        ty0, tx0 = max(0, -y0), max(0, -x0)
        ty1, tx1 = T.shape[0] - max(0, y1 - H), T.shape[1] - max(0, x1 - Wd)
        return (slice(max(0, y0), min(H, y1)), slice(max(0, x0), min(Wd, x1))), T[ty0:ty1, tx0:tx1]
    E = np.zeros((H, Wd))
    occ = {s: np.zeros((H, Wd), bool) for s in shapes}
    picks: List[Dict] = []
    for cv, s, k, y, x in order:
        if occ[s][y, x]:
            continue
        T_full = outline_template(s, m, w, _HALF_OFFSETS[k])
        sl, T = window(T_full, y, x)
        Tm = T * Rf[sl]
        nov = float((np.clip(Wc[sl] - E[sl], 0, 1) * Tm).sum() / max(1e-9, float(Tm.sum())))
        if nov < gain_min:
            continue
        E[sl] = np.maximum(E[sl], np.minimum(Wc[sl], T))
        ro = 2
        occ[s][max(0, y - ro):y + ro + 1, max(0, x - ro):x + ro + 1] = True
        picks.append({"shape": s, "k": k, "y": y, "x": x, "cov": cv, "novel": nov, "sl": sl, "T": T,
                      "inside": float(Tm.sum() / T_full.sum())})
    S = np.zeros((H, Wd))                  # sum of all placed strokes: what the others explain = S - own
    for p in picks:
        S[p["sl"]] += p["T"]

    def uniq(p):
        sl, T = p["sl"], p["T"]
        Tm = T * Rf[sl]
        return float(np.clip(np.minimum(Wc[sl], Tm) - (S[sl] - T), 0, 1).sum() / max(1e-9, float(Tm.sum())))
    while picks:
        u = [uniq(p) for p in picks]
        i = int(np.argmin(u))
        if u[i] >= uniq_min:
            for p, v in zip(picks, u):
                p["unique"] = v
            break
        S[picks[i]["sl"]] -= picks[i]["T"]
        picks.pop(i)
    info = {"residual_spots": 0}
    if picks:
        left = (Wc - np.minimum(S, 1.0)) > 0.5
        lb, nl = ndi.label(left, structure=np.ones((3, 3), bool))
        if nl:
            sizes = ndi.sum(left, lb, range(1, nl + 1))
            m_avg = float(np.mean([p["T"].sum() for p in picks]))
            info["residual_spots"] = int((sizes >= 0.25 * m_avg).sum())
    for p in picks:
        oy, ox = _HALF_OFFSETS[p["k"]]
        p["cy"], p["cx"] = p["y"] + oy, p["x"] + ox
        p.pop("sl")
        p.pop("T")
    return picks, info


def detect_open_markers(rgb: np.ndarray, fr: Frame, color=None, tol: float = 35.0, shape: Optional[str] = None,
                        marker_size: Optional[float] = None, exclude: Sequence = (), legend: Sequence = (),
                        geometry: Optional[Dict] = None, inks: Optional[List[Dict]] = None
                        ) -> Tuple[List[Marker], Dict]:
    """Open (hollow) markers of one colour (or all ink), several shapes at once, however much they overlap.

    The ink of the colour (nearest-ink assignment, soft anti-aliasing weights) is explained by outline
    templates synthesised for every shape present (circle, square, diamond / thin diamond, triangles) at the
    marker size and edge width fitted on the figure's isolated markers (``geometry`` from all ink can be passed
    in when this colour has no isolated marker).  Shapes that never fit confidently in this colour are dropped
    and the fit repeated, so ink of one shape is not handed to another.  Markers of every shape are returned
    (Marker.shape = the family name: thin diamonds are 'diamond'); filter by shape afterwards."""
    bg = background_color(rgb, fr)
    region = _inner_region(rgb.shape[:2], fr)
    for box in list(exclude) + [tuple(b) for b in legend]:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        region[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    if inks is None:
        inks = ink_colours(rgb, _inner_region(rgb.shape[:2], fr, pad=fr.thickness + 1), bg)
    W, ink = ink_weight(rgb, region, bg, color=parse_color(color) if color is not None else None, tol=tol,
                        inks=inks)
    W[~region] = 0
    # long straight lines (gridlines, axes) are not marker outlines (a row of touching triangles is a straight
    # run of a few markers: only much longer lines go)
    _, lines = _remove_straight_lines(W > 0.5, max(40, int(0.25 * (fr.right - fr.left))))
    W[ndi.binary_dilation(lines)] = 0
    ys, xs = np.nonzero(W > 0.05)
    if not len(ys):
        return [], {"note": "no ink of that colour inside the frame"}
    # work on the bounding box of the ink (+ a margin): much faster for a sparse colour
    pad = int(3 * (marker_size or (geometry or {}).get("m") or 15)) + 10
    y0, y1 = max(0, ys.min() - pad), min(W.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W.shape[1], xs.max() + pad + 1)
    Wb, Rb = W[y0:y1, x0:x1], region[y0:y1, x0:x1]
    geo = estimate_outline_geometry(Wb, Rb, size=marker_size)
    if geometry and (not geo or geo.get("n_examples", 0) < 5):
        geo = {**geometry, "from": "all ink"}
    if not geo:
        return [], {"note": "no marker-sized ink of that colour inside the frame"}
    m, w = float(geo["m"]), float(geo["w"])
    shapes = list(OUTLINE_SHAPES)
    inside_frame = np.zeros(W.shape, bool)
    e = fr.thickness / 2 + 1           # a marker centred on the frame line (a point at the axis limit) counts
    inside_frame[max(0, int(math.floor(fr.top - e))):int(math.ceil(fr.bottom + e)) + 1,
                 max(0, int(math.floor(fr.left - e))):int(math.ceil(fr.right + e)) + 1] = True
    cands = _outline_candidates(np.clip(Wb, 0, 1), Rb.astype(float), shapes, m, w,
                                centre_ok=inside_frame[y0:y1, x0:x1])
    for _ in range(4):
        picks, info = outline_pursuit(Wb, Rb, shapes, m, w, cands=cands)
        # a shape with no confident instance in this colour (clean stroke, mostly its own ink) is not one of
        # its shapes: drop it and fit again (its picks were other markers' ink)
        confident = {p["shape"] for p in picks if p["cov"] >= 0.8 and p["novel"] >= 0.6}
        keep = [s for s in shapes if s in confident or (shape and shape_family(s) == shape_family(shape))]
        if set(keep) == set(shapes) or not keep:
            break
        shapes = keep
    for p in picks:
        p["cy"] += y0
        p["cx"] += x0
    hexc = "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in (ink if ink is not None else (0, 0, 0)))
    size = m + w
    markers = []
    for p in picks:
        note = OUTLINE_NOTE if p["novel"] < 0.75 else ""
        if p["inside"] < 0.9:
            note = (note + "; " if note else "") + "clipped by the frame"
        mk = Marker(cx=float(p["cx"]), cy=float(p["cy"]), area=math.pi * (size / 2) ** 2, size_px=size,
                    shape=shape_family(p["shape"]), filled=False, color=hexc, score=round(p["cov"], 3), note=note)
        mk.features = [p["cov"], p["novel"], p.get("unique", 0.0)]
        markers.append(mk)
    markers.sort(key=lambda mk: (mk.cx, mk.cy))
    info.update({"outline_geometry": {k: geo[k] for k in ("m", "w", "dice", "n_examples") if k in geo},
                 "outline_shapes": shapes, "marker_diameter_px": round(size, 1),
                 "shape_counts": {s: sum(1 for p in picks if p["shape"] == s) for s in shapes}})
    return markers, info


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------
def trace_curve(rgb: np.ndarray, fr: Frame, color, tol: float = 35.0, step_px: int = 2,
                direction: str = "x", start_px: Optional[Tuple[float, float]] = None,
                remove_markers: bool = True) -> List[Tuple[float, float]]:
    """Follow a line of a given colour; returns pixel points (x, y).  direction 'x' scans columns, 'y' rows.
    With start_px the trace starts there and follows the curve outward in both directions (picks one of several
    same-coloured curves); otherwise it starts at the first column that contains the colour."""
    mask, _ = _foreground(rgb, background_color(rgb, fr), color=parse_color(color), tol=tol,
                          rivals=rival_colors(rgb, fr))
    pad = int(math.ceil(fr.thickness / 2 + 1))
    t, b, l, r = int(fr.top) + pad, int(math.ceil(fr.bottom)) - pad, int(fr.left) + pad, int(math.ceil(fr.right)) - pad
    inner = np.zeros_like(mask)
    inner[t:b + 1, l:r + 1] = mask[t:b + 1, l:r + 1]
    if remove_markers:                       # drop compact blobs much thicker than the line
        thick = ndi.binary_opening(inner, structure=_disk(2.5))
        inner = inner & ~ndi.binary_dilation(thick, iterations=1)
    lo, hi = (l, r) if direction == "x" else (t, b)
    jump = max(25, 8 * step_px)

    def centres(k):
        line = inner[:, k] if direction == "x" else inner[k, :]
        s_, e_ = _runs(line)
        return (s_ + e_) / 2.0

    def walk(ks, prev):
        pts = []
        for k in ks:
            c = centres(k)
            if not len(c):
                continue
            if prev is None:
                cc = c[0]
            else:
                cc = c[np.argmin(np.abs(c - prev))]
                if abs(cc - prev) > jump:        # jumped to another curve: skip this column
                    continue
            prev = cc
            pts.append((float(k), float(cc)) if direction == "x" else (float(cc), float(k)))
        return pts

    if start_px is None:
        return walk(range(lo, hi + 1, step_px), None)
    k0 = int(round(start_px[0] if direction == "x" else start_px[1]))
    p0 = start_px[1] if direction == "x" else start_px[0]
    k0 = min(max(k0, lo), hi)
    right = walk(range(k0, hi + 1, step_px), p0)
    left = walk(range(k0 - step_px, lo - 1, -step_px), right[0][1] if (right and direction == "x") else
                (right[0][0] if right else p0))
    return left[::-1] + right


# ---------------------------------------------------------------------------
# overlays
# ---------------------------------------------------------------------------
def _overlay(rgb, fr: Frame, ticks=None, markers=None, curve=None, legend=(), out_png="overlay.png",
             title: str = "", tri: Optional["Triangle"] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = rgb.shape[:2]
    dpi = 100
    scale = max(1.0, 900 / max(H, W))
    fig = plt.figure(figsize=(W * scale / dpi, H * scale / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")
    fs = max(6, min(10, int(0.018 * max(H, W) * scale)))
    if tri is not None:
        c = np.array(tri.corners + [tri.corners[0]])
        ax.plot(c[:, 0], c[:, 1], color="lime", lw=1, alpha=0.8)
        cen = c[:3].mean(axis=0)
        for i, (x, y) in enumerate(tri.corners):
            off = (np.array([x, y]) - cen)
            off = off / (np.linalg.norm(off) or 1) * 14
            ax.text(x + off[0], y + off[1], f"c{i}", color="darkgreen", fontsize=fs + 1, ha="center", va="center",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        for name, tks in (tri.ticks or {}).items():
            for t in tks:
                if "index" in t:
                    ax.plot(*t["px"], "o", ms=3, color="red")
                    ax.text(t["px"][0], t["px"][1] - 8, f"{name}{t['index']}", color="red", fontsize=max(5, fs - 2),
                            ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none",
                                                                 alpha=0.7))
    else:
        ax.plot([fr.left, fr.right, fr.right, fr.left, fr.left], [fr.bottom, fr.bottom, fr.top, fr.top, fr.bottom],
                color="lime", lw=1, alpha=0.8)
    if ticks and tri is None:
        for t in ticks.get("x", []):
            if "index" in t:
                ax.plot([t["px"]] * 2, [fr.bottom - 6, fr.bottom + 6], color="red", lw=1)
                ax.text(t["px"], fr.bottom - 9, f"x{t['index']}", color="red", fontsize=fs, ha="center", va="bottom",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        for t in ticks.get("y", []):
            if "index" in t:
                ax.plot([fr.left - 6, fr.left + 6], [t["px"]] * 2, color="blue", lw=1)
                ax.text(fr.left + 9, t["px"], f"y{t['index']}", color="blue", fontsize=fs, ha="left", va="center",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    for bx in legend or []:
        ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="gray", ls="--"))
    for i, m in enumerate(markers or []):
        note = getattr(m, "note", "") or ""
        flag = is_flagged(note)
        col = "gray" if m.in_legend else ("red" if flag else ("orange" if m.score < 0.5 else
                                                              ("deepskyblue" if note else "magenta")))
        r = max(3, m.size_px * 0.75)
        ax.add_patch(plt.Circle((m.cx, m.cy), r, fill=False, ec=col, lw=1.2))
        if flag and not m.in_legend:        # flagged: a second, dashed ring - look here
            ax.add_patch(plt.Circle((m.cx, m.cy), r * 1.5, fill=False, ec=col, lw=1.2, ls="--"))
        ax.text(m.cx + m.size_px * 0.8, m.cy - m.size_px * 0.8, str(i) + ("?" if flag and not m.in_legend else ""),
                color=col, fontsize=max(5, fs - 2), fontweight="bold" if flag else "normal")
    if curve:
        c = np.asarray(curve)
        ax.plot(c[:, 0], c[:, 1], color="magenta", lw=1.2)
    if title:
        ax.text(5, 5, title, color="black", fontsize=fs, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.8))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# provenance figure: the FINAL data values mapped back onto the original figure
# ---------------------------------------------------------------------------
PROV_COLOURS = ["#ff00d4", "#00a2ff", "#ff7b00", "#00b33c", "#8c3cff", "#e0b000", "#00d5e0", "#4d4dff", "#8db600",
                "#b5651d"]
PROV_WARN = "#e8000b"
PROV_CHECK = "#0a7d8c"


def _hex_lab(c) -> np.ndarray:
    return to_lab(parse_color(c).reshape(1, 3))[0]


def provenance_colours(series_colours: Sequence[Optional[str]], bg="#ffffff") -> List[str]:
    """One overlay colour per series: far (in Lab) from that series' own ink, from the other series' inks, from
    the background and from the colours already chosen."""
    inks = [_hex_lab(c) for c in series_colours if c]
    bgl = _hex_lab(bg)
    out: List[str] = []
    reserved = [_hex_lab(PROV_CHECK), _hex_lab(PROV_WARN)]      # the check marks and the warning style
    for c in series_colours:
        own = _hex_lab(c) if c else None
        best = None
        for p in PROV_COLOURS:
            pl = _hex_lab(p)
            d = [np.linalg.norm(pl - bgl) * 0.9]
            if own is not None:                   # it is drawn on top of its own series: most important
                d.append(np.linalg.norm(pl - own) * 0.8)
            d += [np.linalg.norm(pl - q) * 2.0 for q in inks if own is None or q is not own]
            d += [np.linalg.norm(pl - _hex_lab(u)) * 1.2 for u in out]
            d += [np.linalg.norm(pl - q) * 1.2 for q in reserved]
            sc = min(d) - (25 if p in out else 0)
            if best is None or sc > best[0]:
                best = (sc, p)
        out.append(best[1])
    return out


def _nice(v: float, step: float) -> float:
    """v rounded to a tenth of the tick step (step 200 -> nearest 20, 0.2 -> 0.02): sub-pixel tick
    jitter disappears, a wrong calibration (off by a step or a scale error) still shows."""
    if not np.isfinite(v) or not step:
        return v
    e = 10.0 ** math.floor(math.log10(abs(step)))
    step = min((m * e for m in (1, 2, 2.5, 5, 10)), key=lambda c: abs(math.log(c / abs(step))))  # measured -> nice
    q = step / 10.0
    dec = max(0, -int(math.floor(math.log10(q))) + 1)
    return float(round(round(v / q) * q, dec))


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.3g}"
    return f"{v:.6g}"


def calibration_checks(cal: AxisCal, ticks: List[Dict], lo: float, hi: float) -> List[Tuple[float, str]]:
    """Check marks for one axis: the value the calibration gives at each detected major tick, rounded to the
    tick spacing, mapped back to pixels.  Drawn next to the printed labels, a wrong calibration shows at once
    (the labels disagree, or the marks drift off the ticks)."""
    majors = sorted(t["px"] for t in ticks if "index" in t)
    vals = [float(cal.to_value(p)) for p in majors]
    out = []
    if len(vals) >= 2:
        if cal.scale == "log":
            vals = [10 ** round(math.log10(v), 2) if v > 0 else v for v in vals]
        else:
            step = float(np.median(np.abs(np.diff(vals))))
            vals = [_nice(v, step) for v in vals]
    else:                                   # no detected ticks: nice values across the calibrated range
        from matplotlib.ticker import LogLocator, MaxNLocator
        a, b = sorted([float(cal.to_value(lo)), float(cal.to_value(hi))])
        loc = LogLocator() if cal.scale == "log" else MaxNLocator(6)
        vals = [float(v) for v in loc.tick_values(a, b) if a <= v <= b]
    for v in vals:
        p = float(cal.to_px(v))
        if min(lo, hi) - 2 <= p <= max(lo, hi) + 2:
            out.append((p, _fmt(v)))
    return out


def axis_checks(calx: AxisCal, caly: AxisCal, ticks: Dict, fr: Frame) -> Dict:
    return {"x": calibration_checks(calx, ticks.get("x", []), fr.left, fr.right),
            "y": calibration_checks(caly, ticks.get("y", []), fr.top, fr.bottom)}


def ternary_checks(tcal: "TriCal") -> Dict:
    """Corners labelled with the element names + unlabelled marks every 0.2 along the edges, all mapped through
    the ternary calibration."""
    chk = [(float(q[0]), float(q[1]), e) for e, q in zip(tcal.elements, tcal.to_px(np.eye(3)))]
    for i in range(3):
        for j in range(i + 1, 3):
            for f in (0.2, 0.4, 0.6, 0.8):
                c = np.zeros(3)
                c[i], c[j] = f, 1 - f
                q = tcal.to_px(c)[0]
                chk.append((float(q[0]), float(q[1]), ""))
    return {"ternary": chk}


def ternary_description(tcal: "TriCal") -> str:
    return (f"ternary {'-'.join(tcal.elements)} from {tcal.n_points} known compositions "
            f"(max residual {tcal.residual:.3f})")


def axes_description(spec: Dict, cal: AxisCal, name: str) -> str:
    """'x: x0 = 0, x5 = 100 (linear, max residual 0.0 px)'"""
    spec = spec or {}
    parts = []
    if spec.get("ticks"):
        parts.append(", ".join(f"{name}{str(k).lstrip(name)} = {_fmt(float(v))}" for k, v in spec["ticks"].items()))
    if spec.get("range"):
        parts.append(f"frame edges {_fmt(float(spec['range'][0]))} .. {_fmt(float(spec['range'][1]))}")
    if spec.get("pixels"):
        parts.append(f"{len(spec['pixels'])} pixel/value pairs")
    return f"{name}: {'; '.join(parts) or '?'} ({cal.scale}, residual {cal.residual_px:.1f} px)"


def provenance_figure(rgb: np.ndarray, series: Sequence[Dict], out_png: str, header: Sequence[str],
                      fr: Optional[Frame] = None, checks: Optional[Dict] = None, tri: Optional["Triangle"] = None,
                      metadata: Optional[Dict[str, str]] = None) -> str:
    """Draw extracted data (already mapped to pixels of rgb) over the original figure for a check at a glance.

    series: [{name, kind: points|curve, px: (N, 2), flagged: [bool], colour (the series' ink, for contrast),
    size_px, notes}].  checks: {"x": [(px, label)], "y": [(py, label)]} or {"ternary": [(x, y, label)]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    H, W = rgb.shape[:2]
    scale = max(1.0, 1000 / max(H, W))
    dpi = 100
    w_in = W * scale / dpi
    n_leg_rows = max(1, int(math.ceil(len(series) / 3)))
    head_px = 18 * len(header) + 12
    leg_px = 22 * n_leg_rows + 30
    h_in = (H * scale + head_px + leg_px) / dpi
    fig = plt.figure(figsize=(max(w_in, 6.0), h_in), dpi=dpi)
    fw, fh = fig.get_size_inches() * dpi
    img_w = W * scale / fw
    ax = fig.add_axes([(1 - img_w) / 2, leg_px / fh, img_w, H * scale / fh])
    ax.imshow(rgb, interpolation="antialiased" if scale > 1.5 else "nearest")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis("off")
    for i, line in enumerate(header):
        fig.text(0.01, 1 - (8 + 18 * i) / fh, line, fontsize=9 if i else 10, va="top", ha="left",
                 fontweight="bold" if i == 0 else "normal", color="0.1")
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    cols = provenance_colours([s.get("colour") for s in series])
    handles = []
    for s, c in zip(series, cols):
        px = np.asarray(s.get("px", np.zeros((0, 2))), float).reshape(-1, 2)
        flags = list(s.get("flagged") or [False] * len(px))
        d = float(s.get("size_px") or 8.0)
        if s.get("kind") == "curve":
            ok = np.all(np.isfinite(px), axis=1)
            ax.plot(px[ok, 0], px[ok, 1], "-", color=c, lw=1.1, path_effects=halo, zorder=4)
            if ok.sum() <= 80:
                ax.plot(px[ok, 0], px[ok, 1], "o", color=c, ms=2.6, mec="white", mew=0.5, zorder=5)
            handles.append(Line2D([], [], color=c, lw=1.5, label=f"{s['name']}: line, {int(ok.sum())} samples"))
            continue
        rr = max(3.5, 0.8 * d)
        th = np.linspace(0, 2 * np.pi, 49)
        for (x, y), f in zip(px, flags):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if f:
                r2 = rr * 1.35
                ax.plot(x + r2 * np.cos(th), y + r2 * np.sin(th), "--", color=PROV_WARN, lw=1.3,
                        path_effects=halo, zorder=6)
                ax.text(x + r2 * 0.8, y - r2 * 0.8, "?", color=PROV_WARN, fontsize=9, fontweight="bold",
                        ha="left", va="bottom", path_effects=halo, zorder=7)
            ax.plot(x + rr * np.cos(th), y + rr * np.sin(th), "-", color=c, lw=0.9, path_effects=halo, zorder=5)
            for a in (0, 0.5 * np.pi, np.pi, 1.5 * np.pi):   # crosshair arms outside the ring
                ax.plot([x + rr * np.cos(a), x + 1.6 * rr * np.cos(a)], [y + rr * np.sin(a), y + 1.6 * rr * np.sin(a)],
                        "-", color=c, lw=0.9, path_effects=halo, zorder=5, solid_capstyle="butt")
            ax.plot([x], [y], ".", color=c, ms=1.6, zorder=6)
        nf = int(sum(bool(f) for f in flags))
        handles.append(Line2D([], [], color=c, marker="o", mfc="none", ms=8, lw=0,
                              label=f"{s['name']}: {len(px)} points" + (f" ({nf} flagged)" if nf else "")))
    if any(any(s.get("flagged") or []) for s in series):
        handles.append(Line2D([], [], color=PROV_WARN, marker="o", mfc="none", ms=10, lw=0, ls="--",
                              label="? dashed = flagged: may be overlapping / hidden markers - check"))
    # calibration check marks
    checks = checks or {}
    fs = 8
    if fr is not None and (checks.get("x") or checks.get("y")):
        L = max(6.0, 0.012 * max(H, W))
        for p, lab in checks.get("x", []):
            ax.plot([p, p], [fr.bottom - 1, fr.bottom - L], "-", color=PROV_CHECK, lw=1.4, path_effects=halo, zorder=6)
            ax.text(p, fr.bottom - L - 1, lab, color=PROV_CHECK, fontsize=fs, ha="center", va="bottom",
                    fontweight="bold", path_effects=halo, zorder=6)
        for p, lab in checks.get("y", []):
            ax.plot([fr.left + 1, fr.left + L], [p, p], "-", color=PROV_CHECK, lw=1.4, path_effects=halo, zorder=6)
            ax.text(fr.left + L + 2, p, lab, color=PROV_CHECK, fontsize=fs, ha="left", va="center",
                    fontweight="bold", path_effects=halo, zorder=6)
        handles.append(Line2D([], [], color=PROV_CHECK, lw=1.4,
                              label="teal ticks = tick values from the calibration (must match the printed labels)"))
    if checks.get("ternary"):
        cen = np.mean([(x, y) for x, y, lab in checks["ternary"] if lab], axis=0) if any(
            lab for *_, lab in checks["ternary"]) else None
        for x, y, lab in checks["ternary"]:
            if lab:
                off = np.array([x, y]) - cen if cen is not None else np.zeros(2)
                off = off / (np.linalg.norm(off) or 1) * 0.05 * max(H, W)
                ax.plot([x], [y], "+", color=PROV_CHECK, ms=9, mew=1.6, path_effects=halo, zorder=6)
                ax.text(x + off[0], y + off[1], lab, color=PROV_CHECK, fontsize=9, fontweight="bold", ha="center",
                        va="center", path_effects=halo, zorder=6)
            else:
                ax.plot([x], [y], "+", color=PROV_CHECK, ms=5, mew=1.1, path_effects=halo, zorder=6)
        handles.append(Line2D([], [], color=PROV_CHECK, marker="+", lw=0, ms=8,
                              label="teal + = corners / 0.2 steps from the calibration (labels must match)"))
    lax = fig.add_axes([0, 0, 1, (leg_px - 4) / fh])
    lax.axis("off")
    if handles:
        lax.legend(handles=handles, loc="center", ncol=min(3, len(handles)), fontsize=8, frameon=False,
                   handlelength=1.6, columnspacing=1.4)
    meta = {"Software": "calphad_tools digitize"}
    meta.update(metadata or {})
    # where the original image sits in this figure (figure pixels, origin top left) and the colour per series:
    # figure_x = left + (image_x + 0.5) * scale
    ax.apply_aspect()
    pos = ax.get_position()
    meta["image_placement"] = json.dumps({"left": round(pos.x0 * fw, 3), "top": round((1 - pos.y1) * fh, 3),
                                          "scale": round(pos.width * fw / W, 6)})
    meta["series_colours"] = json.dumps({s["name"]: c for s, c in zip(series, cols)})
    fig.savefig(out_png, dpi=dpi, metadata=meta)
    plt.close(fig)
    return out_png


def provenance_from_csvs(work: str, axes: Dict, csvs: Sequence[str], out_png: str, source: str = "",
                         tool: str = "digitized_to_zpf") -> Optional[str]:
    """Provenance figure straight from digitize_points / digitize_curve CSVs: their values mapped back onto the
    working image through the calibration ``axes`` (as given to digitize_points).  None if nothing to draw."""
    import csv as _csv
    import datetime
    rgb = load_rgb(work)
    tern = (axes or {}).get("ternary")
    if tern:
        tri = detect_triangle(rgb)
        tcal = calibrate_triangle(tern, tri)
        fr = tri.frame() if tri is not None else detect_frame(rgb)
        checks, cal_desc = ternary_checks(tcal), ternary_description(tcal)
    else:
        tri = tcal = None
        fr = detect_frame(rgb)
        ticks = detect_ticks(rgb, fr)
        calx, caly = _cal(axes or {}, ticks, fr)
        checks = axis_checks(calx, caly, ticks, fr)
        cal_desc = (axes_description((axes or {}).get("x"), calx, "x") + "; " +
                    axes_description((axes or {}).get("y"), caly, "y"))
    series: Dict[Tuple[str, str], Dict] = {}
    for path in csvs:
        with open(path, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            continue
        kind = "points" if "x_px" in rows[0] else "curve"
        for r in rows:
            name = (r.get("series") or Path(path).stem).strip()
            e = series.setdefault((kind, name), {"name": name, "kind": kind, "px": [], "flagged": [], "cols": [],
                                                 "sizes": [], "csv": Path(path).name})
            try:
                if tcal is not None:
                    f = [float(r[f"X({el})"]) for el in tcal.elements]
                    q = tcal.to_px(np.array(f))[0]
                else:
                    q = (float(calx.to_px(float(r["x"]))), float(caly.to_px(float(r["y"]))))
            except (KeyError, ValueError):
                continue
            e["px"].append((float(q[0]), float(q[1])))
            e["flagged"].append(is_flagged(r.get("note") or "") or (r.get("score") not in (None, "") and
                                                                     float(r["score"]) < 0.5))
            if r.get("color"):
                e["cols"].append(r["color"])
            if r.get("size_px"):
                e["sizes"].append(float(r["size_px"]))
    if not series:
        return None
    out = []
    for e in series.values():
        cols = e.pop("cols")
        sizes = e.pop("sizes")
        e["px"] = np.array(e["px"]).reshape(-1, 2)
        e["colour"] = cols[len(cols) // 2] if cols else None
        e["size_px"] = float(np.median(sizes)) if sizes else 8.0
        out.append(e)
    head = [f"Extracted data on the original figure: {Path(source or work).name}",
            f"calibration: {cal_desc}",
            f"{datetime.datetime.now():%Y-%m-%d %H:%M}, calphad_tools {tool}: the values in "
            + (Path(csvs[0]).name if len(csvs) == 1 else f"{len(csvs)} CSV files")
            + " mapped back to the image through the calibration"]
    return provenance_figure(rgb, out, out_png, head, fr=fr, checks=checks, tri=tri,
                             metadata={"Title": f"provenance of data extracted from {Path(source or work).name}",
                                       "Source": str(source or work)})


# ---------------------------------------------------------------------------
# ternary (Gibbs triangle) figures
# ---------------------------------------------------------------------------
@dataclass
class Triangle:
    """A composition triangle drawn in the image: corners c0 (bottom left), c1 (bottom right), c2 (top) in px,
    and the ticks found along the edges b (bottom, c0->c1), r (right, c1->c2), l (left, c2->c0)."""
    corners: List[Tuple[float, float]]
    thickness: float
    ticks: Dict[str, List[Dict]] = field(default_factory=dict)

    EDGES = {"b": (0, 1), "r": (1, 2), "l": (2, 0)}

    def frame(self) -> Frame:
        c = np.array(self.corners)
        return Frame(left=float(c[:, 0].min()), right=float(c[:, 0].max()), top=float(c[:, 1].min()),
                     bottom=float(c[:, 1].max()), thickness=self.thickness, boxed=False)

    def inside(self, px, py, margin: float = 0.0) -> np.ndarray:
        """True for points inside the triangle, or within ``margin`` px outside its edges."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        c = np.array(self.corners)
        ok = np.ones(np.broadcast(px, py).shape, bool)
        for i, j in self.EDGES.values():
            k = 3 - i - j
            (x0, y0), (x1, y1) = c[i], c[j]
            L = math.hypot(x1 - x0, y1 - y0)
            side = lambda x, y: ((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / L   # noqa: E731
            s_k = np.sign(side(*c[k]))
            ok &= side(px, py) * s_k >= -margin
        return ok


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _line_fit(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through points: (a point on it, unit direction)."""
    m = pts.mean(axis=0)
    u = np.linalg.svd(pts - m, full_matrices=False)[2][0]
    return m, u


def _intersect(l1, l2) -> Optional[np.ndarray]:
    (p, u), (q, w) = l1, l2
    A = np.array([u, -w]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * u


def _coverage(dark: np.ndarray, p0, p1, pad: int = 1, trim: float = 0.0) -> float:
    """Fraction of samples along the segment p0-p1 that have a dark pixel within ``pad`` px."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.hypot(*(p1 - p0)))
    if L < 2:
        return 0.0
    t = np.linspace(trim / L, 1 - trim / L, max(2, int(L - 2 * trim)))
    if len(t) < 2:
        return 0.0
    xy = p0[None] + t[:, None] * (p1 - p0)[None]
    H, W = dark.shape
    hit = np.zeros(len(xy), bool)
    xi, yi = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            xx, yy = np.clip(xi + dx, 0, W - 1), np.clip(yi + dy, 0, H - 1)
            hit |= dark[yy, xx]
    return float(hit.mean())


def detect_triangle(rgb: np.ndarray) -> Optional[Triangle]:
    """Find a composition triangle (equilateral Gibbs triangle or right triangle) from its three long straight
    edges: Hough segments -> line clusters -> the largest triangle whose three edges are drawn."""
    from itertools import combinations
    from skimage.transform import probabilistic_hough_line
    gray = rgb.mean(axis=2)
    dark = gray < dark_threshold(gray)
    H, W = dark.shape
    segs = probabilistic_hough_line(dark, threshold=10, line_length=int(0.25 * min(H, W)), line_gap=4,
                                    theta=np.deg2rad(np.linspace(-90, 90, 721)[:-1]), rng=0)
    clusters: List[Dict] = []
    for (x0, y0), (x1, y1) in segs:
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        nrm = np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
        rho = float(nrm @ np.array([x0, y0], float))
        L = math.hypot(x1 - x0, y1 - y0)
        for c in clusters:
            da = min(abs(c["ang"] - ang), 180 - abs(c["ang"] - ang))
            if da < 2.5 and abs(c["rho"] - rho * (1 if abs(c["ang"] - ang) < 90 else -1)) < 5:
                c["segs"].append(((x0, y0), (x1, y1)))
                c["len"] += L
                break
        else:
            clusters.append({"ang": ang, "rho": rho, "segs": [((x0, y0), (x1, y1))], "len": L})
    clusters = sorted(clusters, key=lambda c: -c["len"])[:8]
    ys, xs = np.nonzero(dark)
    allpts = np.column_stack([xs, ys]).astype(float)
    lines = []
    for c in clusters:
        pts = np.array([p for s in c["segs"] for p in s], float)
        p, u = _line_fit(pts)
        n = np.array([-u[1], u[0]])
        d = np.abs((allpts - p) @ n)
        t = (allpts - p) @ u
        tt = (pts - p) @ u
        sel = (d <= 2.5) & (t >= tt.min() - 3) & (t <= tt.max() + 3)
        if sel.sum() >= 10:
            p, u = _line_fit(allpts[sel])
        thick = float(sel.sum() / max(1.0, tt.max() - tt.min()))
        lines.append((p, u, thick))
    best = None
    for (a, b, c) in combinations(range(len(lines)), 3):
        ls = [lines[a], lines[b], lines[c]]
        angs = [math.degrees(math.atan2(l[1][1], l[1][0])) % 180 for l in ls]
        if any(min(abs(x - y), 180 - abs(x - y)) < 20 for x, y in combinations(angs, 2)):
            continue
        cs = [_intersect(ls[i][:2], ls[j][:2]) for i, j in ((0, 1), (1, 2), (2, 0))]
        if any(q is None for q in cs):
            continue
        cs = np.array(cs)
        if np.any(cs[:, 0] < -0.03 * W) or np.any(cs[:, 0] > 1.03 * W) or np.any(cs[:, 1] < -0.03 * H) \
                or np.any(cs[:, 1] > 1.03 * H):
            continue
        area = 0.5 * abs(_cross2(cs[1] - cs[0], cs[2] - cs[0]))
        if area < 0.08 * H * W:
            continue
        if min(_coverage(dark, cs[i], cs[j], pad=2, trim=6) for i, j in ((0, 1), (1, 2), (2, 0))) < 0.85:
            continue
        if best is None or area > best[0]:
            best = (area, cs, float(np.median([l[2] for l in ls])))
    if best is None:
        return None
    cs = best[1]
    top = int(np.argmin(cs[:, 1]))
    rest = sorted([i for i in range(3) if i != top], key=lambda i: cs[i, 0])
    order = [rest[0], rest[1], top]
    tri = Triangle(corners=[(float(cs[i, 0]), float(cs[i, 1])) for i in order], thickness=max(1.0, best[2]))
    tri.ticks = detect_triangle_ticks(rgb, tri)
    return tri


def detect_triangle_ticks(rgb: np.ndarray, tri: Triangle) -> Dict[str, List[Dict]]:
    """Tick marks along each triangle edge (perpendicular or at +-30 deg, i.e. parallel to another edge; outside
    or inside).  Per edge: [{"px": [x, y], "t": fraction along the edge, "major": bool, "index": i}]."""
    gray = rgb.mean(axis=2)
    bg = float(np.median(gray))
    thr = dark_threshold(gray)
    dark = gray < thr + 0.3 * (bg - thr)
    H, W = dark.shape
    max_len = int(max(6, min(30, 0.035 * min(H, W))))
    c = np.array(tri.corners)
    out: Dict[str, List[Dict]] = {}
    for name, (i, j) in Triangle.EDGES.items():
        p0, p1 = c[i], c[j]
        L = float(np.hypot(*(p1 - p0)))
        u = (p1 - p0) / L
        n = np.array([-u[1], u[0]])
        if n @ (c[3 - i - j] - p0) > 0:
            n = -n                                            # outward normal
        best = []
        for side in (1, -1):
            for rot in (0.0, 30.0, -30.0):
                r = math.radians(rot)
                d = side * (n * math.cos(r) + u * math.sin(r))
                start = tri.thickness / 2 + 1.5
                ts = np.arange(int(0.02 * L), int(0.98 * L))
                counts = np.zeros(len(ts), int)
                alive = np.ones(len(ts), bool)
                for k in range(max_len):
                    q = p0[None] + ts[:, None] * u[None] + (start + k) * d[None]
                    xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
                    okk = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                    hit = np.zeros(len(ts), bool)
                    hit[okk] = dark[yi[okk], xi[okk]]
                    alive &= hit
                    counts += alive
                is_t = counts >= 3
                s, e = _runs(is_t)
                ticks = []
                for a, b in zip(s, e):
                    if b - a + 1 > max(4, 2 * tri.thickness + 3):
                        continue
                    seg = counts[a:b + 1]
                    pos = float(np.average(ts[a:b + 1], weights=seg))
                    ticks.append((pos, int(seg.max())))
                if len(ticks) >= 2 and (not best or (_regularity([p for p, _ in ticks]) + 0.1 * len(ticks)
                                                     > _regularity([p for p, _ in best]) + 0.1 * len(best))):
                    best = ticks
        lens = np.array([l for _, l in best]) if best else np.array([])
        major_len = lens.max() if lens.size else 0
        res = []
        for pos, l in sorted(best):
            q = p0 + pos * u
            res.append({"px": [round(float(q[0]), 2), round(float(q[1]), 2)], "t": round(pos / L, 4),
                        "major": bool(l >= 0.7 * major_len)})
        k = 0
        for t in res:
            if t["major"]:
                t["index"] = k
                k += 1
        out[name] = res
    return out


@dataclass
class TriCal:
    """Affine map pixel -> mole (or atomic) fractions of the three corner elements."""
    elements: List[str]
    A: np.ndarray                 # 3 x 3: fractions = A @ [px, py, 1]
    residual: float               # max |fraction error| at the calibration points
    n_points: int

    def to_fractions(self, px, py) -> np.ndarray:
        px, py = np.atleast_1d(np.asarray(px, float)), np.atleast_1d(np.asarray(py, float))
        return (self.A @ np.vstack([px, py, np.ones_like(px)])).T

    def to_px(self, fractions) -> np.ndarray:
        f = np.atleast_2d(np.asarray(fractions, float))
        M = self.A[:2, :2]
        return np.linalg.solve(M, (f[:, :2] - self.A[:2, 2]).T).T


def calibrate_triangle(spec: Dict, tri: Optional[Triangle]) -> TriCal:
    """spec (any combination, >= 3 non-collinear known compositions in total):
    corners: {"c0": "CO", "c1": "NI", "c2": "CR"} (detected corners) | corners_px: {"CO": [x, y], ...} |
    ticks: {"b2": {"CR": 0.2, "NI": 0.8}, ...} (detected edge ticks b/r/l = bottom/right/left; unnamed elements
    = 0) |
    points: [[px, py, {"CR": 0.2, "NI": 0.3}], ...]."""
    pairs: List[Tuple[float, float, Dict[str, float]]] = []
    els: List[str] = []

    def add_el(e):
        e = str(e).upper()
        if e not in els:
            els.append(e)
        return e
    corners = spec.get("corners") or {}
    for k, el in corners.items():
        if tri is None:
            raise ValueError("no triangle was detected: give corners_px or points instead of corner indices")
        i = int(str(k).lower().lstrip("c"))
        if not 0 <= i <= 2:
            raise ValueError(f"corner index {k} must be c0, c1 or c2")
        e = add_el(el)
        pairs.append((tri.corners[i][0], tri.corners[i][1], {e: 1.0}))
    for el, xy in (spec.get("corners_px") or {}).items():
        e = add_el(el)
        pairs.append((float(xy[0]), float(xy[1]), {e: 1.0}))
    for key, comp in (spec.get("ticks") or {}).items():
        if tri is None:
            raise ValueError("no triangle was detected: tick indices are not available")
        m = str(key).lower().strip()
        edge, idx = m[0], int(m[1:])
        majors = {t["index"]: t["px"] for t in tri.ticks.get(edge, []) if "index" in t}
        if idx not in majors:
            raise ValueError(f"tick {key} not detected (edge {edge}: {edge}0..{edge}{len(majors) - 1})")
        pairs.append((majors[idx][0], majors[idx][1], {add_el(e): float(v) for e, v in comp.items()}))
    for p in spec.get("points") or []:
        pairs.append((float(p[0]), float(p[1]), {add_el(e): float(v) for e, v in p[2].items()}))
    for el in spec.get("elements") or []:
        add_el(el)
    if len(els) != 3:
        raise ValueError(f"a ternary calibration needs exactly three elements, got {els}")
    rows, rhs = [], []
    for px, py, comp in pairs:
        vals = {e: float(comp.get(e, 0.0)) for e in els}
        given = [e for e in els if e in comp]
        if len(given) == 2 and len(comp) == 2 and abs(sum(comp.values()) - 1) > 1e-6:
            raise ValueError(f"composition {comp} does not sum to 1 (edge points: give both non-zero elements)")
        if len(given) == 1 and abs(vals[given[0]] - 1.0) > 1e-9:
            raise ValueError(f"composition {comp}: give at least two element fractions")
        if any(v > 1.0 + 1e-9 for v in vals.values()):
            raise ValueError(f"composition {comp}: give fractions (0..1), not percent")
        rows.append([px, py, 1.0])
        rhs.append([vals[e] for e in els])
    X, Y = np.array(rows), np.array(rhs)
    if len(X) < 3 or np.linalg.matrix_rank(X) < 3:
        raise ValueError("ternary calibration needs >= 3 known points that are not on one line (e.g. the three "
                         "corners, or ticks on two edges)")
    A = np.linalg.lstsq(X, Y, rcond=None)[0].T
    res = float(np.max(np.abs(X @ A.T - Y))) if len(X) > 3 else 0.0
    return TriCal(elements=els, A=A, residual=res, n_points=len(X))


def mask_outside(rgb: np.ndarray, tri: Triangle, margin: float) -> np.ndarray:
    """Copy of the image with everything farther than ``margin`` px outside the triangle painted background."""
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    ok = tri.inside(xx + 0.0, yy + 0.0, margin=margin)
    out = rgb.copy()
    out[~ok] = background_color(rgb)
    return out


# ---------------------------------------------------------------------------
# regions of a phase diagram bounded by its drawn lines
# ---------------------------------------------------------------------------
def segment_regions(rgb: np.ndarray, fr: Frame, tri: Optional[Triangle] = None, discs: Sequence = (),
                    boxes: Sequence = (), ink: float = 40.0, color=None, tol: float = 35.0,
                    text_frac: float = 0.05, min_area: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """Split the plot area into the regions enclosed by the figure's drawn lines.

    Walls = ink strokes (Lab distance from the paper > ``ink``; or only strokes of ``color``) inside the frame /
    triangle, with text glyphs (small isolated components, smaller than ``text_frac`` of the plot) removed, plus
    the given marker discs [(cx, cy, r)] and boxes [l, t, r, b] (legend).  Small gaps are bridged.  Returns
    (region id per pixel, -1 = wall / outside / too small; info)."""
    H, W = rgb.shape[:2]
    bg = background_color(rgb, fr)
    if color is not None:
        walls, _ = _foreground(rgb, bg, color=parse_color(color), tol=tol)
    else:
        lab = to_lab(rgb)
        walls = np.linalg.norm(lab - to_lab(np.asarray(bg).reshape(1, 3))[0], axis=2) > ink
    pad = fr.thickness / 2 + 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    if tri is not None:
        area = tri.inside(xx + 0.0, yy + 0.0, margin=-pad)
    else:
        area = (xx > fr.left + pad) & (xx < fr.right - pad) & (yy > fr.top + pad) & (yy < fr.bottom - pad)
    walls &= area
    # text glyphs: small components that do not touch the frame
    size = text_frac * min(fr.right - fr.left, fr.bottom - fr.top)
    lbl, n = ndi.label(walls, structure=np.ones((3, 3), bool))
    border = ndi.binary_dilation(~area, iterations=2)
    touching = set(np.unique(lbl[border & (lbl > 0)]))
    removed = 0
    for k, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None or (k in touching):
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < size:
            walls[sl][lbl[sl] == k] = False
            removed += 1
    for cx, cy, r in discs:
        walls |= ((xx - cx) ** 2 + (yy - cy) ** 2 <= (r + 1.0) ** 2) & area
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        walls[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
    walls = walls | ndi.binary_closing(walls, structure=_disk(1.5))
    free = area & ~walls
    reg, nreg = ndi.label(free)              # 4-connectivity: diagonal leaks through thin lines are closed
    reg = reg - 1
    min_area = min_area or max(25, int(0.0004 * area.sum()))
    sizes = np.bincount(reg[reg >= 0].ravel(), minlength=nreg)
    small = np.flatnonzero(sizes < min_area)
    if len(small):
        reg[np.isin(reg, small)] = -1
    ids = [int(i) for i in np.flatnonzero(sizes >= min_area)]
    return reg, {"regions": ids, "sizes": {i: int(sizes[i]) for i in ids}, "text_components_removed": removed,
                 "wall_fraction": float(walls[area].mean()) if area.any() else 0.0}


def region_at(reg: np.ndarray, px: float, py: float, search: int = 15) -> int:
    """Region id at a pixel, or the most common region id within ``search`` px (label text, line under it)."""
    H, W = reg.shape
    x, y = int(round(px)), int(round(py))
    if 0 <= x < W and 0 <= y < H and reg[y, x] >= 0:
        return int(reg[y, x])
    for r in range(2, search + 1, 2):
        win = reg[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        vals = win[win >= 0]
        if vals.size:
            return int(np.bincount(vals).argmax())
    return -1


def regions_around(reg: np.ndarray, px: float, py: float, r0: float, r_max: float, n_ang: int = 48,
                   extra: float = 5.0) -> Dict[int, float]:
    """Region ids seen on growing rings around a point -> smallest ring radius (px) at which each was seen
    (rings grow until ``extra`` px beyond the radius where the second region appeared)."""
    H, W = reg.shape
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    seen: Dict[int, float] = {}
    r = r0
    while r <= r_max:
        xs = np.round(px + r * np.cos(ang)).astype(int)
        ys = np.round(py + r * np.sin(ang)).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        for v in np.unique(reg[ys[ok], xs[ok]]):
            if v >= 0 and int(v) not in seen:
                seen[int(v)] = float(r)
        if len(seen) >= 2 and r >= sorted(seen.values())[1] + extra:
            break
        r += 1.5
    return seen


def horizontal_neighbours(reg: np.ndarray, rid: int, max_wall: int = 15) -> Tuple[Optional[int], Optional[int]]:
    """Most common region to the left and to the right of a region across one wall (binary T-x diagrams)."""
    ys, xs = np.nonzero(reg == rid)
    if not len(ys):
        return None, None
    H, W = reg.shape
    votes = {"l": {}, "r": {}}
    rows = np.unique(ys)
    for y in rows[:: max(1, len(rows) // 60)]:
        row = reg[y]
        s, e = _runs(row == rid)
        for a, b in zip(s, e):
            for side, start, step in (("l", a - 1, -1), ("r", b + 1, 1)):
                x, k = start, 0
                while 0 <= x < W and row[x] < 0 and k < max_wall:
                    x += step
                    k += 1
                if 0 <= x < W and row[x] >= 0 and row[x] != rid:
                    v = int(row[x])
                    votes[side][v] = votes[side].get(v, 0) + 1
    pick = lambda d: max(d, key=d.get) if d else None   # noqa: E731
    return pick(votes["l"]), pick(votes["r"])


def region_neighbours(reg: np.ndarray, rid: int, gap: int = 6) -> Dict[int, int]:
    """Regions touching a region across a wall (dilation by ``gap`` px) -> number of contact pixels."""
    m = reg == rid
    ring = ndi.binary_dilation(m, iterations=gap) & ~m
    vals = reg[ring]
    vals = vals[(vals >= 0) & (vals != rid)]
    return {int(k): int(c) for k, c in zip(*np.unique(vals, return_counts=True))} if vals.size else {}


# ---------------------------------------------------------------------------
# markers joined by straight lines (tie-lines, tie-triangles)
# ---------------------------------------------------------------------------
def marker_links(rgb: np.ndarray, fr: Frame, markers: List["Marker"], color=None, tol: float = 35.0,
                 min_cover: float = 0.85) -> Dict[int, str]:
    """Tie-line / tie-triangle ids for markers joined by drawn straight lines.

    Two markers are linked when the straight segment between their outlines is covered by ink.  Triangles of three
    mutually linked, non-collinear markers are tie-triangles; any other link is a tie-line, unless the line
    continues straight through one of its markers to a further marker (then it is a drawn boundary curve).  Ids
    are built from pixel positions."""
    from itertools import combinations
    bg = background_color(rgb, fr)
    if color is not None:
        ink, _ = _foreground(rgb, bg, color=parse_color(color), tol=tol)
    else:
        ink, _ = _foreground(rgb, bg)
    n = len(markers)
    P = np.array([[m.cx, m.cy] for m in markers]) if n else np.zeros((0, 2))
    links = set()
    for i, j in combinations(range(n), 2):
        d = float(np.hypot(*(P[i] - P[j])))
        ri, rj = markers[i].size_px / 2, markers[j].size_px / 2
        if d < 1.6 * (ri + rj) + 4:
            continue
        u = (P[j] - P[i]) / d
        a, b = P[i] + u * (1.3 * ri + 1), P[j] - u * (1.3 * rj + 1)
        # no third marker sitting on the segment (then only the direct neighbours are linked)
        if any(k not in (i, j) and abs(_cross2(u, P[k] - P[i])) < 0.6 * markers[k].size_px
               and 0 < (P[k] - P[i]) @ u < d for k in range(n)):
            continue
        if _coverage(ink, a, b, pad=1) >= min_cover:
            links.add((i, j))
    adj: Dict[int, set] = {k: set() for k in range(n)}
    for i, j in links:
        adj[i].add(j)
        adj[j].add(i)

    def ident(members):
        k = min(members, key=lambda q: (round(P[q][0]), round(P[q][1])))
        return f"{int(round(P[k][0]))}_{int(round(P[k][1]))}"
    out: Dict[int, str] = {}
    for i, j, k in combinations(range(n), 3):
        if j in adj[i] and k in adj[i] and k in adj[j]:
            area = abs(_cross2(P[j] - P[i], P[k] - P[i])) / 2
            per = sum(np.hypot(*(P[a] - P[b])) for a, b in ((i, j), (j, k), (k, i)))
            if area > 0.01 * per ** 2 and not any(q in out for q in (i, j, k)):
                tid = "tri" + ident((i, j, k))
                for q in (i, j, k):
                    out[q] = tid

    def continued(i, j):
        """The line i-j goes on through marker i to another marker (a drawn boundary curve, not a tie-line)."""
        u = (P[j] - P[i]) / (np.hypot(*(P[j] - P[i])) or 1)
        for k in adj[i]:
            if k != j:
                w = (P[k] - P[i]) / (np.hypot(*(P[k] - P[i])) or 1)
                if float(u @ w) < -0.87:
                    return True
        return False
    cand = [(i, j) for i, j in links if i not in out and j not in out and not continued(i, j) and not continued(j, i)]
    deg: Dict[int, int] = {}
    for i, j in cand:
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    for i, j in cand:
        if deg[i] == 1 and deg[j] == 1:
            tid = "tie" + ident((i, j))
            out[i] = out[j] = tid
    return out


# ---------------------------------------------------------------------------
# agent tools
# ---------------------------------------------------------------------------
_AXIS_SPEC ={"type": "object", "description": "{ticks: {index: value, ...} (>=2, from the numbered preview) | "
              "range: [value at left/top frame edge, value at right/bottom frame edge] | pixels: [[px, value], ...]; "
              "scale: linear|log}"}


def _mostly_open(markers: List[Marker], shape: Optional[str] = None) -> bool:
    """Most detected markers (of the requested shape, if it was detected) are open (hollow)."""
    ms = [m for m in markers if not m.in_legend]
    if not ms:
        return False
    if shape:
        same = [m for m in ms if shape_family(m.shape) == shape_family(shape)]
        if same and sum(m.filled for m in same) > 0.3 * len(same):
            return False
    return sum(not m.filled for m in ms) >= 0.7 * len(ms)


def _prefer_outline(markers: List[Marker], outer_px: Optional[float] = None) -> bool:
    """Auto mode, open markers: fit outlines (``detect_open_markers``) when they overlap a lot.

    The component detector handles isolated and lightly touching markers well; when many detections come from
    split or unresolved blobs, or the blobs are much bigger than one marker of the figure's fitted outline size
    (dense bands, rows of touching circles, mixed shapes in one colour), the joint outline fit is used."""
    ms = [m for m in markers if not m.in_legend]
    if not ms:
        return False
    crowded = sum(1 for m in ms if m.note)
    if crowded >= max(2, 0.2 * len(ms)) or (len(ms) <= 5 and crowded >= 1):
        return True
    return bool(outer_px) and float(np.median([m.size_px for m in ms])) > 1.4 * outer_px


def _cal(axes_spec: Dict, ticks, fr: Frame) -> Tuple[AxisCal, AxisCal]:
    xs = dict(axes_spec.get("x") or {})
    ys = dict(axes_spec.get("y") or {})
    if "range" in ys:           # users give [bottom value, top value]; pixel low edge is the top
        lo, hi = ys["range"]
        ys["range"] = [hi, lo]
    return (calibrate_axis(xs, ticks["x"], fr.left, fr.right, "x"),
            calibrate_axis(ys, ticks["y"], fr.top, fr.bottom, "y"))


def register(tb) -> None:
    S = {"type": "string"}
    N = {"type": "number"}
    src = {"image": {**S, "description": "image file, or a PDF with page"}, "page": {"type": "integer"},
           "crop": {"type": "array", "items": N, "description": "[left, top, right, bottom] fractions of the page/"
                    "image (or pixels) around ONE plot; loose is fine"},
           "dpi": {"type": "integer", "description": "PDF render dpi (default 300)"}}
    state: Dict[str, Dict] = {}
    prov: Dict[str, Dict] = {}          # working image -> series extracted from it in this session

    def _provenance(work, image, page, crop, st, tool, key, entry, cal_line, checks) -> str:
        """Add/replace one series of this working image and redraw the combined provenance figure."""
        import datetime
        rec = prov.setdefault(work, {"series": {}, "tools": [], "cal": {}})
        rec["series"][key] = entry
        rec["cal"][key] = cal_line
        if tool not in rec["tools"]:
            rec["tools"].append(tool)
        src = Path(str(image)).name + (f", page {page}" if page else "") + (
            f", crop {[round(float(c), 3) for c in crop]}" if crop else "")
        head = [f"Extracted data on the original figure: {src}",
                f"calibration: {cal_line}",
                f"{datetime.datetime.now():%Y-%m-%d %H:%M}, calphad_tools {' + '.join(rec['tools'])}; ring + "
                f"crosshair = each CSV value mapped back to the image through its calibration"]
        if len(set(rec["cal"].values())) > 1:
            head.insert(2, "NOTE: the series were extracted with different calibrations (each is drawn with its "
                           "own; the teal check marks use the latest)")
        out = tb.outp(None, f"{Path(work).stem}_provenance.png")
        meta = {"Title": f"provenance of data extracted from {src}",
                "Description": "; ".join(f"{e['name']}: {len(e['px'])} {e['kind']} -> {e.get('csv', '')}"
                                         for e in rec["series"].values()),
                "Source": str(image)}
        provenance_figure(st["rgb"], list(rec["series"].values()), out, head, fr=st["frame"], checks=checks,
                          tri=st.get("tri"), metadata=meta)
        return out

    def _work(image, page, crop, dpi):
        out_dir = tb.outdir(None, "digitize")
        return prepare_image(tb.p(image), out_dir, page, crop, dpi or 300)

    def _setup(work, ternary=None):
        st = state.get(work)
        if st is not None and (ternary is None or bool(st.get("tri")) == bool(ternary)):
            return st
        rgb = load_rgb(work)
        fr = tri = None
        err = None
        if not ternary:
            try:
                fr = detect_frame(rgb)
            except ValueError as e:
                err = e
        if fr is None:
            tri = detect_triangle(rgb)
            if tri is None:
                if ternary:
                    raise ValueError("no composition triangle found (three long straight edges). Crop tighter "
                                     "around the triangle, or calibrate with axes.ternary.corners_px / points.")
                raise err
            fr = tri.frame()
        ticks = detect_ticks(rgb, fr) if tri is None else {"x": [], "y": []}
        st = {"rgb": rgb, "frame": fr, "ticks": ticks, "tri": tri}
        if tri is not None:          # blank everything outside the triangle (labels, legend, tick labels)
            st["rgb_det"] = mask_outside(rgb, tri, margin=max(8.0, 0.015 * max(rgb.shape[:2])))
        state[work] = st
        return st

    @tb.add("digitize_setup", "Step 1 of extracting data from a figure (paper PDF page or image file): detects the "
            "plot frame and tick marks and returns a preview with ticks numbered x0.. / y0.. plus the colours and "
            "marker size found. Then read the tick LABEL values next to a few numbered ticks and call digitize_points "
            "/ digitize_curve with axes={x: {ticks: {i: value}}, y: {...}}. Gibbs-triangle (ternary) figures are "
            "detected automatically (or ternary=true): the preview numbers the corners c0..c2 and the edge ticks "
            "(b0.. bottom, r0.. right, l0.. left); then axes={ternary: {corners: {c0: EL, c1: EL, c2: EL}}}.",
            {**src, "ternary": {"type": "boolean", "description": "the figure is a composition triangle"}}, ["image"])
    def digitize_setup(image, page=None, crop=None, dpi=None, ternary=None):
        work = _work(image, page, crop, dpi)
        st = _setup(work, ternary)
        rgb, fr, ticks, tri = st["rgb"], st["frame"], st["ticks"], st.get("tri")
        rgb_det = st.get("rgb_det", rgb)
        bg = background_color(rgb, fr)
        fg, _ = _foreground(rgb_det, bg)
        inner = np.zeros_like(fg)
        pad = int(fr.thickness) + 2
        inner[int(fr.top) + pad:int(fr.bottom) - pad, int(fr.left) + pad:int(fr.right) - pad] = True
        inks = ink_colours(rgb_det, inner, bg)
        clusters = ([{"color": "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in o["rgb"]),
                      "name": nearest_name(o["rgb"]), "pixels": o["pixels"]} for o in inks]
                    or color_clusters(rgb_det, fg & inner))
        try:
            markers, info = detect_markers(rgb_det, fr)
        except Exception:  # noqa: BLE001
            markers, info = [], {}
        prev = str(Path(work).with_name(Path(work).stem + "_setup.png"))
        colours = "ink colours inside the plot (most first; give one as digitize_points color): " + ", ".join(
            f"{c['name']} {c['color']} ({c['pixels']} px)" for c in clusters[:8])
        n_fl = sum(is_flagged(m) for m in markers if not m.in_legend)
        auto = (f"auto marker estimate: {len(markers)} candidates, diameter ~{info.get('marker_diameter_px', '?')} px"
                + (f" ({n_fl} may be overlapping markers)" if n_fl else "")
                + (f", legend box(es) at {info['legend_boxes']}" if info.get("legend_boxes") else ""))
        if tri is not None:
            _overlay(rgb, fr, tri=tri, out_png=prev,
                     title="ternary: name the element at c0, c1, c2 (or read >=2 tick labels on two edges)")
            tk = {e: [t for t in v if "index" in t] for e, v in tri.ticks.items()}
            txt = [f"working image: {tb.rel(work)} ({rgb.shape[1]}x{rgb.shape[0]} px)",
                   "composition TRIANGLE: corners (px) " + ", ".join(
                       f"c{i} ({x:.1f}, {y:.1f})" for i, (x, y) in enumerate(tri.corners))
                   + " (c0 bottom left, c1 bottom right, c2 top)",
                   "edge ticks: " + "; ".join(f"{e} ({'c%d->c%d' % Triangle.EDGES[e]}): {e}0..{e}{len(v) - 1}"
                                              for e, v in tk.items() if v) if any(tk.values()) else
                   "edge ticks: none detected",
                   colours, auto,
                   "Next: digitize_points with axes={\"ternary\": {\"corners\": {\"c0\": EL, \"c1\": EL, \"c2\": EL}}} "
                   "(read the corner labels), or ticks {\"b2\": {\"EL1\": 0.2, \"EL2\": 0.8}, ...} on two edges."]
            return "\n".join(txt), [prev]
        _overlay(rgb, fr, ticks=ticks, legend=info.get("legend_boxes"), out_png=prev,
                 title="numbered ticks: read the label values of >=2 x and >=2 y ticks")
        nx = sum(1 for t in ticks["x"] if "index" in t)
        ny = sum(1 for t in ticks["y"] if "index" in t)
        txt = [f"working image: {tb.rel(work)} ({rgb.shape[1]}x{rgb.shape[0]} px)",
               f"frame (px): left {fr.left:.1f}, right {fr.right:.1f}, top {fr.top:.1f}, bottom {fr.bottom:.1f}"
               f"{' (closed box)' if fr.boxed else ''}",
               f"major ticks: x0..x{nx - 1} at px {[t['px'] for t in ticks['x'] if 'index' in t]}",
               f"             y0..y{ny - 1} at px {[t['px'] for t in ticks['y'] if 'index' in t]}  (y0 = top)",
               colours, auto,
               "If ticks are missing/wrong, calibrate with range (frame-edge values) or pixels instead."]
        return "\n".join(txt), [prev]

    @tb.add("digitize_points", "Step 2: extract markers (data points) from a figure after digitize_setup. Filters: "
            "color (e.g. 'red' or '#1f77b4'), shape (circle|square|triangle_up|triangle_down|diamond|plus|cross), "
            "filled true/false, group (appearance group from a previous run), or template_index (copy the look of "
            "detection #i: finds any marker shape incl. markers on lines). Legend markers are excluded. Touching / "
            "overlapping markers are split (open markers of several shapes overlapping in one colour: outline "
            "fitting, picked automatically); blobs that cannot be split are flagged ('may be N+ overlapping "
            "markers', CSV column note). Writes CSV + an overlay with numbered detections - LOOK at it and re-run "
            "with other filters/exclude_regions if points are missed or spurious - and updates the provenance "
            "figure (all series of this image, values drawn on the original) to SHOW the user. Composition "
            "triangle: axes={ternary: {corners: {c0: EL, ...}}}.",
            {**src, "axes": {"type": "object", "properties": {
                "x": _AXIS_SPEC, "y": _AXIS_SPEC,
                "ternary": {"type": "object", "description": "triangle calibration: corners {c0: EL, c1: EL, c2: EL} | "
                            "corners_px {EL: [x, y]} | ticks {b2: {EL: frac, EL: frac}} | points [[px, py, {EL: frac}]]"}}},
             "color": S, "tolerance": {"type": "number", "description": "colour distance (Lab), default 35"},
             "shape": S, "filled": {"type": "boolean"}, "group": {"type": "integer"},
             "template_index": {"type": "integer"},
             "mode": {"type": "string", "enum": ["auto", "components", "template", "outline"],
                      "description": "auto: template matching from the cleanest example when shape/group is given; "
                                     "open (hollow) markers that overlap -> outline: outlines of all shapes fitted "
                                     "jointly to the colour's ink (dense bands of mixed shapes), then filtered by "
                                     "shape"},
             "min_score": N, "marker_size_px": N,
             "exclude_regions": {"type": "array", "items": {"type": "array", "items": N},
                                 "description": "[[l,t,r,b] px of the working image], e.g. an inset"},
             "errorbars": {"type": "boolean"}, "keep_legend": {"type": "boolean"},
             "series": {**S, "description": "name for this series (CSV column)"}, "output_csv": S},
            ["image", "axes"])
    def digitize_points(image, axes, page=None, crop=None, dpi=None, color=None, tolerance: float = 35.0,
                        shape=None, filled=None, group=None, template_index=None, mode: str = "auto",
                        min_score=None, marker_size_px=None, exclude_regions=(), errorbars=False, keep_legend=False,
                        series=None, output_csv=None):
        import csv
        work = _work(image, page, crop, dpi)
        tern = (axes or {}).get("ternary")
        st = _setup(work, True if tern else None)
        rgb, fr, ticks, tri = st["rgb"], st["frame"], st["ticks"], st.get("tri")
        rgb_det = st.get("rgb_det", rgb)
        if tern:
            tcal = calibrate_triangle(tern, tri)
            calx = caly = None
        else:
            if tri is not None and not (axes or {}).get("x"):
                raise ValueError("this figure is a composition triangle: calibrate with axes={\"ternary\": "
                                 "{\"corners\": {\"c0\": EL, \"c1\": EL, \"c2\": EL}}}")
            calx, caly = _cal(axes, ticks, fr)
            tcal = None
        markers, info = detect_markers(rgb_det, fr, color=color, tol=tolerance, marker_size=marker_size_px,
                                       exclude=exclude_regions or (), errorbars=errorbars, keep_legend=keep_legend)
        def outline_geo():
            if "inks" not in st:
                st["inks"] = ink_colours(rgb_det, _inner_region(rgb_det.shape[:2], fr, pad=fr.thickness + 1),
                                         background_color(rgb_det, fr))
            key = ("outline_geo", marker_size_px)
            if key not in st:          # size / edge width from all ink (for colours without isolated markers)
                reg0 = _inner_region(rgb_det.shape[:2], fr)
                w_all, _ = ink_weight(rgb_det, reg0, background_color(rgb_det, fr), inks=st["inks"])
                st[key] = estimate_outline_geometry(w_all, reg0, size=marker_size_px)
            return st[key]
        outline = mode == "outline"
        if mode == "auto" and template_index is None and group is None and filled is not True and not errorbars:
            ms_ = [m for m in markers if not m.in_legend]
            if _mostly_open(markers, shape):
                g0 = outline_geo()
                outline = _prefer_outline(markers, (g0["m"] + g0["w"]) if g0 else None)
            elif ms_ and len(ms_) <= 5:
                # a few tiny 'filled' bits of this colour in a figure of open markers: fragments of overlapping
                # outlines (e.g. three markers clipped by the frame), not filled markers
                g0 = outline_geo()
                outline = bool(g0 and (g0.get("dice") or 0) >= 0.8 and g0.get("n_examples", 0) >= 3
                               and float(np.median([m.size_px for m in ms_])) < 0.6 * (g0["m"] + g0["w"]))
        if outline:
            if filled:
                raise ValueError("mode 'outline' finds open (hollow) markers only; drop filled=true")
            g0 = outline_geo()
            legend_boxes = [tuple(b) for b in info.get("legend_boxes", [])] if not keep_legend else []
            key = ("outline", color, float(tolerance), marker_size_px, json.dumps(exclude_regions or [], default=str),
                   bool(keep_legend))
            if key not in st:
                st[key] = detect_open_markers(rgb_det, fr, color=color, tol=tolerance, marker_size=marker_size_px,
                                              exclude=exclude_regions or (), legend=legend_boxes,
                                              geometry=g0 or None, inks=st["inks"])
            om, oinfo = st[key]
            if shape and not any(shape_family(s) == shape_family(shape) for s in oinfo.get("outline_shapes", [])):
                # the requested shape never fitted confidently in this colour: fit again keeping it in play
                om, oinfo = detect_open_markers(rgb_det, fr, color=color, tol=tolerance, shape=shape,
                                                marker_size=marker_size_px, exclude=exclude_regions or (),
                                                legend=legend_boxes, geometry=g0 or None, inks=st["inks"])
            om = [Marker(**{**m.__dict__}) for m in om]          # the cached detections stay untouched
            if om or mode == "outline":
                markers = om
                info = {**info, **oinfo, "groups": assign_groups(om, use_color=color is None),
                        "legend_markers_excluded": 0}
                template_index = None
            else:
                outline = False
        if not outline and template_index is None and mode != "components" and (shape or group is not None):
            # pick the cleanest isolated example of the requested class and template-match it (robust for
            # markers on lines, overlapping symbols and black-and-white figures)
            cls = [i for i, m in enumerate(markers) if not m.in_legend and not m.note     # never a merged blob
                   and (shape is None or m.shape == shape) and (filled is None or m.filled == bool(filled))
                   and (group is None or m.group == group)]
            def isolated(i, f):
                return all(math.hypot(markers[i].cx - o.cx, markers[i].cy - o.cy) > f * markers[i].size_px
                           for o in markers if o is not markers[i])
            # examples whose neighbours stay outside the correlation window; an overlap only adds area, so the
            # smallest of them (a hidden second marker must not become the template), the most isolated first
            iso = [i for i in cls if isolated(i, 1.3)]
            if iso:
                a_ref = min(markers[i].area for i in iso)
                iso = [i for i in iso if markers[i].area <= 1.05 * a_ref]
                iso = [i for i in iso if isolated(i, 2.5)] or iso
            if iso:
                template_index = pick_example(rgb_det, fr, markers, iso)
            elif mode == "template":
                raise ValueError("no isolated example of that marker class to use as a template; give template_index"
                                 " (or mode 'outline' for open markers)")
        mode_used = "components"
        if outline:
            g = info.get("outline_geometry", {})
            mode_used = (f"outline fit: open {', '.join(sorted({shape_family(s) for s in info.get('outline_shapes', [])}))}"
                         f", size {g.get('m', '?')} px, edge {g.get('w', '?')} px")
        if template_index is not None:
            if not (0 <= template_index < len(markers)):
                raise ValueError(f"template_index {template_index} out of range (0..{len(markers) - 1})")
            ex = markers[template_index]
            markers = template_markers(rgb_det, fr, (ex.cx, ex.cy), ex.size_px / 2, min_score=min_score or 0.65,
                                       shape=ex.shape,
                                       exclude=list(exclude_regions or ()) +
                                       [tuple(b) for b in info.get("legend_boxes", [])] if not keep_legend
                                       else list(exclude_regions or ()), color=color, tol=tolerance)
            ex_shape, ex_filled = ex.shape, ex.filled
            for m in markers:
                m.shape, m.filled = ex_shape, ex_filled
            mode_used = f"template from #{template_index} ({'filled' if ex_filled else 'open'} {ex_shape})"
        mode = mode_used
        kept = []
        for m in markers:
            if m.in_legend and not keep_legend:
                continue
            if outline:
                if shape and shape_family(m.shape) != shape_family(shape):
                    continue
                if min_score is not None and m.score < min_score:
                    continue
            elif mode == "components":
                if shape and m.shape != shape:
                    continue
                if filled is not None and m.filled != bool(filled):
                    continue
                if group is not None and m.group != group:
                    continue
                if min_score is not None and m.score < min_score:
                    continue
            if tri is not None and not tri.inside(m.cx, m.cy, margin=m.size_px):
                continue
            kept.append(m)
        if tcal is not None:
            fr_ = tcal.to_fractions([m.cx for m in kept], [m.cy for m in kept]) if kept else np.zeros((0, 3))
            xs, ys = fr_[:, 1], fr_[:, 2]
        else:
            xs = calx.to_value([m.cx for m in kept])
            ys = caly.to_value([m.cy for m in kept])
        name = series or (color or shape or (f"group{group}" if group is not None else "points"))
        out_csv = tb.outp(output_csv, f"{Path(work).stem}_{str(name).replace('#', '')}_points.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            head = ["index", "series", "x", "y"]
            if tcal is not None:
                head += [f"X({e})" for e in tcal.elements]
            head += ["x_px", "y_px", "shape", "filled", "color", "size_px", "score"]
            if errorbars and tcal is None:
                head += ["y_err_up", "y_err_down"]
            head += ["note"]
            w.writerow(head)
            for i, (m, x, y) in enumerate(zip(kept, xs, ys)):
                row = [i, name, f"{x:.6g}", f"{y:.6g}"]
                if tcal is not None:
                    row += [f"{v:.6g}" for v in fr_[i]]
                row += [round(m.cx, 2), round(m.cy, 2), m.shape, m.filled, m.color, round(m.size_px, 1), m.score]
                if errorbars and tcal is None:
                    up = (caly.to_value(m.cy - m.err_up_px) - y) if m.err_up_px else ""
                    dn = (y - caly.to_value(m.cy + m.err_down_px)) if m.err_down_px else ""
                    row += [f"{up:.4g}" if up != "" else "", f"{dn:.4g}" if dn != "" else ""]
                row += [m.note]
                w.writerow(row)
        ov = str(Path(out_csv).with_suffix(".overlay.png"))
        _overlay(rgb, fr, ticks=ticks, markers=kept + [m for m in markers if m.in_legend],
                 legend=info.get("legend_boxes"), out_png=ov, tri=tri,
                 title=f"{len(kept)} points ({mode}); gray = legend (excluded); red dashed ? = check (overlap)")
        # provenance: the written VALUES mapped back through the calibration
        if tcal is not None:
            ppx = tcal.to_px(fr_) if len(kept) else np.zeros((0, 2))
            checks, cal_desc = ternary_checks(tcal), ternary_description(tcal)
        else:
            ppx = np.column_stack([calx.to_px(xs), caly.to_px(ys)]) if len(kept) else np.zeros((0, 2))
            checks = axis_checks(calx, caly, ticks, fr)
            cal_desc = (axes_description((axes or {}).get("x"), calx, "x") + "; " +
                        axes_description((axes or {}).get("y"), caly, "y"))
        flagged = [is_flagged(m) or m.score < 0.5 for m in kept]
        ink = "#%02x%02x%02x" % tuple(int(v) for v in np.median(
            [[int(m.color[i:i + 2], 16) for i in (1, 3, 5)] for m in kept], axis=0)) if kept else None
        prov_png = _provenance(work, image, page, crop, st, "digitize_points", ("points", str(name)),
                               {"name": str(name), "kind": "points", "px": ppx, "flagged": flagged, "colour": ink,
                                "size_px": float(np.median([m.size_px for m in kept])) if kept else 8.0,
                                "csv": tb.rel(out_csv)}, cal_desc, checks)
        shapes: Dict[str, int] = {}
        for m in kept:
            shapes[f"{'filled' if m.filled else 'open'} {m.shape}"] = shapes.get(f"{'filled' if m.filled else 'open'} {m.shape}", 0) + 1
        if tcal is not None:
            e = tcal.elements
            cal_line = (f"ternary calibration from {tcal.n_points} known points (max residual {tcal.residual:.4f} "
                        f"in fraction); x = X({e[1]}), y = X({e[2]}); columns X({e[0]}), X({e[1]}), X({e[2]})")
        else:
            cal_line = (f"calibration: x residual {calx.residual_px:.2f} px, y residual {caly.residual_px:.2f} px "
                        f"({calx.scale}/{caly.scale}); 1 px = {abs(calx.b):.4g} x-units, {abs(caly.b):.4g} y-units"
                        + ("" if calx.scale == "linear" else " (log10)"))
        res = [f"{len(kept)} points ({mode}) -> {tb.rel(out_csv)}", cal_line,
               f"marker diameter ~{info.get('marker_diameter_px', '?')} px; shapes: {shapes}",
               "appearance groups (all detections): " + "; ".join(
                   f"group {g['group']}: {g['n']} x {g['shape']} {g['color']}" + (" (incl. legend entry)" if g['n_legend'] else "")
                   for g in info.get("groups", [])[:8]),
               f"legend markers excluded: {info.get('legend_markers_excluded', 0)}; oversized blobs ignored: "
               f"{info.get('rejected_blobs', 0)}" if not outline else
               "outline fit, markers per shape in this colour (all shapes, before the shape filter): "
               + ", ".join(f"{shape_family(k)} {v}" for k, v in info.get("shape_counts", {}).items())
               + (f"; {info['residual_spots']} spot(s) of this colour's ink explained by no marker (other shapes? "
                  "check the overlay)" if info.get("residual_spots") else ""),
               "first points: " + "; ".join(f"#{i} ({x:.4g}, {y:.4g})" for i, (x, y) in enumerate(zip(xs[:8], ys[:8]))),
               "Check the overlay; low-score detections are orange."]
        fl = [(i, m) for i, m in enumerate(kept) if is_flagged(m)]
        if fl:
            by_note: Dict[str, List[int]] = {}
            for i, m in fl:
                by_note.setdefault(m.note.split(";")[0], []).append(i)
            res.append("CHECK: " + "; ".join(f"{', '.join('#%d' % i for i in ix)} {nt}" for nt, ix in by_note.items())
                       + " - look at the overlay (red dashed rings) and the provenance figure; the CSV 'note' column "
                         "has the same flags.")
        sp = [i for i, m in enumerate(kept) if m.note.startswith("split")]
        if sp:
            res.append(f"split touching/overlapping markers: {', '.join('#%d' % i for i in sp)} (blue rings; "
                       f"centres fitted with a one-marker template)")
        res.append(f"provenance figure (all series from this image so far, values mapped back onto the original): "
                   f"{tb.rel(prov_png)} - SHOW it to the user so they can check the points at a glance.")
        if tcal is not None and tcal.residual > 0.01:
            res.append("WARNING: ternary calibration residual > 0.01 - a corner/tick composition may be wrong.")
        if tcal is None and (calx.residual_px > 2 or caly.residual_px > 2):
            res.append("WARNING: calibration residual > 2 px - a tick index/value pair may be wrong.")
        _save_run(out_csv, {"image": image, "page": page, "crop": crop, "dpi": dpi, "axes": axes, "color": color,
                            "tolerance": tolerance, "series": str(name)})
        images = [ov, prov_png]
        sheet = _check_sheet_for(out_csv, st, str(name))
        if sheet:
            images.append(sheet)
            res.append(f"CHECK SHEET (enlarged crop of every doubtful detection): {tb.rel(sheet)} - show it to the "
                       f"user and ask how many points each numbered blob is (e.g. '#7: 3, #13: 2, #23: not a point'), "
                       f"then digitize_fix csv={tb.rel(out_csv)} counts={{...}}.")
        return "\n".join(res), images

    # ------------------------------------------------------------------ human corrections
    def _save_run(out_csv, spec):
        import json
        Path(str(out_csv) + ".run.json").write_text(json.dumps(spec, default=str, indent=1))

    def _load_run(csv_path):
        import json
        p = Path(str(csv_path) + ".run.json")
        if not p.exists():
            raise ValueError(f"{Path(csv_path).name} was not written by digitize_points in this version (no "
                             f"{p.name}); re-run digitize_points first")
        return json.loads(p.read_text())

    def _read_rows(csv_path):
        import csv
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))

    def _check_sheet_for(out_csv, st, name):
        from .digitize_fix import check_sheet, doubtful
        rows = _read_rows(out_csv)
        items = doubtful(rows)
        if not items:
            return None
        conv = lambda r: {"index": int(r["index"]), "cx": float(r["x_px"]), "cy": float(r["y_px"]),  # noqa: E731
                          "size": float(r.get("size_px") or 10), "why": r.get("why", "")}
        return check_sheet(st["rgb"], [conv(r) for r in items], [conv(r) for r in rows],
                           str(Path(out_csv).with_suffix(".check.png")),
                           title=f"{name}: doubtful detections")

    @tb.add("digitize_fix", "Correct a digitize_points result with the user's answers (after showing the check "
            "sheet): counts {'7': 3, '13': 2, '23': 0} refits each numbered blob with exactly that many markers "
            "(0 = not a point; other detections inside the same blob are replaced); add [[x, y], ...] data values "
            "(snapped to a marker nearby); remove [indices]. Rewrites the CSV (renumbered) and the overlay / "
            "provenance figure.",
            {"csv": {"type": "string", "description": "the CSV written by digitize_points"},
             "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
             "add": {"type": "array", "items": {"type": "array", "items": N},
                     "description": "[[x, y]] in axis values (ternary: [X(el1), X(el2)] as in the CSV x, y)"},
             "remove": {"type": "array", "items": {"type": "integer"}}}, ["csv"])
    def digitize_fix(csv, counts=None, add=None, remove=None):
        import csv as _csv
        from .digitize_fix import fill_mask, component_at, learn_template, refit_blob, snap_point
        path = tb.p(csv)
        spec = _load_run(path)
        rows = _read_rows(path)
        work = _work(spec["image"], spec.get("page"), spec.get("crop"), spec.get("dpi"))
        axes = spec.get("axes") or {}
        tern = axes.get("ternary")
        st = _setup(work, True if tern else None)
        rgb, fr, ticks, tri = st["rgb"], st["frame"], st["ticks"], st.get("tri")
        rgb_det = st.get("rgb_det", rgb)
        if tern:
            tcal, calx, caly = calibrate_triangle(tern, tri), None, None
        else:
            calx, caly = _cal(axes, ticks, fr)
            tcal = None
        bg = background_color(rgb_det, fr)
        color = spec.get("color")
        rivals = rival_colors(rgb_det, fr, bg) if color is not None else None
        fg, _ = _foreground(rgb_det, bg, color=parse_color(color) if color is not None else None,
                            tol=float(spec.get("tolerance") or 35.0), rivals=rivals)
        size = float(np.median([float(r.get("size_px") or 10) for r in rows])) if rows else 10.0
        # as the detector does: only the inside of the frame, straight lines (frame, gridlines, tie-lines) out
        H_, W_ = fg.shape
        pad = int(math.ceil(fr.thickness / 2 + 1))
        region = np.zeros((H_, W_), bool)
        region[int(fr.top) + pad:int(math.ceil(fr.bottom)) - pad + 1,
               int(fr.left) + pad:int(math.ceil(fr.right)) - pad + 1] = True
        nolines, _ = _remove_straight_lines(fg & region, max(15, int(size * 2.2)))
        mask = fill_mask(nolines | (ndi.binary_closing(nolines, structure=_disk(2.0)) & fg & region))
        clean = [(float(r["x_px"]), float(r["y_px"])) for r in rows
                 if not r.get("note") and float(r.get("score") or 0) >= 0.7]
        T, tsrc = learn_template(mask, clean, size)
        by_idx = {int(r["index"]): r for r in rows}
        counts = {int(str(k).lstrip("#")): int(v) for k, v in (counts or {}).items()}
        for i in remove or []:
            counts[int(i)] = 0
        bad = [i for i in counts if i not in by_idx]
        if bad:
            raise ValueError(f"no detection {bad} in {Path(path).name} (indices 0..{len(rows) - 1})")
        drop, new, log = set(), [], []
        blobs = {}
        for i, k in sorted(counts.items()):
            r = by_idx[i]
            cx, cy = float(r["x_px"]), float(r["y_px"])
            comp = component_at(mask, cx, cy, 0.6 * size)
            if comp is None:
                drop.add(i)
                log.append(f"#{i}: no ink found there - removed" if k == 0 else f"#{i}: no ink found - skipped")
                continue
            if comp.sum() > 12 * max(1, int(T.sum())):     # not a marker blob (a line / region left over)
                log.append(f"#{i}: the ink there is not a marker-sized blob - left unchanged")
                continue
            key = int(np.flatnonzero(comp)[0])
            members = [j for j, q in by_idx.items() if comp[int(round(float(q["y_px"]))) % comp.shape[0],
                                                             int(round(float(q["x_px"]))) % comp.shape[1]]]
            drop.update(members + [i])
            if key in blobs:                          # two numbers in one blob: the larger count wins
                if k <= blobs[key][0]:
                    continue
            blobs[key] = (k, i, comp, members)
        for key, (k, i, comp, members) in blobs.items():
            cen, info = refit_blob(comp, T, k)
            others = sorted(set(members) - {i})
            log.append(f"#{i}: {k} point(s)" + (f" (replaces #{', #'.join(map(str, others))} in the same blob)"
                                               if others else ""))
            base = by_idx[i]
            for (px, py) in cen:
                new.append({**base, "x_px": px, "y_px": py, "score": 0.9,
                            "note": f"count {k} confirmed by user (blob of #{i})"})
        for a in add or []:
            if tcal is not None:
                fr3 = np.array([[1 - a[0] - a[1], a[0], a[1]]])
                px, py = [float(v) for v in tcal.to_px(fr3)[0]]
            else:
                px, py = float(calx.to_px([a[0]])[0]), float(caly.to_px([a[1]])[0])
            sx, sy, ok = snap_point(mask, T, px, py, 0.8 * size)
            new.append({**(rows[0] if rows else {}), "x_px": sx, "y_px": sy, "score": 0.9 if ok else 0.6,
                        "note": "added by user" + (" (snapped to a marker)" if ok else " (at the given position)")})
            log.append(f"added a point near ({a[0]:g}, {a[1]:g})" + (" - snapped to a marker" if ok else
                                                                   " - nothing marker-like there, kept as given"))
        keep = [r for r in rows if int(r["index"]) not in drop] + new
        keep.sort(key=lambda r: (float(r["x_px"]), float(r["y_px"])))
        pxs = np.array([[float(r["x_px"]), float(r["y_px"])] for r in keep]) if keep else np.zeros((0, 2))
        if tcal is not None:
            fr_ = tcal.to_fractions(pxs[:, 0], pxs[:, 1]) if len(keep) else np.zeros((0, 3))
            xs, ys = fr_[:, 1], fr_[:, 2]
        else:
            xs = calx.to_value(pxs[:, 0]) if len(keep) else np.array([])
            ys = caly.to_value(pxs[:, 1]) if len(keep) else np.array([])
        head = list(rows[0].keys()) if rows else ["index", "series", "x", "y", "x_px", "y_px", "note"]
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
            w.writeheader()
            for n, (r, x, y) in enumerate(zip(keep, xs, ys)):
                r = dict(r)
                r.update(index=n, x=f"{x:.6g}", y=f"{y:.6g}", x_px=round(float(r["x_px"]), 2),
                         y_px=round(float(r["y_px"]), 2))
                if tcal is not None:
                    for e, v in zip(tcal.elements, fr_[n]):
                        r[f"X({e})"] = f"{v:.6g}"
                w.writerow(r)
        rows2 = _read_rows(path)
        mk = [Marker(cx=float(r["x_px"]), cy=float(r["y_px"]), area=0.0, size_px=float(r.get("size_px") or size),
                     shape=r.get("shape") or "circle", filled=str(r.get("filled")) == "True",
                     color=r.get("color") or "#000000", score=float(r.get("score") or 0.9), note=r.get("note") or "")
              for r in rows2]
        ov = str(Path(path).with_suffix(".overlay.png"))
        _overlay(rgb, fr, ticks=ticks, markers=mk, out_png=ov, tri=tri,
                 title=f"{len(mk)} points after corrections; red dashed ? = still to check")
        name = spec.get("series") or Path(path).stem
        if tcal is not None:
            ppx, checks, cal_desc = pxs, ternary_checks(tcal), ternary_description(tcal)
        else:
            ppx = pxs
            checks = axis_checks(calx, caly, ticks, fr)
            cal_desc = axes_description(axes.get("x"), calx, "x") + "; " + axes_description(axes.get("y"), caly, "y")
        prov_png = _provenance(work, spec["image"], spec.get("page"), spec.get("crop"), st, "digitize_fix",
                               ("points", str(name)),
                               {"name": str(name), "kind": "points", "px": ppx,
                                "flagged": [is_flagged(m) for m in mk], "colour": mk[0].color if mk else None,
                                "size_px": size, "csv": tb.rel(path)}, cal_desc, checks)
        # a fresh check sheet showing the corrected blobs, for confirmation
        from .digitize_fix import check_sheet
        fixed = [r for r in rows2 if "confirmed by user" in (r.get("note") or "") or "added by user" in
                 (r.get("note") or "")]
        conv = lambda r: {"index": int(r["index"]), "cx": float(r["x_px"]), "cy": float(r["y_px"]),  # noqa: E731
                          "size": float(r.get("size_px") or size), "why": (r.get("note") or "")[:40]}
        sheet = check_sheet(rgb, [conv(r) for r in fixed], [conv(r) for r in rows2],
                            str(Path(path).with_suffix(".fixed.png")),
                            title=f"{name}: corrected blobs - check the red + sit on the markers", question=False)
        still = [r["index"] for r in rows2 if is_flagged(r.get("note") or "")]
        txt = [f"{len(rows2)} points -> {tb.rel(path)} (renumbered; template {tsrc})", *log,
               f"provenance figure updated: {tb.rel(prov_png)}"]
        if sheet:
            txt.append(f"corrected blobs, enlarged: {tb.rel(sheet)} - show it to the user to confirm")
        if still:
            txt.append(f"still flagged: #{', #'.join(map(str, still))}")
        return "\n".join(txt), [p for p in (sheet, prov_png) if p]

    @tb.add("digitize_curve", "Step 2 (lines): trace a plotted line of a given colour after digitize_setup and "
            "sample it (e.g. a liquidus curve). direction 'x' (default) scans columns; use 'y' for steep curves. "
            "Optional x_values to interpolate onto; start_at [x, y] data values to pick one of several curves.",
            {**src, "axes": {"type": "object", "properties": {"x": _AXIS_SPEC, "y": _AXIS_SPEC}}, "color": S,
             "tolerance": N, "direction": {"type": "string", "enum": ["x", "y"]}, "step_px": {"type": "integer"},
             "x_values": {"type": "array", "items": N}, "start_at": {"type": "array", "items": N},
             "series": S, "output_csv": S}, ["image", "axes", "color"])
    def digitize_curve(image, axes, color, page=None, crop=None, dpi=None, tolerance: float = 35.0,
                       direction: str = "x", step_px: int = 2, x_values=None, start_at=None, series=None,
                       output_csv=None):
        import csv
        work = _work(image, page, crop, dpi)
        st = _setup(work)
        rgb, fr, ticks = st["rgb"], st["frame"], st["ticks"]
        calx, caly = _cal(axes, ticks, fr)
        start_px = None
        if start_at:
            start_px = (float(calx.to_px(start_at[0])), float(caly.to_px(start_at[1])))
        pts = trace_curve(rgb, fr, color, tol=tolerance, step_px=step_px, direction=direction, start_px=start_px)
        if len(pts) < 2:
            raise ValueError(f"no line of colour {color!r} found (try a larger tolerance or check digitize_setup colours)")
        px = np.array(pts)
        xs, ys = calx.to_value(px[:, 0]), caly.to_value(px[:, 1])
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if x_values:
            xv = np.asarray(x_values, dtype=float)
            inside = (xv >= xs.min()) & (xv <= xs.max())
            ys = np.interp(xv[inside], xs, ys)
            xs = xv[inside]
        name = series or color
        out_csv = tb.outp(output_csv, f"{Path(work).stem}_{str(name).replace('#', '')}_curve.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["series", "x", "y"])
            for x, y in zip(xs, ys):
                w.writerow([name, f"{x:.6g}", f"{y:.6g}"])
        ov = str(Path(out_csv).with_suffix(".overlay.png"))
        _overlay(rgb, fr, ticks=ticks, curve=pts, out_png=ov, title=f"traced {len(pts)} px samples")
        checks = axis_checks(calx, caly, ticks, fr)
        cal_desc = (axes_description((axes or {}).get("x"), calx, "x") + "; " +
                    axes_description((axes or {}).get("y"), caly, "y"))
        prov_png = _provenance(work, image, page, crop, st, "digitize_curve", ("curve", str(name)),
                               {"name": str(name), "kind": "curve",
                                "px": np.column_stack([calx.to_px(xs), caly.to_px(ys)]),
                                "colour": "#%02x%02x%02x" % tuple(int(v * 255) for v in parse_color(color)),
                                "csv": tb.rel(out_csv)}, cal_desc, checks)
        return (f"{len(xs)} curve points -> {tb.rel(out_csv)}; x {xs.min():.4g}..{xs.max():.4g}, "
                f"y {ys.min():.4g}..{ys.max():.4g}. Check the overlay (magenta trace).\n"
                f"provenance figure (all series from this image so far, values mapped back onto the original): "
                f"{tb.rel(prov_png)} - SHOW it to the user so they can check the data at a glance."), [ov, prov_png]
