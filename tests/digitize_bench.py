"""Ground-truth benchmark for the plot digitiser: synthetic figures with known points.

    .venv/bin/python tests/digitize_bench.py [outdir]
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calphad_tools import digitize as D


def truth_ticks(ax, fig, axis):
    """What the model would read: values of the major ticks drawn inside the axis limits."""
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    locs = ax.get_xticks() if axis == "x" else ax.get_yticks()
    return [v for v in locs if min(lo, hi) - 1e-9 <= v <= max(lo, hi) + 1e-9]


def make_case(name, out, seed=0, series=None, dpi=100, jpeg=None, blur=0.0, grid=False, legend=True,
              curves=False, errorbars=False, touching=0, logy=False, figsize=(6.4, 4.8), bw=False,
              rotate=0.0, noise=0.0, paper=None, by_shape=False, prepare=False):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)
    truth = {}
    series = series or [("C0", "o", True, 25)]
    for k, (col, mk, filled, n) in enumerate(series):
        x = rng.uniform(0.05, 0.95, n)
        y = rng.uniform(0.1, 0.9, n) if not logy else 10 ** rng.uniform(-2.5, 1.5, n)
        if touching and k == 0:
            x = np.concatenate([x, x[:touching] + 0.012])
            y = np.concatenate([y, y[:touching] * (1.0 if logy else 1) + (0 if logy else 0.01)])
        kw = dict(marker=mk, linestyle="none", markersize=7, label=f"series {k}",
                  color=("k" if bw else col))
        if not filled:
            kw.update(markerfacecolor="none", markeredgewidth=1.2)
        if errorbars and k == 0:
            ax.errorbar(x, y, yerr=0.04 if not logy else y * 0.2, capsize=0, **kw)
        else:
            ax.plot(x, y, **kw)
        truth[f"series{k}"] = {"color": ("k" if bw else col), "marker": mk, "filled": filled,
                               "x": x.tolist(), "y": y.tolist()}
    if curves:
        t = np.linspace(0, 1, 200)
        ax.plot(t, 0.5 + 0.3 * np.sin(6 * t), color="0.3", lw=1.2)
        ax.plot(t, 0.2 + 0.6 * t ** 2, color="C2" if not bw else "k", lw=1.0)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e2)
    else:
        ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    if grid:
        ax.grid(True, alpha=0.6)
    if legend:
        ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png")
    xt, yt = truth_ticks(ax, fig, "x"), truth_ticks(ax, fig, "y")
    plt.close(fig)
    from PIL import Image, ImageFilter
    im = Image.open(buf).convert("RGB")
    if paper is not None:           # off-white paper: multiply
        arr = np.asarray(im, dtype=float) * np.array(paper)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC, expand=False, fillcolor=tuple(int(255 * v) for v in (paper or (1, 1, 1))))
    if noise:
        arr = np.asarray(im, dtype=float) + np.random.default_rng(seed).normal(0, noise * 255, (im.size[1], im.size[0], 3))
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    path = Path(out) / f"{name}.{'jpg' if jpeg else 'png'}"
    if jpeg:
        im.save(path, quality=jpeg)
    else:
        im.save(path)
    return str(path), truth, xt, yt


SHAPES = {"o": "circle", "s": "square", "^": "triangle_up", "v": "triangle_down", "D": "diamond"}


def evaluate(path, truth, xt, yt, logy=False, use_color=True, by_shape=False, prepare=False):
    if prepare:                     # the tool's own loading path (deskew etc.)
        path = D.prepare_image(path, Path(path).parent / "prepared")
    rgb = D.load_rgb(path)
    fr = D.detect_frame(rgb)
    ticks = D.detect_ticks(rgb, fr)
    nx = sum(1 for t in ticks["x"] if "index" in t)
    ny = sum(1 for t in ticks["y"] if "index" in t)
    res = {"ticks_ok": nx == len(xt) and ny == len(yt), "nx": nx, "ny": ny, "nx_true": len(xt), "ny_true": len(yt)}
    # the "model" maps the first and last numbered ticks to the printed values
    ax_spec = {"x": {"ticks": {0: xt[0], nx - 1: xt[-1]} if res["ticks_ok"] else {}, "scale": "linear"},
               "y": {"ticks": {0: yt[-1], ny - 1: yt[0]} if res["ticks_ok"] else {}, "scale": "log" if logy else "linear"}}
    if not res["ticks_ok"]:
        return res
    calx = D.calibrate_axis(ax_spec["x"], ticks["x"], fr.left, fr.right, "x")
    caly = D.calibrate_axis(ax_spec["y"], ticks["y"], fr.top, fr.bottom, "y")
    per = {}
    for name, s in truth.items():
        col = s["color"] if use_color else None
        mk, info = D.detect_markers(rgb, fr, color=col)
        mk = [m for m in mk if not m.in_legend]
        if by_shape:
            per_shape = [m for m in mk if m.shape == SHAPES[s["marker"]] and m.filled == s["filled"]]
            mk = per_shape

        px = np.array([[m.cx, m.cy] for m in mk]) if mk else np.zeros((0, 2))
        dx = calx.to_value(px[:, 0]) if len(px) else np.array([])
        dy = caly.to_value(px[:, 1]) if len(px) else np.array([])
        tx, ty = np.array(s["x"]), np.array(s["y"])
        # match in pixel space (tolerance = 0.6 marker diameter)
        tpx = np.column_stack([calx.to_px(tx), caly.to_px(ty)])
        tol = 0.6 * (info.get("marker_diameter_px") or 8)
        used, errs, tp = set(), [], 0
        for i, p in enumerate(tpx):
            if not len(px):
                break
            d = np.hypot(*(px - p).T)
            j = int(np.argmin(d))
            if d[j] <= tol and j not in used:
                used.add(j)
                tp += 1
                errs.append(d[j])
        per[name] = {"true": len(tx), "found": len(mk), "tp": tp, "recall": tp / len(tx),
                     "precision": tp / max(1, len(mk)), "err_px_rms": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                     "err_pct_x_range": float(np.sqrt(np.mean(np.square(errs))) * abs(calx.b) * 100) if errs and not logy else None}
    res["series"] = per
    res["cal_resid_px"] = (calx.residual_px, caly.residual_px)
    return res


CASES = {
    "basic": dict(series=[("C0", "o", True, 30)], legend=False),
    "three_series_legend_grid": dict(series=[("red", "o", True, 20), ("blue", "s", False, 20), ("green", "^", True, 20)],
                                     grid=True),
    "curves_touching_errorbars": dict(series=[("C3", "o", True, 25)], curves=True, touching=4, errorbars=True),
    "jpeg_blur_150dpi": dict(series=[("C0", "o", True, 20), ("C1", "D", True, 20)], dpi=150, jpeg=60, blur=0.6,
                             grid=True),
    "log_y": dict(series=[("purple", "v", True, 25)], logy=True),
    "open_markers": dict(series=[("black", "o", False, 25), ("C0", "s", False, 20)], curves=True),
    "small_low_dpi": dict(series=[("C0", "o", True, 30)], dpi=72, figsize=(4, 3)),
    "bw_shapes_on_lines": dict(series=[("k", "o", True, 15), ("k", "s", False, 15), ("k", "^", False, 15)],
                               bw=True, curves=True, by_shape=True),
    "scan_rotated_noisy": dict(series=[("C3", "o", True, 20), ("C0", "s", True, 20)], dpi=150, rotate=0.7,
                               noise=0.03, paper=(0.97, 0.95, 0.9), jpeg=50, blur=0.5, prepare=True),
}
EVAL_KEYS = ("by_shape", "prepare")

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_out")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, kw in CASES.items():
        path, truth, xt, yt = make_case(name, out, **kw)
        r = evaluate(path, truth, xt, yt, logy=kw.get("logy", False), use_color=not kw.get("by_shape"),
                     by_shape=kw.get("by_shape", False), prepare=kw.get("prepare", False))
        summary[name] = r
        print(name, json.dumps(r, default=lambda v: round(v, 3) if isinstance(v, float) else v))
