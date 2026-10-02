"""Plot digitiser: ground-truth benchmark + end-to-end tool workflows."""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import digitize_bench as B  # noqa: E402
from calphad_tools import digitize as D  # noqa: E402


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


# ---------------------------------------------------------------------------------- benchmark (synthetic truth)
def _visible(truth_pts_px, legend_boxes, pad):
    """Points not hidden under a legend box (legends are drawn opaque over the data)."""
    return [not any(b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad for b in legend_boxes)
            for x, y in truth_pts_px]


@pytest.mark.parametrize("case", [c for c in B.CASES if c != "bw_shapes_on_lines"])
def test_benchmark_colour_cases(case, tmp_path):
    kw = B.CASES[case]
    fig_kw = {k: v for k, v in kw.items() if k not in B.EVAL_KEYS}
    path, truth, xt, yt = B.make_case(case, tmp_path, **fig_kw)
    if kw.get("prepare"):
        path = D.prepare_image(path, tmp_path / "prep")
    rgb = D.load_rgb(path)
    fr = D.detect_frame(rgb)
    ticks = D.detect_ticks(rgb, fr)
    nx = sum(1 for t in ticks["x"] if "index" in t)
    ny = sum(1 for t in ticks["y"] if "index" in t)
    assert (nx, ny) == (len(xt), len(yt)), f"ticks {nx}/{len(xt)}, {ny}/{len(yt)}"
    calx = D.calibrate_axis({"ticks": {0: xt[0], nx - 1: xt[-1]}}, ticks["x"], fr.left, fr.right, "x")
    caly = D.calibrate_axis({"ticks": {0: yt[-1], ny - 1: yt[0]}, "scale": "log" if kw.get("logy") else "linear"},
                            ticks["y"], fr.top, fr.bottom, "y")
    for name, s in truth.items():
        mk, info = D.detect_markers(rgb, fr, color=s["color"])
        mk = [m for m in mk if not m.in_legend]
        tp = np.column_stack([calx.to_px(s["x"]), caly.to_px(s["y"])])
        vis = _visible(tp, info.get("legend_boxes", []), info["marker_diameter_px"])
        f = np.array([[m.cx, m.cy] for m in mk])
        d = np.array([np.min(np.hypot(*(f - p).T)) for p in tp])
        tol = 0.6 * info["marker_diameter_px"]
        recall = np.mean(d[vis] <= tol)
        matched = sum(np.min(np.hypot(*(tp - q).T)) <= tol for q in f)
        precision = matched / max(1, len(f))
        err = float(np.sqrt(np.mean(np.square(d[(d <= tol)]))))
        assert recall >= 0.95, (case, name, recall)
        assert precision >= 0.95, (case, name, precision)
        assert err <= 1.2, (case, name, err)


def test_benchmark_black_and_white_via_templates(tmp_path):
    kw = {k: v for k, v in B.CASES["bw_shapes_on_lines"].items() if k not in B.EVAL_KEYS}
    path, truth, xt, yt = B.make_case("bw", tmp_path, **kw)
    rgb = D.load_rgb(path)
    fr = D.detect_frame(rgb)
    ticks = D.detect_ticks(rgb, fr)
    calx = D.calibrate_axis({"ticks": {0: xt[0], 5: xt[-1]}}, ticks["x"], fr.left, fr.right, "x")
    caly = D.calibrate_axis({"ticks": {0: yt[-1], 5: yt[0]}}, ticks["y"], fr.top, fr.bottom, "y")
    mk, info = D.detect_markers(rgb, fr, color="black")
    legend = [tuple(b) for b in info["legend_boxes"]]
    assert len(legend) == 1
    for name, s in truth.items():
        want = (B.SHAPES[s["marker"]], s["filled"])
        cands = [m for m in mk if (m.shape, m.filled) == want and not m.in_legend and
                 all(math.hypot(m.cx - o.cx, m.cy - o.cy) > 2.5 * m.size_px for o in mk if o is not m)]
        assert cands, f"no clean {want} found"
        ex = mk[D.pick_example(rgb, fr, mk, [mk.index(c) for c in cands])]
        found = D.template_markers(rgb, fr, (ex.cx, ex.cy), ex.size_px / 2, min_score=0.65, exclude=legend,
                                   color="black", shape=ex.shape)
        f = np.array([[m.cx, m.cy] for m in found])
        tp = np.column_stack([calx.to_px(s["x"]), caly.to_px(s["y"])])
        d = np.array([np.min(np.hypot(*(f - p).T)) for p in tp])
        assert np.mean(d < 5) >= 0.85, (name, np.mean(d < 5))
        assert sum(np.min(np.hypot(*(tp - q).T)) < 5 for q in f) / len(f) >= 0.95
        assert np.sqrt(np.mean(np.square(d[d < 5]))) <= 1.0


def test_deskew_estimate(tmp_path):
    kw = {k: v for k, v in B.CASES["scan_rotated_noisy"].items() if k not in B.EVAL_KEYS}
    path, *_ = B.make_case("skew", tmp_path, **kw)
    assert abs(abs(D.estimate_skew(D.load_rgb(path))) - 0.7) < 0.1


# ---------------------------------------------------------------------------------- end-to-end tool workflows
def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _major_ticks(text, axis):
    return json.loads(re.search(rf"{axis}0\.\.{axis}\d+ at px (\[.*?\])", text).group(1))


def test_workflow_phase_diagram_with_experimental_points(tb, tmp_path):
    """Digitise coloured experimental points drawn on a real computed phase diagram (lines, tie-lines, legend)."""
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                             "tmax": 2000, "temperature_unit": "C", "style": "color"})
    stars = [(0.30, 1450), (0.45, 1300), (0.60, 1100), (0.75, 1500), (0.2, 950)]
    squares = [(0.10, 1700), (0.52, 1650), (0.85, 1250)]
    _, imgs2 = call(tb, "edit_figure", {"image": imgs[0], "output": "with_exp.png", "operations": [
        {"op": "scatter", "x": [p[0] for p in stars], "y": [p[1] for p in stars], "color": "crimson",
         "marker": "*", "s": 140, "label": "exp A", "zorder": 5},
        {"op": "scatter", "x": [p[0] for p in squares], "y": [p[1] for p in squares], "color": "navy",
         "marker": "s", "s": 50, "label": "exp B", "zorder": 5}]})
    # a "foreign" copy: flatten to JPEG so no editable figure / metadata can help
    from PIL import Image
    foreign = tmp_path / "paper_fig.jpg"
    Image.open(imgs2[0]).convert("RGB").save(foreign, quality=85)
    text, prev = call(tb, "digitize_setup", {"image": str(foreign)})
    assert Path(prev[0]).exists() and "major ticks" in text
    # the model reads the tick labels on the preview: x 0.0..1.0 in 0.2 steps; y 800..2000 in 200 steps
    xpx, ypx = _major_ticks(text, "x"), _major_ticks(text, "y")
    assert len(xpx) == 6 and len(ypx) == 7
    axes = {"x": {"ticks": {"0": 0.0, "5": 1.0}}, "y": {"ticks": {"0": 2000, str(len(ypx) - 1): 800}}}
    for col, truth in (("crimson", stars), ("navy", squares)):
        t, ov = call(tb, "digitize_points", {"image": str(foreign), "axes": axes, "color": col, "series": col})
        rows = _read_csv(re.search(r"-> (\S+\.csv)", t).group(1))
        got = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        assert len(got) == len(truth), (col, t)
        for x, y in truth:
            d = np.min(np.hypot((got[:, 0] - x) / 1.0, (got[:, 1] - y) / 1200.0))
            assert d < 0.006, (col, x, y, d)          # < 0.6 % of the axis ranges
        assert Path(ov[0]).exists()


def test_workflow_trace_liquidus(tb, tmp_path):
    """Trace the liquidus of a computed diagram and compare with the exact boundary data."""
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                "tmax": 2000, "temperature_unit": "C", "tielines": 0,
                                                "style": "color"})          # phase colours: trace the liquidus by colour
    info = json.loads(call(tb, "describe_figure", {"image": imgs[0]})[0])
    liq_color = info["axes"][0]["legend"]["LIQUID"]
    t, _ = call(tb, "digitize_setup", {"image": imgs[0]})
    ypx = _major_ticks(t, "y")
    axes = {"x": {"ticks": {"0": 0.0, "5": 1.0}}, "y": {"ticks": {"0": 2000, str(len(ypx) - 1): 800}}}
    t, _ = call(tb, "digitize_curve", {"image": imgs[0], "axes": axes, "color": liq_color,
                                       "x_values": [0.1, 0.2, 0.3, 0.6, 0.8, 0.9], "start_at": [0.1, 1800]})
    rows = _read_csv(re.search(r"-> (\S+\.csv)", t).group(1))
    got = {round(float(r["x"]), 2): float(r["y"]) for r in rows}
    # exact liquidus from the mapping data export
    bcsv = re.search(r"boundaries_csv: (\S+)", text).group(1)
    b = [r for r in _read_csv(bcsv) if r["phase"] == "LIQUID"]
    bx = np.array([float(r["X(NI)"]) for r in b])
    bt = np.array([float(r["T_C"]) for r in b])
    for x, y in got.items():
        near = np.abs(bx - x) < 0.01
        assert near.any()
        assert abs(np.median(bt[near]) - y) < 12, (x, y, np.median(bt[near]))    # within ~1 % of the T range


def test_workflow_pdf_page(tb, tmp_path):
    """A figure embedded in a PDF page with text: render page, loose crop, extract."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    xs, ys = rng.uniform(0.1, 0.9, 12), rng.uniform(900, 1500, 12)
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.1, 0.95, "Journal of Phase Equilibria (synthetic test page)", fontsize=14, weight="bold")
    for i in range(8):
        fig.text(0.1, 0.9 - 0.025 * i, "Lorem ipsum dolor sit amet, consectetur adipiscing elit " * 2, fontsize=8)
    ax = fig.add_axes([0.2, 0.35, 0.6, 0.35])
    ax.plot(xs, ys, "o", color="darkorange", markersize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(800, 1600)
    ax.set_xlabel("x(B)")
    ax.set_ylabel("T (K)")
    fig.text(0.1, 0.28, "Fig. 3. Solidus data.", fontsize=9)
    pdf = tmp_path / "paper.pdf"
    fig.savefig(pdf)
    xt = [v for v in ax.get_xticks() if 0 <= v <= 1]
    yt = [v for v in ax.get_yticks() if 800 <= v <= 1600]
    plt.close(fig)
    crop = [0.1, 0.25, 0.9, 0.7]                 # loose crop around the figure (fractions of the page)
    text, _ = call(tb, "digitize_setup", {"image": str(pdf), "page": 1, "crop": crop})
    assert len(_major_ticks(text, "x")) == len(xt) and len(_major_ticks(text, "y")) == len(yt)
    axes = {"x": {"ticks": {"0": xt[0], str(len(xt) - 1): xt[-1]}},
            "y": {"ticks": {"0": yt[-1], str(len(yt) - 1): yt[0]}}}
    t, _ = call(tb, "digitize_points", {"image": str(pdf), "page": 1, "crop": crop, "axes": axes,
                                        "color": "darkorange"})
    rows = _read_csv(re.search(r"-> (\S+\.csv)", t).group(1))
    got = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    assert len(got) == 12
    for x, y in zip(xs, ys):
        assert np.min(np.hypot(got[:, 0] - x, (got[:, 1] - y) / 800)) < 0.004


def test_calibration_errors_are_actionable(tb, tmp_path):
    path, truth, xt, yt = B.make_case("err", tmp_path, **{k: v for k, v in B.CASES["basic"].items()})
    text, _, err = tb.call("digitize_points", {"image": path, "axes": {"x": {"ticks": {"0": 0}}, "y": {"range": [0, 1]}}})
    assert err and "at least two calibration values" in text
    text, _, err = tb.call("digitize_points", {"image": path, "axes": {"x": {"ticks": {"0": 0, "99": 1}},
                                                                       "y": {"range": [0, 1]}}})
    assert err and "not detected" in text
    # a wrong tick value is flagged by the calibration residual
    t, _ = call(tb, "digitize_points", {"image": path, "axes": {"x": {"ticks": {"0": 0, "1": 0.2, "2": 0.5}},
                                                                "y": {"range": [0, 1]}}, "color": "C0"})
    assert "WARNING: calibration residual" in t


def test_tool_black_and_white_by_shape(tb, tmp_path):
    """B&W figure through the tool: shape filter switches to template matching from the cleanest example."""
    kw = {k: v for k, v in B.CASES["bw_shapes_on_lines"].items() if k not in B.EVAL_KEYS}
    path, truth, xt, yt = B.make_case("bw_tool", tmp_path, **kw)
    text, _ = call(tb, "digitize_setup", {"image": path})
    axes = {"x": {"ticks": {"0": xt[0], "5": xt[-1]}}, "y": {"ticks": {"0": yt[-1], "5": yt[0]}}}
    for s in truth.values():
        shape, filled = B.SHAPES[s["marker"]], s["filled"]
        t, ov = call(tb, "digitize_points", {"image": path, "axes": axes, "color": "black", "shape": shape,
                                             "filled": filled, "series": shape})
        assert "template from" in t, t
        rows = _read_csv(re.search(r"-> (\S+\.csv)", t).group(1))
        got = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        tp = np.column_stack([s["x"], s["y"]])
        d = np.array([np.min(np.hypot(*(got - p).T)) for p in tp])
        assert np.mean(d < 0.012) >= 0.85, (shape, np.mean(d < 0.012))
        prec = np.mean([np.min(np.hypot(*(tp - q).T)) < 0.012 for q in got])
        assert prec >= 0.95, (shape, prec)
