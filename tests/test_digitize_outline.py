"""Plot digitiser: dense, overlapping OPEN markers of several shapes in one colour (outline fitting), close colours
kept apart (nearest ink, not a fixed tolerance), and the axis-projected TDB partner inference of digitized_to_zpf."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from calphad_tools import digitize as D  # noqa: E402

CRIMSON, ORANGE_RED = "#c10020", "#f13a13"     # 24 Lab apart: a fixed 35-Lab tolerance cannot separate them
MS = 7.0                                        # marker size, pt (dpi 100: 9.7 px)


def _figure(path):
    """Crimson open circles in rows of touching markers, a dense band of crimson squares and triangles (down)
    overlapping each other, orange-red triangles (up) touching crimson markers, a few isolated markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    d = MS * 100 / 72 / ax.bbox.width             # marker diameter in x units
    asp = ax.bbox.width / ax.bbox.height
    truth = {"circle": [], "square": [], "triangle_down": [], "triangle_up": []}
    for y in (0.85, 0.75, 0.65):                  # rows of 4 circles, 0.55-0.65 diameters apart
        x0 = 0.1 + rng.uniform(0, 0.05)
        for k in range(4):
            truth["circle"].append((x0 + k * d * rng.uniform(0.55, 0.65), y))
    for k in range(14):                           # band: squares and triangles alternating, 0.75 d apart
        x = 0.35 + k * 0.75 * d * 1.0
        y = 0.35 + 0.3 * (x - 0.35) + (0.3 * d * asp if k % 2 else 0.0)
        truth["square" if k % 2 == 0 else "triangle_down"].append((x, y))
    for x, y in ((0.12, 0.2), (0.2, 0.3), (0.85, 0.15), (0.9, 0.85)):     # isolated ones
        truth["circle"].append((x, y))
    for x, y in ((0.3, 0.1), (0.6, 0.85)):
        truth["square"].append((x, y))
        truth["triangle_down"].append((x + 0.1, y + 0.05))
    reds = [(truth["circle"][1][0] + 0.4 * d, 0.75 + 0.5 * d * asp), (truth["circle"][5][0], 0.65 - 0.7 * d * asp),
            (truth["square"][3][0] + 0.2 * d, truth["square"][3][1] - 0.8 * d * asp), (0.7, 0.3), (0.75, 0.32)]
    truth["triangle_up"] = reds
    style = {"circle": ("o", CRIMSON), "square": ("s", CRIMSON), "triangle_down": ("v", CRIMSON),
             "triangle_up": ("^", ORANGE_RED)}
    for s, pts in truth.items():
        mk, col = style[s]
        for x, y in pts:
            ax.plot(x, y, marker=mk, ms=MS, mew=1.0, fillstyle="none", color=col, ls="")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return truth, d, asp


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    tmp = tmp_path_factory.mktemp("outline")
    img = tmp / "open_markers.png"
    truth, d, asp = _figure(img)
    tb = ToolBox(workdir=str(tmp), output_dir=str(tmp / "out"), ui=U(), python_tool="off")
    text, _, err = tb.call("digitize_setup", {"image": str(img)})
    assert not err, text
    axes = {"x": {"ticks": {"0": 0, "5": 1.0}}, "y": {"ticks": {"0": 1.0, "5": 0}}}

    def query(**f):
        name = "_".join(str(v).lstrip("#") for v in f.values())
        t, _, e = tb.call("digitize_points", {"image": str(img), "axes": axes, **f, "output_csv": f"{name}.csv"})
        assert not e, t
        with open(tmp / "out" / f"{name}.csv") as fh:
            return [(float(r["x"]), float(r["y"]), r["shape"]) for r in csv.DictReader(fh)], t
    return truth, (d, asp), query


def _score(truth, det, dd):
    """(hits, false detections): one-to-one matching within 0.3 marker diameters (in pixels)."""
    d, asp = dd
    used, hits = set(), 0
    for x, y in truth:
        best = None
        for j, (u, v, _) in enumerate(det):
            if j in used:
                continue
            e = math.hypot(u - x, (v - y) / asp)
            if e <= 0.3 * d and (best is None or e < best[0]):
                best = (e, j)
        if best:
            used.add(best[1])
            hits += 1
    return hits, len(det) - len(used)


def test_dense_open_markers_by_shape(run):
    truth, d, query = run
    for shape in ("circle", "square", "triangle_down"):
        det, text = query(color=CRIMSON, shape=shape)
        assert "outline fit" in text, text
        assert all(s == shape for *_, s in det)
        hits, false = _score(truth[shape], det, d)
        assert hits >= len(truth[shape]) - 1, (shape, hits, len(truth[shape]), text)
        assert false <= 1, (shape, false, text)


def test_close_reds_kept_apart(run):
    truth, d, query = run
    crimson, _ = query(color=CRIMSON)
    red, text = query(color=ORANGE_RED)
    # every orange-red triangle found, none of them reported as crimson and no crimson marker as orange-red
    hits, false = _score(truth["triangle_up"], red, d)
    assert hits == len(truth["triangle_up"]) and false == 0, text
    assert _score(truth["triangle_up"], crimson, d)[0] == 0
    n_crimson = sum(len(truth[s]) for s in ("circle", "square", "triangle_down"))
    assert _score([p for s in ("circle", "square", "triangle_down") for p in truth[s]], crimson, d)[0] \
        >= n_crimson - 2


def test_ink_colours_finds_a_rare_close_colour():
    rgb = np.ones((60, 200, 3))
    rgb[10:12, 10:190] = D.parse_color(CRIMSON)            # lots of crimson ...
    rgb[30:32, 10:190] = D.parse_color(CRIMSON)
    rgb[50:52, 20:40] = D.parse_color(ORANGE_RED)          # ... and a little orange-red
    inks = D.ink_colours(rgb, np.ones(rgb.shape[:2], bool), np.ones(3))
    names = sorted("#%02x%02x%02x" % tuple(int(round(v * 255)) for v in o["rgb"]) for o in inks)
    assert names == sorted([CRIMSON, ORANGE_RED])


def test_outline_template_geometry():
    T = D.outline_template("circle", 10.0, 2.0)
    assert T.sum() == pytest.approx(math.pi * 10.0 * 2.0, rel=0.03)            # ring area = pi * m * w
    S = D.outline_template("square", 10.0, 2.0, off=(0.5, 0.5))
    assert S.sum() == pytest.approx(12 ** 2 - 8 ** 2, rel=0.02)                  # miter-joined square stroke
    ys, xs = np.nonzero(D.outline_template("triangle_down", 10.0, 2.0) > 0.5)
    c = D.outline_template("triangle_down", 10.0, 2.0).shape[0] // 2
    assert ys.mean() < c - 0.5                 # 'v': the wide edge on top, the tip down (image y grows downwards)


# ------------------------------------------------------------------------------------- digitized_to_zpf
def _fake_inferencer(shift=100.0):
    """Two liquidus branches of a binary A-B (solid S1 for x < 0.3, S2 above), 100 K below the 'data', and their
    eutectic at x = 0.3 (the data's minimum is at 0.25)."""
    from calphad_tools.digitize_zpf import Inferencer, System
    inf = Inferencer.__new__(Inferencer)
    inf.sys = System(kind="binary", elements=["A", "B"], indep=["B"], comp_unit="fraction", T_unit="K")
    inf.T_scale = 1000.0
    x1 = np.linspace(0, 0.3, 16)
    x2 = np.linspace(0.3, 0.7, 21)
    T1 = 1300 - 1000 * x1 - shift          # S1 liquidus: 1300 K at x=0 down to 1000 K at the eutectic
    T2 = 1000 + 300 * (x2 - 0.3) - shift
    inf.traces = [(("2P", "LIQUID", ("S1",)), np.c_[x1, T1 / 1000]), (("2P", "LIQUID", ("S2",)), np.c_[x2, T2 / 1000])]
    inf.inv = [{"phases": ["S1", "LIQUID", "S2"], "T": 1000.0 - shift,
                "xy": {"S1": np.array([0.05, (1000 - shift) / 1000]), "LIQUID": np.array([0.3, (1000 - shift) / 1000]),
                       "S2": np.array([0.6, (1000 - shift) / 1000])}}]
    inf.liquid_warp = None
    return inf


def _pt(x, T, i=0):
    from calphad_tools.digitize_zpf import DPoint
    return DPoint(idx=i, group="L", source="", kind="marker", raw=(x, T), X={"A": 1 - x, "B": x}, T=T)


def test_liquid_partner_by_vertical_projection():
    inf = _fake_inferencer()
    # just right of the eutectic, 100 K above the calculated S2 liquidus: plain 2-D distance prefers the S1 branch
    # (it ends nearby); the liquid's branch is the one below it at the same composition
    p = _pt(0.34, 1150.0)
    plain = min(inf.candidates(p).items(), key=lambda kv: kv[1])[0]
    by_phase = min(inf.candidates(p, phase="LIQUID").items(), key=lambda kv: kv[1])[0]
    assert plain == ("2P", "LIQUID", ("S1",))
    assert by_phase == ("2P", "LIQUID", ("S2",))


def test_liquid_warp_from_digitized_eutectic():
    inf = _fake_inferencer(shift=0.0)
    xs = np.linspace(0.0, 0.6, 25)
    data = [_pt(x, 1350 - 1400 * x if x <= 0.25 else 1000 + 300 * (x - 0.25), i) for i, x in enumerate(xs)]
    pairs = inf.fit_liquid_warp(data)
    assert len(pairs) == 1 and pairs[0][0] == pytest.approx(0.25, abs=0.03) and pairs[0][1] == pytest.approx(0.3)
    p = _pt(0.27, 1010.0)                 # right of the digitized minimum: the S2 branch although x < 0.3
    best = min(inf.candidates(p, phase="LIQUID").items(), key=lambda kv: kv[1])[0]
    assert best == ("2P", "LIQUID", ("S2",))
