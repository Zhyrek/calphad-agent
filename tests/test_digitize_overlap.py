"""Plot digitiser: close / overlapping markers (never merged silently) and the provenance figure."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import digitize_bench as B  # noqa: E402
import digitize_overlap_bench as OB  # noqa: E402
from calphad_tools import digitize as D  # noqa: E402


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(tmp_path), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------- overlapping markers
@pytest.fixture(scope="module")
def realistic(tmp_path_factory):
    """Pairs / triples at 2.0 ... 0.25 marker diameters plus 15 isolated markers per series, through the tools."""
    out = tmp_path_factory.mktemp("overlap")
    path, truth, ux, uy, ytop = OB.build("realistic", out)
    rep, iso, texts, glob = OB.run("realistic", path, truth, ux, uy, ytop, out)
    print(OB.table(rep, iso, glob))
    return rep, iso, texts, glob


def test_overlap_no_silent_merges(realistic):
    rep, iso, texts, glob = realistic
    silent = [(label, c["sep"]) for label, line in rep for c in line if c["silent"]]
    assert not silent, silent


def test_overlap_targets_by_separation(realistic):
    rep, *_ = realistic
    for label, line in rep:
        for c in line:
            flagged = c["flagged"] + c["flag_on_hit"] > 0
            if c["sep"] >= 1.0:                     # all found, centres within a quarter diameter
                assert c["hit"] == c["true"], (label, c)
                assert c["err_diam"] < 0.25, (label, c)
            elif c["sep"] >= 0.6:                   # found, or flagged for a human look
                assert c["hit"] == c["true"] or flagged, (label, c)
            else:                                   # heavy overlap: resolved, or flagged
                assert c["hit"] == c["true"] or flagged, (label, c)
            assert c["false"] == 0 or flagged, (label, c)


def test_overlap_isolated_markers_untouched(realistic):
    """The splitting must not cost precision on ordinary, isolated markers."""
    _, iso, _, glob = realistic
    assert iso["hit"] == iso["true"], iso
    assert iso["false"] == 0 and iso["flagged"] == 0, iso
    assert iso["err_diam_max"] < 0.15, iso
    # whole figure, every filter: extra detections are flagged; the only unflagged ones allowed are the known
    # limitation (a square template matching two black circles merged at <= 0.4 d)
    assert glob["false"] <= 2, glob


def test_overlap_reported_in_text_csv_and_overlay(realistic, tmp_path):
    rep, iso, texts, glob = realistic
    t = texts["r0_0"]                               # red filled circles: one unresolvable pair at 0.25 d
    assert re.search(r"CHECK: #\d+.* may be 2\+ overlapping markers", t), t
    assert "split touching/overlapping markers" in t
    csvp = re.search(r"-> (\S+\.csv)", t).group(1)
    out = Path(re.search(r"provenance figure .*?: (\S+\.png)", t).group(1))
    assert out.name.endswith("_provenance.png")


def test_split_notes_in_csv(tb, tmp_path):
    """Touching markers are split (note 'split from a blob of 2'), with lower scores than clean markers."""
    kw = {k: v for k, v in B.CASES["curves_touching_errorbars"].items() if k not in B.EVAL_KEYS}
    path, truth, xt, yt = B.make_case("touch", tmp_path, **kw)
    call(tb, "digitize_setup", {"image": path})
    axes = {"x": {"ticks": {"0": xt[0], "5": xt[-1]}}, "y": {"ticks": {"0": yt[-1], "5": yt[0]}}}
    t, imgs = call(tb, "digitize_points", {"image": path, "axes": axes, "color": "C3", "series": "s"})
    rows = _rows(tb.p(re.search(r"-> (\S+\.csv)", t).group(1)))
    assert "note" in rows[0]
    split = [r for r in rows if r["note"].startswith("split from a blob of 2")]
    assert len(split) >= 6
    assert max(float(r["score"]) for r in split) <= 0.7 < max(float(r["score"]) for r in rows)
    # every split point is on a true marker
    tp = np.column_stack([truth["series0"]["x"], truth["series0"]["y"]])
    for r in split:
        assert np.min(np.hypot(tp[:, 0] - float(r["x"]), tp[:, 1] - float(r["y"]))) < 0.01


def test_pursuit_resolves_pair_and_flags_heavy_overlap():
    """Unit test of the matching pursuit on ideal discs: a pair 0.8 d apart is resolved at the true centres (not
    at the midpoint); a pair 0.25 d apart is ambiguous with one template and is not split."""
    r = 6.0
    R = 9
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    T = np.hypot(yy, xx) <= r
    for sep, want in ((0.8, 2), (1.0, 2), (0.25, 1)):
        H, W = 40, 60
        y, x = np.mgrid[0:H, 0:W]
        c1, c2 = (20, 30 - sep * r), (20, 30 + sep * r)
        B_ = (np.hypot(y - c1[0], x - c1[1]) <= r) | (np.hypot(y - c2[0], x - c2[1]) <= r)
        picks, info = D._explain(B_, [T], 3)
        assert len(picks) == want, (sep, picks, info)
        if want == 2:
            xs = sorted(p[1] for p in picks)
            assert abs(xs[0] - c1[1]) <= 1 and abs(xs[1] - c2[1]) <= 1, (sep, xs)


# ---------------------------------------------------------------------------------- provenance figure
def _placement(png):
    from PIL import Image
    with Image.open(png) as im:
        info = dict(im.info)
        arr = np.asarray(im.convert("RGB"), dtype=float) / 255
    return json.loads(info["image_placement"]), json.loads(info["series_colours"]), info, arr


def _hits_colour(arr, pl, pts_px, colour, offset_px):
    """Fraction of points whose right crosshair arm (offset_px image pixels right of the centre) has the colour."""
    target = D.to_lab(D.parse_color(colour).reshape(1, 3))[0]
    lab = D.to_lab(arr)
    ok = 0
    for x, y in pts_px:
        fx = pl["left"] + (x + 0.5 + offset_px) * pl["scale"]
        fy = pl["top"] + (y + 0.5) * pl["scale"]
        win = lab[int(fy) - 1:int(fy) + 2, int(fx) - 1:int(fx) + 2].reshape(-1, 3)
        ok += float(np.min(np.linalg.norm(win - target, axis=1))) < 25
    return ok / max(1, len(pts_px))


def test_provenance_figure_combined_and_at_calibrated_positions(tb, tmp_path):
    kw = dict(series=[("red", "o", True, 15), ("blue", "s", False, 15)], curves=True, legend=True)
    path, truth, xt, yt = B.make_case("prov", tmp_path, **kw)
    text, _ = call(tb, "digitize_setup", {"image": path})
    axes = {"x": {"ticks": {"0": xt[0], "5": xt[-1]}}, "y": {"ticks": {"0": yt[-1], "5": yt[0]}}}
    t1, imgs1 = call(tb, "digitize_points", {"image": path, "axes": axes, "color": "red", "series": "red"})
    prov = Path(re.search(r"provenance figure .*?: (\S+\.png)", t1).group(1))
    prov = prov if prov.is_absolute() else Path(tb.p(str(prov)))
    assert prov.exists() and str(prov) in [str(Path(i)) for i in imgs1] and "SHOW it to the user" in t1
    pl, cols, info, arr = _placement(prov)
    assert list(cols) == ["red"]
    # the provenance figure is drawn from the VALUES: map them back through an independent calibration
    rgb = D.load_rgb(tb.p(re.search(r"working image: (\S+)", text).group(1)))
    fr = D.detect_frame(rgb)
    ticks = D.detect_ticks(rgb, fr)
    calx, caly = D._cal(axes, ticks, fr)
    rows = _rows(tb.p(re.search(r"-> (\S+\.csv)", t1).group(1)))
    pts = [(float(calx.to_px(float(r["x"]))), float(caly.to_px(float(r["y"])))) for r in rows]
    size = float(np.median([float(r["size_px"]) for r in rows]))
    arm = 1.3 * max(3.5, 0.8 * size)                    # middle of the right crosshair arm
    assert _hits_colour(arr, pl, pts, cols["red"], arm) >= 0.8
    # ... and nowhere else: shifted by half an arm the colour is not there
    assert _hits_colour(arr, pl, [(x, y + 2.2 * arm) for x, y in pts], cols["red"], arm) < 0.3
    # second series + a traced line: the SAME figure accumulates them
    t2, _ = call(tb, "digitize_points", {"image": path, "axes": axes, "color": "blue", "series": "blue"})
    t3, _ = call(tb, "digitize_curve", {"image": path, "axes": axes, "color": "C2", "series": "C2 line"})
    for t in (t2, t3):
        assert Path(re.search(r"provenance figure .*?: (\S+\.png)", t).group(1)).name == prov.name
    pl, cols, info, arr = _placement(prov)
    assert list(cols) == ["red", "blue", "C2 line"]
    assert len(set(cols.values())) == 3                    # one distinct colour per series
    assert _hits_colour(arr, pl, pts, cols["red"], arm) >= 0.8        # the first series is still drawn
    rows_b = _rows(tb.p(re.search(r"-> (\S+\.csv)", t2).group(1)))
    pts_b = [(float(calx.to_px(float(r["x"]))), float(caly.to_px(float(r["y"])))) for r in rows_b]
    assert _hits_colour(arr, pl, pts_b, cols["blue"], 1.3 * max(3.5, 0.8 * float(rows_b[0]["size_px"]))) >= 0.8
    assert "red: " in info["Description"] and "blue: " in info["Description"] and "C2 line" in info["Description"]
    # the toolkit's provenance sidecar is written for it like for every output PNG
    assert Path(str(prov) + ".provenance.json").exists()
    # re-running a series replaces it (no duplicate entry)
    call(tb, "digitize_points", {"image": path, "axes": axes, "color": "red", "series": "red"})
    assert list(_placement(prov)[1]) == ["red", "blue", "C2 line"]


def test_calibration_check_marks_expose_a_wrong_calibration(tmp_path):
    path, truth, xt, yt = B.make_case("cal", tmp_path, **B.CASES["basic"])
    rgb = D.load_rgb(path)
    fr = D.detect_frame(rgb)
    ticks = D.detect_ticks(rgb, fr)
    good = D.calibrate_axis({"ticks": {0: xt[0], 5: xt[-1]}}, ticks["x"], fr.left, fr.right, "x")
    labels = [lab for _, lab in D.calibration_checks(good, ticks["x"], fr.left, fr.right)]
    assert labels == [D._fmt(v) for v in xt]                # exactly the printed tick labels
    wrong = D.calibrate_axis({"ticks": {0: xt[0], 5: 1.2}}, ticks["x"], fr.left, fr.right, "x")
    wl = [lab for _, lab in D.calibration_checks(wrong, ticks["x"], fr.left, fr.right)]
    assert wl != labels and wl[-1] == "1.2"


def test_provenance_in_digitized_to_zpf_and_show_code(tb, tmp_path):
    """digitized_to_zpf draws the CSV values on the original figure too; show_code redraws it standalone."""
    kw = dict(series=[("red", "o", True, 12)], legend=False)
    path, truth, xt, yt = B.make_case("zpfprov", tmp_path, **kw)
    call(tb, "digitize_setup", {"image": path})
    axes = {"x": {"ticks": {"0": xt[0], "5": xt[-1]}}, "y": {"ticks": {"0": 1000, "5": 2000}}}
    t, _ = call(tb, "digitize_points", {"image": path, "axes": axes, "color": "red", "series": "liq"})
    csvp = re.search(r"-> (\S+\.csv)", t).group(1)
    text, imgs = call(tb, "digitized_to_zpf", {"components": ["CR", "NI"], "csv": csvp, "image": path, "axes": axes,
                                               "assignments": {"liq": {"boundary": "liquidus"}},
                                               "name_map": {"L": "LIQUID"}, "preview": True})
    m = re.search(r"extracted points on the original figure: (\S+\.png)", text)
    assert m, text
    png = Path(tb.p(m.group(1)))
    assert png.exists() and str(png) in [str(Path(i)) for i in imgs]
    pl, cols, info, arr = _placement(png)
    assert list(cols) == ["liq"]
    # show_code of the digitize step: the recorded values + a standalone provenance redraw
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, tb.p(csvp))
    script = codegen.to_script(tb, cells, ctx)
    assert "digitized = pd.DataFrame(" in script and "# provenance check" in script
    assert "calphad_tools" not in script
