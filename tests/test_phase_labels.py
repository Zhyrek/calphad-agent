"""Phase-region labels on phase diagrams (phase_names, phase_regions, label_place, region_labels).

Run:  .venv/bin/python -m pytest -q -p no:warnings tests/test_phase_labels.py
"""
import csv
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,label", [
    ("BCC_A2", "BCC"), ("A2_BCC", "BCC"), ("FCC_A1", "FCC"), ("HCP_A3", "HCP"), ("LIQUID", "L"), ("LIQ", "L"),
    ("B2_BCC", "B2"), ("BCC_B2", "B2"), ("L12_FCC", "L1₂"), ("FCC_L12", "L1₂"), ("DHCP", "DHCP"),
    ("CBCC_A12", "αMn"), ("CUB_A13", "βMn"), ("LAVES_C14", "C14"), ("C14_LAVES", "C14"), ("LAVES_C15", "C15"),
    ("C15_LAVES", "C15"), ("LAVES_C36", "C36"), ("SIGMA", "σ"), ("SIGMA_D8B", "σ"), ("SIGMA_PHASE", "σ"),
    ("SIGMA_SGTE", "σ"), ("MU_PHASE", "μ"), ("MU", "μ"), ("MU_D85", "μ"), ("CHI", "χ"), ("CHI_A12", "χ"),
    ("DELTA", "δ"), ("GAMMA_PRIME", "γ′"), ("ETA", "η"), ("R_PHASE", "R"), ("P_PHASE", "P"),
    ("BCC_A2#2", "BCC"), ("SIGMA#3", "σ"), ("FCC_A1_1", "FCC"), ("FOOBAR_XYZ", "FOOBAR_XYZ"),
])
def test_format_phase_table(raw, label):
    from calphad_tools.phase_names import format_phase
    assert format_phase(raw) == label


def test_format_phase_formulas():
    from calphad_tools.phase_names import format_phase, split_formula, strukturbericht
    assert format_phase("FE3CR4", ["FE", "CR"]) == "Fe₃Cr₄"
    assert format_phase("FE3CR4") == "Fe₃Cr₄"                       # periodic table when no element list
    assert format_phase("AL13FE4", ["AL", "FE"]) == "Al₁₃Fe₄"
    assert format_phase("M23C6", ["CR", "FE", "C"]) == "M₂₃C₆"
    assert format_phase("M7C3") == "M₇C₃"
    assert format_phase("NI3AL", ["AL", "NI"]) == "Ni₃Al"
    assert format_phase("CR3SI", ["CR", "SI"]) == "Cr₃Si"
    assert format_phase("CRNI2", ["CR", "NI"]) == "CrNi₂"
    assert format_phase("AL3NI1", ["AL", "NI"]) == "Al₃Ni"
    # CO+SI, not C+O+SI (longest symbols first, with backtracking)
    assert split_formula("COSI", ["CO", "SI", "C", "O"]) == [("CO", ""), ("SI", "")]
    assert format_phase("COSI", ["CO", "SI", "C", "O"]) == "CoSi"
    assert split_formula("CO", ["C", "O"]) == [("C", ""), ("O", "")]
    assert format_phase("NI3TI_D024", ["NI", "TI"]) == "Ni₃Ti"         # structure suffix dropped
    assert format_phase("TI3AL_D0_19", ["TI", "AL"]) == "Ti₃Al"
    assert format_phase("AL8CR5_H", ["AL", "CR"]) == "Al₈Cr₅-HT"
    assert format_phase("AL8CR5_L", ["AL", "CR"]) == "Al₈Cr₅-LT"
    assert format_phase("LAVES", ["LA", "V", "ES"]) == "Laves"          # table before formulas
    assert strukturbericht("D019") == "D0₁₉" and strukturbericht("B2") == "B2" and strukturbericht("D8B") == "D8b"


def test_region_labels_and_overrides():
    from calphad_tools.phase_names import format_phase, region_key, region_label
    assert region_label(["BCC_A2", "BCC_A2#2"]) == "BCC+BCC"
    assert region_label(["BCC_A2#1", "BCC_A2#2"]) == "BCC+BCC"
    assert region_label(["FCC_A1", "LIQUID"]) == "L+FCC"
    assert region_label(["LIQUID", "BCC_A2"]) == "L+BCC"
    assert region_label(["SIGMA", "FCC_A1", "BCC_A2"]) == "BCC+FCC+σ"
    assert region_label(["SIGMA", "BCC_A2"], overrides={"SIGMA": "σ-phase"}) == "BCC+σ-phase"
    assert format_phase("BCC_A2", overrides={"BCC": "α"}) == "α"        # override of the default label
    assert region_key(["BCC_A2#2", "BCC_A2"]) == ("BCC_A2", "BCC_A2")


# ---------------------------------------------------------------------------------------------------------
# diagrams
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    from calphad_tools.agent.tools import ToolBox, UI

    class MyUI(UI):
        def ask(self, question, options=None):
            return ""

        def confirm(self, title, body):
            return True
    out = tmp_path_factory.mktemp("labels")
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out / "out"), ui=MyUI(), python_tool="off")


def call(tb, name, args):
    text, images, is_error = tb.call(name, args)
    assert not is_error, text
    return text, images


def files_from(text):
    out = {}
    for line in text.split("Files:", 1)[1].splitlines():
        m = re.match(r"\s+(\w+): (.+)$", line)
        if m:
            p = Path(m.group(2).strip())
            out[m.group(1)] = p if p.is_absolute() else ROOT / "examples" / p
    return out


def stored(png):
    from calphad_tools.figures import load_figure
    fig = load_figure(str(png))
    assert fig is not None
    return fig, fig.axes[0]


def labels_of(ax, prefix=("phase_label:", "invariant_label:")):
    return [t for t in ax.texts if (t.get_gid() or "").startswith(prefix)]


def check_anchors(ax):
    """Every phase label (or the end of its leader line) lies in a region with exactly its phase set."""
    from calphad_tools.calc import get_db
    from calphad_tools.phase_regions import key_at, ordered_info
    rec = ax._calphad_diagram["recipe"]
    db = get_db(rec["tdb"])
    info = ordered_info(db, sorted(rec["elements"]) + ["VA"], rec["phases"])
    n = 0
    for t in labels_of(ax, ("phase_label:",)):
        lab = t._calphad_label
        x, y = t.xy if hasattr(t, "xy") and lab.get("leader") else t.get_position()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = key_at(db, rec, info, x, y)
        if lab["kind"] == "compound":
            assert got and lab["key"][0] in got, (t.get_text(), got)
        else:
            assert got == tuple(lab["key"]), (t.get_text(), (x, y), got)
        n += 1
    return n


def text_rect(t, r):
    """Corners (display px) of a label's (rotated) text rectangle."""
    from matplotlib.text import Text
    if hasattr(t, "update_positions"):
        t.update_positions(r)
    rot = t.get_rotation()
    t.set_rotation(0)
    e = Text.get_window_extent(t, r)
    t.set_rotation(rot)
    cx, cy = t.get_transform().transform(t.get_unitless_position())
    a = np.radians(rot)
    hw, hh = e.width / 2, e.height / 2
    return np.array([(cx + u * np.cos(a) - v * np.sin(a), cy + u * np.sin(a) + v * np.cos(a))
                     for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))])


def rects_overlap(p, q):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() <= b.min() + 1e-6 or b.max() <= a.min() + 1e-6:
                return False
    return True


def point_in_rect(p, u, v, pad=0.0):
    c = p.mean(axis=0)
    ex, ey = p[1] - p[0], p[3] - p[0]
    d = np.array([u, v]) - c
    return abs(d @ ex) / np.linalg.norm(ex) <= np.linalg.norm(ex) / 2 + pad and \
        abs(d @ ey) / np.linalg.norm(ey) <= np.linalg.norm(ey) / 2 + pad


def check_layout(fig, ax):
    """Labels do not overlap each other and stay inside the axes box.  Returns [(text, corners)]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    rects = [(t.get_text(), text_rect(t, r)) for t in labels_of(ax)]
    axb = ax.bbox
    for s, p in rects:
        assert p[:, 0].min() >= axb.x0 - 1 and p[:, 0].max() <= axb.x1 + 1, s
        assert p[:, 1].min() >= axb.y0 - 1 and p[:, 1].max() <= axb.y1 + 1, s
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects_overlap(rects[i][1], rects[j][1]), (rects[i][0], rects[j][0])
    return rects


def test_binary_crni(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                  "tmax": 2200, "temperature_unit": "C", "label_regions": True,
                                                  "output_prefix": "crni_labels"})
    f = files_from(text)
    for k in ("png", "svg", "pdf"):
        assert f[k].exists() and f[k].stat().st_size > 5000, k
    assert "phase labels" in text
    fig, ax = stored(images[0])
    texts = {t.get_text() for t in labels_of(ax)}
    assert {"L", "BCC", "FCC", "L+BCC", "L+FCC", "BCC+FCC"} <= texts, texts
    # eutectic temperature, in °C, as in the invariants CSV
    with open(f["invariants_csv"], newline="") as fh:
        T_C = float(next(r for r in csv.DictReader(fh) if r["type"] == "eutectic")["T_C"])
    inv = [t for t in labels_of(ax, ("invariant_label:",))]
    assert [t.get_text() for t in inv] == [f"{T_C:.0f} °C"] and inv[0].get_gid() == f"invariant_label:{T_C:.0f}C"
    x, y = inv[0].get_position()
    assert abs(y - T_C) < 60                                          # next to the eutectic line
    assert check_anchors(ax) >= 6
    check_layout(fig, ax)
    pickle.dumps(fig)                                                 # figures stay picklable
    # vector output keeps the text as text
    svg = f["svg"].read_text()
    for s in ("L+BCC", "BCC+FCC", f"{T_C:.0f} °C"):
        assert re.search(r"<text[^>]*>" + re.escape(s) + "</text>", svg), s
    assert 'id="phase_label:BCC_A2.FCC_A1"' in svg and "DejaVu Sans" in svg
    assert b"/FontFile2" in f["pdf"].read_bytes()                     # TrueType (Type 42) font embedded
    assert "BCC+FCC" in _pdf_text(f["pdf"]) and f"{T_C:.0f}" in _pdf_text(f["pdf"])


def _pdf_text(path):
    try:
        from pypdf import PdfReader
        return "".join(p.extract_text() for p in PdfReader(str(path)).pages)
    except ImportError:
        return subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True).stdout


def test_binary_fecr_sigma_and_gap(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrFe-82Her.tdb", "elements": ["FE", "CR"], "tmin": 300,
                                                  "tmax": 1000, "temperature_unit": "C", "label_regions": True,
                                                  "label_style": {"invariant_type": True},
                                                  "output_prefix": "fecr_labels"})
    fig, ax = stored(images[0])
    texts = [t.get_text() for t in labels_of(ax)]
    assert "σ" in texts and "BCC+BCC" in texts and "FCC" in texts and "BCC" in texts
    assert texts.count("BCC+σ") == 2                                  # both sides of the σ field
    assert any(re.fullmatch(r"431 °C \(eutectoid\)", s) for s in texts), texts
    assert not any("#" in s or "₁" in s for s in texts)               # gap is BCC+BCC, never BCC#1 / BCC₁
    assert check_anchors(ax) >= 6
    check_layout(fig, ax)
    svg = files_from(text)["svg"].read_text()
    assert re.search(r"<text[^>]*>σ</text>", svg)
    assert "σ" in _pdf_text(files_from(text)["pdf"])


def test_ternary_tie_triangles(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["NI", "CO", "CR"],
                                                  "temperature": 900, "temperature_unit": "C",
                                                  "label_regions": True, "output_prefix": "nicocr_labels"})
    fig, ax = stored(images[0])
    import json
    summary = json.loads(files_from(text)["json"].read_text())
    tris = summary["invariants"]
    assert len(tris) >= 2
    from calphad_tools.phase_names import region_label, region_key
    labs = labels_of(ax, ("phase_label:",))
    for inv in tris:
        key = region_key(inv["phases"])
        hit = [t for t in labs if tuple(t._calphad_label["key"]) == key]
        assert hit, (inv["phases"], [t.get_text() for t in labs])
        t = hit[0]
        assert t.get_text() == region_label(inv["phases"]) and t.get_text().count("+") == 2
        # anchor (or leader end) inside the triangle
        x, y = t.xy if t._calphad_label.get("leader") else t.get_position()
        P = np.array([[inv["compositions"][n][e] for e in ("CO", "CR")] for n in inv["phases"]])
        lam = np.linalg.solve(np.vstack([P.T, np.ones(3)]), np.array([x, y, 1.0]))
        assert (lam > -1e-6).all(), (t.get_text(), lam)
    assert check_anchors(ax) >= 8
    check_layout(fig, ax)


def test_obstacles_edit_ops_and_show_code(tb):
    # a diagram without labels, then data points where the L+BCC label would go, then labels on the stored figure
    _, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                               "tmax": 2200, "temperature_unit": "C", "output_prefix": "crni_plain"})
    ref, imgs_ref = call(tb, "edit_figure", {"image": images[0], "operations": [{"op": "label_phase_regions"}],
                                             "output": "crni_ref.png"})
    assert Path(imgs_ref[0]).with_suffix(".svg").exists() and Path(imgs_ref[0]).with_suffix(".pdf").exists()
    fig0, ax0 = stored(imgs_ref[0])
    lb = next(t for t in labels_of(ax0) if t.get_text() == "L+BCC")
    x0, y0 = lb.get_position()
    pts_x = [x0 - 0.03, x0, x0 + 0.03, x0 - 0.015, x0 + 0.015]
    pts_y = [y0, y0 + 8, y0, y0 - 12, y0 - 12]
    ops = [{"op": "scatter", "x": pts_x, "y": pts_y, "color": "k", "s": 25, "label": "measured"},
           {"op": "label_phase_regions"}]
    text, imgs = call(tb, "edit_figure", {"image": images[0], "operations": ops, "output": "crni_data.png"})
    assert "phase labels" in text
    fig, ax = stored(imgs[0])
    lb = next(t for t in labels_of(ax) if t.get_text() == "L+BCC")
    assert np.hypot(*(np.subtract(lb.get_position(), (x0, y0)) / [1.0, 1400.0])) > 0.01      # moved
    rects = check_layout(fig, ax)
    sc = [c for c in ax.collections if c.get_label() == "measured"][0]
    px = sc.get_offset_transform().transform(sc.get_offsets())
    rad = np.sqrt(sc.get_sizes()[0]) / 2 * fig.dpi / 72
    for s, p in rects:
        for (u, v) in px:
            assert not point_in_rect(p, u, v, rad), (s, u, v)
    check_anchors(ax)
    # edit labels by id / text
    lab = next(t for t in labels_of(ax) if t.get_gid() == "phase_label:LIQUID")
    before = ax.transData.transform(lab.get_position())
    ops = [{"op": "move_label", "id": "phase_label:LIQUID", "dx": 20, "dy": -10},
           {"op": "delete_label", "text": "BCC+FCC"},
           {"op": "relabel", "id": "FCC_A1", "new_text": "γ (FCC)", "color": "navy"}]
    text, imgs2 = call(tb, "edit_figure", {"image": imgs[0], "operations": ops, "output": "crni_edited.png"})
    fig2, ax2 = stored(imgs2[0])
    lab2 = next(t for t in labels_of(ax2) if t.get_gid() == "phase_label:LIQUID")
    after = ax2.transData.transform(lab2.get_position())
    assert np.allclose((after - before) * 72 / fig2.dpi, (20, -10), atol=0.05)
    assert "BCC+FCC" not in {t.get_text() for t in labels_of(ax2)}
    assert next(t for t in labels_of(ax2) if t.get_gid() == "phase_label:FCC_A1").get_text() == "γ (FCC)"
    _, _, err = tb.call("edit_figure", {"image": imgs[0], "operations": [{"op": "delete_label", "id": "NOPE"}]})
    assert err
    # standalone code reproduces the labels (same strings, nearby positions) and does not import the toolkit
    code_text, _ = call(tb, "show_code", {"target": imgs2[0], "format": "script"})
    py = Path(re.search(r"saved (\S+\.py)", code_text).group(1))
    py = py if py.is_absolute() else ROOT / "examples" / py
    src = py.read_text()
    assert "calphad_tools" not in src
    run = subprocess.run([sys.executable, str(py)], cwd=str(py.parent), capture_output=True, text=True, timeout=900,
                         env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-3000:]
    got = {}
    for line in run.stdout.splitlines():
        m = re.match(r"label: (.+) at \(([-\d.e]+), ([-\d.e]+)\) gid (\S+)", line)
        if m:
            got[m.group(4)] = (m.group(1), float(m.group(2)), float(m.group(3)))
    want = {t.get_gid(): (t.get_text(), *t.get_position()) for t in labels_of(ax)}
    assert set(got) == set(want), (set(got) ^ set(want))
    for gid, (s, x, y) in want.items():
        assert got[gid][0] == s
        assert abs(got[gid][1] - x) < 0.08 and abs(got[gid][2] - y) < 100, (gid, got[gid], (x, y))
    out = py.parent / "standalone_outputs"
    svg = next(out.glob("crni_edited*.svg")).read_text()
    assert re.search(r"<text[^>]*>γ \(FCC\)</text>", svg) and "phase_label:LIQUID" in svg


def test_plot_diagram_with_data_labels(tb, tmp_path):
    from calphad_tools import expdata
    d = tmp_path / "ds"
    zpf, _ = expdata.pop_to_zpf(str(ROOT / "examples/expdata/CrNi_boundaries.POP"), ["CR", "NI"],
                                tdb_phases=["LIQUID", "FCC_A1", "BCC_A2"])
    for i, ds in enumerate(zpf):
        expdata.write_dataset(ds, str(d / f"z{i}.json"))
    text, imgs = call(tb, "plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": str(d),
                                                     "tdb": "CrNi-06Hal.tdb", "tmin": 800, "tmax": 2000,
                                                     "temperature_unit": "C", "label_regions": True,
                                                     "phase_labels": {"LIQUID": "Liquid"},
                                                     "output_png": "pdd_labels.png"})
    assert "phase labels" in text and Path(imgs[0]).with_suffix(".svg").exists()
    fig, ax = stored(imgs[0])
    texts = {t.get_text() for t in labels_of(ax)}
    assert {"Liquid", "Liquid+BCC", "BCC+FCC", "1345 °C"} <= texts, texts
    rects = check_layout(fig, ax)
    n = 0
    for ln in ax.get_lines():                          # no label covers a data marker
        if ln.get_marker() in (None, "None", "", " "):
            continue
        px = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
        rad = ln.get_markersize() / 2 * fig.dpi / 72
        for s, p in rects:
            for u, v in px:
                assert not point_in_rect(p, u, v, rad), (s, u, v)
                n += 1
    assert n > 0
    check_anchors(ax)


def test_label_errors(tb):
    _, _, err = tb.call("map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1000,
                                              "tmax": 1200, "label_regions": True, "label_style": {"size": 3}})
    assert err
    _, images = call(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                       "output": "plain"})
    text, _, err = tb.call("edit_figure", {"image": images[0], "operations": [{"op": "label_phase_regions"}]})
    assert err and "recipe" in text


def test_label_diagram_synthetic(tmp_path):
    """Placement on a hand-made diagram: a two-phase box, a line compound, an invariant line and data points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calphad_tools.label_place import label_diagram, save_vector
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1000)
    ax.plot([0.1, 0.5, 0.5, 0.1, 0.1], [100, 100, 500, 500, 100], color="C0")
    ax.plot([0.7, 0.7], [0, 800], color="C1")
    ax.plot([0.1, 0.9], [600, 600], color="red")
    ax.scatter([0.3, 0.28, 0.32], [300, 280, 320], s=30, color="k")      # data right where the label would go
    regions = {"polygons": [{"key": ("FE3CR4", "LIQUID"), "xy": [[0.1, 100], [0.5, 100], [0.5, 500], [0.1, 500]]}],
               "compounds": [{"key": ("FE3CR4",), "x": 0.7, "y0": 0.0, "y1": 800.0}],
               "invariants": [{"key": ("A", "B", "C"), "T_K": 873.15, "y": 600.0, "x0": 0.1, "x1": 0.9,
                               "type": "eutectic"}]}
    rep = label_diagram(ax, regions, None, None, kind="binary", elements=["FE", "CR"], T_unit="C",
                        style={"invariant_type": True})
    texts = {lab["text"]: lab for lab in rep["labels"]}
    assert {"L+Fe₃Cr₄", "Fe₃Cr₄", "600 °C (eutectic)"} <= set(texts), texts
    x, y = texts["L+Fe₃Cr₄"]["x"], texts["L+Fe₃Cr₄"]["y"]
    assert 0.1 < x < 0.5 and 100 < y < 500 and np.hypot((x - 0.3) / 0.4, (y - 300) / 400) > 0.1
    assert texts["Fe₃Cr₄"]["rotation"] in (0.0, 90.0) and abs(texts["Fe₃Cr₄"]["x"] - 0.7) < 0.1
    assert abs(texts["600 °C (eutectic)"]["y"] - 600) < 60
    pickle.dumps(fig)
    save_vector(fig, tmp_path / "s.svg")
    save_vector(fig, tmp_path / "s.pdf")
    svg = (tmp_path / "s.svg").read_text()
    for s in ("L+Fe₃Cr₄", "Fe₃Cr₄", "600 °C (eutectic)"):
        assert re.search(r"<text[^>]*>" + re.escape(s) + "</text>", svg), s
    assert 'id="phase_label:FE3CR4.LIQUID"' in svg and 'id="invariant_label:600C"' in svg
    assert "Fe₃Cr₄" in _pdf_text(tmp_path / "s.pdf")
    plt.close(fig)
