"""Tests for the matplotlib-control tools (describe_figure / edit_figure / make_plot)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


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


def _points(fig, label):
    from calphad_tools.figures import main_axes
    for c in main_axes(fig)[0].collections:
        if c.get_label() == label:
            return np.asarray(c.get_offsets())
    raise AssertionError(label)


def test_kelvin_axis_shown_in_celsius_from_worker_process(tb, tmp_path):
    """Batch diagrams are drawn in K with °C tick labels in a worker process; edits given in °C must land in K."""
    from calphad_tools import diagrams
    from calphad_tools.figures import load_figure
    out = str(tmp_path / "out" / "crni.png")
    res = diagrams.run_tasks([{"name": "CR-NI", "kind": "binary", "kwargs": dict(
        tdb=str(ROOT / "examples/tdbs/CrNi-06Hal.tdb"), elements=["CR", "NI"], out_png=out, tmin_c=800,
        tmax_c=2000, x_step=0.05, t_step=20)}], n_workers=1)
    assert res[0].ok, res[0].error
    info = json.loads(call(tb, "describe_figure", {"image": out})[0])
    ax = info["axes"][0]
    assert "data*1 + -273.15" in ax["y_display_transform"] and ax["ylim_displayed"] == [800.0, 2000.0]
    _, imgs = call(tb, "edit_figure", {"image": out, "operations": [
        {"op": "scatter", "x": [0.5], "y": [1300], "color": "red", "label": "sample"},
        {"op": "errorbar", "x": [0.7], "y": [1400], "yerr": [20], "label": "dsc"}]})
    fig = load_figure(imgs[0])
    assert np.allclose(_points(fig, "sample"), [[0.5, 1573.15]])


def test_ternary_edit_and_restyle_by_phase(tb):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                "temperature": 1000, "temperature_unit": "K",
                                                "style": "color"})   # phase colours + legend (restyle by phase)
    png = imgs[0]
    info = json.loads(call(tb, "describe_figure", {"image": png})[0])
    assert info["axes"][0]["projection"] == "triangular" and "SIGMA" in info["axes"][0]["legend"]
    text, imgs = call(tb, "edit_figure", {"image": png, "output": "tern_edit.png", "operations": [
        {"op": "scatter", "x": [0.4, 0.2], "y": [0.3, 0.1], "color": "black", "marker": "D", "label": "alloys"},
        {"op": "text", "x": 0.45, "y": 0.32, "text": "HEA-1", "fontsize": 9},
        {"op": "style", "match": "SIGMA", "color": "magenta", "linewidth": 2.5}]})
    assert "styled" in text and Path(imgs[0]).name == "tern_edit.png"
    from calphad_tools.figures import load_figure
    assert np.allclose(_points(load_figure(imgs[0]), "alloys"), [[0.4, 0.3], [0.2, 0.1]])


def test_raster_fallback_with_calibration(tb, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image
    img = tmp_path / "foreign.png"          # an image not made by the tools (no stored figure)
    Image.new("RGB", (800, 600), "white").save(img)
    text, _ = call(tb, "describe_figure", {"image": str(img)})
    assert "no editable copy" in text
    text, imgs = call(tb, "edit_figure", {"image": str(img), "operations": [
        {"op": "scatter", "x": [0.5], "y": [1500], "color": "red", "s": 80}],
        "calibration": {"pixel_box": [100, 50, 700, 550], "xlim": [0, 1], "ylim": [1000, 2000]}})
    assert "calibrated" in text
    arr = np.asarray(Image.open(imgs[0]).convert("RGB")).astype(int)
    ys, xs = np.where((arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80))
    assert abs(xs.mean() - 400) < 6 and abs(ys.mean() - 300) < 6        # centre of the calibrated box
    text, _ = call(tb, "edit_figure", {"image": str(img), "operations": [
        {"op": "text", "x": 0.05, "y": 0.95, "text": "fig. 3b", "va": "top"}], "output": "foreign_label.png"})
    assert "0-1 image fractions" in text


def test_make_plot_multi_panel_csv_and_formats(tb, tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"T": [300, 500, 700, 900], "cp": [25, 27, 30, 33], "err": [1, 1, 2, 2],
                  "phase": ["A", "A", "B", "B"]}).to_csv(csv, index=False)
    text, imgs = call(tb, "make_plot", {"layout": [1, 2], "output": "panels", "formats": ["pdf", "svg"],
                                        "style": "ggplot", "suptitle": "demo", "panels": [
        {"operations": [{"op": "errorbar", "csv": str(csv), "x_col": "T", "y_col": "cp", "yerr_col": "err",
                         "color": "C3", "label": "measured"},
                        {"op": "line", "x": [300, 900], "y": [25, 33], "color": "k", "linestyle": ":", "label": "fit"},
                        {"op": "set", "xlabel": "T (K)", "ylabel": "Cp (J/mol/K)"}]},
        {"operations": [{"op": "bar", "csv": str(csv), "x_col": "T", "height_col": "cp", "where": {"phase": "B"},
                         "width": 80, "color": "teal", "label": "phase B"},
                        {"op": "twinx", "ylabel": "other"},
                        {"op": "set", "yscale": "log", "axes": 1}]}]})
    base = Path(imgs[0])
    assert base.name == "panels.png" and base.with_suffix(".pdf").exists() and base.with_suffix(".svg").exists()
    assert "twin y-axis created" in text


@pytest.mark.parametrize("ops, msg", [
    ([{"op": "scatter", "x": [0.5], "y": [1300], "color": "not-a-colour"}], "not a matplotlib colour"),
    ([{"op": "wiggle"}], "unknown op"),
    ([{"op": "scatter", "x": [0.5], "y": [1], "axes": 7}], "out of range"),
    ([{"op": "style", "match": "NOPE_PHASE", "color": "red"}], "no artist matches"),
])
def test_helpful_errors(tb, ops, msg):
    _, imgs = call(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                     "output": "base.png"})
    text, _, err = tb.call("edit_figure", {"image": imgs[0], "operations": ops})
    assert err and msg in text, text


def test_make_plot_layout_styles_and_projections(tb):
    from PIL import Image
    from calphad_tools.figures import load_figure, main_axes
    text, imgs = call(tb, "make_plot", {
        "output": "layout.png", "preset": "journal-double", "palette": "okabe-ito",
        "font": {"size": 9}, "rc": {"axes.spines.top": False}, "transparent": True,
        "width_ratios": [2, 1], "panels": [
            {"row": 0, "col": 0, "rowspan": 2, "operations": [
                {"op": "line", "x": [0, 1, 2], "y": [0, 1, 4], "label": "a"},
                {"op": "line", "x": [0, 1, 2], "y": [0, 2, 3], "label": "b"},
                {"op": "twinx", "ylabel": "right"}]},
            {"row": 0, "col": 1, "projection": "polar", "operations": [
                {"op": "line", "x": [0, 1.5, 3.0], "y": [1, 2, 1]}]},
            {"row": 1, "col": 1, "projection": "triangular", "operations": [
                {"op": "scatter", "x": [0.2], "y": [0.3], "color": "k"}]}]})
    fig = load_figure(imgs[0])
    axs = main_axes(fig)
    names = sorted(a.name for a in axs)
    assert names.count("polar") == 1 and names.count("triangular") == 1
    main = [a for a in axs if a.name == "rectilinear" and a.get_ylabel() != "right"][0]
    cols = [l.get_color() for l in main.get_lines()]
    assert cols[:2] == ["#E69F00", "#56B4E9"]                       # okabe-ito cycle
    # the triangular panel got its scatter even though panel 1 created a twin axis before it
    tri = [a for a in axs if a.name == "triangular"][0]
    assert len(tri.collections) == 1
    assert Image.open(imgs[0]).mode == "RGBA"                      # transparent export


def test_plot_ops_help_and_unknown_op(tb):
    text, _ = call(tb, "plot_ops_help", {})
    assert "Common keys" not in text or "coords" in text
    assert "scatter" in text and "legend" in text
    text, _ = call(tb, "plot_ops_help", {"ops": ["errorbar", "nonsense"]})
    assert "yerr" in text and "unknown" in text
    _, imgs = call(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                     "output": "u.png"})
    text, _, err = tb.call("edit_figure", {"image": imgs[0], "operations": [{"op": "scater", "x": [1], "y": [1]}]})
    assert err and "similar" in text and "scatter" in text


def test_norms_and_bad_preset(tb):
    from calphad_tools.figures import make_norm
    import matplotlib.colors as mc
    assert isinstance(make_norm({"type": "log", "vmin": 1, "vmax": 100}), mc.LogNorm)
    assert isinstance(make_norm({"type": "diverging", "center": 0}), mc.TwoSlopeNorm)
    _, imgs = call(tb, "make_plot", {"panels": [{"operations": [
        {"op": "scatter", "x": [1, 2, 3], "y": [1, 2, 3], "c": [1, 10, 100], "norm": "log",
         "colorbar": {"label": "value", "extend": "both"}}]}], "output": "norm.png"})
    text, _, err = tb.call("make_plot", {"panels": [{"operations": []}], "output": "x.png", "preset": "tiny"})
    assert err and "presets" in text


def test_integration_fixes(tb):
    """convert as dict, binary 'other' element from the title, Arrhenius fits in log space by default,
    inset child axes, hidden legend stays hidden."""
    from calphad_tools.figures import load_figure, main_axes
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 900,
                                             "tmax": 1900, "temperature_unit": "C"})
    text, _ = call(tb, "edit_figure", {"image": imgs[0], "operations": [
        {"op": "secondary_xaxis", "convert": {"type": "mole_to_mass", "percent": True}, "label": "wt.% Ni"}]})
    assert "taken from the title: CR" in text
    T = np.array([900, 1000, 1100, 1200, 1300.])
    D = 1e-4 * np.exp(-250e3 / (8.314462618 * T)) * np.exp(np.random.default_rng(1).normal(0, 0.05, 5))
    text, imgs = call(tb, "make_plot", {"output": "arr.png", "panels": [{"operations": [
        {"op": "fit", "x": T.tolist(), "y": D.tolist(), "model": "arrhenius"},
        {"op": "line", "x": [900, 1300], "y": [1e-18, 1e-14], "label": "guide"},
        {"op": "inset", "bounds": [0.1, 0.5, 0.3, 0.3]},
        {"op": "legend", "show": False}]}]})
    ea = float(re.search(r"Ea = ([\d.]+)", text).group(1))
    assert abs(ea - 250) < 3 and "fitted ln(y)" in text
    fig = load_figure(imgs[0])
    axs = main_axes(fig)
    assert len(axs) == 2                                       # main + inset child axes
    leg = axs[0].get_legend()
    assert leg is None or not leg.get_visible()
