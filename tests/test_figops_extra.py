"""Tests for calphad_tools.figops_extra (grid/field plots, plot types, axes & layout, annotations)."""
import pickle
from pathlib import Path

import numpy as np
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


def plot(tb, panels, output="p.png", **kw):
    text, imgs = call(tb, "make_plot", {"panels": panels, "output": output, **kw})
    return text, imgs[0], _load(imgs[0])


def _load(png):
    from calphad_tools.figures import load_figure
    fig = load_figure(png)
    assert fig is not None, "figure was not stored (not picklable?)"
    pickle.dumps(fig)
    return fig


def _axes(fig):
    from calphad_tools.figures import main_axes
    return main_axes(fig)


def _pixel(fig, ax, x, y):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.0
    px, py = ax.transData.transform((x, y))
    return img[int(round(img.shape[0] - py)), int(round(px))]


# ---------------------------------------------------------------------------
# grid & fields
# ---------------------------------------------------------------------------
def test_heatmap_long_data_annotated(tb):
    recs = [{"alloy": a, "T": t, "hv": 100 + 10 * i + t / 100} for i, a in enumerate(["A1", "B2", "C3"])
            for t in (300, 600, 900, 1200)]
    text, png, fig = plot(tb, [{"operations": [
        {"op": "heatmap", "data": recs, "x_col": "alloy", "y_col": "T", "z_col": "hv", "annotate": True,
         "fmt": "{:.0f}", "cmap": "magma", "colorbar": {"label": "HV"}}]}])
    ax = _axes(fig)[0]
    im = ax.images[0]
    arr = im.get_array()
    assert arr.shape == (4, 3) and arr[0, 0] == pytest.approx(103) and arr[3, 2] == pytest.approx(132)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A1", "B2", "C3"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["300", "600", "900", "1200"]
    texts = {t.get_text(): t.get_color() for t in ax.texts}
    assert len(ax.texts) == 12 and texts["103"] == "white" and texts["132"] == "black"   # contrast on magma
    assert im.origin == "lower"
    assert any(a.get_label() == "<colorbar>" for a in fig.axes)


def test_contourf_long_data_and_legacy_inline_contour(tb):
    xs, ys = np.linspace(0, 1, 11), np.linspace(0, 2, 9)
    recs = [{"x": x, "y": y, "z": x * y} for x in xs for y in ys]
    z2 = [[float(x + y) for x in xs] for y in ys]
    text, png, fig = plot(tb, [
        {"operations": [{"op": "contourf", "data": recs, "x_col": "x", "y_col": "y", "z_col": "z", "levels": 8,
                         "cmap": "viridis", "lines": True, "clabel": True, "colorbar": True}]},
        {"operations": [{"op": "contour", "x": xs.tolist(), "y": ys.tolist(), "z": z2, "levels": [0.5, 1, 1.5, 2],
                         "colors": "k", "linestyles": "--"}]}], layout=[1, 2])
    a0, a1 = _axes(fig)[:2]
    from matplotlib.contour import ContourSet
    cs0 = [c for c in a0.collections if isinstance(c, ContourSet)]
    assert len(cs0) == 2 and cs0[0].filled and not cs0[1].filled
    assert cs0[0].zmax == pytest.approx(2.0) and len(a0.texts) > 0            # z = x*y on [0,1]x[0,2]; clabels
    cs1 = [c for c in a1.collections if isinstance(c, ContourSet)][0]
    assert list(cs1.levels) == [0.5, 1, 1.5, 2] and len(a1.texts) > 0         # old syntax: clabel default on


def test_tricontourf_on_triangular_panel(tb):
    rng = np.random.default_rng(0)
    pts = rng.random((400, 2))
    xb, xc = pts[:, 0], pts[:, 1]                    # includes points outside the triangle (dropped)
    z = xb + 2 * xc
    text, png, fig = plot(tb, [{"projection": "triangular", "operations": [
        {"op": "tricontourf", "x": xb.tolist(), "y": xc.tolist(), "z": z.tolist(), "levels": 40, "cmap": "viridis",
         "vmin": 0, "vmax": 2, "colorbar": {"label": "f"}},
        {"op": "tricontour", "x": xb.tolist(), "y": xc.tolist(), "z": z.tolist(), "levels": [0.5, 1.0],
         "colors": "w"}]}], figsize=[6, 5])
    assert "outside the composition triangle" in text
    ax = _axes(fig)[0]
    assert ax.name == "triangular"
    from matplotlib.tri import TriContourSet
    cs = [c for c in ax.collections if isinstance(c, TriContourSet)]
    assert len(cs) == 2 and cs[0].filled and cs[0].zmax <= 2.0 + 1e-9    # max of z inside the triangle is 2
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    for (b, c) in [(0.1, 0.1), (0.7, 0.1), (0.1, 0.7), (0.3, 0.3)]:
        got = _pixel(fig, ax, b, c)
        want = np.array(cmap((b + 2 * c) / 2.0))[:3]
        assert np.abs(got - want).max() < 0.12, (b, c, got, want)


def test_pcolormesh_tripcolor_quiver_smoke(tb):
    x, y = np.linspace(0, 1, 6), np.linspace(0, 1, 5)
    z = np.add.outer(y, x)
    text, png, fig = plot(tb, [
        {"operations": [{"op": "pcolormesh", "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                         "norm": {"type": "power", "gamma": 0.5}, "colorbar": True}]},
        {"operations": [{"op": "tripcolor", "x": [0, 1, 0, 1, 0.5], "y": [0, 0, 1, 1, 0.5], "z": [0, 1, 1, 2, 1],
                         "shading": "gouraud"}]},
        {"operations": [{"op": "quiver", "x": x.tolist(), "y": y.tolist(), "u": np.ones_like(z).tolist(),
                         "v": z.tolist(), "c": z.ravel().tolist(), "key": {"U": 1, "label": "1 unit"}}]}],
        layout=[1, 3])
    a0, a1, a2 = _axes(fig)[:3]
    assert type(a0.collections[0]).__name__ == "QuadMesh"
    assert type(a1.collections[0]).__name__ == "TriMesh"
    assert type(a2.collections[0]).__name__ == "Quiver"


# ---------------------------------------------------------------------------
# plot types
# ---------------------------------------------------------------------------
def test_stackplot_phase_fractions_sum_to_one(tb):
    T = np.arange(800, 1601, 100)
    liq = np.clip((T - 1200) / 400, 0, 1)
    fcc = (1 - liq) * 0.7
    recs = [{"T": t, "phase": p, "NP": v} for t, l, f in zip(T, liq, fcc)
            for p, v in (("LIQUID", l), ("FCC_A1", f), ("BCC_A2", 1 - l - f), ("SIGMA", 0.0))]
    text, png, fig = plot(tb, [{"operations": [
        {"op": "stackplot", "data": recs, "x_col": "T", "group_col": "phase", "value_col": "NP",
         "colors": {"LIQUID": "tab:blue", "FCC_A1": "tab:orange"}, "alpha": 0.9},
        {"op": "set", "xlabel": "T (K)", "ylabel": "phase fraction"}]}])
    ax = _axes(fig)[0]
    polys = [c for c in ax.collections if type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection")]
    assert [p.get_label() for p in polys] == ["LIQUID", "FCC_A1", "BCC_A2"]       # all-zero SIGMA dropped
    top = polys[-1].get_paths()[0].vertices
    upper = [top[np.isclose(top[:, 0], t), 1].max() for t in T]
    assert np.allclose(upper, 1.0)                                               # stack sums to 1 everywhere
    assert np.allclose(polys[0].get_facecolor()[0][:3], (0.1216, 0.4667, 0.7059), atol=1e-3)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["LIQUID", "FCC_A1", "BCC_A2"]


def test_bars_grouped_and_stacked(tb):
    cats = ["Cr", "Ni", "Fe"]
    series = {"model": [1.0, 2.0, 3.0], "exp": [1.5, 1.8, 2.5]}
    recs = [{"el": c, "src": s, "v": v} for s, vals in series.items() for c, v in zip(cats, vals)]
    text, png, fig = plot(tb, [
        {"operations": [{"op": "bars", "categories": cats, "series": series, "value_labels": "{:.1f}",
                         "errors": {"exp": [0.1, 0.1, 0.2]}}]},
        {"operations": [{"op": "bars", "data": recs, "category_col": "el", "group_col": "src", "value_col": "v",
                         "mode": "stacked", "colors": ["#444444", "#bbbbbb"]}]},
        {"operations": [{"op": "bars", "categories": cats, "series": series, "horizontal": True}]}], layout=[1, 3])
    a0, a1, a2 = _axes(fig)[:3]
    r0 = [p for p in a0.patches if type(p).__name__ == "Rectangle"]
    assert len(r0) == 6 and r0[0].get_width() == pytest.approx(0.4) and r0[3].get_height() == pytest.approx(1.5)
    assert [t.get_text() for t in a0.get_xticklabels()] == cats
    assert "3.0" in [t.get_text() for t in a0.texts]                               # value labels
    r1 = [p for p in a1.patches if type(p).__name__ == "Rectangle"]
    tops = sorted(round(p.get_y() + p.get_height(), 6) for p in r1[3:])
    assert tops == sorted([2.5, 3.8, 5.5])                                       # stacked totals
    assert [t.get_text() for t in a1.get_legend().get_texts()] == ["model", "exp"]
    r2 = [p for p in a2.patches if type(p).__name__ == "Rectangle"]
    assert r2[2].get_width() == pytest.approx(3.0)                               # horizontal: widths = values


def test_pie_donut(tb):
    text, png, fig = plot(tb, [{"operations": [
        {"op": "pie", "values": [60, 30, 10], "labels": ["FCC", "BCC", "SIGMA"], "autopct": True, "width": 0.4,
         "colors": "tol-muted" if False else ["#88CCEE", "#CC6677", "#DDCC77"], "center_text": "1000 K"}]}])
    ax = _axes(fig)[0]
    wedges = [p for p in ax.patches if type(p).__name__ == "Wedge"]
    assert len(wedges) == 3 and wedges[0].width == pytest.approx(0.4)
    txt = [t.get_text() for t in ax.texts]
    assert "60.0%" in txt and "FCC" in txt and "1000 K" in txt
    assert ax.get_legend() is None or not ax.get_legend().get_visible()


def test_radar_on_polar_and_error_on_cartesian(tb):
    ops = [{"op": "radar", "categories": ["YS", "UTS", "El", "HV", "K1c"],
            "series": {"alloy A": [0.8, 0.7, 0.4, 0.9, 0.5], "alloy B": [0.5, 0.6, 0.9, 0.4, 0.8]}, "rlim": [0, 1]}]
    text, png, fig = plot(tb, [{"projection": "polar", "operations": ops}])
    ax = _axes(fig)[0]
    lines = [ln for ln in ax.get_lines() if ln.get_label() in ("alloy A", "alloy B")]
    assert len(lines) == 2 and len(lines[0].get_xdata()) == 6                     # closed polygon
    assert [t.get_text() for t in ax.get_xticklabels()] == ["YS", "UTS", "El", "HV", "K1c"]
    text, _, err = tb.call("make_plot", {"panels": [{"operations": ops}], "output": "bad.png"})
    assert err and "polar" in text


def test_surface_on_3d_and_clear_error_on_2d(tb):
    x, y = np.linspace(-1, 1, 15), np.linspace(-1, 1, 12)
    z = np.exp(-np.add.outer(y ** 2, x ** 2)).tolist()
    text, png, fig = plot(tb, [{"projection": "3d", "operations": [
        {"op": "surface", "x": x.tolist(), "y": y.tolist(), "z": z, "cmap": "coolwarm", "colorbar": True},
        {"op": "scatter3d", "x": [0], "y": [0], "z": [1.1], "color": "k", "label": "peak"},
        {"op": "line3d", "x": [-1, 1], "y": [-1, 1], "z": [0, 0], "color": "r"},
        {"op": "view", "elev": 25, "azim": -60, "zlabel": "G"}]}])
    ax = _axes(fig)[0]
    assert ax.name == "3d" and ax.elev == 25 and ax.azim == -60 and ax.get_zlabel() == "G"
    assert any(type(c).__name__ == "Poly3DCollection" for c in ax.collections)
    text, _, err = tb.call("make_plot", {"panels": [{"operations": [
        {"op": "surface", "x": x.tolist(), "y": y.tolist(), "z": z}]}], "output": "bad3d.png"})
    assert err and "3-D" in text and "projection" in text


# ---------------------------------------------------------------------------
# axes & layout
# ---------------------------------------------------------------------------
def test_secondary_axes_on_phase_diagram_and_reciprocal(tb):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1000,
                                                "tmax": 2200, "composition_step": 0.05, "temperature_step": 20})
    text, imgs = call(tb, "edit_figure", {"image": imgs[0], "output": "crni_wt.png", "operations": [
        {"op": "secondary_xaxis", "convert": "mole_to_mass", "other": "CR", "percent": True,
         "label": "Mass percent Ni"},
        {"op": "secondary_yaxis", "convert": "linear", "offset": -273.15, "label": "T (°C)"}]})
    assert "mole_to_mass for Ni" in text
    fig = _load(imgs[0])
    ax = _axes(fig)[0]
    secs = [c for c in ax.child_axes if type(c).__name__ == "SecondaryAxis"]
    assert len(secs) == 2
    fwd, inv = secs[0]._functions
    # x(Ni)=0.5 -> w(Ni) = 58.693 / (58.693 + 51.996) = 53.02 wt.%
    assert float(fwd(0.5)) == pytest.approx(53.025, abs=0.01)
    assert float(inv(fwd(0.3))) == pytest.approx(0.3) and float(fwd(0.0)) == 0 and float(fwd(1.0)) == 100
    assert np.all(np.diff(fwd(np.linspace(-0.1, 1.1, 50))) > 0)                   # monotonic incl. margins
    assert float(secs[1]._functions[0](1273.15)) == pytest.approx(1000.0)
    # reciprocal axis: 1000/T on top of a T axis, and reload + edit again
    text, png, fig = plot(tb, [{"operations": [
        {"op": "line", "x": [500, 1000, 1500], "y": [1, 2, 3]},
        {"op": "secondary_xaxis", "convert": "reciprocal", "k": 1000, "label": "1000/T (1/K)"}]}], output="rec.png")
    sec = [c for c in _axes(fig)[0].child_axes if type(c).__name__ == "SecondaryAxis"][0]
    assert float(sec._functions[0](500.0)) == pytest.approx(2.0) and float(sec._functions[1](2.0)) == pytest.approx(500)
    text, imgs = call(tb, "edit_figure", {"image": png, "operations": [{"op": "twiny", "xlabel": "other"}]})
    assert "twin x-axis created as axes 1" in text
    _load(imgs[0])


def test_ticks_formats_and_celsius_axis(tb, tmp_path):
    import matplotlib.pyplot as plt
    from calphad_tools.figures import ScaledFormatter
    fig, ax = plt.subplots()
    ax.plot([1000, 1500], [0, 1])
    ax.set_xlim(1000, 1500)
    ax.xaxis.set_major_formatter(ScaledFormatter(offset=-273.15, fmt="{:.0f}"))
    png = str(tb.out / "kelvin.png")
    fig.savefig(png)
    plt.close(fig)
    text, imgs = call(tb, "edit_figure", {"image": png, "operations": [
        {"op": "ticks", "axis": "x", "step": 100, "format": "{x:.1f}", "minor": True, "direction": "in",
         "mirror": True},
        {"op": "ticks", "axis": "y", "format": "percent", "nbins": 4, "grid": "both", "rotation": 30},
        {"op": "spines", "hide": ["right"], "offset": {"left": 4}, "linewidth": 1.5}]})
    fig = _load(imgs[0])
    ax = _axes(fig)[0]
    fmt = ax.xaxis.get_major_formatter()
    assert isinstance(fmt, ScaledFormatter) and fmt.offset == -273.15                # display transform kept
    locs = ax.xaxis.get_major_locator()()
    shown = locs - 273.15
    assert np.allclose(shown, np.round(shown / 100) * 100) and len(locs) >= 4        # round °C ticks
    assert fmt(1273.15) == "1000.0"
    assert ax.yaxis.get_major_formatter()(0.5) == "50%"
    assert not ax.spines["right"].get_visible()
    text, _, err = tb.call("edit_figure", {"image": png, "operations": [{"op": "ticks", "axis": "x", "format": "eng"}]})
    assert err and "display transform" in text
    text, png2, fig = plot(tb, [{"operations": [{"op": "line", "x": [1, 2], "y": [1e-5, 3e-5]},
                                                {"op": "ticks", "axis": "y", "format": "sci"},
                                                {"op": "ticks", "axis": "x", "format": "{x:.2f}", "step": 0.25}]}])
    ax = _axes(fig)[0]
    assert ax.xaxis.get_major_formatter()(1.5) == "1.50"


def test_panel_labels_2x2_and_shared_colorbar(tb):
    z = np.arange(12.0).reshape(3, 4)
    panels = [{"operations": [{"op": "heatmap", "z": (z * (i + 1)).tolist()}]} for i in range(4)]
    panels[-1]["operations"] += [
        {"op": "panel_labels"},
        {"op": "colorbar", "axes_list": [0, 1, 2, 3], "label": "value", "extend": "max"}]
    text, png, fig = plot(tb, panels, layout=[2, 2])
    axs = _axes(fig)
    assert len(axs) == 4
    labs = [t.get_text() for a in axs for t in a.texts]
    assert labs == ["(a)", "(b)", "(c)", "(d)"]
    cbs = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cbs) == 1
    for a in axs:
        assert a.images[0].get_clim() == (0.0, 44.0)                               # synced limits
    cb = cbs[0].get_position()
    assert all(cb.x0 > a.get_position().x1 for a in axs)                          # no overlap after tight_layout
    assert cb.y0 <= min(a.get_position().y0 for a in axs) + 1e-6
    text, png2, fig2 = plot(tb, [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]},
                                 {"operations": [{"op": "line", "x": [0, 1], "y": [1, 0]}, {"op": "twinx"},
                                                 {"op": "panel_labels", "uppercase": True, "parentheses": False,
                                                  "position": "top left"}]}], layout=[1, 2], output="pl2.png")
    assert [t.get_text() for a in _axes(fig2) for t in a.texts] == ["A", "B"]    # twin axes not labelled


def test_broken_yaxis(tb):
    text, png, fig = plot(tb, [{"operations": [
        {"op": "line", "x": [0, 1, 2, 3], "y": [1, 2, 1.5, 100], "marker": "o", "label": "data"},
        {"op": "bar", "x": [0.5], "height": [1.0], "width": 0.2, "color": "grey"},
        {"op": "set", "ylabel": "signal", "title": "broken"},
        {"op": "broken_yaxis", "ylims": [[0, 3], [95, 105]], "ratio": [2, 1]},
        {"op": "text", "x": 1, "y": 2.5, "text": "lower"}]}])
    assert "upper part is axes 1" in text
    lower, upper = _axes(fig)[:2]
    assert lower.get_ylim() == pytest.approx((0, 3)) and upper.get_ylim() == pytest.approx((95, 105))
    assert upper.get_position().y0 > lower.get_position().y1
    assert [ln.get_label() for ln in upper.get_lines() if not ln.get_label().startswith("_")] == ["data"]
    assert upper.get_title() == "broken" and upper.get_legend().get_visible()
    assert any(type(p).__name__ == "Rectangle" for p in upper.patches)


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------
def test_annotate_points_avoids_overlaps(tb):
    x = [0.50, 0.505, 0.51, 0.2, 0.8]
    y = [0.50, 0.505, 0.51, 0.3, 0.7]
    labs = ["alloy-1", "alloy-2", "alloy-3", "B", "C"]
    text, png, fig = plot(tb, [{"operations": [
        {"op": "scatter", "x": x, "y": y},
        {"op": "annotate_points", "data": [{"x": a, "y": b, "name": n} for a, b, n in zip(x, y, labs)],
         "x_col": "x", "y_col": "y", "labels_col": "name", "fontsize": 9},
        {"op": "set", "xlim": [0, 1], "ylim": [0, 1]}]}])
    ax = _axes(fig)[0]
    assert sorted(t.get_text() for t in ax.texts) == sorted(labs)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (ax.texts[i].get_text(), ax.texts[j].get_text())


def test_label_lines(tb):
    x = np.linspace(0, 10, 50)
    text, png, fig = plot(tb, [{"operations": [
        {"op": "line", "x": x.tolist(), "y": (x * 1.0).tolist(), "color": "tab:red", "label": "FCC"},
        {"op": "line", "x": x.tolist(), "y": (x * 1.02).tolist(), "color": "tab:blue", "label": "BCC"},
        {"op": "line", "x": x.tolist(), "y": (0.1 * x).tolist(), "color": "tab:green", "label": "LIQ"},
        {"op": "label_lines", "legend": False},
        {"op": "label_lines", "match": "LIQ", "x": 5, "fontsize": 7}]}])
    ax = _axes(fig)[0]
    import matplotlib.colors as mc
    col = {t.get_text(): mc.to_hex(t.get_color()) for t in ax.texts if t.get_fontsize() == 9}
    assert col == {"FCC": mc.to_hex("tab:red"), "BCC": mc.to_hex("tab:blue"), "LIQ": mc.to_hex("tab:green")}
    liq5 = [t for t in ax.texts if t.get_fontsize() == 7][0]
    assert liq5.xy == pytest.approx((5.0, 0.5))
    assert ax.get_legend() is None or not ax.get_legend().get_visible()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ends = [t for t in ax.texts if t.get_text() in ("FCC", "BCC")]
    assert not ends[0].get_window_extent(r).overlaps(ends[1].get_window_extent(r))


def test_help_lists_new_ops(tb):
    text, _ = call(tb, "plot_ops_help", {"ops": ["heatmap", "secondary_xaxis", "bars", "panel_labels"]})
    assert "mole_to_mass" in text and "grouped|stacked" in text and "unknown" not in text
    text, _ = call(tb, "plot_ops_help", {})
    assert "grid & fields" in text and "tricontourf" in text
