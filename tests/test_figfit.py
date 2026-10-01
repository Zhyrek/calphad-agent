"""Tests for calphad_tools.figfit: fit / residuals ops, statistics ops, the safe evaluator and fit_data."""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
R = 8.314462618


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


def plot(tb, ops, output="p.png", **kw):
    """make_plot with one panel per list of ops; returns (text, reloaded figure) and checks pickling."""
    from calphad_tools.figures import load_figure
    panels = [{"operations": o} for o in ops]
    text, imgs = call(tb, "make_plot", {"panels": panels, "output": output, **kw})
    fig = load_figure(imgs[0])
    assert fig is not None
    pickle.dumps(fig)
    return text, fig, imgs[0]


def axes(fig):
    from calphad_tools.figures import main_axes
    return main_axes(fig)


def num(text, name):
    m = re.search(rf"{re.escape(name)} = (-?[\d.e+-]+) ± ([\d.e+-]+|nan)", text)
    assert m, text
    return float(m.group(1)), float(m.group(2))


# ---------------------------------------------------------------- engine
def _noisy(f, x, sd, seed=1):
    return f(x) + np.random.default_rng(seed).normal(0, sd, len(x))


def test_linear_and_poly_recovery():
    from calphad_tools.figfit import build_model, fit_arrays
    x = np.linspace(0, 10, 50)
    res = fit_arrays(build_model({"model": "linear"}), x, _noisy(lambda x: 2.0 + 0.5 * x, x, 0.1))
    a, b = res.p
    assert abs(a - 2) < 0.1 and abs(b - 0.5) < 0.02 and res.r2 > 0.99 and res.dof == 48
    assert np.all(res.se > 0) and res.se[1] < 0.01
    # exact data -> R² = 1
    res = fit_arrays(build_model({"model": "poly", "deg": 2}), x, 1 - 2 * x + 0.3 * x ** 2)
    assert np.allclose(res.p, [1, -2, 0.3], atol=1e-8) and res.r2 == pytest.approx(1.0)
    res = fit_arrays(build_model({"model": "poly", "deg": 2}), x, _noisy(lambda x: 1 - 2 * x + 0.3 * x ** 2, x, 0.2))
    assert np.allclose(res.p, [1, -2, 0.3], atol=[0.2, 0.1, 0.01])
    assert "x²" in res.model.equation(res.p)


def test_arrhenius_nonlinear_and_linearized():
    from calphad_tools.figfit import build_model, fit_arrays
    T = np.linspace(900, 1400, 15)
    D0, Ea = 2e-5, 250e3
    y = D0 * np.exp(-Ea / (R * T)) * np.exp(np.random.default_rng(3).normal(0, 0.03, len(T)))
    for lin in (False, True):
        res = fit_arrays(build_model({"model": "arrhenius"}), T, y, linearize=lin)
        rep = {n: (v, s) for n, v, s, _ in res.model.report(res.p, res.se)}
        assert abs(rep["Ea"][0] - 250) < 5, rep            # kJ/mol
        assert 0 < rep["Ea"][1] < 5
        assert abs(np.log(rep["A"][0]) - np.log(D0)) < 0.5
        assert res.space == ("ln y" if lin else "y")
    # same thing given in °C and as 10000/T
    res = fit_arrays(build_model({"model": "arrhenius", "T_unit": "C"}), T - 273.15, y)
    assert abs(res.p[1] - 250) < 5
    res = fit_arrays(build_model({"model": "arrhenius", "T_unit": "10000/K"}), 1e4 / T, np.log(y), y_is="ln")
    assert abs(res.p[1] - 250) < 5


def test_custom_expr_exp_logistic_gaussian():
    from calphad_tools.figfit import build_model, fit_arrays
    x = np.linspace(1, 10, 60)
    f = lambda x: 0.3 * x ** 2 + 5 * np.exp(-2 / x)  # noqa: E731
    res = fit_arrays(build_model({"model": "custom", "expr": "a*x**2 + b*exp(-c/x)",
                                  "params": {"a": 1, "b": 1, "c": 1}}), x, _noisy(f, x, 0.05))
    assert np.allclose(res.p, [0.3, 5, 2], rtol=0.1)
    assert "exp(-(" in res.model.equation(res.p)          # numbers substituted into the expression
    # model given as a dict
    res = fit_arrays(build_model({"model": {"expr": "k*x", "params": {"k": 1}}}), x, 3 * x)
    assert res.p[0] == pytest.approx(3)
    res = fit_arrays(build_model({"model": "exp"}), x, _noisy(lambda x: 4 * np.exp(-0.3 * x), x, 0.01))
    assert np.allclose(res.p, [4, -0.3], rtol=0.05)
    xs = np.linspace(-5, 5, 80)
    res = fit_arrays(build_model({"model": "logistic", "offset": True}),
                     xs, _noisy(lambda x: 1 + 3 / (1 + np.exp(-2 * (x - 0.5))), xs, 0.02))
    assert np.allclose(res.p, [3, 2, 0.5, 1], rtol=0.1, atol=0.05)
    res = fit_arrays(build_model({"model": "gaussian"}), xs, _noisy(lambda x: 2 * np.exp(-0.5 * ((x - 1) / 0.7) ** 2),
                                                                     xs, 0.02))
    assert np.allclose(res.p, [2, 1, 0.7], rtol=0.05)
    res = fit_arrays(build_model({"model": "power"}), x, 2 * x ** 1.5)
    assert np.allclose(res.p, [2, 1.5])


def test_weights_and_bands():
    from calphad_tools.figfit import build_model, fit_arrays
    x = np.linspace(0, 10, 21)
    y = 1 + 2 * x
    y[5] += 5.0                                       # outlier with a huge uncertainty
    sig = np.full_like(x, 0.1)
    sig[5] = 100.0
    w = fit_arrays(build_model({}), x, y, sig)
    u = fit_arrays(build_model({}), x, y)
    assert abs(w.p[1] - 2) < abs(u.p[1] - 2) and abs(w.p[0] - 1) < 0.01
    assert w.chi2red is not None and "weighted" in w.stats_text()
    # confidence band: narrowest near the data centroid, prediction band wider
    rng = np.random.default_rng(0)
    y2 = 1 + 2 * x + rng.normal(0, 0.5, len(x))
    res = fit_arrays(build_model({}), x, y2)
    g = np.linspace(-5, 15, 201)
    yh, lo, hi = res.predict(g, "confidence")
    width = hi - lo
    assert abs(g[np.argmin(width)] - x.mean()) < 0.2
    assert np.all(lo < yh) and np.all(yh < hi)
    _, plo, phi = res.predict(g, "prediction")
    assert np.all(phi - plo > width)
    # 95% prediction band at the centroid ≈ ±t·s·sqrt(1 + 1/n)
    from scipy import stats
    s = np.sqrt(res.s2)
    i = np.argmin(np.abs(g - x.mean()))
    assert (phi[i] - plo[i]) / 2 == pytest.approx(stats.t.ppf(0.975, res.dof) * s * np.sqrt(1 + 1 / len(x)), rel=1e-3)
    assert res.predict(g, "confidence", 0.99)[2][i] > hi[i]


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "x.__class__", "(lambda: 1)()", "[a for a in x]", "a[0]",
    "open('f')", "np.exp(x)", "x^2", "exp(x, 2)", "exp(x=1)", "1 if x else 2", "a < x", "'s'", "exp(*x)",
    "globals()", "x" * 500, "b*x",
])
def test_safe_evaluator_rejects(expr):
    from calphad_tools.figfit import SafeExpr
    with pytest.raises(ValueError):
        SafeExpr(expr, "x", ["a"])


def test_safe_evaluator_values():
    from calphad_tools.figfit import SafeExpr
    e = SafeExpr("a*exp(-b/x) + sqrt(abs(x)) - log10(100) + pi*0 + -x**2", "x", ["a", "b"])
    x = np.array([1.0, 2.0])
    assert np.allclose(e(x, 2, 1), 2 * np.exp(-1 / x) + np.sqrt(x) - 2 - x ** 2)
    assert np.isinf(SafeExpr("10**10**10", "x", [])(x)).all()        # floats: overflow, no hang
    assert SafeExpr("a", "x", ["a"])(x, 3).shape == (2,)


# ---------------------------------------------------------------- fit / residuals ops
def test_fit_op_make_plot_with_band_annotation_and_residuals(tb):
    x = np.linspace(0, 10, 30)
    y = _noisy(lambda x: 3 + 1.5 * x - 0.1 * x ** 2, x, 0.2)
    spec = {"x": x.tolist(), "y": y.tolist(), "model": "poly", "deg": 2}
    text, fig, png = plot(tb, [
        [{"op": "fit", **spec, "band": "confidence", "show_data": True, "annotate": True, "label": True,
          "range": [-2, 12], "color": "C3"}],
        [{"op": "residuals", **spec}]], layout=[2, 1], output="fit.png")
    assert "poly2 fit" in text and "R² = 0.9" in text and "n = 30, dof = 27" in text and "95% confidence band" in text
    c0, se0 = num(text, "c0")
    assert abs(c0 - 3) < 3 * se0 + 0.1
    a0, a1 = axes(fig)
    line = [ln for ln in a0.get_lines() if ln.get_label().startswith("poly2 fit")][0]
    assert line.get_xdata()[0] == pytest.approx(-2) and line.get_xdata()[-1] == pytest.approx(12)
    assert len(a0.collections) == 2                        # band + data points
    assert any("R² =" in t.get_text() for t in a0.texts)
    offs = a1.collections[0].get_offsets()
    assert len(offs) == 30 and abs(np.mean(offs[:, 1])) < 0.1
    assert "residuals of poly2 fit" in text


def test_fit_from_csv_with_weights_and_arrhenius_annotation(tb, tmp_path):
    T = np.linspace(800, 1300, 12)
    D = 1e-4 * np.exp(-180e3 / (R * T)) * np.exp(np.random.default_rng(5).normal(0, 0.05, len(T)))
    csv = tmp_path / "diff.csv"
    pd.DataFrame({"T (K)": T, "D": D, "dD": 0.05 * D, "set": ["a"] * 6 + ["b"] * 6}).to_csv(csv, index=False)
    text, fig, _ = plot(tb, [[
        {"op": "fit", "data": str(csv), "x_col": "T (", "y_col": "D", "sigma_col": "dD", "model": "arrhenius",
         "show_data": True, "annotate": "params", "band": "prediction", "level": 0.9},
        {"op": "set", "yscale": "log"}]])
    ea, se = num(text, "Ea")
    assert abs(ea - 180) < 10 and "kJ/mol" in text and "χ²/dof" in text and "90% prediction band" in text
    ax = axes(fig)[0]
    assert any("Ea = " in t.get_text() for t in ax.texts)
    assert any(type(c).__name__ != "PathCollection" for c in ax.collections)      # band
    # where + linearized
    text, _, _ = plot(tb, [[{"op": "fit", "data": str(csv), "x_col": "T (K)", "y_col": "D", "where": {"set": "a"},
                             "model": "arrhenius", "linearize": True}]], output="lin.png")
    assert "n = 6" in text and "fitted ln y" in text


def _celsius_figure(tb, name="c.png"):
    """A stored figure whose y-axis is kelvin shown in °C (diagrams.to_celsius_axis), x = composition."""
    import matplotlib.pyplot as plt
    from calphad_tools import diagrams
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 11)
    ax.plot(x, 1273.15 + 100 * x, color="0.7")
    ax.set_ylim(1200, 1500)
    diagrams.to_celsius_axis(ax)
    out = str(tb.out / name)
    fig.savefig(out)
    plt.close(fig)
    return out


def test_fit_on_celsius_axis_is_done_in_displayed_units(tb):
    from calphad_tools.figures import ScaledFormatter, load_figure
    png = _celsius_figure(tb)
    x = np.linspace(0, 1, 11)
    ops = [{"op": "fit", "x": x.tolist(), "y": (1000 + 100 * x).tolist(), "label": "fit °C", "show_data": True}]
    text, imgs = call(tb, "edit_figure", {"image": png, "operations": ops})
    assert num(text, "intercept")[0] == pytest.approx(1000, abs=1e-6)       # °C, as displayed
    fig = load_figure(imgs[0])
    ln = [l for l in axes(fig)[0].get_lines() if l.get_label() == "fit °C"][0]
    assert np.allclose(ln.get_ydata(), 1273.15 + 100 * ln.get_xdata())     # drawn in kelvin data coordinates
    # data given in kelvin (units: data) are still fitted in °C
    text, _ = call(tb, "edit_figure", {"image": png, "output": "c2.png", "operations": [
        {"op": "fit", "x": x.tolist(), "y": (1273.15 + 100 * x).tolist(), "units": "data"}]})
    assert num(text, "intercept")[0] == pytest.approx(1000, abs=1e-6)
    pickle.dumps(fig)
    # arrhenius on an x axis in °C: T_unit defaults to C
    import matplotlib.pyplot as plt
    f2, a2 = plt.subplots()
    Tk = np.linspace(900, 1300, 10)
    a2.plot(Tk, np.zeros_like(Tk))
    a2.xaxis.set_major_formatter(ScaledFormatter(offset=-273.15))
    p2 = str(tb.out / "tc.png")
    f2.savefig(p2)
    plt.close(f2)
    y = 3.0 * np.exp(-120e3 / (R * Tk))
    text, _ = call(tb, "edit_figure", {"image": p2, "operations": [
        {"op": "fit", "x": (Tk - 273.15).tolist(), "y": y.tolist(), "model": "arrhenius"}]})
    assert num(text, "Ea")[0] == pytest.approx(120, rel=1e-4) and "T = x + 273.15 K" in text


@pytest.mark.parametrize("op, msg", [
    ({"op": "fit", "x": [1, 2], "y": [1, 2], "model": "poly", "deg": 2}, "need more than 3"),
    ({"op": "fit", "x": [1, 2, 3], "y": [1, 2, 3], "model": "banana"}, "unknown model"),
    ({"op": "fit", "x": [1, 2, 3], "y": [1, 2, 3], "model": "custom", "expr": "a*x + os", "params": {"a": 1}},
     "unknown name"),
    ({"op": "fit", "x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "linearize": True}, "linearize"),
    ({"op": "fit", "x": [1, 2, 3], "y": [1, 2, 3], "p0": {"zz": 1}}, "unknown parameter"),
    ({"op": "fit", "y": [1, 2, 3]}, "needs x and y"),
    ({"op": "kde", "values": [1, 1, 1]}, "distinct"),
    ({"op": "hexbin", "x": [1, 2], "y": [1, 2], "norm": "log", "bins": "log"}, "either norm or bins"),
])
def test_errors(tb, op, msg):
    text, _, err = tb.call("make_plot", {"panels": [{"operations": [op]}], "output": "e.png"})
    assert err and msg in text, text


def test_fit_rejects_3d(tb):
    text, _, err = tb.call("make_plot", {"panels": [{"projection": "3d", "operations": [
        {"op": "fit", "x": [1, 2, 3], "y": [1, 2, 3]}]}], "output": "e3.png"})
    assert err and "3-D" in text


# ---------------------------------------------------------------- statistics ops
@pytest.fixture
def groups_csv(tmp_path):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"alloy": ["A"] * 200 + ["B"] * 200,
                       "hv": np.r_[rng.normal(200, 10, 200), rng.normal(250, 20, 200)],
                       "ys": np.r_[rng.normal(500, 30, 200), rng.normal(600, 40, 200)],
                       "el": rng.normal(20, 3, 400)})
    df["uts"] = df["ys"] * 1.3 + rng.normal(0, 5, 400)
    p = tmp_path / "mech.csv"
    df.to_csv(p, index=False)
    return str(p), df


def test_hist_old_syntax_density_groups(tb, groups_csv):
    csv, df = groups_csv
    text, fig, _ = plot(tb, [
        [{"op": "hist", "values": [1, 2, 2, 3, 3, 3], "bins": 3, "color": "teal", "alpha": 0.5, "label": "old"}],
        [{"op": "hist", "data": csv, "values_col": "hv", "group_col": "alloy", "density": True, "bins": 25,
          "histtype": "stepfilled", "alpha": 0.5, "colors": ["C0", "C1"]}],
        [{"op": "hist", "data": csv, "value_cols": ["ys", "uts"], "stacked": True, "cumulative": True, "log": True}],
        [{"op": "hist", "values": [[1, 2, 3], [2, 3, 4, 5]], "weights": [[1, 1, 2], [1, 1, 1, 1]],
          "orientation": "horizontal", "labels": ["w1", "w2"]}]], layout=[2, 2], output="h.png")
    a0, a1, a2, a3 = axes(fig)
    heights = sorted(p.get_height() for p in a0.patches)
    assert heights == [1, 2, 3] and a0.patches[0].get_facecolor()[:3] == pytest.approx((0, 0.502, 0.502), abs=0.01)
    for poly in a1.patches:                      # stepfilled: one polygon per group, area 1
        xy = poly.get_xy()
        area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
        assert area == pytest.approx(1.0, rel=1e-6)
    assert "A: n=200" in text and "B: n=200" in text
    assert a2.get_yscale() == "log" and a2.get_legend() is not None
    assert max(p.get_width() for p in a3.patches) == pytest.approx(2.0)   # weighted, horizontal


def test_box_violin_points(tb, groups_csv):
    csv, df = groups_csv
    text, fig, png = plot(tb, [
        [{"op": "box", "data": csv, "values_col": "hv", "group_col": "alloy", "showmeans": True, "jitter": True}],
        [{"op": "violin", "groups": {"x": [1, 2, 3, 4, 5.0], "y": [2, 4, 6, 8, 10.0]}, "showmeans": True,
          "jitter": 0.1, "colors": ["C2", "C4"]}]], layout=[1, 2], output="bv.png")
    a0, a1 = axes(fig)
    meds = sorted(float(l.get_ydata()[0]) for l in a0.get_lines() if l.get_color() == "k" and len(l.get_ydata()) == 2
                  and l.get_ydata()[0] == l.get_ydata()[1])
    exp = sorted(df.groupby("alloy")["hv"].median())
    assert all(any(abs(m - e) < 1e-9 for m in meds) for e in exp)
    assert [t.get_text() for t in a0.get_xticklabels()] == ["A", "B"]
    assert len([c for c in a0.collections if len(c.get_offsets()) == 200]) == 2      # jittered points
    assert "median=" in text and "IQR" in text
    assert [t.get_text() for t in a1.get_xticklabels()] == ["x", "y"]
    pts = [c for c in a1.collections if type(c).__name__ == "PathCollection"]
    assert len(pts) == 2 and np.all(np.abs(pts[0].get_offsets()[:, 0] - 1) <= 0.1)


def test_kde_ecdf_qq(tb, groups_csv):
    csv, df = groups_csv
    rng = np.random.default_rng(11)
    v = rng.normal(5, 2.0, 400)
    text, fig, _ = plot(tb, [
        [{"op": "kde", "values": v.tolist(), "fill": True, "rug": True, "label": "kde"}],
        [{"op": "ecdf", "data": csv, "values_col": "hv", "group_col": "alloy"}],
        [{"op": "qq", "values": v.tolist()}]], layout=[1, 3], output="k.png")
    a0, a1, a2 = axes(fig)
    kl = [l for l in a0.get_lines() if l.get_label() == "kde"][0]
    assert np.trapezoid(kl.get_ydata(), kl.get_xdata()) == pytest.approx(1.0, abs=0.01)
    assert len(a0.collections) == 1 and len(a0.get_lines()) == 2            # fill + rug
    ec = a1.get_lines()
    assert len(ec) == 2 and all(l.get_ydata()[-1] == pytest.approx(1.0) for l in ec)
    m = re.search(r"slope=([\d.]+), intercept=([\d.]+)", text)
    assert abs(float(m.group(1)) - 2.0) < 0.2 and abs(float(m.group(2)) - 5) < 0.3
    assert a2.get_xlabel().startswith("theoretical quantiles")


def test_hist2d_hexbin_lognorm_colorbar(tb):
    from matplotlib.colors import LogNorm
    rng = np.random.default_rng(2)
    x, y = rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    text, fig, png = plot(tb, [
        [{"op": "hexbin", "x": x.tolist(), "y": y.tolist(), "gridsize": 20, "norm": "log", "mincnt": 1,
          "cmap": "magma", "colorbar": {"label": "count"}}],
        [{"op": "hist2d", "x": x.tolist(), "y": y.tolist(), "bins": 15, "colorbar": True, "cmin": 1}],
        [{"op": "hexbin", "x": x.tolist(), "y": y.tolist(), "c": (x + y).tolist(), "reduce": "max", "gridsize": 10}]],
        layout=[1, 3], output="hex.png")
    assert len(fig.axes) == 5                                           # 3 panels + 2 colour bars
    hb = axes(fig)[0].collections[0]
    assert isinstance(hb.norm, LogNorm) and hb.get_array().min() >= 1
    assert sum(hb.get_array()) == 3000
    assert fig.axes[-2].get_ylabel() == "count" or any(a.get_ylabel() == "count" for a in fig.axes)
    assert "hist2d: 15x15 bins, n=3000" in text


def test_band_inline_and_aggregated(tb, groups_csv, tmp_path):
    rng = np.random.default_rng(4)
    rows = [{"t": t, "rep": r, "val": 2 * t + rng.normal(0, 1), "grp": g} for t in range(10) for r in range(8)
            for g in ("u", "v")]
    p = tmp_path / "reps.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    text, fig, _ = plot(tb, [
        [{"op": "band", "x": [0, 1, 2], "y": [1, 2, 3], "yerr": [0.5, 0.5, 1], "label": "sym"},
         {"op": "band", "x": [0, 1, 2], "y": [1, 1, 1], "ylo": [0, 0.5, 0.9], "yhi": [2, 1.5, 1.1]}],
        [{"op": "band", "data": str(p), "x_col": "t", "y_col": "val", "agg": "std", "group_col": "grp"},
         {"op": "band", "data": str(p), "x_col": "t", "y_col": "val", "agg": "ci", "where": {"grp": "u"},
          "label": "ci", "color": "k"}]], layout=[1, 2], output="band.png")
    a0, a1 = axes(fig)
    fb = a0.collections[0]
    ys = fb.get_paths()[0].vertices[:, 1]
    assert ys.min() == pytest.approx(0.5) and ys.max() == pytest.approx(4.0)
    assert len(a1.get_lines()) == 3 and len(a1.collections) == 3
    d = pd.DataFrame(rows)
    mean0 = d[(d.grp == "u") & (d.t == 0)].val.mean()
    assert a1.get_lines()[0].get_ydata()[0] == pytest.approx(mean0)
    assert "mean half-width" in text


def test_corr_heatmap_and_scatter_matrix(tb, groups_csv):
    csv, df = groups_csv
    text, fig, png = plot(tb, [
        [{"op": "corr_heatmap", "data": csv, "method": "spearman"}],
        [{"op": "scatter_matrix", "data": csv, "cols": ["hv", "ys", "uts"], "group_col": "alloy", "diag": "kde"}]],
        layout=[1, 2], figsize=[12, 5], output="corr.png")
    assert "ys~uts +0.9" in text
    a0 = axes(fig)[0]
    assert [t.get_text() for t in a0.get_yticklabels()] == ["hv", "ys", "el", "uts"]
    assert len(a0.texts) == 16
    im = a0.images[0].get_array()
    assert im[1, 3] == pytest.approx(df[["ys", "uts"]].corr("spearman").iloc[0, 1])
    # heatmap axes + its colour bar + 9 scatter-matrix axes
    assert len(axes(fig)) == 1 + 9 and "target axes replaced by 9 axes, indices 1..9" in text
    grid = axes(fig)[1:]
    assert grid[3].get_ylabel() == "ys" and grid[8].get_xlabel() == "uts"
    assert len(grid[1].collections[0].get_offsets()) == 200          # (0,1): ys-vs-hv? group A points
    # corner version, then keep editing the saved figure (new axes addressable)
    text, fig, png = plot(tb, [[{"op": "scatter_matrix", "data": csv, "cols": ["hv", "ys", "el"], "corner": True,
                                 "color": "purple"}]], output="corner.png")
    assert len(axes(fig)) == 6
    text, imgs = call(tb, "edit_figure", {"image": png, "operations": [
        {"op": "fit", "axes": 1, "data": csv, "x_col": "hv", "y_col": "ys", "color": "k"}]})  # (1,0) = ys vs hv
    assert "linear fit" in text


def test_fit_data_tool(tb, tmp_path):
    x = np.linspace(1, 5, 20)
    text, _ = call(tb, "fit_data", {"x": x.tolist(), "y": (2 * x ** 1.5).tolist(), "model": "power",
                                    "output_csv": "pw.csv", "range": [0.5, 6], "npoints": 50})
    assert "A = 2 ±" in text and "b = 1.5 ±" in text and "R² = 1" in text
    cur = pd.read_csv(tb.out / "pw.csv")
    assert len(cur) == 50 and {"confidence_lo", "confidence_hi"} <= set(cur.columns)
    assert cur["y_fit"].iloc[-1] == pytest.approx(2 * 6 ** 1.5, rel=1e-6)
    resid = pd.read_csv(tb.out / "pw_residuals.csv")
    assert len(resid) == 20 and np.allclose(resid["residual"], 0, atol=1e-8)
    csv = tmp_path / "d.csv"
    pd.DataFrame({"T": [800, 900, 1000, 1100, 1200], "k": 5 * np.exp(-90e3 / (R * np.array([800, 900, 1000, 1100,
                                                                                            1200.0])))}).to_csv(csv)
    text, _ = call(tb, "fit_data", {"data": str(csv), "x_col": "T", "y_col": "k", "model": "arrhenius",
                                    "linearize": True})
    assert num(text, "Ea")[0] == pytest.approx(90, rel=1e-6) and (tb.out / "fit_arrhenius.csv").exists()


def test_help_strings_registered():
    from calphad_tools import figures
    figures.ensure_extensions()
    for name in ("fit", "residuals", "hist", "box", "violin", "kde", "ecdf", "qq", "hist2d", "hexbin", "band",
                 "corr_heatmap", "scatter_matrix"):
        assert name in figures.OPS and figures.op_help(name).startswith(name)
    assert figures.OP_GROUP["fit"] == "fit" and figures.OP_GROUP["hexbin"] == "statistics"
