"""show_code generators of the calculation tools (calphad_tools/codegen_calc.py).

Every test runs the REAL tool, generates the standalone code for its output, executes that code in a subprocess
(pycalphad/numpy/pandas/scipy/matplotlib only) and compares the numbers: CSV files column by column, and the data
of the plotted artists against the tool's stored figure.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
KELVIN = 273.15


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
ARTISTS = '''
def _artists(fig):
    """Plotted data of every (non-colorbar) axes: line xy, line-collection segments, scatter offsets."""
    import numpy as _np
    from matplotlib.collections import LineCollection as _LC
    out = []
    for a in fig.axes:
        if a.get_label() == "<colorbar>":
            continue
        lines = [[_np.asarray(ln.get_xdata(), float).tolist(), _np.asarray(ln.get_ydata(), float).tolist()]
                 for ln in a.get_lines()]
        segs, pts = [], []
        for c in a.collections:
            if isinstance(c, _LC):
                segs += [_np.asarray(s, float).tolist() for s in c.get_segments()]
            elif type(c).__name__ == "PathCollection":
                o = _np.asarray(c.get_offsets(), float)
                pts += o.tolist() if o.size else []
        out.append({"lines": lines, "segs": segs, "pts": pts})
    return out
'''
exec(ARTISTS)  # noqa: S102 - the same extraction runs in the test process and in the generated script

DUMP = ARTISTS + '''
import json as _json
_json.dump(_artists(fig), open("artists.json", "w"))
'''


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def standalone(tb, target="1"):
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, target)
    assert all(c.standalone for c in cells), [(c.title, c.note) for c in cells]
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    return script


def run(script, tmp_path, dump=True):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "gen.py").write_text(script + (DUMP if dump else ""))
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    art = json.loads((d / "artists.json").read_text()) if dump else None
    return d / "standalone_outputs", r.stdout, art


def tool_artists(png):
    from calphad_tools.figures import load_figure
    fig = load_figure(str(png))
    assert fig is not None, png
    return _artists(fig)


def frames_equal(a_path, b_path, rtol=1e-6, atol=1e-9, skip=()):
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    assert list(a.columns) == list(b.columns), (list(a.columns), list(b.columns))
    assert a.shape == b.shape, (a.shape, b.shape)
    for c in a.columns:
        if c in skip:
            continue
        if a[c].dtype.kind in "fi" and b[c].dtype.kind in "fi":
            assert np.allclose(a[c], b[c], rtol=rtol, atol=atol, equal_nan=True), c
        else:
            assert (a[c].fillna("").astype(str) == b[c].fillna("").astype(str)).all(), c
    return a, b


def lines_equal(A, B, dy=0.0, rtol=1e-7, atol=1e-9):
    """Same plotted lines (as multisets); B's y values shifted by dy before comparing."""
    left = [(np.asarray(x, float), np.asarray(y, float) + dy) for x, y in B]
    assert len(A) == len(left), (len(A), len(left))
    for x, y in A:
        x, y = np.asarray(x, float), np.asarray(y, float)
        hit = next((k for k, (bx, by) in enumerate(left) if bx.shape == x.shape and
                    np.allclose(bx, x, rtol=rtol, atol=atol, equal_nan=True) and
                    np.allclose(by, y, rtol=rtol, atol=atol, equal_nan=True)), None)
        assert hit is not None, (x[:5], y[:5])
        left.pop(hit)


def points_near(A, B, scale=(1.0, 1.0), tol=0.02, mapping=False):
    """Every point of A lies within tol of a point of B (coordinates divided by scale).

    mapping=True: pycalphad's mapping depends slightly on Python's hash seed (set iteration order), so a run can
    trace an extra short line segment; then 98 % of the points must be within tol and all within 5 tol."""
    A = np.asarray(A, float) / scale
    B = np.asarray(B, float) / scale
    A, B = A[np.isfinite(A).all(1)], B[np.isfinite(B).all(1)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    if mapping:
        assert np.quantile(d, 0.98) < tol and d.max() < 5 * tol, (np.quantile(d, 0.98), d.max())
    else:
        assert d.max() < tol, d.max()


def line_points(art):
    return [p for x, y in art["lines"] for p in zip(x, y)]


def boundaries_near(a, b, xcol, tcol="T_K", tol=0.02):
    """Boundary points of every phase of a lie on the curves of the same phase in b (mapping may trace extra
    duplicate lines depending on the hash seed, so rows are compared by proximity), both ways."""
    assert set(a.phase) == set(b.phase)
    for ph in set(a.phase):
        pa = a[a.phase == ph][[xcol, tcol]].to_numpy() / [1, 1000]
        pb = b[b.phase == ph][[xcol, tcol]].to_numpy() / [1, 1000]
        points_near(pa, pb, tol=tol, mapping=True)
        points_near(pb, pa, tol=tol, mapping=True)


# ---------------------------------------------------------------------------------------------------------
# map_phase_diagram
# ---------------------------------------------------------------------------------------------------------
def test_map_binary_celsius(tb, tmp_path):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                             "tmax": 2000, "temperature_unit": "C", "composition_step": 0.05,
                                             "temperature_step": 20})
    script = standalone(tb, imgs[0])
    assert "BinaryStrategy" in script and "B2_BCC" in script          # disordered-state renaming kept
    out, stdout, art = run(script, tmp_path)
    png = Path(imgs[0])
    assert (out / png.name).exists()
    stem = png.stem
    a, b = [pd.read_csv(p / f"{stem}_boundaries.csv") for p in (png.parent, out)]
    boundaries_near(a, b, "X(NI)")
    ia, ib = frames_equal(png.parent / f"{stem}_invariants.csv", out / f"{stem}_invariants.csv", rtol=1e-6)
    assert list(ia.type) == ["eutectic"] * 3 and abs(ia.T_K[0] - 1617.855) < 0.5
    # figure: same boundary curves, plotted in °C and mole fraction (the displayed units)
    ta = tool_artists(png)
    points_near(line_points(art[0]), line_points(ta[0]), scale=(1, 1000), tol=0.02, mapping=True)
    points_near(line_points(ta[0]), line_points(art[0]), scale=(1, 1000), tol=0.02, mapping=True)
    assert max(p[1] for p in line_points(art[0])) < 2001            # °C as data


def test_map_binary_weight_percent_nodes(tb, tmp_path):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "AlCoCrNi_Liu2016.tdb", "elements": ["NI", "AL"], "tmin": 800,
                                             "tmax": 1800, "composition_step": 0.05, "temperature_step": 20,
                                             "composition_unit": "weight_percent", "label_nodes": True,
                                             "tielines": 0})
    out, _, art = run(standalone(tb, imgs[0]), tmp_path)
    png = Path(imgs[0])
    a, b = [pd.read_csv(p / f"{png.stem}_boundaries.csv") for p in (png.parent, out)]
    assert "W(AL)" in b.columns
    boundaries_near(a, b, "X(AL)")
    ia, ib = [pd.read_csv(p / f"{png.stem}_invariants.csv") for p in (png.parent, out)]
    assert np.allclose(sorted(ia.T_K.unique()), sorted(ib.T_K.unique()), atol=0.5)
    assert sorted(set(zip(ia.type, ia.reaction_on_cooling))) == sorted(set(zip(ib.type, ib.reaction_on_cooling)))
    ta = tool_artists(png)
    # weight percent on x, K on y; invariant node markers present
    points_near(line_points(art[0]), line_points(ta[0]), scale=(100, 1000), tol=0.02, mapping=True)
    assert len(art[0]["pts"]) >= 3 * 6 and max(p[0] for p in line_points(art[0])) > 50


def test_map_ternary_percent(tb, tmp_path):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                             "temperature": 1200, "temperature_unit": "C", "composition_step": 0.05,
                                             "composition_unit": "mole_percent"})
    script = standalone(tb, imgs[0])
    assert "TernaryStrategy" in script and "projection=\"triangular\"" in script
    out, _, art = run(script, tmp_path)
    png = Path(imgs[0])
    for suffix in ("_boundaries.csv", "_invariants.csv", "_tielines.csv"):
        frames_equal(png.parent / (png.stem + suffix), out / (png.stem + suffix), rtol=1e-6)
    ta = tool_artists(png)
    lines_equal(ta[0]["lines"], art[0]["lines"])
    assert np.allclose(np.asarray(ta[0]["segs"]), np.asarray(art[0]["segs"]))


def test_map_isopleth(tb, tmp_path):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                             "vary": "CR", "fixed": {"NI": 0.1}, "tmin": 900, "tmax": 1900,
                                             "composition_step": 0.05, "temperature_step": 20})
    script = standalone(tb, imgs[0])
    assert "IsoplethStrategy" in script
    out, _, art = run(script, tmp_path)
    png = Path(imgs[0])
    a, b = [pd.read_csv(p / f"{png.stem}_zpf_lines.csv") for p in (png.parent, out)]
    for ph in set(a.zero_phase) | set(b.zero_phase):
        pa = a[a.zero_phase == ph][["X(CR)", "T_K"]].to_numpy() / [1, 1000]
        pb = b[b.zero_phase == ph][["X(CR)", "T_K"]].to_numpy() / [1, 1000]
        points_near(pa, pb, mapping=True)
        points_near(pb, pa, mapping=True)
    ta = tool_artists(png)
    points_near(line_points(art[0]), line_points(ta[0]), scale=(1, 1000), mapping=True)


def test_map_potential_axis_falls_back(tb, tmp_path):
    from calphad_tools import codegen
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1400,
                                             "tmax": 1800, "composition_step": 0.1, "temperature_step": 40,
                                             "potential_axis": "NI"})
    cells, _ = codegen.build_cells(tb, imgs[0])
    assert not cells[0].standalone and "potential_axis" in cells[0].note


# ---------------------------------------------------------------------------------------------------------
# step_map
# ---------------------------------------------------------------------------------------------------------
def test_step_map_temperature(tb, tmp_path):
    _, imgs = call(tb, "step_map", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "vary": "T", "start": 1000,
                                    "stop": 1500, "step": 10, "fixed": {"X(NI)": 0.45}, "temperature_unit": "C",
                                    "plot": ["NP", "X(*,CR)", "hm"]})
    out, stdout, art = run(standalone(tb, imgs[0]), tmp_path)
    png = Path(imgs[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    assert "BCC_A2 + FCC_A1 -> BCC_A2 + LIQUID" in stdout
    ta = tool_artists(png)
    for p in range(3):
        lines_equal(ta[p]["lines"], art[p]["lines"])


def test_step_map_composition(tb, tmp_path):
    _, imgs = call(tb, "step_map", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                    "vary": "X(CR)", "start": 0.1, "stop": 0.6, "step": 0.01,
                                    "fixed": {"T": 1400, "X(NI)": 0.2}, "properties": ["GM", "DOO"]})
    out, _, art = run(standalone(tb, imgs[0]), tmp_path)
    png = Path(imgs[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    lines_equal(tool_artists(png)[0]["lines"], art[0]["lines"])


# ---------------------------------------------------------------------------------------------------------
# equilibrium_property_plot / export_equilibrium_data
# ---------------------------------------------------------------------------------------------------------
def test_equilibrium_property_1d(tb, tmp_path):
    _, imgs = call(tb, "equilibrium_property_plot", {
        "tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "conditions": {"T": [1200, 1500, 20], "X(NI)": 0.45},
        "quantities": ["NP", "X(*,CR)", "HM[kJ/mol]", "HM.T"], "temperature_unit": "C"})
    script = standalone(tb, imgs[0])
    assert "Workspace(" in script
    out, _, art = run(script, tmp_path)
    png = Path(imgs[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    ta = tool_artists(png)
    assert len(ta) == len(art) == 4
    for p in range(4):
        lines_equal(ta[p]["lines"], art[p]["lines"])


def test_equilibrium_property_meta_quantities(tb, tmp_path):
    _, imgs = call(tb, "equilibrium_property_plot", {
        "tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "conditions": {"T": [1000, 1600, 50], "X(NI)": 0.45},
        "quantities": ["DF(LIQUID)", "ISOLATED(FCC_A1):GM", "T0(FCC_A1,BCC_A2)", "DORMANT(LIQUID):X(NI)"],
        "exclude_phases": ["LIQUID"]})
    script = standalone(tb, imgs[0])
    assert "DormantPhase" in script and "IsolatedPhase" in script and "T0(" in script
    out, _, art = run(script, tmp_path)
    png = Path(imgs[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name, rtol=1e-6)
    for ta, ga in zip(tool_artists(png), art):
        lines_equal(ta["lines"], ga["lines"], rtol=1e-6)


def test_equilibrium_property_activity_and_2d(tb, tmp_path):
    call(tb, "equilibrium_property_plot", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                           "conditions": {"T": 1400, "X(NI)": [0.05, 0.95, 0.1]},
                                           "quantities": ["ACR(CR)", "MUR(NI,FCC_A1)", "MU(NI)"]})
    _, imgs = call(tb, "equilibrium_property_plot", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                     "conditions": {"T": 1200, "X(CR)": [0.1, 0.6, 0.1],
                                                                    "X(NI)": [0.1, 0.6, 0.1]},
                                                     "quantities": ["NP(*)"], "temperature_unit": "C"})
    s1 = standalone(tb, "1")
    assert "pure_element_gm" in s1 and "'BCC_A2'" in s1          # SER reference phase resolved at generation time
    out, _, _ = run(s1, tmp_path)
    csv1 = next(Path(tb.out).glob("eqprop_CR-NI_ACR*.csv"))
    a, _ = frames_equal(csv1, out / csv1.name)
    assert "ACR(CR,BCC_A2)" in a.columns
    out, _, art = run(standalone(tb, imgs[0]), tmp_path)
    png = Path(imgs[0])
    a, _ = frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    assert (a.stable_phases == "(X sum > 1)").sum() > 0
    assert len(art) == len(tool_artists(png))


def test_equilibrium_property_single_point_mass_fraction(tb, tmp_path):
    call(tb, "equilibrium_property_plot", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                           "conditions": {"T": 1400, "W(NI)": 0.3}, "quantities": ["NP(*)", "CPM"]})
    script = standalone(tb, "1")
    assert "mass fractions converted to mole fractions" in script
    out, stdout, _ = run(script, tmp_path, dump=False)
    csv = next(Path(tb.out).glob("eqprop_*.csv"))
    frames_equal(csv, out / csv.name)


def test_export_equilibrium_data_eqplot_celsius(tb, tmp_path):
    _, imgs = call(tb, "export_equilibrium_data", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                   "conditions": {"T": [1200, 1500, 50], "X(NI)": [0.1, 0.9, 0.1]},
                                                   "extra_outputs": ["HM", "DOO"], "temperature_unit": "C",
                                                   "eqplot": True})
    out, _, art = run(standalone(tb, imgs[0]), tmp_path)
    png = Path(imgs[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    import xarray as xr
    with xr.open_dataset(png.with_suffix(".nc")) as a, xr.open_dataset(out / png.with_suffix(".nc").name) as b:
        assert np.allclose(a.GM.values, b.GM.values, equal_nan=True)
        assert np.allclose(a.NP.values, b.NP.values, equal_nan=True)
    ta = tool_artists(png)
    # the tool draws K with °C tick labels; the standalone figure has °C as data
    tp, gp = np.asarray(ta[0]["pts"]), np.asarray(art[0]["pts"])
    assert tp.shape == gp.shape and np.allclose(tp - [0, KELVIN], gp)


# ---------------------------------------------------------------------------------------------------------
# phase_property_plot / scheil / fit_data
# ---------------------------------------------------------------------------------------------------------
def test_phase_property_composition_and_temperature(tb, tmp_path):
    _, img1 = call(tb, "phase_property_plot", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "prop": "GM",
                                               "temperature": 1200, "temperature_unit": "C",
                                               "reference": {"CR": "BCC_A2", "NI": "FCC_A1"},
                                               "common_tangent": True})
    out, stdout, art = run(standalone(tb, img1[0]), tmp_path)
    png = Path(img1[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    assert "two-phase" in stdout
    lines_equal(tool_artists(png)[0]["lines"], art[0]["lines"])
    _, img2 = call(tb, "phase_property_plot", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "prop": "HM",
                                               "temperature": [1000, 1800, 100], "composition": {"NI": 0.4},
                                               "phases": ["FCC_A1", "BCC_A2", "LIQUID"]})
    out, _, art = run(standalone(tb, img2[0]), tmp_path)
    png = Path(img2[0])
    frames_equal(png.with_suffix(".csv"), out / png.with_suffix(".csv").name)
    lines_equal(tool_artists(png)[0]["lines"], art[0]["lines"])


def test_scheil(tb, tmp_path):
    _, imgs = call(tb, "scheil_solidification", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                 "composition": {"NI": 0.3}, "step_temperature": 5,
                                                 "temperature_unit": "C"})
    out, stdout, art = run(standalone(tb, imgs[0]), tmp_path)
    png = Path(imgs[0])
    for suffix in ("_scheil.csv", "_equilibrium.csv"):
        frames_equal(png.with_name(png.stem + suffix), out / (png.stem + suffix))
    ta = tool_artists(png)
    for p in range(2):
        lines_equal(ta[p]["lines"], art[p]["lines"])


def test_fit_data_models(tb, tmp_path):
    # the fit_data generator lives in codegen_plot (shared with the 'fit' plot op)
    call(tb, "fit_data", {"x": [800, 900, 1000, 1100, 1200], "y": [1.1e-15, 2.3e-14, 3.1e-13, 2.4e-12, 1.5e-11],
                          "model": "arrhenius", "T_unit": "C"})
    call(tb, "fit_data", {"data": "expdata/CrNi_digitized.csv", "x_col": "T_C", "y_col": "X(CR)_1",
                          "where": {"reference": ["Jones 1975", "Brown 1980"]}, "model": "linear",
                          "band": "prediction", "level": 0.9, "output_csv": "poly_fit.csv"})
    call(tb, "fit_data", {"x": [-2, -1, -0.5, 0, 0.5, 1, 2, 3], "y": [0.3, 1.2, 2.1, 2.9, 2.2, 1.4, 0.5, 0.2],
                          "sigma": 0.1, "model": "gaussian", "offset": True, "range": [-3, 4], "npoints": 50})
    for i, (curve, resid) in enumerate([("fit_arrhenius.csv", "fit_arrhenius_residuals.csv"),
                                        ("poly_fit.csv", "poly_fit_residuals.csv"),
                                        ("fit_gaussian.csv", "fit_gaussian_residuals.csv")]):
        script = standalone(tb, str(i + 1))
        if i == 0:
            assert "np.log" in script                         # arrhenius: fitted in ln y by default
        if i == 1:
            assert "expdata/CrNi_digitized.csv" in script and "calphad_tools" not in script
        out, stdout, _ = run(script, tmp_path, dump=False)
        for name in (curve, resid):
            frames_equal(Path(tb.out) / name, out / name, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------------------------------------
# diagram overviews and ESPEI plots
# ---------------------------------------------------------------------------------------------------------
def _overview_files_match(png_dir: Path, out: Path, stems, xcol=None):
    """Every subsystem the tool computed is recomputed by the code (map_phase_diagram per subsystem): the same
    files, and boundaries on the same curves (the code runs with a random hash seed here)."""
    for stem in stems:
        for suffix in (".png", ".svg", ".pdf", ".json", "_boundaries.csv", "_tielines.csv", "_invariants.csv"):
            assert (out / (stem + suffix)).exists(), stem + suffix
        info = json.loads((png_dir / (stem + ".json")).read_text())
        x = xcol or info["orientation"]["x_axis"]
        a, b = [pd.read_csv(d / f"{stem}_boundaries.csv") for d in (png_dir, out)]
        if info["kind"] == "binary":
            boundaries_near(a, b, x)
        else:                   # ternary: (x, y) mole fractions, unscaled
            y = info["orientation"]["y_axis"]
            assert set(a.phase) == set(b.phase)
            for ph in set(a.phase):
                pa, pb = a[a.phase == ph][[x, y]].to_numpy(), b[b.phase == ph][[x, y]].to_numpy()
                points_near(pa, pb, tol=0.02, mapping=True)
                points_near(pb, pa, tol=0.02, mapping=True)
        ia, ib = [pd.read_csv(d / f"{stem}_invariants.csv") for d in (png_dir, out)]
        assert np.allclose(sorted(ia.T_K.unique()), sorted(ib.T_K.unique()), atol=0.5)


def test_plot_all_binaries_and_ternaries(tb, tmp_path):
    _, imgs = call(tb, "plot_all_binaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                             "tmin_c": 800, "tmax_c": 1800})
    out, _, _ = run(standalone(tb, imgs[0]), tmp_path, dump=False)
    png_dir = Path(imgs[0]).parent
    assert (out / Path(imgs[0]).name).exists() and (out / "binaries_CoCrFeNiV_Choi2019.pptx").exists()
    single = sorted(png_dir.glob("CoCrFeNiV_Choi2019_*-*.png"))
    assert len(single) == 3
    _overview_files_match(png_dir, out, [p.stem for p in single])
    _, imgs = call(tb, "plot_all_ternaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                              "temperatures_c": [1200]})
    out, _, _ = run(standalone(tb, imgs[0]), tmp_path, dump=False)
    assert (out / Path(imgs[0]).name).exists() and (out / "ternaries_CoCrFeNiV_Choi2019.pptx").exists()
    _overview_files_match(Path(imgs[0]).parent, out, ["CoCrFeNiV_Choi2019_CO-CR-NI_1200C"])


def test_espei_plots(tb, tmp_path):
    call(tb, "write_zpf_dataset", {"components": ["CR", "NI"], "temperatures": [1200, 1300],
                                   "temperature_unit": "C",
                                   "values": [[["FCC_A1", ["CR"], [0.42]], ["BCC_A2", ["CR"], [0.9]]],
                                              [["FCC_A1", ["CR"], [0.45]], ["BCC_A2", ["CR"], [0.88]]]],
                                   "reference": "test", "output": str(Path(tb.out) / "datasets" / "crni_zpf.json")})
    _, imgs = call(tb, "plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": str(Path(tb.out) /
                                                                                              "datasets"),
                                                  "tdb": "CrNi-06Hal.tdb", "tmin": 1000, "tmax": 1800,
                                                  "temperature_unit": "C"})
    assert Path(imgs[0]).with_suffix(".ipynb").exists()            # auto-saved reproduction notebook
    out, _, art = run(standalone(tb, imgs[0]), tmp_path)
    ta = tool_artists(imgs[0])[0]
    tpts = np.array(line_points(ta) + ta["pts"]) - [0, KELVIN]
    gpts = np.array(line_points(art[0]) + art[0]["pts"])
    points_near(gpts, tpts, scale=(1, 1000), tol=0.02, mapping=True)
    points_near(tpts, gpts, scale=(1, 1000), tol=0.02, mapping=True)
    _, imgs = call(tb, "plot_parameter_vs_data", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                  "phase": "LIQUID", "configuration": [["CR", "NI"]],
                                                  "output": "HM_MIX"})
    out, _, art = run(standalone(tb, imgs[0]), tmp_path)
    lines_equal(tool_artists(imgs[0])[0]["lines"], art[0]["lines"])


# ---------------------------------------------------------------------------------------------------------
# model / database / MCMC outputs
# ---------------------------------------------------------------------------------------------------------
def test_export_phase_model_and_write_database(tb, tmp_path):
    t1, _ = call(tb, "export_phase_model", {"tdb": "CrNi-06Hal.tdb", "phase": "FCC_A1", "elements": ["CR", "NI"],
                                            "evaluate_at": {"T": 1000, "X(CR)": 0.3, "X(NI)": 0.7}, "latex": True})
    call(tb, "export_phase_model", {"tdb": "CrNi-06Hal.tdb", "phase": "B2_BCC", "expand_functions": True})
    call(tb, "write_database", {"tdb": "CrNi-06Hal.tdb", "output_tdb": "crni_rewritten.tdb"})
    for i, files in enumerate([["model_FCC_A1_CR-NI_CrNi-06Hal.txt", "model_FCC_A1_CR-NI_CrNi-06Hal.tex"],
                               ["model_B2_BCC_all_CrNi-06Hal.txt"], ["crni_rewritten.tdb"]]):
        out, _, _ = run(standalone(tb, str(i + 1)), tmp_path, dump=False)
        for f in files:
            assert _no_stamp(out / f) == _no_stamp(Path(tb.out) / f), f
        if i < 2:
            csv = files[0].replace(".txt", "_parameters.csv")
            frames_equal(Path(tb.out) / csv, out / csv)
    assert "GM = " in (Path(tb.out) / "model_FCC_A1_CR-NI_CrNi-06Hal.txt").read_text()
    from calphad_tools import codegen
    call(tb, "write_database", {"tdb": "CrNi-06Hal.tdb", "output_tdb": "ni_only.tdb", "elements": ["NI"]})
    cells, _ = codegen.build_cells(tb, "4")
    assert not cells[0].standalone


def _no_stamp(path):
    """File text without the TDB writer's '$ ...' header comments (they carry a time stamp)."""
    return "\n".join(ln for ln in Path(path).read_text().splitlines() if not ln.startswith("$"))


def _mcmc_run(d: Path) -> Path:
    """Synthetic ESPEI-like run: a TDB with three VV symbols, trace (6 chains, 60 + 10 zero-filled iterations)."""
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "examples/tdbs/CrNi-06Hal.tdb").read_text()
    src += ("\n FUNCTION VV0000 298.15 -1000.0; 6000 N !\n FUNCTION VV0001 298.15 2.5; 6000 N !\n"
            " FUNCTION VV0002 298.15 300.0; 6000 N !\n PARAMETER L(LIQUID,CR,NI;3) 298.15 VV0000+VV0001*T; 6000 N !\n"
            " PARAMETER L(FCC_A1,CR,NI:VA;3) 298.15 VV0002; 6000 N !\n")
    (d / "dft.tdb").write_text(src)
    rng = np.random.default_rng(1)
    nch, nit, k = 6, 60, 3
    tr, lp = np.zeros((nch, nit + 10, k)), np.zeros((nch, nit + 10))
    x = np.array([-1000.0, 2.5, 300.0]) + rng.normal(0, [200, 0.5, 50], (nch, k))
    for i in range(nit):
        prop = x + rng.normal(0, [30, 0.05, 8], (nch, k))
        x = np.where((rng.random(nch) < 0.4)[:, None], prop, x)
        tr[:, i] = x
        lp[:, i] = -500 * np.exp(-i / 8) - 50 - rng.normal(0, 2, nch) ** 2
    np.save(d / "trace.npy", tr)
    np.save(d / "lnprob.npy", lp)
    return d


def test_mcmc_analysis_and_tdb(tb, tmp_path):
    run_dir = _mcmc_run(tmp_path / "mcmc_run")
    _, imgs = call(tb, "analyze_mcmc", {"trace": str(run_dir)})
    call(tb, "mcmc_to_tdb", {"trace": str(run_dir / "trace.npy"), "statistic": "mean", "burn_in": "20"})
    call(tb, "mcmc_to_tdb", {"trace": str(run_dir), "statistic": "map"})
    corner = next(p for p in imgs if p.endswith("_corner.png"))
    script = standalone(tb, corner)
    assert "truncate_arrays" in script and "L(LIQUID,CR,NI;3) [T]" in script
    out, _, art = run(script, tmp_path)
    frames_equal(Path(tb.out) / "mcmc_analysis" / "mcmc_run_summary.csv", out / "mcmc_run_summary.csv")
    # fig/ax are left on the requested image (the corner plot): compare its artists
    ta = tool_artists(corner)
    assert len(ta) == len(art)
    for a_, g_ in zip(ta, art):
        lines_equal(a_["lines"], g_["lines"])
    for p in imgs:
        assert (out / Path(p).name).exists()
    for i, name in ((2, "mcmc_run_mean.tdb"), (3, "mcmc_run_map.tdb")):
        out, _, _ = run(standalone(tb, str(i)), tmp_path, dump=False)
        assert _no_stamp(out / name) == _no_stamp(Path(tb.out) / name)
