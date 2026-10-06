"""Diagram orientation (which element on which side) and data that follow it.

map_phase_diagram right / corners, plot_diagram_with_data (binary + ternary ZPF data), the composition_points
figure operation, the element_fractions transform step, overview element_order, and their show_code scripts.

Run:  .venv/bin/python -m pytest -q -p no:warnings tests/test_orientation.py
"""
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calphad_tools.phase_names import plain_text as name   # the name a text shows (Al₃Ni₂)

ROOT = Path(__file__).resolve().parents[1]
KELVIN = 273.15


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return True
    out = tmp_path_factory.mktemp("orient")
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out / "out"), ui=U(), python_tool="off")


def call(tb, name, args):
    text, images, err = tb.call(name, args)
    assert not err, text
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
    assert fig is not None, png
    return fig, fig.axes[0]


def curve_points(ax):
    """(x, y) of every plotted curve point (phase-boundary lines, no markers)."""
    pts = []
    for ln in ax.get_lines():
        if ln.get_marker() in (None, "None", "", " ") and len(ln.get_xdata()) > 2:
            pts += list(zip(np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)))
    return np.array(pts)


def near(A, B, scale=(1.0, 1.0), tol=0.02, q=0.95, max_mul=5.0):
    """A fraction q of the points of A lie within tol of B (pycalphad's mapping may trace an extra short segment
    depending on the hash seed and the axis variables), all within max_mul * tol (max_mul None: no limit)."""
    A, B = np.asarray(A, float) / scale, np.asarray(B, float) / scale
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    ok = np.quantile(d, q) < tol and (max_mul is None or d.max() < max_mul * tol)
    assert ok, (np.quantile(d, q), d.max())


def data_markers(ax):
    """[(x, y)] of all single-marker lines (data points drawn with plot(..., marker, linestyle=''))."""
    out = []
    for ln in ax.get_lines():
        if ln.get_marker() not in (None, "None", "", " ") and ln.get_linestyle() in ("None", "", " ", "none"):
            out += list(zip(np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)))
    return np.array(out)


def labels_of(ax, prefix=("phase_label:", "invariant_label:")):
    return [t for t in ax.texts if (t.get_gid() or "").startswith(prefix)]


def check_anchors(ax):
    """Every phase label lies in a region with exactly its phase set (equilibrium check; a stoichiometric phase
    named outside the plot: that phase is stable at the compound)."""
    from calphad_tools.calc import get_db
    from calphad_tools.phase_regions import key_at, ordered_info
    rec = ax._calphad_diagram["recipe"]
    db = get_db(rec["tdb"])
    info = ordered_info(db, sorted(rec["elements"]) + ["VA"], rec["phases"])
    n = 0
    for t in labels_of(ax, ("phase_label:",)):
        lab = t._calphad_label
        x, y = lab["at"] if lab.get("outside") else t.get_position()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = key_at(db, rec, info, x, y)
        if lab["kind"] == "compound":
            assert got and lab["key"][0] in got, (name(t), got)
        else:
            assert got == tuple(lab["key"]), (name(t), (x, y), got)
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


def point_in_rect(p, u, v, pad=0.0):
    c = p.mean(axis=0)
    ex, ey = p[1] - p[0], p[3] - p[0]
    d = np.array([u, v]) - c
    return abs(d @ ex) / np.linalg.norm(ex) <= np.linalg.norm(ex) / 2 + pad and \
        abs(d @ ey) / np.linalg.norm(ey) <= np.linalg.norm(ey) / 2 + pad


def label_pos(ax, text):
    t = next(t for t in labels_of(ax) if name(t) == text)
    return t.get_position()


def run_script(tb, target, tmp_path, name="gen"):
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, target)
    assert all(c.standalone for c in cells), [(c.title, c.note) for c in cells]
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    dump = '''
import json as _json, numpy as _np
from matplotlib.collections import LineCollection as _LC
_ax = fig.axes[0]
_json.dump({"lines": [[_np.asarray(l.get_xdata(), float).tolist(), _np.asarray(l.get_ydata(), float).tolist(),
                       str(l.get_marker()), str(l.get_linestyle())] for l in _ax.get_lines()],
            "pts": [_np.asarray(c.get_offsets(), float).tolist() for c in _ax.collections
                    if type(c).__name__ == "PathCollection"],
            "segs": [_np.asarray(s, float).tolist() for c in _ax.collections if isinstance(c, _LC)
                     for s in c.get_segments()],
            "texts": [getattr(t, "_calphad_plain", None) or t.get_text() for t in _ax.texts],
            "xlabel": _ax.get_xlabel()}, open("art.json", "w"))
'''
    (d / "gen.py").write_text(script + dump)
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return script, json.loads((d / "art.json").read_text()), d / "standalone_outputs"


# ---------------------------------------------------------------------------------------------------------
# orientation resolution (no calculation)
# ---------------------------------------------------------------------------------------------------------
def test_orientation_resolution():
    from calphad_tools.orientation import (binary_order, orientation_text, preferred_order, subsystem_order,
                                           ternary_order)
    assert binary_order(["FE", "CR"]) == ["FE", "CR"]                       # fallback: element order
    assert binary_order(["FE", "CR"], "CR") == ["FE", "CR"]
    assert binary_order(["FE", "CR"], "fe") == ["CR", "FE"]
    assert binary_order(["FE", "CR"], left="CR") == ["CR", "FE"]
    with pytest.raises(ValueError, match="not one of"):
        binary_order(["FE", "CR"], "NI")
    assert ternary_order(["FE", "CR", "NI"]) == ["FE", "CR", "NI"]
    assert ternary_order(["FE", "CR", "NI"], {"left": "NI", "right": "FE", "top": "CR"}) == ["NI", "FE", "CR"]
    assert ternary_order(["FE", "CR", "NI"], {"bottom-right": "FE"}) == ["CR", "FE", "NI"]   # rest in given order
    assert ternary_order(["FE", "CR", "NI"], right="NI") == ["FE", "NI", "CR"]
    for bad in ({"left": "NI", "right": "NI"}, {"left": "CO"}, {"middle": "FE"}):
        with pytest.raises(ValueError):
            ternary_order(["FE", "CR", "NI"], bad)
    assert preferred_order(["NI", "CR", "FE"]) == ["CR", "FE", "NI"]         # default: alphabetical
    assert preferred_order(["NI", "CR", "FE"], ["FE", "CR", "NI"]) == ["FE", "CR", "NI"]
    assert preferred_order(["CO", "NI", "FE"], ["NI"]) == ["NI", "CO", "FE"]
    assert subsystem_order(["CR", "NI"], ["FE", "CR", "NI"], {"Ni-Cr": ["NI", "CR"]}) == ["NI", "CR"]
    assert subsystem_order(["CR", "FE", "NI"], None, {"FE-CR-NI": {"left": "NI", "right": "FE", "top": "CR"}}) == \
        ["NI", "FE", "CR"]
    assert subsystem_order(["CR", "FE"], None, {"CR-FE": "CR"}) == ["FE", "CR"]
    assert orientation_text("binary", ["FE", "CR"]) == "orientation: x axis = X(Cr); Fe on the left, Cr on the right"
    assert "Ni bottom-left, Fe bottom-right, Cr top" in orientation_text("ternary", ["NI", "FE", "CR"])


def test_element_fractions_step():
    from calphad_tools.figdata import run_transform
    df = pd.DataFrame({"x": [10.0, 30.0], "T": [1000.0, 1100.0]})
    at = run_transform(df, [{"step": "element_fractions", "cols": {"x": "FE"}, "elements": ["CR", "FE"],
                             "unit": "at%"}])
    assert np.allclose(at["X(FE)"], [0.1, 0.3]) and np.allclose(at["X(CR)"], [0.9, 0.7])
    wt = run_transform(df, [{"step": "element_fractions", "cols": {"x": "FE"}, "elements": ["CR", "FE"],
                             "unit": "wt%"}])
    w = 0.3 / 55.845
    assert abs(wt["X(FE)"][1] - w / (w + 0.7 / 51.996)) < 1e-4 and np.allclose(wt["X(FE)"] + wt["X(CR)"], 1)
    with pytest.raises(ValueError, match="balance"):
        run_transform(df, [{"step": "element_fractions", "cols": {"x": "FE"}, "elements": ["CR", "FE", "NI"]}])


# ---------------------------------------------------------------------------------------------------------
# binary: Cr on the right vs Fe on the right
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fecr(tb):
    out = {}
    for right in ("CR", "FE"):
        text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrFe-82Her.tdb", "elements": ["FE", "CR"], "tmin": 300,
                                                    "tmax": 1000, "temperature_unit": "C", "right": right,
                                                    "label_regions": True, "formats": []})
        out[right] = (text, imgs[0])
    return out


def test_binary_mirroring(fecr):
    (tc, pc), (tf, pf) = fecr["CR"], fecr["FE"]
    assert "x axis = mole fraction Cr; Fe on the left, Cr on the right" in tc
    assert "x axis = mole fraction Fe; Cr on the left, Fe on the right" in tf
    assert Path(pc).name != Path(pf).name                                  # FE-CR vs CR-FE file names
    figc, axc = stored(pc)
    figf, axf = stored(pf)
    assert axc.get_xlabel() == "Mole fraction Cr" and axf.get_xlabel() == "Mole fraction Fe"
    assert axc._calphad_diagram["recipe"]["elements"] == ["FE", "CR"]
    assert axf._calphad_diagram["recipe"]["elements"] == ["CR", "FE"]
    # plotted boundaries are mirror images (x <-> 1 - x), in °C
    a, b = curve_points(axc), curve_points(axf) * [-1, 1] + [1, 0]
    near(a, b, scale=(1, 1000))
    near(b, a, scale=(1, 1000))
    # the CSVs store compositions by element name: the same whichever element is on the right
    ba, bb = [pd.read_csv(files_from(t)["boundaries_csv"]) for t in (tc, tf)]
    assert {"X(FE)", "X(CR)"} <= set(ba.columns) & set(bb.columns)
    for ph in set(ba.phase) | set(bb.phase):
        pa = ba[ba.phase == ph][["X(CR)", "T_K"]].to_numpy() / [1, 1000]
        pb = bb[bb.phase == ph][["X(CR)", "T_K"]].to_numpy() / [1, 1000]
        near(pa, pb)
        near(pb, pa)
    ia, ib = [pd.read_csv(files_from(t)["invariants_csv"]) for t in (tc, tf)]
    assert np.allclose(ia.T_K.unique(), ib.T_K.unique(), atol=0.5)
    assert np.allclose(sorted(ia["X(CR)"]), sorted(ib["X(CR)"]), atol=5e-3)
    assert set(ia.type) == set(ib.type) == {"eutectoid"}
    # labels land in the correct regions, on the mirrored side
    assert check_anchors(axc) >= 6 and check_anchors(axf) >= 6
    assert label_pos(axc, "FCC")[0] < 0.2 and label_pos(axf, "FCC")[0] > 0.8          # Fe-rich γ loop
    s_c, s_f = label_pos(axc, "σ"), label_pos(axf, "σ")
    assert abs(s_c[0] - (1 - s_f[0])) < 0.05


# ---------------------------------------------------------------------------------------------------------
# ternary: two corner permutations
# ---------------------------------------------------------------------------------------------------------
TERN = {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"], "temperature": 800,
        "temperature_unit": "C", "composition_step": 0.04}


@pytest.fixture(scope="module")
def fecrni(tb):
    out = {}
    for key, corners in (("fe_br", {"left": "NI", "right": "FE", "top": "CR"}),
                         ("ni_top", {"left": "FE", "right": "CR", "top": "NI"})):
        text, imgs = call(tb, "map_phase_diagram", {**TERN, "corners": corners,
                                                    "label_regions": key == "fe_br", "formats": []})
        out[key] = (text, imgs[0])
    return out


def corner_xy(ax):
    ax.figure.canvas.draw()
    return {k: ax.transData.transform(p) for k, p in (("bl", (0, 0)), ("br", (1, 0)), ("top", (0, 1)))}


def test_ternary_permutations(fecrni):
    (ta, pa), (tb_, pb) = fecrni["fe_br"], fecrni["ni_top"]
    assert "corners Ni bottom-left, Fe bottom-right, Cr top" in ta
    assert "corners Fe bottom-left, Cr bottom-right, Ni top" in tb_
    figa, axa = stored(pa)
    figb, axb = stored(pb)
    assert (axa.get_xlabel(), axa.get_ylabel()) == ("Mole fraction Fe", "Mole fraction Cr")
    assert (axb.get_xlabel(), axb.get_ylabel()) == ("Mole fraction Cr", "Mole fraction Ni")
    # corner labels sit at their corners
    for ax, want in ((axa, ("Ni", "Fe", "Cr")), (axb, ("Fe", "Cr", "Ni"))):
        cx = corner_xy(ax)
        r = ax.figure.canvas.get_renderer()
        got = {t.get_gid(): t for t in ax.texts if (t.get_gid() or "").startswith("corner_label:")}
        assert sorted(name(t) for t in got.values()) == sorted(want)
        for corner, el in zip(("bl", "br", "top"), want):
            t = got[f"corner_label:{el.upper()}"]
            e = t.get_window_extent(r)
            assert abs((e.x0 + e.x1) / 2 - cx[corner][0]) < 25, (el, corner)
            assert (e.y1 < cx[corner][1]) if corner != "top" else (e.y0 > cx[corner][1])
    # a known composition maps to the barycentric point of the corners
    X = {"NI": 0.2, "FE": 0.5, "CR": 0.3}
    for ax, (xe, ye), (bl, br, tp) in ((axa, ("FE", "CR"), ("NI", "FE", "CR")), (axb, ("CR", "NI"), ("FE", "CR", "NI"))):
        cx = corner_xy(ax)
        want = X[bl] * cx["bl"] + X[br] * cx["br"] + X[tp] * cx["top"]
        assert np.allclose(ax.transData.transform((X[xe], X[ye])), want, atol=0.5)
    # the same phase boundaries, by element name, in both permutations
    ba, bb = [pd.read_csv(files_from(t)["boundaries_csv"]) for t in (ta, tb_)]
    # two independent mappings (0.02 composition steps, different axis variables): the sampling differs by 1-2
    # steps and one of them may trace a short extra segment the other does not (seen for ~1 in 6 hash seeds), so
    # compare phases with enough points on 90 % of them; exact positions are checked above
    for ph in set(ba.phase) | set(bb.phase):
        a = ba[ba.phase == ph][["X(FE)", "X(CR)"]].to_numpy()
        b = bb[bb.phase == ph][["X(FE)", "X(CR)"]].to_numpy()
        if min(len(a), len(b)) < 20:
            continue
        near(a, b, tol=0.05, q=0.9, max_mul=None)
        near(b, a, tol=0.05, q=0.9, max_mul=None)
    # plotted curves: (X(Fe), X(Cr)) in A  ->  (X(Cr), X(Ni)) in B
    ca = curve_points(axa)
    ca_in_b = np.column_stack([ca[:, 1], 1 - ca[:, 0] - ca[:, 1]])
    near(ca_in_b, curve_points(axb), tol=0.05, q=0.9, max_mul=None)
    # tie-triangles inside the triangle agree (a degenerate one lying on a binary edge may be traced in one run only)
    def inner(t):
        invs = json.loads(files_from(t)["json"].read_text())["invariants"]
        return [sorted(sorted(c[e] for e in ("FE", "CR", "NI")) and [c["FE"], c["CR"]]
                       for c in i["compositions"].values()) for i in invs
                if not any(all(c[e] < 1e-4 for c in i["compositions"].values()) for e in ("FE", "CR", "NI"))]
    ia, ib = inner(ta), inner(tb_)
    assert len(ia) == len(ib) >= 2
    for tri in ia:
        assert any(np.allclose(tri, t2, atol=0.02) for t2 in ib), tri
    assert check_anchors(axa) >= 4      # labels of the Fe-bottom-right diagram (thin fields: no room for 8 pt)
    # BCC, stable for pure Cr and pure Fe at 800 °C: named at those corners (top, bottom-right) in this orientation
    corner = {(name(t), t._calphad_label.get("corner")) for t in labels_of(axa, ("phase_label:",))
              if t._calphad_label.get("outside") == "corner"}
    assert corner == {("BCC", 2), ("BCC", 1)}, corner
    assert "named at its corner: " in ta and "BCC (Cr)" in ta and "BCC (Fe)" in ta


# ---------------------------------------------------------------------------------------------------------
# ZPF data follow the orientation (plot_diagram_with_data)
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def zpf_dirs(tmp_path_factory):
    from calphad_tools import expdata
    d = tmp_path_factory.mktemp("zpf")
    ds2 = expdata.make_zpf_dataset(["FE", "CR"], [900, 1000], [
        [["BCC_A2", ["CR"], [0.3]], ["SIGMA_D8B", ["CR"], [0.45]]],
        [["BCC_A2", ["CR"], [0.2]], ["SIGMA_D8B", ["CR"], [None]]]], reference="bin")
    expdata.write_dataset(ds2, str(d / "b" / "fecr.json"))
    ds3 = expdata.make_zpf_dataset(["FE", "CR", "NI"], [1073.15, 1075, 1073.15, 1273], [
        [["FCC_A1", ["CR", "NI"], [0.20, 0.30]], ["BCC_A2", ["CR", "NI"], [0.40, 0.05]]],     # tie-line
        [["FCC_A1", ["CR", "FE"], [0.25, 0.55]], ["BCC_A2", ["CR", "NI"], [None, None]]],     # broken: 1 point
        [["FCC_A1", ["CR", "NI"], [0.28, 0.25]], ["BCC_A2", ["CR", "NI"], [0.55, 0.10]],
         ["SIGMA", ["CR", "NI"], [0.45, 0.05]]],                                             # tie-triangle
        [["FCC_A1", ["CR", "NI"], [0.10, 0.10]], ["BCC_A2", ["CR", "NI"], [0.20, 0.05]]]],    # other T: skipped
        reference="tern")
    expdata.write_dataset(ds3, str(d / "t" / "fecrni.json"))
    return d


def test_zpf_binary_follows(tb, zpf_dirs):
    want_cr = {(0.3, 900.0), (0.45, 900.0), (0.2, 1000.0)}          # (X(CR), T_K) of the data points
    for right in ("CR", "FE"):
        text, imgs = call(tb, "plot_diagram_with_data", {"elements": ["FE", "CR"], "datasets_dir": str(zpf_dirs / "b"),
                                                         "tdb": "CrFe-82Her.tdb", "tmin": 300, "tmax": 1000,
                                                         "temperature_unit": "C", "right": right,
                                                         "output_png": f"pdd_{right}.png"})
        other = "Fe" if right == "CR" else "Cr"
        assert f"{other} on the left, {right.capitalize()} on the right" in text
        fig, ax = stored(imgs[0])
        assert ax.get_xlabel() == f"Mole fraction {right.capitalize()}"
        pts = {(round(x, 6), round(y, 2)) for x, y in data_markers(ax)}
        exp = {(round(x if right == "CR" else 1 - x, 6), T) for x, T in want_cr}
        assert pts == exp, (pts, exp)
        assert ax._calphad_diagram["recipe"]["elements"] == (["FE", "CR"] if right == "CR" else ["CR", "FE"])


def test_zpf_ternary_follows(tb, zpf_dirs, tmp_path):
    corners = {"left": "NI", "right": "FE", "top": "CR"}
    text, imgs = call(tb, "plot_diagram_with_data", {"elements": ["FE", "CR", "NI"], "datasets_dir": str(zpf_dirs / "t"),
                                                     "tdb": "CoCrFeNiV_Choi2019.tdb", "temperature": 800,
                                                     "temperature_unit": "C", "corners": corners,
                                                     "label_regions": True, "formats": [],
                                                     "output_png": "pdd_tern.png"})
    assert "3 ZPF equilibria" in text and "1 tie-lines, 1 single points of broken tie-lines, 1 tie-triangles" in text
    assert "corners Ni bottom-left, Fe bottom-right, Cr top" in text
    fig, ax = stored(imgs[0])
    comps = [(0.20, 0.30), (0.40, 0.05), (0.25, 1 - 0.25 - 0.55), (0.28, 0.25), (0.55, 0.10), (0.45, 0.05)]
    exp = {(round(1 - cr - ni, 6), round(cr, 6)) for cr, ni in comps}      # (X(Fe), X(Cr)) = (x, y)
    got = {(round(x, 6), round(y, 6)) for x, y in data_markers(ax)}
    assert got == exp, (got, exp)
    # tie-line (black) and tie-triangle (red dashed) of the data
    tie = [ln for ln in ax.get_lines() if ln.get_marker() in ("None", None, "") and len(ln.get_xdata()) == 2
           and ln.get_color() == "k"]
    assert any(np.allclose(sorted(zip(ln.get_xdata(), ln.get_ydata())), sorted([(0.5, 0.2), (0.55, 0.4)]))
               for ln in tie)
    assert any(ln.get_linestyle() == "--" and len(ln.get_xdata()) == 4 for ln in ax.get_lines())
    # labels keep clear of the data points and sit in their regions
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = ax.transData.transform(data_markers(ax))
    for t in labels_of(ax):
        p = text_rect(t, r)
        for u, v in px:
            assert not point_in_rect(p, u, v, 5.0 / 2 * fig.dpi / 72), (name(t), u, v)
    assert check_anchors(ax) >= 4
    # the same figure from show_code
    script, art, _ = run_script(tb, imgs[0], tmp_path, "tern_data")
    assert "TernaryStrategy" in script and "draw_shaded" in script            # shaded style (default)
    assert "v.X('FE')" in script and "'Ni'" in script
    gpts = {(round(x[0], 6), round(y[0], 6)) for x, y, m, ls in art["lines"] if m not in ("None", "") and len(x) == 1}
    assert gpts == exp
    assert {"Ni", "Fe", "Cr"} <= set(art["texts"])


# ---------------------------------------------------------------------------------------------------------
# composition_points on existing diagrams
# ---------------------------------------------------------------------------------------------------------
def scatter_offsets(ax, label=None):
    cs = [c for c in ax.collections if type(c).__name__ == "PathCollection" and (label is None or
                                                                                c.get_label() == label)]
    return np.asarray(cs[-1].get_offsets(), float)


def test_composition_points_binary(tb, fecr, tmp_path):
    pts = [{"X(CR)": 0.3, "T": 600}, {"X(CR)": 0.8, "T_K": 1000}]
    for right, (_, png) in fecr.items():
        text, imgs = call(tb, "edit_figure", {"image": png, "operations": [
            {"op": "composition_points", "points": pts, "color": "red", "label": "measured", "tielines": [[0, 1]]},
            {"op": "composition_points", "points": [{"wt% Cr": 30, "T": 500}], "unit": "wt%", "label": "wt"},
            {"op": "composition_points", "points": [{"X(FE)": 0.25, "X(CR)": 0.75, "T": 1073.15}], "T_unit": "K",
             "label": "all"}],
            "output": f"cp_{right}.png"})
        assert "balance Fe computed" in text
        fig, ax = stored(imgs[0])
        xs = np.array([0.3, 0.8]) if right == "CR" else np.array([0.7, 0.2])
        assert np.allclose(scatter_offsets(ax, "measured"), np.column_stack([xs, [600, 1000 - KELVIN]]))
        w = 30 / 51.996
        xcr = w / (w + 70 / 55.845)                                          # wt% -> mole fraction
        assert np.allclose(scatter_offsets(ax, "wt"), [[xcr if right == "CR" else 1 - xcr, 500]], atol=1e-4)
        assert np.allclose(scatter_offsets(ax, "all"), [[0.75 if right == "CR" else 0.25, 800]])
        seg = [c for c in ax.collections if type(c).__name__ == "LineCollection" and len(c.get_segments()) == 1]
        assert any(np.allclose(s.get_segments()[0], [[xs[0], 600], [xs[1], 1000 - KELVIN]]) for s in seg)
        # show_code reproduces the positions (map + edit cells)
        if right == "FE":
            script, art, _ = run_script(tb, imgs[0], tmp_path, f"cp_{right}")
            assert "comp['FE'] = 1.0 - sum(comp.values())" in script and "x = shown['FE']" in script
            got = [np.asarray(p) for p in art["pts"] if len(p)]
            for want in (scatter_offsets(ax, "measured"), scatter_offsets(ax, "wt"), scatter_offsets(ax, "all")):
                assert any(g.shape == want.shape and np.allclose(g, want, atol=1e-9) for g in got), want


def test_composition_points_after_labels_and_csv(tb, tmp_path):
    # a mole-percent diagram, then region labels, then points from a CSV (at%, °C, tie-line groups)
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrFe-82Her.tdb", "elements": ["CR", "FE"], "tmin": 500,
                                             "tmax": 1000, "temperature_unit": "C", "right": "CR",
                                             "composition_step": 0.05, "temperature_step": 20,
                                             "composition_unit": "mole_percent"})
    _, lab = call(tb, "edit_figure", {"image": imgs[0], "operations": [{"op": "label_phase_regions"}],
                                      "output": "cp_labelled.png", "formats": []})
    csv = tmp_path / "pts.csv"
    pd.DataFrame({"at% Cr": [20.0, 40.0, 60.0], "T_C": [700.0, 700.0, 900.0], "tie": [1, 1, 2]}).to_csv(csv, index=False)
    text, out = call(tb, "edit_figure", {"image": lab[0], "operations": [
        {"op": "composition_points", "csv": str(csv), "unit": "at%", "tieline_col": "tie", "label": "csv"}],
        "output": "cp_csv.png"})
    fig, ax = stored(out[0])
    assert np.allclose(scatter_offsets(ax, "csv"), [[20, 700], [40, 700], [60, 900]])       # mole percent, °C
    assert labels_of(ax)                                                                     # labels kept
    # the diagram of plot_diagram_with_data: data in K shown in °C; points given in °C land at K data values
    from calphad_tools import expdata
    d = tmp_path / "ds"
    expdata.write_dataset(expdata.make_zpf_dataset(["CR", "FE"], [900], [[["BCC_A2", ["CR"], [0.3]],
                                                                          ["SIGMA_D8B", ["CR"], [0.45]]]]),
                          str(d / "z.json"))
    _, pdd = call(tb, "plot_diagram_with_data", {"elements": ["CR", "FE"], "datasets_dir": str(d), "tdb": "CrFe-82Her.tdb",
                                                 "tmin": 300, "tmax": 1000, "temperature_unit": "C", "right": "FE",
                                                 "output_png": "cp_pdd.png"})
    text, out = call(tb, "edit_figure", {"image": pdd[0], "operations": [
        {"op": "composition_points", "points": [{"X(CR)": 0.1, "T": 800}], "label": "c"}], "output": "cp_pdd2.png"})
    fig, ax = stored(out[0])
    assert np.allclose(scatter_offsets(ax, "c"), [[0.9, 800 + KELVIN]])
    # a figure without a recipe: clear error
    _, plain = call(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                      "output": "plain"})
    t, _, err = tb.call("edit_figure", {"image": plain[0], "operations": [
        {"op": "composition_points", "points": [{"X(CR)": 0.1, "T": 800}]}]})
    assert err and "orientation is unknown" in t


def test_composition_points_ternary(tb, fecrni, tmp_path):
    pts = [{"X(CR)": 0.2, "X(NI)": 0.3}, {"X(CR)": 0.4, "X(NI)": 0.05, "T": 1400}]
    geo = {"fe_br": lambda X: (X["FE"], X["CR"]), "ni_top": lambda X: (X["CR"], X["NI"])}
    for key, (_, png) in fecrni.items():
        text, imgs = call(tb, "edit_figure", {"image": png, "operations": [
            {"op": "composition_points", "points": pts, "tielines": [[0, 1]], "label": "p"},
            {"op": "composition_points", "points": [{"W(CR)": 0.3, "W(NI)": 0.3}], "label": "w"},
            {"op": "composition_points", "points": pts, "T_unit": "K", "T_tolerance": 5, "label": "near"}],
            "output": f"cpt_{key}.png"})
        fig, ax = stored(imgs[0])
        X = [{"CR": 0.2, "NI": 0.3, "FE": 0.5}, {"CR": 0.4, "NI": 0.05, "FE": 0.55}]
        assert np.allclose(scatter_offsets(ax, "p"), [geo[key](x) for x in X])
        n = {"CR": 0.3 / 51.996, "NI": 0.3 / 58.69, "FE": 0.4 / 55.847}
        Xw = {e: v / sum(n.values()) for e, v in n.items()}
        assert np.allclose(scatter_offsets(ax, "w"), [geo[key](Xw)], atol=1e-3)
        assert np.allclose(scatter_offsets(ax, "near"), [geo[key](X[0])])      # the 1400 K point is dropped
        assert "1 point(s) more than 5 K from the section temperature skipped" in text
        if key == "fe_br":
            script, art, _ = run_script(tb, imgs[0], tmp_path, "cpt")
            got = [np.asarray(p) for p in art["pts"] if len(p)]
            for lab in ("p", "w"):
                want = scatter_offsets(ax, lab)
                assert any(g.shape == want.shape and np.allclose(g, want, atol=1e-9) for g in got), lab


def test_isopleth_right_and_points(tb, tmp_path):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                                "right": "CR", "fixed": {"NI": 0.1}, "tmin": 900, "tmax": 1900,
                                                "composition_step": 0.05, "temperature_step": 20})
    assert "x axis = mole fraction Cr increasing to the right; balance Fe" in text
    assert "_isopleth_CR_NI0.1" in imgs[0]
    text, out = call(tb, "edit_figure", {"image": imgs[0], "operations": [
        {"op": "composition_points", "points": [{"X(CR)": 0.2, "X(NI)": 0.1, "T": 1500},
                                                {"X(CR)": 0.3, "X(NI)": 0.2, "T": 1400}], "label": "iso"}]})
    assert "off the section" in text
    fig, ax = stored(out[0])
    assert np.allclose(scatter_offsets(ax, "iso"), [[0.2, 1500], [0.3, 1400]])
    script, art, _ = run_script(tb, out[0], tmp_path, "iso")
    assert any(np.allclose(p, [[0.2, 1500], [0.3, 1400]]) for p in art["pts"] if len(p))
    t, _, err = tb.call("map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                              "vary": "NI", "right": "CR", "fixed": {"FE": 0.5}, "tmin": 900,
                                              "tmax": 1000})
    assert err and "contradicts" in t


# ---------------------------------------------------------------------------------------------------------
# overview decks: element_order
# ---------------------------------------------------------------------------------------------------------
def test_overview_element_order(tb, tmp_path):
    text, imgs = call(tb, "plot_all_binaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["NI", "CR", "FE"],
                                                "tmin_c": 900, "tmax_c": 1600, "element_order": ["FE", "CR", "NI"],
                                                "orientation": {"Cr-Ni": ["NI", "CR"]}, "output_pptx": "bin_ord.pptx"})
    assert "FE-CR:" in text and "FE-NI:" in text and "NI-CR:" in text
    d = Path(imgs[0]).parent
    for stem, xl in (("FE-CR", "Cr"), ("FE-NI", "Ni"), ("NI-CR", "Cr")):
        fig, ax = stored(d / f"CoCrFeNiV_Choi2019_{stem}.png")
        assert ax.get_xlabel() == f"Mole fraction {xl}"
    script, art, _ = run_script(tb, imgs[0], tmp_path, "bin_ord")
    # one cell per subsystem, each with the overview's element order (first = balance / left element)
    assert "elements = ['FE', 'CR']" in script and "elements = ['NI', 'CR']" in script
    assert "elements = ['FE', 'NI']" in script
    text, imgs = call(tb, "plot_all_ternaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["NI", "CR", "FE"],
                                                 "temperatures_c": [1000], "element_order": ["FE", "CR", "NI"],
                                                 "output_pptx": "tern_ord.pptx"})
    fig, ax = stored(Path(imgs[0]).parent / "CoCrFeNiV_Choi2019_FE-CR-NI_1000C.png")
    gids = {t.get_gid(): name(t) for t in ax.texts if (t.get_gid() or "").startswith("corner_label:")}
    assert gids == {"corner_label:FE": "Fe", "corner_label:CR": "Cr", "corner_label:NI": "Ni"}
    assert ax.get_title().startswith("Fe–Cr–Ni")
    with pytest.raises(ValueError, match="match no subsystem"):
        from calphad_tools.orientation import check_overrides
        check_overrides({"CO-FE": ["FE", "CO"]}, [("CR", "FE")])


# ---------------------------------------------------------------------------------------------------------
# show_code reproduces the orientation
# ---------------------------------------------------------------------------------------------------------
def test_show_code_orientation(tb, tmp_path):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrFe-82Her.tdb", "elements": ["FE", "CR"], "tmin": 300,
                                             "tmax": 1000, "temperature_unit": "C", "right": "FE",
                                             "composition_step": 0.05, "temperature_step": 20})
    script, art, out = run_script(tb, imgs[0], tmp_path, "bin")
    assert "v.X('FE')" in script and "Cr on the left, Fe on the right" in script
    assert art["xlabel"] == "Mole fraction Fe"
    fig, ax = stored(imgs[0])
    gen = np.array([p for x, y, m, ls in art["lines"] if m in ("None", "") and len(x) > 2 for p in zip(x, y)])
    near(gen, curve_points(ax), scale=(1, 1000))
    _, imgs = call(tb, "map_phase_diagram", {**TERN, "composition_step": 0.05,
                                             "corners": {"left": "FE", "right": "CR", "top": "NI"}})
    script, art, out = run_script(tb, imgs[0], tmp_path, "tern")
    assert "v.X('CR'): (0, 1, 0.05)" in script and "v.X('NI'): (0, 1, 0.05)" in script
    assert {"Fe", "Cr", "Ni"} <= set(art["texts"]) and art["xlabel"] == "Mole fraction Cr"
    fig, ax = stored(imgs[0])
    gen = np.array([p for x, y, m, ls in art["lines"] if m in ("None", "") and len(x) > 2 for p in zip(x, y)])
    near(gen, curve_points(ax), tol=0.02)
    # transform_data element_fractions
    csv = tmp_path / "dig.csv"
    pd.DataFrame({"x": [10.0, 40.0], "T": [900.0, 1000.0]}).to_csv(csv, index=False)
    call(tb, "transform_data", {"source": str(csv), "output": "dig_x.csv", "transform": [
        {"step": "element_fractions", "cols": {"x": "FE"}, "elements": ["CR", "FE"], "unit": "wt%"}]})
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, str(Path(tb.out) / "dig_x.csv"))
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    d = tmp_path / "td"
    d.mkdir()
    (d / "g.py").write_text(script)
    r = subprocess.run([sys.executable, "-W", "ignore", "g.py"], cwd=d, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    a, b = pd.read_csv(Path(tb.out) / "dig_x.csv"), pd.read_csv(d / "standalone_outputs" / "dig_x.csv")
    assert list(a.columns) == list(b.columns) == ["x", "T", "X(CR)", "X(FE)"]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float))
