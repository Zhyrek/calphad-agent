"""Diagram style of the phase-diagram tools (calphad_tools/diagram_style.py).

shaded (default): black boundaries / invariants (lw 1.25), two-phase fields filled aliceblue, grey (#999999) ternaries,
no tie-lines, no legend (phases named by region labels, on by default), ticks pointing in on all four sides, SVG + PDF
next to the PNG.  color: the old look (phase colours, tie-lines, red invariants, legend).

Run:  .venv/bin/python -m pytest -q -p no:warnings tests/test_diagram_style.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KELVIN = 273.15
BLACK = "#000000"
GREY = "#999999"
ALICEBLUE = mcolors.to_hex("aliceblue")


def _toolbox(out, style="shaded"):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out), ui=U(), python_tool="off",
                   diagram_style=style)


@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    return _toolbox(tmp_path_factory.mktemp("style") / "out")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def stored(png):
    from calphad_tools.figures import load_figure
    fig = load_figure(str(png))
    assert fig is not None, png
    return fig, fig.axes[0]


def plain_lines(ax):
    """Calculated lines: Line2D artists without markers."""
    return [ln for ln in ax.get_lines() if str(ln.get_marker()) in ("None", "", " ", "none")]


def fills(ax):
    return [c for c in ax.collections if type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection")]


def facecolors(c):
    return {mcolors.to_hex(f) for f in c.get_facecolor()}


def labels_of(ax, prefix=("phase_label:", "invariant_label:")):
    return [t for t in ax.texts if (t.get_gid() or "").startswith(prefix)]


def assert_ticks_in(ax):
    for axis, sides in ((ax.xaxis, ("bottom", "top")), (ax.yaxis, ("left", "right"))):
        for which in ("major", "minor"):
            kw = axis.get_tick_params(which=which)
            assert kw.get("direction") == "in", (which, kw)
            assert all(kw.get(s) for s in sides), (which, kw)
            assert kw.get("pad") == 7
    assert ax.xaxis.get_minor_ticks() and ax.yaxis.get_minor_ticks()


def assert_shaded_binary(ax):
    lines = plain_lines(ax)
    assert lines
    for ln in lines:
        assert mcolors.to_hex(ln.get_color()) == BLACK and ln.get_linewidth() == 1.25 and ln.get_linestyle() == "-"
    assert any(ALICEBLUE in facecolors(c) for c in fills(ax))
    assert not any(type(c).__name__ == "LineCollection" for c in ax.collections)        # no tie-lines
    assert ax.get_legend() is None
    assert_ticks_in(ax)


def check_anchors(ax):
    """Every phase label (or the end of its leader line) lies in a region with exactly its phase set."""
    import warnings
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


# ---------------------------------------------------------------------------------------------------------
# map_phase_diagram
# ---------------------------------------------------------------------------------------------------------
def test_binary_shaded_default(tb):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                "tmax": 2000, "temperature_unit": "C", "output_prefix": "crni_shaded"})
    fig, ax = stored(imgs[0])
    assert_shaded_binary(ax)
    # titles and axis labels are the toolkit's usual ones
    assert ax.get_title() == "Cr-Ni" and ax.get_xlabel() == "Mole fraction Ni"
    assert ax.get_ylabel() == "Temperature (°C)"
    assert np.allclose(ax.get_xlim(), (0, 1)) and np.allclose(ax.get_ylim(), (800, 2000))
    # the invariant line at the eutectic temperature, black
    inv = [ln for ln in plain_lines(ax) if np.ptp(ln.get_ydata()) < 1e-9 and np.ptp(ln.get_xdata()) > 0.05]
    assert inv and abs(inv[0].get_ydata()[0] - 1344.70) < 0.2
    # labels instead of a legend: on by default, in black (also the invariant temperature)
    labs = labels_of(ax)
    texts = {t.get_text() for t in labs}
    assert {"L", "BCC", "FCC", "L+BCC", "BCC+FCC", "1345 °C"} <= texts, texts
    assert all(mcolors.to_hex(t.get_color()) == BLACK for t in labs)
    assert all(t.get_zorder() > max(c.get_zorder() for c in fills(ax)) for t in labs)   # fills never hide labels
    assert check_anchors(ax) >= 5
    # SVG + PDF by default, text kept as text
    svg = Path(imgs[0]).with_suffix(".svg")
    assert svg.exists() and Path(imgs[0]).with_suffix(".pdf").exists()
    s = svg.read_text()
    assert re.search(r"<text[^>]*>BCC\+FCC</text>", s) and 'id="phase_label:BCC_A2.FCC_A1"' in s
    assert "#f0f8ff" in s.lower()                                       # the aliceblue fills


def test_binary_shaded_fecr_kelvin_no_labels(tb):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrFe-82Her.tdb", "elements": ["FE", "CR"], "tmin": 600,
                                                "tmax": 1300, "label_regions": False, "output_prefix": "fecr_k"})
    fig, ax = stored(imgs[0])
    assert_shaded_binary(ax)
    assert ax.get_title() == "Fe-Cr" and ax.get_xlabel() == "Mole fraction Cr" and ax.get_ylabel() == "Temperature (K)"
    assert not labels_of(ax) and "phase labels" not in text


def test_ternary_shaded(tb):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                "temperature": 1000, "temperature_unit": "C",
                                                "output_prefix": "cocrni_shaded"})
    fig, ax = stored(imgs[0])
    lines = plain_lines(ax)
    assert lines and all(mcolors.to_hex(ln.get_color()) == GREY for ln in lines)
    tri = [ln for ln in lines if len(ln.get_xdata()) == 4 and ln.get_xdata()[0] == ln.get_xdata()[-1]]
    assert len(tri) >= 1                                                 # tie-triangle edges, grey
    assert not any(type(c).__name__ == "LineCollection" for c in ax.collections)        # no tie-lines
    assert not fills(ax) and ax.get_legend() is None
    assert tuple(fig.get_size_inches()) == (7.0, 6.0)
    assert {t.get_text() for t in ax.texts if (t.get_gid() or "").startswith("corner_label:")} == {"Co", "Cr", "Ni"}
    assert ax.get_xlabel() == "Mole fraction Cr" and ax.get_title().startswith("Co-Cr-Ni   1000 °C")
    texts = {t.get_text() for t in labels_of(ax)}
    assert {"FCC", "BCC", "BCC+FCC+σ"} <= texts, texts
    assert check_anchors(ax) >= 4


def test_isopleth_shaded_names_lines(tb):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                                "vary": "CR", "fixed": {"NI": 0.1}, "tmin": 900, "tmax": 1700,
                                                "x_range": [0, 0.4], "output_prefix": "iso_shaded"})
    fig, ax = stored(imgs[0])
    assert all(mcolors.to_hex(ln.get_color()) == BLACK and ln.get_linewidth() == 1.25 for ln in plain_lines(ax))
    assert ax.get_legend() is None
    names = [t.get_text() for t in ax.texts if (t.get_gid() or "").startswith("line_label:")]
    assert names and set(names) <= {"L", "BCC", "FCC", "σ"}, names
    assert_ticks_in(ax)


def test_color_style_is_the_old_look(tmp_path):
    from calphad_tools.agent.config import AgentConfig
    assert AgentConfig().diagram_style == "shaded"
    tbc = _toolbox(tmp_path / "out", style="color")                     # config diagram_style = "color"
    text, imgs = call(tbc, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                 "tmax": 2000, "temperature_unit": "C"})
    fig, ax = stored(imgs[0])
    from calphad_tools.mapping_tools import INVARIANT_COLOR, phase_color
    cols = {mcolors.to_hex(ln.get_color()) for ln in plain_lines(ax)}
    assert {phase_color("LIQUID"), phase_color("FCC_A1"), phase_color("BCC_A2"), INVARIANT_COLOR} <= cols
    assert any(type(c).__name__ == "LineCollection" for c in ax.collections)            # tie-lines
    assert not fills(ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["BCC_A2", "FCC_A1", "LIQUID"]
    assert not labels_of(ax) and not Path(imgs[0]).with_suffix(".svg").exists()
    assert ax.get_title() == "Cr-Ni"
    # an explicit style beats the configured default
    text, imgs = call(tbc, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                 "tmax": 2000, "temperature_unit": "C", "style": "shaded",
                                                 "label_regions": False, "output_prefix": "explicit"})
    assert_shaded_binary(stored(imgs[0])[1])


def test_style_argument_is_checked(tb):
    text, imgs, err = tb.call("map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                    "tmax": 2000, "style": "neon"})
    assert err


# ---------------------------------------------------------------------------------------------------------
# overviews and plot_diagram_with_data
# ---------------------------------------------------------------------------------------------------------
def test_overviews_shaded(tb):
    text, imgs = call(tb, "plot_all_binaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CR", "NI"],
                                                "tmin_c": 800, "tmax_c": 1800, "output_pptx": "bin_style.pptx"})
    png = next(Path(imgs[0]).parent.glob("CoCrFeNiV_Choi2019_CR-NI.png"))
    fig, ax = stored(png)
    assert_shaded_binary(ax)
    assert ax.get_title() == "Cr–Ni" and ax.get_xlabel() == "Mole fraction Ni"
    assert ax.get_ylabel() == "Temperature (°C)" and np.allclose(ax.get_ylim(), (800, 1800))
    assert {"L", "BCC", "FCC", "BCC+FCC"} <= {t.get_text() for t in labels_of(ax)}
    assert png.with_suffix(".svg").exists() and png.with_suffix(".pdf").exists()
    text, imgs = call(tb, "plot_all_ternaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                 "temperatures_c": [1200], "output_pptx": "tern_style.pptx"})
    png = next(Path(imgs[0]).parent.glob("CoCrFeNiV_Choi2019_CO-CR-NI_1200C.png"))
    fig, ax = stored(png)
    assert all(mcolors.to_hex(ln.get_color()) == GREY for ln in plain_lines(ax))
    assert ax.get_legend() is None and labels_of(ax)
    assert ax.get_title() == "Co–Cr–Ni   1200 °C (1473 K)"


def test_plot_diagram_with_data_shaded(tb, tmp_path):
    from calphad_tools import expdata
    d = tmp_path / "ds"
    zpf, _ = expdata.pop_to_zpf(str(ROOT / "examples/expdata/CrNi_boundaries.POP"), ["CR", "NI"],
                                tdb_phases=["LIQUID", "FCC_A1", "BCC_A2"])
    for i, ds in enumerate(zpf):
        expdata.write_dataset(ds, str(d / f"z{i}.json"))
    text, imgs = call(tb, "plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": str(d),
                                                     "tdb": "CrNi-06Hal.tdb", "tmin": 800, "tmax": 2000,
                                                     "temperature_unit": "C", "output_png": "pdd_shaded.png"})
    fig, ax = stored(imgs[0])
    assert any(ALICEBLUE in facecolors(c) for c in fills(ax))
    calc = [ln for ln in plain_lines(ax) if ln.get_linewidth() == 1.25]
    assert calc and all(mcolors.to_hex(ln.get_color()) == BLACK for ln in calc)
    data = [ln for ln in ax.get_lines() if ln not in plain_lines(ax)]
    assert data and any(mcolors.to_hex(ln.get_color()) != BLACK for ln in data)      # markers keep phase colours
    leg = ax.get_legend()
    texts = [t.get_text() for t in leg.get_texts()]
    assert texts and not {"LIQUID", "FCC_A1", "BCC_A2"} & set(texts)                # data references only
    fig.canvas.draw()
    lb, axb = leg.get_window_extent(), ax.get_window_extent()
    assert axb.x0 - 1 <= lb.x0 and lb.x1 <= axb.x1 + 1                              # inside the axes
    assert {"L", "BCC+FCC", "1345 °C"} <= {t.get_text() for t in labels_of(ax)}       # labels on by default
    assert ax.get_xlabel() == "Mole fraction Ni" and ax.get_title().startswith("Cr–Ni")
    assert_ticks_in(ax)
    assert Path(imgs[0]).with_suffix(".svg").exists()


# ---------------------------------------------------------------------------------------------------------
# show_code reproduces the style
# ---------------------------------------------------------------------------------------------------------
STYLE_DUMP = '''
import json as _json
import matplotlib.colors as _mc


def _style_of(fig):
    out = []
    for a in fig.axes:
        lines = sorted({(_mc.to_hex(l.get_color()), l.get_linewidth(), l.get_linestyle()) for l in a.get_lines()
                        if str(l.get_marker()) in ("None", "", " ", "none")})
        fills = sorted({_mc.to_hex(f) for c in a.collections
                        if type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection")
                        for f in c.get_facecolor()})
        ticks = [a.xaxis.get_tick_params(which=w).get("direction") for w in ("major", "minor")]
        labels = sorted({t.get_text() for t in a.texts if (t.get_gid() or "").startswith("phase_label:")})
        out.append({"lines": [list(x) for x in lines], "fills": fills, "legend": a.get_legend() is not None,
                    "ticks": ticks, "labels": labels,
                    "tielines": any(type(c).__name__ == "LineCollection" for c in a.collections)})
    return out
'''
exec(STYLE_DUMP)  # noqa: S102 - same extraction in the test and in the generated script


def _run_show_code(tb, png, tmp_path):
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, png)
    assert all(c.standalone for c in cells), [(c.title, c.note) for c in cells]
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script and "def draw_shaded" in script
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "gen.py").write_text(script + STYLE_DUMP + "\n_json.dump(_style_of(fig), open('style.json', 'w'))\n")
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads((d / "style.json").read_text()), d / "standalone_outputs"


def test_show_code_reproduces_shaded_binary(tb, tmp_path):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                "tmax": 2000, "temperature_unit": "C", "composition_step": 0.05,
                                                "output_prefix": "crni_code"})
    got, out = _run_show_code(tb, imgs[0], tmp_path)
    want = _style_of(stored(imgs[0])[0])
    assert got[0]["lines"] == want[0]["lines"] == [[BLACK, 1.25, "-"]]
    assert got[0]["fills"] == want[0]["fills"] == [ALICEBLUE]
    assert got[0]["ticks"] == want[0]["ticks"] == ["in", "in"]
    assert not got[0]["legend"] and not got[0]["tielines"]
    assert got[0]["labels"] == want[0]["labels"] and len(got[0]["labels"]) >= 5
    assert (out / "crni_code.svg").exists() and (out / "crni_code.pdf").exists()


def test_show_code_reproduces_shaded_ternary_overview(tb, tmp_path):
    text, imgs = call(tb, "plot_all_ternaries", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                 "temperatures_c": [1000], "output_pptx": "tern_code.pptx"})
    got, out = _run_show_code(tb, imgs[0], tmp_path)
    single = next(Path(imgs[0]).parent.glob("CoCrFeNiV_Choi2019_CO-CR-NI_1000C.png"))
    want = _style_of(stored(single)[0])
    assert got[0]["lines"] == want[0]["lines"] == [[GREY, 1.0, "-"]]
    assert not got[0]["fills"] and not got[0]["legend"] and not got[0]["tielines"]
    assert len(set(want[0]["labels"]) & set(got[0]["labels"])) >= 3 and len(got[0]["labels"]) >= 4


def test_color_overview_and_its_code(tmp_path):
    tbc = _toolbox(tmp_path / "out", style="color")
    text, imgs = call(tbc, "plot_all_binaries", {"tdb": "CrNi-06Hal.tdb", "tmin_c": 900, "tmax_c": 1800})
    png = next(Path(imgs[0]).parent.glob("CrNi-06Hal_CR-NI.png"))
    fig, ax = stored(png)
    assert ax.get_legend() is not None and not fills(ax)
    assert any(type(c).__name__ == "LineCollection" for c in ax.collections)            # binplot tie-lines
    assert not png.with_suffix(".svg").exists()
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tbc, imgs[0])
    script = codegen.to_script(tbc, cells, ctx)
    assert "binplot(" in script and "draw_shaded" not in script
    compile(script, "gen.py", "exec")
