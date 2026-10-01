"""Phase-diagram figure -> ESPEI ZPF datasets (digitize_setup -> digitize_points -> digitized_to_zpf).

Synthetic "paper figures" with known ground truth (tests/digitize_zpf_bench.py).  The main path uses no database:
the phases come from the figure (region labels + the regions bounded by its drawn lines, and/or the legend);
the true TDB is then used only to prove the datasets are physically right (ESPEI's ZPF residual).
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import digitize_zpf_bench as ZB  # noqa: E402

NAME_MAP = {"(Cr)": "BCC_A2", "(Ni)": "FCC_A1"}
LEGEND_WORDS = {"liquidus": "liquidus", "solidus": "solidus", "bcc_solvus": "solvus", "fcc_solvus": "solvus",
                "eutectic": "invariant"}


class _UI:
    def ask(self, q, o=None):
        return ""

    def confirm(self, t, b):
        return False

    def progress(self, m):
        pass

    def notice(self, k, t):
        pass


def _toolbox(out):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(_UI, UI):
        pass
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out), ui=U(), python_tool="off")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def _major_ticks(text, axis):
    return json.loads(re.search(rf"{axis}0\.\.{axis}\d+ at px (\[.*?\])", text).group(1))


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def digitize_binary(tb, d, name, wt=False, labels=ZB.CRNI_LABELS):
    """What the agent does: set up, read the tick labels, extract each legend entry's markers."""
    path, truth, info = ZB.make_binary_figure(d, name=name, wt=wt, labels=labels)
    text, prev = call(tb, "digitize_setup", {"image": path})
    xt, yt = info["xticks"], info["yticks"]
    nx, ny = len(_major_ticks(text, "x")), len(_major_ticks(text, "y"))
    assert (nx, ny) == (len(xt), len(yt)), text            # inward ticks on a boxed frame
    axes = {"x": {"ticks": {"0": xt[0], str(nx - 1): xt[-1]}}, "y": {"ticks": {"0": yt[-1], str(ny - 1): yt[0]}}}
    csvs = []
    for g, st in ZB.STYLES.items():
        t, _ = call(tb, "digitize_points", {"image": path, "axes": axes, "color": st["color"], "series": g})
        csvs.append(re.search(r"-> (\S+\.csv)", t).group(1))
    return {"path": path, "truth": truth, "info": info, "axes": axes, "csvs": csvs, "wt": wt}


def regions_for(fig, labels=None):
    conv = (lambda x: float(ZB.x_to_wt(x))) if fig["wt"] else (lambda x: 100 * x)
    return [{"label": lab, "at": [conv(x), T]} for lab, x, T in (labels or fig["info"]["labels"])]


def to_zpf(tb, fig, **kw):
    args = {"components": ["CR", "NI"], "csv": fig["csvs"], "image": fig["path"], "axes": fig["axes"],
            "composition_unit": "wt%" if fig["wt"] else "at%", "temperature_unit": "C",
            "pair_tielines": {"T_tol": 8}, "figure": "synthetic Cr-Ni paper figure", **kw}
    text, imgs = call(tb, "digitized_to_zpf", args)
    summary = re.search(r"summary CSV: (\S+)", text).group(1)
    return text, imgs, _rows(tb.p(summary)), args


def match(truth, rows, tol_x=0.012, tol_T=12.0):
    """truth index -> summary row (nearest digitized point within tolerance)."""
    X = np.array([[float(r["x"]), float(r["T"])] for r in rows])
    out = {}
    for i, t in enumerate(truth):
        d = np.hypot((X[:, 0] - t["x"]) / tol_x, (X[:, 1] - t["T"]) / tol_T)
        j = int(np.argmin(d))
        if d[j] <= 1.0:
            out[i] = rows[j]
    return out


def accuracy(truth, m):
    ok = [i for i, r in m.items() if r["phase"] == truth[i]["phase"]
          and set(filter(None, r["coexisting"].split("+"))) == set(truth[i]["coexisting"])]
    return len(ok) / len(truth), [i for i in m if i not in ok]


def check_pairs_and_invariant(truth, m):
    for pid in ("s0", "s1", "l0"):
        idx = [i for i, t in enumerate(truth) if t["pair"] == pid]
        assert all(i in m for i in idx)
        rows = {m[i]["zpf_row"] for i in idx}
        assert len(rows) == 1 and "" not in rows, (pid, [m[i] for i in idx])
    inv = [i for i, t in enumerate(truth) if t["pair"] == "inv"]
    assert len({m[i]["zpf_row"] for i in inv}) == 1
    assert all(m[i]["zpf_file"].endswith("_inv.json") for i in inv)


def zpf_residual(tdb, datasets_dir):
    from calphad_tools.param_tools import Objective
    o = Objective(tdb, str(datasets_dir), [])
    return np.abs(o.blocks(o.x0)["ZPF"]["r"])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("zpf")
    return _toolbox(d / "out"), d


@pytest.fixture(scope="module")
def crni(env):
    tb, d = env
    return digitize_binary(tb, d, "crni_paper")


# ---------------------------------------------------------------------------------------------- binary, no TDB
def test_binary_from_figure_labels_no_tdb(env, crni):
    """Main path: region labels in paper notation + name map; no legend, no database."""
    tb, d = env
    text, imgs, rows, _ = to_zpf(tb, crni, regions=regions_for(crni), name_map=NAME_MAP, prefix="crni_regions")
    truth = crni["truth"]
    m = match(truth, rows)
    assert len(m) == len(truth) == len(rows)                          # every point recovered, nothing spurious
    err_x = [abs(float(r["x"]) - truth[i]["x"]) for i, r in m.items()]
    err_T = [abs(float(r["T"]) - truth[i]["T"]) for i, r in m.items()]
    assert max(err_x) < 0.006 and np.sqrt(np.mean(np.square(err_T))) < 4.0, (max(err_x), max(err_T))
    acc, wrong = accuracy(truth, m)
    assert acc >= 0.95, (acc, [(truth[i], m[i]) for i in wrong])
    check_pairs_and_invariant(truth, m)
    assert "ESPEI check: 4 valid, 0 invalid; load_datasets: 4 loaded" in text
    assert "(Cr) -> BCC_A2" in text and "(Ni) -> FCC_A1" in text and "phase models" in text
    assert "assignment preview" in text and Path(imgs[0]).exists()
    ds_dir = tb.out / "datasets" / "crni_regions"
    # physically right: ESPEI's ZPF driving forces of the true database are ~0, a perturbed one is clearly worse
    from calphad_tools.param_tools import edit_tdb
    r0 = zpf_residual(ZB.CRNI, ds_dir)
    pert = str(d / "CrNi_perturbed.tdb")
    edit_tdb(ZB.CRNI, [{"target": "L(LIQUID,CR,NI;0)", "op": "add", "value": 8000}], pert)
    r1 = zpf_residual(pert, ds_dir)
    assert np.mean(r0) < 60, np.mean(r0)
    assert np.mean(r1) > 5 * np.mean(r0), (np.mean(r0), np.mean(r1))


def test_binary_missing_label_is_inferred(env):
    """'(Cr)+(Ni)' is not printed on the figure: the field between (Cr) and (Ni) is inferred as their two-phase
    field (the L+(Ni) lens is never labelled: too narrow)."""
    tb, d = env
    labels = [lab for lab in ZB.CRNI_LABELS if lab[0] != "(Cr)+(Ni)"]
    fig = digitize_binary(tb, d, "crni_paper_nolabel", labels=labels)
    text, imgs, rows, _ = to_zpf(tb, fig, regions=regions_for(fig), name_map=NAME_MAP, prefix="crni_missing")
    assert re.search(r"inferred as BCC_A2\+FCC_A1 from its neighbours", text), text
    m = match(fig["truth"], rows)
    acc, wrong = accuracy(fig["truth"], m)
    assert acc >= 0.95, acc
    check_pairs_and_invariant(fig["truth"], m)


def test_binary_legend_plus_regions(env, crni):
    """Legend words (liquidus / solidus / solvus / invariant) combined with the region labels."""
    tb, _ = env
    text, _, rows, _ = to_zpf(tb, crni, regions=regions_for(crni), name_map=NAME_MAP, assignments=LEGEND_WORDS,
                              prefix="crni_legend_regions")
    truth = crni["truth"]
    m = match(truth, rows)
    acc, wrong = accuracy(truth, m)
    assert acc == 1.0, [(truth[i], m[i]) for i in wrong]
    check_pairs_and_invariant(truth, m)


def test_binary_legend_only(env, crni):
    """No region labels: the legend alone (read by the model), with composition ranges where a boundary changes
    partner at the eutectic."""
    tb, _ = env
    sides = {"(Cr)": [0, 46], "(Ni)": [46, 100]}
    legend = {"liquidus": {"boundary": "liquidus", "with": sides},
              "solidus": {"boundary": "solidus", "phase": sides},
              "bcc_solvus": {"phase": "(Cr)", "coexisting": ["(Ni)"]},
              "fcc_solvus": {"phase": "(Ni)", "coexisting": ["(Cr)"]},
              "eutectic": {"boundary": "invariant", "phases": ["(Cr)", "L", "(Ni)"]}}
    text, _, rows, _ = to_zpf(tb, crni, assignments=legend, name_map=NAME_MAP, prefix="crni_legend")
    truth = crni["truth"]
    m = match(truth, rows)
    acc, wrong = accuracy(truth, m)
    assert acc == 1.0, [(truth[i], m[i]) for i in wrong]
    check_pairs_and_invariant(truth, m)
    assert "ESPEI check: 4 valid" in text


def test_name_map_errors_and_suggestions(env, crni):
    tb, _ = env
    args = {"components": ["CR", "NI"], "csv": crni["csvs"], "image": crni["path"], "axes": crni["axes"],
            "regions": regions_for(crni), "composition_unit": "at%", "temperature_unit": "C", "preview": True}
    text, _, err = tb.call("digitized_to_zpf", args)
    assert err and "'(Cr)' (suggestion: BCC_A2" in text and "'(Ni)' (suggestion: FCC_A1" in text
    text, imgs = call(tb, "digitized_to_zpf", {**args, "accept_suggestions": True})
    assert "PREVIEW ONLY" in text and "(Cr) -> BCC_A2 (suggestion accepted" in text
    assert Path(imgs[0]).exists()


def test_overrides_and_exclusions(env, crni):
    tb, _ = env
    text, _, rows, _ = to_zpf(tb, crni, regions=regions_for(crni), name_map=NAME_MAP, prefix="crni_override",
                              overrides={"0": {"phase": "(Cr)", "coexisting": ["L"]}}, exclude_points=[1])
    r = {int(x["point"]): x for x in rows}
    assert r[0]["phase"] == "BCC_A2" and r[0]["coexisting"] == "LIQUID" and r[0]["method"] == "override"
    assert r[1]["zpf_row"] == "" and "excluded" in r[1]["flag"]


# ---------------------------------------------------------------------------------------------- with a TDB
def test_binary_tdb_inference(env, crni):
    """Optional database: inference for points nothing else assigns (points generated on its own boundaries)."""
    tb, _ = env
    text, imgs, rows, _ = to_zpf(tb, crni, tdb="CrNi-06Hal.tdb", prefix="crni_tdb")
    truth = crni["truth"]
    m = match(truth, rows)
    acc, wrong = accuracy(truth, m)
    assert acc >= 0.95, (acc, [(truth[i], m[i]) for i in wrong])
    check_pairs_and_invariant(truth, m)
    assert "SUGGESTIONS" in text or "suggestions" in text
    assert Path(imgs[-1]).exists()


def test_figure_vs_tdb_disagreement_and_far_points(env):
    """Points shifted off the database boundary are flagged, never silently re-assigned; with a figure/legend
    assignment the visual one is kept and the disagreement reported."""
    tb, d = env
    truth, lines, inv = ZB.crni_truth(seed=3)
    pts = [[100 * t["x"], t["T"] - ZB.KELVIN, "fcc_solvus"] for t in truth if t["group"] == "fcc_solvus"]
    shifted = [0, 2]
    for k in shifted:
        pts[k][0] += 9.0                                          # 0.09 in x: well inside the (Ni) field
    common = {"components": ["CR", "NI"], "points": pts, "composition_unit": "at%", "temperature_unit": "C",
              "tdb": "CrNi-06Hal.tdb"}
    text, _ = call(tb, "digitized_to_zpf", {**common, "prefix": "far_infer"})
    rows = _rows(tb.p(re.search(r"summary CSV: (\S+)", text).group(1)))
    far = [int(r["point"]) for r in rows if "far from calculated boundary" in r["flag"]]
    assert sorted(far) == shifted, rows
    text, _ = call(tb, "digitized_to_zpf", {**common, "prefix": "far_legend", "name_map": NAME_MAP,
                                            "assignments": {"fcc_solvus": {"phase": "(Ni)", "coexisting": ["(Cr)"]}}})
    rows = _rows(tb.p(re.search(r"summary CSV: (\S+)", text).group(1)))
    assert all(r["phase"] == "FCC_A1" and r["method"] == "legend" for r in rows)
    far = [int(r["point"]) for r in rows if "far from calculated boundary" in r["flag"]]
    assert sorted(far) == shifted


# ---------------------------------------------------------------------------------------------- units
def test_wt_percent_axis(env):
    tb, d = env
    fig = digitize_binary(tb, d, "crni_paper_wt", wt=True)
    text, _, rows, _ = to_zpf(tb, fig, regions=regions_for(fig), name_map=NAME_MAP, prefix="crni_wt")
    truth = fig["truth"]
    m = match(truth, rows)
    assert len(m) == len(truth)
    assert max(abs(float(r["x"]) - truth[i]["x"]) for i, r in m.items()) < 0.006
    assert accuracy(truth, m)[0] >= 0.95
    assert "from wt%" in text
    ds = json.loads(next((tb.out / "datasets" / "crni_wt").glob("*BCC_A2-FCC_A1.json")).read_text())
    assert "composition converted from wt%" in ds["comment"]


def test_swapped_axes_inline_points(env):
    """x = temperature (°C), y = at% Ni; inline points with the legend's assignment."""
    tb, _ = env
    text, _ = call(tb, "digitized_to_zpf", {
        "components": ["CR", "NI"], "points": [[1200, 55.0, "fcc"], [1300, 52.0, "fcc"]], "swap_axes": True,
        "composition_unit": "at%", "temperature_unit": "C", "prefix": "swapped",
        "assignments": {"fcc": {"phase": "FCC_A1", "coexisting": ["BCC_A2"]}}})
    ds = json.loads(next((tb.out / "datasets" / "swapped").glob("*.json")).read_text())
    assert ds["conditions"]["T"] == [1473.15, 1573.15]
    assert ds["values"][0] == [["FCC_A1", ["NI"], [0.55]], ["BCC_A2", ["NI"], [None]]]


def test_drawn_line_pseudo_data(env, crni):
    """A traced line (no markers): sampled at T steps, phases from the regions, marked as pseudo-data."""
    tb, _ = env
    t, _ = call(tb, "digitize_curve", {"image": crni["path"], "axes": crni["axes"], "color": "black",
                                       "series": "drawn_liquidus", "x_values": list(np.arange(6, 40, 2.0)),
                                       "start_at": [20, 1690]})
    curve = re.search(r"-> (\S+\.csv)", t).group(1)
    text, _ = call(tb, "digitized_to_zpf", {"components": ["CR", "NI"], "csv": [curve], "image": crni["path"],
                                            "axes": crni["axes"], "regions": regions_for(crni), "name_map": NAME_MAP,
                                            "composition_unit": "at%", "temperature_unit": "C",
                                            "line_sampling": {"T_step": 50}, "prefix": "crni_line"})
    rows = _rows(tb.p(re.search(r"summary CSV: (\S+)", text).group(1)))
    assert len(rows) >= 5
    good = [r for r in rows if r["phase"] == "LIQUID" and r["coexisting"] == "BCC_A2"]
    assert len(good) >= 0.9 * len(rows), rows
    assert all("drawn line" in r["method"] for r in good)
    f = next((tb.out / "datasets" / "crni_line").glob("*_line.json"))
    assert "PSEUDO-DATA digitized from a drawn line" in json.loads(f.read_text())["comment"]


# ---------------------------------------------------------------------------------------------- ternary
@pytest.fixture(scope="module")
def tern(env):
    tb, d = env
    path, truth = ZB.make_ternary_figure(d)
    text, prev = call(tb, "digitize_setup", {"image": path})
    assert "composition TRIANGLE" in text
    c = [tuple(map(float, m)) for m in re.findall(r"c\d \(([\d.]+), ([\d.]+)\)", text)]
    assert len(c) == 3
    axes = {"ternary": {"corners": {"c0": "NI", "c1": "CR", "c2": "CO"}}}
    csvs = []
    for ph, st in ZB.TERN_STYLES.items():
        t, _ = call(tb, "digitize_points", {"image": path, "axes": axes, "color": st["color"], "series": st["label"]})
        csvs.append(re.search(r"-> (\S+\.csv)", t).group(1))
    return {"path": path, "truth": truth, "axes": axes, "csvs": csvs, "setup": text}


def _tern_check(truth, rows):
    X = np.array([[float(r["x"]), float(r["y"])] for r in rows])          # X(CR), X(NI)
    ok, errs, rowid = 0, [], {}
    for t in truth:
        d = np.hypot(X[:, 0] - t["X"]["CR"], X[:, 1] - t["X"]["NI"])
        j = int(np.argmin(d))
        assert d[j] < 0.02, (t, d[j])
        errs.append(d[j])
        r = rows[j]
        ok += r["phase"] == t["phase"] and set(r["coexisting"].split("+")) == set(t["coexisting"])
        rowid.setdefault(t["tie"], set()).add(r["zpf_row"])
    return ok / len(truth), max(errs), rowid


@pytest.mark.parametrize("mode", ["legend", "regions"])
def test_ternary_isotherm(env, tern, mode):
    """Gibbs triangle calibrated by its corners; tie-lines / the tie-triangle joined by drawn lines."""
    tb, _ = env
    args = {"components": ["CO", "CR", "NI"], "csv": tern["csvs"], "image": tern["path"], "axes": tern["axes"],
            "temperature": 1000, "temperature_unit": "C", "tie_lines": True, "prefix": f"cocrni_{mode}",
            "name_map": {"fcc": "FCC_A1", "σ": "SIGMA", "bcc": "BCC_A2"}}
    if mode == "legend":
        args["assignments"] = {"fcc": {"phase": "fcc"}, "σ": {"phase": "σ"}, "bcc": {"phase": "bcc"}}
    else:
        args["regions"] = [{"label": lab, "at": X} for lab, X in ZB.TERN_LABELS]
    text, imgs = call(tb, "digitized_to_zpf", args)
    rows = _rows(tb.p(re.search(r"summary CSV: (\S+)", text).group(1)))
    acc, err, rowid = _tern_check(tern["truth"], rows)
    assert acc == 1.0 and err < 0.01, (acc, err, text)
    assert all(len(r) == 1 and "" not in r for r in rowid.values()), rowid     # every tie-line / triangle is one row
    assert "tie-triangle(s)" in text and "1 tie-triangle(s)" in text
    files = list((tb.out / "datasets" / f"cocrni_{mode}").glob("*.json"))
    from espei.datasets import load_datasets
    dbs = load_datasets([str(f) for f in files])
    for ds in dbs.all():
        for row in ds["values"]:
            assert all(len(v[1]) == 2 and len(v[2]) == 2 for v in row)       # two independent components
    tri = json.loads(next(f for f in files if f.name.endswith("_inv.json")).read_text())
    assert len(tri["values"][0]) == 3 and all(None not in v[2] for v in tri["values"][0])


def test_ternary_calibration_by_edge_ticks(tern):
    """Corners and edge ticks give the same affine pixel -> composition map."""
    from calphad_tools import digitize as D
    rgb = D.load_rgb(tern["path"])
    tri = D.detect_triangle(rgb)
    assert tri is not None and len(tri.ticks["b"]) >= 5
    by_corners = D.calibrate_triangle({"corners": {"c0": "NI", "c1": "CR", "c2": "CO"}}, tri)
    a, b = [t for t in tri.ticks["b"] if "index" in t], [t for t in tri.ticks["r"] if "index" in t]
    # bottom edge runs Ni -> Cr, right edge Cr -> Co; the model reads the tick labels (here: fractions along the edge)
    ticks = {f"b{t['index']}": {"CR": round(t["t"], 1), "NI": round(1 - t["t"], 1)} for t in a[1:-1:3]}
    ticks.update({f"r{t['index']}": {"CO": round(t["t"], 1), "CR": round(1 - t["t"], 1)} for t in b[2:4]})
    by_ticks = D.calibrate_triangle({"ticks": ticks}, tri)
    pts = np.array([[300, 400], [450, 450], [360, 250]], float)
    f1 = by_corners.to_fractions(pts[:, 0], pts[:, 1])
    f2 = by_ticks.to_fractions(pts[:, 0], pts[:, 1])
    order = [by_ticks.elements.index(e) for e in by_corners.elements]
    assert np.max(np.abs(f1 - f2[:, order])) < 0.006
    assert by_ticks.residual < 0.006


# ---------------------------------------------------------------------------------------------- show_code
def test_show_code_reproduces_json(env, crni, tmp_path):
    tb, _ = env
    text, _, rows, args = to_zpf(tb, crni, regions=regions_for(crni), name_map=NAME_MAP, prefix="crni_code")
    summary = re.search(r"summary CSV: (\S+)", text).group(1)
    t, _ = call(tb, "show_code", {"target": summary, "format": "script"})
    py = re.search(r"saved (\S+\.py)", t).group(1)
    src = Path(tb.p(py)).read_text()
    assert "calphad_tools" not in src and "rows_from_points" in src
    r = subprocess.run([sys.executable, tb.p(py)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    made = sorted((tmp_path / "standalone_outputs" / "datasets" / "crni_code").glob("*.json"))
    ref = sorted((tb.out / "datasets" / "crni_code").glob("*.json"))
    assert [f.name for f in made] == [f.name for f in ref] and len(made) == 4
    for a, b in zip(made, ref):
        assert json.loads(a.read_text()) == json.loads(b.read_text())


def test_data_plot_orientation(tmp_path):
    """right / corners orient only the data plot: mirrored binary x, permuted ternary corners, same ZPF JSON."""
    from calphad_tools import digitize_zpf as Z
    pts = [[1200, 55.0, "fcc"], [1300, 52.0, "fcc"]]
    common = dict(points=pts, swap_axes=True, composition_unit="at%", temperature_unit="C",
                  assignments={"fcc": {"phase": "FCC_A1", "coexisting": ["BCC_A2"]}})
    xs, jsons = {}, {}
    for right in ("NI", "CR"):
        od = tmp_path / right
        res = Z.convert(["CR", "NI"], str(od), overlay_png=str(tmp_path / f"{right}.png"), right=right, **common)
        assert res["display_order"][1] == right
        jsons[right] = sorted(json.loads(f.read_text())["values"] for f in od.glob("*.json"))
        # re-draw and capture what is actually scattered (x of the data points)
        from matplotlib.axes import Axes
        seen, orig = [], Axes.scatter

        def spy(self, x, y, *a, **k):
            if k.get("zorder") == 4:
                seen.extend(np.atleast_1d(x).tolist())
            return orig(self, x, y, *a, **k)
        Axes.scatter = spy
        try:
            Z.plot_overlay(res, str(tmp_path / f"{right}_2.png"), right=right)
        finally:
            Axes.scatter = orig
        xs[right] = sorted(seen)
    assert jsons["NI"] == jsons["CR"]                       # datasets do not depend on the orientation
    assert np.allclose(xs["NI"], [0.52, 0.55]) and np.allclose(xs["CR"], [0.45, 0.48])
    sysm = Z.System("ternary", ["CO", "CR", "NI"], ["CR", "NI"], "fraction", "C", 1273.15)
    assert Z.display_order(sysm) == ["CO", "CR", "NI"]
    assert Z.display_order(sysm, corners={"left": "NI", "right": "CO", "top": "CR"}) == ["NI", "CO", "CR"]
    with pytest.raises(ValueError):
        Z.display_order(sysm, corners={"left": "FE"})
