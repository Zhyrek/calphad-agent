"""Tests for the data preparation layer (calphad_tools/figdata.py): sources, where, transform steps, tools."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calphad_tools import figdata as fd

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


def T(df, steps):
    return fd.run_transform(df, steps)


@pytest.fixture
def cp_frame():
    return pd.DataFrame({"T [K]": [300., 500, 700, 900, 1100, 1300], "cp": [25., 27, 30, 33, 35, 36],
                         "phase": ["FCC_A1", "FCC_A1", "FCC_A1", "BCC_A2", "BCC_A2", "LIQUID"]})


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def test_text_sources_autodetect(tmp_path):
    (tmp_path / "a.csv").write_text("# measured 2024\n# by DSC\nT,cp\n300,25.5\n500,27\n")
    (tmp_path / "b.csv").write_text("T;cp\n300;25,5\n500;27,0\n")                    # European
    (tmp_path / "c.txt").write_text("T   cp\n300  25.5\n500  27\n")                   # whitespace
    (tmp_path / "d.tsv").write_text("T\tcp\n300\t25.5\n500\t27\n")
    for name in "a.csv", "b.csv", "c.txt", "d.tsv":
        df = fd.load_source(str(tmp_path / name))
        assert list(df.columns) == ["T", "cp"], name
        assert np.allclose(df["cp"], [25.5, 27]), name
    pd.DataFrame({"X(CR)": [0.1, 0.2], "ACR(CR,BCC_A2)": [0.05, 0.1], "ph": ["FCC_A1", "BCC_A2"]}).to_csv(
        tmp_path / "quoted.csv", index=False)                                        # quoted comma in a header
    assert list(fd.load_source(str(tmp_path / "quoted.csv")).columns) == ["X(CR)", "ACR(CR,BCC_A2)", "ph"]
    df = fd.load_source(str(tmp_path / "a.csv"), read={"skiprows": [0, 1, 3]})       # explicit read options
    assert list(df["T"]) == [500]


def test_excel_json_netcdf_inline_sources(tmp_path):
    xr = pytest.importorskip("xarray")
    x = tmp_path / "book.xlsx"
    with pd.ExcelWriter(x) as w:
        pd.DataFrame({"T": [1, 2]}).to_excel(w, sheet_name="first", index=False)
        pd.DataFrame({"T": [300, 400], "H": [1.5, 2.5]}).to_excel(w, sheet_name="calc", index=False)
    assert list(fd.load_source(str(x))["T"]) == [1, 2]
    assert list(fd.load_source(str(x), sheet="calc")["H"]) == [1.5, 2.5]
    with pytest.raises(ValueError, match="sheets: \\['first', 'calc'\\]"):
        fd.load_source(str(x), sheet="nope")

    (tmp_path / "rec.json").write_text(json.dumps([{"T": 300, "p": {"name": "FCC"}}, {"T": 400, "p": {"name": "BCC"}}]))
    (tmp_path / "cols.json").write_text(json.dumps({"T": [300, 400], "cp": [25, 26]}))
    (tmp_path / "lines.jsonl").write_text('{"T": 300}\n{"T": 400}\n')
    assert list(fd.load_source(str(tmp_path / "rec.json")).columns) == ["T", "p.name"]
    assert list(fd.load_source(str(tmp_path / "cols.json"))["cp"]) == [25, 26]
    assert list(fd.load_source(str(tmp_path / "lines.jsonl"))["T"]) == [300, 400]

    ds = xr.Dataset({"GM": (("T", "X"), np.arange(6.0).reshape(3, 2)), "other": (("T",), [1.0, 2, 3])},
                    coords={"T": [300.0, 400, 500], "X": [0.1, 0.2]})
    ds["GM"].attrs["units"] = "J/mol"
    nc = tmp_path / "calc.nc"
    ds.to_netcdf(nc, engine="scipy")
    df = fd.load_source(str(nc), variables=["GM"])
    assert list(df.columns) == ["T", "X", "GM"] and len(df) == 6
    assert df.set_index(["T", "X"]).loc[(500.0, 0.2), "GM"] == 5.0
    assert any("GM(T, X)" in ln and "J/mol" in ln for ln in fd.file_notes(str(nc)))
    with pytest.raises(ValueError, match="data variables"):
        fd.load_source(str(nc), variables=["nope"])

    assert list(fd.load_source([{"a": 1}, {"a": 2}])["a"]) == [1, 2]
    assert list(fd.load_source({"a": [1, 2], "b": [3, 4]})["b"]) == [3, 4]
    assert list(fd.load_source("a,b\n1,2\n3,4\n")["b"]) == [2, 4]
    spec = {"data": {"a": [1, 2, 3]}, "where": {"a": {">": 1}}}                        # nested spec dict
    assert list(fd.load_source(spec)["a"]) == [2, 3]


def test_where_comparisons(cp_frame):
    w = lambda cond: fd.apply_where(cp_frame, cond)  # noqa: E731
    assert list(w({"T": {">": 800, "<=": 1100}})["cp"]) == [33, 35]                   # prefix "T" -> "T [K]"
    assert list(w({"T": {"between": [500, 700]}})["cp"]) == [27, 30]
    assert list(w({"phase": {"contains": "bcc"}})["cp"]) == [33, 35]
    assert list(w({"phase": {"not": ["FCC_A1", "LIQUID"]}})["cp"]) == [33, 35]
    assert list(w({"phase": "LIQUID"})["cp"]) == [36]                                 # old equality semantics
    assert list(w({"phase": ["LIQUID", "BCC_A2"], "cp": {"!=": 33}})["cp"]) == [35, 36]
    with pytest.raises(ValueError, match="columns: \\['T \\[K\\]', 'cp', 'phase'\\]"):
        w({"nope": 1})
    with pytest.raises(ValueError, match="unknown condition"):
        w({"cp": {"~": 1}})


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------
def test_derive_and_filter_expressions(cp_frame):
    df = T(cp_frame, [
        {"step": "derive", "new": "T_C", "expr": "`T [K]` - 273.15"},
        {"step": "derive", "exprs": {"invT": "1e4 / col('T [K]')", "lncp": "log(cp)", "S": "cp / R",
                                     "hot": "where(T_C > 500, 'hot', 'cold')", "cpmax": "max(cp)",
                                     "clipped": "clip(cp, 26, 34)"}},
        {"step": "filter", "expr": "T_C >= 200 and not phase == 'LIQUID' and phase in ['FCC_A1', 'BCC_A2']"}])
    assert np.allclose(df["T_C"], [226.85, 426.85, 626.85, 826.85])
    assert np.allclose(df["invT"], 1e4 / np.array([500, 700, 900, 1100]))
    assert np.allclose(df["S"], df["cp"] / 8.314462618) and (df["cpmax"] == 36).all()
    assert list(df["hot"]) == ["cold", "cold", "hot", "hot"] and list(df["clipped"]) == [27, 30, 33, 34]
    assert list(T(cp_frame, [{"step": "filter", "expr": "contains(phase, 'fcc')"}])["cp"]) == [25, 27, 30]


@pytest.mark.parametrize("expr, msg", [
    ("__import__('os')", "function '__import__' not allowed"),
    ("__import__('os').system('ls')", "only plain function calls"),
    ("cp.__class__", "attribute access"),
    ("(lambda: 1)()", "only plain function calls"),
    ("lambda x: x", "Lambda is not allowed"),
    ("[c for c in cp]", "ListComp is not allowed"),
    ("open('/etc/passwd')", "function 'open' not allowed"),
    ("__builtins__", "not allowed"),
    ("cp[0]", "Subscript is not allowed"),
    ("nosuchcol + 1", "unknown name 'nosuchcol'"),
    ("phase = 'FCC'", "use == for equality"),
    ("round(cp, ndigits=1)", "keyword arguments"),
])
def test_safe_evaluator_rejects(cp_frame, expr, msg):
    with pytest.raises(ValueError, match=msg.replace("(", "\\(").replace("[", "\\[")):
        fd.evaluate(cp_frame, expr)


def test_evaluator_huge_power_does_not_hang(cp_frame):
    assert np.isinf(fd.evaluate(cp_frame, "10 ** 10 ** 10"))


# ---------------------------------------------------------------------------
# reshaping steps
# ---------------------------------------------------------------------------
def test_basic_steps(cp_frame):
    df = T(cp_frame, [{"step": "rename", "map": {"T": "temp"}}, {"step": "sort", "by": "cp", "ascending": False},
                      {"step": "select", "cols": ["temp", "cp"]}, {"step": "head", "n": 2}])
    assert list(df.columns) == ["temp", "cp"] and list(df["cp"]) == [36, 35]
    df = T(cp_frame, [{"step": "dedupe", "cols": ["phase"], "keep": "last"}, {"step": "drop", "cols": ["phase"]}])
    assert list(df["cp"]) == [30, 35, 36] and "phase" not in df
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, None]})
    assert len(T(df, [{"step": "dropna", "cols": ["a"]}])) == 2 and len(T(df, [{"step": "dropna"}])) == 1
    assert list(T(cp_frame, [{"step": "tail", "n": 1, "by": "phase"}])["cp"]) == [30, 35, 36]


def test_groupby_and_melt(cp_frame):
    g = T(cp_frame, [{"step": "groupby", "by": ["phase"], "agg": {"cp": ["mean", "std", "count", "q50"]}}])
    assert list(g.columns) == ["phase", "cp_mean", "cp_std", "cp_count", "cp_q50", "n"]
    row = g.set_index("phase").loc["FCC_A1"]
    assert row["cp_mean"] == pytest.approx(82 / 3) and row["cp_std"] == pytest.approx(np.std([25, 27, 30], ddof=1))
    all_mean = T(cp_frame, [{"step": "groupby", "by": "phase"}])
    assert set(all_mean.columns) == {"phase", "T [K]_mean", "cp_mean", "n"}
    wide = pd.DataFrame({"T": [300, 400], "calc": [1.0, 2.0], "exp": [1.1, 2.1]})
    long = T(wide, [{"step": "melt", "id_vars": ["T"], "var_name": "source", "value_name": "H"}])
    assert len(long) == 4 and set(long["source"]) == {"calc", "exp"}


def test_pivot_contour_grid(tb, tmp_path):
    xs, ts = np.array([0.1, 0.2, 0.3]), np.array([300.0, 400.0])
    long = pd.DataFrame([{"X(CR)": x, "T": t, "G": 1000 * x + t} for t in ts for x in xs])
    csv = tmp_path / "grid.csv"
    long.sample(frac=1, random_state=0).to_csv(csv, index=False)                    # shuffled rows
    wide = fd.load_source(str(csv), transform=[{"step": "pivot", "index": "T", "columns": "X(CR)", "values": "G"}])
    assert list(wide.columns) == ["T", "0.1", "0.2", "0.3"] and list(wide["T"]) == [300, 400]
    z = wide.drop(columns="T").to_numpy()                                           # z[len(T)][len(X)]
    assert np.allclose(z, 1000 * xs[None, :] + ts[:, None])
    # transform_data -> json -> contourf in make_plot
    text, _ = call(tb, "transform_data", {"source": str(csv), "output": "grid.json",
                                          "transform": [{"step": "pivot", "index": "T", "columns": "X",
                                                         "values": "G"}]})
    rec = json.loads((tmp_path / "out" / "grid.json").read_text())
    zz = [[r[k] for k in ("0.1", "0.2", "0.3")] for r in rec]
    _, imgs = call(tb, "make_plot", {"output": "contour.png", "panels": [{"operations": [
        {"op": "contourf", "x": xs.tolist(), "y": [r["T"] for r in rec], "z": zz, "levels": 5}]}]})
    assert Path(imgs[0]).exists()


def test_join_calc_vs_experiment(tmp_path):
    calc = tmp_path / "calc.csv"
    pd.DataFrame({"T": np.arange(300.0, 1301, 100), "cp": np.linspace(25, 35, 11)}).to_csv(calc, index=False)
    exp = pd.DataFrame({"T": [500.0, 800.0, 1000.0, 1234.0], "cp": [27.3, 29.8, 32.2, 34.0]})
    j = T(exp, [{"step": "join", "with": str(calc), "on": "T", "suffixes": ["_exp", "_calc"]},
                {"step": "derive", "new": "resid", "expr": "cp_exp - cp_calc"}])
    assert list(j["T"]) == [500, 800, 1000] and np.allclose(j["resid"], [0.3, -0.2, 0.2])
    n = T(exp, [{"step": "join", "with": {"data": str(calc), "where": {"T": {">=": 400}}}, "on": "T",
                 "how": "nearest", "tolerance": 50, "suffixes": ["_exp", "_calc"]}])
    assert len(n) == 4 and n.loc[n["T"] == 1234, "cp_calc"].item() == pytest.approx(34.0)
    lo = T(exp.rename(columns={"T": "temp"}), [{"step": "join", "with": str(calc), "left_on": "temp",
                                                 "right_on": "T", "how": "left"}])
    assert len(lo) == 4 and "T" not in lo and lo["cp_2"].isna().sum() == 1


def test_concat_with_labels(tmp_path):
    for i in range(3):
        pd.DataFrame({"T": [300 + i], "cp": [25 + i]}).to_csv(tmp_path / f"run{i}.csv", index=False)
    df = fd.load_source(str(tmp_path / "run0.csv"), resolve=lambda p: p,
                        transform=[{"step": "concat", "with": str(tmp_path / "run[12].csv"), "label_col": "src"}])
    assert list(df["src"]) == ["run0", "run1", "run2"] and list(df["cp"]) == [25, 26, 27]


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
def test_interpolate_onto_grid():
    x = np.linspace(0, 10, 21)
    df = pd.concat([pd.DataFrame({"g": "a", "x": x, "y": 2 * x + 1, "z": x ** 3}),
                    pd.DataFrame({"g": "b", "x": x[::-1], "y": -x[::-1], "z": x})], ignore_index=True)
    out = T(df, [{"step": "interpolate", "x": "x", "y": ["y", "z"], "by": "g",
                  "onto": {"start": 0.25, "stop": 12, "step": 0.5}, "method": "cubic"}])
    a = out[out["g"] == "a"]
    inside = a["x"] <= 10
    assert np.allclose(a.loc[inside, "y"], 2 * a.loc[inside, "x"] + 1)
    assert np.allclose(a.loc[inside, "z"], a.loc[inside, "x"] ** 3, rtol=1e-3, atol=1e-3)  # cubic spline
    assert a.loc[~inside, "y"].isna().all()                                                # no extrapolation
    b = T(df[df["g"] == "b"], [{"step": "interpolate", "x": "x", "y": "y", "onto": [2.5, 7.25]}])
    assert np.allclose(b["y"], [-2.5, -7.25])
    p = T(df[df["g"] == "a"], [{"step": "interpolate", "x": "x", "y": "y", "onto": {"num": 5}, "method": "pchip"}])
    assert np.allclose(p["x"], np.linspace(0, 10, 5)) and np.allclose(p["y"], 2 * p["x"] + 1)


def test_smooth_reduces_noise():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 2 * np.pi, 400)
    truth = np.sin(x)
    df = pd.DataFrame({"x": x, "y": truth + rng.normal(0, 0.1, x.size)})
    err0 = np.sqrt(np.mean((df["y"] - truth) ** 2))
    for method, w in (("rolling", 15), ("savgol", 31), ("ewm", 5)):
        s = T(df, [{"step": "smooth", "cols": "y", "method": method, "window": w, "order": 3, "suffix": "_s"}])
        err = np.sqrt(np.mean((s["y_s"] - truth) ** 2))
        assert err < 0.6 * err0, (method, err, err0)
        assert np.allclose(s["y"], df["y"])                                             # original kept


def test_derivative_and_integrate_analytic():
    x = np.linspace(0, np.pi, 201)
    df = pd.DataFrame({"x": x[::-1], "y": np.sin(x[::-1])})                        # unsorted input
    d = T(df, [{"step": "derivative", "x": "x", "y": "y", "new": "dy"}])
    assert np.allclose(d["dy"], np.cos(d["x"]), atol=1e-4)
    i = T(df, [{"step": "integrate", "x": "x", "y": "y", "new": "Y"}])
    assert np.allclose(i["Y"], 1 - np.cos(i["x"]), atol=1e-4)
    tot = T(df, [{"step": "integrate", "x": "x", "y": "y", "cumulative": False}])
    assert tot["int_y"].item() == pytest.approx(2.0, abs=1e-4)
    # Cp -> H(T) - H(298) with a group key; exact for a linear Cp
    cp = pd.DataFrame({"ph": ["A"] * 5 + ["B"] * 5, "T": list(np.linspace(298, 698, 5)) * 2,
                       "cp": list(20 + 0.01 * np.linspace(298, 698, 5)) + [30.0] * 5})
    h = T(cp, [{"step": "integrate", "x": "T", "y": "cp", "new": "dH", "by": "ph"}])
    ha = h[h["ph"] == "A"]
    assert np.allclose(ha["dH"], 20 * (ha["T"] - 298) + 0.005 * (ha["T"] ** 2 - 298 ** 2))
    assert np.allclose(h[h["ph"] == "B"]["dH"], 30 * (h[h["ph"] == "B"]["T"] - 298))


def test_normalize_and_bin():
    df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "y": [1.0, 2, 4, -2, 0, 1]})
    n = T(df, [{"step": "normalize", "cols": "y", "method": "max", "by": "g", "suffix": "_n"},
               {"step": "normalize", "cols": "y", "method": "minmax", "new": "mm"},
               {"step": "normalize", "cols": "y", "method": "first", "by": "g", "new": "f"},
               {"step": "normalize", "cols": "y", "method": "zscore", "new": "z"},
               {"step": "normalize", "cols": "y", "method": "sum", "by": "g", "new": "s"}])
    assert np.allclose(n["y_n"], [0.25, 0.5, 1, -1, 0, 0.5]) and np.allclose(n["mm"], [0.5, 2 / 3, 1, 0, 1 / 3, 0.5])
    assert np.allclose(n["f"], [1, 2, 4, 1, 0, -0.5]) and abs(n["z"].mean()) < 1e-12
    assert np.allclose(n.groupby("g")["s"].sum(), 1)
    x = np.arange(0.0, 100.0)
    b = T(pd.DataFrame({"x": x, "y": 2 * x}), [{"step": "bin", "col": "x", "bins": [0, 50, 99],
                                                "agg": {"y": ["mean", "std"]}}])
    assert list(b.columns) == ["x", "y_mean", "y_std", "n"]
    assert np.allclose(b["x"], [25, 74.5]) and list(b["n"]) == [51, 49]
    assert np.allclose(b["y_mean"], [50, 2 * np.mean(np.arange(51, 100))])
    b = T(pd.DataFrame({"x": x, "y": 2 * x}), [{"step": "bin", "col": "x", "bins": 4, "label": "interval"}])
    assert b["x"].iloc[0] == "[0, 24.75]" and "y_mean" in b and b["n"].sum() == 100


def test_convert_units_and_composition():
    df = pd.DataFrame({"T": [273.15, 1273.15], "H": [1000.0, -2000.0], "P": [1e5, 101325.0]})
    c = T(df, [{"step": "convert", "col": "T", "from": "K", "to": "°C", "new": "T_C"},
               {"step": "convert", "col": "T_C", "from": "C", "to": "F", "new": "T_F"},
               {"step": "convert", "col": "H", "from": "J/mol", "to": "kJ/mol"},
               {"step": "convert", "col": "P", "from": "Pa", "to": "atm", "new": "P_atm"}])
    assert np.allclose(c["T_C"], [0, 1000]) and np.allclose(c["T_F"], [32, 1832])
    assert np.allclose(c["H"], [1, -2]) and c["P_atm"].iloc[1] == pytest.approx(1.0)
    assert T(pd.DataFrame({"f": [0.25]}), [{"step": "convert", "col": "f", "from": "fraction",
                                           "to": "percent"}])["f"].item() == 25
    with pytest.raises(ValueError, match="cannot convert K to bar"):
        T(df, [{"step": "convert", "col": "T", "from": "K", "to": "bar"}])
    # Fe-18Cr-8Ni (wt%) -> at%
    alloy = pd.DataFrame({"wt% Cr": [18.0], "wt% Ni": [8.0]})
    at = T(alloy, [{"step": "convert", "cols": ["wt% Cr", "wt% Ni"], "from": "wt%", "to": "at%",
                    "balance": "FE", "new": ["Cr_at", "Ni_at"], "balance_col": "Fe_at"}])
    assert at["Cr_at"].item() == pytest.approx(19.15, abs=0.01)
    assert at[["Cr_at", "Ni_at", "Fe_at"]].sum(axis=1).item() == pytest.approx(100)
    back = T(at, [{"step": "convert", "cols": ["Cr_at", "Ni_at"], "from": "at%", "to": "wt%", "balance": "FE",
                   "elements": ["CR", "NI"]}])
    assert np.allclose(back[["Cr_at", "Ni_at"]].to_numpy(), [[18, 8]])
    # binary wt fraction -> mole fraction, matches expdata.to_mole_fraction
    from calphad_tools.expdata import to_mole_fraction
    b = T(pd.DataFrame({"w": [0.3]}), [{"step": "convert", "col": "w", "from": "wt_fraction",
                                        "to": "mole_fraction", "element": "CR", "other": "NI"}])
    assert b["w"].item() == pytest.approx(to_mole_fraction(0.3, "wt", "CR", "NI"))
    with pytest.raises(ValueError, match="balance element"):
        T(alloy, [{"step": "convert", "cols": ["wt% Cr"], "from": "wt%", "to": "at%"}])


def test_step_errors_are_actionable(cp_frame):
    with pytest.raises(ValueError) as e:
        T(cp_frame, [{"step": "derive", "new": "a", "expr": "cp * 2"},
                     {"step": "groupby", "by": "phse", "agg": {"cp": "mean"}}])
    msg = str(e.value)
    assert "transform step 2 (groupby) failed" in msg and "syntax: groupby {by:" in msg
    assert "'phase'" in msg and "columns before this step" in msg
    with pytest.raises(ValueError, match="unknown step 'frobnicate'"):
        T(cp_frame, [{"step": "frobnicate"}])
    with pytest.raises(ValueError, match="unknown key\\(s\\) \\['colz'\\]"):
        T(cp_frame, [{"step": "select", "colz": ["cp"]}])
    with pytest.raises(ValueError, match="unknown aggregate 'avg'"):
        T(cp_frame, [{"step": "groupby", "by": "phase", "agg": {"cp": "avg"}}])


# ---------------------------------------------------------------------------
# through the plotting tools
# ---------------------------------------------------------------------------
def _replicates(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"T [K]": t, "cp": 20 + 0.01 * t + rng.normal(0, 0.5), "run": r, "phase": "FCC" if t < 800 else "BCC"}
            for t in (400, 600, 800, 1000) for r in range(4)]
    df = pd.DataFrame(rows)
    p = tmp_path / "replicates.xlsx"
    df.to_excel(p, sheet_name="dsc", index=False)
    return p, df


def test_groupby_errorbar_make_plot_and_pickle(tb, tmp_path):
    from calphad_tools.figures import load_figure, main_axes
    p, raw = _replicates(tmp_path)
    ops = [{"op": "errorbar", "data": str(p), "sheet": "dsc", "where": {"T": {">=": 600}},
            "transform": [{"step": "groupby", "by": "T", "agg": {"cp": ["mean", "std"]}},
                          {"step": "convert", "col": "T", "from": "K", "to": "C", "new": "T_C"}],
            "x_col": "T_C", "y_col": "cp_mean", "yerr_col": "cp_std", "label": "DSC mean±std"},
           {"op": "line", "data": str(p), "sheet": "dsc",
            "transform": [{"step": "derive", "new": "fit", "expr": "20 + 0.01 * `T [K]`"},
                          {"step": "dedupe", "cols": ["T"]}], "x_col": "T", "y_col": "fit", "label": "fit"}]
    _, imgs = call(tb, "make_plot", {"output": "cp_err.png", "panels": [{"operations": ops}]})
    fig = load_figure(imgs[0])
    ax = main_axes(fig)[0]
    pickle.dumps(fig)
    g = raw[raw["T [K]"] >= 600].groupby("T [K]")["cp"]
    cont = [c for c in ax.containers if c.get_label() == "DSC mean±std"][0]
    data_line = cont.lines[0]
    assert np.allclose(data_line.get_xdata(), [326.85, 526.85, 726.85])
    assert np.allclose(data_line.get_ydata(), g.mean().to_numpy())
    segs = cont.lines[2][0].get_segments()
    assert np.allclose([s[1][1] - s[0][1] for s in segs], 2 * g.std().to_numpy())
    fit = [ln for ln in ax.get_lines() if ln.get_label() == "fit"][0]
    assert np.allclose(fit.get_ydata(), 20 + 0.01 * np.array([400, 600, 800, 1000]))


def test_edit_figure_with_transform_and_errors(tb, tmp_path):
    from calphad_tools.figures import load_figure, main_axes
    csv = tmp_path / "exp.csv"
    pd.DataFrame({"T_C": [500, 700, 900], "H_J": [1000.0, 2000, 3000]}).to_csv(csv, index=False)
    _, imgs = call(tb, "make_plot", {"output": "base.png", "panels": [{"operations": [
        {"op": "line", "x": [700, 1200], "y": [0, 4], "label": "calc"}]}]})
    _, out = call(tb, "edit_figure", {"image": imgs[0], "operations": [
        {"op": "scatter", "data": str(csv), "x_col": "T_C", "y_col": "H_J", "label": "exp",
         "transform": [{"step": "convert", "col": "T_C", "from": "C", "to": "K"},
                       {"step": "convert", "col": "H_J", "from": "J", "to": "kJ"}]}]})
    ax = main_axes(load_figure(out[0]))[0]
    pts = [c for c in ax.collections if c.get_label() == "exp"][0].get_offsets()
    assert np.allclose(pts, [[773.15, 1], [973.15, 2], [1173.15, 3]])
    text, _, err = tb.call("edit_figure", {"image": imgs[0], "operations": [
        {"op": "scatter", "data": str(csv), "x_col": "T_C", "y_col": "H",
         "transform": [{"step": "derive", "new": "bad", "expr": "T_C.__class__"}]}]})
    assert err and "transform step 1 (derive) failed" in text and "attribute access" in text


def test_plot_ops_help_topics(tb):
    text, _ = call(tb, "plot_ops_help", {"ops": ["transform", "data"]})
    assert "interpolate {x, y" in text and "between" in text and "convert {" in text


# ---------------------------------------------------------------------------
# agent tools
# ---------------------------------------------------------------------------
def test_inspect_data(tb, tmp_path):
    p, _ = _replicates(tmp_path)
    text, _ = call(tb, "inspect_data", {"path": str(p), "sheet": "dsc"})
    assert "shape: 16 rows x 4 columns" in text and "sheets: ['dsc']" in text
    assert "T [K]: int64" in text and "phase (2 unique): FCC (8), BCC (8)" in text and "numeric stats" in text
    text, _ = call(tb, "inspect_data", {"path": str(p), "rows": 2, "transform": [
        {"step": "groupby", "by": "phase", "agg": {"cp": "mean"}}]})
    assert "shape: 2 rows x 3 columns" in text and "after where/transform" in text
    units = tmp_path / "units.csv"                             # units row under the header: hinted
    units.write_text("T,cp\nK,J/mol/K\n" + "".join(f"{300 + i},{25 + i}\n" for i in range(10)))
    text, _ = call(tb, "inspect_data", {"path": str(units)})
    assert "units row?" in text
    text, _, err = tb.call("inspect_data", {"path": str(tmp_path / "missing.csv")})
    assert err and "not found" in text


def test_transform_data_outputs(tb, tmp_path):
    p, raw = _replicates(tmp_path)
    tf = [{"step": "groupby", "by": ["phase", "T"], "agg": {"cp": ["mean", "sem"]}}]
    text, _ = call(tb, "transform_data", {"source": str(p), "transform": tf, "output": "means.xlsx"})
    assert "wrote" in text and "cp_sem" in text
    back = pd.read_excel(tmp_path / "out" / "means.xlsx")
    assert list(back["phase"]) == ["BCC", "BCC", "FCC", "FCC"] and list(back["T [K]"]) == [800, 1000, 400, 600]
    assert np.allclose(back["cp_mean"], raw.groupby("T [K]")["cp"].mean().loc[[800, 1000, 400, 600]])
    call(tb, "transform_data", {"source": [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "where": {"a": {">": 1}},
                                "transform": [{"step": "derive", "new": "c", "expr": "a + b"}], "output": "t.csv"})
    assert pd.read_csv(tmp_path / "out" / "t.csv").to_dict("records") == [{"a": 3, "b": 4, "c": 7}]
    call(tb, "transform_data", {"source": str(p), "output": "plain"})
    assert (tmp_path / "out" / "plain.csv").exists()
