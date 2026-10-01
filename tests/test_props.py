"""Tests for calphad_tools.props (equilibrium / single-phase property tools).

Run:  .venv/bin/python -m pytest -q tests/test_props.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calphad_tools.agent.tools import UI, ToolBox

ROOT = Path(__file__).resolve().parents[1]
CRNI = "CrNi-06Hal.tdb"
CHOI = "CoCrFeNiV_Choi2019.tdb"


class _UI(UI):
    def ask(self, question, options=None):
        return "ok"

    def confirm(self, title, body):
        return False


@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    out = tmp_path_factory.mktemp("props") / "out"
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out), ui=_UI(), python_tool="off")


def call(tb, name, args):
    text, images, is_error = tb.call(name, args)
    assert not is_error, text
    for im in images:
        assert Path(im).exists() and Path(im).stat().st_size > 5000, im
    return text, images


def _path(tb, text, kind):
    for line in text.splitlines():
        if line.startswith(kind + ":"):
            return Path(tb.p(line.split(":", 1)[1].strip().split(" ")[0]))
    raise AssertionError(f"no {kind} line in:\n{text}")


def test_tools_registered(tb):
    for n in ("equilibrium_property_plot", "phase_property_plot", "export_equilibrium_data", "make_plot"):
        assert n in tb.tools


def test_phase_fractions_fe18cr8ni(tb):
    text, imgs = call(tb, "equilibrium_property_plot", {
        "tdb": CHOI, "elements": ["FE", "CR", "NI"], "temperature_unit": "C",
        "conditions": {"T": [500, 1500, 25], "X(CR)": 0.18, "X(NI)": 0.08}, "quantities": ["NP(*)"]})
    assert len(imgs) == 1
    df = pd.read_csv(_path(tb, text, "CSV"))
    assert "T [°C]" in df.columns and df["T [°C]"].iloc[0] == pytest.approx(500)
    npc = [c for c in df.columns if c.startswith("NP(")]
    assert {"NP(FCC_A1)", "NP(BCC_A2)", "NP(LIQUID)"} <= set(npc)
    assert np.allclose(df[npc].sum(axis=1), 1.0, atol=1e-6)
    # austenite at 1000 °C, liquid at 1500 °C
    assert df.loc[df["T [°C]"] == 1000, "NP(FCC_A1)"].iloc[0] > 0.99
    assert df.loc[df["T [°C]"] == 1500, "NP(LIQUID)"].iloc[0] > 0.99
    assert "Stable phases along T" in text and "FCC_A1" in text


def test_activity_vs_composition_crni(tb):
    text, _ = call(tb, "equilibrium_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": 1273, "X(CR)": [0.02, 0.98, 0.04]},
        "quantities": ["MU(CR)", "ACR(CR,BCC_A2)", "ACR(NI)"], "exclude_phases": ["B2_BCC", "L12_FCC"]})
    df = pd.read_csv(_path(tb, text, "CSV"))
    a = df["ACR(CR,BCC_A2)"].values
    assert np.all((a > 0) & (a <= 1.0 + 1e-6))
    assert np.all(np.diff(a) >= -1e-6)                  # activity rises with X(CR)
    assert np.all(np.diff(df["MU(CR) [J/mol]"].values) >= -1e-3)
    ani = df["ACR(NI,FCC_A1)"].values                   # default reference = stable pure-Ni phase (FCC)
    assert np.all((ani > 0) & (ani <= 1 + 1e-6)) and ani[0] > ani[-1]
    # two-phase field: constant activity
    two = df["stable_phases"] == "BCC_A2+FCC_A1"
    assert two.sum() >= 5 and np.ptp(a[two.values]) < 1e-3


def test_enthalpy_heat_capacity_vs_T(tb):
    text, imgs = call(tb, "equilibrium_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": [400, 2000, 40], "X(CR)": 0.3},
        "quantities": ["HM[kJ/mol]", "CPM", "HM.T"], "output_name": "hm_cp"})
    df = pd.read_csv(_path(tb, text, "CSV"))
    assert (df["CPM [J/mol/K]"] > 15).all() and (df["CPM [J/mol/K]"] < 80).all()
    assert np.all(np.diff(df["HM [kJ/mol]"].values) > 0)
    assert (df[[c for c in df.columns if c.startswith("HM.T")][0]] > 0).all()
    assert Path(imgs[0]).name == "hm_cp.png"


def test_driving_force_sigma_fecr(tb):
    base = {"tdb": CHOI, "elements": ["CR", "FE"], "conditions": {"T": [700, 1300, 50], "X(CR)": 0.45}}
    text, _ = call(tb, "equilibrium_property_plot", {
        **base, "quantities": ["DF(SIGMA)", "NP(SIGMA)"], "phases": ["BCC_A2", "SIGMA", "LIQUID"]})
    df = pd.read_csv(_path(tb, text, "CSV"))
    df_s, np_s = df["DF(SIGMA) [J/mol]"].values, df["NP(SIGMA)"].values
    assert np.all(df_s < 1e-3)                            # sigma allowed: DF <= 0, = 0 where stable
    assert np.all(np.abs(df_s[np_s > 1e-6]) < 1e-3) and (np_s > 0.1).any()
    # sigma suspended: positive driving force for precipitation in the sigma range
    text2, _ = call(tb, "equilibrium_property_plot", {
        **base, "quantities": ["DF(SIGMA)", "DORMANT(SIGMA):X(CR)"], "phases": ["BCC_A2", "LIQUID"]})
    df2 = pd.read_csv(_path(tb, text2, "CSV"))
    d2 = df2["DF(SIGMA) [J/mol]"].values
    assert d2.max() > 50 and d2[-1] < 0
    assert np.all(np.sign(d2[np_s > 1e-6]) > 0)
    xc = df2["DORMANT(SIGMA):X(SIGMA,CR)"].values
    assert np.all((xc > 0.3) & (xc < 0.7))
    assert "DF>=0" in text2


def test_2d_map_with_netcdf(tb):
    import xarray as xr
    text, imgs = call(tb, "equilibrium_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": [1000, 2200, 100], "X(CR)": [0, 1, 0.1]},
        "quantities": ["NP(FCC_A1)", "NP(LIQUID)", "HM"], "write_netcdf": True})
    assert len(imgs) == 1
    ds = xr.open_dataset(_path(tb, text, "netCDF"), engine="scipy")
    assert set(ds.dims) == {"T", "X_CR"}
    f = ds["NP_FCC_A1"].values
    assert np.nanmin(f) >= -1e-9 and np.nanmax(f) <= 1 + 1e-9
    assert ds["NP_LIQUID"].sel(T=2200).values.min() > 0.99
    assert "Stable phase regions" in text and "LIQUID" in text
    df = pd.read_csv(_path(tb, text, "CSV"))
    assert len(df) == 13 * 11


def test_single_point_isolated_t0(tb):
    text, imgs = call(tb, "equilibrium_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": 1000, "X(CR)": 0.35},
        "quantities": ["GM", "ISOLATED(LIQUID):GM", "T0(FCC_A1,BCC_A2)", "X(FCC_A1,CR)"]})
    assert imgs == [] and "Single point" in text
    df = pd.read_csv(_path(tb, text, "CSV"))
    gm = df["GM [J/mol]"].iloc[0]
    gliq = df["ISOLATED(LIQUID):GM(LIQUID) [J/mol]"].iloc[0]
    assert gliq > gm                                       # metastable liquid is above equilibrium
    assert 1200 < df["T0(FCC_A1,BCC_A2) [K]"].iloc[0] < 2200


def test_gm_curves_reference_and_common_tangent(tb):
    text, imgs = call(tb, "phase_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "prop": "GM", "temperature": 1273,
        "reference": {"CR": "BCC_A2", "NI": "FCC_A1"}, "common_tangent": True,
        "phases": ["BCC_A2", "FCC_A1", "LIQUID"]})
    df = pd.read_csv(_path(tb, text, "CSV"))
    bcc = df[df.phase == "BCC_A2"].sort_values("X(NI)")
    fcc = df[df.phase == "FCC_A1"].sort_values("X(NI)")
    assert abs(bcc["GM [J/mol]"].iloc[0]) < 300 and abs(fcc["GM [J/mol]"].iloc[-1]) < 500
    assert "two-phase BCC_A2 + FCC_A1" in text
    import re
    lo, hi = map(float, re.search(r"two-phase BCC_A2 \+ FCC_A1: X\(NI\) ([\d.]+) – ([\d.]+)", text).groups())
    assert 0.0 < lo < 0.12 and 0.45 < hi < 0.6       # equilibrium: X(Cr) ~0.46-0.94 at 1273 K


def test_gm_mix_curves(tb):
    text, _ = call(tb, "phase_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "prop": "GM_MIX", "temperature": 1000, "temperature_unit": "C",
        "show_all_points": True})
    df = pd.read_csv(_path(tb, text, "CSV"))
    liq = df[df.phase == "LIQUID"]["GM_MIX [J/mol]"]
    assert (liq < 1e-6).all() and liq.min() < -5000
    assert "partitioned" in text                         # ordered phases give NaN _MIX -> explained


def test_phase_property_vs_T(tb):
    text, _ = call(tb, "phase_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "prop": "GM", "temperature": [800, 2000, 100],
        "composition": {"CR": 0.3}, "phases": ["FCC_A1", "LIQUID", "BCC_A2"],
        "reference": {"CR": "BCC_A2", "NI": "FCC_A1"}})
    df = pd.read_csv(_path(tb, text, "CSV"))
    d = df["GM(LIQUID)"] - df["GM(FCC_A1)"]
    assert d.iloc[0] > 0 and d.iloc[-1] < 0                # liquid becomes more stable than fcc on heating
    assert "Lowest-G single phase" in text


def test_export_equilibrium_data(tb):
    import xarray as xr
    text, imgs = call(tb, "export_equilibrium_data", {
        "tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": [1000, 2200, 100], "X(CR)": [0, 1, 0.05]},
        "extra_outputs": ["HM", "CPM"], "eqplot": True, "output_name": "crni_grid"})
    assert len(imgs) == 1
    ds = xr.open_dataset(_path(tb, text, "nc"), engine="scipy")
    assert {"NP", "GM", "MU", "X", "Y", "Phase", "HM", "CPM"} <= set(ds.data_vars)
    df = pd.read_csv(_path(tb, text, "csv"))
    key = ["T [K]", "X(CR)"]
    sums = df.groupby(key)["NP"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-6)
    assert len(sums) == 13 * 21
    assert {"phase", "X(phase,CR)", "Y0", "MU(CR) [J/mol]", "HM"} <= set(df.columns)


def test_plot_csv_overlay(tb, tmp_path):
    text, _ = call(tb, "equilibrium_property_plot", {
        "tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": 1273, "X(CR)": [0.05, 0.95, 0.1]},
        "quantities": ["ACR(CR,BCC_A2)"], "plot": False, "output_name": "acr_for_overlay"})
    exp = tmp_path / "exp.csv"
    pd.DataFrame({"x": [0.2, 0.5], "a": [0.25, 0.9]}).to_csv(exp, index=False)
    t2, imgs = call(tb, "make_plot", {"panels": [{"operations": [
        {"op": "line", "csv": str(_path(tb, text, "CSV")), "x_col": "X(CR)", "y_col": "ACR(CR", "label": "calc"},
        {"op": "scatter", "csv": str(exp), "x_col": "x", "y_col": "a", "color": "red", "label": "exp"},
        {"op": "set", "yscale": "log", "ylabel": "a(Cr)"}]}], "output": "overlay.png"})
    assert Path(imgs[0]).name == "overlay.png" and "saved" in t2


@pytest.mark.parametrize("args, msg", [
    ({"quantities": ["NP(FOO)"]}, "unknown phase"),
    ({"quantities": ["WHATEVER("]}, "Quantity syntax"),
    ({"quantities": ["NP(*)"], "phases": ["SIGMA"]}, "available"),
    ({"quantities": ["NP(*)"], "conditions": {"T": [300, 3000, 0.1], "X(CR)": [0, 1, 0.01]}}, "limit"),
    ({"quantities": ["NP(*)"], "conditions": {"X(CR)": 0.3}}, "temperature"),
])
def test_helpful_errors(tb, args, msg):
    a = {"tdb": CRNI, "elements": ["CR", "NI"], "conditions": {"T": 1000, "X(CR)": 0.3}, **args}
    text, _, is_error = tb.call("equilibrium_property_plot", a)
    assert is_error and msg in text, text
