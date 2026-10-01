"""Tests for calphad_tools.mapping_tools (agent tools for mapping data, Scheil, TDB/model export).

Run:  .venv/bin/python -m pytest -q tests/test_mapping_tools.py
"""
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"


@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    from calphad_tools.agent.tools import ToolBox, UI

    class MyUI(UI):
        def ask(self, question, options=None):
            return ""

        def confirm(self, title, body):
            return True

    out = tmp_path_factory.mktemp("maptools")
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out / "out"), ui=MyUI(), python_tool="off")


def call(tb, name, args):
    text, images, is_error = tb.call(name, args)
    assert not is_error, text
    return text, images


def files_from(text):
    """Parse the 'Files:' block of a tool result into {key: Path}."""
    out = {}
    for line in text.split("Files:", 1)[1].splitlines():
        m = re.match(r"\s+(\w+): (.+)$", line)
        if m:
            p = Path(m.group(2).split(" (")[0].strip())
            out[m.group(1)] = p if p.is_absolute() else ROOT / "examples" / p
    return out


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_tools_registered(tb):
    for name in ("map_phase_diagram", "step_map", "scheil_solidification", "write_database", "export_phase_model"):
        assert name in tb.tools, name


def test_binary_crni_data_export(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                  "tmin": 1000, "tmax": 2300, "composition_step": 0.04,
                                                  "temperature_step": 20, "label_nodes": True,
                                                  "output_prefix": "crni_bin"})
    assert "eutectic" in text and "LIQUID -> BCC_A2 + FCC_A1" in text
    f = files_from(text)
    assert Path(images[0]).stat().st_size > 20000
    data = json.loads(f["json"].read_text())
    assert data["kind"] == "binary"
    eut = [i for i in data["invariants"] if i["type"] == "eutectic"]
    assert len(eut) == 1
    e = eut[0]
    assert abs(e["T_K"] - 1618) < 3
    assert set(e["phases"]) == {"LIQUID", "BCC_A2", "FCC_A1"}
    x = {p: c["NI"] for p, c in e["compositions"].items()}
    assert x["BCC_A2"] < x["LIQUID"] < x["FCC_A1"]
    assert abs(x["LIQUID"] - 0.46) < 0.02
    # invariant CSV: one row per phase
    inv_rows = [r for r in read_csv(f["invariants_csv"]) if r["type"] == "eutectic"]
    assert len(inv_rows) == 3 and abs(float(inv_rows[0]["T_K"]) - 1618) < 3
    # boundaries: long format with both phase compositions per tie-line, BCC+LIQUID up to Cr melting point
    rows = read_csv(f["boundaries_csv"])
    assert {"line", "region", "phase", "T_K", "T_C", "X(CR)", "X(NI)", "MU(NI)"} <= set(rows[0])
    liq_bcc = [r for r in rows if r["region"] == "BCC_A2 + LIQUID"]
    assert liq_bcc and max(float(r["T_K"]) for r in liq_bcc) == pytest.approx(2180, abs=15)
    for r in rows[:50]:
        assert abs(float(r["X(CR)"]) + float(r["X(NI)"]) - 1) < 1e-6
    ties = read_csv(f["tielines_csv"])
    assert {"phase_1", "phase_2", "X1(NI)", "X2(NI)"} <= set(ties[0])
    assert len(ties) * 2 == len(rows)
    assert any(r["region"] == "FCC_A1 + LIQUID" for r in data["regions"])


def test_binary_options_weight_celsius_mu(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                  "tmin": 1200, "tmax": 1900, "temperature_unit": "C",
                                                  "composition_unit": "weight_percent", "tielines": 0,
                                                  "composition_step": 0.05, "temperature_step": 20,
                                                  "output_prefix": "crni_wt"})
    f = files_from(text)
    rows = read_csv(f["boundaries_csv"])
    assert "W(NI)" in rows[0]
    assert Path(images[0]).exists()
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                  "tmin": 1500, "tmax": 2200, "potential_axis": "NI",
                                                  "composition_step": 0.05, "temperature_step": 20,
                                                  "output_prefix": "crni_mu"})
    assert Path(images[0]).exists() and "eutectic" in text


def test_ternary_isotherm_cocrni(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                  "temperature": 1273, "composition_step": 0.05,
                                                  "composition_unit": "mole_percent", "output_prefix": "cocrni"})
    f = files_from(text)
    data = json.loads(f["json"].read_text())
    assert data["kind"] == "ternary"
    assert {"FCC_A1", "SIGMA"} <= set(data["phases_found"])
    assert data["invariants"], "expected at least one three-phase tie-triangle"
    tri = data["invariants"][0]
    assert len(tri["phases"]) == 3
    for comp in tri["compositions"].values():
        assert abs(sum(comp.values()) - 1) < 1e-6
    rows = read_csv(f["boundaries_csv"])
    assert {"X(CO)", "X(CR)", "X(NI)"} <= set(rows[0])
    assert Path(images[0]).stat().st_size > 15000


def test_isopleth_fecrni(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                                  "vary": "CR", "fixed": {"NI": 0.1}, "tmin": 1400, "tmax": 1900,
                                                  "x_range": [0, 0.4], "composition_step": 0.04,
                                                  "temperature_step": 25, "output_prefix": "fecrni_iso"})
    f = files_from(text)
    data = json.loads(f["json"].read_text())
    assert data["kind"] == "isopleth" and "LIQUID" in data["phases_found"]
    rows = read_csv(f["boundaries_csv"])
    assert {"zero_phase", "T_K", "X(CR)", "phase_1", "X1(FE)"} <= set(rows[0])
    assert Path(images[0]).exists()


def test_step_map_T(tb):
    text, images = call(tb, "step_map", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "vary": "T",
                                         "start": 1500, "stop": 1900, "step": 5, "fixed": {"X(NI)": 0.4},
                                         "properties": ["GM", "HM", "HM.T"], "plot": ["NP", "X(*,NI)", "HM"],
                                         "output_prefix": "crni_step"})
    assert "LIQUID" in text and "Transitions" in text
    csvp = Path(re.search(r"csv: (\S+)", text).group(1))
    csvp = csvp if csvp.is_absolute() else ROOT / "examples" / csvp
    rows = read_csv(csvp)
    assert {"T_K", "T_C", "NP(LIQUID)", "NP(BCC_A2)", "X(LIQUID,NI)", "MU(CR)", "GM", "HM", "HM.T"} <= set(rows[0])
    T = np.array([float(r["T_K"]) for r in rows])
    fl = np.array([float(r["NP(LIQUID)"]) for r in rows])
    assert np.all(np.diff(T) >= 0)
    assert fl[0] == pytest.approx(0, abs=1e-6) and fl[-1] == pytest.approx(1, abs=1e-6)
    # the eutectic (1618 K) and liquidus transitions are reported
    ts = [float(m) for m in re.findall(r"at ([\d.]+) K", text)]
    assert any(abs(t - 1617.9) < 2 for t in ts)
    assert any(1650 < t < 1720 for t in ts)
    for r in rows:
        assert sum(float(r[k]) for k in r if k.startswith("NP(")) == pytest.approx(1, abs=1e-6)
    assert Path(images[0]).exists()


def test_scheil_crni(tb):
    pytest.importorskip("scheil")
    text, images = call(tb, "scheil_solidification", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                      "composition": {"NI": 0.3}, "step_temperature": 4,
                                                      "output_prefix": "crni_scheil"})
    f = files_from(text)
    assert "Scheil" in text and "Equilibrium" in text
    liq = float(re.search(r"liquidus ([\d.]+) K", text).group(1))
    sol = float(re.search(r"solidus ([\d.]+) K", text).group(1))
    assert 1790 < liq < 1830 and 1660 < sol < 1700
    end = float(re.search(r"ends at ([\d.]+) K", text).group(1))
    assert abs(end - 1618) < 6          # Scheil ends at the eutectic
    s = read_csv(f["scheil_csv"])
    assert {"Temperature (K)", "fraction_solid", "NP(LIQUID)", "NP(BCC_A2)"} <= set(s[0])
    fs = np.array([float(r["fraction_solid"]) for r in s])
    assert np.all(np.diff(fs) >= -1e-12) and fs[-1] == pytest.approx(1.0)
    e = read_csv(f["equilibrium_csv"])
    assert float(e[-1]["fraction_solid"]) == pytest.approx(1.0, abs=1e-6)
    assert Path(images[0]).stat().st_size > 20000


def test_scheil_fecrni(tb):
    pytest.importorskip("scheil")
    text, _ = call(tb, "scheil_solidification", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                                 "composition": {"CR": 0.2, "NI": 0.1}, "step_temperature": 4,
                                                 "include_equilibrium": False, "output_prefix": "fecrni_scheil"})
    assert "BCC_A2" in text and "Equilibrium" not in text


def test_write_database_subset_matches_full(tb):
    from pycalphad import Database, equilibrium, variables as v
    text, _ = call(tb, "write_database", {"tdb": "CoCrFeNiV_Choi2019.tdb", "output_tdb": "CrNi_subset.tdb",
                                          "elements": ["CR", "NI"]})
    assert "reloaded OK" in text
    path = Path(tb.outp("CrNi_subset.tdb", "x.tdb"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub = Database(str(path))
        full = Database(str(TDBS / "CoCrFeNiV_Choi2019.tdb"))
    assert {str(e) for e in sub.elements} - {"VA", "/-"} == {"CR", "NI"}
    assert len(sub._parameters) < len(full._parameters)
    worst = float(re.search(r"max \|dGM\| = ([\deE.+-]+)", text).group(1))
    assert worst < 1e-6
    from calphad_tools.diagrams import phases_for
    ph = phases_for(full, ["CR", "NI"])
    assert phases_for(sub, ["CR", "NI"]) == ph
    conds = {v.T: [900.0, 1500.0], v.P: 101325, v.N: 1, v.X("NI"): 0.35}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = equilibrium(full, ["CR", "NI", "VA"], ph, conds)
        b = equilibrium(sub, ["CR", "NI", "VA"], ph, conds)
    assert np.allclose(a.GM.values, b.GM.values, atol=1e-3)


def test_write_database_phase_selection(tb):
    text, _ = call(tb, "write_database", {"tdb": "CrNi-06Hal.tdb", "output_tdb": "crni_liq_fcc.tdb",
                                          "phases": ["LIQUID", "L12_FCC"], "groupby": "phase"})
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(tb.outp("crni_liq_fcc.tdb", "x.tdb"))
    # the disordered part of L12_FCC is kept automatically
    assert set(db.phases) == {"LIQUID", "L12_FCC", "A1_FCC"}
    assert db.phases["L12_FCC"].model_hints.get("disordered_phase") == "A1_FCC"
    # refusing to overwrite the source
    _, _, err = tb.call("write_database", {"tdb": "CrNi-06Hal.tdb", "output_tdb": str(TDBS / "CrNi-06Hal.tdb")})
    assert err


def test_export_phase_model_fcc_crni(tb):
    text, _ = call(tb, "export_phase_model", {"tdb": "CrNi-06Hal.tdb", "phase": "FCC_A1",
                                              "evaluate_at": {"T": 1000, "X(CR)": 0.3, "X(NI)": 0.7},
                                              "latex": True, "output_prefix": "fcc_crni"})
    assert "xsmix" in text and "L(FCC_A1,CR,NI:VA;0)" in text and "GHSERCR" in text
    f = files_from(text)
    full = f["txt"].read_text()
    assert "=== xsmix" in full and "=== FUNCTIONS" in full
    assert "\\begin{dmath}" in f["tex"].read_text()
    rows = read_csv(f["parameters_csv"])
    assert len(rows) == 11
    assert {r["parameter"] for r in rows} >= {"L(FCC_A1,CR,NI:VA;0)", "L(FCC_A1,CR,NI:VA;1)", "TC(FCC_A1,CR:VA;0)"}
    ev = json.loads(text.split("evaluation: ", 1)[1].splitlines()[0])
    # hand-computed Redlich-Kister excess at T=1000 K, x(Cr)=0.3
    L0, L1, L2 = 5320 - 10.5 * 1000, 32030 - 14.4 * 1000, 700.0
    d = 0.3 - 0.7
    assert ev["xsmix"] == pytest.approx(0.21 * (L0 + L1 * d + L2 * d * d), abs=1e-6)
    # total GM matches pycalphad calculate()
    from pycalphad import Database, calculate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(str(TDBS / "CrNi-06Hal.tdb"))
        r = calculate(db, ["CR", "NI", "VA"], "FCC_A1", T=1000, P=101325, N=1, points=np.array([[0.3, 0.7, 1.0]]))
    assert ev["GM"] == pytest.approx(float(r.GM.values.squeeze()), abs=1e-4)


def test_bad_input_messages(tb):
    text, _, err = tb.call("map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "XX"], "tmin": 1000,
                                                 "tmax": 2000})
    assert err and "available" in text
    text, _, err = tb.call("export_phase_model", {"tdb": "CrNi-06Hal.tdb", "phase": "NOPE"})
    assert err and "FCC_A1" in text


def test_phase_colors_consistent():
    from calphad_tools.mapping_tools import phase_color
    assert phase_color("FCC_A1") == phase_color("FCC_A1#2") == phase_color("fcc_a1")
    assert phase_color("SOMEPHASE") == phase_color("SOMEPHASE")
    assert phase_color("LIQUID", {"LIQUID": "#000000"}) == "#000000"
