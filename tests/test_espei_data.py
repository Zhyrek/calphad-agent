"""Tests for calphad_tools.espei_data: thermochemical ESPEI datasets + goodness-of-fit report.

Synthetic data are computed FROM examples/tdbs/CrNi-06Hal.tdb, so the fit report of that TDB must give ~zero
residuals, while a TDB with a modified liquid interaction parameter must rank the liquid datasets worst.
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TDB = ROOT / "examples" / "tdbs" / "CrNi-06Hal.tdb"
COMPS = ["CR", "NI", "VA"]


# ---------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def dbf():
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(TDB))


def make_tb(out):
    from calphad_tools.agent.tools import UI, ToolBox

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out), ui=U(), python_tool="off")


@pytest.fixture
def tb(tmp_path):
    return make_tb(tmp_path / "out")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def espei_load(paths):
    from espei.datasets import check_dataset, load_datasets
    for p in paths:
        check_dataset(json.loads(Path(p).read_text()))
    return load_datasets([str(p) for p in paths])


def hm_mix_fixed(dbf, phase, T, x_ni):
    from pycalphad import calculate
    pts = np.array([[1 - x, x] + ([1.0] if phase != "LIQUID" else []) for x in x_ni])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = calculate(dbf, COMPS, phase, T=T, P=101325, points=pts, output="HM_MIX")
    return r.HM_MIX.values.squeeze().tolist()


def hm_form_crni2(dbf, Ts):
    from pycalphad import Model, ReferenceState, calculate
    mod = Model(dbf, COMPS, "CRNI2")
    mod.shift_reference_state([ReferenceState("CR", "BCC_A2"), ReferenceState("NI", "FCC_A1")], dbf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = calculate(dbf, COMPS, "CRNI2", T=Ts, P=101325, output="HMR", model={"CRNI2": mod})
    return np.atleast_1d(r.HMR.values.squeeze()).tolist()


def eq_prop(dbf, phases, T, x_ni, output):
    from pycalphad import equilibrium, variables as v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(dbf, COMPS, phases, {v.T: T, v.P: 101325, v.X("NI"): list(x_ni), v.N: 1}, output=output)
    return np.atleast_1d(getattr(eq, output).values.squeeze()).tolist()


def activity(dbf, el, T, x_ni, ref_phase):
    from pycalphad import equilibrium, variables as v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(dbf, COMPS, list(dbf.phases), {v.T: T, v.P: 101325, v.X("NI"): list(x_ni), v.N: 1})
        ref = equilibrium(dbf, COMPS, [ref_phase], {v.T: T, v.P: 101325, v.X("NI"): 1.0 if el == "NI" else 0.0,
                                                    v.N: 1})
    mu = eq.MU.sel(component=el).values.squeeze()
    mu0 = float(ref.MU.sel(component=el).values.squeeze())
    return np.exp((mu - mu0) / (8.3145 * T)).tolist()


def tieline(dbf, T, x_cr, phases=("LIQUID", "FCC_A1", "BCC_A2", "CRNI2")):
    from pycalphad import equilibrium, variables as v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(dbf, COMPS, list(phases), {v.T: T, v.P: 101325, v.X("CR"): x_cr, v.N: 1})
    ph = [str(p) for p in eq.Phase.values.squeeze() if p]
    xs = eq.X.values.squeeze()[: len(ph), 0]
    assert len(ph) == 2, (T, x_cr, ph)
    return [[p, ["CR"], [round(float(x), 6)]] for p, x in zip(ph, xs)]


def perturbed_tdb(path):
    txt = TDB.read_text()
    old = "L(LIQUID,CR,NI;0),,               -14190+2.3*T;"
    assert old in txt
    path.write_text(txt.replace(old, "L(LIQUID,CR,NI;0),,               -4190+2.3*T;"))
    return path


# ---------------------------------------------------------------- dataset writers
def test_write_thermo_dataset_mixing_and_formation(tb, dbf, tmp_path):
    xs = [0.0, 0.2, 0.5, 0.8]
    vals = [0.0] + hm_mix_fixed(dbf, "FCC_A1", 1200, xs[1:])
    text, _ = call(tb, "write_thermo_dataset", {
        "components": ["CR", "NI"], "phase": "FCC_A1", "output": "HM_MIX", "values": vals, "T": 1200,
        "mixing_x": xs, "x_component": "NI", "tdb": "CrNi-06Hal.tdb", "reference": "synthetic fcc",
        "file": str(tmp_path / "fcc_hmix.json")})
    assert "dropped 1 point" in text
    ds = json.loads((tmp_path / "fcc_hmix.json").read_text())
    assert ds["components"] == ["CR", "NI", "VA"] and ds["solver"]["sublattice_site_ratios"] == [1.0, 1.0]
    assert ds["solver"]["sublattice_configurations"][0] == [["CR", "NI"], "VA"]
    assert ds["solver"]["sublattice_occupancies"][0] == [[0.8, 0.2], 1.0]
    assert np.asarray(ds["values"]).shape == (1, 1, 3)

    # unsorted interaction is sorted together with its occupancies; per-formula -> per-atom conversion
    from calphad_tools import espei_data as ed
    pm = ed.phase_info("FCC_A1", str(TDB), components=["CR", "NI"])
    d2, _ = ed.make_thermo_dataset(["CR", "NI"], "FCC_A1", "HM_MIX", [1000.0], T=1000,
                                   configurations=[[["NI", "CR"], "VA"]], occupancies=[[[0.7, 0.3], 1]], phase_model=pm)
    assert d2["solver"]["sublattice_configurations"] == [[["CR", "NI"], "VA"]]
    assert d2["solver"]["sublattice_occupancies"] == [[[0.3, 0.7], 1.0]]
    pm2 = ed.phase_info("CRNI2", str(TDB), components=["CR", "NI"])
    d3, w = ed.make_thermo_dataset(["CR", "NI"], "CRNI2", "HM_FORM", [-9.0, -9.3], T=[298.15, 400],
                                   configurations=[["CR", "NI"]], phase_model=pm2, value_unit="kJ/mol-formula")
    assert np.allclose(np.asarray(d3["values"]).ravel(), [-3000.0, -3100.0]) and any("per mole of atoms" in x for x in w)
    assert d3["components"] == ["CR", "NI"] and "sublattice_occupancies" not in d3["solver"]

    # sanity checks
    _, w = ed.make_thermo_dataset(["CR", "NI"], "CRNI2", "HM_FORM", [-3.1], configurations=[["CR", "NI"]],
                                  phase_model=pm2)
    assert any("kJ" in x for x in w)
    with pytest.raises(ed.ThermoDataError, match="not a constituent"):
        ed.make_thermo_dataset(["CR", "NI"], "CRNI2", "HM_FORM", [-3000], configurations=[["NI", "CR"]], phase_model=pm2)
    with pytest.raises(ed.ThermoDataError, match="shape"):
        ed.make_thermo_dataset(["CR", "NI"], "CRNI2", "HM_FORM", [-3000, 1, 2], T=[300, 400],
                               configurations=[["CR", "NI"]], phase_model=pm2)
    with pytest.raises(ed.ThermoDataError, match="write_equilibrium_dataset"):
        ed.make_thermo_dataset(["CR", "NI"], "CRNI2", "HMR", [-3000], configurations=[["CR", "NI"]], phase_model=pm2)
    text, _, err = tb.call("write_thermo_dataset", {"components": ["CR", "NI"], "phase": "FCC_A1", "output": "HM_MIX",
                                                    "values": [1.0], "configurations": [[["CR", "NI"], "VA"]],
                                                    "reference": "x", "file": "bad.json"})
    assert err and "occupancies" in text
    espei_load([tmp_path / "fcc_hmix.json"])


def test_write_equilibrium_dataset_split_and_refs(tb, tmp_path):
    from calphad_tools import espei_data as ed
    text, _ = call(tb, "write_equilibrium_dataset", {
        "components": ["CR", "NI"], "phases": ["LIQUID"], "output": "ACR(CR)", "tdb": "CrNi-06Hal.tdb",
        "conditions": {"T": [2200, 2300], "X(NI)": [0.2, 0.5]}, "values": [[[0.8, 0.5], [0.81, 0.52]]],
        "reference": "act", "file": str(tmp_path / "acr.json")})
    files = sorted(tmp_path.glob("acr_*.json"))
    assert len(files) == 2 and "SER" in text
    d = json.loads(files[1].read_text())
    assert d["output"] == "ACR_CR" and d["conditions"]["T"] == 2300.0 and d["conditions"]["X_NI"] == [0.2, 0.5]
    assert d["reference_state"] == {"phases": ["BCC_A2"], "conditions": {"P": 101325.0, "T": 2300.0, "X_NI": 0.0}}
    assert "VA" in d["components"]
    # multi-T x multi-X equilibrium data are split per T (ESPEI 0.9.1 evaluates X-major)
    dss, w = ed.make_equilibrium_datasets(["CR", "NI"], ["LIQUID"], "HMR", {"T": [2200, 2300], "X_NI": [0.2, 0.5, 0.8]},
                                          [[[-2000, -3000, -2000], [-2100, -3100, -2100]]],
                                          reference_states={"CR": "LIQUID", "NI": "liquid"})
    assert len(dss) == 2 and dss[0]["reference_states"]["NI"] == {"phase": "LIQUID"}
    # single varying condition: flat values
    dss, _ = ed.make_equilibrium_datasets(["CR", "NI"], ["FCC_A1"], "CPM", {"T_C": [900, 1000, 1100], "X(NI)": [0.8]},
                                          [30.1, 31.0, 32.2])
    assert len(dss) == 1 and dss[0]["conditions"]["T"] == [1173.15, 1273.15, 1373.15]
    assert np.asarray(dss[0]["values"]).shape == (1, 3, 1)
    dss, w = ed.make_equilibrium_datasets(["CR", "NI"], ["LIQUID"], "HMR", {"T": 2200, "X_NI": [0.5]}, [-3000])
    assert dss[0]["output"] == "HM" and any("SER-referenced" in x for x in w)
    dss, w = ed.make_equilibrium_datasets(["CR", "NI"], ["CRNI2"], "HM_FORM", {"T": 298.15, "X_NI": [2 / 3]}, [-3000],
                                          ser_phases=ed.ser_phases_from_tdb(str(TDB)))
    assert dss[0]["output"] == "HMR" and dss[0]["reference_states"]["CR"] == {"phase": "BCC_A2"}
    with pytest.raises(ed.ThermoDataError, match="composition condition"):
        ed.make_equilibrium_datasets(["CR", "NI"], ["LIQUID"], "HM", {"T": 2200}, [1.0])
    with pytest.raises(ed.ThermoDataError, match="> 0"):
        ed.make_equilibrium_datasets(["CR", "NI"], ["LIQUID"], "ACR_CR", {"T": 2200, "X_NI": [0.5]}, [0.0],
                                     reference_states={"CR": "LIQUID"})
    with pytest.raises(ed.ThermoDataError, match="not in the database"):
        ed.make_equilibrium_datasets(["CR", "NI"], ["LIQ"], "HM", {"T": 2200, "X_NI": [0.5]}, [1.0],
                                     all_phases=["LIQUID"])
    for p in files:
        espei_load([p])
    for d in dss:
        ed._check_espei(d)


# ---------------------------------------------------------------- synthetic dataset directory
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory, dbf):
    base = tmp_path_factory.mktemp("espei_data")
    tb = make_tb(base / "out")
    d = base / "datasets"
    xs = [0.2, 0.4, 0.6, 0.8]
    call(tb, "write_thermo_dataset", {"components": ["CR", "NI"], "phase": "LIQUID", "output": "HM_MIX", "T": 2200,
                                      "values": hm_mix_fixed(dbf, "LIQUID", 2200, xs), "mixing_x": xs,
                                      "x_component": "NI", "tdb": "CrNi-06Hal.tdb", "reference": "liq hmix",
                                      "file": str(d / "liq_hmix.json")})
    call(tb, "write_thermo_dataset", {"components": ["CR", "NI"], "phase": "FCC_A1", "output": "HM_MIX", "T": 1200,
                                      "values": hm_mix_fixed(dbf, "FCC_A1", 1200, xs), "mixing_x": xs,
                                      "x_component": "NI", "tdb": "CrNi-06Hal.tdb", "reference": "fcc hmix",
                                      "file": str(d / "fcc_hmix.json")})
    call(tb, "write_thermo_dataset", {"components": ["CR", "NI"], "phase": "CRNI2", "output": "HM_FORM",
                                      "T": [298.15, 600], "values": hm_form_crni2(dbf, [298.15, 600]),
                                      "configurations": [["CR", "NI"]], "tdb": "CrNi-06Hal.tdb",
                                      "reference": "crni2 form", "file": str(d / "crni2_hform.json")})
    call(tb, "write_equilibrium_dataset", {"components": ["CR", "NI"], "phases": ["LIQUID"], "output": "HMR",
                                           "conditions": {"T": 2200, "X_NI": xs},
                                           "values": eq_prop(dbf, ["LIQUID"], 2200, xs, "HM_MIX"),
                                           "reference_states": {"CR": "LIQUID", "NI": "LIQUID"},
                                           "reference": "liq hmr", "file": str(d / "liq_hmr.json")})
    call(tb, "write_equilibrium_dataset", {"components": ["CR", "NI"], "phases": ["LIQUID"], "output": "ACR_CR",
                                           "conditions": {"T": 2200, "X_NI": xs},
                                           "values": activity(dbf, "CR", 2200, xs, "LIQUID"),
                                           "reference_states": {"CR": "LIQUID"}, "reference": "liq acr",
                                           "file": str(d / "liq_acr.json")})
    xf = [0.7, 0.8, 0.9]
    call(tb, "write_equilibrium_dataset", {"components": ["CR", "NI"], "phases": ["FCC_A1"], "output": "ACR_NI",
                                           "conditions": {"T": 1300, "X_NI": xf}, "tdb": "CrNi-06Hal.tdb",
                                           "values": activity(dbf, "NI", 1300, xf, "FCC_A1"), "reference": "fcc acr",
                                           "file": str(d / "fcc_acr.json")})
    from calphad_tools import expdata
    liq = expdata.make_zpf_dataset(COMPS, [1650, 1700, 1650], [tieline(dbf, 1650, 0.6), tieline(dbf, 1700, 0.62),
                                                              tieline(dbf, 1650, 0.44)], reference="liq zpf")
    sol = expdata.make_zpf_dataset(COMPS, [1300, 1400, 1500], [tieline(dbf, T, 0.65) for T in (1300, 1400, 1500)],
                                   reference="solid zpf")
    # one vertex without composition (null) to exercise ESPEI's unknown-composition branch
    sol["values"][0][1][2] = [None]
    expdata.write_dataset(liq, str(d / "liq_zpf.json"))
    expdata.write_dataset(sol, str(d / "solid_zpf.json"))
    files = sorted(d.glob("*.json"))
    espei_load(files)
    bad = perturbed_tdb(base / "CrNi-perturbed.tdb")
    r_true = call(tb, "fit_report", {"tdb": "CrNi-06Hal.tdb", "datasets_dir": str(d), "elements": ["CR", "NI"],
                                     "diagram": False, "output_prefix": str(base / "true" / "fit")})
    r_bad = call(tb, "fit_report", {"tdb": str(bad), "datasets_dir": str(d), "output_prefix": str(base / "bad" / "fit"),
                                    "pptx": True})
    return dict(base=base, d=d, files=files, true=r_true, bad=r_bad)


def _summary(prefix):
    import csv
    return list(csv.DictReader(open(str(prefix) + "_summary.csv")))


def test_fit_report_same_tdb_is_zero(synthetic):
    text, imgs = synthetic["true"]
    rows = _summary(synthetic["base"] / "true" / "fit")
    assert len(rows) == 8, text
    for r in rows:
        tol = 5.0 if r["kind"] == "ZPF" else 1e-4          # ZPF: tie-lines from minimizer, J/mol
        assert float(r["max_abs"]) < tol, r
    assert "worst" in text and any(i.endswith("_parity.png") for i in imgs)
    resid = (synthetic["base"] / "true" / "fit_residuals.csv").read_text().splitlines()
    assert len(resid) == 1 + 4 + 4 + 2 + 4 + 4 + 3 + 6 + 6
    acr = [ln for ln in resid if "liq_acr.json" in ln][0].split(",")
    assert abs(float(acr[7]) - float(acr[8])) < 1e-6          # calculated activity == experimental


def test_fit_report_perturbed_ranking(synthetic):
    text, imgs = synthetic["bad"]
    rows = _summary(synthetic["base"] / "bad" / "fit")
    order = [r["file"] for r in rows]
    by = {r["file"]: r for r in rows}
    # L0(liquid) changed by +10 kJ: HM_MIX = +10000 x(1-x) -> RMS 2139 J/mol; mu_Cr shifts by 10000 x_Ni^2
    assert set(order[:3]) == {"liq_hmix.json", "liq_hmr.json", "liq_acr.json"}, text
    assert abs(float(by["liq_hmix.json"]["rms"]) - 10000 * np.sqrt(np.mean([(x * (1 - x)) ** 2 for x in
                                                                        [0.2, 0.4, 0.6, 0.8]]))) < 1.0
    assert abs(float(by["liq_hmix.json"]["rms"]) - float(by["liq_hmr.json"]["rms"])) < 1e-3
    assert float(by["liq_zpf.json"]["rms"]) > 50
    for f in ("fcc_hmix.json", "crni2_hform.json", "fcc_acr.json"):
        assert float(by[f]["max_abs"]) < 1e-4, by[f]
    assert float(by["solid_zpf.json"]["max_abs"]) < 5.0
    assert order.index("liq_zpf.json") < order.index("solid_zpf.json")
    names = [Path(i).name for i in imgs]
    assert "fit_parity.png" in names and "fit_residual_hist.png" in names and "fit_zpf_CR-NI.png" in names
    assert (synthetic["base"] / "bad" / "fit_fit_report.pptx").stat().st_size > 30000


def test_figures_picklable(synthetic, dbf):
    import csv
    from calphad_tools import espei_data as ed
    rows = []
    for r in csv.DictReader(open(synthetic["base"] / "bad" / "fit_residuals.csv")):
        rows.append(dict(file=r["file"], reference=r["reference"], kind=r["kind"], output=r["output"],
                         exp=float(r["experimental"]) if r["experimental"] else None,
                         calc=float(r["calculated"]) if r["calculated"] else None, residual=float(r["residual"]),
                         sigma=float(r["sigma"]), z=float(r["residual_over_sigma"]), T=float(r["T_K"])))
    zrows, notes = ed.evaluate_dataset(dbf, ed._load_one(str(synthetic["d"] / "solid_zpf.json")), COMPS)
    assert len(zrows) == 6 and any("x(CR=?)" in r["point"] for r in zrows)
    for r in zrows:
        r.update(file="solid_zpf.json", kind="ZPF")
    overlay = ed.zpf_overlay_figure(dbf, ["CR", "NI"], zrows, x_step=0.05, t_step=25)
    for fig in (ed.parity_figure(rows), ed.histogram_figure(rows), overlay):
        assert fig is not None
        pickle.loads(pickle.dumps(fig))


# ---------------------------------------------------------------- POP / CSV import
POP = """$ thermochemical experiments (values computed from CrNi-06Hal)
TABLE_HEAD 100
CREATE_NEW_EQUILIBRIUM @@,1
CHANGE_STATUS PHASE LIQ=FIX 1
SET_REFERENCE_STATE CR LIQ * 101325
SET_REFERENCE_STATE NI LIQ,*,101325
SET_CONDITION P=101325 T=2200 X(NI)=@1
EXPERIMENT HMR=@2:200 ACR(CR)=@3:5%
LABEL AHL
TABLE_VALUES
{rows}
TABLE_END
CREATE_NEW_EQUILIBRIUM 200,1
CHANGE_STATUS PHASE FCC=FIX 1
SET_CONDITION P=101325 T=1200 X(NI)=0.8
EXPERIMENT HM.T={cp}:1
LABEL ACP
CREATE_NEW_EQUILIBRIUM 201,1
CHANGE_STATUS PHASE FCC=FIX 1
SET_CONDITION P=101325 T=1250 X(NI)=0.8
EXPERIMENT HM.T={cp2}:1
LABEL ACP
CREATE_NEW_EQUILIBRIUM 300,1
CHANGE_STATUS PHASE LIQ=FIX 1
SET_CONDITION P=101325 T=2200 X(NI)=0.5
EXPERIMENT AC(CR)=0.001 DGM(BCC)=0
LABEL AXX
CREATE_NEW_EQUILIBRIUM 400,1
CHANGE_STATUS PHASE LIQ,FCC=FIX 1
SET_CONDITION P=101325 T=1650
EXPERIMENT X(LIQ,CR)=0.45:0.01, X(FCC,CR)=0.42:0.01
LABEL ATIE
"""


def test_pop_thermo_conversion(tb, dbf, tmp_path):
    xs = [0.3, 0.5, 0.7]
    hmr = eq_prop(dbf, ["LIQUID"], 2200, xs, "HM_MIX")
    acr = activity(dbf, "CR", 2200, xs, "LIQUID")
    cp = eq_prop(dbf, ["FCC_A1"], 1200, [0.8], "CPM")[0]
    cp2 = eq_prop(dbf, ["FCC_A1"], 1250, [0.8], "CPM")[0]
    rows = "\n".join(f"{x} {h:.6f} {a:.8f}" for x, h, a in zip(xs, hmr, acr))
    (tmp_path / "thermo.POP").write_text(POP.format(rows=rows, cp=f"{cp:.6f}", cp2=f"{cp2:.6f}"))
    text, _ = call(tb, "convert_pop_to_thermo", {"path": str(tmp_path / "thermo.POP"), "components": ["CR", "NI"],
                                                 "tdb": "CrNi-06Hal.tdb", "output_dir": str(tmp_path / "pop")})
    files = sorted((tmp_path / "pop").glob("*.json"))
    assert len(files) == 3, text
    assert "AC(CR)" in text and "DGM(BCC)" in text and "not supported" in text and "X(LIQ" not in text
    dss = {json.loads(f.read_text())["output"]: json.loads(f.read_text()) for f in files}
    assert set(dss) == {"HMR", "ACR_CR", "CPM"}
    assert dss["HMR"]["reference_states"] == {"CR": {"phase": "LIQUID"}, "NI": {"phase": "LIQUID"}}
    assert dss["HMR"]["conditions"]["X_NI"] == xs and dss["HMR"]["weight"] == 2.5          # 500 / 200
    assert dss["ACR_CR"]["reference_state"]["phases"] == ["LIQUID"]
    assert dss["CPM"]["phases"] == ["FCC_A1"] and dss["CPM"]["conditions"]["T"] == [1200.0, 1250.0]
    espei_load(files)
    from calphad_tools import espei_data as ed
    r = ed.fit_report(str(TDB), str(tmp_path / "pop"), str(tmp_path / "popfit" / "f"), diagram=False)
    assert all(s["max_abs"] < 1e-3 for s in r["summary"]), r["text"]


def test_csv_thermo_conversion(tb, dbf, tmp_path):
    xs = [0.25, 0.5, 0.75]
    h = hm_mix_fixed(dbf, "LIQUID", 2200, xs)
    a = activity(dbf, "NI", 2200, xs, "LIQUID")
    lines = ["T_C,X(NI),HM_MIX,uncertainty,phase,reference"]
    lines += [f"{2200 - 273.15},{x},{v / 1000:.8f},0.25,LIQUID,Calo 1999" for x, v in zip(xs, h)]
    (tmp_path / "calo.csv").write_text("\n".join(lines) + "\n")
    lines = ["T_K,X(NI),ACR(NI),HMR,ref_phase,reference"]
    lines += [f"2200,{x},{ai:.8f},{hi:.6f},LIQUID,Emf 2001" for x, ai, hi in
              zip(xs, a, eq_prop(dbf, ["LIQUID"], 2200, xs, "HM_MIX"))]
    (tmp_path / "emf.csv").write_text("\n".join(lines) + "\n")
    text, _ = call(tb, "convert_csv_to_thermo", {"path": str(tmp_path / "calo.csv"), "components": ["CR", "NI"],
                                                 "tdb": "CrNi-06Hal.tdb", "value_unit": "kJ/mol-atom",
                                                 "output_dir": str(tmp_path / "csv")})
    text2, _ = call(tb, "convert_csv_to_thermo", {"path": str(tmp_path / "emf.csv"), "components": ["CR", "NI"],
                                                  "phase": "LIQUID", "tdb": "CrNi-06Hal.tdb",
                                                  "output_dir": str(tmp_path / "csv")})
    files = sorted((tmp_path / "csv").glob("*.json"))
    dss = [json.loads(f.read_text()) for f in files]
    assert sorted(d["output"] for d in dss) == ["ACR_NI", "HMR", "HM_MIX"], text + text2
    hm = [d for d in dss if d["output"] == "HM_MIX"][0]
    assert hm["reference"] == "Calo 1999" and hm["weight"] == 2.0 and hm["conditions"]["T"] == pytest.approx(2200)
    assert hm["solver"]["sublattice_occupancies"] == [[[0.75, 0.25]], [[0.5, 0.5]], [[0.25, 0.75]]]
    espei_load(files)
    from calphad_tools import espei_data as ed
    r = ed.fit_report(str(TDB), str(tmp_path / "csv"), str(tmp_path / "csvfit" / "f"), diagram=False)
    assert len(r["summary"]) == 3 and all(s["max_abs"] < 1e-3 for s in r["summary"]), r["text"]
    # HMR without a reference phase is refused with a clear message
    (tmp_path / "bad.csv").write_text("T_K,X(NI),HMR\n2200,0.5,-3000\n")
    text, _ = call(tb, "convert_csv_to_thermo", {"path": str(tmp_path / "bad.csv"), "components": ["CR", "NI"],
                                                 "phase": "LIQUID", "output_dir": str(tmp_path / "csv2")})
    assert "wrote 0 datasets" in text and "ref_phase" in text
