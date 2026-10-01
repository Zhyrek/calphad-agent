"""Parameter what-if edits, sensitivities and the gradient-based optimizer (calphad_tools/param_tools.py) and
their show_code generators (calphad_tools/codegen_params.py).

Synthetic ESPEI datasets (ZPF tie-lines + HM_MIX of FCC_A1) are generated from the ORIGINAL Cr-Ni database, so the
original parameters are the exact optimum; the optimizer must recover a known perturbation.
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

ROOT = Path(__file__).resolve().parents[1]
CRNI = ROOT / "examples" / "tdbs" / "CrNi-06Hal.tdb"
L0 = "L(FCC_A1,CR,NI:VA;0)"


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """ZPF tie-lines (FCC/BCC, FCC/LIQUID) and HM_MIX(FCC_A1) calculated with the original CrNi-06Hal.tdb."""
    from pycalphad import Database, calculate, equilibrium, variables as v
    from calphad_tools.espei_data import make_thermo_dataset, phase_info, write_dataset as write_thermo
    from calphad_tools.expdata import make_zpf_dataset, write_dataset
    d = tmp_path_factory.mktemp("synth_datasets")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(str(CRNI))
        comps = ["CR", "NI", "VA"]
        vals, Ts = [], []
        for T, x, phs in [(1100, 0.6, ("FCC_A1", "BCC_A2")), (1300, 0.6, ("FCC_A1", "BCC_A2")),
                          (1500, 0.6, ("FCC_A1", "BCC_A2")), (1640, 0.465, ("FCC_A1", "LIQUID")),
                          (1660, 0.40, ("FCC_A1", "LIQUID"))]:
            eq = equilibrium(db, comps, list(phs), {v.T: T, v.P: 101325, v.N: 1, v.X("CR"): x})
            names = [str(p) for p in np.squeeze(eq.Phase.values)]
            X = np.squeeze(eq.X.sel(component="CR").values)
            row = [[n, ["CR"], [round(float(X[i]), 6)]] for i, n in enumerate(names) if n]
            assert len(row) == 2
            vals.append(row)
            Ts.append(T)
        write_dataset(make_zpf_dataset(["CR", "NI"], Ts, vals, reference="synthetic"), str(d / "zpf_synth.json"))
        xs = [0.1, 0.2, 0.3, 0.4]
        res = calculate(db, comps, "FCC_A1", T=1200.0, P=101325, N=1, output="HM_MIX",
                        points=np.array([[x, 1 - x, 1.0] for x in xs]))
        ds, _ = make_thermo_dataset(comps, "FCC_A1", "HM_MIX", [np.squeeze(res.HM_MIX.values).tolist()], T=1200.0,
                                    phase_model=phase_info("FCC_A1", tdb=str(CRNI)), mixing_x=xs, x_component="CR",
                                    reference="synthetic")
        write_thermo(ds, str(d / "hm_synth.json"))
    return d


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def espei_nll(tdb_text, datasets_dir):
    """Independent -lnL: ESPEI's own setup_context + get_likelihood (dummy symbol = no fitted parameters)."""
    from espei.datasets import load_datasets, recursive_glob
    from espei.error_functions.context import setup_context
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database.from_string(tdb_text + "\n FUNCTION DV9999 298.15 0; 6000 N !\n", fmt="tdb")
        ctx = setup_context(db, load_datasets(recursive_glob(str(datasets_dir), "*.json")), ["DV9999"])
        return -sum(float(o.get_likelihood(np.zeros(1))) for o in ctx["residual_objs"])


def espei_nll_by_type(tdb_text, datasets_dir):
    from espei.datasets import load_datasets, recursive_glob
    from espei.error_functions.context import setup_context
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database.from_string(tdb_text + "\n FUNCTION DV9999 298.15 0; 6000 N !\n", fmt="tdb")
        ctx = setup_context(db, load_datasets(recursive_glob(str(datasets_dir), "*.json")), ["DV9999"])
        return {type(o).__name__: -float(o.get_likelihood(np.zeros(1))) for o in ctx["residual_objs"]}


# ---------------------------------------------------------------------------------------------------------
# expression algebra and in-place edits
# ---------------------------------------------------------------------------------------------------------
def test_expression_terms():
    from calphad_tools.param_tools import add_to_expr, get_coef, negate_expr, scale_expr, split_terms
    e = "-8856.94+157.48*T-26.908*T*LN(T)+0.00189435*T**2-1.47721E-06*T**3+139250*T**(-1)"
    t = split_terms(e)
    assert [x.basis for x in t] == ["const", "T", "TLNT", "T2", "T3", "TINV"]
    assert get_coef(e, "T3") == pytest.approx(-1.47721e-06)
    assert add_to_expr(e, 10.0) == e.replace("-8856.94", "-8846.94")
    assert add_to_expr("+GHSERCR+7284+0.163*T", -0.5, "T") == "+GHSERCR+7284-0.337*T"
    assert add_to_expr("+5320-10.5*T", "DV0001", "T") == "+5320-10.5*T+DV0001*T"
    assert add_to_expr("+700", 1.0, "TLNT") == "+700+T*LN(T)"
    assert scale_expr("+5320-10.5*T", 2) == "+10640-21*T"
    assert scale_expr("+GHSERCR-8260", 2) == "2*(GHSERCR-8260)"
    assert negate_expr("-1.5*CRNI3+1.5*CR2NI2") == "+1.5*CRNI3-1.5*CR2NI2"


def test_edits_round_trip_byte_identical(tmp_path):
    from calphad_tools.param_tools import TDBText, apply_pairs, edit_tdb, scan_commands
    from calphad_tools.tdbparse import parse_tdb
    out = tmp_path / "e.tdb"
    edits = [{"target": "L(FCC_A1,CR,NI;0)", "op": "add", "value": 2000},          # no :VA, as users write it
             {"target": "L(BCC_A2,NI,CR:VA;1)", "op": "add", "value": 100, "term": "T"},   # reversed order: sign
             {"target": "GHSERCR", "op": "add", "value": 10, "t_range": 1},       # second T range only
             {"target": "G(CRNI2,CR:NI)", "op": "scale", "value": 1.01},
             {"target": "TC(HCP_A3,CR,NI:VA;0)", "op": "remove"},
             {"target": "L(HCP_A3,CR,NI:VA;3)", "op": "add_parameter", "value": "-500+T", "reference": "test"},
             {"target": "VV0001", "op": "add_parameter", "value": 1234.5},
             {"target": "VV0001", "op": "set", "value": -42}]
    r = edit_tdb(str(CRNI), edits, str(out))
    orig = CRNI.read_text()
    new = out.read_text()
    assert CRNI.read_text() == orig                     # original untouched
    # applying the recorded replacements reproduces the file; undoing them gives the original back, byte for byte
    assert apply_pairs(orig, r["pairs"]) == new
    back = new
    for old, nw in reversed(r["pairs"]):
        assert back.count(nw) == 1
        back = back.replace(nw, old)
    assert back == orig
    # every command that was not edited is textually identical
    touched = {"L(FCC_A1,CR,NI:VA;0)", "L(A1_FCC,CR,NI:VA;0)", "L(BCC_A2,CR,NI:VA;1)", "L(A2_BCC,CR,NI:VA;1)",
               "GHSERCR", "G(CRNI2,CR:NI)", "TC(HCP_A3,CR,NI:VA;0)", "L(HCP_A3,CR,NI:VA;3)", "VV0001"}
    o = {(c.kind, c.name): orig[c.start:c.end] for c in scan_commands(orig) if c.name not in touched}
    n = {(c.kind, c.name): new[c.start:c.end] for c in scan_commands(new) if c.name not in touched}
    assert o == n and len(o) > 80
    # values
    db0, db1 = parse_tdb(orig, is_text=True), parse_tdb(new, is_text=True)
    p0, p1 = TDBText(orig), TDBText(new)

    def val(tt, db, name, T):
        c, s, _ = tt.find(name)
        return s * c.expr.evaluate(np.array([T]), db.evaluator())[0]
    assert val(p1, db1, L0, 1000) - val(p0, db0, L0, 1000) == pytest.approx(2000)
    assert val(p1, db1, "L(A1_FCC,CR,NI:VA;0)", 1000) - val(p0, db0, "L(A1_FCC,CR,NI:VA;0)", 1000) == \
        pytest.approx(2000)                                           # disordered-part twin of L12_FCC follows
    assert p1.find("L(BCC_A2,CR,NI:VA;1)")[0].expr.ranges[0][2] == "+15650-101.5*T"
    assert val(p1, db1, "L(BCC_A2,NI,CR:VA;1)", 1000) - val(p0, db0, "L(BCC_A2,NI,CR:VA;1)", 1000) == \
        pytest.approx(1000 * 100)
    g0, g1 = db0.evaluate_function("GHSERCR", [1000, 2500]), db1.evaluate_function("GHSERCR", [1000, 2500])
    assert g1 - g0 == pytest.approx([0, 10])
    assert "TC(HCP_A3,CR,NI:VA;0)" not in {c.name for c in p1.parameters()}
    assert p1.functions()["VV0001"].expr.ranges[0][2] == "-42"
    # a single-line edit keeps the original layout
    line = next(ln for ln in new.splitlines() if "L(FCC_A1,CR,NI:VA;0)" in ln)
    assert line == " PAR  L(FCC_A1,CR,NI:VA;0),,            +7320-10.5*T;,,               N 06Hal !"
    assert any("sign flipped" in w for w in r["warnings"])
    from pycalphad import Database
    Database(str(out))


def test_update_twins_off_and_errors(tmp_path):
    from calphad_tools.param_tools import edit_tdb
    r = edit_tdb(str(CRNI), [{"target": L0, "op": "add", "value": 1}], str(tmp_path / "a.tdb"), update_twins=False)
    assert len(r["pairs"]) == 1 and any("A1_FCC" in w for w in r["warnings"])
    with pytest.raises(KeyError):
        edit_tdb(str(CRNI), [{"target": "L(FCC_A1,CR,NI:VA;7)", "op": "add", "value": 1}], str(tmp_path / "b.tdb"))
    with pytest.raises(ValueError):
        edit_tdb(str(CRNI), [{"target": L0, "op": "add_parameter", "value": 1}], str(tmp_path / "c.tdb"))
    with pytest.raises(ValueError, match="used by"):
        edit_tdb(str(CRNI), [{"target": "GHSERCR", "op": "remove"}], str(tmp_path / "d.tdb"))


# ---------------------------------------------------------------------------------------------------------
# ESPEI objective
# ---------------------------------------------------------------------------------------------------------
def test_objective_matches_espei_and_scipy_optimizer_sign(synth):
    from espei.optimizers.opt_scipy import SciPyOptimizer
    from calphad_tools.param_tools import Objective, TDBText, resolve_free_params
    fps = resolve_free_params(TDBText.read(str(CRNI)), [L0, L0 + "@T", "L(FCC_A1,NI,CR:VA;1)@T"])
    assert fps[2].x0 == pytest.approx(14.4)             # file has -14.4*T for L(CR,NI;1): user spelling flips it
    ob = Objective(str(CRNI), str(synth), fps)
    for dx in ([0, 0, 0], [2000, 0, 0], [0, 1.5, -2.0]):
        x = ob.x0 + np.array(dx)
        by = ob.nll_from_blocks(ob.blocks(x))
        assert sum(by.values()) == pytest.approx(ob.nll(x), rel=1e-10)
        lp = float(SciPyOptimizer.predict(ob.vec(x), ob.ctx))
        assert lp == pytest.approx(-ob.nll(x), rel=1e-10)     # predict = +lnL, which SciPyOptimizer MINIMISES
    # the original database is the optimum of the synthetic data: moving away increases -lnL
    assert ob.nll(ob.x0 + np.array([2000, 0, 0])) > ob.nll(ob.x0)


def test_edit_tool_compare_and_objective_change(tb, synth):
    from calphad_tools.param_tools import objective_compare
    text, imgs = call(tb, "edit_parameters", {
        "tdb": "CrNi-06Hal.tdb", "edits": [{"target": "L(FCC_A1,CR,NI;0)", "op": "add", "value": 2000}],
        "compare": {"elements": ["NI", "CR"], "tmin": 1000, "tmax": 2000, "boundary_temperatures": [1200, 1500]},
        "datasets_dir": str(synth)})
    out = Path(tb.out) / "CrNi-06Hal_edited.tdb"
    assert out.exists() and len(imgs) == 1 and Path(imgs[0]).exists()
    m = re.search(r"BCC_A2\+FCC_A1\+LIQUID eutectic: ([\d.]+) -> ([\d.]+) K \(dT ([-+\d.]+) K\)", text)
    assert m, text
    assert float(m.group(1)) == pytest.approx(1617.86, abs=0.05)
    assert -60 < float(m.group(3)) < -30                          # more positive L0(fcc): fcc less stable
    shifts = pd.read_csv(Path(imgs[0]).with_suffix(".csv"))
    fcc = shifts[shifts["boundary"].astype(str).str.endswith("|FCC_A1")]
    assert (fcc["dX"] < 0).all() and len(fcc) == 2
    # the objective change equals ESPEI's own computation on the two files
    n0, n1 = espei_nll(CRNI.read_text(), synth), espei_nll(out.read_text(), synth)
    oc = objective_compare(str(CRNI), str(out), str(synth))
    assert oc["nll0"] == pytest.approx(n0, rel=1e-10) and oc["nll1"] == pytest.approx(n1, rel=1e-10)
    assert f"change {n1 - n0:+.4f}" in text and n1 > n0


# ---------------------------------------------------------------------------------------------------------
# sensitivities
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def perturbed(tmp_path_factory):
    from calphad_tools.param_tools import edit_tdb
    p = tmp_path_factory.mktemp("pert") / "CrNi_perturbed.tdb"
    edit_tdb(str(CRNI), [{"target": L0, "op": "add", "value": 2000}], str(p))
    return p


def test_sensitivity_matches_recomputation(tmp_path, synth, perturbed):
    """At the perturbed database (not the optimum, where dNLL/dp ~ 0 is only solver noise)."""
    from pycalphad import Database, Workspace, variables as v
    from calphad_tools.param_tools import edit_tdb, sensitivity
    r = sensitivity(str(perturbed), str(tmp_path / "s"), parameters=[L0, L0 + "@T"], datasets_dir=str(synth),
                    elements=["NI", "CR"], tmin=1400, tmax=1800, invariants=True,
                    conditions={"T": 1200, "X(CR)": 0.3}, properties=["HM", "MU(CR)"])
    tab = pd.read_csv(r["files"]["csv"])

    def d(q, p):
        return float(tab[(tab.quantity == q) & (tab.parameter == p)].derivative.iloc[0])
    h = r["steps"][0]                                   # 0.01 * 7320 J/mol
    assert h == pytest.approx(73.2)
    # independent recomputation: ESPEI likelihood of +-h databases, and pycalphad HM
    txt = {}
    for s in (1, -1):
        p = tmp_path / f"pm{s}.tdb"
        edit_tdb(str(perturbed), [{"target": L0, "op": "add", "value": s * h}], str(p))
        txt[s] = p
    # fixed-configuration (HM_MIX) part: exact against fresh ESPEI evaluations of the two edited files
    fx = {s: espei_nll_by_type(txt[s].read_text(), synth)["FixedConfigurationPropertyResidual"] for s in (1, -1)}
    assert d("NLL_thermo_fixed", L0) == pytest.approx((fx[1] - fx[-1]) / (2 * h), rel=1e-8)
    # total: against ESPEI's symbol route (as in MCMC), built by hand: +DV0001 in the fcc L0 lines
    from espei.datasets import load_datasets, recursive_glob
    from espei.error_functions.context import setup_context
    ptxt = perturbed.read_text()
    assert ptxt.count("+7320-10.5*T;") == 2                  # FCC_A1 and its twin A1_FCC
    ptxt = ptxt.replace("+7320-10.5*T;", "+7320-10.5*T+DV0001;") + "\n FUNCTION DV0001 298.15 0; 6000 N !\n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = setup_context(Database.from_string(ptxt, fmt="tdb"),
                            load_datasets(recursive_glob(str(synth), "*.json")), ["DV0001"])
    nl = {s: -sum(float(o.get_likelihood(np.array([s * h]))) for o in ctx["residual_objs"]) for s in (1, -1)}
    dn = (nl[1] - nl[-1]) / (2 * h)
    assert abs(dn) > 1e-4
    assert d("NLL_total", L0) == pytest.approx(dn, rel=1e-8)
    assert "disordered part of L12_FCC" in r["text"]           # pycalphad limitation is reported
    hm = {}
    for s in (1, -1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = Database(str(txt[s]))
            wks = Workspace(db, ["CR", "NI", "VA"], sorted(db.phases),
                            {v.T: 1200.0, v.X("CR"): 0.3, v.P: 101325.0, v.N: 1.0})
            hm[s] = float(np.squeeze(wks.get("HM")))
    q = 'HM@{"T": 1200, "X(CR)": 0.3}'
    assert d(q, L0) == pytest.approx((hm[1] - hm[-1]) / (2 * h), rel=1e-8)
    assert d(q, L0) == pytest.approx(0.3 * 0.7, rel=1e-4)        # single-phase fcc: dHM/dL0 = x(1-x)
    assert abs(d(q, L0 + "@T")) < 1e-3                           # the T coefficient does not enter H
    # eutectic temperature: about -20 K per kJ/mol (edit_parameters: +2 kJ/mol -> -45 K)
    dT = d("T_inv[BCC_A2+FCC_A1+LIQUID]", L0)
    assert -0.03 < dT < -0.012
    assert Path(r["files"]["tornado_png"]).exists() and Path(r["files"]["fisher_csv"]).exists()
    assert "condition number" in r["text"]


def test_fisher_identifiability_warnings():
    from calphad_tools.param_tools import FreeParam, fisher_analysis, jacobian_degeneracy
    fps = [FreeParam(f"P{i}", f"P{i}", "const", f"VV000{i}", 1000.0, 1000.0, True) for i in range(3)]
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    J = np.column_stack([a, 2 * a + 1e-7 * rng.normal(size=20), np.zeros(20)])
    f = fisher_analysis(J, fps)
    w = " | ".join(f["warnings"])
    assert "P2: no data depend on it" in w
    assert "P0 and P1 are correlated" in w and "near-degenerate direction" in w
    J2 = np.column_stack([a, rng.normal(size=20), rng.normal(size=20)]) / 1000
    assert not [x for x in fisher_analysis(J2, fps)["warnings"] if "correlated" in x]
    D = {"q1": np.array([1.0, 2.0, 0.3]), "q2": np.array([2.0, 4.0, -1.0])}
    assert "P0 and P1 shift all requested quantities" in " ".join(jacobian_degeneracy(D, fps))


def test_equilibrium_dataset_order_warning():
    from espei.utils import PickleableTinyDB
    from tinydb.storages import MemoryStorage
    from calphad_tools.param_tools import dataset_warnings
    db = PickleableTinyDB(storage=MemoryStorage)
    db.insert({"components": ["CR", "NI"], "phases": ["FCC_A1"], "output": "HM", "reference": "x",
               "conditions": {"P": 101325, "T": [1000, 1100], "X_CR": [0.1, 0.2, 0.3]},
               "values": np.zeros((1, 2, 3)).tolist()})
    assert "wrong order" in " ".join(dataset_warnings(db))


# ---------------------------------------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------------------------------------
def test_optimizer_recovers_known_perturbation(tmp_path, synth, perturbed):
    from calphad_tools.param_tools import TDBText, optimize_job
    msgs = []
    r = optimize_job(str(perturbed), str(synth), str(tmp_path / "opt"), parameters=[L0],
                     progress=lambda m, f=None, **k: msgs.append(m))
    assert r["x"][0] == pytest.approx(5320.0, abs=2.0), r["text"]
    assert r["nll1"] < r["nll0"] and msgs[-1] == "done"
    tt = TDBText.read(r["file_map"]["tdb"])
    for name in (L0, "L(A1_FCC,CR,NI:VA;0)"):                  # the twin was optimised along
        e = tt.find(name)[0].expr.ranges[0][2]
        c = float(re.match(r"\+?(-?[\d.]+)", e).group(1))
        assert c == pytest.approx(5320, abs=2.0)
    tab = pd.read_csv(r["file_map"]["parameters_csv"])
    assert tab["change"].iloc[0] == pytest.approx(-2000, abs=2) and tab["approx_std"].iloc[0] > 0
    hist = pd.read_csv(r["file_map"]["history_csv"])
    assert hist["best_nll"].iloc[-1] == pytest.approx(r["nll1"], abs=1e-3)
    fit = pd.read_csv(r["file_map"]["fit_before_after_csv"])
    assert (fit["rms_over_sigma_after"] < 0.01).all() and (fit["rms_over_sigma_before"] > 0.2).all()
    assert Path(r["file_map"]["diagram_CR-NI"]).exists()


def test_optimizer_lbfgsb_two_parameters_bounds(tmp_path, synth, perturbed):
    from calphad_tools.param_tools import optimize_job
    r = optimize_job(str(perturbed), str(synth), str(tmp_path / "opt2"), parameters=[L0, L0 + "@T"],
                     method="lbfgsb", compare=False, bounds={L0 + "@T": [-11.0, -10.0]})
    assert r["x"][0] == pytest.approx(5320.0, abs=5.0), r["text"]
    assert r["x"][1] == pytest.approx(-10.5, abs=0.01)
    r = optimize_job(str(perturbed), str(synth), str(tmp_path / "opt3"), parameters=[L0], compare=False,
                     bounds={L0: [6500, 8000]})
    assert r["x"][0] == pytest.approx(6500.0, abs=1e-6) and "AT BOUND" in r["text"]


def test_optimize_tool_background_job(tb, synth, perturbed):
    text, _ = call(tb, "optimize_parameters", {"tdb": str(perturbed), "datasets_dir": str(synth),
                                               "parameters": [L0], "background": True})
    jid = re.search(r"job (\S+):", text).group(1)
    text, _ = call(tb, "wait_job", {"job_id": jid, "timeout_s": 240})
    assert "state: done" in text, text
    tdb = Path(tb.out) / "optimize_CrNi_perturbed" / "CrNi_perturbed_optimized.tdb"
    assert tdb.exists()
    res = json.loads((Path(tb.out) / "jobs" / jid / "result.json").read_text())
    assert res["x"][0] == pytest.approx(5320.0, abs=2.0)


def test_tool_descriptions_compact(tb):
    for name in ("edit_parameters", "parameter_sensitivity", "optimize_parameters"):
        assert name in tb.tools
        assert len(tb.tools[name].spec.description) < 650, name


# ---------------------------------------------------------------------------------------------------------
# show_code generators
# ---------------------------------------------------------------------------------------------------------
def standalone(tb, target):
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, target)
    assert all(c.standalone for c in cells), [(c.title, c.note) for c in cells]
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    return script


def run(script, tmp_path):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "gen.py").write_text(script)
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return d / "standalone_outputs", r.stdout


def test_codegen_edit_parameters(tb, tmp_path, synth):
    text, imgs = call(tb, "edit_parameters", {
        "tdb": "CrNi-06Hal.tdb", "edits": [{"target": "L(FCC_A1,CR,NI;0)", "op": "add", "value": 2000},
                                           {"target": "L(LIQUID,CR,NI;1)", "op": "add", "value": 500}],
        "compare": {"elements": ["NI", "CR"], "tmin": 1000, "tmax": 2000, "boundary_temperatures": [1300]},
        "datasets_dir": str(synth)})
    out, stdout = run(standalone(tb, "1"), tmp_path)
    assert (out / "CrNi-06Hal_edited.tdb").read_bytes() == (Path(tb.out) / "CrNi-06Hal_edited.tdb").read_bytes()
    tool = pd.read_csv(Path(imgs[0]).with_suffix(".csv"))
    gen = pd.read_csv(out / "compare_NI-CR_shifts.csv")
    ti = tool.dropna(subset=["invariant"]).set_index("invariant")
    gi = gen.dropna(subset=["invariant"]).set_index("invariant")
    # pycalphad mapping may pick up extra metastable nodes in one process and not in another: compare common ones
    common = set(ti.index) & set(gi.index)
    assert "BCC_A2+FCC_A1+LIQUID" in common
    for k in common:
        for c in ("T0_K", "T1_K"):
            assert ti.loc[k, c] == pytest.approx(gi.loc[k, c], abs=1e-3, nan_ok=True)
    tb_ = tool.dropna(subset=["boundary"]).set_index("boundary")
    gb = gen.dropna(subset=["boundary"]).set_index("boundary")
    common = set(tb_.index) & set(gb.index)
    assert {"BCC_A2 + FCC_A1|FCC_A1", "BCC_A2 + FCC_A1|BCC_A2"} <= common
    for k in common:
        assert tb_.loc[k, "dX"] == pytest.approx(gb.loc[k, "dX"], abs=1e-6)
    m = re.search(r"ESPEI -lnL: ([\d.]+) -> ([\d.]+)", stdout)
    assert m and f"{float(m.group(1)):.4f} -> {float(m.group(2)):.4f}" in text


def test_codegen_parameter_sensitivity(tb, tmp_path, synth):
    call(tb, "parameter_sensitivity", {"tdb": "CrNi-06Hal.tdb", "parameters": [L0, "L(FCC_A1,NI,CR:VA;1)@T"],
                                       "datasets_dir": str(synth), "elements": ["CR", "NI"],
                                       "conditions": {"T": 1200, "X(CR)": 0.3}, "properties": ["HM", "X(FCC_A1,CR)"],
                                       "background": False})
    out, _ = run(standalone(tb, "1"), tmp_path)
    tool = pd.read_csv(Path(tb.out) / "sensitivity_CrNi-06Hal" / "sens_CrNi-06Hal_sensitivity.csv")
    gen = pd.read_csv(out / "sens_CrNi-06Hal_sensitivity.csv")
    t = tool.set_index(["quantity", "parameter"])["derivative"]
    g = gen.set_index(["quantity", "parameter"])["derivative"]
    assert set(t.index) == set(g.index) and len(t) == 2 * 5
    for k in t.index:
        assert t[k] == pytest.approx(g[k], rel=1e-6, abs=1e-9), k


def test_codegen_optimize_parameters(tb, tmp_path, synth, perturbed):
    text, _ = call(tb, "optimize_parameters", {"tdb": str(perturbed), "datasets_dir": str(synth),
                                               "parameters": [L0, L0 + "@T"], "compare": False,
                                               "background": False})
    out, stdout = run(standalone(tb, "1"), tmp_path)
    tool = pd.read_csv(Path(tb.out) / "optimize_CrNi_perturbed" / "CrNi_perturbed_parameters.csv")
    gen = pd.read_csv(out / "CrNi_perturbed_parameters.csv")
    assert list(tool["parameter"]) == list(gen["parameter"])
    assert np.allclose(tool["after"], gen["after"], rtol=0, atol=[1.0, 1e-3])
    assert gen["after"].iloc[0] == pytest.approx(5320, abs=2.0)
    # the generated optimized TDB (fitted DV symbols) describes the same database as the tool's numeric edit
    gtext = (out / "CrNi_perturbed_optimized.tdb").read_text()
    ttext = (Path(tb.out) / "optimize_CrNi_perturbed" / "CrNi_perturbed_optimized.tdb").read_text()
    assert espei_nll(gtext, synth) == pytest.approx(espei_nll(ttext, synth), abs=1e-3)
