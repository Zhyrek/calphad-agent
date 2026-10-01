"""Tests for ESPEI MCMC post-processing (calphad_tools.espei_post).

A real (short) ESPEI run is made once per session: Cr-Ni with LIQUID/FCC_A1/BCC_A2, synthetic
HM_MIX/SM_MIX single-phase data (generate_parameters -> VV symbols) plus the Cr-Ni ZPF datasets of
outputs/espei_demo, then MCMC with scheduler null, 2 chains per parameter.
"""
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ZPF_DIR = ROOT / "outputs/espei_demo/CrNi/datasets"
MCMC_ITERATIONS = 14
R = 8.31451

# Hallstedt (2006) Cr-Ni Redlich-Kister terms L0, L1 = a + b*T (L2 dropped to keep the fit small)
_L = {"LIQUID": [(-14190.0, 2.3), (-950.0, 4.7)],
      "FCC_A1": [(5320.0, -10.5), (32030.0, -14.4)],
      "BCC_A2": [(22780.0, -14.6), (15650.0, -1.5)]}
_SUBL = {"LIQUID": ([["CR", "NI"]], [1.0]), "FCC_A1": ([["CR", "NI"], ["VA"]], [1.0, 1.0]),
         "BCC_A2": ([["CR", "NI"], ["VA"]], [1.0, 3.0])}


def _write_inputs(d: Path):
    ds = d / "datasets"
    ds.mkdir(parents=True)
    xs = np.array([0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    for ph, Ls in _L.items():
        sl, ratios = _SUBL[ph]
        xcr, xni = xs, 1 - xs
        dd = xcr - xni
        H = xcr * xni * sum(a * dd ** k for k, (a, b) in enumerate(Ls))
        S = -xcr * xni * sum(b * dd ** k for k, (a, b) in enumerate(Ls)) - R * (xcr * np.log(xcr) + xni * np.log(xni))
        cfg = [[["CR", "NI"]] + (["VA"] if len(sl) > 1 else []) for _ in xs]
        occ = [[[float(a), float(1 - a)]] + ([1.0] if len(sl) > 1 else []) for a in xs]
        for out, vals in (("HM_MIX", H), ("SM_MIX", S)):
            (ds / f"{ph}_{out}.json").write_text(json.dumps({
                "components": ["CR", "NI"] + (["VA"] if len(sl) > 1 else []), "phases": [ph],
                "solver": {"mode": "manual", "sublattice_site_ratios": ratios, "sublattice_configurations": cfg,
                           "sublattice_occupancies": occ},
                "conditions": {"P": 101325.0, "T": 1600.0}, "output": out, "values": [[[float(v) for v in vals]]],
                "excluded_model_contributions": ["mag"], "reference": "synthetic (Hallstedt 2006 L0, L1)",
                "comment": "test data"}, indent=1))
    for f in ZPF_DIR.glob("*.json"):
        shutil.copy(f, ds / f.name)
    pm = {"components": ["CR", "NI", "VA"],
          "phases": {p: {"sublattice_model": _SUBL[p][0], "sublattice_site_ratios": _SUBL[p][1]} for p in _L}}
    (d / "phase_models.json").write_text(json.dumps(pm, indent=1))
    gen = {"system": {"phase_models": "phase_models.json", "datasets": "datasets"},
           "output": {"verbosity": 0, "output_db": "dft.tdb", "logfile": "generate.log"},
           "generate_parameters": {"excess_model": "linear", "ref_state": "SGTE91"}}
    mcmc = {"system": {"phase_models": "phase_models.json", "datasets": "datasets"},
            "output": {"verbosity": 0, "output_db": "mcmc.tdb", "tracefile": "trace.npy", "probfile": "lnprob.npy",
                       "logfile": "mcmc.log"},
            "mcmc": {"iterations": MCMC_ITERATIONS, "input_db": "dft.tdb", "scheduler": None,
                     "chains_per_parameter": 2, "chain_std_deviation": 0.1, "save_interval": 1,
                     "deterministic": True}}
    (d / "espei_generate.yaml").write_text(yaml.safe_dump(gen, sort_keys=False))
    (d / "espei_mcmc.yaml").write_text(yaml.safe_dump(mcmc, sort_keys=False))


_RUNNER = ("import sys, yaml\nfrom espei.espei_script import run_espei\n"
           "run_espei(yaml.safe_load(open(sys.argv[1])))\n")


@pytest.fixture(scope="session")
def espei_run(tmp_path_factory):
    """Real ESPEI parameter generation + short MCMC (run in a subprocess, ESPEI configures logging)."""
    d = tmp_path_factory.mktemp("espei_crni")
    _write_inputs(d)
    t0 = time.time()
    for y in ("espei_generate.yaml", "espei_mcmc.yaml"):
        r = subprocess.run([sys.executable, "-c", _RUNNER, y], cwd=d, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
    secs = time.time() - t0
    print(f"\nESPEI generate + {MCMC_ITERATIONS}-iteration MCMC: {secs:.1f} s")
    assert (d / "trace.npy").exists() and (d / "lnprob.npy").exists() and (d / "mcmc.tdb").exists()
    return d


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(tmp_path), output_dir="out", ui=U(), python_tool="off")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


# ---------------------------------------------------------------------------
# statistics on synthetic chains
# ---------------------------------------------------------------------------
def _synthetic(n_ch=24, n=600, burn=120, seed=1):
    rng = np.random.default_rng(seed)
    mu = np.array([1000.0, -5.0, 3e4])
    sd = np.array([20.0, 0.5, 300.0])
    tr = mu + sd * rng.standard_normal((n_ch, n, 3))
    # burn-in: start far away and relax exponentially
    relax = np.exp(-np.arange(n) / (burn / 6.0))[None, :, None]
    tr = tr + relax * 30 * sd
    lp = -0.5 * np.sum(((tr - mu) / sd) ** 2, axis=2)
    return tr, lp, mu, sd


def test_statistics_recovered_on_known_distribution():
    from calphad_tools import espei_post as ep
    tr, lp, mu, sd = _synthetic()
    burn, info = ep.estimate_burn_in(lp)
    assert 40 <= burn <= 200, (burn, info)
    st = ep.summarize_chains(tr, lp, ["VV0000", "VV0001", "VV0002"], burn)
    for k, p in enumerate(st["parameters"]):
        assert abs(p["mean"] - mu[k]) < 0.05 * sd[k] + 1e-9, p
        assert abs(p["std"] / sd[k] - 1) < 0.05, p
        lo, hi = p["ci95"]
        assert abs(lo - (mu[k] - 1.96 * sd[k])) < 0.1 * sd[k]
        assert abs(hi - (mu[k] + 1.96 * sd[k])) < 0.1 * sd[k]
        assert 0.9 < p["tau"] < 1.6 and p["tau_reliable"]         # iid draws
        assert p["ess"] > 0.6 * st["n_samples"]
        assert abs(p["rhat"] - 1) < 0.02
    assert st["acceptance_fraction"]["mean"] == pytest.approx(1.0)
    assert not st["strong_correlations"]


def test_burn_in_detection_exponential_approach():
    from calphad_tools import espei_post as ep
    rng = np.random.default_rng(3)
    n = 400
    lp = -50 - 1000 * np.exp(-np.arange(n) / 20.0)[None, :] + rng.standard_normal((20, n))
    burn, info = ep.estimate_burn_in(lp)
    assert 100 <= burn <= 200, (burn, info)          # 1000 exp(-i/20) < ~1 at i ~ 140
    assert "warning" not in info
    # a run that never stops rising is flagged
    lp2 = np.linspace(-1000, -10, n)[None, :] + rng.standard_normal((20, n))
    b2, info2 = ep.estimate_burn_in(lp2)
    assert b2 > n / 2 and "warning" in info2


def test_acceptance_autocorrelation_and_rhat():
    from calphad_tools import espei_post as ep
    rng = np.random.default_rng(5)
    n_ch, n = 16, 3000
    # Metropolis-like sticky chains: move with probability 0.3
    tr = np.zeros((n_ch, n, 1))
    for c in range(n_ch):
        x = 0.0
        for i in range(n):
            if rng.random() < 0.3:
                x = rng.standard_normal()
            tr[c, i, 0] = x
    acc = ep.acceptance_fraction(tr)
    assert np.allclose(acc.mean(), 0.3, atol=0.02)
    tau = ep.integrated_time(tr[:, :, 0].T)
    # AR-like: rho(k) = 0.7^k -> tau = (1 + 0.7) / (1 - 0.7) = 5.67
    assert 4.5 < tau < 7.0, tau
    # chains stuck at different values -> R-hat >> 1
    bad = rng.standard_normal((8, 200)) * 0.01 + np.arange(8)[:, None]
    assert ep.split_rhat(bad) > 2
    # a constant (stuck) walker does not poison tau
    tr2 = tr.copy()
    tr2[0, :, 0] = 1.234
    assert np.isfinite(ep.integrated_time(tr2[:, :, 0].T))


def test_zero_filled_truncation():
    from calphad_tools import espei_post as ep
    tr, lp, _, _ = _synthetic(n_ch=6, n=50)
    full_tr, full_lp = np.zeros((6, 80, 3)), np.zeros((6, 80))
    full_tr[:, :50], full_lp[:, :50] = tr, lp
    t2, l2, how = ep.truncate_zero_filled(full_tr, full_lp)
    assert t2.shape == (6, 50, 3) and l2.shape == (6, 50) and "truncate_arrays" in how
    # a parameter that is exactly zero in one chain breaks espei's reshape -> robust fallback
    full_tr[2, :50, 1] = 0.0
    full_tr[3, 10, :] = 0.0
    t3, _, how3 = ep.truncate_zero_filled(full_tr, full_lp)
    assert t3.shape[1] == 50


# ---------------------------------------------------------------------------
# real ESPEI run, end to end
# ---------------------------------------------------------------------------
def test_symbol_mapping_matches_espei(espei_run):
    from espei.utils import database_symbols_to_fit
    from pycalphad import Database
    from calphad_tools import espei_post as ep
    db = Database(str(espei_run / "dft.tdb"))
    tr = np.load(espei_run / "trace.npy")
    syms = ep.fitted_symbols(db, tr.shape[2])
    assert syms == database_symbols_to_fit(db) and len(syms) == tr.shape[2]
    use = ep.symbol_usage(db, syms)
    labels = {s: ep._describe(use, s) for s in syms}
    assert all("not used" not in v for v in labels.values()), labels
    # every L parameter is a + b*T: one 'const' and one 'T' symbol each
    joined = " ".join(labels.values())
    assert "L(LIQUID,CR,NI;0) [const]" in joined and "L(LIQUID,CR,NI;0) [T]" in joined
    assert "L(BCC_A2,CR,NI:VA;1)" in joined
    # the MCMC starting point is the generated value: column medians of iteration 0 ~ TDB values
    start = np.median(tr[:, 0, :], axis=0)
    for s, v in zip(syms, start):
        ref = use[s]["value_in_tdb"]
        assert abs(v - ref) <= 0.25 * abs(ref) + 1e-9
    with pytest.raises(ValueError, match="parameter columns"):
        ep.fitted_symbols(db, tr.shape[2] + 1)


def test_analyze_mcmc_tool(tb, espei_run):
    from calphad_tools.figures import load_figure
    text, imgs = call(tb, "analyze_mcmc", {"trace": str(espei_run)})
    assert "burn-in" in text and "acceptance fraction" in text and "VV0000" in text
    assert "L(LIQUID,CR,NI;0)" in text
    assert len(imgs) == 4 and all(Path(p).exists() for p in imgs)
    js = json.loads(Path(tb.out / "mcmc_analysis" / f"{espei_run.name}_summary.json").read_text())
    n_par = np.load(espei_run / "trace.npy").shape[2]
    assert js["n_parameters"] == n_par == len(js["parameters"])
    assert js["n_chains"] == 2 * n_par and js["completed_iterations"] == MCMC_ITERATIONS
    for p in js["parameters"]:
        assert p["ci95"][0] <= p["median"] <= p["ci95"][1] and p["tdb_parameters"]
    # figures are stored editable (pickled) -> must be picklable
    for p in imgs:
        fig = load_figure(p)
        assert fig is not None, p
        pickle.dumps(fig)
    # explicit burn-in + corner subset + thin
    text2, imgs2 = call(tb, "analyze_mcmc", {"trace": str(espei_run / "trace.npy"),
                                             "lnprob": str(espei_run / "lnprob.npy"),
                                             "tdb": str(espei_run / "mcmc.tdb"), "burn_in": "4", "thin": 2,
                                             "corner_parameters": ["VV0001", "VV0003", "VV0005"],
                                             "output_prefix": "an2"})
    assert "burn-in 4 (user)" in text2 and Path(tb.out / "an2_corner.png").exists()


def test_analyze_unfinished_run(tb, espei_run, tmp_path):
    """Zero-fill the last iterations as ESPEI does for an interrupted run."""
    tr, lp = np.load(espei_run / "trace.npy"), np.load(espei_run / "lnprob.npy")
    k = tr.shape[1] - 5
    tr2, lp2 = tr.copy(), lp.copy()
    tr2[:, k:], lp2[:, k:] = 0, 0
    d = tmp_path / "partial"
    d.mkdir()
    np.save(d / "trace.npy", tr2)
    np.save(d / "lnprob.npy", lp2)
    shutil.copy(espei_run / "dft.tdb", d / "dft.tdb")
    text, _ = call(tb, "analyze_mcmc", {"trace": str(d), "output_prefix": "partial"})
    assert f"iterations completed {k}/{tr.shape[1]}" in text and "zero-filled" in text


def test_mcmc_to_tdb(tb, espei_run):
    from pycalphad import Database
    from calphad_tools import espei_post as ep
    tr, lp = np.load(espei_run / "trace.npy"), np.load(espei_run / "lnprob.npy")
    text, _ = call(tb, "mcmc_to_tdb", {"trace": str(espei_run), "tdb": str(espei_run / "dft.tdb"),
                                       "statistic": "map", "output": "crni_map.tdb"})
    assert "wrote" in text and "VV0000" in text
    db = Database(str(tb.out / "crni_map.tdb"))
    syms = ep.fitted_symbols(db, tr.shape[2])
    c, it = divmod(int(np.argmax(lp)), lp.shape[1])
    got = np.array([ep._unpack(db.symbols[s]) for s in syms])
    assert np.allclose(got, tr[c, it], rtol=1e-6)
    # ESPEI's own mcmc.tdb is the MAP (optimal_parameters) -> same values
    esp = Database(str(espei_run / "mcmc.tdb"))
    assert np.allclose(got, [ep._unpack(esp.symbols[s]) for s in syms], rtol=1e-6)
    call(tb, "mcmc_to_tdb", {"trace": str(espei_run), "statistic": "median", "burn_in": "3", "output": "med"})
    db2 = Database(str(tb.out / "med.tdb"))
    med = np.median(tr[:, 3:, :].reshape(-1, tr.shape[2]), axis=0)
    assert np.allclose([ep._unpack(db2.symbols[s]) for s in syms], med, rtol=1e-6)


def test_propagate_binary_diagram(tb, espei_run):
    t0 = time.time()
    text, imgs = call(tb, "propagate_uncertainty", {
        "trace": str(espei_run), "kind": "binary_diagram", "elements": ["CR", "NI"], "tmin": 1300, "tmax": 2300,
        "n_samples": 4, "composition_step": 0.05, "temperature_step": 25, "output_prefix": "unc_diagram"})
    print(f"\nbinary propagation (4 samples + median): {time.time() - t0:.1f} s")
    assert "BCC_A2 + FCC_A1 + LIQUID" in text and "eutectic" in text
    js = json.loads(Path(tb.out / "unc_diagram_summary.json").read_text())
    assert js["n_samples_ok"] == 4
    inv = next(g for g in js["invariants"] if g["phases"] == "BCC_A2 + FCC_A1 + LIQUID")
    assert inv["found_in_samples"] >= 3
    assert 1450 < inv["T_K"]["median"] < 1800
    assert imgs[0].endswith("unc_diagram.png") and Path(imgs[0]).exists() and len(imgs) == 2
    from calphad_tools.figures import load_figure
    assert all(load_figure(p) is not None for p in imgs)
    assert Path(tb.out / "unc_diagram_boundaries.csv").exists()


def test_propagate_property(tb, espei_run):
    text, imgs = call(tb, "propagate_uncertainty", {
        "trace": str(espei_run), "kind": "property", "elements": ["CR", "NI"], "quantity": "NP(LIQUID)",
        "conditions": {"T": [1550, 1850, 10], "X(NI)": 0.7}, "n_samples": 5, "threshold": 0.5,
        "output_prefix": "unc_np"})
    assert "NP(LIQUID) crosses 0.5" in text
    js = json.loads(Path(tb.out / "unc_np_summary.json").read_text())
    assert js["n_samples_ok"] == 5
    c = js["threshold_crossings"]["NP(LIQUID)"]["crossing_T"]
    assert c["n"] == 5 and 1550 < c["median"] < 1850
    band = Path(tb.out / "unc_np_band.csv").read_text().splitlines()
    assert len(band) == 1 + 31


def test_guards(tb, espei_run):
    text, _, err = tb.call("propagate_uncertainty", {"trace": str(espei_run), "kind": "binary_diagram",
                                                      "elements": ["CR", "NI"]})
    assert err and "tmin" in text
    text, _, err = tb.call("propagate_uncertainty", {"trace": str(espei_run), "kind": "property",
                                                      "elements": ["CR", "NI"], "quantity": "GM",
                                                      "conditions": {"T": 1500, "X(NI)": 0.5}})
    assert err and "one varying condition" in text
    text, _, err = tb.call("analyze_mcmc", {"trace": str(espei_run), "burn_in": "999"})
    assert err and "burn_in" in text
