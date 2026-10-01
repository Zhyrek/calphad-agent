"""Tests for the TDB sanity checker (calphad_tools.tdb_checks / check_tdb tool)."""
import json
from pathlib import Path

import pytest

from calphad_tools.tdb_checks import run_checks

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"
BASE = TDBS / "CrFe-82Her.tdb"          # clean SGTE-based Cr-Fe: every broken variant is made from this one


def _run(path, tmp_path, **kw):
    return run_checks(str(path), output_prefix=str(tmp_path / "chk" / Path(path).stem), **kw)


def _broken(tmp_path, name, *edits):
    """Copy BASE with exact text replacements (each must match)."""
    text = BASE.read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new, 1)
    p = tmp_path / f"{name}.tdb"
    p.write_text(text)
    return p


def _find(res, code, severity=None, responsible=None):
    hits = [f for f in res["findings"] if f.code == code and (severity is None or f.severity == severity)
            and (responsible is None or responsible in f.responsible)]
    assert hits, (f"no {severity} {code} naming {responsible}:\n" + res["report"])
    return hits[0]


# ---------------------------------------------------------------------------
# clean databases
# ---------------------------------------------------------------------------
KNOWN_TRUE_POSITIVES = {
    # GHSERVZ (metastable V for fcc/hcp) has its middle range running to 4000 K instead of 2183 K -> G jumps
    "CoCrFeNiV_Choi2019.tdb": {("breakpoint_jump", "FUNCTION GHSERVZ")},
}


@pytest.mark.parametrize("tdb", sorted(p.name for p in TDBS.glob("*.tdb")))
def test_clean_examples_have_no_errors(tdb, tmp_path):
    res = _run(TDBS / tdb, tmp_path)
    errors = {(f.code, f.responsible[0] if f.responsible else "") for f in res["findings"] if f.severity == "ERROR"}
    assert errors == KNOWN_TRUE_POSITIVES.get(tdb, set()), res["report"]
    assert res["meta"]["seconds"] < 30
    mps = res["meta"]["melting_points"]
    expected = {"CR": 2180.0, "NI": 1728.0, "FE": 1811.0, "CO": 1768.0, "AL": 933.47, "V": 2183.0}
    for el, v in mps.items():
        if el in expected and v["absolute"]:
            assert abs(v["Tm"] - expected[el]) < 5, (el, v)
    # output files
    js = json.loads(Path(res["json"]).read_text())
    assert js["meta"]["counts"]["ERROR"] == len(errors) and all("fix" in f for f in js["findings"])
    assert len(res["images"]) == 4 and all(Path(i).stat().st_size > 10000 for i in res["images"])


def test_clean_melting_points_and_non_ser(tmp_path):
    r = _run(TDBS / "CrNi-06Hal.tdb", tmp_path)
    mp = r["meta"]["melting_points"]
    assert abs(mp["CR"]["Tm"] - 2180) < 5 and abs(mp["NI"]["Tm"] - 1728) < 5
    r = _run(TDBS / "FeNi-00Ans-2SL.tdb", tmp_path)
    assert abs(r["meta"]["melting_points"]["FE"]["Tm"] - 1811) < 5
    # Kaufman lattice stabilities (bcc Cr = 0): recognised as non-SER, so no Cp / SGTE-Tm complaints
    r = _run(TDBS / "Cr-Fe_her.tdb", tmp_path)
    f = _find(r, "non_ser_reference", "INFO", "G(BCC_A2,CR;0)")
    assert f.evidence["S298"] < 1
    assert not [x for x in r["findings"] if x.code in ("negative_cp", "tm_mismatch", "liquid_negative_cp")]


# ---------------------------------------------------------------------------
# broken databases
# ---------------------------------------------------------------------------
def test_undefined_function(tmp_path):
    p = _broken(tmp_path, "undef", ("+GLIQCR;,,", "+GLIQCRX;,,"))
    r = _run(p, tmp_path, checks=["syntax", "load"])
    f = _find(r, "undefined_function", "ERROR", "G(LIQUID,CR)")
    assert f.responsible[0] == "GLIQCRX"
    _find(r, "unused_function", "INFO", "GLIQCR")


def test_range_gap_and_overlap(tmp_path):
    p = _broken(tmp_path, "gap",
                ("PAR  G(FCC_A1,CR:VA),,                 +GHSERCR+10460+0.628*T;,,     N 82Her !",
                 "PAR  G(FCC_A1,CR:VA) 1000 +GHSERCR+10460+0.628*T; 6000 N 82Her !"),
                ("      1811.00  Y  -3957.195+5.249009*T+GHSERFE+4.9251E+30*T**(-9);\n      6000.00  N !",
                 "      1811.00  Y  -3957.195+5.249009*T+GHSERFE+4.9251E+30*T**(-9);\n"
                 "      1500.00  Y  -3957.195+5.249009*T+GHSERFE+4.9251E+30*T**(-9);\n      6000.00  N !"))
    r = _run(p, tmp_path, checks=["ranges"])
    f = _find(r, "range_gap_low", "WARNING", "G(FCC_A1,CR:VA)")
    assert f.evidence["limits"]["G(FCC_A1,CR:VA)"] == 1000
    f = _find(r, "range_order", "ERROR", "FUNCTION GHCPFE")
    assert f.evidence["low"] == 1811 and f.evidence["high"] == 1500
    # upper limit below tmax -> extrapolation warning
    r = _run(BASE, tmp_path, checks=["ranges"], tmax=6500)
    _find(r, "range_extrapolated_high", "WARNING", "FUNCTION GHSERCR")


def test_discontinuous_breakpoints(tmp_path):
    p = _broken(tmp_path, "jump", ("1811.00  Y  -10838.83+291.302*T-46*T*LN(T);",       # +500 J/mol jump
                                   "1811.00  Y  -10338.83+291.302*T-46*T*LN(T);"),
                ("-34869.344+344.18*T-50*T*LN(T)", "-35305.344+344.38*T-50*T*LN(T)"))  # dS jump only
    r = _run(p, tmp_path, checks=["breakpoints"])
    f = _find(r, "breakpoint_jump", "ERROR", "FUNCTION GLIQFE")
    assert f.evidence["T"] == 1811 and abs(f.evidence["dG_J"] - 500) < 1
    f = _find(r, "breakpoint_slope", "WARNING", "FUNCTION GHSERCR")
    assert abs(f.evidence["dS_J_K"] + 0.2) < 1e-3 and abs(f.evidence["dG_J"]) < 1
    assert not _run(BASE, tmp_path, checks=["breakpoints"])["findings"]


def test_huge_liquid_interaction_gap(tmp_path):
    p = _broken(tmp_path, "gap_liq", ("+17638-11.375*T;,,", "+217638-11.375*T;,,"))
    r = _run(p, tmp_path, checks=["interactions", "binary"])
    f = _find(r, "liquid_miscibility_gap", "ERROR", "L(LIQUID,CR,FE;0)")
    assert f.evidence["T_gap_max"] > 5900 and f.evidence["x_min"] < 0.2 < 0.8 < f.evidence["x_max"]
    _find(r, "large_interaction", "WARNING", "L(LIQUID,CR,FE;0)")
    # inverted gap: L0 = -20000 + 25*T exceeds 2RT above ~2400 K
    p = _broken(tmp_path, "gap_inv", ("+17638-11.375*T;,,", "-20000+25*T;,,"))
    r = _run(p, tmp_path, checks=["binary"])
    f = _find(r, "liquid_miscibility_gap", "ERROR", "L(LIQUID,CR,FE;0)")
    assert f.evidence["inverted"] and 2000 < f.evidence["T_gap_min"] < 2900


def test_solid_stabilised_over_liquid(tmp_path):
    # hcp-Cr with a too negative T coefficient: fine below Tm, but beats the liquid again above ~3000 K
    p = _broken(tmp_path, "hcp", ("+GHSERCR+4438;,,", "+GHSERCR+28000-12*T;,,"))
    r = _run(p, tmp_path, checks=["unary", "binary"])
    f = _find(r, "inverse_melting", "ERROR", "G(HCP_A3,CR:VA)")
    assert 2500 < f.evidence["phases"]["HCP_A3"]["T_ranges"][0][0] < 3300
    assert abs(r["meta"]["melting_points"]["CR"]["Tm"] - 2180) < 5
    f = _find(r, "binary_solid_above_liquid", "ERROR", "G(HCP_A3,CR:VA)")
    assert "HCP_A3" in f.evidence["phases"] and f.evidence["phases"]["HCP_A3"]["x_min"] < 0.05
    # -20*T on the stable solid end-member: Cr never melts
    p = _broken(tmp_path, "bcc", ("PAR  G(BCC_A2,CR:VA),,                 +GHSERCR;,,",
                                  "PAR  G(BCC_A2,CR:VA),,                 +GHSERCR-20*T;,,"))
    r = _run(p, tmp_path, checks=["unary", "binary"])
    _find(r, "no_melting", "ERROR", "G(BCC_A2,CR:VA)")
    _find(r, "binary_solid_above_liquid", "ERROR", "G(BCC_A2,CR:VA)")


def test_parameters_without_phase(tmp_path):
    p = _broken(tmp_path, "nophase", (" PHASE HCP_A3 %A 2 1 0.5 !\n CONST HCP_A3 : CR FE : VA : !\n", ""))
    r = _run(p, tmp_path, checks=["syntax", "load"])
    f = _find(r, "param_undefined_phase", "ERROR", "G(HCP_A3,CR:VA)")
    assert f.evidence["phase"] == "HCP_A3" and "L(HCP_A3,CR,FE:VA;0)" in f.responsible
    # PHASE removed but CONSTITUENT left behind, plus a parameter with an undeclared species
    p = _broken(tmp_path, "nophase2", (" PHASE HCP_A3 %A 2 1 0.5 !\n", ""),
                ("PAR  L(FCC_A1,CR,FE:VA;0)", "PAR  L(FCC_A1,CR,FE:VA;0),, +1;,, N !\n PAR  L(FCC_A1,CR,MN:VA;0)"))
    r = _run(p, tmp_path, checks=["syntax"])
    _find(r, "constituent_undefined_phase", "ERROR")
    _find(r, "undeclared_species", "ERROR", "L(FCC_A1,CR,MN:VA;0)")


def test_negative_cp(tmp_path):
    # a large T**(-1) term makes Cp of bcc-Fe (the stable, reference phase) negative near room temperature
    p = _broken(tmp_path, "cp", ("PAR  G(BCC_A2,FE:VA),,                 +GHSERFE;,,",
                                 "PAR  G(BCC_A2,FE:VA),,                 +GHSERFE+2E6*T**(-1)-6700;,,"))
    r = _run(p, tmp_path, checks=["unary"])
    f = _find(r, "negative_cp", "ERROR", "G(BCC_A2,FE:VA)")
    assert f.evidence["phase"] == "BCC_A2" and f.evidence["extreme_Cp"] < -10 and f.evidence["stable"]
    assert f.evidence["T_ranges"][0][0] == pytest.approx(298.15)
    _find(r, "ser_offset", "WARNING", "G(BCC_A2,FE:VA)")


def test_duplicate_parameter_and_dis_part(tmp_path):
    p = _broken(tmp_path, "dup", ("PAR  L(FCC_A1,CR,FE:VA;0),,            +7100-9.375*T;,,              N 82Her !",
                                  "PAR  L(FCC_A1,CR,FE:VA;0),,            +7100-9.375*T;,,              N 82Her !\n"
                                  " PAR  L(FCC_A1,FE,CR:VA;0),, +5000;,, N !"))
    r = _run(p, tmp_path, checks=["syntax"])
    f = _find(r, "duplicate_parameter", "ERROR", "L(FCC_A1,FE,CR:VA;0)")
    assert len(f.evidence["values_1000K"]) == 2
    # order/disorder pair with inconsistent sublattices
    text = (TDBS / "CrNi-06Hal.tdb").read_text().replace(" PHASE B2_BCC %BO 3 0.5 0.5 3 !",
                                                         " PHASE B2_BCC %BO 3 0.5 0.6 3 !")
    q = tmp_path / "dis.tdb"
    q.write_text(text)
    r = _run(q, tmp_path, checks=["syntax"])
    _find(r, "dis_part_mismatch", "ERROR", "B2_BCC")


def test_tool_output_and_picklable_figures(tmp_path):
    from calphad_tools.agent.tools import UI, ToolBox
    from calphad_tools.figures import load_figure

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    tb = ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")
    assert "check_tdb" in tb.tools and len(tb.tools["check_tdb"].spec.description) < 800
    text, imgs, err = tb.call("check_tdb", {"tdb": "CrNi-06Hal.tdb", "output_prefix": "crni"})
    assert not err, text
    assert "ERROR 0" in text and "melting points: CR 2180" in text and "crni_findings.json" in text
    assert len(imgs) == 4
    for i in imgs:
        assert load_figure(i) is not None, i
    text, imgs, err = tb.call("check_tdb", {"tdb": "CrNi-06Hal.tdb", "checks": ["bogus"]})
    assert err and "unknown check" in text
