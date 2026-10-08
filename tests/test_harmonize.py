"""Phase harmonisation (calphad_tools.harmonize): safe piecewise scaling, formula-unit normalisation, extend_phase and
the 'extend' decision of combine_databases, on small synthetic Ce-Co-Fe databases:

* base C15_LAVES (CE)1(FE)2 vs add CET2 (CO,FE)2(CE)1 (sublattices swapped, names differ), with a multi-range
  Piecewise parameter written with FUNCTIONs;
* the magnetic variants: add magnetic on a sum-one basis with a non-magnetic base (the base is rescaled to the
  magnetic side's basis, exact, add's magnetic parameters untouched) and both magnetic (refuse / keep / refit);
* RE2TM17 / CE2T17 2:17 -> 2/19:17/19;
* combine_databases on three files with an 'extend' decision.

Run:  .venv/bin/python -m pytest -q tests/test_harmonize.py
"""
import math
import re
import warnings

import numpy as np
import pytest

from calphad_tools import harmonize as hz
from calphad_tools import tdbstage as ts
from calphad_tools.tdbparse import parse_tdb, parse_piecewise, scale_piecewise

HEAD = """ TEMPERATURE_LIMITS 298.15 6000 !
 ELEMENT /-   ELECTRON_GAS 0 0 0 !
 ELEMENT VA   VACUUM 0 0 0 !
"""
EL = {"CE": " ELEMENT CE   FCC_A1 140.12 0 0 !\n", "CO": " ELEMENT CO   HCP_A3 58.933 0 0 !\n",
      "FE": " ELEMENT FE   BCC_A2 55.847 0 0 !\n"}
FN = {"CE": " FUNCTION GHSERCE 298.15 -7160.519+84.23*T-22.39*T*LN(T); 1000 Y -14198.42+190.4*T-37.6*T*LN(T); "
            "3000 N !\n",
      "FE": " FUNCTION GHSERFE 298.15 1225.7+124.134*T-23.5143*T*LN(T); 1811 Y -25383.581+299.31255*T-46*T*LN(T); "
            "3000 N !\n",
      "CO": " FUNCTION GHSERCO 298.15 310.241+133.36601*T-25.0861*T*LN(T); 1768 Y -17197.666+253.28374*T-40.5*T*LN(T); "
            "3000 N !\n"}
G0 = {"LIQUID": {"CE": "GHSERCE+5000-5*T", "CO": "GHSERCO+16000-9*T", "FE": "GHSERFE+12000-6.5*T"},
      "FCC_A1": {"CE": "GHSERCE", "CO": "GHSERCO+430-0.6*T", "FE": "GHSERFE+1500-1.5*T"},
      "BCC_A2": {"CE": "GHSERCE+3000", "CO": "GHSERCO+2900-1*T", "FE": "GHSERFE"},
      "HCP_A3": {"CE": "GHSERCE+2000", "CO": "GHSERCO", "FE": "GHSERFE+5000"}}
LMIX = {("CE", "FE"): "-1000", ("CE", "CO"): "-50000", ("CO", "FE"): "-5000"}


def solutions(els, phases=("LIQUID", "FCC_A1", "BCC_A2")):
    s = ""
    for ph in phases:
        s += f" PHASE {ph} % 1 1 !\n CONSTITUENT {ph} :{','.join(els)}: !\n"
        for e in els:
            s += f" PARAMETER G({ph},{e};0) 298.15 {G0[ph][e]}; 3000 N !\n"
        for i, a in enumerate(els):
            for b in els[i + 1:]:
                s += f" PARAMETER L({ph},{a},{b};0) 298.15 {LMIX[(a, b)]}; 3000 N !\n"
    return s


def tdb(els, body, typedefs="", solution_phases=("LIQUID", "FCC_A1", "BCC_A2"), extra_fn=""):
    els = sorted(els)
    return (HEAD + "".join(EL[e] for e in els) + "".join(FN[e] for e in els) + extra_fn
            + " TYPE_DEFINITION % SEQ * !\n" + typedefs + solutions(els, solution_phases) + body)


# base: C15_LAVES (CE)1(FE)2, its G a two-range piecewise with FUNCTIONs
BASE_C15 = """ PHASE C15_LAVES % 2 1 2 !
 CONSTITUENT C15_LAVES :CE:FE: !
 PARAMETER G(C15_LAVES,CE:FE;0) 298.15 GHSERCE+2*GHSERFE-60000+12*T; 1000 Y GHSERCE+2*GHSERFE-61000+13*T; 3000 N !
"""
# add: CET2 (CO,FE)2(CE)1 - sublattices in the other order, Co-bearing end-member written with a 2-range FUNCTION
ADD_FN = " FUNCTION GCOCE 298.15 -45000+10*T; 900 Y -46000+11*T; 3000 N !\n"
ADD_CET2 = """ PHASE CET2 % 2 2 1 !
 CONSTITUENT CET2 :CO,FE:CE: !
 PARAMETER G(CET2,CO:CE;0) 298.15 2*GHSERCO+GHSERCE+GCOCE; 3000 N !
 PARAMETER G(CET2,FE:CE;0) 298.15 2*GHSERFE+GHSERCE-58000+12*T; 3000 N !
 PARAMETER L(CET2,CO,FE:CE;0) 298.15 -4000+2*T; 3000 N !
 PARAMETER L(CET2,CO,FE:CE;1) 298.15 1500; 3000 N !
"""
# add, magnetic, on one atom per formula unit (2/3 : 1/3)
MAG_TD = " TYPE_DEFINITION M GES A_P_D CET2 MAGNETIC -3.0 0.28 !\n"
ADD_CET2_MAG = """ PHASE CET2 %M 2 0.666666666666667 0.333333333333333 !
 CONSTITUENT CET2 :CO,FE:CE: !
 PARAMETER G(CET2,CO:CE;0) 298.15 (2*GHSERCO+GHSERCE+GCOCE)/3; 3000 N !
 PARAMETER TC(CET2,CO:CE;0) 298.15 900; 3000 N !
 PARAMETER BMAGN(CET2,CO:CE;0) 298.15 1.2; 3000 N !
 PARAMETER TC(CET2,CO,FE:CE;0) 298.15 -200; 3000 N !
 PARAMETER BMAGN(CET2,CO,FE:CE;0) 298.15 0.3; 3000 N !
 PARAMETER G(CET2,FE:CE;0) 298.15 (2*GHSERFE+GHSERCE-58000+12*T)/3; 3000 N !
 PARAMETER TC(CET2,FE:CE;0) 298.15 600; 3000 N !
 PARAMETER BMAGN(CET2,FE:CE;0) 298.15 0.9; 3000 N !
 PARAMETER L(CET2,CO,FE:CE;0) 298.15 -1333+0.6*T; 3000 N !
"""
BASE_C15_MAG = """ TYPE_DEFINITION B GES A_P_D C15_LAVES MAGNETIC -3.0 0.28 !
 PHASE C15_LAVES %B 2 1 2 !
 CONSTITUENT C15_LAVES :CE:FE: !
 PARAMETER G(C15_LAVES,CE:FE;0) 298.15 GHSERCE+2*GHSERFE-60000+12*T; 3000 N !
 PARAMETER TC(C15_LAVES,CE:FE;0) 298.15 550; 3000 N !
 PARAMETER BMAGN(C15_LAVES,CE:FE;0) 298.15 2.5; 3000 N !
"""


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("harm")
    f = {"base": tdb(["CE", "FE"], BASE_C15), "add": tdb(["CE", "CO", "FE"], ADD_CET2, extra_fn=ADD_FN),
         "addmag": tdb(["CE", "CO", "FE"], ADD_CET2_MAG, MAG_TD, extra_fn=ADD_FN),
         "basemag": tdb(["CE", "FE"], BASE_C15_MAG)}
    out = {}
    for k, v in f.items():
        out[k] = str(d / f"{k}.tdb")
        (d / f"{k}.tdb").write_text(v)
    out["dir"] = d
    return out


def _db(path):
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(path))


def _params(path, phase):
    return {p.key.as_tuple(): p for p in parse_tdb(str(path)).parameters if p.key.phase == phase}


# ================================================================================================ 1. safe scaling
def test_scale_piecewise_each_range_and_pycalphad_reads_it(tmp_path):
    expr = "298.15 GHSERCE+2*GHSERFE-60000+12*T; 1000 Y +GHSERCE+2*GHSERFE-61000+13*T; 2000 Y -5000; 3000 N REF1"
    pw = parse_piecewise(expr)
    for k in (1 / 3, 19.0, 2.5, -1.0, 1 / 19):
        s = scale_piecewise(pw, k)
        assert [(lo, hi) for lo, hi, _ in s.ranges] == [(lo, hi) for lo, hi, _ in pw.ranges]   # same breakpoints
        assert s.reference == "REF1"
        assert "Piecewise" not in s.to_tdb() and s.to_tdb().count(";") == 3
        text = tdb(["CE", "FE"], f" PHASE X % 1 1 !\n CONSTITUENT X :CE: !\n PARAMETER G(X,CE;0) {s.to_tdb()} !\n"
                                 f" PHASE Y % 1 1 !\n CONSTITUENT Y :CE: !\n PARAMETER G(Y,CE;0) {pw.to_tdb()} !\n")
        p = tmp_path / "s.tdb"
        p.write_text(text)
        db = _db(p)                                   # pycalphad parses the scaled text
        from pycalphad import Model, variables as v
        mx, my = Model(db, ["CE", "VA"], "X"), Model(db, ["CE", "VA"], "Y")
        for T in (400.0, 999.0, 1500.0, 2500.0):      # every range
            sub = {v.T: T, v.P: 1e5}
            gx = float(mx.GM.xreplace({**sub, mx.site_fractions[0]: 1.0}))
            gy = float(my.GM.xreplace({**sub, my.site_fractions[0]: 1.0}))
            assert gx == pytest.approx(k * gy, rel=1e-12), (k, T)
    # tdbstage's G/L rescale goes through the helper
    assert ts._scale_pw(pw, 0.5).ranges == scale_piecewise(pw, 0.5).ranges
    assert scale_piecewise(pw, 1 / 3).ranges[2][2] == repr(-5000 / 3)        # plain numbers are multiplied out
    assert scale_piecewise(pw, 1 / 3).ranges[0][2] == "(GHSERCE+2*GHSERFE-60000+12*T)/3"


# ================================================================================================ 2. normalisation
RE217 = """ PHASE RE2TM17 % 2 2 17 !
 CONSTITUENT RE2TM17 :CE:CO,FE: !
 PARAMETER G(RE2TM17,CE:CO;0) 298.15 2*GHSERCE+17*GHSERCO-150000+20*T; 1200 Y 2*GHSERCE+17*GHSERCO-152000+21.5*T; 3000 N !
 PARAMETER G(RE2TM17,CE:FE;0) 298.15 2*GHSERCE+17*GHSERFE-90000+10*T; 3000 N !
 PARAMETER L(RE2TM17,CE:CO,FE;0) 298.15 -30000; 3000 N !
 PHASE CE2T17 %M 2 2 17 !
 CONSTITUENT CE2T17 :CE:CO,FE: !
 PARAMETER G(CE2T17,CE:CO;0) 298.15 2*GHSERCE+17*GHSERCO-150000+20*T; 3000 N !
 PARAMETER G(CE2T17,CE:FE;0) 298.15 2*GHSERCE+17*GHSERFE-90000+10*T; 3000 N !
 PARAMETER L(CE2T17,CE:CO,FE;0) 298.15 -30000; 3000 N !
 PARAMETER TC(CE2T17,CE:CO;0) 298.15 1100; 3000 N !
 PARAMETER BMAGN(CE2T17,CE:CO;0) 298.15 25; 3000 N !
 PARAMETER TC(CE2T17,CE:FE;0) 298.15 300; 3000 N !
 PARAMETER BMAGN(CE2T17,CE:FE;0) 298.15 30; 3000 N !
 PARAMETER BMAGN(CE2T17,CE:CO,FE;0) 298.15 10; 3000 N !
"""


def test_normalize_2_17_to_sum_one(tmp_path):
    src = tmp_path / "re217.tdb"
    src.write_text(tdb(["CE", "CO", "FE"], RE217, " TYPE_DEFINITION M GES A_P_D CE2T17 MAGNETIC -3.0 0.28 !\n"))
    out = tmp_path / "one.tdb"
    r = hz.normalize_phase_formula_unit(str(src), ["RE2TM17"], "sum_to_one", str(out))
    rep = r["phases"][0]
    assert rep["k"] == pytest.approx(1 / 19) and rep["k_text"] == "1/19"
    assert " PHASE RE2TM17 % 2 0.105263157894737 0.894736842105263 !" in out.read_text()
    assert sum(_db(out).phases["RE2TM17"].sublattices) == pytest.approx(1.0, abs=1e-15)
    assert rep["pycalphad_sum"] == pytest.approx(1.0, abs=1e-15)
    assert rep["gm"]["max_rel"] <= 1e-9 and rep["gm"]["n"] >= 400            # GM per atom unchanged
    assert parse_tdb(str(out)).phases["CE2T17"].site_ratios == [2.0, 17.0]    # other phases untouched
    assert r["validator"] is None or r["validator"]["verdict"] == "PASS"
    print("2:17 -> sum one:", rep["text"])
    # back to integers
    r2 = hz.normalize_phase_formula_unit(str(out), ["RE2TM17"], "integers", str(tmp_path / "int.tdb"))
    assert r2["phases"][0]["new"] == [2.0, 17.0] and r2["phases"][0]["gm"]["max_rel"] <= 1e-9
    # the magnetic 2:17 phase: refused with the size of the error the other options give
    with pytest.raises(hz.MagneticBasisError) as e:
        hz.normalize_phase_formula_unit(str(src), ["CE2T17"], "sum_to_one", str(tmp_path / "x.tdb"))
    msg = str(e.value)
    keep = float(re.search(r"change GM by up to ([-0-9.e+]+) J/mol", msg).group(1))
    refit = float(re.search(r"refit leaves ([-0-9.e+]+) J/mol", msg).group(1))
    assert "not linear in BMAGN" in msg and keep > 1000 and refit < keep / 100
    rk = hz.normalize_phase_formula_unit(str(src), ["CE2T17"], "sum_to_one", str(tmp_path / "k.tdb"),
                                         magnetic="keep_parameters")["phases"][0]
    rf = hz.normalize_phase_formula_unit(str(src), ["CE2T17"], "sum_to_one", str(tmp_path / "r.tdb"),
                                         magnetic="refit")["phases"][0]
    assert rk["gm"]["max_abs"] == pytest.approx(keep, rel=5e-3)          # (the message has 3 digits)
    assert rf["gm"]["max_abs"] == pytest.approx(refit, rel=5e-3) and rf["gm"]["max_abs"] < 1.0
    # end-members transformed exactly: beta' = (1+beta)**k - 1 (beta > 0), k = 1/19
    bm = _params(tmp_path / "r.tdb", "CE2T17")[("BMAGN", "CE2T17", (("CE",), ("CO",)), 0, "")]
    assert float(bm.expr.ranges[0][2]) == pytest.approx(26 ** (1 / 19) - 1, rel=1e-11)
    print(f"2:17 magnetic: keep_parameters max |dG| {keep:.4g} J/mol of atoms, refit {refit:.4g}")
    # match another database's phase (the formula unit of the other side)
    r3 = hz.normalize_phase_formula_unit(str(src), ["RE2TM17"], output_tdb=str(tmp_path / "m.tdb"),
                                         match_tdb=str(out))
    assert r3["phases"][0]["new"] == parse_tdb(str(out)).phases["RE2TM17"].site_ratios


def test_sum_to_one_decimals():
    for r in ([2, 17], [1, 2], [1, 1, 1], [3, 5, 7], [0.6667, 0.3333]):
        s = hz.sum_to_one_ratios(r)
        assert abs(sum(s) - 1.0) <= 1e-15
        assert all(len(f"{x:.15g}".replace("0.", "")) <= 16 for x in s)
    assert hz.sum_to_one_ratios([2, 17]) == [0.105263157894737, 0.894736842105263]


# ================================================================================================ 3. extend_phase
def test_extend_nonmagnetic(files, tmp_path):
    out = tmp_path / "ext.tdb"
    r = hz.extend_phase(files["base"], files["add"], "C15_LAVES", "CET2", ["CO"], output_tdb=str(out))
    ph = parse_tdb(str(out)).phases["C15_LAVES"]
    assert ph.constituents == [["CE"], ["FE", "CO"]] and ph.site_ratios == [1.0, 2.0]       # (CE)1(CO,FE)2
    assert r["new_from_old"] == [1, 0]
    # only Co-bearing parameters imported; the base's Ce-Fe parameter untouched
    assert sorted(r["imported"]) == ["G(C15_LAVES,CE:CO;0)", "L(C15_LAVES,CE:CO,FE;0)", "L(C15_LAVES,CE:CO,FE;1)"]
    pb, po = _params(files["base"], "C15_LAVES"), _params(out, "C15_LAVES")
    key = ("G", "C15_LAVES", (("CE",), ("FE",)), 0, "")
    assert po[key].expr.ranges == pb[key].expr.ranges
    assert {"GHSERCO", "GCOCE"} <= set(parse_tdb(str(out)).functions)
    ch = r["checks"]
    assert ch["base"]["max_rel"] <= 1e-9 and ch["base"]["n"] >= 400                     # base GM unchanged
    by = {c["system"]: c for c in ch["add"]}
    assert by["Ce-Co"]["max_rel"] <= 1e-9 and by["Ce-Co"]["expected_exact"]              # add reproduced
    assert not by["Ce-Co-Fe"]["expected_exact"] and "G(C15_LAVES,CE:FE;0)" in by["Ce-Co-Fe"]["why_not_exact"][0]
    assert r["load"]["loaded"] and not r["load"]["errors"]
    assert r["validator"] is None or r["validator"]["verdict"] == "PASS"
    assert r["missing_g"] == []
    print("non-magnetic: base", hz.gm_text(ch["base"]), "| Ce-Co", hz.gm_text(by["Ce-Co"]))


def test_extend_base_nonmagnetic_add_magnetic(files, tmp_path):
    """The magnetic side sets the basis: the base's non-magnetic C15 (1:2, 3 sites) is rescaled to add's magnetic
    sum-one basis (k = 1/3, exact); add's magnetic parameters come over bit for bit; nothing of add is overwritten."""
    out = tmp_path / "extmag.tdb"
    r = hz.extend_phase(files["base"], files["addmag"], "C15_LAVES", "CET2", output_tdb=str(out))
    bc = r["basis"]
    assert bc["basis"] == "add" and bc["exact"] and bc["k_base"] == pytest.approx(1 / 3) and bc["k_add"] == 1.0
    assert "base's non-magnetic model rescaled to add's magnetic basis (sum 1), k = 1/3, exact" in bc["text"]
    new = parse_tdb(str(out))
    assert new.phases["C15_LAVES"].site_ratios == [0.333333333333333, 0.666666666666667]   # exactly add's (mapped)
    assert sum(_db(out).phases["C15_LAVES"].sublattices) == 1.0
    # add's magnetic (and G) parameters untouched: same expression text, same type definition factors
    add = _params(files["addmag"], "CET2")
    for p in new.parameters:
        if p.key.phase == "C15_LAVES" and "CO" in str(p.key.constituents):
            src = [q for q in add.values() if q.key.ptype == p.key.ptype and q.key.order == p.key.order
                   and sorted(map(tuple, q.key.constituents)) == sorted(map(tuple, p.key.constituents))]
            assert src and src[0].expr.ranges == p.expr.ranges, p.raw_name
    assert {"TC", "BMAGN"} <= {p.key.ptype for p in new.parameters if p.key.phase == "C15_LAVES"}
    assert hz.magnetic_factors(new, "C15_LAVES") == (-3.0, 0.28)
    # the base's own parameter is rescaled (G x 1/3), and its GM per atom is unchanged
    g = _params(out, "C15_LAVES")[("G", "C15_LAVES", (("CE",), ("FE",)), 0, "")]
    assert g.expr.ranges[0][2] == "(GHSERCE+2*GHSERFE-60000+12*T)/3"
    ch = r["checks"]
    assert ch["base"]["max_rel"] <= 1e-9
    by = {c["system"]: c for c in ch["add"]}
    assert by["Ce-Co"]["max_rel"] <= 1e-9          # magnetic contribution included: exact
    assert r["validator"] is None or r["validator"]["verdict"] == "PASS"
    print("base non-magnetic / add magnetic:", bc["text"], "| base", hz.gm_text(ch["base"]), "| Ce-Co",
          hz.gm_text(by["Ce-Co"]))


def test_extend_both_magnetic(files, tmp_path):
    with pytest.raises(hz.MagneticBasisError) as e:
        hz.extend_phase(files["basemag"], files["addmag"], "C15_LAVES", "CET2", output_tdb=str(tmp_path / "x.tdb"))
    assert "k = 3" in str(e.value) and "keep_parameters' would change GM by up to" in str(e.value)
    res = {}
    for mode in ("keep_parameters", "refit"):
        r = hz.extend_phase(files["basemag"], files["addmag"], "C15_LAVES", "CET2", magnetic=mode,
                            output_tdb=str(tmp_path / f"{mode}.tdb"))
        assert not r["basis"]["exact"] and r["checks"]["base"]["max_rel"] <= 1e-9     # base untouched
        res[mode] = {c["system"]: c for c in r["checks"]["add"]}["Ce-Co"]["max_abs"]
        assert r["load"]["loaded"] and not r["load"]["errors"]
    assert res["keep_parameters"] > 100 and res["refit"] < 1e-6 * res["keep_parameters"] + 1e-6
    print(f"both magnetic, Ce-Co max |dG|: keep_parameters {res['keep_parameters']:.4g}, refit {res['refit']:.3g} "
          f"J/mol of atoms")


def test_collisions_and_missing_endmembers(files, tmp_path):
    base_co = tdb(["CE", "CO", "FE"], """ PHASE C15_LAVES % 2 1 2 !
 CONSTITUENT C15_LAVES :CE,FE:CO,FE: !
 PARAMETER G(C15_LAVES,CE:FE;0) 298.15 GHSERCE+2*GHSERFE-60000+12*T; 3000 N !
 PARAMETER G(C15_LAVES,CE:CO;0) 298.15 GHSERCE+2*GHSERCO-70000; 3000 N !
 PARAMETER G(C15_LAVES,FE:FE;0) 298.15 3*GHSERFE+15000; 3000 N !
""")
    p = tmp_path / "base_co.tdb"
    p.write_text(base_co)
    key = ("G", "C15_LAVES", (("CE",), ("CO",)), 0, "")
    rb = hz.extend_phase(str(p), files["add"], "C15_LAVES", "CET2", ["CO"], output_tdb=str(tmp_path / "b.tdb"),
                         verify=False)
    assert [c["kept"] for c in rb["collisions"]] == ["base"]
    assert _params(tmp_path / "b.tdb", "C15_LAVES")[key].expr.ranges == _params(p, "C15_LAVES")[key].expr.ranges
    ra = hz.extend_phase(str(p), files["add"], "C15_LAVES", "CET2", ["CO"], collision="add",
                         output_tdb=str(tmp_path / "a.tdb"), verify=False)
    assert [c["kept"] for c in ra["collisions"]] == ["add"]
    assert "GCOCE" in _params(tmp_path / "a.tdb", "C15_LAVES")[key].expr.ranges[0][2]
    with pytest.raises(hz.HarmonizeError, match="collision"):
        hz.extend_phase(str(p), files["add"], "C15_LAVES", "CET2", ["CO"], collision="error", verify=False)
    # a base with FE on the Ce sublattice (antisite) and no Co -> FE:CO is a new end-member nobody gives G for
    p2 = tmp_path / "base_anti.tdb"
    p2.write_text(tdb(["CE", "FE"], """ PHASE C15_LAVES % 2 1 2 !
 CONSTITUENT C15_LAVES :CE,FE:FE: !
 PARAMETER G(C15_LAVES,CE:FE;0) 298.15 GHSERCE+2*GHSERFE-60000+12*T; 3000 N !
 PARAMETER G(C15_LAVES,FE:FE;0) 298.15 3*GHSERFE+15000; 3000 N !
"""))
    rm = hz.extend_phase(str(p2), files["add"], "C15_LAVES", "CET2", verify=False)
    miss = {m["end_member"]: m for m in rm["missing_g"]}
    assert set(miss) == {"FE:CO"} and miss["FE:CO"]["matters"] and "MATTERS" in miss["FE:CO"]["text"]
    assert "FE:CO" in [m["end_member"] for m in rb["missing_g"]] and not rb["missing_g"][0]["matters"]


def test_odd_order_interaction_in_reverse_order(tmp_path):
    """L(CET2,FE,CO:CE;1) written against the alphabet: pycalphad sorts the constituents WITHOUT flipping the sign.
    The extension writes it in sorted order with the value pycalphad uses, so pycalphad's GM of add is reproduced
    (here on the whole Ce-Co-Fe system: base and add agree on Ce-Fe)."""
    add = tdb(["CE", "CO", "FE"], """ PHASE CET2 % 2 2 1 !
 CONSTITUENT CET2 :CO,FE:CE: !
 PARAMETER G(CET2,CO:CE;0) 298.15 2*GHSERCO+GHSERCE-70000; 3000 N !
 PARAMETER G(CET2,FE:CE;0) 298.15 2*GHSERFE+GHSERCE-60000+12*T; 3000 N !
 PARAMETER L(CET2,FE,CO:CE;1) 298.15 -9000; 3000 N !
 PARAMETER L(CET2,FE,CO:CE;0) 298.15 -4000; 3000 N !
""")
    base = tdb(["CE", "FE"], """ PHASE C15_LAVES % 2 1 2 !
 CONSTITUENT C15_LAVES :CE:FE: !
 PARAMETER G(C15_LAVES,CE:FE;0) 298.15 GHSERCE+2*GHSERFE-60000+12*T; 3000 N !
""")
    (tmp_path / "a.tdb").write_text(add)
    (tmp_path / "b.tdb").write_text(base)
    r = hz.extend_phase(str(tmp_path / "b.tdb"), str(tmp_path / "a.tdb"), "C15_LAVES", "CET2",
                        output_tdb=str(tmp_path / "o.tdb"))
    raws = [p.raw_name for p in parse_tdb(str(tmp_path / "o.tdb")).parameters if p.key.phase == "C15_LAVES"]
    assert "L(C15_LAVES,CE:CO,FE;1)" in raws                          # sorted, value as pycalphad reads it
    assert parse_tdb(str(tmp_path / "o.tdb")).parameters[-1].expr.ranges[0][2] in ("-9000", "-4000")
    assert any("Thermo-Calc reads the original with the opposite sign" in n for n in r["notes"])
    by = {c["system"]: c for c in r["checks"]["add"]}
    assert by["Ce-Co-Fe"]["expected_exact"] and by["Ce-Co-Fe"]["max_rel"] <= 1e-9
    assert r["validator"] is None or r["validator"]["verdict"] == "PASS"


def test_extension_spec_yaml_shape():
    A = parse_tdb(tdb(["CE", "FE"], BASE_C15), is_text=True)
    B = parse_tdb(tdb(["CE", "CO", "FE"], ADD_CET2, extra_fn=ADD_FN), is_text=True)
    spec = hz.extension_spec("C15_LAVES", {
        "source_phase": "CET2", "target_phase": "C15_LAVES", "structure_from": "base", "sublattice_map": [1, 0],
        "add_constituents": {0: ["CO"]}, "import_parameters": {"require_components": ["CO"]},
        "site_ratio_normalization": "sum_to_one", "scale_parameters_with_formula_unit": True,
        "magnetic_model": "source"}, A, B)
    assert spec["phase"] == "CET2" and spec["components"] == ["CO"] and spec["normalization"] == "per_atom"
    assert spec["magnetic"] == "auto" and spec["add_constituents"] == {0: ["CO"]}
    new, rep = hz.extend_phase_db(A, B, "C15_LAVES", spec["phase"], spec["components"], spec["sublattice_map"],
                                  spec["normalization"], add_constituents=spec["add_constituents"])
    assert new.phases["C15_LAVES"].site_ratios == [0.333333333333333, 0.666666666666667]   # per atom, exact
    assert rep["added_constituents"] == {1: ["CO"]}
    for bad in ({"structure_from": "add"}, {"scale_parameters_with_formula_unit": False}, {"bogus": 1}):
        with pytest.raises(hz.HarmonizeError):
            hz.extension_spec("C15_LAVES", {"extend": {"phase": "CET2", **bad}}, A, B)
    assert hz.is_extend("extend") and hz.is_extend({"extend": {}}) and not hz.is_extend("db1")
    cands = hz.harmonization_candidates(A, B)
    assert [(c["a_phase"], c["b_phase"], c["components"]) for c in cands] == [("C15_LAVES", "CET2", ["CO"])]


# ================================================================================================ 4. combine
CEFE = tdb(["CE", "FE"], BASE_C15)
COFE = tdb(["CO", "FE"], "", solution_phases=("LIQUID", "FCC_A1", "BCC_A2", "HCP_A3"))
CECO = tdb(["CE", "CO"], """ PHASE CET2 %M 2 0.666666666666667 0.333333333333333 !
 CONSTITUENT CET2 :CO:CE: !
 PARAMETER G(CET2,CO:CE;0) 298.15 (2*GHSERCO+GHSERCE-75000+15*T)/3; 3000 N !
 PARAMETER TC(CET2,CO:CE;0) 298.15 900; 3000 N !
 PARAMETER BMAGN(CET2,CO:CE;0) 298.15 1.2; 3000 N !
""", " TYPE_DEFINITION M GES A_P_D CET2 MAGNETIC -3.0 0.28 !\n", ("LIQUID", "FCC_A1", "HCP_A3"))


@pytest.fixture(scope="module")
def trio_ext(tmp_path_factory):
    from calphad_tools.combine import combine_databases
    d = tmp_path_factory.mktemp("combine_ext")
    srcs = []
    for name, text in (("cefe", CEFE), ("cofe", COFE), ("ceco", CECO)):
        (d / f"{name}.tdb").write_text(text)
        srcs.append(str(d / f"{name}.tdb"))
    kw = dict(order=srcs, unary50="off", n_workers=3, lint=False, tmax_c=1800, composition_step=0.04,
              temperature_step=25)
    r1 = combine_databases(srcs, str(d / "c"), **kw)
    r2 = combine_databases(srcs, str(d / "c"), decisions={"step2:phase:C15_LAVES": {"extend": {"phase": "CET2"}}},
                           decision_reasons={"step2:phase:C15_LAVES": "user: keep the base Laves, add Co"}, **kw)
    return {"plan": r1, "run": r2, "srcs": srcs, "dir": d, "kw": kw}


def test_combine_plan_suggests_extend(trio_ext):
    r = trio_ext["plan"]
    assert not r["written"] and [n.id for n in r["plan"].pending] == ["step2:phase:C15_LAVES"]
    t = r["text"]
    assert "ceco's CET2" in t and "looks like" in t and "'extend':" in t and "'separate':" in t
    assert "non-magnetic model rescaled to ceco's magnetic basis (sum 1), k = 1/3, exact" in t


def test_combine_with_extend_decision(trio_ext):
    r = trio_ext["run"]
    assert r["written"], r["text"]
    st = r["state"]
    e = next(x for x in st["log"] if x["id"] == "step2:phase:C15_LAVES")
    assert e["chosen"] == "extend" and e["who"] == "user" and e["kind"] == "harmonize"
    ev = " | ".join(e["evidence"])
    for s in ("B sublattice 2->1, 1->2", "k = 1/3, exact", "imported 3 parameter(s)", "constituents added",
              "GM A's C15_LAVES on Ce-Fe", "GM B's CET2 reproduced on Ce-Co"):
        assert s in ev, s
    ex = [x for x in st["exactness"] if x["transform"].startswith("extend")]
    assert len(ex) == 2 and all(x["ok"] for x in ex)
    rows = {(x["source"], x["subsystem"]): x for x in st["validation"]["rows"]}
    assert all(x["verdict"] != "CHANGED, UNEXPLAINED" for x in rows.values())
    assert "step2:phase:C15_LAVES" in rows[("ceco", "Ce-Co")]["explained_by"]
    assert rows[("cefe", "Ce-Fe")]["verdict"] == "reproduced"
    # the final TDB: base's C15_LAVES on ceco's magnetic sum-one basis, ceco's magnetic parameters untouched
    F = parse_tdb(st["final"])
    assert F.phases["C15_LAVES"].site_ratios == [0.333333333333333, 0.666666666666667] and "CET2" not in F.phases
    fp = {p.key.label(): p.expr.ranges for p in F.parameters if p.key.phase == "C15_LAVES"}
    assert fp["TC(C15_LAVES,CE:CO;0)"] == [(298.15, 3000.0, "900")]
    assert fp["BMAGN(C15_LAVES,CE:CO;0)"] == [(298.15, 3000.0, "1.2")]
    assert st["steps"][1]["extensions"]["C15_LAVES"]["phase"] == "CET2"


def test_combine_notebook_rebuilds_extension(trio_ext, tmp_path):
    import json
    import os
    import subprocess
    import sys
    st = trio_ext["run"]["state"]
    nb = json.loads(open(st["files"]["notebook"]).read())
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "staged_extensions" in code and "the 'extend' decisions through calphad_tools.harmonize" in json.dumps(nb)
    (tmp_path / "nb.py").write_text(code)
    env = {**os.environ, "COMBINE_NOTEBOOK_VALIDATION": "0", "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
    p = subprocess.run([sys.executable, "-W", "ignore", "nb.py"], cwd=tmp_path, capture_output=True, text=True,
                       env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "parameter-equivalent to the tool's combined TDB: True" in p.stdout


def test_phase_harmonization_argument_plans_the_extension(trio_ext):
    from calphad_tools.combine import plan_combination
    d = trio_ext["dir"]
    plan = plan_combination(trio_ext["srcs"], str(d / "p2"), order=trio_ext["srcs"], unary50="off",
                            phase_harmonization={"C15_LAVES": {"source_phase": "CET2", "sublattice_map": [1, 0],
                                                               "import_parameters": {"require_components": ["CO"]},
                                                               "magnetic_model": "source"}},
                            decision_reasons={"phase_harmonization": "user: the Ce-Co Laves is C15"})
    assert not plan.pending
    st = plan.steps[1]
    assert st.extensions["C15_LAVES"]["phase"] == "CET2" and st.extensions["C15_LAVES"]["components"] == ["CO"]
    n = next(x for x in st.needed if x.id == "step2:phase:C15_LAVES")
    assert n.answer == "extend" and n.who == "user"
    assert any("k = 1/3, exact" in h for h in st.plan_info["harmonize"])
    assert "phase:C15_LAVES" not in st.decisions          # not a merge decision: the merge never sees CET2
