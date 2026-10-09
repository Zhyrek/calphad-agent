"""The magnetic formula-unit basis rule (the user's directive), on a C15 Laves case like the one that lost Co:

* A (base): (Fe,Mo,Zr)2(Ce,Fe,Mo,Zr)1, magnetic (BMAGN(FE:ZR) = 3, Fe2Zr is ferromagnetic);
* B: (Ce)0.3333(Co,Fe)0.6667, magnetic on Co.

Rule, for a matched pair with proportional site ratios (factor k != 1):

  A magnetic | B magnetic | rescaled
  yes        | yes        | A to B's basis: G/L x 1/k, TC kept, BMAGN end-members (1+beta)**(1/k)-1 exactly,
             |            | interactions refitted; GM exact at A's end-members, error between them reported
  no         | yes        | A to B's basis (exact), whatever B's atoms per formula unit
  yes        | no         | B to A's basis (exact)
  no         | no         | B to A's basis (exact)

Plus: the db1 / db2 options of a remaining phase-model conflict name what they drop, no policy ever drops an
element from a phase automatically, and 'extend' is listed first (recommended) where it is possible.

Run:  .venv/bin/python -m pytest -q tests/test_magnetic_basis.py
"""
import math
import re
from pathlib import Path

import pytest

from calphad_tools import harmonize as hz
from calphad_tools import tdbstage as ts
from calphad_tools.tdbparse import parse_tdb

HEAD = """ELEMENT VA VACUUM 0 0 0 !
ELEMENT CE FCC_A1 140.12 0 0 !
ELEMENT CO HCP_A3 58.93 0 0 !
ELEMENT FE BCC_A2 55.85 0 0 !
ELEMENT MO BCC_A2 95.94 0 0 !
ELEMENT ZR HCP_A3 91.22 0 0 !
TYPE_DEFINITION % SEQ * !
"""
MAG_TD = "TYPE_DEFINITION A GES AMEND_PHASE_DESCRIPTION C15_LAVES MAGNETIC -3 0.28 !\n"


def _liquid(els, ref):
    return (f"PHASE LIQUID % 1 1 !\nCONSTITUENT LIQUID :{','.join(els)}: !\n"
            + "".join(f"PARAMETER G(LIQUID,{e};0) 298.15 5000-5*T; 6000 N {ref} !\n" for e in els))


A_C15 = """CONSTITUENT C15_LAVES :FE,MO,ZR:CE,FE,MO,ZR: !
PARAMETER G(C15_LAVES,FE:CE;0) 298.15 -30000; 6000 N REFA !
PARAMETER G(C15_LAVES,FE:FE;0) 298.15 15000; 6000 N REFA !
PARAMETER G(C15_LAVES,MO:ZR;0) 298.15 -20000; 6000 N REFA !
PARAMETER G(C15_LAVES,FE:ZR;0) 298.15 -25000; 6000 N REFA !
PARAMETER G(C15_LAVES,ZR:ZR;0) 298.15 15000; 6000 N REFA !
PARAMETER G(C15_LAVES,MO:MO;0) 298.15 15000; 6000 N REFA !
PARAMETER G(C15_LAVES,FE:MO;0) 298.15 10000; 6000 N REFA !
PARAMETER L(C15_LAVES,FE:CE,FE;0) 298.15 30000; 6000 N REFA !
"""
A_MAG = """PARAMETER TC(C15_LAVES,FE:ZR;0) 298.15 600; 6000 N REFA !
PARAMETER BMAGN(C15_LAVES,FE:ZR;0) 298.15 3.0; 6000 N REFA !
"""


def a_text(mag=True):
    return (HEAD + _liquid(["CE", "FE", "MO", "ZR"], "REFA") + (MAG_TD if mag else "")
            + f"PHASE C15_LAVES %{'A' if mag else ''} 2 2 1 !\n" + A_C15 + (A_MAG if mag else ""))


def b_text(mag=True, ratios="0.3333 0.6667", scale=1):
    g = lambda v: f"{v * scale:g}"   # noqa: E731
    return (HEAD + _liquid(["CE", "CO", "FE"], "REFB") + (MAG_TD if mag else "")
            + f"PHASE C15_LAVES %{'A' if mag else ''} 2 {ratios} !\n"
            + "CONSTITUENT C15_LAVES :CE:CO,FE: !\n"
            + f"PARAMETER G(C15_LAVES,CE:CO;0) 298.15 {g(-12000)}; 6000 N REFB !\n"
            + f"PARAMETER G(C15_LAVES,CE:FE;0) 298.15 {g(-10000)}; 6000 N REFB !\n"
            + f"PARAMETER L(C15_LAVES,CE:CO,FE;0) 298.15 {g(-2000)}; 6000 N REFB !\n"
            + ("PARAMETER TC(C15_LAVES,CE:CO;0) 298.15 1000; 6000 N REFB !\n"
               "PARAMETER BMAGN(C15_LAVES,CE:CO;0) 298.15 1.0; 6000 N REFB !\n" if mag else ""))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("magbasis")
    out = {"dir": d}
    for name, text in (("A_mag", a_text(True)), ("A", a_text(False)), ("B_mag", b_text(True)),
                       ("B_nomag", b_text(False)), ("B_mag42", b_text(True, "2 4", 6)),
                       ("B_bad", b_text(True, "0.4 0.6"))):
        (d / f"{name}.tdb").write_text(text)
        out[name] = str(d / f"{name}.tdb")
    return out


def _params(path_or_db, phase="C15_LAVES"):
    db = path_or_db if not isinstance(path_or_db, (str, Path)) else parse_tdb(str(path_or_db))
    return {p.raw_name: p for p in db.parameters if p.key.phase == phase}


def _value(p) -> float:
    return float(p.expr.ranges[0][2])


def _gm(orig, new, phase="C15_LAVES", n=400, comps=None, a_to_b=None):
    o = orig if not isinstance(orig, (str, Path)) else parse_tdb(str(orig))
    w = new if not isinstance(new, (str, Path)) else parse_tdb(str(new))
    return hz.gm_compare(hz.pyc_db(o), phase, hz.pyc_db(w), phase, comps or hz.phase_comps(o, phase), a_to_b, n=n)


# ================================================================================================ A_mag + B_mag
@pytest.fixture(scope="module")
def both(files):
    from calphad_tools.staged_merge import staged_merge
    r = staged_merge(files["A_mag"], files["B_mag"], str(files["dir"] / "both"), plot=False,
                     ternary_temperatures_c=[])
    return r


def test_both_magnetic_no_decision_and_co_kept(both):
    r = both
    assert r["written"] and r.get("pending") is None            # no phase-model question any more
    for lab in ("B3", "Q"):
        ph = parse_tdb(r["versions"][lab].path).phases["C15_LAVES"]
        assert [set(c) for c in ph.constituents] == [{"CO", "FE", "MO", "ZR"}, {"CE", "FE", "MO", "ZR"}], lab
        assert ph.site_ratios == [0.666666666666667, 0.333333333333333], lab      # B's sum-1 basis, A's order
    assert not [n for n in r["notes"] if n.action in ("not_rescaled", "incompatible")]


def test_both_magnetic_B_bit_for_bit_in_Q(both, files):
    """B's C15 parameters reach Q unchanged: the same expression text (only the sublattice order is A's).  The one
    key both databases have, G(FE:CE) (Ce-Fe), follows the staged rule (the base's binary wins) - equal here."""
    q = _params(both["versions"]["Q"].path)
    b = _params(files["B_mag"])
    a = _params(files["A_mag"])
    assert len(b) == 5
    for name, p in b.items():
        flipped = re.sub(r"\(C15_LAVES,([^:;]+):([^;]+);", r"(C15_LAVES,\2:\1;", name)
        if flipped in a:
            assert _value(q[flipped]) == pytest.approx(_value(a[flipped]) / 3, rel=1e-15) == _value(p), name
        else:
            assert q[flipped].expr.ranges == p.expr.ranges, name          # bit for bit (Co-bearing: 4 terms)
    assert sum(1 for n in b if "CO" in n) == 4


def test_both_magnetic_A_transformed(both, files):
    """A's G/L x 1/3 per T range, TC unchanged, BMAGN end-member (1+beta)**(1/3) - 1 exactly; A's GM per atom
    unchanged at its end-members (relative 1e-9) and the error between them reported."""
    a, a0, q = _params(files["A_mag"]), _params(both["A_eff"]), _params(both["versions"]["Q"].path)
    for name, p in a.items():
        if p.key.ptype in ("G", "L"):
            assert _value(a0[name]) == pytest.approx(_value(p) / 3, rel=1e-15), name
            assert _value(q[name]) == pytest.approx(_value(p) / 3, rel=1e-15), name
    assert a0["TC(C15_LAVES,FE:ZR;0)"].expr.ranges == a["TC(C15_LAVES,FE:ZR;0)"].expr.ranges
    beta = 4.0 ** (1 / 3) - 1
    assert _value(a0["BMAGN(C15_LAVES,FE:ZR;0)"]) == pytest.approx(beta, rel=1e-15)
    assert _value(q["BMAGN(C15_LAVES,FE:ZR;0)"]) == pytest.approx(beta, rel=1e-15)
    refit = [n for n, p in a0.items() if p.key.ptype == "BMAGN" and p.expr.reference == "REFIT"]
    assert refit and all("FE" in n for n in refit)                  # interactions next to the magnetic end-member
    em = _gm(files["A_mag"], both["A_eff"], n=0)
    assert em["n"] > 0 and em["max_rel"] <= 1e-9                    # exact at every end-member of A
    note = next(n for n in both["notes"] if n.action == ts.A_RESCALED)
    it = note.magnetic["interior"]
    assert note.magnetic["endmembers_exact"] and 0 < it["max_abs"] < 1.0 and it["n"] >= 400
    full = _gm(files["A_mag"], both["A_eff"])                       # independent check: same order of magnitude
    assert full["max_abs"] == pytest.approx(it["max_abs"], rel=1e-6)
    print(f"C15 interior error {it['max_abs']:.4g} J/mol of atoms; end-members {em['max_abs']:.2g}")


def test_both_magnetic_log_text(both):
    note = next(n for n in both["notes"] if n.action == ts.A_RESCALED)
    t = note.text()
    m = re.search(r"C15_LAVES: A rescaled to B's basis - A \(magnetic\) rescaled to B's magnetic basis \(sum 1\), "
                  r"k = 1/3; BMAGN end-members exact, interior error <= ([0-9.e-]+) J/mol of atoms \(directive: "
                  r"tolerated\)", t)
    assert m, t
    assert float(m.group(1)) == pytest.approx(note.magnetic["interior"]["max_abs"], rel=1e-2)
    assert "BMAGN(C15_LAVES,FE:ZR;0) 3 -> 0.587401" in t and "TC unchanged" in t
    assert "A0 " in __import__("calphad_tools.staged_merge", fromlist=["summary_text"]).summary_text(both)


def test_validate_edit_accepts_magnetic_basis_rescale(both, files):
    from calphad_tools.tdb_validate import validate_edit
    r = validate_edit(files["A_mag"], both["A_eff"], expect=["energy_per_atom_preserved"])
    g = {x["phase"]: x for x in r["gm"]}["C15_LAVES"]
    assert r["verdict"] == "PASS" and g["status"] == "pass"
    assert "magnetic basis rescale" in g["detail"] and "end-members exact" in g["detail"]
    assert "magnetic contribution changed by" in g["detail"] and g["mag_abs"] > 0


def test_version_diff_counts_bmagn_not_gl(both, files):
    """A -> A0: G/L compared x1/3 (exact), BMAGN terms are changes where they apply (Fe-Zr), Mo-Zr is not."""
    els = ["CE", "FE", "MO", "ZR"]
    d = ts.diff_versions(ts.StageView(parse_tdb(files["A_mag"]), "A", elements=els),
                         ts.StageView(parse_tdb(both["A_eff"]), "A0", elements=els), ts.stage_subsystems(els, (2,)))
    fe_zr, mo_zr = d.get(("FE", "ZR")), d.get(("MO", "ZR"))
    assert fe_zr.changed and {p.ptype for p in fe_zr.params} == {"BMAGN"}
    assert not mo_zr.changed and any("magnetic basis rescale" in n for n in mo_zr.notes)


# ================================================================================================ other rows
def test_A_magnetic_B_not_B_rescaled_exact(files):
    A, B = parse_tdb(files["A_mag"]), parse_tdb(files["B_nomag"])
    An, Bn, notes = ts.normalize_pair(A, B)
    acts = [n.action for n in notes]
    assert "rescaled" in acts and ts.A_RESCALED not in acts
    note = next(n for n in notes if n.action == "rescaled")
    assert "B rescaled to A's basis (A is magnetic and B is not" in note.detail and "k = 3" in note.detail
    assert note.detail.endswith("exact")
    assert An.phases["C15_LAVES"].site_ratios == [2.0, 1.0] and Bn.phases["C15_LAVES"].site_ratios == [2.0, 1.0]
    assert {k: p.expr.ranges for k, p in _params(An).items()} == {k: p.expr.ranges for k, p in _params(A).items()}
    ref = parse_tdb(files["B_nomag"])
    ts.snap_phase_in(ref, "C15_LAVES")                               # compare with the exact-fraction B
    res = _gm(ref, Bn, a_to_b=[1, 0])
    assert res["n"] > 0 and res["max_rel"] <= 1e-9


@pytest.mark.parametrize("b", ["B_mag", "B_mag42"])
def test_A_nonmagnetic_B_magnetic_A_rescaled_exact(files, b):
    """Whatever B's atoms per formula unit (1/3:2/3 -> k = 1/3; 2:4 -> k = 2): A takes B's basis, exactly."""
    A, B = parse_tdb(files["A"]), parse_tdb(files[b])
    An, Bn, notes = ts.normalize_pair(A, B)
    note = next(n for n in notes if n.action == ts.A_RESCALED)
    k = {"B_mag": "1/3", "B_mag42": "2"}[b]
    assert f"A (non-magnetic) rescaled to B's magnetic basis (sum {'1' if b == 'B_mag' else '6'}), k = {k}: exact" \
        in note.detail and note.magnetic is None
    want = {"B_mag": [0.666666666666667, 0.333333333333333], "B_mag42": [4.0, 2.0]}[b]
    assert An.phases["C15_LAVES"].site_ratios == want and Bn.phases["C15_LAVES"].site_ratios == want
    res = _gm(A, An)
    assert res["n"] > 0 and res["max_rel"] <= 1e-9
    # B's magnetic parameters untouched (values; only reordered)
    vb = sorted((p.key.ptype, p.expr.ranges[0][2]) for p in _params(B).values() if p.key.ptype in ("TC", "BMAGN"))
    vn = sorted((p.key.ptype, p.expr.ranges[0][2]) for p in _params(Bn).values() if p.key.ptype in ("TC", "BMAGN"))
    assert vb == vn and len(vb) == 2


def test_neither_magnetic_unchanged(files):
    An, Bn, notes = ts.normalize_pair(parse_tdb(files["A"]), parse_tdb(files["B_nomag"]))
    note = next(n for n in notes if n.action == "rescaled")
    assert "B rescaled to A's basis (neither is magnetic), k = 3" in note.detail
    assert not [n for n in notes if n.action == ts.A_RESCALED]
    assert An.phases["C15_LAVES"].site_ratios == [2.0, 1.0]
    assert _value(_params(Bn)["G(C15_LAVES,CO:CE;0)"]) == pytest.approx(-36000.0)


def test_order_disorder_still_excluded(tmp_path):
    """An order-disorder A phase is never rescaled: B's magnetic basis cannot win there (reordered only)."""
    a = ("ELEMENT VA VACUUM 0 0 0 !\nELEMENT FE BCC_A2 55.85 0 0 !\nELEMENT MO BCC_A2 95.94 0 0 !\n"
         "TYPE_DEFINITION % SEQ * !\nTYPE_DEFINITION D GES A_P_D L12 DIS_PART FCC_A1 !\n"
         "PHASE FCC_A1 % 1 1 !\nCONSTITUENT FCC_A1 :FE,MO: !\n"
         "PARAMETER G(FCC_A1,FE;0) 298.15 1000; 6000 N !\nPARAMETER G(FCC_A1,MO;0) 298.15 2000; 6000 N !\n"
         "PHASE L12 %D 2 3 1 !\nCONSTITUENT L12 :FE,MO:FE,MO: !\n"
         "PARAMETER G(L12,FE:MO;0) 298.15 -4000; 6000 N !\nPARAMETER G(L12,MO:FE;0) 298.15 4000; 6000 N !\n")
    b = ("ELEMENT VA VACUUM 0 0 0 !\nELEMENT FE BCC_A2 55.85 0 0 !\nELEMENT W BCC_A2 183.84 0 0 !\n"
         "TYPE_DEFINITION % SEQ * !\nTYPE_DEFINITION M GES A_P_D L12 MAGNETIC -3 0.28 !\n"
         "PHASE L12 %M 2 0.75 0.25 !\nCONSTITUENT L12 :FE,W:FE,W: !\n"
         "PARAMETER G(L12,FE:W;0) 298.15 -1000; 6000 N !\nPARAMETER TC(L12,FE:FE;0) 298.15 900; 6000 N !\n"
         "PARAMETER BMAGN(L12,FE:FE;0) 298.15 1.5; 6000 N !\n")
    (tmp_path / "a.tdb").write_text(a)
    (tmp_path / "b.tdb").write_text(b)
    An, Bn, notes = ts.normalize_pair(parse_tdb(str(tmp_path / "a.tdb")), parse_tdb(str(tmp_path / "b.tdb")))
    nr = [n for n in notes if n.action == "not_rescaled"]
    assert nr and "order-disorder" in nr[0].detail and ts.A_RESCALED not in [n.action for n in notes]
    assert An.phases["L12"].site_ratios == [3.0, 1.0] and Bn.phases["L12"].site_ratios == [0.75, 0.25]


# ================================================================================================ harmonize
def test_extend_phase_auto_same_rule(files, both, tmp_path):
    """extend_phase(magnetic='auto') with both magnetic: the same rule - A's C15 on B's sum-1 basis, A's BMAGN
    end-member transformed exactly (the same value as the staged normalisation), B's Co parameters imported."""
    r = hz.extend_phase(files["A_mag"], files["B_mag"], "C15_LAVES", components=["CO"],
                        output_tdb=str(tmp_path / "ext.tdb"))
    bc = r["basis"]
    assert bc["basis"] == "add" and bc["magnetic_transform"] and "k = 1/3" in bc["text"]
    assert "BMAGN end-members exact" in bc["text"] and "interior error <=" in bc["text"]
    new = parse_tdb(str(tmp_path / "ext.tdb"))
    assert new.phases["C15_LAVES"].site_ratios == [0.666666666666667, 0.333333333333333]
    pe, pq = _params(new), _params(both["versions"]["Q"].path)
    assert _value(pe["BMAGN(C15_LAVES,FE:ZR;0)"]) == _value(pq["BMAGN(C15_LAVES,FE:ZR;0)"])
    assert "BMAGN(C15_LAVES,CO:CE;0)" in pe and _value(pe["BMAGN(C15_LAVES,CO:CE;0)"]) == 1.0
    assert r["checks"]["base_endmembers"]["max_rel"] <= 1e-9


def test_normalize_formula_unit_match_follows_rule(files, tmp_path):
    r = hz.normalize_phase_formula_unit(files["A_mag"], ["C15_LAVES"], output_tdb=str(tmp_path / "m.tdb"),
                                        match_tdb=files["B_mag"])
    rep = r["phases"][0]
    assert rep["magnetic"] and rep["k_text"] == "1/3" and rep["gm_endmembers"]["max_rel"] <= 1e-9
    assert "BMAGN end-members exact" in rep["text"]
    with pytest.raises(hz.MagneticBasisError, match="rescale .*C15_LAVES to this phase instead"):
        hz.normalize_phase_formula_unit(files["A_mag"], ["C15_LAVES"], output_tdb=str(tmp_path / "n.tdb"),
                                        match_tdb=files["B_nomag"])


# ================================================================================================ dangerous options
def test_db1_names_what_it_drops_and_policy_never_drops(files):
    from calphad_tools import merge
    from calphad_tools.staged_merge import staged_merge, summary_text
    for pol in ("ask", "first", "last"):
        r = staged_merge(files["A_mag"], files["B_bad"], str(files["dir"] / f"bad_{pol}"), plot=False,
                         policy=pol, ternary_temperatures_c=[])
        assert not r["written"], pol                          # never resolved by a policy: it would drop elements
        t = summary_text(r)
        assert "* phase:C15_LAVES" in t
        assert re.search(r"'db1': use the model from A_mag\.tdb - drops CO from C15_LAVES \(B_bad\.tdb's 5 "
                         r"C15_LAVES parameters lost: Ce-Co 3, Ce-Fe 1, Ce-Co-Fe 1\)", t), t
        assert "'db2': use the model from B_bad.tdb - drops MO, ZR from C15_LAVES" in t
    plan = merge.plan_merge([files["A_mag"], files["B_bad"]], {}, "first")
    assert [c.id for c in plan.pending] == ["phase:C15_LAVES"]
    assert not merge.plan_merge([files["A_mag"], files["B_bad"]], {}, "first:predict").pending


def test_extend_listed_first_when_possible(files):
    from calphad_tools.staged_merge import staged_merge, summary_text
    r = staged_merge(files["A_mag"], files["B_mag"], str(files["dir"] / "keep"), plot=False,
                     sublattice_map={"C15_LAVES": "keep"}, ternary_temperatures_c=[])
    assert not r["written"]
    opts = [o.key for o in r["pending"].pending[0].options]
    assert opts[0] == "extend" and "'extend': RECOMMENDED" in summary_text(r)
    r2 = staged_merge(files["A_mag"], files["B_mag"], str(files["dir"] / "keep2"), plot=False,
                      sublattice_map={"C15_LAVES": "keep"}, decisions={"phase:C15_LAVES": "extend"},
                      ternary_temperatures_c=[])
    assert r2["written"]
    q = parse_tdb(r2["versions"]["Q"].path).phases["C15_LAVES"]
    assert "CO" in q.constituents[0] and {"MO", "ZR"} <= set(q.constituents[0])


# ================================================================================================ combine
@pytest.fixture(scope="module")
def combined(files):
    from calphad_tools.combine import combine_databases
    srcs = [files["A_mag"], files["B_mag"]]
    return combine_databases(srcs, str(files["dir"] / "comb"), order=srcs, unary50="off", n_workers=3, lint=False,
                             tmax_c=1800, composition_step=0.05, temperature_step=25)


def test_combine_c15_co_kept_logged_and_validated(combined):
    r = combined
    assert r["written"], r["text"]
    st = r["state"]
    assert not any("C15" in n["id"] for n in st["steps"][0]["needed"])            # no question
    F = parse_tdb(st["final"])
    assert "CO" in F.phases["C15_LAVES"].constituents[0]
    assert F.phases["C15_LAVES"].site_ratios == [0.666666666666667, 0.333333333333333]
    e = next(x for x in st["log"] if x["id"] == "step1:norm:C15_LAVES:magnetic-basis")
    assert e["who"] == "rule" and e["chosen"] == "base rescaled"
    assert re.search(r"C15_LAVES: A \(magnetic\) rescaled to B's magnetic basis \(sum 1\), k = 1/3; BMAGN "
                     r"end-members exact, interior error <= [0-9.e-]+ J/mol of atoms \(directive: tolerated\)",
                     e["reason"]), e["reason"]
    assert any("GM identity at the end-members" in x and "exact" in x for x in e["evidence"])
    ex = next(x for x in st["exactness"] if x["transform"].startswith("magnetic basis rule"))
    assert ex["ok"] and ex["max_rel"] <= 1e-9
    rows = {(x["source"], x["subsystem"]): x for x in st["validation"]["rows"]}
    assert all(x["verdict"] != "CHANGED, UNEXPLAINED" for x in rows.values()), \
        [x["basis"] for x in rows.values() if x["verdict"] == "CHANGED, UNEXPLAINED"]
    fe_zr = rows[("A_mag", "Fe-Zr")]
    assert fe_zr["verdict"] in ("changed, explained", "reproduced")
    assert fe_zr["verdict"] == "reproduced" or "step1:norm:C15_LAVES:magnetic-basis" in fe_zr["explained_by"]


def test_combine_policy_never_drops_an_element(files):
    from calphad_tools.combine import combine_databases
    srcs = [files["A_mag"], files["B_bad"]]
    r = combine_databases(srcs, str(files["dir"] / "comb_bad"), order=srcs, unary50="off", policy="first")
    assert not r["written"]
    n = next(x for x in r["plan"].pending if x.id == "step1:phase:C15_LAVES")
    assert n.answer is None and n.drops["db1"] == ["CO"]
    t = r["text"]
    assert re.search(r"'db1': A_mag's C15_LAVES model only - drops CO from C15_LAVES \(B_bad's 5 C15_LAVES "
                     r"parameters lost: Ce-Co 3, Ce-Fe 1, Ce-Co-Fe 1\)", t), t
    assert "policy='first' not applied: 'db1' would drop CO from the phase - asked" in t


def test_combine_lists_extend_first(files):
    """B's C15 kept as is (sublattice_map 'keep'): the remaining phase-model question offers 'extend' first, as
    recommended, and each db option says what it drops."""
    from calphad_tools.combine import combine_databases
    srcs = [files["A_mag"], files["B_mag"]]
    r = combine_databases(srcs, str(files["dir"] / "comb_keep"), order=srcs, unary50="off",
                          sublattice_map={"C15_LAVES": "keep"})
    assert not r["written"]
    n = next(x for x in r["plan"].pending if x.id == "step1:phase:C15_LAVES")
    assert [k for k, _ in n.options][:3] == ["extend", "db1", "db2"]
    assert n.options[0][1].startswith("RECOMMENDED - ")
    assert "drops CO from C15_LAVES" in dict(n.options)["db1"]
