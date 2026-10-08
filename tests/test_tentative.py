"""Tentative parameters (calphad_tools.tentative) added when TDBs are combined.

* rule 1 (TENT_MIX): (Fe)2(Fe,W) + (Fe)2(Nb) -> (Fe)2(Fe,Nb,W) gets L(C14_LAVES,FE:FE,NB;0) = +50000 J/mol-atom x
  3 atoms/fu = +150000 (2:1) or +50000 (0.666667:0.333333); Fe,W (assessed) is left alone, Nb,W is new too;
* rule 2 (TENT_VA): bcc (Fe,Cr,Ni,Va)(Va) + (Co)(Va) -> L(BCC_A2,CO,VA:VA;0) = +200000; Fe/Cr/Ni keep their
  assessed terms; (Fe)(C,Va) never triggers it (an interstitial sublattice);
* precedence: an assessed term replaces a tentative one whichever database is the base - merge_tdbs, staged Q,
  a 3-source combine_databases (a tentative term from step 1 replaced by step 2's assessment), extend_phase;
* citations: TENT_MIX / TENT_VA reference-list entries written, re-parsed as tentative, shown by compare_system and
  query_tdb.

Run:  .venv/bin/python -m pytest -q tests/test_tentative.py
"""
import json
import re
from pathlib import Path

import pytest

from calphad_tools import tentative as tent
from calphad_tools.merge import merge, plan_merge
from calphad_tools.tdbparse import is_tentative, parse_tdb

from test_compatibility import GHSER, FNAME

HDR = """ ELEMENT VA VACUUM 0 0 0 !
 ELEMENT FE BCC_A2 55.847 4489 27.28 !
 ELEMENT NB BCC_A2 92.906 5220 36.27 !
 ELEMENT W BCC_A2 183.84 4973 32.66 !
 ELEMENT CR BCC_A2 51.996 4050 23.54 !
 ELEMENT NI FCC_A1 58.69 4787 29.79 !
 ELEMENT CO HCP_A3 58.933 4765 30.04 !
 ELEMENT C GRAPHITE 12.011 1054 5.74 !
 TYPE_DEFINITION % SEQ * !
"""


def write(path, text):
    Path(path).write_text(text)
    return str(path)


def c14(cons, params, ratios="2 2 1"):
    return HDR + f" PHASE C14_LAVES % {ratios} !\n CONSTITUENT C14_LAVES :{cons}: !\n" + "\n".join(params) + "\n"


FEW = [" PARAMETER G(C14_LAVES,FE:FE;0) 298.15 15000-10*T; 6000 N REFA !",
       " PARAMETER G(C14_LAVES,FE:W;0) 298.15 -30000+5*T; 6000 N REFA !",
       " PARAMETER L(C14_LAVES,FE:FE,W;0) 298.15 -5000; 6000 N REFA !"]
FENB = [" PARAMETER G(C14_LAVES,FE:NB;0) 298.15 -40000+5*T; 6000 N REFB !"]


def params(tdb_path):
    t = parse_tdb(tdb_path)
    return t, {p.key.label(): p for p in t.parameters}


# ------------------------------------------------------------------------------------------------- rule 1
@pytest.mark.parametrize("ratios,value", [("2 2 1", 150000.0), ("2 0.666667 0.333333", 50000.0)])
def test_rule1_new_mixing_scales_with_atoms_per_formula_unit(tmp_path, ratios, value):
    a = write(tmp_path / "FeW.tdb", c14("FE:FE,W", FEW, ratios))
    b = write(tmp_path / "FeNb.tdb", c14("FE:NB", FENB, ratios))
    for paths in ([a, b], [b, a]):
        plan = plan_merge(paths, policy="first")
        got = {t.label: t for t in plan.tentative}
        assert set(got) == {"L(C14_LAVES,FE:FE,NB;0)", "L(C14_LAVES,FE:NB,W;0)"}
        assert all(t.ref == "TENT_MIX" and t.value == value for t in got.values())
        assert "FE:FE" in got["L(C14_LAVES,FE:FE,NB;0)"].evidence and "FE:NB" in got["L(C14_LAVES,FE:FE,NB;0)"].evidence
        # the assessed Fe,W interaction is not touched
        assert plan.params[("G", "C14_LAVES", (("FE",), ("FE", "W")), 0, "")].expr.short() == "-5000"
    out = merge([a, b], str(tmp_path / "m.tdb"), policy="first")
    assert out["written"] and not out["validation"]["errors"]
    t, ps = params(tmp_path / "m.tdb")
    p = ps["L(C14_LAVES,FE:FE,NB;0)"]
    assert p.reference == "TENT_MIX" and float(p.expr.ranges[0][2]) == value and is_tentative(p, t)
    assert not is_tentative(ps["L(C14_LAVES,FE:FE,W;0)"], t)
    # the value is configurable and the rule can be switched off
    plan = plan_merge([a, b], policy="first", tentative={"mixing_per_atom": 30000})
    assert {x.value for x in plan.tentative} == {round(30000 * (3.0 if ratios == "2 2 1" else 1.0), 6)}
    assert not plan_merge([a, b], policy="first", tentative=False).tentative


def test_rule1_needs_both_end_members_and_skips_solution_phases(tmp_path):
    # no G for FE:NB: nothing to add for Fe,Nb (pycalphad would use 0 for the end-member anyway)
    a = write(tmp_path / "FeW.tdb", c14("FE:FE,W", FEW))
    b = write(tmp_path / "FeNb.tdb", c14("FE:NB", []))
    assert not plan_merge([a, b], policy="first").tentative
    # a solution phase (one substitutional sublattice): a pair no source describes is a new binary, left alone
    liq = lambda els: HDR + f" PHASE LIQUID % 1 1 !\n CONSTITUENT LIQUID :{els}: !\n" + "\n".join(  # noqa: E731
        f" PARAMETER G(LIQUID,{e};0) 298.15 -1000*{i + 1}; 6000 N REFL !" for i, e in enumerate(els.split(","))) + "\n"
    plan = plan_merge([write(tmp_path / "l1.tdb", liq("FE,NB")), write(tmp_path / "l2.tdb", liq("FE,W"))],
                      policy="first")
    assert not plan.tentative


# ------------------------------------------------------------------------------------------------- rule 2
def bcc(cons, params, ratios="2 1 3"):
    return HDR + f" PHASE BCC_A2 % {ratios} !\n CONSTITUENT BCC_A2 :{cons}: !\n" + "\n".join(params) + "\n"


BCC_A = [" PARAMETER G(BCC_A2,FE:VA;0) 298.15 -1000; 6000 N REFA !",
         " PARAMETER G(BCC_A2,CR:VA;0) 298.15 -2000; 6000 N REFA !",
         " PARAMETER G(BCC_A2,NI:VA;0) 298.15 -3000; 6000 N REFA !",
         " PARAMETER G(BCC_A2,VA:VA;0) 298.15 0; 6000 N REFA !",
         " PARAMETER L(BCC_A2,FE,VA:VA;0) 298.15 180000; 6000 N REFA !",
         " PARAMETER L(BCC_A2,CR,VA:VA;0) 298.15 100000; 6000 N REFA !",
         " PARAMETER L(BCC_A2,NI,VA:VA;0) 298.15 210000; 6000 N REFA !"]
BCC_B = [" PARAMETER G(BCC_A2,CO:VA;0) 298.15 -4000; 6000 N REFB !"]


def test_rule2_vacancy_on_substitutional_sublattice(tmp_path):
    a = write(tmp_path / "FeCrNiVa.tdb", bcc("FE,CR,NI,VA:VA", BCC_A))
    b = write(tmp_path / "Co.tdb", bcc("CO:VA", BCC_B))
    for paths in ([a, b], [b, a]):
        plan = plan_merge(paths, policy="first")
        assert [(t.label, t.ref, t.value) for t in plan.tentative] == [("L(BCC_A2,CO,VA:VA;0)", "TENT_VA", 200000.0)]
        assert not plan.tentative_notes                       # G(BCC_A2,VA:VA) exists
        assert plan.params[("G", "BCC_A2", (("FE", "VA"), ("VA",)), 0, "")].expr.short() == "180000"
        assert plan.params[("G", "BCC_A2", (("CR", "VA"), ("VA",)), 0, "")].expr.short() == "100000"
    out = merge([a, b], str(tmp_path / "m.tdb"), policy="first", tentative={"vacancy": 250000})
    t, ps = params(tmp_path / "m.tdb")
    assert ps["L(BCC_A2,CO,VA:VA;0)"].reference == "TENT_VA" and ps["L(BCC_A2,CO,VA:VA;0)"].expr.short() == "+250000"
    assert t.reference_list["TENT_VA"].startswith("Tentative parameter added by calphad_tools when combining")
    assert "+250000 J/mol" in t.reference_list["TENT_VA"]
    # the vacancy end-member missing: reported, not invented
    a2 = write(tmp_path / "noVA.tdb", bcc("FE,CR,NI,VA:VA", [x for x in BCC_A if "VA:VA;0) 298.15 0;" not in x]))
    plan = plan_merge([a2, b], policy="first")
    assert plan.tentative and any("no G(BCC_A2,VA:VA) for the vacancy end-member" in n and "G = 0" in n
                                  for n in plan.tentative_notes)
    assert ("G", "BCC_A2", (("VA",), ("VA",)), 0, "") not in plan.params


def test_rule2_not_on_interstitial_sublattice(tmp_path):
    a = write(tmp_path / "FeC.tdb", bcc("FE:C,VA", [" PARAMETER G(BCC_A2,FE:VA;0) 298.15 -1000; 6000 N R !",
                                                    " PARAMETER G(BCC_A2,FE:C;0) 298.15 50000; 6000 N R !"]))
    b = write(tmp_path / "Cr.tdb", bcc("CR:VA", [" PARAMETER G(BCC_A2,CR:VA;0) 298.15 -2000; 6000 N R !"]))
    plan = plan_merge([a, b], policy="first")
    assert plan.phases["BCC_A2"][0].constituents == [["FE", "CR"], ["C", "VA"]]
    assert not plan.tentative                 # (C,Va) is interstitial; (Cr,Fe) mix in a solution phase
    assert tent.is_main(["FE", "CR", "NI", "VA"], tent.default_elements_of()) and not tent.is_main(
        ["C", "VA"], tent.default_elements_of())


# ------------------------------------------------------------------------------------------------- precedence
TENT_FENB = (" PARAMETER L(C14_LAVES,FE:FE,NB;0) 298.15 +150000; 6000 N TENT_MIX !\n LIST_OF_REFERENCES\n"
             " NUMBER SOURCE\n  TENT_MIX 'Tentative parameter added by calphad_tools when combining x, y (2026-01-01):"
             " new Fe,Nb mixing'\n !")
ASSESSED_FENB = " PARAMETER L(C14_LAVES,FE:FE,NB;0) 298.15 -20000+3*T; 6000 N 99ASS !"


@pytest.fixture
def prec(tmp_path):
    g = [" PARAMETER G(C14_LAVES,FE:FE;0) 298.15 15000-10*T; 6000 N REFA !",
         " PARAMETER G(C14_LAVES,FE:NB;0) 298.15 -40000+5*T; 6000 N REFB !"]
    tdb_t = write(tmp_path / "Tent.tdb", c14("FE:FE,NB", g + [TENT_FENB]))
    tdb_a = write(tmp_path / "Assessed.tdb", c14("FE:FE,NB", g + [ASSESSED_FENB]))
    return tdb_t, tdb_a


def test_precedence_merge_both_ways(prec, tmp_path):
    tdb_t, tdb_a = prec
    key = ("G", "C14_LAVES", (("FE",), ("FE", "NB")), 0, "")
    for paths in ([tdb_t, tdb_a], [tdb_a, tdb_t]):
        for policy in ("first", "last", "ask"):
            plan = plan_merge(paths, policy=policy, decisions={"system:FE-NB": "db1"})
            assert plan.params[key].expr.short() == "-20000+3*T"
            assert not plan.pending and not [c for c in plan.conflicts if c.kind == "parameter"]
            assert [r.text() for r in plan.tentative_replaced] == [
                "rule:tentative-replaced: L(C14_LAVES,FE:FE,NB;0) TENT_MIX (Tent.tdb) -> assessed value from "
                "Assessed.tdb"]
    r = merge([tdb_t, tdb_a], str(tmp_path / "m.tdb"), policy="first")
    t, ps = params(tmp_path / "m.tdb")
    assert not is_tentative(ps["L(C14_LAVES,FE:FE,NB;0)"], t) and "TENT_MIX" not in t.reference_list
    assert "rule:tentative-replaced" in r["report"]
    # both tentative: the larger value is kept
    tdb_t2 = write(tmp_path / "Tent2.tdb", Path(tdb_t).read_text().replace("+150000", "+90000"))
    plan = plan_merge([tdb_t2, tdb_t], policy="first")
    assert plan.params[key].expr.short() == "+150000" and plan.tentative_replaced[0].kind == "both"


def test_described_pair_removes_tentative_term(tmp_path):
    """A source that mixes Fe,Nb on sublattice 2 (no L term: ideal, as assessed) removes the tentative term."""
    g = [" PARAMETER G(C14_LAVES,FE:FE;0) 298.15 15000-10*T; 6000 N REFA !",
         " PARAMETER G(C14_LAVES,FE:NB;0) 298.15 -40000+5*T; 6000 N REFB !"]
    q = write(tmp_path / "Q.tdb", c14("FE:FE,NB", g + [TENT_FENB.replace("FE:FE,NB;0", "FE:NB,FE;0")]))
    s = write(tmp_path / "S.tdb", c14("FE,NB:FE,NB", g))
    for paths in ([q, s], [s, q]):
        plan = plan_merge(paths, policy="first")
        assert ("G", "C14_LAVES", (("FE",), ("FE", "NB")), 0, "") not in plan.params
        assert plan.tentative_replaced[0].kind == "described" and "S.tdb describes" in plan.tentative_replaced[0].text()


def test_precedence_staged_q_both_ways(prec, tmp_path):
    from calphad_tools.staged_merge import staged_merge, summary_text
    tdb_t, tdb_a = prec
    for i, (a, b) in enumerate(((tdb_t, tdb_a), (tdb_a, tdb_t))):
        r = staged_merge(a, b, str(tmp_path / f"sm{i}"), plot=False, policy="first")
        assert r["written"]
        t, ps = params(r["versions"]["Q"].path)
        assert ps["L(C14_LAVES,FE:FE,NB;0)"].expr.short() == "-20000+3*T"
        assert not is_tentative(ps["L(C14_LAVES,FE:FE,NB;0)"], t)
        assert "Tentative replaced: L(C14_LAVES,FE:FE,NB;0) TENT_MIX" in summary_text(r)


def test_precedence_extend_phase_both_ways(prec, tmp_path):
    """extend_phase: an assessed collision partner beats a tentative one whatever ``collision`` says (components
    NB, W: add's Fe,Nb terms are imported, so L(C14_LAVES,FE:FE,NB;0) collides)."""
    from calphad_tools.harmonize import extend_phase
    g = [" PARAMETER G(C14_LAVES,FE:FE;0) 298.15 15000-10*T; 6000 N REFA !",
         " PARAMETER G(C14_LAVES,FE:NB;0) 298.15 -40000+5*T; 6000 N REFB !",
         " PARAMETER G(C14_LAVES,FE:W;0) 298.15 -30000+5*T; 6000 N REFA !"]
    base_t = write(tmp_path / "BaseT.tdb", c14("FE:FE,NB", g[:2] + [TENT_FENB]))
    add_a = write(tmp_path / "AddA.tdb", c14("FE:FE,NB,W", g + [ASSESSED_FENB.replace("FE:FE,NB", "FE:NB,FE")]))
    base_a = write(tmp_path / "BaseA.tdb", c14("FE:FE,NB", g[:2] + [ASSESSED_FENB]))
    add_t = write(tmp_path / "AddT.tdb", c14("FE:FE,NB,W", g + [TENT_FENB]))
    key = "L(C14_LAVES,FE:FE,NB;0)"
    for base, add, collision, frm in ((base_t, add_a, "base", "BaseT"), (base_a, add_t, "add", "AddT")):
        r = extend_phase(base, add, "C14_LAVES", components=["NB", "W"], collision=collision, verify=False,
                         output_tdb=str(tmp_path / f"ext_{Path(base).stem}.tdb"))
        t, ps = params(r["output"])
        assert ps[key].expr.short() == "-20000+3*T" and not is_tentative(ps[key], t)
        rep = [x for x in r["tentative_replaced"] if x.label == key]
        assert len(rep) == 1 and rep[0].kind == "assessed"
        assert rep[0].tentative_from == ("base" if frm == "BaseT" else "add")
        assert not r["tentative"]               # every new pair (Nb,W / Fe,W on sublattice 2) is described by add


def test_extend_phase_adds_tentative_terms(tmp_path):
    from calphad_tools.harmonize import extend_phase
    base = write(tmp_path / "FeNb.tdb", c14("FE:NB", FENB))
    add = write(tmp_path / "FeW.tdb", c14("FE:FE,W", FEW))
    r = extend_phase(base, add, "C14_LAVES", components=["W"], verify=False, output_tdb=str(tmp_path / "x.tdb"))
    assert [(t.label, t.value) for t in r["tentative"]] == [("L(C14_LAVES,FE:NB,W;0)", 150000.0)]
    t, ps = params(tmp_path / "x.tdb")
    assert is_tentative(ps["L(C14_LAVES,FE:NB,W;0)"], t) and "TENT_MIX" in t.reference_list
    assert "tentative rule 1 (TENT_MIX, new mixing): L(C14_LAVES,FE:NB,W;0) = +150000" in r["text"]
    r = extend_phase(base, add, "C14_LAVES", components=["W"], verify=False, tentative=False)
    assert not r["tentative"]


# ------------------------------------------------------------------------------------------------- citations
def test_citation_round_trip_compare_system_and_query_tdb(tmp_path):
    from calphad_tools.compare_system import compare_system, summary_text, write_csv
    from calphad_tools.tdb_query import query
    from calphad_tools.tdbtables import ref_display
    a = write(tmp_path / "FeW.tdb", c14("FE:FE,W", FEW))
    b = write(tmp_path / "FeNb.tdb", c14("FE:NB", FENB))
    m = str(tmp_path / "merged.tdb")
    merge([a, b], m, policy="first")
    text = Path(m).read_text()
    assert "LIST_OF_REFERENCES" in text and "  TENT_MIX 'Tentative parameter added by calphad_tools" in text
    assert "!" not in text[text.index("TENT_MIX 'Tentative"):].rstrip().rstrip("!")
    cit = parse_tdb(m).reference_list["TENT_MIX"]
    assert re.fullmatch(r"Tentative parameter added by calphad_tools when combining FeW, FeNb \(\d{4}-\d\d-\d\d\): "
                        r"new Fe,Nb mixing on sublattice 2 of C14_LAVES, \+50000 J/mol-atom x 3 atoms/fu; new Nb,W "
                        r"mixing on sublattice 2 of C14_LAVES, \+50000 J/mol-atom x 3 atoms/fu; not assessed - "
                        r"replace with an assessed value", cit), cit
    t = parse_tdb(m)
    tent_params = [p.key.label() for p in t.parameters if is_tentative(p, t)]
    assert tent_params == ["L(C14_LAVES,FE:FE,NB;0)", "L(C14_LAVES,FE:NB,W;0)"]
    # a hand-written tentative term (citation starts with "Tentative") counts too
    hand = write(tmp_path / "hand.tdb", c14("FE:FE,NB", FENB + [
        " PARAMETER L(C14_LAVES,FE:FE,NB;0) 298.15 90000; 6000 N MYGUESS !",
        " LIST_OF_REFERENCES\n NUMBER SOURCE\n  MYGUESS 'Tentative: my estimate'\n !"]))
    th = parse_tdb(hand)
    assert is_tentative(th.parameters[-1], th) and ref_display("MYGUESS", th.citation("MYGUESS")) == "MYGUESS · tentative"
    # compare_system: the reference sub-line says tentative
    cmp = compare_system([b, m], ["FE", "NB"], workdir=tmp_path)
    row = {x.key: x for x in cmp.written}["G(C14_LAVES,FE:FE,NB;0)"]
    lab = cmp.labels[1]
    assert row.refs[lab] == "TENT_MIX" and row.ref_text(lab) == "TENT_MIX · tentative"
    s = summary_text(cmp, {})
    assert "TENTATIVE parameters (not assessed; TENT_MIX / TENT_VA" in s and "L(C14_LAVES,FE:FE,NB;0)" in s
    csv = write_csv(cmp, str(tmp_path / "cmp.csv"))
    assert "tentative" in Path(csv).read_text()
    # query_tdb: phase mode and element mode
    txt, tables, _ = query([m], "phase", phases=["C14_LAVES"])
    assert "TENTATIVE (not assessed; TENT_MIX / TENT_VA): 2:" in txt and "TENT_MIX · tentative" in txt
    assert [r["tentative"] for r in tables["parameters"] if r["parameter"].startswith("L(C14_LAVES,FE:FE,NB")] == [True]
    txt, _, _ = query([m], "element", elements=["NB"])
    assert "TENT_MIX · tentative" in txt
    assessed = write(tmp_path / "assessed.tdb", c14("FE:FE,NB", FENB + [ASSESSED_FENB]))
    txt, tables, _ = query([assessed, m], "collisions")
    hit = [r for r in tables["collisions"] if r["parameter"] == "L(C14_LAVES,FE:FE,NB;0)"]
    assert hit and "tentative in merged: a merge takes the assessed value" in hit[0]["status"]
    assert "tentative in merged" in txt


def test_settings_and_reference_stripping():
    assert tent.settings(True) == tent.DEFAULTS and tent.settings(False) is None and tent.settings("off") is None
    assert tent.settings({"vacancy": None})["vacancy"] is None
    with pytest.raises(ValueError):
        tent.settings({"mixing": 1})
    cmd = "LIST_OF_REFERENCES\n NUMBER SOURCE\n  REF1 'A. Author, 1999'\n  TENT_MIX 'Tentative x\n   y'\n  REF2 'B'"
    assert "TENT_MIX" not in tent.strip_entries(cmd) and "REF1" in tent.strip_entries(cmd)


def test_merge_config_defaults(tmp_path):
    from calphad_tools.agent.config import MergeConfig, load_config
    p = tmp_path / "c.toml"
    p.write_text("[merge]\ntentative = true\ntentative_mixing_per_atom = 40000\ntentative_vacancy = 150000\n")
    cfg = load_config(str(p))
    assert cfg.merge.tentative_setting() == {"mixing_per_atom": 40000.0, "vacancy": 150000.0}
    assert MergeConfig(tentative=False).tentative_setting() is False
    root = Path(__file__).resolve().parents[1]
    assert load_config(str(root / "calphad_agent.example.toml")).merge.tentative_setting() == {
        "mixing_per_atom": 50000.0, "vacancy": 200000.0}


# ------------------------------------------------------------------------------------------------- combine chain
def sys_tdb(els, phases, params):
    """A small mappable database (LIQUID, BCC_A2, C14_LAVES) with the SGTE unaries of test_compatibility."""
    ELS = {"FE": " ELEMENT FE BCC_A2 55.847 4489 27.28 !", "MO": " ELEMENT MO BCC_A2 95.94 4589 28.56 !",
           "W": " ELEMENT W BCC_A2 183.84 4973 32.66 !"}
    L = [" ELEMENT VA VACUUM 0 0 0 !"] + [ELS[e] for e in els]
    L += [f" FUNCTION {FNAME[e]} 298.15 {GHSER[e]} !" for e in els]
    L += [" TYPE_DEFINITION % SEQ * !"]
    return "\n".join(L + phases + params) + "\n"


def chain_sources(tmp_path):
    liq = {"FE": "12040.17-6.55843*T+GHSERFE", "MO": "34085.045-12.47*T+GHSERMO", "W": "52160.584-14.10999*T+GHSERWW"}

    def base(els, c14_cons, c14_params, extra=()):
        ph = [" PHASE LIQUID % 1 1 !", f" CONSTITUENT LIQUID :{','.join(els)}: !", " PHASE BCC_A2 % 2 1 3 !",
              f" CONSTITUENT BCC_A2 :{','.join(els)}:VA: !", " PHASE C14_LAVES % 2 2 1 !",
              f" CONSTITUENT C14_LAVES :{c14_cons}: !"]
        pr = [f" PARAMETER G(LIQUID,{e};0) 298.15 {liq[e]}; 6000 N SGTE !" for e in els]
        pr += [f" PARAMETER G(BCC_A2,{e}:VA;0) 298.15 {FNAME[e]}; 6000 N SGTE !" for e in els]
        return sys_tdb(els, ph, pr + list(c14_params) + list(extra))
    few = base(["FE", "W"], "FE:FE,W", [" PARAMETER G(C14_LAVES,FE:FE;0) 298.15 3*GHSERFE+5000; 6000 N A1 !",
                                         " PARAMETER G(C14_LAVES,FE:W;0) 298.15 2*GHSERFE+GHSERWW-30000; 6000 N A1 !",
                                         " PARAMETER L(C14_LAVES,FE:FE,W;0) 298.15 -5000; 6000 N A1 !"],
               [" PARAMETER L(LIQUID,FE,W;0) 298.15 -15000; 6000 N A1 !",
                " PARAMETER L(BCC_A2,FE,W:VA;0) 298.15 10000; 6000 N A1 !"])
    femo_bin = [" PARAMETER L(LIQUID,FE,MO;0) 298.15 -6973-0.37*T; 6000 N A2 !",
                " PARAMETER L(BCC_A2,FE,MO:VA;0) 298.15 36818-9.141*T; 6000 N A2 !",
                " PARAMETER G(C14_LAVES,FE:MO;0) 298.15 2*GHSERFE+GHSERMO-45000+2*T; 6000 N A2 !"]
    femo = base(["FE", "MO"], "FE:MO", [], femo_bin)
    femo2 = base(["FE", "MO"], "FE:FE,MO", [" PARAMETER G(C14_LAVES,FE:FE;0) 298.15 3*GHSERFE+5000; 6000 N A3 !",
                                             " PARAMETER L(C14_LAVES,FE:FE,MO;0) 298.15 +20000; 6000 N A3 !"],
                 femo_bin)
    return [write(tmp_path / "S0_FeW.tdb", few), write(tmp_path / "S1_FeMo.tdb", femo),
            write(tmp_path / "S2_FeMoC14.tdb", femo2)]


def test_combine_chain_replaces_step1_tentative_term(tmp_path):
    from calphad_tools.combine import combine_databases, plan_combination, plan_text
    srcs = chain_sources(tmp_path)
    # the plan lists what each step will add / replace
    plan = plan_combination(srcs, str(tmp_path / "plan"), order=srcs, unary50="off")
    t = plan_text(plan)
    assert "TENTATIVE step 1: rule 1 (new mixing, TENT_MIX) will add L(C14_LAVES,FE:FE,MO;0) = +150000" in t
    assert "TENTATIVE step 2: replaces L(C14_LAVES,FE:FE,MO;0) TENT_MIX" in t
    r = combine_databases(srcs, str(tmp_path / "c"), order=srcs, unary50="off", n_workers=4, tmax_c=1800.0,
                          composition_step=0.05, temperature_step=25.0, lint=False)
    assert r["written"], r["text"]
    st = r["state"]
    q1 = parse_tdb(st["steps"][0]["Q"])
    p1 = {p.key.label(): p for p in q1.parameters}
    assert is_tentative(p1["L(C14_LAVES,FE:FE,MO;0)"], q1) and p1["L(C14_LAVES,FE:FE,MO;0)"].expr.short() == "+150000"
    F = parse_tdb(st["final"])
    pf = {p.key.label(): p for p in F.parameters}
    assert pf["L(C14_LAVES,FE:FE,MO;0)"].expr.short() == "+20000" and not is_tentative(pf["L(C14_LAVES,FE:FE,MO;0)"], F)
    assert is_tentative(pf["L(C14_LAVES,FE:MO,W;0)"], F)          # still needed: nobody assesses Mo,W on sublattice 2
    log = {e["id"]: e for e in st["log"]}
    add1 = log["step1:rule:tentative:L(C14_LAVES,FE:FE,MO;0)"]
    assert add1["who"] == "rule" and add1["kind"] == "tentative" and "rule 1 (TENT_MIX)" in add1["reason"]
    assert any("G(C14_LAVES,FE:FE) from" in x and "no source mixes Fe,Mo" in x for x in add1["evidence"])
    rep2 = log["step2:rule:tentative-replaced:L(C14_LAVES,FE:FE,MO;0)"]
    assert rep2["who"] == "rule" and rep2["chosen"] == "TENT_MIX (Q_S0_FeW+S1_FeMo.tdb) -> assessed value from " \
                                                       "S2_FeMoC14.tdb"
    # validation: S1's Fe-Mo changed by the C14 union + the tentative term (explained), never unexplained
    rows = {(x["source"], x["subsystem"]): x for x in st["validation"]["rows"]}
    assert not [x for x in rows.values() if x["verdict"] == "CHANGED, UNEXPLAINED"]
    md = Path(st["files"]["report"]).read_text()
    assert "## Tentative parameters" in md and "L(C14_LAVES,FE:MO,W;0)" in md
    assert "Tentative" in r["text"]
    json.dumps(st)                                                       # the state stays JSON
