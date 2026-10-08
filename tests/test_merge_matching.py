"""How the merge tools match phases: site ratios written as rounded fractions are read exactly (0.333:0.667 = 1/3:2/3,
k = 3 against 1:2, no question, one rule entry), sublattices are flipped automatically when the snapped normalised
ratios say so, and pairing is one-to-one (a phase paired with one of the other database is never offered again;
competing candidates are one question).  Small synthetic Fe-Mo / Fe-Mo-W databases.

Run:  .venv/bin/python -m pytest -q tests/test_merge_matching.py
"""
import re
import warnings
from pathlib import Path

import pytest

from calphad_tools import tdbstage as ts
from calphad_tools.tdbparse import parse_tdb, snap_ratios

from test_compatibility import HDR, tdb_text, write

# ------------------------------------------------------------------------------------------------ databases
A_C15 = tdb_text(HDR, [" PHASE C15 % 2 1 2 !", " CONSTITUENT C15 :MO:FE: !"],
                 [" PARAMETER G(C15,MO:FE;0) 298.15 GHSERMO+2*GHSERFE-15000; 6000 N !"])
B_PARAMS = [" PARAMETER G(C15,W:FE;0) 298.15 (GHSERWW+2*GHSERFE-30000+6*T)/3; 6000 N !",
            " PARAMETER G(C15,W:W;0) 298.15 GHSERWW+5000; 6000 N !",
            " PARAMETER L(C15,W:FE,W;0) 298.15 -4000+T; 6000 N !"]


def b_c15(ratios: str, flip: bool = False) -> str:
    """B's C15 per atom: (W)r1(FE,W)r2, or written the other way round, (FE,W)r2(W)r1."""
    if not flip:
        return tdb_text(HDR, [f" PHASE C15 % 2 {ratios} !", " CONSTITUENT C15 :W:FE,W: !"], B_PARAMS)
    params = [p.replace("W:FE;0", "FE:W;0").replace("W:FE,W;0", "FE,W:W;0") for p in B_PARAMS]
    return tdb_text(HDR, [f" PHASE C15 % 2 {ratios} !", " CONSTITUENT C15 :FE,W:W: !"], params)


def _norm(tmp_path, a_text, b_text, **kw):
    a, b = write(tmp_path, "A.tdb", a_text), write(tmp_path, "B.tdb", b_text)
    An, Bn, notes = ts.normalize_pair(parse_tdb(a), parse_tdb(b), **kw)
    an, bn = ts.write_tdb(An, str(tmp_path / "An.tdb")), ts.write_tdb(Bn, str(tmp_path / "Bn.tdb"))
    return a, b, an, bn, notes


def _validate(before, after, phase):
    from calphad_tools.tdb_validate import validate_edit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = validate_edit(before, after, affected_phases=[phase], expect=["energy_per_atom_preserved"])
    return r, next(x for x in r["gm"] if x["phase"] == phase) if r.get("gm") else None


# ------------------------------------------------------------------------------------------------ 1. snapping
def test_snap_ratios_rule():
    cases = {(0.333, 0.667): ([1, 2], True), (0.6667, 0.3333): ([2, 1], True), (0.25, 0.75): ([1, 3], False),
             (0.105, 0.895): ([2, 17], True), (0.2632, 0.7368): ([5, 14], True),
             (0.538, 0.154, 0.308): ([7, 2, 4], True), (7, 2, 4): ([7, 2, 4], False), (1, 2): ([1, 2], False),
             (0.4, 0.6): ([2, 3], False)}
    for r, (ints, changed) in cases.items():
        s = snap_ratios(r)
        assert s is not None and s.integers == ints and s.changed == changed, r
    assert snap_ratios([0.41, 0.59]) is None                      # not a rounded simple fraction: left as written
    assert snap_ratios([0.333, 0.667]).values == [0.333333333333333, 0.666666666666667]
    assert snap_ratios([0.4, 0.6]).values == [0.4, 0.6]           # 2:3 exactly: snaps to itself
    assert snap_ratios([0.538, 0.154, 0.308]).text() == "7/13:2/13:4/13"
    assert snap_ratios([0.105, 0.895]).text() == "2/19:17/19"


def test_rounded_ratios_same_order_proportional_k3(tmp_path):
    """A (MO)1(FE)2 vs B (W)0.333(FE,W)0.667: proportional, k = 3 exactly, B written 1:2, no question, one rule note
    with a tiny GM change; validate_edit passes (exact once the ratios are read exactly)."""
    a, b, an, bn, notes = _norm(tmp_path, A_C15, b_c15("0.333 0.667"))
    m = ts.sublattice_mapping(parse_tdb(a).phases["C15"], parse_tdb(b).phases["C15"])
    assert m.k == 3.0 and m.perm == [0, 1] and m.mode == "snapped" and m.a_snap is None
    assert [n.action for n in notes] == ["snapped", "rescaled"]
    snaps = ts.snap_notes(notes)
    assert len(snaps) == 1 and "0.333:0.667 read as 1/3:2/3" in snaps[0].detail
    gm = float(re.search(r"GM changes by at most ([0-9.e+-]+) J/mol", snaps[0].detail).group(1))
    assert 0 < gm < 10                                            # tiny: configurational-entropy weights only
    assert " PHASE C15 % 2 1 2 !" in Path(bn).read_text()          # the merged phase: exact ratios
    assert parse_tdb(bn).phases["C15"].site_ratios == [1.0, 2.0]
    assert not ts.a_changed(notes) and ts.b_changed(notes)
    r, row = _validate(b, bn, "C15")
    assert row["status"] == "pass" and "read exactly" in row["detail"], row
    eff = float(re.search(r"that alone changes GM by max ([0-9.e+-]+) J/mol", row["detail"]).group(1))
    assert 0 < eff < gm
    # the summary is silent about it (no warning, not in the 'Normalised' list), the Excel sheet has it
    from calphad_tools.compatibility import evaluate_pair, summary_text
    ev = evaluate_pair(a, b, unary50="off", workdir=str(tmp_path / "w"))
    s = summary_text(ev)
    assert "read as" not in s and "snapped" not in s and "C15: site ratios" not in ev.stages[4].line
    assert ev.stages[4].verdict in ("OK", "CHANGES", "WARNING")
    assert any(n.action == "snapped" for n in ev.norm_notes)


def test_rounded_ratios_flipped_automatically(tmp_path):
    """B written (FE,W)0.667(W)0.333 against A (MO)1(FE)2: normalised 2/3:1/3 vs 1/3:2/3 -> flipped [1, 0], k = 3;
    no SublatticeMapError, no question; GM checked with validate_edit."""
    a, b, an, bn, notes = _norm(tmp_path, A_C15, b_c15("0.667 0.333", flip=True))
    m = ts.sublattice_mapping(parse_tdb(a).phases["C15"], parse_tdb(b).phases["C15"])
    assert m.flipped and m.perm == [1, 0] and m.k == 3.0
    assert [n.action for n in notes] == ["snapped", "reordered", "rescaled"]
    assert "flipped sublattices to match A's order" in notes[1].detail
    ph = parse_tdb(bn).phases["C15"]
    assert ph.site_ratios == [1.0, 2.0] and ph.constituents == [["W"], ["FE", "W"]]
    assert {p.raw_name for p in parse_tdb(bn).parameters if p.key.phase == "C15"} == {
        "G(C15,W:FE;0)", "G(C15,W:W;0)", "L(C15,W:FE,W;0)"}
    r, row = _validate(b, bn, "C15")
    assert row["status"] == "pass" and row["perm"] == "1,0", row
    # extend_phase uses the same matcher: A's C15 extended with B's W, flipped and snapped, GM checks exact
    from calphad_tools import harmonize as hz
    rep = hz.extend_phase(a, b, "C15", components=["W"], output_tdb=str(tmp_path / "ext.tdb"))
    assert rep["new_from_old"] == [1, 0] and parse_tdb(str(tmp_path / "ext.tdb")).phases["C15"].site_ratios == [1, 2]
    assert any("read as 1/3:2/3" in x for x in rep["snaps"]) and "site ratios read exactly" in rep["text"]
    assert rep["checks"]["base"]["max_rel"] <= 1e-9
    assert all(c["max_rel"] <= 1e-9 for c in rep["checks"]["add"] if c.get("expected_exact"))


def test_sum_to_one_written_exactly(tmp_path):
    """A's 0.333:0.667 (base basis, per atom) read as 1/3:2/3 and written 0.333333333333333 / 0.666666666666667
    (pycalphad's sum exactly 1); B's 1:2 rescaled onto it (k = 1/3)."""
    a_text = A_C15.replace(" PHASE C15 % 2 1 2 !", " PHASE C15 % 2 0.333 0.667 !").replace(
        "GHSERMO+2*GHSERFE-15000", "(GHSERMO+2*GHSERFE-15000)/3")
    b_text = b_c15("1 2").replace("(GHSERWW+2*GHSERFE-30000+6*T)/3", "GHSERWW+2*GHSERFE-30000+6*T").replace(
        "GHSERWW+5000", "3*GHSERWW+15000").replace("-4000+T", "-12000+3*T")
    a, b, an, bn, notes = _norm(tmp_path, a_text, b_text)
    assert [n.action for n in notes] == [ts.A_SNAPPED, "rescaled"] and ts.a_changed(notes)
    exact = [0.333333333333333, 0.666666666666667]
    assert parse_tdb(an).phases["C15"].site_ratios == exact and parse_tdb(bn).phases["C15"].site_ratios == exact
    assert " PHASE C15 % 2 0.333333333333333 0.666666666666667 !" in Path(an).read_text()
    from calphad_tools.harmonize import pyc_db
    assert sum(pyc_db(parse_tdb(bn)).phases["C15"].sublattices) == 1.0
    # the formula-unit tool reads it the same way
    from calphad_tools import harmonize as hz
    r = hz.normalize_phase_formula_unit(a, ["C15"], "sum_to_one", str(tmp_path / "one.tdb"))
    assert " PHASE C15 % 2 0.333333333333333 0.666666666666667 !" in (tmp_path / "one.tdb").read_text()
    assert "read as 1/3:2/3" in r["text"] and r["phases"][0]["pycalphad_sum"] == 1.0
    r2 = hz.normalize_phase_formula_unit(a, ["C15"], "integers", str(tmp_path / "int.tdb"))
    assert r2["phases"][0]["new"] == [1.0, 2.0] and r2["phases"][0]["gm"]["max_rel"] <= 1e-9


def test_odd_ratios_stay_a_model_difference(tmp_path):
    a, b, an, bn, notes = _norm(tmp_path, A_C15, b_c15("0.41 0.59"))
    assert ts.sublattice_mapping(parse_tdb(a).phases["C15"], parse_tdb(b).phases["C15"]) is None
    assert [n.action for n in notes] == ["incompatible"] and "not proportional" in notes[0].detail
    assert parse_tdb(bn).phases["C15"].site_ratios == [0.41, 0.59]
    from calphad_tools.compatibility import evaluate_pair
    ev = evaluate_pair(a, b, unary50="off", workdir=str(tmp_path / "w"))
    assert ev.stages[4].verdict == "INCOMPATIBLE" and "C15: site ratios 1:2 (A) vs 0.41:0.59 (B)" in ev.stages[4].line


def test_ambiguous_equal_size_still_asks(tmp_path):
    """Equal-size sublattices whose element overlap points both ways (or shares nothing): still an error / question,
    also when the ratios are written rounded."""
    a = tdb_text(HDR, [" PHASE PH11 % 2 1 1 !", " CONSTITUENT PH11 :FE:MO: !"], [])
    for ratios in ("1 1", "0.5 0.5", "0.333 0.333"):
        b = tdb_text(HDR, [f" PHASE PH11 % 2 {ratios} !", " CONSTITUENT PH11 :FE,MO:W: !"], [])
        with pytest.raises(ts.SublatticeMapError, match="points both ways"):
            _norm(tmp_path, a, b)
    b = tdb_text(HDR, [" PHASE PH11 % 2 1 1 !", " CONSTITUENT PH11 :FE,MO:W: !"],
                 [" PARAMETER G(PH11,FE:W;0) 298.15 GHSERFE+GHSERWW; 6000 N !"])
    from calphad_tools.combine import plan_combination
    pa, pb = write(tmp_path, "amb_a.tdb", a), write(tmp_path, "amb_b.tdb", b)
    plan = plan_combination([pa, pb], str(tmp_path / "plan"), order=[pa, pb], unary50="off")
    assert "step1:sublattice:PH11" in [n.id for n in plan.pending]


# ------------------------------------------------------------------------------------------------ 2. combine (log)
@pytest.fixture(scope="module")
def snapped_combine(tmp_path_factory):
    from calphad_tools.combine import combine_databases
    d = tmp_path_factory.mktemp("snap_combine")
    a = tdb_text(["FE", "MO"], [" PHASE LIQUID % 1 1 !", " CONSTITUENT LIQUID :FE,MO: !", " PHASE BCC_A2 %B 2 1 3 !",
                                " CONSTITUENT BCC_A2 :FE,MO:VA: !", " PHASE C15 % 2 1 2 !",
                                " CONSTITUENT C15 :MO:FE: !"],
                 [" PARAMETER G(LIQUID,FE;0) 298.15 12040.17-6.55843*T+GHSERFE; 6000 N !",
                  " PARAMETER G(LIQUID,MO;0) 298.15 34085.045-12.47*T+GHSERMO; 5000 N !",
                  " PARAMETER G(BCC_A2,FE:VA;0) 298.15 GHSERFE; 6000 N !",
                  " PARAMETER G(BCC_A2,MO:VA;0) 298.15 GHSERMO; 5000 N !",
                  " PARAMETER L(LIQUID,FE,MO;0) 298.15 -6973-0.37*T; 6000 N !",
                  " PARAMETER L(BCC_A2,FE,MO:VA;0) 298.15 36818-9.141*T; 6000 N !",
                  " PARAMETER G(C15,MO:FE;0) 298.15 GHSERMO+2*GHSERFE-15000; 6000 N !"])
    b = tdb_text(["FE", "W"], [" PHASE LIQUID % 1 1 !", " CONSTITUENT LIQUID :FE,W: !", " PHASE BCC_A2 %B 2 1 3 !",
                               " CONSTITUENT BCC_A2 :FE,W:VA: !", " PHASE C15 % 2 0.333 0.667 !",
                               " CONSTITUENT C15 :W:FE,W: !"],
                 [" PARAMETER G(LIQUID,FE;0) 298.15 12040.17-6.55843*T+GHSERFE; 6000 N !",
                  " PARAMETER G(LIQUID,W;0) 298.15 52160.584-14.10999*T+GHSERWW; 6000 N !",
                  " PARAMETER G(BCC_A2,FE:VA;0) 298.15 GHSERFE; 6000 N !",
                  " PARAMETER G(BCC_A2,W:VA;0) 298.15 GHSERWW; 6000 N !",
                  " PARAMETER L(LIQUID,FE,W;0) 298.15 -15000; 6000 N !",
                  " PARAMETER L(BCC_A2,FE,W:VA;0) 298.15 10000; 6000 N !"] + B_PARAMS)
    srcs = [write(d, "FeMo_c15.tdb", a), write(d, "FeW_c15.tdb", b)]
    r = combine_databases(srcs, str(d / "c"), order=srcs, unary50="off", n_workers=3, lint=False, tmax_c=1800,
                          composition_step=0.05, temperature_step=25)
    return r


def test_combine_snap_is_one_rule_entry(snapped_combine):
    r = snapped_combine
    assert r["written"], r["text"]
    st = r["state"]
    assert not any("C15" in n["id"] for n in st["steps"][0]["needed"])          # no question about C15
    snaps = [e for e in st["log"] if e["id"].endswith(":snapped")]
    assert len(snaps) == 1 and snaps[0]["id"] == "step1:norm:C15:snapped" and snaps[0]["who"] == "rule"
    ev = " | ".join(snaps[0]["evidence"])
    assert "0.333:0.667 read as 1/3:2/3" in ev and "the snap alone changes GM" in ev
    rel = float(re.search(r"by max relative ([0-9.e+-]+)", ev).group(1))
    assert rel < 1e-3                                                           # tiny
    ex = [x for x in st["exactness"] if x["phase"] == "C15"]
    assert ex and all(x["ok"] for x in ex)                                      # exact against the exact-fraction B
    F = parse_tdb(st["final"])
    assert F.phases["C15"].site_ratios == [1.0, 2.0]
    rows = st["validation"]["rows"]
    assert all(x["verdict"] != "CHANGED, UNEXPLAINED" for x in rows), [(x["subsystem"], x["verdict"]) for x in rows]
    assert "read as" not in r["text"]                         # silent in the reply: a rule entry of the log only


# ------------------------------------------------------------------------------------------------ 3. one-to-one pairing
LAVES_A = tdb_text(["FE", "MO"], [
    " PHASE LIQUID % 1 1 !", " CONSTITUENT LIQUID :FE,MO: !", " PHASE LAVES % 2 1 2 !", " CONSTITUENT LAVES :MO:FE: !",
    " PHASE C14_LAVES % 2 1 3 !", " CONSTITUENT C14_LAVES :MO:FE: !"], [
    " PARAMETER G(LIQUID,FE;0) 298.15 12040.17-6.55843*T+GHSERFE; 6000 N !",
    " PARAMETER G(LIQUID,MO;0) 298.15 34085.045-12.47*T+GHSERMO; 5000 N !",
    " PARAMETER G(LAVES,MO:FE;0) 298.15 GHSERMO+2*GHSERFE-15000; 6000 N !",
    " PARAMETER G(C14_LAVES,MO:FE;0) 298.15 GHSERMO+3*GHSERFE-12000; 6000 N !"])
LAVES_B = tdb_text(["FE", "MO", "W"], [
    " PHASE LIQUID % 1 1 !", " CONSTITUENT LIQUID :FE,MO,W: !", " PHASE LAVES_C15 % 2 1 2 !",
    " CONSTITUENT LAVES_C15 :MO,W:FE: !", " PHASE LAVES_C14 % 2 1 2 !", " CONSTITUENT LAVES_C14 :MO:FE: !"], [
    " PARAMETER G(LIQUID,FE;0) 298.15 12040.17-6.55843*T+GHSERFE; 6000 N !",
    " PARAMETER G(LIQUID,MO;0) 298.15 34085.045-12.47*T+GHSERMO; 5000 N !",
    " PARAMETER G(LIQUID,W;0) 298.15 52160.584-14.10999*T+GHSERWW; 6000 N !",
    " PARAMETER G(LAVES_C15,MO:FE;0) 298.15 GHSERMO+2*GHSERFE-15000; 6000 N !",
    " PARAMETER G(LAVES_C15,W:FE;0) 298.15 GHSERWW+2*GHSERFE-21000; 6000 N !",
    " PARAMETER G(LAVES_C14,MO:FE;0) 298.15 GHSERMO+2*GHSERFE-14000; 6000 N !"])


@pytest.fixture(scope="module")
def laves(tmp_path_factory):
    d = tmp_path_factory.mktemp("laves")
    return d, write(d, "laves_a.tdb", LAVES_A), write(d, "laves_b.tdb", LAVES_B)


def _plan(laves, tag, **kw):
    from calphad_tools.combine import plan_combination
    d, a, b = laves
    return plan_combination([a, b], str(d / tag), order=[a, b], unary50="off", **kw)


def test_competing_candidates_are_one_question(laves):
    """No decisions: LAVES_C15 (extend: it brings W) and LAVES_C14 (same model) both look like LAVES -> ONE question
    for LAVES with both as options; LAVES_C14 ~ C14_LAVES (same structure token) is its own question."""
    plan = _plan(laves, "p0")
    ids = [n.id for n in plan.pending]
    assert ids == ["step1:name:LAVES", "step1:name:LAVES_C14"], ids
    q = plan.pending[0]
    keys = [k for k, _ in q.options]
    assert set(keys) == {"extend:LAVES_C15", "LAVES_C15", "LAVES_C14", "separate"} and q.kind == "harmonize"
    assert "at most one" in q.question
    assert "C14_LAVES" in plan.pending[1].question


@pytest.mark.parametrize("how", ["phase_map", "grouped_answer", "single_answer", "extend"])
def test_paired_phase_never_offered_again(laves, how):
    """LAVES (db1) paired with LAVES_C15 (db2) - by phase_map, by answering the grouped question, by the single-id
    answer 'name:LAVES_C15' = 'same', or by an 'extend' decision: no 'LAVES with LAVES_C14?' question remains (plan
    and compatibility stage 5); LAVES_C14 is still offered for db1's unpaired C14_LAVES."""
    kw = {"phase_map": {"phase_map": {"LAVES_C15": "LAVES"}},
          "grouped_answer": {"decisions": {"step1:name:LAVES": "LAVES_C15"}},
          "single_answer": {"decisions": {"step1:name:LAVES_C15": "same"}},
          "extend": {"decisions": {"step1:phase:LAVES": {"extend": {"phase": "LAVES_C15"}}}}}[how]
    plan = _plan(laves, f"p_{how}", **kw)
    pend = plan.pending
    assert [n.id for n in pend] == ["step1:name:LAVES_C14"], [n.id for n in pend]
    assert "C14_LAVES" in pend[0].question and "LAVES's" not in pend[0].question
    assert not any(re.search(r"\bLAVES\b", k) for k, _ in pend[0].options)
    st = plan.steps[0]
    if how == "extend":
        assert st.extensions["LAVES"]["phase"] == "LAVES_C15"
    else:
        assert st.phase_map.get("LAVES_C15") == "LAVES"
    rows = [r for r in st.compat.stages[4].rows if r.get("name candidate")]
    assert all(not re.search(r"~ (A's )?LAVES\b", r["name candidate"]) for r in rows), rows
    assert all(r["phase"] not in ("LAVES", "LAVES_C15") for r in rows)
    if how in ("grouped_answer", "single_answer"):                # the answer is kept (decision log), not re-asked
        answered = [n for n in plan.needed if n.answer is not None and n.kind in ("name", "harmonize")]
        assert answered and all(n.who != "policy" for n in answered)


def test_auto_alias_pairs_before_candidates(tmp_path):
    """db1 C15_LAVES ~ db2 LAVES_C15 is a clear alias (paired automatically); LAVES_C14 is then not offered for
    C15_LAVES, only for db1's C14_LAVES."""
    a_text = re.sub(r"\bLAVES\b", "C15_LAVES", LAVES_A)
    a, b = write(tmp_path, "al_a.tdb", a_text), write(tmp_path, "al_b.tdb", LAVES_B)
    assert "C15_LAVES" in parse_tdb(a).phases
    _, _, notes = ts.normalize_pair(parse_tdb(a), parse_tdb(b))
    assert any(n.action == "renamed" and n.phase == "LAVES_C15" for n in notes)
    assert not any(n.action == "candidate" and "C15_LAVES" in n.detail for n in notes)
    from calphad_tools.combine import plan_combination
    plan = plan_combination([a, b], str(tmp_path / "plan"), order=[a, b], unary50="off")
    assert [n.id for n in plan.pending] == ["step1:name:LAVES_C14"]
    assert "C14_LAVES" in plan.pending[0].question and "C15_LAVES" not in plan.pending[0].question
