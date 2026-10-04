"""A real three-database merge: Fe-Nb (Huang 1990) + Fe-W (Jacob 2015) + Nb-W (Huang 1997), Hallstedt's
datasets from phasediagrams.org (examples/tdbs).

* MU_D85 is (FE)7(NB)2(FE,NB)4 in Fe-Nb and (FE,W)1(W)4(FE,W)2(FE,W)6 in Fe-W: evaluate_compatibility reports it
  instead of stopping, and the merge decision 'rename' keeps Fe-W's model as MU_D85_2 (phase, constituents,
  parameters) so Fe-W in Q is B's Fe-W diagram.
* C14_LAVES is (FE)2(NB) in Fe-Nb and (FE,W)2(FE,W) in Fe-W: B's FE:FE end-member widens Fe-Nb's C14.
* Q + Nb-W: C14 enters Nb-W as (W)2(NB,W) with no G for W:NB - a warning in the overall verdict.
"""
import shutil
import warnings
from pathlib import Path

import pytest

from calphad_tools.compatibility import run as compat_run, summary_text as compat_summary
from calphad_tools.staged_merge import staged_merge, summary_text

from test_compatibility import A_FeMo, B_FeMoW, write

EX = Path(__file__).resolve().parent.parent / "examples" / "tdbs"
FENB, FEW, NBW = (str(EX / n) for n in ("FeNb-90Hua-LB.tdb", "FeW-15Jac-static.tdb", "NbW-97Hua-LB.tdb"))
FAST = dict(ternary_temperatures_c=[], n_workers=4)


@pytest.fixture(scope="module")
def fenb_few(tmp_path_factory):
    d = tmp_path_factory.mktemp("trio")
    return staged_merge(FENB, FEW, str(d / "out"), decisions={"phase:MU_D85": "rename"}, **FAST)


def _c14_xnb(plot):
    import json
    data = json.loads(Path(plot["png"]).with_suffix(".json").read_text())
    xs = [v for reg in data["regions"] for ph, r in reg["X_ranges"].items() if ph.startswith("C14") for v in r["X(NB)"]]
    return min(xs), max(xs)


def test_rename_keeps_mu_d85_2_and_fe_w_unchanged(fenb_few):
    r = fenb_few
    assert r["written"] and r["phase_renames"] == {"MU_D85": "MU_D85_2"}
    q = r["versions"]["Q"].path
    text = Path(q).read_text()
    assert " PHASE MU_D85_2 % 4 1 4 2 6 !" in text and "CONSTITUENT MU_D85_2 :FE,W : W : FE,W : FE,W :" in text
    assert "refers to undefined phase" not in text
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(q)
    assert {"MU_D85", "MU_D85_2"} <= set(db.phases)
    assert len([p for p in db._parameters.all() if p["phase_name"] == "MU_D85_2"]) == 8
    assert db.phases["MU_D85_2"].sublattices == (1.0, 4.0, 2.0, 6.0)
    # B's Fe-W and Q's Fe-W: the same invariants (μ called MU_D85_2 in Q), within 0.5 K
    inv_b = r["plots"][("B", "Fe-W")]["invariants"]
    inv_q = r["plots"][("Q", "Fe-W")]["invariants"]
    assert not r["plots"][("Q", "Fe-W")].get("reused_from")         # really mapped again, as a check
    key = lambda inv, ren: (" + ".join(sorted(ren.get(p, p) for p in inv["phases"])), inv["T_K"])  # noqa: E731
    kb = sorted(key(i, {"MU_D85": "MU_D85_2"}) for i in inv_b)
    kq = sorted(key(i, {}) for i in inv_q)
    assert len(kb) == len(kq) == 3 and [k for k, _ in kb] == [k for k, _ in kq]
    assert all(abs(tb - tq) <= 0.5 for (_, tb), (_, tq) in zip(kb, kq))
    assert any("MU_D85_2" in k for k, _ in kq) and min(abs(t - 1333.0) for _, t in kq) < 1.0
    # the report: kept under a new name, compared as the same phase, nothing "removed"
    st = next(s for s in r["steps"] if s.title == "B3 → Q")
    assert not st.diff.changed and not st.changed_items
    assert st.forced_shift("Fe-W")[2] and "MU_D85 kept as MU_D85_2" in st.forced_kind["Fe-W"]
    s = summary_text(r)
    assert "Kept under a new name: MU_D85 of FeW-15Jac-static.tdb is MU_D85_2 in Q" in s
    assert "MU_D85 removed" not in s and "FAILED" not in s


def test_c14_fe_fe_widens_fe_nb(fenb_few, tmp_path):
    r = fenb_few
    a_min, a_max = _c14_xnb(r["plots"][("A", "Fe-Nb")])
    a1_min, a1_max = _c14_xnb(r["plots"][("A1", "Fe-Nb")])
    assert abs(a_min - 1 / 3) < 1e-3 and abs(a_max - 1 / 3) < 1e-3
    assert a1_min < 0.33 and abs(a1_max - 1 / 3) < 1e-3
    st = next(s for s in r["steps"] if s.title == "A → A1")
    m = [x for x in st.diff.get(("FE", "NB")).models if x.phase == "C14_LAVES"][0]
    assert m.old == "(FE)2(NB)1" and m.new == "(FE)2(FE,NB)1" and m.new_endmembers == ["FE:FE"] and not m.missing_g
    assert "G(C14_LAVES,FE:FE;0)" in {p.key for p in st.diff.params}
    ev = compat_run(FENB, FEW, str(tmp_path / "compat"), unary50="off")
    assert "new FE:FE in A's space (Fe-Nb; G from B)" in ev.stages[4].line


def test_evaluate_without_sublattice_map_reports_mu_d85(tmp_path):
    ev = compat_run(FENB, FEW, str(tmp_path / "compat"), unary50="off")
    text = compat_summary(ev)
    assert "MU_D85: incompatible A (FE)7(NB)2(FE,NB)4 has 3 non-vacancy sublattices" in text
    s5 = ev.stages[4]
    assert s5.verdict == "INCOMPATIBLE"
    assert "MU_D85: site ratios 7:2:4 (A) vs 1:4:2:6 (B) - INCOMPATIBLE" in s5.line
    assert "{'phase:MU_D85': 'rename'}" in s5.line
    assert [s.n for s in ev.stages] == [1, 2, 3, 4, 5, 6] and ev.overall.startswith("INCOMPATIBLE")
    # staged_merge asks in the decisions format (no traceback)
    r = staged_merge(FENB, FEW, str(tmp_path / "sm"), plot=False)
    s = summary_text(r)
    assert not r["written"] and "* phase:MU_D85:" in s and "'rename'" in s and "'db1'" in s


def test_zero_g_end_member_warning_q_plus_nb_w(fenb_few, tmp_path):
    q = shutil.copy(fenb_few["versions"]["Q"].path, tmp_path / "FeNbW_Q1.tdb")
    ev = compat_run(str(q), NBW, str(tmp_path / "compat"), unary50="off")
    assert ev.stages[4].verdict == "WARNING"
    assert "C14_LAVES (W)2(NB,W)1 [no G for W:NB: pycalphad uses 0]" in ev.stages[4].line
    assert ev.overall.startswith("COMPATIBLE with warnings: C14_LAVES W:NB has no G (pycalphad uses 0")
    r = staged_merge(str(q), NBW, str(tmp_path / "sm"), **FAST)
    st = next(s for s in r["steps"] if s.title == "B2 → B3")
    assert st.zero_g and "C14_LAVES W:NB: C14_LAVES not stable in the mapped Nb-W" in st.zero_g[0]
    assert "no G (pycalphad uses 0): C14_LAVES W:NB" in summary_text(r)


def test_zero_g_severity_rule(tmp_path):
    """No G for a new end-member: INCOMPATIBLE where that database assessed the phase in the binary, else WARNING."""
    a = write(tmp_path, "A_FeMo.tdb", A_FeMo())
    b = write(tmp_path, "B.tdb", B_FeMoW().replace(" PARAMETER G(C14,FE:FE;0) 298.15 3*GHSERFE+5000; 6000 N !\n", ""))
    ev = compat_run(a, b, str(tmp_path / "c1"), unary50="off")
    rows = {r["new end-member"]: r for r in ev.stages[4].rows if r.get("new end-member")}
    # FE:FE lies in A's Fe-Mo, where A assessed C14 (G(C14,FE:MO)): the zero G changes it
    assert "changes A's assessed C14 in Fe-Mo" in rows["FE:FE"]["provided by"]
    assert ev.stages[4].verdict == "INCOMPATIBLE"
    # W:MO lies in Mo-W, where neither database assessed C14 (B has only W:W there): a warning
    ev = compat_run(a, write(tmp_path, "B2.tdb", B_FeMoW()), str(tmp_path / "c2"), unary50="off")
    assert ev.stages[4].verdict == "WARNING" and ev.overall.startswith("COMPATIBLE with warnings: C14 W:MO has no G")


def test_ambiguous_sublattices_reported_not_raised(tmp_path):
    from test_compatibility import HDR, tdb_text
    a = write(tmp_path, "a.tdb", tdb_text(HDR, [" PHASE PH11 % 2 1 1 !", " CONSTITUENT PH11 :FE:MO: !"], []))
    b = write(tmp_path, "b.tdb", tdb_text(HDR, [" PHASE PH11 % 2 1 1 !", " CONSTITUENT PH11 :FE,MO:W: !"], []))
    ev = compat_run(a, b, str(tmp_path / "c"), unary50="off")
    assert ev.stages[4].verdict == "INCOMPATIBLE" and "PH11: sublattice mapping undecided" in ev.stages[4].line
    assert "sublattice_map {'PH11': [...]}" in ev.stages[4].line
    r = staged_merge(a, b, str(tmp_path / "sm"), plot=False)
    s = summary_text(r)
    assert not r["written"] and "* sublattice_map:PH11:" in s and "{'PH11': 'keep'}" in s
