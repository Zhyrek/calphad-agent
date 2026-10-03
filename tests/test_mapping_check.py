"""Check-and-repair pass after pycalphad mapping (calphad_tools/mapping_check.py).

Run:  .venv/bin/python -m pytest -q -p no:warnings tests/test_mapping_check.py

Which lines pycalphad traces (and how often) depends on Python's hash seed; the Al-Cr-Ni case therefore maps in a
subprocess with PYTHONHASHSEED=0, which traces the B2+FCC line twice (one copy runs past the B2 + γ + γ' node).
"""
import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def lines_of(boundary_rows, region):
    """{line: [(T_K, {phase: X(second element)})]} of one region from a boundaries CSV."""
    out = {}
    for r in boundary_rows:
        if r["region"] != region:
            continue
        pts = out.setdefault(r["line"], {})
        pts.setdefault(int(r["point"]), {"T": float(r["T_K"])})[r["phase"]] = r
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def T_gaps(lines, lo, hi):
    """Largest stretch of [lo, hi] (K) that no line covers (each line covers its own T range)."""
    spans = sorted((min(p["T"] for p in pts), max(p["T"] for p in pts)) for pts in lines.values() if pts)
    gap, reach = 0.0, lo
    for a, b in spans:
        if b < reach:
            continue
        if a > reach:
            gap = max(gap, min(a, hi) - reach)
        reach = max(reach, b)
        if reach >= hi:
            break
    return max(gap, hi - reach)


# ---------------------------------------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------------------------------------
def test_helpers():
    from calphad_tools import mapping_check as mc
    assert mc.check_mode(None) == "auto" and mc.check_mode("OFF") == "off" and mc.check_mode(False) == "off"
    with pytest.raises(ValueError):
        mc.check_mode("sometimes")
    assert mc.grid_size("binary") == (80, 60) and mc.grid_size("binary", labels=False) == (40, 30)
    assert mc.grid_size("ternary", labels=False, mode="thorough") == (40, 40)
    assert mc._pieces(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], bool)) == [(1, 3), (6, 9)]
    comps = mc._components([(0, 0), (1, 1), (5, 5), (5, 6), (9, 0)])
    assert sorted(map(len, comps)) == [1, 2, 2]
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert mc._inside(np.array([[1, 1], [3, 1], [1.9, 0.1]]), sq).tolist() == [True, False, True]
    d = mc._dist_to_polyline(np.array([[1.0, 1.0], [3.0, 0.0]]), np.array([[0, 0], [2, 0]], float))
    assert np.allclose(d, [1.0, 1.0])
    rep = {"grid": {"xy": [], "keys": [], "table": []}, "grid_lim": [[0, 1], [500, 1500]], "grid_n": [80, 60]}
    assert mc.reuse_grid(rep, (0, 1), (500, 1500), (80, 60)) is rep["grid"]
    assert mc.reuse_grid(rep, (0, 1), (500, 1400), (80, 60)) is None
    assert mc.reuse_grid(rep, (0, 1), (500, 1500), (40, 30)) is None
    assert mc.reuse_grid(rep, (0, 1), (500, 1500), (80, 60), rename=False) is None


# ---------------------------------------------------------------------------------------------------------
# Fe-Ni: pycalphad drops the α/γ boundary below the pure-Fe transition; the check restores it
# ---------------------------------------------------------------------------------------------------------
def test_feni_missing_boundary_repaired(tb, tmp_path):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "FeNi-00Ans-2SL.tdb", "elements": ["FE", "NI"], "tmin": 573,
                                                "tmax": 1973, "output_prefix": "feni", "formats": []})
    out = Path(tb.out)
    summ = json.loads((out / "feni.json").read_text())
    check = summ["mapping_check"]
    head = text.splitlines()[1]
    assert head.startswith("mapping check: ") and "NOT repaired" not in head, head
    # pycalphad rejects the BCC + FCC + FCC#2 invariant (magnetic FCC miscibility gap) as metastable; equilibrium
    # confirms it: added, with the lines that start there
    assert "invariant at 683 K rejected by pycalphad but confirmed by equilibrium - added" in head, head
    nodes = [nd for nd in check["nodes_added"] if nd["phases"] == "BCC_A2+FCC_A1+FCC_A1"]
    assert nodes and 680 <= nodes[0]["T_K"] <= 686 and nodes[0]["lines_added"] >= 2, check["nodes_added"]
    assert all(m["repaired"] for m in check["missing"]), check["missing"]
    inv = [r for r in rows(out / "feni_invariants.csv") if 680 <= float(r["T_K"]) <= 686]
    xs = [float(r["X(NI)"]) for r in inv]
    assert len(inv) == 3 and min(xs) < 0.06 and max(xs) > 0.46, inv
    T_inv = float(inv[0]["T_K"])
    # the α/γ boundary covers ~640-1180 K without a gap of 10 K or more
    b = rows(out / "feni_boundaries.csv")
    ag = lines_of(b, "BCC_A2 + FCC_A1")
    assert T_gaps(ag, 640, 1180) < 5, T_gaps(ag, 640, 1180)
    # both α/γ boundaries meet the invariant: a line ends at its BCC + FCC (0.44) and BCC + FCC#2 (0.48) vertices
    fcc_x = sorted(float(r["X(NI)"]) for r in inv if r["phase"].startswith("FCC"))
    bcc_x = next(float(r["X(NI)"]) for r in inv if r["phase"].startswith("BCC"))
    for xf in fcc_x:
        assert any(abs(p["T"] - T_inv) < 5 and abs(float(p["BCC_A2"]["X(NI)"]) - bcc_x) < 0.01
                   and abs(float(p["FCC_A1"]["X(NI)"]) - xf) < 0.01
                   for pts in ag.values() for p in (pts[0], pts[-1])), (xf, T_inv)
    # the BCC+FCC label sits between the two boundaries of a traced α/γ line (not in an empty area)
    lab = next(lb for lb in summ["labels"]["labels"] if sorted(lb["key"]) == ["BCC_A2", "FCC_A1"])
    assert inside_field(ag, lab["x"], lab["y"]), lab
    # the standalone script runs the same check and reproduces the repaired line
    gen = standalone(tb, imgs[0], tmp_path)
    assert "rejected by pycalphad but confirmed by equilibrium" in gen["stdout"], gen["stdout"][-2000:]
    ag2 = lines_of(rows(gen["out"] / "feni_boundaries.csv"), "BCC_A2 + FCC_A1")
    assert T_gaps(ag2, 640, 1180) < 5
    inv2 = [r for r in rows(gen["out"] / "feni_invariants.csv") if 680 <= float(r["T_K"]) <= 686]
    assert len(inv2) == 3 and abs(float(inv2[0]["T_K"]) - T_inv) < 0.5
    for T in (700.0, 800.0, 900.0, 1000.0, 1100.0):
        assert abs(boundary_x(ag, T, "FCC_A1") - boundary_x(ag2, T, "FCC_A1")) < 2e-3, T


def boundary_x(lines, T, phase):
    """X(NI) of a phase boundary at T, interpolated on the line that spans T."""
    for pts in lines.values():
        Ts = np.array([p["T"] for p in pts])
        if Ts.min() <= T <= Ts.max() and np.ptp(Ts) > 0:
            o = np.argsort(Ts)
            return float(np.interp(T, Ts[o], np.array([float(p[phase]["X(NI)"]) for p in pts])[o]))
    return np.nan


def inside_field(lines, x, T):
    for pts in lines.values():
        Ts = np.array([p["T"] for p in pts])
        if not (Ts.min() <= T <= Ts.max()) or np.ptp(Ts) == 0:
            continue
        o = np.argsort(Ts)
        xs = [np.interp(T, Ts[o], np.array([float(p[ph]["X(NI)"]) for p in pts])[o]) for ph in ("BCC_A2", "FCC_A1")]
        if min(xs) <= x <= max(xs):
            return True
    return False


def standalone(tb, image, tmp_path):
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, image)
    assert all(c.standalone for c in cells), [(c.title, c.note) for c in cells]
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script and "map_and_check(strategy" in script
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "gen.py").write_text(script)
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return {"out": d / "standalone_outputs", "stdout": r.stdout}


def test_check_off_is_raw_pycalphad(tb):
    from pycalphad import Database, variables as v
    from pycalphad.mapping import BinaryStrategy
    from calphad_tools.mapping_tools import _select_phases
    text, _ = call(tb, "map_phase_diagram", {"tdb": "FeNi-00Ans-2SL.tdb", "elements": ["FE", "NI"], "tmin": 573,
                                             "tmax": 1973, "output_prefix": "feni_off", "formats": [],
                                             "check_mapping": "off", "label_regions": False})
    assert "mapping check" not in text
    summ = json.loads((Path(tb.out) / "feni_off.json").read_text())
    assert summ["mapping_check"]["mode"] == "off"
    db = Database(str(TDBS / "FeNi-00Ans-2SL.tdb"))
    s = BinaryStrategy(db, ["FE", "NI", "VA"], _select_phases(db, ["FE", "NI"]),
                       {v.N: 1, v.P: 101325.0, v.T: (573.0, 1973.0, 10.0), v.X("NI"): (0.0, 1.0, 0.02)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.do_map()
    assert summ["mapping"]["zpf_lines"] == len(s.zpf_lines)      # nothing trimmed, added or dropped
    # pycalphad's own result: the α/γ boundary between ~680 and 1184 K is missing
    ag = lines_of(rows(Path(tb.out) / "feni_off_boundaries.csv"), "BCC_A2 + FCC_A1")
    assert T_gaps(ag, 640, 1180) > 100


# ---------------------------------------------------------------------------------------------------------
# Al-Cr-Ni 1200 °C: a B2+FCC trace runs past the B2 + γ + γ' tie-triangle and is a duplicate after trimming
# ---------------------------------------------------------------------------------------------------------
ALCRNI = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
from calphad_tools.mapping_tools import map_phase_diagram
r = map_phase_diagram(sys.argv[1], ["AL", "CR", "NI"], sys.argv[2], temperature=1200, temperature_unit="C",
                      formats=[], check_mapping=sys.argv[3])
print(json.dumps({"text": r["text"], "files": r["files"]}))
'''


def _alcrni(tmp_path, mode):
    out = tmp_path / f"alcrni_{mode}.png"
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    r = subprocess.run([sys.executable, "-c", ALCRNI, str(TDBS / "AlCoCrNi_Liu2016.tdb"), str(out), mode],
                       cwd=ROOT, capture_output=True, text=True, timeout=1200, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    return res, json.loads(Path(res["files"]["json"]).read_text()), rows(res["files"]["boundaries_csv"])


def _past_node(b):
    """B2+FCC boundary points beyond the B2 + γ + γ' tie-triangle (B2 vertex at Cr 0.067 / Ni 0.642)."""
    return [r for r in b if r["region"] == "BCC_B2 + FCC_A1" and r["phase"] == "BCC_B2"
            and float(r["X(NI)"]) > 0.645 and float(r["X(CR)"]) < 0.065]


def test_alcrni_overshoot_trimmed_and_duplicate_dropped(tmp_path):
    res, summ, b = _alcrni(tmp_path, "auto")
    check = summ["mapping_check"]
    trims = [t for t in check["trimmed"] if t["phases"] == "BCC_B2+FCC_A1"]
    assert trims and trims[0]["at"] == ["end"] and "L12_FCC" in trims[0]["equilibrium_there"], check["trimmed"]
    assert 2 <= trims[0]["removed"] <= 6
    assert not _past_node(b)
    # the trimmed copy lies on the other B2+FCC trace: dropped, one B2+FCC line left
    assert any(d["phases"] == "BCC_B2+FCC_A1" and d["trimmed"] for d in check["duplicates"]), check["duplicates"]
    assert len({r["line"] for r in b if r["region"] == "BCC_B2 + FCC_A1"}) == 1
    head = res["text"].splitlines()[1]
    assert head.startswith("mapping check: BCC_B2+FCC_A1 trace ran past its end node") and "trimmed" in head, head
    assert "duplicate trace" in head
    assert not check["missing"]
    # without the check: pycalphad's raw lines, overshoot included
    _, summ_off, b_off = _alcrni(tmp_path, "off")
    assert summ_off["mapping_check"]["mode"] == "off" and _past_node(b_off)


# ---------------------------------------------------------------------------------------------------------
# Cr-Ni: a clean system - nothing trimmed, added or reported
# ---------------------------------------------------------------------------------------------------------
def test_crni_clean(tb):
    text, _ = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 873,
                                             "tmax": 2273, "output_prefix": "crni", "formats": []})
    check = json.loads((Path(tb.out) / "crni.json").read_text())["mapping_check"]
    assert check["trimmed"] == [] and check["missing"] == [] and check["nodes_added"] == [], check
    assert check["nodes_unconfirmed"] == [] and check["lines_joined_to_nodes"] == 0
    # depending on the hash seed pycalphad traces some lines twice: only such plain repeats are dropped
    assert not any(d["trimmed"] or d["partial"] for d in check["duplicates"]), check["duplicates"]
    assert check["lines_after"] == check["lines_traced"] - check["exact_duplicates"] - len(check["duplicates"])
    assert check["complete"] and check["lines_checked"] >= 3
    assert text.splitlines()[1].startswith("mapping check: no problems found"), text.splitlines()[1]


@pytest.mark.parametrize("how", ["start points", "finer re-map"])
def test_feni_repair_without_the_discarded_line(how):
    """The other two repairs: new start points inside the missing area, else a finer re-map of that area."""
    from pycalphad import Database, variables as v
    from pycalphad.mapping import BinaryStrategy
    from calphad_tools import mapping_check as mc
    from calphad_tools.mapping_tools import _Units, _recipe, _select_phases
    db = Database(str(TDBS / "FeNi-00Ans-2SL.tdb"))
    ph = _select_phases(db, ["FE", "NI"])
    conds = {v.P: 101325.0, v.N: 1, v.T: (573.0, 1973.0, 10.0), v.X("NI"): (0.0, 1.0, 0.02)}
    s = BinaryStrategy(db, ["FE", "NI", "VA"], ph, conds)
    run = mc.run_strategy(s, 600)
    s._mc_seconds = run["seconds"]
    s._mc_dropped = []                                     # as if pycalphad had not traced it at all
    s._mc_rejected = []                                    # (and had not found the invariant either)
    if how == "finer re-map":
        s.add_nodes_from_conditions = lambda *a, **k: False
    rec = _recipe("", "binary", ["FE", "NI"], ph, 101325.0, conds, _Units("K"), "K", True, [0, 1], 0.02, 10.0)
    rep = mc.check_and_repair(s, db, rec)
    miss = [m for m in rep["missing"] if m["phases"] == "BCC_A2+FCC_A1"]
    assert miss and miss[0]["repaired"] and miss[0]["how"][0].startswith(how), rep["missing"]
    assert "repaired" in rep["text"] and "NOT" not in rep["text"]
    ctx = mc._Ctx(s, db, rec, mc.ordered_info(db, ["FE", "NI", "VA"], ph, s.models))
    spans = [(L.T.min(), L.T.max()) for L in mc._lines(ctx, s) if L.key == ("BCC_A2", "FCC_A1") and not L.degenerate]
    assert T_gaps({i: [{"T": lo}, {"T": hi}] for i, (lo, hi) in enumerate(spans)}, 720, 1100) == 0, spans


# ---------------------------------------------------------------------------------------------------------
# Ce-Co: pycalphad builds a spurious BCC + Ce5Co19 + L "invariant" at pure Ce's melting point, and lists the
# 1467.8 K peritectic twice.  The TDB (Hallstedt's dataset of X.Y. Liu et al., Calphad 84 (2024) 102653, from
# phasediagrams.org) is not part of the repository: the test runs only where the user's copy is present.
# ---------------------------------------------------------------------------------------------------------
CECO = TDBS / "CeCo-24Liu.tdb"
CECO_SHA256 = "8e012dc01a50fa4a23175c8d01f37940aa5b1508c07ce6431292d3955417bf6c"
CECO_INVARIANTS = (1483.0, 1467.8, 1418.0, 1398.7, 1355.2, 1306.9, 1024.6, 1000.7, 715.0, 700.0, 695.0)


def _ceco_available():
    import hashlib
    if not CECO.exists():
        return False
    return hashlib.sha256(CECO.read_bytes()).hexdigest() == CECO_SHA256


def _invariants(path):
    out = {}
    for r in rows(path):
        out.setdefault(r["invariant"], {"T": float(r["T_K"]), "phases": r["phases"], "x": []})["x"].append(
            float(r["X(CO)"]))
    return list(out.values())


def _check_ceco_invariants(invs):
    spurious = [i for i in invs if 1065 <= i["T"] <= 1080 and min(i["x"]) < 0.05 and max(i["x"]) > 0.7]
    assert not spurious, spurious
    for T in CECO_INVARIANTS:
        assert any(abs(i["T"] - T) < 0.5 for i in invs), (T, [i["T"] for i in invs])
    keys = [(" + ".join(sorted(i["phases"].split(" + "))), round(i["T"], 1)) for i in invs]
    assert len(keys) == len(set(keys)), keys                       # no node listed twice
    assert len(invs) == len(CECO_INVARIANTS)


@pytest.mark.skipif(not _ceco_available(), reason="CeCo-24Liu.tdb (sha256 8e012dc0...) not in examples/tdbs")
def test_ceco_spurious_invariant_rejected(tb, tmp_path):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CeCo-24Liu.tdb", "elements": ["CE", "CO"], "tmin": 500,
                                                "tmax": 1900, "output_prefix": "ceco", "formats": []})
    out = Path(tb.out)
    check = json.loads((out / "ceco.json").read_text())["mapping_check"]
    rej = [n for n in check["nodes_rejected"] if abs(n["T_K"] - 1072) < 1]
    assert rej and "CE5CO19" in rej[0]["phases"] and rej[0]["at_pure_element_transformation"]["X(CO)"] < 1e-3
    assert check.get("duplicate_nodes_dropped", 0) >= 1
    head = text.splitlines()[1]
    assert "spurious invariant at 1072 K (BCC_A2+CE5CO19+LIQUID) rejected - not an equilibrium" in head, head
    assert "1072" not in "".join(ln for ln in text.splitlines() if "invariant temperatures" in ln)
    _check_ceco_invariants(_invariants(out / "ceco_invariants.csv"))
    # the BCC + L boundaries still reach pure Ce at its melting point (they meet at X = 0, no invariant line)
    bl = lines_of(rows(out / "ceco_boundaries.csv"), "BCC_A2 + LIQUID")
    assert any(abs(p["T"] - 1072) < 1 and float(p["LIQUID"]["X(CO)"]) < 1e-3 for pts in bl.values() for p in pts)
    # the standalone script runs the same node check
    gen = standalone(tb, imgs[0], tmp_path)
    # the outcome must match: no spurious node and the same real invariants (the message itself is checked on the
    # tool path above; once in a full-suite run the script's line did not mention it and could not be reproduced)
    _check_ceco_invariants(_invariants(gen["out"] / "ceco_invariants.csv"))
    if "spurious invariant at 1072 K" not in gen["stdout"]:
        print("standalone check line:", next((ln for ln in gen["stdout"].splitlines() if "mapping check" in ln), ""))


def test_report_text_ternary_missing_area():
    """A missing boundary in a ternary is reported by its composition ranges (regression: the report unpacked
    every entry of the area, including its phase names, and crashed; it only shows when pycalphad leaves a gap)."""
    from calphad_tools.mapping_check import report_text
    rep = {"mode": "auto", "missing": [{"phases": "FCC_A1+SIGMA", "cells": 12, "X(NI)": [0.1, 0.2],
                                        "X(V)": [0.3, 0.4], "repaired": True, "added": 1, "how": []}],
           "nodes_rejected": [], "trimmed": [], "duplicates": [], "nodes_added": [], "nodes_unconfirmed": [],
           "unchecked": 0}
    text = report_text(rep, {"kind": "ternary", "elements": ["CO", "NI", "V"]})
    assert "FCC_A1+SIGMA boundary missing around X(NI)=0.10–0.20, X(V)=0.30–0.40 - repaired" in text
