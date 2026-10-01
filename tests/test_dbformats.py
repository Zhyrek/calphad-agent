"""ChemSage .dat input, convert_database and dialect-safe write_database (dbformats.py, tdb_dialect.py).

Run:  .venv/bin/python -m pytest -q tests/test_dbformats.py
"""
import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pycalphad
import pytest

from calphad_tools import dbformats, tdb_dialect
from calphad_tools.agent.tools import UI, ToolBox

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"
DATS = Path(pycalphad.__file__).parent / "tests" / "databases"     # public ChemSage files shipped with pycalphad


class _UI(UI):
    def ask(self, question, options=None):
        return "ok"

    def confirm(self, title, body):
        return False


@pytest.fixture
def work(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    shutil.copy(TDBS / "CrNi-06Hal.tdb", d)
    for f in ("Viitala.dat", "Kaye_Pd-Ru-Tc-Mo.dat", "Shishin_Fe-Sb-O-S_slag.dat"):
        shutil.copy(DATS / f, d)
    src = (TDBS / "CrNi-06Hal.tdb").read_text()      # long function / phase names that need renaming
    src = re.sub(r"\bGHSERCR\b", "GHSER_CHROMIUM_LONG", src)
    src = re.sub(r"\bGHSERNI\b", "GHSER_NICKEL_LONGNAME", src)
    src = re.sub(r"\bLIQUID\b", "LIQUID_PHASE_WITH_A_VERY_LONG_NAME", src)
    (d / "long.tdb").write_text(src)
    return d


@pytest.fixture
def tb(work):
    return ToolBox(workdir=str(work), output_dir="out", ui=_UI(), python_tool="off")


def call(tb, name, args):
    text, images, err = tb.call(name, args)
    assert not err, text
    return text


def _db(path):
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(path))


# ---------------------------------------------------------------------------------------------------------
# .dat input
# ---------------------------------------------------------------------------------------------------------
def test_calculation_tools_accept_dat(tb):
    r = json.loads(call(tb, "calculate_equilibrium", {"tdb": "Kaye_Pd-Ru-Tc-Mo.dat", "elements": ["MO", "RU"],
                                                      "conditions": {"T": 2500, "X(RU)": 0.5},
                                                      "temperature_unit": "K"}))
    assert r["points"][0]["stable_phases"]
    info = json.loads(call(tb, "tdb_info", {"tdb": "Viitala.dat"}))
    assert "LIQUIDSOLN" in info["not_expressible_in_TDB"] and "CU" in info["elements"]
    txt = call(tb, "equilibrium_property_plot", {"tdb": "Kaye_Pd-Ru-Tc-Mo.dat", "elements": ["MO", "RU"],
                                                 "conditions": {"T": [2200, 2600, 200], "X(RU)": 0.5},
                                                 "quantities": ["GM"], "plot": False})
    assert "CSV:" in txt
    text, _, err = tb.call("compare_tdbs", {"tdb_a": "Viitala.dat", "tdb_b": "CrNi-06Hal.tdb"})
    assert err and "convert_database" in text


def test_convert_dat_pycalphad_dialect(tb, work):
    txt = call(tb, "convert_database", {"source": "Viitala.dat", "output_tdb": "viitala.tdb"})
    out = work / "out" / "viitala.tdb"
    assert "LIQUIDSOLN" in txt and "quasichemical" in txt                 # reported, not written
    assert "CL[-] -> CL-1" in txt and "CU_SOLID(S) -> CU_SOLID_S" in txt  # renames listed
    back = _db(out)
    assert "LIQUIDSOLN" not in back.phases and "CU_SOLID_S" in back.phases
    info = dbformats.convert_database(str(work / "Viitala.dat"), str(work / "v2.tdb"))
    chk = info["model_check_max_abs_dGM_J_per_mol"]
    assert len(chk) >= 10 and max(chk.values()) < 1e-6
    assert tdb_dialect.check_tdb_text(out.read_text(), "pycalphad") == []


def test_convert_reports_qkto_and_strict_exponents(tb, work):
    txt = call(tb, "convert_database", {"source": "Kaye_Pd-Ru-Tc-Mo.dat", "output_tdb": "kaye"})
    assert "FCCN" in txt and "Kohler-Toop" in txt
    text, _, err = tb.call("convert_database", {"source": "Viitala.dat", "output_tdb": "v_tc.tdb",
                                                "dialect": "thermocalc"})
    assert err and "non-integer exponents" in text and "pycalphad" in text


def test_convert_dat_opencalphad_limits(tb, work):
    txt = call(tb, "convert_database", {"source": "Shishin_Fe-Sb-O-S_slag.dat", "output_tdb": "shishin_oc.tdb",
                                        "dialect": "opencalphad"})
    text = (work / "out" / "shishin_oc.tdb").read_text()
    assert tdb_dialect.check_tdb_text(text, "opencalphad") == []
    assert "opencalphad limits: all met" in txt and "SLAG-LIQ" in txt
    assert "SPECIES FE2++2 FE2/+2 !" in text


# ---------------------------------------------------------------------------------------------------------
# dialect-safe export
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dialect", ["thermocalc", "opencalphad", "pycalphad"])
def test_write_database_dialects(tb, work, dialect):
    txt = call(tb, "write_database", {"tdb": "long.tdb", "output_tdb": f"long_{dialect}.tdb", "dialect": dialect})
    out = work / "out" / f"long_{dialect}.tdb"
    text = out.read_text()
    assert tdb_dialect.check_tdb_text(text, dialect) == []
    assert all(len(ln) <= 78 for ln in text.splitlines())
    lim = tdb_dialect.DIALECTS[dialect]
    fnames = re.findall(r"^FUNCTION (\S+)", text, re.M)
    pnames = [p.split(":")[0] for p in re.findall(r"^PHASE (\S+)", text, re.M)]
    if lim["function"]:
        assert max(map(len, fnames)) <= lim["function"] and max(map(len, pnames)) <= lim["phase"]
        assert "renamed functions (2)" in txt and "GHSER_CHROMIUM_LONG ->" in txt
        assert "renamed phases (1)" in txt
        new = re.search(r"GHSER_CHROMIUM_LONG -> (\w+)", txt).group(1)
        assert new in fnames and re.search(rf"[+\-*]{new}\b|\b{new}[+\-*]", text)    # used consistently
        assert "LN(T)" in text and "LOG(" not in text
        cmds = tdb_dialect._logical_lines(text)
        assert all(c.rstrip().endswith("!") for c in cmds if c.strip() and not c.startswith("$"))
        for c in cmds:
            if c.startswith(("FUNCTION", "PARAMETER")):
                assert re.search(r"\bN\s*!$", c), c
        assert "TYPE_DEFINITION A GES A_P_D A1_FCC MAGNETIC -3.0 2.80000E-01 !" in text
        assert re.search(r"^PHASE A1_FCC %A 2 ", text, re.M)            # DIS_PART code only on L12_FCC
        assert text.index("PHASE A1_FCC") < text.index("PHASE L12_FCC")
        assert "ELEMENT VA VACUUM" in text
    else:
        assert "renamed" not in txt
    assert "max |dGM| = 0 J/mol" in txt
    # reload and compare Gibbs energies at sample points, phase by phase
    ren = tdb_dialect.plan_renames(_db(work / "long.tdb"), dialect)
    diffs = dbformats.gm_check(_db(work / "long.tdb"), _db(out), ren, n=6, seed=3)
    assert len(diffs) == 9 and max(diffs.values()) < 1e-9       # float round-off only (term order)


def test_export_examples_thermocalc_identical_gm(tmp_path):
    for f in ("CrNi-06Hal.tdb", "FeNi-00Ans-2SL.tdb", "AlCoCrNi_Liu2016.tdb"):
        db = _db(TDBS / f)
        info = dbformats.export_verified(db, db, str(tmp_path / f), "thermocalc")
        assert info["violations"] == [], (f, info["violations"])
        chk = info["model_check_max_abs_dGM_J_per_mol"]
        assert len(chk) == len(db.phases) and max(chk.values()) < 1e-9, (f, chk)
        text = (tmp_path / f).read_text()
        assert "EXP(" in text or "EXP" not in (TDBS / f).read_text().upper()


def test_raw_writer_output_flagged():
    import io
    from pycalphad.io.tdb import write_tdb
    buf = io.StringIO()
    write_tdb(_db(TDBS / "CrNi-06Hal.tdb"), buf, if_incompatible="ignore")
    bad = tdb_dialect.check_tdb_text(buf.getvalue(), "thermocalc")
    assert any("DIS_PART type code" in b for b in bad)


# ---------------------------------------------------------------------------------------------------------
# show_code
# ---------------------------------------------------------------------------------------------------------
def _standalone_run(tb, target, tmp_path):
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, target)
    assert all(c.standalone for c in cells), [(c.title, c.note) for c in cells]
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gen.py").write_text(script)
    r = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return d / "standalone_outputs", r.stdout


def _no_stamp(path):
    return "\n".join(ln for ln in Path(path).read_text().splitlines() if not ln.startswith("$"))


def test_show_code_for_dialect_and_convert(tb, work, tmp_path):
    call(tb, "write_database", {"tdb": "long.tdb", "output_tdb": "long_tc.tdb", "dialect": "thermocalc"})
    call(tb, "convert_database", {"source": "Viitala.dat", "output_tdb": "viitala.tdb"})
    for i, name in ((1, "long_tc.tdb"), (2, "viitala.tdb")):
        out, stdout = _standalone_run(tb, str(i), tmp_path / str(i))
        assert _no_stamp(out / name) == _no_stamp(work / "out" / name)
        assert "renamed:" in stdout
