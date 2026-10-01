"""Process-wide calculation cache (calphad_tools/pcache.py).

Run:  .venv/bin/python -m pytest -q tests/test_cache.py
"""
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from calphad_tools import pcache
from calphad_tools.agent.tools import UI, ToolBox

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"


class _UI(UI):
    def ask(self, question, options=None):
        return "ok"

    def confirm(self, title, body):
        return False


@pytest.fixture
def work(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    for f in ("CrNi-06Hal.tdb", "AlCoCrNi_Liu2016.tdb"):
        shutil.copy(TDBS / f, d / f)
    return d


def _tb(work):
    return ToolBox(workdir=str(work), output_dir="out", ui=_UI(), python_tool="off")


def call(tb, name, args):
    text, images, err = tb.call(name, args)
    assert not err, text
    return text


@pytest.fixture(autouse=True)
def _fresh_cache():
    pcache.set_enabled(True)
    pcache.clear()
    yield
    pcache.set_enabled(True)
    pcache.clear()


def _files(tb, text, exts=(".csv", ".json")):
    """Output files named in a tool's text (relative to the workdir)."""
    out = []
    for m in re.finditer(r"([\w./-]+\.(?:csv|json))", text):
        p = Path(tb.p(m.group(1)))
        if p.suffix in exts and p.exists():
            out.append(p)
    return sorted(set(out))


def _uncached(fn):
    pcache.set_enabled(False)
    try:
        return fn()
    finally:
        pcache.set_enabled(True)


CRNI_CALLS = [
    ("calculate_equilibrium", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                               "conditions": {"T": [1100, 1500], "X(CR)": 0.35}, "temperature_unit": "K"}),
    ("equilibrium_property_plot", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "plot": False,
                                   "conditions": {"T": [900, 1900, 50], "X(CR)": 0.45},
                                   "quantities": ["NP(*)", "GM", "MU(CR)"]}),
    ("step_map", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "vary": "T", "start": 900, "stop": 1900,
                  "step": 20, "fixed": {"X(CR)": 0.45}}),
    ("map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["NI", "CR"], "tmin": 1000, "tmax": 2200,
                           "composition_step": 0.05, "temperature_step": 20}),
    ("phase_property_plot", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "prop": "HM", "temperature": 1300}),
    ("export_equilibrium_data", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "formats": ["csv"],
                                 "conditions": {"T": [1000, 1600, 100], "X(CR)": [0.2, 0.5, 0.1]}}),
    ("scheil_solidification", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "composition": {"CR": 0.3},
                               "step_temperature": 5.0}),
]
# a first call on the same system with other conditions, to warm the cache before the compared call
WARMUP = {
    "calculate_equilibrium": {"conditions": {"T": 1300, "X(CR)": 0.6}},
    "equilibrium_property_plot": {"conditions": {"T": [1000, 1600, 100], "X(CR)": 0.2}},
    "step_map": {"fixed": {"X(CR)": 0.25}},
    "map_phase_diagram": {"tmin": 1200, "tmax": 2000},
    "phase_property_plot": {"temperature": 1100},
    "export_equilibrium_data": {"conditions": {"T": 1200, "X(CR)": 0.3}},
    "scheil_solidification": {"composition": {"CR": 0.2}},
}


@pytest.mark.parametrize("name,args", CRNI_CALLS, ids=[c[0] for c in CRNI_CALLS])
def test_cached_results_identical(work, name, args):
    tb = _tb(work)
    ref_text = _uncached(lambda: call(tb, name, args))
    ref_files = {p.name: p.read_bytes() for p in _files(tb, ref_text)}
    call(tb, name, {**args, **WARMUP[name]})                    # warm: same system, other conditions
    misses = pcache.info()["models"]["misses"]
    text = call(tb, name, args)
    assert pcache.info()["models"]["misses"] == misses          # nothing rebuilt
    strip = (lambda s: re.sub(r"\(?[\d.]+ ?s\)?|\d+(\.\d+)? s\b", "", s))
    assert strip(text) == strip(ref_text)
    files = {p.name: p.read_bytes() for p in _files(tb, text)}
    assert files.keys() == ref_files.keys()
    assert files or name == "calculate_equilibrium"
    for k in files:
        a, b = files[k], ref_files[k]
        if k.endswith(".json"):                                # drop the run time ("seconds") only
            a, b = (re.sub(rb'"seconds": [0-9.]+', b"", x) for x in (a, b))
        assert a == b, k                                       # byte-identical numbers


def test_second_call_is_faster(work):
    tb = _tb(work)
    els = ["AL", "CO", "CR", "NI"]
    a = {"tdb": "AlCoCrNi_Liu2016.tdb", "elements": els, "temperature_unit": "K",
         "conditions": {"T": 1200, "X(AL)": 0.1, "X(CR)": 0.2, "X(NI)": 0.5}}
    t0 = time.perf_counter()
    call(tb, "calculate_equilibrium", a)
    cold = time.perf_counter() - t0
    info1 = pcache.info()
    times = []
    for T, xal in ((1000, 0.15), (1400, 0.05), (1100, 0.12)):
        t0 = time.perf_counter()
        call(tb, "calculate_equilibrium", {**a, "conditions": {"T": T, "X(AL)": xal, "X(CR)": 0.2, "X(NI)": 0.5}})
        times.append(time.perf_counter() - t0)
    info2 = pcache.info()
    assert info2["models"]["misses"] == info1["models"]["misses"]
    assert info2["compiled"]["factory_misses"] == info1["compiled"]["factory_misses"]
    assert min(times) < cold / 3, (cold, times)
    # a different tool on the same system reuses the compiled code too
    t0 = time.perf_counter()
    call(tb, "equilibrium_property_plot", {"tdb": "AlCoCrNi_Liu2016.tdb", "elements": els, "plot": False,
                                           "conditions": {"T": [1000, 1600, 100], "X(AL)": 0.1, "X(CR)": 0.2,
                                                          "X(NI)": 0.5}, "quantities": ["NP(*)"]})
    assert time.perf_counter() - t0 < cold
    assert pcache.info()["models"]["misses"] == info1["models"]["misses"]


def _gm_liquid(tb, tdb="CrNi-06Hal.tdb"):
    import json
    r = json.loads(call(tb, "calculate_equilibrium", {"tdb": tdb, "elements": ["CR", "NI"], "phases": ["LIQUID"],
                                                      "conditions": {"T": 2200, "X(CR)": 0.5},
                                                      "temperature_unit": "K"}))
    return r["points"][0]["GM_J_per_mol"]


def _edit(path: Path, new_L0: str):
    txt = path.read_text()
    txt2 = re.sub(r"(L\(LIQUID,CR,NI;0\),,\s+)[^;]+;", rf"\g<1>{new_L0};", txt, count=1)
    assert txt2 != txt
    path.write_text(txt2)


def test_editing_tdb_invalidates(work):
    tb = _tb(work)
    g0 = _gm_liquid(tb)
    _edit(work / "CrNi-06Hal.tdb", "-5000+2*T")               # same size or not: the content hash decides
    g1 = _gm_liquid(tb)
    assert g1 != g0
    assert g1 == _uncached(lambda: _gm_liquid(tb))


def test_edited_copy_gets_own_entry(work):
    tb = _tb(work)
    shutil.copy(work / "CrNi-06Hal.tdb", work / "crni_mod.tdb")
    _edit(work / "crni_mod.tdb", "-5000+2*T")
    g_orig = _gm_liquid(tb)
    g_mod = _gm_liquid(tb, "crni_mod.tdb")
    assert g_mod != g_orig
    assert pcache.info()["databases"]["entries"] == 2
    assert _gm_liquid(tb) == g_orig and _gm_liquid(tb, "crni_mod.tdb") == g_mod
    assert g_mod == _uncached(lambda: _gm_liquid(tb, "crni_mod.tdb"))


def test_identical_copy_shares_entry_and_cache_spans_toolboxes(work):
    tb1, tb2 = _tb(work), _tb(work)
    shutil.copy(work / "CrNi-06Hal.tdb", work / "same.tdb")
    g = _gm_liquid(tb1)
    misses = pcache.info()["models"]["misses"]
    assert _gm_liquid(tb2, "same.tdb") == g
    info = pcache.info()
    assert info["databases"]["entries"] == 1 and info["models"]["misses"] == misses


def test_mutated_shared_database_is_not_reused(work):
    from calphad_tools.calc import get_db
    tb = _tb(work)
    g0 = _gm_liquid(tb)
    db = get_db(str(work / "CrNi-06Hal.tdb"))
    db.symbols["GHSERCR"] = db.symbols["GHSERCR"] + 1000.0     # someone modifies the shared object in place
    assert get_db(str(work / "CrNi-06Hal.tdb")) is not db      # re-read from disk
    assert _gm_liquid(tb) == g0


def test_lru_limits(work):
    old = pcache._DBS.maxsize
    pcache._DBS.maxsize = 1
    try:
        tb = _tb(work)
        shutil.copy(work / "CrNi-06Hal.tdb", work / "b.tdb")
        _edit(work / "b.tdb", "-1000")
        g1, g2 = _gm_liquid(tb), _gm_liquid(tb, "b.tdb")
        assert pcache.info()["databases"]["entries"] == 1
        assert _gm_liquid(tb) == g1 and g1 != g2
    finally:
        pcache._DBS.maxsize = old


def test_cache_tool_and_show_code_stays_standalone(work):
    from calphad_tools import codegen
    tb = _tb(work)
    _gm_liquid(tb)
    txt = call(tb, "calc_cache", {})
    assert '"entries": 1' in txt
    cells, ctx = codegen.build_cells(tb, "1")
    script = codegen.to_script(tb, cells, ctx)
    assert "pcache" not in script and "calphad_tools" not in script
    assert "cleared" in call(tb, "calc_cache", {"clear": True})
    assert pcache.info()["databases"]["entries"] == 0


def test_worker_pool_reuses_workers(tmp_path):
    from calphad_tools import diagrams
    tdb = str(TDBS / "CrNi-06Hal.tdb")
    tasks = [{"name": f"T{i}", "kind": "binary", "kwargs": dict(
        tdb=tdb, elements=["CR", "NI"], out_png=str(tmp_path / f"b{i}.png"), tmin_c=900 + 50 * i, tmax_c=1900,
        x_step=0.1, t_step=40)} for i in range(3)]
    tasks.append({"name": "bad", "kind": "binary", "kwargs": dict(tdb=tdb, elements=["CR", "XX"],
                                                                 out_png=str(tmp_path / "bad.png"))})
    res = diagrams.run_tasks(tasks, n_workers=2, timeout=300)
    assert [r.name for r in res] == ["T0", "T1", "T2", "bad"]
    assert all(r.ok for r in res[:3]), [r.error for r in res]
    assert not res[3].ok and res[3].error
    assert all(Path(r.png).exists() for r in res[:3])


def test_worker_timeout_is_replaced(tmp_path):
    from calphad_tools import diagrams
    tdb = str(TDBS / "AlCoCrNi_Liu2016.tdb")
    slow = {"name": "slow", "kind": "ternary", "kwargs": dict(tdb=tdb, elements=["AL", "CR", "NI"], temperature=900,
                                                              out_png=str(tmp_path / "slow.png"), step=0.01)}
    fast = {"name": "fast", "kind": "binary", "kwargs": dict(tdb=str(TDBS / "CrNi-06Hal.tdb"), elements=["CR", "NI"],
                                                             out_png=str(tmp_path / "fast.png"), tmin_c=1000,
                                                             tmax_c=1800, x_step=0.1, t_step=50)}
    res = diagrams.run_tasks([slow, fast], n_workers=1, timeout=8)
    assert not res[0].ok and "timed out" in res[0].error
    assert res[1].ok, res[1].error


def test_cached_models_match_fresh_models(work):
    """Model expressions from the cache are the same objects' content as freshly built ones."""
    from pycalphad import Model, variables as v
    from calphad_tools.calc import get_db
    db = get_db(str(work / "CrNi-06Hal.tdb"))
    with pcache.use():
        from pycalphad.core import utils
        from pycalphad.core import workspace
        m1 = workspace.instantiate_models(db, ["CR", "NI", "VA"], ["LIQUID", "FCC_A1"])
        m2 = workspace.instantiate_models(db, ["CR", "NI", "VA"], ["FCC_A1"])
        assert m1["FCC_A1"] is m2["FCC_A1"]
        assert utils.instantiate_models is not workspace.instantiate_models
    for ph, m in m1.items():
        fresh = Model(db, ["CR", "NI", "VA"], ph)
        pt = {v.T: 1234.0, v.P: 101325.0, **{y: 0.5 for y in fresh.site_fractions}}
        assert float(m.GM.xreplace(pt)) == float(fresh.GM.xreplace(pt))
    from pycalphad.core import workspace as w2
    assert w2.instantiate_models is utils.instantiate_models     # patch removed outside use()
    assert np.isfinite(float(m1["LIQUID"].GM.xreplace({v.T: 2000.0, **{y: 0.5 for y in m1["LIQUID"].site_fractions}})))
