"""show_code framework: standalone code reproduces tool results; fallbacks are labelled; chaining."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tb(tmp_path):
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="auto")


def _script(text):
    return text.split("```python\n", 1)[1].rsplit("```", 1)[0]


def _run(code, cwd):
    p = Path(cwd) / "gen.py"
    p.write_text(code)
    r = subprocess.run([sys.executable, "-W", "ignore", str(p)], cwd=cwd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_equilibrium_code_reproduces_tool(tb, tmp_path):
    t, _, err = tb.call("calculate_equilibrium", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                  "conditions": {"T": [1000, 1400], "X(NI)": 0.4},
                                                  "temperature_unit": "C"})
    assert not err
    tool = json.loads(t)
    text, _, err = tb.call("show_code", {"target": "1"})
    assert not err and "1 standalone" in text
    code = _script(text)
    assert "calphad_tools" not in code
    out = _run(code, tmp_path)
    for pt in tool["points"]:
        for ph in pt["stable_phases"]:
            assert f"{ph['phase']} {round(ph['fraction'], 5)}" in out.replace("  ", " "), (ph, out)


def test_run_python_code_and_helpers(tb, tmp_path):
    code_in = ("db = load_db('tdbs/CrNi-06Hal.tdb')\nfrom pycalphad import calculate\n"
               "r = calculate(db, ['CR','NI','VA'], 'LIQUID', T=1800, P=101325, output='GM')\n"
               "print('GMMIN', round(float(r.GM.min()), 3))\nimport matplotlib.pyplot as plt\n"
               "plt.plot([0, 1], [0, 1]); savefig('line')")
    t, imgs, err = tb.call("run_python", {"code": code_in})
    assert not err
    val = re.search(r"GMMIN (\S+)", t).group(1)
    text, _, _ = tb.call("show_code", {"target": imgs[0], "format": "notebook"})
    assert ".ipynb" in text
    out = _run(_script(text), tmp_path)
    assert f"GMMIN {val}" in out and (tmp_path / "standalone_outputs" / "line.png").exists()


def test_fallback_is_labelled_and_runs(tb, tmp_path):
    from calphad_tools import codegen
    saved = codegen.GENERATORS.pop("tdb_info", None)
    try:
        t, _, err = tb.call("tdb_info", {"tdb": "CrNi-06Hal.tdb"})
        text, _, _ = tb.call("show_code", {"target": "1"})
        assert "via calphad_tools" in text and "reproduced through calphad_tools" in text
        out = _run(_script(text), tmp_path)
        assert "CR" in out
    finally:
        if saved:
            codegen.GENERATORS["tdb_info"] = saved


def test_unknown_target_is_clear(tb):
    text, _, err = tb.call("show_code", {"target": "no_such_file.png"})
    assert err and "no record" in text
    text, _, err = tb.call("show_code", {"target": "99"})
    assert err and "does not exist" in text


def test_output_from_an_earlier_session_via_provenance(tb, tmp_path):
    """A fresh ToolBox (new session) can still produce code for a PNG made in a previous session."""
    t, imgs, err = tb.call("run_python", {"code": "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])\n"
                                                  "savefig('old_fig')"})
    from calphad_tools.agent.tools import ToolBox, UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    tb2 = ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")
    text, _, err = tb2.call("show_code", {"target": imgs[0]})
    assert not err and "plt.plot([1, 2], [3, 4])" in text


def test_edit_chain_has_one_cell_per_step(tb):
    """A call's own `output` must not be treated as its parent (regression: duplicated edit cells)."""
    _, im, _ = tb.call("make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                     "output": "p1.png"})
    _, im2, _ = tb.call("edit_figure", {"image": im[0], "operations": [{"op": "scatter", "x": [0.5], "y": [0.5]}],
                                        "output": "p2.png"})
    _, im3, _ = tb.call("edit_figure", {"image": im2[0], "operations": [{"op": "set", "title": "x"}],
                                        "output": "p3.png"})
    text, _, _ = tb.call("show_code", {"target": im3[0]})
    heads = [l for l in text.splitlines() if l.startswith("# %%")]
    assert heads == ["# %% make_plot (call #1)", "# %% edit_figure (call #2)", "# %% edit_figure (call #3)"]


def test_show_code_is_never_its_own_producer(tb, tmp_path):
    _, im, _ = tb.call("make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                     "output": "q.png"})
    for _ in range(2):
        text, _, err = tb.call("show_code", {"target": "q.png"})
        assert not err
        heads = [l for l in text.splitlines() if l.startswith("# %%")]
        assert heads == ["# %% make_plot (call #1)"], heads
