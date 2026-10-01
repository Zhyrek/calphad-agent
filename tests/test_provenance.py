"""Session log, provenance sidecars and replay export (calphad_tools/provenance.py)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
TDB = ROOT / "examples" / "tdbs" / "CrNi-06Hal.tdb"
SECRET = "sk-test-0123456789abcdefghijklmnop"


def _ui():
    from calphad_tools.agent.tools import UI

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return U()


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _toolbox(wd, python_tool="off"):
    from calphad_tools.agent.tools import ToolBox
    return ToolBox(str(wd), "out", _ui(), python_tool=python_tool, interactive=False)


def _call(tb, name, args, ok=True):
    text, imgs, err = tb.call(name, args)
    assert err is (not ok), text
    return text, imgs


@pytest.fixture(scope="module")
def sess(tmp_path_factory):
    wd = tmp_path_factory.mktemp("wd")
    (wd / "tdbs").mkdir()
    (wd / "data").mkdir()
    shutil.copy(TDB, wd / "tdbs")
    (wd / "data" / "d.csv").write_text("T,cp\n300,25\n500,27\n700,30\n900,33\n")
    tb = _toolbox(wd)
    _call(tb, "tdb_info", {"tdb": "CrNi-06Hal.tdb"})                                                         # 1
    _call(tb, "make_plot", {"output": "cp.png", "panels": [{"operations": [                                   # 2
        {"op": "scatter", "csv": "data/d.csv", "x_col": "T", "y_col": "cp", "label": "measured"},
        {"op": "set", "xlabel": "T (K)", "ylabel": "Cp"}]}]})
    _call(tb, "phase_property_plot", {"tdb": "tdbs/CrNi-06Hal.tdb", "elements": ["CR", "NI"],                # 3
                                      "prop": "GM", "temperature": 1200})
    _call(tb, "convert_composition", {"composition": {"CR": 20}, "balance": "NI", "from_unit": "wt%"})     # 4
    _call(tb, "tdb_info", {"tdb": "CrNi-06Hal.tdb", "api_key": SECRET}, ok=False)                           # 5
    _call(tb, "edit_figure", {"image": "out/cp.png", "output": "cp_edit.png", "operations": [               # 6
        {"op": "text", "x": 500, "y": 30, "text": "note"}]})
    big = list(range(3000))
    _call(tb, "make_plot", {"output": "big.png", "panels": [{"operations": [                                  # 7
        {"op": "line", "x": big, "y": [v * 0.5 for v in big]}]}]})
    return tb


def _log(tb):
    lines = [json.loads(line) for line in tb.provenance.log_path.read_text().splitlines()]
    return lines[0], lines[1:]


def test_session_log_file_and_redaction(sess):
    tb = sess
    assert tb.provenance.log_path.parent == tb.out / ".session"
    header, calls = _log(tb)
    assert header["type"] == "header" and header["python_tool"] == "off"
    assert set(header["versions"]) >= {"pycalphad", "espei", "numpy", "scipy", "matplotlib", "pandas",
                                       "calphad_tools"}
    assert "environ" not in json.dumps(header).lower()
    assert [c["tool"] for c in calls[:7]] == ["tdb_info", "make_plot", "phase_property_plot", "convert_composition",
                                              "tdb_info", "edit_figure", "make_plot"]
    assert [c["index"] for c in calls[:7]] == list(range(1, 8))
    # redaction: the arg is kept (as a key) but its value never reaches the log
    raw = tb.provenance.log_path.read_text()
    assert SECRET not in raw
    assert calls[4]["args"]["api_key"] == "[REDACTED]" and calls[4]["ok"] is False
    # inputs: TDB hash, CSV hash; outputs: PNG + CSV of phase_property_plot
    tdb_rel = "tdbs/CrNi-06Hal.tdb"
    assert calls[0]["inputs"][tdb_rel]["sha256"] == _sha(TDB)
    assert calls[1]["inputs"]["data/d.csv"]["sha256"] == _sha(tb.workdir / "data/d.csv")
    outs = {o["out_rel"] for o in calls[2]["outputs"]}
    assert outs == {"phaseprop_CR-NI_GM_1200K.png", "phaseprop_CR-NI_GM_1200K.csv"}
    assert not calls[0]["outputs"] and not calls[3]["outputs"]
    # edit_figure: the source image is an input (generated in the session), the new PNG an output
    assert calls[5]["inputs"]["out/cp.png"]["generated"] is True
    assert [o["out_rel"] for o in calls[5]["outputs"]] == ["cp_edit.png"]
    # huge args are kept in full
    assert len(calls[6]["args"]["panels"][0]["operations"][0]["x"]) == 3000


def test_redact_helper(monkeypatch):
    from calphad_tools.provenance import redact, _secret_values
    monkeypatch.setenv("MY_SERVICE_TOKEN", "abcdefgh12345678")
    r = redact({"a": {"Password": "x", "token_count": None, "n": 1}, "key": {"U": 1},
                "s": "use abcdefgh12345678 and sk-ant-aaaaaaaaaaaaaaaaaaaaaaaa"}, _secret_values())
    assert r["a"]["Password"] == "[REDACTED]" and r["a"]["n"] == 1 and r["key"] == {"U": 1}
    assert "abcdefgh12345678" not in r["s"] and "sk-ant" not in r["s"]


def test_png_sidecar_and_embedded_metadata(sess):
    from calphad_tools.provenance import read_png_text
    tb = sess
    png = tb.out / "phaseprop_CR-NI_GM_1200K.png"
    side = json.loads((tb.out / (png.name + ".provenance.json")).read_text())
    assert side["tool"] == "phase_property_plot" and side["call_index"] == 3
    assert side["inputs"]["tdbs/CrNi-06Hal.tdb"]["sha256"] == _sha(TDB)
    assert side["file_info"]["sha256"] == _sha(png) and side["versions"]["pycalphad"]
    emb = json.loads(read_png_text(str(png)))
    assert emb["tool"] == "phase_property_plot" and emb["session"] == side["session"]
    from PIL import Image
    with Image.open(png) as im:                       # still a valid PNG
        im.load()
        assert list(im.size) == side["file_info"]["png"]
    csv_side = tb.out / "phaseprop_CR-NI_GM_1200K.csv.provenance.json"
    assert not csv_side.exists()                      # sidecars only for PNGs


def test_editable_figure_store_unaffected(sess):
    from calphad_tools import figures
    tb = sess
    for name in ("phaseprop_CR-NI_GM_1200K.png", "cp.png", "cp_edit.png"):
        fig = figures.load_figure(str(tb.out / name))
        assert fig is not None, name
    text, _ = _call(tb, "describe_figure", {"image": "out/phaseprop_CR-NI_GM_1200K.png"})
    assert "stored" in text.lower() or "axes" in text.lower()


def test_session_log_tool(sess):
    tb = sess
    text, _ = _call(tb, "session_log", {})
    lines = text.splitlines()
    assert lines[0].startswith("session ") and ".session/" in lines[0]
    assert any(ln.startswith("#3 ") and "phase_property_plot" in ln and "phaseprop_CR-NI_GM_1200K.png" in ln
               for ln in lines)
    assert any(ln.startswith("#5 ") and "FAIL" in ln and "[REDACTED]" in ln for ln in lines)
    big = next(ln for ln in lines if ln.startswith("#7 "))
    assert "[3000 items]" in big and len(big) < 300
    text, _ = _call(tb, "session_log", {"last": 2})
    assert "(last 2 of" in text and "#1 " not in text


def test_provenance_tool(sess):
    tb = sess
    text, _ = _call(tb, "provenance", {"file": "out/cp.png"})
    info = json.loads(text)
    assert info["tool"] == "make_plot" and info["call_index"] == 2 and info["unchanged_since_recorded"] is True
    assert "data/d.csv" in info["inputs"]
    text, _ = _call(tb, "provenance", {"file": "phaseprop_CR-NI_GM_1200K.csv"})   # no sidecar -> session log
    info = json.loads(text)
    assert info["tool"] == "phase_property_plot" and info["source"].endswith(".jsonl")
    assert info["inputs"]["tdbs/CrNi-06Hal.tdb"]["sha256"] == _sha(TDB)
    text, _ = _call(tb, "provenance", {"file": "data/d.csv"})
    assert "no provenance" in text


def test_export_script_replays(sess, tmp_path):
    tb = sess
    text, _ = _call(tb, "export_session", {"format": "script", "output": "full"})
    script = tb.out / "full.py"
    assert script.exists() and "skipped: " in text and "#5 tdb_info (failed)" in text
    src = script.read_text()
    assert "R.step(3, 'phase_property_plot'" in src and SECRET not in src and "api_key" not in src
    # run from an unrelated cwd: the default workdir is found relative to the script
    r = subprocess.run([PY, str(script), "--output-dir", "rep", "--strict"], cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "replay finished: 0 problem(s)" in r.stdout
    rep = tb.workdir / "rep"
    for name in ("cp.png", "cp_edit.png", "big.png", "phaseprop_CR-NI_GM_1200K.png"):
        assert (rep / name).exists(), name
    assert (rep / "phaseprop_CR-NI_GM_1200K.csv").read_text() == (tb.out / "phaseprop_CR-NI_GM_1200K.csv").read_text()
    from PIL import Image
    for name in ("cp.png", "phaseprop_CR-NI_GM_1200K.png"):
        with Image.open(rep / name) as a, Image.open(tb.out / name) as b:
            assert a.size == b.size
    assert "identical phaseprop_CR-NI_GM_1200K.csv" in r.stdout
    # edit_figure's image arg was remapped into the replay dir (edited the replayed figure, not the original)
    assert (rep / ".session").is_dir()
    rlog = next((rep / ".session").glob("*.jsonl")).read_text()
    assert str(tb.out / "cp.png") not in rlog and str(rep / "cp.png") in rlog


def test_export_input_hash_warning(sess, tmp_path):
    tb = sess
    text, _ = _call(tb, "export_session", {"format": "script", "output": "out/replay/partial.py", "steps": "1-2"})
    script = tb.out / "replay" / "partial.py"
    assert "out/replay/partial.py" in text
    assert "R.step(3," not in script.read_text()
    wd2 = tmp_path / "wd2"
    shutil.copytree(tb.workdir / "tdbs", wd2 / "tdbs")
    (wd2 / "data").mkdir()
    (wd2 / "data" / "d.csv").write_text("T,cp\n300,26\n500,27\n700,30\n900,33\n")   # changed input
    r = subprocess.run([PY, str(script), "--workdir", str(wd2), "--output-dir", "o"], cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "WARNING: input data/d.csv changed" in r.stdout and "tdbs/CrNi-06Hal.tdb" not in r.stdout.split(
        "replaying")[1].split("[1]")[0]
    assert (wd2 / "o" / "cp.png").exists()


def test_export_notebook(sess):
    from calphad_tools.provenance import validate_notebook
    tb = sess
    text, _ = _call(tb, "export_session", {"format": "notebook", "since": 2, "output": "nb"})
    nbp = tb.out / "nb.ipynb"
    nb = json.loads(nbp.read_text())
    validate_notebook(nb)
    assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "markdown"
    head = "".join(nb["cells"][0]["source"])
    assert "pycalphad" in head and "tdbs/CrNi-06Hal.tdb" in head
    md = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert any("Step 3: `phase_property_plot`" in m and "GM vs X(NI)" in m for m in md)
    assert not any("Step 1:" in m for m in md)
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    try:
        import nbclient  # noqa: F401
        import nbformat
        from nbclient import NotebookClient
        NotebookClient(nbformat.read(str(nbp), as_version=4), timeout=600,
                       resources={"metadata": {"path": str(nbp.parent)}}).execute()
    except ImportError:
        # no Jupyter: run the code cells in order as a script from the notebook's directory
        flat = nbp.with_name("nb_flat.py")
        flat.write_text("\n\n".join(code))
        r = subprocess.run([PY, str(flat)], cwd=str(nbp.parent), capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "replay finished: 0 problem(s)" in r.stdout
    assert (tb.workdir / "out_replay" / "phaseprop_CR-NI_GM_1200K.png").exists()


def test_run_python_steps_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                  # run_python chdirs into the workdir
    wd = tmp_path / "w"
    wd.mkdir()
    tb = _toolbox(wd, python_tool="auto")
    code = "import numpy as np\nx = np.linspace(0, 1, 5)\nplt.plot(x, x**2)\nprint('saved', savefig('py_fig'))\n" \
           "s = '''tri'''\n"
    _call(tb, "run_python", {"code": code})
    _call(tb, "convert_composition", {"composition": {"CR": 20}, "balance": "NI", "from_unit": "wt%"})
    _, calls = _log(tb)
    assert calls[0]["executed"] is True and [o["out_rel"] for o in calls[0]["outputs"]] == ["py_fig.png"]
    _call(tb, "export_session", {"format": "both"})
    script = next((tb.out / "replay").glob("*.py"))
    src = script.read_text()
    assert "NOT run unless --run-python" in src and "CODE_1 = " in src and "python=True" in src
    r = subprocess.run([PY, str(script), "--output-dir", "r1"], cwd=str(tmp_path), capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0 and "SKIPPED" in r.stdout, r.stdout + r.stderr
    assert not (wd / "r1" / "py_fig.png").exists()
    r = subprocess.run([PY, str(script), "--output-dir", "r2", "--run-python", "--strict"], cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (wd / "r2" / "py_fig.png").exists()


def test_concurrent_sessions_do_not_clobber(tmp_path):
    from calphad_tools.provenance import Recorder
    wd = tmp_path / "w"
    wd.mkdir()
    tb1, tb2 = _toolbox(wd), _toolbox(wd)
    tb2.provenance.session_id = tb1.provenance.session_id         # force an id collision
    for tb in (tb1, tb2):
        _call(tb, "convert_composition", {"composition": {"CR": 20}, "balance": "NI", "from_unit": "wt%"})
    p1, p2 = tb1.provenance.log_path, tb2.provenance.log_path
    assert p1 != p2 and p1.parent == p2.parent
    for p in (p1, p2):
        lines = p.read_text().splitlines()
        assert len(lines) == 2 and json.loads(lines[1])["tool"] == "convert_composition"
    assert isinstance(tb1.provenance, Recorder)


def test_non_json_args_and_listener_robustness(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    tb = _toolbox(wd)
    text, _, err = tb.call("convert_composition", {"composition": {"CR": 20}, "balance": "NI",
                                                   "from_unit": "wt%", "when": object()})
    assert err                                              # unknown arg -> failed call, still logged
    _, calls = _log(tb)
    assert calls[0]["args"]["when"].startswith("<object object")
    with pytest.raises(ValueError):
        from calphad_tools.provenance import export_session
        export_session(tb, tb.provenance, "script")         # nothing successful to export
