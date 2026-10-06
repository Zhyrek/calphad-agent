"""Reproduction notebooks saved next to every phase diagram (codegen.save_figure_notebook) and the fixed hash seed
(calphad_tools/hashseed.py).

The notebooks' code cells are run (concatenated; nbclient is not a dependency) in a fresh directory with
PYTHONHASHSEED=0 - the seed the tool's figures record - and must write the same files as the tool: PNG/SVG/PDF, the
boundary / tie-line / invariant CSVs and the summary JSON (same keys, equivalent content).

Run:  .venv/bin/python -m pytest -q tests/test_notebooks.py
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


def call(tb, name, args):
    text, imgs, err = tb.call(name, args)
    assert not err, text
    return text, imgs


def code_of(nb_path: Path) -> str:
    cells = json.loads(nb_path.read_text())["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def run_notebook(nb_path: Path, d: Path, seed="0"):
    """The notebook's code in a fresh directory (as Jupyter would run it top to bottom) -> its output folder."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "nb.py").write_text(code_of(nb_path))
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env.pop("PYTHONHASHSEED", None)
    if seed is not None:
        env["PYTHONHASHSEED"] = seed
    r = subprocess.run([sys.executable, "-W", "ignore", "nb.py"], cwd=d, capture_output=True, text=True, env=env,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return d / "standalone_outputs", r.stdout


def tool_files(tb, index: int):
    """Files call ``index`` wrote (its provenance record), without the notebook."""
    rec = next(r for r in tb.provenance.records if r["index"] == index)
    return {Path(o["path"]).name for o in rec["outputs"] if not o["path"].endswith(".ipynb")}


def recorded_seed(json_path: Path) -> str:
    return json.loads(json_path.read_text())["hash_seed"]


def frames_close(a: Path, b: Path, atol=1e-7):
    x, y = pd.read_csv(a), pd.read_csv(b)
    assert list(x.columns) == list(y.columns), (list(x.columns), list(y.columns))
    assert x.shape == y.shape, (a.name, x.shape, y.shape)
    for c in x.columns:
        if x[c].dtype.kind in "fi" and y[c].dtype.kind in "fi":
            assert np.allclose(x[c], y[c], rtol=1e-7, atol=atol, equal_nan=True), (a.name, c)
        else:
            assert (x[c].fillna("").astype(str) == y[c].fillna("").astype(str)).all(), (a.name, c)


def boundaries_close(a: Path, b: Path, cols, tol=0.02):
    """Different hash seeds: every boundary point of a lies on the same phase's curve in b and vice versa."""
    x, y = pd.read_csv(a), pd.read_csv(b)
    assert set(x.phase) == set(y.phase)
    scale = np.array([1.0, 1000.0]) if cols[1] == "T_K" else np.array([1.0, 1.0])
    for ph in set(x.phase):
        pa, pb = x[x.phase == ph][cols].to_numpy() / scale, y[y.phase == ph][cols].to_numpy() / scale
        for p, q in ((pa, pb), (pb, pa)):
            d = np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(1)
            assert np.quantile(d, 0.98) < tol and d.max() < 5 * tol, (ph, d.max())


def summaries_equivalent(a: dict, b: dict, strict: bool):
    """Same keys; same phases, regions, invariants (T within 0.5 K), labels and mapping-check findings.  strict:
    both ran with the same hash seed, so the traced lines (and everything computed from them) agree exactly."""
    assert set(a) == set(b), set(a) ^ set(b)
    same = ["kind", "system", "elements", "phases_considered", "phases_found", "conditions", "orientation",
            "renamed_when_disordered", "style"]
    for k in same + (["regions", "hash_seed"] if strict else []):
        assert a[k] == b[k], k
    assert [r["region"] for r in a["regions"]] == [r["region"] for r in b["regions"]]
    assert set(a["files"]) == set(b["files"])
    assert {Path(p).name for p in a["files"].values()} == {Path(p).name for p in b["files"].values()}
    ia, ib = a["invariants"], b["invariants"]
    assert len(ia) == len(ib)
    for x, y in zip(ia, ib):
        assert sorted(x["phases"]) == sorted(y["phases"]) and abs(x["T_K"] - y["T_K"]) < 0.5
        assert x.get("type") == y.get("type") and set(x) == set(y)
    assert set(a["mapping_check"]) == set(b["mapping_check"])
    if strict:
        for k in ("lines_checked", "tests", "lines_after", "trimmed", "missing", "nodes_added", "complete"):
            assert a["mapping_check"].get(k) == b["mapping_check"].get(k), k
        assert {k: v for k, v in a["mapping"].items() if k != "seconds"} == \
            {k: v for k, v in b["mapping"].items() if k != "seconds"}
    assert set(a["mapping"]) == set(b["mapping"])
    if "labels" in a:
        la, lb = a["labels"], b["labels"]
        assert set(la) == set(lb)
        assert sorted(t["text"] for t in la["labels"]) == sorted(t["text"] for t in lb["labels"])
        if strict:
            assert [t["gid"] for t in la["labels"]] == [t["gid"] for t in lb["labels"]]
            for x, y in zip(la["labels"], lb["labels"]):
                assert np.allclose([x["x"], x["y"]], [y["x"], y["y"]], rtol=1e-6)
            assert [s["label"] for s in la["skipped"]] == [s["label"] for s in lb["skipped"]]


def check_regenerates(tb, text, imgs, tmp_path):
    """The auto-saved notebook of the last call regenerates every file the call wrote, equivalently.  It runs with
    the hash seed the figure recorded (0 when the tests themselves run with PYTHONHASHSEED=0: then the lines are
    identical; otherwise the original seed was random and the comparison allows for differently traced lines)."""
    png = Path(imgs[0])
    nb = png.with_suffix(".ipynb")
    assert nb.exists()
    assert imgs == [str(png)]                                    # the notebook is no extra result image
    lines = [ln for ln in text.splitlines() if ln.startswith("notebook: ")]
    assert lines == [f"notebook: {tb.rel(str(nb))} (regenerates all files below)"]
    assert text.index("notebook: ") < text.index("\nFiles:\n")
    written = tool_files(tb, len(tb.call_log))
    assert {png.name, png.stem + ".json", png.stem + ".svg", png.stem + ".pdf"} <= written
    seed = recorded_seed(png.with_suffix(".json"))
    strict = seed.isdigit()
    out, _ = run_notebook(nb, tmp_path / "nbrun", seed if strict else "0")
    produced = {p.name for p in out.iterdir()}
    assert produced == written, (produced ^ written)
    a, b = (json.loads((d / (png.stem + ".json")).read_text()) for d in (png.parent, out))
    for name in sorted(written):
        if not name.endswith(".csv"):
            continue
        if strict:
            frames_close(png.parent / name, out / name)
        elif name.endswith("_boundaries.csv"):
            x = a["orientation"]["x_axis"]
            boundaries_close(png.parent / name, out / name, [x, "T_K" if a["kind"] == "binary"
                                                             else a["orientation"]["y_axis"]])
        elif name.endswith("_invariants.csv"):
            ia, ib = pd.read_csv(png.parent / name), pd.read_csv(out / name)
            assert np.allclose(sorted(ia.T_K.unique()), sorted(ib.T_K.unique()), atol=0.5)
    summaries_equivalent(a, b, strict)
    return a, b


# ---------------------------------------------------------------------------------------------------------
# notebooks next to diagrams
# ---------------------------------------------------------------------------------------------------------
def test_binary_notebook_regenerates_all_files(tb, tmp_path):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                "tmax": 2000, "temperature_unit": "C", "composition_step": 0.05,
                                                "temperature_step": 20})
    a, b = check_regenerates(tb, text, imgs, tmp_path)
    assert [i["type"] for i in a["invariants"]] == ["eutectic"] and "labels" in a
    nb = json.loads(Path(imgs[0]).with_suffix(".ipynb").read_text())
    first = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    from calphad_tools.hashseed import current
    assert f"HASH_SEED = {current()!r}" in first or current() == "random"


def test_ternary_notebook_regenerates_all_files(tb, tmp_path):
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["CO", "CR", "NI"],
                                                "temperature": 1000, "temperature_unit": "C"})
    a, _ = check_regenerates(tb, text, imgs, tmp_path)
    assert a["kind"] == "ternary" and any(i["phases"] for i in a["invariants"])


def test_isopleth_notebook_writes_summary(tb, tmp_path):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": ["FE", "CR", "NI"],
                                             "vary": "CR", "fixed": {"NI": 0.1}, "tmin": 900, "tmax": 1900,
                                             "composition_step": 0.05, "temperature_step": 20})
    png = Path(imgs[0])
    out, _ = run_notebook(png.with_suffix(".ipynb"), tmp_path / "iso")      # (no mapping check: plain do_map)
    assert {p.name for p in out.iterdir()} == tool_files(tb, 1)
    a, b = (json.loads((d / (png.stem + ".json")).read_text()) for d in (png.parent, out))
    assert set(a) == set(b) and a["regions"] == b["regions"] and a["phases_found"] == b["phases_found"]
    assert [round(i["T_K"]) for i in a["invariants"]] == [round(i["T_K"]) for i in b["invariants"]]


def test_edited_figure_notebook_includes_the_diagram(tb):
    """An edit (also in place, output = image) refreshes the notebook next to the edited PNG with the diagram it was
    drawn on; an edit of an ordinary plot gets none."""
    from calphad_tools.codegen import build_cells
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                             "tmax": 2000, "temperature_unit": "C", "composition_step": 0.1,
                                             "temperature_step": 40})
    text, im2 = call(tb, "edit_figure", {"image": imgs[0], "operations": [
        {"op": "relabel", "text": "BCC", "new_text": "α-Cr"}, {"op": "move_label", "id": "phase_label:LIQUID",
                                                               "dx": 5, "dy": 5}]})
    nb = Path(im2[0]).with_suffix(".ipynb")
    assert nb.exists() and f"notebook: {tb.rel(str(nb))}" in text
    text, im3 = call(tb, "edit_figure", {"image": im2[0], "output": Path(im2[0]).name,
                                         "operations": [{"op": "delete_label", "id": "invariant_label"}]})
    assert im3 == im2 and "notebook: " in text
    cells, _ = build_cells(tb, "3")
    assert [c.title for c in cells] == ["map_phase_diagram (call #1)", "edit_figure (call #2)",
                                        "edit_figure (call #3)"]
    assert all(c.standalone for c in cells)
    code = code_of(nb)
    assert "α-Cr" in code and "remove_label" in code and "calphad_tools" not in code
    _, im4 = call(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                    "output": "plain.png"})
    text, im5 = call(tb, "edit_figure", {"image": im4[0], "operations": [{"op": "set", "title": "x"}]})
    assert "notebook: " not in text and not Path(im5[0]).with_suffix(".ipynb").exists()


def test_notebook_failure_never_breaks_the_tool(tb, monkeypatch, caplog):
    from calphad_tools import codegen

    def boom(*a, **k):
        raise RuntimeError("generator exploded")
    monkeypatch.setattr(codegen, "build_cells", boom)
    with caplog.at_level(logging.WARNING, logger="calphad_tools.codegen"):
        text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1200,
                                                    "tmax": 2000, "composition_step": 0.1, "temperature_step": 40})
    assert "notebook: " not in text and "Files:" in text and imgs
    assert not Path(imgs[0]).with_suffix(".ipynb").exists()
    msgs = [r.getMessage() for r in caplog.records if r.name == "calphad_tools.codegen"]
    assert len(msgs) == 1 and "generator exploded" in msgs[0] and "\n" not in msgs[0]


def test_switch_off(tmp_path):
    from calphad_tools.agent.config import AgentConfig
    from calphad_tools.agent.tools import ToolBox, UI
    assert AgentConfig().save_notebooks is True
    tb = ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "o"), ui=UI(), python_tool="off",
                 save_notebooks=False)
    text, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1200,
                                                "tmax": 2000, "composition_step": 0.1, "temperature_step": 40})
    assert "notebook: " not in text and not Path(imgs[0]).with_suffix(".ipynb").exists()


def test_show_code_defaults_to_notebook(tb):
    _, imgs = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1200,
                                             "tmax": 2000, "composition_step": 0.1, "temperature_step": 40})
    text, _ = call(tb, "show_code", {"target": imgs[0]})
    stem = Path(imgs[0]).stem
    assert f"code/{stem}_standalone.py" in text and f"code/{stem}_standalone.ipynb" in text
    nb = json.loads((tb.out / "code" / f"{stem}_standalone.ipynb").read_text())
    assert "PYTHONHASHSEED" in "".join(nb["cells"][1]["source"])          # first code cell: the seed check
    text, _ = call(tb, "show_code", {"target": imgs[0], "format": "script"})
    assert ".ipynb" not in text.split("\n")[0]


# ---------------------------------------------------------------------------------------------------------
# the hash seed
# ---------------------------------------------------------------------------------------------------------
_MAP = r'''
import sys
from calphad_tools.mapping_tools import map_phase_diagram
r = map_phase_diagram(sys.argv[1], ["CO", "CR"], sys.argv[2], tmin=800, tmax=1800, temperature_unit="C",
                      composition_step=0.05, temperature_step=20, label_regions=False, formats=[])
print(r["hash_seed"])
'''


def _map_in_process(tmp_path, name, seed):
    d = tmp_path / name
    d.mkdir()
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env.pop("PYTHONHASHSEED", None)
    if seed is not None:
        env["PYTHONHASHSEED"] = seed
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", _MAP,
                        str(ROOT / "examples" / "tdbs" / "CoCrFeNiV_Choi2019.tdb"), str(d / "m.png")],
                       capture_output=True, text=True, env=env, timeout=600, cwd=d)
    assert r.returncode == 0, r.stderr[-3000:]
    return d, r.stdout.strip().splitlines()[-1]


def test_same_seed_same_lines(tmp_path):
    """Two separate processes with PYTHONHASHSEED=0 trace identical lines (byte-identical CSVs); the seed is
    recorded in the summary JSON."""
    (a, sa), (b, sb) = _map_in_process(tmp_path, "a", "0"), _map_in_process(tmp_path, "b", "0")
    assert sa == sb == "0"
    for f in ("m_boundaries.csv", "m_tielines.csv", "m_invariants.csv"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert json.loads((a / "m.json").read_text())["hash_seed"] == "0"
    _, sr = _map_in_process(tmp_path, "r", None)
    assert sr == "random"


def test_seed_check_cell(tmp_path):
    from calphad_tools.codegen import Ctx, seed_code
    ctx = Ctx(None)
    ctx.seeds = ["0", "0"]
    cell = seed_code(ctx, "x.ipynb")
    assert "HASH_SEED = '0'" in cell and "PYTHONHASHSEED=" in cell and "def run_with_recorded_seed" in cell
    for seed, warned in (("0", False), ("7", True), (None, True)):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        if seed is not None:
            env["PYTHONHASHSEED"] = seed
        r = subprocess.run([sys.executable, "-c", cell], capture_output=True, text=True, env=env, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert ("PYTHONHASHSEED=0 jupyter lab" in r.stdout) == warned, (seed, r.stdout)
    ctx.seeds = ["0", "random"]
    assert "random (not recorded)" in seed_code(ctx)
    ctx.seeds = []
    assert seed_code(ctx) == ""


def test_cli_reexecs_with_seed_and_workers_get_it(tmp_path):
    """Without PYTHONHASHSEED a command-line program re-executes itself with PYTHONHASHSEED=0; a seed the user set
    is kept; worker / job processes get the default too."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHASHSEED", "CALPHAD_HASHSEED_REEXEC")}
    prog = ("import sys; sys.argv = ['prog']; from calphad_tools.hashseed import reexec_with_seed, current; "
            "reexec_with_seed('calphad_tools.hashseed'); print('not re-executed', current())")
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "0", (r.stdout, r.stderr)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                       env={**env, "PYTHONHASHSEED": "123"}, cwd=tmp_path)
    assert r.stdout.strip() == "not re-executed 123"
    from calphad_tools import hashseed
    assert hashseed.child_env({})["PYTHONHASHSEED"] == "0"
    assert hashseed.child_env({"PYTHONHASHSEED": "5"})["PYTHONHASHSEED"] == "5"
    before = os.environ.get("PYTHONHASHSEED")
    with hashseed.for_children():
        assert os.environ.get("PYTHONHASHSEED") == (before or "0")
    assert os.environ.get("PYTHONHASHSEED") == before


def test_launchers_set_the_seed():
    sh = (ROOT / "install.sh").read_text()
    assert 'export PYTHONHASHSEED="\\${PYTHONHASHSEED:-0}"' in sh
    assert "if not defined PYTHONHASHSEED set PYTHONHASHSEED=0" in (ROOT / "install.ps1").read_text()
