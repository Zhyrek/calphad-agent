"""Check sheet + digitize_fix: the user tells how many markers a doubtful blob is; the centres are refitted."""
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import digitize_overlap_bench as B  # noqa: E402


@pytest.fixture(scope="module")
def fig(tmp_path_factory):
    out = tmp_path_factory.mktemp("fix")
    path, truth, ux, uy, ytop = B.build("realistic", out)
    from calphad_tools.agent.tools import UI, ToolBox
    tb = ToolBox(workdir=str(out), output_dir=str(out / "out"), ui=UI(), python_tool="off")
    text, _, err = tb.call("digitize_setup", {"image": str(path)})
    assert not err, text
    nx = int(re.search(r"x0\.\.x(\d+)", text).group(1)) + 1
    ny = int(re.search(r"y0\.\.y(\d+)", text).group(1)) + 1
    axes = {"x": {"ticks": {"0": 0, str(nx - 1): 100}}, "y": {"ticks": {"0": ytop, str(ny - 1): 0}}}
    return {"tb": tb, "path": str(path), "truth": truth, "ux": ux, "uy": uy, "axes": axes, "out": out / "out"}


def _rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _extract(fig, r):
    tb = fig["tb"]
    text, imgs, err = tb.call("digitize_points", {"image": fig["path"], "axes": fig["axes"], **B.FILTERS[r][0],
                                                  "series": f"r{r}", "output_csv": f"r{r}.csv"})
    assert not err, text
    return text, imgs, fig["out"] / f"r{r}.csv"


def _cell(fig, rows, r, sep):
    tr = [t for t in fig["truth"] if t["row"] == r and t["sep"] == sep]
    det = [(float(q["x"]), float(q["y"])) for q in rows]
    err = max(min(np.hypot((x - t["x"]) / fig["ux"], (y - t["y"]) / fig["uy"]) for x, y in det) for t in tr)
    xc = np.mean([t["x"] for t in tr])
    n = sum(1 for x, y in det if abs(x - xc) < 4 and abs(y - tr[0]["y"]) < 4)
    return err, n, len(tr)


def _answers(fig, rows, r, k):
    """What the user would say: the true count for every doubtful blob of the close cells."""
    counts = {}
    for c, sep in enumerate(B.SEPS):
        if sep > 0.6:
            continue
        xc, yc = 8 + c * 11.7, 90 - r * 15
        inside = [q for q in rows if abs(float(q["x"]) - xc) < 4 and abs(float(q["y"]) - yc) < 4]
        if inside and (len(inside) != k or any(q["note"] for q in inside)):
            counts[inside[0]["index"]] = k
    return counts


@pytest.mark.parametrize("r,k", [(0, 2), (4, 2), (5, 3)])
def test_counts_resolve_heavy_overlap(fig, r, k):
    text, imgs, p = _extract(fig, r)
    sheet = next(i for i in imgs if i.endswith(".check.png"))
    assert Path(sheet).exists() and "CHECK SHEET" in text and "digitize_fix" in text
    before = _rows(p)
    n_other = len(before) - sum(_cell(fig, before, r, s)[1] for s in B.SEPS if s <= 0.6)
    counts = _answers(fig, before, r, k)
    assert counts
    t, imgs2, err = fig["tb"].call("digitize_fix", {"csv": str(p), "counts": counts})
    assert not err, t
    after = _rows(p)
    for s in B.SEPS:
        if s <= 0.6:
            e, n, ntrue = _cell(fig, after, r, s)
            assert n == ntrue and e < 0.15, (s, e, n)
    assert len(after) - sum(_cell(fig, after, r, s)[1] for s in B.SEPS if s <= 0.6) == n_other  # rest untouched
    assert all("confirmed by user" in q["note"] for q in after if q["note"].startswith("count"))
    assert any(i.endswith(".fixed.png") for i in imgs2) and any(i.endswith("_provenance.png") for i in imgs2)


def test_check_sheet_one_tile_per_blob():
    from calphad_tools.digitize_fix import group_blobs
    items = [{"index": 11, "cx": 100, "cy": 100, "size": 12, "why": "split from a blob"},
             {"index": 12, "cx": 110, "cy": 100, "size": 12, "why": "split from a blob"},
             {"index": 13, "cx": 105, "cy": 91, "size": 12, "why": "split from a blob"},
             {"index": 32, "cx": 300, "cy": 100, "size": 12, "why": "may be several markers"}]
    g = group_blobs(items)
    assert [t["members"] for t in g] == [[32], [11, 12, 13]]         # flagged first, split blob as one tile


def test_zero_add_remove_and_show_code(fig, tmp_path):
    text, imgs, p = _extract(fig, 1)
    rows = _rows(p)
    iso = [q for q in rows if not q["note"] and float(q["y"]) > 105]  # isolated markers in the upper half
    victim, n0 = iso[0], len(rows)
    tx, ty = float(victim["x"]), float(victim["y"])
    t, _, err = fig["tb"].call("digitize_fix", {"csv": str(p), "counts": {victim["index"]: 0}})
    assert not err, t
    rows = _rows(p)
    assert len(rows) == n0 - 1 and not any(abs(float(q["x"]) - tx) < 0.5 and abs(float(q["y"]) - ty) < 1
                                           for q in rows)
    # the user says "there is one here": given roughly (off by ~0.3 marker diameters), snapped back onto it
    t, _, err = fig["tb"].call("digitize_fix", {"csv": str(p), "add": [[tx + 0.3 * fig["ux"], ty]]})
    assert not err and "snapped" in t, t
    rows = _rows(p)
    got = min(rows, key=lambda q: np.hypot(float(q["x"]) - tx, float(q["y"]) - ty))
    assert np.hypot((float(got["x"]) - tx) / fig["ux"], (float(got["y"]) - ty) / fig["uy"]) < 0.1
    assert "added by user" in got["note"]
    t, _, err = fig["tb"].call("digitize_fix", {"csv": str(p), "remove": [int(got["index"])]})
    assert not err and len(_rows(p)) == n0 - 1
    code, _, err = fig["tb"].call("show_code", {"target": str(p), "format": "script"})
    assert not err, code
    script = code.split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert "calphad_tools" not in script and "digitize_fix" in script
    f = tmp_path / "s.py"
    f.write_text(script)
    r = subprocess.run([sys.executable, "-W", "ignore", str(f)], cwd=tmp_path, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
