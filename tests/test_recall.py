"""'@last' references, recent_results and rerun_call (calphad_tools/recall.py)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tb(out):
    from calphad_tools.agent.tools import UI, ToolBox

    class U(UI):
        def ask(self, q, o=None):
            return ""

        def confirm(self, t, b):
            return False
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out), ui=U(), python_tool="off")


def ok(tb, name, args):
    text, images, err = tb.call(name, args)
    assert not err, text
    return text, images


def _plot(tb, name="first.png", y=(0, 1)):
    return ok(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": list(y)}]}],
                                "output": name})


def test_add_point_to_last_plot(tmp_path):
    from calphad_tools.figures import load_figure
    tb = _tb(tmp_path / "out")
    _, im1 = _plot(tb)
    text, im2 = ok(tb, "edit_figure", {"image": "@last", "operations": [
        {"op": "scatter", "x": [0.5], "y": [0.7], "color": "red", "label": "new point"}], "output": "edited.png"})
    assert "@last = " in text and "first.png" in text.splitlines()[0]
    fig = load_figure(im2[0])
    assert len(fig.axes[0].collections) == 1 and len(fig.axes[0].lines) == 1
    # the log holds the resolved path, not '@last' (provenance / show_code / replays see a real file)
    assert tb.call_log[-1]["args"]["image"].endswith("first.png")
    code, _, err = tb.call("show_code", {"target": "@last"})
    assert not err and [l for l in code.splitlines() if l.startswith("# %%")] == [
        "# %% make_plot (call #1)", "# %% edit_figure (call #2)"]


def test_extension_tool_and_call_refs(tmp_path):
    tb = _tb(tmp_path / "out")
    _plot(tb, "a.png")
    ok(tb, "fit_data", {"x": [1, 2, 3, 4], "y": [2.1, 3.9, 6.2, 7.9], "model": "linear", "output_csv": "fit.csv"})
    _plot(tb, "b.png")
    from calphad_tools.recall import resolve
    assert resolve(tb, "@last").endswith("b.png")
    assert resolve(tb, "@last.csv").endswith(".csv")
    assert resolve(tb, "@last:make_plot").endswith("b.png")
    assert resolve(tb, "@call:1").endswith("a.png")
    assert resolve(tb, "@call:2.csv").endswith(".csv")
    text, _, err = tb.call("inspect_data", {"path": "@last.csv"})
    assert not err, text
    text, _, err = tb.call("edit_figure", {"image": "@call:99", "operations": []})
    assert err and "no call 99" in text
    listing, _ = ok(tb, "recent_results", {"n": 3})
    lines = [l for l in listing.splitlines() if l.startswith("- ")]
    assert lines[0].startswith("- @call:3") and "b.png" in listing and len(lines) == 3


def test_rerun_with_changes(tmp_path):
    from calphad_tools.figures import load_figure
    tb = _tb(tmp_path / "out")
    ok(tb, "calculate_equilibrium", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                     "conditions": {"T": 1000, "X(NI)": 0.4}})
    _plot(tb, "p.png", y=(0, 1))
    text, images = ok(tb, "rerun_call", {"call": "last", "changes": {"output": "p2.png",
                                                                        "panels": [{"operations": [
                                                                            {"op": "line", "x": [0, 1],
                                                                             "y": [5, 6]}]}]}})
    assert text.startswith("re-ran call 2 (make_plot)") and images[0].endswith("p2.png")
    assert list(load_figure(images[0]).axes[0].lines[0].get_ydata()) == [5, 6]
    # the inner call is logged as the producer; rerun_call is neither exported nor a show_code producer
    assert [c["tool"] for c in tb.call_log][-2:] == ["make_plot", "rerun_call"]
    text, _ = ok(tb, "rerun_call", {"call": "last:calculate_equilibrium",
                                    "changes": {"conditions": {"T": 1200, "X(NI)": 0.4}}})
    assert "re-ran call 1 (calculate_equilibrium)" in text and "1200" in json.dumps(tb.call_log[-2]["args"])
    code, _, err = tb.call("show_code", {"target": "p2.png"})
    assert not err and "# %% make_plot" in code and "rerun_call" not in code


def test_falls_back_to_previous_session(tmp_path):
    out = tmp_path / "out"
    tb1 = _tb(out)
    _plot(tb1, "yesterday.png")
    tb2 = _tb(out)                                     # a new session (e.g. after restarting the agent)
    from calphad_tools.recall import resolve
    assert resolve(tb2, "@last").endswith("yesterday.png")
    listing, _ = ok(tb2, "recent_results", {})
    assert "earlier session" in listing and "yesterday.png" in listing
    text, images = ok(tb2, "rerun_call", {"call": "last", "changes": {"output": "today.png"}})
    assert images[0].endswith("today.png")
    with pytest.raises(ValueError):
        resolve(tb2, "@call:1.csv")
