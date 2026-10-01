"""Tests for the offline pycalphad documentation tools (calphad_tools/pycalphad_docs.py)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    from calphad_tools.agent.tools import ToolBox, UI

    class NoHumanUI(UI):
        def ask(self, question, options=None):
            return "no"

        def confirm(self, title, body):
            return False

    out = tmp_path_factory.mktemp("docs_out")
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out / "out"), ui=NoHumanUI(), python_tool="off")


def _call(tb, name, args):
    text, images, err = tb.call(name, args)
    assert not err, text
    assert images == []
    return text


def test_tools_registered(tb):
    assert "pycalphad_help" in tb.tools
    assert "list_calphad_capabilities" in tb.tools
    # descriptions stay short (they cost prompt tokens)
    assert len(tb.tools["pycalphad_help"].spec.description) < 400
    assert len(tb.tools["list_calphad_capabilities"].spec.description) < 300


@pytest.mark.parametrize("query, expected", [
    ("isopleth", ["IsoplethStrategy", "plot_isopleth"]),
    ("driving force", ["DormantPhase", "driving_force"]),
    ("reference state", ["ReferenceState", "shift_reference_state"]),
    ("to_file", ["Database.to_file", "if_exists"]),
    ("export tieline data", ["get_tieline_data"]),
    ("scheil", ["simulate_scheil_solidification"]),
    ("property strings", ["NP(FCC_A1)", "X(FCC_A1,CR)", "MU(CR)"]),
])
def test_help_search(tb, query, expected):
    text = _call(tb, "pycalphad_help", {"query": query})
    for word in expected:
        assert word in text, f"{word!r} missing from help for {query!r}:\n{text[:2000]}"
    assert text.count("\n### ") <= 5          # at most 5 results


def test_help_exact_path_workspace(tb):
    text = _call(tb, "pycalphad_help", {"query": "pycalphad.core.workspace.Workspace"})
    assert text.startswith("pycalphad.core.workspace.Workspace")
    assert "Reactive equilibrium calculator" in text
    assert "signature: Workspace(database, components, phases, conditions" in text
    assert "public members:" in text
    for m in (".get(", ".get_dict(", ".enumerate_composition_sets("):
        assert m in text


def test_help_exact_path_strategy_and_short_name(tb):
    text = _call(tb, "pycalphad_help", {"query": "pycalphad.mapping.BinaryStrategy"})
    assert "get_tieline_data" in text and "get_invariant_data" in text and "do_map" in text
    text = _call(tb, "pycalphad_help", {"query": "Workspace"})      # resolves pycalphad.Workspace
    assert "Reactive equilibrium calculator" in text


def test_help_unknown_and_empty(tb):
    assert "No offline pycalphad documentation" in _call(tb, "pycalphad_help", {"query": "zzqqxx"})
    assert "dotted path" in _call(tb, "pycalphad_help", {"query": ""})


def test_capabilities(tb):
    text = _call(tb, "list_calphad_capabilities", {})
    # task map is built from what is actually registered
    assert "TASK -> TOOL" in text
    assert "map_phase_diagram" in text and "equilibrium_property_plot" in text and "calculate_equilibrium" in text
    assert "pycalphad_help" in text
    for name in tb.tools:
        assert name in text
    # conventions for offline use
    for s in ("v.X('CR')", "v.MU", "v.NP", "'NP(FCC_A1)'", "'X(FCC_A1,CR)'", "'MU(CR)'", "Y(PHASE,subl,SPECIES)",
              "Kelvin", "ReferenceState"):
        assert s in text, s
    assert len(text) < 20000


def test_capabilities_survives_broken_plugin(tb, monkeypatch):
    import calphad_tools.pycalphad_docs as d
    monkeypatch.setattr(d, "PLUGIN_MODULES", d.PLUGIN_MODULES + ["calphad_tools.does_not_exist_xyz"])
    text = d.capabilities_text(tb)
    assert "does_not_exist_xyz: not available" in text


def test_inventory_files_consistent(tmp_path):
    import calphad_tools.pycalphad_docs as d
    jpath, mpath = d.write_inventory_files(str(tmp_path))
    data = json.loads(Path(jpath).read_text())
    assert len(data) > 100
    assert {"name", "path", "kind", "signature", "summary"} <= set(data[0])
    assert all(e["kind"] in ("plotting", "data-output", "both", "support") for e in data)
    assert all(e["signature"] != "(unavailable)" for e in data), [e["path"] for e in data
                                                                  if e["signature"] == "(unavailable)"]
    paths = {e["path"] for e in data}
    for p in ("pycalphad.equilibrium", "pycalphad.mapping.plot_isopleth", "pycalphad.Database.to_file",
              "pycalphad.property_framework.IsolatedPhase", "pycalphad.plot.utils.phase_legend"):
        assert p in paths
    # the committed docs are in sync with the module
    committed = ROOT / "docs" / "pycalphad_inventory.json"
    if committed.exists():
        assert {e["path"] for e in json.loads(committed.read_text())} == paths
    assert "## mapping" in Path(mpath).read_text()
