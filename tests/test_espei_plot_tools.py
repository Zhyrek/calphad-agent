import json
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
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(tmp_path / "out"), ui=U(), python_tool="off")


@pytest.fixture
def datasets(tmp_path):
    from calphad_tools import expdata
    d = tmp_path / "ds"
    zpf, _ = expdata.pop_to_zpf(str(ROOT / "examples/expdata/CrNi_boundaries.POP"), ["CR", "NI"],
                                tdb_phases=["LIQUID", "FCC_A1", "BCC_A2"])
    for i, ds in enumerate(zpf):
        expdata.write_dataset(ds, str(d / f"z{i}.json"))
    (d / "hm.json").write_text(json.dumps({
        "components": ["CR", "NI"], "phases": ["CRNI2"], "output": "HM_FORM", "reference": "synthetic",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1, 2], "sublattice_configurations": [["CR", "NI"]]},
        "conditions": {"P": 101325, "T": 298.15}, "values": [[[-8000]]]}))
    return str(d)


def test_overlay_and_data_only(tb, datasets):
    text, imgs, err = tb.call("plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": datasets,
                                                         "tdb": "CrNi-06Hal.tdb", "tmin": 800, "tmax": 2000,
                                                         "temperature_unit": "C"})
    assert not err and "4 ZPF points" in text and Path(imgs[0]).stat().st_size > 20000
    text, imgs, err = tb.call("plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": datasets})
    assert not err and "data only" in text


def test_parameter_vs_data(tb, datasets):
    text, imgs, err = tb.call("plot_parameter_vs_data", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                         "phase": "CRNI2", "configuration": ["CR", "NI"],
                                                         "output": "HM_FORM", "datasets_dir": datasets})
    assert not err and "endmember" in text and imgs
    text, imgs, err = tb.call("plot_parameter_vs_data", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                         "phase": "FCC_A1", "configuration": [["CR", "NI"], "VA"],
                                                         "output": "HM_MIX"})
    assert not err and "interaction" in text
    text, _, err = tb.call("plot_parameter_vs_data", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"],
                                                      "phase": "NOPE", "configuration": ["CR"], "output": "HM_FORM"})
    assert err and "phases:" in text


def test_convert_composition(tb):
    text, _, err = tb.call("convert_composition", {"composition": {"CR": 18, "NI": 8}, "balance": "FE",
                                                   "from_unit": "wt%"})
    assert not err and "19.15" in text and '"X(CR)": 0.1915' in text
    text, _, err = tb.call("convert_composition", {"composition": {"CR": 0.2}, "balance": "NI",
                                                   "from_unit": "mole_fraction", "tdb": "CrNi-06Hal.tdb"})
    assert not err and "18.13" in text
    _, _, err = tb.call("convert_composition", {"composition": {"CR": 80, "NI": 30}, "balance": "FE",
                                                "from_unit": "wt%"})
    assert err
