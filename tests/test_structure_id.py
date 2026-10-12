"""Structure identification for compound labels (calphad_tools.structure_id): offline, with saved AFLOW / OQMD
responses (tests/data/structure_id) and a monkeypatched HTTP layer."""
import json
import os
import re
import sys
from pathlib import Path

import pytest

from calphad_tools import structure_id as sid

_real_http = sid._http_get_json

DATA = Path(__file__).parent / "data" / "structure_id"


class FakeHTTP:
    """Serves the saved responses; records every URL; ``fail`` = raise like a dead network."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.fail:
            raise OSError("network unreachable")
        m = re.search(r"species\(([^)]*)\)", url)
        if m:
            p = DATA / f"aflow_{'_'.join(sorted(m.group(1).split(',')))}.json"
            return json.loads(p.read_text()) if p.exists() else []
        m = re.search(r"composition=([A-Za-z0-9]+)", url)
        if m:
            key = sid.formula_key(sid.parse_formula(m.group(1)))
            for p in DATA.glob("oqmd_*.json"):
                if sid.formula_key(sid.parse_formula(p.stem[5:])) == key:
                    return json.loads(p.read_text())
            return {"data": [], "response_message": "OK"}
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setenv("CALPHAD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("CALPHAD_STRUCTURE_ONLINE", raising=False)
    sid._DOWN.clear()
    sid._SYSTEM_MEMO.clear()
    sid.configure(online=True, overrides={})
    http = FakeHTTP()
    monkeypatch.setattr(sid, "_http_get_json", http)
    yield http
    sid._SETTINGS = None


def test_table_families():
    assert sid.identify("CeCo5").suffix == "1:5" and sid.identify("CeCo5").source == "table"
    for f in ("ZrNi5", "HfNi5", "UNi5"):                                  # AuBe5-type: C15b, not 1:5
        r = sid.identify(f)
        assert r.suffix == "C15_{b}" and r.prototype == "AuBe5" and r.source == "table", r.line()
    for f, proto, sfx in (("WAl12", "WAl12", None), ("MoSi2", "MoSi2", "C11_{b}"), ("CuAl2", "CuAl2", "C16")):
        r = sid.identify(f)
        assert r.suffix == sfx and r.prototype == proto
    assert sid.identify("Fe2Nb").suffix == "C14" and sid.identify("MgCu2").suffix == "C15"
    assert sid.identify("Fe7Mo6").suffix == "μ" and sid.identify("FeCr").suffix == "σ"
    assert sid.identify({"CR": 0.47, "FE": 0.53}).suffix == "σ"
    assert sid.identify("CECO5").formula == "CeCo5"                  # TDB capitals
    r = sid.identify("Ce2Fe17")
    assert r.suffix == "2:17R" and "user, 2026-10-12" in r.evidence
    assert r.alternatives[0]["suffix"] == "2:17H" and r.alternatives[0]["kind"] == "model"
    assert sid.identify("Ce2Co17").suffix == "2:17H"


def test_strukturbericht_suffixes():
    expect = {"Nb3Sn": ("A15", "A15"), "Ni3Al": ("L1_{2}", "L12"), "MoSi2": ("C11_{b}", "C11b"),
              "ZrNi5": ("C15_{b}", "C15b"), "CeCo5": ("1:5", "1:5"), "FeCr": ("σ", "σ"), "Fe2Nb": ("C14", "C14"),
              "Ti3Al": ("D0_{19}", "D019"), "Ni3Nb": ("D0_{a}", "D0a"), "Ti2Ni": ("E9_{3}", "E93"),
              "Cr23C6": ("D8_{4}", "D84"), "NiAl": ("B2", "B2"), "TiAl": ("L1_{0}", "L10")}
    for f, (sfx, pl) in expect.items():
        r = sid.identify(f, allow_online=False)
        assert (r.suffix, r.suffix_plain) == (sfx, pl), r.line()
        assert r.as_dict()["suffix_plain"] == pl
    assert sid.identify("CeCo5").strukturbericht == "D2_{d}"          # the family name wins over D2d
    assert sid.identify("FeCr").strukturbericht == "D8_{b}"           # σ wins over D8b
    for el in ("Cu", "Fe", "Mn"):                                     # pure elements: never A1 / A2 / A12
        r = sid.identify(el)
        assert r.suffix is None and "pure element" in r.evidence
    assert sid.name_evidence("CBCC_A12") is None and sid.name_evidence("FCC_A1") is None and sid.name_evidence("CUB_A13") is None
    assert sid.name_evidence("NI3TI_D024")[1] == "D0_{24}" and sid.name_evidence("TIAL_L1_0")[1] == "L1_{0}"
    assert sid.identify("Fe2Nb", overrides={"FE2NB": "L12"}).suffix == "L1_{2}"   # plain -> markup
    assert sid.plain("D0_{19}") == "D019" and sid.plain("1:5") == "1:5"
    # same space group / cell / ratio: told apart by chemistry (Th6Mn23 D8a vs Cr23C6 D84)
    assert sid.label_structure("A23B6_cF116_225_ad2f_e", {"Fe": 23, "Zr": 6}).prototype == "Th6Mn23"
    assert sid.match_structure(225, 116, (6, 23), {"Cr", "C"}).prototype == "Cr23C6"
    # a σ-type aluminide is not "σ" but keeps the (non-elemental) designation
    assert sid._suffix_for(sid.BY_PROTO["σ-CrFe"], {"Al", "Nb"}) == "D8_{b}"
    assert sid._suffix_for(sid.BY_PROTO["α-Mn"], {"Al", "Mg"}) is None   # A12 is an element type
    # rhombohedral Al8Cr5 is D8_10, not the cubic γ-brass D8_2
    assert sid.label_structure("A8B5_hR26_160_a3bc_a3b", {"Al": 8, "Cr": 5}).strukturbericht == "D8_{10}"
    assert sid.identify("Al8Cr5", allow_online=False).suffix == "D8_{10}"


def test_computed_only_structure_gives_no_suffix():
    # Al4Cr: AFLOW's only answer is a hypothetical MoNi4-type decoration (no ICSD entry) - quoted, no D1a
    g = {"label": "A4B_tI10_87_h_a", "sg": 87, "n": 1, "n_icsd": 0, "H": -0.072, "id": "aflow:x"}
    r = sid._combine({"aflow": [g]}, {"Al": 4, "Cr": 1}, "Al4Cr", {"aflow": False})
    assert r.suffix is None and r.prototype == "MoNi4" and r.confidence == "low"
    assert "no ICSD report" in r.evidence
    g["n_icsd"] = 2                                                     # reported: the suffix stands
    assert sid._combine({"aflow": [g]}, {"Al": 4, "Cr": 1}, "Al4Cr", {"aflow": False}).suffix == "D1_{a}"


def test_217_polymorphs_and_assign():
    r = sid.identify("Sm2Co17")
    assert r.suffix == "2:17" and {a["suffix"] for a in r.alternatives} == {"2:17R", "2:17H"}
    by_name = sid.assign_217([{"phase": "SM2CO17_H"}, {"phase": "SM2CO17_R"}])
    assert by_name == {"SM2CO17_H": "2:17H", "SM2CO17_R": "2:17R"}
    assert sid.assign_217([{"phase": "TH2NI17_TYPE"}, {"phase": "X217"}]) == {"TH2NI17_TYPE": "2:17H",
                                                                                "X217": "2:17R"}
    by_t = sid.assign_217_detail([{"phase": "A", "T": 1200}, {"phase": "B", "T": 1550}], allow_online=False)
    assert by_t["B"]["suffix"] == "2:17H" and by_t["A"]["suffix"] == "2:17R" and by_t["A"]["confidence"] == "low"
    by_f = sid.assign_217([{"phase": "P1", "formula": "Ce2Co17"}, {"phase": "P2", "formula": "Ce2Fe17"}])
    assert by_f == {"P1": "2:17H", "P2": "2:17R"}
    assert sid.assign_217([{"phase": "A"}, {"phase": "B"}], allow_online=False) == {"A": "2:17", "B": "2:17"}


def test_override_beats_everything_and_names_beat_table():
    ov = {"C14_LAVES": "C15", "CECO5": "CeCo₅ (1:5)", "ZRNI5": "1:5"}
    r = sid.identify("Fe2Nb", phase_name="C14_LAVES", overrides=ov)
    assert r.suffix == "C15" and r.source == "override"
    r = sid.identify("CeCo5", overrides=ov)
    assert r.label == "CeCo₅ (1:5)" and r.suffix == "1:5"
    assert sid.identify("ZrNi5", overrides=ov).suffix == "1:5"           # the user is always right
    sid.configure(online=False, overrides={"FE2NB": ""})
    assert sid.identify("Fe2Nb").suffix is None and sid.identify("Fe2Nb").source == "override"
    sid.configure(online=False)
    r = sid.identify("Fe2Nb", phase_name="C15_LAVES")                     # name beats the table (noted)
    assert r.suffix == "C15" and r.source == "phase_name" and "the name wins" in r.evidence
    r = sid.identify("Fe2Nb", phase_name="LAVES")                         # family-level name refined by the table
    assert r.suffix == "C14" and r.source == "table" and "LAVES" in r.evidence
    assert sid.identify("Fe7Mo6", phase_name="FE7MO6_MU").source == "phase_name"
    assert sid.name_evidence("MUMETAL") is None and sid.name_evidence("CBCC_A12") is None
    assert sid.name_evidence("D2_D")[1] == "1:5" and sid.name_evidence("SIGMA_D8B")[1] == "σ"
    assert sid.name_evidence("CE2CO17_R", {"Ce": 2, "Co": 17})[1] == "2:17R"
    assert sid.name_evidence("AL8CR5_H") is None                          # _H outside a 2:17 context
    assert sid.identify("Ce2Ni17", phase_name="RE2TM17", allow_online=False).suffix == "2:17"


def test_online_aflow_oqmd_fixtures(clean):
    r, notes = sid.lookup_online("CeCo5")
    assert r.suffix == "1:5" and r.source == "aflow" and r.aflow_label == "AB5_hP6_191_a_cg", r.line()
    assert "OQMD: Cmmm" in r.evidence and not notes                      # OQMD's relaxed cell is not counted
    for f in ("ZrNi5", "HfNi5", "UNi5"):                                  # mapped to the Strukturbericht type
        r, _ = sid.lookup_online(f)
        assert r.suffix == "C15_{b}" and r.suffix_plain == "C15b" and r.prototype == "AuBe5", r.line()
    r, _ = sid.lookup_online("WAl12")
    assert r.prototype == "WAl12" and r.suffix is None
    r, _ = sid.lookup_online("Fe2Nb")
    assert r.suffix == "C14" and {a["suffix"] for a in r.alternatives} >= {"C15", "C36"}
    assert sid.lookup_online("Fe7Mo6")[0].suffix == "μ"
    assert sid.lookup_online("Y2Co17")[0].suffix == "2:17"                # both polymorphs in ICSD
    r, _ = sid.lookup_online("Sm2Co17")                                   # AFLOW has none; OQMD has R and H
    assert r.suffix == "2:17" and r.source == "oqmd" and len(r.alternatives) == 2
    r, _ = sid.lookup_online("HfNi5")                                     # both databases agree
    assert r.prototype == "AuBe5" and r.confidence == "high" and "OQMD entry 19424" in r.evidence
    r, _ = sid.lookup_online("Fe2Nb")       # OQMD's DFT ground state is C15; the ICSD reports decide
    assert r.suffix == "C14" and r.confidence == "medium" and "MgZn2 has most ICSD reports" in r.evidence
    assert any("species(Ce,Co)" in u for u in clean.calls) and any("composition=CeCo5" in u for u in clean.calls)
    assert all(u.startswith(("https://aflowlib.duke.edu/", "https://oqmd.org/")) for u in clean.calls)


def test_cache_round_trip(clean, monkeypatch):
    r1, _ = sid.lookup_online("ZrNi5")
    assert r1.source == "aflow" and sid.cache_path().exists()
    cached = json.loads(sid.cache_path().read_text())["entries"]["Ni5Zr"]
    assert set(cached) == {"aflow", "oqmd"} and cached["aflow"]["groups"]
    dead = FakeHTTP(fail=True)
    monkeypatch.setattr(sid, "_http_get_json", dead)
    r2, notes = sid.lookup_online("ZrNi5")
    assert r2.source == "cache" and r2.prototype == "AuBe5" and not dead.calls and not notes
    r3 = sid.identify("ZrCo5", allow_online=False)                        # not cached, offline: no suffix
    assert r3.suffix is None and r3.source == "none" and not dead.calls


def test_network_failure_falls_through(monkeypatch):
    dead = FakeHTTP(fail=True)
    monkeypatch.setattr(sid, "_http_get_json", dead)
    r = sid.identify("LaCu5")
    assert r.suffix == "1:5" and r.source == "chemistry_rule" and r.confidence == "low"
    assert any("not reachable" in n for n in r.notes)
    n = len(dead.calls)
    sid.identify("PrCu5")                                                 # a dead host is not retried at once
    assert len(dead.calls) == n
    for f in ("ZrCo5", "Al5La", "HfCo5"):
        assert sid.chemistry_rule(f) is None
    assert sid.chemistry_rule("Ce2Zn17").suffix == "2:17"
    assert sid.chemistry_rule("CeZn5") is None


def test_offline_makes_no_http_call(clean, monkeypatch):
    sid.identify("LaCu5", allow_online=False)
    sid.configure(online=False)
    sid.identify("PrCu5")
    sid.identify("PrCu5", allow_online=True)                              # a call cannot override the config
    sid.configure(online=True)
    monkeypatch.setenv("CALPHAD_STRUCTURE_ONLINE", "0")
    r = sid.identify("NdCu5")
    assert not clean.calls and "CALPHAD_STRUCTURE_ONLINE" in " ".join(r.notes)


def test_config_fields(tmp_path, monkeypatch):
    from calphad_tools.agent.config import load_config
    p = tmp_path / "c.toml"
    p.write_text('[structure]\nonline = false\ntimeout_s = 5\n[labels]\nphase_labels = { CECO5 = "1:5" }\n')
    cfg = load_config(str(p))
    assert cfg.structure.online is False and cfg.structure.timeout_s == 5
    assert cfg.labels.phase_labels == {"CECO5": "1:5"}
    s = sid.configure(cfg)
    assert s.online is False and s.overrides == {"CECO5": "1:5"}
    monkeypatch.setenv("CALPHAD_STRUCTURE_ONLINE", "0")
    p.write_text("[structure]\nonline = true\n")
    assert load_config(str(p)).structure.online is False


def test_phase_family_endmembers_and_warning():
    pf = sid.phase_family("CE2TM17", ["Ce2Fe17", "Ce2Co17"], online=False)
    assert pf.per_endmember["Ce2Fe17"].suffix == "2:17R" and pf.per_endmember["Ce2Co17"].suffix == "2:17H"
    assert pf.consensus is None and pf.ambiguous_label == "2:17" and pf.suffix is None
    assert pf.warning and "2:17R (Ce2Fe17)" in pf.warning and "2:17H (Ce2Co17" in pf.warning
    assert "be careful with these labels" in pf.warning
    pf = sid.phase_family("CE2FE17_R", ["Ce2Fe17", "Ce2Co17"], online=False)
    assert pf.name_suffix == "2:17R" and pf.suffix == "2:17R" and "R" in pf.name_evidence
    pf = sid.phase_family("CECO5", ["CeCo5"], online=False)
    assert pf.consensus == "1:5" and pf.suffix == "1:5" and pf.warning is None
    pf = sid.phase_family("SM2CO17", ["Sm2Co17"], online=False)           # one end-member, two polymorphs
    assert pf.consensus is None and pf.ambiguous_label == "2:17" and "polymorph" in pf.warning
    pf = sid.phase_family("CE2FE17", ["Ce2Fe17"], online=False)           # a binary: modelling note, no warning
    assert pf.consensus == "2:17R" and pf.warning is None
    pf = sid.phase_family("C14_LAVES", ["Fe2Nb"], online=False, overrides={"C14_LAVES": "C36"})
    assert pf.suffix == "C36" and "override" in pf.name_evidence


TDB = """
ELEMENT VA   VACUUM     0 0 0 !
ELEMENT CE   FCC_A1     140.1 0 0 !
ELEMENT CO   HCP_A3     58.9 0 0 !
ELEMENT FE   BCC_A2     55.8 0 0 !
ELEMENT NB   BCC_A2     92.9 0 0 !
PHASE LIQUID % 1 1.0 !
CONSTITUENT LIQUID :CE,CO,FE,NB: !
PHASE FCC_A1 % 2 1 1 !
CONSTITUENT FCC_A1 :CE,CO,FE:VA: !
PHASE CECO5 % 2 1 5 !
CONSTITUENT CECO5 :CE:CO: !
PHASE CE2TM17 % 2 2 17 !
CONSTITUENT CE2TM17 :CE:CO,FE: !
PHASE C14_LAVES % 2 2 1 !
CONSTITUENT C14_LAVES :FE,NB:FE,NB: !
PHASE MYSTERY % 2 3 2 !
CONSTITUENT MYSTERY :CO:NB: !
"""


def _toolbox(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from calphad_tools.agent.tools import UI, ToolBox
    (tmp_path / "t.tdb").write_text(TDB)
    return ToolBox(str(tmp_path), "out", UI(), python_tool="off", interactive=False)


def test_tool_output_and_help(tmp_path, clean):
    tb = _toolbox(tmp_path)
    text, _, err = tb.call("identify_structure", {"formula": "CeCo5, ZrNi5", "online": False})
    assert not err
    lines = text.splitlines()
    assert lines[0].startswith("CeCo5: 1:5 | CaCu5-type") and "table, high" in lines[0]
    assert lines[1].startswith("ZrNi5: C15_{b} | AuBe5-type") and "off for this call" in lines[-1]
    text, _, err = tb.call("identify_structure", {"tdb": "t.tdb", "online": False})
    assert not err and "4 compound phase(s)" in text, text
    assert "CECO5: 1:5" in text and "C14_LAVES: C14" in text and "MYSTERY: no suffix" in text
    assert "WARNING: CE2TM17 is a single CALPHAD phase representing both 2:17R (Ce2Fe17) and 2:17H (Ce2Co17" \
        in text
    assert "LIQUID" not in text and "FCC_A1" not in text and not clean.calls
    text, _, err = tb.call("identify_structure", {"tdb": "t.tdb", "phase": "CE2TM17", "online": False})
    assert not err and text.startswith("1 compound phase") and "Ce2Fe17: 2:17R" in text
    text, _, err = tb.call("tool_help", {"names": ["identify_structure"]})
    assert not err and "## identify_structure" in text and "never TDB content" in text
    compact = {t.name: t for t in tb.model_specs("compact")}
    assert "identify_structure" in compact and len(compact["identify_structure"].description) < 200


@pytest.mark.skipif(os.environ.get("CALPHAD_LIVE") != "1", reason="live AFLOW / OQMD test: set CALPHAD_LIVE=1")
def test_live(monkeypatch):
    monkeypatch.setattr(sid, "_http_get_json", _real_http)
    r, notes = sid.lookup_online("CeCo5", use_cache=False)
    assert r is not None and r.suffix == "1:5", (r and r.line(), notes)
    r, notes = sid.lookup_online("ZrNi5", use_cache=False)
    assert r is not None and r.suffix == "C15_{b}" and r.prototype == "AuBe5", (r and r.line(), notes)

