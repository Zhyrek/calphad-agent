"""Crystal-structure family of a compound, for phase-diagram labels written "Formula (family)".

"Fe₂Nb (C14)", "CeCo₅ (1:5)", "Ce₂Fe₁₇ (2:17R)", "Fe₇Mo₆ (μ)", "Nb₃Sn (A15)", "Ni₃Al (L1₂)", "ZrNi₅ (C15b)": the
suffix must come from structural evidence, never from the ratio alone - ZrNi5 / HfNi5 / UNi5 are AuBe5-type (C15b),
not CaCu5; WAl12 is WAl12-type (no designation), not ThMn12; MoSi2 (C11b) and CuAl2 (C16) are not Laves.

Suffix priority for an identified prototype: a Greek letter (σ, μ, χ, R), the CaCu5 family (1:5, 1:12, 2:17R/H),
Laves (C14 / C15 / C36), else its Strukturbericht designation in subscript markup ("L1_{2}", "D0_{19}", "C11_{b}",
"A15"; ``StructureID.suffix_plain`` gives "L12"), else none.  Elemental designations (A1-A12) are never used, nor
anything for a pure element.  Use it for COMPOUND phases only: solution phases, even ordered ones named B2_BCC or
L12_FCC, keep their short region names (the caller's decision).  An unknown prototype gives no suffix.

API
---
* :func:`identify` (formula, phase_name=..., allow_online=...) -> :class:`StructureID` (suffix, prototype, space
  group, AFLOW label, source, confidence, evidence, alternatives).  Identification is per FORMULA (one end-member).
* :func:`phase_family` (phase_name, end-member formulas) -> :class:`PhaseFamily`: the per-end-member answers, their
  consensus, the family-level label ("2:17", "Laves"), the phase name's own evidence (authoritative for the whole
  phase) and a warning when one CALPHAD phase covers several structures (Ce2(Co,Fe)17: 2:17H and 2:17R).
* :func:`assign_217` (phases) -> {phase: "2:17R" | "2:17H"} for two 2:17 phases of one diagram.
* :func:`compound_phases` (tdb) -> the stoichiometric / near-stoichiometric phases of a TDB and their end-members.

Sources, first hit wins
-----------------------
1. ``override``: ``[labels] phase_labels`` of the config (keyed by phase name or formula; "1:5" or a full label
   "CeCo₅ (1:5)"; "" = no suffix);
2. ``phase_name``: structural tokens of the TDB phase name (C14, LAVES, D2D, TH2ZN17, SIGMA, MU, CHI_A12 ...);
3. ``table``: the curated offline table :data:`TABLE` (well-established binaries, Pearson / Strukturbericht data);
4. ``cache``: earlier online answers, ``$CALPHAD_STATE_DIR/structure_cache.json`` (default ~/.calphad_agent);
5. ``aflow`` / ``oqmd``: the keyless AFLOW AFLUX and OQMD REST APIs (only the chemical formula is sent - never any
   database content); switched off by ``[structure] online = false`` or ``CALPHAD_STRUCTURE_ONLINE=0``;
6. ``chemistry_rule``: rare earth / Th + Mn, Fe, Co, Ni, Cu (Zn for 2:17) at 1:5, 1:12, 2:17 -> CaCu5 family, low
   confidence (never Zr, Hf, U, Al; never Laves or a Greek letter from a ratio);
7. ``none``: no suffix.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ----------------------------------------------------------------------------------------------- elements
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr "
            "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
            "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
_EL_UP = {e.upper(): e for e in ELEMENTS}
RARE_EARTHS = frozenset("Sc Y La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
CACU5_TM = frozenset("Mn Fe Co Ni Cu".split())           # + Zn for 2:17
NEVER_CACU5 = frozenset("Zr Hf U Al".split())
D_BLOCK = frozenset("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg".split())
_SUBDIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

SG_SYMBOLS = {12: "C2/m", 51: "Pmma", 59: "Pmmn", 62: "Pnma", 63: "Cmcm", 65: "Cmmm", 70: "Fddd", 87: "I4/m",
              123: "P4/mmm", 129: "P4/nmm", 136: "P4_2/mnm", 139: "I4/mmm", 140: "I4/mcm", 148: "R-3", 164: "P-3m1",
              166: "R-3m", 180: "P6_222", 181: "P6_422", 186: "P6_3mc", 189: "P-62m", 191: "P6/mmm", 193: "P6_3/mcm",
              194: "P6_3/mmc", 198: "P2_13", 204: "Im-3", 205: "Pa-3", 213: "P4_132", 215: "P-43m", 216: "F-43m",
              217: "I-43m", 221: "Pm-3m", 223: "Pm-3n", 225: "Fm-3m", 227: "Fd-3m"}
_SG_BY_NORM = {re.sub(r"[_\s]", "", s).lower(): n for n, s in SG_SYMBOLS.items()}


def sg_symbol(n: Optional[int]) -> Optional[str]:
    return None if n is None else SG_SYMBOLS.get(int(n), f"SG {int(n)}")


def sg_number(symbol: Any) -> Optional[int]:
    """'P6/mmm' / 'P6_3/mmc' / 'P63/mmc' / 194 -> number (only the space groups of :data:`STRUCTURES`)."""
    if symbol is None:
        return None
    if isinstance(symbol, int) or str(symbol).strip().isdigit():
        return int(symbol)
    return _SG_BY_NORM.get(re.sub(r"[_\s]", "", str(symbol)).lower())


def plain(suffix: Optional[str]) -> Optional[str]:
    """Strukturbericht markup without the subscript braces: 'L1_{2}' -> 'L12', 'D0_{19}' -> 'D019'; others as is."""
    return None if suffix is None else re.sub(r"_\{([^{}]*)\}", r"\1", suffix)


NONMETALS = frozenset("B C N O".split())


# ----------------------------------------------------------------------------------------------- structures
@dataclass(frozen=True)
class Structure:
    """A recognised structure type.  ``suffix``: a family name (C14, 1:5, σ ...); without one the label suffix is
    the Strukturbericht designation (``strukturbericht``, markup 'L1_{2}': the part after '_' is a subscript), and
    a type with neither (WAl12, PuNi3) gets no suffix."""
    suffix: Optional[str]
    prototype: str
    strukturbericht: str              # markup: "D0_{19}", "C11_{b}", "A15"; "" = none
    sg: int
    pearson: str                      # Pearson symbol as AFLOW writes it (hR = rhombohedral cell)
    aflow_label: str = ""             # a representative AFLOW prototype label (verified against AFLOW where given)
    ratio: Optional[Tuple[int, ...]] = None   # sorted reduced stoichiometry; None = any (sigma, mu, chi, R)
    tcp: bool = False                 # Greek-letter TCP phase: the letter only for transition-metal compounds
    aliases: Tuple[str, ...] = ()     # prototype names as OQMD may write them
    nonmetal: Optional[bool] = None   # True: only with B/C/N/O (Cr23C6), False: never (Th6Mn23); None: either
    match: bool = True                # False: offline table only (shares sg / cell / ratio with another type)

    @property
    def space_group(self) -> str:
        return sg_symbol(self.sg) or ""

    @property
    def prim(self) -> int:
        n = int(re.sub(r"\D", "", self.pearson))
        return n if self.pearson[1] in "PR" else n // {"F": 4, "I": 2}.get(self.pearson[1], 2)

    @property
    def conv(self) -> int:
        n = int(re.sub(r"\D", "", self.pearson))
        return 3 * n if self.pearson[1] == "R" else n


_ST = Structure
STRUCTURES: Tuple[Structure, ...] = (
    # named families (beat the Strukturbericht symbol)
    _ST("C14", "MgZn2", "C14", 194, "hP12", "A2B_hP12_194_ah_f", (1, 2), aliases=("MgZn2",)),
    _ST("C15", "MgCu2", "C15", 227, "cF24", "A2B_cF24_227_d_a", (1, 2), aliases=("MgCu2", "Cu2Mg")),
    _ST("C36", "MgNi2", "C36", 194, "hP24", "A2B_hP24_194_fgh_ef", (1, 2), aliases=("MgNi2",)),
    _ST("1:5", "CaCu5", "D2_{d}", 191, "hP6", "AB5_hP6_191_a_cg", (1, 5), aliases=("CaCu5",)),
    _ST("1:12", "ThMn12", "D2_{b}", 139, "tI26", "A12B_tI26_139_fij_a", (1, 12), aliases=("ThMn12",)),
    _ST("2:17R", "Th2Zn17", "", 166, "hR19", "A2B17_hR19_166_c_cdfh", (2, 17), aliases=("Th2Zn17",)),
    _ST("2:17H", "Th2Ni17", "", 194, "hP38", "A17B2_hP38_194_fgjk_bc", (2, 17), aliases=("Th2Ni17",)),
    _ST("σ", "σ-CrFe", "D8_{b}", 136, "tP30", "", None, True, ("CrFe", "sigma")),
    _ST("μ", "W6Fe7", "D8_{5}", 166, "hR13", "A7B6_hR13_166_ah_3c", None, True, ("W6Fe7", "Fe7W6")),
    _ST("χ", "α-Mn", "A12", 217, "cI58", "A_cI58_217_ac2g", None, True, ("alpha-Mn", "Mn")),
    _ST("R", "R-(Co,Cr,Mo)", "", 148, "hR53", "", None, True, ()),
    # Strukturbericht types: the designation is the suffix
    _ST(None, "Cr3Si", "A15", 223, "cP8", "A3B_cP8_223_c_a", (1, 3), aliases=("Cr3Si", "Nb3Sn")),
    _ST(None, "β-Mn", "A13", 213, "cP20", "", None, aliases=("beta-Mn",)),
    _ST(None, "AuBe5", "C15_{b}", 216, "cF24", "AB5_cF24_216_a_ce", (1, 5), aliases=("AuBe5",)),
    _ST(None, "MoSi2", "C11_{b}", 139, "tI6", "AB2_tI6_139_a_e", (1, 2), aliases=("MoSi2",)),
    _ST(None, "CuAl2", "C16", 140, "tI12", "A2B_tI12_140_h_a", (1, 2), aliases=("CuAl2", "Al2Cu")),
    _ST(None, "CrSi2", "C40", 180, "hP9", "AB2_hP9_180_c_i", (1, 2), aliases=("CrSi2",)),
    _ST(None, "AlB2", "C32", 191, "hP3", "A2B_hP3_191_d_a", (1, 2), aliases=("AlB2",)),
    _ST(None, "CaF2", "C1", 225, "cF12", "AB2_cF12_225_a_c", (1, 2), aliases=("CaF2",)),
    _ST(None, "FeS2", "C2", 205, "cP12", "A2B_cP12_205_c_a", (1, 2), aliases=("FeS2", "pyrite")),
    _ST(None, "Fe2P", "C22", 189, "hP9", "", (1, 2), aliases=("Fe2P",)),
    _ST(None, "Cu2Sb", "C38", 129, "tP6", "", (1, 2), aliases=("Cu2Sb",)),
    _ST(None, "ZrSi2", "C49", 63, "oC12", "", (1, 2), aliases=("ZrSi2",)),
    _ST(None, "TiSi2", "C54", 70, "oF24", "", (1, 2), aliases=("TiSi2",)),
    _ST(None, "MgAgAs", "C1_{b}", 216, "cF12", "", (1, 1, 1), aliases=("MgAgAs", "half-Heusler")),
    _ST(None, "Ti2Ni", "E9_{3}", 227, "cF96", "AB2_cF96_227_e_cf", (1, 2), aliases=("Ti2Ni", "NiTi2")),
    _ST(None, "NaCl", "B1", 225, "cF8", "AB_cF8_225_a_b", (1, 1), aliases=("NaCl", "rocksalt")),
    _ST(None, "CsCl", "B2", 221, "cP2", "AB_cP2_221_a_b", (1, 1), aliases=("CsCl",)),
    _ST(None, "ZnS", "B3", 216, "cF8", "AB_cF8_216_a_c", (1, 1), aliases=("ZnS", "zincblende", "sphalerite")),
    _ST(None, "ZnO", "B4", 186, "hP4", "AB_hP4_186_b_b", (1, 1), aliases=("wurtzite",)),
    _ST(None, "NiAs", "B8_{1}", 194, "hP4", "AB_hP4_194_a_c", (1, 1), aliases=("NiAs",)),
    _ST(None, "Ni2In", "B8_{2}", 194, "hP6", "", (1, 2), aliases=("Ni2In",)),
    _ST(None, "γ-CuTi", "B11", 129, "tP4", "", (1, 1), aliases=("CuTi",)),
    _ST(None, "AuCd", "B19", 51, "oP4", "AB_oP4_51_e_f", (1, 1), aliases=("AuCd",)),
    _ST(None, "FeSi", "B20", 198, "cP8", "AB_cP8_198_a_a", (1, 1), aliases=("FeSi",)),
    _ST(None, "CrB", "B33", 63, "oC8", "AB_oC8_63_c_c", (1, 1), aliases=("CrB", "TlI")),
    _ST(None, "Cu3Au", "L1_{2}", 221, "cP4", "AB3_cP4_221_a_c", (1, 3), aliases=("Cu3Au", "AuCu3")),
    _ST(None, "CuAu", "L1_{0}", 123, "tP2", "AB_tP2_123_a_d", (1, 1), aliases=("CuAu", "AuCu")),
    _ST(None, "Cu2MnAl", "L2_{1}", 225, "cF16", "", (1, 1, 2), aliases=("Cu2MnAl", "Heusler")),
    _ST(None, "BiF3", "D0_{3}", 225, "cF16", "AB3_cF16_225_a_bc", (1, 3), aliases=("BiF3", "Fe3Al", "AlFe3")),
    _ST(None, "Fe3C", "D0_{11}", 62, "oP16", "AB3_oP16_62_c_cd", (1, 3), aliases=("Fe3C", "cementite")),
    _ST(None, "NiAl3", "D0_{20}", 62, "oP16", "", (1, 3), match=False),
    _ST(None, "Ni3Sn", "D0_{19}", 194, "hP8", "A3B_hP8_194_h_c", (1, 3), aliases=("Ni3Sn", "Ti3Al", "Mg3Cd")),
    _ST(None, "TiAl3", "D0_{22}", 139, "tI8", "A3B_tI8_139_bd_a", (1, 3), aliases=("TiAl3", "Al3Ti")),
    _ST(None, "ZrAl3", "D0_{23}", 139, "tI16", "", (1, 3), aliases=("ZrAl3", "Al3Zr")),
    _ST(None, "Ni3Ti", "D0_{24}", 194, "hP16", "", (1, 3), aliases=("Ni3Ti", "TiNi3")),
    _ST(None, "β-Cu3Ti", "D0_{a}", 59, "oP8", "", (1, 3), aliases=("Cu3Ti", "Ni3Nb", "NbNi3")),
    _ST(None, "BaAl4", "D1_{3}", 139, "tI10", "AB4_tI10_139_a_de", (1, 4), aliases=("BaAl4", "ThZn4")),
    _ST(None, "MoNi4", "D1_{a}", 87, "tI10", "", (1, 4), aliases=("MoNi4", "Ni4Mo")),
    _ST(None, "Ni2Al3", "D5_{13}", 164, "hP5", "A3B2_hP5_164_ad_d", (2, 3), aliases=("Ni2Al3", "Al3Ni2")),
    _ST(None, "Th6Mn23", "D8_{a}", 225, "cF116", "A23B6_cF116_225_ad2f_e", (6, 23), aliases=("Th6Mn23",),
        nonmetal=False),
    _ST(None, "Cr23C6", "D8_{4}", 225, "cF116", "", (6, 23), aliases=("Cr23C6",), nonmetal=True),
    _ST(None, "Cu5Zn8", "D8_{2}", 217, "cI52", "", (5, 8), aliases=("Cu5Zn8", "gamma-brass")),
    _ST(None, "Cr5Al8", "D8_{10}", 160, "hR26", "A8B5_hR26_160_a3bc_a3b", (5, 8), aliases=("Cr5Al8", "Al8Cr5")),
    _ST(None, "Cu9Al4", "D8_{3}", 215, "cP52", "A4B9_cP52_215_ei_3efgi", (4, 9), aliases=("Cu9Al4", "Al4Cu9")),
    _ST(None, "Mn5Si3", "D8_{8}", 193, "hP16", "A5B3_hP16_193_dg_g", (3, 5), aliases=("Mn5Si3",)),
    _ST(None, "W5Si3", "D8_{m}", 140, "tI32", "A5B3_tI32_140_bk_ah", (3, 5), aliases=("W5Si3",), nonmetal=False),
    _ST(None, "Cr5B3", "D8_{l}", 140, "tI32", "", (3, 5), aliases=("Cr5B3",), nonmetal=True),
    # recognised, no Strukturbericht designation: no suffix
    _ST(None, "WAl12", "", 204, "cI26", "A12B_cI26_204_g_a", (1, 12), aliases=("WAl12", "Al12W")),
    _ST(None, "PuNi3", "", 166, "hR12", "A3B_hR12_166_bch_ac", (1, 3), aliases=("PuNi3",)),
    _ST(None, "Ce2Ni7", "", 194, "hP36", "A7B2_hP36_194_aefhk_2f", (2, 7), aliases=("Ce2Ni7",)),
    _ST(None, "Gd2Co7", "", 166, "hR18", "A7B2_hR18_166_a2cdh_2c", (2, 7), aliases=("Gd2Co7",)),
)
BY_PROTO = {s.prototype: s for s in STRUCTURES}
BY_SUFFIX = {s.suffix: s for s in STRUCTURES if s.suffix}
BY_SB = {plain(s.strukturbericht).upper(): s for s in STRUCTURES if s.strukturbericht and not s.suffix}
# family level of a suffix: the label to fall back to when the variant is unknown
GROUP = {"C14": "Laves", "C15": "Laves", "C36": "Laves", "Laves": "Laves",
         "2:17R": "2:17", "2:17H": "2:17", "2:17": "2:17"}


def elemental_designation(sb: Optional[str]) -> bool:
    """A1-A12 (and other A-types of pure elements) are never compound suffixes; A13 and A15 are."""
    p = plain(sb) or ""
    return bool(re.fullmatch(r"A\d+", p)) and p not in ("A13", "A15")


def group_of(suffix: Optional[str]) -> Optional[str]:
    return GROUP.get(suffix, suffix) if suffix else None


# ----------------------------------------------------------------------------------------------- formulas
def _split_upper(s: str) -> Optional[List[Tuple[str, str]]]:
    """'CECO5' -> [('Ce',''), ('Co','5')]: element symbols of an all-capitals formula, two-letter symbols first,
    with backtracking."""
    if not s:
        return []
    for n in (2, 1):
        el = _EL_UP.get(s[:n])
        if el is None:
            continue
        m = re.match(r"\d*\.?\d*", s[n:])
        num = m.group(0) if m else ""
        rest = _split_upper(s[n + len(num):])
        if rest is not None:
            return [(el, num)] + rest
    return None


def parse_formula(formula: Union[str, Dict[str, float]]) -> Dict[str, float]:
    """'CeCo5', 'CECO5', 'Ce₂Co₁₇', 'Cr0.47Fe0.53' or {'CE': 1, 'CO': 5} -> {'Ce': 1.0, 'Co': 5.0} (input order)."""
    out: Dict[str, float] = {}
    if isinstance(formula, dict):
        for k, v in formula.items():
            el = _EL_UP.get(str(k).strip().upper())
            if el is None:
                raise ValueError(f"unknown element {k!r} in {formula!r}")
            if float(v) > 0:
                out[el] = out.get(el, 0.0) + float(v)
        if not out:
            raise ValueError(f"empty formula {formula!r}")
        return out
    s = str(formula).translate(_SUBDIGITS).replace(" ", "")
    if not s:
        raise ValueError("empty formula")
    if s == s.upper():
        parts = _split_upper(s)
    else:
        parts = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", s)
        if "".join(a + b for a, b in parts) != s or any(a not in _EL_UP.values() for a, _ in parts):
            parts = None
    if not parts:
        raise ValueError(f"cannot read the formula {formula!r}")
    for el, num in parts:
        out[el] = out.get(el, 0.0) + (float(num) if num else 1.0)
    return out


def reduced(counts: Dict[str, float], tol: float = 0.004) -> Optional[Dict[str, int]]:
    """Smallest integer counts with the same ratio ({'Ce': 2, 'Co': 4} -> {'Ce': 1, 'Co': 2}); None when the
    ratio is not close to small integers (a sigma composition like Cr0.47Fe0.53 is kept as Cr47Fe53)."""
    tot = float(sum(counts.values()))
    if tot <= 0:
        return None
    fr = {e: Fraction(v / tot).limit_denominator(120) for e, v in counts.items()}
    if any(abs(float(f) - v / tot) > tol for (e, f), v in zip(fr.items(), counts.values())):
        return None
    den = 1
    for f in fr.values():
        den = den * f.denominator // gcd(den, f.denominator)
    ints = {e: int(round(float(f * den))) for e, f in fr.items()}
    g = 0
    for i in ints.values():
        g = gcd(g, i)
    return {e: i // max(g, 1) for e, i in ints.items()}


def fractions(counts: Dict[str, float]) -> Dict[str, float]:
    tot = float(sum(counts.values()))
    return {e: v / tot for e, v in counts.items()}


def formula_key(counts: Dict[str, float]) -> Optional[str]:
    """Canonical key: reduced integers, elements alphabetical ('CeCo5', 'Co17Sm2', 'CrFe'); None if not integral."""
    ints = reduced(counts)
    if not ints:
        return None
    return "".join(e + ("" if ints[e] == 1 else str(ints[e])) for e in sorted(ints))


def _num(v: float) -> str:
    return "" if abs(v - 1) < 1e-9 else (str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}")


def display_formula(formula: Union[str, Dict[str, float]], counts: Optional[Dict[str, float]] = None) -> str:
    """The formula as given, capitalised ('CECO5' -> 'CeCo5'); dicts in their own order."""
    if isinstance(formula, str) and formula != formula.upper():
        return formula.translate(_SUBDIGITS).replace(" ", "")
    counts = counts or parse_formula(formula)
    return "".join(e + _num(v) for e, v in counts.items())


def _ratio(counts: Dict[str, float]) -> Optional[Tuple[int, ...]]:
    ints = reduced(counts)
    return tuple(sorted(ints.values())) if ints else None


# ----------------------------------------------------------------------------------------------- results
@dataclass
class StructureID:
    """Answer for one formula.  ``suffix`` is the label suffix: a family name ("1:5", "C14", "2:17R", "σ"; "2:17" /
    "Laves" when only the family is known), else the prototype's Strukturbericht designation in subscript markup
    ("L1_{2}", "D0_{19}", "C15_{b}", "A15"; ``suffix_plain`` is "L12", "D019", "C15b"), else None (no designation:
    WAl12, PuNi3; never A1-A12, so no FCC / BCC / HCP relabelling).  Meant for compound phases only: callers should
    keep the short names of solution phases, even ordered ones (B2_BCC, L12_FCC)."""
    formula: str
    suffix: Optional[str] = None
    prototype: Optional[str] = None
    space_group: Optional[str] = None
    aflow_label: Optional[str] = None
    source: str = "none"              # override | phase_name | table | cache | aflow | oqmd | chemistry_rule | none
    confidence: str = "low"           # high | medium | low
    evidence: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    strukturbericht: Optional[str] = None
    label: Optional[str] = None       # a complete label from an override ("CeCo₅ (1:5)")
    notes: List[str] = field(default_factory=list)       # e.g. an online source that could not be reached
    temperature_K: Optional[float] = None

    @property
    def family(self) -> Optional[str]:
        return self.suffix

    @property
    def suffix_plain(self) -> Optional[str]:
        """The suffix without subscript markup ("L1_{2}" -> "L12"), for reports and lookups."""
        return plain(self.suffix)

    @property
    def group(self) -> Optional[str]:
        """Family level: "2:17" for 2:17R / 2:17H, "Laves" for C14 / C15 / C36, else the suffix."""
        return group_of(self.suffix)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["family"], d["suffix_plain"] = self.family, self.suffix_plain
        return d

    def line(self) -> str:
        """One compact line (the identify_structure tool's output)."""
        head = f"{self.formula}: " + (self.suffix if self.suffix else "no suffix")
        st = ", ".join(x for x in (
            (f"{self.prototype}-type" + (f" ({self.strukturbericht})" if self.strukturbericht else ""))
            if self.prototype else "", self.space_group or "", self.aflow_label or "") if x)
        parts = [head] + ([st] if st else []) + [f"{self.source}, {self.confidence}"]
        if self.label:
            parts.append(f"label {self.label!r}")
        if self.evidence:
            parts.append(self.evidence)
        if self.alternatives:
            parts.append("alt: " + "; ".join(_alt_text(a) for a in self.alternatives))
        if self.notes:
            parts.append("note: " + "; ".join(self.notes))
        return " | ".join(parts)


def _alt_text(a: Dict[str, Any]) -> str:
    s = (a.get("suffix") or "no suffix") + (f" {a['prototype']}-type" if a.get("prototype") else "")
    return s + (f" ({a['note']})" if a.get("note") else "")


def _alt(st: Optional[Structure], note: str = "", kind: str = "polymorph", prototype: Optional[str] = None,
         elements: Iterable[str] = ()) -> Dict[str, Any]:
    sfx = _suffix_for(st, elements) if st else None
    return {"suffix": sfx, "suffix_plain": plain(sfx), "prototype": st.prototype if st else prototype,
            "space_group": st.space_group if st else None, "aflow_label": (st.aflow_label or None) if st else None,
            "kind": kind, "note": note}


def _suffix_for(st: Structure, elements: Iterable[str]) -> Optional[str]:
    """Suffix of a structure for a compound of these elements: the family name (Greek letter, 1:5 ..., Laves),
    else the Strukturbericht designation (never an elemental A1-A12 one, never for a pure element).  A TCP letter
    (σ, μ, χ, R) only for transition-metal compounds (Al12Mg17 is α-Mn-type but called γ): others get the
    designation if it is not elemental (D8_{b} for a σ-type Al compound), else nothing."""
    els = set(elements)
    if len(els) == 1:
        return None
    sb = st.strukturbericht if st.strukturbericht and not elemental_designation(st.strukturbericht) else None
    if st.suffix:
        if st.tcp and els and not els <= D_BLOCK:
            return sb
        return st.suffix
    return sb


def _make(st: Optional[Structure], formula: str, counts: Dict[str, float], source: str, confidence: str,
          evidence: str, **kw) -> StructureID:
    r = StructureID(formula=formula, source=source, confidence=confidence, evidence=evidence)
    if st is not None:
        r.suffix = _suffix_for(st, counts)
        r.prototype, r.space_group = st.prototype, st.space_group
        r.aflow_label, r.strukturbericht = st.aflow_label or None, st.strukturbericht or None
        if st.tcp and r.suffix != st.suffix:
            r.evidence += f"; {st.prototype}-type, but {st.suffix} is kept for transition-metal phases"
    for k, v in kw.items():
        if v is not None:
            setattr(r, k, v)
    return r


AMBIGUOUS_217 = "2:17"


def _make_217(formula: str, source: str, confidence: str, evidence: str, notes=("", "")) -> StructureID:
    return StructureID(formula=formula, suffix=AMBIGUOUS_217, prototype="Th2Zn17 / Th2Ni17", source=source,
                       confidence=confidence, evidence=evidence,
                       alternatives=[_alt(BY_PROTO["Th2Zn17"], notes[0]), _alt(BY_PROTO["Th2Ni17"], notes[1])])


# ----------------------------------------------------------------------------------------------- offline table
@dataclass(frozen=True)
class TableEntry:
    formula: str
    prototypes: Tuple[Any, ...]       # first = primary; others (prototype, kind, note): kind polymorph | model
    confidence: str = "high"
    note: str = ""
    either_217: bool = False          # both 2:17 polymorphs, neither preferred -> family "2:17"


_PS = "Pearson / Strukturbericht"
_TABLE_ROWS: Tuple[TableEntry, ...] = (
    # CaCu5 (D2d) -> 1:5
    *(TableEntry(f, ("CaCu5",), "high", f"{_PS}: CaCu5-type (D2d)") for f in (
        "LaNi5", "LaCo5", "CeCo5", "CeNi5", "PrCo5", "PrNi5", "NdCo5", "NdNi5", "SmCo5", "SmNi5", "GdCo5", "YCo5",
        "YNi5", "ThNi5", "ThCo5", "CaCu5")),
    # AuBe5 (C15b): a 1:5 ratio that is NOT CaCu5
    *(TableEntry(f, ("AuBe5",), "high", f"{_PS}: AuBe5-type (C15b), not CaCu5") for f in (
        "ZrNi5", "HfNi5", "UNi5", "AuBe5")),
    # WAl12-type (Im-3): a 1:12 ratio that is NOT ThMn12
    *(TableEntry(f, ("WAl12",), "high", f"{_PS}: WAl12-type (Im-3), not ThMn12") for f in ("WAl12", "MoAl12")),
    # ThMn12 (D2b) -> 1:12
    *(TableEntry(f, ("ThMn12",), "high", f"{_PS}: ThMn12-type (D2b)") for f in (
        "ThMn12", "YMn12", "SmFe11Ti", "NdFe11Ti")),
    TableEntry("SmFe12", ("ThMn12",), "medium", "ThMn12-type; metastable as a binary (thin films)"),
    # 2:17
    TableEntry("Ce2Fe17", ("Th2Zn17", ("Th2Ni17", "model", "some Ce-Fe-X CALPHAD models use a 2:17H formulation")),
               "high", "user, 2026-10-12: Ce2Fe17 Th2Zn17-type (R); AFLOW: 4 ICSD entries Th2Zn17-type"),
    TableEntry("Ce2Co17", ("Th2Ni17",), "high", "user, 2026-10-12: Ce2Co17 Th2Ni17-type (H)"),
    *(TableEntry(f, ("Th2Zn17",), "high", f"{_PS}: Th2Zn17-type (R)") for f in (
        "Nd2Fe17", "Sm2Fe17", "Pr2Fe17", "Th2Zn17")),
    *(TableEntry(f, ("Th2Ni17",), "high", f"{_PS}: Th2Ni17-type (H)") for f in ("Th2Ni17", "Lu2Fe17")),
    TableEntry("Sm2Co17", ("Th2Zn17", ("Th2Ni17", "polymorph", "high-temperature form")), "medium",
               "Th2Zn17-type (R) at low T, Th2Ni17-type (H) at high T", either_217=True),
    TableEntry("Y2Co17", ("Th2Zn17", ("Th2Ni17", "polymorph", "")), "medium",
               "both polymorphs reported (AFLOW: 3 ICSD entries each)", either_217=True),
    TableEntry("Y2Fe17", ("Th2Ni17", ("Th2Zn17", "polymorph", "")), "medium",
               "mostly Th2Ni17-type (H); Th2Zn17-type also reported", either_217=True),
    # Laves
    *(TableEntry(f, ("MgZn2",), "high", f"{_PS}: MgZn2-type (C14)") for f in (
        "Fe2Nb", "Fe2Ti", "Fe2Ta", "MgZn2")),
    *(TableEntry(f, ("MgZn2",), "high", f"{_PS}: MgZn2-type (C14), the λ phase") for f in ("Fe2Mo", "Fe2W")),
    *(TableEntry(f, ("MgCu2",), "high", f"{_PS}: MgCu2-type (C15)") for f in (
        "Fe2Zr", "Co2Zr", "CeFe2", "CeCo2", "MgCu2", "Co2Y", "Fe2Y", "ZrV2", "ZrMo2", "ZrW2", "HfV2", "CaAl2")),
    TableEntry("MgNi2", ("MgNi2",), "high", f"{_PS}: MgNi2-type (C36)"),
    TableEntry("Cr2Nb", ("MgCu2", ("MgZn2", "polymorph", "high-temperature form")), "medium",
               "C15 at low T, C14 at high T"),
    TableEntry("Fe2Hf", ("MgCu2", ("MgNi2", "polymorph", "high T"), ("MgZn2", "polymorph", "high T")), "medium",
               "C15; C36 / C14 at high temperature"),
    TableEntry("Cr2Ti", ("MgCu2", ("MgNi2", "polymorph", "high T"), ("MgZn2", "polymorph", "high T")), "medium",
               "C15 at low T; C36 / C14 at high T"),
    TableEntry("Cr2Zr", ("MgCu2", ("MgNi2", "polymorph", "high T"), ("MgZn2", "polymorph", "high T")), "medium",
               "C15 at low T; C36 / C14 at high T"),
    TableEntry("Cr2Hf", ("MgCu2", ("MgZn2", "polymorph", "high T")), "medium", "C15 at low T; C14 at high T"),
    TableEntry("Co2Nb", ("MgCu2", ("MgNi2", "polymorph", "Co-rich"), ("MgZn2", "polymorph", "Nb-rich")), "medium",
               "C15 near Co2Nb; C36 and C14 at other compositions"),
    # mu (D85)
    *(TableEntry(f, ("W6Fe7",), "high", f"{_PS}: W6Fe7-type (D85), μ") for f in (
        "Fe7Mo6", "Fe7W6", "Co7Mo6", "Co7W6", "Fe7Nb6", "Co7Nb6")),
    # chi (A12, alpha-Mn type)
    TableEntry("Fe36Cr12Mo10", ("α-Mn",), "high", f"{_PS}: α-Mn-type (A12), χ"),
    # A15
    *(TableEntry(f, ("Cr3Si",), "high", f"{_PS}: Cr3Si-type (A15)") for f in (
        "Cr3Si", "Mo3Si", "Nb3Sn", "Nb3Al", "Nb3Ge", "V3Si", "V3Ga")),
    # recognised, no suffix
    *(TableEntry(f, ("MoSi2",), "high", f"{_PS}: MoSi2-type (C11b), not Laves") for f in ("MoSi2", "WSi2")),
    TableEntry("CuAl2", ("CuAl2",), "high", f"{_PS}: CuAl2-type (C16, θ), not Laves"),
    TableEntry("CrSi2", ("CrSi2",), "high", f"{_PS}: CrSi2-type (C40), not Laves"),
)


def _sb_rows(proto: str, formulas: str, confidence: str = "high", note: str = "") -> Tuple[TableEntry, ...]:
    st = BY_PROTO[proto]
    text = f"{_PS}: {proto}-type ({plain(st.strukturbericht)})" + (f", {note}" if note else "")
    return tuple(TableEntry(f, (proto,), confidence, text) for f in formulas.split())


# common compounds by Strukturbericht type (the designation is their suffix)
_TABLE_ROWS += (
    *_sb_rows("Cu3Au", "Ni3Al FeNi3 Al3Sc Cu3Au"),
    *_sb_rows("CsCl", "NiAl FeAl CoAl CuZn FeTi CoTi"),
    *_sb_rows("BiF3", "Fe3Al Fe3Si"),
    *_sb_rows("CuAu", "TiAl CuAu FePt"),
    *_sb_rows("Ni3Sn", "Ti3Al Ni3Sn Co3W Co3Mo"),
    *_sb_rows("TiAl3", "Al3Ti Al3V Ni3V"),
    *_sb_rows("ZrAl3", "Al3Zr"),
    *_sb_rows("Ni3Ti", "Ni3Ti"),
    *_sb_rows("β-Cu3Ti", "Ni3Nb Ni3Mo"),
    *_sb_rows("NiAl3", "NiAl3"),
    *_sb_rows("Ti2Ni", "Ti2Ni"),
    *_sb_rows("Fe3C", "Fe3C", note="cementite"),
    *_sb_rows("Cr23C6", "Cr23C6"),
    *_sb_rows("Mn5Si3", "Mn5Si3 Ti5Si3"),
    *_sb_rows("W5Si3", "W5Si3 Mo5Si3"),
    *_sb_rows("NiAs", "NiAs"),
    *_sb_rows("AlB2", "AlB2 TiB2 ZrB2 HfB2"),
    *_sb_rows("CuAl2", "Fe2B Co2B"),
    *_sb_rows("CuAl2", "NiZr2", "medium"),
    *_sb_rows("CrSi2", "NbSi2 TaSi2"),
    *_sb_rows("ZrSi2", "ZrSi2"),
    *_sb_rows("TiSi2", "TiSi2", note="the stable C54 form"),
    *_sb_rows("Ni2Al3", "Ni2Al3"),
    *_sb_rows("MoNi4", "MoNi4"),
    *_sb_rows("BaAl4", "BaAl4"),
    *_sb_rows("FeSi", "FeSi CoSi CrSi"),
    *_sb_rows("CrB", "CrB"),
    *_sb_rows("Fe2P", "Fe2P"),
    *_sb_rows("Cu2Sb", "Cu2Sb"),
    *_sb_rows("Cu5Zn8", "Cu5Zn8", note="γ-brass"),
    *_sb_rows("Cr5Al8", "Al8Cr5", note="the rhombohedral (LT) form; HT Al8Cr5 is γ-brass-like"),
    *_sb_rows("Cu9Al4", "Cu9Al4"),
)
TABLE: Dict[str, TableEntry] = {formula_key(parse_formula(e.formula)): e for e in _TABLE_ROWS}

# sigma: a composition range rather than a formula: (elements, element x refers to, x_min, x_max, confidence)
SIGMA_RANGES: Tuple[Tuple[frozenset, str, float, float, str], ...] = (
    (frozenset({"Cr", "Fe"}), "Cr", 0.40, 0.55, "high"),
    (frozenset({"Fe", "V"}), "V", 0.35, 0.60, "high"),
    (frozenset({"Co", "Cr"}), "Cr", 0.52, 0.66, "high"),
    (frozenset({"Cr", "Mn"}), "Mn", 0.65, 0.85, "medium"),
    (frozenset({"Fe", "Mo"}), "Mo", 0.40, 0.55, "medium"),
    (frozenset({"Mo", "Re"}), "Re", 0.50, 0.75, "medium"),
    (frozenset({"Re", "W"}), "Re", 0.55, 0.80, "medium"),
)


def table_lookup(formula: Union[str, Dict[str, float]]) -> Optional[StructureID]:
    """The offline table's answer for a formula (source "table"), or None."""
    counts = parse_formula(formula)
    disp = display_formula(formula, counts)
    e = TABLE.get(formula_key(counts) or "")
    if e is not None:
        if e.either_217:
            r = _make_217(disp, "table", e.confidence, e.note)
            first = e.prototypes[0]
            r.alternatives.sort(key=lambda a: a["prototype"] != first)
            for a in r.alternatives:
                for p in e.prototypes[1:]:
                    if isinstance(p, tuple) and p[0] == a["prototype"] and p[2]:
                        a["note"] = p[2]
            return r
        st = BY_PROTO[e.prototypes[0]]
        alts = [_alt(BY_PROTO[p[0]], p[2], p[1], elements=counts) for p in e.prototypes[1:]]
        return _make(st, disp, counts, "table", e.confidence, e.note, alternatives=alts)
    x = fractions(counts)
    for els, el, lo, hi, conf in SIGMA_RANGES:
        if set(counts) == els and lo - 1e-9 <= x[el] <= hi + 1e-9:
            return _make(BY_PROTO["σ-CrFe"], disp, counts, "table", conf,
                         f"{_PS}: σ (D8b) in {'-'.join(sorted(els))} near x({el}) {lo:.2f}-{hi:.2f}")
    return None


# ----------------------------------------------------------------------------------------------- phase names
def _name_tokens(name: str) -> List[str]:
    n = re.sub(r"#\d+$", "", str(name or "").strip().upper())
    n = re.sub(r":[A-Z]$", "", n)
    return [t for t in re.split(r"[^A-Z0-9]+", n) if t]


_ALNUM_EDGE = r"(?<![A-Z0-9]){}(?![A-Z0-9])"
# (regex on the "_"-joined tokens, structure prototype or a family-level label)
NAME_RULES: Tuple[Tuple[str, str], ...] = (
    (r"C15B|AUBE5|BE5AU", "AuBe5"), (r"C11B|MOSI2|SI2MO", "MoSi2"), (r"C16|CUAL2|AL2CU", "CuAl2"),
    (r"C40|CRSI2", "CrSi2"),
    (r"C14|MGZN2|ZN2MG", "MgZn2"), (r"C15|MGCU2|CU2MG", "MgCu2"), (r"C36|MGNI2|NI2MG", "MgNi2"),
    (r"CACU5|CU5CA|D2_?D|RT5|R1T5|RE1TM5|RETM5|1_5", "CaCu5"),
    (r"THMN12|MN12TH|D2_?B|RT12|R1T12|RE1TM12|RETM12|1_12", "ThMn12"),
    (r"TH2ZN17|ZN17TH2|2_?17_?R", "Th2Zn17"), (r"TH2NI17|NI17TH2|2_?17_?H", "Th2Ni17"),
    (r"SIGMA|D8_?B", "σ-CrFe"), (r"MU|D8_?5|W6FE7|FE7W6", "W6Fe7"), (r"CHI", "α-Mn"),
    (r"R_PHASE|RPHASE", "R-(Co,Cr,Mo)"), (r"A15|CR3SI", "Cr3Si"),
    (r"LAVES", "Laves"),
    (r"2_17|RT17|R2T17|RE2TM17|[A-Z]{1,2}2_?TM?_?17", "2:17"),
)


def _sb_rules() -> Tuple[Tuple[str, str], ...]:
    """Name tokens of the other Strukturbericht types: D019, D0_19, L12, B8_1 ... (not the one- or two-character
    ones that read like other things: B1, B3, B4, C1, C2; not A13, which in a TDB is the β-Mn solution CUB_A13)."""
    out = []
    for st in STRUCTURES:
        sb = st.strukturbericht
        if st.suffix or not sb or elemental_designation(sb) or plain(sb) in ("B1", "B3", "B4", "C1", "C2", "A13"):
            continue
        m = re.fullmatch(r"([A-Z]\d+)(?:_\{(\w+)\})?", sb)
        if m:
            out.append((m.group(1) + (f"_?{m.group(2).upper()}" if m.group(2) else ""), st.prototype))
    return tuple(out)


NAME_RULES = NAME_RULES[:-2] + _sb_rules() + NAME_RULES[-2:]
R217_TOKENS = {"R": "2:17R", "RHOMB": "2:17R", "RHOMBOHEDRAL": "2:17R", "R3M": "2:17R",
               "H": "2:17H", "HEX": "2:17H", "HEXAGONAL": "2:17H", "P63MMC": "2:17H"}


def name_evidence(phase_name: str, counts: Optional[Dict[str, float]] = None
                  ) -> Optional[Tuple[Optional[Structure], str, str]]:
    """Structural evidence in a phase name: (Structure or None, label, token), label = suffix ("C14", "2:17R"),
    "" for a recognised no-suffix type (C15B), or a family-level "Laves" / "2:17".  R / H tokens (CE2FE17_R) count
    in a 2:17 context (a 2:17 token, or a 2:17 formula).  Tokens match whole: MU never matches inside a word."""
    toks = _name_tokens(phase_name)
    if not toks:
        return None
    s = "_".join(toks)
    hit = None
    for rx, target in NAME_RULES:
        m = re.search(_ALNUM_EDGE.format(f"(?:{rx})"), s)
        if m:
            hit = (target, m.group(0))
            break
    is_217 = (hit is not None and hit[0] in ("2:17", "Th2Zn17", "Th2Ni17")) or (
        counts is not None and _ratio(counts) == (2, 17))
    if is_217 and (hit is None or hit[0] == "2:17"):
        for t in toks[1:]:
            if t in R217_TOKENS:
                st = BY_SUFFIX[R217_TOKENS[t]]
                return st, st.suffix, t
    if hit is None:
        return None
    target, tok = hit
    if target in ("Laves", "2:17"):
        return None, target, tok
    st = BY_PROTO[target]
    return st, _suffix_for(st, counts or {}) or "", tok


# ----------------------------------------------------------------------------------------------- settings
@dataclass
class Settings:
    online: bool = True
    timeout_s: float = 15.0
    cache_days: float = 180.0
    sources: Tuple[str, ...] = ("aflow", "oqmd")
    overrides: Dict[str, str] = field(default_factory=dict)


_SETTINGS: Optional[Settings] = None


def configure(cfg=None, **kw) -> Settings:
    """Set the defaults from an agent Config (``[structure]`` and ``[labels] phase_labels``) and/or keywords
    (online, timeout_s, cache_days, sources, overrides).  The agent calls this once per session."""
    global _SETTINGS
    s = Settings()
    sc = getattr(cfg, "structure", None)
    if sc is not None:
        s.online = bool(getattr(sc, "online", True))
        s.timeout_s = float(getattr(sc, "timeout_s", s.timeout_s))
        s.cache_days = float(getattr(sc, "cache_days", s.cache_days))
        s.sources = tuple(str(x).lower() for x in (getattr(sc, "sources", None) or s.sources))
    lc = getattr(cfg, "labels", None)
    if lc is not None:
        s.overrides = dict(getattr(lc, "phase_labels", None) or {})
    for k, v in kw.items():
        if not hasattr(s, k):
            raise TypeError(f"unknown structure setting {k!r}")
        setattr(s, k, tuple(v) if k == "sources" else v)
    _SETTINGS = s
    return s


def settings() -> Settings:
    if _SETTINGS is None:
        try:
            import warnings
            from .agent.config import load_config
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                configure(load_config())
        except Exception:  # noqa: BLE001 - no / broken config: built-in defaults
            configure()
    return _SETTINGS  # type: ignore[return-value]


def online_allowed(allow_online: Optional[bool] = None) -> Tuple[bool, str]:
    """(allowed, reason).  CALPHAD_STRUCTURE_ONLINE (0/1) beats ``[structure] online``; a call can only switch
    lookups off, never on against the config / environment."""
    env = os.environ.get("CALPHAD_STRUCTURE_ONLINE", "").strip().lower()
    if env:
        on, why = env not in ("0", "false", "no", "off"), "CALPHAD_STRUCTURE_ONLINE"
    else:
        on, why = bool(settings().online), "[structure] online"
    if not on:
        return False, f"online lookups off ({why})"
    if allow_online is False:
        return False, "online lookups off for this call"
    return True, ""


# ----------------------------------------------------------------------------------------------- cache
def cache_path() -> Path:
    return Path(os.environ.get("CALPHAD_STATE_DIR") or (Path.home() / ".calphad_agent")) / "structure_cache.json"


def _cache_load() -> Dict[str, Any]:
    try:
        d = json.loads(cache_path().read_text())
        return d.get("entries", {}) if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _cache_save(entries: Dict[str, Any]) -> None:
    p = cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": entries}, indent=1, sort_keys=True))
        os.replace(tmp, p)
    except OSError:
        pass


# ----------------------------------------------------------------------------------------------- online
AFLUX_URL = "https://aflowlib.duke.edu/API/aflux/"
AFLUX_FIELDS = ("aflow_prototype_label_orig,aflow_prototype_label_relax,spacegroup_orig,spacegroup_relax,"
                "Pearson_symbol_orig,Pearson_symbol_relax,enthalpy_formation_atom,catalog,compound,auid")
OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
OQMD_FIELDS = "name,entry_id,icsd_id,spacegroup,prototype,natoms,delta_e,stability"
_DOWN: Dict[str, float] = {}          # source -> time until which it is not tried again (after a failure)
_SYSTEM_MEMO: Dict[Tuple[str, ...], List[Dict]] = {}   # AFLOW answers per element system (this process)


def _http_get_json(url: str, timeout: float, retries: int = 2) -> Any:
    """GET a JSON answer; a 5xx (OQMD answers 502 now and then) is retried after a second, twice."""
    import urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": "calphad-tools structure_id",
                                               "Accept": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - fixed https hosts
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                raise
            time.sleep(1.0)


def aflow_url(elements: Sequence[str]) -> str:
    els = ",".join(sorted(elements))
    return f"{AFLUX_URL}?species({els}),nspecies({len(elements)}),{AFLUX_FIELDS},$paging(0)"


def oqmd_url(formula: str) -> str:
    return f"{OQMD_URL}?" + urllib.parse.urlencode(
        {"composition": formula, "fields": OQMD_FIELDS, "limit": 100}, safe=",")


_LABEL_RX = re.compile(r"^(?P<st>[A-Za-z0-9]+?)_(?P<pearson>[a-z][A-Z]\d+)_(?P<sg>\d+)(?:_|$)")


def label_structure(label: Optional[str], counts: Dict[str, float]) -> Optional[Structure]:
    """Structure of an AFLOW prototype label ('AB5_hP6_191_a_cg' -> CaCu5): space group, cell size and
    stoichiometry (independent of the alphabetical A/B order and the Wyckoff origin choice)."""
    m = _LABEL_RX.match(label or "")
    if not m:
        return None
    nums = [int(n) if n else 1 for n in re.findall(r"[A-Z](\d*)", m.group("st"))]
    ratio = None
    if nums:
        g = 0
        for n in nums:
            g = gcd(g, n)
        ratio = tuple(sorted(n // g for n in nums))
    pearson, sg = m.group("pearson"), int(m.group("sg"))
    return match_structure(sg, Structure(None, "", "", sg, pearson).prim, ratio, counts)


def match_structure(sg: Optional[int], natoms: Optional[int], ratio: Optional[Tuple[int, ...]],
                    elements: Iterable[str] = ()) -> Optional[Structure]:
    """Structure with this space group, atoms per primitive cell (a conventional-cell count only when no
    primitive one fits: Ti2Ni cF96 has 24 atoms per primitive cell, as many as MgCu2's conventional cell) and
    reduced stoichiometry (sorted)."""
    if sg is None:
        return None
    lean = ratio is not None and min(ratio) / sum(ratio) < 0.15     # TCP phases are not that lopsided
    nm = bool(set(elements) & NONMETALS)
    hits = [s for s in STRUCTURES if s.match and s.sg == sg and (s.ratio == ratio or (s.ratio is None and not lean))
            and (s.nonmetal is None or s.nonmetal == nm)]
    if natoms is None:
        return hits[0] if len(hits) == 1 else None
    best = [s for s in hits if natoms == s.prim] or [s for s in hits if natoms == s.conv]
    best.sort(key=lambda s: s.ratio is None)      # a ratio-specific type beats an any-ratio one
    return best[0] if best else None


def _proto_structure(name: Optional[str], sg: Optional[int], ratio, elements: Iterable[str] = ()
                     ) -> Optional[Structure]:
    if not name:
        return None
    pieces = {str(name).strip().lower()} | {p.lower() for p in re.split(r"[#_()\s]+", str(name)) if p}
    for s in STRUCTURES:          # OQMD writes 'MgZn2', 'Ni17Th2', 'C15_MgCu2', 'Laves(cub)#MgCu2'
        for a in s.aliases + (s.prototype,):
            if a.lower() in pieces:
                if (sg is None or s.sg == sg) and (s.ratio is None or s.ratio == ratio) and (
                        s.nonmetal is None or s.nonmetal == bool(set(elements) & NONMETALS)):
                    return s
    return None


def group_structure(g: Dict[str, Any], counts: Dict[str, float]) -> Optional[Structure]:
    """Structure of one grouped observation (AFLOW label, or OQMD space group / cell size / prototype name)."""
    if g.get("label"):
        return label_structure(g["label"], counts)
    sg = sg_number(g.get("sg_symbol"))
    ratio = _ratio(counts)
    return (match_structure(sg, g.get("natoms"), ratio, counts)
            or _proto_structure(g.get("proto"), sg, ratio, counts))


def _same_composition(name: str, target: Dict[str, int], tol: float = 0.003) -> bool:
    try:
        c = parse_formula(name)
    except ValueError:
        return False
    if set(c) != set(target):
        return False
    x, t = fractions(c), fractions({k: float(v) for k, v in target.items()})
    return all(abs(x[e] - t[e]) <= tol for e in t)


def parse_aflow(entries: Any, counts: Dict[str, float]) -> List[Dict[str, Any]]:
    """AFLUX entries -> observations at this composition, grouped by structure.  ICSD entries: the original
    (experimental) label - their relaxed one is often symmetry-broken by magnetism; LIB entries: the relaxed label
    and the formation enthalpy (ICSD entries have none)."""
    target = reduced(counts) or {}
    groups: Dict[str, Dict[str, Any]] = {}
    for e in entries if isinstance(entries, list) else []:
        if not isinstance(e, dict) or not _same_composition(str(e.get("compound", "")), target):
            continue
        icsd = str(e.get("catalog", "")).upper() == "ICSD"
        label = (e.get("aflow_prototype_label_orig") if icsd else
                 (e.get("aflow_prototype_label_relax") or e.get("aflow_prototype_label_orig")))
        if not label:
            continue
        sg = e.get("spacegroup_orig") if icsd else e.get("spacegroup_relax")
        st = label_structure(label, counts)
        key = st.prototype if st else label
        g = groups.setdefault(key, {"label": label, "sg": sg, "n": 0, "n_icsd": 0, "H": None, "id": None})
        g["n"] += 1
        g["n_icsd"] += int(icsd)
        h = e.get("enthalpy_formation_atom")
        if isinstance(h, (int, float)) and (g["H"] is None or h < g["H"]):
            g["H"], g["id"] = float(h), e.get("auid")
            if not icsd:
                g["label"] = label
    return list(groups.values())


def parse_oqmd(resp: Any, counts: Dict[str, float]) -> List[Dict[str, Any]]:
    """OQMD formationenergy response -> observations at this composition, grouped by structure."""
    target = reduced(counts) or {}
    data = resp.get("data", []) if isinstance(resp, dict) else (resp if isinstance(resp, list) else [])
    groups: Dict[str, Dict[str, Any]] = {}
    for d in data:
        if not isinstance(d, dict) or not _same_composition(str(d.get("name", "")), target):
            continue
        g0 = {"proto": d.get("prototype"), "sg_symbol": d.get("spacegroup"), "natoms": d.get("natoms")}
        st = group_structure(g0, counts)
        key = st.prototype if st else f"{g0['proto']}|{g0['sg_symbol']}"
        g = groups.setdefault(key, dict(g0, n=0, n_icsd=0, H=None, stability=None, id=None))
        g["n"] += 1
        g["n_icsd"] += int(bool(d.get("icsd_id")))
        h, stab = d.get("delta_e"), d.get("stability")
        if isinstance(h, (int, float)) and (g["H"] is None or h < g["H"]):
            g["H"], g["id"] = float(h), d.get("entry_id")
            g["stability"] = float(stab) if isinstance(stab, (int, float)) else None
    return list(groups.values())


def _query(source: str, counts: Dict[str, float], timeout: float) -> List[Dict[str, Any]]:
    """Ask one database (only the element symbols / the formula leave this machine)."""
    ints = reduced(counts) or {}
    if source == "aflow":
        key = tuple(sorted(ints))
        if key not in _SYSTEM_MEMO:
            _SYSTEM_MEMO[key] = _http_get_json(aflow_url(key), timeout)
        return parse_aflow(_SYSTEM_MEMO[key], counts)
    if source == "oqmd":
        return parse_oqmd(_http_get_json(oqmd_url(formula_key(counts) or ""), timeout), counts)
    raise ValueError(f"unknown structure source {source!r}")


def _pick(groups: List[Dict[str, Any]], counts: Dict[str, float]) -> Dict[str, Any]:
    """The answer of one source: among the experimentally reported structures (ICSD) the lowest formation energy
    (the most-reported one if no energy is known); computed-only structures when nothing is reported."""
    rows = [(g, group_structure(g, counts)) for g in groups]
    observed = [r for r in rows if r[0].get("n_icsd")]
    pool = observed or rows
    with_h = [r for r in pool if r[0].get("H") is not None]
    best = (min(with_h, key=lambda r: r[0]["H"]) if with_h else
            max(pool, key=lambda r: (r[0].get("n_icsd", 0), r[0].get("n", 0))))
    sufs = {s.suffix for g, s in observed if s}
    amb = {"2:17R", "2:17H"} <= sufs
    alts = []
    for g, s in sorted(rows, key=lambda r: (not r[0].get("n_icsd"), r[0].get("H") if r[0].get("H") is not None
                                            else 9.0)):
        if g is best[0] or (s is not None and best[1] is not None and s.prototype == best[1].prototype):
            continue
        if not g.get("n_icsd") and (s is None or g.get("H") is None or g["H"] >= 0):
            continue                                  # computed only and unstable / unrecognised: noise
        alts.append((g, s))
    return {"best": best, "observed": bool(observed), "amb217": amb, "alts": alts[:3]}


def _obs_text(src: str, g: Dict[str, Any], s: Optional[Structure], chosen: bool = True) -> str:
    proto = s.prototype if s else (g.get("proto") or g.get("label"))
    h = g.get("H")
    if src == "aflow":
        t = f"AFLOW {g.get('label')}" + (f" ({proto}-type)" if s else "")
        n = g.get("n_icsd", 0)
        t += f", {n} ICSD entr{'y' if n == 1 else 'ies'}" if n else ", computed only"
        if h is not None:
            t += f", ΔH_f {h:+.3f} eV/atom"
        return t
    t = f"OQMD entry {g.get('id')}, " + (f"{proto} prototype" if proto else f"{g.get('sg_symbol')}, no prototype")
    stab = g.get("stability")
    if stab is not None:
        t += ", ground state" if stab <= 1e-3 else f", {stab:.3f} eV/atom above the hull"
    if h is not None:
        t += f" (ΔH_f {h:+.2f} eV/atom)"
    if not g.get("n_icsd"):
        t += ", computed only"
    return t


def _combine(obs: Dict[str, List[Dict[str, Any]]], counts: Dict[str, float], disp: str,
             from_cache: Dict[str, bool]) -> Optional[StructureID]:
    picks = {src: _pick(gs, counts) for src, gs in obs.items() if gs}
    # computed-only answers count only for a recognised structure with a negative formation energy
    picks = {src: p for src, p in picks.items() if p["observed"] or (
        p["best"][1] is not None and p["best"][0].get("H") is not None and p["best"][0]["H"] < 0)}
    if not picks:
        return None

    def key(p):
        if p["amb217"]:
            return "2:17"
        g, s = p["best"]
        return s.prototype if s else (g.get("proto") or g.get("label"))

    order = list(picks)
    # a source whose answer is no recognised structure (OQMD's relaxed CeCo5 is Cmmm, no prototype) does not
    # contradict one that recognises a structure: it is quoted, not counted
    known = [s for s in order if picks[s]["amb217"] or picks[s]["best"][1] is not None]
    quiet = [s for s in order if s not in known] if known else []
    order = [s for s in order if s not in quiet]
    keys = {src: key(picks[src]) for src in order}
    agree = len(set(keys.values())) == 1
    clear = False
    if agree:
        main = order[0]
    else:   # disagreement (DFT ground states differ): the structure with the most ICSD reports over all sources
        reports: Dict[str, int] = {}
        for gs in obs.values():
            for g in gs:
                st = group_structure(g, counts)
                k = st.prototype if st else (g.get("proto") or g.get("label"))
                reports[k] = reports.get(k, 0) + int(g.get("n_icsd") or 0)
        ranked = sorted(order, key=lambda s: -reports.get(keys[s], 0))
        main = ranked[0]
        top = sorted(reports.values(), reverse=True) + [0, 0]
        clear = top[0] >= 2 * max(top[1], 1) and reports.get(keys[main], 0) == top[0]
    p = picks[main]
    if len(order) > 1:
        conf = "high" if agree and any(picks[s]["observed"] for s in order) else (
            "medium" if agree or clear else "low")
    else:
        conf = "medium" if p["observed"] else "low"
    if p["best"][1] is None and not p["amb217"]:
        conf = "low"                                  # no recognised structure: no suffix, weakly
    source = "cache" if all(from_cache.get(s) for s in order) else main
    ev = "; ".join(_obs_text(s, *picks[s]["best"]) for s in order)
    if not agree:
        ev += (" - the databases' ground states disagree; " + (f"{keys[main]} has most ICSD reports" if clear
                                                               else "no clear majority"))
    for s in quiet:
        g = picks[s]["best"][0]
        ev += (f"; {s.upper()}: {g.get('sg_symbol') or sg_symbol(g.get('sg'))}"
               f"{', ' + str(g.get('proto')) if g.get('proto') else ''} (no recognised prototype; not counted)")
    if p["amb217"]:
        r = _make_217(disp, source, "medium" if conf == "high" else conf, ev + "; both 2:17 polymorphs reported")
        return r
    g, s = p["best"]
    r = _make(s, disp, counts, source, conf, ev)
    if not any(picks[x]["observed"] for x in order) and r.suffix:
        # a computed-only structure is a hypothetical decoration (AFLOW / OQMD prototype runs), not a report of
        # the real one (Al4Cr: MoNi4-type computed, μ-Al4Mn-type observed): quoted, never a suffix
        r.evidence += f"; no ICSD report - the computed {r.prototype}-type gives no suffix"
        r.suffix, r.confidence = None, "low"
    if s is None:
        r.prototype = g.get("proto") or g.get("label")
        r.space_group = g.get("sg_symbol") or sg_symbol(g.get("sg"))
        r.aflow_label = g.get("label")
    elif g.get("label"):
        r.aflow_label = g["label"]
    seen = {r.prototype}
    for src in order:
        q = picks[src]
        cand = q["alts"] + ([q["best"]] if src != main else [])
        for g2, s2 in cand:
            proto = s2.prototype if s2 else (g2.get("proto") or g2.get("label"))
            if proto in seen:
                continue
            seen.add(proto)
            r.alternatives.append(_alt(s2, _obs_text(src, g2, s2), "database", proto, counts))
    r.alternatives = r.alternatives[:4]
    return r


def lookup_online(formula: Union[str, Dict[str, float]], *, allow_online: Optional[bool] = None,
                  use_cache: bool = True) -> Tuple[Optional[StructureID], List[str]]:
    """Cache, then AFLOW / OQMD.  (result or None, notes).  Failures fall through quietly with a note."""
    counts = parse_formula(formula)
    disp = display_formula(formula, counts)
    notes: List[str] = []
    key = formula_key(counts)
    if not key or len(counts) < 2:
        return None, notes
    st = settings()
    allowed, why = online_allowed(allow_online)
    cache = _cache_load() if use_cache else {}
    entry = dict(cache.get(key, {}))
    obs: Dict[str, List[Dict[str, Any]]] = {}
    from_cache: Dict[str, bool] = {}
    changed = False
    now = time.time()
    for src in st.sources:
        c = entry.get(src)
        if c and now - float(c.get("t", 0)) <= st.cache_days * 86400:
            obs[src], from_cache[src] = c.get("groups", []), True
            continue
        if not allowed:
            continue
        if _DOWN.get(src, 0) > now:
            notes.append(f"{src.upper()} skipped (unreachable earlier in this session)")
            continue
        try:
            obs[src] = _query(src, counts, st.timeout_s)
            from_cache[src] = False
            entry[src] = {"t": now, "groups": obs[src]}
            changed = True
        except Exception as e:  # noqa: BLE001 - network / parsing: fall through
            _DOWN[src] = now + 600
            notes.append(f"{src.upper()} not reachable ({type(e).__name__}); offline sources only")
    if not allowed and not obs:
        notes.append(why)
    if changed and use_cache:
        cache = _cache_load()
        cache[key] = entry
        _cache_save(cache)
    return _combine(obs, counts, disp, from_cache), notes


# ----------------------------------------------------------------------------------------------- chemistry rule
def chemistry_rule(formula: Union[str, Dict[str, float]]) -> Optional[StructureID]:
    """Rare earth (La-Lu, Y, Sc) or Th with Mn, Fe, Co, Ni, Cu (Zn for 2:17) at 1:5 / 1:12 / 2:17 -> the CaCu5
    family, low confidence.  Never for Zr, Hf, U, Al; never Laves or a Greek letter."""
    counts = parse_formula(formula)
    ints = reduced(counts)
    if not ints or len(ints) != 2 or set(ints) & NEVER_CACU5:
        return None
    r_el = [e for e in ints if e in RARE_EARTHS or e == "Th"]
    if len(r_el) != 1:
        return None
    r = r_el[0]
    t = next(e for e in ints if e != r)
    ratio = (ints[r], ints[t])
    disp = display_formula(formula, counts)
    ev = f"chemistry rule: {r} + {t} at {ratio[0]}:{ratio[1]} (no structure data)"
    if ratio == (2, 17) and (t in CACU5_TM or t == "Zn"):
        return _make_217(disp, "chemistry_rule", "low", ev)
    if t in CACU5_TM and ratio in ((1, 5), (1, 12)):
        return _make(BY_PROTO["CaCu5" if ratio == (1, 5) else "ThMn12"], disp, counts, "chemistry_rule", "low", ev)
    return None


# ----------------------------------------------------------------------------------------------- overrides
_SUFFIX_RX = re.compile(r"^(?:\d+:\d+[RH]?|[A-L]\d+(?:_\{\w+\}|\w*)?|Laves|[σμχηδελτθκωζ]|R|none|)$", re.I)


def override_for(overrides: Optional[Dict[str, str]], phase_name: str = "", formula: Any = None
                 ) -> Optional[Tuple[str, str]]:
    """(key, value) of the override for a phase name or formula (case-insensitive), or None."""
    if not overrides:
        return None
    norm = {str(k).strip().upper(): (str(k), v) for k, v in overrides.items()}
    cands = []
    if phase_name:
        cands += [phase_name.strip().upper(), re.sub(r"#\d+$", "", phase_name.strip().upper())]
    if formula not in (None, "", {}):
        try:
            c = parse_formula(formula)
            cands += [display_formula(formula, c).upper(), (formula_key(c) or "").upper()]
        except ValueError:
            pass
    for c in cands:
        if c and c in norm:
            return norm[c][0], str(norm[c][1] if norm[c][1] is not None else "")
    return None


def _from_override(key: str, value: str, disp: str, counts: Dict[str, float]) -> StructureID:
    v = value.strip()
    m = re.search(r"\(([^()]*)\)\s*$", v)
    if m:
        suffix, label = m.group(1).strip() or None, v
    elif _SUFFIX_RX.match(v):
        suffix, label = (None if v.lower() in ("", "none") else v), None
    else:
        suffix, label = None, v
    st = BY_SUFFIX.get(suffix or "")
    sb = BY_SB.get((plain(suffix) or "").upper())
    if st is None and sb is not None:          # "L12" or "L1_{2}" -> the designation in markup
        st, suffix = sb, sb.strukturbericht
    r = _make(st, disp, counts, "override", "high", f"override {key} = {value!r}") if st else \
        StructureID(formula=disp, suffix=suffix, source="override", confidence="high",
                    evidence=f"override {key} = {value!r}")
    r.suffix, r.label = suffix, label
    return r


# ----------------------------------------------------------------------------------------------- identify
def identify(formula: Union[str, Dict[str, float], None], *, phase_name: str = "",
             allow_online: Optional[bool] = None, temperature_K: Optional[float] = None,
             overrides: Optional[Dict[str, str]] = None, use_cache: bool = True) -> StructureID:
    """Structure family of one compound formula (one end-member).

    ``phase_name``: the TDB phase, whose structural tokens beat the table and the databases; ``allow_online``:
    False = no network for this call (None / True: as configured - a call never switches lookups on against
    ``[structure] online = false`` or CALPHAD_STRUCTURE_ONLINE=0); ``temperature_K``: kept on the result (a single
    temperature cannot pick a polymorph - see :func:`assign_217`); ``overrides``: phase name / formula -> suffix or
    full label (default: ``[labels] phase_labels`` of the config)."""
    counts = parse_formula(formula) if formula not in (None, "", {}) else {}
    disp = display_formula(formula, counts) if counts else (phase_name or "?")
    ov = settings().overrides if overrides is None else overrides
    hit = override_for(ov, phase_name, formula if counts else None)
    if hit:
        return _with_t(_from_override(hit[0], hit[1], disp, counts), temperature_K)

    if len(counts) == 1:
        return _with_t(StructureID(formula=disp, source="none", confidence="high",
                                   evidence="a pure element: no compound suffix (FCC / BCC / HCP keep their names)"),
                       temperature_K)
    nm = name_evidence(phase_name, counts or None) if phase_name else None
    if nm and nm[1] not in ("Laves", "2:17"):
        st, label, tok = nm
        r = _make(st, disp, counts, "phase_name", "high", f"phase name token {tok}")
        tab = table_lookup(counts) if counts else None
        if tab is not None and tab.prototype != r.prototype and tab.suffix != r.suffix:
            r.confidence = "medium"
            r.evidence += f"; the table says {tab.suffix or 'no suffix'} ({tab.prototype}) - the name wins"
            r.alternatives.append({k: getattr(tab, k) for k in ("suffix", "prototype", "space_group",
                                                                "aflow_label")} | {"kind": "table",
                                                                                   "note": tab.evidence})
        return _with_t(r, temperature_K)

    data, notes = _data(counts, allow_online, use_cache) if counts else (None, [])
    if nm:                                              # family-level name: LAVES / 2_17 / RE2TM17
        _, fam, tok = nm
        if data is not None and data.group == fam:
            if data.suffix != fam:
                data.evidence = f"phase name token {tok}; " + data.evidence
                return _with_t(data, temperature_K, notes)
        if fam == "2:17":
            r = _make_217(disp, "phase_name", "medium", f"phase name token {tok}; R / H not determined")
        else:
            r = StructureID(formula=disp, suffix="Laves", source="phase_name", confidence="medium",
                            evidence=f"phase name token {tok}; C14 / C15 / C36 not determined",
                            alternatives=[_alt(BY_SUFFIX[s]) for s in ("C14", "C15", "C36")])
        if data is not None and data.group != fam:
            r.evidence += f"; data say {data.suffix or 'no suffix'} ({data.prototype}, {data.source})"
        return _with_t(r, temperature_K, notes)
    if data is not None:
        return _with_t(data, temperature_K, notes)
    chem = chemistry_rule(counts) if counts else None
    if chem is not None:
        return _with_t(chem, temperature_K, notes)
    return _with_t(StructureID(formula=disp, source="none", confidence="low",
                               evidence="no structural evidence (an unknown prototype gives no suffix)"),
                   temperature_K, notes)


def _with_t(r: StructureID, t: Optional[float], notes: Sequence[str] = ()) -> StructureID:
    r.temperature_K = t
    r.notes.extend(n for n in notes if n not in r.notes)
    return r


def _data(counts, allow_online, use_cache) -> Tuple[Optional[StructureID], List[str]]:
    """Table, then cache / online."""
    tab = table_lookup(counts)
    if tab is not None:
        return tab, []
    return lookup_online(counts, allow_online=allow_online, use_cache=use_cache)


# ----------------------------------------------------------------------------------------------- phases
@dataclass
class PhaseFamily:
    """Structure family of one CALPHAD phase from its end-members (each identified on its own) and its name.

    ``name_suffix`` / ``name_evidence``: the phase name's own structural tokens (CE2FE17_R, TH2NI17, C14,
    or an override of the phase) - authoritative for the whole phase.  ``consensus``: the suffix when every
    end-member with an answer agrees and none lists a conflicting polymorph.  ``ambiguous_label``: the family-level
    label they share ("2:17", "Laves", or the suffix itself).  ``warning``: set when one phase covers several
    structures (end-members, or the polymorphs of one end-member, of different variants)."""
    phase: str
    per_endmember: Dict[str, StructureID] = field(default_factory=dict)
    consensus: Optional[str] = None
    ambiguous_label: Optional[str] = None
    name_suffix: Optional[str] = None
    name_evidence: Optional[str] = None
    warning: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def suffix(self) -> Optional[str]:
        """Recommended suffix: the name's (unless it only names the family: LAVES, RE2TM17), else the consensus,
        else None (the labeller decides)."""
        return self.name_suffix if self.name_definite else self.consensus

    @property
    def name_definite(self) -> bool:
        return bool(self.name_evidence) and self.name_suffix not in ("2:17", "Laves")

    def lines(self) -> List[str]:
        sfx = self.suffix
        head = f"{self.phase}: " + (sfx if sfx else (f"no single suffix (family {self.ambiguous_label})"
                                                     if self.ambiguous_label else "no suffix"))
        if self.name_evidence:
            head += f" [{self.name_evidence}]"
        out = [head]
        out += ["  " + r.line() for r in self.per_endmember.values()]
        if self.warning:
            out.append("  WARNING: " + self.warning)
        return out


def phase_family(phase_name: str, endmember_formulas: Sequence[Union[str, Dict[str, float]]], *,
                 online: Optional[bool] = None, overrides: Optional[Dict[str, str]] = None,
                 use_cache: bool = True) -> PhaseFamily:
    """Identify each end-member of a phase (per formula: Ce2Fe17 -> 2:17R, Ce2Co17 -> 2:17H) and combine."""
    ov = settings().overrides if overrides is None else overrides
    pf = PhaseFamily(phase=phase_name)
    for f in endmember_formulas:
        r = identify(f, allow_online=online, overrides=ov, use_cache=use_cache)
        pf.per_endmember[r.formula] = r
    hit = override_for(ov, phase_name)
    if hit:
        o = _from_override(hit[0], hit[1], phase_name, {})
        pf.name_suffix, pf.name_evidence = o.suffix, f"override {hit[0]} = {hit[1]!r}"
    else:
        counts = None
        for f in endmember_formulas:
            c = parse_formula(f)
            if _ratio(c) == (2, 17):
                counts = c
        nm = name_evidence(phase_name, counts)
        if nm:
            pf.name_suffix, pf.name_evidence = (nm[1] or None), f"phase name token {nm[2]}"
    answers = [(f, r.suffix) for f, r in pf.per_endmember.items() if r.suffix]
    poly = [(f, a["suffix"]) for f, r in pf.per_endmember.items() for a in r.alternatives
            if a.get("suffix") and a.get("kind") == "polymorph"]
    model = [(f, a["suffix"]) for f, r in pf.per_endmember.items() for a in r.alternatives
             if a.get("suffix") and a.get("kind") == "model"]
    sufs = {s for _, s in answers}
    definite = {s for s in sufs if s not in ("2:17", "Laves")}
    if len(sufs) == 1 and definite and not any(s not in sufs for _, s in poly):
        pf.consensus = next(iter(sufs))
    groups = {group_of(s) for _, s in answers + poly}
    if len(groups) == 1:
        pf.ambiguous_label = next(iter(groups))
    # warning: one phase, several structures
    span: Dict[str, List[str]] = {}
    for f, s in answers:
        for v in (["2:17R", "2:17H"] if s == "2:17" else [s]):
            span.setdefault(v, []).append(f)
    if not pf.name_definite:
        for f, s in poly:
            span.setdefault(s, []).append(f"{f} polymorph")
    if len(pf.per_endmember) > 1:
        for f, s in model:
            span.setdefault(s, []).append(f"{f} in some models")
    if len(span) > 1:
        parts = [f"{s} ({', '.join(dict.fromkeys(fs))})"
                 for s, fs in sorted(span.items(), key=lambda kv: kv[0].replace("2:17R", "2:17A"))]
        both = " and ".join(parts) if len(parts) == 2 else ", ".join(parts)
        pf.warning = (f"{phase_name} is a single CALPHAD phase representing "
                      f"{'both ' if len(parts) == 2 else ''}{both} - be careful with these labels")
    for r in pf.per_endmember.values():
        pf.notes.extend(n for n in r.notes if n not in pf.notes)
    return pf


_HT_TOKENS = {"HT": "2:17H", "HIGH": "2:17H", "LT": "2:17R", "LOW": "2:17R"}


def assign_217_detail(phases: Sequence[Dict[str, Any]], *, allow_online: Optional[bool] = None
                      ) -> Dict[str, Dict[str, str]]:
    """R vs H for the 2:17 phases of one diagram: {phase: {"suffix", "source", "confidence", "evidence"}}.

    Each phase: {"phase": name, "formula": "Ce2Co17" (or "formulas": [...]), "T": highest temperature where it is
    stable (or "T_max" / "temperature_K")}.  Order: name tokens (TH2ZN17, _R, _H, HEX ...); the structure data of
    its formula(s); the complement of an assigned partner; the higher-temperature phase is 2:17H (low confidence).
    Undecidable: "2:17"."""
    out: Dict[str, Dict[str, str]] = {}
    names = [str(p.get("phase") or p.get("name")) for p in phases]
    for p, n in zip(phases, names):
        nm = name_evidence(n, {"Ce": 2, "Co": 17})          # a 2:17 context by definition
        if nm and nm[1] in ("2:17R", "2:17H"):
            out[n] = {"suffix": nm[1], "source": "phase_name", "confidence": "high",
                      "evidence": f"phase name token {nm[2]}"}
            continue
        ht = [t for t in _name_tokens(n)[1:] if t in _HT_TOKENS]
        if ht:
            out[n] = {"suffix": _HT_TOKENS[ht[0]], "source": "phase_name", "confidence": "medium",
                      "evidence": f"phase name token {ht[0]} (the high-T form is usually 2:17H)"}
    for p, n in zip(phases, names):
        if n in out:
            continue
        forms = p.get("formulas") or ([p["formula"]] if p.get("formula") else [])
        got = {identify(f, allow_online=allow_online).suffix for f in forms}
        got = {s for s in got if s}
        if len(got) == 1 and next(iter(got)) in ("2:17R", "2:17H"):
            s = next(iter(got))
            taken = [m for m, v in out.items() if v["suffix"] == s and v["source"] == "structure"]
            if not taken:
                out[n] = {"suffix": s, "source": "structure", "confidence": "medium",
                          "evidence": f"structure data of {', '.join(map(str, forms))}"}
    left = [n for n in names if n not in out]
    if len(names) == 2 and len(left) == 1:
        other = out[next(n for n in names if n != left[0])]["suffix"]
        out[left[0]] = {"suffix": "2:17H" if other == "2:17R" else "2:17R", "source": "complement",
                        "confidence": "medium", "evidence": f"the other 2:17 phase is {other}"}
        left = []

    def t_of(p):
        for k in ("T", "T_max", "t_max", "temperature_K"):
            if isinstance(p.get(k), (int, float)):
                return float(p[k])
        return None
    temps = {n: t_of(p) for p, n in zip(phases, names) if n in left}
    known = {n: t for n, t in temps.items() if t is not None}
    if len(known) >= 2 and len(set(known.values())) > 1:
        hi = max(known, key=known.get)
        for n in known:
            out[n] = {"suffix": "2:17H" if n == hi else "2:17R", "source": "temperature", "confidence": "low",
                      "evidence": "the higher-temperature 2:17 phase is usually 2:17H"}
    for n in names:
        out.setdefault(n, {"suffix": AMBIGUOUS_217, "source": "none", "confidence": "low",
                           "evidence": "R / H not decidable (no name token, structure data or temperatures)"})
    return out


def assign_217(phases: Sequence[Dict[str, Any]], *, allow_online: Optional[bool] = None) -> Dict[str, str]:
    """{phase: "2:17R" | "2:17H"} for the 2:17 phases of one diagram ("2:17" if undecidable); see
    :func:`assign_217_detail` for the evidence."""
    return {n: d["suffix"] for n, d in assign_217_detail(phases, allow_online=allow_online).items()}


# ----------------------------------------------------------------------------------------------- TDB phases
_SOLUTION_TOKENS = {"LIQUID", "LIQ", "GAS", "FCC", "BCC", "HCP", "DHCP", "CBCC", "CUB", "BCT", "DIAMOND",
                    "IONIC", "SLAG", "A1", "A2", "A3", "GRAPHITE"}


def _endmembers(subl: List[List[str]], sites: List[float]) -> List[Dict[str, float]]:
    import itertools
    out, seen = [], set()
    for ch in itertools.product(*subl):
        amt: Dict[str, float] = {}
        for c, s in zip(ch, sites):
            if c != "VA":
                amt[c] = amt.get(c, 0.0) + float(s)
        if len(amt) < 2:
            continue
        k = formula_key({_EL_UP.get(e, e): v for e, v in amt.items()}) if all(e in _EL_UP for e in amt) else None
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append({_EL_UP[e]: v for e, v in amt.items()})
        if len(out) >= 12:
            break
    return out


def compound_phases(tdb_path: str, elements: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Stoichiometric and near-stoichiometric phases of a TDB: [{"phase", "endmembers": [formula], "how"}].

    Solutions (liquid, gas, fcc / bcc / hcp ..., one-sublattice phases, order-disorder parts) are skipped.  A phase
    with several end-members keeps those the offline table knows (Ce2(Co,Fe)17 -> Ce2Co17, Ce2Fe17), else the one
    marked with % in CONSTITUENT, else the one its name spells, else the first constituents."""
    from .tdbparse import parse_tdb
    db = parse_tdb(tdb_path)
    allowed = {e.upper() for e in elements} | {"VA"} if elements else None
    dis_parts = set()
    for td in db.typedefs.values():
        m = re.search(r"DIS_PART\s+(\S+)", td.text, re.I)
        if m:
            dis_parts.add(m.group(1).upper().rstrip(","))
    out = []
    for name, ph in db.phases.items():
        toks = set(_name_tokens(name))
        if toks & _SOLUTION_TOKENS or name.upper() in dis_parts:
            continue
        subl = [[c.upper() for c in cs if allowed is None or c.upper() in allowed] for cs in ph.constituents]
        if not subl or any(not s for s in subl):
            continue
        real = [s for s in subl if s != ["VA"]]
        if len(real) < 2 and not (len(real) == 1 and len(real[0]) == 1):
            continue
        if max(len([c for c in s if c != "VA"]) for s in subl) > 3:
            continue
        ends = _endmembers(subl, ph.site_ratios)
        if not ends:
            continue
        how = "stoichiometric" if len(ends) == 1 and all(len(s) == 1 for s in subl) else "model end-members"
        if len(ends) > 1:
            known = [e for e in ends if table_lookup(e) is not None]
            majors = [[c.upper() for c in m] for m in (ph.majors or [])]
            if known:
                ends, how = known, "end-members in the offline table"
            elif majors and len(majors) == len(subl) and all(len(m) == 1 for m in majors):
                ends = _endmembers(majors, ph.site_ratios) or ends[:1]
                how = "major constituents (%)"
            else:
                try:
                    named = parse_formula(re.split(r"[_#]", name)[0])
                    nk = formula_key(named)
                    hit = [e for e in ends if formula_key(e) == nk]
                except ValueError:
                    hit = []
                ends, how = (hit, "formula in the name") if hit else (ends[:1], "first constituents (a guess)")
        out.append({"phase": name, "endmembers": [display_formula(e) for e in ends], "how": how})
    return out


# ----------------------------------------------------------------------------------------------- tool
def register(tb) -> None:
    S = {"type": "string"}

    @tb.add("identify_structure",
            "Structure family of a compound for labels (CeCo5 -> 1:5, Ni3Al -> L1_{2}, ZrNi5 -> C15_{b}). "
            "Suffix: Greek letter, 1:5 / 2:17R family, Laves, else the Strukturbericht symbol (_{..} = subscript; "
            "never A1-A12), for compound phases only. Sources in order: [labels] phase_labels overrides, TDB phase-name tokens, a built-in table, cached / "
            "online AFLOW + OQMD (only the chemical formula is sent, never TDB content; [structure] online = false "
            "or CALPHAD_STRUCTURE_ONLINE=0 for offline), a rare-earth CaCu5 rule. Give formula (comma-separated "
            "for several), or tdb + phase, or tdb alone to check every compound phase (one line per end-member, "
            "warnings when one phase covers several structures).",
            {"formula": {"type": "string", "description": "e.g. 'CeCo5' or 'Ce2Co17, Ce2Fe17'"},
             "tdb": S, "phase": {"type": "string", "description": "a phase of tdb (or a name to read tokens from)"},
             "online": {"type": "boolean", "description": "false = no network for this call"},
             "elements": {"type": "array", "items": {"type": "string"},
                          "description": "tdb mode: only phases / end-members in these elements"}}, [])
    def identify_structure(formula=None, tdb=None, phase=None, online=None, elements=None):
        on, why = online_allowed(online)
        foot = ("online: AFLOW + OQMD (formulas only)" if on else why)
        if formula:
            forms = [f.strip() for f in str(formula).split(",") if f.strip()]
            lines = [identify(f, phase_name=phase or "", allow_online=online).line() for f in forms]
            return "\n".join(lines + [f"({foot})"]), []
        if not tdb:
            raise ValueError("give formula, or tdb (+ phase)")
        path = tb._find_tdb(tdb)
        rows = compound_phases(path, elements)
        if phase:
            rows = [r for r in rows if r["phase"].upper() == str(phase).upper()]
            if not rows:
                raise ValueError(f"{phase} is not a stoichiometric / near-stoichiometric phase of {tdb} "
                                 f"(within {elements or 'all elements'})")
        out = [f"{len(rows)} compound phase(s) in {Path(path).name}" + (f" ({'-'.join(elements)})" if elements
                                                                          else "")]
        for r in rows:
            pf = phase_family(r["phase"], r["endmembers"], online=online)
            ls = pf.lines()
            ls[0] += f"  <{r['how']}>"
            out += ls
        out.append(f"({foot})")
        return "\n".join(out), []
