"""Readable labels for CALPHAD phase names (plain Unicode, one string per label).

Examples: BCC_A2 -> BCC, LIQUID -> L, L12_FCC -> L1₂, SIGMA_D8B -> σ, MU_PHASE -> μ, GAMMA_PRIME -> γ′,
LAVES_C14 -> C14, FE3CR4 -> Fe₃Cr₄, M23C6 -> M₂₃C₆, AL8CR5_H -> Al₈Cr₅-HT, BCC_A2#2 -> BCC.

The rules are table driven (``NAMES``, ``GREEK``, ``BASES``, ``ORDERED``): add a row to extend them.
Order of the rules in :func:`format_phase`:

1. composition-set markers are removed (``#2``; ``_0``/``_1`` when the rest is a known name);
2. the exact-name table ``NAMES``;
3. Greek-letter names (``SIGMA``, ``SIGMA_D8B``, ``D8B_SIGMA``, ``MU_PHASE``, ...);
4. structure names: base lattice + Strukturbericht symbol in either order (``FCC_A1``, ``A2_BCC``, ``BCC_B2``,
   ``FCC_L12`` ...): a disordered symbol (A1, A2, A3) gives the base name, an ordered one gives the symbol;
5. chemical formulas, split into element symbols of the database (longest symbol first, with backtracking, so
   COSI with elements CO, SI, C, O is Co+Si); ``M`` is a generic metal (M23C6); a trailing structure symbol
   (``_D0_19``, ``_L12``) is dropped, ``_H``/``_L`` become ``-HT``/``-LT``;
6. anything else is kept as written.

Choices: CBCC_A12 is written αMn and CUB_A13 βMn (the element prototypes, as in most assessments); the
alternative A12 / A13 can be set per call through ``overrides``. HIGH_SIGMA is written σ-HT.

Compound labels (:func:`compound_label`): the own label of a stoichiometric (or practically stoichiometric) phase
is its formula from the sublattice model, plus a family name for well-known structures: "Fe₂Nb (C14)",
"CeCo₂ (C15)", "CeCo₅ (1:5)", "Ce₂Co₁₇ (2:17R)", "Fe₇Mo₆ (μ)", "Al₃Ni₂" (no family).  Laves phases were labelled
"C14" alone before; now "formula (C14)" when the Laves phase is labelled as a compound on its own, and still the
short "C14" in a multi-phase region label ("BCC+C14", also CECO2_C15 -> "L+C15") and for a Laves field with a
solubility range.

* formula: one constituent per sublattice - the one that, with the others, gives the composition closest to where
  the phase sits in the diagram (Fe on both sublattices of (Fe)2(Fe,Nb) is Fe₃, but at X(Nb) ~ 1/3 it is Fe₂Nb);
  vacancies dropped, counts reduced to the smallest integers (0.6667:0.3333 -> Fe₂Nb, 2:17 -> Ce₂Co₁₇).  A
  ternary line compound whose two ends differ gets the mixed sublattice as "(Co,Ni)₃V".  Element order: as in
  the phase name when the name is a formula of the same elements (AL8CR5_L -> Al₈Cr₅-LT, its -HT/-LT kept;
  ``M`` stands for the one remaining element: M3V -> Ni₃V on the Ni-V edge), else the order of the sublattices;
* family (:func:`phase_family`), only from evidence - never from the ratio alone (ZrNi₅, UNi₅ are not CaCu5-type,
  WAl₁₂ is not ThMn12-type, MoSi₂ / CuAl₂ / Ni₂Zr-type AB2 phases are not Laves):
  Laves from name tokens C14 / C15 / C36 / LAVES / MGZN2 / MGCU2 / MGNI2; Greek letters from Greek names (SIGMA,
  MU_D85, CHI_A12, ETA, TAU ..., not ALPHA / BETA / GAMMA); the CaCu5 family (1:5, 1:12, 2:17) from tokens
  CACU5 / D2D / THMN12 / D2B / TH2ZN17 / TH2NI17 / RT5 / RT12 / RE2TM17 / 1_5 / 1_12 / 2_17 or an R-T name
  (CE2T17), and - as the only fallback - a phase named by its formula (CECO5, CE2CO17) of a rare earth (La-Lu, Y,
  Sc) or Th with Mn, Fe, Co, Ni, Cu (and Zn for 2:17) in the ratio 1:5, 1:12 or 2:17.  When in doubt: formula only.
  2:17R (Th2Zn17-type: tokens TH2ZN17, R, RHOMB, R3M) / 2:17H (Th2Ni17-type: TH2NI17, H, HEX, P63MMC) from the
  name decide for the whole phase; end-member structures (structure_id), two 2:17 phases in one diagram and the
  warnings: label_place.phase_structures / resolve_families.  Multi-phase labels use the family as the short name
  (L+C15, 1:5+Ce₅Co₁₉, L+2:17).
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# exact names (upper case, composition-set marker removed) -> label.  Listed in the preferred label order
NAMES: Dict[str, str] = {
    "LIQUID": "L", "LIQ": "L", "LIQUID_PHASE": "L", "IONIC_LIQ": "L", "IONIC_LIQUID": "L", "SLAG": "Slag",
    "GAS": "Gas", "GAS_PHASE": "Gas", "VAPOR": "Gas", "VAPOUR": "Gas",
    "BCC": "BCC", "FCC": "FCC", "HCP": "HCP", "DHCP": "DHCP", "HCP_ZN": "HCP", "DHCP_A3": "DHCP",
    "CBCC_A12": "αMn", "A12_CBCC": "αMn", "CBCC": "αMn", "CUB_A13": "βMn", "A13_CUB": "βMn", "CUB": "βMn",
    "BCT_A5": "BCT", "A5_BCT": "BCT", "TETRAGONAL_A6": "A6", "RHOMBOHEDRAL_A7": "A7", "ORTHORHOMBIC_A20": "A20",
    "DIAMOND_A4": "Diamond", "A4_DIAMOND": "Diamond", "DIAMOND": "Diamond", "GRAPHITE": "Graphite",
    "GRAPHITE_A9": "Graphite", "CEMENTITE": "Fe₃C", "CEMENTITE_D011": "Fe₃C",
    "R_PHASE": "R", "P_PHASE": "P", "Z_PHASE": "Z", "G_PHASE": "G", "M_PHASE": "M", "T_PHASE": "T",
    "HIGH_SIGMA": "σ-HT", "LAVES": "Laves", "LAVES_PHASE": "Laves",
}

# Greek-letter phase names (a name token) -> letter.  GAMMA_PRIME etc. are matched as whole names first
GREEK_NAMES: Dict[str, str] = {
    "GAMMA_PRIME": "γ′", "GAMMA_P": "γ′", "GPRIME": "γ′", "GAMMAPRIME": "γ′", "GAMMA_DPRIME": "γ″",
    "GAMMA_DOUBLE_PRIME": "γ″", "GAMMA_PP": "γ″", "GDPRIME": "γ″",
}
GREEK: Dict[str, str] = {
    "SIGMA": "σ", "MU": "μ", "CHI": "χ", "DELTA": "δ", "ETA": "η", "EPSILON": "ε", "ZETA": "ζ", "THETA": "θ",
    "KAPPA": "κ", "LAMBDA": "λ", "TAU": "τ", "OMEGA": "ω", "ALPHA": "α", "BETA": "β", "GAMMA": "γ", "PI": "π",
    "PHI": "φ", "PSI": "ψ", "XI": "ξ", "NU": "ν", "RHO": "ρ", "IOTA": "ι", "OMICRON": "ο", "UPSILON": "υ",
}
# tokens that may follow (or precede) a Greek name without changing it
_NOISE = {"PHASE", "SGTE", "D8B", "D85", "A12", "A13", "D8B1", "PH"}

# base lattices and the Strukturbericht symbols of their disordered forms
BASES: Dict[str, str] = {"FCC": "FCC", "BCC": "BCC", "HCP": "HCP", "DHCP": "DHCP", "BCT": "BCT", "CBCC": "αMn",
                         "CUB": "βMn"}
DISORDERED_SYMBOLS = {"A1", "A2", "A3", "A3'", "A12", "A13", "A5"}
# Laves phases: symbol only
LAVES = {"C14", "C15", "C36"}

# Strukturbericht symbol pattern: letter + digits (+ one lower-case/upper-case letter as in D8b)
_STRUK = re.compile(r"^([ABCDEL])(\d)(\d*)([A-Z]?)$")

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr "
            "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
            "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()


def strukturbericht(sym: str) -> Optional[str]:
    """'L12' -> 'L1₂', 'D019' -> 'D0₁₉', 'D8B' -> 'D8b', 'B2' -> 'B2', 'C14' -> 'C14'; None if not a symbol."""
    s = sym.upper()
    m = _STRUK.match(s)
    if not m:
        return None
    letter, d1, rest, suffix = m.groups()
    if letter in "LD":
        return letter + d1 + rest.translate(SUB) + suffix.lower()
    return letter + d1 + rest + suffix.lower()


def strip_compset(name: str, known: Optional[Iterable[str]] = None) -> str:
    """Composition-set marker removed: BCC_A2#2 -> BCC_A2; FCC_A1_1 -> FCC_A1 when FCC_A1 is a known name."""
    n = str(name).strip().upper()
    n = re.sub(r"#\d+$", "", n)
    m = re.fullmatch(r"(.+)_(\d)", n)
    if m:
        base = m.group(1)
        kn = {k.upper() for k in (known or ())}
        # also a merge's renamed copy of a phase (MU_D85_2 -> MU_D85 -> μ, decision 'rename')
        if base in kn or base in NAMES or base in GREEK or _structure(base) or base in GREEK_NAMES or _greek(base):
            n = base
    return n


def _structure(name: str) -> Optional[str]:
    """Label of 'BASE_SYMBOL' / 'SYMBOL_BASE' structure names, else None."""
    toks = name.split("_")
    if len(toks) != 2:
        return None
    for base, sym in (toks, toks[::-1]):
        if base == "LAVES" and sym in LAVES:
            return sym
        if base in BASES:
            if sym in DISORDERED_SYMBOLS:
                return BASES[base]
            s = strukturbericht(sym)
            if s:
                return s
    return None


def _greek(name: str) -> Optional[str]:
    if name in GREEK_NAMES:
        return GREEK_NAMES[name]
    toks = [t for t in name.split("_") if t]
    greek = [t for t in toks if t in GREEK]
    if len(greek) != 1:
        return None
    rest = [t for t in toks if t not in GREEK]
    if all(t in _NOISE or strukturbericht(t) for t in rest):
        return GREEK[greek[0]]
    return None


def split_formula(word: str, elements: Optional[Sequence[str]] = None) -> Optional[List[Tuple[str, str]]]:
    """[(element, count), ...] for a formula such as FE3CR4 / COSI / M23C6, or None.

    Element symbols are matched longest first with backtracking against ``elements`` (the database elements; the
    periodic table if none).  'M' stands for any metal.  Counts are digit strings ('' for 1)."""
    w = word.upper()
    if not re.fullmatch(r"[A-Z0-9.]+", w) or not w[0].isalpha():
        return None
    allowed = {e.upper() for e in (elements or ELEMENTS) if e and e.upper() not in ("VA", "/-")} | {"M"}

    def rec(i: int) -> Optional[List[Tuple[str, str]]]:
        if i == len(w):
            return []
        for L in (2, 1):
            sym = w[i:i + L]
            if len(sym) == L and sym.isalpha() and sym in allowed:
                j = i + L
                m = re.match(r"\d+(?:\.\d+)?", w[j:])
                num = m.group(0) if m else ""
                tail = rec(j + len(num))
                if tail is not None:
                    return [(sym, num)] + tail
        return None

    parts = rec(0)
    if not parts:
        return None
    has_digit = any(n for _, n in parts)
    if elements is None and not has_digit:
        return None                     # plain words (no database elements to check against) are not formulas
    if not has_digit and len(parts) < 2:
        return None
    return parts


def format_formula(parts: List[Tuple[str, str]]) -> str:
    out = []
    for sym, num in parts:
        el = sym.capitalize()
        if num in ("", "1"):
            out.append(el)
        else:
            out.append(el + num.translate(SUB))
    return "".join(out)


_TEMP_SUFFIX = {"H": "-HT", "HT": "-HT", "L": "-LT", "LT": "-LT", "RT": "-RT"}


def _formula_name(name: str, elements) -> Optional[str]:
    toks = name.split("_")
    head = split_formula(toks[0], elements)
    if head is None:
        return None
    rest = toks[1:]
    # merge split symbols such as D0_19 -> D019
    merged: List[str] = []
    for t in rest:
        if merged and t.isdigit() and re.fullmatch(r"[A-Z]\d", merged[-1]):
            merged[-1] += t
        else:
            merged.append(t)
    suffix = ""
    for t in merged:
        if t in _TEMP_SUFFIX:
            suffix += _TEMP_SUFFIX[t]
        elif t in LAVES or strukturbericht(t) or t in ("PHASE", "SGTE"):
            continue                    # structure designation: dropped
        else:
            return None                 # unknown extra token: keep the name as written
    return format_formula(head) + suffix


def format_phase(name: str, elements: Optional[Sequence[str]] = None,
                 overrides: Optional[Dict[str, str]] = None, known: Optional[Iterable[str]] = None) -> str:
    """Display label of one phase name (see the module docstring)."""
    raw = str(name).strip()
    ov = {str(k).upper(): v for k, v in (overrides or {}).items()}
    if raw.upper() in ov:
        return ov[raw.upper()]
    n = strip_compset(raw, known)
    if n in ov:
        return ov[n]
    lab = _default_label(n, elements)
    return ov.get(lab.upper(), lab) if ov else lab


def _default_label(n: str, elements) -> str:
    if n in NAMES:
        return NAMES[n]
    g = _greek(n)
    if g:
        return g
    if n in BASES:
        return BASES[n]
    s = _structure(n)
    if s:
        return s
    if n in LAVES:
        return n
    f = _formula_name(n, elements)
    if f:
        return f
    return n


def is_liquid_name(name: str) -> bool:
    n = strip_compset(name)
    return NAMES.get(n) == "L" or n.startswith(("LIQ", "IONIC_LIQ"))


_ORDER = list(dict.fromkeys(["L", "Slag", "Gas", "BCC", "B2", "D0₃", "L2₁", "FCC", "L1₂", "L1₀", "HCP", "D0₁₉",
                             "DHCP", "αMn", "βMn", "BCT", "Diamond", "Graphite"]))
_GREEK_ORDER = list(dict.fromkeys(list(GREEK_NAMES.values()) + list(GREEK.values()) + ["σ-HT"]))


def sort_key(name: str, elements=None, liquids: Iterable[str] = ()) -> tuple:
    """Stable order of phases in a region label: liquid, solutions (BCC, FCC, HCP ...), Greek-letter phases,
    Laves, other names alphabetically."""
    n = strip_compset(name)
    lab = _default_label(n, elements)
    liq = {strip_compset(x) for x in liquids}
    if n in liq or lab == "L":
        return (0, 0, lab)
    if lab in _ORDER:
        return (1, _ORDER.index(lab), lab)
    if lab in _GREEK_ORDER:
        return (2, _GREEK_ORDER.index(lab), lab)
    if lab in LAVES:
        return (3, sorted(LAVES).index(lab), lab)
    return (4, 0, lab.lower())


def region_label(phases: Sequence[str], elements=None, overrides=None, liquids: Iterable[str] = (),
                 sep: str = "+") -> str:
    """'L+FCC', 'BCC+BCC' (miscibility gap; composition sets are not numbered), 'BCC+FCC+σ' ..."""
    names = sorted(phases, key=lambda p: sort_key(p, elements, liquids))
    return sep.join(format_phase(p, elements, overrides) for p in names)


def region_key(phases: Sequence[str]) -> Tuple[str, ...]:
    """Name-independent identity of a phase region: sorted phase names without composition-set markers."""
    return tuple(sorted(strip_compset(p) for p in phases))


# ---------------------------------------------------------------------------------------------------------
# compound labels: formula from the sublattice model + structure family ("Fe₂Nb (C14)")
# ---------------------------------------------------------------------------------------------------------
RARE_EARTHS = {"SC", "Y", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
               "TH"}
CACU5_TM = {"MN", "FE", "CO", "NI", "CU"}           # (+ ZN for the 2:17 type)
LAVES_TOKENS = {"C14": "C14", "C15": "C15", "C36": "C36", "MGZN2": "C14", "MGCU2": "C15", "MGNI2": "C36"}
# Greek letters that name intermetallic phases (not α / β / γ: those are the terminal solutions)
GREEK_FAMILY = {"σ", "μ", "χ", "δ", "η", "ε", "ζ", "θ", "κ", "λ", "τ", "ω", "φ", "ψ", "ξ", "ν", "π", "ρ", "ι",
                "γ′", "γ″", "σ-HT"}
CACU5_TOKENS = {"CACU5": "1:5", "D2D": "1:5", "D2_D": "1:5", "RT5": "1:5", "1_5": "1:5",
                "THMN12": "1:12", "D2B": "1:12", "D2_B": "1:12", "RT12": "1:12", "1_12": "1:12",
                "TH2ZN17": "2:17", "TH2NI17": "2:17", "RT17": "2:17", "R2T17": "2:17", "2_17": "2:17"}
CACU5_RATIOS = {(1, 5): "1:5", (1, 12): "1:12", (2, 17): "2:17"}
R217_TOKENS = {"TH2ZN17": "R", "R": "R", "RHOMB": "R", "RHOMBOHEDRAL": "R", "R3M": "R", "R3_M": "R",
               "TH2NI17": "H", "H": "H", "HEX": "H", "HEXAGONAL": "H", "P63MMC": "H", "P63_MMC": "H"}
_RT_NAME = re.compile(r"^(RE|R|[A-Z][A-Z]?)(\d*)(TM|T)(\d+)$")       # RT5, RE2TM17, CE2T17


def _reduce(amounts: Sequence[float]) -> List[int]:
    """Smallest integers in the ratio of ``amounts`` (2:1, 0.6667:0.3333 -> 2:1; 24:11; 8:4:18 -> 4:2:9)."""
    from fractions import Fraction
    from math import gcd
    tot = float(sum(amounts))
    fr = [Fraction(a / tot).limit_denominator(100) for a in amounts]
    den = 1
    for f in fr:
        den = den * f.denominator // gcd(den, f.denominator)
    ints = [int(round(float(f * den))) for f in fr]
    g = 0
    for i in ints:
        g = gcd(g, i)
    return [i // max(g, 1) for i in ints]


def _name_formula(name: str, elements=None):
    """(formula parts [(EL, count str)], temperature suffix, other tokens) of a phase name that starts with a formula
    (AL8CR5_L -> ([('AL','8'),('CR','5')], '-LT', [])), else (None, '', tokens)."""
    toks = [t for t in strip_compset(name).split("_") if t]
    if not toks:
        return None, "", []
    head = split_formula(toks[0], elements)
    if head is None:
        return None, "", toks
    suffix, rest = "", []
    for t in toks[1:]:
        if t in _TEMP_SUFFIX:
            suffix += _TEMP_SUFFIX[t]
        else:
            rest.append(t)
    return head, suffix, rest


def phase_family(name: str, elements=None) -> Optional[Dict]:
    """Structure family of a phase, from evidence in its name only: {"family": "C14" | "σ" | "1:5" | "1:12" | "2:17",
    "variant": "R" | "H" | None (2:17), "evidence": text}, or None.  The ratio alone never gives a family; the
    one fallback is a formula name of a rare earth / Th with Mn-Cu (Zn) in the ratio 1:5, 1:12 or 2:17."""
    n = strip_compset(name)
    toks = [t for t in n.split("_") if t]
    pairs = [a + "_" + b for a, b in zip(toks, toks[1:])]
    for t in toks:
        if t in LAVES_TOKENS:
            return {"family": LAVES_TOKENS[t], "variant": None, "evidence": f"name token {t}"}
    if "LAVES" in toks:
        return {"family": "Laves", "variant": None, "evidence": "name token LAVES"}
    g = NAMES.get(n) or _greek(n)
    if g is None:
        hits = [GREEK[t] for t in toks if t in GREEK]
        g = hits[0] if len(hits) == 1 else None
    if g in GREEK_FAMILY:
        return {"family": g, "variant": None, "evidence": f"Greek-letter name {n}"}
    variant = None
    for t in toks[1:] + [t for t in toks[:1] if t in ("TH2ZN17", "TH2NI17")]:
        if t in R217_TOKENS:
            variant = R217_TOKENS[t]
    for t in pairs + toks:
        if t in CACU5_TOKENS:
            fam = CACU5_TOKENS[t]
            return {"family": fam, "variant": variant if fam == "2:17" else None, "evidence": f"name token {t}"}
        m = _RT_NAME.match(t)
        if m and (m.group(1) in ("R", "RE") or m.group(1) in RARE_EARTHS):
            ratio = (int(m.group(2) or 1), int(m.group(4)))
            if ratio in CACU5_RATIOS:
                fam = CACU5_RATIOS[ratio]
                return {"family": fam, "variant": variant if fam == "2:17" else None,
                        "evidence": f"R-T name {t}"}
    head, _, _ = _name_formula(n, elements)
    if head and len(head) == 2 and "M" not in (head[0][0], head[1][0]):
        (a, na), (b, nb) = head
        cnt = {a: int(float(na or 1)), b: int(float(nb or 1))}
        re_ = [e for e in cnt if e in RARE_EARTHS]
        if len(re_) == 1:
            tm = b if re_[0] == a else a
            r = _reduce([cnt[re_[0]], cnt[tm]])
            fam = CACU5_RATIOS.get(tuple(r))
            if fam and (tm in CACU5_TM or (fam == "2:17" and tm == "ZN")):
                return {"family": fam, "variant": variant if fam == "2:17" else None,
                        "evidence": f"R-T {fam} by chemistry ({format_formula(head)})"}
    return None


def _assignments(model: Dict, allowed: Sequence[str], cap: int = 20000):
    """Every choice of one constituent per sublattice (elements in ``allowed`` or VA) -> [(choice, {EL: amount})]."""
    import itertools
    ok = {str(e).upper() for e in allowed} | {"VA"}
    subs = [[str(c).upper() for c in cs if str(c).upper() in ok] for cs in model["constituents"]]
    if not subs or any(not s for s in subs):
        return []
    out = []
    for k, ch in enumerate(itertools.product(*subs)):
        if k >= cap:
            break
        amt: Dict[str, float] = {}
        for site, c in zip(model["sites"], ch):
            if c != "VA":
                amt[c] = amt.get(c, 0.0) + float(site)
        if amt:
            out.append((ch, amt))
    return out


def ideal_compositions(model: Dict, allowed: Sequence[str]) -> List[Tuple[tuple, Dict[str, float]]]:
    """Compositions (mole fractions) of the model's end-members within ``allowed`` elements: [(choice, X)]."""
    out = []
    for ch, amt in _assignments(model, allowed):
        tot = sum(amt.values())
        out.append((ch, {e: a / tot for e, a in amt.items()}))
    return out


def _closest(model: Dict, X: Dict[str, float], allowed: Sequence[str]):
    best = None
    for ch, Xi in ideal_compositions(model, allowed):
        d = max(abs(float(X.get(e, 0.0)) - Xi.get(e, 0.0)) for e in set(X) | set(Xi))
        if best is None or d < best[0] - 1e-12:
            best = (d, ch)
    return None if best is None else best[1]


def model_formula(model: Dict, X=None, allowed: Optional[Sequence[str]] = None, X_ends=None,
                  order: Optional[Sequence[str]] = None) -> Optional[List[Tuple[str, int]]]:
    """[(element or '(A,B)', count)] of a phase at composition X (or along a line between X_ends): one constituent
    per sublattice, the end-member closest to X; vacancies dropped, sublattices of one element summed, counts
    reduced.  Ordered by ``order`` (elements), then by sublattice."""
    allowed = list(allowed or sorted({str(c).upper() for cs in model["constituents"] for c in cs} - {"VA"}))
    ends = [X_ends[0], X_ends[-1]] if X_ends else [X]
    choices = []
    for Xe in ends:
        if Xe is None:
            ch = _assignments(model, allowed)
            if len(ch) != 1:
                return None
            choices.append(ch[0][0])
        else:
            c = _closest(model, {str(k).upper(): float(v) for k, v in Xe.items()}, allowed)
            if c is None:
                return None
            choices.append(c)
    amt: Dict[str, float] = {}
    for i, site in enumerate(model["sites"]):
        cs = sorted({ch[i] for ch in choices} - {"VA"})
        if not cs:
            continue
        key = cs[0] if len(cs) == 1 else "(" + ",".join(sorted(c.capitalize() for c in cs)) + ")"
        amt[key] = amt.get(key, 0.0) + float(site)
    if not amt:
        return None
    keys = list(amt)
    if order:
        pos = {str(e).upper(): i for i, e in enumerate(order)}
        keys.sort(key=lambda k: (pos.get(k, len(pos)), keys.index(k)))
    return list(zip(keys, _reduce([amt[k] for k in keys])))


def formula_text(parts: Sequence[Tuple[str, int]]) -> str:
    """'Fe₂Nb', 'Ce₂Co₁₇', '(Co,Ni)₃V' from [(EL or group, count)]."""
    out = []
    for el, n in parts:
        s = el if el.startswith("(") else el.capitalize()
        out.append(s + ("" if int(n) == 1 else str(int(n)).translate(SUB)))
    return "".join(out)


def compound_label(name: str, model: Optional[Dict] = None, X=None, allowed: Optional[Sequence[str]] = None,
                   X_ends=None, elements=None, overrides=None, family: Optional[Dict] = None) -> Tuple[str, Dict]:
    """Own label of a compound: 'formula (family)' (see the module docstring) and {"short", "formula", "family",
    "evidence"}.  ``model`` {"sites", "constituents"}; X {EL: mole fraction} where the phase sits (on an edge of a
    ternary, ``allowed`` = its two elements); X_ends: the two ends of a ternary line compound; ``family``:
    phase_family's result, its "label" set by the caller (2:17R / 2:17H).  Without a model: the formula of the
    name.  An override (phase_labels) is used as written."""
    short = format_phase(name, elements, overrides)
    info = {"short": short, "formula": None, "family": None, "evidence": None}
    if overrides and short != format_phase(name, elements):
        return short, info
    fam = (family if family is not None else phase_family(name, elements)) or None    # (False: no family)
    head, suffix, _ = _name_formula(name, elements)
    if fam and fam["family"] == "2:17" and any(t in R217_TOKENS for t in strip_compset(name).split("_")[1:]):
        suffix = ""                                     # _H / _R name the 2:17 structure here, not a temperature
    named = [e for e, _ in head or []]
    formula = None
    parts = model_formula(model, X, allowed, X_ends) if model else None
    if parts:
        got = [e for e, _ in parts]
        if "M" in named:                                # M3V: M is the one element the name does not give
            extra = [e for e in got if e not in named]
            named = [extra[0] if (e == "M" and len(extra) == 1) else e for e in named]
        if named and set(got) == set(named):            # the name's elements: its order and -HT / -LT
            parts = sorted(parts, key=lambda p: named.index(p[0]))
            formula = formula_text(parts) + suffix
        else:
            formula = formula_text(parts)
    elif head and "M" not in named or (head and model is None):
        formula = format_formula(head) + suffix
    fname = (fam.get("label") or fam["family"] + (fam.get("variant") or "")) if fam else None
    info.update(formula=formula, family=fname, evidence=(fam or {}).get("evidence"))
    if formula and fname:
        return f"{formula} ({fname})", info
    if formula:
        return formula, info
    return (fname if fname and short == strip_compset(name) else short), info


def prettify_legend(ax, elements: Optional[Sequence[str]] = None,
                    overrides: Optional[Dict[str, str]] = None) -> None:
    """Rewrite a phase legend with display names (BCC_A2 -> BCC, SIGMA_D8B -> σ); composition sets
    (BCC_A2#2) share one entry.  Placement and style of the legend are kept."""
    leg = ax.get_legend()
    if leg is None:
        return
    handles, seen = [], {}
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        lab = format_phase(t.get_text(), elements, overrides)
        if lab not in seen:
            seen[lab] = h
            handles.append(h)
    kw = {"loc": leg._loc, "fontsize": leg.get_texts()[0].get_fontsize() if leg.get_texts() else 8,
          "frameon": leg.get_frame_on()}
    if leg._bbox_to_anchor is not None:
        (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(leg.get_bbox_to_anchor().get_points())
        kw["bbox_to_anchor"] = (x0, y0) if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9 else (x0, y0, x1 - x0, y1 - y0)
    title = leg.get_title().get_text()
    if title:
        kw.update(title=title, title_fontsize=leg.get_title().get_fontsize())
    leg = ax.legend(handles, [to_mathtext(lab) for lab in seen], **kw)
    for t, lab in zip(leg.get_texts(), seen):
        if to_mathtext(lab) != lab:
            t._calphad_plain = lab


# ---------------------------------------------------------------------------------------------------------
# subscripts on figures: typeset in PNG / PDF (mathtext), Inkscape-native in SVG
# ---------------------------------------------------------------------------------------------------------
SUB_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
_SUB_RUN = re.compile(f"[{SUB_DIGITS}]+")
_UNSUB = str.maketrans(SUB_DIGITS, "0123456789")
SVG_SUB_STYLE = "font-size:65%;baseline-shift:sub"       # what Inkscape writes for a subscript


_MARK = re.compile(r"_\{([^{}]*)\}")                  # structure suffix markup: L1_{2}, C11_{b}, D0_{19}
_SUB_ANY = re.compile(r"_\{([^{}]*)\}|[" + SUB_DIGITS + "]+")


def plain_label(s: str) -> str:
    """Plain (Unicode) form of a name with subscript markup: 'Ni₃Al (L1_{2})' -> 'Ni₃Al (L1₂)', 'MoSi₂ (C11_{b})' ->
    'MoSi₂ (C11b)' (digits as Unicode subscripts, letters as they are) - what editing tools match."""
    return _MARK.sub(lambda m: m.group(1).translate(SUB) if m.group(1).isdigit() else m.group(1), str(s))


def to_mathtext(s: str) -> str:
    """Display name with Unicode subscript digits (or ``_{...}`` markup) -> matplotlib text with real subscripts:
    'Al₃Ni₂' -> 'Al$_\\mathregular{3}$Ni$_\\mathregular{2}$', 'C11_{b}' -> 'C11$_\\mathregular{b}$' (only the
    subscripts are math, in the text font; the rest - Greek, '+', '-LT', spaces - stays plain text).  A text
    without subscripts is returned unchanged (no mathtext)."""
    s = str(s)
    if not _SUB_ANY.search(s):
        return s
    out, pos = [], 0
    for m in _SUB_ANY.finditer(s):
        out.append(s[pos:m.start()].replace("$", r"\$"))
        body = m.group(1) if m.group(1) is not None else m.group().translate(_UNSUB)
        out.append(r"$_\mathregular{" + body.replace("$", "") + "}$")
        pos = m.end()
    out.append(s[pos:].replace("$", r"\$"))
    return "".join(out)


def set_name(t, s: str):
    """Show a (phase) name on a Text with typeset subscripts; the plain (Unicode) name stays on the text as
    ``_calphad_plain`` (what editing tools match), the subscript markup (``_{...}``, letters too) as
    ``_calphad_markup`` (what the SVG output uses).  Returns the text."""
    m = to_mathtext(s)
    t.set_text(m)
    t._calphad_plain = plain_label(s) if m != s else None
    t._calphad_markup = s if _MARK.search(str(s)) else None
    return t


def plain_text(t) -> str:
    """The name a text shows, in Unicode (Al₃Ni₂), whether drawn as mathtext or not."""
    return getattr(t, "_calphad_plain", None) or t.get_text()


def svg_subscripts(svg: str) -> str:
    """Inkscape-native subscripts in an SVG written with svg.fonttype 'none': in every <text>, each run of Unicode
    subscript digits and each ``_{...}`` markup (letters too: C11_{b}) becomes <tspan
    style="font-size:65%;baseline-shift:sub">24</tspan>.  The <text> keeps its x / y, anchor, transform, font and
    id: one editable text object per label."""
    def fix(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        if "<" in body or not _SUB_ANY.search(body):
            return m.group(0)
        body = _SUB_ANY.sub(lambda r: f'<tspan style="{SVG_SUB_STYLE}">'
                            f'{r.group(1) if r.group(1) is not None else r.group().translate(_UNSUB)}</tspan>', body)
        return head + body + tail
    return re.sub(r"(<text\b[^>]*>)(.*?)(</text>)", fix, svg, flags=re.S)


def save_vector(fig, path, **kw):
    """fig.savefig with text kept as text (svg.fonttype 'none', Type 42 fonts in PDF / EPS).  SVG: names drawn with
    mathtext subscripts are written as one <text> each, with Inkscape subscript tspans (see svg_subscripts);
    PNG / PDF and the figure itself keep the typeset form."""
    import matplotlib
    from matplotlib.text import Text
    rc = {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}
    svg = str(path).lower().endswith(".svg") or str(kw.get("format", "")).lower() == "svg"
    swapped = []
    if svg:
        for t in fig.findobj(Text):
            plain = getattr(t, "_calphad_markup", None) or getattr(t, "_calphad_plain", None)
            if plain:
                swapped.append((t, t.get_text()))
                t.set_text(plain)
    try:
        with matplotlib.rc_context(rc):
            res = fig.savefig(path, **kw)
    finally:
        for t, m in swapped:
            t.set_text(m)
    if svg and swapped and isinstance(path, (str, os.PathLike)):     # (a file-like target is left as written)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg_subscripts(text))
    return res
