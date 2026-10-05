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


def to_mathtext(s: str) -> str:
    """Display name with Unicode subscript digits -> matplotlib text with real subscripts: 'Al₃Ni₂' ->
    'Al$_\\mathregular{3}$Ni$_\\mathregular{2}$' (only the digits are math, in the text font; the rest - Greek,
    '+', '-LT', spaces - stays plain text).  A text without subscript digits is returned unchanged (no mathtext)."""
    s = str(s)
    if not _SUB_RUN.search(s):
        return s
    out, pos = [], 0
    for m in _SUB_RUN.finditer(s):
        out.append(s[pos:m.start()].replace("$", r"\$"))
        out.append(r"$_\mathregular{" + m.group().translate(_UNSUB) + "}$")
        pos = m.end()
    out.append(s[pos:].replace("$", r"\$"))
    return "".join(out)


def set_name(t, s: str):
    """Show a (phase) name on a Text with typeset subscripts; the plain (Unicode) name stays on the text as
    ``_calphad_plain`` (what editing tools and the SVG output use).  Returns the text."""
    m = to_mathtext(s)
    t.set_text(m)
    t._calphad_plain = s if m != s else None
    return t


def plain_text(t) -> str:
    """The name a text shows, in Unicode (Al₃Ni₂), whether drawn as mathtext or not."""
    return getattr(t, "_calphad_plain", None) or t.get_text()


def svg_subscripts(svg: str) -> str:
    """Inkscape-native subscripts in an SVG written with svg.fonttype 'none': in every <text>, each run of Unicode
    subscript digits becomes <tspan style="font-size:65%;baseline-shift:sub">24</tspan>.  The <text> keeps its
    x / y, anchor, transform, font and id: one editable text object per label."""
    def fix(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        if "<" in body or not _SUB_RUN.search(body):
            return m.group(0)
        body = _SUB_RUN.sub(lambda r: f'<tspan style="{SVG_SUB_STYLE}">{r.group().translate(_UNSUB)}</tspan>', body)
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
            plain = getattr(t, "_calphad_plain", None)
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
