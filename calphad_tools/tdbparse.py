"""Text-preserving TDB parser.

pycalphad's own parser converts everything to sympy and throws away the
original text, ordering, type-definition letters, comments, etc.  For
comparing and merging databases we want to keep *exactly* what was written,
so this module parses TDB commands into lightweight dataclasses that can be
written back out, and provides a fast numpy evaluator for the piecewise
expressions (so two parameters can be compared numerically without
pycalphad having to load either database).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------
# Order matters for abbreviation resolution: the first keyword that starts
# with the (normalised) abbreviation wins.
KEYWORDS = [
    "ELEMENT", "SPECIES", "PHASE", "PARAMETER", "CONSTITUENT", "FUNCTION",
    "TYPE_DEFINITION", "DEFINE_SYSTEM_DEFAULT", "DEFAULT_COMMAND",
    "DATABASE_INFO", "VERSION_DATE", "VERSION_DATA", "TEMPERATURE_LIMITS",
    "LIST_OF_REFERENCES", "ADD_REFERENCES", "REFERENCE_FILE", "ASSESSED_SYSTEMS",
    "DIFFUSION", "ZERO_VOLUME_SPECIES",
]


def resolve_keyword(word: str) -> Optional[str]:
    w = word.upper().replace("-", "_")
    if not w:
        return None
    if w in KEYWORDS:
        return w
    # Thermo-Calc allows abbreviating each underscore separated part
    # (e.g. TEMP-LIM, DEF_SYS_DEF, LIST-OF-REF, TYPE-DEF).
    wparts = w.split("_")
    for kw in KEYWORDS:
        kparts = kw.split("_")
        if len(wparts) <= len(kparts) and all(k.startswith(p) for p, k in zip(wparts, kparts)):
            if len(wparts) == 1 and len(w) < 3:
                continue
            return kw
    return None


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*#?)"
    r"|(?P<op>\*\*|[-+*/(),^]))"
)
_MATH_FUNCS = {"LN": "_np.log", "LOG": "_np.log", "EXP": "_np.exp", "SQRT": "_np.sqrt",
               "ABS": "_np.abs"}


class ExpressionError(ValueError):
    pass


def tokenize(expr: str) -> List[Tuple[str, str]]:
    pos, out = 0, []
    s = expr.strip()
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            raise ExpressionError(f"cannot tokenize {s[pos:pos+20]!r} in {expr!r}")
        pos = m.end()
        for kind in ("num", "name", "op"):
            if m.group(kind) is not None:
                out.append((kind, m.group(kind)))
                break
    return out


def referenced_symbols(expr: str) -> List[str]:
    """Function/variable names referenced by an expression (excluding T, P, math funcs)."""
    names = []
    toks = tokenize(expr)
    for i, (kind, val) in enumerate(toks):
        if kind != "name":
            continue
        u = val.upper().rstrip("#")
        nxt = toks[i + 1][1] if i + 1 < len(toks) else None
        if u in _MATH_FUNCS and nxt == "(":
            continue
        if u in ("T", "P"):
            continue
        names.append(u)
    return names


def compile_expr(expr: str):
    """Compile a TDB math expression to a Python code object.

    Identifiers are turned into ``_v("NAME")`` look-ups so the only thing the
    compiled code can do is arithmetic on numbers/arrays we supply.
    """
    parts = []
    toks = tokenize(expr)
    for i, (kind, val) in enumerate(toks):
        if kind == "num":
            parts.append(re.sub(r"[Dd]", "E", val))
        elif kind == "op":
            parts.append("**" if val == "^" else val)
        else:
            u = val.upper().rstrip("#")
            nxt = toks[i + 1][1] if i + 1 < len(toks) else None
            if u in _MATH_FUNCS and nxt == "(":
                parts.append(_MATH_FUNCS[u])
            else:
                parts.append(f'_v("{u}")')
    src = "".join(parts) or "0"
    try:
        return compile(src, "<tdb-expr>", "eval")
    except SyntaxError as e:  # pragma: no cover - malformed TDB
        raise ExpressionError(f"bad expression {expr!r}: {e}") from None


def _fmt_num(x: float) -> str:
    if x == int(x) and abs(x) < 1e6:
        return f"{x:.2f}" if x != 298.15 else "298.15"
    return repr(x)


@dataclass
class Piecewise:
    """A TDB piecewise-in-T expression: [(T_low, T_high, expr), ...]."""
    ranges: List[Tuple[float, float, str]]
    reference: str = ""

    # -- textual helpers ---------------------------------------------------
    @property
    def tmin(self) -> float:
        return self.ranges[0][0] if self.ranges else float("nan")

    @property
    def tmax(self) -> float:
        return self.ranges[-1][1] if self.ranges else float("nan")

    def canonical(self) -> str:
        """Whitespace/case/number-format independent representation."""
        out = []
        for lo, hi, e in self.ranges:
            toks = []
            for kind, val in tokenize(e):
                if kind == "num":
                    toks.append(repr(float(re.sub(r"[Dd]", "E", val))))
                elif kind == "name":
                    toks.append(val.upper().rstrip("#"))
                else:
                    toks.append("**" if val == "^" else val)
            if toks and toks[0] == "+":
                toks = toks[1:]
            out.append(f"{lo:g}<{''.join(toks)}<{hi:g}")
        return ";".join(out)

    def canonical_exprs(self) -> str:
        """Like canonical() but ignoring the temperature limits."""
        return ";".join(c.split("<", 1)[1].rsplit("<", 1)[0] for c in self.canonical().split(";"))

    def short(self, width: int = 70) -> str:
        """Human-friendly single line (first range + number of ranges)."""
        if not self.ranges:
            return ""
        e = self.ranges[0][2]
        if len(self.ranges) > 1:
            e = f"{e}  [+{len(self.ranges) - 1} T-range(s)]"
        return e if len(e) <= width else e[: width - 1] + "…"

    def to_tdb(self) -> str:
        """Body of a FUNCTION/PARAMETER line after the name."""
        segs = []
        for i, (lo, hi, e) in enumerate(self.ranges):
            if i == 0:
                segs.append(f"{_fmt_num(lo)} {e};")
            else:
                segs.append(f"{_fmt_num(lo)} Y {e};")
        tail = f"{_fmt_num(self.tmax)} N"
        if self.reference:
            tail += f" {self.reference}"
        return " ".join(segs) + " " + tail

    def rename_symbols(self, mapping: Dict[str, str]) -> "Piecewise":
        if not mapping:
            return self
        new = []
        for lo, hi, e in self.ranges:
            def sub(m):
                u = m.group(1).upper()
                return mapping.get(u, m.group(0))
            new.append((lo, hi, re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)#?", sub, e)))
        return Piecewise(new, self.reference)

    def symbols(self) -> List[str]:
        s = []
        for _, _, e in self.ranges:
            for n in referenced_symbols(e):
                if n not in s:
                    s.append(n)
        return s

    # -- numerical evaluation ---------------------------------------------
    def evaluate(self, T: np.ndarray, lookup: Callable[[str, np.ndarray], np.ndarray],
                 P: float = 101325.0) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        out = np.full(T.shape, np.nan)
        for lo, hi, e in self.ranges:
            mask = (T >= lo - 1e-9) & (T <= hi + 1e-9) & np.isnan(out)
            if not mask.any():
                continue
            code = compile_expr(e)
            Tm = T[mask]

            def _v(name, _Tm=Tm):
                if name == "T":
                    return _Tm
                if name == "P":
                    return np.full(_Tm.shape, P)
                return lookup(name, _Tm)

            with np.errstate(all="ignore"):
                val = eval(code, {"__builtins__": {}}, {"_v": _v, "_np": np})
            out[mask] = np.broadcast_to(np.asarray(val, dtype=float), Tm.shape)
        return out


def parse_piecewise(text: str, tlow_default: float = 298.15,
                    thigh_default: float = 6000.0) -> Piecewise:
    """Parse ``Tlow expr; Thigh Y expr; Thigh N ref`` (whitespace-insensitive)."""
    segs = [s.strip() for s in text.split(";")]
    if not segs:
        raise ExpressionError("empty expression")

    def _temp(tok: str, default: float) -> float:
        tok = tok.strip()
        if tok == "" or set(tok) == {","}:
            return default
        return float(re.sub(r"[Dd]", "E", tok))

    # first segment: "Tlow expr" or ",, expr"
    first = segs[0]
    m = re.match(r"^\s*(,+|[-+]?[\d.]+(?:[EeDd][+-]?\d+)?)\s*(.*)$", first, re.S)
    if not m:
        raise ExpressionError(f"cannot find low temperature in {first!r}")
    lo = _temp(m.group(1), tlow_default)
    cur_expr = re.sub(r"\s+", "", m.group(2))
    ranges: List[Tuple[float, float, str]] = []
    reference = ""
    for seg in segs[1:]:
        seg = seg.strip()
        if not seg:
            continue
        # "Thigh Y expr" | "Thigh N ref" | ",, N ref" | ",,N ref"
        m = re.match(r"^(,+|[-+]?[\d.]+(?:[EeDd][+-]?\d+)?)?\s*([YN])\b\s*(.*)$", seg, re.S | re.I)
        if not m:
            # "6000.00 REF" (terminating N omitted) - seen in the wild
            m2 = re.match(r"^(,+|[-+]?[\d.]+(?:[EeDd][+-]?\d+)?)\s*(.*)$", seg, re.S)
            if not m2:
                raise ExpressionError(f"cannot parse range segment {seg!r}")
            ranges.append((lo, _temp(m2.group(1), thigh_default), cur_expr or "0"))
            reference, cur_expr = m2.group(2).strip(), None
            break
        hi = _temp(m.group(1) or ",,", thigh_default)
        ranges.append((lo, hi, cur_expr or "0"))
        if m.group(2).upper() == "Y":
            lo = hi
            cur_expr = re.sub(r"\s+", "", m.group(3))
        else:
            reference = m.group(3).strip()
            cur_expr = None
            break
    if cur_expr is not None:  # no terminating N (malformed but seen in the wild)
        ranges.append((lo, thigh_default, cur_expr or "0"))
    return Piecewise(ranges, reference)


# ---------------------------------------------------------------------------
# Database objects
# ---------------------------------------------------------------------------
@dataclass
class Element:
    name: str
    ref_phase: str
    mass: float
    h298: float
    s298: float

    def to_tdb(self) -> str:
        return (f" ELEMENT {self.name:<3s} {self.ref_phase:<24s} {self.mass:.6E} "
                f"{self.h298:.6E} {self.s298:.6E}!")


@dataclass
class Species:
    name: str
    formula: str

    def to_tdb(self) -> str:
        return f" SPECIES {self.name:<24s} {self.formula}!"


@dataclass
class Function:
    name: str
    expr: Piecewise

    def to_tdb(self) -> str:
        return f" FUNCTION {self.name:<10s} {self.expr.to_tdb()} !"


@dataclass
class TypeDef:
    code: str
    text: str            # everything after the code, whitespace-normalised

    @property
    def kind(self) -> str:
        t = self.text.upper()
        if t.startswith("SEQ"):
            return "SEQ"
        if "MAGNETIC" in t:
            return "MAGNETIC"
        if "DIS_PART" in t or "DIS_P" in t:
            return "DIS_PART"
        return "OTHER"

    @property
    def target_phase(self) -> Optional[str]:
        # "GES A_P_D <phase|@> ..."
        m = re.match(r"GES\S*\s+(?:A_P_D|AMEND_PHASE_DESCRIPTION|AM_PH_DES|A_P_DES)\s+(\S+)",
                     self.text, re.I)
        return m.group(1).upper() if m else None

    def canonical(self, phase: Optional[str] = None) -> str:
        t = re.sub(r"\s+", " ", self.text.upper())
        t = re.sub(r"AMEND_PHASE_DESCRIPTION|AM_PH_DES|A_P_DES", "A_P_D", t)
        if phase is not None:
            tp = self.target_phase
            if tp and (tp == "@" or tp == phase):
                t = t.replace(f" {tp} ", " <PHASE> ", 1)

        # normalise numbers (-1 vs -1.0, 0.4 vs 4.00000E-01)
        def num(m):
            try:
                return repr(float(m.group(0)))
            except ValueError:
                return m.group(0)
        return re.sub(r"(?<![A-Z_])[-+]?\d*\.?\d+(?:E[-+]?\d+)?", num, t)

    def to_tdb(self) -> str:
        return f" TYPE_DEFINITION {self.code} {self.text} !"


@dataclass
class Phase:
    name: str                      # without :G/:L suffix
    suffix: str                    # e.g. ":L" or ""
    type_codes: str                # e.g. "%&" (without % if absent)
    site_ratios: List[float]
    constituents: List[List[str]] = field(default_factory=list)
    majors: List[List[str]] = field(default_factory=list)   # species marked with %

    @property
    def n_sublattices(self) -> int:
        return len(self.site_ratios)

    def model_signature(self) -> Tuple:
        return (self.n_sublattices, tuple(round(r, 6) for r in self.site_ratios))

    def va_only_sublattices(self) -> List[int]:
        return [i for i, sl in enumerate(self.constituents) if sl == ["VA"]]

    def reduced_signature(self) -> Tuple:
        """Model signature ignoring sublattices that contain only vacancies.

        Such a sublattice contributes nothing to G (no configurational entropy,
        G(VA)=0), so e.g. BCC_A2 (CR,FE)1(VA)3 == BCC_A2 (CR,FE)1.
        """
        drop = set(self.va_only_sublattices())
        r = [x for i, x in enumerate(self.site_ratios) if i not in drop]
        return (len(r), tuple(round(x, 6) for x in r))

    def to_tdb(self) -> List[str]:
        ratios = " ".join(f"{r:g}" for r in self.site_ratios)
        lines = [f" PHASE {self.name}{self.suffix} {self.type_codes or '%'} "
                 f"{self.n_sublattices} {ratios} !"]
        if self.constituents:
            subl = []
            for i, sl in enumerate(self.constituents):
                majors = set(self.majors[i]) if i < len(self.majors) else set()
                subl.append(",".join(s + ("%" if s in majors else "") for s in sl))
            lines.append(f"    CONSTITUENT {self.name}{self.suffix} :" + " : ".join(subl) + " :  !")
        return lines


@dataclass
class ParamKey:
    ptype: str                          # G, L, TC, BMAGN, MQ, ...
    phase: str
    constituents: Tuple[Tuple[str, ...], ...]
    order: int
    diffusing: str = ""                  # for mobility params: MQ(PHASE&CR,...)

    def label(self) -> str:
        c = ":".join(",".join(sl) for sl in self.constituents)
        d = f"&{self.diffusing}" if self.diffusing else ""
        pt = self.ptype
        if pt == "G" and any(len(sl) > 1 for sl in self.constituents):
            pt = "L"   # G and L are synonyms; show the conventional one
        return f"{pt}({self.phase}{d},{c};{self.order})"

    def without_sublattices(self, idxs: Sequence[int]) -> "ParamKey":
        cons = tuple(sl for i, sl in enumerate(self.constituents) if i not in set(idxs))
        return ParamKey(self.ptype, self.phase, cons, self.order, self.diffusing)

    def as_tuple(self) -> Tuple:
        return (self.ptype, self.phase, self.constituents, self.order, self.diffusing)

    def sort_key(self):
        return (self.phase, len([s for sl in self.constituents for s in sl]),
                self.constituents, self.ptype, self.order)


# Types of parameter which are "the same thing" under different spellings.
PTYPE_ALIASES = {"BM": "BMAGN", "BMAG": "BMAGN", "BMAGN": "BMAGN", "L": "G"}


@dataclass
class Parameter:
    raw_name: str            # exactly as written, e.g. "G(BCC_A2,CR,FE:VA;0)"
    key: ParamKey            # normalised (sorted) key
    sign: float              # multiply raw expression by this to get normalised value
    order_map: Optional[Dict[int, int]]  # ternary order remapping (raw->normalised), if any
    expr: Piecewise

    def to_tdb(self) -> str:
        return f" PARAMETER {self.raw_name:<30s} {self.expr.to_tdb()} !"


def parse_param_name(raw: str) -> Tuple[ParamKey, float, Optional[Dict[int, int]]]:
    """Parse ``G(BCC_A2,CR,FE:VA;1)`` into a normalised key.

    Returns (key, sign, order_map).  If constituents in an interacting
    sublattice had to be re-ordered to put them in alphabetical order, odd
    Redlich-Kister orders change sign (L(A,B;1) == -L(B,A;1)).
    """
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*$", raw, re.S)
    if not m:
        raise ExpressionError(f"bad parameter name {raw!r}")
    ptype = m.group(1).upper()
    ptype = PTYPE_ALIASES.get(ptype, ptype)
    inner = re.sub(r"\s+", "", m.group(2)).upper()
    order = 0
    if ";" in inner:
        inner, o = inner.rsplit(";", 1)
        order = int(o) if o else 0
    phase, _, rest = inner.partition(",")
    diffusing = ""
    if "&" in phase:
        phase, diffusing = phase.split("&", 1)
    phase = phase.split(":")[0]
    subls = [tuple(s.strip("%#") for s in sl.split(",") if s) for sl in rest.split(":")]
    sign = 1.0
    order_map = None
    interacting = [i for i, sl in enumerate(subls) if len(sl) > 1]
    norm = []
    for i, sl in enumerate(subls):
        srt = tuple(sorted(sl))
        if srt != sl and len(interacting) == 1:
            if len(sl) == 2 and order % 2 == 1:
                sign = -1.0
            elif len(sl) == 3:
                # ternary L(A,B,C;k): k refers to the k-th listed species
                new_order = srt.index(sl[order]) if order < 3 else order
                order_map = {order: new_order}
        norm.append(srt)
    if order_map:
        order = list(order_map.values())[0]
    return ParamKey(ptype, phase, tuple(norm), order, diffusing), sign, order_map


@dataclass
class TDB:
    path: str
    elements: Dict[str, Element] = field(default_factory=dict)
    species: Dict[str, Species] = field(default_factory=dict)
    functions: Dict[str, Function] = field(default_factory=dict)
    typedefs: Dict[str, TypeDef] = field(default_factory=dict)
    phases: Dict[str, Phase] = field(default_factory=dict)
    parameters: List[Parameter] = field(default_factory=list)
    other_commands: List[str] = field(default_factory=list)  # DEFINE_SYSTEM_DEFAULT etc.
    references: List[str] = field(default_factory=list)
    header_comments: List[str] = field(default_factory=list)
    temp_limits: Tuple[float, float] = (298.15, 6000.0)
    warnings: List[str] = field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def real_elements(self) -> List[str]:
        return sorted(e for e in self.elements if e not in ("VA", "/-"))

    def param_dict(self) -> Dict[Tuple, Parameter]:
        """Normalised key -> parameter (last definition wins, like Thermo-Calc)."""
        d = {}
        for p in self.parameters:
            d[_key_tuple(p.key)] = p
        return d

    def phase_typedefs(self, phase: Phase) -> List[TypeDef]:
        return [self.typedefs[c] for c in phase.type_codes if c in self.typedefs]

    # -- numerical evaluation ---------------------------------------------
    def evaluator(self, P: float = 101325.0):
        cache: Dict[Tuple[str, bytes], np.ndarray] = {}
        stack: List[str] = []

        def lookup(name: str, T: np.ndarray) -> np.ndarray:
            key = (name, T.tobytes())
            if key in cache:
                return cache[key]
            if name in stack:
                raise ExpressionError(f"circular function reference: {' -> '.join(stack + [name])}")
            f = self.functions.get(name)
            if f is None:
                if name == "R":
                    return np.full(T.shape, 8.31451)
                raise ExpressionError(f"undefined function {name} in {self.name}")
            stack.append(name)
            try:
                val = f.expr.evaluate(T, lookup, P)
            finally:
                stack.pop()
            cache[key] = val
            return val
        return lookup

    def evaluate(self, expr: Piecewise, T: Sequence[float]) -> np.ndarray:
        return expr.evaluate(np.asarray(T, float), self.evaluator())

    def evaluate_function(self, name: str, T: Sequence[float]) -> np.ndarray:
        return self.evaluator()(name.upper(), np.asarray(T, float))


def _key_tuple(k: ParamKey) -> Tuple:
    return (k.ptype, k.phase, k.constituents, k.order, k.diffusing)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _split_commands(text: str) -> Tuple[List[str], List[str]]:
    """Strip $-comments and split the file into '!'-terminated commands."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept, header, seen_command = [], [], False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("$"):
            if not seen_command:
                header.append(stripped)
            continue
        if "$" in ln:
            # inline comment - but only outside quotes (references/db info)
            if ln.count("'") % 2 == 0 and "'" not in ln.split("$", 1)[0]:
                ln = ln.split("$", 1)[0]
        if stripped:
            seen_command = True
        kept.append(ln)
    body = "\n".join(kept)
    cmds = [c.strip() for c in body.split("!")]
    return [c for c in cmds if c], header


def parse_tdb(path_or_text: str, *, is_text: bool = False, name: str = "") -> TDB:
    if is_text:
        text = path_or_text
        path = name or "<string>"
    else:
        path = str(path_or_text)
        if Path(path).suffix.lower() == ".dat":
            raise ValueError(f"{Path(path).name} is a ChemSage/FactSage .dat database; this tool reads TDB text. "
                             "Convert it first with convert_database (calculation tools accept .dat directly).")
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    db = TDB(path=path)
    commands, db.header_comments = _split_commands(text)
    # first pass: temperature limits (needed for ,, defaults)
    for cmd in commands:
        words = cmd.split()
        if words and resolve_keyword(words[0]) == "TEMPERATURE_LIMITS" and len(words) >= 3:
            try:
                db.temp_limits = (float(words[1]), float(words[2]))
            except ValueError:
                pass
    tlo, thi = db.temp_limits
    pending_constituents: Dict[str, str] = {}
    for cmd in commands:
        words = cmd.split()
        kw = resolve_keyword(words[0])
        try:
            if kw == "ELEMENT":
                nm = words[1].upper()
                nums = [float(w) for w in words[3:6]] + [0.0] * 3
                db.elements[nm] = Element(nm, words[2].upper() if len(words) > 2 else "",
                                          *nums[:3])
            elif kw == "SPECIES":
                db.species[words[1].upper()] = Species(words[1].upper(), "".join(words[2:]))
            elif kw == "FUNCTION":
                nm = words[1].upper().rstrip("#")
                body = cmd.split(None, 2)[2] if len(words) > 2 else ""
                db.functions[nm] = Function(nm, parse_piecewise(body, tlo, thi))
            elif kw == "TYPE_DEFINITION":
                code = words[1]
                db.typedefs[code] = TypeDef(code, " ".join(words[2:]))
            elif kw == "PHASE":
                full = words[1].upper()
                nm, _, suf = full.partition(":")
                codes = words[2]
                n = int(float(words[3]))
                ratios = [float(w) for w in words[4:4 + n]]
                db.phases[nm] = Phase(nm, (":" + suf) if suf else "", codes, ratios)
            elif kw == "CONSTITUENT":
                nm = words[1].upper().split(":")[0]
                rest = cmd.split(None, 2)[2] if len(words) > 2 else ""
                pending_constituents[nm] = rest
            elif kw == "PARAMETER":
                body = cmd.split(None, 1)[1]
                # name runs up to the first ')' (names can contain spaces)
                close = body.index(")")
                raw_name = re.sub(r"\s+", "", body[: close + 1])
                rest = body[close + 1:]
                key, sign, omap = parse_param_name(raw_name)
                db.parameters.append(Parameter(raw_name, key, sign, omap,
                                               parse_piecewise(rest, tlo, thi)))
            elif kw in ("LIST_OF_REFERENCES", "ADD_REFERENCES"):
                db.references.append(cmd)
            elif kw is None:
                db.warnings.append(f"unrecognised command: {cmd[:60]!r}")
            else:
                db.other_commands.append(re.sub(r"\s+", " ", cmd))
        except (ValueError, IndexError, ExpressionError) as e:
            db.warnings.append(f"could not parse {cmd[:60]!r}: {e}")
    for nm, rest in pending_constituents.items():
        ph = db.phases.get(nm)
        if ph is None:
            db.warnings.append(f"CONSTITUENT for undefined phase {nm}")
            continue
        rest = rest.strip()
        if rest.startswith(":"):
            rest = rest[1:]
        if rest.endswith(":"):
            rest = rest[:-1]
        subls = rest.split(":")
        ph.constituents, ph.majors = [], []
        for sl in subls:
            sp = [s for s in re.split(r"[\s,]+", sl.strip()) if s]
            ph.constituents.append([s.upper().rstrip("%") for s in sp])
            ph.majors.append([s.upper().rstrip("%") for s in sp if s.endswith("%")])
    return db


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def wrap_tdb_line(line: str, width: int = 78) -> str:
    """Wrap a long TDB command for Thermo-Calc's 80-column limit.

    Breaks at whitespace or before a +/- operator (never inside a number).
    """
    if len(line) <= width:
        return line
    out, cur = [], line
    while len(cur) > width:
        cut = -1
        for i in range(width, 20, -1):
            ch = cur[i]
            if ch == " ":
                cut = i
                break
            if ch in "+-" and cur[i - 1] not in "EeDd(*/^" and not cur[i - 1].isspace():
                cut = i
                break
        if cut <= 0:
            break
        out.append(cur[:cut].rstrip())
        cur = "      " + cur[cut:].lstrip()
    out.append(cur)
    return "\n".join(out)


def iter_constituent_species(db: TDB) -> Iterable[str]:
    for ph in db.phases.values():
        for sl in ph.constituents:
            yield from sl


def evaluation_grid(t_lo: float = 298.15, t_hi: float = 3000.0, n: int = 60) -> np.ndarray:
    return np.linspace(t_lo, t_hi, n)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return math.nan
    return float(np.max(np.abs(a[mask] - b[mask])))
