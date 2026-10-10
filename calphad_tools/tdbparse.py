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
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
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


def fmt_ratio(r: float) -> str:
    """A site ratio as written in a PHASE line: up to 15 significant digits, so a value parsed from text (or set from
    an exact decimal such as 0.105263157894737) is written back unchanged ('{:g}' would cut it to 6 digits)."""
    return f"{float(r):.15g}"


# ---------------------------------------------------------------------------
# Site ratios written as rounded decimals (0.333:0.667 for 1/3:2/3)
# ---------------------------------------------------------------------------
SNAP_RTOL = 2e-3          # a ratio matches a simple fraction within this relative error (or is it, rounded) ...
SNAP_MAX_DEN = 30         # ... and the smallest integer formula has at most this many sites in total


def sum_one_decimals(fractions: Sequence[Fraction]) -> List[float]:
    """Exact fractions summing to 1 as 15-significant-digit decimals whose decimal sum is exactly 1 and whose float sum
    (what pycalphad adds) is 1 to 1e-15: 1/3, 2/3 -> 0.333333333333333 / 0.666666666666667."""
    with localcontext() as ctx:
        ctx.prec = 15
        dec = [+(Decimal(x.numerator) / Decimal(x.denominator)) for x in fractions]
    big = max(range(len(dec)), key=lambda i: dec[i])
    dec[big] = Decimal(1) - sum(d for i, d in enumerate(dec) if i != big)
    out = [float(str(d)) for d in dec]
    if abs(sum(out) - 1.0) > 1e-15:          # nudge the largest by one unit in the last place
        for step in (1, -1, 2, -2):
            trial = list(out)
            trial[big] = float(str(dec[big] + Decimal(step) * Decimal(10) ** (dec[big].adjusted() - 14)))
            if abs(sum(trial) - 1.0) <= 1e-15:
                return trial
    return out


@dataclass
class RatioSnap:
    """Site ratios read as exact fractions (see snap_ratios)."""
    ratios: List[float]                       # as given
    integers: List[int]                       # the smallest integer formula (0.333:0.667 -> 1:2)
    fractions: Optional[List[Fraction]]       # exact values (None when the total is not a simple number)
    values: List[float]                       # what to write: exact decimals (summing to exactly 1 if the total is 1)
    changed: bool                             # values differ from the given ratios

    def normalised(self) -> List[Fraction]:
        n = sum(self.integers)
        return [Fraction(i, n) for i in self.integers]

    def text(self) -> str:
        if self.fractions is None:
            return ":".join(fmt_ratio(v) for v in self.values)
        return ":".join(str(f) for f in self.fractions)

    def given_text(self) -> str:
        return ":".join(fmt_ratio(r) for r in self.ratios)


def _ratio_digits(r: float) -> Optional[Tuple[int, int]]:
    """(significant digits, decimals) of ``r`` as written (fmt_ratio), None for exponent notation."""
    s = fmt_ratio(r)
    if "e" in s.lower():
        return None
    return len(s.lstrip("-").replace(".", "").lstrip("0")), (len(s.split(".", 1)[1]) if "." in s else 0)


def _ratio_matches(r: float, t, rtol: float) -> bool:
    """``r`` is ``t`` within ``rtol`` (relative), or ``t`` rounded to the decimals ``r`` is written with (at least 3
    significant digits: 0.105 is 2/19 = 0.10526 rounded; 0.41 is too short to stand for 9/22)."""
    tf = float(t)
    if abs(r - tf) <= rtol * abs(tf):
        return True
    dig = _ratio_digits(r)
    if dig is None or dig[0] < 3 or dig[1] == 0:
        return False
    q = Decimal(1).scaleb(-dig[1])
    return Decimal(repr(tf)).quantize(q, rounding=ROUND_HALF_UP) == Decimal(fmt_ratio(r)).quantize(q)


def snap_ratios(ratios: Sequence[float], rtol: float = SNAP_RTOL, max_den: int = SNAP_MAX_DEN) -> Optional[RatioSnap]:
    """Site ratios read as the simple fractions they are a rounded form of: 0.333:0.667 -> 1/3:2/3, 0.105:0.895 ->
    2/19:17/19, 0.538:0.154:0.308 -> 7/13:2/13:4/13; integer ratios (1:2, 7:2:4) are their own exact form.  The
    smallest integer formula n_1:...:n_k with n_1+...+n_k <= ``max_den`` whose proportions match every ratio (within
    ``rtol`` relative, or as its rounded decimal) is used; the total keeps its value (1, 3, 13 ...) when that is a
    simple number too.  None when there is no such formula (0.41:0.59 stays as written; 0.4:0.6 is 2:3 exactly)."""
    r = [float(x) for x in ratios]
    if not r or any(not math.isfinite(x) or x <= 0 for x in r):
        return None
    S = sum(r)
    if all(x == int(x) and x < 1e6 for x in r):              # integer formulas are exact as written (24:11)
        g = 0
        for x in r:
            g = math.gcd(g, int(x))
        return RatioSnap(r, [int(x) // g for x in r], [Fraction(int(x)) for x in r], list(r), False)
    fs = Fraction(S).limit_denominator(max_den)
    S_exact = fs if fs > 0 and abs(float(fs) - S) <= rtol * S else None
    for D in range(1, max_den + 1):
        n = [int(round(x / S * D)) for x in r]
        if sum(n) != D or min(n) < 1:
            continue
        if S_exact is not None:
            target = [S_exact * Fraction(i, D) for i in n]
        else:
            target = [S * i / D for i in n]
        if not all(_ratio_matches(x, t, rtol) for x, t in zip(r, target)):
            continue
        if S_exact is None:
            values = [float(t) for t in target]
            fr = None
        elif S_exact == 1:
            fr = list(target)
            values = sum_one_decimals(fr)
        else:
            fr = list(target)
            values = [float(t) for t in fr]
        changed = any(abs(v - x) > 1e-12 * max(r) for v, x in zip(values, r))
        return RatioSnap(r, n, fr, values, changed)
    return None


def ratios_proportional(ra: Sequence[float], rb: Sequence[float], rtol: float = 1e-9) -> bool:
    """``rb`` = k * ``ra`` (same order) to ``rtol`` (relative to the largest)."""
    if len(ra) != len(rb) or not ra:
        return False
    sa, sb = sum(ra), sum(rb)
    xa, xb = [x / sa for x in ra], [x / sb for x in rb]
    return all(abs(x - y) <= rtol * max(xa) for x, y in zip(xa, xb))


def snapped_proportional(ra: Sequence[float], rb: Sequence[float], rtol: float = SNAP_RTOL) -> bool:
    """``ra`` and ``rb`` (same order) proportional once read as exact fractions (snap_ratios)."""
    if len(ra) != len(rb) or not ra:
        return False
    sa, sb = snap_ratios(ra, rtol), snap_ratios(rb, rtol)
    if sa is None or sb is None:
        return False
    return sa.normalised() == sb.normalised()


def _is_number(e: str) -> bool:
    return re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?", e.strip()) is not None


def scale_expr(e: str, k: float) -> str:
    """One T-range expression multiplied by ``k`` as valid TDB text.  Simplified where possible: ``k`` is
    distributed into the terms (``(GHSERWW+2*GHSERFE-30000+6*T)/3`` x 3 -> ``GHSERWW+2*GHSERFE-30000+6*T``), as the
    VV#### substitution does, checked numerically against the exact product; otherwise ``k*(e)`` (``(e)/n`` when
    k = 1/n, ``-(e)`` for k = -1, a plain number when ``e`` is one)."""
    k = float(k)
    if k == 1.0:
        return e
    simple = None if _is_number(e) else _scale_terms(e, k)      # a plain number stays a number (below)
    if simple is not None:
        return simple
    if _is_number(e):
        x = float(re.sub(r"[Dd]", "E", e))
        n = 1.0 / k if k != 0.0 else 0.0
        if k != 0.0 and abs(k) < 1.0 and abs(n - round(n)) <= 1e-12 * abs(n):
            return repr(x / round(n))                  # exact division (x/3, not x*0.333...)
        return repr(x * k)
    if k == -1.0:
        return f"-({e})"
    if k != 0.0 and abs(k) < 1.0:
        n = 1.0 / k
        if abs(n - round(n)) <= 1e-12 * abs(n):
            return f"({e})/{int(round(n))}" if n > 0 else f"-({e})/{int(round(-n))}"
    if k == int(k) and abs(k) < 1e15:
        return f"{int(k)}*({e})"
    return f"{k!r}*({e})"


def _short_num(x: float):
    """``x`` as a short decimal (<= 12 significant digits) if that is exactly ``x``, else None."""
    t = f"{x:.12g}".upper()
    return t if float(t) == x else None


def _scale_terms(e: str, k: float):
    """``e`` x ``k`` with ``k`` folded into each term's numeric coefficient (a coefficient c/n that has no short
    decimal stays written as c.../n), or None when ``e`` does not parse or the result does not evaluate to exactly
    (rel 1e-12) the product at random values of T and the names."""
    try:
        from .tdbtables import _parse, _fmt_sum
        terms = _parse(e)
    except Exception:  # noqa: BLE001 - unparsable: the safe k*(e) form
        return None
    n_inv = 1.0 / k
    div = int(round(n_inv)) if abs(k) < 1.0 and abs(n_inv - round(n_inv)) <= 1e-12 * abs(n_inv) else None

    def num(f) -> float:
        return float(re.sub(r"[Dd]", "E", f[1]))

    def split(terms):
        """[(coefficient, other factors)], with a lone (sum) term - optionally times / over numbers - opened."""
        if len(terms) == 1:
            sign, fac = terms[0]
            groups = [f for op, f in fac if op == "*" and f[0] == "group"]
            nums = [(op, f) for op, f in fac if f[0] == "num"]
            if len(groups) == 1 and len(nums) == len(fac) - 1:
                c = float(sign)
                for op, f in nums:
                    c = c * num(f) if op == "*" else c / num(f)
                return [(c * cc, rest) for cc, rest in split(groups[0][1])]
        out = []
        for sign, fac in terms:
            c, rest = float(sign), []
            for op, f in fac:
                if f[0] == "num":
                    c = c * num(f) if op == "*" else c / num(f)
                else:
                    rest.append((op, f))
            out.append((c, rest))
        return out
    out = []
    for c, rest in split(terms):
        if c == 0.0:
            continue
        v = c * k
        tail = []
        if div is not None and _short_num(abs(v)) is None:
            mag, tail = _short_num(abs(c)) or repr(abs(c)), [("/", ("num", str(div)))]
        else:
            mag = _short_num(abs(v)) or repr(abs(v))
        if rest and rest[0][0] == "/":
            rest = [("*", ("num", "1"))] + rest
        body = ([] if (mag == "1" and rest) else [("*", ("num", mag))]) + rest + tail
        if body and body[0][0] == "/":
            body = [("*", ("num", "1"))] + body
        out.append([1 if v > 0 else -1, body])
    if not out:
        return "0"
    text = _fmt_sum(out, False, False)
    try:
        ok = _same_value(e, text, k)
    except Exception:  # noqa: BLE001
        ok = False
    return text if ok else None

def _same_value(old: str, new: str, k: float, n: int = 6) -> bool:
    """``new`` == ``k`` x ``old`` at random positive values of T and every name (rel 1e-12)."""
    import math as _m
    import random as _r
    from .tdbtables import _parse
    to, tn = _parse(old), _parse(new)
    rng = _r.Random(1)
    names = {}

    def node(x, env):
        kd = x[0]
        if kd == "num":
            return float(re.sub(r"[Dd]", "E", x[1]))
        if kd == "name":
            return env.setdefault(x[1], rng.uniform(0.5, 3.0))
        if kd == "group":
            return tsum(x[1], env)
        if kd == "neg":
            return -node(x[1], env)
        if kd == "pow":
            return node(x[1], env) ** node(x[2], env)
        if kd == "call":
            a = [tsum(arg, env) for arg in x[2]]
            fn = x[1].upper()
            if fn in ("LN", "LOG"):
                return _m.log(a[0])
            if fn == "EXP":
                return _m.exp(a[0])
            return env.setdefault(f"{fn}()", rng.uniform(0.5, 3.0))
        raise ValueError(kd)

    def tsum(terms, env):
        tot = 0.0
        for sign, fac in terms:
            v = 1.0
            for op, f in fac:
                x = node(f, env)
                v = v * x if op == "*" else v / x
            tot += sign * v
        return tot
    for _ in range(n):
        env = dict(names)
        env["T"] = rng.uniform(300.0, 3000.0)
        a, b = k * tsum(to, env), tsum(tn, env)
        if not _m.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9):
            return False
        names = {key: v for key, v in env.items() if key != "T"}
    return True


def scale_piecewise(pw: "Piecewise", k: float) -> "Piecewise":
    """``pw`` times ``k``, scaling EACH temperature range's expression separately (never the piecewise object as a
    whole), so the result is a valid TDB piecewise with the same breakpoints.  The one helper every rescale of a
    parameter goes through (tdbstage normalisation, harmonize)."""
    if float(k) == 1.0:
        return Piecewise(list(pw.ranges), pw.reference)
    return Piecewise([(lo, hi, scale_expr(e, k)) for lo, hi, e in pw.ranges], pw.reference)


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
        ratios = " ".join(fmt_ratio(r) for r in self.site_ratios)
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
    # reference key as written after the last range ("... 6000 N 91Din !" -> "91Din"); when the command has none,
    # a short comment right after its "!" ("... N ! $1967Kau") is used.  "" if there is neither.
    reference: str = ""

    def __post_init__(self):
        if not self.reference:
            self.reference = " ".join(str(self.expr.reference or "").split())

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
    # reference key (upper case) -> full citation, from LIST_OF_REFERENCES / ADD_REFERENCES or a $-comment list
    reference_list: Dict[str, str] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def name(self) -> str:
        return Path(self.path).name

    def citation(self, ref: str) -> str:
        """Full citation of a parameter's reference key ("" if the database's reference list does not have it)."""
        ref = " ".join((ref or "").split())
        if not ref:
            return ""
        hit = self.reference_list.get(ref.upper())
        if hit is None and " " in ref:
            hit = self.reference_list.get(ref.split()[0].upper())
        return hit or ""

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
    cmds, header, _, _ = _split_commands_ex(text)
    return cmds, header


def _split_commands_ex(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """(commands, header comments, comment right after each command's '!' on the same line, comment lines).

    In the comment lines, "$" (an empty comment) marks where commands interrupted a run of comment lines."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept, header, comments, seen_command = [], [], [], False
    trailing: Dict[int, str] = {}         # number of '!' up to the comment -> comment text
    n_bang = 0
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("$"):
            comments.append(stripped)
            if not seen_command:
                header.append(stripped)
            continue
        if "$" in ln:
            # inline comment - but only outside quotes (references/db info)
            if ln.count("'") % 2 == 0 and "'" not in ln.split("$", 1)[0]:
                code, com = ln.split("$", 1)
                if "!" in code and not code.rsplit("!", 1)[1].strip():
                    trailing[n_bang + code.count("!")] = com.strip()
                ln = code
        if stripped:
            seen_command = True
            if comments and comments[-1] != "$":
                comments.append("$")
        n_bang += ln.count("!")
        kept.append(ln)
    body = "\n".join(kept)
    cmds, trail = [], []
    for i, c in enumerate(body.split("!")):
        c = c.strip()
        if c:
            cmds.append(c)
            trail.append(trailing.get(i + 1, ""))
    return cmds, header, trail, comments


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
_REF_ENTRY_RE = re.compile(r"([^\s']+)((?:[ \t]+[^\s']+)*?)\s*'([^']*)'?", re.S)


def parse_reference_command(cmd: str) -> Dict[str, str]:
    """``LIST_OF_REFERENCES`` / ``ADD_REFERENCES`` body -> {KEY: citation}.

    Thermo-Calc style: an optional ``NUMBER SOURCE`` heading, then ``KEY 'citation'`` (the quoted text may run over
    several lines; ``KEY ALIAS 'citation'`` defines both).  Without quotes, each line is ``KEY citation``."""
    parts = cmd.split(None, 1)
    body = parts[1] if len(parts) > 1 else ""
    body = re.sub(r"^\s*NUMBER\s+SOURCE\b", "", body, flags=re.I)
    out: Dict[str, str] = {}
    if "'" in body:
        for m in _REF_ENTRY_RE.finditer(body):
            text = " ".join(m.group(3).split())
            for k in [m.group(1)] + m.group(2).split():
                out[k.upper()] = text
    else:
        for ln in body.split("\n"):
            w = ln.split(None, 1)
            if len(w) == 2:
                out[w[0].upper()] = " ".join(w[1].split())
    return out


# Tentative parameters (calphad_tools.tentative): temporary interaction terms a TDB combination adds where no source
# assesses a new mixing; recognised by these reference keys, or by a citation that starts with "Tentative".
TENT_MIX = "TENT_MIX"            # rule 1: new mixing of two constituents on one sublattice
TENT_VA = "TENT_VA"              # rule 2: vacancies mixing with an element on a substitutional sublattice
TENTATIVE_KEYS = (TENT_MIX, TENT_VA)


def _ref_of(param) -> str:
    if isinstance(param, str):
        return " ".join(param.split())
    ref = getattr(param, "reference", None)
    if not ref:
        ref = getattr(getattr(param, "expr", param), "reference", "") or ""
    return " ".join(str(ref).split())


def is_tentative(param, tdb: Optional["TDB"] = None, citation: Optional[str] = None) -> bool:
    """True if ``param`` (a Parameter, a merge plan's parameter, a Piecewise or a reference key) is tentative - not
    assessed: its reference key is TENT_MIX / TENT_VA, or its citation (``citation``, else from ``tdb``'s reference
    list) starts with "Tentative", so hand-written tentative terms count too."""
    ref = _ref_of(param)
    if not ref:
        return False
    if ref.split()[0].upper() in TENTATIVE_KEYS:
        return True
    cit = citation if citation is not None else (tdb.citation(ref) if tdb is not None else "")
    return bool(cit) and cit.strip().lower().startswith("tentative")


def calc_text(text: str) -> str:
    """TDB text without what cannot change a calculation - comment lines and LIST_OF_REFERENCES / ADD_REFERENCES
    commands - for content-hash caches (a reference list that carries a date must not invalidate them)."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("$")]
    body = "\n".join(lines)
    return re.sub(r"(?im)^\s*(LIST[_-]OF[_-]REF\w*|ADD[_-]REF\w*)\b[^!]*!", "", body)


def comment_references(comments: Sequence[str], keys: Iterable[str]) -> Dict[str, str]:
    """Citations for ``keys`` found in $-comment lines: ``$ [KEY] citation`` or ``$ KEY: citation`` /
    ``$ KEY citation`` (the unbracketed forms only for keys with a digit, e.g. REF283, 91Din).  The citation may
    continue on the following comment lines until a blank line, a separator or the next key."""
    want = {k.upper(): k for k in keys if k and len(k) <= 30}
    if not want:
        return {}
    lines = [c.lstrip("$").strip() for c in comments]

    def key_at(ln: str) -> Optional[Tuple[str, str]]:
        m = re.match(r"^\[([^\]]+)\]\s*[:=\-]?\s*(.*)$", ln)
        if m and m.group(1).strip().upper() in want:
            return m.group(1).strip().upper(), m.group(2)
        m = re.match(r"^(\S+?)\s*(?:[:=]\s*|\s+-\s+|\s+)(.+)$", ln)
        if m and m.group(1).upper() in want and re.search(r"\d", m.group(1)):
            return m.group(1).upper(), m.group(2)
        return None

    out: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        hit = key_at(lines[i])
        i += 1
        if hit is None:
            continue
        key, text = hit
        parts = [text.strip()] if text.strip() else []
        while i < len(lines) and len(parts) < 6:
            nxt = lines[i]
            if not nxt or set(nxt) <= set("-=*_#~ ") or nxt.startswith("[") or key_at(nxt) is not None:
                break
            parts.append(nxt)
            i += 1
        if parts and key not in out:
            out[key] = " ".join(" ".join(parts).split())
    return out


_YEAR_RE = re.compile(r"(?<![\dA-Za-z\-])(1[89]\d\d|20\d\d)(?!\d|\s*-\s*\d)")   # not page ranges
_INIT = r"(?:[A-Z][a-z]?\.(?:\s*-?\s*[A-Z]\.)*\s*)"
_SURNAME = r"[A-Z][a-zA-Z'’\-]*[a-z][a-zA-Z'’\-]*"
_AUTHOR_FULL = re.compile(rf"^{_INIT}+\s*({_SURNAME}(?:\s+{_SURNAME})?)$")       # A.T. Dinsdale, A. Fernandez G.
_AUTHOR_PREFIX = re.compile(rf"^{_INIT}+\s*({_SURNAME})(?![\w.])")                # B. Sundman Thermodynamic ...
_AUTHOR_NAMES = re.compile(rf"^[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?\s+(?:{_INIT}\s*)?({_SURNAME})$")   # Alan Dinsdale


_NOT_NAMES = {"DATA", "DATABASE", "REPORT", "THESIS", "UNARY", "ELEMENTS", "PHASE", "PHASES", "SYSTEM", "SYSTEMS",
              "ASSESSMENT", "WORK", "SOURCE", "UNPUBLISHED", "LABORATORY", "UNIVERSITY", "INSTITUTE", "VALUES",
              "PRESENT", "THIS", "ESTIMATED", "ESTIMATE"}


def _author(chunk: str) -> Optional[str]:
    c = chunk.strip()
    for rx in (_AUTHOR_FULL, _AUTHOR_NAMES, _AUTHOR_PREFIX):
        m = rx.match(c)
        if m and not any(w.upper() in _NOT_NAMES for w in m.group(0).split()):
            return m.group(1)
    return None


def short_citation(text: str, width: int = 30) -> str:
    """First author + year of a citation ("S. Hertzman, B. Sundman, Calphad, 6, 67-80(1982)" -> "Hertzman &
    Sundman 1982"; three or more authors -> "Dupin et al. 2001").  Text that does not start with author names is
    cut to ``width`` characters (plus the year when it is not already in the cut text)."""
    t = " ".join((text or "").split())
    if not t:
        return ""
    years = _YEAR_RE.findall(t)
    year = years[0] if years else ""
    authors = []
    chunks = re.split(r",|;|\band\b|&|(?<=[a-z])\.\s+(?=[A-Z]\.)", t)     # "C.P.Chin. S.Hertzman" = 2 authors
    for chunk in chunks:
        a = _author(chunk)
        if a is None:
            break
        authors.append(a)
        if not _AUTHOR_FULL.match(chunk.strip()) and not _AUTHOR_NAMES.match(chunk.strip()):
            break                                   # "B. Sundman Thermodynamic ...": the title starts here
    if not authors:                                 # "Title, L. Kaufman, eds. ...": an author after the title
        for chunk in chunks[1:3]:
            if _AUTHOR_FULL.match(chunk.strip()) and _author(chunk):
                authors.append(_author(chunk))
                break
    if authors:
        who = (authors[0] if len(authors) == 1 else f"{authors[0]} & {authors[1]}" if len(authors) == 2
               else f"{authors[0]} et al.")
        return f"{who} {year}".strip()
    first = re.split(r"[,;]", t)[0].strip().rstrip(".")
    s = first if len(first) <= width else first[: width - 1].rstrip() + "…"
    return s + (f" {year}" if year and year not in s else "")


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
    commands, db.header_comments, trailing, comments = _split_commands_ex(text)
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
    for cmd, trail in zip(commands, trailing):
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
                p = Parameter(raw_name, key, sign, omap, parse_piecewise(rest, tlo, thi))
                if not p.reference and trail and len(trail) <= 40:
                    p.reference = " ".join(trail.split())          # "... 6000 N ! $1967Kau"
                db.parameters.append(p)
            elif kw in ("LIST_OF_REFERENCES", "ADD_REFERENCES"):
                db.references.append(cmd)
                db.reference_list.update(parse_reference_command(cmd))
            elif kw is None:
                db.warnings.append(f"unrecognised command: {cmd[:60]!r}")
            else:
                db.other_commands.append(re.sub(r"\s+", " ", cmd))
        except (ValueError, IndexError, ExpressionError) as e:
            db.warnings.append(f"could not parse {cmd[:60]!r}: {e}")
    used = {p.reference for p in db.parameters if p.reference}
    missing = [r for r in used if r.upper() not in db.reference_list]
    for k, v in comment_references(comments, missing).items():
        db.reference_list.setdefault(k, v)
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
