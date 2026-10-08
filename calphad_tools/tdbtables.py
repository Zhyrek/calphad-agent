"""Subsystem selection and written-expression tables for comparing databases.

* :class:`Subsystem` - one parsed TDB restricted to an element subsystem: the phases that can exist with the elements
  (+VA) and the parameters whose constituents all lie inside, keyed so that the same parameter matches across
  databases (canonical phase name, vacancy-only sublattices dropped, constituents sorted with the odd
  Redlich-Kister sign handled).
* Written expressions: every parameter / FUNCTION is shown as the TDB writes it, tidied (spacing, ``+-`` -> ``-``,
  numeric coefficient first) and with ESPEI's ``VV####`` optimisation handles replaced by their values in the
  precision the file wrote.  Other functions (GHSERFE, ...) keep their names.
* Tables: every parameter across databases (sorted by phase, then G terms, L0/L1/..., ternary terms, TC/BMAGN/...),
  the rows that differ (values beyond the tolerance, or missing somewhere), the rows written differently but
  numerically equal, and per-database parameter / function lists.  Each row keeps every database's reference key
  (``ExprRow.refs``) and its full citation from the TDB's reference list (``ExprRow.citations``).

* Numbers: every parameter is also inlined (all FUNCTIONs substituted, :mod:`calphad_tools.pwinline`), so a row is
  a *difference* only if the values differ (or the parameter is missing somewhere); rows written differently but
  numerically equal are marked as such.  Where the databases of a row do not use the same functions (or define a
  shared one differently) the row is shown with the functions expanded so the cells compare term by term.

Uses the standard library, numpy, symengine, :mod:`calphad_tools.tdbparse` and :mod:`calphad_tools.pwinline`:
show_code copies this module's source (after those two) into standalone scripts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import symengine as se

from .pwinline import P, T, MissingRef, evaluate, fmt_expr, inline
from .tdbparse import TDB, ExpressionError, is_tentative, short_citation, tokenize

VV_RE = re.compile(r"^VV\d+$")
MATH_NAMES = {"LN", "LOG", "EXP", "SQRT", "ABS"}
PTYPE_GROUP = {"G": 0, "TC": 1, "BMAGN": 2}


def make_key(ptype: str, phase: str, reduced, order: int, diffusing: str = "") -> str:
    """Row identity of a parameter: canonical phase, vacancy-only sublattices dropped, constituents sorted."""
    c = ":".join(",".join(sl) for sl in reduced)
    d = f"&{diffusing}" if diffusing else ""
    return f"{ptype}({phase}{d},{c};{order})"


def make_label(ptype: str, phase: str, full, order: int, diffusing: str = "") -> str:
    pt = "L" if ptype == "G" and any(len(sl) > 1 for sl in full) else ptype
    return make_key(pt, phase, full, order, diffusing)


def key_constituents(key: str) -> List[List[str]]:
    inner = key[key.index("(") + 1: key.rindex(";")]
    cons = inner.split(",", 1)[1] if "," in inner else ""
    return [sl.split(",") for sl in cons.split(":")]


def key_order(key: str) -> int:
    return int(key.rsplit(";", 1)[1].rstrip(")"))


def is_vv(name: str) -> bool:
    return bool(VV_RE.match(name.upper().rstrip("#")))


_SE_MATH = {"LN": "_log", "LOG": "_log", "EXP": "_exp", "SQRT": "_sqrt", "ABS": "_abs"}
_SE_NS = {"__builtins__": {}, "_I": se.Integer, "_F": lambda s: se.RealDouble(float(s)), "_T": T, "_P": P,
          "_S": se.Symbol, "_log": se.log, "_exp": se.exp, "_sqrt": se.sqrt, "_abs": se.Abs}


def tdb_to_se(text: str):
    """TDB math expression -> symengine expression (FUNCTION names become symbols)."""
    toks = tokenize(text)
    parts = []
    for i, (kind, val) in enumerate(toks):
        if kind == "num":
            v = re.sub(r"[Dd]", "E", val)
            parts.append(f"_I({v})" if re.fullmatch(r"\d+", v) else f"_F('{v}')")
        elif kind == "op":
            parts.append("**" if val == "^" else val)
        else:
            u = val.upper().rstrip("#")
            nxt = toks[i + 1][1] if i + 1 < len(toks) else None
            if u in _SE_MATH and nxt == "(":
                parts.append(_SE_MATH[u])
            elif u == "T":
                parts.append("_T")
            elif u == "P":
                parts.append("_P")
            else:
                parts.append(f"_S('{u}')")
    src = "".join(parts) or "0"
    try:
        return eval(compile(src, "<expr>", "eval"), dict(_SE_NS))   # noqa: S307 - tokens are numbers/names/ops only
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse {text!r}: {e}") from None


def pw_segments(pw, sign: float = 1.0):
    """tdbparse Piecewise -> [(T_low, T_high, symengine expression)]."""
    return [(lo, hi, tdb_to_se(e) * sign if sign != 1.0 else tdb_to_se(e)) for lo, hi, e in pw.ranges]


# =============================================================================================== selection
@dataclass
class PhaseView:
    canon: str
    native: str
    site_ratios: List[float]
    constituents: List[List[str]]          # restricted to the subsystem
    va_only: List[int]
    magnetic: Optional[Tuple[float, float]] = None
    dis_part: Optional[str] = None         # this (ordered) phase's disordered part
    disordered_of: Optional[str] = None    # this phase is the DIS_PART of that phase
    ordered_like: bool = False

    @property
    def n_sub(self) -> int:
        return len(self.site_ratios)

    def ratios_text(self) -> str:
        return ":".join(f"{r:g}" + ("(VA)" if i in self.va_only else "") for i, r in enumerate(self.site_ratios))

    def reduced_ratios(self) -> Tuple[float, ...]:
        return tuple(round(r, 6) for i, r in enumerate(self.site_ratios) if i not in self.va_only)

    def reduced_constituents(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(sorted(c)) for i, c in enumerate(self.constituents) if i not in self.va_only)

    def hints(self) -> List[str]:
        h = []
        if self.magnetic:
            h.append(f"magnetic {self.magnetic[0]:g}/{self.magnetic[1]:g}")
        if self.dis_part:
            h.append(f"ordered, DIS_PART {self.dis_part}")
        if self.disordered_of:
            h.append(f"disordered part of {self.disordered_of}")
        if self.ordered_like and not self.dis_part:
            h.append("equivalent sublattices (ordering)")
        return h


@dataclass
class ParamRow:
    key: str                  # row identity, e.g. "G(BCC_A2,CR,FE;0)" (canonical phase, VA-only sublattices dropped)
    label: str                # display, e.g. "L(BCC_A2,CR,FE:VA;0)"
    phase: str                # canonical phase
    ptype: str
    n_species: int
    raw_name: str
    raw_text: str
    sign: float
    order_changed: bool
    raw_pw: object = None     # tdbparse.Piecewise as written
    segs: Optional[list] = None
    error: str = ""
    reference: str = ""       # reference key as written ("91Din", "REF283"; "" = none)

    def sort_key(self):
        return row_sort_key(self.key, self.ptype)


def row_sort_key(key: str, ptype: str) -> tuple:
    """G terms (unaries, then compound end-members), binary L0, L1, ... per constituent set, ternary/reciprocal
    terms; then the same for TC, BMAGN and other parameter types."""
    cons = [tuple(s for s in sl) for sl in key_constituents(key)]
    extra = sum(len(sl) - 1 for sl in cons)
    elements = {s for sl in cons for s in sl if s not in ("VA", "*")}
    if extra == 0:
        cat = (0, len(elements))
    elif extra == 1:
        cat = (1, 0)
    else:
        cat = (2, extra)
    return (PTYPE_GROUP.get(ptype, 3), ptype, cat, tuple(cons), key_order(key), key)


def phase_order(names) -> List[str]:
    """LIQUID first, then alphabetical."""
    return sorted(set(names), key=lambda p: (p != "LIQUID", p))


class Subsystem:
    """A parsed TDB restricted to ``elements`` (+VA); ``rename`` maps native -> canonical phase names."""

    def __init__(self, tdb: TDB, label: str, rel: str, elements: Sequence[str], rename: Dict[str, str]):
        self.tdb = tdb
        self.label, self.rel = label, rel
        self.elements = [e.upper() for e in elements]
        self.E = set(self.elements) | {"VA"}
        self.rename = {k.upper(): v.upper() for k, v in rename.items()}
        self.notes: List[str] = []
        self.missing_elements = [e for e in self.elements if e not in self.tdb.elements]
        self._written: Dict[str, "Written"] = {}
        self._fwritten: Dict[str, "Written"] = {}
        self.inline_notes: set = set()
        self._fn: Dict[str, list] = {}
        self._build_phases()
        self._build_rows()
        for row in self.rows.values():            # every parameter with all FUNCTIONs substituted
            try:
                row.segs = inline(pw_segments(row.raw_pw, row.sign), self.fn, self.inline_notes)
            except MissingRef as e:
                row.error = f"undefined function {e.args[0]}"
            except (ExpressionError, RecursionError, ValueError, ZeroDivisionError) as e:
                row.error = f"{type(e).__name__}: {e}"

    def fn(self, name: str, _stack: Tuple[str, ...] = ()) -> list:
        """Fully inlined segments of FUNCTION ``name``."""
        name = name.upper()
        if name in self._fn:
            return self._fn[name]
        f = self.tdb.functions.get(name)
        if f is None:
            if name == "R":
                return [(-math.inf, math.inf, se.RealDouble(8.31451))]
            raise MissingRef(name)
        if name in _stack:
            raise ExpressionError("circular function reference: " + " -> ".join(_stack + (name,)))
        segs = inline(pw_segments(f.expr), lambda n: self.fn(n, _stack + (name,)), self.inline_notes)
        self._fn[name] = segs
        return segs

    # ------------------------------------------------------------------ helpers
    def canon(self, native: str) -> str:
        return self.rename.get(native.upper(), native.upper())

    def canon_of(self, name: str) -> Optional[str]:
        """Canonical name for a phase given by canonical or native name (None if not in this database)."""
        u = name.upper()
        if u in self.phases:
            return u
        c = self.rename.get(u)
        if c and c in self.phases:
            return c
        return None

    def species_elements(self, sp: str) -> set:
        s = sp.upper().strip("%")
        if s in ("VA", "*") or s in self.tdb.elements:
            return {s}
        spc = self.tdb.species.get(s)
        if spc is not None:
            formula = spc.formula.split("/")[0]
            els = set(re.findall(r"([A-Z][A-Z]?)\d+\.?\d*", formula.upper()))
            if els:
                return els
        return {re.sub(r"[+-]\d*$", "", s)}

    def in_sub(self, sp: str) -> bool:
        return sp == "*" or self.species_elements(sp) <= self.E

    # ------------------------------------------------------------------ phases
    def _build_phases(self):
        self.phases: Dict[str, PhaseView] = {}
        seen: Dict[str, str] = {}
        dis_targets: Dict[str, str] = {}
        for nm, ph in self.tdb.phases.items():
            restricted = [[s for s in sl if self.in_sub(s)] for sl in ph.constituents]
            if not restricted or any(not sl for sl in restricted):
                continue
            if all(s == "VA" for sl in restricted for s in sl):
                continue
            cn = self.canon(nm)
            if cn in seen:
                raise ValueError(f"phase_map merges two phases of {self.label}: {seen[cn]} and {nm} -> {cn}")
            seen[cn] = nm
            pv = PhaseView(cn, nm, list(ph.site_ratios), restricted,
                           [i for i, sl in enumerate(restricted) if sl == ["VA"]])
            for td in self.tdb.phase_typedefs(ph):
                t = td.text.upper()
                if td.kind == "MAGNETIC":
                    m = re.search(r"MAGNETIC\s+(\S+)\s+(\S+)", t)
                    if m:
                        try:
                            pv.magnetic = (float(m.group(1)), float(m.group(2)))
                        except ValueError:
                            pass
                if "DIS_P" in t:
                    m = re.search(r"DIS_P\w*\s+([A-Z0-9_]+)", t)
                    if m:
                        pv.dis_part = m.group(1)
            red = [(tuple(sorted(c)), round(r, 6)) for i, (c, r) in enumerate(zip(restricted, ph.site_ratios))
                   if i not in pv.va_only]
            pv.ordered_like = len(red) > 1 and len(set(red)) < len(red) and all(len(c) > 1 for c, _ in red)
            self.phases[cn] = pv
        for pv in self.phases.values():
            if pv.dis_part:
                pv.dis_part = self.canon(pv.dis_part)
                dis_targets[pv.dis_part] = pv.canon
        for d, o in dis_targets.items():
            if d in self.phases:
                self.phases[d].disordered_of = o
            else:
                self.notes.append(f"{o}: DIS_PART {d} is not a phase of the subsystem")

    # ------------------------------------------------------------------ parameters
    def _build_rows(self):
        self.rows: Dict[str, ParamRow] = {}
        self.pyc_sign_notes: List[str] = []
        by_native = {pv.native: pv for pv in self.phases.values()}
        for p in self.tdb.parameters:
            pv = by_native.get(p.key.phase)
            if pv is None:
                continue
            if not all(self.in_sub(s) for sl in p.key.constituents for s in sl):
                continue
            if p.key.diffusing and not self.in_sub(p.key.diffusing):
                continue
            cons = p.key.constituents
            if len(cons) == pv.n_sub:
                reduced = tuple(sl for i, sl in enumerate(cons) if i not in pv.va_only)
            else:
                reduced = cons
                self.notes.append(f"{p.raw_name}: {len(cons)} sublattices, phase {pv.native} has {pv.n_sub}")
            key = make_key(p.key.ptype, pv.canon, reduced, p.key.order, p.key.diffusing)
            if key in self.rows:
                self.notes.append(f"{key} defined twice; the last definition is used")
            order_changed = p.sign != 1.0 or bool(p.order_map and any(a != b for a, b in p.order_map.items()))
            self.rows[key] = ParamRow(key, make_label(p.key.ptype, pv.canon, cons, p.key.order, p.key.diffusing),
                                      pv.canon, p.key.ptype, sum(1 for sl in reduced for s in sl if s != "VA"),
                                      p.raw_name, p.expr.to_tdb(), p.sign, order_changed, p.expr,
                                      reference=getattr(p, "reference", "") or "")
            if order_changed:
                self.pyc_sign_notes.append(p.raw_name)

    def citation(self, key: str) -> str:
        """Full citation of row ``key``'s reference from the database's reference list ("" if not resolved)."""
        r = self.rows.get(key)
        cite = getattr(self.tdb, "citation", None)
        return cite(r.reference) if r is not None and r.reference and cite else ""

    def used_functions(self, keys: Optional[Sequence[str]] = None, include_vv: bool = False) -> List[str]:
        """FUNCTIONs reachable from the subsystem's parameters (VV#### optimisation handles left out)."""
        out: List[str] = []
        todo: List[str] = []
        for k, r in self.rows.items():
            if keys is not None and k not in keys:
                continue
            if r.raw_pw is not None:
                todo += r.raw_pw.symbols()
        while todo:
            n = todo.pop(0)
            if n in out or n not in self.tdb.functions:
                continue
            out.append(n)
            todo += self.tdb.functions[n].expr.symbols()
        return out if include_vv else [n for n in out if not is_vv(n)]

    # ------------------------------------------------------------------ written expressions
    def written(self, key: str) -> "Written":
        """The parameter as written (VV#### substituted, tidied), sign-corrected for reordered constituents."""
        if key not in self._written:
            r = self.rows[key]
            self._written[key] = written_expr(r.raw_pw.ranges, self.tdb.functions, r.sign)
        return self._written[key]

    def function_written(self, name: str) -> Optional["Written"]:
        f = self.tdb.functions.get(name)
        if f is None:
            return None
        if name not in self._fwritten:
            self._fwritten[name] = written_expr(f.expr.ranges, self.tdb.functions)
        return self._fwritten[name]

    def function_closure(self, names) -> List[str]:
        """Non-VV functions referenced directly or through other functions (undefined names included)."""
        out: List[str] = []
        todo = list(names)
        while todo:
            n = todo.pop(0)
            if n in out or is_vv(n):
                continue
            out.append(n)
            w = self.function_written(n)
            if w is not None:
                todo += sorted(w.refs)
        return out


# =============================================================================================== written expressions
# tiny expression tree: ('num', text) ('name', NAME) ('call', NAME, [sum, ...]) ('group', sum) ('pow', base, exp)
# ('neg', node);  sum = [[sign, [(op, factor), ...]], ...]  with op '*' or '/'
class _Parser:
    def __init__(self, text: str):
        self.t = [(k, v.upper().rstrip("#") if k == "name" else v) for k, v in tokenize(text)]
        self.i = 0

    def peek(self):
        return self.t[self.i][1] if self.i < len(self.t) else None

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def sum(self):
        terms = []
        while self.i < len(self.t) and self.peek() not in (")", ","):
            sign = 1
            while self.peek() in ("+", "-"):
                if self.take()[1] == "-":
                    sign = -sign
            factors = [("*", self.power())]
            while self.peek() in ("*", "/"):
                op = self.take()[1]
                factors.append((op, self.power()))
            terms.append([sign, factors])
        return terms

    def power(self):
        base = self.atom()
        if self.peek() in ("**", "^"):
            self.take()
            neg = False
            while self.peek() in ("+", "-"):
                neg ^= self.take()[1] == "-"
            ex = self.atom()
            base = ("pow", base, ("neg", ex) if neg else ex)
        return base

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return ("num", val)
        if kind == "name":
            if self.peek() == "(":
                self.take()
                args = [self.sum()]
                while self.peek() == ",":
                    self.take()
                    args.append(self.sum())
                self.take()                      # ')'
                return ("call", val, args)
            return ("name", val)
        if val == "(":
            s = self.sum()
            self.take()                          # ')'
            return ("group", s)
        if val in ("+", "-"):                    # unary sign inside a factor: 2*-3
            node = self.atom()
            return ("neg", node) if val == "-" else node
        raise ValueError(f"unexpected {val!r}")


def _parse(text: str):
    p = _Parser(text)
    s = p.sum()
    if p.i != len(p.t):
        raise ValueError(f"cannot parse {text!r}")
    return s


def _vv_sum(name: str, functions, stack: Tuple[str, ...]):
    f = functions.get(name)
    if f is None or len(f.expr.ranges) != 1 or name in stack:
        return None
    return _subst(_parse(f.expr.ranges[0][2]), functions, stack + (name,))


def _subst_node(node, functions, stack):
    kind = node[0]
    if kind == "call":
        return ("call", node[1], [_subst(a, functions, stack) for a in node[2]])
    if kind == "group":
        return ("group", _subst(node[1], functions, stack))
    if kind == "pow":
        return ("pow", _subst_node(node[1], functions, stack), _subst_node(node[2], functions, stack))
    if kind == "neg":
        return ("neg", _subst_node(node[1], functions, stack))
    if kind == "name" and is_vv(node[1]):
        s = _vv_sum(node[1], functions, stack)
        if s is None:
            return node
        if len(s) == 1 and len(s[0][1]) == 1 and s[0][0] == 1:
            return s[0][1][0][1]                 # a plain positive number / name
        return ("group", s)
    return node


def _subst(terms, functions, stack=()):
    """Replace VV#### names by their values; a negative value moves its sign to the term; numbers go first."""
    out = []
    for sign, factors in terms:
        new = []
        for op, f in factors:
            g = _subst_node(f, functions, stack)
            # a (sub-)product spliced into this product: take over its sign and factors
            if op == "*" and g[0] == "group" and len(g[1]) == 1:
                s2, f2 = g[1][0]
                sign *= s2
                new += f2
                continue
            new.append((op, g))
        if len(new) == 1 and new[0][1][0] == "group" and new[0][0] == "*":   # a lone sum: splice its terms
            for s2, f2 in new[0][1][1]:
                out.append([sign * s2, f2])
            continue
        nums = [x for x in new if x[0] == "*" and x[1][0] == "num"]
        rest = [x for x in new if not (x[0] == "*" and x[1][0] == "num")]
        if rest:                                  # 1.0*GHSERCR -> GHSERCR
            nums = [x for x in nums if _num_value(x[1][1]) != 1.0]
        if rest and rest[0][0] == "/":
            rest.insert(0, ("*", ("num", "1"))) if not nums else None
        out.append([sign, nums + rest])
    return out


def _num_value(text: str) -> float:
    return float(re.sub(r"[Dd]", "E", text))


def _fmt_number(text: str, canonical: bool) -> str:
    if canonical:
        return repr(float(re.sub(r"[Dd]", "E", text)))
    return text


def _fmt_node(node, canonical: bool) -> str:
    kind = node[0]
    if kind == "num":
        return _fmt_number(node[1], canonical)
    if kind == "name":
        return node[1]
    if kind == "call":
        name = "LN" if canonical and node[1] == "LOG" else node[1]
        return f"{name}(" + ",".join(_fmt_sum(a, canonical, False) for a in node[2]) + ")"
    if kind == "group":
        return "(" + _fmt_sum(node[1], canonical, False) + ")"
    if kind == "pow":
        ex = node[2]
        if ex[0] == "group" and len(ex[1]) == 1 and len(ex[1][0][1]) == 1 and ex[1][0][1][0][1][0] == "num":
            sign, ((_, num),) = ex[1][0]                       # T**(7) -> T**7, T**(-1) stays
            ex = ("neg", num) if sign < 0 else num
        ex_s = _fmt_node(ex, canonical)
        if ex[0] == "neg" or (ex[0] not in ("num", "name", "group", "call")):
            ex_s = f"({ex_s})"
        return f"{_fmt_node(node[1], canonical)}**{ex_s}"
    if kind == "neg":
        return "-" + _fmt_node(node[1], canonical)
    raise ValueError(kind)


def _fmt_term(factors, canonical: bool) -> str:
    if canonical:                                 # product order does not matter for the comparison
        factors = ([f for f in factors if f[0] == "*"] and
                   sorted((f for f in factors if f[0] == "*"), key=lambda f: _fmt_node(f[1], True))) \
            + [f for f in factors if f[0] == "/"]
    s = ""
    for i, (op, f) in enumerate(factors):
        s += ("" if i == 0 else op) + _fmt_node(f, canonical)
    return s


def _fmt_sum(terms, canonical: bool, top: bool = True) -> str:
    parts = [(sign, _fmt_term(f, canonical)) for sign, f in terms]
    if canonical:
        parts.sort(key=lambda x: (x[1], x[0]))
    if not parts:
        return "0"
    sep_p, sep_m = (" + ", " - ") if top and not canonical else ("+", "-")
    s = ("-" if parts[0][0] < 0 else "") + parts[0][1]
    for sign, t in parts[1:]:
        s += (sep_m if sign < 0 else sep_p) + t
    return s


def _refs(terms, acc: set):
    for _, factors in terms:
        for _, f in factors:
            _refs_node(f, acc)
    return acc


def _refs_node(node, acc):
    kind = node[0]
    if kind == "name" and node[1] not in ("T", "P"):
        acc.add(node[1])
    elif kind == "call":
        for a in node[2]:
            _refs(a, acc)
    elif kind == "group":
        _refs(node[1], acc)
    elif kind in ("pow",):
        _refs_node(node[1], acc)
        _refs_node(node[2], acc)
    elif kind == "neg":
        _refs_node(node[1], acc)


@dataclass
class Written:
    lines: List[str]                  # one per T range ("298.15–2180: ..." when there are several)
    canon: tuple                      # comparable form (numbers normalised, terms sorted, ranges included)
    refs: set = field(default_factory=set)   # non-VV names referenced (functions)
    ranges: List[Tuple[float, float, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def tidy(text: str, functions=None, sign: float = 1.0) -> Tuple[str, str, set]:
    """(display, canonical, referenced names) of one TDB expression with VV#### replaced by their values."""
    try:
        terms = _subst(_parse(text), functions or {})
    except (ValueError, IndexError):
        return text, re.sub(r"\s+", "", text.upper()), set()
    if sign < 0:
        terms = [[-s, f] for s, f in terms]
    return _fmt_sum(terms, False), _fmt_sum(terms, True), {n for n in _refs(terms, set()) if not is_vv(n)}


def written_expr(ranges, functions, sign: float = 1.0) -> Written:
    lines, canon, refs, rng = [], [], set(), []
    for lo, hi, text in ranges:
        d, c, r = tidy(text, functions, sign)
        refs |= r
        rng.append((lo, hi, d))
        canon.append(c)
    if len(ranges) == 1:
        lines = [rng[0][2]]
        canon_t = (canon[0],)
    else:
        lines = [f"{lo:g}–{hi:g}: {d}" for lo, hi, d in rng]
        canon_t = tuple((round(lo, 4), round(hi, 4), c) for (lo, hi, _), c in zip(rng, canon))
    return Written(lines, canon_t, refs, rng)


# =============================================================================================== tables
@dataclass
class ExprRow:
    key: str
    phase: str
    label: str
    ptype: str
    cells: Dict[str, Optional[Written]]            # database label -> as written (None: absent)
    status: str = "same"                           # same | equal (written differently) | differs
    kind: str = ""                                 # differs because: value | absent
    note: str = ""
    shown: Dict[str, Optional[str]] = field(default_factory=dict)      # cell text (expanded when needed)
    shown_short: Dict[str, Optional[str]] = field(default_factory=dict)  # same, long non-polynomial terms cut
    expanded: List[str] = field(default_factory=list)                  # functions expanded for the comparison
    cell_differs: Dict[str, bool] = field(default_factory=dict)        # value differs from the first database
    max_diff: float = math.nan
    refs: Dict[str, str] = field(default_factory=dict)       # database label -> reference key as written ("" none)
    citations: Dict[str, str] = field(default_factory=dict)  # database label -> full citation ("" unresolved)

    def ref_text(self, label: str, width: int = 30) -> str:
        """'91Din', '91Din · Dinsdale 1991' (short citation when the reference list resolves it) or ''."""
        return ref_display(self.refs.get(label, ""), self.citations.get(label, ""), width)


def ref_display(ref: str, citation: str = "", width: int = 30) -> str:
    """Reference shorthand as written plus, when resolved, the short citation: '82Her · Hertzman & Sundman 1982';
    a tentative parameter (TENT_MIX / TENT_VA, or a citation starting "Tentative") shows 'TENT_MIX · tentative'."""
    if not ref:
        return ""
    if is_tentative(ref, citation=citation):
        return f"{ref} · tentative"
    short = short_citation(citation, width) if citation else ""
    return f"{ref} · {short}" if short and short.lower() != ref.lower() else ref


def tolerance(ptype: str) -> float:
    """Absolute tolerance: 1e-3 J/mol for Gibbs-energy terms, 1e-6 for TC/BMAGN-type magnitudes."""
    return 1e-3 if ptype == "G" else 1e-6


def segs_close(a, b, grid, rtol: float, atol: float):
    """(equal over the common T range, max |difference|)."""
    lo, hi = max(a[0][0], b[0][0], grid[0]), min(a[-1][1], b[-1][1], grid[-1])
    g = grid[(grid >= lo) & (grid <= hi)] if lo < hi else np.asarray([0.5 * (lo + hi)])
    va, vb = evaluate(a, g)[0], evaluate(b, g)[0]
    m = np.isfinite(va) & np.isfinite(vb)
    if not m.any():
        return False, math.nan
    d = float(np.max(np.abs(va[m] - vb[m])))
    scale = float(max(np.max(np.abs(va[m])), np.max(np.abs(vb[m]))))   # over the range, as compare_tdbs does
    return d <= max(atol, rtol * scale), d


def expanded_text(segs, max_other: int = 0) -> str:
    """Inlined piecewise polynomial in T, terms collected, one line per T range (non-polynomial terms such as
    pressure/volume contributions are cut to ``max_other`` characters if given)."""
    if len(segs) == 1:
        return fmt_expr(segs[0][2], 10, max_other)
    return "\n".join(f"{lo:g}–{hi:g}: {fmt_expr(e, 10, max_other)}" for lo, hi, e in segs)


def expression_table(subs: List[Subsystem], phases: Optional[Sequence[str]] = None, rtol: float = 1e-4,
                     t_grid: Tuple[float, float] = (298.15, 3000.0)) -> List[ExprRow]:
    """Every parameter of the subsystem across databases, ordered by phase then parameter type/order.

    status: "same" (identical as written, functions included), "equal" (written differently, numerically equal
    within rtol over t_grid) or "differs" (values differ, or absent from some database)."""
    names = {r.phase for s in subs for r in s.rows.values()}
    order = phase_order(names)
    if phases:
        want = {p.upper() for p in phases}
        order = [p for p in order if p in want]
    grid = np.linspace(t_grid[0], t_grid[1], 121)
    fcanon: Dict[Tuple[str, str], Optional[tuple]] = {}

    def fdef(s: Subsystem, name: str):
        k = (s.label, name)
        if k not in fcanon:
            w = s.function_written(name)
            fcanon[k] = None if w is None else w.canon
        return fcanon[k]

    out: List[ExprRow] = []
    for ph in order:
        keys: Dict[str, Tuple[tuple, str, str]] = {}
        for s in subs:
            for k, r in s.rows.items():
                if r.phase != ph:
                    continue
                prev = keys.get(k)
                if prev is None or len(r.label) > len(prev[1]):    # label that shows the :VA sublattices
                    keys[k] = (row_sort_key(k, r.ptype), r.label, r.ptype)
        for k, (_, label, ptype) in sorted(keys.items(), key=lambda kv: kv[1][0]):
            out.append(_compare_row(subs, k, ph, label, ptype, fdef, grid, rtol))
    return out


def _compare_row(subs, k, ph, label, ptype, fdef, grid, rtol) -> ExprRow:
    cells = {s.label: (s.written(k) if k in s.rows else None) for s in subs}
    row = ExprRow(k, ph, label, ptype, cells)
    present = [s for s in subs if cells[s.label] is not None]
    row.refs = {s.label: s.rows[k].reference for s in present}
    row.citations = {s.label: s.citation(k) for s in present}
    absent = [s.label for s in subs if cells[s.label] is None]
    row.shown = {lab: (w.text if w is not None else None) for lab, w in cells.items()}
    row.shown_short = dict(row.shown)
    # functions used by each database for this parameter (direct + indirect), and whether they agree
    closures = {s.label: s.function_closure(sorted(cells[s.label].refs)) for s in present}
    union = sorted({f for c in closures.values() for f in c})
    differing = [f for f in union
                 if len({fdef(s, f) for s in present if f in closures[s.label]}) > 1
                 or any(f not in closures[s.label] for s in present)]
    textual_same = (len({cells[s.label].canon for s in present}) <= 1 and not differing)
    # numbers
    segs = {s.label: s.rows[k].segs for s in present}
    numeric = len(present) > 1 and all(v is not None for v in segs.values())
    if numeric:
        first = present[0].label
        atol = tolerance(ptype)
        for s in present[1:]:
            ok, d = segs_close(segs[first], segs[s.label], grid, rtol, atol)
            row.cell_differs[s.label] = not ok
            if math.isfinite(d):
                row.max_diff = d if math.isnan(row.max_diff) else max(row.max_diff, d)
    values_differ = any(row.cell_differs.values()) if numeric else not textual_same
    # expand the functions when the databases do not share identical ones
    if len(present) > 1 and differing and all(v is not None for v in segs.values()):
        row.expanded = sorted({f for s in present for f in cells[s.label].refs})    # the names written in the cells
        for s in present:
            row.shown[s.label] = expanded_text(segs[s.label])
            row.shown_short[s.label] = expanded_text(segs[s.label], 50)
    notes = []
    if row.expanded:
        notes.append("functions expanded for comparison: " + ", ".join(row.expanded[:6])
                     + (" …" if len(row.expanded) > 6 else ""))
    if absent and present:
        row.status, row.kind = "differs", "absent"
        notes.append("absent in " + ", ".join(absent) + ("; values differ" if values_differ and len(present) > 1
                                                          else ""))
    elif values_differ:
        row.status, row.kind = "differs", "value"
        if math.isfinite(row.max_diff):
            notes.append(f"max |Δ| {row.max_diff:.3g}")
    elif not textual_same:
        row.status = "equal"
        notes.append("numerically equal (written differently)")
    row.note = "; ".join(notes)
    return row


def differences(rows: List[ExprRow]) -> List[ExprRow]:
    """Rows whose values differ (beyond the tolerance) or that are missing from some database.  Rows equal within
    the tolerance - identical, written differently, or differing only in their reference - are never listed."""
    return [r for r in rows if r.status == "differs"]


def equal_rows(rows: List[ExprRow]) -> List[ExprRow]:
    """Rows written differently but numerically equal within the tolerance (status "equal")."""
    return [r for r in rows if r.status == "equal"]


def database_keys(sub: Subsystem, phases: Optional[Sequence[str]] = None) -> List[str]:
    """Row keys of one database's subsystem parameters, in table order."""
    order = phase_order({r.phase for r in sub.rows.values()})
    if phases:
        want = {p.upper() for p in phases}
        order = [p for p in order if p in want]
    out = []
    for ph in order:
        out += sorted((k for k, r in sub.rows.items() if r.phase == ph),
                      key=lambda k: row_sort_key(k, sub.rows[k].ptype))
    return out


def database_table(sub: Subsystem, phases: Optional[Sequence[str]] = None) -> List[Tuple[str, str, Written]]:
    """(phase, label, written) of one database's subsystem parameters, in table order."""
    return [(sub.rows[k].phase, sub.rows[k].label, sub.written(k)) for k in database_keys(sub, phases)]


def database_functions(sub: Subsystem, phases: Optional[Sequence[str]] = None) -> List[Tuple[str, Written]]:
    """Non-VV FUNCTIONs used by the database's (shown) subsystem parameters, with their written expressions."""
    keys = [k for k, r in sub.rows.items() if not phases or r.phase in {p.upper() for p in phases}]
    names = set()
    for k in keys:
        names |= sub.written(k).refs
    out = []
    for n in sub.function_closure(sorted(names)):
        w = sub.function_written(n)
        if w is not None:
            out.append((n, w))
    return out
