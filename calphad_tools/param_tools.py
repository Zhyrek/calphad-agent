"""Parameter "what-if" edits, sensitivities and a gradient-based optimizer on top of ESPEI's residual objects.

Registered by :func:`register` (called from ``calphad_tools.agent.tools``):

* ``edit_parameters``       - edit TDB parameters / FUNCTIONs in place (the rest of the file stays byte-identical),
  optional before/after diagram overlay with invariant and boundary shifts, and the change of the ESPEI objective.
* ``parameter_sensitivity`` - central finite-difference derivatives of the ESPEI objective (total and per residual
  type), invariant temperatures/compositions, phase-boundary compositions or equilibrium properties with respect to
  chosen parameters; CSV, tornado plot, Fisher-matrix correlations / identifiability warnings.
* ``optimize_parameters``   - gradient-based minimisation of ESPEI's negative log-likelihood (scipy least_squares
  'trf' on the whitened residuals, or L-BFGS-B on -lnL) over chosen symbols or parameter coefficients.

Notes verified against the installed ESPEI 0.9.1 source
-------------------------------------------------------
* ``SciPyOptimizer.predict`` returns the summed log-likelihood (larger = better) and ``_fit`` passes it to
  ``scipy.optimize.minimize`` - it therefore moves AWAY from the data.  Here the objective is ``-lnL``.
* ``get_residuals`` returns ``(residuals, weights)``; for ZPF the second item is the dataset weight and the likelihood
  scale is ``1000 / data_weight / weight`` J/mol, for all other types it already is the standard deviation.  With
  ``z = residual / sigma``:  ``-lnL = sum(z**2)/2 + sum(ln(sigma)) + n ln(2 pi)/2``  (checked in the tests), so
  least squares on ``z`` minimises exactly ESPEI's objective.
* ``EquilibriumPropertyResidual`` compares datasets with several T (or P) AND several compositions in the wrong
  order (X-major vs P/T-major flattening).  Such datasets give a wrong objective; they are flagged, split them.
* pycalphad 0.11.2: a fitted symbol inside the disordered part of an order-disorder phase (A1_FCC of L12_FCC) is
  replaced by its database value when the ordered phase's Model is built with ``parameters`` (the disordered-part
  Model does not get them), so ESPEI's residuals see the ordered phase at the start value.  Warned about.
* Analytic gradients: ESPEI PR #268 (Kunselman, Bocklund et al., "Analytical gradient-based optimization of CALPHAD
  model parameters", Acta Mater. 2025) makes ``get_likelihood`` return ``(lnL, dlnL/dparams)``.  It is not released;
  if such an ESPEI is installed it is detected and used for L-BFGS-B, otherwise central finite differences are used.

Free parameters
---------------
``"VV0001"`` (any FUNCTION that is a constant: fitted directly), ``"L(FCC_A1,CR,NI:VA;0)"`` (constant term) or
``"L(FCC_A1,CR,NI:VA;0)@T"`` (coefficient of T; also TLNT, T2, T3, TINV).  A coefficient is fitted by adding a
temporary FUNCTION ``DVnnnn`` (initial 0) times the basis to the parameter expression; the result is written back as
a plain numeric edit of the original TDB.
"""
from __future__ import annotations

import csv
import difflib
import json
import math
import re
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .tdbparse import (ExpressionError, Piecewise, _fmt_num, parse_param_name, parse_piecewise,  # noqa: E402
                       parse_tdb, resolve_keyword, tokenize, wrap_tdb_line)

KELVIN = 273.15
LN2PI = math.log(2 * math.pi)
SYMBOL_RE = re.compile(r"^V[V]?([0-9]+)$")
# typical magnitude of a coefficient of each basis (J/mol, J/mol/K, ...) - FD steps and scaling
TYPICAL = {"const": 1000.0, "T": 1.0, "TLNT": 0.1, "T2": 1e-4, "T3": 1e-7, "TINV": 1e4, "LNT": 10.0}
BASIS_TEXT = {"const": "", "T": "*T", "TLNT": "*T*LN(T)", "T2": "*T**2", "T3": "*T**3", "TINV": "*T**(-1)",
              "LNT": "*LN(T)"}
BASIS_ALIASES = {"": "const", "CONST": "const", "C": "const", "0": "const", "A": "const", "T": "T", "B": "T",
                 "TLNT": "TLNT", "TLN(T)": "TLNT", "T*LN(T)": "TLNT", "C_TLNT": "TLNT", "T2": "T2", "T**2": "T2",
                 "T^2": "T2", "T3": "T3", "T**3": "T3", "T^3": "T3", "TINV": "TINV", "1/T": "TINV", "T**(-1)": "TINV",
                 "T-1": "TINV", "LNT": "LNT", "LN(T)": "LNT"}
RESIDUAL_TYPES = {"ZPFResidual": "ZPF", "FixedConfigurationPropertyResidual": "thermo_fixed",
                  "EquilibriumPropertyResidual": "thermo_equilibrium", "ActivityResidual": "activity"}


def _basis_key(term: Optional[str]) -> str:
    if term is None:
        return "const"
    t = str(term).strip().upper().replace(" ", "")
    if t in BASIS_ALIASES:
        return BASIS_ALIASES[t]
    m = re.fullmatch(r"T(?:\*\*|\^)?\(?([+-]?\d+)\)?", t)
    if m:
        n = int(m.group(1))
        return {1: "T", -1: "TINV"}.get(n, f"T{n}")
    raise ValueError(f"unknown term {term!r}: use const, T, TLNT, T2, T3, TINV (or T<n>)")


def _basis_text(b: str) -> str:
    if b in BASIS_TEXT:
        return BASIS_TEXT[b]
    return f"*T**({int(b[1:])})"


def _typical(b: str) -> float:
    if b in TYPICAL:
        return TYPICAL[b]
    n = int(b[1:])
    return 1000.0 / 1000.0 ** n if n > 0 else 1000.0 * 1000.0 ** (-n)


# ---------------------------------------------------------------------------------------------------------
# expressions as sums of terms
# ---------------------------------------------------------------------------------------------------------
_NUM = r"(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"


@dataclass
class Term:
    sign: int                 # +1 / -1
    text: str                 # unsigned text as written (whitespace free)
    coef: Optional[float]     # numeric coefficient if the term is coef * basis
    basis: Optional[str]      # "const", "T", "TLNT", ... (None: symbolic term)

    def render(self) -> str:
        if self.basis is None:
            return ("+" if self.sign > 0 else "-") + self.text
        c = self.sign * self.coef
        s = "+" if c >= 0 else "-"
        a = abs(c)
        if self.basis == "const":
            return s + _fmt_coef(a)
        return s + ("" if a == 1.0 else _fmt_coef(a)) + (_basis_text(self.basis)[1:] if a == 1.0
                                                         else _basis_text(self.basis))


def _fmt_coef(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = f"{x:.12g}".upper()
    if "E" in s and "." not in s.split("E")[0]:
        m, e = s.split("E")
        s = f"{m}.0E{e}"
    return s


def _classify(text: str) -> Tuple[Optional[float], Optional[str]]:
    """'10.5*T' -> (10.5, 'T'); '5320' -> (5320, 'const'); 'GHSERCR' -> (None, None)."""
    u = text.upper()
    m = re.fullmatch(rf"({_NUM})(?:\*(.+))?", u)
    if m:
        coef = float(re.sub(r"[D]", "E", m.group(1)))
        rest = m.group(2)
    else:
        coef, rest = 1.0, u
    if rest is None:
        return coef, "const"
    rest = rest.replace("LOG(", "LN(")
    table = {"T": "T", "T*LN(T)": "TLNT", "LN(T)*T": "TLNT", "LN(T)": "LNT"}
    if rest in table:
        return coef, table[rest]
    pm = re.fullmatch(r"T(?:\*\*|\^)\(?([+-]?\d+)\)?", rest)
    if pm:
        n = int(pm.group(1))
        return coef, {1: "T", -1: "TINV"}.get(n, f"T{n}")
    return None, None


def split_terms(expr: str) -> List[Term]:
    """Split a TDB expression into top-level signed terms (numbers with exponents are never split)."""
    toks = tokenize(expr) if expr.strip() else []
    terms: List[Term] = []
    cur: List[str] = []
    sign, depth = 1, 0
    last_ends_value = False
    for kind, val in toks:
        if kind == "op" and val in "+-" and depth == 0 and (not cur or last_ends_value):
            if cur:
                terms.append(_mk_term(sign, cur))
                cur, sign = [], 1
            sign *= -1 if val == "-" else 1
            continue
        if val == "(":
            depth += 1
        elif val == ")":
            depth -= 1
        cur.append(val)
        last_ends_value = kind in ("num", "name") or val == ")"
    if cur:
        terms.append(_mk_term(sign, cur))
    return terms


def _mk_term(sign: int, toks: List[str]) -> Term:
    text = "".join(toks)
    coef, basis = _classify(text)
    return Term(sign, text, coef, basis)


def join_terms(terms: List[Term]) -> str:
    s = "".join(t.render() for t in terms if not (t.basis is not None and t.coef == 0.0))
    return s or "0"


def add_to_expr(expr: str, value, basis: str = "const") -> str:
    """expr + value*basis; numeric coefficients of the same basis are merged."""
    terms = split_terms(expr)
    if isinstance(value, (int, float)):
        add_terms = [Term(1 if value >= 0 else -1, "", abs(float(value)), basis)]
    else:
        v = str(value).strip()
        if basis != "const":
            v = f"({v}){_basis_text(basis)}" if split_terms(v).__len__() > 1 else f"{v}{_basis_text(basis)}"
        add_terms = split_terms(v)
    for a in add_terms:
        if a.basis is not None:
            hit = next((t for t in terms if t.basis == a.basis), None)
            if hit is not None:
                c = hit.sign * hit.coef + a.sign * a.coef
                hit.sign, hit.coef = (1 if c >= 0 else -1), abs(c)
                continue
        terms.append(a)
    return join_terms(terms)


def set_coef(expr: str, value: float, basis: str) -> str:
    terms = split_terms(expr)
    hit = next((t for t in terms if t.basis == basis), None)
    if hit is None:
        terms.append(Term(1 if value >= 0 else -1, "", abs(value), basis))
    else:
        hit.sign, hit.coef = (1 if value >= 0 else -1), abs(value)
    return join_terms(terms)


def get_coef(expr: str, basis: str) -> float:
    return sum(t.sign * t.coef for t in split_terms(expr) if t.basis == basis)


def scale_expr(expr: str, factor: float, basis: Optional[str] = None) -> str:
    terms = split_terms(expr)
    if basis is not None:
        for t in terms:
            if t.basis == basis:
                c = t.sign * t.coef * factor
                t.sign, t.coef = (1 if c >= 0 else -1), abs(c)
        return join_terms(terms)
    if all(t.basis is not None for t in terms):
        for t in terms:
            c = t.sign * t.coef * factor
            t.sign, t.coef = (1 if c >= 0 else -1), abs(c)
        return join_terms(terms)
    return f"{_fmt_coef(factor)}*({expr.lstrip('+')})"


def negate_expr(expr: str) -> str:
    terms = split_terms(expr)
    for t in terms:
        t.sign = -t.sign
    return join_terms(terms)


# ---------------------------------------------------------------------------------------------------------
# commands of a TDB text with their byte spans
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Cmd:
    start: int
    end: int                  # exclusive; text[end-1] == '!'
    kind: str                 # PARAMETER / FUNCTION / other keyword
    word: str                 # keyword as written (PAR, PARAMETER, FUNCTION, ...)
    name: str = ""            # raw parameter name (whitespace free) or FUNCTION name
    expr: Optional[Piecewise] = None
    key: Any = None           # ParamKey (parameters)
    sign: float = 1.0


def _masked(text: str) -> str:
    """Same length as text, with $-comments replaced by spaces (same rules as tdbparse)."""
    out = []
    for ln in text.splitlines(keepends=True):
        body = ln.rstrip("\r\n")
        nl = ln[len(body):]
        stripped = body.strip()
        if stripped.startswith("$"):
            out.append(" " * len(body) + nl)
            continue
        if "$" in body and body.count("'") % 2 == 0 and "'" not in body.split("$", 1)[0]:
            i = body.index("$")
            body = body[:i] + " " * (len(body) - i)
        out.append(body + nl)
    return "".join(out)


def scan_commands(text: str, tlims: Tuple[float, float] = (298.15, 6000.0)) -> List[Cmd]:
    m = _masked(text)
    cmds, prev = [], 0
    for e in [i for i, ch in enumerate(m) if ch == "!"]:
        seg = m[prev:e]
        if seg.strip():
            start = prev + (len(seg) - len(seg.lstrip()))
            body = m[start:e]
            words = body.split()
            kw = resolve_keyword(words[0]) or words[0].upper()
            c = Cmd(start, e + 1, kw, words[0])
            try:
                if kw == "PARAMETER":
                    rest = body.split(None, 1)[1]
                    close = rest.index(")")
                    c.name = re.sub(r"\s+", "", rest[:close + 1])
                    c.expr = parse_piecewise(rest[close + 1:], *tlims)
                    c.key, c.sign, _ = parse_param_name(c.name)
                elif kw == "FUNCTION":
                    c.name = words[1].upper().rstrip("#")
                    c.expr = parse_piecewise(body.split(None, 2)[2], *tlims)
            except (ValueError, IndexError, ExpressionError):
                c.kind = "UNPARSED"
            cmds.append(c)
        prev = e + 1
    return cmds


def _key_t(k) -> Tuple:
    return (k.ptype, k.phase, k.constituents, k.order, k.diffusing)


class TDBText:
    """A TDB file as text + located commands; edits replace single command spans."""

    def __init__(self, text: str, path: str = "<text>"):
        self.text = text
        self.path = path
        self.db = parse_tdb(text, is_text=True, name=path)
        self.tlims = self.db.temp_limits
        self.cmds = scan_commands(text, self.tlims)
        self.pairs: List[Tuple[str, str]] = []      # (old, new) text replacements made, in order

    @classmethod
    def read(cls, path: str) -> "TDBText":
        return cls(Path(path).read_text(encoding="utf-8", errors="replace"), str(path))

    def refresh(self):
        self.db = parse_tdb(self.text, is_text=True, name=self.path)
        self.cmds = scan_commands(self.text, self.tlims)

    # -- lookup -------------------------------------------------------------------------------------
    def functions(self) -> Dict[str, Cmd]:
        return {c.name: c for c in self.cmds if c.kind == "FUNCTION"}

    def parameters(self) -> List[Cmd]:
        return [c for c in self.cmds if c.kind == "PARAMETER"]

    def _reduced(self, key):
        ph = self.db.phases.get(key.phase)
        if ph is None or len(key.constituents) != ph.n_sublattices:
            return _key_t(key)
        return _key_t(key.without_sublattices(ph.va_only_sublattices()))

    def find(self, target: str) -> Tuple[Cmd, float, List[str]]:
        """(command, sign, notes). sign converts the user's spelling to the file's: file_expr = sign * user_expr."""
        t = target.strip()
        notes: List[str] = []
        if "(" not in t:
            f = self.functions().get(t.upper().rstrip("#"))
            if f is None:
                raise KeyError(f"no FUNCTION {t} in {Path(self.path).name}")
            return f, 1.0, notes
        ku, su, _ = parse_param_name(t)
        hits = [c for c in self.parameters() if _key_t(c.key) == _key_t(ku)]
        if not hits:
            red = self._reduced(ku)
            hits = [c for c in self.parameters() if self._reduced(c.key) == red or _key_t(c.key) == red]
        if not hits:
            raise KeyError(f"parameter {t} not found in {Path(self.path).name} (use op add_parameter to create it)")
        if len(hits) > 1:
            notes.append(f"{t} is defined {len(hits)} times; the last definition (the one used) was edited")
        c = hits[-1]
        return c, su * c.sign, notes

    def label(self, c: Cmd) -> str:
        return c.name if c.kind == "PARAMETER" else c.name

    # -- modification -------------------------------------------------------------------------------
    def _splice(self, a: int, b: int, seg: str):
        """text[a:b] = seg, recording an (old, new) pair whose old text is unique in the current text."""
        t = self.text
        nt = t[:a] + seg + t[b:]
        s, e = a, b

        def ok():   # old unique in the old text, new unique in the new text (so the pair can also be undone)
            old, new = t[s:e], t[s:a] + seg + t[b:e]
            return old and new and t.count(old) == 1 and nt.count(new) == 1
        while not ok() and (s > 0 or e < len(t)):
            s, e = max(0, s - 8), min(len(t), e + 8)
        self.pairs.append((t[s:e], t[s:a] + seg + t[b:e]))
        self.text = nt
        self.refresh()

    def replace(self, c: Cmd, new_cmd: Optional[str]):
        """Replace the span of command c (None: delete it together with its now empty line)."""
        if new_cmd is None:
            ls = self.text.rfind("\n", 0, c.start) + 1
            le = self.text.find("\n", c.end)
            le = len(self.text) if le < 0 else le + 1
            before, after = self.text[ls:c.start], self.text[c.end:le]
            if not before.strip() and not after.strip():
                self._splice(ls, le, "")
            else:
                self._splice(c.start, c.end, "")
        else:
            self._splice(c.start, c.end, new_cmd)

    def insert_after(self, c: Optional[Cmd], new_cmd: str):
        if c is None:
            pos = len(self.text.rstrip("\n"))
            self._splice(pos, len(self.text), "\n " + new_cmd + "\n")
        else:
            self._splice(c.end, c.end, "\n " + new_cmd)

    def render(self, c: Cmd, pw: Piecewise) -> str:
        """New text for command c with expression pw: the old expression text is substituted in place when
        possible (keeps alignment and comments), otherwise the command is re-written, one T range per line."""
        span = self.text[c.start:c.end]
        if c.expr is not None and len(c.expr.ranges) == len(pw.ranges) and \
                all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 for a, b in zip(c.expr.ranges, pw.ranges)):
            new = span
            ok = True
            for (_, _, eo), (_, _, en) in zip(c.expr.ranges, pw.ranges):
                if eo == en:
                    continue
                if new.count(eo) != 1:
                    ok = False
                    break
                new = new.replace(eo, en)
            if ok:
                lead = self.text[self.text.rfind("\n", 0, c.start) + 1:c.start]
                lines = (lead + new).split("\n")
                for i, ln in enumerate(lines):          # take padding spaces out of over-long lines
                    while len(ln) > 80:
                        runs = [m for m in re.finditer(r" {2,}", ln) if m.start() > len(lead)]
                        if not runs:
                            break
                        m = runs[-1]
                        ln = ln[:m.start()] + ln[m.start() + 1:]
                    lines[i] = ln
                new = "\n".join(lines)[len(lead):]
                chk = scan_commands(new, self.tlims)
                if len(chk) == 1 and chk[0].expr is not None and chk[0].expr.canonical() == pw.canonical() \
                        and max(len(ln) for ln in lines) <= 80:
                    return new
        return render_command(c.word, c.name, pw)


def render_command(word: str, name: str, pw: Piecewise) -> str:
    """'PAR G(...) 298.15 expr; 2180 Y expr2; 6000 N ref !' - one T range per line, wrapped at 78 columns."""
    lines = []
    for i, (lo, hi, e) in enumerate(pw.ranges):
        lines.append(f" {word} {name} {_fmt_num(lo)} {e};" if i == 0 else f"      {_fmt_num(lo)} Y {e};")
    tail = f"{_fmt_num(pw.tmax)} N" + (f" {pw.reference}" if pw.reference else "") + " !"
    if len(pw.ranges) == 1 and len(lines[0]) + 1 + len(tail) <= 78:
        lines[0] += " " + tail
    else:
        lines.append("      " + tail)
    return "\n".join(wrap_tdb_line(ln) for ln in lines)[1:]


# ---------------------------------------------------------------------------------------------------------
# edits
# ---------------------------------------------------------------------------------------------------------
OPS = ("set", "add", "scale", "remove", "add_parameter")


def _value(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        return float(re.sub(r"[Dd]", "E", s))
    except ValueError:
        tokenize(s)        # raises on garbage
        return s


def _ranges_sel(pw: Piecewise, t_range) -> List[int]:
    if t_range is None:
        return list(range(len(pw.ranges)))
    idx = [int(t_range)] if not isinstance(t_range, (list, tuple)) else [int(i) for i in t_range]
    for i in idx:
        if not 0 <= i < len(pw.ranges):
            raise ValueError(f"t_range {i} out of range (parameter has {len(pw.ranges)} T ranges, 0-based)")
    return idx


def _apply_to_pw(pw: Piecewise, op: str, value, basis: Optional[str], sign: float, sel: List[int]) -> Piecewise:
    ranges = list(pw.ranges)
    if op == "set" and basis is None and len(sel) == len(ranges):
        v = value if isinstance(value, float) else str(value)
        e = add_to_expr("0", v) if isinstance(v, float) else v
        e = e if sign > 0 else negate_expr(e)
        return Piecewise([(ranges[0][0], ranges[-1][1], e)], pw.reference)
    out = []
    for i, (lo, hi, e) in enumerate(ranges):
        if i in sel:
            if op == "add":
                if isinstance(value, float):
                    e = add_to_expr(e, sign * value, basis or "const")
                else:
                    e = add_to_expr(e, value if sign > 0 else negate_expr(value), basis or "const")
            elif op == "scale":
                if not isinstance(value, float):
                    raise ValueError("scale needs a number")
                e = scale_expr(e, value, basis)
            elif op == "set":
                if basis is not None:
                    if not isinstance(value, float):
                        raise ValueError("set with a term needs a number")
                    e = set_coef(e, sign * value, basis)
                else:
                    e = str(value) if not isinstance(value, float) else _fmt_coef_signed(value)
                    e = e if sign > 0 else negate_expr(e)
        out.append((lo, hi, e))
    return Piecewise(out, pw.reference)


def _fmt_coef_signed(x: float) -> str:
    return ("-" if x < 0 else "+") + _fmt_coef(abs(x))


def _user_value(tt: TDBText, c: Cmd, sign: float, T: Sequence[float]) -> np.ndarray:
    look = tt.db.evaluator()
    return sign * c.expr.evaluate(np.asarray(T, float), look)


def apply_edits(tt: TDBText, edits: Sequence[Dict]) -> List[Dict]:
    """Apply edits to tt (in memory). Returns one record per edit."""
    recs = []
    for k, ed in enumerate(edits):
        ed = dict(ed)
        op = str(ed.get("op", "set")).lower()
        if op not in OPS:
            raise ValueError(f"edit {k}: op must be one of {OPS}")
        target = str(ed.get("target") or ed.get("parameter") or ed.get("name") or "").strip()
        if not target:
            raise ValueError(f"edit {k}: missing target")
        term = ed.get("term")
        basis = None if term in (None, "", "all") else _basis_key(term)
        rec = {"target": target, "op": op, "value": ed.get("value"), "term": basis, "notes": []}
        if op == "add_parameter":
            if ed.get("value") is None:
                raise ValueError(f"edit {k}: add_parameter needs value (expression)")
            val = _value(ed["value"])
            e = add_to_expr("0", val) if isinstance(val, float) else str(val)
            lo = float(ed.get("tmin", tt.tlims[0]))
            hi = float(ed.get("tmax", tt.tlims[1]))
            pw = Piecewise([(lo, hi, e)], str(ed.get("reference", "") or ""))
            if "(" in target:
                try:
                    tt.find(target)
                    raise ValueError(f"edit {k}: {target} already exists - use op set/add")
                except KeyError:
                    pass
                key, _, _ = parse_param_name(target)
                if key.phase not in tt.db.phases:
                    raise ValueError(f"edit {k}: phase {key.phase} is not defined in the TDB")
                name = re.sub(r"\s+", "", target).upper()
                same = [c for c in tt.parameters() if c.key.phase == key.phase]
                anchor = same[-1] if same else (tt.parameters()[-1] if tt.parameters() else None)
                new = render_command("PARAMETER", name, pw)
            else:
                name = target.upper()
                if name in tt.functions():
                    raise ValueError(f"edit {k}: FUNCTION {name} already exists - use op set/add")
                fns = list(tt.functions().values())
                anchor = fns[-1] if fns else None
                new = render_command("FUNCTION", name, pw)
            tt.insert_after(anchor, new)
            rec.update(old=None, new=pw, file_name=name, kind="PARAMETER" if "(" in target else "FUNCTION")
            recs.append(rec)
            continue
        c, sign, notes = tt.find(target)
        rec["notes"] += notes
        rec.update(file_name=c.name, kind=c.kind, sign=sign, old=c.expr)
        if sign < 0:
            rec["notes"].append(f"{target} is written as {c.name} in the file: odd Redlich-Kister order, sign flipped")
        if op == "remove":
            if c.kind == "FUNCTION":
                users = [p.name for p in tt.cmds if p.expr is not None and c.name in p.expr.symbols()]
                if users:
                    raise ValueError(f"edit {k}: FUNCTION {c.name} is used by {users[:5]} - cannot remove")
            tt.replace(c, None)
            rec["new"] = None
            recs.append(rec)
            continue
        if ed.get("value") is None:
            raise ValueError(f"edit {k}: op {op} needs value")
        val = _value(ed["value"])
        sel = _ranges_sel(c.expr, ed.get("t_range"))
        pw = _apply_to_pw(c.expr, op, val, basis, sign, sel)
        try:
            pw.to_tdb()
            for _, _, e in pw.ranges:
                tokenize(e)
        except ExpressionError as e:
            raise ValueError(f"edit {k}: resulting expression is invalid: {e}") from None
        tt.replace(c, tt.render(c, pw))
        rec["new"] = pw
        recs.append(rec)
    return recs


def _twin_warnings(orig: TDBText, recs: List[Dict]) -> List[str]:
    """Parameters identical to an edited one in another phase (e.g. A1_FCC = disordered part of L12_FCC)."""
    out = []
    edited = {r["file_name"] for r in recs if r.get("kind") == "PARAMETER"}
    by_name = {c.name: c for c in orig.parameters()}
    for r in recs:
        if r.get("kind") != "PARAMETER" or r.get("old") is None:
            continue
        c = by_name.get(r["file_name"])
        if c is None:
            continue
        k = c.key
        for o in orig.parameters():
            if o.name in edited or o.key.phase == k.phase:
                continue
            if (o.key.ptype, o.key.constituents, o.key.order) == (k.ptype, k.constituents, k.order) and \
                    o.expr.canonical_exprs() == c.expr.canonical_exprs():
                out.append(f"{o.name} is identical to {c.name} before the edit (e.g. a disordered-part twin); "
                           f"edit it too if both should change")
    return out


def edit_pairs(text: str, edits: Sequence[Dict], update_twins: bool = True, path: str = "<text>"
               ) -> Tuple[List[Tuple[str, str]], str]:
    """(old, new) text replacements that turn text into the edited text (applied in order), and that text."""
    tt = TDBText(text, path)
    if update_twins:
        edits = list(edits) + twin_edits(tt, edits)[0]
    apply_edits(tt, edits)
    return tt.pairs, tt.text


def apply_pairs(text: str, pairs: Sequence[Tuple[str, str]]) -> str:
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError(f"replacement source not unique/found: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def _diff(a: str, b: str, name: str) -> str:
    d = difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=f"{name} (original)",
                             tofile=f"{name} (modified)", n=0, lineterm="")
    return "\n".join(d)


def disordered_parts(tt: TDBText) -> Dict[str, str]:
    """{disordered phase: ordered phase} from DIS_PART type definitions."""
    out = {}
    for td in tt.db.typedefs.values():
        if td.kind == "DIS_PART":
            words = td.text.upper().split()
            try:
                i = next(k for k, w in enumerate(words) if w.startswith("DIS_P"))
                ordered = td.target_phase
                if ordered and ordered != "@" and i + 1 < len(words):
                    out[words[i + 1]] = ordered
            except StopIteration:
                continue
    return out


def twin_edits(tt: TDBText, edits: Sequence[Dict]) -> Tuple[List[Dict], List[str]]:
    """Copies of parameter edits for identical parameters of disordered-part phases (FCC_A1 -> A1_FCC of
    L12_FCC): without them the ordered phase's disordered state keeps the old description and can take over."""
    dis = disordered_parts(tt)
    extra, notes = [], []
    named = set()
    for ed in edits:
        try:
            named.add(tt.find(str(ed.get("target", "")))[0].name)
        except Exception:  # noqa: BLE001
            pass
    for ed in edits:
        tgt = str(ed.get("target", ""))
        if "(" not in tgt or str(ed.get("op", "set")).lower() not in ("set", "add", "scale", "remove"):
            continue
        try:
            c, _, _ = tt.find(tgt)
        except Exception:  # noqa: BLE001
            continue
        k = c.key
        ph = tt.db.phases.get(k.phase)
        for o in tt.parameters():
            if o.key.phase == k.phase or o.key.phase not in dis or o.name in named:
                continue
            oph = tt.db.phases.get(o.key.phase)
            if ph is None or oph is None or oph.site_ratios != ph.site_ratios:
                continue                    # same site ratios: FCC_A1 ~ A1_FCC, not HCP_A3 ~ A1_FCC
            if (o.key.ptype, o.key.constituents, o.key.order, o.key.diffusing) == \
                    (k.ptype, k.constituents, k.order, k.diffusing) and \
                    o.expr.canonical_exprs() == c.expr.canonical_exprs() and o.sign == c.sign:
                m = re.match(r"^\s*([A-Za-z0-9_]+\s*\()\s*([^,&]+)", tgt)
                twin_target = m.group(1) + o.key.phase + tgt[m.end():] if m else o.name
                extra.append({**ed, "target": twin_target})
                named.add(o.name)
                notes.append(f"also applied to {twin_target} (disordered part of {dis[o.key.phase]}, identical "
                             f"to {c.name} before the edit); update_twins=false edits only the named parameter")
    return extra, notes


def edit_tdb(tdb: str, edits: Sequence[Dict], out_tdb: str, in_place: bool = False,
             eval_T: Sequence[float] = (298.15, 1000.0, 1500.0, 2000.0), update_twins: bool = True) -> Dict:
    """Apply edits to a TDB file and write the result. Returns records, diff, warnings and value changes."""
    orig = TDBText.read(tdb)
    tt = TDBText(orig.text, tdb)
    tnotes: List[str] = []
    if update_twins:
        extra, tnotes = twin_edits(orig, edits)
        edits = list(edits) + extra
    recs = apply_edits(tt, edits)
    warns = tnotes + [n for r in recs for n in r["notes"]] + _twin_warnings(orig, recs)
    # values before/after (user's sign convention) at a few temperatures
    table = []
    for r in recs:
        row = {"target": r["target"], "op": r["op"], "term": r["term"] or "", "value": r["value"],
               "old": r["old"].short(90) if r.get("old") is not None else "(none)",
               "new": r["new"].short(90) if r.get("new") is not None else "(removed)"}
        s = r.get("sign", 1.0)
        for T in eval_T:
            try:
                before = float(s * r["old"].evaluate(np.array([T]), orig.db.evaluator())[0]) if r.get("old") else 0.0
            except ExpressionError:
                before = float("nan")
            try:
                after = float(s * r["new"].evaluate(np.array([T]), tt.db.evaluator())[0]) if r.get("new") else 0.0
            except ExpressionError:
                after = float("nan")
            row[f"d@{T:g}K"] = after - before
        table.append(row)
    target = tdb if in_place else out_tdb
    if Path(target).resolve() == Path(tdb).resolve() and not in_place:
        raise ValueError("output would overwrite the original; set in_place=true to do that")
    diff = _diff(orig.text, tt.text, Path(tdb).name)
    # validate with pycalphad before writing
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Database.from_string(tt.text, fmt="tdb")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"the edited TDB does not load in pycalphad ({type(e).__name__}: {e}); nothing written")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(tt.text, encoding="utf-8")
    return {"tdb": target, "records": recs, "table": table, "diff": diff, "warnings": warns, "pairs": tt.pairs,
            "original_text": orig.text, "new_text": tt.text}


# ---------------------------------------------------------------------------------------------------------
# free parameters (symbols or coefficients) and the ESPEI objective
# ---------------------------------------------------------------------------------------------------------
@dataclass
class FreeParam:
    label: str            # "VV0001", "L(FCC_A1,CR,NI:VA;0)" or "L(FCC_A1,CR,NI:VA;0)@T"
    target: str           # parameter / FUNCTION name (user spelling)
    basis: str            # const, T, TLNT, ... ('const' for symbols)
    symbol: str           # name fitted by ESPEI (the FUNCTION itself, or a temporary DVnnnn)
    x0: float             # current value (user sign convention)
    typ: float            # typical magnitude of this kind of coefficient
    direct: bool          # True: symbol is a constant FUNCTION of the TDB (value = x); False: delta (value = x - x0)

    @property
    def scale(self) -> float:
        return max(abs(self.x0), self.typ)

    @property
    def unit(self) -> str:
        return {"const": "J/mol", "T": "J/mol/K", "TLNT": "J/mol/K", "T2": "J/mol/K2", "T3": "J/mol/K3",
                "TINV": "J K/mol", "LNT": "J/mol"}.get(self.basis, "")


def _parse_spec(spec) -> Tuple[str, Optional[str]]:
    if isinstance(spec, dict):
        return str(spec.get("target") or spec.get("name")), spec.get("term")
    s = str(spec).strip()
    if "@" in s:
        t, term = s.rsplit("@", 1)
        return t.strip(), term.strip()
    return s, None


def _symbol_kind(tt: TDBText, name: str) -> str:
    """Basis a symbol multiplies in the parameters that use it (for its typical magnitude)."""
    for c in tt.cmds:
        if c.expr is None or c.name == name:
            continue
        for _, _, e in c.expr.ranges:
            if name not in e.upper():
                continue
            for t in split_terms(e):
                if name in re.findall(r"[A-Z_][A-Z0-9_]*", t.text.upper()):
                    rest = re.sub(rf"(^|\*){name}(\*|$)", r"\1\2", t.text.upper()).strip("*")
                    rest = re.sub(r"^\d+\.?\d*(E[+-]?\d+)?\*?", "", rest)
                    _, b = _classify(rest) if rest else (1.0, "const")
                    return b or "const"
    return "const"


def phase_interaction_specs(tt: TDBText, phase: str, terms: Sequence[str] = ("const", "T")) -> List[str]:
    """'L(PHASE,...)' (and '@T' where the T coefficient is non-zero) for every interaction parameter of a phase."""
    out = []
    ph = phase.upper()
    for c in tt.parameters():
        k = c.key
        if k.phase != ph or k.ptype != "G" or not any(len(sl) > 1 for sl in k.constituents):
            continue
        e = c.expr.ranges[0][2]
        for t in terms:
            b = _basis_key(t)
            if b == "const" or abs(get_coef(e, b)) > 0:
                out.append(c.name + ("" if b == "const" else f"@{b}"))
    return out


def resolve_free_params(tt: TDBText, specs=None, phase: Optional[str] = None) -> List[FreeParam]:
    specs = list(specs or [])
    if phase:
        specs += phase_interaction_specs(tt, phase)
    if not specs:
        specs = sorted(n for n in tt.functions() if SYMBOL_RE.match(n))
        if not specs:
            raise ValueError("no parameters given and the TDB has no VV symbols: pass parameters (e.g. "
                             "'L(FCC_A1,CR,NI:VA;0)', '...@T') or phase")
    taken = set(tt.functions())
    out: List[FreeParam] = []
    k = 0
    for spec in specs:
        target, term = _parse_spec(spec)
        basis = _basis_key(term)
        c, sign, _ = tt.find(target)
        label = target + ("" if basis == "const" else f"@{basis}")
        if any(f.label == label for f in out):
            continue
        if c.kind == "FUNCTION" and basis == "const" and len(c.expr.ranges) == 1:
            terms = split_terms(c.expr.ranges[0][2])
            if len(terms) == 1 and terms[0].basis == "const":
                x0 = terms[0].sign * terms[0].coef
                kind = _symbol_kind(tt, c.name)
                out.append(FreeParam(label, c.name, "const", c.name, x0, _typical(kind), True))
                continue
        while True:
            k += 1
            sym = f"DV{k:04d}"
            if sym not in taken:
                break
        e0 = c.expr.ranges[0][2]
        x0 = sign * get_coef(e0, basis)
        out.append(FreeParam(label, target, basis, sym, x0, _typical(basis), False))
    return out


def parameterized_text(tt: TDBText, fps: Sequence[FreeParam], update_twins: bool = True) -> str:
    """TDB text in which every delta parameter carries its DVnnnn symbol (initial value 0)."""
    t2 = TDBText(tt.text, tt.path)
    edits, syms = [], []
    for f in fps:
        if f.direct:
            continue
        edits.append({"target": f.target, "op": "add", "value": f.symbol, "term": f.basis})
        syms.append({"target": f.symbol, "op": "add_parameter", "value": 0})
    if update_twins:
        edits += twin_edits(tt, edits)[0]
    apply_edits(t2, edits + syms)
    return t2.text


def result_edits(fps: Sequence[FreeParam], x: Sequence[float]) -> List[Dict]:
    """Numeric edits that write the values x into the ORIGINAL TDB."""
    eds = []
    for f, xi in zip(fps, x):
        if f.direct:
            if xi != f.x0:
                eds.append({"target": f.target, "op": "set", "value": float(xi)})
        elif xi != f.x0:
            eds.append({"target": f.target, "op": "add", "value": float(xi - f.x0), "term": f.basis})
    return eds


def load_datasets_dir(path: str):
    from espei.datasets import load_datasets, recursive_glob
    files = recursive_glob(str(path), "*.json")
    if not files:
        raise FileNotFoundError(f"no *.json datasets under {path}")
    return load_datasets(files)


def _read_phase_models(pm) -> Optional[Dict]:
    if pm is None or isinstance(pm, dict):
        return pm
    return json.loads(Path(pm).read_text())


def dataset_warnings(datasets) -> List[str]:
    out = []
    for d in datasets.all():
        if d.get("output") == "ZPF" or "solver" in d or str(d.get("output", "")).startswith("ACR"):
            continue
        vals = np.asarray(d.get("values", []), dtype=float)
        if vals.ndim == 3 and vals.shape[2] > 1 and vals.shape[0] * vals.shape[1] > 1:
            out.append(f"equilibrium dataset {d.get('reference', '')!r} ({d.get('output')}) has several T/P AND "
                       f"several compositions: ESPEI 0.9.1 compares it in the wrong order (objective is wrong for "
                       f"it) - split it into one-T datasets")
    return out


def count_data_points(datasets_dir: str) -> int:
    n = 0
    for f in Path(datasets_dir).rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("output") == "ZPF":
            n += sum(len(v) for v in d.get("values", []))
        else:
            n += int(np.size(np.asarray(d.get("values", []), dtype=object)))
    return n


class Objective:
    """ESPEI's negative log-likelihood as a function of free parameters x (user units, absolute values).

    * ``nll(x)``        -  -sum(get_likelihood) over ESPEI's residual objects (ESPEI's own computation)
    * ``blocks(x)``     - per residual type: residuals, sigma, whitened z = residual/sigma
    * ``zvec(x)``       - all whitened residuals (fixed length; failed evaluations -> penalty)
    * ``nll_from_z``    - sum(z^2)/2 + sum(ln sigma) + n ln(2 pi)/2 per type (== nll(x), see tests)
    """

    def __init__(self, tdb: str, datasets, fps: Sequence[FreeParam], phase_models=None, data_weights=None,
                 tdb_text: Optional[str] = None, update_twins: bool = True):
        from espei.error_functions.context import setup_context
        from pycalphad import Database
        self.tt = TDBText(tdb_text, tdb) if tdb_text is not None else TDBText.read(tdb)
        self.fps = list(fps)
        text = parameterized_text(self.tt, self.fps, update_twins)
        syms = [f.symbol for f in self.fps]
        if not syms:           # ESPEI needs >= 1 degree of freedom: an unused dummy symbol
            text = TDBText(text, tdb)
            apply_edits(text, [{"target": "DV9999", "op": "add_parameter", "value": 0}])
            text, syms = text.text, ["DV9999"]
        self.text = text
        if isinstance(datasets, (str, Path)):
            datasets = load_datasets_dir(str(datasets))
        self.datasets = datasets
        self.warnings = dataset_warnings(datasets)
        ptt = TDBText(text, tdb)
        for dis, ordered in disordered_parts(ptt).items():
            hit = sorted({f.symbol for f in self.fps for c in ptt.parameters() if c.key.phase == dis
                          and f.symbol in c.expr.symbols()})
            if hit:
                self.warnings.append(
                    f"{', '.join(hit)} also enter {dis}, the disordered part of {ordered}: pycalphad 0.11 builds "
                    f"{ordered} with the database value of fitted symbols in its disordered part, so ESPEI's "
                    f"objective sees {ordered} at the start value (edit_parameters / fresh files are exact)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dbf = Database.from_string(text, fmt="tdb")
            self.ctx = setup_context(self.dbf, datasets, syms, data_weights=data_weights,
                                     phase_models=_read_phase_models(phase_models))
        self.order = list(self.ctx["symbols_to_fit"])
        self.pos = [self.order.index(f.symbol) for f in self.fps]
        self.objs = [(RESIDUAL_TYPES.get(type(o).__name__, type(o).__name__), o) for o in self.ctx["residual_objs"]]
        self.x0 = np.array([f.x0 for f in self.fps], dtype=float)
        self.n_evals = 0
        b0 = self.blocks(self.x0)
        self.sizes = {k: len(v["z"]) for k, v in b0.items()}
        self.types = [k for k, n in self.sizes.items() if n > 0]
        self.analytic = False
        try:
            r = self.objs[0][1].get_likelihood(self.vec(self.x0))
            self.analytic = isinstance(r, tuple)          # ESPEI PR #268: (lnL, gradient)
        except Exception:  # noqa: BLE001
            pass

    # -- parameter vectors ------------------------------------------------------------------------
    def vec(self, x) -> np.ndarray:
        v = np.zeros(len(self.order))
        for f, p, xi in zip(self.fps, self.pos, np.atleast_1d(np.asarray(x, dtype=float))):
            v[p] = xi if f.direct else xi - f.x0
        return v

    # -- ESPEI's own likelihood ---------------------------------------------------------------------
    def lnlike_by_type(self, x) -> Dict[str, float]:
        v = self.vec(x)
        out: Dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, o in self.objs:
                r = o.get_likelihood(v)
                r = r[0] if isinstance(r, tuple) else r
                out[name] = out.get(name, 0.0) + float(r)
        self.n_evals += 1
        return out

    def nll(self, x) -> float:
        return -sum(self.lnlike_by_type(x).values())

    def nll_and_grad(self, x) -> Tuple[float, np.ndarray]:
        """Analytic -lnL and gradient (ESPEI with PR #268 only)."""
        v = self.vec(x)
        f, g = 0.0, np.zeros(len(self.order))
        for _, o in self.objs:
            lk, gr = o.get_likelihood(v)
            f -= float(lk)
            if len(np.atleast_1d(gr)):
                g -= np.asarray(gr, dtype=float)
        return f, g[self.pos]

    # -- residuals --------------------------------------------------------------------------------------
    def blocks(self, x) -> Dict[str, Dict[str, np.ndarray]]:
        v = self.vec(x)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, o in self.objs:
                r, w = o.get_residuals(v)
                r = np.asarray(r, dtype=float).ravel()
                w = np.asarray(w, dtype=float).ravel()
                sig = 1000.0 / float(getattr(o, "weight", 1.0)) / w if name == "ZPF" else w
                b = out.setdefault(name, {"r": np.array([]), "sigma": np.array([]), "z": np.array([])})
                b["r"] = np.concatenate([b["r"], r])
                b["sigma"] = np.concatenate([b["sigma"], sig])
                b["z"] = np.concatenate([b["z"], r / sig])
        self.n_evals += 1
        return out

    @staticmethod
    def nll_from_blocks(bl) -> Dict[str, float]:
        out = {}
        for k, b in bl.items():
            z, s = b["z"], b["sigma"]
            if not np.all(np.isfinite(z)):
                out[k] = float("inf")
            else:
                out[k] = float(0.5 * np.sum(z ** 2) + np.sum(np.log(s)) + 0.5 * len(z) * LN2PI)
        return out

    def zvec(self, x, penalty: float = 1e3) -> np.ndarray:
        bl = self.blocks(x)
        parts = []
        for k in self.types:
            z = bl.get(k, {"z": np.array([])})["z"]
            if len(z) != self.sizes[k] or not np.all(np.isfinite(z)):
                z = np.full(self.sizes[k], penalty)       # failed calculation (ESPEI: lnL = -inf)
            parts.append(z)
        return np.concatenate(parts) if parts else np.zeros(0)


# ---------------------------------------------------------------------------------------------------------
# diagram quantities (mapping) and before/after comparison
# ---------------------------------------------------------------------------------------------------------
def _el(s: str) -> str:
    return str(s).strip().upper()


def run_map(tdb: str, elements: Sequence[str], out_png: str, tmin=None, tmax=None, temperature=None,
            temperature_unit: str = "K", temperature_step: float = 10.0, composition_step: Optional[float] = None,
            phases=None, time_limit_s: float = 600.0) -> Dict:
    from .mapping_tools import map_phase_diagram
    return map_phase_diagram(tdb, list(elements), out_png, tmin=tmin, tmax=tmax, temperature=temperature,
                             temperature_unit=temperature_unit, temperature_step=temperature_step,
                             composition_step=composition_step, phases=phases, time_limit_s=time_limit_s,
                             tielines=0)


def _read_rows(path: str) -> List[Dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def boundary_lines(res: Dict) -> List[Dict]:
    """[{line, region, phase, T: array K, X: {el: array}}] from a map_phase_diagram result."""
    rows = _read_rows(res["files"]["boundaries_csv"])
    els = res["elements"]
    groups: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        g = groups.setdefault((r["line"], r["phase"]), {"line": r["line"], "region": r["region"],
                                                        "phase": r["phase"], "T": [], "X": {e: [] for e in els}})
        g["T"].append(float(r["T_K"]))
        for e in els:
            g["X"][e].append(float(r[f"X({e})"]))
    out = []
    for g in groups.values():
        g["T"] = np.asarray(g["T"])
        g["X"] = {e: np.asarray(v) for e, v in g["X"].items()}
        out.append(g)
    return out


def invariants_of(res: Dict) -> List[Dict]:
    """De-duplicated invariants with a stable key ('BCC_A2+FCC_A1+LIQUID' plus '#2' for repeats)."""
    out: List[Dict] = []
    for inv in res.get("invariants", []):
        ph = "+".join(sorted(inv["phases"]))
        if any(o["phases_key"] == ph and abs(o["T_K"] - inv["T_K"]) < 0.5 for o in out):
            continue
        n = sum(o["phases_key"] == ph for o in out)
        out.append({**inv, "phases_key": ph, "key": ph + (f"#{n + 1}" if n else "")})
    return out


def boundaries_at(lines: List[Dict], T: float, el: str) -> List[Dict]:
    """Phase-boundary compositions X(el) where the mapped ZPF lines cross temperature T (binary)."""
    pts = []
    for ln in lines:
        Ts, Xs = ln["T"], ln["X"][el]
        for i in range(len(Ts) - 1):
            a, b = Ts[i], Ts[i + 1]
            if a == b or not (min(a, b) <= T <= max(a, b)):
                continue
            f = (T - a) / (b - a)
            pts.append({"region": ln["region"], "phase": ln["phase"], "X": float(Xs[i] + f * (Xs[i + 1] - Xs[i]))})
    pts.sort(key=lambda p: (p["region"], p["phase"], p["X"]))
    uniq: List[Dict] = []
    for p in pts:                           # the same crossing found from two lines / segment ends
        if uniq and uniq[-1]["region"] == p["region"] and uniq[-1]["phase"] == p["phase"] and \
                abs(uniq[-1]["X"] - p["X"]) < 1e-4:
            continue
        uniq.append(p)
    counts: Dict[Tuple[str, str], int] = {}
    for p in uniq:
        k = (p["region"], p["phase"])
        counts[k] = counts.get(k, 0) + 1
        p["key"] = f"{p['region']}|{p['phase']}" + (f"#{counts[k]}" if counts[k] > 1 else "")
    return uniq


def map_quantities(res: Dict, boundary_T: Sequence[float] = ()) -> Dict[str, float]:
    """Scalar quantities of a binary map: invariant T and phase compositions, boundary X at given T (K)."""
    els = res["elements"]
    b = els[1]
    q: Dict[str, float] = {}
    for inv in invariants_of(res):
        q[f"T_inv[{inv['key']}]"] = float(inv["T_K"])
        for ph, comp in inv["compositions"].items():
            q[f"X({b})_inv[{inv['key']}]:{ph}"] = float(comp[b])
    if boundary_T:
        lines = boundary_lines(res)
        for T in boundary_T:
            for p in boundaries_at(lines, float(T), b):
                q[f"X({b})@{float(T):g}K[{p['key']}]"] = p["X"]
    return q


def compare_maps(r0: Dict, r1: Dict, out_png: str, boundary_T: Sequence[float] = (), title: str = "",
                 temperature_unit: str = "K") -> Dict:
    """Overlay original (dashed) and modified (solid) diagrams; shifts of invariants and boundaries."""
    from .mapping_tools import phase_color
    els = r0["elements"]
    kind = r0["kind"]
    tern = kind == "ternary"
    off = KELVIN if str(temperature_unit).upper().startswith("C") else 0.0
    L0, L1 = boundary_lines(r0), boundary_lines(r1)
    fig = plt.figure(figsize=(7.4, 5.6) if not tern else (7.2, 6.2))
    if tern:
        from pycalphad.plot import triangular  # noqa: F401
        ax = fig.add_subplot(projection="triangular")
    else:
        ax = fig.add_subplot()
    phases = set()
    for lines, ls, lw, alpha in ((L0, (0, (4, 3)), 1.2, 0.8), (L1, "-", 1.4, 1.0)):
        for ln in lines:
            c = phase_color(ln["phase"])
            phases.add(ln["phase"])
            if tern:
                ax.plot(ln["X"][els[1]], ln["X"][els[2]], ls=ls, lw=lw, color=c, alpha=alpha)
            else:
                ax.plot(ln["X"][els[1]], ln["T"] - off, ls=ls, lw=lw, color=c, alpha=alpha)
    for res, ls in ((r0, (0, (4, 3))), (r1, "-")):
        for inv in invariants_of(res):
            xs = [inv["compositions"][p][els[1]] for p in inv["phases"]]
            if tern:
                ys = [inv["compositions"][p][els[2]] for p in inv["phases"]]
                ax.plot(xs + xs[:1], ys + ys[:1], color="#d62728", ls=ls, lw=1.0)
            else:
                ax.plot([min(xs), max(xs)], [inv["T_K"] - off] * 2, color="#d62728", ls=ls, lw=1.0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=phase_color(p), lw=2, label=p) for p in sorted(phases)]
    handles += [Line2D([], [], color="k", ls=(0, (4, 3)), label="original"),
                Line2D([], [], color="k", label="modified")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    if tern:
        ax.set_xlabel(f"X({els[1]})")
        ax.set_ylabel(f"X({els[2]})")
    else:
        ax.set_xlabel(f"Mole fraction {els[1]}")
        ax.set_ylabel("Temperature (°C)" if off else "Temperature (K)")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
    ax.set_title(title or f"{'-'.join(els)}: original (dashed) vs modified (solid)", fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # shifts
    I0 = {i["key"]: i for i in invariants_of(r0)}
    I1 = {i["key"]: i for i in invariants_of(r1)}
    inv_rows = []
    for k in list(I0) + [k for k in I1 if k not in I0]:
        a, b = I0.get(k), I1.get(k)
        side = b or a
        row = {"invariant": k, "type": side.get("type", ""),
               "compositions": "; ".join(f"{ph} (" + ", ".join(f"{e} {side['compositions'][ph][e]:.4f}"
                                                              for e in els[1:]) + ")" for ph in side["phases"]),
               "T0_K": a["T_K"] if a else None, "T1_K": b["T_K"] if b else None,
               "dT_K": (b["T_K"] - a["T_K"]) if a and b else None}
        for ph in (a or b)["phases"]:
            for e in els[1:]:
                x0 = a["compositions"][ph][e] if a and ph in a["compositions"] else None
                x1 = b["compositions"][ph][e] if b and ph in b["compositions"] else None
                row[f"dX({e}):{ph}"] = (x1 - x0) if x0 is not None and x1 is not None else None
        inv_rows.append(row)
    bnd_rows = []
    if not tern:
        for T in boundary_T:
            Tk = float(T) + off
            p0 = {p["key"]: p for p in boundaries_at(L0, Tk, els[1])}
            p1 = {p["key"]: p for p in boundaries_at(L1, Tk, els[1])}
            for k in list(p0) + [k for k in p1 if k not in p0]:
                a, b = p0.get(k), p1.get(k)
                bnd_rows.append({"T": float(T), "boundary": k, "X0": a["X"] if a else None,
                                 "X1": b["X"] if b else None,
                                 "dX": (b["X"] - a["X"]) if a and b else None})
    reg0 = {r.get("region") for r in r0.get("regions", [])}
    reg1 = {r.get("region") for r in r1.get("regions", [])}
    return {"png": out_png, "kind": kind, "invariants": inv_rows, "boundaries": bnd_rows,
            "regions_new": sorted(reg1 - reg0), "regions_gone": sorted(reg0 - reg1)}


def _fmtv(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:+.{nd}g}" if isinstance(v, float) else str(v)


def compare_text(cmp: Dict, temperature_unit: str = "K") -> str:
    off = KELVIN if str(temperature_unit).upper().startswith("C") else 0.0
    u = "°C" if off else "K"
    if cmp.get("kind") == "ternary":
        L = ["Three-phase triangles (original -> modified):"]
        for r in cmp["invariants"]:
            state = ("new" if r["T0_K"] is None else "gone" if r["T1_K"] is None else "shifted")
            dx = ", ".join(f"{k.split(':', 1)[1]} {k.split(':', 1)[0]} {_fmtv(v)}" for k, v in r.items()
                           if k.startswith("dX(") and v is not None)
            L.append(f"  {r['invariant']} {state}: {r['compositions']}" + (f"; {dx}" if dx else ""))
        return "\n".join(L + [f"new phase regions: {', '.join(cmp['regions_new'])}"] * bool(cmp["regions_new"])
                         + [f"phase regions gone: {', '.join(cmp['regions_gone'])}"] * bool(cmp["regions_gone"]))
    L = ["Invariants (original -> modified):"]
    for r in cmp["invariants"]:
        t0 = "absent" if r["T0_K"] is None else f"{r['T0_K'] - off:.2f}"
        t1 = "absent" if r["T1_K"] is None else f"{r['T1_K'] - off:.2f}"
        dx = ", ".join(f"{k.split(':', 1)[1]} {k.split(':', 1)[0]} {_fmtv(v)}" for k, v in r.items()
                       if k.startswith("dX(") and v is not None)
        L.append(f"  {r['invariant']} {r['type']}: {t0} -> {t1} {u}" +
                 (f" (dT {r['dT_K']:+.2f} K)" if r["dT_K"] is not None else "") + (f"; {dx}" if dx else ""))
    if cmp["boundaries"]:
        L.append("Phase boundaries at the requested temperatures:")
        for r in cmp["boundaries"]:
            L.append(f"  {r['T']:g} {u} {r['boundary']}: {_fmtv(r['X0'])} -> {_fmtv(r['X1'])} (dX {_fmtv(r['dX'])})")
    if cmp["regions_new"]:
        L.append("new phase regions: " + ", ".join(cmp["regions_new"]))
    if cmp["regions_gone"]:
        L.append("phase regions gone: " + ", ".join(cmp["regions_gone"]))
    return "\n".join(L)


def objective_compare(tdb0: str, tdb1: str, datasets_dir: str, phase_models=None, data_weights=None) -> Dict:
    """ESPEI -lnL per residual type for two TDBs (same datasets)."""
    ds = load_datasets_dir(datasets_dir)
    a = Objective(tdb0, ds, [], phase_models, data_weights)
    b = Objective(tdb1, ds, [], phase_models, data_weights)
    la, lb = a.lnlike_by_type([]), b.lnlike_by_type([])
    ba, bb = a.blocks([]), b.blocks([])
    rows = []
    for k in la:
        n = len(ba[k]["z"])
        if n == 0 and la[k] == 0 and lb[k] == 0:
            continue
        rows.append({"type": k, "n": n, "nll0": -la[k], "nll1": -lb[k], "dnll": -(lb[k] - la[k]),
                     "rms_z0": float(np.sqrt(np.mean(ba[k]["z"] ** 2))) if n else float("nan"),
                     "rms_z1": float(np.sqrt(np.mean(bb[k]["z"] ** 2))) if n else float("nan")})
    tot0, tot1 = -sum(la.values()), -sum(lb.values())
    return {"rows": rows, "nll0": tot0, "nll1": tot1, "dnll": tot1 - tot0, "warnings": a.warnings}


def objective_text(oc: Dict) -> str:
    L = [f"ESPEI objective -lnL: {oc['nll0']:.4f} -> {oc['nll1']:.4f} (change {oc['dnll']:+.4f}; "
         f"negative = better fit)"]
    for r in oc["rows"]:
        L.append(f"  {r['type']:<18} n={r['n']:<4} {r['nll0']:.4f} -> {r['nll1']:.4f} ({r['dnll']:+.4f}); "
                 f"RMS(res/sigma) {r['rms_z0']:.3g} -> {r['rms_z1']:.3g}")
    L += [f"  warning: {w}" for w in oc.get("warnings", [])]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# sensitivities
# ---------------------------------------------------------------------------------------------------------
def perturbed_tdb(tt: TDBText, fp: FreeParam, value: float, path: str, update_twins: bool = True) -> str:
    t2 = TDBText(tt.text, tt.path)
    eds = result_edits([fp], [value])
    apply_edits(t2, eds + (twin_edits(tt, eds)[0] if update_twins else []))
    Path(path).write_text(t2.text, encoding="utf-8")
    return path


def property_quantities(tdb: str, elements: Sequence[str], conditions: Dict, props: Sequence[str],
                        phases=None, temperature_unit: str = "K") -> Dict[str, float]:
    """Equilibrium properties at ONE condition point (pycalphad Workspace)."""
    from pycalphad import Database, Workspace, variables as v
    from pycalphad.core.utils import filter_phases, unpack_species
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(tdb)
        els = sorted(_el(e) for e in elements)
        comps = els + (["VA"] if "VA" in db.elements else [])
        ph = [p.upper() for p in phases] if phases else sorted(filter_phases(db, unpack_species(db, comps)))
        conds = {v.P: 101325.0, v.N: 1.0}
        for k, val in conditions.items():
            ku = k.upper().replace(" ", "")
            if ku == "T":
                conds[v.T] = float(val) + (KELVIN if str(temperature_unit).upper().startswith("C") else 0.0)
            elif ku == "P":
                conds[v.P] = float(val)
            else:
                conds[v.X(re.sub(r"^X\(|\)$", "", ku))] = float(val)
        wks = Workspace(db, comps, ph, conds)
        vals = wks.get(*props)
    return {p: float(np.squeeze(val)) for p, val in zip(props, vals)}


def _fd_step(fp: FreeParam, rel_step: float) -> float:
    return rel_step * fp.scale


def sensitivity(tdb: str, out_prefix: str, parameters=None, phase: Optional[str] = None,
                datasets_dir: Optional[str] = None, phase_models=None, data_weights=None,
                elements: Optional[Sequence[str]] = None, tmin=None, tmax=None, temperature_unit: str = "K",
                invariants: bool = False, boundary_temperatures: Sequence[float] = (),
                conditions: Optional[Dict] = None, properties: Sequence[str] = (), phases=None,
                rel_step: float = 0.01, temperature_step: float = 10.0, fisher: bool = True,
                update_twins: bool = True, progress: Optional[Callable] = None) -> Dict:
    """Central finite-difference derivatives dQ/dp.  Returns text + files; writes CSV/PNG under out_prefix."""
    prog = progress or (lambda *a, **k: None)
    tt = TDBText.read(tdb)
    fps = resolve_free_params(tt, parameters, phase)
    n = len(fps)
    if not (datasets_dir or invariants or boundary_temperatures or properties):
        raise ValueError("choose quantities: datasets_dir (ESPEI objective), invariants / boundary_temperatures "
                         "(binary map, needs elements+tmin+tmax), or properties + conditions")
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    notes: List[str] = []
    files: Dict[str, str] = {}
    base: Dict[str, float] = {}
    D: Dict[str, np.ndarray] = {}
    J = None
    obj = None
    # ---- ESPEI objective (one context, parameters changed through symbols) ----
    if datasets_dir:
        prog("building ESPEI residual objects", 0.0)
        obj = Objective(tdb, datasets_dir, fps, phase_models, data_weights, update_twins=update_twins)
        notes += obj.warnings
        b0 = obj.blocks(obj.x0)
        nll0 = obj.nll_from_blocks(b0)
        espei0 = -sum(obj.lnlike_by_type(obj.x0).values())
        chk = sum(nll0.values())
        if abs(chk - espei0) > 1e-6 * max(1.0, abs(espei0)):
            notes.append(f"-lnL from residuals {chk:.6g} differs from ESPEI get_likelihood {espei0:.6g}")
        base["NLL_total"] = espei0
        for k in obj.types:
            base[f"NLL_{k}"] = nll0[k]
        zs0 = obj.zvec(obj.x0)
        J = np.zeros((len(zs0), n))
        for i, f in enumerate(fps):
            h = _fd_step(f, rel_step)
            vals = []
            for sgn in (1, -1):
                x = obj.x0.copy()
                x[i] += sgn * h
                bl = obj.blocks(x)
                nl = obj.nll_from_blocks(bl)
                vals.append((nl, obj.zvec(x)))
            (np_, zp), (nm_, zm) = vals
            D.setdefault("NLL_total", np.zeros(n))[i] = (sum(np_.values()) - sum(nm_.values())) / (2 * h)
            for k in obj.types:
                D.setdefault(f"NLL_{k}", np.zeros(n))[i] = (np_[k] - nm_[k]) / (2 * h)
            J[:, i] = (zp - zm) / (2 * h)
            prog(f"objective derivatives {i + 1}/{n}", (i + 1) / n * (0.5 if (invariants or boundary_temperatures
                                                                              or properties) else 1.0))
    # ---- diagram / property quantities (perturbed TDB files) ----
    use_map = bool(invariants or boundary_temperatures)
    if use_map or properties:
        if not elements:
            raise ValueError("elements are needed for invariants / boundaries / properties")
        if use_map and (len(elements) != 2 or tmin is None or tmax is None):
            raise ValueError("invariants / boundary_temperatures need a binary (2 elements) and tmin, tmax")
        off = KELVIN if str(temperature_unit).upper().startswith("C") else 0.0
        bT = [float(t) + off for t in boundary_temperatures]
        tmpd = Path(tempfile.mkdtemp(prefix="sens_", dir=str(Path(out_prefix).parent)))

        def quantities(path: str, tag: str) -> Dict[str, float]:
            q: Dict[str, float] = {}
            if use_map:
                r = run_map(path, elements, str(tmpd / f"{tag}.png"), tmin=tmin, tmax=tmax,
                            temperature_unit=temperature_unit, temperature_step=temperature_step, phases=phases)
                if not r["mapping"]["finished"]:
                    notes.append(f"mapping {tag} hit its time limit")
                mq = map_quantities(r, bT)
                if not invariants:
                    mq = {k: val for k, val in mq.items() if not k.startswith(("T_inv", f"X({_el(elements[1])})_inv"))}
                q.update(mq)
            if properties:
                q.update({f"{p}@{json.dumps(conditions, sort_keys=True)}": val for p, val in property_quantities(
                    path, elements, conditions or {}, list(properties), phases, temperature_unit).items()})
            return q
        base_path = str(tmpd / "base.tdb")
        Path(base_path).write_text(tt.text, encoding="utf-8")
        q0 = quantities(base_path, "base")
        base.update(q0)
        for i, f in enumerate(fps):
            h = _fd_step(f, rel_step)
            qp = quantities(perturbed_tdb(tt, f, f.x0 + h, str(tmpd / f"p{i}.tdb"), update_twins), f"p{i}")
            qm = quantities(perturbed_tdb(tt, f, f.x0 - h, str(tmpd / f"m{i}.tdb"), update_twins), f"m{i}")
            for k in q0:
                a, b = qp.get(k), qm.get(k)
                if a is not None and b is not None:
                    d = (a - b) / (2 * h)
                elif a is not None:
                    d = (a - q0[k]) / h
                elif b is not None:
                    d = (q0[k] - b) / h
                else:
                    d = float("nan")
                D.setdefault(k, np.full(n, np.nan))[i] = d
            lost = [k for k in q0 if k not in qp or k not in qm]
            if lost:
                notes.append(f"{f.label}: {len(lost)} quantities not found in a perturbed calculation "
                             f"(e.g. {lost[0]}); one-sided differences used")
            prog(f"diagram/property derivatives {i + 1}/{n}", 0.5 + 0.5 * (i + 1) / n if datasets_dir else (i + 1) / n)
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)
    # ---- table ----
    for qn, d in D.items():
        for i, f in enumerate(fps):
            eff = d[i] * f.typ
            el = d[i] * f.x0 / base[qn] if base.get(qn) not in (0, None) and f.x0 != 0 else float("nan")
            rows.append({"quantity": qn, "base_value": base.get(qn), "parameter": f.label, "parameter_value": f.x0,
                         "parameter_unit": f.unit, "step": _fd_step(f, rel_step), "derivative": d[i],
                         "typical_change": f.typ, "effect_of_typical_change": eff, "elasticity": el})
    csv_path = out_prefix + "_sensitivity.csv"
    _write_rows(csv_path, rows)
    files["csv"] = csv_path
    png = out_prefix + "_tornado.png"
    tornado(D, fps, png)
    files["tornado_png"] = png
    text = [f"{n} parameters x {len(D)} quantities; central differences, step = {rel_step:g} x max(|p|, typical)"]
    fis = None
    if J is not None and fisher:
        fis = fisher_analysis(J, fps)
        fcsv = out_prefix + "_fisher.csv"
        _write_rows(fcsv, fis["table"])
        files["fisher_csv"] = fcsv
        np.savetxt(out_prefix + "_jacobian.csv", J, delimiter=",", header=",".join(f.label for f in fps), comments="")
        files["jacobian_csv"] = out_prefix + "_jacobian.csv"
        cpng = out_prefix + "_correlation.png"
        correlation_plot(fis["corr"], fps, cpng)
        files["correlation_png"] = cpng
    elif len(D) > 1:
        fis = {"warnings": jacobian_degeneracy({k: val for k, val in D.items() if not k.startswith("NLL")}, fps)}
    text += summary_lines(D, base, fps)
    if fis:
        if "text" in fis:
            text.append(fis["text"])
        text += [f"WARNING: {w}" for w in fis.get("warnings", [])]
    text += [f"note: {x}" for x in dict.fromkeys(notes)]
    return {"text": "\n".join(text), "files": files, "D": {k: v.tolist() for k, v in D.items()}, "base": base,
            "parameters": [f.label for f in fps], "steps": [_fd_step(f, rel_step) for f in fps],
            "fisher": {k: v for k, v in (fis or {}).items() if k in ("std", "eig")}}


def _write_rows(path: str, rows: List[Dict]) -> str:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


def summary_lines(D: Dict[str, np.ndarray], base: Dict[str, float], fps: Sequence[FreeParam],
                  top: int = 4) -> List[str]:
    L = ["dQ per typical parameter change (1000 J/mol constant, 1 J/mol/K T term, 0.1 TLNT ...):"]
    for qn, d in list(D.items())[:40]:
        eff = d * np.array([f.typ for f in fps])
        order = np.argsort(-np.nan_to_num(np.abs(eff), nan=-1))[:top]
        b = base.get(qn)
        L.append(f"  {qn} = {b:.6g}: " + ", ".join(f"{fps[i].label} {eff[i]:+.4g}" for i in order))
    if len(D) > 40:
        L.append(f"  ... {len(D) - 40} more quantities in the CSV")
    return L


def fisher_analysis(J: np.ndarray, fps: Sequence[FreeParam], corr_warn: float = 0.95) -> Dict:
    """Gauss-Newton Hessian F = J^T J of -lnL (J = d(residual/sigma)/dp): std, correlations, weak directions."""
    F = J.T @ J
    s = np.array([f.scale for f in fps])
    Fs = F * np.outer(s, s)                          # dimensionless (parameters in units of their scale)
    n = len(fps)
    warn = []
    zero = [i for i in range(n) if not np.any(J[:, i])]
    for i in zero:
        warn.append(f"{fps[i].label}: no data depend on it (not identifiable from these datasets)")
    ev, evec = np.linalg.eigh(Fs) if n else (np.array([]), np.zeros((0, 0)))
    cov_s = np.linalg.pinv(Fs, rcond=1e-12) if n else np.zeros((0, 0))
    std = np.sqrt(np.clip(np.diag(cov_s), 0, None)) * s
    d = np.sqrt(np.clip(np.diag(cov_s), 1e-300, None))
    corr = cov_s / np.outer(d, d) if n else cov_s
    for i in range(n):
        for j in range(i + 1, n):
            if i in zero or j in zero:
                continue
            if abs(corr[i, j]) >= corr_warn:
                warn.append(f"{fps[i].label} and {fps[j].label} are correlated (r = {corr[i, j]:+.3f}): the data "
                            f"mostly fix a combination of them")
    if n and ev.max() > 0:
        for k in range(n):
            if ev[k] < 1e-8 * ev.max() and not any(abs(evec[i, k]) > 0.99 for i in zero):
                comp = ", ".join(f"{evec[i, k]:+.2f} {fps[i].label}" for i in np.argsort(-np.abs(evec[:, k]))[:4]
                                 if abs(evec[i, k]) > 0.05)
                warn.append(f"near-degenerate direction (eigenvalue {ev[k]:.2g} vs max {ev.max():.2g}): {comp}")
    for i in range(n):
        if i not in zero and std[i] > 10 * fps[i].scale:
            warn.append(f"{fps[i].label}: approximate std {std[i]:.3g} {fps[i].unit} > 10x its magnitude "
                        f"(poorly determined)")
    table = [{"parameter": f.label, "value": f.x0, "unit": f.unit, "approx_std": std[i],
              **{f"corr[{g.label}]": corr[i, j] for j, g in enumerate(fps)}} for i, f in enumerate(fps)]
    cond = float(ev.max() / ev[ev > 0].min()) if n and np.any(ev > 0) else float("inf")
    text = (f"Fisher/Gauss-Newton Hessian (J^T J of residual/sigma, parameters scaled by max(|p|, typical)): "
            f"condition number {cond:.3g}; approx std: " +
            ", ".join(f"{f.label} {std[i]:.3g}" for i, f in enumerate(fps)))
    return {"F": F, "std": std.tolist(), "corr": corr, "eig": ev.tolist(), "warnings": warn, "table": table,
            "text": text, "cond": cond}


def jacobian_degeneracy(D: Dict[str, np.ndarray], fps: Sequence[FreeParam], cos_warn: float = 0.995) -> List[str]:
    if not D:
        return []
    M = np.array([np.nan_to_num(v) for v in D.values()]) * np.array([f.typ for f in fps])
    norms = np.linalg.norm(M, axis=0)
    out = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if norms[i] > 0 and norms[j] > 0 and len(D) > 1:
                c = float(M[:, i] @ M[:, j] / norms[i] / norms[j])
                if abs(c) >= cos_warn:
                    out.append(f"{fps[i].label} and {fps[j].label} shift all requested quantities almost "
                               f"proportionally (cos = {c:+.4f}): they cannot be separated by these quantities")
    return out


def tornado(D: Dict[str, np.ndarray], fps: Sequence[FreeParam], png: str, max_panels: int = 6) -> str:
    names = list(D)[:max_panels]
    npan = max(1, len(names))
    ncol = 2 if npan > 1 else 1
    nrow = int(math.ceil(npan / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, max(2.2, 0.32 * len(fps) + 1.2) * nrow), squeeze=False)
    for ax in axes.ravel()[npan:]:
        ax.set_visible(False)
    typ = np.array([f.typ for f in fps])
    for ax, qn in zip(axes.ravel(), names):
        eff = np.nan_to_num(D[qn] * typ)
        order = np.argsort(np.abs(eff))
        ax.barh(range(len(fps)), eff[order], color=["#1f77b4" if e >= 0 else "#d62728" for e in eff[order]])
        ax.set_yticks(range(len(fps)))
        ax.set_yticklabels([fps[i].label for i in order], fontsize=7)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(qn if len(qn) < 60 else qn[:57] + "...", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("change for a typical parameter change", fontsize=7)
    fig.suptitle("Sensitivities: effect of +1 kJ/mol (constant) / +1 J/mol/K (T term) ...", fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png


def correlation_plot(corr: np.ndarray, fps: Sequence[FreeParam], png: str) -> str:
    n = len(fps)
    fig, ax = plt.subplots(figsize=(1.2 + 0.45 * n + 2, 1.0 + 0.45 * n))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f.label for f in fps], rotation=90, fontsize=7)
    ax.set_yticklabels([f.label for f in fps], fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("parameter correlation (from J^T J)", fontsize=9)
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------------------------------------
class _Stop(Exception):
    pass


def optimize_job(tdb: str, datasets_dir: str, out_dir: str, parameters=None, phase: Optional[str] = None,
                 method: str = "least_squares", bounds: Optional[Dict[str, Sequence]] = None,
                 prior_std=None, max_nfev: Optional[int] = None, time_limit_s: Optional[float] = None,
                 phase_models=None, data_weights=None, compare: bool = True, output_tdb: Optional[str] = None,
                 diff_step: float = 1e-3, gradient: str = "auto", elements: Optional[Sequence[str]] = None,
                 update_twins: bool = True, progress: Optional[Callable] = None) -> Dict:
    """Gradient-based minimisation of ESPEI's -lnL over free parameters.

    Plain function with the background-job signature ``target(**kwargs, progress=callable)``.
    method: 'least_squares' (scipy trf on residual/sigma, 3-point Jacobian, x_scale = parameter scale) or
    'lbfgsb' (-lnL with central-difference or analytic gradient).  bounds: {label: [lo, hi]} (null = open).
    prior_std: number = Gaussian prior (centred on the start value) with std = prior_std * max(|p|, typical),
    or {label: std}.  Returns {"text", "files", ...}; a stop (SIGTERM) or the time limit keeps the best point.
    """
    from scipy.optimize import least_squares, minimize
    prog = progress or (lambda *a, **k: None)
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(tdb).stem
    prog("building ESPEI residual objects", 0.0)
    tt = TDBText.read(tdb)
    fps = resolve_free_params(tt, parameters, phase)
    obj = Objective(tdb, datasets_dir, fps, phase_models, data_weights, update_twins=update_twins)
    n = len(fps)
    s = np.array([f.scale for f in fps])
    x0 = obj.x0.copy()
    # priors
    psd = None
    if prior_std is not None:
        if isinstance(prior_std, dict):
            psd = np.array([float(prior_std.get(f.label, prior_std.get(f.target, np.inf))) for f in fps])
        else:
            psd = float(prior_std) * s
    # bounds in scaled units u = (x - x0) / s
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    for k, (a, b) in (bounds or {}).items():
        hits = [i for i, f in enumerate(fps) if k in (f.label, f.target, f.symbol)]
        if not hits:
            raise ValueError(f"bounds for unknown parameter {k!r}; parameters are {[f.label for f in fps]}")
        for i in hits:
            if a is not None:
                lo[i] = (float(a) - x0[i]) / s[i]
            if b is not None:
                hi[i] = (float(b) - x0[i]) / s[i]
    if np.any(lo > 0) or np.any(hi < 0):
        raise ValueError("the start value lies outside the bounds")
    nll_const = {k: float(np.sum(np.log(b["sigma"])) + 0.5 * len(b["z"]) * LN2PI)
                 for k, b in obj.blocks(x0).items() if len(b["z"])}
    const = sum(nll_const.values())
    hist: List[Dict] = []
    best = {"nll": np.inf, "x": x0.copy()}
    state = {"last_report": 0.0, "nfev": 0}
    limit = float(time_limit_s) if time_limit_s else None
    cap = max_nfev or (60 * max(1, n) if method == "least_squares" else 200)

    def xof(u):
        return x0 + np.asarray(u, dtype=float) * s

    def prior_res(x):
        return (x - x0) / psd if psd is not None else np.zeros(0)

    def record(x, nll, kind):
        state["nfev"] += 1
        if nll < best["nll"]:
            best.update(nll=nll, x=np.array(x, dtype=float))
        hist.append({"eval": state["nfev"], "time_s": round(time.time() - t0, 3), "kind": kind, "nll": nll,
                     "best_nll": best["nll"], **{f.label: float(xi) for f, xi in zip(fps, x)}})
        if time.time() - state["last_report"] > 1.0:
            state["last_report"] = time.time()
            prog(f"eval {state['nfev']}: -lnL {nll:.4f} (best {best['nll']:.4f})",
                 min(0.95, state["nfev"] / max(1, cap * (2 * n + 1 if method == 'least_squares' else 2 * n + 1))),
                 nll=round(float(nll), 5), best_nll=round(float(best["nll"]), 5))
        if limit and time.time() - t0 > limit:
            raise _Stop("time limit")

    def zfun(u):
        x = xof(u)
        z = np.concatenate([obj.zvec(x), prior_res(x)])
        nll = 0.5 * float(np.sum(z ** 2)) + const
        record(x, nll, "residuals")
        return z

    def ffun(u):
        x = xof(u)
        f = obj.nll(x)
        if psd is not None:
            f += 0.5 * float(np.sum(prior_res(x) ** 2))
        f = f if np.isfinite(f) else 1e30
        record(x, f, "nll")
        return f

    use_analytic = obj.analytic and gradient in ("auto", "analytic")
    if gradient == "analytic" and not obj.analytic:
        raise ValueError("analytic gradients need an ESPEI whose get_likelihood returns (lnL, gradient) (PR #268)")

    def gfun(u):
        if use_analytic:
            x = xof(u)
            f, g = obj.nll_and_grad(x)
            if psd is not None:
                g = g + (x - x0) / psd ** 2
            return g * s
        g = np.zeros(n)
        for i in range(n):
            e = np.zeros(n)
            e[i] = diff_step
            g[i] = (ffun(u + e) - ffun(u - e)) / (2 * diff_step)
        return g

    status, res = "converged", None
    prog(f"optimizing {n} parameter(s) with {method}", 0.02)
    try:
        if method in ("least_squares", "trf"):
            res = least_squares(zfun, np.zeros(n), jac="3-point", bounds=(lo, hi), method="trf",
                                diff_step=diff_step, x_scale=1.0, max_nfev=cap, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            status = res.message
        elif method in ("lbfgsb", "l-bfgs-b", "L-BFGS-B"):
            res = minimize(ffun, np.zeros(n), jac=gfun, method="L-BFGS-B",
                           bounds=list(zip(np.where(np.isfinite(lo), lo, None), np.where(np.isfinite(hi), hi, None))),
                           options={"maxiter": cap, "ftol": 1e-12, "gtol": 1e-8})
            status = str(res.message)
        else:
            raise ValueError("method must be least_squares or lbfgsb")
    except _Stop:
        status = f"stopped at the time limit ({limit:.0f} s); best point kept"
    except KeyboardInterrupt:
        status = "stopped by the user; best point kept"
    xb = best["x"]
    # uncertainty from J^T J at the best point
    prog("Jacobian at the optimum", 0.96)
    std = np.full(n, np.nan)
    corr = None
    fis = None
    try:
        zb = obj.zvec(xb)
        J = np.zeros((len(zb), n))
        for i in range(n):
            h = diff_step * s[i]
            xp, xm = xb.copy(), xb.copy()
            xp[i] += h
            xm[i] -= h
            J[:, i] = (obj.zvec(xp) - obj.zvec(xm)) / (2 * h)
        if psd is not None:
            J = np.vstack([J, np.diag(1.0 / psd)])
        fps_b = [FreeParam(f.label, f.target, f.basis, f.symbol, float(v), f.typ, f.direct) for f, v in zip(fps, xb)]
        fis = fisher_analysis(J, fps_b)
        std = np.array(fis["std"])
        corr = fis["corr"]
    except (KeyboardInterrupt, Exception) as e:  # noqa: BLE001
        fis = {"warnings": [f"uncertainty estimate failed: {type(e).__name__}: {e}"]}
    # outputs
    files: Dict[str, str] = {}
    out_tdb = output_tdb or str(out / f"{stem}_optimized.tdb")
    r = edit_tdb(tdb, result_edits(fps, xb), out_tdb, update_twins=update_twins) if result_edits(fps, xb) else None
    if r is None:
        Path(out_tdb).write_text(tt.text, encoding="utf-8")
    files["tdb"] = out_tdb
    by0 = obj.lnlike_by_type(x0)
    by1 = obj.lnlike_by_type(xb)
    ptab = [{"parameter": f.label, "unit": f.unit, "before": f.x0, "after": float(xb[i]),
             "change": float(xb[i] - f.x0), "approx_std": float(std[i]), "symbol": f.symbol,
             "at_bound": bool(np.isfinite(lo[i]) and abs((xb[i] - x0[i]) / s[i] - lo[i]) < 1e-9
                              or np.isfinite(hi[i]) and abs((xb[i] - x0[i]) / s[i] - hi[i]) < 1e-9)}
            for i, f in enumerate(fps)]
    files["parameters_csv"] = _write_rows(str(out / f"{stem}_parameters.csv"), ptab)
    files["history_csv"] = _write_rows(str(out / f"{stem}_history.csv"), hist)
    files["history_png"] = history_plot(hist, str(out / f"{stem}_history.png"))
    if corr is not None and n > 1:
        files["correlation_png"] = correlation_plot(corr, fps, str(out / f"{stem}_correlation.png"))
    comp_txt = []
    if compare:
        prog("before/after comparison with the data", 0.97)
        try:
            comp = before_after(tdb, out_tdb, datasets_dir, str(out / stem), phase_models, data_weights, elements)
            files.update(comp["files"])
            comp_txt = comp["text"]
        except Exception as e:  # noqa: BLE001
            comp_txt = [f"comparison failed: {type(e).__name__}: {e}"]
    nll0, nll1 = -sum(by0.values()), -sum(by1.values())
    L = [f"{method}: {status}; {state['nfev']} objective evaluations, {time.time() - t0:.1f} s"
         + (" (analytic gradients)" if use_analytic and method != "least_squares" else ""),
         f"-lnL (ESPEI): {nll0:.4f} -> {nll1:.4f}" + (" [+ prior]" if psd is not None else ""),
         "  per type: " + ", ".join(f"{k} {-by0[k]:.3f} -> {-by1[k]:.3f}" for k in by0 if by0[k] or by1[k]),
         "parameter                          before          after         change      approx std"]
    for p in ptab:
        L.append(f"{p['parameter']:<32} {p['before']:>12.6g} {p['after']:>14.8g} {p['change']:>+14.6g} "
                 f"{p['approx_std']:>12.4g} {p['unit']}" + ("  AT BOUND" if p["at_bound"] else ""))
    L += [f"WARNING: {w}" for w in (fis or {}).get("warnings", [])]
    L += [f"note: {w}" for w in obj.warnings]
    L += comp_txt
    prog("done", 1.0, nll=round(nll1, 5), best_nll=round(float(best["nll"]), 5))
    return {"text": "\n".join(L), "files": list(files.values()), "file_map": files, "x": xb.tolist(),
            "x0": x0.tolist(), "labels": [f.label for f in fps], "std": std.tolist(), "nll0": nll0, "nll1": nll1,
            "status": status, "nfev": state["nfev"]}


def history_plot(hist: List[Dict], png: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if hist:
        ev = [h["eval"] for h in hist]
        ax.plot(ev, [h["nll"] for h in hist], ".", ms=3, color="0.6", label="evaluations (incl. finite differences)")
        ax.plot(ev, [h["best_nll"] for h in hist], "-", color="#1f77b4", label="best so far")
        lo = min(h["best_nll"] for h in hist)
        hi = np.nanpercentile([h["nll"] for h in hist if np.isfinite(h["nll"])] or [lo + 1], 90)
        if hi > lo:
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("-lnL (ESPEI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png


def before_after(tdb0: str, tdb1: str, datasets_dir: str, prefix: str, phase_models=None, data_weights=None,
                 elements=None) -> Dict:
    """fit_report numbers before/after (+ binary diagrams with the ZPF data, side by side)."""
    from .espei_data import fit_report
    files: Dict[str, str] = {}
    txt = []
    pm = _read_phase_models(phase_models)
    r0 = fit_report(tdb0, datasets_dir, prefix + "_before", elements=elements, phase_models=pm,
                    data_weights=data_weights, diagram=False)
    r1 = fit_report(tdb1, datasets_dir, prefix + "_after", elements=elements, phase_models=pm,
                    data_weights=data_weights, diagram=False)
    s1 = {s["file"]: s for s in r1["summary"]}
    rows = []
    for s in r0["summary"]:
        a = s1.get(s["file"], {})
        rows.append({"file": s["file"], "output": s["output"], "n": s["n"], "rms_over_sigma_before": s["rms_z"],
                     "rms_over_sigma_after": a.get("rms_z"), "rms_before": s["rms"], "rms_after": a.get("rms"),
                     "unit": s["unit"]})
    files["fit_before_after_csv"] = _write_rows(prefix + "_fit_before_after.csv", rows)
    txt.append("Data fit per dataset, RMS(residual/sigma) before -> after:")
    for r in rows[:15]:
        txt.append(f"  {r['file']} ({r['output']}, n={r['n']}): {r['rms_over_sigma_before']:.3g} -> "
                   f"{(r['rms_over_sigma_after'] if r['rms_over_sigma_after'] is not None else float('nan')):.3g}")
    systems = sorted({tuple(r["components"]) for r in r0.get("rows", []) if r["kind"] == "ZPF"})
    for sysc in systems:
        if len(sysc) == 2:
            png = prefix + f"_diagram_{'-'.join(sysc)}.png"
            try:
                diagram_with_data_pair(tdb0, tdb1, list(sysc), datasets_dir, png)
                files[f"diagram_{'-'.join(sysc)}"] = png
            except Exception as e:  # noqa: BLE001
                txt.append(f"diagram {'-'.join(sysc)} failed: {type(e).__name__}: {e}")
    return {"files": files, "text": txt}


def diagram_with_data_pair(tdb0: str, tdb1: str, elements: Sequence[str], datasets_dir: str, png: str,
                           tmin: Optional[float] = None, tmax: Optional[float] = None) -> str:
    """Before | after binary diagrams with the ZPF data (as the plot_diagram_with_data tool draws them)."""
    from espei.plot import dataplot
    from pycalphad import binplot, variables as v
    from .calc import get_db
    from .diagrams import TIELINE_COLOR, phases_for
    from .espei_plot_tools import _consistent_legend, _load_datasets
    a, b = sorted(_el(e) for e in elements)
    comps = [a, b, "VA"]
    datasets, _ = _load_datasets(datasets_dir)
    Ts = [t for d in datasets.all() if d.get("output") == "ZPF" and set(d["components"]) <= set(comps)
          for t in np.atleast_1d(d["conditions"]["T"])]
    lo = tmin if tmin is not None else (max(300.0, min(Ts) - 200) if Ts else 300.0)
    hi = tmax if tmax is not None else (max(Ts) + 200 if Ts else 2500.0)
    conds = {v.P: 101325, v.T: (lo, hi, 10), v.X(b): (0, 1, 0.02), v.N: 1}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax, t, name in ((axes[0], tdb0, "before"), (axes[1], tdb1, "after")):
            db = get_db(t)
            ph = phases_for(db, [a, b])
            data_phases = {p for d in datasets.all() for p in d.get("phases", [])}
            legend = _consistent_legend(sorted(set(ph) | data_phases))
            binplot(db, comps, ph, conds, plot_kwargs=dict(ax=ax, tielines=4, tieline_color=TIELINE_COLOR,
                                                           legend_generator=legend))
            dataplot(comps, ph, conds, datasets, ax=ax, legend_generator=legend)
            ax.set_xlim(0, 1)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{name}: {Path(t).name}", fontsize=10)
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------------------------------------
# job wrappers (JSON-safe results: {"text", "files": [...]})
# ---------------------------------------------------------------------------------------------------------
def sensitivity_job(progress: Optional[Callable] = None, **kw) -> Dict:
    r = sensitivity(progress=progress, **kw)
    return {"text": r["text"], "files": list(r["files"].values()), "file_map": r["files"], "D": r["D"],
            "base": r["base"], "parameters": r["parameters"]}


def edit_text(r: Dict, tb=None, max_diff_lines: int = 60) -> str:
    rel = tb.rel if tb is not None else str
    L = [f"wrote {rel(r['tdb'])} ({len(r['records'])} edit(s))"]
    for row in r["table"]:
        dv = ", ".join(f"{k[2:]} {v:+.6g}" for k, v in row.items() if k.startswith("d@") and np.isfinite(v))
        L.append(f"  {row['op']} {row['target']}{'@' + row['term'] if row['term'] else ''}: {row['old']}  ->  "
                 f"{row['new']}" + (f"   (change at {dv})" if dv else ""))
    d = r["diff"].splitlines()
    L.append("diff:\n" + "\n".join(d[:max_diff_lines]) + (f"\n... ({len(d) - max_diff_lines} more lines)"
                                                            if len(d) > max_diff_lines else ""))
    L += [f"warning: {w}" for w in r["warnings"]]
    return "\n".join(L)


def register(tb) -> None:  # noqa: C901 - flat list of tool definitions
    S = {"type": "string"}
    N = {"type": "number"}
    B = {"type": "boolean"}
    I = {"type": "integer"}
    STRS = {"type": "array", "items": {"type": "string"}}
    NUMS = {"type": "array", "items": {"type": "number"}}
    OBJ = {"type": "object"}
    UNIT = {"type": "string", "enum": ["K", "C"]}
    PARAMS = {**STRS, "description": "'VV0001', 'L(FCC_A1,CR,NI:VA;0)' (constant term) or '...;0)@T' (T coefficient; "
                                     "also TLNT, T2, T3, TINV). Default: all VV symbols, or all L of `phase`"}
    TWINS = {**B, "description": "also change identical disordered-part twins (A1_FCC of L12_FCC); default true"}
    EDIT = {"type": "object", "properties": {
        "target": {**S, "description": "parameter name, e.g. L(FCC_A1,CR,NI:VA;0), G(SIGMA,CR:FE:CR;0), "
                                       "TC(BCC_A2,CR,NI:VA;0), or a FUNCTION/symbol, e.g. VV0001"},
        "op": {"type": "string", "enum": list(OPS)},
        "value": {"description": "number or TDB expression"},
        "term": {**S, "description": "act on one coefficient: const, T, TLNT, T2, T3, TINV"},
        "t_range": {**I, "description": "T-range index (0-based); default all ranges"},
        "tmin": N, "tmax": N, "reference": S}, "required": ["target", "op"]}

    def _jobs():
        try:
            from . import jobs
            return jobs
        except ImportError:        # TODO hook: runs synchronously when calphad_tools.jobs is missing
            return None

    @tb.add("edit_parameters",
            "What-if edits of TDB parameters/FUNCTIONs (e.g. 'L0 of FCC Cr-Ni 2 kJ more positive'); the rest of the "
            "file stays byte-identical. ops: set, add (to the whole expression or one coefficient via term), "
            "scale, remove, add_parameter. Writes a new TDB (in_place overwrites) and returns a diff. compare: "
            "overlay original (dashed) vs modified diagram - binary {elements, tmin, tmax, boundary_temperatures} "
            "or ternary {elements, temperature} - with invariant and boundary shifts. datasets_dir: change of the "
            "ESPEI objective per data type.",
            {"tdb": S, "edits": {"type": "array", "items": EDIT}, "output_tdb": S, "in_place": B,
             "compare": {**OBJ, "description": "{elements, tmin, tmax | temperature, temperature_unit, "
                                               "boundary_temperatures, phases}"},
             "datasets_dir": S, "phase_models": S, "data_weights": OBJ, "update_twins": TWINS},
            ["tdb", "edits"])
    def edit_parameters(tdb, edits, output_tdb=None, in_place=False, compare=None, datasets_dir=None,
                        phase_models=None, data_weights=None, update_twins=True):
        t = tb._find_tdb(tdb)
        stem = Path(t).stem
        original = Path(t).read_text(encoding="utf-8", errors="replace")
        wdir = tb.outdir(None, f"whatif_{stem}")
        orig_copy = str(wdir / f"{stem}_original.tdb")
        Path(orig_copy).write_text(original, encoding="utf-8")
        out = t if in_place else tb.outp(output_tdb, f"{stem}_edited.tdb")
        r = edit_tdb(t, edits, out, in_place=bool(in_place), update_twins=update_twins is not False)
        text = [edit_text(r, tb)]
        if in_place:
            text.append(f"original kept as {tb.rel(orig_copy)}")
        images = []
        if compare:
            c = dict(compare)
            els = c.get("elements")
            if not els:
                raise ValueError("compare needs elements")
            unit = c.get("temperature_unit", "K")
            kw = dict(tmin=c.get("tmin"), tmax=c.get("tmax"), temperature=c.get("temperature"), temperature_unit=unit,
                      phases=c.get("phases"), temperature_step=c.get("temperature_step", 10.0))
            tag = "-".join(_el(e) for e in els)
            r0 = run_map(orig_copy, els, str(wdir / f"map_{tag}_original.png"), **kw)
            r1 = run_map(r["tdb"], els, str(wdir / f"map_{tag}_modified.png"), **kw)
            png = tb.outp(None, f"whatif_{stem}/compare_{tag}.png")
            cm = compare_maps(r0, r1, png, c.get("boundary_temperatures") or (), temperature_unit=unit)
            _write_rows(str(Path(png).with_suffix(".csv")), cm["invariants"] + cm["boundaries"])
            text.append(compare_text(cm, unit))
            text.append(f"overlay: {tb.rel(png)}; shifts CSV: {tb.rel(str(Path(png).with_suffix('.csv')))}")
            images.append(png)
        if datasets_dir:
            oc = objective_compare(orig_copy, r["tdb"], tb.p(datasets_dir),
                                   tb.p(phase_models) if phase_models else None, data_weights)
            text.append(objective_text(oc))
        return "\n".join(text), images

    @tb.add("parameter_sensitivity",
            "Central finite-difference sensitivities dQ/dp of: the ESPEI objective (datasets_dir; total and per data "
            "type, plus Fisher-matrix std, correlations and identifiability warnings), binary invariant T/X and "
            "boundary X at given T (elements, tmin, tmax, invariants / boundary_temperatures), or equilibrium "
            "properties at one condition point (conditions + properties). CSV + tornado plot.",
            {"tdb": S, "parameters": PARAMS, "phase": S, "datasets_dir": S, "phase_models": S, "data_weights": OBJ,
             "elements": STRS, "tmin": N, "tmax": N, "temperature_unit": UNIT, "invariants": B,
             "boundary_temperatures": NUMS,
             "conditions": {**OBJ, "description": "e.g. {\"T\": 1200, \"X(CR)\": 0.3}"},
             "properties": {**STRS, "description": "pycalphad properties, e.g. NP(FCC_A1), X(FCC_A1,CR), HM"},
             "rel_step": {**N, "description": "FD step / max(|p|, typical), default 0.01"},
             "output_prefix": S, "background": B, "update_twins": TWINS}, ["tdb"])
    def parameter_sensitivity(tdb, parameters=None, phase=None, datasets_dir=None, phase_models=None,
                              data_weights=None, elements=None, tmin=None, tmax=None, temperature_unit="K",
                              invariants=False, boundary_temperatures=None, conditions=None, properties=None,
                              rel_step=0.01, output_prefix=None, background=None, update_twins=True):
        t = tb._find_tdb(tdb)
        pre = tb.outp(output_prefix, f"sensitivity_{Path(t).stem}/sens_{Path(t).stem}")
        kw = dict(tdb=t, out_prefix=pre, parameters=parameters, phase=phase,
                  datasets_dir=tb.p(datasets_dir) if datasets_dir else None,
                  phase_models=tb.p(phase_models) if phase_models else None, data_weights=data_weights,
                  elements=elements, tmin=tmin, tmax=tmax, temperature_unit=temperature_unit,
                  invariants=bool(invariants), boundary_temperatures=list(boundary_temperatures or []),
                  conditions=conditions, properties=list(properties or []), rel_step=rel_step,
                  update_twins=update_twins is not False)
        n = len(resolve_free_params(TDBText.read(t), parameters, phase))
        maps = 2 * n + 1 if (invariants or boundary_temperatures) else 0
        J = _jobs()
        if background is None:
            background = maps > 12
        if background and J is not None:
            jid = J.submit(tb, f"sensitivity {Path(t).name} ({n} parameters)",
                           "calphad_tools.param_tools:sensitivity_job", kw)
            return (f"started background job {jid}: {n} parameters" + (f", {maps} phase-diagram maps" if maps else "")
                    + ". Check with job_status / wait_job."), []
        r = sensitivity(progress=lambda m, f=None, **k: tb.ui.progress(m), **kw)
        files = "\n".join(f"  {k}: {tb.rel(v)}" for k, v in r["files"].items())
        imgs = [r["files"][k] for k in ("tornado_png", "correlation_png") if k in r["files"]]
        return f"{r['text']}\nFiles:\n{files}", imgs

    @tb.add("optimize_parameters",
            "Gradient-based fit of TDB parameters to ESPEI datasets: minimizes ESPEI's -lnL (least_squares 'trf' "
            "on residual/sigma, or lbfgsb) with finite-difference (or analytic, ESPEI PR #268) gradients, bounds "
            "and an optional Gaussian prior. Writes the optimized TDB, parameter table (before/after/approx std from "
            "J^T J), objective history CSV/PNG and a before/after comparison with the data. background (default "
            "for >3 parameters or >50 data points) runs a job; see job_status.",
            {"tdb": S, "datasets_dir": S, "parameters": PARAMS, "phase": S,
             "method": {"type": "string", "enum": ["least_squares", "lbfgsb"]},
             "bounds": {**OBJ, "description": "{parameter: [lo, hi]} (null = open)"},
             "prior_std": {**N, "description": "Gaussian prior around the start, std = prior_std x max(|p|, "
                                               "typical)"},
             "max_nfev": I, "time_limit_s": {**N, "description": "default 300 s when not in background"},
             "phase_models": S, "data_weights": OBJ, "output_tdb": S, "compare": B, "background": B,
             "update_twins": TWINS}, ["tdb", "datasets_dir"])
    def optimize_parameters(tdb, datasets_dir, parameters=None, phase=None, method="least_squares", bounds=None,
                            prior_std=None, max_nfev=None, time_limit_s=None, phase_models=None, data_weights=None,
                            output_tdb=None, compare=True, background=None, update_twins=True):
        t = tb._find_tdb(tdb)
        stem = Path(t).stem
        n = len(resolve_free_params(TDBText.read(t), parameters, phase))
        npts = count_data_points(tb.p(datasets_dir))
        J = _jobs()
        if background is None:
            background = n > 3 or npts > 50
        od = tb.outdir(None, f"optimize_{stem}")
        kw = dict(tdb=t, datasets_dir=tb.p(datasets_dir), out_dir=str(od), parameters=parameters, phase=phase,
                  method=method, bounds=bounds, prior_std=prior_std, max_nfev=max_nfev, time_limit_s=time_limit_s,
                  phase_models=tb.p(phase_models) if phase_models else None, data_weights=data_weights,
                  compare=compare, output_tdb=tb.outp(output_tdb, f"optimize_{stem}/{stem}_optimized.tdb"),
                  update_twins=update_twins is not False)
        note = ""
        if background:
            if J is not None:
                jid = J.submit(tb, f"optimize {Path(t).name} ({n} parameters, {npts} data points)",
                               "calphad_tools.param_tools:optimize_job", kw)
                return (f"started background job {jid}: {method}, {n} parameters, {npts} data points; outputs in "
                        f"{tb.rel(str(od))}. Check with job_status / wait_job (stop_job keeps the best point)."), []
            note = "(background jobs unavailable: ran synchronously)\n"
        if kw["time_limit_s"] is None:
            kw["time_limit_s"] = 300.0
        r = optimize_job(progress=lambda m, f=None, **k: tb.ui.progress(m), **kw)
        fm = r["file_map"]
        imgs = [fm[k] for k in fm if k.startswith("diagram_")][:1] + [fm["history_png"]]
        text = r["text"]
        return note + text + "\nFiles:\n" + "\n".join(f"  {k}: {tb.rel(v)}" for k, v in fm.items()), imgs
