"""Piecewise-in-T expression algebra for database comparison.

A piecewise expression is a list of segments ``[(T_low, T_high, expr), ...]`` (``expr`` a symengine expression in
``T``).  This module inlines FUNCTION references (recursively, merging temperature ranges: the union of breakpoints,
one explicit expression per sub-range), collects terms (const, T, T*ln(T), T**2, T**-1, T**7, T**-9, ...) so
databases written differently can be compared term by term, and evaluates segments numerically.

Only the standard library, numpy and symengine are used: show_code copies this module's source into standalone
scripts.
"""
from __future__ import annotations

import math

import numpy as np
import symengine as se

T = se.Symbol("T")
P = se.Symbol("P")
P_REF = 101325.0                  # pressure substituted into P-dependent functions (Pa)
LN_T = se.log(T)
PTYPE_ALIASES = {"BM": "BMAGN", "BMAG": "BMAGN", "BMAGN": "BMAGN", "L": "G"}


class MissingRef(KeyError):
    """A FUNCTION (or parameter) referenced by an expression is not defined in the database."""


# ------------------------------------------------------------------------------------------ canonical form
def canon(e, pressure: float = P_REF):
    """Substitute P, turn E**x into exp(x), integral float exponents into integers, expand products."""
    def walk(x):
        if not x.args:
            return x
        args = [walk(a) for a in x.args]
        if x.is_Pow:
            base, ex = args
            if base.is_Number and abs(float(base) - math.e) < 1e-12:
                return se.exp(ex)
            if ex.is_Number and not ex.is_integer:
                f = float(ex)
                if f == int(f):
                    return base ** se.Integer(int(f))
            return base ** ex
        return x.func(*args)
    e = walk(se.sympify(e))
    # state variables by name (pycalphad's T/P are Symbol subclasses that do not compare equal to plain symbols)
    rep = {s: (T if s.name == "T" else se.RealDouble(float(pressure)))
           for s in e.free_symbols if getattr(s, "name", "") in ("T", "P")}
    rep = {s: r for s, r in rep.items() if not (r == T and type(s) is type(T))}
    if rep:
        e = e.subs(rep)
    return se.expand(e)


# ------------------------------------------------------------------------------------------ segments
def seg_at(segs, t: float):
    """(index of the segment containing t, extrapolated?).  Ranges are [lo, hi); the last one includes hi."""
    for i, (a, b, _) in enumerate(segs):
        if a <= t < b:
            return i, False
    if t == segs[-1][1]:
        return len(segs) - 1, False
    if t < segs[0][0]:
        return 0, True
    if t > segs[-1][1]:
        return len(segs) - 1, True
    return max(i for i, s in enumerate(segs) if s[0] <= t), True     # gap between ranges


def _mid(a: float, b: float) -> float:
    if math.isfinite(a) and math.isfinite(b):
        return 0.5 * (a + b)
    if math.isfinite(a):
        return a + 1.0
    if math.isfinite(b):
        return b - 1.0
    return 1000.0


def merge_equal(segs):
    out = []
    for a, b, e in segs:
        if out and out[-1][2] == e and out[-1][1] == a:
            out[-1] = (out[-1][0], b, e)
        else:
            out.append((a, b, e))
    return out


def inline(segs, resolve, notes=None):
    """Replace every function symbol by its (already inlined) piecewise definition.

    ``resolve(name)`` returns the segments of a function or raises MissingRef.  Sub-ranges are split at the union
    of all breakpoints; a referenced function used outside its own T range is extrapolated (noted in ``notes``)."""
    out = []
    for lo, hi, e in segs:
        e = canon(e)
        names = sorted({s.name for s in e.free_symbols} - {"T", "P"})
        subs = {nm: resolve(nm) for nm in names}
        bps = {lo, hi}
        for pw in subs.values():
            for a, b, _ in pw:
                for x in (a, b):
                    if lo < x < hi and math.isfinite(x):
                        bps.add(x)
        bps = sorted(bps)
        for a, b in zip(bps[:-1], bps[1:]):
            m = _mid(a, b)
            rep = {}
            for nm, pw in subs.items():
                i, ext = seg_at(pw, m)
                if ext and notes is not None:
                    notes.add(f"{nm} extrapolated outside its range {pw[0][0]:g}-{pw[-1][1]:g} K")
                rep[se.Symbol(nm)] = pw[i][2]
            out.append((a, b, se.expand(e.subs(rep)) if rep else e))
    return merge_equal(out)


def scale(segs, factor):
    return [(a, b, se.expand(e * factor)) for a, b, e in segs]


def t_range(segs):
    return (segs[0][0], segs[-1][1]) if segs else (math.nan, math.nan)


# ------------------------------------------------------------------------------------------ terms
def _mono_key(m):
    """(power of T, power of ln T) of a monomial, or None if it is not T**n * ln(T)**k."""
    if m == 1 or m.is_Number:
        return (0, 0)
    factors = m.args if m.is_Mul else (m,)
    n, k = 0, 0
    for f in factors:
        if f == T:
            n += 1
        elif f == LN_T:
            k += 1
        elif f.is_Pow and f.args[0] == T and f.args[1].is_Number:
            x = float(f.args[1])
            n += int(x) if x == int(x) else x
        else:
            return None
    return (n, k) if k <= 1 else None


def terms(expr):
    """({(n, k): coefficient of T**n*ln(T)**k}, other) with ``other`` the sum of non-polynomial terms (or 0)."""
    d, other = {}, se.Integer(0)
    for mono, c in se.expand(expr).as_coefficients_dict().items():
        key = _mono_key(mono)
        if key is None:
            other = other + mono * c
            continue
        coef = float(c) * (float(mono) if mono.is_Number else 1.0)
        d[key] = d.get(key, 0.0) + coef
    def negligible(key, c):          # floating-point residue of cancelled terms (< 1e-6 J/mol on 298-6000 K)
        n, k = key
        tmax = 6000.0 if n >= 0 else 298.15
        return abs(c) * tmax ** n * (math.log(6000.0) if k else 1.0) < 1e-6
    return {k: v for k, v in d.items() if v != 0.0 and not negligible(k, v)}, other


def term_order(key) -> tuple:
    n, k = key
    if (n, k) == (0, 0):
        return (0, 0)
    if (n, k) == (1, 0):
        return (1, 0)
    if (n, k) == (1, 1):
        return (2, 0)
    if k == 0 and 2 <= n <= 6:
        return (3, n)
    if (n, k) == (-1, 0):
        return (4, 0)
    if k == 0 and n > 6:
        return (5, n)
    if k == 0 and n < -1:
        return (6, -n)
    return (7, n, k)


def term_name(key) -> str:
    n, k = key
    if n == 0:
        return "1" if k == 0 else "ln(T)"
    tp = "T" if n == 1 else (f"T**{n}" if n > 0 else f"T**({n})")
    return tp + ("*ln(T)" if k else "")


def _num(c: float, digits: int = 10) -> str:
    s = f"{c:.{digits}g}"
    return s


def fmt_expr(expr, digits: int = 10, max_other: int = 0) -> str:
    d, other = terms(expr)
    parts = []
    for key in sorted(d, key=term_order):
        c = d[key]
        mag = _num(abs(c), digits)
        body = mag if key == (0, 0) else (term_name(key) if mag == "1" else f"{mag}*{term_name(key)}")
        parts.append(("-" if c < 0 else "+") + body)
    if other != 0:
        o = str(other)
        if max_other and len(o) > max_other:
            o = o[:max_other] + "…"
        parts.append("+(" + o + ")")
    if not parts:
        return "0"
    s = " ".join(parts)
    return s[1:] if s.startswith("+") else s


def pw_text(segs, digits: int = 10) -> str:
    return " ; ".join(f"[{a:g}-{b:g} K] {fmt_expr(e, digits)}" for a, b, e in segs)


# ------------------------------------------------------------------------------------------ numbers
_LAMBDAS = {}


def _lam(e):
    f = _LAMBDAS.get(e)
    if f is None:
        f = se.Lambdify([T], [e])
        _LAMBDAS[e] = f
    return f


def evaluate(segs, temps):
    """(values, extrapolated mask) at the given temperatures."""
    temps = np.atleast_1d(np.asarray(temps, dtype=float))
    vals = np.full(temps.shape, np.nan)
    ext = np.zeros(temps.shape, dtype=bool)
    if not segs:
        return vals, ext
    idx = [seg_at(segs, float(t)) for t in temps]
    for i in {j for j, _ in idx}:
        mask = np.array([j == i for j, _ in idx])
        out = np.asarray(_lam(segs[i][2])(temps[mask]), dtype=float).reshape(-1)
        vals[mask] = np.broadcast_to(out, (int(mask.sum()),)) if out.size in (1, mask.sum()) else out[: mask.sum()]
    ext[:] = [x for _, x in idx]
    return vals, ext


def close(va, vb, rtol: float = 1e-4, atol: float = 1e-6):
    """(equal within rtol*|value| (point by point, floor atol), max |difference|)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    m = np.isfinite(va) & np.isfinite(vb)
    if not m.any():
        return False, math.nan
    d = np.abs(va[m] - vb[m])
    tol = np.maximum(atol, rtol * np.maximum(np.abs(va[m]), np.abs(vb[m])))
    return bool(np.all(d <= tol)), float(d.max())


def sort_constituents(subls, order: int):
    """Sort species in each sublattice (Thermo-Calc convention: odd Redlich-Kister orders change sign when a
    binary interaction is reordered; a ternary order index follows its species).  Returns (subls, sign, order)."""
    subls = [tuple(s) for s in subls]
    interacting = [i for i, sl in enumerate(subls) if len(sl) > 1]
    sign, new_order, out = 1.0, order, []
    for sl in subls:
        srt = tuple(sorted(sl))
        if srt != sl and len(interacting) == 1:
            if len(sl) == 2 and order % 2 == 1:
                sign = -1.0
            elif len(sl) == 3 and order < 3:
                new_order = srt.index(sl[order])
        out.append(srt)
    return out, sign, new_order


def fmt_value(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    a = abs(v)
    if a >= 1e5:
        s = f"{v:,.0f}"
    elif a >= 100:
        s = f"{v:.1f}"
    elif a == 0:
        s = "0"
    else:
        s = f"{v:.4g}"
    return s.replace("-", "−")


def segments_from_piecewise(pw, default_hi: float = math.inf):
    """Segments of a pycalphad (symengine) Piecewise((expr, cond), ..., (0, True))."""
    pw = se.sympify(pw)
    if not isinstance(pw, se.Piecewise):
        return [(-math.inf, default_hi, pw)]
    a = pw.args
    segs = []
    for expr, cond in zip(a[0::2], a[1::2]):
        if cond == se.true:
            continue
        lo, hi = -math.inf, default_hi
        rels = cond.args if isinstance(cond, se.And) else (cond,)
        for r in rels:
            lhs, rhs = r.args
            if getattr(lhs, "name", "") == "T" and rhs.is_Number:      # T < hi  /  T <= hi
                hi = float(rhs)
            elif getattr(rhs, "name", "") == "T" and lhs.is_Number:    # lo <= T
                lo = float(lhs)
        segs.append((lo, hi, expr))
    segs.sort(key=lambda s: s[0])
    return segs
