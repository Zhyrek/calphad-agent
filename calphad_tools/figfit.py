"""Curve fitting and statistical plot operations for edit_figure / make_plot (plus the ``fit_data`` tool).

Registered operations (see :mod:`calphad_tools.figures` for the extension contract):

* ``fit`` / ``residuals`` - least-squares fits (linear, poly, exp, power, log, arrhenius, logistic, gaussian or a
  custom expression evaluated by a small AST-whitelisted evaluator - never ``eval``), with confidence /
  prediction bands (delta method, t distribution) and a parameter report (± standard errors, R², RMSE, n, dof).
  Fits are done in the units *as displayed* (a kelvin axis labelled in °C is fitted in °C).
* statistics: ``hist`` (extends the built-in), ``box``, ``violin``, ``kde``, ``ecdf``, ``qq``, ``hist2d``,
  ``hexbin``, ``band``, ``corr_heatmap``, ``scatter_matrix``.

Everything drawn is plain matplotlib artists (no callables attached), so figures stay picklable.
"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .figures import (ScaledFormatter, _to_data, _to_display, find_col, main_axes, make_norm,
                      register_op)

R_GAS = 8.314462618           # J/(mol K)
KELVIN = 273.15

# ---------------------------------------------------------------------------
# safe expression evaluator (custom fit models)
# ---------------------------------------------------------------------------
SAFE_FUNCS: Dict[str, Callable] = {
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
    "arctan": np.arctan}
SAFE_CONSTS = {"pi": math.pi, "R": R_GAS}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
MAX_EXPR_LEN = 400
MAX_EXPR_NODES = 200


class SafeExpr:
    """A formula in one variable and named parameters: numbers, + - * / **, unary minus and the functions
    in SAFE_FUNCS. Anything else (attributes, subscripts, other calls, lambdas, comprehensions...) is refused."""

    def __init__(self, expr: str, var: str = "x", params: Sequence[str] = ()):
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("expr must be a non-empty formula string, e.g. 'a*x**2 + b*exp(-c/x)'")
        if len(expr) > MAX_EXPR_LEN:
            raise ValueError(f"expr is too long (max {MAX_EXPR_LEN} characters)")
        self.expr, self.var, self.params = expr, var, list(params)
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            raise ValueError(f"cannot parse expr {expr!r}: {e}") from None
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_EXPR_NODES:
            raise ValueError(f"expr is too complex (max {MAX_EXPR_NODES} syntax nodes)")
        self.names = set()
        for nd in nodes:
            self._check(nd)
        unknown = sorted(self.names - {var} - set(self.params) - set(SAFE_CONSTS))
        if unknown:
            raise ValueError(f"unknown name(s) {unknown} in expr: declare parameters in 'params' "
                             f"(the variable is {var!r}; functions: {', '.join(SAFE_FUNCS)}; constants: pi, R)")
        self.tree = tree.body

    def _check(self, nd):
        if isinstance(nd, (ast.Expression, ast.Load, ast.operator, ast.unaryop)):
            if isinstance(nd, ast.operator) and type(nd) not in _BINOPS:
                hint = " (use ** for powers)" if isinstance(nd, ast.BitXor) else ""
                raise ValueError(f"operator {type(nd).__name__} not allowed in expr{hint}")
            if isinstance(nd, ast.unaryop) and not isinstance(nd, (ast.USub, ast.UAdd)):
                raise ValueError(f"operator {type(nd).__name__} not allowed in expr")
            return
        if isinstance(nd, (ast.BinOp, ast.UnaryOp)):
            return
        if isinstance(nd, ast.Constant):
            if isinstance(nd.value, bool) or not isinstance(nd.value, (int, float)):
                raise ValueError(f"only numbers are allowed as constants in expr (got {nd.value!r})")
            return
        if isinstance(nd, ast.Name):
            if nd.id.startswith("_"):
                raise ValueError(f"name {nd.id!r} not allowed in expr")
            if nd.id not in SAFE_FUNCS:
                self.names.add(nd.id)
            return
        if isinstance(nd, ast.Call):
            if not isinstance(nd.func, ast.Name) or nd.func.id not in SAFE_FUNCS:
                raise ValueError(f"only these functions may be called in expr: {', '.join(SAFE_FUNCS)}")
            if len(nd.args) != 1 or nd.keywords or any(isinstance(a, ast.Starred) for a in nd.args):
                raise ValueError(f"{nd.func.id}() takes exactly one argument")
            return
        raise ValueError(f"{type(nd).__name__} is not allowed in expr (only numbers, {self.var!r}, parameters, "
                         f"+ - * / ** and {', '.join(SAFE_FUNCS)})")

    def __call__(self, x, *p):
        env = dict(SAFE_CONSTS)
        env.update(zip(self.params, (float(v) for v in p)))
        env[self.var] = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            out = self._eval(self.tree, env)
        return np.broadcast_to(np.asarray(out, dtype=float), np.shape(env[self.var])).astype(float)

    def _eval(self, nd, env):
        if isinstance(nd, ast.Constant):
            return float(nd.value)            # floats only: 10**10**10 overflows to inf instead of hanging
        if isinstance(nd, ast.Name):
            return env[nd.id]
        if isinstance(nd, ast.BinOp):
            return _BINOPS[type(nd.op)](self._eval(nd.left, env), self._eval(nd.right, env))
        if isinstance(nd, ast.UnaryOp):
            v = self._eval(nd.operand, env)
            return -v if isinstance(nd.op, ast.USub) else v
        if isinstance(nd, ast.Call):
            return SAFE_FUNCS[nd.func.id](self._eval(nd.args[0], env))
        raise ValueError(f"{type(nd).__name__} not allowed")      # unreachable after _check


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _g(v) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.4g}"


def _poly_str(coefs, var="x") -> str:
    out = ""
    for i, c in enumerate(coefs):
        term = "" if i == 0 else (f"·{var}" if i == 1 else f"·{var}{'²³'[i - 2] if i <= 3 else '^' + str(i)}")
        mag = _g(abs(c))
        if i == 0:
            out = ("−" if c < 0 else "") + mag
        else:
            out += (" − " if c < 0 else " + ") + mag + term
    return out


def _plus_c(p, has_offset: bool) -> str:
    if not has_offset:
        return ""
    return (" − " if p[-1] < 0 else " + ") + _g(abs(p[-1]))


class Model:
    """A fit model: value function f(x, *p), optional ln f for linearized fits, start values and a formatter."""

    def __init__(self, name, pnames, f, p0, eq, logf=None, report=None, xvar="x"):
        self.name, self.pnames, self.f, self._p0, self._eq = name, list(pnames), f, p0, eq
        self.logf, self._report, self.xvar = logf, report, xvar

    def p0(self, x, y):
        with np.errstate(all="ignore"):
            p = np.asarray(self._p0(x, y), dtype=float)
        return np.where(np.isfinite(p), p, 1.0)

    def equation(self, p) -> str:
        return self._eq(p)

    def report(self, p, se) -> List[Tuple[str, float, float, str]]:
        if self._report:
            return self._report(p, se)
        return [(n, v, s, "") for n, v, s in zip(self.pnames, p, se)]


def _linfit(x, y, deg=1):
    try:
        return np.polyfit(x, y, deg)[::-1]      # ascending
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return np.ones(deg + 1)


def _t_of_x(unit: str):
    unit = (unit or "K").replace(" ", "")
    table = {"K": (1.0, 0.0, False), "C": (1.0, KELVIN, False), "°C": (1.0, KELVIN, False),
             "1/K": (1.0, 0.0, True), "1000/K": (1000.0, 0.0, True), "1e3/K": (1000.0, 0.0, True),
             "10000/K": (1e4, 0.0, True), "1e4/K": (1e4, 0.0, True)}
    if unit not in table:
        raise ValueError(f"T_unit {unit!r} unknown; use K, C, 1/K, 1000/K or 10000/K")
    return table[unit]


class _TConv:
    """x (as fitted) -> absolute temperature in K."""

    def __init__(self, unit):
        self.unit = unit
        self.factor, self.offset, self.inverse = _t_of_x(unit)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return self.factor / x if self.inverse else x * self.factor + self.offset


MODELS_HELP = ("linear | poly (deg) | exp (A·e^(b·x)) | power (A·x^b) | log (a+b·ln x) | arrhenius "
               "(A·exp(-Ea/(R·T)), Ea in kJ/mol) | logistic (L/(1+e^(-k(x-x0)))) | gaussian (A·e^(-(x-mu)²/2s²)) | "
               "custom")


def build_model(op: Dict, t_unit_default: str = "K") -> Model:
    """Model from an op/tool spec: model name (+ deg / offset / T_unit) or a custom {expr, params}."""
    spec = op.get("model", "linear")
    if isinstance(spec, dict):              # "model": {"expr": ..., "params": {...}}
        op = {**op, **spec, "model": "custom"}
        spec = "custom"
    if op.get("expr") and spec in (None, "linear") and "model" not in op:
        spec = "custom"
    name = str(spec).lower()
    offset = bool(op.get("offset"))
    oname = ["c"] if offset else []

    def _off(p):
        return p[-1] if offset else 0.0

    if name in ("linear", "line", "lin"):
        return Model("linear", ["intercept", "slope"], lambda x, a, b: a + b * x,
                     lambda x, y: _linfit(x, y, 1), lambda p: "y = " + _poly_str(p))
    if name in ("poly", "polynomial", "quadratic", "cubic"):
        deg = int(op.get("deg", {"quadratic": 2, "cubic": 3}.get(name, 2)))
        if not 0 <= deg <= 10:
            raise ValueError("poly deg must be 0..10")

        def fpoly(x, *c):
            return np.polynomial.polynomial.polyval(x, c)
        return Model(f"poly{deg}", [f"c{i}" for i in range(deg + 1)], fpoly, lambda x, y: _linfit(x, y, deg),
                     lambda p: "y = " + _poly_str(p))
    if name in ("exp", "exponential"):
        def fexp(x, A, b, *c):
            return A * np.exp(b * x) + (c[0] if c else 0.0)

        def p0exp(x, y):
            yy = y - (np.min(y) - 0.1 * np.ptp(y) if offset else 0.0)
            ok = yy > 0
            b, lnA = (_linfit(x[ok], np.log(yy[ok]), 1)[::-1] if ok.sum() >= 2 else (0.0, np.log(np.mean(np.abs(y)))))
            return [np.exp(lnA), b] + ([np.min(y) - 0.1 * np.ptp(y)] if offset else [])
        return Model("exp", ["A", "b"] + oname, fexp, p0exp,
                     lambda p: f"y = {_g(p[0])}·exp({_g(p[1])}·x)" + _plus_c(p, offset),
                     logf=None if offset else (lambda x, A, b: np.log(np.abs(A)) + b * x))
    if name in ("power", "powerlaw", "power_law"):
        def fpow(x, A, b, *c):
            return A * np.power(x, b) + (c[0] if c else 0.0)

        def p0pow(x, y):
            ok = (x > 0) & (y > 0)
            b, lnA = _linfit(np.log(x[ok]), np.log(y[ok]), 1)[::-1] if ok.sum() >= 2 else (1.0, 0.0)
            return [np.exp(lnA), b] + ([0.0] if offset else [])
        return Model("power", ["A", "b"] + oname, fpow, p0pow,
                     lambda p: f"y = {_g(p[0])}·x^{_g(p[1])}" + _plus_c(p, offset),
                     logf=None if offset else (lambda x, A, b: np.log(np.abs(A)) + b * np.log(x)))
    if name in ("log", "logarithmic", "ln"):
        return Model("log", ["a", "b"], lambda x, a, b: a + b * np.log(x),
                     lambda x, y: _linfit(np.log(x), y, 1),
                     lambda p: f"y = {_g(p[0])} {'−' if p[1] < 0 else '+'} {_g(abs(p[1]))}·ln(x)")
    if name in ("arrhenius", "arr"):
        tc = _TConv(op.get("T_unit") or t_unit_default)

        def farr(x, A, Ea, *c):
            return A * np.exp(-Ea * 1000.0 / (R_GAS * tc(x))) + (c[0] if c else 0.0)

        def logarr(x, A, Ea):
            return np.log(np.abs(A)) - Ea * 1000.0 / (R_GAS * tc(x))

        def p0arr(x, y):
            ok = y > 0
            if ok.sum() >= 2:
                slope, lnA = np.polyfit(1.0 / tc(x[ok]), np.log(y[ok]), 1)
                return [np.exp(lnA), -slope * R_GAS / 1000.0] + ([0.0] if offset else [])
            return [1.0, 100.0] + ([0.0] if offset else [])

        def rep(p, se):
            out = [("A", p[0], se[0], ""), ("Ea", p[1], se[1], "kJ/mol"),
                   ("ln A", float(np.log(abs(p[0]))), float(se[0] / abs(p[0])), "")]
            return out + ([("c", p[2], se[2], "")] if offset else [])
        tnote = "" if tc.unit == "K" else f", T = {'x + 273.15' if tc.offset else tc.unit.replace('/K', '/x')} K"
        return Model("arrhenius", ["A", "Ea"] + oname, farr, p0arr,
                     lambda p: f"y = {_g(p[0])}·exp(−{_g(p[1])} kJ/mol / (R·T)){_plus_c(p, offset)}{tnote}",
                     logf=None if offset else logarr, report=rep, xvar="T")
    if name in ("logistic", "sigmoid"):
        def flog(x, L, k, x0, *c):
            return L / (1.0 + np.exp(-k * (x - x0))) + (c[0] if c else 0.0)

        def p0log(x, y):
            lo, hi = np.min(y), np.max(y)
            sgn = np.sign(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 and np.ptp(y) > 0 else 1.0
            L = (hi - lo) if offset else (hi if abs(hi) >= abs(lo) else lo)
            return [L, 4.0 * (sgn or 1.0) / (np.ptp(x) or 1.0), float(np.median(x))] + ([lo] if offset else [])
        return Model("logistic", ["L", "k", "x0"] + oname, flog, p0log,
                     lambda p: f"y = {_g(p[0])} / (1 + exp(−{_g(p[1])}·(x − {_g(p[2])}))){_plus_c(p, offset)}")
    if name in ("gaussian", "gauss", "normal"):
        def fgau(x, A, mu, s, *c):
            return A * np.exp(-0.5 * ((x - mu) / s) ** 2) + (c[0] if c else 0.0)

        def p0gau(x, y):
            base = np.min(y) if offset else 0.0
            w = np.clip(y - base, 0, None)
            i = int(np.argmax(y))
            mu = float(np.sum(w * x) / np.sum(w)) if np.sum(w) > 0 else float(x[i])
            s = float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w))) if np.sum(w) > 0 else float(np.ptp(x) / 4)
            return [y[i] - base, mu, s or 1.0] + ([base] if offset else [])
        return Model("gaussian", ["A", "mu", "sigma"] + oname, fgau, p0gau,
                     lambda p: f"y = {_g(p[0])}·exp(−(x − {_g(p[1])})² / (2·{_g(abs(p[2]))}²)){_plus_c(p, offset)}")
    if name in ("custom", "expr", "expression"):
        params = op.get("params")
        if isinstance(params, dict):
            pnames, start = list(params), [float(v) for v in params.values()]
        elif isinstance(params, (list, tuple)):
            pnames, start = [str(v) for v in params], [1.0] * len(params)
        else:
            raise ValueError("custom model needs 'params': {name: start value, ...}")
        if not pnames:
            raise ValueError("custom model needs at least one parameter")
        var = str(op.get("var", "x"))
        fx = SafeExpr(op.get("expr", ""), var=var, params=pnames)
        expr_s = fx.expr

        def eqc(p):
            s = expr_s
            for n, v in sorted(zip(pnames, p), key=lambda t: -len(t[0])):
                s = re.sub(rf"\b{re.escape(n)}\b", f"({_g(v)})", s)
            return "y = " + s
        return Model("custom", pnames, fx, lambda x, y: start, eqc, xvar=var)
    raise ValueError(f"unknown model {spec!r}; models: {MODELS_HELP}")


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
class FitResult:
    """Result of a least-squares fit in 'fit space' (y, or ln y for linearized fits)."""

    def __init__(self, model, p, pcov, x, y, t, g, back, sigma_t, space, absolute_sigma):
        self.model, self.p, self.pcov, self.x, self.y = model, p, pcov, x, y
        self.t, self.g, self.back, self.sigma_t, self.space = t, g, back, sigma_t, space
        self.n, self.k = len(x), len(p)
        self.dof = self.n - self.k
        self.absolute_sigma = absolute_sigma
        self.se = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) else np.full(self.k, np.nan)
        self.fitted_t = g(x, *p)
        self.resid = t - self.fitted_t
        ssr = float(np.sum(self.resid ** 2))
        self.rmse = math.sqrt(ssr / self.n)
        if sigma_t is not None:
            w = 1.0 / sigma_t ** 2
            tw = np.sum(w * t) / np.sum(w)
            sst = float(np.sum(w * (t - tw) ** 2))
            self.chi2 = float(np.sum(w * self.resid ** 2))
            self.r2 = 1.0 - self.chi2 / sst if sst > 0 else float("nan")
            self.chi2red = self.chi2 / self.dof
        else:
            sst = float(np.sum((t - t.mean()) ** 2))
            self.r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
            self.chi2 = self.chi2red = None
        self.s2 = ssr / self.dof                      # residual variance (unweighted)

    def jac(self, x):
        J = np.empty((len(x), self.k))
        for i in range(self.k):
            h = 1e-6 * abs(self.p[i]) if self.p[i] != 0 else 1e-8
            pp, pm = self.p.copy(), self.p.copy()
            pp[i] += h
            pm[i] -= h
            J[:, i] = (self.g(x, *pp) - self.g(x, *pm)) / (2 * h)
        return J

    def predict(self, xg, band: Optional[str] = None, level: float = 0.95):
        """(yhat, lo, hi) in displayed y units; lo/hi None without a band."""
        from scipy import stats
        xg = np.asarray(xg, dtype=float)
        with np.errstate(all="ignore"):
            yt = self.g(xg, *self.p)
            if not band or band in ("none", False):
                return self.back(yt), None, None
            if not np.all(np.isfinite(self.pcov)):
                return self.back(yt), None, None
            J = self.jac(xg)
            var = np.einsum("ij,jk,ik->i", J, self.pcov, J)
            if band == "prediction":
                if self.sigma_t is None:
                    var = var + self.s2
                else:        # new-observation sigma: log-interpolated from the data's sigmas
                    o = np.argsort(self.x)
                    s_new = np.exp(np.interp(xg, self.x[o], np.log(self.sigma_t[o])))
                    var = var + (1.0 if self.absolute_sigma else self.chi2red) * s_new ** 2
            elif band != "confidence":
                raise ValueError("band must be 'confidence', 'prediction' or none")
            tq = stats.t.ppf(0.5 + level / 2.0, self.dof)
            half = tq * np.sqrt(np.clip(var, 0, None))
            return self.back(yt), self.back(yt - half), self.back(yt + half)

    def params_text(self, sep="; ") -> str:
        parts = []
        for n, v, s, unit in self.model.report(self.p, self.se):
            parts.append(f"{n} = {_g(v)} ± {_g(s)}{(' ' + unit) if unit else ''}")
        return sep.join(parts)

    def stats_text(self) -> str:
        sp = "" if self.space == "y" else f" (in {self.space})"
        s = f"R²{sp} = {self.r2:.5g}{' (weighted)' if self.sigma_t is not None else ''}, RMSE{sp} = {_g(self.rmse)}"
        if self.chi2red is not None:
            s += f", χ²/dof = {self.chi2red:.4g}"
        return s + f", n = {self.n}, dof = {self.dof}"

    def summary(self) -> str:
        return (f"{self.model.name} fit: {self.model.equation(self.p)}; {self.params_text()}; {self.stats_text()}"
                + ("" if self.space == "y" else f" [fitted {self.space}]"))


def _ident(v):
    return v


def fit_arrays(model: Model, x, y, sigma=None, linearize: bool = False, y_is: str = "value",
               p0: Optional[Dict] = None, bounds: Optional[Dict] = None, absolute_sigma: bool = False,
               maxfev: int = 20000) -> FitResult:
    """Least-squares fit of ``model`` to displayed-unit data."""
    from scipy.optimize import curve_fit
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sig = None if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), x.shape).copy()
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(sig) & (sig > 0) if sig is not None else True)
    x, y = x[ok], y[ok]
    sig = sig[ok] if sig is not None else None
    y_is = (y_is or "value").lower()
    g, back, t, space, sig_t = model.f, _ident, y, "y", sig
    if y_is in ("ln", "log10") or linearize:
        if model.logf is None:
            raise ValueError(f"linearize/y_is needs a positive model without offset (exp, power, arrhenius), "
                             f"not {model.name}")
        lf = model.logf
        if y_is == "ln":
            g, space = lf, "ln y"
        elif y_is == "log10":
            g, space = _Log10(lf), "log10 y"
        else:
            if np.any(y <= 0):
                raise ValueError("linearize needs y > 0 (it fits ln y)")
            g, back, t, space = lf, np.exp, np.log(y), "ln y"
            sig_t = None if sig is None else sig / y
    if linearize and y_is in ("ln", "log10"):
        pass                                  # already linear in the displayed space
    k = len(model.pnames)
    if len(x) <= k:
        raise ValueError(f"{model.name} has {k} parameters; need more than {k} valid points (got {len(x)})")
    if model.name == "log" and np.any(x <= 0):
        raise ValueError("log model needs x > 0")
    if model.name == "power" and np.any(x <= 0):
        raise ValueError("power model needs x > 0")
    start = model.p0(x, y)
    for name, v in (p0 or {}).items():
        if name not in model.pnames:
            raise ValueError(f"p0 has unknown parameter {name!r}; parameters: {model.pnames}")
        start[model.pnames.index(name)] = float(v)
    lo, hi = np.full(k, -np.inf), np.full(k, np.inf)
    for name, (b0, b1) in (bounds or {}).items():
        if name not in model.pnames:
            raise ValueError(f"bounds has unknown parameter {name!r}; parameters: {model.pnames}")
        i = model.pnames.index(name)
        lo[i] = -np.inf if b0 is None else float(b0)
        hi[i] = np.inf if b1 is None else float(b1)
    kw = {}
    if bounds:
        start = np.clip(start, np.where(np.isfinite(lo), lo + 1e-12 * np.abs(lo), lo),
                        np.where(np.isfinite(hi), hi - 1e-12 * np.abs(hi), hi))
        kw["bounds"] = (lo, hi)
    else:
        kw["maxfev"] = maxfev
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            p, pcov = curve_fit(g, x, t, p0=start, sigma=sig_t, absolute_sigma=absolute_sigma and sig_t is not None,
                                **kw)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"{model.name} fit did not converge ({e}); give start values 'p0': "
                             f"{{{', '.join(repr(n) + ': ...' for n in model.pnames)}}} or 'bounds'") from None
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{model.name} fit gave non-finite parameters; give p0 or bounds")
    return FitResult(model, np.asarray(p, dtype=float), np.asarray(pcov, dtype=float), x, y, t, g, back, sig_t,
                     space, absolute_sigma)


class _Log10:
    def __init__(self, lf):
        self.lf = lf

    def __call__(self, x, *p):
        return self.lf(x, *p) / np.log(10.0)


# ---------------------------------------------------------------------------
# op helpers
# ---------------------------------------------------------------------------
def _scale(axis) -> float:
    f = axis.get_major_formatter()
    return f.scale if isinstance(f, ScaledFormatter) else 1.0


def _require_rect(ctx, what):
    if ctx.ax.name != "rectilinear":
        raise ValueError(f"{what} needs ordinary x-y axes (this axes is {ctx.ax.name!r})")
    if ctx.coords != "data":
        raise ValueError(f"{what} works in data coordinates only (remove 'coords')")


def _require_2d(ctx, what):
    if ctx.ax.name == "3d":
        raise ValueError(f"{what} does not work on 3-D axes")
    if ctx.coords != "data":
        raise ValueError(f"{what} works in data coordinates only (remove 'coords')")


def _fit_spec(op: Dict) -> Dict:
    return {k: op[k] for k in ("model", "deg", "offset", "T_unit", "expr", "params", "var") if k in op}


def _default_t_unit(ctx) -> str:
    f = ctx.ax.xaxis.get_major_formatter()
    if isinstance(f, ScaledFormatter) and f.scale == 1 and abs(f.offset + KELVIN) < 1e-6:
        return "C"
    return "K"


def default_linearize(op: Dict, y, sigma) -> bool:
    """Arrhenius data usually span decades with multiplicative scatter: fit ln y unless told otherwise.

    An explicit "linearize" wins; supplying sigma/yerr keeps the (weighted) value-space fit; offsets can't be
    linearized.
    """
    if op.get("linearize") is not None:
        return bool(op["linearize"])
    name = op.get("model") if isinstance(op.get("model"), str) else ""
    y = np.asarray(y, dtype=float)
    return (str(name).lower() == "arrhenius" and sigma is None and not op.get("offset")
            and op.get("y_is", "value") == "value" and y.size > 0 and bool(np.all(y > 0)))


def fit_from_ctx(ctx) -> FitResult:
    """Fit the op's data in the units as displayed on ctx.ax."""
    op = ctx.op
    if ctx.raw("x") is None or ctx.raw("y") is None:
        raise ValueError("needs x and y: inline lists, or a data source with x_col and y_col")
    xd = _to_display(ctx.ax.xaxis, ctx.X())
    yd = _to_display(ctx.ax.yaxis, ctx.Y())
    if len(xd) != len(yd):
        raise ValueError(f"x and y have different lengths ({len(xd)} vs {len(yd)})")
    sig = None
    for key in ("sigma", "yerr"):
        if ctx.raw(key) is not None:
            s = ctx.dY(key)
            sig = np.abs(s * _scale(ctx.ax.yaxis))
            break
    model = build_model(op, _default_t_unit(ctx))
    lin = default_linearize(op, yd, sig)
    res = fit_arrays(model, xd, yd, sig, linearize=lin, y_is=op.get("y_is", "value"),
                     p0=op.get("p0"), bounds=op.get("bounds"), absolute_sigma=bool(op.get("absolute_sigma")))
    _decade_note(ctx, op, yd, sig, lin)
    return res


def _decade_note(ctx, op, y, sig, lin):
    y = np.asarray(y, dtype=float)
    name = str(op.get("model", "")).lower() if isinstance(op.get("model"), str) else ""
    if lin and op.get("linearize") is None and name == "arrhenius":
        ctx.note("fitted ln(y) vs 1/T (default for arrhenius: multiplicative scatter); set linearize: false "
                 "for an unweighted fit of y itself")
    elif (not lin and sig is None and name in ("exp", "power") and y.size and np.all(y > 0)
          and np.nanmax(y) / max(np.nanmin(y), 1e-300) > 10):
        ctx.note(f"y spans {np.log10(np.nanmax(y) / np.nanmin(y)):.1f} decades: an unweighted fit is dominated by "
                 "the largest values; consider linearize: true")


def _grid(ctx_ax, lo, hi, n, logx):
    if logx and lo > 0 and hi > 0:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


_LOCS = {"upper left": (0.03, 0.97, "left", "top"), "upper right": (0.97, 0.97, "right", "top"),
         "lower left": (0.03, 0.03, "left", "bottom"), "lower right": (0.97, 0.03, "right", "bottom"),
         "upper center": (0.5, 0.97, "center", "top"), "lower center": (0.5, 0.03, "center", "bottom"),
         "center": (0.5, 0.5, "center", "center")}


def _style(op, allowed, prefix=""):
    """Style kwargs: op[prefix+key] for key in allowed (colours validated)."""
    import matplotlib.colors as mc
    kw = {}
    for k in allowed:
        v = op.get(prefix + k)
        if v is None:
            continue
        if k in ("color", "edgecolor", "facecolor") and isinstance(v, str) and not mc.is_color_like(v):
            raise ValueError(f"{prefix + k}={v!r} is not a matplotlib colour (name, '#hex' or 'C0'..'C9')")
        kw[k] = v
    return kw


_LINE_KEYS = ("color", "linestyle", "linewidth", "alpha", "zorder")


# ---------------------------------------------------------------------------
# fit / residuals ops
# ---------------------------------------------------------------------------
FIT_HELP = ("fit {x, y | data + x_col, y_col, [where], sigma|yerr (or *_col) weights, absolute_sigma, model: " + MODELS_HELP
            + ", deg, offset: true (+c), T_unit: K|C|1/K|1000/K|10000/K (arrhenius; default K, C on a °C axis), "
            "linearize: true (fit ln y), y_is: value|ln|log10 (y axis already shows ln y), custom: expr: 'a*x**2 + "
            "b*exp(-c/x)', params: {a: 1, ...} (funcs exp log log10 sqrt abs sin cos tan tanh, consts pi R), "
            "p0: {name: v}, bounds: {name: [lo, hi]}, range: [x0, x1] (extrapolate), npoints, band: confidence|"
            "prediction, level: 0.95, band_alpha, show_data: true, marker, s, data_label, annotate: true|equation|"
            "params, annotate_loc: 'upper left'.., fontsize, label (true = auto), color, linestyle, linewidth}. "
            "Fits in units as displayed; the result reports params ± SE, R², RMSE, n, dof")


@register_op("fit", FIT_HELP, group="fit")
def _op_fit(ctx) -> None:
    _require_2d(ctx, "fit")
    op, ax = ctx.op, ctx.ax
    res = fit_from_ctx(ctx)
    if op.get("range") is not None:
        r = np.asarray(op["range"], dtype=float)
        lo, hi = _to_display(ax.xaxis, _to_data(ax.xaxis, r, ctx.mode))[:2]
    else:
        lo, hi = float(np.min(res.x)), float(np.max(res.x))
    xg = _grid(ax, lo, hi, int(op.get("npoints", 200)), ax.get_xscale() == "log")
    band = op.get("band")
    band = None if band in (None, False, "none") else ("confidence" if band is True else str(band).lower())
    level = float(op.get("level", 0.95))
    yh, ylo, yhi = res.predict(xg, band, level)
    X = _to_data(ax.xaxis, xg, "displayed")
    kw = _style(op, _LINE_KEYS)
    label = op.get("label")
    if label is True:
        label = f"{res.model.name} fit (R² = {res.r2:.4f})"
    if label:
        kw["label"] = str(label)
    line, = ax.plot(X, _to_data(ax.yaxis, yh, "displayed"), **kw)
    color = line.get_color()
    if ylo is not None:
        ax.fill_between(X, _to_data(ax.yaxis, ylo, "displayed"), _to_data(ax.yaxis, yhi, "displayed"),
                        color=op.get("band_color", color), alpha=float(op.get("band_alpha", 0.2)), linewidth=0,
                        label=op.get("band_label"), zorder=line.get_zorder() - 0.1)
    if op.get("show_data"):
        xd = _to_data(ax.xaxis, res.x, "displayed")
        yd = _to_data(ax.yaxis, res.y, "displayed")
        dkw = {"color": op.get("data_color", color), "marker": op.get("marker", "o"), "label": op.get("data_label")}
        if res.sigma_t is not None and res.space == "y":
            ax.errorbar(xd, yd, yerr=res.sigma_t / _scale(ax.yaxis), linestyle="none", capsize=2,
                        markersize=op.get("markersize", 5), **dkw)
        else:
            ax.scatter(xd, yd, s=op.get("s", 25), zorder=line.get_zorder() + 0.1, **dkw)
    ann = op.get("annotate")
    if ann:
        eq = res.model.equation(res.p)
        if ann == "equation":
            txt = f"{eq}\nR² = {res.r2:.4f}"
        elif ann == "params":
            txt = res.params_text("\n") + f"\nR² = {res.r2:.4f}"
        else:
            txt = f"{eq}\n" + res.params_text("\n") + f"\nR² = {res.r2:.4f}, n = {res.n}"
        loc = op.get("annotate_loc", "upper left")
        if loc not in _LOCS:
            raise ValueError(f"annotate_loc must be one of {list(_LOCS)}")
        x0, y0, ha, va = _LOCS[loc]
        ax.text(x0, y0, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=op.get("fontsize", 8),
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.6"})
    extra = f"; {int(level * 100)}% {band} band drawn" if ylo is not None else ""
    ctx.note(res.summary() + extra)


RESID_HELP = ("residuals {same data/model keys as fit, standardized: true (divide by sigma or RMSE), color, marker, s, "
              "label, zero_line: true}: residuals (y - fit) vs x, e.g. in a second panel")


@register_op("residuals", RESID_HELP, group="fit")
def _op_residuals(ctx) -> None:
    _require_2d(ctx, "residuals")
    op, ax = ctx.op, ctx.ax
    res = fit_from_ctx(ctx)
    r = res.resid
    if op.get("standardized"):
        r = r / (res.sigma_t if res.sigma_t is not None else math.sqrt(res.s2))
    xd = _to_data(ax.xaxis, res.x, "displayed")
    kw = {"color": op.get("color"), "marker": op.get("marker", "o"), "s": op.get("s", 25)}
    if op.get("label"):
        kw["label"] = op["label"]
    if op.get("alpha") is not None:
        kw["alpha"] = op["alpha"]
    ax.scatter(xd, r, **{k: v for k, v in kw.items() if v is not None})
    if op.get("zero_line", True):
        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    if not ax.get_ylabel():
        ax.set_ylabel(("standardized residual" if op.get("standardized") else "residual")
                      + ("" if res.space == "y" else f" ({res.space})"))
    ctx.note(f"residuals of {res.model.name} fit: RMSE = {_g(res.rmse)}, max |r| = {_g(np.max(np.abs(res.resid)))}, "
             f"{res.stats_text()}")


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def _num(v) -> np.ndarray:
    return np.asarray(v, dtype=float).ravel()


def _groups(ctx, key: str = "values", extra: Sequence[str] = ()) -> List[Tuple[str, np.ndarray, Dict]]:
    """[(label, values, {extra_key: values})] from inline values / groups or a data source with
    <key>_col | value_cols [+ group_col, groups]. Non-finite values are dropped."""
    op = ctx.op
    out = []
    src_cols = op.get(f"{key}_col") or op.get("value_cols") or op.get("group_col")
    if ctx.has_source and src_cols:
        df = ctx.frame()
        src = op.get("csv") or op.get("data")
        vcols = op.get("value_cols")
        cols = [find_col(df, c, src) for c in (vcols if isinstance(vcols, list) else [vcols])] if vcols else \
            ([ctx.colname(key)] if op.get(f"{key}_col") else [])
        if not cols:
            raise ValueError(f"give {key}_col (or value_cols) with the data source")
        ecols = {e: find_col(df, op[f"{e}_col"], src) for e in extra if op.get(f"{e}_col")}
        if op.get("group_col"):
            if len(cols) != 1:
                raise ValueError("group_col works with a single value column")
            gc = find_col(df, op["group_col"], src)
            order = op.get("groups")
            import pandas as pd
            keys = list(order) if isinstance(order, list) else list(pd.unique(df[gc].dropna()))
            for gv in keys:
                sub = df[df[gc] == gv]
                if not len(sub) and isinstance(gv, str):
                    sub = df[df[gc].astype(str) == gv]
                out.append((str(gv), sub[cols[0]].to_numpy(), {e: sub[c].to_numpy() for e, c in ecols.items()}))
        else:
            for c in cols:
                lab = str(c) if len(cols) > 1 else (op.get("label") or "")
                out.append((lab, df[c].to_numpy(), {e: df[cc].to_numpy() for e, cc in ecols.items()}))
    else:
        v = op.get(key)
        grp = op.get("groups")
        if isinstance(grp, dict):
            v = grp
        if v is None:
            raise ValueError(f"needs '{key}' (list, list of lists or {{group: list}}), 'groups': {{name: list}}, or a "
                             f"data source with {key}_col / value_cols (+ group_col)")
        ex = {e: op.get(e) for e in extra if op.get(e) is not None}
        if isinstance(v, dict):
            for i, (k, vals) in enumerate(v.items()):
                out.append((str(k), vals, {e: (w[k] if isinstance(w, dict) else w[i]) for e, w in ex.items()}))
        elif isinstance(v, (list, tuple)) and v and all(isinstance(t, (list, tuple, np.ndarray)) for t in v):
            for i, vals in enumerate(v):
                out.append((f"group {i + 1}", vals, {e: w[i] for e, w in ex.items()}))
        else:
            out.append((op.get("label") or "", v, ex))
    labels = op.get("labels")
    if isinstance(labels, list):
        if len(labels) != len(out):
            raise ValueError(f"labels has {len(labels)} entries for {len(out)} groups")
        out = [(str(lab), v, e) for lab, (_, v, e) in zip(labels, out)]
    clean = []
    for lab, vals, ex in out:
        vals = _num(vals)
        exa = {e: _num(w) for e, w in ex.items()}
        ok = np.isfinite(vals)
        for e, w in exa.items():
            if len(w) != len(vals):
                raise ValueError(f"{e} has {len(w)} values for {len(vals)} data values")
            ok &= np.isfinite(w)
        clean.append((lab, vals[ok], {e: w[ok] for e, w in exa.items()}))
    if not clean or all(len(v) == 0 for _, v, _ in clean):
        raise ValueError("no finite data values")
    return clean


def _colors(op, n) -> Optional[List]:
    import matplotlib.colors as mc
    c = op.get("colors", op.get("color"))
    if c is None:
        return None
    cl = [c] * n if isinstance(c, str) or (isinstance(c, (list, tuple)) and len(c) in (3, 4) and
                                           all(isinstance(t, (int, float)) for t in c)) else list(c)
    if len(cl) < n:
        raise ValueError(f"{len(cl)} colours for {n} groups")
    for x in cl[:n]:
        if not mc.is_color_like(x):
            raise ValueError(f"{x!r} is not a matplotlib colour")
    return cl[:n]


def _summ(groups) -> str:
    parts = []
    for lab, v, _ in groups[:10]:
        if len(v):
            parts.append(f"{lab + ': ' if lab else ''}n={len(v)}, mean={_g(v.mean())}, std={_g(v.std(ddof=1) if len(v) > 1 else 0)}, "
                         f"median={_g(np.median(v))}")
    return "; ".join(parts) + (f"; ... ({len(groups)} groups)" if len(groups) > 10 else "")


def _vdata(ctx, v, axis):
    """Values given as displayed -> data coordinates of the value axis."""
    return _to_data(axis, v, ctx.mode) if len(v) else v


# ---------------------------------------------------------------------------
# statistics ops
# ---------------------------------------------------------------------------
GROUP_KEYS = "values | values_col | value_cols | groups: {name: [..]} | group_col (+ groups: order)"


@register_op("hist", f"hist {{{GROUP_KEYS}, bins (n|edges|'auto'), range, density, cumulative, histtype: "
             "bar|barstacked|step|stepfilled, weights|weights_col, orientation, stacked, log, rwidth, color|colors, "
             "alpha, label|labels, edgecolor, linewidth, hatch}", group="statistics")
def _op_hist(ctx) -> None:
    _require_rect(ctx, "hist")
    op, ax = ctx.op, ctx.ax
    groups = _groups(ctx, "values", ("weights",))
    horiz = op.get("orientation", "vertical") == "horizontal"
    axis = ax.yaxis if horiz else ax.xaxis
    data = [_vdata(ctx, v, axis) for _, v, _ in groups]
    bins = op.get("bins", 20)
    if isinstance(bins, list):
        bins = _to_data(axis, bins, ctx.mode)
    kw = {"bins": bins, "density": bool(op.get("density", False)), "cumulative": op.get("cumulative", False),
          "histtype": op.get("histtype", "bar"), "orientation": "horizontal" if horiz else "vertical",
          "stacked": bool(op.get("stacked", False)), "log": bool(op.get("log", False))}
    if op.get("range") is not None:
        kw["range"] = tuple(_to_data(axis, op["range"], ctx.mode))
    if op.get("rwidth") is not None:
        kw["rwidth"] = op["rwidth"]
    ws = [e.get("weights") for _, _, e in groups]
    if any(w is not None for w in ws):
        kw["weights"] = ws if len(ws) > 1 else ws[0]
    cols = _colors(op, len(groups))
    if cols:
        kw["color"] = cols if len(groups) > 1 else cols[0]
    labs = [lab for lab, _, _ in groups]
    if any(labs):
        kw["label"] = labs if len(groups) > 1 else labs[0]
    kw.update(_style(op, ("alpha", "edgecolor", "linewidth", "linestyle", "hatch", "zorder")))
    ax.hist(data if len(data) > 1 else data[0], **kw)
    ctx.note("hist: " + _summ(groups))


def _positions(op, n):
    pos = op.get("positions")
    if pos is None:
        return list(range(1, n + 1))
    if len(pos) != n:
        raise ValueError(f"positions has {len(pos)} entries for {n} groups")
    return [float(p) for p in pos]


def _points_overlay(ctx, data, pos, horiz, colors, dark=False):
    op, ax = ctx.op, ctx.ax
    jit = op.get("jitter")
    if not (jit or op.get("points")):
        return
    rng = np.random.default_rng(int(op.get("seed", 0)))
    w = 0.15 if jit is True else float(jit or 0.0)
    for i, (v, p) in enumerate(zip(data, pos)):
        off = p + (rng.uniform(-w, w, len(v)) if w else np.zeros(len(v)))
        xs, ys = (v, off) if horiz else (off, v)
        ax.scatter(xs, ys, s=op.get("s", 10), alpha=op.get("point_alpha", 0.6), zorder=3,
                   color=op.get("point_color", "k" if (colors is None or dark) else colors[i]), linewidths=0)


def _set_cat_ticks(ax, pos, labels, horiz):
    if horiz:
        ax.set_yticks(pos, labels)
    else:
        ax.set_xticks(pos, labels)


@register_op("box", f"box {{{GROUP_KEYS}, labels, positions, widths, orientation, showmeans, notch, whis, "
             "showfliers, color|colors (fill), alpha, median_color, jitter: true|width, points: true, s, "
             "point_color}", group="statistics")
def _op_box(ctx) -> None:
    _require_rect(ctx, "box")
    op, ax = ctx.op, ctx.ax
    groups = _groups(ctx)
    horiz = op.get("orientation", "vertical") == "horizontal"
    axis = ax.xaxis if horiz else ax.yaxis
    data = [_vdata(ctx, v, axis) for _, v, _ in groups]
    pos = _positions(op, len(data))
    labs = [lab for lab, _, _ in groups]
    kw = {"positions": pos, "orientation": "horizontal" if horiz else "vertical", "patch_artist": True,
          "showmeans": bool(op.get("showmeans", False)), "notch": bool(op.get("notch", False)),
          "showfliers": bool(op.get("showfliers", not (op.get("jitter") or op.get("points"))))}
    if op.get("widths") is not None:
        kw["widths"] = op["widths"]
    if op.get("whis") is not None:
        kw["whis"] = op["whis"]
    if any(labs):
        kw["tick_labels"] = labs
    if op.get("median_color"):
        kw["medianprops"] = {"color": _style(op, ("color",), "median_")["color"], "linewidth": 1.5}
    bp = ax.boxplot(data, **kw)
    cols = _colors(op, len(data)) or [f"C{i % 10}" for i in range(len(data))]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(float(op.get("alpha", 0.6)))
    if not op.get("median_color"):
        for m in bp["medians"]:
            m.set_color("k")
    _points_overlay(ctx, data, pos, horiz, cols)
    parts = []
    for lab, v, _ in groups[:10]:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        parts.append(f"{lab + ': ' if lab else ''}n={len(v)}, median={_g(med)}, IQR=[{_g(q1)}, {_g(q3)}], "
                     f"mean={_g(v.mean())}")
    ctx.note("box: " + "; ".join(parts))


@register_op("violin", f"violin {{{GROUP_KEYS}, labels, positions, widths, orientation, showmeans, showmedians "
             "(default true), showextrema, quantiles, bw, side: both|low|high, color|colors, alpha, jitter: true|"
             "width, points: true, s, point_color}", group="statistics")
def _op_violin(ctx) -> None:
    _require_rect(ctx, "violin")
    op, ax = ctx.op, ctx.ax
    groups = _groups(ctx)
    if any(len(v) < 2 for _, v, _ in groups):
        raise ValueError("violin needs at least 2 values per group")
    horiz = op.get("orientation", "vertical") == "horizontal"
    axis = ax.xaxis if horiz else ax.yaxis
    data = [_vdata(ctx, v, axis) for _, v, _ in groups]
    pos = _positions(op, len(data))
    kw = {"positions": pos, "orientation": "horizontal" if horiz else "vertical",
          "showmeans": bool(op.get("showmeans", False)), "showmedians": bool(op.get("showmedians", True)),
          "showextrema": bool(op.get("showextrema", True)), "widths": op.get("widths", 0.7),
          "side": op.get("side", "both")}
    if op.get("bw") is not None:
        kw["bw_method"] = op["bw"]
    if op.get("quantiles") is not None:
        q = op["quantiles"]
        kw["quantiles"] = [q] * len(data) if q and not isinstance(q[0], list) else q
    parts = ax.violinplot(data, **kw)
    cols = _colors(op, len(data)) or [f"C{i % 10}" for i in range(len(data))]
    for body, c in zip(parts["bodies"], cols):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(float(op.get("alpha", 0.5)))
    for k in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars", "cquantiles"):
        if k in parts:
            parts[k].set_color("0.2")
    labs = [lab for lab, _, _ in groups]
    if any(labs):
        _set_cat_ticks(ax, pos, labs, horiz)
    _points_overlay(ctx, data, pos, horiz, cols, dark=True)
    ctx.note("violin: " + _summ(groups))


@register_op("kde", f"kde {{{GROUP_KEYS}, weights|weights_col, bw ('scott'|'silverman'|factor), npoints, cut (3), "
             "range, fill: true, alpha, rug: true, scale: density|count, orientation, color|colors, linestyle, "
             "linewidth, label|labels}", group="statistics")
def _op_kde(ctx) -> None:
    from scipy.stats import gaussian_kde
    _require_rect(ctx, "kde")
    op, ax = ctx.op, ctx.ax
    groups = _groups(ctx, "values", ("weights",))
    horiz = op.get("orientation", "vertical") == "horizontal"
    axis = ax.yaxis if horiz else ax.xaxis
    cols = _colors(op, len(groups))
    info = []
    for i, (lab, v, ex) in enumerate(groups):
        v = _vdata(ctx, v, axis)
        if len(v) < 2 or np.ptp(v) == 0:
            raise ValueError(f"kde needs at least 2 distinct values (group {lab or i + 1})")
        k = gaussian_kde(v, bw_method=op.get("bw"), weights=ex.get("weights"))
        bw = float(np.sqrt(k.covariance[0, 0]))
        if op.get("range") is not None:
            lo, hi = _to_data(axis, op["range"], ctx.mode)[:2]
        else:
            cut = float(op.get("cut", 3))
            lo, hi = v.min() - cut * bw, v.max() + cut * bw
        g = np.linspace(lo, hi, int(op.get("npoints", 256)))
        d = k(g) * (len(v) if op.get("scale") == "count" else 1.0)
        kw = _style(op, ("linestyle", "linewidth", "zorder"))
        if cols:
            kw["color"] = cols[i]
        if lab:
            kw["label"] = lab
        line, = ax.plot(d, g, **kw) if horiz else ax.plot(g, d, **kw)
        c = line.get_color()
        if op.get("fill"):
            fb = ax.fill_betweenx if horiz else ax.fill_between
            fb(g, 0, d, color=c, alpha=float(op.get("alpha", 0.25)), linewidth=0)
        if op.get("rug"):
            tr = ax.get_yaxis_transform() if horiz else ax.get_xaxis_transform()
            r = np.zeros_like(v)
            if horiz:
                ax.plot(r, v, "_", color=c, markersize=10, transform=tr, alpha=0.7)
            else:
                ax.plot(v, r, "|", color=c, markersize=10, transform=tr, alpha=0.7)
        info.append(f"{lab + ': ' if lab else ''}n={len(v)}, bandwidth={_g(bw)}, mode≈{_g(g[np.argmax(d)])}")
    ctx.note("kde: " + "; ".join(info[:10]))


@register_op("ecdf", f"ecdf {{{GROUP_KEYS}, weights|weights_col, complementary, orientation, color|colors, "
             "linestyle, linewidth, alpha, label|labels}", group="statistics")
def _op_ecdf(ctx) -> None:
    _require_rect(ctx, "ecdf")
    op, ax = ctx.op, ctx.ax
    groups = _groups(ctx, "values", ("weights",))
    horiz = op.get("orientation", "vertical") == "horizontal"
    axis = ax.yaxis if horiz else ax.xaxis
    cols = _colors(op, len(groups))
    for i, (lab, v, ex) in enumerate(groups):
        kw = _style(op, ("linestyle", "linewidth", "alpha", "zorder"))
        if cols:
            kw["color"] = cols[i]
        if lab:
            kw["label"] = lab
        ax.ecdf(_vdata(ctx, v, axis), weights=ex.get("weights"), complementary=bool(op.get("complementary")),
                orientation="horizontal" if horiz else "vertical", **kw)
    ctx.note("ecdf: " + _summ(groups))


_QQ_DISTS = ("norm", "expon", "uniform", "logistic", "laplace", "gumbel_r", "gumbel_l", "cauchy")


@register_op("qq", f"qq {{{GROUP_KEYS}, dist: {'|'.join(_QQ_DISTS)} (default norm), line: true|false, color|colors, "
             "marker, s, label|labels}: probability plot, ordered values vs theoretical quantiles + fitted line "
             "(slope ≈ σ, intercept ≈ μ)", group="statistics")
def _op_qq(ctx) -> None:
    from scipy import stats
    _require_rect(ctx, "qq")
    op, ax = ctx.op, ctx.ax
    dist = op.get("dist", "norm")
    if dist not in _QQ_DISTS:
        raise ValueError(f"dist must be one of {_QQ_DISTS}")
    groups = _groups(ctx)
    cols = _colors(op, len(groups))
    info = []
    for i, (lab, v, _) in enumerate(groups):
        if len(v) < 3:
            raise ValueError("qq needs at least 3 values")
        (osm, osr), (slope, icpt, r) = stats.probplot(v, dist=dist)
        kw = {"marker": op.get("marker", "o"), "s": op.get("s", 18)}
        if cols:
            kw["color"] = cols[i]
        if lab:
            kw["label"] = lab
        sc = ax.scatter(osm, _vdata(ctx, osr, ax.yaxis), **kw)
        if op.get("line", True):
            xx = np.array([osm.min(), osm.max()])
            ax.plot(xx, _vdata(ctx, icpt + slope * xx, ax.yaxis), color=sc.get_facecolor()[0], linestyle="--",
                    linewidth=1.2)
        info.append(f"{lab + ': ' if lab else ''}slope={_g(slope)}, intercept={_g(icpt)}, r={r:.4f}, n={len(v)}")
    if not ax.get_xlabel():
        ax.set_xlabel(f"theoretical quantiles ({dist})")
    if not ax.get_ylabel():
        ax.set_ylabel("ordered values")
    ctx.note(f"qq ({dist}): " + "; ".join(info[:10]))


def _xy(ctx, what):
    x, y = ctx.X(), ctx.Y()
    if x is None or y is None:
        raise ValueError(f"{what} needs x and y (inline or x_col/y_col with a data source)")
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths ({len(x)} vs {len(y)})")
    return x, y


@register_op("hist2d", "hist2d {x, y (or x_col, y_col), bins (n|[nx, ny]|[xedges, yedges]), range: [[x0, x1], "
             "[y0, y1]], density, weights|weights_col, cmin, cmax, cmap, norm, vmin, vmax, alpha, colorbar}",
             group="statistics")
def _op_hist2d(ctx) -> None:
    _require_rect(ctx, "hist2d")
    op, ax = ctx.op, ctx.ax
    x, y = _xy(ctx, "hist2d")
    w = ctx.raw("weights")
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(w) if w is not None else True)
    kw = {"bins": op.get("bins", 30), "density": bool(op.get("density", False)), "cmap": op.get("cmap", "viridis")}
    if op.get("range") is not None:
        (x0, x1), (y0, y1) = op["range"]
        kw["range"] = [list(_to_data(ax.xaxis, [x0, x1], ctx.mode)), list(_to_data(ax.yaxis, [y0, y1], ctx.mode))]
    for k in ("cmin", "cmax", "alpha"):
        if op.get(k) is not None:
            kw[k] = op[k]
    if op.get("norm") is not None:
        kw["norm"] = make_norm(op["norm"])
    else:
        kw.update({k: op[k] for k in ("vmin", "vmax") if op.get(k) is not None})
    h, xe, ye, qm = ax.hist2d(x[ok], y[ok], weights=w[ok] if w is not None else None, **kw)
    ctx.colorbar(qm)
    ctx.note(f"hist2d: {h.shape[0]}x{h.shape[1]} bins, n={int(ok.sum())}, max bin={_g(np.nanmax(h))}")


_REDUCE = {"mean": np.mean, "median": np.median, "sum": np.sum, "max": np.max, "min": np.min, "std": np.std,
           "count": np.size}


@register_op("hexbin", "hexbin {x, y (or *_col), c|c_col (values to reduce), reduce: mean|median|sum|max|min|std|count, "
             "gridsize, mincnt, bins: 'log'|n, norm, cmap, vmin, vmax, xscale, yscale, extent, edgecolor, linewidth, "
             "alpha, colorbar}", group="statistics")
def _op_hexbin(ctx) -> None:
    _require_rect(ctx, "hexbin")
    op, ax = ctx.op, ctx.ax
    x, y = _xy(ctx, "hexbin")
    C = ctx.raw("c")
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(C) if C is not None else True)
    red = op.get("reduce", "mean")
    if red not in _REDUCE:
        raise ValueError(f"reduce must be one of {list(_REDUCE)}")
    kw = {"gridsize": op.get("gridsize", 30), "cmap": op.get("cmap", "viridis"),
          "xscale": op.get("xscale", "linear"), "yscale": op.get("yscale", "linear"),
          "reduce_C_function": _REDUCE[red], "edgecolors": op.get("edgecolor", "face")}
    for k in ("mincnt", "alpha", "vmin", "vmax"):
        if op.get(k) is not None:
            kw[k] = op[k]
    if op.get("linewidth") is not None:
        kw["linewidths"] = op["linewidth"]
    if op.get("extent") is not None:
        e = op["extent"]
        kw["extent"] = list(_to_data(ax.xaxis, e[:2], ctx.mode)) + list(_to_data(ax.yaxis, e[2:4], ctx.mode))
    if op.get("norm") is not None:
        if op.get("bins") is not None:
            raise ValueError("give either norm or bins, not both")
        kw["norm"] = make_norm(op["norm"])
        kw.pop("vmin", None)
        kw.pop("vmax", None)
    elif op.get("bins") is not None:
        kw["bins"] = op["bins"]
    hb = ax.hexbin(x[ok], y[ok], C=C[ok] if C is not None else None, **kw)
    ctx.colorbar(hb)
    arr = hb.get_array()
    arr = np.asarray(arr.compressed() if hasattr(arr, "compressed") else arr, dtype=float)
    ctx.note(f"hexbin: {len(arr)} cells shown, n={int(ok.sum())}, "
             f"cell {'count' if C is None else red} range [{_g(arr.min()) if len(arr) else '-'}, "
             f"{_g(arr.max()) if len(arr) else '-'}]")


_AGGS = ("std", "sem", "ci", "minmax", "iqr", "pct")


@register_op("band", "band {x, y + yerr (±) | ylo, yhi (or *_col); OR a data source with x_col, y_col, agg: "
             "std|sem|ci|minmax|iqr|pct (rows grouped by x), center: mean|median, nstd (1), level (ci, 0.95), "
             "pct: [lo, hi]; group_col (one band per group), color|colors, alpha (band, 0.25), linestyle, linewidth, "
             "marker, label|labels, band_label}: line with shaded band", group="statistics")
def _op_band(ctx) -> None:
    _require_2d(ctx, "band")
    op, ax = ctx.op, ctx.ax
    series = []                  # (label, x, y, lo, hi) in data coordinates
    if op.get("agg"):
        series = _band_agg(ctx)
    else:
        x, y = _xy(ctx, "band")
        if ctx.raw("yerr") is not None:
            e = ctx.dY("yerr")
            lo, hi = y - e, y + e
        elif ctx.raw("ylo") is not None and ctx.raw("yhi") is not None:
            lo, hi = ctx.Y("ylo"), ctx.Y("yhi")
        else:
            raise ValueError("band needs yerr, or ylo and yhi, or a data source with agg")
        o = np.argsort(x)
        series = [(op.get("label"), x[o], y[o], np.broadcast_to(lo, x.shape)[o], np.broadcast_to(hi, x.shape)[o])]
    cols = _colors(op, len(series))
    labels = op.get("labels")
    for i, (lab, x, y, lo, hi) in enumerate(series):
        kw = _style(op, ("linestyle", "linewidth", "marker", "markersize", "zorder"))
        if cols:
            kw["color"] = cols[i]
        lab = labels[i] if isinstance(labels, list) else lab
        if lab:
            kw["label"] = str(lab)
        line, = ax.plot(x, y, **kw)
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=float(op.get("alpha", 0.25)), linewidth=0,
                        label=op.get("band_label") if i == 0 else None)
    ctx.note(f"band: {len(series)} series, " + "; ".join(
        f"{(str(s[0]) + ': ') if s[0] else ''}{len(s[1])} points, mean half-width {_g(np.nanmean((s[4] - s[3]) / 2))}"
        for s in series[:10]))


def _band_agg(ctx):
    from scipy import stats
    op, ax = ctx.op, ctx.ax
    if not (ctx.has_source and op.get("x_col") and op.get("y_col")):
        raise ValueError("agg needs a data source with x_col and y_col")
    agg = op["agg"]
    if agg not in _AGGS:
        raise ValueError(f"agg must be one of {_AGGS}")
    df = ctx.frame()
    src = op.get("csv") or op.get("data")
    xc, yc = ctx.colname("x"), ctx.colname("y")
    parts = [(None, df)]
    if op.get("group_col"):
        import pandas as pd
        gc = find_col(df, op["group_col"], src)
        keys = op.get("groups") or list(pd.unique(df[gc].dropna()))
        parts = [(str(k), df[df[gc] == k]) for k in keys]
    out = []
    center = op.get("center", "mean")
    for lab, sub in parts:
        sub = sub[[xc, yc]].apply(lambda s: s.astype(float)).dropna()
        gb = sub.groupby(xc)[yc]
        xs = np.asarray(sorted(gb.groups), dtype=float)
        c = (gb.median() if center == "median" else gb.mean()).reindex(xs).to_numpy()
        if agg in ("std", "sem", "ci"):
            sd = gb.std(ddof=1).reindex(xs).to_numpy()
            cnt = gb.count().reindex(xs).to_numpy()
            if agg == "std":
                h = float(op.get("nstd", 1.0)) * sd
            else:
                h = sd / np.sqrt(cnt)
                if agg == "ci":
                    h = h * stats.t.ppf(0.5 + float(op.get("level", 0.95)) / 2, np.maximum(cnt - 1, 1))
            lo, hi = c - np.nan_to_num(h), c + np.nan_to_num(h)
        else:
            q = {"minmax": (0, 100), "iqr": (25, 75)}.get(agg) or tuple(op.get("pct", (2.5, 97.5)))
            lo = gb.quantile(q[0] / 100).reindex(xs).to_numpy()
            hi = gb.quantile(q[1] / 100).reindex(xs).to_numpy()
        tx = _to_data(ax.xaxis, xs, ctx.mode)
        out.append((lab if lab is not None else op.get("label"), tx, _to_data(ax.yaxis, c, ctx.mode),
                    _to_data(ax.yaxis, lo, ctx.mode), _to_data(ax.yaxis, hi, ctx.mode)))
    return out


def _numeric_cols(ctx, maxn=None):
    df = ctx.frame()
    src = ctx.op.get("csv") or ctx.op.get("data")
    want = ctx.op.get("cols") or ctx.op.get("value_cols")
    if want:
        cols = [find_col(df, c, src) for c in want]
    else:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype
        cols = [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
                and c != ctx.op.get("group_col")]
    if len(cols) < 2:
        raise ValueError(f"need at least 2 numeric columns; columns: {list(df.columns)}")
    if maxn and len(cols) > maxn:
        raise ValueError(f"{len(cols)} columns is too many (max {maxn}); choose some with 'cols'")
    return df, cols


@register_op("corr_heatmap", "corr_heatmap {data|csv (+ where), cols (default: all numeric), method: pearson|"
             "spearman|kendall, annotate: true, fmt: '.2f', fontsize, triangle: lower|upper, cmap (RdBu_r), "
             "colorbar (default true)}: correlation matrix", group="statistics")
def _op_corr(ctx) -> None:
    _require_rect(ctx, "corr_heatmap")
    op, ax = ctx.op, ctx.ax
    df, cols = _numeric_cols(ctx, 30)
    method = op.get("method", "pearson")
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be pearson, spearman or kendall")
    C = df[cols].apply(lambda s: s.astype(float)).corr(method=method).to_numpy()
    M = C.copy()
    tri = op.get("triangle")
    if tri in ("lower", "upper"):
        mask = np.triu(np.ones_like(M, bool), 1) if tri == "lower" else np.tril(np.ones_like(M, bool), -1)
        M[mask] = np.nan
    n = len(cols)
    im = ax.imshow(M, cmap=op.get("cmap", "RdBu_r"), vmin=-1, vmax=1, interpolation="nearest")
    labels = [str(c) for c in cols]
    ax.set_xticks(range(n), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    if op.get("annotate", True):
        fmt = "{:" + str(op.get("fmt", ".2f")) + "}"
        fs = op.get("fontsize", max(5, 10 - n // 3))
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs,
                            color="white" if abs(M[i, j]) > 0.6 else "black")
    if "colorbar" not in op:
        ctx.op = {**op, "colorbar": {"label": f"{method} r"}}
    ctx.colorbar(im)
    pairs = sorted(((abs(C[i, j]), labels[i], labels[j], C[i, j]) for i in range(n) for j in range(i + 1, n)),
                   reverse=True)
    ctx.note(f"corr_heatmap ({method}, {n} columns, n={len(df)}): strongest pairs "
             + ", ".join(f"{a}~{b} {r:+.3f}" for _, a, b, r in pairs[:10]))


@register_op("scatter_matrix", "scatter_matrix {data|csv (+ where), cols (default: numeric, max 8), group_col "
             "(colour by group), diag: hist|kde, bins, s, alpha, color, corner: true (lower triangle)}: replaces the "
             "target axes with an n×n grid of new axes (indices reported); use as the panel's last op",
             group="statistics")
def _op_scatter_matrix(ctx) -> None:
    from scipy.stats import gaussian_kde
    _require_rect(ctx, "scatter_matrix")
    op, ax, fig = ctx.op, ctx.ax, ctx.fig
    ss = ax.get_subplotspec()
    if ss is None:
        raise ValueError("scatter_matrix needs a grid panel axes (not an inset or twin axes)")
    df, cols = _numeric_cols(ctx, 8)
    n = len(cols)
    src = op.get("csv") or op.get("data")
    if op.get("group_col"):
        import pandas as pd
        gc = find_col(df, op["group_col"], src)
        keys = op.get("groups") or list(pd.unique(df[gc].dropna()))
        parts = [(str(k), df[df[gc] == k]) for k in keys]
    else:
        parts = [(None, df)]
    base = op.get("color")
    colors = [base] if (base and len(parts) == 1) else [f"C{i % 10}" for i in range(len(parts))]
    corner = bool(op.get("corner"))
    gs = ss.subgridspec(n, n, wspace=0.08, hspace=0.08)
    ax.remove()
    axs = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if corner and j > i:
                continue
            share = {}
            col_top = axs[j][j] if corner else axs[0][j]
            if col_top is not None:
                share["sharex"] = col_top
            if i != j:
                row_first = next((axs[i][k] for k in range(n) if k != i and axs[i][k] is not None), None)
                if row_first is not None:
                    share["sharey"] = row_first
            a = fig.add_subplot(gs[i, j], **share)
            axs[i][j] = a
            for (lab, sub), c in zip(parts, colors):
                xv = sub[cols[j]].to_numpy(dtype=float)
                if i == j:
                    xv = xv[np.isfinite(xv)]
                    if op.get("diag", "hist") == "kde" and len(xv) > 1 and np.ptp(xv) > 0:
                        k = gaussian_kde(xv)
                        g = np.linspace(xv.min(), xv.max(), 128)
                        a.plot(g, k(g), color=c)
                        a.fill_between(g, 0, k(g), color=c, alpha=0.25, linewidth=0)
                    else:
                        a.hist(xv, bins=op.get("bins", 15), color=c, alpha=0.6 if len(parts) > 1 else 0.8)
                    a.tick_params(left=False, labelleft=False)
                else:
                    yv = sub[cols[i]].to_numpy(dtype=float)
                    a.scatter(xv, yv, s=op.get("s", 8), alpha=op.get("alpha", 0.7), color=c, linewidths=0,
                              label=lab if (lab and i == 1 and j == 0) else None)
            if i < n - 1:
                a.tick_params(labelbottom=False)
            else:
                a.set_xlabel(str(cols[j]))
            if j > 0 and i != j:
                a.tick_params(labelleft=False)
            if j == 0 and i > 0:
                a.set_ylabel(str(cols[i]))
            if i == 0 and j == n - 1 and n > 1 and not corner:     # row 0 starts with a diagonal: label right
                a.tick_params(labelright=True)
                a.yaxis.set_label_position("right")
                a.set_ylabel(str(cols[0]))
            a.tick_params(labelsize=7)
    if len(parts) > 1:
        axs[1][0].legend(fontsize=7, markerscale=1.5, loc="best")
    made = [a for row in axs for a in row if a is not None]
    idx = [main_axes(fig).index(a) for a in made]
    ctx.note(f"scatter_matrix of {cols}: target axes replaced by {len(made)} axes, indices {idx[0]}..{idx[-1]} "
             f"(row-major{', lower triangle' if corner else ''})")


# ---------------------------------------------------------------------------
# agent tool: numbers only
# ---------------------------------------------------------------------------
def register(tb) -> None:
    from matplotlib.figure import Figure
    from .figures import OpContext
    S = {"type": "string"}
    N = {"type": "number"}
    NUMS = {"type": "array", "items": {"type": "number"}}

    @tb.add("fit_data", "Fit a model to x/y data (file or inline) WITHOUT drawing: parameters ± standard errors, R², "
            "RMSE, n, dof, plus CSVs of the fitted curve (with confidence band) and residuals. Models as the fit "
            "op: " + MODELS_HELP + ". To draw the fit use make_plot/edit_figure op 'fit'.",
            {"data": {"description": "path (csv/xlsx/json...) or inline records / {col: [..]}"},
             "x": NUMS, "y": NUMS, "sigma": NUMS, "x_col": S, "y_col": S, "sigma_col": S,
             "where": {"type": "object"}, "transform": {"type": "array", "items": {"type": "object"}},
             "model": {"description": "model name, or {expr, params} for a custom formula"},
             "deg": {"type": "integer"}, "offset": {"type": "boolean"}, "T_unit": S, "expr": S,
             "params": {"type": "object", "description": "custom: {name: start value}"},
             "p0": {"type": "object"}, "bounds": {"type": "object"}, "linearize": {"type": "boolean"},
             "y_is": S, "absolute_sigma": {"type": "boolean"}, "range": NUMS, "npoints": {"type": "integer"},
             "level": N, "band": {"type": "string", "enum": ["confidence", "prediction"]},
             "output_csv": {"type": "string", "description": "default fit_<model>.csv (+ _residuals.csv)"}}, [])
    def fit_data(output_csv: Optional[str] = None, npoints: int = 200, level: float = 0.95,
                 band: str = "confidence", range=None, **kw):     # noqa: A002 - JSON name
        import pandas as pd
        op = {k: v for k, v in kw.items() if v is not None}
        if "data" not in op and ("x_col" in op or "y_col" in op):
            raise ValueError("x_col/y_col need 'data' (a file path or inline records)")
        fig = Figure()
        ax = fig.add_subplot()
        notes: List[str] = []
        ctx = OpContext(fig, ax, op, 1, tb.p, notes)
        res = fit_from_ctx(ctx)
        lo, hi = (float(range[0]), float(range[1])) if range else (float(res.x.min()), float(res.x.max()))
        xg = np.linspace(lo, hi, int(npoints))
        yh, ylo, yhi = res.predict(xg, band, float(level))
        out = tb.outp(output_csv, f"fit_{res.model.name}.csv")
        if not out.lower().endswith(".csv"):
            out = str(Path(out).with_suffix(".csv"))
        curve = {"x": xg, "y_fit": yh}
        if ylo is not None:
            curve[f"{band}_lo"] = ylo
            curve[f"{band}_hi"] = yhi
        pd.DataFrame(curve).to_csv(out, index=False)
        rout = str(Path(out).with_name(Path(out).stem + "_residuals.csv"))
        pd.DataFrame({"x": res.x, "y": res.y, "y_fit": res.back(res.fitted_t),
                      "residual" + ("" if res.space == "y" else f" ({res.space})"): res.resid}).to_csv(rout, index=False)
        return (res.summary() + f"\ncurve ({len(xg)} points, {int(level * 100)}% {band} band): {tb.rel(out)}; "
                f"residuals: {tb.rel(rout)}"), []
