"""Data preparation layer for plot operations: file loaders, "where" filters and "transform" pipelines.

Every figure operation with a data source (``"data"``/``"csv"``) is loaded through :func:`load_source`:

1. the source: a path to CSV/TSV/TXT (delimiter detected), Excel, JSON, netCDF, or inline records/columns/CSV text,
2. ``"where"``: equality and comparison filters,
3. ``"transform"``: a list of pipeline steps (derive, groupby, pivot, join, interpolate, smooth, convert ...).

Expressions (``derive``/``filter``) are evaluated by a small AST interpreter (:class:`SafeExpr`): arithmetic,
comparisons, and/or/not, numbers, strings, column names and a fixed set of numpy functions.  No Python ``eval``,
no attribute access, no imports.

The module also adds the agent tools ``inspect_data`` and ``transform_data``.
"""
from __future__ import annotations

import ast
import glob
import io
import json
import operator
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import figures
from .expdata import ATOMIC_MASS
from .figures import find_col

# ---------------------------------------------------------------------------
# syntax help (returned by plot_ops_help(["data"]) / plot_ops_help(["transform"]))
# ---------------------------------------------------------------------------
DATA_HELP = """Data sources (any op): "data" (or "csv"): path to .csv/.tsv/.txt/.dat (delimiter auto-detected, leading
# comment lines skipped; "read": {sep, skiprows, header, comment, decimal, thousands, names, encoding, na_values,
nrows, usecols}), .xlsx/.xls ("sheet": name|index), .json (records, {col: [..]} or JSON lines), .nc netCDF (flat
table of coordinates + variables; "variables": [..]); or inline [{col: v}, ..] | {col: [..]} | "csv text".
Columns: "<key>_col" = exact name or an unambiguous prefix/substring ("T" -> "T [K]").
"where": {col: value | [values] | {">": 800, "<=": 1200, "==": v, "!=": v, "between": [a, b], "in": [..],
"not": v|[..], "contains": "fcc" (case-insensitive), "startswith": s, "isna": true|false}} (all AND-ed), applied
before "transform": [steps] (plot_ops_help(["transform"])). inspect_data shows an unknown file's columns first."""

EXPR_HELP = ("Expressions: + - * / // % **, == != < <= > >=, and/or/not, x if c else y, `col in ['A','B']`; "
             "names = columns (`T [K]` or col('T [K]') for odd names) or constants pi e R(8.314462618) F kB nan inf; "
             "functions log ln log10 log2 exp sqrt abs sin cos tan arcsin arccos arctan tanh floor ceil "
             "round(x,n) clip(x,lo,hi) where(c,a,b) min max (1 arg = column min, 2+ = elementwise) mean std sum "
             "median cumsum diff isna num(text->number) contains(col,'s') startswith(col,'s').")


# ---------------------------------------------------------------------------
# safe expression evaluator
# ---------------------------------------------------------------------------
CONSTANTS = {"pi": np.pi, "e": np.e, "R": 8.314462618, "F": 96485.33212, "kB": 1.380649e-23, "nan": np.nan,
             "inf": np.inf}


def _reduce(name):
    def f(x):
        if isinstance(x, pd.Series):
            return getattr(x, name)()
        return getattr(np, "nan" + name)(np.asarray(x, dtype=float))
    return f


def _minmax(np_fn, name):
    red = _reduce(name)

    def f(*a):
        if not a:
            raise ValueError(f"{name}() needs arguments")
        if len(a) == 1:
            return red(a[0])
        out = a[0]
        for v in a[1:]:
            out = np_fn(out, v)
        return out
    return f


def _series_method(name):
    def f(x):
        if not isinstance(x, pd.Series):
            raise ValueError(f"{name}() needs a column")
        return getattr(x, name)()
    return f


def _contains(s, pat):
    if isinstance(s, pd.Series):
        return s.astype(str).str.contains(str(pat), case=False, regex=False)
    return str(pat).lower() in str(s).lower()


def _startswith(s, pat):
    if isinstance(s, pd.Series):
        return s.astype(str).str.startswith(str(pat))
    return str(s).startswith(str(pat))


def _num(x):
    return pd.to_numeric(x, errors="coerce") if isinstance(x, pd.Series) else float(x)


FUNCS: Dict[str, Callable] = {
    "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "tanh": np.tanh, "floor": np.floor, "ceil": np.ceil,
    "round": lambda x, n=0: np.round(x, int(n)),
    "clip": lambda x, lo, hi: np.clip(x, lo, hi),
    "where": lambda c, a, b: np.where(c, a, b),
    "min": _minmax(np.minimum, "min"), "max": _minmax(np.maximum, "max"),
    "mean": _reduce("mean"), "std": _reduce("std"), "sum": _reduce("sum"), "median": _reduce("median"),
    "cumsum": _series_method("cumsum"), "diff": _series_method("diff"),
    "isna": pd.isna, "isnan": pd.isna, "num": _num, "contains": _contains, "startswith": _startswith,
}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMPOPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
           ast.Gt: operator.gt, ast.GtE: operator.ge}
MAX_EXPR_CHARS, MAX_EXPR_NODES = 2000, 400


class SafeExpr:
    """Evaluate an expression over DataFrame columns without Python eval (AST whitelist interpreter)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, expr):
        if isinstance(expr, (int, float, bool)):
            return expr
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError(f"expression must be a non-empty string, got {expr!r}")
        if len(expr) > MAX_EXPR_CHARS:
            raise ValueError(f"expression longer than {MAX_EXPR_CHARS} characters")
        text = re.sub(r"`([^`]*)`", lambda m: "col(" + repr(m.group(1)) + ")", expr.strip())
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as e:
            hint = " (use == for equality)" if re.search(r"[^=!<>]=[^=]", text) else ""
            raise ValueError(f"cannot parse expression {expr!r}: {e.msg}{hint}. Column names with spaces or "
                             "brackets: `T [K]` or col('T [K]')") from None
        if sum(1 for _ in ast.walk(tree)) > MAX_EXPR_NODES:
            raise ValueError("expression too complex")
        out = self._ev(tree.body)
        if isinstance(out, np.ndarray) and out.ndim == 1 and len(out) == len(self.df):
            out = pd.Series(out, index=self.df.index)
        return out

    # -- names ------------------------------------------------------------
    def column(self, name: str) -> pd.Series:
        return self.df[find_col(self.df, name)]

    def _name(self, name: str):
        if name.startswith("_"):
            raise ValueError(f"name {name!r} is not allowed (names starting with '_' are reserved)")
        if name in self.df.columns:
            return self.df[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in FUNCS or name == "col":
            raise ValueError(f"{name} is a function: call it, e.g. {name}(...)")
        try:
            return self.column(name)
        except ValueError:
            raise ValueError(f"unknown name {name!r}: not a column or constant. Columns: {list(self.df.columns)}; "
                             f"constants: {sorted(CONSTANTS)}. Write odd column names as `T [K]` or col('T [K]')."
                             ) from None

    # -- nodes ------------------------------------------------------------
    def _ev(self, node):  # noqa: C901 - one branch per allowed node type
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or v is None:
                return np.nan if v is None else v
            if isinstance(v, (int, float)):
                return np.float64(v)       # float: overflow gives inf instead of huge ints
            if isinstance(v, str):
                return v
            raise ValueError(f"constant {v!r} not allowed")
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.BinOp):
            fn = _BINOPS.get(type(node.op))
            if fn is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            a, b = self._ev(node.left), self._ev(node.right)
            if isinstance(a, str) != isinstance(b, str) or (isinstance(a, str) and fn is not operator.add):
                raise ValueError("strings only support + with other strings")
            with np.errstate(all="ignore"):
                return fn(a, b)
        if isinstance(node, ast.UnaryOp):
            v = self._ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.Not):
                return ~v.astype(bool) if isinstance(v, pd.Series) else (not v)
            raise ValueError(f"operator {type(node.op).__name__} not allowed (use not)")
        if isinstance(node, ast.BoolOp):
            vals = [self._ev(v) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (_as_bool(out) & _as_bool(v)) if isinstance(node.op, ast.And) else (_as_bool(out) | _as_bool(v))
            return out
        if isinstance(node, ast.Compare):
            left, out = self._ev(node.left), None
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    if not isinstance(comp, (ast.List, ast.Tuple, ast.Set)):
                        raise ValueError("'in' needs a literal list, e.g. phase in ['FCC_A1', 'BCC_A2']")
                    vals = [self._ev(e) for e in comp.elts]
                    r = left.isin(vals) if isinstance(left, pd.Series) else (left in vals)
                    r = (~r if isinstance(r, pd.Series) else (not r)) if isinstance(op, ast.NotIn) else r
                    right = None
                else:
                    fn = _CMPOPS.get(type(op))
                    if fn is None:
                        raise ValueError(f"comparison {type(op).__name__} not allowed")
                    right = self._ev(comp)
                    r = fn(left, right)
                out = r if out is None else (_as_bool(out) & _as_bool(r))
                left = right
            return out
        if isinstance(node, ast.IfExp):
            return np.where(_as_bool(self._ev(node.test)), self._ev(node.body), self._ev(node.orelse))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only plain function calls are allowed, e.g. log10(x) (no attribute access or "
                                 "methods)")
            name = node.func.id
            if node.keywords:
                raise ValueError(f"{name}(): keyword arguments are not allowed; pass values in order")
            if name == "col":
                if len(node.args) != 1 or not (isinstance(node.args[0], ast.Constant)
                                               and isinstance(node.args[0].value, str)):
                    raise ValueError("col() takes one string: col('T [K]')")
                return self.column(node.args[0].value)
            fn = FUNCS.get(name)
            if fn is None:
                raise ValueError(f"function {name!r} not allowed; allowed: {sorted(FUNCS) + ['col']}")
            args = [self._ev(a) for a in node.args]
            try:
                with np.errstate(all="ignore"):
                    return fn(*args)
            except TypeError as e:
                raise ValueError(f"{name}(): {e}") from None
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access ('.') is not allowed in expressions")
        if isinstance(node, (ast.Starred, ast.List, ast.Tuple, ast.Set)):
            raise ValueError("lists are only allowed after 'in'")
        raise ValueError(f"{type(node).__name__} is not allowed in expressions. {EXPR_HELP}")


def _as_bool(v):
    if isinstance(v, pd.Series):
        return v.fillna(False).astype(bool) if v.dtype == object else v.astype(bool)
    if isinstance(v, np.ndarray):
        return v.astype(bool)
    return bool(v)


def evaluate(df: pd.DataFrame, expr) -> Any:
    return SafeExpr(df)(expr)


# ---------------------------------------------------------------------------
# reading files
# ---------------------------------------------------------------------------
READ_OPTS_TEXT = {"sep", "delimiter", "skiprows", "header", "comment", "decimal", "thousands", "names", "encoding",
                  "na_values", "nrows", "usecols", "skipfooter"}
READ_OPTS_EXCEL = {"skiprows", "header", "names", "na_values", "nrows", "usecols", "decimal", "thousands"}
TEXT_EXT = (".csv", ".tsv", ".txt", ".dat", ".tab")
EXCEL_EXT = (".xlsx", ".xlsm", ".xls")
NC_EXT = (".nc", ".nc4", ".cdf", ".netcdf")
JSON_EXT = (".json", ".jsonl", ".ndjson")
_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_CACHE_MAX = 8


def _sniff(lines: List[str], ext: str) -> Tuple[str, Optional[str]]:
    """(sep, decimal) for the first lines of a text table (quote-aware field counts)."""
    import csv
    if ext == ".tsv":
        return "\t", None
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return ",", None
    for cand in ("\t", ",", ";", "|"):
        try:
            counts = [len(r) for r in csv.reader(body, delimiter=cand)]
        except csv.Error:
            continue
        if counts[0] > 1 and sum(c == counts[0] for c in counts) >= 0.8 * len(counts):
            dec = "," if cand == ";" and any(re.search(r"\d,\d", ln) for ln in body[1:]) else None
            return cand, dec
    return r"\s+", None


def _read_text(path: str, opts: Dict, src_text: Optional[str] = None) -> pd.DataFrame:
    ext = Path(path).suffix.lower() if path else ""
    opts = dict(opts)
    if "delimiter" in opts:
        opts["sep"] = opts.pop("delimiter")
    if "sep" not in opts:
        if src_text is not None:
            lines = src_text.splitlines()[:40]
        else:
            lines = []
            with open(path, "r", encoding=opts.get("encoding", "utf-8"), errors="replace") as f:
                for ln in f:
                    lines.append(ln.rstrip("\r\n"))
                    if len(lines) >= 60:
                        break
        skip = 0
        while skip < len(lines) and (not lines[skip].strip() or lines[skip].lstrip().startswith("#")):
            skip += 1
        sep, dec = _sniff(lines[skip:skip + 30], ext)
        opts["sep"] = sep
        if dec and "decimal" not in opts:
            opts["decimal"] = dec
        if skip and "skiprows" not in opts:
            opts["skiprows"] = skip
    sep = opts.get("sep")
    if (isinstance(sep, str) and len(sep) > 1 and sep != r"\s+") or opts.get("skipfooter"):
        opts["engine"] = "python"
    return pd.read_csv(io.StringIO(src_text) if src_text is not None else path, **opts)


def _read_json(path: str) -> pd.DataFrame:
    if path.lower().endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True)
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return pd.read_json(io.StringIO(txt), lines=True)
    return _inline_frame(obj, path)


def _inline_frame(obj, where="inline data") -> pd.DataFrame:
    if isinstance(obj, list):
        if obj and all(isinstance(r, dict) for r in obj):
            return pd.json_normalize(obj)
        if obj and all(isinstance(r, (list, tuple)) for r in obj):
            return pd.DataFrame(obj)
        raise ValueError(f"{where}: a list must hold records [{{col: value}}, ...]")
    if isinstance(obj, dict):
        for k in ("data", "records", "rows"):        # {"data": [records]} wrappers
            v = obj.get(k)
            if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
                return _inline_frame(obj[k], where)
        if all(not isinstance(v, (list, tuple, dict)) for v in obj.values()):
            return pd.DataFrame([obj])
        try:
            return pd.DataFrame(obj)
        except ValueError as e:
            raise ValueError(f"{where}: columns {{col: [values]}} must have equal lengths ({e})") from None
    if isinstance(obj, str):
        return _read_text("", {}, src_text=obj)
    raise ValueError(f"{where}: expected records, columns or CSV text, got {type(obj).__name__}")


def _read_netcdf(path: str, variables=None) -> pd.DataFrame:
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover
        raise ValueError("reading netCDF needs xarray") from None
    try:
        ds = xr.open_dataset(path)
    except (ValueError, OSError) as e:
        raise ValueError(f"cannot open {path} as netCDF ({e}). netCDF4/HDF5 files need the netCDF4 or h5netcdf "
                         "package; netCDF3 files work with scipy") from None
    with ds:
        if variables:
            variables = [variables] if isinstance(variables, str) else list(variables)
            missing = [v for v in variables if v not in ds.variables]
            if missing:
                raise ValueError(f"variables {missing} not in {Path(path).name}; data variables: "
                                 f"{list(ds.data_vars)}, coordinates: {list(ds.coords)}")
            ds = ds[variables]
        ds = ds.load()
    if not ds.data_vars:
        return ds.coords.to_dataset().to_dataframe().reset_index()
    return ds.to_dataframe().reset_index()


def read_path(path: str, sheet=None, variables=None, read: Optional[Dict] = None) -> pd.DataFrame:
    """Read one data file (cached by path + mtime)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"data file not found: {path}")
    ext = Path(path).suffix.lower()
    read = dict(read or {})
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, json.dumps([sheet, variables, read], default=str,
                                                                          sort_keys=True))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key].copy()
    if ext in EXCEL_EXT:
        bad = set(read) - READ_OPTS_EXCEL
        if bad:
            raise ValueError(f"read options {sorted(bad)} not allowed for Excel; allowed: {sorted(READ_OPTS_EXCEL)}")
        try:
            df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, **read)
        except ImportError as e:
            raise ValueError(f"reading {ext} needs an extra package: {e}") from None
        except ValueError as e:
            if "not found" in str(e).lower() or "worksheet" in str(e).lower():
                names = pd.ExcelFile(path).sheet_names
                raise ValueError(f"sheet {sheet!r} not in {Path(path).name}; sheets: {names}") from None
            raise
    elif ext in NC_EXT:
        df = _read_netcdf(path, variables)
    elif ext in JSON_EXT:
        df = _read_json(path)
    else:
        bad = set(read) - READ_OPTS_TEXT
        if bad:
            raise ValueError(f"read options {sorted(bad)} not allowed; allowed: {sorted(READ_OPTS_TEXT)}")
        if ext not in TEXT_EXT and ext:
            with open(path, "rb") as f:
                head = f.read(2048)
            if b"\0" in head:
                raise ValueError(f"{Path(path).name}: unsupported binary format {ext}; use csv/tsv/txt, xlsx/xls, "
                                 "json or nc")
        df = _read_text(path, read)
    df.columns = [str(c).strip() if not isinstance(c, str) else c.strip() for c in df.columns]
    _CACHE[key] = df
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return df.copy()


# ---------------------------------------------------------------------------
# sources, where, pipelines
# ---------------------------------------------------------------------------
SPEC_KEYS = {"data", "csv", "path", "sheet", "variables", "read", "where", "transform"}


def _is_spec(obj) -> bool:
    """{data|csv|path: path | records | {col: [..]}, sheet, where, transform ...} (not a plain columns dict)."""
    if not isinstance(obj, dict) or not set(obj) <= SPEC_KEYS:
        return False
    for k in ("data", "csv", "path"):
        v = obj.get(k)
        if isinstance(v, (str, dict)) or (isinstance(v, list) and v and all(isinstance(r, dict) for r in v)):
            return True
    return False


def load_source(src, resolve: Callable = lambda p: p, sheet=None, variables=None, read=None, where=None,
                transform=None, _depth: int = 0) -> pd.DataFrame:
    """DataFrame from a source (path | inline | spec dict), then ``where`` filters and ``transform`` steps."""
    if _depth > 5:
        raise ValueError("data sources nested too deeply")
    if _is_spec(src):
        spec = src
        src = spec.get("data", spec.get("csv", spec.get("path")))
        sheet = spec.get("sheet", sheet)
        variables = spec.get("variables", variables)
        read = spec.get("read", read)
        where = spec.get("where", where)
        transform = spec.get("transform", transform)
        return load_source(src, resolve, sheet, variables, read, where, transform, _depth + 1)
    if src is None:
        raise ValueError("no data source: give data/csv as a path or inline records")
    if isinstance(src, str) and ("\n" not in src):
        path = resolve(src)
        df = read_path(path, sheet=sheet, variables=variables, read=read)
        name = Path(src).stem
    else:
        df = _inline_frame(src)
        name = "data"
    if where:
        df = apply_where(df, where)
    if transform:
        df = run_transform(df, transform, resolve, name, _depth)
    return df


def _cond(s: pd.Series, spec: Dict, col: str) -> pd.Series:
    m = pd.Series(True, index=s.index)
    for k, v in spec.items():
        kk = str(k).lower()
        try:
            if kk in (">", "gt"):
                m &= s > v
            elif kk in (">=", "ge", "gte"):
                m &= s >= v
            elif kk in ("<", "lt"):
                m &= s < v
            elif kk in ("<=", "le", "lte"):
                m &= s <= v
            elif kk in ("==", "eq", "="):
                m &= s == v
            elif kk in ("!=", "ne"):
                m &= s != v
            elif kk == "between":
                lo, hi = v
                m &= (s >= lo) & (s <= hi)
            elif kk == "in":
                m &= s.isin(v if isinstance(v, list) else [v])
            elif kk in ("not", "not_in"):
                m &= ~s.isin(v if isinstance(v, list) else [v])
            elif kk == "contains":
                m &= s.astype(str).str.contains(str(v), case=False, regex=False)
            elif kk == "startswith":
                m &= s.astype(str).str.startswith(str(v))
            elif kk == "isna":
                m &= s.isna() if v else s.notna()
            else:
                raise ValueError(f"unknown condition {k!r}; use >, >=, <, <=, ==, !=, between, in, not, contains, "
                                 "startswith, isna")
        except TypeError as e:
            raise ValueError(f"where {col!r} {k} {v!r}: cannot compare ({e}); column dtype {s.dtype}") from None
    return m.fillna(False).astype(bool)


def apply_where(df: pd.DataFrame, where: Dict) -> pd.DataFrame:
    if not isinstance(where, dict):
        raise ValueError('"where" must be an object {col: value | [values] | {">": v, ...}}')
    mask = pd.Series(True, index=df.index)
    for col, val in where.items():
        c = find_col(df, col)
        if isinstance(val, dict):
            mask &= _cond(df[c], val, c)
        else:
            mask &= df[c].isin(val if isinstance(val, list) else [val])
    return df[mask]


STEPS: Dict[str, Tuple[Callable, str, set]] = {}
ALIASES = {"mutate": "derive", "assign": "derive", "query": "filter", "where": "filter", "group": "groupby",
           "aggregate": "groupby", "merge": "join", "append": "concat", "stack": "concat", "resample": "bin",
           "units": "convert", "unique": "dedupe", "drop_duplicates": "dedupe", "diff": "derivative",
           "differentiate": "derivative", "gradient": "derivative", "integral": "integrate", "cumtrapz": "integrate",
           "wide": "pivot", "long": "melt", "unpivot": "melt", "limit": "head", "fillna": "dropna"}


def step(name: str, syntax: str, keys: Sequence[str] = ()):
    """Register a transform step: fn(df, st, env) -> DataFrame."""
    def deco(fn):
        STEPS[name] = (fn, syntax, set(keys) | {"step"})
        return fn
    return deco


def run_transform(df: pd.DataFrame, steps, resolve: Callable = lambda p: p, src_name: str = "data",
                  _depth: int = 0) -> pd.DataFrame:
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list):
        raise ValueError('"transform" must be a list of steps [{"step": NAME, ...}]')
    env = {"resolve": resolve, "src": src_name, "depth": _depth}
    for i, st in enumerate(steps, 1):
        if not isinstance(st, dict) or "step" not in st:
            raise ValueError(f"transform step {i} must be an object {{\"step\": NAME, ...}}; steps: {sorted(STEPS)}")
        name = str(st["step"]).lower()
        name = ALIASES.get(name, name)
        if name not in STEPS:
            close = [k for k in STEPS if name in k or k in name]
            raise ValueError(f"transform step {i}: unknown step {st['step']!r}" + (f" (similar: {close})" if close
                             else "") + f"; steps: {sorted(STEPS)}")
        fn, syntax, keys = STEPS[name]
        try:
            bad = set(st) - keys
            if bad:
                raise ValueError(f"unknown key(s) {sorted(bad)}")
            out = fn(df, st, env)
        except Exception as e:  # noqa: BLE001 - report which step failed, with its syntax and the columns
            msg = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            raise ValueError(f"transform step {i} ({name}) failed: {msg}\n  syntax: {syntax}\n  columns before "
                             f"this step: {list(df.columns)}") from None
        df = out.reset_index(drop=True) if not isinstance(out.index, pd.RangeIndex) else out
    return df


# -- helpers ----------------------------------------------------------------
def _cols(df, v, key="cols") -> List[str]:
    if v is None:
        raise ValueError(f"missing '{key}'")
    return [find_col(df, c) for c in ([v] if isinstance(v, str) else v)]


def _opt_cols(df, v) -> List[str]:
    return [] if v is None else _cols(df, v)


def _groups(df, by: List[str]):
    if not by:
        yield (), df
        return
    for k, g in df.groupby(by, sort=True, dropna=False):
        yield (k if isinstance(k, tuple) else (k,)), g


def _num(df, c) -> np.ndarray:
    try:
        return pd.to_numeric(df[c]).to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"column {c!r} is not numeric (dtype {df[c].dtype}); e.g. {df[c].head(3).tolist()}") from None


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.10g}"
    return str(v)


def _grid(onto, x: np.ndarray) -> np.ndarray:
    if isinstance(onto, (list, tuple)):
        return np.asarray(onto, dtype=float)
    if isinstance(onto, (int, float)):
        onto = {"num": onto}
    if not isinstance(onto, dict):
        raise ValueError("onto must be a list of values or {start, stop, num | step}")
    lo = float(onto.get("start", np.nanmin(x)))
    hi = float(onto.get("stop", np.nanmax(x)))
    if onto.get("step"):
        s = float(onto["step"])
        n = int(np.floor((hi - lo) / s + 1e-9)) + 1
        if n > 1_000_000:
            raise ValueError("grid too large")
        return lo + s * np.arange(n)
    n = int(onto.get("num", 50))
    if not 1 <= n <= 1_000_000:
        raise ValueError("num must be 1..1e6")
    return np.linspace(lo, hi, n)


AGG_FUNCS = {"mean", "std", "sem", "min", "max", "count", "sum", "median", "first", "last", "var", "nunique"}


def _aggspec(df, agg, by: List[str]) -> Dict[str, Tuple[str, Any]]:
    """{out_name: (col, func)} with flat names like cp_mean."""
    if agg is None:
        agg = {"*": "mean"}
    if not isinstance(agg, dict):
        raise ValueError("agg must be {col: func | [funcs]} (funcs: mean std sem min max count sum median first last "
                         "var nunique qNN)")
    out = {}
    for col, fns in agg.items():
        if col == "*":
            cols = [c for c in df.select_dtypes("number").columns if c not in by]
        else:
            cols = [find_col(df, col)]
        for c in cols:
            for fn in ([fns] if isinstance(fns, str) else fns):
                f = str(fn).lower()
                m = re.fullmatch(r"(?:q|p)(\d{1,2}(?:\.\d+)?)", f)
                if m:
                    q = float(m.group(1)) / 100.0
                    func: Any = (lambda s, q=q: s.quantile(q))
                elif f in AGG_FUNCS:
                    func = f
                else:
                    raise ValueError(f"unknown aggregate {fn!r}; use mean std sem min max count sum median first last "
                                     "var nunique or q25/q75 (quantiles)")
                out[f"{c}_{f}"] = (c, func)
    if not out:
        raise ValueError("nothing to aggregate (no numeric columns besides the group columns)")
    return out


# -- steps ------------------------------------------------------------------
@step("derive", "derive {new, expr} | {exprs: {new: expr, ...}}  e.g. {new: 'T_C', expr: '`T [K]` - 273.15'}",
      ["new", "expr", "exprs"])
def _derive(df, st, env):
    pairs = list((st.get("exprs") or {}).items())
    if "new" in st or "expr" in st:
        if "new" not in st or "expr" not in st:
            raise ValueError("give both 'new' (column name) and 'expr'")
        pairs.insert(0, (st["new"], st["expr"]))
    if not pairs:
        raise ValueError("give new + expr, or exprs {new: expr}")
    df = df.copy()
    for new, expr in pairs:
        v = evaluate(df, expr)
        if isinstance(v, pd.Series):
            v = v.to_numpy()
        df[str(new)] = v
    return df


@step("filter", "filter {expr: boolean expression} | {where: {col: cond}}  e.g. \"T > 800 and phase == 'FCC_A1'\"",
      ["expr", "where"])
def _filter(df, st, env):
    if st.get("where"):
        df = apply_where(df, st["where"])
    if st.get("expr") is not None:
        m = evaluate(df, st["expr"])
        if isinstance(m, pd.Series):
            if m.dtype != bool and not pd.api.types.is_bool_dtype(m):
                raise ValueError(f"expression {st['expr']!r} is not a condition (got {m.dtype} values)")
            df = df[m.fillna(False).astype(bool).to_numpy()]
        elif not bool(m):
            df = df.iloc[0:0]
    return df


@step("select", "select {cols: [..]} (keeps and orders columns)", ["cols"])
def _select(df, st, env):
    return df[_cols(df, st.get("cols"))]


@step("drop", "drop {cols: [..]}", ["cols"])
def _drop(df, st, env):
    return df.drop(columns=_cols(df, st.get("cols")))


@step("rename", "rename {map: {old: new}}", ["map"])
def _rename(df, st, env):
    m = st.get("map")
    if not isinstance(m, dict):
        raise ValueError("map must be {old_name: new_name}")
    return df.rename(columns={find_col(df, k): str(v) for k, v in m.items()})


@step("sort", "sort {by: col | [cols], ascending: true | [bools]}", ["by", "ascending"])
def _sort(df, st, env):
    return df.sort_values(_cols(df, st.get("by"), "by"), ascending=st.get("ascending", True), kind="stable")


@step("dropna", "dropna {cols?: [..] (default all), how: any|all}", ["cols", "how"])
def _dropna(df, st, env):
    return df.dropna(subset=_opt_cols(df, st.get("cols")) or None, how=st.get("how", "any"))


@step("dedupe", "dedupe {cols?: [..], keep: first|last}", ["cols", "keep"])
def _dedupe(df, st, env):
    return df.drop_duplicates(subset=_opt_cols(df, st.get("cols")) or None, keep=st.get("keep", "first"))


@step("head", "head {n: 5, by?: [cols] (per group)}", ["n", "by"])
def _head(df, st, env):
    by = _opt_cols(df, st.get("by"))
    n = int(st.get("n", 5))
    return df.groupby(by, sort=False).head(n) if by else df.head(n)


@step("tail", "tail {n: 5, by?}", ["n", "by"])
def _tail(df, st, env):
    by = _opt_cols(df, st.get("by"))
    n = int(st.get("n", 5))
    return df.groupby(by, sort=False).tail(n) if by else df.tail(n)


@step("groupby", "groupby {by: [cols], agg: {col: mean|std|sem|min|max|count|sum|median|first|last|var|nunique|q25 "
      "| [several]} ('*' = all numeric; default mean)} -> flat names cp_mean, cp_std + n (group size)",
      ["by", "agg"])
def _groupby(df, st, env):
    by = _cols(df, st.get("by"), "by")
    spec = _aggspec(df, st.get("agg"), by)
    g = df.groupby(by, sort=True, dropna=False)
    out = g.agg(**spec)
    out["n"] = g.size()
    return out.reset_index()


@step("pivot", "pivot {index: x col, columns: col whose values become columns, values: col | [cols], agg: mean, "
      "prefix?} long->wide (z grid: rows = index values, one column per 'columns' value)",
      ["index", "columns", "values", "agg", "prefix"])
def _pivot(df, st, env):
    idx = _cols(df, st.get("index"), "index")
    col = find_col(df, st["columns"]) if st.get("columns") else None
    if col is None:
        raise ValueError("missing 'columns'")
    vals = _cols(df, st.get("values"), "values")
    agg = st.get("agg", "mean")
    t = df.pivot_table(index=idx, columns=col, values=vals, aggfunc=agg, dropna=False, sort=True)
    prefix = st.get("prefix", "")
    multi = len(vals) > 1
    t.columns = [(f"{v}_" if multi else prefix) + _fmt(c) for v, c in t.columns]
    return t.reset_index()


@step("melt", "melt {id_vars: [..], value_vars?: [..] (default all others), var_name: variable, value_name: value} "
      "wide->long", ["id_vars", "value_vars", "var_name", "value_name"])
def _melt(df, st, env):
    ids = _opt_cols(df, st.get("id_vars"))
    vv = _opt_cols(df, st.get("value_vars")) or None
    return df.melt(id_vars=ids, value_vars=vv, var_name=st.get("var_name", "variable"),
                   value_name=st.get("value_name", "value"))


def _other(env, spec) -> pd.DataFrame:
    return load_source(spec, env["resolve"], _depth=env["depth"] + 1)


@step("join", "join {with: path | inline | {data, sheet, where, transform}, on: col | [cols] | left_on/right_on, "
      "how: inner|left|right|outer|nearest (nearest: tolerance, direction, by), suffixes: ['', '_2']}",
      ["with", "on", "left_on", "right_on", "how", "suffixes", "tolerance", "direction", "by"])
def _join(df, st, env):
    if "with" not in st:
        raise ValueError("missing 'with' (the other table)")
    other = _other(env, st["with"])
    if st.get("on") is not None:
        lk = _cols(df, st["on"], "on")
        rk = [find_col(other, c) for c in ([st["on"]] if isinstance(st["on"], str) else st["on"])]
    else:
        lk = _cols(df, st.get("left_on"), "left_on (or on)")
        rk = [find_col(other, c) for c in ([st["right_on"]] if isinstance(st.get("right_on"), str)
                                            else st.get("right_on") or [])]
    if len(lk) != len(rk):
        raise ValueError("left_on and right_on need the same number of columns")
    other = other.rename(columns={r: l for l, r in zip(lk, rk) if r != l and l not in other.columns})
    rk = [l if l in other.columns else r for l, r in zip(lk, rk)]
    suffixes = tuple(st.get("suffixes", ["", "_2"]))
    how = st.get("how", "inner")
    if how == "nearest":
        if len(lk) != 1:
            raise ValueError("how=nearest joins on one numeric column (use by: [...] for exact-match keys)")
        by = _opt_cols(df, st.get("by")) or None
        left = df.assign(**{lk[0]: pd.to_numeric(df[lk[0]]).astype(float)}).sort_values(lk[0], kind="stable")
        right = other.assign(**{rk[0]: pd.to_numeric(other[rk[0]]).astype(float)}).sort_values(rk[0], kind="stable")
        tol = st.get("tolerance")
        keys = {"on": lk[0]} if lk[0] == rk[0] else {"left_on": lk[0], "right_on": rk[0]}
        return pd.merge_asof(left, right, by=by, suffixes=suffixes, direction=st.get("direction", "nearest"),
                             tolerance=None if tol is None else float(tol), **keys)
    if how not in ("inner", "left", "right", "outer"):
        raise ValueError("how must be inner, left, right, outer or nearest")
    out = df.merge(other, left_on=lk, right_on=rk, how=how, suffixes=suffixes)
    return out.drop(columns=[r for l, r in zip(lk, rk) if r != l and r in out.columns])


@step("concat", "concat {with: path | glob | [paths/specs], label_col?: name recording the source, labels?: [..] "
      "(first = this table)}", ["with", "label_col", "labels"])
def _concat(df, st, env):
    items = st.get("with")
    if items is None:
        raise ValueError("missing 'with'")
    if not isinstance(items, list) or (items and all(isinstance(i, dict) and not _is_spec(i) for i in items)):
        items = [items]                          # one path / spec / inline table
    srcs, names = [], []
    for it in items:
        if isinstance(it, str) and any(ch in it for ch in "*?["):
            hits = sorted(glob.glob(env["resolve"](it)))
            if not hits:
                raise ValueError(f"no files match {it!r}")
            for h in hits:
                srcs.append(h)
                names.append(Path(h).stem)
        else:
            srcs.append(it)
            names.append(Path(it).stem if isinstance(it, str) else
                         Path(str(it.get("data", it.get("csv", it.get("path", ""))))).stem
                         if _is_spec(it) else f"data{len(names) + 2}")
    frames = [df] + [_other(env, s) for s in srcs]
    labels = st.get("labels") or [env["src"]] + names
    if len(labels) != len(frames):
        raise ValueError(f"labels needs {len(frames)} entries (this table first)")
    if st.get("label_col"):
        frames = [f.assign(**{st["label_col"]: lab}) for f, lab in zip(frames, labels)]
    return pd.concat(frames, ignore_index=True, sort=False)


_INTERP = ("linear", "cubic", "pchip", "akima", "nearest")


def _interp1(x, y, xn, method, extrapolate):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.full(len(xn), np.nan)
    ux, inv = np.unique(x, return_inverse=True)          # average duplicate x values
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    if len(ux) == 1:
        return np.where(np.isclose(xn, ux[0]), uy[0], np.nan)
    if method == "linear" and not extrapolate:
        out = np.interp(xn, ux, uy)
    elif method == "linear":
        from scipy import interpolate as si
        out = si.interp1d(ux, uy, bounds_error=False, fill_value="extrapolate")(xn)
    else:
        from scipy import interpolate as si
        if method == "cubic" and len(ux) >= 4:
            f = si.CubicSpline(ux, uy, extrapolate=True)
        elif method == "pchip" or (method == "cubic"):
            f = si.PchipInterpolator(ux, uy, extrapolate=True)
        elif method == "akima":
            f = si.Akima1DInterpolator(ux, uy, extrapolate=True) if len(ux) >= 3 else \
                si.PchipInterpolator(ux, uy, extrapolate=True)
        else:
            f = si.interp1d(ux, uy, kind="nearest", bounds_error=False, fill_value="extrapolate")
        out = np.asarray(f(xn), dtype=float)
    if not extrapolate:
        out = np.where((xn < ux[0]) | (xn > ux[-1]), np.nan, out)
    return out


@step("interpolate", "interpolate {x, y: col | [cols], onto: [values] | {start, stop, num | step} (default: data "
      "range, num 50), method: linear|cubic|pchip|akima|nearest, by?: [group cols], extrapolate: false} -> table of "
      "by + x + y on the new grid (NaN outside the data)", ["x", "y", "onto", "method", "by", "extrapolate"])
def _interpolate(df, st, env):
    x = find_col(df, st["x"]) if st.get("x") else None
    if x is None:
        raise ValueError("missing 'x'")
    ys = _cols(df, st.get("y"), "y")
    by = _opt_cols(df, st.get("by"))
    method = str(st.get("method", "linear")).lower()
    if method not in _INTERP:
        raise ValueError(f"method must be one of {_INTERP}")
    parts = []
    for key, g in _groups(df, by):
        xv = _num(g, x)
        xn = _grid(st.get("onto", {"num": 50}), xv)
        part = {b: [k] * len(xn) for b, k in zip(by, key)}
        part[x] = xn
        for y in ys:
            part[y] = _interp1(xv, _num(g, y), xn, method, bool(st.get("extrapolate", False)))
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)


def _out_names(df, cols, st, default_suffix=""):
    if st.get("new") is not None:
        new = [st["new"]] if isinstance(st["new"], str) else list(st["new"])
        if len(new) != len(cols):
            raise ValueError("new needs one name per column")
        return new
    return [c + st.get("suffix", default_suffix) for c in cols]


@step("smooth", "smooth {cols: col | [..], method: rolling|savgol|ewm, window: 5 (points), order: 2 (savgol), by?, "
      "x? (sort by it first), new? | suffix? (default: overwrite)}",
      ["cols", "col", "method", "window", "order", "by", "x", "new", "suffix"])
def _smooth(df, st, env):
    cols = _cols(df, st.get("cols", st.get("col")))
    by = _opt_cols(df, st.get("by"))
    method = str(st.get("method", "rolling")).lower()
    w = int(st.get("window", 5))
    if w < 1:
        raise ValueError("window must be >= 1")
    if st.get("x"):
        df = df.sort_values(by + [find_col(df, st["x"])], kind="stable")
    df = df.reset_index(drop=True)
    outs = _out_names(df, cols, st)
    res = {o: np.full(len(df), np.nan) for o in outs}
    for _, g in _groups(df, by):
        pos = df.index.get_indexer(g.index)
        for c, o in zip(cols, outs):
            y = _num(g, c)
            if method == "rolling":
                v = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
            elif method == "ewm":
                v = pd.Series(y).ewm(span=w).mean().to_numpy()
            elif method == "savgol":
                from scipy.signal import savgol_filter
                order = int(st.get("order", 2))
                ww = min(w, len(y) if len(y) % 2 else len(y) - 1)
                ww = ww if ww % 2 else ww - 1
                if ww <= order:
                    v = y.copy()
                else:
                    v = savgol_filter(y, ww, order, mode="interp")
            else:
                raise ValueError("method must be rolling, savgol or ewm")
            res[o][pos] = v
    return df.assign(**res)


@step("derivative", "derivative {x, y, new?: 'dy_dx', by?} (np.gradient, sorted by x, 2nd-order accurate)",
      ["x", "y", "new", "by"])
def _derivative(df, st, env):
    x, y = find_col(df, st["x"]), find_col(df, st["y"])
    by = _opt_cols(df, st.get("by"))
    new = st.get("new") or f"d{y}_d{x}"
    df = df.sort_values(by + [x], kind="stable").reset_index(drop=True)
    out = np.full(len(df), np.nan)
    for _, g in _groups(df, by):
        pos = df.index.get_indexer(g.index)
        xv, yv = _num(g, x), _num(g, y)
        if len(xv) < 2:
            continue
        if np.any(np.diff(xv) == 0):
            raise ValueError(f"duplicate {x} values; dedupe or groupby first")
        with np.errstate(all="ignore"):
            out[pos] = np.gradient(yv, xv, edge_order=2 if len(xv) >= 3 else 1)
    return df.assign(**{new: out})


@step("integrate", "integrate {x, y, new?: 'int_y', cumulative: true (running trapezoid, 0 at the first x) | false "
      "(one total per group), by?}", ["x", "y", "new", "cumulative", "by"])
def _integrate(df, st, env):
    from scipy.integrate import cumulative_trapezoid
    x, y = find_col(df, st["x"]), find_col(df, st["y"])
    by = _opt_cols(df, st.get("by"))
    new = st.get("new") or f"int_{y}"
    df = df.sort_values(by + [x], kind="stable").reset_index(drop=True)
    if not st.get("cumulative", True):
        rows = []
        for key, g in _groups(df, by):
            rows.append({**dict(zip(by, key)), new: float(np.trapezoid(_num(g, y), _num(g, x)))})
        return pd.DataFrame(rows)
    out = np.full(len(df), np.nan)
    for _, g in _groups(df, by):
        pos = df.index.get_indexer(g.index)
        out[pos] = cumulative_trapezoid(_num(g, y), _num(g, x), initial=0.0)
    return df.assign(**{new: out})


@step("normalize", "normalize {cols, method: max (y/max|y|) | minmax | zscore | first (y/y[0]) | sum, by?, "
      "new? | suffix?}", ["cols", "col", "method", "by", "new", "suffix"])
def _normalize(df, st, env):
    cols = _cols(df, st.get("cols", st.get("col")))
    by = _opt_cols(df, st.get("by"))
    method = str(st.get("method", "max")).lower()
    outs = _out_names(df, cols, st)
    df = df.reset_index(drop=True)
    res = {o: np.full(len(df), np.nan) for o in outs}
    for _, g in _groups(df, by):
        pos = df.index.get_indexer(g.index)
        for c, o in zip(cols, outs):
            y = _num(g, c)
            with np.errstate(all="ignore"):
                if method == "max":
                    v = y / np.nanmax(np.abs(y))
                elif method == "minmax":
                    v = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
                elif method == "zscore":
                    v = (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
                elif method == "first":
                    v = y / y[0]
                elif method == "sum":
                    v = y / np.nansum(y)
                else:
                    raise ValueError("method must be max, minmax, zscore, first or sum")
            res[o][pos] = v
    return df.assign(**res)


@step("bin", "bin {col, bins: n | [edges], agg: {col: func | [..]} (default: mean of numeric cols), label: center|"
      "interval|left|right, by?} -> one row per non-empty bin: col (label) + agg columns + n",
      ["col", "bins", "agg", "label", "by"])
def _bin(df, st, env):
    c = find_col(df, st["col"]) if st.get("col") else None
    if c is None:
        raise ValueError("missing 'col'")
    by = _opt_cols(df, st.get("by"))
    xv = _num(df, c)
    bins = st.get("bins", 10)
    if isinstance(bins, (int, float)):
        if int(bins) < 1:
            raise ValueError("bins must be >= 1")
        edges = np.linspace(np.nanmin(xv), np.nanmax(xv), int(bins) + 1)
    else:
        edges = np.asarray(sorted(float(v) for v in bins))
        if len(edges) < 2:
            raise ValueError("bins needs at least 2 edges")
    code = pd.cut(xv, bins=edges, labels=False, include_lowest=True)
    label = st.get("label", "center")
    if label not in ("center", "interval", "left", "right"):
        raise ValueError("label must be center, interval, left or right")
    spec = _aggspec(df, st.get("agg"), by + [c] if st.get("agg") is None else by)
    key = "__bin__"
    g = df.assign(**{key: code}).dropna(subset=[key]).groupby(by + [key], sort=True)
    out = g.agg(**spec)
    out["n"] = g.size()
    out = out.reset_index()
    i = out[key].astype(int).to_numpy()
    lo, hi = edges[i], edges[i + 1]
    out[key] = {"center": (lo + hi) / 2, "left": lo, "right": hi}.get(label) if label != "interval" else \
        [f"{'[' if j == 0 else '('}{_fmt(a)}, {_fmt(b)}]" for a, b, j in zip(lo, hi, i)]
    return out.rename(columns={key: c})[[*by, c] + [k for k in out.columns if k not in (*by, key)]]


# -- unit conversion ---------------------------------------------------------
def _unit_table() -> Dict[str, Tuple[str, float]]:
    t: Dict[str, Tuple[str, float]] = {}
    energy = {"j": 1.0, "kj": 1e3, "mj": 1e6, "cal": 4.184, "kcal": 4184.0}
    for u, f in energy.items():
        t[u] = ("energy", f)
        t[u + "/mol"] = ("molar energy", f)
        t[u + "/mol/k"] = ("molar entropy", f)
        t[u + "/g"] = ("specific energy", f)
        t[u + "/kg"] = ("specific energy", f / 1000.0)
        t[u + "/g/k"] = ("specific heat", f)
        t[u + "/kg/k"] = ("specific heat", f / 1000.0)
    t["ev"] = ("energy", 1.602176634e-19)
    t["ev/atom"] = ("molar energy", 96485.33212)
    t["mev/atom"] = ("molar energy", 96.48533212)
    t["ev/atom/k"] = ("molar entropy", 96485.33212)
    for u, f in {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "gpa": 1e9, "bar": 1e5, "mbar": 100.0, "kbar": 1e8,
                 "atm": 101325.0, "torr": 101325.0 / 760, "mmhg": 133.322387415, "psi": 6894.757293168}.items():
        t[u] = ("pressure", f)
    for u, f in {"fraction": 1.0, "frac": 1.0, "1": 1.0, "percent": 0.01, "%": 0.01, "ppm": 1e-6,
                 "ppb": 1e-9}.items():
        t[u] = ("fraction", f)
    for u, f in {"s": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0, "day": 86400.0, "d": 86400.0,
                 "ms": 1e-3}.items():
        t[u] = ("time", f)
    for u, f in {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "angstrom": 1e-10,
                 "å": 1e-10, "a": 1e-10, "pm": 1e-12}.items():
        t[u] = ("length", f)
    return t


UNITS = _unit_table()
TEMP = {"k": "K", "kelvin": "K", "c": "C", "celsius": "C", "f": "F", "fahrenheit": "F"}
COMP = {"at%": ("mole", 0.01), "at.%": ("mole", 0.01), "mol%": ("mole", 0.01), "atomic%": ("mole", 0.01),
        "at_percent": ("mole", 0.01), "mole_fraction": ("mole", 1.0), "mol_fraction": ("mole", 1.0),
        "at_fraction": ("mole", 1.0), "x": ("mole", 1.0), "mole": ("mole", 1.0),
        "wt%": ("mass", 0.01), "wt.%": ("mass", 0.01), "mass%": ("mass", 0.01), "weight%": ("mass", 0.01),
        "wt_percent": ("mass", 0.01), "wt_fraction": ("mass", 1.0), "mass_fraction": ("mass", 1.0),
        "weight_fraction": ("mass", 1.0), "w": ("mass", 1.0), "wt": ("mass", 1.0)}


def _norm_unit(u) -> str:
    s = str(u).strip().lower().replace("°", "").replace("deg", "").replace("μ", "µ")
    s = re.sub(r"[\s.·*]+(mol|atom|k|g|kg)(\^|\*\*)?-1\b", r"/\1", s)     # J mol-1 K-1 -> j/mol/k
    if re.search(r"/.*(mol|atom|g)", s):                 # J/(mol K), J/mol-K, J/mol·K -> j/mol/k
        s = re.sub(r"[()]", "", s)
        s = re.sub(r"[·*\-\s]+", "/", s.strip())
        s = re.sub(r"/+", "/", s)
    return s.replace(" ", "")


def _to_kelvin(v, u):
    return v if u == "K" else (v + 273.15 if u == "C" else (v + 459.67) * 5.0 / 9.0)


def _from_kelvin(v, u):
    return v if u == "K" else (v - 273.15 if u == "C" else v * 9.0 / 5.0 - 459.67)


_COMP_WORDS = {"wt", "at", "mol", "mole", "mass", "weight", "atomic", "fraction", "frac", "x", "w", "pct", "percent"}


def _element(name: str) -> Optional[str]:
    """Element symbol from a column name like 'Cr', 'wt% Cr', 'x(CR)', 'Cr_at'."""
    toks = re.sub(r"[_()\[\]%.:,;/\-]", " ", str(name)).split()
    if len(toks) > 1:
        toks = [t for t in toks if t.lower() not in _COMP_WORDS]
    if len(toks) == 1 and toks[0].upper() in ATOMIC_MASS:
        return toks[0].upper()
    return None


def _mass(el: str) -> float:
    m = ATOMIC_MASS.get(str(el).upper())
    if m is None:
        raise ValueError(f"unknown element {el!r} (no atomic mass)")
    return m


@step("convert", "convert {col | cols, from, to, new? | suffix?} units: K C F; J kJ cal kcal eV (+ /mol, /mol/K, "
      "eV/atom); Pa kPa MPa GPa bar atm torr psi; fraction percent ppm; s min h; m mm um nm angstrom. Composition "
      "at%|mole_fraction <-> wt%|wt_fraction: binary {col, element, other} or multi {cols, elements? (default from "
      "names like 'Cr' or 'wt% Cr'), balance: FE, balance_col?: name to also write the balance element}",
      ["col", "cols", "from", "to", "new", "suffix", "element", "other", "elements", "balance", "balance_col"])
def _convert(df, st, env):
    cols = _cols(df, st.get("cols", st.get("col")), "col")
    if "from" not in st or "to" not in st:
        raise ValueError("give from and to units")
    fu, tu = _norm_unit(st["from"]), _norm_unit(st["to"])
    outs = _out_names(df, cols, st)
    df = df.copy()
    if fu in TEMP or tu in TEMP:
        if not (fu in TEMP and tu in TEMP):
            raise ValueError(f"cannot convert {st['from']} to {st['to']} (temperature units: K, C, F)")
        for c, o in zip(cols, outs):
            df[o] = _from_kelvin(_to_kelvin(_num(df, c), TEMP[fu]), TEMP[tu])
        return df
    if fu in COMP or tu in COMP:
        fb, fs = COMP.get(fu) or (None, UNITS.get(fu, ("", None))[1])
        tb_, ts = COMP.get(tu) or (None, UNITS.get(tu, ("", None))[1])
        if fs is None or ts is None:
            raise ValueError(f"unknown composition unit {st['from'] if fs is None else st['to']!r}; use at%, "
                             "mole_fraction, wt%, wt_fraction (or percent/fraction)")
        fb, tb_ = fb or tb_, tb_ or fb
        vals = [_num(df, c) * fs for c in cols]            # fractions in the source basis
        if fb != tb_:
            if st.get("element"):
                els = [st["element"]] if isinstance(st["element"], str) else list(st["element"])
            else:
                els = st.get("elements") or [_element(c) for c in cols]
            if len(els) != len(cols) or any(e is None for e in els):
                raise ValueError(f"cannot tell the element of {cols}; give element (binary) or elements: [...]")
            bal = st.get("balance") or st.get("other")
            if not bal:
                raise ValueError("give the balance element: other (binary) or balance, e.g. 'FE'")
            m = np.array([_mass(e) for e in els])
            mb = _mass(bal)
            v = np.vstack(vals)                                  # (n_el, n_rows)
            vb = 1.0 - v.sum(axis=0)
            if np.any(vb < -1e-9):
                raise ValueError("fractions of the listed elements sum above 1 (check the from unit)")
            if fb == "mass":                                     # w -> x
                n, nb = v / m[:, None], vb / mb
            else:                                                # x -> w
                n, nb = v * m[:, None], vb * mb
            tot = n.sum(axis=0) + nb
            vals = list(n / tot)
            if st.get("balance_col"):
                df[st["balance_col"]] = nb / tot / ts
        for o, v in zip(outs, vals):
            df[o] = v / ts
        return df
    a, b = UNITS.get(fu), UNITS.get(tu)
    if a is None or b is None:
        bad = st["from"] if a is None else st["to"]
        raise ValueError(f"unknown unit {bad!r} (units are listed in the syntax below; energies may carry /mol, "
                         "/mol/K, /g, /kg)")
    if a[0] != b[0]:
        raise ValueError(f"cannot convert {st['from']} ({a[0]}) to {st['to']} ({b[0]})")
    for c, o in zip(cols, outs):
        df[o] = _num(df, c) * (a[1] / b[1])
    return df


@step("element_fractions", "element_fractions {cols: {column: EL} (what each column is, e.g. {\"x\": \"FE\"} for "
      "x = at% Fe), elements: [all elements] (one not in cols = balance), unit: mole_fraction|at%|wt%|wt_fraction} -> "
      "X(EL) mole-fraction columns for every element (orientation-free; for composition_points / ZPF)",
      ["cols", "elements", "unit"])
def _element_fractions(df, st, env):
    spec = st.get("cols")
    if not isinstance(spec, dict) or not spec:
        raise ValueError('give cols: {column: element}, e.g. {"x": "FE"}')
    given = {}
    for c, e in spec.items():
        el = str(e).strip().upper()
        if el not in ATOMIC_MASS:
            raise ValueError(f"{e!r} is not an element")
        given[el] = find_col(df, c)
    els = [str(e).strip().upper() for e in (st.get("elements") or list(given))]
    if not set(given) <= set(els):
        raise ValueError(f"cols name {sorted(set(given) - set(els))}, which are not in elements {els}")
    missing = [e for e in els if e not in given]
    if len(missing) > 1:
        raise ValueError(f"all elements but one (the balance) need a column; missing {missing}")
    u = _norm_unit(st.get("unit", "mole_fraction"))
    if u not in COMP:
        raise ValueError(f"unit {st.get('unit')!r}: use mole_fraction, at%, wt% or wt_fraction")
    basis, f = COMP[u]
    fr = {e: _num(df, c) * f for e, c in given.items()}
    if missing:
        fr[missing[0]] = 1.0 - sum(fr.values())
    if basis == "mass":
        n = {e: fr[e] / _mass(e) for e in els}
        tot = sum(n.values())
        fr = {e: n[e] / tot for e in els}
    df = df.copy()
    for e in els:
        df[f"X({e})"] = fr[e]
    return df


TRANSFORM_HELP = ('"transform": [steps] applied in order after "where"; each {"step": NAME, ...}. Columns: exact or '
                  "unambiguous prefix. " + EXPR_HELP + "\nSteps:\n" +
                  "\n".join(f"- {s[1]}" for s in STEPS.values()))


# ---------------------------------------------------------------------------
# figure loader
# ---------------------------------------------------------------------------
def data_loader(op: Dict, resolve) -> Optional[pd.DataFrame]:
    src = op.get("csv") if op.get("csv") is not None else op.get("data")
    if src is None:
        return None
    return load_source(src, resolve, sheet=op.get("sheet"), variables=op.get("variables"), read=op.get("read"),
                       where=op.get("where"), transform=op.get("transform"))


figures.add_data_loader(data_loader)
figures.OP_HELP["data"] = DATA_HELP
figures.OP_HELP["transform"] = TRANSFORM_HELP


# ---------------------------------------------------------------------------
# summaries + agent tools
# ---------------------------------------------------------------------------
def _g(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{v:.5g}"
    return str(v)


def file_notes(path: str) -> List[str]:
    """Format-specific facts (sheet names, netCDF dims/variables)."""
    ext = Path(path).suffix.lower()
    out = []
    try:
        if ext in EXCEL_EXT:
            out.append(f"sheets: {pd.ExcelFile(path).sheet_names}")
        elif ext in NC_EXT:
            import xarray as xr
            with xr.open_dataset(path) as ds:
                out.append("netCDF dims: " + ", ".join(f"{k}={v}" for k, v in ds.sizes.items()))
                for name, v in list(ds.variables.items())[:40]:
                    attrs = ", ".join(f"{k}={v.attrs[k]}" for k in ("units", "long_name") if k in v.attrs)
                    out.append(f"  {name}({', '.join(v.dims)}) {v.dtype}" + (f" [{attrs}]" if attrs else ""))
    except Exception as e:  # noqa: BLE001 - notes are best effort
        out.append(f"(could not read format details: {e})")
    return out


def summarize(df: pd.DataFrame, rows: int = 5, max_cols: int = 60) -> str:
    lines = [f"shape: {len(df)} rows x {df.shape[1]} columns"]
    cols = list(df.columns)
    shown = cols[:max_cols]
    lines.append("columns (name: dtype, missing):")
    for c in shown:
        miss = int(df[c].isna().sum())
        lines.append(f"  {c}: {df[c].dtype}" + (f", {miss} missing" if miss else ""))
    if len(cols) > max_cols:
        lines.append(f"  ... {len(cols) - max_cols} more: {cols[max_cols:max_cols + 100]}")
    num = [c for c in shown if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if num and len(df):
        d = df[num].describe().T[["count", "mean", "std", "min", "50%", "max"]]
        lines.append("numeric stats:")
        lines.append(d.to_string(float_format=lambda v: f"{v:.5g}"))
    text = [c for c in shown if c not in num]
    if text and len(df):
        lines.append("text/other columns:")
        for c in text:
            s = df[c]
            nun = s.nunique(dropna=True)
            if nun <= 25:
                vc = s.value_counts(dropna=True)
                lines.append(f"  {c} ({nun} unique): " + ", ".join(f"{_g(k)} ({n})" for k, n in vc.items()))
            else:
                lines.append(f"  {c} ({nun} unique), e.g. {[_g(v) for v in s.dropna().unique()[:5]]}")
            conv = pd.to_numeric(s, errors="coerce")
            frac = conv.notna().mean()
            if 0.8 <= frac < 1.0 and s.notna().all():
                bad = s[conv.isna()].head(3).tolist()
                lines.append(f"    mostly numeric; non-numeric entries {bad} (a units row? use read: "
                             "{skiprows: [1]}, or derive with num())")
    if rows and len(df):
        lines.append(f"first {min(rows, len(df))} rows:")
        lines.append(df.head(rows).to_string(max_cols=30, max_colwidth=40, float_format=lambda v: f"{v:.6g}"))
    return "\n".join(lines)


WRITERS = {".csv", ".tsv", ".txt", ".xlsx", ".json"}


def write_table(df: pd.DataFrame, path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in WRITERS:
        path = str(Path(path).with_suffix(".csv")) if not ext else path
        ext = Path(path).suffix.lower()
        if ext not in WRITERS:
            raise ValueError(f"output format {ext} not supported; use .csv, .tsv, .xlsx or .json")
    if ext in (".csv", ".txt"):
        df.to_csv(path, index=False)
    elif ext == ".tsv":
        df.to_csv(path, index=False, sep="\t")
    elif ext == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_json(path, orient="records", indent=1)
    return path


def register(tb) -> None:
    STEPS_SCHEMA = {"type": "array", "items": {"type": "object"},
                    "description": "pipeline steps (syntax: plot_ops_help(['transform']))"}
    WHERE = {"type": "object", "description": "{col: value | [values] | {'>': v, 'between': [a, b], ...}}"}

    @tb.add("inspect_data", "Summarize a data file (csv/tsv/txt, xlsx, json, nc) before plotting it: shape, columns + "
            "dtypes, numeric stats, text-column values, first rows; optionally after where/transform. Works on files "
            "too big for read_document.",
            {"path": {"type": "string"}, "sheet": {"description": "Excel sheet name or index"},
             "variables": {"type": "array", "items": {"type": "string"}, "description": "netCDF variables"},
             "where": WHERE, "transform": STEPS_SCHEMA, "rows": {"type": "integer", "description": "head rows (5)"}},
            ["path"])
    def inspect_data(path: str, sheet=None, variables=None, where=None, transform=None, rows: int = 5):
        p = tb.p(path)
        df = load_source(path, tb.p, sheet=sheet, variables=variables, where=where, transform=transform)
        head = [f"{tb.rel(p)}"] + file_notes(p)
        if where or transform:
            head.append("(after where/transform)")
        return "\n".join(head) + "\n" + summarize(df, rows=max(0, min(int(rows), 50))), []

    @tb.add("transform_data", "Load a data file or inline table, apply where + transform steps (group means, pivot, "
            "join, interpolate, unit/composition conversion ...), write the result (.csv/.tsv/.xlsx/.json) and "
            "preview it.",
            {"source": {"description": "path, or inline records [{col: v}] / columns {col: [..]}"},
             "transform": STEPS_SCHEMA, "where": WHERE, "sheet": {"description": "Excel sheet"},
             "output": {"type": "string", "description": "file name; extension sets the format (default .csv)"},
             "rows": {"type": "integer", "description": "preview rows (10)"}},
            ["source"])
    def transform_data(source, transform=None, where=None, sheet=None, output: Optional[str] = None,
                       rows: int = 10):
        df = load_source(source, tb.p, sheet=sheet, where=where, transform=transform or [])
        stem = Path(source).stem if isinstance(source, str) and "\n" not in source else "table"
        out = tb.outp(output, f"{stem}_transformed.csv")
        out = write_table(df, out)
        rows = max(0, min(int(rows), 100))
        prev = df.head(rows).to_string(max_cols=30, max_colwidth=40, float_format=lambda v: f"{v:.6g}")
        return (f"wrote {tb.rel(out)}: {len(df)} rows x {df.shape[1]} columns {list(df.columns)[:80]}\n" + prev), []
