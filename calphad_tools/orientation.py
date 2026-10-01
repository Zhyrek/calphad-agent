"""Which element goes on which side of a phase diagram, and compositions -> plotted coordinates.

Orientation options (map_phase_diagram, plot_diagram_with_data, plot_all_binaries / plot_all_ternaries):

* binary ``right``: the element whose fraction is the x axis - it is on the right, the other element on the left;
* ternary ``corners``: ``{"left": EL, "right": EL, "top": EL}`` (bottom-left, bottom-right, top);
* overviews ``element_order``: earlier elements go left / bottom-left in every subsystem (+ per-system overrides).

Everything resolves to an element ORDER, the convention used by the diagram code, the recipes stored on the axes
(``ax._calphad_diagram``) and the region labels: ``[left, right]`` for a binary (x = fraction of the second
element), ``[bottom-left, bottom-right, top]`` for a ternary (x = second, y = third element; pycalphad's
triangular projection), ``[.., x-axis element, ..]`` for an isopleth.

Data follow the orientation automatically: :func:`recipe_xy` places a composition (all elements, by name) on the
diagram through its recipe, and the ``composition_points`` figure operation (edit_figure / make_plot) uses it, so
points are never converted to plot coordinates by hand.  :func:`zpf_equilibria` reads ESPEI ZPF datasets (which store
only the independent components) as complete compositions.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KELVIN = 273.15
CORNER_KEYS = {"left": 0, "bottom_left": 0, "bottomleft": 0, "right": 1, "bottom_right": 1, "bottomright": 1,
               "top": 2}
CORNER_NAMES = ("bottom-left", "bottom-right", "top")


def _el(s) -> str:
    s = str(s).strip().upper()
    m = re.fullmatch(r"[XW]\(([A-Z]{1,2})\)", s)
    return m.group(1) if m else s


def _cap(e: str) -> str:
    return str(e).capitalize()


# ---------------------------------------------------------------------------------------------------------
# orientation -> element order
# ---------------------------------------------------------------------------------------------------------
def binary_order(elements: Sequence[str], right: Optional[str] = None, left: Optional[str] = None) -> List[str]:
    """[left, right] of a binary; without right/left the given order is kept (x axis = second element)."""
    els = [_el(e) for e in elements]
    if len(els) != 2:
        raise ValueError(f"a binary diagram has 2 elements, got {els}")
    if right is None and left is None:
        return els
    r = _el(right) if right is not None else None
    lft = _el(left) if left is not None else None
    for side, e in (("right", r), ("left", lft)):
        if e is not None and e not in els:
            raise ValueError(f"{side}={e!r} is not one of the diagram's elements {els}")
    if r is not None and lft is not None and r == lft:
        raise ValueError(f"left and right are both {r}")
    if r is None:
        r = next(e for e in els if e != lft)
    return [e for e in els if e != r] + [r]


def ternary_order(elements: Sequence[str], corners: Optional[Dict[str, str]] = None,
                  right: Optional[str] = None) -> List[str]:
    """[bottom-left, bottom-right, top] of a ternary isotherm.  ``corners`` may name 1-3 corners; the others are
    filled with the remaining elements in the given order."""
    els = [_el(e) for e in elements]
    if len(els) != 3:
        raise ValueError(f"a ternary diagram has 3 elements, got {els}")
    if not corners and right is None:
        return els
    if corners is not None and not isinstance(corners, dict):
        raise ValueError('corners must be an object like {"left": "NI", "right": "FE", "top": "CR"}')
    pos: Dict[int, str] = {}
    for k, val in (corners or {}).items():
        key = CORNER_KEYS.get(re.sub(r"[\s\-]+", "_", str(k).strip().lower()))
        if key is None:
            raise ValueError(f"corners key {k!r}: use left (bottom-left), right (bottom-right) and top")
        if key in pos:
            raise ValueError(f"corner {CORNER_NAMES[key]} given twice in {corners}")
        pos[key] = _el(val)
    if right is not None:
        if 1 in pos and pos[1] != _el(right):
            raise ValueError(f"right={right!r} contradicts corners {corners}")
        pos[1] = _el(right)
    bad = [e for e in pos.values() if e not in els]
    if bad:
        raise ValueError(f"corners {bad} are not elements of this diagram {els}; use exactly {els}")
    if len(set(pos.values())) != len(pos):
        raise ValueError(f"each element can be at one corner only: {corners}")
    rest = [e for e in els if e not in pos.values()]
    return [pos[i] if i in pos else rest.pop(0) for i in range(3)]


def preferred_order(elements: Sequence[str], element_order: Optional[Sequence[str]] = None) -> List[str]:
    """Subsystem order for overviews: elements of ``element_order`` first (in that order), the others
    alphabetically after them.  No preference = alphabetical (the historical default)."""
    els = sorted(_el(e) for e in elements)
    if not element_order:
        return els
    rank = {}
    for i, e in enumerate(_el(x) for x in element_order):
        rank.setdefault(e, i)
    return sorted(els, key=lambda e: (rank.get(e, len(rank)), e))


def _system_key(name) -> frozenset:
    parts = name if isinstance(name, (list, tuple)) else re.split(r"[-–\s,/]+", str(name).strip())
    return frozenset(_el(p) for p in parts if str(p).strip())


def subsystem_order(elements: Sequence[str], element_order: Optional[Sequence[str]] = None,
                    overrides: Optional[Dict] = None) -> List[str]:
    """Orientation of one overview subsystem: a per-system override (key "CR-FE" / "Fe-Cr-Ni" in any order; value
    [left, right] / [bottom-left, bottom-right, top], an element for the right side, or {"right": ..} /
    {"left": .., "right": .., "top": ..}), else ``element_order``, else alphabetical."""
    base = preferred_order(elements, element_order)
    ov = None
    for k, val in (overrides or {}).items():
        if _system_key(k) == frozenset(base):
            ov = val
            break
    if ov is None:
        return base
    if isinstance(ov, (list, tuple)):
        o = [_el(e) for e in ov]
        if sorted(o) != sorted(base):
            raise ValueError(f"orientation for {'-'.join(base)}: {list(ov)} must list exactly {base}")
        return o
    if isinstance(ov, str):
        return binary_order(base, ov) if len(base) == 2 else ternary_order(base, right=ov)
    if isinstance(ov, dict):
        if len(base) == 2:
            return binary_order(base, ov.get("right"), ov.get("left"))
        return ternary_order(base, ov)
    raise ValueError(f"orientation for {'-'.join(base)}: give a list, an element or a corners object")


def check_overrides(overrides: Optional[Dict], systems: Sequence[Sequence[str]]) -> None:
    """Every per-system override names one of the computed subsystems."""
    keys = {frozenset(s) for s in systems}
    bad = [k for k in (overrides or {}) if _system_key(k) not in keys]
    if bad:
        raise ValueError(f"orientation overrides {bad} match no subsystem; systems: "
                         f"{sorted('-'.join(sorted(s)) for s in keys)}")


# ---------------------------------------------------------------------------------------------------------
# describing the orientation
# ---------------------------------------------------------------------------------------------------------
def orientation_info(kind: str, els: Sequence[str], axis_el: Optional[str] = None,
                     balance: Optional[str] = None) -> Dict:
    if kind == "binary":
        return {"left": els[0], "right": els[1], "x_axis": f"X({els[1]})"}
    if kind == "ternary":
        return {"left": els[0], "right": els[1], "top": els[2], "x_axis": f"X({els[1]})", "y_axis": f"X({els[2]})"}
    return {"x_axis": f"X({axis_el})", "balance": balance}


def orientation_text(kind: str, els: Sequence[str], axis_label: Optional[str] = None,
                     axis_el: Optional[str] = None, balance: Optional[str] = None) -> str:
    """One line stating the orientation actually used."""
    if kind == "binary":
        what = axis_label or f"X({_cap(els[1])})"
        return f"orientation: x axis = {what}; {_cap(els[0])} on the left, {_cap(els[1])} on the right"
    if kind == "ternary":
        return (f"orientation: corners {_cap(els[0])} bottom-left, {_cap(els[1])} bottom-right, {_cap(els[2])} top "
                f"(x axis = X({_cap(els[1])}), y axis = X({_cap(els[2])}))")
    return (f"orientation: x axis = {axis_label or 'X(' + _cap(axis_el) + ')'} increasing to the right"
            + (f"; balance {_cap(balance)}" if balance else ""))


def corner_labels(ax, els: Sequence[str], fontsize: float = 11) -> None:
    """Element names at the three corners of a triangular (pycalphad) axes: bottom-left, bottom-right, top."""
    for t in list(ax.texts):
        if str(t.get_gid() or "").startswith("corner_label:"):
            t.remove()
    kw = {"ha": "center", "fontsize": fontsize, "fontweight": "bold"}
    ax.text(0.0, -0.08, _cap(els[0]), transform=ax.transAxes, va="top", gid=f"corner_label:{els[0]}", **kw)
    ax.text(1.0, -0.08, _cap(els[1]), transform=ax.transAxes, va="top", gid=f"corner_label:{els[1]}", **kw)
    ax.annotate(_cap(els[2]), xy=(0.0, 1.0), xycoords="data", xytext=(0, 7), textcoords="offset points",
                va="bottom", gid=f"corner_label:{els[2]}", annotation_clip=False, **kw)


def corner_label_code(els: Sequence[str], ax: str = "ax") -> str:
    """The same corner labels as standalone code."""
    a, b, c = (_cap(e) for e in els)
    kw = "ha='center', fontsize=11, fontweight='bold'"
    return (f"# corners: {a} bottom-left, {b} bottom-right (x axis), {c} top (y axis)\n"
            f"{ax}.text(0.0, -0.08, {a!r}, transform={ax}.transAxes, va='top', {kw})\n"
            f"{ax}.text(1.0, -0.08, {b!r}, transform={ax}.transAxes, va='top', {kw})\n"
            f"{ax}.annotate({c!r}, xy=(0.0, 1.0), xytext=(0, 7), textcoords='offset points', va='bottom', "
            f"annotation_clip=False, {kw})\n")


# ---------------------------------------------------------------------------------------------------------
# compositions -> plotted coordinates
# ---------------------------------------------------------------------------------------------------------
def diagram_recipe(ax) -> Dict:
    """The recipe (orientation, units) stored on a diagram axes, or a clear error."""
    d = getattr(ax, "_calphad_diagram", None) or {}
    rec = d.get("recipe") or d.get("orientation")
    if not rec:
        raise ValueError("this axes has no phase-diagram recipe, so its orientation is unknown: composition "
                         "points work on diagrams made by map_phase_diagram (binary, ternary isotherm, isopleth) or "
                         "plot_diagram_with_data; on other plots give x/y coordinates (scatter)")
    return rec


def masses_for(recipe: Dict, els: Sequence[str]) -> Dict[str, float]:
    """Atomic masses: the recipe's (weight diagrams), else the database's ELEMENT lines, else a table."""
    m = dict((recipe.get("comp") or {}).get("masses") or {})
    need = [e for e in els if not m.get(e)]
    if need and recipe.get("tdb"):
        try:
            from .calc import get_db
            db = get_db(recipe["tdb"])
            for e in need:
                v = float(db.refstates.get(e, {}).get("mass", 0.0) or 0.0)
                if v > 0:
                    m[e] = v
        except Exception:  # noqa: BLE001 - fall back to the table
            pass
    from .expdata import ATOMIC_MASS
    for e in els:
        if not m.get(e):
            if e not in ATOMIC_MASS:
                raise ValueError(f"no atomic mass for {e}")
            m[e] = float(ATOMIC_MASS[e])
    return {e: float(m[e]) for e in els}


def parse_unit(unit: Optional[str]) -> Tuple[bool, float]:
    """(is_weight, scale) of a composition unit: mole_fraction, at%, mole_percent, wt%, weight_fraction ..."""
    u = str(unit or "mole_fraction").strip().lower().replace(" ", "").replace(".", "")
    table = {"mole_fraction": (False, 1.0), "molefraction": (False, 1.0), "x": (False, 1.0), "fraction": (False, 1.0),
             "at_fraction": (False, 1.0), "atomic_fraction": (False, 1.0), "mol_fraction": (False, 1.0),
             "at%": (False, 100.0), "mol%": (False, 100.0), "mole_percent": (False, 100.0),
             "at_percent": (False, 100.0), "atomic_percent": (False, 100.0), "percent": (False, 100.0),
             "wt%": (True, 100.0), "mass%": (True, 100.0), "weight_percent": (True, 100.0),
             "wt_percent": (True, 100.0), "mass_percent": (True, 100.0), "wt_fraction": (True, 1.0),
             "weight_fraction": (True, 1.0), "mass_fraction": (True, 1.0), "w": (True, 1.0)}
    if u not in table:
        raise ValueError(f"unit {unit!r}: use mole_fraction, at%, wt% or weight_fraction")
    return table[u]


def complete_fractions(given: Dict[str, np.ndarray], elements: Sequence[str], weight: bool = False,
                       scale: float = 1.0, masses: Optional[Dict[str, float]] = None
                       ) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
    """Mole fractions of every element from the given ones (one element may be missing: the balance).
    Returns (fractions, balance element or None)."""
    els = [_el(e) for e in elements]
    bad = [e for e in given if e not in els]
    if bad:
        raise ValueError(f"elements {bad} are not in this diagram ({'-'.join(els)})")
    fr = {e: np.asarray(v, dtype=float) / scale for e, v in given.items()}
    missing = [e for e in els if e not in fr]
    balance = None
    if len(missing) > 1:
        raise ValueError(f"give the composition of all elements but one (the balance); missing {missing}")
    if missing:
        balance = missing[0]
        fr[balance] = 1.0 - sum(fr.values())
        if np.any(fr[balance] < -1e-6):
            raise ValueError(f"the given fractions sum above 1 (balance {balance} negative); check unit (at%/wt% "
                             f"need unit 'at%'/'wt%')")
    else:
        tot = sum(fr.values())
        if np.any(np.abs(tot - 1.0) > 0.02):
            raise ValueError(f"the fractions of all elements sum to {np.round(tot, 4).tolist()}, not 1; leave out "
                             "the balance element or check the unit")
        fr = {e: v / tot for e, v in fr.items()}
    if weight:
        m = masses or {}
        n = {e: fr[e] / m[e] for e in els}
        tot = sum(n.values())
        fr = {e: n[e] / tot for e in els}
    return {e: fr[e] for e in els}, balance


def recipe_xy(recipe: Dict, X: Dict[str, np.ndarray], T_K=None) -> Tuple[np.ndarray, np.ndarray]:
    """Data coordinates of compositions X (mole fractions of every element, arrays) at T (K) on a diagram."""
    kind = recipe["kind"]
    els = recipe["elements"]
    c = recipe.get("comp") or {}
    s = float(c.get("scale", 1.0))
    fr = {e: np.asarray(X[e], dtype=float) for e in els}
    if c.get("weight"):
        m = {e: float(c["masses"][e]) for e in els}
        tot = sum(fr[e] * m[e] for e in els)
        fr = {e: fr[e] * m[e] / tot for e in els}
    if kind == "ternary":
        return fr[els[1]] * s, fr[els[2]] * s
    x_el = els[1] if kind == "binary" else recipe["axis_element"]
    if T_K is None:
        raise ValueError(f"points on a {kind} T-x diagram need a temperature (T)")
    a, b = recipe.get("tmap", (1.0, 0.0))
    return fr[x_el] * s, (np.asarray(T_K, dtype=float) - b) / a


def recipe_axes_text(recipe: Dict) -> str:
    """Orientation line of a stored recipe (for notes)."""
    kind, els = recipe["kind"], recipe["elements"]
    if kind == "isopleth":
        return orientation_text(kind, els, axis_el=recipe["axis_element"], balance=recipe.get("balance"))
    return orientation_text(kind, els)


# ---------------------------------------------------------------------------------------------------------
# ESPEI ZPF datasets -> complete compositions
# ---------------------------------------------------------------------------------------------------------
def zpf_equilibria(datasets, elements: Sequence[str], T: Optional[float] = None,
                   T_tolerance: Optional[float] = None) -> List[Dict]:
    """ZPF data of exactly the system ``elements``: [{"T", "P", "reference", "phases": [(phase, {el: X} | None)]}].
    Datasets store only the independent components: the missing (balance) element is computed.  A phase whose
    composition is unknown (broken tie-line) has None.  With T: only points within T_tolerance (K)."""
    els = [_el(e) for e in elements]
    want = set(els)
    rows = datasets.all() if hasattr(datasets, "all") else list(datasets)
    out = []
    for d in rows:
        if d.get("output") != "ZPF":
            continue
        if {c for c in d.get("components", []) if c != "VA"} != want:
            continue
        vals = d.get("values", [])
        Ts = np.atleast_1d(np.asarray(d["conditions"]["T"], dtype=float))
        Ts = np.full(len(vals), Ts[0]) if Ts.size == 1 else Ts
        Ps = np.atleast_1d(np.asarray(d["conditions"].get("P", 101325.0), dtype=float))
        Ps = np.full(len(vals), Ps[0]) if Ps.size == 1 else Ps
        for row, t, p in zip(vals, Ts, Ps):
            if T is not None and abs(float(t) - float(T)) > float(T_tolerance if T_tolerance is not None else 1e-6):
                continue
            phases = []
            for ph, cs, xs in row:
                if ph == "__HYPERPLANE__":
                    continue
                if any(x is None for x in xs):
                    phases.append((ph, None))
                    continue
                X = {_el(cc): float(x) for cc, x in zip(cs, xs)}
                missing = [e for e in els if e not in X]
                if len(missing) == 1:
                    X[missing[0]] = 1.0 - sum(X.values())
                elif missing:
                    phases.append((ph, None))
                    continue
                phases.append((ph, {e: X[e] for e in els}))
            out.append({"T": float(t), "P": float(p), "reference": d.get("reference", ""), "phases": phases})
    return out


def plot_zpf(ax, recipe: Dict, eqs: List[Dict], colors: Dict[str, str], marker_map: Dict,
             ms: float = 5.0) -> Dict[str, int]:
    """Draw ZPF data on a diagram through its recipe: markers per phase (colour) and reference (marker), two-phase
    tie-lines as black segments, tie-triangles as dashed red triangles; a broken tie-line (one composition unknown) is a single
    point.  Returns counts."""
    n = {"tielines": 0, "single": 0, "triangles": 0, "points": 0}
    for eq in eqs:
        pts = []
        for ph, X in eq["phases"]:
            if X is None:
                continue
            x, y = recipe_xy(recipe, {e: np.asarray([X[e]]) for e in recipe["elements"]}, [eq["T"]])
            pts.append((ph, float(x[0]), float(y[0])))
        sym = marker_map.get(eq["reference"], {}).get("markers", {"marker": "o", "fillstyle": "full"})
        for ph, x, y in pts:
            ax.plot(x, y, marker=sym["marker"], fillstyle=sym["fillstyle"], linestyle="", color=colors.get(ph, "k"),
                    markersize=ms, markeredgewidth=0.6, zorder=4)
            n["points"] += 1
        if len(pts) == 2:
            ax.plot([pts[0][1], pts[1][1]], [pts[0][2], pts[1][2]], color="k", linewidth=0.8, zorder=3.5)
            n["tielines"] += 1
        elif len(pts) == 3:
            loop = pts + pts[:1]
            ax.plot([p[1] for p in loop], [p[2] for p in loop], color="r", linewidth=0.8, linestyle="--",
                    zorder=3.5)
            n["triangles"] += 1
        elif len(pts) == 1:
            n["single"] += 1
    return n


# ---------------------------------------------------------------------------------------------------------
# figure operation: composition_points
# ---------------------------------------------------------------------------------------------------------
from .figures import _SCATTER_STYLE, OpContext, find_col, register_op  # noqa: E402

POINTS_HELP = ('composition_points {points: [{"X(FE)": 0.3, "X(NI)": 0.2, "T": 1273}, ..] | data/csv with element '
               "columns (X(FE), Fe, wt% Cr ...; or element_cols {EL: col}) + T_col; unit: mole_fraction|at%|wt%|"
               "wt_fraction; T_unit: K|C (default: the diagram's); the balance element is computed; tielines: "
               "[[i, j], ..] (point index pairs) | tieline_col (rows sharing a value are joined); T_tolerance "
               "(ternary: skip points off the section T); scatter style (color, marker, s, label ..), tieline_color, "
               "tieline_width}: points placed BY COMPOSITION through the diagram's orientation (figures of "
               "map_phase_diagram / plot_diagram_with_data), never by hand-converted x/y")
_T_KEYS = {"T": None, "T_K": "K", "T_C": "C"}


def points_input(op: Dict, frame, recipe: Dict) -> Dict:
    """What a composition_points op gives: {"given": {EL: array (as given)}, "cols": {EL: column}, "T": array|None,
    "T_col", "T_unit", "unit", "groups": [[row indices], ..]} (shared by the op and its show_code translation)."""
    els = [_el(e) for e in recipe["elements"]]
    T_unit = op.get("T_unit")
    unit = op.get("unit")
    out: Dict = {"cols": {}, "T_col": None}
    if frame is not None:
        if op.get("element_cols"):
            cols = {_el(k): find_col(frame, v) for k, v in op["element_cols"].items()}
        else:
            from .figdata import _element
            cols = {}
            for c in frame.columns:
                e = _element(c)
                if e in els and e not in cols:
                    cols[e] = c
            if not cols:
                raise ValueError(f"no element columns in the data (columns {list(frame.columns)}); name them like "
                                 f"X(FE) / Fe / wt% Fe or give element_cols {{EL: column}}")
            if unit is None and all(str(c).strip().upper().startswith("W(") for c in cols.values()):
                unit = "weight_fraction"
        given = {e: frame[c].to_numpy(dtype=float) for e, c in cols.items()}
        n = len(frame)
        tcol = op.get("T_col")
        if tcol:
            tcol = find_col(frame, tcol)
        else:
            tcol = next((c for c in ("T_K", "T_C", "T") if c in frame.columns), None)
            if tcol in ("T_K", "T_C") and T_unit is None:
                T_unit = tcol[-1]
        T = frame[tcol].to_numpy(dtype=float) if tcol else None
        out.update(cols=cols, T_col=tcol)
    else:
        pts = op.get("points")
        if isinstance(pts, dict):
            pts = [pts]
        if not pts or not isinstance(pts, list) or not all(isinstance(p, dict) for p in pts):
            raise ValueError('give points [{"X(FE)": 0.3, "X(NI)": 0.2, "T": 1273}, ...] or a data/csv source')
        n = len(pts)
        keys = []
        for p in pts:
            for k in p:
                if k not in keys:
                    keys.append(k)
        t_keys = [k for k in keys if str(k).strip().upper() in ("T", "T_K", "T_C")]
        comp_keys = [k for k in keys if k not in t_keys and str(k).lower() not in ("label", "phase")]
        given = {}
        for k in comp_keys:
            e = _el(re.sub(r"^\s*(wt|at|mol|mass|weight)\s*%?\s*", "", str(k), flags=re.I))
            if e not in els:
                raise ValueError(f"point key {k!r}: not an element of this diagram ({'-'.join(els)}); use X(EL), "
                                 f"W(EL) or EL, and T")
            vals = [p.get(k) for p in pts]
            if any(v is None for v in vals):
                raise ValueError(f"every point needs {k}: give the same elements in all points (leave out the same "
                                 f"balance element everywhere)")
            given[e] = np.asarray(vals, dtype=float)
            if unit is None and str(k).strip().upper().startswith("W("):
                unit = "weight_fraction"
        T = None
        if t_keys:                        # temperatures of inline points are kept in K
            T = np.full(n, np.nan)
            default = (T_unit or recipe.get("T_unit") or "K").upper()[0]
            for i, p in enumerate(pts):
                hit = [k for k in t_keys if p.get(k) is not None]
                if len(hit) > 1:
                    raise ValueError(f"point {i}: give one temperature (T, T_K or T_C), not {hit}")
                if hit:
                    u = _T_KEYS[str(hit[0]).strip().upper()] or default
                    T[i] = float(p[hit[0]]) + (KELVIN if u == "C" else 0.0)
            T_unit = "K"
    groups = []
    if op.get("tielines") is not None:
        for g in op["tielines"]:
            idx = [int(i) for i in g]
            if len(idx) < 2 or any(not 0 <= i < n for i in idx):
                raise ValueError(f"tielines entry {g}: give 2 (or 3) point indices 0..{n - 1}")
            groups.append(idx)
    elif op.get("tieline_col"):
        if frame is None:
            raise ValueError("tieline_col needs a data/csv source; with inline points use tielines [[i, j], ..]")
        col = find_col(frame, op["tieline_col"])
        for _, g in frame.reset_index(drop=True).groupby(col, sort=False):
            if len(g) >= 2:
                groups.append([int(i) for i in g.index])
    out.update(given=given, T=T, T_unit=(T_unit or recipe.get("T_unit") or "K").upper()[0],
               unit=unit or "mole_fraction", groups=groups, n=n)
    return out


def place_points(recipe: Dict, inp: Dict, T_tolerance: Optional[float] = None) -> Dict:
    """Plot coordinates of composition_points input: {"x", "y", "keep", "X", "T_K", "balance", "notes"}."""
    els = recipe["elements"]
    weight, scale = parse_unit(inp["unit"])
    masses = masses_for(recipe, els) if weight else None
    X, balance = complete_fractions(inp["given"], els, weight, scale, masses)
    T_K = None
    if inp["T"] is not None:
        T_K = inp["T"] + (KELVIN if inp["T_unit"] == "C" else 0.0)
    notes = []
    keep = np.ones(inp["n"], dtype=bool)
    kind = recipe["kind"]
    if kind != "ternary" and (T_K is None or np.isnan(T_K).any()):
        raise ValueError(f"points on a {kind} T-x diagram need a temperature (T, T_K or T_C) for every point")
    if kind == "ternary" and T_K is not None:
        off = np.abs(T_K - float(recipe["T_K"]))          # NaN (no T given) counts as on the section
        if T_tolerance is not None:
            keep = ~(off > float(T_tolerance))
            if (~keep).any():
                notes.append(f"{int((~keep).sum())} point(s) more than {T_tolerance:g} K from the section "
                             f"temperature skipped")
        elif (off > 10).any():
            notes.append(f"{int((off > 10).sum())} point(s) are more than 10 K off the section temperature "
                         f"({recipe['T_K']:.2f} K); T_tolerance drops them")
    if kind == "isopleth":
        dev = [e for e, v in (recipe.get("fixed") or {}).items() if np.any(np.abs(X[e] - float(v)) > 0.01)]
        if dev:
            notes.append(f"some points are off the section (fixed {', '.join(f'X({e})' for e in dev)} differ by "
                         f"more than 0.01)")
    x, y = recipe_xy(recipe, X, T_K)
    return {"x": np.asarray(x, float), "y": np.asarray(y, float), "keep": keep, "X": X, "T_K": T_K,
            "balance": balance, "notes": notes, "masses": masses}


def point_style(op: Dict) -> Dict:
    kw = {k: v for k, v in op.items() if k in _SCATTER_STYLE}
    kw.setdefault("color", "k")
    kw.setdefault("marker", "o")
    kw.setdefault("s", 30)
    kw.setdefault("zorder", 4)
    return kw


def tieline_segments(groups, x, y, keep) -> List:
    segs = []
    for g in groups:
        idx = [i for i in g if keep[i]]
        if len(idx) < 2:
            continue
        ring = idx + [idx[0]] if len(idx) >= 3 else idx
        segs += [[(float(x[a]), float(y[a])), (float(x[b]), float(y[b]))] for a, b in zip(ring, ring[1:])]
    return segs


@register_op("composition_points", POINTS_HELP, group="draw")
def _op_composition_points(ctx: OpContext):
    from matplotlib.collections import LineCollection
    from .figures import _pick
    ax, op = ctx.ax, ctx.op
    rec = diagram_recipe(ax)
    inp = points_input(op, ctx.frame() if ctx.has_source else None, rec)
    res = place_points(rec, inp, op.get("T_tolerance"))
    keep = res["keep"]
    kw = _pick(point_style(op), _SCATTER_STYLE)
    ax.scatter(res["x"][keep], res["y"][keep], **kw)
    segs = tieline_segments(inp["groups"], res["x"], res["y"], keep)
    if segs:
        ax.add_collection(LineCollection(segs, colors=[op.get("tieline_color", kw.get("color", "k"))],
                                         linewidths=float(op.get("tieline_width", 0.9)),
                                         linestyles=op.get("tieline_style", "solid"), zorder=kw["zorder"] - 0.5))
    msg = f"{int(keep.sum())} point(s) placed by composition ({recipe_axes_text(rec)})"
    if res["balance"]:
        msg += f"; balance {_cap(res['balance'])} computed"
    if segs:
        msg += f"; {len(segs)} tie-line segment(s)"
    ctx.note("; ".join([msg] + res["notes"]))
