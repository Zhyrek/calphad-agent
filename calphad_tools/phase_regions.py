"""Phase regions of a binary T-x diagram, a ternary isotherm or an isopleth, in the coordinates of the plot.

Needs pycalphad and numpy only.  A *recipe* describes the diagram::

    {"kind": "binary" | "ternary" | "isopleth", "elements": [balance, x_el(, y_el)], "phases": [...], "P": 101325,
     "T_K": 1173.15 (ternary), "tmap": [a, b] (binary / isopleth: T in K = a * y + b),
     "comp": {"scale": 1 | 100, "weight": False, "masses": {el: g/mol}}}   # plotted composition / scale

An isopleth's recipe also has "axis_element" (x axis), "balance" and "fixed" ({element: mole fraction}); its x is
the fraction (scale, weight) of the axis element of the whole alloy.  Its regions are built by isopleth_regions.

Region identity is a *key*: the sorted phase names with composition-set markers removed, e.g.
('BCC_A2', 'BCC_A2') for a miscibility gap.  An ordered phase whose ordered sublattices are equivalent is named
after its disordered phase (B2_BCC -> BCC_A2), as in the phase-boundary data.

* :func:`regions_from_lines` - exact two-phase polygons (the two phase-boundary traces of each ZPF line),
  the boundary points of every phase (``phase_points``),
  tie-triangles, line compounds and binary invariant lines from mapped ZPF data;
* :func:`map_lines` - that ZPF data from a pycalphad mapping (checked and repaired: mapping_check);
  :func:`strategy_lines` - the same from a finished mapping strategy;
* :func:`equilibrium_grid` - stable phase set on a coarse grid (finds regions the mapping did not trace);
* :func:`key_at` - phase set at one plotted point (used to confirm every label position);
* :func:`lowest_key` - which of two phase sets is stable at a point, by Gibbs energy (where a full equilibrium's
  global minimisation misses one).
"""
from __future__ import annotations

import itertools
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KELVIN = 273.15


# ---------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------
def base_name(name: str) -> str:
    """Phase name without a composition-set marker (#2)."""
    return re.sub(r"#\d+$", "", str(name).strip().upper())


def key_of(names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(base_name(n) for n in names))


def ordered_info(db, comps, phases, models=None) -> Dict[str, tuple]:
    """Ordered phase -> (disordered name, number of ordered sublattices, {(sublattice, species): dof index})."""
    out = {}
    for name in phases:
        ph = db.phases.get(name)
        if ph is None or ph.model_hints.get("ordered_phase") != name:
            continue
        dis = ph.model_hints.get("disordered_phase")
        if not dis or dis not in db.phases:
            continue
        n_sub = len(db.phases[name].sublattices) - (len(db.phases[dis].sublattices) - 1)
        if n_sub < 2:
            continue
        m = re.fullmatch(r"([A-Z]\d+)_([A-Z]+)", dis)       # A1_FCC -> FCC_A1 if that phase also exists
        if m and f"{m.group(2)}_{m.group(1)}" in db.phases:
            dis = f"{m.group(2)}_{m.group(1)}"
        mod = (models or {}).get(name)
        if mod is None:
            from pycalphad import Model
            mod = Model(db, comps, name)
        idx = {(int(y.sublattice_index), str(y.species.name)): j for j, y in enumerate(mod.site_fractions)}
        out[name] = (dis, n_sub, idx)
    return out


def state_name(phase: str, y, info: Dict[str, tuple], tol: float = 1e-4) -> str:
    """Name of a phase state: the disordered phase's name if the ordered sublattices are equivalent."""
    inf = info.get(phase)
    if inf is None:
        return phase
    dis, n_sub, idx = inf
    y = np.asarray(y, dtype=float)
    for sp in {sp for (s, sp) in idx if s < n_sub}:
        vals = [y[idx[(s, sp)]] if (s, sp) in idx else 0.0 for s in range(n_sub)]
        if np.ptp(vals) > tol:
            return phase
    return dis


def compset_names(compsets, info: Dict[str, tuple]) -> List[str]:
    """Names of the composition sets of a mapping point (second set of one phase: PHASE#2)."""
    out, seen = [], {}
    for cs in compsets:
        n = state_name(cs.phase_record.phase_name, np.asarray(cs.dof)[cs.phase_record.num_statevars:], info)
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


# ---------------------------------------------------------------------------------------------------------
# plotted coordinates <-> conditions
# ---------------------------------------------------------------------------------------------------------
def _to_mole(fr: Dict[str, float], masses: Dict[str, float]) -> Dict[str, float]:
    tot = sum(fr[e] / masses[e] for e in fr)
    return {e: (fr[e] / masses[e]) / tot for e in fr}


def iso_axis_fraction(recipe: Dict, x) -> np.ndarray:
    """Isopleth: mole fraction of the axis element at plotted x (mass fraction / percent converted with the other
    elements' fixed mole fractions)."""
    c = recipe.get("comp") or {}
    w = np.asarray(x, float) / float(c.get("scale", 1.0))
    if not c.get("weight"):
        return w
    m = {e: float(c["masses"][e]) for e in recipe["elements"]}
    a, b, fixed = recipe["axis_element"], recipe["balance"], recipe["fixed"]
    C = sum(float(val) * m[e] for e, val in fixed.items())
    F = sum(float(val) for val in fixed.values())
    return w * (C + (1.0 - F) * m[b]) / (m[a] * (1.0 - w) + w * m[b])


def plot_to_fractions(recipe: Dict, x, y=None) -> Dict[str, np.ndarray]:
    """Mole fractions of the axis elements (isopleth: of every element) at plotted composition(s)."""
    els = recipe["elements"]
    c = recipe.get("comp") or {}
    s = float(c.get("scale", 1.0))
    if recipe["kind"] == "isopleth":
        xa = iso_axis_fraction(recipe, x)
        fr = {e: np.full_like(xa, float(val)) for e, val in recipe["fixed"].items()}
        fr[recipe["axis_element"]] = xa
        fr[recipe["balance"]] = 1.0 - xa - sum(float(val) for val in recipe["fixed"].values())
        return fr
    if recipe["kind"] == "binary":
        fr = {els[1]: np.asarray(x, float) / s}
        fr[els[0]] = 1.0 - fr[els[1]]
    else:
        fr = {els[1]: np.asarray(x, float) / s, els[2]: np.asarray(y, float) / s}
        fr[els[0]] = 1.0 - fr[els[1]] - fr[els[2]]
    if c.get("weight"):
        fr = _to_mole(fr, {e: float(c["masses"][e]) for e in fr})
    return fr


def fractions_to_plot(recipe: Dict, X: Dict[str, float]) -> Tuple[float, float]:
    """Plotted composition(s) of a phase with mole fractions X (ternary: (x, y); binary / isopleth: x only in
    [0]; isopleth: X is the overall composition)."""
    els = recipe["elements"]
    c = recipe.get("comp") or {}
    s = float(c.get("scale", 1.0))
    fr = {e: float(X[e]) for e in els}
    if c.get("weight"):
        m = {e: float(c["masses"][e]) for e in els}
        tot = sum(fr[e] * m[e] for e in els)
        fr = {e: fr[e] * m[e] / tot for e in els}
    if recipe["kind"] == "binary":
        return fr[els[1]] * s, float("nan")
    if recipe["kind"] == "isopleth":
        return fr[recipe["axis_element"]] * s, float("nan")
    return fr[els[1]] * s, fr[els[2]] * s


def T_of_y(recipe: Dict, y):
    a, b = recipe.get("tmap", (1.0, 0.0))
    return np.asarray(y, float) * a + b


def y_of_T(recipe: Dict, T):
    a, b = recipe.get("tmap", (1.0, 0.0))
    return (np.asarray(T, float) - b) / a


def xy_of(recipe: Dict, X: Dict[str, float], T: float) -> Tuple[float, float]:
    """Plotted point of a phase composition X at temperature T (K)."""
    px, py = fractions_to_plot(recipe, X)
    if recipe["kind"] in ("binary", "isopleth"):
        return px, float(y_of_T(recipe, T))
    return px, py


# ---------------------------------------------------------------------------------------------------------
# regions from mapped ZPF data
# ---------------------------------------------------------------------------------------------------------
def composition_dof(db, phase: str, elements: Sequence[str]) -> Optional[int]:
    """Independent composition variables of a phase in a system (0: a stoichiometric phase, a point; 1: a line
    compound of a ternary, e.g. (Co,Ni)3V), from its sublattice model; None if the database does not have it."""
    ph = db.phases.get(base_name(phase)) if db is not None else None
    if ph is None:
        return None
    comps = {str(e).upper() for e in elements} | {"VA"}
    n = 0
    for sub in ph.constituents:
        k = sum(1 for sp in sub if {str(c).upper() for c in getattr(sp, "constituents", {str(sp): 1})} <= comps)
        n += max(0, k - 1)
    return n


# solution phases (ordered ones too: B2_BCC, L12_FCC) - never a compound (the same set as label_place's)
SOLUTION_TOKENS = {"LIQUID", "LIQ", "GAS", "FCC", "BCC", "HCP", "DHCP", "CBCC", "CUB", "BCT", "DIAMOND", "IONIC",
                   "SLAG", "A1", "A2", "A3", "GRAPHITE"}


def model_info(db, phase: str, elements: Sequence[str]) -> Optional[Dict]:
    """Sublattice model of a phase within a system: {"sites": [site ratios], "constituents": [[EL or VA, ...], ...]}
    (constituents that are elements of the system or VA; other species dropped); None if the database lacks it."""
    ph = db.phases.get(base_name(phase)) if db is not None else None
    if ph is None:
        return None
    els = {str(e).upper() for e in elements} | {"VA"}
    cons = []
    for sub in ph.constituents:
        cons.append(sorted({str(getattr(sp, "name", sp)).upper() for sp in sub} & els))
    return {"sites": [float(x) for x in ph.sublattices], "constituents": cons}


def ideal_points(model: Dict, elements: Sequence[str], cap: int = 20000) -> List[Tuple[tuple, np.ndarray]]:
    """End-members of a model (one constituent per sublattice) that are compounds (two or more elements): [(choice,
    mole fractions in the order of ``elements``)]."""
    els = [str(e).upper() for e in elements]
    out = []
    for k, ch in enumerate(itertools.product(*model["constituents"])):
        if k >= cap:
            break
        amt = np.zeros(len(els))
        for site, c in zip(model["sites"], ch):
            if c in els:
                amt[els.index(c)] += float(site)
        if amt.sum() > 0 and (amt > 0).sum() >= 2:
            out.append((ch, amt / amt.sum()))
    return out


def _near_stoichiometric(kind, recipe, n, pts, model) -> Optional[Dict]:
    """How close a phase's field (its boundary compositions ``pts``, mole fractions in the recipe's element order)
    is to stoichiometric: the largest deviation of any point from the nearest compound end-member of its model
    (point), or - ternary - from the nearest line between two end-members that differ on one sublattice and lie in
    different binaries (a line compound, e.g. (Co,Ni)3V; a field stretched along one binary edge has a range in
    that binary and is not one), and the field's width (largest range of one mole fraction).  None if the model has no compound end-member or the
    field is far from it (> 0.05)."""
    cap = 0.05
    ideal = ideal_points(model, recipe["elements"])
    if not ideal or not len(pts):
        return None
    X = np.asarray(pts, float)
    width = float(np.ptp(X, axis=0).max())
    out = {"key": (n,), "width": width}
    dev = [float(np.abs(X - P).max()) for _, P in ideal]
    k = int(np.argmin(dev))
    els = recipe["elements"]
    if dev[k] < cap:
        Xm = X.mean(axis=0)
        px, py = fractions_to_plot(recipe, dict(zip(els, Xm)))
        out["point"] = {"dev": dev[k], "X": {e: float(v) for e, v in zip(els, Xm)},
                        "ideal": {e: float(v) for e, v in zip(els, ideal[k][1])}, "x": float(px)}
        if kind == "ternary":
            out["point"]["y"] = float(py)
    if kind == "ternary":
        best = None
        for (c1, P1), (c2, P2) in itertools.combinations(ideal, 2):
            if sum(a != b for a, b in zip(c1, c2)) != 1 or ((P1 > 0) | (P2 > 0)).sum() < 3:
                continue                         # (a line along one binary edge is a range in that binary)
            d = P2 - P1
            t = np.clip((X - P1) @ d / max(float(d @ d), 1e-12), 0.0, 1.0)
            dv = float(np.abs(X - (P1 + t[:, None] * d)).max())
            if best is None or dv < best[0]:
                best = (dv, P1, d, t)
        if best is not None and best[0] < cap:
            dv, P1, d, t = best
            ends = [P1 + t.min() * d, P1 + t.max() * d]
            if float(np.abs(ends[1] - ends[0]).max()) > 2e-3:
                line = [list(map(float, fractions_to_plot(recipe, dict(zip(els, E))))) for E in ends]
                out["line"] = {"dev": dv, "line": line,
                               "X_ends": [{e: float(v) for e, v in zip(els, E)} for E in ends]}
    return out if ("point" in out or "line" in out) else None


def regions_from_lines(lines, invariants, recipe: Dict, db=None) -> Dict:
    """Polygons / compounds / invariant lines (plotted coordinates) from ZPF lines and three-phase nodes.

    lines: [{"phases": [n1, n2], "points": [{"T": K, "phases": [(n1, X1), (n2, X2)]}, ...]}, ...]
    invariants: [{"T_K": K, "phases": [...], "compositions": {name: {el: x}}, "type": optional}, ...]
    db: when given, a phase is a compound (named outside the plot) only if its model has no composition freedom
    in this system (a ternary line: one) - a solution phase whose boundary hardly moves is not one.  With db, also
    "models" (sublattice model of each phase: the formula of a compound label), "near_compounds" (phases whose
    field lies within 0.05 of a compound end-member of their model: label_place treats those within
    label_style.stoich_tolerance as compounds), "phase_T" (binary: temperature range of each phase's boundaries, K)
    and "axes_elements"; each compound has "X" (mole fractions; a ternary line: "X_ends")."""
    kind = recipe["kind"]
    polys, traces, phase_T = [], {}, {}
    for ln in lines:
        pts = ln["points"]
        if len(pts) < 2 or len(ln["phases"]) != 2:
            continue
        a = [xy_of(recipe, p["phases"][0][1], p["T"]) for p in pts]
        b = [xy_of(recipe, p["phases"][1][1], p["T"]) for p in pts]
        polys.append({"key": key_of(ln["phases"]), "xy": [list(map(float, q)) for q in a + b[::-1]],
                      "ties": [[list(map(float, p)), list(map(float, q))] for p, q in zip(a, b)]})
        for k, n in enumerate(ln["phases"]):
            X = np.array([[p["phases"][k][1][e] for e in recipe["elements"]] for p in pts], float)
            traces.setdefault(base_name(n), []).append((X, a if k == 0 else b))
            T_seen = phase_T.setdefault(base_name(n), [np.inf, -np.inf])
            T_seen[:] = [min(T_seen[0], min(p["T"] for p in pts)), max(T_seen[1], max(p["T"] for p in pts))]
    compounds = []
    for n, trs in traces.items():
        for X, path in trs:
            xs = np.array(path, float)
            ptp = np.ptp(X, axis=0)
            if kind == "ternary" and ptp.max() > 1e-3:
                # a line phase (one element fixed, e.g. M3V between Ni3V and Co3V): label along the line
                fixed = [i for i, d in enumerate(ptp) if d < 1e-4 and X[:, i].mean() > 2e-3]
                span = np.ptp(xs, axis=0)
                if not fixed or np.hypot(*span) < 0.05:
                    continue
                d = xs - xs.mean(axis=0)
                u = np.linalg.svd(d, full_matrices=False)[2][0]           # direction of the line
                t = d @ u
                ends = [xs.mean(axis=0) + t.min() * u, xs.mean(axis=0) + t.max() * u]
                def off_line(c):                 # distance of this trace from a line found before
                    a0, a1 = np.asarray(c["line"][0]), np.asarray(c["line"][1])
                    e = (a1 - a0) / (np.linalg.norm(a1 - a0) or 1.0)
                    q = np.asarray(ends[0]) - a0
                    return abs(float(e[0] * q[1] - e[1] * q[0]))
                hit = next((c for c in compounds if c["key"] == (n,) and "line" in c and off_line(c) < 2e-3), None)
                if hit:                          # the same line seen from another field: extend it
                    pts = np.array(hit["line"] + [list(e) for e in ends], float)
                    dd = pts - pts.mean(axis=0)
                    tt = dd @ u
                    hit["line"] = [list(map(float, pts[int(np.argmin(tt))])), list(map(float, pts[int(np.argmax(tt))]))]
                else:
                    compounds.append({"key": (n,), "line": [list(map(float, e)) for e in ends]})
                continue
            if ptp.max() > 1e-3:
                continue
            if kind == "binary":
                xc = float(np.mean(xs[:, 0]))
                x_frac = float(np.mean(X[:, 1]))
                if min(x_frac, 1 - x_frac) < 2e-3:
                    continue                     # pure element edge
                hit = next((c for c in compounds if c["key"] == (n,) and abs(c["x"] - xc) < 1e-6 + 1e-3 * abs(xc)),
                           None)
                y0, y1 = float(np.min(xs[:, 1])), float(np.max(xs[:, 1]))
                if hit:
                    hit["y0"], hit["y1"] = min(hit["y0"], y0), max(hit["y1"], y1)
                else:
                    compounds.append({"key": (n,), "x": xc, "y0": y0, "y1": y1,
                                      "X": {e: float(v) for e, v in zip(recipe["elements"], X.mean(axis=0))}})
            else:
                xc, yc = float(np.mean(xs[:, 0])), float(np.mean(xs[:, 1]))
                if max(X.mean(axis=0)) > 1 - 2e-3:
                    continue                     # a pure element (corner); compounds on an edge are kept
                if not any(c["key"] == (n,) and abs(c["x"] - xc) < 2e-3 and abs(c["y"] - yc) < 2e-3
                           for c in compounds):
                    compounds.append({"key": (n,), "x": xc, "y": yc,
                                      "X": {e: float(v) for e, v in zip(recipe["elements"], X.mean(axis=0))}})
    if db is not None:
        def is_compound(c):
            dof = composition_dof(db, c["key"][0], recipe["elements"])
            return dof is None or dof <= (1 if "line" in c else 0)
        compounds = [c for c in compounds if is_compound(c)]
    for c in compounds:
        if "line" in c:
            c["X_ends"] = [{e: float(np.squeeze(v)) for e, v in plot_to_fractions(recipe, *q).items()}
                           for q in c["line"]]
    tris, inv_lines = [], []
    edge_pts: Dict[str, list] = {n: [p for _, path in trs for p in path] for n, trs in traces.items()}
    for inv in invariants or []:
        names = list(inv["phases"])
        pts = [xy_of(recipe, inv["compositions"][n], inv["T_K"]) for n in names]
        for n, p in zip(names, pts):
            edge_pts.setdefault(base_name(n), []).append(p)
        if kind == "ternary":
            tris.append({"key": key_of(names), "xy": [list(map(float, p)) for p in pts], "tri": True})
        else:
            xs = [p[0] for p in pts]
            inv_lines.append({"key": key_of(names), "T_K": float(inv["T_K"]), "y": float(pts[0][1]),
                              "x0": float(min(xs)), "x1": float(max(xs)), "type": inv.get("type", "")})
    phase_points = {}                    # boundary points of each phase's own field (where it is named if its
    for n, pts in edge_pts.items():      # field is too thin to be a region of the plot)
        arr = np.asarray(pts, float).reshape(-1, 2)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr):
            idx = np.unique(np.linspace(0, len(arr) - 1, min(len(arr), 300)).astype(int))
            phase_points[n] = [list(map(float, q)) for q in arr[idx]]
    out = {"polygons": tris + polys, "compounds": compounds, "invariants": inv_lines, "phase_points": phase_points}
    if db is not None and kind in ("binary", "ternary"):
        els = recipe["elements"]
        exact = {c["key"][0] for c in compounds}
        models, near = {}, []
        for n in sorted(set(recipe.get("phases") or []) | set(traces)):
            m = model_info(db, n, els)
            if m is not None:
                models[base_name(n)] = m
        for n, trs in sorted(traces.items()):
            ph = db.phases.get(n)
            if n in exact or n not in models or (ph is not None and ph.model_hints.get("liquid")) \
                    or n.startswith(("LIQ", "IONIC_LIQ", "GAS")) or set(n.split("_")) & SOLUTION_TOKENS:
                continue                         # (solution phases, ordered ones too, keep their own names)
            Xs = [X for X, _ in trs] + [np.array([[inv["compositions"][q][e] for e in els]], float)
                                         for inv in invariants or [] for q in inv["phases"] if base_name(q) == n]
            cand = _near_stoichiometric(kind, recipe, n, np.vstack(Xs), models[n])
            if cand is None:
                continue
            if kind == "binary":
                ys = np.concatenate([np.asarray(path, float)[:, 1] for _, path in trs]
                                    + [np.asarray([q[1]], float) for q in edge_pts.get(n, [])])
                cand["y0"], cand["y1"] = float(np.nanmin(ys)), float(np.nanmax(ys))
            near.append(cand)
        out.update(models=models, near_compounds=near, axes_elements=list(els),
                   phase_T={n: [float(a), float(b)] for n, (a, b) in phase_T.items()})
    return out


def mapping_conditions(recipe: Dict, step: float = 0.02, T_step: float = 10.0) -> Dict:
    from pycalphad import variables as v
    els = recipe["elements"]
    conds = {v.P: float(recipe.get("P", 101325.0)), v.N: 1}
    if recipe["kind"] == "binary":
        conds[v.T] = tuple(float(t) for t in recipe["T_range"]) + (float(T_step),)
        xr = recipe.get("x_range") or [0.0, 1.0]
        conds[v.X(els[1])] = (float(xr[0]), float(xr[1]), float(step))
    else:
        conds[v.T] = float(recipe["T_K"])
        conds[v.X(els[1])] = (0, 1, float(step))
        conds[v.X(els[2])] = (0, 1, float(step))
    return conds


def map_lines(db, recipe: Dict, info: Dict[str, tuple], step: float = 0.02, T_step: float = 10.0,
              time_limit: float = 600.0, check: str = "auto", report: Optional[Dict] = None):
    """(lines, invariants) of a pycalphad mapping strategy for the recipe (same structures as regions_from_lines
    reads).  check: the mapping check (mapping_check.check_and_repair: "auto", "thorough" or "off"); its report
    goes into ``report`` (a dict) when given."""
    import time
    from pycalphad.mapping import BinaryStrategy, TernaryStrategy
    try:                    # a standalone script has the mapping check in the same file
        _run, _check = run_strategy, check_and_repair  # noqa: F821
    except NameError:
        from .mapping_check import check_and_repair as _check, run_strategy as _run
    els = recipe["elements"]
    comps = sorted(els) + ["VA"]
    Strat = BinaryStrategy if recipe["kind"] == "binary" else TernaryStrategy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strategy = Strat(db, comps, list(recipe["phases"]), mapping_conditions(recipe, step, T_step))
    t0 = time.time()
    run = _run(strategy, time_limit, check)
    strategy._mc_seconds = run["seconds"]
    rep = _check(strategy, db, recipe, mode=check if run["finished"] else "off", deadline=t0 + time_limit)
    if report is not None:
        report.update(rep)
    return strategy_lines(strategy, db, recipe, info)


def strategy_lines(strategy, db, recipe: Dict, info: Optional[Dict[str, tuple]] = None):
    """(lines, invariants) of a finished BinaryStrategy / TernaryStrategy (the structures regions_from_lines
    reads); zero-length lines along an edge are left out."""
    from pycalphad import variables as v
    els = recipe["elements"]
    comps = sorted(els) + ["VA"]
    info = {**ordered_info(db, comps, recipe["phases"], strategy.models), **(info or {})}
    s_els = [e for e in strategy.elements if e != "VA"]

    def comp(cs):
        return dict(zip(s_els, np.asarray(cs.X, dtype=float)))

    def temp(p):
        return float(np.squeeze(p.global_conditions[v.T]))
    lines = []
    for z in strategy.zpf_lines:
        if len(z.points) < 2:
            continue
        if all(max(np.max(np.abs(np.asarray(a.X, float) - np.asarray(b.X, float)))
                   for a, b in itertools.combinations(p.stable_composition_sets, 2)) < 1e-5
               for p in z.points if len(p.stable_composition_sets) > 1):
            continue                        # zero-length tie-lines along an edge
        names = compset_names(z.points[len(z.points) // 2].stable_composition_sets, info)
        lines.append({"phases": names, "points": [{"T": temp(p), "phases": [(n, comp(cs)) for n, cs in
                                                                          zip(names, p.stable_composition_sets)]}
                                                  for p in z.points]})
    invariants = []
    for node in strategy.node_queue.nodes:
        if len(node.stable_composition_sets) != 3:
            continue
        names = compset_names(node.stable_composition_sets, info)
        T = temp(node)
        X = {n: comp(cs) for n, cs in zip(names, node.stable_composition_sets)}
        key = sorted(tuple(round(c[e], 3) for e in els) for c in X.values())
        if any(abs(i["T_K"] - T) < 0.05 and i["_key"] == key for i in invariants):
            continue
        invariants.append({"T_K": T, "phases": names, "compositions": X, "_key": key})
    return lines, invariants


# ---------------------------------------------------------------------------------------------------------
# equilibrium classification
# ---------------------------------------------------------------------------------------------------------
def _keys_from_eq(eq, info: Dict[str, tuple], np_min: float = 1e-6, dx: float = 2e-3) -> List[Optional[tuple]]:
    ph = np.asarray(eq.Phase.values)
    nv = ph.shape[-1]
    ph = ph.reshape(-1, nv)
    npf = np.asarray(eq.NP.values, float).reshape(-1, nv)
    X = np.asarray(eq.X.values, float)
    X = X.reshape(-1, nv, X.shape[-1])
    Y = np.asarray(eq.Y.values, float)
    Y = Y.reshape(-1, nv, Y.shape[-1])
    out: List[Optional[tuple]] = []
    for i in range(ph.shape[0]):
        items = []
        for k in range(nv):
            name = str(ph[i, k])
            if not name or not np.isfinite(npf[i, k]) or npf[i, k] <= np_min:
                continue
            n = state_name(name, Y[i, k], info)
            if any(m == n and np.max(np.abs(X[i, k] - xk)) < dx for m, xk in items):
                continue                    # the same phase state found twice
            items.append((n, X[i, k]))
        out.append(tuple(sorted(n for n, _ in items)) if items else None)
    return out


def conditions_at(recipe: Dict, x, y) -> Dict:
    from pycalphad import variables as v
    els = recipe["elements"]
    fr = plot_to_fractions(recipe, x, y)
    conds = {v.P: float(recipe.get("P", 101325.0)), v.N: 1}
    if recipe["kind"] == "binary":
        conds[v.T] = float(T_of_y(recipe, y))
        conds[v.X(els[1])] = float(np.clip(fr[els[1]], 1e-6, 1 - 1e-6))
    elif recipe["kind"] == "isopleth":
        conds[v.T] = float(T_of_y(recipe, y))
        conds[v.X(recipe["axis_element"])] = float(np.clip(fr[recipe["axis_element"]], 1e-6, 1 - 1e-6))
        for e, val in recipe["fixed"].items():
            conds[v.X(e)] = float(val)
    else:
        conds[v.T] = float(recipe["T_K"])
        for e in els[1:]:
            conds[v.X(e)] = float(np.clip(fr[e], 1e-6, 1 - 1e-6))
    return conds


def key_at(db, recipe: Dict, info: Dict[str, tuple], x: float, y: float,
           pdens: Optional[int] = None) -> Optional[tuple]:
    """Stable phase set (a region key) at a plotted point, or None (outside the diagram / no result).  pdens: point
    sampling of the equilibrium (None: pycalphad's default)."""
    from pycalphad import equilibrium
    els = recipe["elements"]
    fr = plot_to_fractions(recipe, x, y)
    if any(not (-1e-9 <= float(fr[e]) <= 1 + 1e-9) for e in els) or not np.isfinite(float(T_of_y(recipe, y))):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, sorted(els) + ["VA"], list(recipe["phases"]), conditions_at(recipe, x, y),
                         **({"calc_opts": {"pdens": int(pdens)}} if pdens else {}))
    return _keys_from_eq(eq, info)[0]


def key_phases(key, phases: Sequence[str], info: Dict[str, tuple]) -> List[str]:
    """Database phases that give the states of a region key (an ordered phase also for its disordered name)."""
    out = set()
    for n in key:
        b = base_name(n)
        if b in phases:
            out.add(b)
        out |= {o for o, inf in info.items() if inf[0] == b and o in phases}
    return sorted(out)


def lowest_key(db, recipe: Dict, info: Dict[str, tuple], x: float, y: float, keys, tol: float = 0.5,
               pdens: int = 500) -> Optional[tuple]:
    """Of candidate phase sets at a plotted point, the one with the lowest Gibbs energy (an equilibrium restricted
    to each candidate's phases; a candidate whose restricted equilibrium is not that phase set is out; within tol
    J/mol the earlier candidate wins).  Settles a point where the global minimisation of a full equilibrium
    misses the stable phase set (e.g. two nearly equal σ descriptions); None if no candidate holds."""
    from pycalphad import equilibrium
    comps = sorted(recipe["elements"]) + ["VA"]
    cand = []
    for k in keys:
        k = tuple(k)
        phs = key_phases(k, list(recipe["phases"]), info)
        if not phs:
            continue
        for pd in (int(pdens), 10 * int(pdens)):     # (a second composition set can need dense sampling)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eq = equilibrium(db, comps, phs, conditions_at(recipe, x, y), calc_opts={"pdens": pd})
            gm = float(np.squeeze(eq.GM.values))
            if _keys_from_eq(eq, info)[0] == k and np.isfinite(gm):
                cand.append((gm, k))
                break
    if not cand:
        return None
    g0 = min(g for g, _ in cand)
    return next(k for g, k in cand if g <= g0 + tol)


def equilibrium_grid(db, recipe: Dict, info: Dict[str, tuple], xlim, ylim, n: Tuple[int, int] = (80, 60),
                     models=None, phase_records=None, pdens: Optional[int] = None) -> Dict:
    """Stable phase sets on a grid over the plotted area: {"xy": [N, 2] plotted points, "keys": [N] int (index
    into "table", -1 = none), "table": [key, ...]}.  models / phase_records: those of a mapping strategy of the
    same system (saves building and compiling them again; same results).  pdens: point sampling (None:
    pycalphad's default; an isopleth uses 500, which finds e.g. γ + γ' of one TDB phase)."""
    kw = {"model": models, "phase_records": phase_records} if models is not None else {}
    if pdens:
        kw["calc_opts"] = {"pdens": int(pdens)}
    from pycalphad import equilibrium, variables as v
    els = recipe["elements"]
    comps = sorted(els) + ["VA"]
    if recipe["kind"] == "binary":
        nx, ny = int(n[0]), int(n[1])
        xs = xlim[0] + (np.arange(nx) + 0.5) / nx * (xlim[1] - xlim[0])
        ys = ylim[0] + (np.arange(ny) + 0.5) / ny * (ylim[1] - ylim[0])
        Xb = np.clip(plot_to_fractions(recipe, xs)[els[1]], 1e-6, 1 - 1e-6)
        conds = {v.P: float(recipe.get("P", 101325.0)), v.N: 1, v.T: T_of_y(recipe, ys), v.X(els[1]): Xb}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eq = equilibrium(db, comps, list(recipe["phases"]), conds, **kw)
        keys = _keys_from_eq(eq, info)                       # T major, then X
        pts = np.array([[x, y] for y in ys for x in xs], float)
    elif recipe["kind"] == "isopleth":
        nx, ny = int(n[0]), int(n[1])
        xs = xlim[0] + (np.arange(nx) + 0.5) / nx * (xlim[1] - xlim[0])
        ys = ylim[0] + (np.arange(ny) + 0.5) / ny * (ylim[1] - ylim[0])
        a = recipe["axis_element"]
        xa = iso_axis_fraction(recipe, xs)
        ok_x = (xa > 1e-9) & (1.0 - xa - sum(float(val) for val in recipe["fixed"].values()) > 1e-9)
        xs = xs[ok_x]
        conds = {v.P: float(recipe.get("P", 101325.0)), v.N: 1, v.T: T_of_y(recipe, ys), v.X(a): xa[ok_x]}
        conds.update({v.X(e): float(val) for e, val in recipe["fixed"].items()})
        keys, pts = [], np.zeros((0, 2))
        if len(xs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eq = equilibrium(db, comps, list(recipe["phases"]), conds, **kw)
            dims = [str(d) for d in eq.Phase.dims if str(d) != "vertex"]
            flat = _keys_from_eq(eq, info)
            idx = np.arange(len(flat)).reshape([eq.Phase.sizes[d] for d in dims])
            idx = np.moveaxis(idx, [dims.index("T"), dims.index(f"X_{a}")], [0, 1]).reshape(len(ys), len(xs), -1)
            keys = [flat[i] for i in idx[:, :, 0].ravel()]   # T major, then X
            pts = np.array([[x, y] for y in ys for x in xs], float)
    else:
        m = int(n[0])
        g = (np.arange(m) + 0.5) / m
        conds = {v.P: float(recipe.get("P", 101325.0)), v.N: 1, v.T: float(recipe["T_K"]),
                 v.X(els[1]): g, v.X(els[2]): g}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eq = equilibrium(db, comps, list(recipe["phases"]), conds, **kw)
        order = [str(d)[2:] for d in eq.Phase.dims if str(d).startswith("X_")]   # condition dimension order
        keys = _keys_from_eq(eq, info)
        pts = []
        for a in g:
            for b in g:
                X = {order[0]: a, order[1]: b}
                X[els[0]] = 1 - a - b
                pts.append(fractions_to_plot(recipe, X) if X[els[0]] > 0 else (np.nan, np.nan))
        pts = np.array(pts, float)
    table, ids = [], []
    for k in keys:
        if k is None:
            ids.append(-1)
            continue
        if k not in table:
            table.append(k)
        ids.append(table.index(k))
    ok = np.all(np.isfinite(pts), axis=1)
    return {"xy": pts[ok], "keys": np.asarray(ids, int)[ok], "table": table}
