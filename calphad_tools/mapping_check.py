"""Check-and-repair pass after pycalphad phase-diagram mapping (binary T-x diagrams and ternary isotherms).

pycalphad's mapping strategies (BinaryStrategy / TernaryStrategy) can return ZPF lines that are wrong or
incomplete, and nothing downstream checks them.  This pass uses equilibrium calculations as the ground truth:

1. **Line validation** - equilibrium (sampling density PDENS) at the middle of tie-lines of every traced line: both
   ends and the middle (thorough: about every n/12-th tie-line), then bisection where the answer changes.  A
   tie-line is valid if equilibrium gives the same two phases in the same ordered/disordered state (an ordered
   phase whose ordered sublattices are equivalent is named after its disordered phase, as everywhere in the
   toolkit: L12_FCC disordered = FCC_A1, BCC_B2 disordered = BCC_A2), or if the traced two-phase mixture is not
   higher in Gibbs energy than the equilibrium answer (G_TOL: a node, or a field equilibrium missed).  Metastable
   runs at the ends are trimmed, interior ones split the line.  Example: a B2+FCC trace that runs past the
   B2 + γ + γ' tie-triangle (γ and γ' are one TDB phase, so pycalphad sees no phase change there).
2. **Duplicates** - a line that lies on another line of the same phases (after trimming) is dropped; the more
   complete one is kept.
3. **Nodes** - every three-phase node (pycalphad's and the check's own) must be an equilibrium: at its
   temperature, equilibrium at the middle of each pair of its phases (ternary: also the triangle's centre) may not
   be lower in Gibbs energy than its own phases mixed there; failing nodes (e.g. Ce-Co: BCC + Ce5Co19 + L at pure
   Ce's melting point) are removed, duplicates kept once.  Conversely, pycalphad drops a three-phase node (and the
   whole line that found it) when its own metastability check fails; run_strategy keeps those nodes.  A binary
   node is confirmed by point equilibria (pdens PDENS) between its phases within 15 K: where they switch from one
   two-phase pair to another, the invariant is solved and checked at T -/+ 0.5 to 2 K; a ternary tie-triangle by
   equilibrium inside it.  A confirmed node is added (invariant line / tie-triangle) and the lines that start there
   are traced; lines of its phases that end just short of it are joined to it.
4. **Missing boundaries** - an equilibrium grid (phase_regions.equilibrium_grid; the region labels reuse it) is
   compared with the two-phase fields of the traced lines; a connected area of grid cells (at least MIN_CELLS, not
   a one-row / one-column sliver) whose two-phase equilibrium no traced field covers is a missing boundary.
   Repair, in order: a line pycalphad traced but threw away (it removes the whole line when the node it ends on
   fails its metastability check; run_strategy keeps those aside), new start points inside the area, a re-map of
   the area alone with half the steps.  New lines are validated as in 1, and the coverage is checked again.

Depends on numpy, pycalphad and phase_regions only, so show_code copies it into the standalone scripts: the
tools and the generated code run the same check.  Isopleths and T-MU diagrams are not checked.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phase_regions import (base_name, compset_names, equilibrium_grid, fractions_to_plot, key_of, ordered_info,
                            plot_to_fractions, T_of_y, xy_of, y_of_T)

MODES = ("auto", "thorough", "off")
GRID_N = {"binary": (80, 60), "ternary": (40, 40)}   # the region-label grid (region_labels.GRID_N)
MIN_CELLS = 4               # a missing area must cover at least this many grid cells
COVER_TOL = 1.0             # a grid cell within this many cell spacings of a traced field is covered by it
DUP_TOL = 2.5e-3            # duplicate lines: within this distance (mole fraction / fraction of the T range)
G_TOL = 0.5                # J/mol: a traced tie-line this close to the equilibrium Gibbs energy is not metastable
PDENS = 500                 # sampling of the test equilibria (pycalphad's default 60 misses e.g. γ' next to γ)


def check_mode(mode) -> str:
    """'auto' (default), 'thorough' (denser validation) or 'off' (pycalphad's raw lines)."""
    m = "auto" if mode is None or mode is True else ("off" if mode is False else str(mode).strip().lower())
    if m not in MODES:
        raise ValueError(f"check_mapping must be one of {list(MODES)}, not {mode!r}")
    return m


# ---------------------------------------------------------------------------------------------------------
# running a strategy
# ---------------------------------------------------------------------------------------------------------
def run_strategy(strategy, time_limit: float = 600.0, mode: str = "auto") -> Dict:
    """Map with a pycalphad strategy until it is finished or time_limit (s) has passed.  With checking on, a ZPF
    line that pycalphad removes again (BinaryStrategy / TernaryStrategy._process_new_node drop the whole line
    when the node it ends on fails the metastability check) is kept in ``strategy._mc_dropped``: the repair can
    use its points (they passed the global-minimum check at every step)."""
    keep = check_mode(mode) != "off"
    if keep and not hasattr(strategy, "_mc_dropped"):
        strategy._mc_dropped = []
    if keep:
        _watch_nodes(strategy)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(strategy.node_queue.nodes) == 0 and len(strategy.zpf_lines) == 0:
            strategy.generate_automatic_starting_points()
        finished, n = False, 0
        while not finished:
            nl = len(strategy.zpf_lines)
            last = strategy.zpf_lines[-1] if nl else None
            finished = strategy.iterate()
            n += 1
            if keep and last is not None and len(strategy.zpf_lines) < nl and len(last.points) >= 2:
                strategy._mc_dropped.append(last)
            if not finished and time.time() - t0 > time_limit:
                break
    return {"finished": bool(finished), "iterations": n, "seconds": round(time.time() - t0, 1)}


def _watch_nodes(strategy) -> None:
    """Keep the three-phase nodes pycalphad finds but rejects as metastable (strategy._mc_rejected): the check
    confirms or refutes them with equilibrium calculations."""
    if getattr(strategy, "_mc_rejected", None) is not None:
        return
    strategy._mc_rejected = []
    orig = strategy._process_new_node

    def _process_new_node(zpf_line, new_node):
        n = len(strategy.zpf_lines)
        orig(zpf_line, new_node)
        if len(strategy.zpf_lines) < n and len(new_node.stable_composition_sets) == 3:
            strategy._mc_rejected.append(new_node)
    strategy._process_new_node = _process_new_node


def map_and_check(strategy, db, recipe: Dict, mode: str = "auto", time_limit: float = 600.0, grid_n=None) -> Dict:
    """Map with the strategy (run_strategy) and check / repair the lines (check_and_repair) within time_limit
    seconds in all; the check is skipped when the mapping stops at the time limit.  Returns the check report
    with the mapping run ("run": finished, iterations, seconds)."""
    t0 = time.time()
    mode = check_mode(mode)
    run = run_strategy(strategy, time_limit, mode)
    strategy._mc_seconds = run["seconds"]
    if mode != "off" and not run["finished"]:
        rep = {"mode": mode, "text": "mapping check: skipped (the mapping stopped at the time limit)",
               "complete": False}
    else:
        rep = check_and_repair(strategy, db, recipe, mode=mode, grid_n=grid_n, deadline=t0 + time_limit)
    rep["run"] = run
    return rep


# ---------------------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------------------
class _Line:
    """A two-phase ZPF line as arrays: T (K), compositions of both phases, state names per point."""

    def __init__(self, z, ctx):
        self.z = z
        pts = z.points
        self.n = len(pts)
        self.T = np.array([float(np.squeeze(p.global_conditions[ctx.vT])) for p in pts])
        self.X = np.array([[np.asarray(cs.X, float)[ctx.cs_index] for cs in p.stable_composition_sets[:2]]
                           for p in pts])                                  # (n, 2, n_els) in recipe order
        self.names = [compset_names(p.stable_composition_sets[:2], ctx.info) for p in pts]
        self.G = np.array([[float(cs.energy) for cs in p.stable_composition_sets[:2]] for p in pts])
        self.keys = [key_of(nm) for nm in self.names]
        self.key = self.keys[self.n // 2]
        self.degenerate = bool(np.all(np.abs(self.X[:, 0] - self.X[:, 1]).max(axis=1) < 1e-5))
        self.uv = ctx.norm(self.T, self.X)                                # (n, 2, 2) normalized coordinates

    def length(self) -> float:
        if self.n < 2:
            return 0.0
        return float(sum(np.hypot(*np.diff(self.uv[:, k], axis=0).T).sum() for k in (0, 1)))

    def max_step(self) -> float:
        """Largest step of either boundary between consecutive tie-lines (normalized coordinates)."""
        if self.n < 2:
            return 0.0
        return float(max(np.hypot(*np.diff(self.uv[:, k], axis=0).T).max() for k in (0, 1)))

    def sig(self):
        """End points (T, both phase compositions in either order): equal for a line traced twice exactly."""
        return tuple(sorted((round(float(self.T[i]), 3),) + tuple(sorted(tuple(np.round(self.X[i, k], 5))
                                                                         for k in (0, 1)))
                            for i in (0, self.n - 1)))


class _Ctx:
    def __init__(self, strategy, db, recipe, info):
        from pycalphad import variables as v
        self.v = v
        self.vT = v.T
        self.db, self.recipe, self.info = db, recipe, info
        self.kind = recipe["kind"]
        self.els = list(recipe["elements"])
        s_els = [str(e) for e in strategy.elements if str(e) != "VA"]
        self.cs_index = [s_els.index(e) for e in self.els]
        self.P = float(recipe.get("P", 101325.0))
        if self.kind == "binary":
            self.T_lo, self.T_hi = (float(t) for t in recipe["T_range"])
        self.strategy = strategy
        self.calls = 0
        self.eq_seconds = 0.0

    def norm(self, T, X):
        """(n, 2 phases, 2) coordinates scaled to the axes: binary (x, T as a fraction of the range), ternary
        (x1, x2) - mole fractions of the axis elements."""
        if self.kind == "binary":
            w = (np.asarray(T, float) - self.T_lo) / max(self.T_hi - self.T_lo, 1e-9)
            return np.stack([X[:, :, 1], np.repeat(w[:, None], 2, axis=1)], axis=-1)
        return np.stack([X[:, :, 1], X[:, :, 2]], axis=-1)

    def equilibrium(self, T: float, X: Sequence[float], full: bool = False):
        """([(state name, X in recipe order, NP, composition set)] of the stable phases, GM in J/mol of atoms), or
        None; full: also the chemical potentials (a third item)."""
        from pycalphad import Workspace
        v = self.v
        conds = {v.N: 1, v.P: self.P, v.T: float(T)}
        for k, e in enumerate(self.els[1:], start=1):
            conds[v.X(e)] = float(np.clip(X[k], 1e-6, 1 - 1e-6))
        st = self.strategy
        t0 = time.time()
        self.calls += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wks = Workspace(self.db, st.components, st.phases, conds, models=st.models,
                                phase_record_factory=st.phase_records, calc_opts={"pdens": PDENS})
                compsets = wks.get_composition_sets()
                mu = np.atleast_1d(np.squeeze(np.asarray(wks.eq.MU, float))) if full else None
        except Exception:  # noqa: BLE001 - no answer is "unsure", never "invalid"
            return None
        finally:
            self.eq_seconds += time.time() - t0
        items, G, NP = [], 0.0, 0.0
        for cs in compsets:
            if not np.isfinite(cs.NP) or cs.NP <= 1e-6:
                continue
            G += float(cs.NP) * float(cs.energy)
            NP += float(cs.NP)
            name = compset_names([cs], self.info)[0]
            x = np.asarray(cs.X, float)[self.cs_index]
            if any(it[0] == name and np.max(np.abs(x - it[1])) < 2e-3 for it in items):
                continue                                    # the same phase state found twice
            items.append((name, x, float(cs.NP), cs))
        if not items:
            return None
        return (items, G / NP, mu) if full else (items, G / NP)


# ---------------------------------------------------------------------------------------------------------
# 1. line validation
# ---------------------------------------------------------------------------------------------------------
def _test(ctx: _Ctx, L: _Line, i: int, found: Dict):
    """Equilibrium at the middle of tie-line i (T and overall composition).  True = the traced phases are the
    equilibrium: same phases in the same ordered/disordered states, or (another answer) the traced two-phase
    mixture is not higher in Gibbs energy than the equilibrium answer, i.e. equilibrium missed it or the point is
    degenerate (a node); False = the traced tie-line is metastable; None = no equilibrium result."""
    X = L.X[i].mean(axis=0)
    res = ctx.equilibrium(float(L.T[i]), X)
    if res is None:
        return None
    items, G_eq = res
    k = key_of([it[0] for it in items])
    if k == L.keys[i]:
        return True
    G_line = 0.5 * (L.G[i, 0] + L.G[i, 1])          # lever rule: half of each phase at the tie-line middle
    if not np.isfinite(G_line) or not np.isfinite(G_eq) or G_line <= G_eq + G_TOL:
        return True
    found.setdefault("eq", " + ".join(k))
    found.setdefault("dG", round(float(G_line - G_eq), 3))
    return False


def _validate(ctx: _Ctx, L: _Line, mode: str, deadline: float) -> Optional[Dict]:
    """Validity of every point (None if not checked).  Tests both ends and the middle (thorough: about every
    n/12-th point), then bisects between a valid and an invalid test."""
    if L.degenerate or L.n == 0:
        return None
    n = L.n
    if mode == "thorough":
        want = sorted(set(range(0, n, max(1, (n - 1) // 12))) | {n - 1})
    else:
        want = sorted({0, n - 1, (n - 1) // 2})
    res: Dict[int, Optional[bool]] = {}
    found: Dict = {}
    for i in want:
        if time.time() > deadline:
            return None
        res[i] = _test(ctx, L, i, found)
    good = (lambda i: res[i] is not False)
    if all(good(i) for i in res):
        return {"valid": np.ones(n, bool), "tests": len(res), "eq": None}
    tested = sorted(res)
    for a, b in zip(tested, tested[1:]):              # bisect between tests that disagree
        while b - a > 1 and good(a) != good(b):
            if time.time() > deadline:
                return None
            c = (a + b) // 2
            res[c] = _test(ctx, L, c, found)
            if good(c) == good(a):
                a = c
            else:
                b = c
    tested = sorted(res)
    valid = np.ones(n, bool)
    for i in range(n):                              # untested points: as the tests on both sides (they agree
        a = max(t for t in tested if t <= i)        # after bisection)
        valid[i] = good(a)
    return {"valid": valid, "tests": len(res), "eq": found.get("eq"), "dG": found.get("dG")}


def _pieces(valid: np.ndarray) -> List[Tuple[int, int]]:
    """Runs [a, b) of valid points with at least 2 points."""
    out, a = [], None
    for i, ok in enumerate(list(valid) + [False]):
        if ok and a is None:
            a = i
        elif not ok and a is not None:
            if i - a >= 2:
                out.append((a, i))
            a = None
    return out


def _new_zpf(z, pts):
    from pycalphad.mapping.primitives import ZPFLine, ZPFState
    status = z.status if z.status != ZPFState.NOT_FINISHED else ZPFState.FAILED
    return ZPFLine(list(z.fixed_phases), list(z.free_phases), points=list(pts), status=status,
                   axis_var=z.axis_var, axis_direction=z.axis_direction, current_delta=z.current_delta)


def _phase_text(key) -> str:
    return "+".join(key)


def _validate_lines(ctx: _Ctx, strategy, mode: str, deadline: float, rep: Dict) -> None:
    """Validate the ZPF lines not validated yet; trim / split / remove invalid parts in strategy.zpf_lines."""
    out = []
    for z in strategy.zpf_lines:
        if getattr(z, "_mc_checked", False) or len(z.points) == 0 or \
                len(z.points[0].stable_composition_sets) != 2:
            out.append(z)
            continue
        z._mc_checked = True
        L = _Line(z, ctx)
        r = _validate(ctx, L, mode, deadline)
        rep["lines_checked"] += r is not None
        rep["tests"] += r["tests"] if r else 0
        if r is None:
            if not L.degenerate and time.time() > deadline:
                rep["unchecked"] += 1
            out.append(z)
            continue
        if r["valid"].all():
            out.append(z)
            continue
        pcs = _pieces(r["valid"])
        removed = int(L.n - sum(b - a for a, b in pcs))
        ends = []
        if not r["valid"][0]:
            ends.append("start")
        if not r["valid"][-1]:
            ends.append("end")
        entry = {"phases": _phase_text(L.key), "points": L.n, "removed": removed, "pieces": len(pcs),
                 "at": ends or ["interior"], "equilibrium_there": r["eq"], "dG_J_per_mol": r.get("dG"),
                 "T_K": [round(float(L.T.min()), 2), round(float(L.T.max()), 2)],
                 "new": bool(rep.get("_phase") == "repair")}
        rep["trimmed"].append(entry)
        if len(pcs) == 1 and pcs[0] == (0, L.n):
            out.append(z)
            continue
        for a, b in pcs:
            nz = _new_zpf(z, z.points[a:b])
            nz._mc_checked = nz._mc_trimmed = True
            out.append(nz)
    strategy.zpf_lines[:] = out


# ---------------------------------------------------------------------------------------------------------
# 2. duplicates
# ---------------------------------------------------------------------------------------------------------
def _dist_to_polyline(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Distance of every point of P (n, 2) to the polyline Q (m, 2)."""
    if len(Q) == 1:
        return np.hypot(*(P - Q[0]).T)
    a, b = Q[:-1], Q[1:]
    d = b - a
    L2 = (d ** 2).sum(1)
    L2[L2 == 0] = 1e-30
    t = np.clip(((P[:, None, :] - a[None]) * d[None]).sum(-1) / L2[None], 0.0, 1.0)
    C = a[None] + t[..., None] * d[None]
    return np.sqrt(((P[:, None, :] - C) ** 2).sum(-1)).min(1)


def _covers(A: _Line, B: _Line, tol: float = DUP_TOL, frac: float = 0.9) -> bool:
    """Both phase boundaries of A lie on the boundaries of the same phases of B."""
    for k in (0, 1):
        want = base_name(A.names[A.n // 2][k])
        cand = [j for j in (0, 1) if base_name(B.names[B.n // 2][j]) == want] or [0, 1]
        best = 0.0
        for j in cand:
            best = max(best, float(np.mean(_dist_to_polyline(A.uv[:, k], B.uv[:, j]) <= tol)))
        if best < frac:
            return False
    return True


def _dedupe(ctx: _Ctx, strategy, rep: Dict) -> None:
    lines = [_Line(z, ctx) if len(z.points) and len(z.points[0].stable_composition_sets) == 2 else None
             for z in strategy.zpf_lines]
    drop = set()
    order = sorted((i for i, L in enumerate(lines) if L is not None and not L.degenerate),
                   key=lambda i: (lines[i].length(), lines[i].n, -i))
    for pos, i in enumerate(order):                  # shortest first: is it part of a longer (or equal) line?
        if i in drop:
            continue
        A = lines[i]
        for j in order[pos + 1:]:
            if j in drop:
                continue
            B = lines[j]
            if not (set(A.keys) & set(B.keys)) or not _covers(A, B):
                continue
            mutual = _covers(B, A)
            if mutual and A.max_step() < 0.8 * B.max_step():
                A, B, i_drop = B, A, j                      # the same trace twice: keep the finer-sampled one
            else:
                i_drop = i
            drop.add(i_drop)
            if A.sig() == B.sig():
                rep["exact_duplicates"] += 1
            else:       # a repeat of the same trace, or a shorter / trimmed partial one
                rep["duplicates"].append({"phases": _phase_text(A.key), "points": A.n, "kept_points": B.n,
                                          "trimmed": bool(getattr(A.z, "_mc_trimmed", False)),
                                          "partial": not mutual,
                                          "new": bool(rep.get("_phase") == "repair")})
            break
    if drop:
        strategy.zpf_lines[:] = [z for i, z in enumerate(strategy.zpf_lines) if i not in drop]


# ---------------------------------------------------------------------------------------------------------
# 3. nodes pycalphad rejected: confirm with equilibrium, add, trace their lines
# ---------------------------------------------------------------------------------------------------------
def _vertices(ctx: _Ctx, node):
    """(names, compositions in recipe order, T) of a three-phase node."""
    cs = node.stable_composition_sets
    return (compset_names(cs, ctx.info), [np.asarray(c.X, float)[ctx.cs_index] for c in cs],
            float(np.squeeze(node.global_conditions[ctx.vT])))


def _matched(items, names, X):
    """Node vertices (indices) that the equilibrium phases correspond to (same phase, nearest composition)."""
    out = set()
    for it in items:
        cand = [k for k in range(len(names)) if base_name(names[k]) == base_name(it[0])]
        if not cand:
            return None
        out.add(min(cand, key=lambda k: float(np.abs(X[k] - it[1]).max())))
    return frozenset(out) if len(out) == len(items) else None


def _solve_node(ctx: _Ctx, res, extra_cs, conds, free_var):
    """pycalphad node (three composition sets, free_var solved) from a two-phase equilibrium and a third set."""
    from pycalphad.mapping import zpf_equilibrium as zeq
    from pycalphad.mapping import utils as map_utils
    from pycalphad.mapping.primitives import Point
    items, _, mu = res
    pair = [it[3] for it in items]
    tot = sum(float(c.NP) for c in pair)
    for c in pair:
        map_utils.update_cs_phase_frac(c, float(c.NP) / tot)
    map_utils.update_cs_phase_frac(extra_cs, 1e-6)
    new_point = Point(dict(conds), np.asarray(mu, float), [], pair + [extra_cs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        node = zeq.create_node_from_different_points(new_point, pair, [free_var])
    if node is None or len(node.stable_composition_sets) != 3:
        return None
    node.encountered_points = []        # trace all three two-phase boundaries from it
    return node


def _settle(ctx: _Ctx, node, free_var, tries: int = 3):
    """pycalphad's own check of a node (BinaryStrategy._process_new_node): if another composition set lowers the
    energy, it replaces the node's nearest set and the node is solved again.  None if it does not settle."""
    from pycalphad.mapping import zpf_equilibrium as zeq
    st = ctx.strategy
    v = ctx.v
    for _ in range(tries + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand = zeq._find_global_min_cs(node, system_info=st.system_info, pdens=st.GLOBAL_MIN_PDENS,
                                           tol=st.GLOBAL_MIN_TOL, num_candidates=st.GLOBAL_MIN_NUM_CANDIDATES)
            if cand is None or not zeq._detect_degenerate_phase(node, cand[0]):
                return node
        cs_new = cand[0]
        css = node.stable_composition_sets
        xn = np.asarray(cs_new.X, float)
        k = min(range(3), key=lambda i: float(np.abs(np.asarray(css[i].X, float) - xn).max()))
        pair = [c for i, c in enumerate(css) if i != k]
        X = 0.5 * (np.asarray(pair[0].X, float) + np.asarray(pair[1].X, float))[ctx.cs_index]
        conds = {key: float(np.squeeze(val)) for key, val in node.global_conditions.items()}
        for j, e in enumerate(ctx.els[1:], start=1):
            conds[v.X(e)] = float(X[j])
        for c in pair:
            c.fixed = False
        node = _solve_node(ctx, ([(None, None, 0.5, c) for c in pair], None, node.chemical_potentials), cs_new,
                           conds, free_var)
        if node is None:
            return None
    return None


def _node_at_switch(ctx: _Ctx, r_lo, r_hi, lo: float, hi: float, xp: float, extra):
    """Node at the temperature where the equilibrium switches (within hi - lo <= 0.05 K): the two phases below and
    the new one above, from those equilibria (used when the node solver lands on another, metastable combination
    of composition sets)."""
    from pycalphad.mapping import utils as map_utils
    from pycalphad.mapping.primitives import Node
    v = ctx.v
    if r_lo is None or r_hi is None:
        return None
    third = extra(r_lo, r_hi, 0.005)
    if len(third) != 1:
        return None
    pair = [it[3] for it in r_lo[0]]
    tot = sum(float(c.NP) for c in pair)
    for c in pair:
        c.fixed = False
        map_utils.update_cs_phase_frac(c, float(c.NP) / tot)
    cs3 = third[0][3]
    cs3.fixed = True
    map_utils.update_cs_phase_frac(cs3, 0.0)
    conds = {v.N: 1.0, v.P: ctx.P, v.T: 0.5 * (lo + hi), v.X(ctx.els[1]): xp}
    node = Node(conds, np.asarray(r_lo[2], float), [cs3], pair, None)
    node.encountered_points = []
    return node


def _confirm_binary(ctx: _Ctx, node, T_win: float = 15.0):
    """The invariant near a rejected binary node: find where the two-phase equilibrium at a composition between the
    node's phases jumps (another pair of phases, or a phase composition that jumps) within T_win of it, bisect to
    0.05 K, solve the node there (pycalphad's own node solver, T free; if its solution is more than 0.25 K from the
    switch, the node is taken from the equilibria at the switch) and confirm the invariant with
    equilibria at T -/+ 0.5 (1, 2) K: two different two-phase equilibria of its three phases (the equilibrium
    switch and the solved invariant can differ by ~1 K where two phase pairs are almost equal in energy)."""
    from itertools import combinations
    v = ctx.v
    names, X, T0 = _vertices(ctx, node)
    xs = sorted({round(float(x[1]), 6) for x in X})
    probes = [float(np.mean([x[1] for x in X]))] + [float((a + b) / 2) for a, b in combinations(xs, 2)]
    Ts = [T for T in T0 + np.arange(-T_win, T_win + 0.01, 1.0) if ctx.T_lo <= T <= ctx.T_hi]

    def at(T, xp, full=False):
        res = ctx.equilibrium(T, np.array([1 - xp, xp]), full=full)
        return res if res and len(res[0]) == 2 and all(base_name(it[0]) in {base_name(n) for n in names}
                                                        for it in res[0]) else None

    def dist(r1, r2):
        """How different two two-phase equilibria are (inf: other phases)."""
        if sorted(base_name(it[0]) for it in r1[0]) != sorted(base_name(it[0]) for it in r2[0]):
            return np.inf
        a = sorted((base_name(it[0]), float(it[1][1])) for it in r1[0])
        b = sorted((base_name(it[0]), float(it[1][1])) for it in r2[0])
        return max(abs(p[1] - q[1]) for p, q in zip(a, b))

    def extra(r_lo, r_hi, tol):
        """Phases of r_hi that r_lo does not have (same phase within tol)."""
        return [it for it in r_hi[0] if not any(base_name(it[0]) == base_name(jt[0]) and
                                                 float(np.abs(it[1] - jt[1]).max()) < tol for jt in r_lo[0])]
    for xp in probes:
        prev = None
        for T in Ts:
            r = at(T, xp)
            if r is not None and prev is not None and dist(prev[1], r) > 0.01:
                lo, hi, r_lo, r_hi = prev[0], T, prev[1], r
                while hi - lo > 0.05:
                    mid = 0.5 * (lo + hi)
                    rm = at(mid, xp)
                    if rm is None:
                        break
                    if dist(rm, r_lo) <= dist(rm, r_hi):
                        lo, r_lo = mid, rm
                    else:
                        hi, r_hi = mid, rm
                if dist(r_lo, r_hi) <= 0.005:
                    prev = (T, r)
                    continue                                # a steep boundary, not a jump
                if len(extra(r_lo, r_hi, 0.005)) != 1 or len(extra(r_hi, r_lo, 0.005)) != 1:
                    prev = (T, r)
                    continue
                # solve the node from either side (the pair below + the new phase above, and vice versa) and keep
                # the solution nearest the switch: where two descriptions of one phase are nearly equal in energy
                # (Fe-Ni: FCC_A1 and disordered L12_FCC) the two can differ by ~1 K
                sols = []
                for T_a, T_b in ((lo, hi), (hi, lo)):
                    r_a, r_b = at(T_a, xp, True), at(T_b, xp, True)    # fresh composition sets (solving changes them)
                    third = extra(r_a, r_b, 0.005) if r_a is not None and r_b is not None else []
                    if len(third) != 1:
                        continue
                    conds = {v.N: 1, v.P: ctx.P, v.T: T_a, v.X(ctx.els[1]): xp}
                    sol = _solve_node(ctx, r_a, third[0][3], conds, v.T)
                    sol = _settle(ctx, sol, v.T) if sol is not None else None
                    if sol is not None:
                        sols.append((abs(_vertices(ctx, sol)[2] - 0.5 * (lo + hi)), sol))
                new = min(sols, key=lambda t: t[0])[1] if sols else None
                if new is None or min(sols, key=lambda t: t[0])[0] > 0.25:
                    new = _node_at_switch(ctx, at(lo, xp, True), at(hi, xp, True), lo, hi, xp, extra)
                if new is None:
                    return None
                n2, X2, T2 = _vertices(ctx, new)
                if key_of(n2) != key_of(names) or not (lo - 2 <= T2 <= hi + 2) or min(
                        float(np.abs(X2[i] - X2[j]).max()) for i, j in combinations(range(3), 2)) < 1e-3:
                    return None
                for dT in (0.5, 1.0, 2.0):         # equilibria a little below / above: two different pairs
                    below, above = at(T2 - dT, xp), at(T2 + dT, xp)
                    if below is not None and above is not None and dist(below, above) > 0.005 + 0.004 * dT:
                        return new
                return None
            prev = (T, r) if r is not None else None
    return None


def _confirm_ternary(ctx: _Ctx, node):
    """A rejected tie-triangle: equilibrium (pdens PDENS) inside it must give the same three phases."""
    v = ctx.v
    names, X, T = _vertices(ctx, node)
    res = ctx.equilibrium(T, np.mean(X, axis=0), full=True)
    if res is None or len(res[0]) != 3 or _matched(res[0], names, X) != frozenset(range(3)):
        return None
    items, G, mu = res
    order = sorted(range(3), key=lambda k: -items[k][2])
    pair = ([items[k] for k in order[:2]], G, mu)
    extra = items[order[2]][3]
    tot = sum(it[2] for it in pair[0])
    xc = sum(it[2] / tot * it[1] for it in pair[0])
    conds = {v.N: 1, v.P: ctx.P, v.T: T, v.X(ctx.els[1]): float(xc[1]), v.X(ctx.els[2]): float(xc[2])}
    new = _solve_node(ctx, pair, extra, conds, v.X(ctx.els[1]))
    return _settle(ctx, new, v.X(ctx.els[1])) if new is not None else None


def _same_node(ctx: _Ctx, a, b, dT: float = 1.0, dx: float = 0.01) -> bool:
    na, Xa, Ta = _vertices(ctx, a)
    nb, Xb, Tb = _vertices(ctx, b)
    if key_of(na) != key_of(nb) or abs(Ta - Tb) > dT:
        return False
    return all(min(float(np.abs(x - y).max()) for y in Xb) < dx for x in Xa)


def _rejected_nodes(ctx: _Ctx, strategy, mode: str, deadline: float, rep: Dict) -> None:
    """Nodes pycalphad found but rejected as metastable: confirmed by point equilibria (pdens PDENS) -> added to
    the strategy (an invariant / tie-triangle) and the lines that start there traced."""
    rejected = list(getattr(strategy, "_mc_rejected", None) or [])
    strategy._mc_rejected = []
    seen = []
    for node in rejected:
        if time.time() > deadline:
            rep["unchecked"] += 1
            break
        if any(_same_node(ctx, node, o, dT=3.0, dx=0.03) for o in seen):
            continue
        seen.append(node)
        names, _, T0 = _vertices(ctx, node)
        try:
            new = _confirm_binary(ctx, node) if ctx.kind == "binary" else _confirm_ternary(ctx, node)
        except Exception:  # noqa: BLE001 - an unconfirmed node is only reported
            new = None
        entry = {"phases": _phase_text(key_of(names)), "pycalphad_T_K": round(T0, 2)}
        if new is None:
            rep["nodes_unconfirmed"].append(entry)
            continue
        n2, X2, T2 = _vertices(ctx, new)
        entry.update(T_K=round(T2, 2), compositions={n: [round(float(x), 5) for x in xx] for n, xx in zip(n2, X2)})
        if any(len(o.stable_composition_sets) == 3 and _same_node(ctx, new, o)
               for o in strategy.node_queue.nodes):
            continue                                    # pycalphad has it already (found from another line)
        n0, nd0 = len(strategy.zpf_lines), len(getattr(strategy, "_mc_dropped", []))
        strategy.node_queue.add_node(new, True)
        _iterate(strategy, deadline)
        cands = list(strategy.zpf_lines[n0:])
        del strategy.zpf_lines[n0:]
        cands += [_new_zpf(z, z.points) for z in getattr(strategy, "_mc_dropped", [])[nd0:]]
        new_lines = _merge(ctx, strategy, cands, mode, deadline, rep)
        if ctx.kind == "binary":
            _extend_failed(ctx, strategy, new_lines, T2, rep)
            new_lines = [_Line(L.z, ctx) for L in new_lines]
        entry["lines_added"] = len(new_lines)
        if new_lines and ctx.kind == "binary":
            T = np.concatenate([L.T for L in new_lines])
            entry["added_T_K"] = [round(float(T.min()), 1), round(float(T.max()), 1)]
        rep["nodes_added"].append(entry)
        rep["_nodes"].append(new)


def _extend_failed(ctx: _Ctx, strategy, lines: List[_Line], T_node: float, rep: Dict, max_steps: int = 80) -> int:
    """Binary: a line traced from a confirmed node that failed on the way (pycalphad keeps finding a "new" set of
    the same phase, e.g. in a miscibility gap whose two sets are two descriptions of one phase) is continued with
    point equilibria (pdens PDENS) at the middle of its last tie-line, in steps of a quarter of the mapping
    temperature step, while equilibrium has the same two phases.  Returns the number of points added."""
    from pycalphad.mapping.primitives import Point, ZPFState
    v = ctx.v
    step = float(np.atleast_1d(strategy.conditions[v.T])[-1]) / 4
    added = 0
    for L in lines:
        if L.z.status != ZPFState.FAILED or L.n < 2:
            continue
        at_end = abs(L.T[-1] - T_node) >= abs(L.T[0] - T_node)
        e, prev = (L.n - 1, L.n - 2) if at_end else (0, 1)
        sgn = np.sign(L.T[e] - L.T[prev])
        if sgn == 0:
            continue
        T, Xp = float(L.T[e]), L.X[e].copy()
        want = sorted(base_name(n) for n in L.names[e])
        new_pts = []

        def point_at(T, Xp):
            """(Point, tie-line compositions) of the line's two phases at T, or the single-phase result."""
            res = ctx.equilibrium(T, Xp.mean(axis=0), full=True)
            if res is None:
                return None, None, None
            if len(res[0]) != 2 or sorted(base_name(it[0]) for it in res[0]) != want:
                return None, None, res
            items = list(res[0])
            if float(np.abs(items[0][1] - Xp[0]).max() + np.abs(items[1][1] - Xp[1]).max()) > \
                    float(np.abs(items[1][1] - Xp[0]).max() + np.abs(items[0][1] - Xp[1]).max()):
                items = items[::-1]                         # same phase order as the line
            Xn = np.array([items[0][1], items[1][1]])
            conds = {v.N: 1.0, v.P: ctx.P, v.T: T, v.X(ctx.els[1]): float(Xn.mean(axis=0)[1])}
            return Point(conds, np.asarray(res[2], float), [], [items[0][3], items[1][3]]), Xn, res
        for _ in range(max_steps):
            T_last, T = T, T + sgn * step
            if not (ctx.T_lo <= T <= ctx.T_hi):
                break
            pt, Xn, res = point_at(T, Xp)
            if pt is None:
                # a miscibility gap that closes: bisect to it and end the line on the single phase there
                if res is not None and len(res[0]) == 1 and want[0] == want[1] == base_name(res[0][0][0]):
                    lo, hi, single = T_last, T, res
                    for _ in range(6):
                        mid = 0.5 * (lo + hi)
                        pm, Xm, rm = point_at(mid, Xp)
                        if pm is not None:
                            new_pts.append(pm)
                            lo, Xp = mid, Xm
                        elif rm is not None and len(rm[0]) == 1:
                            hi, single = mid, rm
                        else:
                            break
                    cs = single[0][0][3]
                    conds = {v.N: 1.0, v.P: ctx.P, v.T: hi, v.X(ctx.els[1]): float(single[0][0][1][1])}
                    new_pts.append(Point(conds, np.asarray(single[2], float), [], [cs, cs]))
                break
            if float(np.abs(Xn[0] - Xn[1]).max()) < 1e-3:
                break                                       # the two sets merge (critical point)
            new_pts.append(pt)
            Xp = Xn
        if new_pts:
            if at_end:
                L.z.points.extend(new_pts)
            else:
                L.z.points[:0] = new_pts[::-1]
            added += len(new_pts)
    rep["points_added_by_equilibrium"] = rep.get("points_added_by_equilibrium", 0) + added
    return added


def _drop_degenerate_nodes(ctx: _Ctx, strategy, rep: Dict) -> None:
    """Remove three-phase 'nodes' where two of the sets are the same phase state at the same composition (e.g. a
    phase and the disordered state of an ordered phase that describes the same structure, both at a pure element):
    not an invariant reaction."""
    keep = []
    for node in strategy.node_queue.nodes:
        css = node.stable_composition_sets
        if len(css) == 3:
            names, X, _ = _vertices(ctx, node)
            if any(base_name(names[i]) == base_name(names[j]) and float(np.abs(X[i] - X[j]).max()) < 1e-4
                   for i in range(3) for j in range(i + 1, 3)):
                rep["degenerate_nodes_dropped"] = rep.get("degenerate_nodes_dropped", 0) + 1
                continue
        keep.append(node)
    if len(keep) != len(strategy.node_queue.nodes):
        strategy.node_queue.nodes[:] = keep
        strategy.node_queue._current_node_index = len(keep)


NODE_G_TOL = 5.0            # J/mol: a node's tie-lines may be this far above equilibrium (vertices from equilibria
                            # a few hundredths of a K apart); spurious nodes are off by orders of magnitude more


def _validate_nodes(ctx: _Ctx, strategy, rep: Dict) -> None:
    """Every three-phase node (pycalphad's and the check's own) must be an equilibrium: at the node temperature,
    equilibrium at the middle of each pair of its phases (and, ternary, at the centre of the tie-triangle) must not
    be lower in Gibbs energy than the node's own phases mixed there (NODE_G_TOL).  A node that fails is removed
    (so neither its invariant line / tie-triangle nor its label is drawn).  Nodes with the same phases at the same
    temperature (binary, within 0.1 K) or the same vertices (ternary, within 0.01) are kept once."""
    from itertools import combinations
    keep = []
    for node in strategy.node_queue.nodes:
        css = node.stable_composition_sets
        if len(css) != 3:
            keep.append(node)
            continue
        names, X, T = _vertices(ctx, node)
        dup = next((o for o in keep if len(o.stable_composition_sets) == 3 and
                    _same_node(ctx, node, o, dT=0.1 if ctx.kind == "binary" else 1e-6,
                               dx=1.0 if ctx.kind == "binary" else 0.01)), None)
        if dup is not None:
            rep["duplicate_nodes_dropped"] = rep.get("duplicate_nodes_dropped", 0) + 1
            continue
        G = [float(c.energy) for c in css]
        probes = [((i, j), 0.5 * (X[i] + X[j]), 0.5 * (G[i] + G[j])) for i, j in combinations(range(3), 2)
                  if float(np.abs(X[i] - X[j]).max()) > 1e-3]
        if ctx.kind == "ternary":
            probes.append(((0, 1, 2), np.mean(X, axis=0), float(np.mean(G))))
        worst, eq_there = 0.0, None
        for _, Xp, Gmix in probes:
            res = ctx.equilibrium(T, Xp)
            if res is None:
                continue
            dG = Gmix - res[1]
            if dG > worst:
                worst, eq_there = dG, " + ".join(sorted(it[0] for it in res[0]))
        if worst > NODE_G_TOL:
            entry = {"phases": _phase_text(key_of(names)), "T_K": round(T, 2), "dG_J_per_mol": round(worst, 1),
                     "equilibrium_there": eq_there,
                     "compositions": {n: [round(float(x), 5) for x in xx] for n, xx in zip(names, X)}}
            same = [(names[i], names[j]) for i, j in combinations(range(3), 2)
                    if float(np.abs(X[i] - X[j]).max()) <= 1e-3 and base_name(names[i]) != base_name(names[j])]
            if same and ctx.kind == "binary":
                k = [i for i in range(3) if names[i] == same[0][0]][0]
                entry["at_pure_element_transformation"] = {"phases": list(same[0]),
                                                          f"X({ctx.els[1]})": round(float(X[k][1]), 5)}
            ends = 0                                        # line ends that stopped at it (left as traced:
            for L in _lines(ctx, strategy):                 # they were validated with the lines)
                for e in (0, L.n - 1):
                    if abs(L.T[e] - T) < 0.5 and all(min(float(np.abs(L.X[e, k] - x).max()) for x in X) < 1e-3
                                                     for k in (0, 1)):
                        ends += 1
            entry["line_ends_there"] = ends
            rep["nodes_rejected"].append(entry)
            continue
        rep["nodes_checked"] = rep.get("nodes_checked", 0) + 1
        rep["nodes_max_dG_J_per_mol"] = round(max(rep.get("nodes_max_dG_J_per_mol", 0.0), worst), 3)
        keep.append(node)
    if len(keep) != len(strategy.node_queue.nodes):
        strategy.node_queue.nodes[:] = keep
        strategy.node_queue._current_node_index = len(keep)


def _join_nodes(ctx: _Ctx, strategy, nodes, rep: Dict, dT: float = 5.0, dx: float = 0.015) -> None:
    """A line of two of a confirmed node's phases that ends just short of it (within dT K and dx in composition of
    the node's two vertices) gets the node's tie-line as its last point, so its boundaries meet the invariant."""
    from pycalphad.mapping.primitives import Point
    for node in nodes:
        names, X, T = _vertices(ctx, node)
        css = node.stable_composition_sets
        for z in strategy.zpf_lines:
            if not z.points or len(z.points[0].stable_composition_sets) != 2:
                continue
            L = _Line(z, ctx)
            if L.degenerate:
                continue
            for end, nxt in ((0, 1), (L.n - 1, L.n - 2)):
                if abs(L.T[end] - T) > dT:
                    continue
                ks = []
                for k in (0, 1):
                    cand = [j for j in range(3) if base_name(names[j]) == base_name(L.names[end][k])
                            and float(np.abs(X[j] - L.X[end, k]).max()) < dx]
                    ks.append(min(cand, key=lambda j: float(np.abs(X[j] - L.X[end, k]).max())) if cand else None)
                if None in ks or ks[0] == ks[1]:
                    continue
                gap = max(float(np.abs(X[ks[k]] - L.X[end, k]).max()) for k in (0, 1))
                if gap < 1e-5 and abs(L.T[end] - T) < 1e-3:
                    continue                                # ends at the node already
                if 0 <= nxt < L.n and max(float(np.abs(X[ks[k]] - L.X[nxt, k]).max()) for k in (0, 1)) < gap:
                    continue                                # the line runs past the node: not its end
                p = Point(dict(node.global_conditions), np.asarray(node.chemical_potentials, float), [],
                          [css[ks[0]], css[ks[1]]])
                if end == 0:
                    z.points.insert(0, p)
                else:
                    z.points.append(p)
                rep["lines_joined_to_nodes"] += 1
                break


# ---------------------------------------------------------------------------------------------------------
# 4. missing boundaries
# ---------------------------------------------------------------------------------------------------------
def grid_limits(recipe: Dict) -> Tuple[List[float], List[float]]:
    """Plotted x / y limits of the whole diagram (the grid region labelling uses for a default axes)."""
    if recipe["kind"] == "ternary":
        return [0.0, 1.0], [0.0, 1.0]
    els = recipe["elements"]
    xr = recipe.get("x_range") or [0.0, 1.0]
    x0 = fractions_to_plot(recipe, {els[0]: 1 - xr[0], els[1]: xr[0]})[0]
    x1 = fractions_to_plot(recipe, {els[0]: 1 - xr[1], els[1]: xr[1]})[0]
    y = y_of_T(recipe, np.asarray(recipe["T_range"], float))
    return [float(x0), float(x1)], [float(y[0]), float(y[1])]


def grid_size(kind: str, labels: bool = True, mode: str = "auto") -> Tuple[int, int]:
    """The region-label grid (GRID_N) when the labels reuse it or the check is thorough, else half as fine."""
    n = GRID_N[kind]
    if labels or mode == "thorough":
        return tuple(n)
    return tuple(max(10, k // 2) for k in n)


def reuse_grid(report: Optional[Dict], xlim, ylim, n=None, rename: bool = True):
    """The check's equilibrium grid, if it was computed for these axis limits and grid size (else None)."""
    if not report or report.get("grid") is None or not rename:
        return None
    if n is not None and list(report.get("grid_n") or []) != list(n):
        return None
    lim = report.get("grid_lim")
    if lim and np.allclose(lim[0], list(xlim), rtol=0, atol=1e-9) and np.allclose(lim[1], list(ylim), rtol=0,
                                                                                    atol=1e-9):
        return report["grid"]
    return None


class _Cells:
    """Equilibrium grid nodes as integer cells (i, j) with their phase-set keys."""

    def __init__(self, ctx: _Ctx, grid: Dict, xlim, ylim, n):
        self.ctx = ctx
        self.kind = ctx.kind
        self.n = tuple(int(k) for k in n)
        xy = np.asarray(grid["xy"], float).reshape(-1, 2)
        keys = np.asarray(grid["keys"], int)
        table = [tuple(k) for k in grid["table"]]
        if self.kind == "binary":
            self.xlim, self.ylim = list(xlim), list(ylim)
            ij = self.to_ij(xy)
        else:
            fr = plot_to_fractions(ctx.recipe, xy[:, 0], xy[:, 1])
            ij = np.column_stack([fr[ctx.els[1]], fr[ctx.els[2]]]) * self.n[0] - 0.5
        self.ij = np.round(ij).astype(int)
        self.keys = [table[k] if k >= 0 else None for k in keys]

    def to_ij(self, xy):
        """Plotted points (binary) -> grid index coordinates (cell centres at integers)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        i = (xy[:, 0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.n[0] - 0.5
        j = (xy[:, 1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.n[1] - 0.5
        return np.column_stack([i, j])

    def line_ij(self, L: _Line, k: int) -> np.ndarray:
        if self.kind == "binary":
            pts = [xy_of(self.ctx.recipe, dict(zip(self.ctx.els, L.X[i, k])), L.T[i]) for i in range(L.n)]
            return self.to_ij(pts)
        return np.column_stack([L.X[:, k, 1], L.X[:, k, 2]]) * self.n[0] - 0.5

    def centre(self, c) -> Tuple[float, np.ndarray]:
        """(T in K, mole fractions in recipe order) at the centre of cell c."""
        i, j = c
        rec = self.ctx.recipe
        if self.kind == "binary":
            x = self.xlim[0] + (i + 0.5) / self.n[0] * (self.xlim[1] - self.xlim[0])
            y = self.ylim[0] + (j + 0.5) / self.n[1] * (self.ylim[1] - self.ylim[0])
            fr = plot_to_fractions(rec, x)
            return float(T_of_y(rec, y)), np.array([float(fr[e]) for e in self.ctx.els])
        a, b = (i + 0.5) / self.n[0], (j + 0.5) / self.n[0]
        return float(rec["T_K"]), np.array([1 - a - b, a, b])


def _inside(P: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Points P (n, 2) inside the closed polygon poly (m, 2) (even-odd rule)."""
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    a, b = poly, np.roll(poly, -1, axis=0)
    cond = (a[:, 1][None] > y) != (b[:, 1][None] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = a[:, 0][None] + (y - a[:, 1][None]) * (b[:, 0] - a[:, 0])[None] / (b[:, 1] - a[:, 1])[None]
    return (np.sum(cond & (x < xc), axis=1) % 2) == 1


def _uncovered(cells: _Cells, lines: List[_Line]) -> Dict[tuple, List[tuple]]:
    """Two-phase grid cells not inside (or next to) a traced field of the same phases, per key."""
    by_key: Dict[tuple, List[int]] = {}
    for n, k in enumerate(cells.keys):
        if k is not None and len(k) == 2:
            by_key.setdefault(k, []).append(n)
    out = {}
    polys: Dict[tuple, List[np.ndarray]] = {}
    for L in lines:
        if L.degenerate:
            continue
        a, b = cells.line_ij(L, 0), cells.line_ij(L, 1)
        poly = np.vstack([a, b[::-1]])
        poly = poly[np.isfinite(poly).all(1)]
        if not len(poly):
            continue
        for k in set(L.keys):
            polys.setdefault(k, []).append(poly)
    for k, idx in by_key.items():
        P = cells.ij[idx].astype(float)
        cov = np.zeros(len(P), bool)
        for poly in polys.get(k, []):
            rest = ~cov
            if not rest.any():
                break
            Q = P[rest]
            hit = _dist_to_polyline(Q, np.vstack([poly, poly[:1]])) <= COVER_TOL
            if len(poly) >= 3:
                hit |= _inside(Q, poly)
            cov[np.flatnonzero(rest)[hit]] = True
        miss = [tuple(c) for c in cells.ij[np.asarray(idx)[~cov]]]
        if miss:
            out[k] = miss
    return out


def _components(cells_list: List[tuple]) -> List[List[tuple]]:
    """8-connected groups of grid cells."""
    left = set(cells_list)
    out = []
    while left:
        seed = left.pop()
        comp, stack = [seed], [seed]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    c = (i + di, j + dj)
                    if c in left:
                        left.remove(c)
                        comp.append(c)
                        stack.append(c)
        out.append(sorted(comp))
    return out


def _missing(cells: _Cells, lines: List[_Line]) -> List[Dict]:
    areas = []
    for k, miss in _uncovered(cells, lines).items():
        for comp in _components(miss):
            if len(comp) < MIN_CELLS:
                continue                                    # grid-resolution sliver: not a missing boundary
            ii = {c[0] for c in comp}
            jj = {c[1] for c in comp}
            if len(ii) < 2 or len(jj) < 2:
                continue                                    # one row / column of cells: a sliver too
            areas.append({"key": k, "cells": comp})
    return areas


def _area_text(cells: _Cells, area: Dict) -> Dict:
    """Extent of a missing area (T in K for a binary, composition ranges for a ternary)."""
    pts = [cells.centre(c) for c in area["cells"]]
    els = cells.ctx.els
    out = {"phases": _phase_text(area["key"]), "cells": len(area["cells"])}
    if cells.kind == "binary":
        T = [p[0] for p in pts]
        x = [p[1][1] for p in pts]
        dT = (cells.ctx.T_hi - cells.ctx.T_lo) / cells.n[1] / 2
        out["T_K"] = [round(min(T) - dT, 1), round(max(T) + dT, 1)]
        out[f"X({els[1]})"] = [round(min(x), 3), round(max(x), 3)]
    else:
        for k in (1, 2):
            x = [p[1][k] for p in pts]
            out[f"X({els[k]})"] = [round(min(x), 3), round(max(x), 3)]
    return out


def _starts(area: Dict, kind: str) -> List[tuple]:
    """Up to three start cells: the deepest cell of the area, and (tall binary areas) one near each end."""
    cells = area["cells"]
    S = set(cells)

    def depth(c):
        d = 0
        while all((c[0] + a, c[1] + b) in S for a in range(-d - 1, d + 2) for b in range(-d - 1, d + 2)):
            d += 1
        return d
    out = [max(cells, key=lambda c: (depth(c), -abs(c[1] - np.mean([q[1] for q in cells]))))]
    js = sorted({c[1] for c in cells})
    if kind == "binary" and len(js) >= 6:
        for band in (js[: len(js) // 3], js[-(len(js) // 3):]):
            sub = [c for c in cells if c[1] in band]
            if sub:
                out.append(max(sub, key=depth))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------------------------------------
def _area_hit(cells: _Cells, L: _Line, area: Dict) -> bool:
    """Does line L bound (part of) the area?"""
    if L.degenerate or area["key"] not in set(L.keys):
        return False
    P = np.asarray(area["cells"], float)
    poly = np.vstack([cells.line_ij(L, 0), cells.line_ij(L, 1)[::-1]])
    hit = _dist_to_polyline(P, np.vstack([poly, poly[:1]])) <= COVER_TOL
    if len(poly) >= 3:
        hit |= _inside(P, poly)
    return bool(hit.any())


def _iterate(strategy, deadline: float) -> bool:
    """Continue a strategy (new start points) until finished or the deadline; keeps dropped lines."""
    dropped = getattr(strategy, "_mc_dropped", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while True:
            nl = len(strategy.zpf_lines)
            last = strategy.zpf_lines[-1] if nl else None
            if strategy.iterate():
                return True
            if dropped is not None and last is not None and len(strategy.zpf_lines) < nl and len(last.points) >= 2:
                dropped.append(last)
            if time.time() > deadline:
                return False


def _repair_area(ctx: _Ctx, strategy, cells: _Cells, area: Dict, mode: str, deadline: float, rep: Dict) -> Dict:
    """Try to trace the boundary of one missing area; returns what was done."""
    from pycalphad import variables as v
    how, added, spans = [], 0, []
    key = area["key"]

    def missing_now():
        lines = _lines(ctx, strategy)
        left = [a for a in _missing(cells, lines) if a["key"] == key and set(a["cells"]) & set(area["cells"])]
        return left

    def merge(cands, label):
        nonlocal added
        new = _merge(ctx, strategy, cands, mode, deadline, rep)
        if new:
            added += len(new)
            how.append(f"{label}: {len(new)} line(s)")
            spans.extend(L.T for L in new)

    def done():
        out = {"added": added, "how": how, "repaired": not missing_now()}
        if spans and ctx.kind == "binary":
            T = np.concatenate(spans)
            out["added_T_K"] = [round(float(T.min()), 1), round(float(T.max()), 1)]
        return out

    # a. lines pycalphad traced through the area but threw away
    dropped = [z for z in getattr(strategy, "_mc_dropped", []) if not getattr(z, "_mc_rescued", False)]
    cands = []
    for z in dropped:
        L = _Line(z, ctx)
        if _area_hit(cells, L, area):
            z._mc_rescued = True
            cands.append(_new_zpf(z, z.points))
    merge(cands, "kept a line pycalphad discarded")
    if not missing_now() or time.time() > deadline:
        return done()

    # b. start points inside the area
    n0 = len(strategy.zpf_lines)
    nd0 = len(getattr(strategy, "_mc_dropped", []))
    started = 0
    for c in _starts(area, cells.kind):
        T, X = cells.centre(c)
        conds = {v.N: 1, v.P: ctx.P, v.T: T}
        for k, e in enumerate(ctx.els[1:], start=1):
            conds[v.X(e)] = float(np.clip(X[k], 1e-6, 1 - 1e-6))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                started += bool(strategy.add_nodes_from_conditions(conds))
        except Exception:  # noqa: BLE001 - a start point that fails is skipped
            pass
    if started:
        _iterate(strategy, deadline)
        new = list(strategy.zpf_lines[n0:])
        del strategy.zpf_lines[n0:]
        for z in getattr(strategy, "_mc_dropped", [])[nd0:]:
            if _area_hit(cells, _Line(z, ctx), area):
                z._mc_rescued = True
                new.append(_new_zpf(z, z.points))
        merge(new, "start points")
    if not missing_now() or time.time() > deadline:
        return done()

    # c. finer re-map of the area only
    sub = _submap(ctx, strategy, cells, area, deadline)
    if sub:
        merge([z for z in sub if area["key"] in {key_of(nm) for nm in [compset_names(
            z.points[len(z.points) // 2].stable_composition_sets[:2], ctx.info)]}], "finer re-map of the area")
    return done()


def _merge(ctx: _Ctx, strategy, cands, mode: str, deadline: float, rep: Dict) -> List[_Line]:
    """Add candidate lines, validate and de-duplicate everything new; the new lines that remain."""
    if not cands:
        return []
    for z in strategy.zpf_lines:
        z._mc_old = True
    strategy.zpf_lines.extend(z for z in cands if not getattr(z, "_mc_old", False))
    _validate_lines(ctx, strategy, mode, deadline, rep)
    _dedupe(ctx, strategy, rep)
    return [L for L in (_Line(z, ctx) for z in strategy.zpf_lines if not getattr(z, "_mc_old", False)
                        and z.points and len(z.points[0].stable_composition_sets) == 2) if not L.degenerate]


def _submap(ctx: _Ctx, strategy, cells: _Cells, area: Dict, deadline: float):
    """ZPF lines of a re-map of the area's box with half the steps (None if no time)."""
    from pycalphad import variables as v
    pts = [cells.centre(c) for c in area["cells"]]
    conds = {v.N: 1, v.P: ctx.P}
    sc = strategy.conditions
    if ctx.kind == "binary":
        T_step = float(sc[v.T][2]) / 2
        x_step = float(sc[v.X(ctx.els[1])][2]) / 2
        dT = (ctx.T_hi - ctx.T_lo) / cells.n[1]
        Ts = [p[0] for p in pts]
        xs = [p[1][1] for p in pts]
        conds[v.T] = (max(ctx.T_lo, min(Ts) - 2 * dT), min(ctx.T_hi, max(Ts) + 2 * dT), T_step)
        conds[v.X(ctx.els[1])] = (max(0.0, min(xs) - 0.05), min(1.0, max(xs) + 0.05), x_step)
    else:
        conds[v.T] = float(sc[v.T])
        for k in (1, 2):
            xs = [p[1][k] for p in pts]
            conds[v.X(ctx.els[k])] = (max(0.0, min(xs) - 0.05), min(1.0, max(xs) + 0.05),
                                      float(sc[v.X(ctx.els[k])][2]) / 2)
    if time.time() > deadline:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s2 = strategy.__class__(ctx.db, strategy.components, strategy.phases, conds, models=strategy.models,
                                    phase_record_factory=strategy.phase_records)
        s2._mc_dropped = []
        t_left = max(1.0, deadline - time.time())
        run_strategy(s2, t_left, "auto")
    except Exception:  # noqa: BLE001 - the re-map is best effort
        return None
    return [z for z in s2.zpf_lines + s2._mc_dropped if len(z.points) >= 2]


def _lines(ctx: _Ctx, strategy) -> List[_Line]:
    return [_Line(z, ctx) for z in strategy.zpf_lines
            if len(z.points) and len(z.points[0].stable_composition_sets) == 2]


# ---------------------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------------------
def check_and_repair(strategy, db, recipe: Dict, *, mode: str = "auto", grid: Optional[Dict] = None,
                     xlim=None, ylim=None, grid_n=None, budget: Optional[float] = None,
                     deadline: Optional[float] = None, info: Optional[Dict] = None) -> Dict:
    """Check the ZPF lines of a finished BinaryStrategy / TernaryStrategy against equilibrium and repair them in
    place (strategy.zpf_lines is trimmed / split / extended; nodes are not changed).

    recipe: the diagram (see phase_regions: kind, elements [balance, x(, y)], phases, P, T_range / T_K, comp,
    tmap, x_range).  grid / xlim / ylim / grid_n: an equilibrium grid already computed (phase_regions.
    equilibrium_grid, default size GRID_N), else one is computed over xlim / ylim (default the whole diagram) and
    returned ("grid", "grid_lim", "grid_n") for the region labels (see reuse_grid).  budget: seconds for the
    check without the grid (default: 30 s or 1.5 x the mapping time, thorough twice that); deadline: absolute
    time.time() limit for everything (the tool's time_limit_s).  Returns a report with "text"."""
    mode = check_mode(mode)
    rep: Dict = {"mode": mode, "lines_checked": 0, "tests": 0, "trimmed": [], "duplicates": [],
                 "exact_duplicates": 0, "missing": [], "nodes_added": [], "nodes_unconfirmed": [],
                 "nodes_rejected": [], "lines_joined_to_nodes": 0, "unchecked": 0, "complete": True, "text": "",
                 "_nodes": []}
    if mode == "off":
        rep["text"] = "mapping check: off (pycalphad's lines as traced)"
        rep["complete"] = False
        return rep
    if recipe.get("kind") not in ("binary", "ternary"):
        rep["text"] = "mapping check: not available for this diagram kind"
        return rep
    t0 = time.time()
    if budget is None:
        mt = float(getattr(strategy, "_mc_seconds", 0.0) or 0.0)
        budget = max(30.0, 1.5 * mt) * (2 if mode == "thorough" else 1)
    hard = float("inf") if deadline is None else float(deadline)
    end = min(t0 + budget, hard)
    comps = sorted(recipe["elements"]) + ["VA"]
    if info is None:
        info = ordered_info(db, comps, recipe["phases"], strategy.models)
    ctx = _Ctx(strategy, db, recipe, info)
    rep["lines_traced"] = len(strategy.zpf_lines)

    # 1. validate every line, 2. drop duplicates, 3. nodes pycalphad rejected
    _validate_lines(ctx, strategy, mode, end, rep)
    _dedupe(ctx, strategy, rep)
    rep["_phase"] = "repair"
    _rejected_nodes(ctx, strategy, mode, end, rep)
    rep["_phase"] = None
    t1 = time.time()

    # 3. missing boundaries (equilibrium grid, shared with the region labels)
    if xlim is None or ylim is None:
        xlim, ylim = grid_limits(recipe)
    grid_n = tuple(grid_n or GRID_N[recipe["kind"]])
    if grid is None:
        try:
            grid = equilibrium_grid(db, recipe, info, xlim, ylim, grid_n,
                                    models=strategy.models, phase_records=strategy.phase_records)
        except Exception as e:  # noqa: BLE001 - no grid: no search for missing boundaries
            rep["grid_error"] = f"{type(e).__name__}: {e}"
            grid = None
    rep["grid"], rep["grid_lim"] = grid, [list(map(float, xlim)), list(map(float, ylim))]
    rep["grid_n"] = list(grid_n)
    t2 = time.time()
    end = min(end + (t2 - t1), hard)     # the grid is shared with the region labels: not part of the budget
    if grid is not None:
        cells = _Cells(ctx, grid, xlim, ylim, grid_n)
        areas = _missing(cells, _lines(ctx, strategy))
        rep["_phase"] = "repair"
        for area in areas:
            entry = _area_text(cells, area)
            if time.time() > end:
                entry.update(repaired=False, how=["no time left"])
                rep["unchecked"] += 1
            else:
                entry.update(_repair_area(ctx, strategy, cells, area, mode, end, rep))
            rep["missing"].append(entry)
        if areas:       # what is still missing after all repairs (an area can be fixed by another one's lines)
            left = _missing(cells, _lines(ctx, strategy))
            for entry, area in zip(rep["missing"], areas):
                entry["repaired"] = not any(a["key"] == area["key"] and set(a["cells"]) & set(area["cells"])
                                            for a in left)
    _join_nodes(ctx, strategy, rep.pop("_nodes"), rep)
    _drop_degenerate_nodes(ctx, strategy, rep)
    _validate_nodes(ctx, strategy, rep)
    rep["complete"] = rep["unchecked"] == 0
    rep["seconds"] = {"validate": round(t1 - t0, 2), "grid": round(t2 - t1, 2),
                      "repair": round(time.time() - t2, 2), "total": round(time.time() - t0, 2),
                      "equilibrium": round(ctx.eq_seconds, 2)}
    rep["equilibrium_calls"] = ctx.calls
    rep["lines_after"] = len(strategy.zpf_lines)
    rep.pop("_phase", None)
    rep["text"] = report_text(rep, recipe)
    return rep


def _fmt_T(TK: float, recipe: Dict) -> str:
    return f"{TK - 273.15:.0f} °C" if recipe.get("T_unit") == "C" else f"{TK:.0f} K"


def report_text(rep: Dict, recipe: Dict) -> str:
    """1-3 lines for the tool result."""
    if rep["mode"] == "off":
        return rep.get("text") or "mapping check: off"
    parts = []
    for m in rep["missing"]:
        if "T_K" in m:
            where = f"{_fmt_T(m['T_K'][0], recipe)}–{_fmt_T(m['T_K'][1], recipe)}"
        else:
            where = ", ".join(f"{k}={r[0]:.2f}–{r[1]:.2f}" for k, r in m.items() if k.startswith("X("))
        if m.get("repaired"):
            span = (f", {_fmt_T(m['added_T_K'][0], recipe)}–{_fmt_T(m['added_T_K'][1], recipe)}"
                    if m.get("added_T_K") else "")
            parts.append(f"{m['phases']} boundary missing around {where} - repaired ({m.get('added', 0)} line(s) "
                         f"added{span})")
        else:
            parts.append(f"{m['phases']} boundary missing around {where} - NOT repaired; read that area from "
                         f"point equilibria")
    for nd in rep.get("nodes_rejected", []):
        what = "invariant" if recipe.get("kind") == "binary" else "tie-triangle"
        at = f" at {_fmt_T(nd['T_K'], recipe)}" if what == "invariant" else ""
        pure = nd.get("at_pure_element_transformation")
        note = ""
        if pure:
            x = pure[[k for k in pure if k.startswith("X(")][0]]
            note = (f"; it sat on pure {recipe['elements'][0 if x < 0.5 else 1].capitalize()}'s "
                    f"{pure['phases'][0]}/{pure['phases'][1]} transition at the edge")
        parts.append(f"spurious {what}{at} ({nd['phases']}) rejected - not an equilibrium (equilibrium there: "
                     f"{nd['equilibrium_there']}){note}")
    for nd in rep.get("nodes_added", []):
        what = "invariant" if "T_K" in nd and recipe.get("kind") == "binary" else "tie-triangle"
        at = f" at {_fmt_T(nd['T_K'], recipe)}" if what == "invariant" else ""
        span = (f", {_fmt_T(nd['added_T_K'][0], recipe)}–{_fmt_T(nd['added_T_K'][1], recipe)}"
                if nd.get("added_T_K") else "")
        parts.append(f"{nd['phases']} {what}{at} rejected by pycalphad but confirmed by equilibrium - added "
                     f"({nd.get('lines_added', 0)} line(s) traced from it{span})")
    minor = 0
    for t in rep["trimmed"]:
        if t.get("new"):
            continue                                        # part of a repair (in the JSON)
        if t["removed"] <= 1 and t["pieces"] >= 1:
            minor += 1                                      # one tie-line at an end: noted, not spelled out
            continue
        where = {"end": "ran past its end node", "start": "started past its node"}.get(
            t["at"][0] if len(t["at"]) == 1 else "", "had metastable tie-lines")
        parts.append(f"{t['phases']} trace {where} (equilibrium there: {t['equilibrium_there'] or '?'}) - "
                     f"{t['removed']} metastable tie-line(s) {'removed' if t['pieces'] == 0 else 'trimmed'}")
    if minor:
        parts.append(f"{minor} metastable end tie-line{'s' if minor > 1 else ''} trimmed")
    # duplicates worth a mention: trimmed or partial traces (plain repeats of a whole line are only in the JSON)
    n_dup = sum(1 for d in rep["duplicates"] if (d.get("trimmed") or d.get("partial")) and not d.get("new"))
    dup = f"{n_dup} duplicate trace{'s' if n_dup > 1 else ''} dropped" if n_dup else ""
    if not rep.get("complete", True):
        parts.append("check incomplete (time limit)")
    if not parts:
        return (f"mapping check: no problems found ({rep['lines_checked']} lines checked against "
                f"{rep.get('equilibrium_calls', rep['tests'])} point equilibria" + (f"; {dup}" if dup else "") + ")")
    text = "mapping check: " + "; ".join(parts + ([dup] if dup else []))
    if len(text) > 700:
        text = text[:697] + "..."
    return text


def report_json(rep: Dict) -> Dict:
    """The report without the grid (for the tool's JSON)."""
    return {k: val for k, val in rep.items() if k not in ("grid", "grid_lim", "text")}


# ---------------------------------------------------------------------------------------------------------
# pycalphad's binplot / ternplot with the check between mapping and plotting
# ---------------------------------------------------------------------------------------------------------
def recipe_from_conditions(components, phases, conditions) -> Dict:
    """The recipe (see check_and_repair) of a binary T-x or ternary isotherm mapping from its conditions."""
    from pycalphad import variables as v
    els = sorted(str(c).upper() for c in components if str(c).upper() != "VA")
    xs = [(k, val) for k, val in conditions.items() if isinstance(k, v.X) and len(np.atleast_1d(val)) > 1]
    axis = [str(k.species.name).upper() for k, _ in xs]
    bal = [e for e in els if e not in axis]
    rec = {"elements": bal[:1] + axis, "phases": list(phases), "P": float(conditions.get(v.P, 101325.0)),
           "comp": {"scale": 1.0, "weight": False}}
    if len(axis) == 1:
        T = conditions[v.T]
        rec.update(kind="binary", T_range=[float(T[0]), float(T[1])], tmap=[1.0, 0.0],
                   x_range=[float(xs[0][1][0]), float(xs[0][1][1])])
    else:
        rec.update(kind="ternary", T_K=float(np.squeeze(conditions[v.T])))
    return rec


def checked_binplot(database, components, phases, conditions, plot_kwargs=None, check_mapping: str = "auto",
                    time_limit: float = 600.0):
    """pycalphad's binplot (BinaryStrategy + plot_binary) with check_and_repair between mapping and plotting.
    Returns (ax, strategy, check report)."""
    from pycalphad.mapping import BinaryStrategy, plot_binary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strategy = BinaryStrategy(database, components, phases, conditions)
    rep = map_and_check(strategy, database, recipe_from_conditions(components, phases, conditions), check_mapping,
                        time_limit, grid_size("binary", labels=False))
    plot_kwargs = dict(plot_kwargs or {})
    gridlines = plot_kwargs.pop("gridlines", False)
    ax = plot_binary(strategy, **plot_kwargs)
    ax.grid(gridlines)
    return ax, strategy, rep


def checked_ternplot(database, components, phases, conditions, x=None, y=None, check_mapping: str = "auto",
                     time_limit: float = 600.0, **plot_kwargs):
    """pycalphad's ternplot (TernaryStrategy + plot_ternary) with check_and_repair between mapping and plotting.
    Returns (ax, strategy, check report)."""
    from pycalphad.mapping import TernaryStrategy, plot_ternary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strategy = TernaryStrategy(database, components, phases, conditions)
    rep = map_and_check(strategy, database, recipe_from_conditions(components, phases, conditions), check_mapping,
                        time_limit, grid_size("ternary", labels=False))
    ax = plot_ternary(strategy, x, y, **plot_kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax, strategy, rep
