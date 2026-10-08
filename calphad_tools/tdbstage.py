"""Building blocks for ``evaluate_compatibility`` and ``staged_merge``: version views of a TDB, the subsystem
"does it change?" comparator, and the text-level TDB edits that build the intermediate databases.

* :class:`StageView` - a parsed TDB with every parameter's FUNCTIONs inlined (a tdbtables ``Subsystem`` over the
  database's own elements, or a subset), plus the element reference (SER) of each element.
* :func:`restricted_model` / :func:`model_endmembers` - a phase's sublattice model restricted to an element
  subsystem (vacancy-only sublattices dropped), and the end-members it has there.
* :func:`diff_versions` - for two versions of a database (e.g. B and B with A's unaries), which binaries /
  ternaries change: a subsystem is *provably unchanged* when every parameter whose constituents lie inside it is
  numerically equal (functions inlined) and every phase model restricted to it is identical (site ratios,
  constituents per sublattice, magnetic and order-disorder settings).
* TDB edits: :func:`tdb_to_text` (writes a parsed TDB back), :func:`restrict_tdb` (element subset, optionally only
  unary / unary+binary parameters), :func:`substitute_ser` (B1), :func:`substitute_unaries` (B2) and
  :func:`plan_versions` helpers that go through merge.plan_merge (B3, Q, A1, A2).

Only the standard library, numpy, symengine and calphad_tools.tdbparse / pwinline / tdbtables / merge are used, so
show_code can copy this module into a standalone script.  Names are prefixed so they don't clash with the
copied modules.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .merge import _build_raw_name, _split_raw_name, plan_merge, render_tdb
from .pwinline import MissingRef, evaluate, inline, scale
from .tdbparse import (TDB, Function, Parameter, Piecewise, parse_param_name, parse_tdb, scale_piecewise,
                       wrap_tdb_line)
from .tdbtables import Subsystem, key_constituents, make_key, pw_segments, row_sort_key

STAGE_GRID = (298.15, 6000.0)          # temperature range of SER (SGTE unary) comparisons (K)
PARAM_GRID = (298.15, 3000.0)          # other parameters (as compare_system / merge_tdbs): no 6000 K extrapolation
STAGE_RTOL = 1e-4                      # relative tolerance (of the largest |value| over the range), as merge_tdbs
STAGE_ATOL = {"G": 1e-2}               # absolute floor, J/mol for G/L terms; others (TC, BMAGN ...) 1e-6
SKIP_COMMANDS = ("ASSESS", "VERSION", "DATABASE", "REFERENCE_F")


def stage_sysname(elements) -> str:
    """'CR-FE' -> 'Cr-Fe' (sorted)."""
    return "-".join(e.capitalize() for e in sorted(elements))


def stage_subsystems(elements: Sequence[str], sizes=(2, 3)) -> List[Tuple[str, ...]]:
    els = sorted({e.upper() for e in elements})
    return [c for n in sizes for c in itertools.combinations(els, n)]


# =============================================================================================== views
@dataclass
class SerRef:
    element: str
    kind: str                 # "param" (G of the element in its ELEMENT reference phase) | "function" (GHSER<EL>)
    what: str                 # row key or function name
    function: str             # function the SER is written with (GHSERFE), "" if written out
    phase: str                # reference phase (canonical name), "" for a bare function
    segs: list


class StageView(Subsystem):
    """A parsed TDB (restricted to ``elements``, default all of its own) with every parameter inlined."""

    def __init__(self, tdb: TDB, label: str, elements: Optional[Sequence[str]] = None,
                 rename: Optional[Dict[str, str]] = None):
        els = [e.upper() for e in (elements or tdb.real_elements) if e.upper() in tdb.elements]
        super().__init__(tdb, label, tdb.path, els, rename or {})
        self.own_elements = set(els)
        self._row_els: Dict[str, frozenset] = {}

    # -------------------------------------------------------------- rows
    def row_elements(self, key: str) -> frozenset:
        if key not in self._row_els:
            els = set()
            for sl in key_constituents(key):
                for s in sl:
                    if s not in ("VA", "*", ""):
                        els |= self.species_elements(s)
            self._row_els[key] = frozenset(e for e in els if e not in ("VA", "*"))
        return self._row_els[key]

    def key_for(self, ptype: str, phase: str, endmember: Sequence[str], order: int = 0) -> str:
        return make_key(ptype, phase, tuple((s,) for s in endmember), order)

    def ser(self, el: str) -> Optional[SerRef]:
        """The element's SER Gibbs energy: G of the element in its ELEMENT reference phase, else GHSER<EL>."""
        e = self.tdb.elements.get(el)
        if e is not None and e.ref_phase:
            cn = self.canon_of(e.ref_phase)
            if cn:
                pv = self.phases[cn]
                red = tuple((el,) for i in range(pv.n_sub) if i not in pv.va_only)
                key = make_key("G", cn, red, 0)
                row = self.rows.get(key)
                if row is not None and row.segs is not None:
                    fn = ""
                    rng = row.raw_pw.ranges if row.raw_pw is not None else []
                    m = re.fullmatch(r"\+?([A-Z][A-Z0-9_]*)#?", rng[0][2].upper()) if len(rng) == 1 else None
                    if m and m.group(1) in self.tdb.functions:
                        fn = m.group(1)
                    return SerRef(el, "param", key, fn, cn, row.segs)
        for cand in (f"GHSER{el}", f"GHSER{el}{el}"):
            if cand in self.tdb.functions:
                try:
                    return SerRef(el, "function", cand, cand, "", self.fn(cand))
                except (MissingRef, ValueError):
                    return None
        return None

    def segs_in(self, key: str, other: "StageView") -> Optional[list]:
        """This view's parameter ``key`` evaluated with ``other``'s FUNCTIONs where both define a name (what an
        A-priority merge does to B's parameters); None if it cannot be inlined."""
        r = self.rows.get(key)
        if r is None or r.raw_pw is None:
            return None

        def resolve(name, _stack=()):
            if name in other.tdb.functions:
                return other.fn(name)
            return self.fn(name)
        try:
            return inline(pw_segments(r.raw_pw, r.sign), resolve, set())
        except (MissingRef, ValueError, ZeroDivisionError, RecursionError):
            return None


def stage_renames(tdb: TDB, renames: Dict[str, str]) -> Dict[str, str]:
    """The part of ``renames`` ({old: new}, from a merge's 'rename' decision) that applies to ``tdb``: phases it still
    has under the old name and not under the new one (B's own versions, not the merged file)."""
    return {o: n for o, n in (renames or {}).items() if o in tdb.phases and n not in tdb.phases}


def stage_view_of(path_or_tdb, label: str, elements=None, rename=None) -> StageView:
    tdb = path_or_tdb if isinstance(path_or_tdb, TDB) else parse_tdb(str(path_or_tdb))
    return StageView(tdb, label, elements, rename)


# =============================================================================================== numbers
def compare_segs(a, b, rtol: float = STAGE_RTOL, atol: float = 1e-2,
                 trange: Tuple[float, float] = PARAM_GRID, n: int = 241):
    """(equal, max |a-b|, T at the max) over the common temperature range."""
    if not a or not b:
        return False, math.nan, math.nan
    lo, hi = max(a[0][0], b[0][0], trange[0]), min(a[-1][1], b[-1][1], trange[1])
    g = np.linspace(lo, hi, n) if lo < hi else np.asarray([0.5 * (lo + hi)])
    va, vb = evaluate(a, g)[0], evaluate(b, g)[0]
    m = np.isfinite(va) & np.isfinite(vb)
    if not m.any():
        return False, math.nan, math.nan
    d = np.abs(va[m] - vb[m])
    i = int(np.argmax(d))
    big = float(max(np.max(np.abs(va[m])), np.max(np.abs(vb[m]))))
    return bool(d[i] <= max(atol, rtol * big)), float(d[i]), float(g[m][i])


def stage_atol(ptype: str) -> float:
    return STAGE_ATOL.get(ptype, 1e-6)


def fmt_energy(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "?"
    return f"{x / 1000:.3g} kJ/mol" if abs(x) >= 100 else f"{x:.3g} J/mol"


def fmt_delta(x: float, ptype: str = "G") -> str:
    if ptype == "G":
        return fmt_energy(x)
    return "?" if x is None or not math.isfinite(x) else f"{x:.4g}"


# =============================================================================================== phase models
@dataclass(frozen=True)
class PhaseModel:
    ratios: Tuple[float, ...]                         # vacancy-only sublattices dropped
    constituents: Tuple[Tuple[str, ...], ...]         # sorted per sublattice
    magnetic: Optional[Tuple[float, float]] = None
    dis_part: str = ""

    def text(self) -> str:
        return "".join(f"({','.join(c)}){r:g}" for c, r in zip(self.constituents, self.ratios))

    def extras(self) -> str:
        x = []
        if self.magnetic:
            x.append(f"magnetic {self.magnetic[0]:g}/{self.magnetic[1]:g}")
        if self.dis_part:
            x.append(f"DIS_PART {self.dis_part}")
        return ", ".join(x)


def restricted_model(view: StageView, phase: str, elements) -> Optional[PhaseModel]:
    """The phase's model with only species inside ``elements`` (+VA); None if the phase cannot exist there."""
    pv = view.phases.get(phase)
    if pv is None:
        return None
    E = {e.upper() for e in elements} | {"VA"}
    cons = [[s for s in sl if s == "VA" or view.species_elements(s) <= E] for sl in pv.constituents]
    if not cons or any(not c for c in cons) or all(s == "VA" for c in cons for s in c):
        return None
    keep = [i for i, c in enumerate(cons) if c != ["VA"]]
    return PhaseModel(tuple(round(pv.site_ratios[i], 6) for i in keep), tuple(tuple(sorted(cons[i])) for i in keep),
                      pv.magnetic, pv.dis_part or "")


def full_model(view: StageView, phase: str) -> Optional[PhaseModel]:
    return restricted_model(view, phase, view.own_elements)


def model_endmembers(m: Optional[PhaseModel]) -> Set[Tuple[str, ...]]:
    if m is None:
        return set()
    return set(itertools.product(*m.constituents))


def union_model(a: PhaseModel, b: PhaseModel) -> Optional[PhaseModel]:
    """A's model with B's constituents added (None if the site ratios differ)."""
    if a.ratios != b.ratios:
        return None
    return PhaseModel(a.ratios, tuple(tuple(sorted(set(x) | set(y))) for x, y in zip(a.constituents, b.constituents)),
                      a.magnetic, a.dis_part)


def endmember_elements(view: StageView, em: Sequence[str]) -> frozenset:
    els = set()
    for s in em:
        if s not in ("VA", "*"):
            els |= view.species_elements(s)
    return frozenset(e for e in els if e != "VA")


# =============================================================================================== version diff
@dataclass
class ParamChange:
    key: str
    label: str
    phase: str
    ptype: str
    elements: frozenset
    kind: str                    # "value" | "added" | "removed"
    max_diff: float = math.nan
    T_at: float = math.nan
    old: str = ""
    new: str = ""

    def text(self) -> str:
        if self.kind == "value":
            return f"{self.label} Δ{fmt_delta(self.max_diff, self.ptype)}"
        return f"{self.label} {self.kind}"


@dataclass
class ModelChange:
    phase: str
    kind: str                    # "phase_added" | "phase_removed" | "model" | "magnetic" | "ordering"
    old: str = ""
    new: str = ""
    new_endmembers: List[str] = field(default_factory=list)
    lost_endmembers: List[str] = field(default_factory=list)
    missing_g: List[str] = field(default_factory=list)      # new end-members without a G parameter (pycalphad: 0)

    def text(self) -> str:
        if self.kind == "phase_added":
            s = f"{self.phase} added {self.new}"
        elif self.kind == "phase_removed":
            s = f"{self.phase} removed"
        else:
            s = f"{self.phase} {self.old} → {self.new}"
        if self.new_endmembers and self.kind != "phase_added":
            s += (f" (+{', '.join(self.new_endmembers[:3])}"
                  + (f" and {len(self.new_endmembers) - 3} more" if len(self.new_endmembers) > 3 else "") + ")")
        if self.missing_g:
            s += f" [no G for {', '.join(self.missing_g[:3])}: pycalphad uses 0]"
        return s


@dataclass
class SubsystemDiff:
    elements: Tuple[str, ...]
    params: List[ParamChange] = field(default_factory=list)
    models: List[ModelChange] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)          # e.g. exact proportional rescales (not changes)

    @property
    def name(self) -> str:
        return stage_sysname(self.elements)

    @property
    def changed(self) -> bool:
        return bool(self.params or self.models)

    def reasons(self, n: int = 3) -> str:
        extra = "".join(f"; {x}" for x in self.notes)
        if not self.changed:
            return "no change (all parameters numerically equal, same phase models)" + extra
        parts = [m.text() for m in self.models[:n]]
        if len(self.models) > n:
            parts.append(f"+{len(self.models) - n} more model changes")
        direct = [p for p in self.params if len(p.elements) == len(self.elements)]
        inherited = [p for p in self.params if len(p.elements) < len(self.elements)]
        if direct:
            ex = ", ".join(p.text() for p in sorted(direct, key=_pc_rank)[:n])
            parts.append(f"{len(direct)} {self.name} parameter(s) differ/added/removed ({ex}"
                         + (f", +{len(direct) - n} more)" if len(direct) > n else ")"))
        if inherited:
            subs: Dict[str, int] = {}
            for p in inherited:
                k = stage_sysname(p.elements) if p.elements else "VA"
                subs[k] = subs.get(k, 0) + 1
            ex = ", ".join(p.text() for p in sorted(inherited, key=_pc_rank)[:2])
            parts.append(f"{len(inherited)} from lower-order systems ({', '.join(f'{k}:{v}' for k, v in subs.items())}"
                         f"; e.g. {ex})")
        return "; ".join(parts) + extra


def _pc_rank(p: ParamChange):
    return (p.kind != "value", -(p.max_diff if math.isfinite(p.max_diff) else 0.0), p.key)


@dataclass
class VersionDiff:
    old: str
    new: str
    params: List[ParamChange]                    # every parameter change (any subsystem of the views)
    subsystems: List[SubsystemDiff]
    notes: List[str] = field(default_factory=list)

    @property
    def changed(self) -> List[SubsystemDiff]:
        return [s for s in self.subsystems if s.changed]

    def get(self, elements) -> Optional[SubsystemDiff]:
        want = tuple(sorted(e.upper() for e in elements))
        return next((s for s in self.subsystems if tuple(sorted(s.elements)) == want), None)


def _written(view: StageView, key: str) -> str:
    try:
        return view.written(key).text
    except Exception:  # noqa: BLE001 - display only
        r = view.rows.get(key)
        return r.raw_text if r is not None else ""


def phase_rescales(X: StageView, Y: StageView) -> Dict[str, float]:
    """Phases whose site ratios differ only by a factor k (Y = k*X, vacancy-only sublattices ignored) and that have
    no magnetic parameters in X and no order-disorder link: comparing Y's G/L terms with k*X's is then exact (GM per
    mole of atoms is unchanged by such a rescale)."""
    out = {}
    for ph in set(X.phases) & set(Y.phases):
        px, py = X.phases[ph], Y.phases[ph]
        if px.dis_part or py.dis_part or px.disordered_of or py.disordered_of:
            continue
        rx, ry = px.reduced_ratios(), py.reduced_ratios()
        if len(rx) != len(ry) or rx == ry or not rx:
            continue
        k = sum(ry) / sum(rx)
        if abs(k - 1.0) < 1e-9 or any(abs(b - a * k) > 1e-3 * max(ry) for a, b in zip(rx, ry)):
            continue
        if any(r.phase == ph and r.ptype in MAGNETIC_PTYPES for r in X.rows.values()):
            continue
        out[ph] = k
    return out


def param_changes(X: StageView, Y: StageView, rtol: float = STAGE_RTOL,
                  keys: Optional[Sequence[str]] = None, scales: Optional[Dict[str, float]] = None) -> List[ParamChange]:
    """Parameters that differ from X to Y; for a phase in ``scales`` (exact proportional rescale, see
    phase_rescales) Y's G/L terms are compared with k * X's."""
    scales = scales or {}
    out = []
    for k in (keys if keys is not None else sorted(set(X.rows) | set(Y.rows))):
        rx, ry = X.rows.get(k), Y.rows.get(k)
        r = rx or ry
        els = (X if rx is not None else Y).row_elements(k)
        if rx is None or ry is None:
            out.append(ParamChange(k, r.label, r.phase, r.ptype, els, "added" if rx is None else "removed",
                                   old=_written(X, k) if rx else "", new=_written(Y, k) if ry else ""))
            continue
        if rx.segs is None or ry.segs is None:
            if re.sub(r"\s", "", rx.raw_text.upper()) != re.sub(r"\s", "", ry.raw_text.upper()) or rx.sign != ry.sign:
                out.append(ParamChange(k, r.label, r.phase, r.ptype, els, "value", old=_written(X, k),
                                       new=_written(Y, k)))
            continue
        xs = scale(rx.segs, scales[r.phase]) if r.phase in scales and r.ptype in SCALED_PTYPES else rx.segs
        ok, d, t = compare_segs(xs, ry.segs, rtol, stage_atol(r.ptype))
        if not ok:
            out.append(ParamChange(k, r.label, r.phase, r.ptype, els, "value", d, t, _written(X, k), _written(Y, k)))
    return out


def model_change(X: StageView, Y: StageView, phase: str, elements,
                 k: Optional[float] = None) -> Optional[ModelChange]:
    mx, my = restricted_model(X, phase, elements), restricted_model(Y, phase, elements)
    if mx is None and my is None:
        return None
    if k is not None and mx is not None and my is not None and len(mx.ratios) == len(my.ratios) and all(
            abs(b - a * k) <= 1e-3 * max(my.ratios) for a, b in zip(mx.ratios, my.ratios)):
        mx = PhaseModel(my.ratios, mx.constituents, mx.magnetic, mx.dis_part)     # exact rescale: same model
    ems_x, ems_y = model_endmembers(mx), model_endmembers(my)
    new = sorted(ems_y - ems_x)
    missing = [":".join(e) for e in new if Y.key_for("G", phase, e) not in Y.rows]
    if mx is None:
        return ModelChange(phase, "phase_added", "", my.text(), [":".join(e) for e in new], [], missing)
    if my is None:
        return ModelChange(phase, "phase_removed", mx.text(), "")
    if mx.ratios != my.ratios or mx.constituents != my.constituents:
        return ModelChange(phase, "model", mx.text(), my.text(), [":".join(e) for e in new],
                           [":".join(e) for e in sorted(ems_x - ems_y)], missing)
    if mx.magnetic != my.magnetic and any(
            r.phase == phase and r.ptype in MAGNETIC_PTYPES and V.row_elements(key) <= set(elements)
            for V in (X, Y) for key, r in V.rows.items()):
        # a magnetic type definition only matters where the subsystem has TC/BMAGN parameters for the phase
        f = lambda m: f"magnetic {m[0]:g}/{m[1]:g}" if m else "non-magnetic"   # noqa: E731
        return ModelChange(phase, "magnetic", f(mx.magnetic), f(my.magnetic))
    if mx.dis_part != my.dis_part:
        return ModelChange(phase, "ordering", f"DIS_PART {mx.dis_part or '-'}", f"DIS_PART {my.dis_part or '-'}")
    return None


def diff_versions(X: StageView, Y: StageView, subsystems: Sequence[Sequence[str]],
                  rtol: float = STAGE_RTOL) -> VersionDiff:
    """Which of ``subsystems`` change from version X to version Y (see module docstring for the rule)."""
    scales = phase_rescales(X, Y)
    changes = param_changes(X, Y, rtol, scales=scales)
    notes = {ph: (f"{ph}: {X.label} {X.phases[ph].ratios_text()} -> {Y.label} {Y.phases[ph].ratios_text()} is a "
                  f"proportional rescale (x{k:.6g}) with G/L x{k:.6g} and no magnetic terms in {X.label}: exact, "
                  f"not a change")
             for ph, k in scales.items()
             if not any(c.phase == ph and c.kind == "value" for c in changes)}
    phases = sorted(set(X.phases) | set(Y.phases))
    subs = []
    for S in subsystems:
        S = tuple(sorted(e.upper() for e in S))
        Sset = set(S)
        present = {ph for ph in phases if restricted_model(X, ph, Sset) or restricted_model(Y, ph, Sset)}
        sd = SubsystemDiff(S)
        for ph in sorted(present, key=lambda p: (p != "LIQUID", p)):
            mc = model_change(X, Y, ph, Sset, scales.get(ph))
            if mc is not None:
                sd.models.append(mc)
            if ph in notes:
                sd.notes.append(notes[ph])
        sd.params = [c for c in changes if c.elements <= Sset and c.phase in present]
        subs.append(sd)
    return VersionDiff(X.label, Y.label, changes, subs, list(notes.values()))


# =============================================================================================== TDB text
def tdb_to_text(db: TDB, header: Sequence[str] = ()) -> str:
    """Write a parsed TDB back (type-definition codes, expressions and reference keys as parsed)."""
    L = [f"$ {h}" for h in header]
    L += [c[:200] for c in db.header_comments[:60]]
    L.append("$")
    has_tl = False
    cmds = []
    for cmd in db.other_commands:
        kw = cmd.split()[0].upper().replace("-", "_")
        if kw.startswith(SKIP_COMMANDS):
            continue
        if kw.startswith("TEMP"):
            has_tl = True
            L.append(wrap_tdb_line(f" {cmd} !"))
            continue
        cmds.append(cmd)
    if not has_tl:
        L.append(f" TEMPERATURE_LIMITS {db.temp_limits[0]:g} {db.temp_limits[1]:g} !")
    for nm in sorted(db.elements, key=lambda e: (e not in ("/-", "VA"), e)):
        L.append(db.elements[nm].to_tdb())
    for sp in db.species.values():
        L.append(sp.to_tdb())
    L.append("$")
    for fn in db.functions.values():
        L.append(wrap_tdb_line(fn.to_tdb()))
    L.append("$")
    for td in db.typedefs.values():
        L.append(wrap_tdb_line(td.to_tdb()))
    for cmd in cmds:
        L.append(wrap_tdb_line(f" {cmd} !"))
    L.append("$")
    for ph in db.phases.values():
        L += [wrap_tdb_line(x) for x in ph.to_tdb()]
    L.append("$")
    for p in db.parameters:
        L.append(wrap_tdb_line(p.to_tdb()))
    for r in db.references:
        L.append("$")
        L.append(r.rstrip() + " !")
    return "\n".join(L) + "\n"


def write_tdb(db: TDB, path: str, header: Sequence[str] = ()) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(tdb_to_text(db, header))
    return str(path)


def _species_els(db: TDB, sp: str) -> Set[str]:
    s = sp.upper().strip("%")
    if s in ("VA", "*") or s in db.elements:
        return {s}
    spc = db.species.get(s)
    if spc is not None:
        els = set(re.findall(r"([A-Z][A-Z]?)\d+\.?\d*", spc.formula.split("/")[0].upper()))
        if els:
            return els
    return {re.sub(r"[+-]\d*$", "", s)}


def param_element_set(db: TDB, p: Parameter) -> Set[str]:
    els = set()
    for sl in p.key.constituents:
        for s in sl:
            els |= _species_els(db, s)
    return els - {"VA", "*"}


def restrict_tdb(db: TDB, elements: Sequence[str], max_elements: Optional[int] = None,
                 constituents_from_params: bool = False, phases: Optional[Sequence[str]] = None) -> TDB:
    """Copy of ``db`` with only ``elements`` (+VA): species, phases (constituents restricted), parameters inside;
    ``max_elements`` keeps only parameters with at most that many elements (1: unaries, 2: + binaries);
    ``constituents_from_params`` keeps only the constituents those parameters use (per sublattice)."""
    E = {e.upper() for e in elements} | {"VA"}
    out = copy.deepcopy(db)
    ok_sp = lambda s: s == "*" or _species_els(db, s) <= E   # noqa: E731
    out.elements = {k: v for k, v in db.elements.items() if k in E or k == "/-"}
    out.species = {k: v for k, v in db.species.items() if _species_els(db, k) <= E}
    want_ph = {p.upper() for p in phases} if phases else None
    params = []
    for p in db.parameters:
        if not all(ok_sp(s) for sl in p.key.constituents for s in sl):
            continue
        if want_ph is not None and p.key.phase not in want_ph:
            continue
        if max_elements is not None and len(param_element_set(db, p)) > max_elements:
            continue
        params.append(copy.deepcopy(p))
    used: Dict[str, List[Set[str]]] = {}
    for p in params:
        ph = db.phases.get(p.key.phase)
        if ph is None:
            continue
        u = used.setdefault(p.key.phase, [set() for _ in ph.constituents])
        if len(p.key.constituents) == len(u):
            for i, sl in enumerate(p.key.constituents):
                u[i] |= {s for s in sl if s != "*"}
    out.phases = {}
    for nm, ph in db.phases.items():
        if want_ph is not None and nm not in want_ph:
            continue
        ph2 = copy.deepcopy(ph)
        cons = [[s for s in sl if ok_sp(s)] for sl in ph.constituents]
        if constituents_from_params:
            if nm not in used:
                continue
            cons = [[s for s in c if s in u or c == ["VA"]] for c, u in zip(cons, used[nm])]
        if not cons or any(not c for c in cons) or all(s == "VA" for c in cons for s in c):
            continue
        ph2.constituents = cons
        ph2.majors = [[s for s in m if s in c] for m, c in zip(ph.majors or [[]] * len(cons), cons)]
        out.phases[nm] = ph2
    out.parameters = [p for p in params if p.key.phase in out.phases
                      and len(p.key.constituents) == len(out.phases[p.key.phase].constituents)
                      and all(s == "*" or s in c for sl, c in zip(p.key.constituents,
                                                                   out.phases[p.key.phase].constituents) for s in sl)]
    # type definitions whose DIS_PART partner was dropped
    for nm, ph in out.phases.items():
        codes = ""
        for c in ph.type_codes:
            td = db.typedefs.get(c)
            m = re.search(r"DIS_P\w*\s+([A-Z0-9_]+)", td.text.upper()) if td is not None else None
            if m and m.group(1) not in out.phases:
                continue
            codes += c
        ph.type_codes = codes
    out.other_commands = [c for c in db.other_commands if not re.search(r"(REJECT|RESTORE)", c, re.I)]
    out.warnings = []
    return out


def rename_phases(db: TDB, mapping: Dict[str, str]) -> List[str]:
    """Rename phases in place (parameters, type definitions, ELEMENT reference phases too).  Returns notes."""
    notes = []
    for old, new in (mapping or {}).items():
        old, new = old.upper(), new.upper()
        if old not in db.phases or old == new:
            continue
        if new in db.phases:
            raise ValueError(f"phase_map renames {old} to {new}, but {db.name} already has {new}")
        ph = db.phases.pop(old)
        ph.name = new
        db.phases[new] = ph
        for p in db.parameters:
            if p.key.phase == old:
                p.raw_name = re.sub(rf"^(\w+\(\s*){re.escape(old)}(?=[,:&])", rf"\g<1>{new}", p.raw_name, flags=re.I)
                p.key.phase = new
        for td in db.typedefs.values():
            td.text = re.sub(rf"(?<![A-Z0-9_]){re.escape(old)}(?![A-Z0-9_])", new, td.text, flags=re.I)
        for el in db.elements.values():
            if el.ref_phase.upper() == old:
                el.ref_phase = new
        notes.append(f"{old} -> {new}")
    return notes


# =============================================================================================== substitutions
def _neg_piecewise(pw: Piecewise, factor: float) -> Piecewise:
    """``pw`` with the sign ``factor`` (+1 / -1) applied range by range (tdbparse.scale_piecewise)."""
    return scale_piecewise(pw, -1.0 if factor < 0 else 1.0)


def _function_closure(db: TDB, names: Sequence[str]) -> List[str]:
    out: List[str] = []
    todo = list(names)
    while todo:
        n = todo.pop(0)
        if n in out or n not in db.functions:
            continue
        out.append(n)
        todo += db.functions[n].expr.symbols()
    return out


def copy_functions(dst: TDB, src: TDB, names: Sequence[str], log: List[str], why: str) -> None:
    """Give ``dst`` the source's definitions of ``names`` and everything they reference (A-priority: a function
    defined differently in ``dst`` is replaced, which also changes its other users there)."""
    for n in _function_closure(src, names):
        f = src.functions[n]
        if n in dst.functions:
            if dst.functions[n].expr.canonical() == f.expr.canonical():
                continue
            users = sum(n in p.expr.symbols() for p in dst.parameters) + sum(
                n in g.expr.symbols() for k, g in dst.functions.items() if k != n)
            log.append(f"FUNCTION {n} replaced by {src.name}'s ({why}; {users} other reference(s) in the file)")
        else:
            log.append(f"FUNCTION {n} added from {src.name} ({why})")
        dst.functions[n] = Function(n, copy.deepcopy(f.expr))


def _raw_for(view: StageView, key: str) -> str:
    """Raw parameter name in ``view``'s native phase/sublattice layout for a canonical key (constituents sorted,
    vacancy-only sublattices restored)."""
    ptype, rest = key.split("(", 1)
    inner = rest[: rest.rindex(";")]
    phase_part = inner.split(",", 1)[0]
    phase, _, diff = phase_part.partition("&")
    order = key.rsplit(";", 1)[1].rstrip(")")
    pv = view.phases[phase]
    red = iter(key_constituents(key))
    subls = []
    for i in range(pv.n_sub):
        subls.append("VA" if i in pv.va_only else ",".join(next(red)))
    d = f"&{diff}" if diff else ""
    return f"{ptype}({pv.native}{d},{':'.join(subls)};{order})"


def _param_by_raw(db: TDB, raw_name: str) -> Optional[Parameter]:
    hit = None
    for p in db.parameters:
        if p.raw_name == raw_name:
            hit = p          # last definition wins
    return hit


def put_parameter(dst: TDB, dst_view: StageView, src_view: StageView, key: str, log: List[str], why: str) -> str:
    """Set ``dst``'s parameter ``key`` to the source's (replace or add).  Returns 'replaced' | 'added' | reason."""
    rs = src_view.rows[key]
    rd = dst_view.rows.get(key)
    if rd is not None:
        p = _param_by_raw(dst, rd.raw_name)
        if p is None:
            return "not found"
        p.expr = _neg_piecewise(rs.raw_pw, rs.sign * rd.sign)
        log.append(f"{rd.label}: {why}")
        copy_functions(dst, src_view.tdb, rs.raw_pw.symbols(), log, f"used by {rs.label}")
        return "replaced"
    pv = dst_view.phases.get(rs.phase)
    if pv is None:
        return f"phase {rs.phase} not in {dst.name}"
    cons = key_constituents(key)
    keep = [i for i in range(pv.n_sub) if i not in pv.va_only]
    if len(keep) != len(cons):
        return f"{rs.phase} model differs"
    for i, sl in zip(keep, cons):
        if any(s not in pv.constituents[i] and s != "*" for s in sl):
            return f"{rs.phase} has no {'/'.join(sl)} on sublattice {i + 1} in {dst.name}"
    raw = _raw_for(dst_view, key)
    k2, sign, _ = parse_param_name(raw)
    dst.parameters.append(Parameter(raw, k2, sign, None, _neg_piecewise(rs.raw_pw, rs.sign * sign)))
    log.append(f"{rs.label}: added ({why})")
    copy_functions(dst, src_view.tdb, rs.raw_pw.symbols(), log, f"used by {rs.label}")
    return "added"


def substitute_ser(B: TDB, VA: StageView, VB: StageView, elements: Sequence[str],
                   rtol: float = STAGE_RTOL) -> Tuple[TDB, List[str]]:
    """B1: B with A's SER (SGTE unary) Gibbs energies of ``elements``.  Where B writes its SER as a FUNCTION
    (GHSERFE, or G(BCC_A2,FE:VA) = GHSERFE) that FUNCTION gets A's definition, so every B parameter written
    relative to it follows; a SER written out in the parameter itself is replaced in place."""
    out = copy.deepcopy(B)
    log: List[str] = []
    for el in sorted(elements):
        sa, sb = VA.ser(el), VB.ser(el)
        if sa is None or sb is None:
            log.append(f"{el}: no SER reference found in {'A' if sa is None else 'B'}; not substituted")
            continue
        ok, d, _ = compare_segs(sa.segs, sb.segs, rtol, 1e-2, STAGE_GRID)
        if ok:
            log.append(f"{el}: SER already equal (max |Δ| {fmt_energy(d)})")
            continue
        # A's SER as an expression: the function, or the parameter's expression
        if sa.function:
            a_pw, a_funcs = VA.tdb.functions[sa.function].expr, [sa.function]
        else:
            a_pw = VA.rows[sa.what].raw_pw if sa.kind == "param" else VA.tdb.functions[sa.what].expr
            a_funcs = []
        if sb.function:
            target = sb.function
            if a_funcs and a_funcs[0] == target:
                copy_functions(out, VA.tdb, [target], log, f"SER of {el}")
            else:
                out.functions[target] = Function(target, copy.deepcopy(a_pw))
                copy_functions(out, VA.tdb, a_pw.symbols(), log, f"used by SER of {el}")
                log.append(f"FUNCTION {target} (SER of {el}) set to A's {sa.function or sa.what}")
        else:
            rd = VB.rows[sb.what]
            p = _param_by_raw(out, rd.raw_name)
            if p is None:
                log.append(f"{el}: B's SER parameter {rd.raw_name} not found")
                continue
            p.expr = copy.deepcopy(a_pw)
            copy_functions(out, VA.tdb, a_pw.symbols(), log, f"used by SER of {el}")
            log.append(f"{rd.label} (SER of {el}) set to A's")
        log.append(f"{el}: SER differed by up to {fmt_energy(d)}; now A's")
    return out, log


def substitute_unaries(B: TDB, VA: StageView, VB: StageView, elements: Sequence[str],
                       rtol: float = STAGE_RTOL) -> Tuple[TDB, List[str]]:
    """B2 (from B1): every unary parameter of A (G, TC, BMAGN, ... of one element in one phase) for the shared
    elements, in the phases both databases have with the same site ratios; B parameters A lacks are kept."""
    out = copy.deepcopy(B)
    log: List[str] = []
    E = {e.upper() for e in elements}
    for k in sorted(VA.rows, key=lambda k: row_sort_key(k, VA.rows[k].ptype)):
        ra = VA.rows[k]
        els = VA.row_elements(k)
        if len(els) != 1 or not els <= E or ra.phase not in VB.phases:
            continue
        ma, mb = full_model(VA, ra.phase), full_model(VB, ra.phase)
        if ma is None or mb is None or ma.ratios != mb.ratios:
            log.append(f"{ra.label}: skipped ({ra.phase} site ratios differ)")
            continue
        rb = VB.rows.get(k)
        if rb is not None and ra.segs is not None and rb.segs is not None:
            ok, _, _ = compare_segs(ra.segs, rb.segs, rtol, stage_atol(ra.ptype))
            if ok:
                continue
        res = put_parameter(out, VB, VA, k, log, "A's value")
        if res not in ("replaced", "added"):
            log.append(f"{ra.label}: not substituted ({res})")
    return out, log


# =============================================================================================== merges
def alias_separate(paths: Sequence[str], decisions: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Decisions that keep phases with alias names (SIGMA / SIGMA_D8B ...) separate unless already decided:
    names are only unified through an explicit phase_map."""
    d = dict(decisions or {})
    plan = plan_merge(list(paths), d, "ask")
    for c in plan.conflicts:
        if c.kind == "phase_alias" and c.id not in d:
            d[c.id] = "separate"
    return d


def auto_decisions(plan, shared: Sequence[str], max_elements: int = 2) -> Dict[str, str]:
    """Conflicts the staged procedure resolves toward the base (db1): element data and FUNCTIONs of the shared
    elements, unary and binary parameters of shared elements.  Ternary+ parameters, phase models, type
    definitions and default commands are left to the user."""
    E = {e.upper() for e in shared}
    out = {}
    for c in plan.conflicts:
        keys = [o.key for o in c.options]
        if "db1" not in keys:
            continue
        if c.kind == "element" and c.id.split(":", 1)[1] in E:
            out[c.id] = "db1"
        elif c.kind == "function":
            out[c.id] = "db1"
        elif c.kind == "parameter" and c.group and c.group.startswith("system:"):
            els = [e for e in c.group.split(":", 1)[1].split("-") if e and e != "?"]
            if els and set(els) <= E and len(els) <= max_elements:
                out[c.group] = "db1"
    return out


def merged_text(paths: Sequence[str], decisions: Dict[str, str], policy: str = "first"):
    """(plan, text or None if decisions are pending)."""
    plan = plan_merge(list(paths), decisions, policy)
    if plan.pending:
        return plan, None
    return plan, render_tdb(plan)


# =============================================================================================== normalisation
# B (the database to add) adopts A's (the base's) conventions before comparing / merging: phase names, sublattice
# order and formula-unit size.
NAME_TOKENS = {"BCC": "bcc", "A2": "bcc", "FCC": "fcc", "A1": "fcc", "HCP": "hcp", "A3": "hcp", "SIGMA": "sigma",
               "D8B": "sigma", "SIG": "sigma", "LIQUID": "liq", "LIQ": "liq", "C14": "c14", "C15": "c15",
               "C36": "c36", "MU": "mu", "D85": "mu", "CHI": "chi", "A12": "chi", "CBCC": "chi", "CUB": "cub",
               "A13": "cub", "B2": "b2", "L12": "l12", "L1_2": "l12", "DIAMOND": "a4", "A4": "a4", "GRAPHITE": "gra"}
NAME_NOISE = {"LAVES", "PHASE", "SGTE", "SER"}
MAGNETIC_PTYPES = ("TC", "BMAGN", "NT", "BM")
SCALED_PTYPES = ("G", "L")          # per-formula-unit energy terms (G and L are the same parameter type)


class SublatticeMapError(ValueError):
    """B's sublattices cannot be matched to A's unambiguously."""


def phase_tokens(name: str) -> frozenset:
    """Structure tokens of a phase name: LAVES_C15 / C15_LAVES / C15 -> {c15}; BCC_A2 / A2_BCC -> {bcc}."""
    toks = [t for t in re.split(r"[_#\-]", name.upper()) if t and not t.isdigit()]
    return frozenset(NAME_TOKENS.get(t, t.lower()) for t in toks if t not in NAME_NOISE)


@dataclass
class NormNote:
    phase: str
    action: str          # renamed | reordered | rescaled | not_rescaled | incompatible | unmapped | candidate | kept
    detail: str

    def text(self) -> str:
        return f"{self.phase}: {self.action} {self.detail}".strip()


def _va_only(ph) -> List[int]:
    return [i for i, sl in enumerate(ph.constituents) if sl == ["VA"]]


def _subl_text(ph) -> str:
    return "".join(f"({','.join(c)}){r:g}" for c, r in zip(ph.constituents, ph.site_ratios))


def sublattice_mapping(pa, pb, override=None, tol: float = 1e-3):
    """How B's phase ``pb`` maps onto A's ``pa``: (perm, k) with perm[t] = B's non-vacancy sublattice that goes to
    A's t-th non-vacancy sublattice and k = A's total sites / B's.  None if the site ratios are not proportional
    (a genuinely different model).  Raises SublatticeMapError when the assignment is ambiguous or contradicted."""
    name = pa.name
    a_idx = [i for i in range(len(pa.site_ratios)) if i not in _va_only(pa)]
    b_idx = [i for i in range(len(pb.site_ratios)) if i not in _va_only(pb)]
    hint = (f"give sublattice_map {{'{name}': [...]}} (B's sublattice i -> A's position; vacancy-only sublattices "
            f"stay in place) or {{'{name}': 'keep'}} to leave B's model unchanged, or ask the user")
    if len(a_idx) != len(b_idx):
        raise SublatticeMapError(f"{name}: A {_subl_text(pa)} has {len(a_idx)} non-vacancy sublattices, B "
                                 f"{_subl_text(pb)} has {len(b_idx)}: cannot map them; {hint}")
    rA = [pa.site_ratios[i] for i in a_idx]
    rB = [pb.site_ratios[i] for i in b_idx]
    k = sum(rA) / sum(rB)
    if override is not None:
        ov = [int(x) for x in override]                 # B's sublattice i -> A's position (as in A's PHASE line)
        if len(ov) == len(pb.site_ratios) and len(ov) != len(b_idx):
            ov = [ov[i] for i in b_idx]                 # given over all of B's sublattices: the non-vacancy ones
        ov = [a_idx.index(x) if x in a_idx else -1 for x in ov]
        if sorted(ov) != list(range(len(b_idx))):
            raise SublatticeMapError(f"{name}: sublattice_map {list(override)} is not a permutation of "
                                     f"{len(b_idx)} sublattices")
        perm = [0] * len(b_idx)
        for j, t in enumerate(ov):
            perm[t] = j
        ok = all(abs(rB[perm[t]] * k - rA[t]) <= tol * max(rA) for t in range(len(rA)))
        return (perm, k) if ok else None
    # size classes
    match = [[t for t in range(len(rA)) if abs(rB[j] * k - rA[t]) <= tol * max(rA)] for j in range(len(rB))]
    if any(not m for m in match):
        return None
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for j, m in enumerate(match):
        groups.setdefault(tuple(m), []).append(j)
    perm = [None] * len(rA)
    A_cons = [set(pa.constituents[i]) - {"VA"} for i in a_idx]
    B_cons = [set(pb.constituents[i]) - {"VA"} for i in b_idx]
    for ts, js in groups.items():
        if len(ts) != len(js):
            return None
        if len(ts) == 1:
            perm[ts[0]] = js[0]
            continue
        best, scored = None, []
        for p in itertools.permutations(js):
            score = sum(len(A_cons[t] & B_cons[j]) for t, j in zip(ts, p))
            scored.append((score, p))
        top = max(s for s, _ in scored)
        tied = [p for s, p in scored if s == top]
        a_same = len({frozenset(A_cons[t]) for t in ts}) == 1
        b_same = len({frozenset(B_cons[j]) for j in js}) == 1
        sizes = "/".join(f"{rA[t]:g}" for t in ts)
        if top == 0 and not (a_same or b_same):
            raise SublatticeMapError(
                f"{name}: B's equal-size sublattices ({sizes} sites) "
                f"{' vs '.join('(' + ','.join(sorted(B_cons[j])) + ')' for j in js)} share no element with A's "
                f"{' vs '.join('(' + ','.join(sorted(A_cons[t])) + ')' for t in ts)}: which is which is undecidable; {hint}")
        if len(tied) > 1 and not (a_same or b_same):
            raise SublatticeMapError(
                f"{name}: element overlap of B's equal-size sublattices ({sizes} sites) points both ways "
                f"(A {_subl_text(pa)}, B {_subl_text(pb)}); {hint}")
        best = tied[0] if len(tied) == 1 else min(tied, key=lambda p: sum(abs(a - b) for a, b in zip(p, js)))
        for t, j in zip(ts, best):
            perm[t] = j
    return perm, k


def _scale_pw(pw: Piecewise, k: float) -> Piecewise:
    """G/L rescale: each T range multiplied separately (tdbparse.scale_piecewise)."""
    return scale_piecewise(pw, k)


def _permute_raw(raw: str, new_from_old: List[int]) -> str:
    ptype, phase, diff, subls, order = _split_raw_name(raw)
    if len(subls) != len(new_from_old):
        raise ValueError(f"{raw}: {len(subls)} sublattices, phase has {len(new_from_old)}")
    return _build_raw_name(ptype, phase, diff, [subls[o] for o in new_from_old], order)


def _has_magnetic_params(db: TDB, phase: str) -> bool:
    for p in db.parameters:
        if p.key.phase == phase and p.key.ptype in MAGNETIC_PTYPES:
            if not all(re.fullmatch(r"[+-]?0*\.?0*(E[+-]?\d+)?", e.strip().upper()) for _, _, e in p.expr.ranges):
                return True
    return False


def _order_disorder_related(db: TDB, phase: str) -> bool:
    ph = db.phases[phase]
    if any(td.kind == "DIS_PART" for td in db.phase_typedefs(ph)):
        return True
    for other in db.phases.values():
        for td in db.phase_typedefs(other):
            if td.kind == "DIS_PART" and re.search(rf"DIS_P\w*\s+{re.escape(phase)}(?![A-Z0-9_])", td.text.upper()):
                return True
    return False


A_RESCALED = "A rescaled to B (magnetic exception)"
NORM_CHANGES = ("renamed", "reordered", "rescaled", A_RESCALED)


def _dis_part_of(db: TDB, phase: str) -> Optional[str]:
    for td in db.phase_typedefs(db.phases[phase]):
        m = re.search(r"DIS_P\w*\s+([A-Z0-9_]+)", td.text.upper())
        if m:
            return m.group(1)
    return None


def magnetic_exception(A: TDB, B: TDB, phase: str) -> Optional[str]:
    """None if A's phase should be rescaled to B's site ratios instead of the reverse - B's phase has TC/BMAGN
    parameters and ~1 atom per formula unit, A's has no magnetic terms (own or DIS_PART) - else the reason why not
    (the phase is then only reordered)."""
    if _has_magnetic_params(A, phase):
        return "both phases have magnetic parameters"
    dis = _dis_part_of(A, phase)
    if dis and dis in A.phases and _has_magnetic_params(A, dis):
        return "A's disordered part has magnetic parameters"
    if _order_disorder_related(A, phase):
        return "A's phase is order-disorder linked (its partner would need rescaling too)"
    pb = B.phases[phase]
    subs = [i for i in range(len(pb.site_ratios)) if i not in _va_only(pb)]
    if any("VA" in pb.constituents[i] for i in subs):
        return "B's atoms per formula unit are not defined (VA on a substitutional sublattice)"
    n = sum(pb.site_ratios[i] for i in subs)
    if abs(n - 1.0) > 1e-3:
        return f"B has {n:g} atoms per formula unit (the exception needs ~1)"
    return None


def normalize_phase_model(A: TDB, B: TDB, phase: str, override=None) -> List[NormNote]:
    """Make B's ``phase`` use A's sublattice order and formula unit (in place).  G and L terms are multiplied by
    k = A's sites / B's sites; TC/BMAGN are not (pycalphad's IHJ magnetic term R*T*ln(beta+1)*f(tau) is per
    formula unit and not linear in BMAGN, so a magnetic phase with magnetic parameters is only reordered, never
    rescaled).  Order-disorder phases are left alone unless nothing changes."""
    pa, pb = A.phases[phase], B.phases[phase]
    if isinstance(override, str) and override.lower() == "keep":
        return [NormNote(phase, "kept", f"B's model {_subl_text(pb)} left unchanged (sublattice_map 'keep')")]
    na, nb = (len(p.site_ratios) - len(_va_only(p)) for p in (pa, pb))
    if na != nb:
        # a different number of real sublattices is a different model: no sublattice_map can map it, the merge
        # has to keep one model or both (merge decision db1 / db2 / rename)
        return [NormNote(phase, "incompatible", f"A {_subl_text(pa)} has {na} non-vacancy sublattices, B "
                                                f"{_subl_text(pb)} has {nb}: different models, B's kept as is")]
    m = sublattice_mapping(pa, pb, override)
    if m is None:
        return [NormNote(phase, "incompatible", f"site ratios not proportional: A {_subl_text(pa)} vs B "
                                                f"{_subl_text(pb)}")]
    perm, k = m
    b_idx = [i for i in range(len(pb.site_ratios)) if i not in _va_only(pb)]
    new_from_old = list(range(len(pb.site_ratios)))
    for t, j in enumerate(perm):
        new_from_old[b_idx[t]] = b_idx[j]
    reorder = new_from_old != list(range(len(pb.site_ratios)))
    rescale = abs(k - 1.0) > 1e-12
    if not reorder and not rescale:
        return []
    notes = []
    before = _subl_text(pb)
    rescale_a, k_b = False, k
    if rescale and _has_magnetic_params(B, phase):
        why = magnetic_exception(A, B, phase)
        if why is None:
            rescale_a = True
        else:
            notes.append(NormNote(phase, "not_rescaled", f"B has magnetic parameters (IHJ term per formula unit) and "
                                                        f"{why}: B {before} vs A {_subl_text(pa)} stay different "
                                                        f"models"))
        rescale, k = False, 1.0
        if not reorder and not rescale_a:
            return notes
    if (reorder or rescale) and _order_disorder_related(B, phase):
        return notes + [NormNote(phase, "incompatible", f"order-disorder phase: B {before} not reordered/rescaled "
                                                        f"to A {_subl_text(pa)}")]
    pb.site_ratios = [pb.site_ratios[o] * k for o in new_from_old]
    pb.constituents = [pb.constituents[o] for o in new_from_old]
    if pb.majors and len(pb.majors) == len(new_from_old):
        pb.majors = [pb.majors[o] for o in new_from_old]
    # site ratios exactly A's where they correspond (k*B may differ in the last digits, e.g. 0.6667*3)
    for t, i in enumerate(b_idx):
        a_i = [x for x in range(len(pa.site_ratios)) if x not in _va_only(pa)][t]
        if rescale and abs(pb.site_ratios[i] - pa.site_ratios[a_i]) <= 1e-3 * max(pa.site_ratios):
            pb.site_ratios[i] = pa.site_ratios[a_i]
    n_par = 0
    for p in B.parameters:
        if p.key.phase != phase:
            continue
        if reorder:
            p.raw_name = _permute_raw(p.raw_name, new_from_old)
            p.key, p.sign, p.order_map = parse_param_name(p.raw_name)
        if rescale and p.key.ptype in SCALED_PTYPES:
            p.expr = _scale_pw(p.expr, k)
        n_par += 1
    if reorder:
        notes.append(NormNote(phase, "reordered", f"sublattices {before} -> "
                              f"{''.join(f'({c})' for c in (','.join(x) for x in pb.constituents))} "
                              f"(B's sublattice order {new_from_old}); {n_par} parameter(s) permuted"))
    if rescale:
        notes.append(NormNote(phase, "rescaled", f"x{k:g} sites to A's {_subl_text(pa)}: G/L terms x{k:g} "
                                                 f"(per formula unit), TC/BMAGN unchanged"))
    if rescale_a:
        # A takes B's site ratios (in A's sublattice order); A's G/L x 1/k is exact since A has no magnetic terms
        ka = 1.0 / k_b
        a_before = _subl_text(pa)
        a_idx = [i for i in range(len(pa.site_ratios)) if i not in _va_only(pa)]
        pa.site_ratios = [r * ka for r in pa.site_ratios]
        for t, i in enumerate(a_idx):
            pa.site_ratios[i] = pb.site_ratios[b_idx[t]]
        n_a = 0
        for p in A.parameters:
            if p.key.phase == phase and p.key.ptype in SCALED_PTYPES:
                p.expr = _scale_pw(p.expr, ka)
                n_a += 1
        notes.append(NormNote(phase, A_RESCALED, f"x{ka:.6g}: A {a_before} -> {_subl_text(pa)} (B's site ratios, "
                                                 f"A's order) because B is magnetic with 1 atom/fu and A is not; "
                                                 f"{n_a} A G/L term(s) x{ka:.6g}: exact"))
    return notes


def normalize_pair(A: TDB, B: TDB, phase_map: Optional[Dict[str, str]] = None,
                   sublattice_map: Optional[Dict] = None, on_error: str = "raise") -> Tuple[TDB, TDB, List[NormNote]]:
    """(A', B', notes): B in A's conventions (normalize_to_base); A' is A, except for phases where the magnetic
    exception applies (B magnetic with ~1 atom/fu, A not magnetic): there A is rescaled to B's site ratios."""
    An = copy.deepcopy(A)
    Bn, notes = normalize_to_base(An, B, phase_map, sublattice_map, on_error)
    return An, Bn, notes


def unmapped_hint(phase: str) -> str:
    return (f"sublattice_map {{'{phase}': [...]}} (B's sublattice i -> A's position) if the user knows the "
            f"correspondence, or {{'{phase}': 'keep'}} to leave B's model as is (the merge then asks "
            f"'phase:{phase}': db1 / db2 / rename if the models differ)")


def normalize_to_base(A: TDB, B: TDB, phase_map: Optional[Dict[str, str]] = None,
                      sublattice_map: Optional[Dict] = None, on_error: str = "raise") -> Tuple[TDB, List[NormNote]]:
    """Copy of B in A's conventions.  Phase names: an explicit phase_map {B name: A name}, else a clear alias
    (identical structure tokens - LAVES_C15 / C15_LAVES / C15, SIGMA / SIGMA_D8B, A2_BCC / BCC_A2 - one phase on
    each side, the A name unused in B, and proportional site ratios).  Weaker matches are returned as 'candidate'
    notes, never applied.  Then each phase present in both is reordered / rescaled to A's model - except a magnetic
    B phase with ~1 atom/fu against a non-magnetic A phase: there ``A`` itself (modified in place; normalize_pair
    works on a copy) is rescaled to B's site ratios.

    on_error: 'raise' (a phase whose sublattice correspondence is ambiguous raises SublatticeMapError) or 'keep'
    (that phase keeps B's model unchanged and gets an 'unmapped' note with the error - evaluate_compatibility and
    staged_merge report it instead of stopping).  A different number of non-vacancy sublattices is never an error:
    it is a different model ('incompatible' note; the merge decides db1 / db2 / rename)."""
    out = copy.deepcopy(B)
    notes: List[NormNote] = []
    smap = {k.upper(): v for k, v in (sublattice_map or {}).items()}
    user = {k.upper(): v.upper() for k, v in (phase_map or {}).items()}
    for b, a in user.items():
        if b not in out.phases:
            raise ValueError(f"phase_map: {b} is not a phase of {B.name}")
        if a in out.phases and a != b:
            raise ValueError(f"phase_map renames {b} to {a}, but {B.name} also has a phase {a} (two phases of the "
                             f"added database are never merged)")
    mapping = dict(user)
    for b in sorted(out.phases):
        if b in mapping or b in A.phases:
            continue
        tb = phase_tokens(b)
        if not tb:
            continue
        cands = [a for a in A.phases if a not in out.phases and a not in mapping.values() and phase_tokens(a) == tb]
        same_b = [x for x in out.phases if phase_tokens(x) == tb and x not in A.phases]
        if len(cands) == 1 and len(same_b) == 1:
            a = cands[0]
            try:
                ok = sublattice_mapping(A.phases[a], out.phases[b], smap.get(a, smap.get(b))) is not None
            except SublatticeMapError:
                ok = True            # same structure: the reorder step below raises the clear error
            if ok:
                mapping[b] = a
            else:
                notes.append(NormNote(b, "candidate", f"~ A's {a} (same structure tokens) but site ratios differ: "
                                                      f"{_subl_text(A.phases[a])} vs {_subl_text(out.phases[b])}; "
                                                      f"phase_map {{'{b}': '{a}'}} only if the user confirms"))
    for b, a in mapping.items():
        rename_phases(out, {b: a})
        notes.append(NormNote(b, "renamed", f"-> {a}" + (" (phase_map)" if b in user else " (same structure)")))
    # weaker candidates (shared structure token), listed only
    for b in sorted(out.phases):
        if b in A.phases:
            continue
        tb = phase_tokens(b)
        for a in sorted(A.phases):
            if a in out.phases or not (tb & phase_tokens(a)) or any(n.phase == b and a in n.detail for n in notes):
                continue
            notes.append(NormNote(b, "candidate", f"~ A's {a} (shares '{'/'.join(sorted(tb & phase_tokens(a)))}'); "
                                                  f"phase_map {{'{b}': '{a}'}} if the user says it is the same phase"))
    for ph in sorted(set(out.phases) & set(A.phases)):
        try:
            notes += normalize_phase_model(A, out, ph, smap.get(ph))
        except SublatticeMapError as e:
            if on_error != "keep":
                raise
            msg = str(e).split("; give sublattice_map")[0]
            notes.append(NormNote(ph, "unmapped", f"B's model {_subl_text(out.phases[ph])} kept as is: {msg}"))
    return out, notes
