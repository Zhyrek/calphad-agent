"""Feature 4: merge two or more TDB files into one.

The merge is planned deterministically from the input files plus a dict of
decisions.  Anything "obvious" is resolved automatically (and logged):

* identical definitions (ignoring whitespace / number formatting / "+" signs)
* numerically equivalent definitions (e.g. SGTE unary data typed with a
  different number of decimals) -> the higher-priority (earlier) file wins
* phases whose models differ only by a vacancy-only sublattice
  (e.g. BCC_A2 (CR,FE)1 vs (CR,FE)1(VA)3) -> parameters are converted to the
  model *with* the interstitial sublattice
* same phase with different constituents -> union of constituents
* magnetic / order-disorder type definitions -> re-lettered so codes never clash

Everything else becomes a :class:`Conflict` with a stable id (e.g.
``function:GHSERCR`` or ``parameter:L(LIQUID,CR,FE;0)``) that needs a
decision.  Parameter conflicts are grouped by subsystem (``system:CR-FE``) so
one answer can cover a whole binary.  Re-plan with the decisions until no
conflicts are pending, then write.

CLI:  calphad-tools merge a.tdb b.tdb c.tdb -o merged.tdb          (asks interactively)
      calphad-tools merge a.tdb b.tdb -o merged.tdb --prefer first (never asks)
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tdbparse import (TDB, ExpressionError, Function, Parameter, Phase, Piecewise, TypeDef,
                       parse_param_name, parse_tdb, wrap_tdb_line)

# Phase names that are commonly used for the same structure in different databases.
ALIAS_GROUPS = [
    {"SIGMA", "SIGMA_D8B", "SIGMA_SGTE", "D8B_SIGMA"},
    {"BCC_B2", "B2_BCC"},
    {"L12_FCC", "FCC_L12"},
    {"LAVES_C14", "C14_LAVES"},
    {"LAVES_C15", "C15_LAVES"},
    {"LIQUID", "LIQ"},
    {"MU_PHASE", "MU", "MU_D85"},
]
TYPE_CODE_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ&'()*+-./<=>?@[]^_`{|}~"


@dataclass
class Option:
    key: str            # what to put in decisions[conflict.id]
    label: str          # human readable
    detail: str = ""


@dataclass
class Conflict:
    id: str
    kind: str
    question: str
    options: List[Option]
    group: Optional[str] = None
    decision: Optional[str] = None
    decided_by: str = ""        # "user" | "group" | "policy"

    @property
    def pending(self) -> bool:
        return self.decision is None

    def to_dict(self) -> Dict:
        return {"id": self.id, "kind": self.kind, "question": self.question, "group": self.group,
                "options": [{"key": o.key, "label": o.label, "detail": o.detail} for o in self.options],
                "decision": self.decision, "decided_by": self.decided_by}


@dataclass
class MergedParam:
    raw_name: str
    expr: Piecewise
    src: int
    key: Tuple


@dataclass
class MergePlan:
    paths: List[str]
    dbs: List[TDB]
    conflicts: List[Conflict] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    elements: Dict = field(default_factory=dict)
    species: Dict = field(default_factory=dict)
    functions: Dict[str, Tuple[Function, int]] = field(default_factory=dict)
    phases: Dict[str, Tuple[Phase, List[TypeDef], int]] = field(default_factory=dict)
    params: Dict[Tuple, MergedParam] = field(default_factory=dict)
    other_commands: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    temp_limits: Tuple[float, float] = (298.15, 6000.0)
    phase_renames: List[Tuple[int, str, str]] = field(default_factory=list)   # (db index, old name, new name)

    def renames_of(self, i: int) -> Dict[str, str]:
        """{old: new} for the phases of db ``i`` that a 'rename' decision renamed."""
        return {old: new for j, old, new in self.phase_renames if j == i}

    @property
    def pending(self) -> List[Conflict]:
        return [c for c in self.conflicts if c.pending]

    def db_label(self, i: int) -> str:
        return f"db{i + 1} ({self.dbs[i].name})"

    def report(self, max_items: int = 60) -> str:
        lines = [f"Merge of: " + ", ".join(self.db_label(i) for i in range(len(self.dbs)))]
        lines.append(f"Auto-resolved / notes ({len(self.log)}):")
        lines += [f"  - {x}" for x in self.log[:max_items]]
        if len(self.log) > max_items:
            lines.append(f"  ... {len(self.log) - max_items} more")
        pend = self.pending
        lines.append(f"Pending decisions: {len(pend)}")
        groups: Dict[str, List[Conflict]] = {}
        for c in pend:
            groups.setdefault(c.group or c.id, []).append(c)
        for g, cs in groups.items():
            if len(cs) > 1 and cs[0].group:
                lines.append(f"  [{g}] {len(cs)} parameter conflicts - decide the whole group with "
                             f"decisions['{g}'] = one of {[o.key for o in cs[0].options]}, or individually:")
            for c in cs[:max_items]:
                lines.append(f"  * {c.id}: {c.question}")
                for o in c.options:
                    lines.append(f"      - '{o.key}': {o.label}" + (f"  [{o.detail}]" if o.detail else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _split_raw_name(raw: str):
    m = re.match(r"^([A-Za-z0-9_]+)\((.*)\)$", raw)
    ptype, inner = m.group(1), m.group(2)
    order = None
    if ";" in inner:
        inner, order = inner.rsplit(";", 1)
    phase, _, rest = inner.partition(",")
    diff = ""
    if "&" in phase:
        phase, diff = phase.split("&", 1)
    subls = [sl.split(",") for sl in rest.split(":")]
    return ptype, phase, diff, subls, order


def _build_raw_name(ptype, phase, diff, subls, order) -> str:
    d = f"&{diff}" if diff else ""
    s = ":".join(",".join(sl) for sl in subls)
    o = f";{order}" if order not in (None, "") else ";0"
    return f"{ptype}({phase}{d},{s}{o})"


def _rename_phase_in_param(p: Parameter, new_phase: str) -> None:
    ptype, phase, diff, subls, order = _split_raw_name(p.raw_name)
    p.raw_name = _build_raw_name(ptype, new_phase, diff, subls, order)
    p.key.phase = new_phase


def _rename_phase(db: TDB, old: str, new: str) -> None:
    ph = db.phases.pop(old)
    ph.name = new
    db.phases[new] = ph
    for p in db.parameters:
        if p.key.phase == old:
            _rename_phase_in_param(p, new)
    for td in db.typedefs.values():
        td.text = re.sub(rf"(?<![A-Z0-9_]){re.escape(old)}(?![A-Z0-9_])", new, td.text, flags=re.I)


def _rename_function(db: TDB, old: str, new: str) -> None:
    f = db.functions.pop(old)
    f.name = new
    db.functions[new] = f
    mapping = {old: new}
    for fn in db.functions.values():
        fn.expr = fn.expr.rename_symbols(mapping)
    for p in db.parameters:
        p.expr = p.expr.rename_symbols(mapping)


def _values(db: TDB, expr: Piecewise, T: np.ndarray, sign: float = 1.0) -> Optional[np.ndarray]:
    try:
        return sign * expr.evaluate(T, db.evaluator())
    except (ExpressionError, ZeroDivisionError, OverflowError, ValueError):
        return None


def _max_diff(va, vb) -> Tuple[float, float]:
    if va is None or vb is None:
        return math.nan, math.nan
    m = np.isfinite(va) & np.isfinite(vb)
    if not m.any():
        return math.nan, math.nan
    return float(np.max(np.abs(va[m] - vb[m]))), float(max(np.max(np.abs(va[m])), np.max(np.abs(vb[m]))))


def _subsystem(key: Tuple, db: TDB) -> str:
    """Element subsystem of a parameter (for grouping decisions)."""
    els = set()
    for sl in key[2]:
        for sp in sl:
            if sp in ("VA", "*", "/-"):
                continue
            if sp in db.elements:
                els.add(sp)
            else:  # species like AL2, CR+3 -> strip digits/charges
                for e in re.findall(r"[A-Z][A-Z]?", re.sub(r"[\d+\-/]", " ", sp)):
                    if e in db.elements:
                        els.add(e)
    return "-".join(sorted(els)) or "?"


def _decide(c: Conflict, decisions: Dict[str, str], policy: str, default_key: Optional[str] = None) -> Conflict:
    keys = [o.key for o in c.options]
    if c.id in decisions and decisions[c.id] in keys:
        c.decision, c.decided_by = decisions[c.id], "user"
    elif c.group and c.group in decisions and decisions[c.group] in keys:
        c.decision, c.decided_by = decisions[c.group], "group"
    elif policy in ("first", "last"):
        dbk = [k for k in keys if k.startswith("db")]
        pick = default_key if (default_key and default_key in keys) else (
            (dbk[0] if policy == "first" else dbk[-1]) if dbk else keys[0])
        c.decision, c.decided_by = pick, f"policy:{policy}"
    return c


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def plan_merge(paths: Sequence[str], decisions: Optional[Dict[str, str]] = None, policy: str = "ask",
               rtol: float = 1e-4, atol: float = 1e-2, t_min: float = 298.15,
               t_max: float = 3000.0) -> MergePlan:
    """Plan a merge.  ``policy`` = 'ask' (leave conflicts pending) | 'first' | 'last'.

    Files earlier in ``paths`` have higher priority for automatic choices.
    """
    decisions = dict(decisions or {})
    dbs = [parse_tdb(p) for p in paths]
    plan = MergePlan([str(p) for p in paths], dbs)
    T = np.linspace(t_min, t_max, 60)
    n = len(dbs)
    for i, db in enumerate(dbs):
        for w in db.warnings:
            plan.log.append(f"parse warning in {db.name}: {w}")

    def equivalent(va, vb) -> Tuple[bool, float]:
        d, scale = _max_diff(va, vb)
        if math.isnan(d):
            return False, d
        return d <= max(atol, rtol * scale), d

    # ---- 0. phase-name aliases -------------------------------------------
    for group in ALIAS_GROUPS:
        present = {}
        for i, db in enumerate(dbs):
            names = [nm for nm in db.phases if nm in group]
            if len(names) == 1:
                present.setdefault(names[0], []).append(i)
        if len(present) < 2:
            continue
        sigs = {nm: dbs[idx[0]].phases[nm].reduced_signature() for nm, idx in present.items()}
        if len(set(sigs.values())) != 1:
            continue
        names = sorted(present)
        cid = "alias:" + "|".join(names)
        opts = [Option(f"unify:{nm}", f"treat as the same phase, named {nm}",
                       "used in " + ", ".join(dbs[i].name for i in present[nm])) for nm in names]
        opts.append(Option("separate", "keep as separate phases"))
        c = _decide(Conflict(cid, "phase_alias",
                             f"Phases {', '.join(names)} have the same sublattice model; same phase?", opts),
                    decisions, policy, default_key=f"unify:{names[0]}" if policy != "ask" else None)
        # prefer the name used by the highest-priority file under a policy
        if c.decided_by.startswith("policy"):
            first_name = min(present, key=lambda nm: present[nm][0])
            c.decision = f"unify:{first_name}"
        plan.conflicts.append(c)
        if c.decision and c.decision.startswith("unify:"):
            target = c.decision.split(":", 1)[1]
            for nm, idxs in present.items():
                if nm != target:
                    for i in idxs:
                        if target not in dbs[i].phases:
                            _rename_phase(dbs[i], nm, target)
            plan.log.append(f"phase alias: {', '.join(names)} unified as {target}")

    # ---- 1. temperature limits / elements / species -------------------------
    plan.temp_limits = (min(d.temp_limits[0] for d in dbs), max(d.temp_limits[1] for d in dbs))
    for el in sorted({e for d in dbs for e in d.elements}):
        have = [(i, d.elements[el]) for i, d in enumerate(dbs) if el in d.elements]
        first_i, first = have[0]
        plan.elements[el] = first
        diff = [(i, e) for i, e in have[1:] if e.ref_phase != first.ref_phase or not np.allclose(
            [e.mass, e.h298, e.s298], [first.mass, first.h298, first.s298], rtol=2e-3, atol=1e-3)]
        near = [(i, e) for i, e in have[1:] if (e.ref_phase == first.ref_phase and (i, e) not in diff and
                                                [e.mass, e.h298, e.s298] != [first.mass, first.h298, first.s298])]
        if near:
            plan.log.append(f"element {el}: reference data differ only slightly between files; "
                            f"kept {dbs[first_i].name}")
        if diff:
            opts = [Option(f"db{i + 1}", f"{dbs[i].name}: {e.ref_phase} M={e.mass:g} H298={e.h298:g} "
                                          f"S298={e.s298:g}") for i, e in have]
            c = _decide(Conflict(f"element:{el}", "element", f"Element {el} reference data differ", opts),
                        decisions, policy)
            plan.conflicts.append(c)
            if c.decision:
                plan.elements[el] = dbs[int(c.decision[2:]) - 1].elements[el]
    for i, db in enumerate(dbs):
        for nm, sp in db.species.items():
            if nm not in plan.species:
                plan.species[nm] = sp
            elif re.sub(r"\s", "", plan.species[nm].formula.upper()) != re.sub(r"\s", "", sp.formula.upper()):
                plan.log.append(f"species {nm}: formula differs ({plan.species[nm].formula} vs {sp.formula}); "
                                f"kept first")

    # ---- 2. functions ---------------------------------------------------------
    for name in sorted({f for d in dbs for f in d.functions}):
        have = [i for i, d in enumerate(dbs) if name in d.functions]
        base = have[0]
        plan.functions[name] = (dbs[base].functions[name], base)
        differing = []
        for i in have[1:]:
            fa, fb = dbs[base].functions[name], dbs[i].functions[name]
            if fa.expr.canonical_exprs() == fb.expr.canonical_exprs():
                continue
            ok, d = equivalent(_values(dbs[base], fa.expr, T), _values(dbs[i], fb.expr, T))
            if ok:
                plan.log.append(f"function {name}: {dbs[base].name} and {dbs[i].name} differ only numerically "
                                f"by max {d:.3g} (<= tolerance); kept {dbs[base].name}")
            else:
                differing.append((i, d))
        if not differing:
            continue
        opts = [Option(f"db{base + 1}", f"use {dbs[base].name} version", dbs[base].functions[name].expr.short(80))]
        for i, d in differing:
            opts.append(Option(f"db{i + 1}", f"use {dbs[i].name} version",
                               f"{dbs[i].functions[name].expr.short(80)}  (max|Δ|={d:.3g})"))
        opts.append(Option("keep_all", "keep all versions (later files' copies renamed, e.g. "
                                       f"{name}_2, and their references rewritten)"))
        c = _decide(Conflict(f"function:{name}", "function",
                             f"FUNCTION {name} differs between files", opts), decisions, policy)
        plan.conflicts.append(c)
        if c.decision == "keep_all":
            for i, _ in differing:
                new = f"{name}_{i + 1}"
                _rename_function(dbs[i], name, new)
                plan.functions[new] = (dbs[i].functions[new], i)
                plan.log.append(f"function {name}: {dbs[i].name} copy renamed to {new}")
        elif c.decision:
            j = int(c.decision[2:]) - 1
            plan.functions[name] = (dbs[j].functions[name], j)

    # ---- 3. phases --------------------------------------------------------------
    # conversion of parameters for phases aligned by vacancy sublattices: (db, phase) -> index map
    convert: Dict[Tuple[int, str], List[Optional[int]]] = {}
    dropped_phase_src: Dict[str, set] = {}
    # a worklist: a 'rename' decision appends the renamed copies (e.g. MU_D85_2), which are then planned like any
    # other phase (PHASE / CONSTITUENT line, type definitions re-lettered, parameters, DIS_PART links)
    phase_names = sorted({p for d in dbs for p in d.phases})
    k_phase = 0
    while k_phase < len(phase_names):
        name = phase_names[k_phase]
        k_phase += 1
        have = [i for i, d in enumerate(dbs) if name in d.phases]
        # pick the model: prefer the one with more sublattices among reduced-equal models
        sigs = {i: dbs[i].phases[name].model_signature() for i in have}
        red = {i: dbs[i].phases[name].reduced_signature() for i in have}
        base = have[0]
        if len(set(sigs.values())) > 1:
            if len(set(red.values())) == 1:
                base = max(have, key=lambda i: (dbs[i].phases[name].n_sublattices, -i))
                tb = dbs[base].phases[name]
                t_keep = [k for k in range(tb.n_sublattices) if k not in tb.va_only_sublattices()]
                for i in have:
                    if sigs[i] != sigs[base]:
                        ps = dbs[i].phases[name]
                        s_keep = [k for k in range(ps.n_sublattices) if k not in ps.va_only_sublattices()]
                        convert[(i, name)] = [s_keep[t_keep.index(k)] if k in t_keep else None
                                              for k in range(tb.n_sublattices)]
                        plan.log.append(f"phase {name}: {dbs[i].name} model {ps.site_ratios} differs from "
                                        f"{dbs[base].name} {tb.site_ratios} only by a vacancy-only sublattice; "
                                        f"converted {dbs[i].name} parameters to the {tb.site_ratios} model")
            else:
                groups: Dict[Tuple, List[int]] = {}
                for i in have:
                    groups.setdefault(sigs[i], []).append(i)
                opts = []
                for sig, idxs in groups.items():
                    ph = dbs[idxs[0]].phases[name]
                    opts.append(Option(f"db{idxs[0] + 1}", f"use the model from {', '.join(dbs[k].name for k in idxs)}",
                                       f"{ph.n_sublattices} sublattices {ph.site_ratios}; parameters of {name} "
                                       f"from files with other models are dropped"))
                opts.append(Option("rename", f"keep all: rename {name} in later files to {name}_2, ..."))
                c = _decide(Conflict(f"phase:{name}", "phase_model",
                                     f"Phase {name} has incompatible sublattice models", opts), decisions, policy)
                plan.conflicts.append(c)
                if c.decision == "rename":
                    for sig, idxs in list(groups.items())[1:]:
                        for i in idxs:
                            new = f"{name}_{i + 1}"
                            k = 2
                            while any(new in d.phases for d in dbs):
                                new, k = f"{name}_{i + 1}_{k}", k + 1
                            _rename_phase(dbs[i], name, new)
                            plan.phase_renames.append((i, name, new))
                            phase_names.append(new)
                            plan.log.append(f"phase {name} in {dbs[i].name} renamed to {new} (kept with its own "
                                            f"model {dbs[i].phases[new].site_ratios})")
                    have = groups[list(groups)[0]]
                    base = have[0]
                elif c.decision:
                    base = int(c.decision[2:]) - 1
                    keep = [i for i in have if sigs[i] == sigs[base]]
                    dropped_phase_src[name] = set(have) - set(keep)
                    have = keep
                else:
                    have = [i for i in have if sigs[i] == sigs[base]]
        tb = copy.deepcopy(dbs[base].phases[name])
        # union constituents
        for i in have:
            if i == base:
                continue
            ps = dbs[i].phases[name]
            idx_map = convert.get((i, name), list(range(ps.n_sublattices)))
            for k, src_k in enumerate(idx_map):
                if src_k is None or k >= len(tb.constituents) or src_k >= len(ps.constituents):
                    continue
                added = [s for s in ps.constituents[src_k] if s not in tb.constituents[k]]
                if added:
                    tb.constituents[k] += added
                    plan.log.append(f"phase {name}: sublattice {k + 1} constituents {added} added from {dbs[i].name}")
        # type definitions
        tds = {i: [t for t in dbs[i].phase_typedefs(dbs[i].phases[name]) if t.kind != "SEQ"] for i in have}
        canon = {i: {t.canonical(name): t for t in tds[i]} for i in have}
        merged_td = dict(canon[base])
        for i in have:
            for k, t in canon[i].items():
                kinds_same = [x for x in merged_td.values() if x.kind == t.kind]
                if k in merged_td:
                    continue
                if kinds_same and t.kind in ("MAGNETIC", "DIS_PART"):
                    opts = [Option(f"db{base + 1}", f"{dbs[base].name}: {kinds_same[0].text}"),
                            Option(f"db{i + 1}", f"{dbs[i].name}: {t.text}")]
                    c = _decide(Conflict(f"typedef:{name}:{t.kind}", "typedef",
                                         f"{t.kind} type definition for {name} differs", opts), decisions, policy)
                    plan.conflicts.append(c)
                    if c.decision == f"db{i + 1}":
                        for kk in [kk for kk, x in merged_td.items() if x.kind == t.kind]:
                            del merged_td[kk]
                        merged_td[k] = t
                else:
                    merged_td[k] = t
                    plan.log.append(f"phase {name}: {t.kind.lower()} type definition taken from {dbs[i].name}")
        plan.phases[name] = (tb, list(merged_td.values()), base)
        for i in have:
            if i != base and (i, name) not in convert:
                convert[(i, name)] = list(range(dbs[i].phases[name].n_sublattices))
        convert[(base, name)] = list(range(tb.n_sublattices))
        for i in dropped_phase_src.get(name, ()):
            plan.log.append(f"phase {name}: parameters from {dbs[i].name} dropped (incompatible model)")

    # ---- 4. parameters ------------------------------------------------------
    per_key: Dict[Tuple, List[Tuple[int, Parameter, str, Tuple]]] = {}
    order_keys: List[Tuple] = []
    for i, db in enumerate(dbs):
        seen = {}
        for p in db.parameters:
            ph = p.key.phase
            if ph not in plan.phases:
                plan.log.append(f"parameter {p.raw_name} in {db.name} refers to undefined phase; dropped")
                continue
            idx_map = convert.get((i, ph))
            if idx_map is None:     # phase from this db dropped
                continue
            raw = p.raw_name
            if idx_map != list(range(len(p.key.constituents))):
                ptype, phn, diff, subls, order = _split_raw_name(raw)
                subls = [subls[k] if k is not None and k < len(subls) else ["VA"] for k in idx_map]
                raw = _build_raw_name(ptype, phn, diff, subls, order)
            key, sign, _ = parse_param_name(raw)
            kt = key.as_tuple()
            seen[kt] = (i, p, raw, kt, sign)   # last definition in a file wins
        for kt, (i, p, raw, kt_, sign) in seen.items():
            if kt not in per_key:
                per_key[kt] = []
                order_keys.append(kt)
            per_key[kt].append((i, p, raw, sign))
    for kt in order_keys:
        entries = per_key[kt]
        i0, p0, raw0, s0 = entries[0]
        plan.params[kt] = MergedParam(raw0, p0.expr, i0, kt)
        differing = []
        for i, p, raw, s in entries[1:]:
            if p.expr.canonical_exprs() == p0.expr.canonical_exprs() and s == s0:
                continue
            ok, d = equivalent(_values(dbs[i0], p0.expr, T, s0), _values(dbs[i], p.expr, T, s))
            if ok:
                plan.log.append(f"parameter {raw0}: {dbs[i0].name} and {dbs[i].name} equivalent "
                                f"(max|Δ|={d:.3g}); kept {dbs[i0].name}")
            else:
                differing.append((i, p, raw, d))
        if not differing:
            continue
        label = parse_param_name(raw0)[0].label()
        sub = _subsystem(kt, dbs[i0])
        opts = [Option(f"db{i0 + 1}", f"{dbs[i0].name}", p0.expr.short(80))]
        for i, p, raw, d in differing:
            opts.append(Option(f"db{i + 1}", f"{dbs[i].name}", f"{p.expr.short(80)}  (max|Δ|={d:.3g})"))
        c = _decide(Conflict(f"parameter:{label}", "parameter", f"{label} differs", opts,
                             group=f"system:{sub}"), decisions, policy)
        plan.conflicts.append(c)
        if c.decision:
            j = int(c.decision[2:]) - 1
            for i, p, raw, d in [(i0, p0, raw0, 0)] + differing:
                if i == j:
                    plan.params[kt] = MergedParam(raw, p.expr, i, kt)
    # parameters that only exist in a non-chosen db for a *group* decision: keep them (they don't conflict)

    # ---- 5. defaults / references -------------------------------------------
    seen_cmd = set()
    used_phases_by_db = {i: {p.key.phase for p in d.parameters} for i, d in enumerate(dbs)}
    for i, db in enumerate(dbs):
        for cmd in db.other_commands:
            kw = cmd.split()[0].upper().replace("-", "_")
            if kw.startswith(("TEMP", "ASSESS", "VERSION", "DATABASE", "REFERENCE_F")):
                continue
            if kw.startswith("DEFINE_SYS") and any(c.upper().startswith("DEFINE_SYS") for c in seen_cmd):
                continue
            norm = cmd.upper().replace("-", "_")
            m = re.search(r"REJECT_PHASE\s+(.*)$", norm)
            if m:
                rejected = m.group(1).split()
                others_use = [ph for ph in rejected if any(ph in used_phases_by_db[k] for k in range(n) if k != i)]
                if others_use:
                    opts = [Option("keep", f"keep '{cmd}'"),
                            Option("drop", f"drop it (other files model {', '.join(others_use)})")]
                    c = _decide(Conflict(f"default:{i + 1}:{norm}", "default_command",
                                         f"{db.name} rejects {', '.join(others_use)} by default, "
                                         f"but another file has parameters for it", opts), decisions, policy,
                                default_key="drop")
                    if c.decided_by.startswith("policy"):
                        c.decision = "drop"
                    plan.conflicts.append(c)
                    if c.decision != "keep":
                        continue
            if norm not in seen_cmd:
                seen_cmd.add(norm)
                plan.other_commands.append(cmd)
        plan.references += db.references
    return plan


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def render_tdb(plan: MergePlan, allow_pending: bool = False) -> str:
    if plan.pending and not allow_pending:
        raise ValueError(f"{len(plan.pending)} conflicts still need a decision: "
                         + ", ".join(c.id for c in plan.pending[:10]))
    L: List[str] = []
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L.append(f"$ Merged TDB written by calphad_tools on {now}")
    L.append("$ Sources (priority order):")
    for i, p in enumerate(plan.paths):
        L.append(f"$   db{i + 1}: {p}")
    L.append("$")
    decided = [c for c in plan.conflicts if c.decision]
    if decided:
        L.append("$ Decisions:")
        for c in decided:
            L.append(f"$   {c.id} -> {c.decision} ({c.decided_by})"[:200])
    if plan.log:
        L.append("$ Automatic resolutions / notes:")
        for x in plan.log:
            L.append(f"$   {x}"[:200])
    for i, db in enumerate(plan.dbs):
        if db.header_comments:
            L.append(f"$ ---- header of {db.name} ----")
            L += [c[:200] for c in db.header_comments[:40]]
    L.append("$")
    L.append(f" TEMPERATURE_LIMITS {plan.temp_limits[0]:g} {plan.temp_limits[1]:g} !")
    special = {"/-": Element_("/-", "ELECTRON_GAS"), "VA": Element_("VA", "VACUUM")}
    for nm in ["/-", "VA"]:
        if nm in plan.elements or nm == "VA":
            L.append(plan.elements.get(nm, special[nm]).to_tdb())
    for nm, el in sorted(plan.elements.items()):
        if nm not in ("/-", "VA"):
            L.append(el.to_tdb())
    L.append("$")
    for nm, sp in sorted(plan.species.items()):
        L.append(sp.to_tdb())
    L.append("$")
    L.append("$ ---------------------------------------------------------------- FUNCTIONS")
    for nm, (fn, src) in plan.functions.items():
        L.append(wrap_tdb_line(fn.to_tdb()))
    L.append("$")
    # type definitions: assign fresh codes
    code_of: Dict[str, str] = {}
    pool = iter(TYPE_CODE_POOL)
    L.append(" TYPE_DEFINITION % SEQ * !")
    for nm, (ph, tds, src) in plan.phases.items():
        for td in tds:
            txt = re.sub(r"\s+", " ", td.text.strip())
            if txt not in code_of:
                code_of[txt] = next(pool)
                L.append(wrap_tdb_line(f" TYPE_DEFINITION {code_of[txt]} {txt} !"))
    for cmd in plan.other_commands:
        L.append(wrap_tdb_line(f" {cmd} !"))
    L.append("$")
    L.append("$ ---------------------------------------------------------------- PHASES")
    for nm, (ph, tds, src) in plan.phases.items():
        ph = copy.deepcopy(ph)
        ph.type_codes = "%" + "".join(code_of[re.sub(r"\s+", " ", t.text.strip())] for t in tds)
        L += [wrap_tdb_line(x) for x in ph.to_tdb()]
    L.append("$")
    L.append("$ ---------------------------------------------------------------- PARAMETERS")
    by_phase: Dict[str, List[MergedParam]] = {}
    for mp in plan.params.values():
        by_phase.setdefault(mp.key[1], []).append(mp)
    for nm in plan.phases:
        ps = by_phase.get(nm, [])
        if not ps:
            continue
        L.append(f"$ {nm}")
        ps.sort(key=lambda m: (sum(len(s) for s in m.key[2]), m.key[2], m.key[0], m.key[3]))
        for mp in ps:
            L.append(wrap_tdb_line(f" PARAMETER {mp.raw_name:<28s} {mp.expr.to_tdb()} !"))
    if plan.references:
        L.append("$")
        L.append("$ ---------------------------------------------------------------- REFERENCES")
        body = []
        seen = set()
        for r in plan.references:
            for line in r.split("\n")[1:]:
                s = line.strip()
                if not s or s.upper().startswith("NUMBER") or s in seen:
                    continue
                seen.add(s)
                body.append("  " + s)
        if body:
            L.append(" LIST_OF_REFERENCES")
            L.append(" NUMBER  SOURCE")
            L += body
            L.append(" !")
    return "\n".join(L) + "\n"


@dataclass
class Element_:
    name: str
    ref_phase: str

    def to_tdb(self):
        return f" ELEMENT {self.name:<3s} {self.ref_phase:<24s} 0.0 0.0 0.0!"


def validate(tdb_path: str) -> Dict:
    """Load the merged file with pycalphad and build every phase model."""
    import warnings
    from pycalphad import Database, Model
    from pycalphad.core.utils import filter_phases, unpack_species
    out = {"loaded": False, "errors": []}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = Database(tdb_path)
        out["loaded"] = True
        comps = sorted(str(e) for e in db.elements if str(e) != "/-")
        for ph in filter_phases(db, unpack_species(db, comps)):
            try:
                Model(db, comps, ph)
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"{ph}: {type(e).__name__}: {e}")
        out["phases"] = sorted(db.phases)
        out["elements"] = comps
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
    # undefined function references
    tdb = parse_tdb(tdb_path)
    missing = set()
    for p in tdb.parameters:
        for s in p.expr.symbols():
            if s not in tdb.functions and s != "R":
                missing.add(s)
    for f in tdb.functions.values():
        for s in f.expr.symbols():
            if s not in tdb.functions and s != "R":
                missing.add(s)
    if missing:
        out["errors"].append(f"undefined functions referenced: {sorted(missing)}")
    return out


def merge(paths: Sequence[str], output: str, decisions: Optional[Dict[str, str]] = None,
          policy: str = "ask", ask: Optional[Callable[[Conflict, MergePlan], Optional[str]]] = None,
          rtol: float = 1e-4, max_rounds: int = 50) -> Dict:
    """Plan (asking via ``ask`` callback if given), write and validate a merged TDB."""
    decisions = dict(decisions or {})
    plan = plan_merge(paths, decisions, policy, rtol=rtol)
    rounds = 0
    while plan.pending and ask is not None and rounds < max_rounds:
        rounds += 1
        c = plan.pending[0]
        ans = ask(c, plan)
        if ans is None:
            break
        decisions[ans[0]] = ans[1]
        plan = plan_merge(paths, decisions, policy, rtol=rtol)
    if plan.pending:
        return {"written": False, "pending": [c.to_dict() for c in plan.pending],
                "report": plan.report(), "decisions": decisions}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(render_tdb(plan))
    val = validate(output)
    return {"written": True, "output": output, "report": plan.report(), "validation": val,
            "decisions": decisions}


def cli_ask(c: Conflict, plan: MergePlan) -> Optional[Tuple[str, str]]:
    """Interactive terminal resolver.  Offers group-wide answers for parameter conflicts."""
    print("\n" + "=" * 78)
    group_members = [x for x in plan.pending if c.group and x.group == c.group]
    target_id = c.id
    if len(group_members) > 1:
        print(f"{len(group_members)} parameter conflicts in subsystem {c.group.split(':', 1)[1]}:")
        for x in group_members[:12]:
            print(f"   {x.id.split(':', 1)[1]}")
            for o in x.options:
                print(f"       {o.label:<28s} {o.detail}")
        if len(group_members) > 12:
            print(f"   ... and {len(group_members) - 12} more")
        print("Choose one file for the whole subsystem, or 'i' to decide individually:")
        for k, o in enumerate(c.options, 1):
            print(f"  {k}) {o.label}")
        ans = input("> ").strip()
        if ans.lower() != "i":
            try:
                return c.group, c.options[int(ans) - 1].key
            except (ValueError, IndexError):
                print("invalid choice")
                return cli_ask(c, plan)
    print(c.question)
    for k, o in enumerate(c.options, 1):
        print(f"  {k}) {o.label}" + (f"\n       {o.detail}" if o.detail else ""))
    ans = input("> ").strip()
    try:
        return target_id, c.options[int(ans) - 1].key
    except (ValueError, IndexError):
        print("invalid choice")
        return cli_ask(c, plan)


def save_decisions(decisions: Dict[str, str], path: str) -> None:
    Path(path).write_text(json.dumps(decisions, indent=2))
