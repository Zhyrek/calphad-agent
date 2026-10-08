"""``combine_databases``: merge N TDBs end to end - plan, decisions, staged merges, validation, one report.

Stages
------
1. **Plan** (compatibility only, no mapping).  Every pair is checked with ``evaluate_compatibility``; a base and an
   order are proposed with reasons (most elements, most complete unary set, unary50 agreement where a local copy
   exists, most binaries assessed; then the source sharing most elements with what is already in, fewest
   conflicts).  The chain is then simulated step by step (``Q_k = Q_{k-1} + S_k``, built from the same merge plans
   as ``staged_merge``) and EVERY decision it needs is collected in one list with stable ids
   (``step2:phase:MU_D85``, ``step1:system:CR-FE``, ``step1:sublattice:C14``, ``step1:name:LAVES_C15``) and the
   options' consequences.  Later steps are planned with provisional answers (phase models 'rename', binaries
   'db1'); an answer that differs can change their questions, which are then asked in a second round.  Missing
   answers: the list is returned and nothing is merged (``policy='first'/'last'`` fills them, logged as policy).
2. **Run**.  ``staged_merge`` for each step (its versions, diagrams, invariant comparisons and content-hash cache),
   Q of step k being the base of step k+1, and a **decision log**: one entry per decision - the caller's (who =
   user / agent, with ``decision_reasons``), policy fills, and the automatic rules (unary/SER replacement, name
   normalisation, sublattice reorder, rescaling, magnetic exception, kept FUNCTIONs, constituent unions, phases
   entering another database's subsystems, forced replots, end-members without G) - with evidence (ΔG at a few
   T, invariant shifts, GM identity errors, figures) and the subsystems each one affects.  Every parameter or
   phase-model change of every step is attributed to an entry (the attribution map).
3. **Validation** ("proof").  For every source S and every binary (ternary at ``ternary_temperatures_c``) S
   describes: S and the final database F are mapped (mapping check on), invariants matched (ΔT, Δx), ZPF
   boundaries compared (max distance, 1 K ≙ 0.002 in x), an equilibrium grid compared (fraction of points with
   the same stable phases) and H, S, Cp, G of the stable assemblage spot-checked.  Each subsystem is
   *reproduced* (identical description, or within ΔT 0.5 K / Δx 1e-3 / grid 99.5 % / equal properties),
   *changed, explained* (every parameter / model difference S -> F is attributed to decision-log entries) or
   *CHANGED, UNEXPLAINED* (a difference no entry explains, or F edited after the run).  Normalisations and
   rescales are checked to be exact on the actual data (GM per mole of atoms, relative 1e-9).  F is loaded in
   pycalphad, linted (check_tdb), searched for new end-members without G (pycalphad uses 0) and its grids for
   non-converged points.  What cannot be proven - subsystems no source describes - is listed (and mapped as
   "predicted, not validated" at ``validate_temperatures_c``).
4. **Report**.  One deck, one Excel workbook, ``combine_report.md``, the final TDB with a header summarising
   sources, order, decisions and verdicts, ``combine.ipynb`` (rebuilds F; regenerates the validation) and
   ``combine_state.json`` (re-validate a hand-edited F with :func:`validate_from_state`).

Tentative parameters (calphad_tools.tentative, ``tentative``): every merge of the chain adds TENT_MIX / TENT_VA L0
terms for mixings no source describes and lets an assessed term replace a tentative one (a term added at step k goes
when a later source assesses it).  The plan lists the terms each step will add, the decision log has one rule entry
per term added and per replacement, and the validation attributes the changes they make to those entries.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tdbstage as ts
from . import tentative as tent
from .compatibility import clip
from .tdbparse import is_tentative, parse_tdb

KELVIN = 273.15
TOL_T, TOL_X, TOL_GRID, TOL_B = 0.5, 1e-3, 0.995, 2e-3     # reproduced: invariants, grid agreement, boundaries
T_SCALE = 500.0                                             # boundary distance: 1 K counts as 0.002 in x
PROP_TOL = {"GM": 0.01, "HM": 0.01, "SM": 1e-5, "CPM": 1e-4}   # J/mol(/K), plus 1e-7 relative
GM_TOL = 1e-9
EVIDENCE_T = (500.0, 1000.0, 1500.0)
VERDICTS = ("CHANGED, UNEXPLAINED", "changed, explained", "reproduced")
NOT_GIVEN = "not given"
SNAP_REASON = ("rule: site ratios written as a rounded form of simple fractions (0.333:0.667) are read as those "
               "fractions (1/3:2/3), and the merged phase is written with the exact ratios - no question; the GM "
               "change it makes is the evidence")
STATE_FILE = "combine_state.json"


def _stem(p: str) -> str:
    return Path(p).stem


def _sysname(els) -> str:
    return ts.stage_sysname(els)


def _base_name(p: str) -> str:
    return re.sub(r"#\d+$", "", str(p))


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _content(path: str) -> str:
    """TDB text without comment lines and reference lists (what the content-hash caches key on)."""
    from .tdbparse import calc_text
    return calc_text(Path(path).read_text(errors="replace"))


def parse_reason(value) -> Tuple[str, str]:
    """('user' | 'agent', text) from a decision_reasons value: 'user: <their words>', 'agent: <why>',
    {'who': ..., 'reason': ...}, or plain text (the agent's).  No value -> ('agent', '')."""
    if value is None:
        return "agent", ""
    if isinstance(value, dict):
        who, text = str(value.get("who", "")).lower().strip(), str(value.get("reason", "")).strip()
    else:
        text, who = str(value).strip(), ""
        m = re.match(r"^(user|agent)\s*[:\-]\s*(.*)$", text, re.I | re.S)
        if m:
            who, text = m.group(1).lower(), m.group(2).strip()
    return (who if who in ("user", "agent") else "agent"), text


# =============================================================================================== records
@dataclass
class Entry:
    """One decision-log entry."""
    id: str
    step: str
    subject: str
    options: List[str]
    chosen: str
    who: str                                      # user | agent | policy | rule
    reason: str
    kind: str = ""
    evidence: List[str] = field(default_factory=list)
    affected: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)

    def add_evidence(self, *lines: str, limit: int = 16) -> None:
        for s in lines:
            if s and s not in self.evidence and len(self.evidence) < limit:
                self.evidence.append(s)

    def add_affected(self, *names: str) -> None:
        for n in names:
            if n and n not in self.affected:
                self.affected.append(n)

    @property
    def flagged(self) -> bool:
        return self.who in ("user", "agent") and (not self.reason or self.reason == NOT_GIVEN)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Needed:
    """A decision the chain needs (asked once, up front)."""
    id: str
    step: int
    pair: str
    kind: str                                     # phase_model | binary | parameter | typedef | default | ...
    cid: str                                      # the merge's own id / group (staged_merge decisions key)
    question: str
    options: List[Tuple[str, str]]                # (value, consequence)
    answer: Optional[str] = None
    who: str = ""
    reason: str = ""
    provisional: str = ""
    pairs: Dict[str, Tuple[str, str, str]] = field(default_factory=dict)   # answer -> ('same'|'extend', A, B phase)

    def text(self) -> str:
        lines = [f"* {self.id} ({self.pair}): {self.question}"]
        lines += [f"    - '{k}': {c}" for k, c in self.options]
        return "\n".join(lines)


class DecisionLog:
    def __init__(self):
        self.entries: Dict[str, Entry] = {}

    def add(self, e: Entry) -> Entry:
        if e.id in self.entries:
            old = self.entries[e.id]
            old.add_evidence(*e.evidence)
            old.add_affected(*e.affected)
            return old
        self.entries[e.id] = e
        return e

    def get(self, eid: str) -> Optional[Entry]:
        return self.entries.get(eid)

    def rule(self, eid: str, step: str, subject: str, reason: str, kind: str, chosen: str = "applied",
             options: Sequence[str] = ()) -> Entry:
        return self.add(Entry(eid, step, subject, list(options), chosen, "rule", reason, kind))

    def __iter__(self):
        return iter(self.entries.values())

    def __len__(self):
        return len(self.entries)


# =============================================================================================== sources
def _locate_unary50_offline(spec) -> Optional[str]:
    if spec is None or str(spec).lower() == "off":
        return None
    from .compatibility import locate_unary50
    path, _ = locate_unary50(spec, allow_download=False)
    return path


def source_info(path: str, V50=None) -> Dict[str, Any]:
    """Elements, unary parameter count, assessed binaries / ternaries, SER agreement with unary50."""
    tdb = parse_tdb(path)
    V = ts.StageView(tdb, "S")
    els = sorted(V.own_elements)
    unary = [k for k in V.rows if len(V.row_elements(k)) == 1]
    bins = sorted({tuple(sorted(V.row_elements(k))) for k in V.rows if len(V.row_elements(k)) == 2})
    terns_p = sorted({tuple(sorted(V.row_elements(k))) for k in V.rows if len(V.row_elements(k)) == 3})
    described_t = []
    for t in ts.stage_subsystems(els, (3,)):
        if t in terns_p or all(tuple(sorted(b)) in bins for b in ts.stage_subsystems(t, (2,))):
            described_t.append(t)
    u50 = None
    if V50 is not None:
        u50 = 0
        for el in els:
            a, b = V.ser(el), V50.ser(el)
            if a is not None and b is not None and ts.compare_segs(a.segs, b.segs, ts.STAGE_RTOL, 1e-2,
                                                                   ts.STAGE_GRID)[0]:
                u50 += 1
    return {"path": str(path), "name": Path(path).name, "stem": _stem(path), "elements": els,
            "n_unary": len(unary), "binaries": [list(b) for b in bins], "ternaries": [list(t) for t in described_t],
            "unary50": u50, "phases": sorted(tdb.phases), "sha256": _sha(path)}


def _score(info) -> Tuple:
    return (len(info["elements"]), info["n_unary"], -1 if info["unary50"] is None else info["unary50"],
            len(info["binaries"]))


def _conflicts(ev) -> int:
    n = sum(1 for s in ev.stages if s.n in (1, 2, 3, 4) for r in s.rows if r.get("status") == "differs")
    n += sum(1 for r in ev.stages[4].rows if r.get("issue") and ("cannot" in r["issue"] or "undecided" in r["issue"]))
    return n


def propose_order(infos: Dict[str, Dict], pair_eval: Dict[Tuple[str, str], Any], base: Optional[str] = None,
                  order: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    """(paths in merge order, one reason per position)."""
    paths = list(infos)

    def conf(a, b):
        ev = pair_eval.get((a, b)) or pair_eval.get((b, a))
        return _conflicts(ev) if ev is not None else 0
    if order:
        out = list(order)
        reasons = [f"base {infos[out[0]]['name']}: order given by the caller"]
        have = set(infos[out[0]]["elements"])
        for p in out[1:]:
            sh = sorted(have & set(infos[p]["elements"]))
            shares = ", ".join(e.capitalize() for e in sh) or "nothing"
            reasons.append(f"then {infos[p]['name']} (given order; shares {shares})")
            have |= set(infos[p]["elements"])
        return out, reasons
    ranked = sorted(paths, key=lambda p: (_score(infos[p]), -paths.index(p)), reverse=True)
    b = base or ranked[0]
    i = infos[b]
    u50 = ("unary50 not checked (no local copy)" if i["unary50"] is None else
           f"{i['unary50']}/{len(i['elements'])} SER match unary50")
    best = {"elements": max(len(x["elements"]) for x in infos.values()),
            "unary": max(x["n_unary"] for x in infos.values())}
    why = (f"{len(i['elements'])} elements{' (most)' if len(i['elements']) == best['elements'] else ''}, "
           f"{i['n_unary']} unary parameters{' (most complete unary set)' if i['n_unary'] == best['unary'] else ''}, "
           f"{u50}, {len(i['binaries'])} binar{'y' if len(i['binaries']) == 1 else 'ies'} assessed")
    reasons = [f"base {i['name']}: " + ("given by the caller; " if base else "") + why]
    out = [b]
    have = set(i["elements"])
    rest = [p for p in paths if p != b]
    while rest:
        cand = sorted(rest, key=lambda p: (len(have & set(infos[p]["elements"])),
                                           -sum(conf(p, q) for q in out), _score(infos[p]), -paths.index(p)),
                      reverse=True)
        p = cand[0]
        sh = sorted(have & set(infos[p]["elements"]))
        if not sh:
            raise ValueError(f"{infos[p]['name']} shares no element with {', '.join(infos[q]['name'] for q in out)}: "
                             f"a staged merge needs a common element (merge it with merge_tdbs instead)")
        nc = {infos[q]["stem"]: conf(p, q) for q in out}
        reasons.append(f"then {infos[p]['name']}: shares {', '.join(e.capitalize() for e in sh)} with the sources "
                       f"already in; conflicts with them: "
                       + ", ".join(f"{k} {v}" for k, v in nc.items()))
        out.append(p)
        have |= set(infos[p]["elements"])
        rest.remove(p)
    return out, reasons


# =============================================================================================== keys
def _per_pair(spec, k: int, bstem: str) -> Dict:
    """A phase_map / sublattice_map given globally ({B phase: ...}) or per pair ({'step2': {...}},
    {'<added stem>': {...}}): the part for step k."""
    if not spec:
        return {}
    glob, per = {}, {}
    for key, v in spec.items():
        if isinstance(v, dict) and (re.fullmatch(r"step\d+", str(key)) or str(key).lower() in {bstem.lower()}
                                    or str(key).lower().endswith(".tdb")):
            if str(key) == f"step{k}" or _stem(str(key)).lower() == bstem.lower():
                per.update(v)
        elif not isinstance(v, dict):
            glob[key] = v
    return {**glob, **per}


def _lookup(d: Dict, k: int, bstem: str, cid: str, short: str = ""):
    """(key used, value) of a decision for step k: 'step{k}:<id>', '<added stem>:<id>', then '<id>' (id: the
    merge's own conflict id or the short id of the list)."""
    for c in [cid] + ([short] if short and short != cid else []):
        for key in (f"step{k}:{c}", f"{bstem}:{c}", c):
            if key in (d or {}):
                return key, d[key]
    return None, None


def _phase_subsystems(view: ts.StageView, phase: str) -> List[str]:
    return [_sysname(s) for s in ts.stage_subsystems(view.own_elements, (2,))
            if ts.restricted_model(view, phase, s) is not None and any(
                r.phase == phase and view.row_elements(kk) and view.row_elements(kk) <= set(s)
                and len(view.row_elements(kk)) == 2 for kk, r in view.rows.items())]


# =============================================================================================== step plans
def _step_plan(a: str, b: str, pm: Dict, sm: Dict, dec_user: Dict, work: Path, k: int,
               ext_raw: Optional[Dict[str, Any]] = None, ext_global: Sequence[str] = (),
               tentative=True) -> Dict[str, Any]:
    """What staged_merge's build_versions does before the merge, without writing versions: normalisation, 'extend'
    decisions (harmonize.staged_extensions), alias separation, the automatic (staged) decisions and the merge plan's
    conflicts.  ``ext_raw`` {A phase: decision value}; targets in ``ext_global`` (given without a step) are skipped
    where they do not apply."""
    from .harmonize import HarmonizeError, extension_spec, harmonization_candidates, staged_extensions
    work.mkdir(parents=True, exist_ok=True)
    A, B = parse_tdb(a), parse_tdb(b)
    An, Bn, notes = ts.normalize_pair(A, B, pm or None, sm or None, on_error="keep")
    a_eff, b_eff = a, b
    if ts.a_changed(notes):
        a_eff = ts.write_tdb(An, str(work / f"step{k}_A0_{_stem(a)}.tdb"), ["provisional A0 (combine plan)"])
    if ts.b_changed(notes):
        b_eff = ts.write_tdb(Bn, str(work / f"step{k}_B0_{_stem(b)}.tdb"), ["provisional B0 (combine plan)"])
    shared = sorted(set(A.real_elements) & set(parse_tdb(b_eff).real_elements))
    Ae, Be = parse_tdb(a_eff), parse_tdb(b_eff)
    cands = harmonization_candidates(Ae, Be)
    ren = _norm_renames(notes)
    specs: Dict[str, Dict] = {}
    for tgt, v in (ext_raw or {}).items():
        default = next((c["b_phase"] for c in cands if c["a_phase"] == tgt), tgt)
        try:
            sp = extension_spec(tgt, v, Ae, None, default)
            sp["phase"] = ren.get(sp["phase"], sp["phase"])
            if sp["phase"] not in Be.phases:
                raise HarmonizeError(f"extend {tgt}: {sp['phase']} is not a phase of {Path(b).name}")
        except HarmonizeError:
            if tgt in ext_global:
                continue
            raise
        specs[tgt] = sp
    b_merge, ext = b_eff, None
    if specs:
        ext = staged_extensions(a_eff, b_eff, specs, str(work / "extend"), shared, verify=False,
                                tentative=tentative)
        a_eff, b_merge = ext["A_ext"], ext["B_merge"]
    dec = ts.alias_separate([a_eff, b_merge], dec_user)
    plan0 = ts.plan_merge([a_eff, b_merge], dec, "ask")
    auto = ts.auto_decisions(plan0, shared)
    return {"notes": notes, "a_eff": a_eff, "b_eff": b_merge, "shared": shared, "plan0": plan0, "auto": auto,
            "alias": {c: v for c, v in dec.items() if c not in dec_user}, "specs": specs, "ext": ext,
            "cands": cands, "A_pre": Ae, "B_pre": Be}


def _is_binary_group(c) -> bool:
    if c.kind != "parameter" or not c.group or not c.group.startswith("system:"):
        return False
    return len([e for e in c.group.split(":", 1)[1].split("-") if e and e != "?"]) == 2


def _needed_from_plan(sp: Dict, k: int, pair: str, a_label: str, b_label: str, ev) -> List[Needed]:
    """Decisions of one step: pending merge conflicts and the binary assessments the staged rule would give to the
    base (asked: which assessment of a shared binary wins is the user's call)."""
    out: Dict[str, Needed] = {}
    plan0, auto = sp["plan0"], sp["auto"]
    models = {r["phase"]: r for r in ev.stages[4].rows if r.get("A model")} if ev is not None else {}
    for c in plan0.conflicts:
        if c.kind == "phase_alias":
            continue
        cid = c.group if (c.kind == "parameter" and c.group) else c.id
        if cid in out:
            continue
        auto_hit = (c.group in auto) if c.kind == "parameter" else (c.id in auto)
        if auto_hit and not _is_binary_group(c):
            continue                                       # staged rule (unaries, FUNCTIONs, element data)
        if c.kind == "phase_model":
            ph = c.id.split(":", 1)[1]
            row = models.get(ph, {})
            subs = {}
            for side, path in (("db1", sp["a_eff"]), ("db2", sp["b_eff"])):
                try:
                    subs[side] = ", ".join(_phase_subsystems(ts.StageView(parse_tdb(path), side), ph)) or "none"
                except Exception:  # noqa: BLE001 - text only
                    subs[side] = "its subsystems with " + ph
            q = (f"{ph}: incompatible models - {a_label} {row.get('A model', '')} vs {b_label} "
                 f"{row.get('B model', '')}").replace("  ", " ")
            opts = []
            for o in c.options:
                if o.key == "db1":
                    opts.append(("db1", f"{a_label}'s {ph} only; {b_label}'s {ph} parameters dropped "
                                        f"({b_label}'s {subs.get('db2')} change)"))
                elif o.key == "db2":
                    opts.append(("db2", f"{b_label}'s {ph} only; {a_label}'s {ph} parameters dropped "
                                        f"({a_label}'s {subs.get('db1')} change)"))
                elif o.key == "rename":
                    opts.append(("rename", f"keep both, {b_label}'s as {ph}_2: both sources' diagrams are kept; "
                                           f"in new mixed systems {ph} and {ph}_2 are separate phases"))
            ext_opt = _extend_option(sp, ph, ph, a_label, b_label)
            if ext_opt:
                opts.append(ext_opt)
            out[cid] = Needed(f"step{k}:{cid}", k, pair, "phase_model", cid, q, opts)
        elif c.kind == "parameter":
            members = [x for x in plan0.conflicts if x.group == c.group] if c.group else [c]
            sub = c.group.split(":", 1)[1] if c.group else c.id
            dmax = []
            for x in members:
                for o in x.options[1:]:
                    m = re.search(r"max\|Δ\|=([-+0-9.eE]+)", o.detail)
                    if m:
                        dmax.append(float(m.group(1)))
            big = f"; largest |Δ| {ts.fmt_energy(max(dmax))}" if dmax else ""
            binary = _is_binary_group(c)
            q = (f"{sub}: {len(members)} parameter(s) differ{big} - whose assessment of {sub} wins?")
            opts = [("db1", f"{a_label}'s {sub}" + (" (the staged default); " if binary else "; ")
                     + f"{b_label}'s {sub} diagram changes"),
                    ("db2", f"{b_label}'s {sub}" + (f", with {a_label}'s unaries (B's were replaced in B1/B2) - a "
                                                    f"combination nobody assessed; " if binary else "; ")
                     + f"{a_label}'s {sub} diagram changes")]
            out[cid] = Needed(f"step{k}:{cid}", k, pair, "binary" if binary else "parameter", cid, q, opts)
        elif c.kind == "default_command":
            m = re.search(r"rejects (.+?) by default", c.question)
            phs = "+".join(x.strip() for x in m.group(1).split(",")) if m else "phases"
            db = "db" + c.id.split(":")[1]
            who = a_label if db == "db1" else b_label
            opts = [("keep", f"keep {who}'s REJECT_PHASE (Thermo-Calc then ignores {phs.replace('+', ', ')}, which "
                             f"the other database uses)"),
                    ("drop", f"drop it: all phases active (pycalphad ignores REJECT_PHASE anyway, so the diagrams "
                             f"here do not change)")]
            out[cid] = Needed(f"step{k}:reject:{db}:{phs}", k, pair, c.kind, cid, c.question, opts)
        else:
            q = c.question
            opts = [(o.key, o.label + (f" [{clip(o.detail, 90)}]" if o.detail else "")) for o in c.options]
            out[cid] = Needed(f"step{k}:{cid}", k, pair, c.kind, cid, q, opts)
    # phase harmonisation: decided extensions, and B phases that look like a base phase with extra constituents
    for tgt, spec in (sp.get("specs") or {}).items():
        rep = next((r for r in (sp.get("ext") or {}).get("reports", []) if r["base_phase"] == tgt), None)
        q = (f"{tgt}: extend {a_label}'s {tgt} with {b_label}'s {spec['phase']} ("
             + (rep["basis"]["text"] if rep else "") + ")")
        out[f"phase:{tgt}"] = Needed(f"step{k}:phase:{tgt}", k, pair, "harmonize", f"phase:{tgt}", q,
                                     [("extend", _extend_consequence(rep, a_label, b_label) if rep else "extend"),
                                      ("separate", "two phases")])
    for n in sp["notes"]:
        if n.action == "unmapped":
            out[f"sublattice:{n.phase}"] = Needed(
                f"step{k}:sublattice:{n.phase}", k, pair, "sublattice", f"sublattice:{n.phase}",
                f"{n.phase}: {n.detail.split(' kept as is: ', 1)[-1]}",
                [("keep", f"leave {b_label}'s {n.phase} model as is (then the models may differ: a phase decision "
                          f"follows)"),
                 ("[i, j, ...]", f"{b_label}'s sublattice i goes to {a_label}'s position (as in its PHASE line)")])
    for n in pairing_questions(sp, k, pair, a_label, b_label, ev):
        out.setdefault(n.cid, n)
    return list(out.values())


def _cand_target(detail: str) -> Optional[str]:
    m = re.search(r"~ (?:A's )?([A-Z0-9_]+)", detail)
    return m.group(1) if m else None


def pairing_candidates(sp: Dict, ev) -> List[Dict[str, Any]]:
    """Candidate pairings {a, b, name, extend} between phases NEITHER database has paired yet: phases present in
    both under one name (same names, clear aliases, phase_map) and the phases of an 'extend' decision are paired
    already and never offered again.  Sources: compatibility stage-5 name candidates, the normalisation's
    candidate notes and harmonize.harmonization_candidates (A's phase with B's extra constituents)."""
    A, B = sp.get("A_pre"), sp.get("B_pre")
    if A is None or B is None:
        return []
    specs = sp.get("specs") or {}
    both = set(A.phases) & set(B.phases)
    paired_a = both | set(specs)
    paired_b = both | {x["phase"] for x in specs.values()}
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add(a, b, kind, value):
        if a in B.phases and b in A.phases and a not in A.phases:          # written the other way round
            a, b = b, a
        if a not in A.phases or b not in B.phases or a in paired_a or b in paired_b or a == b:
            return
        e = edges.setdefault((a, b), {"a": a, "b": b, "name": None, "extend": None})
        if e[kind] is None:
            e[kind] = value
    if ev is not None:
        for r in ev.stages[4].rows:
            if r.get("name candidate"):
                t = _cand_target(r["name candidate"])
                if t:
                    add(t, r["phase"], "name", r["name candidate"])
    for n in sp["notes"]:
        if n.action == "candidate":
            t = _cand_target(n.detail)
            if t:
                add(t, n.phase, "name", n.detail)
    for c in sp.get("cands") or []:
        add(c["a_phase"], c["b_phase"], "extend", c)
    return sorted(edges.values(), key=lambda e: (e["a"], e["b"]))


def _group_pairings(edges: List[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    """One question per contested phase: candidates competing for the same A phase (then the same B phase) are
    asked together; the rest one by one.  [(hub side 'A' | 'B' | '', hub phase, edges)]."""
    by_a: Dict[str, List[Dict]] = {}
    by_b: Dict[str, List[Dict]] = {}
    for e in edges:
        by_a.setdefault(e["a"], []).append(e)
        by_b.setdefault(e["b"], []).append(e)
    used, out = set(), []
    for a, es in sorted(by_a.items()):
        if len(es) > 1:
            out.append(("A", a, es))
            used |= {(e["a"], e["b"]) for e in es}
    for b, es in sorted(by_b.items()):
        rest = [e for e in es if (e["a"], e["b"]) not in used]
        if len(rest) > 1:
            out.append(("B", b, rest))
            used |= {(e["a"], e["b"]) for e in rest}
    for e in edges:
        if (e["a"], e["b"]) not in used:
            out.append(("", "", [e]))
    return out


def pairing_questions(sp: Dict, k: int, pair: str, a_label: str, b_label: str, ev) -> List[Needed]:
    """The pairing questions of one step (see pairing_candidates / _group_pairings): 'step{k}:phase:<A phase>'
    (extend / separate) or 'step{k}:name:<B phase>' (same / separate) for a single candidate; for candidates
    competing for one phase a single 'step{k}:name:<that phase>' question whose options name the partner."""
    out = []
    A = sp.get("A_pre")
    for side, hub, es in _group_pairings(pairing_candidates(sp, ev)):
        if not side:
            e = es[0]
            a, b = e["a"], e["b"]
            if e["extend"] is not None:
                c = e["extend"]
                opt = _extend_option(sp, a, b, a_label, b_label, c)
                q = (f"{b_label}'s {b} {c['b_model']} looks like {a_label}'s {a} {c['a_model']} with "
                     f"{', '.join(x.capitalize() for x in c['components'])} ({c['why']}) - the same phase? "
                     f"{c['map_note']}".strip())
                opts = ([opt] if opt else [])
                pairs = {"extend": ("extend", a, b)} if opt else {}
                if e["name"] is not None:
                    opts.append(("same", f"phase_map {{'{b}': '{a}'}}: one phase, models merged"))
                    pairs["same"] = ("same", a, b)
                opts.append(("separate", f"two phases (default): {a_label}'s {a} and {b_label}'s {b}"))
                out.append(Needed(f"step{k}:phase:{a}", k, pair, "harmonize", f"phase:{a}", q, opts, pairs=pairs))
            else:
                out.append(Needed(
                    f"step{k}:name:{b}", k, pair, "name", f"name:{b}",
                    f"is {b_label}'s {b} the same phase as {a_label}'s {a}? ({clip(e['name'], 120)})",
                    [("same", f"phase_map {{'{b}': '{a}'}}: one phase, models merged"),
                     ("separate", "two phases (default)")], pairs={"same": ("same", a, b)}))
            continue
        opts, pairs, why = [], {}, []
        for e in es:
            a, b = e["a"], e["b"]
            other = b if side == "A" else a
            opts.append((other, f"{b_label}'s {b} is {a_label}'s {a}: phase_map {{'{b}': '{a}'}}, one phase, "
                                f"models merged"))
            pairs[other] = ("same", a, b)
            if e["name"] is not None:
                why.append(f"{other}: {clip(e['name'], 90)}")
            if e["extend"] is not None:
                c = e["extend"]
                opt = _extend_option(sp, a, b, a_label, b_label, c)
                if opt:
                    opts.append((f"extend:{other}", opt[1].replace(
                        f"(answer {{'extend': {{'phase': '{b}'}}}}", f"(answer 'extend:{other}'").replace(
                        "(answer 'extend'", f"(answer 'extend:{other}'")))
                    pairs[f"extend:{other}"] = ("extend", a, b)
                if e["name"] is None:
                    why.append(f"{other}: {c['why']}")
        if not pairs:
            continue
        others = sorted({(e["b"] if side == "A" else e["a"]) for e in es})
        hub_db = a_label if side == "A" else b_label
        other_db = b_label if side == "A" else a_label
        model = ""
        if side == "A" and A is not None and hub in A.phases:
            model = " " + ts._subl_text(A.phases[hub])
        q = (f"{hub_db}'s {hub}{model} has {len(others)} candidate partners in {other_db}: {', '.join(others)} - "
             f"which one (if any) is the same phase? A phase pairs with at most one ({'; '.join(why)})")
        opts.append(("separate", f"none: {hub} stays a separate phase (default)"))
        out.append(Needed(f"step{k}:name:{hub}", k, pair, "harmonize" if any(v[0] == "extend" for v in pairs.values())
                          else "name", f"name:{hub}", q, opts, pairs=pairs))
    return out


def _extend_consequence(rep: Optional[Dict], a_label: str, b_label: str) -> str:
    if not rep:
        return "extend"
    return (f"keep {a_label}'s {rep['base_phase']} (structure, names, base-element parameters), add "
            f"{', '.join(rep['components'])} and the {len(rep['imported'])} parameter(s) with them from {b_label}'s "
            f"{rep['add_phase']}; {rep['basis']['text']}"
            + (f"; {len(rep['missing_g'])} new end-member(s) without G" if rep.get("missing_g") else ""))


def _extend_option(sp: Dict, a_phase: str, b_phase: str, a_label: str, b_label: str,
                   cand: Optional[Dict] = None) -> Optional[Tuple[str, str]]:
    """('extend', consequence) when B's phase brings elements A's lacks and the models map (else None)."""
    from .harmonize import basis_choice, is_magnetic, phase_comps, _real
    A, B = sp.get("A_pre"), sp.get("B_pre")
    if A is None or B is None or a_phase not in A.phases or b_phase not in B.phases:
        return None
    pa, pb = A.phases[a_phase], B.phases[b_phase]
    extra = sorted(set(phase_comps(B, b_phase)) - set(phase_comps(A, a_phase)))
    if not extra or len(_real(pa)) != len(_real(pb)):
        return None
    try:
        if ts.sublattice_mapping(pa, pb) is None:
            return None
    except ts.SublatticeMapError:
        pass
    bc = basis_choice(is_magnetic(A, a_phase), is_magnetic(B, b_phase), sum(pa.site_ratios[i] for i in _real(pa)),
                      sum(pb.site_ratios[i] for i in _real(pb)), "base", "auto", a_phase, a_label, b_label)
    how = (f"{{'extend': {{'phase': '{b_phase}'}}}}" if b_phase != a_phase else "'extend'")
    return ("extend", f"keep {a_label}'s {a_phase} model and its {', '.join(sorted(set(phase_comps(A, a_phase)) - {'VA'}))}"
                      f" parameters, add {', '.join(x.capitalize() for x in extra)} with only {b_label}'s parameters that "
                      f"involve {'it' if len(extra) == 1 else 'them'}; {bc['text']} (answer {how}; options: "
                      f"components, sublattice_map, normalization, collision, magnetic)")


def _answer_text(val, n: Needed) -> str:
    """A decision value as the answer of ``n``: 'extend' (or 'extend:<phase>' when the question offers several
    sources), the JSON of a list, else the text (an option key matched case-insensitively)."""
    from .harmonize import is_extend
    keys = {kk.lower(): kk for kk in (n.pairs or {})}
    if is_extend(val):
        multi = [kk for kk in (n.pairs or {}) if kk.startswith("extend:")]
        if isinstance(val, str) and val.strip().lower() in keys:
            return keys[val.strip().lower()]
        if multi and isinstance(val, dict):
            body = val.get("extend") if isinstance(val.get("extend"), dict) else val
            ph = str(body.get("phase") or body.get("source_phase") or "").upper()
            hit = [kk for kk in multi if n.pairs[kk][2] == ph or kk.split(":", 1)[1] == ph]
            if len(hit) == 1:
                return hit[0]
        return "extend"
    if isinstance(val, (list, tuple)):
        return json.dumps(val)
    v = str(val)
    return keys.get(v.strip().lower(), v)


def _pairing_lookup(dec: Dict, k: int, bstem: str, n: Needed):
    """A grouped pairing question answered through the single-candidate ids: 'name:<B phase>' = 'same' (or
    'phase:<A phase>' = extend with that source)."""
    for key, (kind, a, b) in n.pairs.items():
        if kind == "same":
            kk, v = _lookup(dec, k, bstem, f"name:{b}")
            if kk is not None and str(v).strip().lower() == "same":
                return kk, key
    return None, None


def _provisional(n: Needed) -> str:
    keys = [k for k, _ in n.options]
    if n.kind == "harmonize":
        return n.answer if n.answer is not None and str(n.answer).startswith("extend") else "separate"
    if n.kind == "phase_model" and "rename" in keys:
        return "rename"
    if n.kind == "sublattice":
        return "keep"
    if n.kind == "name":
        return "separate"
    if n.kind == "default_command" and "drop" in keys:
        return "drop"
    return "db1" if "db1" in keys else keys[0]


def _policy_pick(n: Needed, policy: str) -> Optional[str]:
    if policy not in ("first", "last"):
        return None
    keys = [k for k, _ in n.options]
    if n.kind in ("sublattice", "name", "default_command", "harmonize"):
        return _provisional(n)
    dbk = [k for k in keys if k.startswith("db")]
    return (dbk[0] if policy == "first" else dbk[-1]) if dbk else keys[0]


def _sublattice_value(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    s = str(v).strip()
    if s.lower() == "keep":
        return "keep"
    nums = re.findall(r"-?\d+", s)
    return [int(x) for x in nums] if nums else s


@dataclass
class StepSpec:
    k: int
    a: str                       # base of this step (source 0 or Q_{k-1})
    b: str                       # source added
    a_label: str
    b_label: str
    phase_map: Dict = field(default_factory=dict)
    sublattice_map: Dict = field(default_factory=dict)
    decisions: Dict[str, str] = field(default_factory=dict)       # staged_merge decisions (plain ids)
    extensions: Dict[str, Dict] = field(default_factory=dict)     # 'extend' decisions {A phase: spec}
    needed: List[Needed] = field(default_factory=list)
    compat: Any = None
    plan_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def pair(self) -> str:
        return f"{self.a_label} + {self.b_label}"


@dataclass
class CombinePlan:
    paths: List[str]
    infos: Dict[str, Dict]
    order: List[str]
    order_reasons: List[str]
    pair_eval: Dict[Tuple[str, str], Any]
    steps: List[StepSpec]
    policy: str = "ask"
    files: Dict[str, str] = field(default_factory=dict)
    tentative: Any = True                        # the tentative-parameter setting (calphad_tools.tentative)

    @property
    def needed(self) -> List[Needed]:
        return [n for s in self.steps for n in s.needed]

    @property
    def pending(self) -> List[Needed]:
        return [n for n in self.needed if n.answer is None]


def plan_combination(paths: Sequence[str], workdir: str, *, base: Optional[str] = None,
                     order: Optional[Sequence[str]] = None, decisions: Optional[Dict] = None,
                     decision_reasons: Optional[Dict] = None, phase_map=None, sublattice_map=None,
                     policy: str = "ask", unary50: Optional[str] = "auto", phase_harmonization=None,
                     tentative=True, progress: Callable[[str], None] = lambda s: None) -> CombinePlan:
    from .compatibility import evaluate_pair, write_xlsx as compat_xlsx
    paths = [str(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("combine_databases needs at least 2 databases")
    if len({_stem(p) for p in paths}) != len(paths):
        raise ValueError("two inputs have the same file name: rename one (stems identify the sources)")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    dec = dict(decisions or {})
    reasons = dict(decision_reasons or {})
    u50 = _locate_unary50_offline(unary50)
    V50 = ts.StageView(parse_tdb(u50), "unary50") if u50 else None
    progress("combine: reading sources")
    infos = {p: source_info(p, V50) for p in paths}
    ranked = sorted(paths, key=lambda p: (_score(infos[p]), -paths.index(p)), reverse=True)
    pair_eval = {}
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            if not set(infos[a]["elements"]) & set(infos[b]["elements"]):
                continue
            progress(f"combine: compatibility {infos[a]['stem']} + {infos[b]['stem']}")
            stem = f"compat_{infos[a]['stem']}+{infos[b]['stem']}"
            ev = evaluate_pair(a, b, unary50=unary50, workdir=str(work / stem), tentative=tentative)
            ev.files["xlsx"] = compat_xlsx(ev, str(work / f"{stem}.xlsx"))
            pair_eval[(a, b)] = ev

    def find(name):
        hit = [p for p in paths if p == name or Path(p).name == name or _stem(p) == _stem(str(name))]
        if not hit:
            raise ValueError(f"{name} is not one of the inputs {[Path(p).name for p in paths]}")
        return hit[0]
    ordered = [find(x) for x in order] if order else None
    if ordered and sorted(ordered) != sorted(paths):
        raise ValueError("order must list every input exactly once")
    seq, why = propose_order(infos, pair_eval, find(base) if base else None, ordered)
    plan = CombinePlan(paths, infos, seq, why, pair_eval, [], policy)
    plan.tentative = tentative
    from .harmonize import is_extend
    harm = {str(t).upper(): v for t, v in (phase_harmonization or {}).items()}
    a = seq[0]
    acc = [infos[seq[0]]["stem"]]
    for k, b in enumerate(seq[1:], 1):
        bstem = infos[b]["stem"]
        a_label = acc[0] if k == 1 else f"Q{k - 1} ({'+'.join(acc)})"
        st = StepSpec(k, a, b, a_label, bstem)
        pm = {kk.upper(): vv.upper() for kk, vv in _per_pair(phase_map, k, bstem).items()}
        sm = {kk.upper(): vv for kk, vv in _per_pair(sublattice_map, k, bstem).items()}
        user_plain = {}
        for key, v in dec.items():
            m = re.match(r"^(step\d+):(.+)$", str(key))
            if m:
                if m.group(1) == f"step{k}":
                    user_plain[m.group(2)] = v
            elif str(key).lower().startswith(f"{bstem.lower()}:"):
                user_plain[str(key)[len(bstem) + 1:]] = v
            elif ":" in str(key) and not str(key).startswith(("sublattice:", "name:", "plan:")):
                user_plain.setdefault(str(key), v)
        # 'extend' decisions ({'phase:X': 'extend' | {'extend': {...}}}) and phase_harmonization entries
        ext_raw: Dict[str, Any] = {}
        ext_global: List[str] = []
        for kk, vv in user_plain.items():
            if kk.startswith("phase:") and is_extend(vv):
                ext_raw[kk.split(":", 1)[1].upper()] = vv
                if not any(str(x).startswith((f"step{k}:", f"{bstem}:")) for x in dec if str(x).endswith(kk)):
                    ext_global.append(kk.split(":", 1)[1].upper())
        for tgt, vv in harm.items():
            if tgt in ext_raw:
                continue
            st_k = vv.get("step") if isinstance(vv, dict) else None
            src = vv.get("source") if isinstance(vv, dict) else None
            if (st_k is not None and str(st_k).replace("step", "") != str(k)) or (
                    src is not None and _stem(str(src)).lower() != bstem.lower()):
                continue
            ext_raw[tgt] = {"extend": {kx: x for kx, x in vv.items() if kx not in ("step", "source")}} \
                if isinstance(vv, dict) else vv
            if st_k is None and src is None:
                ext_global.append(tgt)
        merge_user = {kk: str(vv) for kk, vv in user_plain.items() if not kk.startswith(("sublattice:", "name:"))
                      and not (kk.startswith("phase:") and is_extend(vv))}
        progress(f"combine: planning step {k} ({st.pair})")
        needed: Dict[str, Needed] = {}
        kept: Dict[str, Needed] = {}             # applied answers ('same', sublattice maps) whose question goes
        ext_who: Dict[str, Needed] = {}          # extensions chosen through a pairing question
        ev = None
        for _ in range(6):          # answers (names, extensions, sublattices) can remove questions or raise new ones
            sp = _step_plan(a, b, pm, sm, merge_user, work / f"step{k}", k, ext_raw, ext_global, tentative)
            ext_pairs = set(sp.get("specs") or {}) | {x["phase"] for x in (sp.get("specs") or {}).values()}
            ev = evaluate_pair(a, b, phase_map=pm or None, sublattice_map=sm or None, unary50=unary50,
                               workdir=str(work / f"step{k}_compat"), tentative=tentative, paired_phases=ext_pairs)
            needed = {}
            for n in _needed_from_plan(sp, k, st.pair, a_label, bstem, ev):
                if n.id in needed:
                    continue
                short = n.id.split(":", 1)[1]
                key, val = _lookup(dec, k, bstem, n.cid, short)
                if key is None and n.pairs:
                    key, val = _pairing_lookup(dec, k, bstem, n)
                src = None
                if key is None and n.kind == "harmonize" and n.cid.split(":", 1)[1] in (sp.get("specs") or {}):
                    key, val = "phase_harmonization", "extend"
                    src = ext_who.get(n.cid.split(":", 1)[1])
                if key is not None:
                    n.answer = _answer_text(val, n)
                    _, rv = _lookup(reasons, k, bstem, n.cid, short)
                    if rv is None and src is not None:
                        _, rv = _lookup(reasons, k, bstem, src.cid, src.id.split(":", 1)[1])
                    if rv is None and key == "phase_harmonization":
                        rv = reasons.get("phase_harmonization")
                    n.who, n.reason = parse_reason(rv)
                    n.reason = n.reason or NOT_GIVEN
                    if src is not None:
                        n.question += f" [chosen in {src.id} = '{src.answer}']"
                else:
                    pick = _policy_pick(n, policy)
                    if pick is not None:
                        n.answer, n.who = pick, "policy"
                        n.reason = (f"policy='{policy}' ("
                                    + ("the earlier database / the base wins" if policy == "first" else
                                       "the later database wins") + ")")
                n.provisional = n.answer if n.answer is not None else _provisional(n)
                needed[n.id] = n
            pm2, sm2, ext2 = dict(pm), dict(sm), dict(ext_raw)
            for n in list(needed.values()) + list(kept.values()):
                ph = n.cid.split(":", 1)[1]
                act = n.pairs.get(str(n.provisional)) if n.pairs else None
                # an applied answer (or provisional value) removes its question from the next round: keep it
                if act is not None and act[0] == "same":
                    pm2[act[2]] = act[1]
                    kept[n.id] = n
                elif act is not None and act[0] == "extend":
                    if act[1] not in ext2:
                        ext2[act[1]] = {"extend": {"phase": act[2]}}
                        ext_who[act[1]] = n
                elif n.kind == "sublattice":
                    sm2[ph] = _sublattice_value(n.provisional)
                    kept[n.id] = n
            if pm2 == pm and sm2 == sm and ext2 == ext_raw:
                break
            pm, sm, ext_raw = pm2, sm2, ext2
        for kid, n in kept.items():
            needed.setdefault(kid, n)
        st.compat = ev
        st.warnings = _unary_warnings(ev, k, a_label, bstem)
        st.phase_map, st.sublattice_map = pm, sm
        st.needed = list(needed.values())
        st.decisions = {**merge_user, **{n.cid: n.provisional for n in st.needed
                                         if n.kind not in ("sublattice", "name", "harmonize")
                                         and n.provisional != "extend"}}
        st.extensions = dict(sp.get("specs") or {})
        st.plan_info = {"shared": sp["shared"], "auto": sp["auto"], "notes": [n.text() for n in sp["notes"]],
                        "alias": sp["alias"],
                        "harmonize": [r["basis"]["text"] for r in (sp.get("ext") or {}).get("reports", [])]}
        # provisional Q_k for planning the next step
        decs = {**sp["auto"], **sp["alias"], **st.decisions}
        pl = ts.plan_merge([sp["a_eff"], sp["b_eff"]], decs, "first", tentative=tentative)
        from .staged_merge import tentative_info
        st.plan_info["tentative"] = tentative_info(pl)
        from .merge import render_tdb
        qp = work / f"step{k}" / f"provisional_Q{k}.tdb"
        qp.write_text(render_tdb(pl))
        plan.steps.append(st)
        acc.append(bstem)
        a = str(qp)
    return plan


def _unary_warnings(ev, k: int, a_label: str, b_label: str) -> List[str]:
    """Unary data of the added database that the staged rule replaces by the base's: decided by base and order."""
    if ev is None:
        return []
    rows = [r for r in ev.stages[0].rows if r.get("status") == "differs"]
    rows2 = [r for r in ev.stages[1].rows if r.get("status") == "differs"]
    if not rows and not rows2:
        return []
    ex = [f"SER of {r['element'].capitalize()} Δ{ts.fmt_energy(r.get('max |Δ| (J/mol)', math.nan))}" for r in rows]
    ex += [f"{r['parameter']} Δ{ts.fmt_delta(r.get('max |Δ|'), r.get('_ptype', 'G'))}" for r in rows2]
    els = sorted(({r["element"] for r in rows} | {e for r in rows2 for e in re.findall(r"[,(:]([A-Z][A-Z]?)[:;)]",
                                                                                      r["parameter"])})
                 & set(ev.shared))
    return [f"step {k}: {b_label}'s unary data differ from {a_label}'s ({', '.join(ex[:3])}"
            + (f", +{len(ex) - 3} more" if len(ex) > 3 else "") + f"): the base's win, so {b_label}'s subsystems with "
            f"{', '.join(e.capitalize() for e in els) or 'those elements'} change - choose another base or order to "
            f"keep {b_label}'s instead"]


def plan_text(plan: CombinePlan, rel=lambda p: p) -> str:
    L = ["Plan: " + " -> ".join(plan.infos[p]["stem"] for p in plan.order)]
    L += [f"  {r}" for r in plan.order_reasons]
    for (a, b), ev in plan.pair_eval.items():
        L.append(f"  compat {plan.infos[a]['stem']} + {plan.infos[b]['stem']}: {clip(ev.overall, 150)}")
    for st in plan.steps:
        L += [f"  NOTE {w}" for w in st.warnings]
        L += [f"  HARMONISE step {st.k}: {h}" for h in st.plan_info.get("harmonize", [])]
        L += [f"  TENTATIVE step {st.k}: {x}" for x in tentative_plan_lines(st.plan_info.get("tentative"))]
    return "\n".join(L)


def tentative_plan_lines(info: Optional[Dict[str, Any]], max_items: int = 6) -> List[str]:
    """What a step's merge will do with tentative parameters: rule 1 and rule 2 terms (values), replacements."""
    if not info:
        return []
    L = []
    for ref, title in (("TENT_MIX", "rule 1 (new mixing, TENT_MIX)"), ("TENT_VA", "rule 2 (vacancies, TENT_VA)")):
        xs = [a for a in info.get("added", []) if a["ref"] == ref]
        if xs:
            L.append(f"{title} will add " + "; ".join(f"{a['label']} = {a['value']:+g}" for a in xs[:max_items])
                     + (f"; +{len(xs) - max_items} more" if len(xs) > max_items else ""))
    rp = info.get("replaced") or []
    if rp:
        L.append("replaces " + "; ".join(r["text"].replace("rule:tentative-replaced: ", "") for r in rp[:max_items]))
    L += list(info.get("notes") or [])[:2]
    return L


def pending_text(plan: CombinePlan, rel=lambda p: p) -> str:
    pend = plan.pending
    L = [f"NOT FINISHED - {len(pend)} decision(s) needed before merging (nothing merged yet). Ask the user ALL of "
         f"them at once, then call again with decisions={{id: option}} and decision_reasons={{id: 'user: <their "
         f"words>'}} (or 'agent: <why>' if you chose).", plan_text(plan, rel)]
    for n in pend:
        L.append(n.text())
    later = [n for n in pend if any(m.step > n.step for m in plan.needed)]
    if later:
        L.append("Later steps were planned assuming: " + ", ".join(f"{n.id} = '{n.provisional}'" for n in later)
                 + " (a different answer can change their questions).")
    answered = [n for n in plan.needed if n.answer is not None]
    if answered:
        L.append("Already answered: " + ", ".join(f"{n.id} = '{n.answer}'" for n in answered))
    return "\n".join(L)


# =============================================================================================== run
def _norm_renames(notes) -> Dict[str, str]:
    """{B's original phase name: base's name} of the 'renamed' normalisation notes."""
    out = {}
    for n in notes:
        if n.action == "renamed":
            m = re.match(r"->\s*([A-Z0-9_]+)", n.detail)
            if m:
                out[n.phase] = m.group(1)
    return out


def run_steps(plan: CombinePlan, out: Path, settings: Dict[str, Any], *, n_workers=None, force_replot=False,
              ternary_temperatures_c=(), progress=lambda s: None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """staged_merge for every step (Q of step k is the base of step k+1).  (results, pending report or None)."""
    from .staged_merge import staged_merge
    results = []
    a = plan.order[0]
    for st in plan.steps:
        st.a = a
        od = out / f"step{st.k}_{plan.infos[st.b]['stem']}"
        progress(f"combine: step {st.k}/{len(plan.steps)}: {st.pair}")
        r = staged_merge(a, st.b, str(od), phase_map=st.phase_map or None, sublattice_map=st.sublattice_map or None,
                         decisions=st.decisions, policy="ask", extensions=st.extensions or None,
                         tentative=plan.tentative,
                         ternary_temperatures_c=list(ternary_temperatures_c),
                         tmin_c=settings["tmin_c"], tmax_c=settings["tmax_c"],
                         composition_step=settings["x_step"], temperature_step=settings["t_step"],
                         ternary_step=settings["tern_step"], n_workers=n_workers, timeout_s=settings["timeout"],
                         force_replot=force_replot, progress=progress)
        if not r.get("written"):
            from .staged_merge import summary_text
            return results, (f"NOT FINISHED at step {st.k} ({st.pair}): the merge still asks (the plan's "
                             f"provisional answers differed from the given ones). Ask the user, then call again "
                             f"with decisions['step{st.k}:<id>'].\n" + summary_text(r))
        r["k"] = st.k
        results.append(r)
        a = r["versions"]["Q"].path
    return results, None


# =============================================================================================== decision log
STEP_KINDS = {("B", "B1"): ("unaries",), ("B1", "B2"): ("unaries", "function"),
              ("B2", "B3"): ("binary", "parameter", "tentative", "tentative-replaced"),
              ("B3", "Q"): ("phase_model", "constituents", "from-base", "binary", "parameter", "typedef", "tentative",
                            "tentative-replaced"),
              ("A", "A1"): ("from-added", "constituents", "tentative", "tentative-replaced"),
              ("A1", "A2"): ("from-added", "constituents", "tentative", "tentative-replaced")}


class StepContext:
    """What one step decided, for attributing its parameter / model changes to decision-log entries."""

    def __init__(self, k: int, st: StepSpec, r: Dict[str, Any], log: DecisionLog, b_label: str, a_label: str):
        self.k, self.st, self.r, self.log = k, st, r, log
        self.tag = f"step {k}: {a_label} + {b_label}"
        self.a_label, self.b_label = a_label, b_label
        self.shared = set(r["shared"])
        self.plan = ts.plan_merge([r["A_eff"], r.get("B_merge") or r["B"]], r["decisions"], "ask",
                                  tentative=r.get("tentative_cfg", True))
        # 'extend' decisions: every change of the extended phase (and of B's source phase) is theirs
        self.extended: Dict[str, str] = {}
        for x in (r.get("extensions") or {}).get("reports", []):
            eid = f"step{k}:phase:{x['base_phase']}"
            self.extended[x["base_phase"]] = eid
            self.extended[x["add_phase"]] = eid
        self.conf = {c.id: c for c in self.plan.conflicts}
        self.groups: Dict[str, List] = {}
        for c in self.plan.conflicts:
            if c.group:
                self.groups.setdefault(c.group, []).append(c)
        self.constituents: Dict[str, List[str]] = {}
        for line in self.plan.log:
            m = re.match(r"phase (\S+): sublattice (\d+) constituents \[(.*)\] added from (.+)$", line)
            if m:
                self.constituents.setdefault(m.group(1), []).append(line)
        self.needed = {n.cid: n for n in st.needed}

    # ------------------------------------------------------------------ ids
    def _decided(self, cid: str, side: str) -> Optional[str]:
        """Entry id of conflict / group ``cid`` if its choice made ``side`` lose (A's value replaced: 'A')."""
        cs = [self.conf[cid]] if cid in self.conf else self.groups.get(cid, [])
        cs = [c for c in cs if c.decision]
        if not cs:
            return None
        lost = any((c.decision != "db1") if side == "A" else (c.decision != "db2") for c in cs)
        return f"step{self.k}:{cid}" if lost else None

    def classify_param(self, p: ts.ParamChange, side: str, X: ts.StageView, Y: ts.StageView) -> Optional[str]:
        k = self.k
        ph = p.phase
        tid = self._tentative_entry(p, X, Y)
        if tid:
            return tid
        if ph in self.extended:
            return self.extended[ph]
        if p.kind == "value":
            if side == "B" and len(p.elements) == 1 and next(iter(p.elements)) in self.shared:
                return self._unary_entry(next(iter(p.elements)))
            grp = "system:" + "-".join(sorted(p.elements))
            for cid in (grp, f"parameter:{p.label}"):
                e = self._decided(cid, side)
                if e:
                    return self._ensure_conflict_entry(cid, e)
            row = (X if side == "B" else X).rows.get(p.key)
            if row is not None and row.raw_pw is not None:
                names = ts._function_closure(X.tdb, row.raw_pw.symbols())
                for n in names:
                    e = self._decided(f"function:{n}", side)
                    if e:
                        return self._ensure_conflict_entry(f"function:{n}", e)
            if side == "A" and len(p.elements) == 1:
                for n in [x for x in self.conf if x.startswith("function:")]:
                    e = self._decided(n, side)
                    if e:
                        return self._ensure_conflict_entry(n, e)
            return None
        if p.kind == "removed":
            e = self._decided(f"phase:{ph}", side)
            return self._ensure_conflict_entry(f"phase:{ph}", e) if e else None
        if p.kind == "added":
            if ph in self.constituents:
                return self._constituent_entry(ph)
            return self._enters_entry(ph, side)
        return None

    def classify_model(self, m: ts.ModelChange, side: str) -> Optional[str]:
        ph = m.phase
        if ph in self.extended:
            return self.extended[ph]
        if m.kind == "phase_added":
            return self._enters_entry(ph, side)
        if m.kind == "phase_removed":
            e = self._decided(f"phase:{ph}", side)
            return self._ensure_conflict_entry(f"phase:{ph}", e) if e else None
        if m.kind == "model":
            e = self._decided(f"phase:{ph}", side)
            if e:
                return self._ensure_conflict_entry(f"phase:{ph}", e)
            if ph in self.constituents:
                return self._constituent_entry(ph)
            return None
        kind = "MAGNETIC" if m.kind == "magnetic" else "DIS_PART"
        e = self._decided(f"typedef:{ph}:{kind}", side)
        if e:
            return self._ensure_conflict_entry(f"typedef:{ph}:{kind}", e)
        eid = f"step{self.k}:rule:typedef:{ph}"
        if side == "B":
            why = (f"the merged {ph} keeps {self.a_label}'s type definitions ({m.new}); {self.b_label}'s {ph} had "
                   f"{m.old}, so {self.a_label}'s {m.kind} model now applies to {self.b_label}'s {ph}"
                   + (" - TC/BMAGN parameters of the phase in these subsystems now take effect"
                      if m.kind == "magnetic" and m.old == "non-magnetic" else ""))
        else:
            why = (f"{self.b_label}'s {ph} type definition ({m.new}) is added to the merged phase (the base had "
                   f"{m.old})")
        e = self.log.rule(eid, self.tag, ph, "merge rule: a phase's type definitions (magnetic, order-disorder) "
                                             "are the base's, plus those only the other database has", "typedef",
                          chosen=f"{m.old} -> {m.new}")
        e.add_evidence(why)
        return eid

    # ------------------------------------------------------------------ tentative parameters
    def _tentative_entry(self, p: ts.ParamChange, X: ts.StageView, Y: ts.StageView) -> Optional[str]:
        """A change made by a tentative term: one added (or kept the larger of two) -> its 'tentative' entry; a
        tentative term replaced by an assessed one or removed -> its 'tentative-replaced' entry."""
        rx, ry = X.rows.get(p.key), Y.rows.get(p.key)
        tx = rx is not None and is_tentative(rx.reference, X.tdb)
        ty = ry is not None and is_tentative(ry.reference, Y.tdb)
        if ty and not tx:
            return self.tentative_added(p.label)
        if tx:
            return self.tentative_replaced(p.label)
        return None

    def tentative_added(self, label: str, term=None) -> str:
        eid = f"step{self.k}:rule:tentative:{label}"
        e = self.log.get(eid)
        if e is None:
            t = term or next((x for x in self.plan.tentative if x.label == label), None)
            ref = t.ref if t is not None else tent.TENT_MIX
            cfg = self.plan.tentative_settings or tent.DEFAULTS
            why = tent.RULE_TEXT[ref].format(mixing_per_atom=cfg.get("mixing_per_atom") or 0,
                                             vacancy=cfg.get("vacancy") or 0)
            chosen = f"{t.value:+g} ({t.ref}, tentative)" if t is not None else "tentative (from an earlier merge)"
            e = self.log.rule(eid, self.tag, label, why, "tentative", chosen=chosen,
                              options=[chosen, "tentative=false: no term (ideal mixing)",
                                       "an assessed value (replaces it automatically when a source has one)"])
            if t is not None:
                e.add_evidence(t.text(), t.evidence)
            else:
                e.add_evidence(f"{label}: tentative term carried into Q{self.k} (TENT_MIX / TENT_VA; e.g. from an "
                               f"'extend' decision or an earlier step)")
        return eid

    def tentative_replaced(self, label: str, rp=None) -> str:
        eid = f"step{self.k}:rule:tentative-replaced:{label}"
        e = self.log.get(eid)
        if e is None:
            rp = rp or next((x for x in self.plan.tentative_replaced if x.label == label), None)
            chosen = (rp.text().replace(f"rule:tentative-replaced: {label} ", "") if rp is not None
                      else "assessed value replaces the tentative term")
            e = self.log.rule(eid, self.tag, label, "rule: " + tent.PRECEDENCE, "tentative-replaced", chosen=chosen,
                              options=["the assessed value (always)", "the tentative value (never offered)"])
            e.add_evidence(rp.text() if rp is not None else f"rule:tentative-replaced: {label}")
        return eid

    # ------------------------------------------------------------------ entries made on demand
    def _unary_entry(self, el: str) -> str:
        eid = f"step{self.k}:rule:unaries:{el}"
        if self.log.get(eid) is None:
            e = self.log.rule(eid, self.tag, f"unaries of {el.capitalize()}",
                              f"staged-merge rule: {self.b_label} takes {self.a_label}'s SER (B1) and unary "
                              f"parameters (B2) of the shared element {el.capitalize()}, so every subsystem uses "
                              f"one set of pure-element data", "unaries", chosen=f"{self.a_label}'s",
                              options=[f"{self.a_label}'s (base)", f"{self.b_label}'s (not offered: the base's "
                                                                     f"unary set is kept)"])
            ev = self.st.compat
            if ev is not None:
                for r in ev.stages[0].rows:
                    if r.get("element") == el:
                        e.add_evidence(f"SER of {el.capitalize()}: {r.get('status')}"
                                       + (f", max |Δ| {ts.fmt_energy(r.get('max |Δ| (J/mol)', math.nan))} at "
                                          f"{r.get('T at max (K)', math.nan):.0f} K" if r.get("status") == "differs"
                                          else "") + (f" ({r['unary50']} unary50)" if r.get("unary50") else ""))
                for r in ev.stages[1].rows:
                    if r.get("status") == "differs" and el.upper() in r.get("parameter", "").upper():
                        d = ts.fmt_delta(r.get("max |Δ|"), r.get("_ptype", "G"))
                        e.add_evidence(f"{r['parameter']}: compat max |Δ| {d}"
                                       + (f" ({r['unary50']} unary50)" if r.get("unary50") else ""))
        return eid

    def _ensure_conflict_entry(self, cid: str, eid: str) -> str:
        if self.log.get(eid) is None:
            cs = [self.conf[cid]] if cid in self.conf else self.groups.get(cid, [])
            c = cs[0]
            n = self.needed.get(cid)
            if n is not None:
                self.log.add(_needed_entry(n, self.tag))
            else:
                self.log.add(Entry(eid, self.tag, cid.split(":", 1)[1], [o.key for o in c.options], c.decision or "",
                                   "rule", f"staged-merge rule: the base's ({self.a_label}) {c.kind} is kept "
                                           f"(element data, FUNCTIONs and unaries of shared elements)", c.kind))
        return eid

    def _constituent_entry(self, ph: str) -> str:
        eid = f"step{self.k}:rule:constituents:{ph}"
        if self.log.get(eid) is None:
            lines = self.constituents.get(ph, [])
            e = self.log.rule(eid, self.tag, ph,
                              f"merge rule: the merged {ph} has the union of both databases' constituents; the "
                              f"new end-members get their G from whichever database has it", "constituents",
                              chosen="union of constituents", options=["union (merge rule)",
                                                                       "a phase decision (db1/db2/rename) instead"])
            e.add_evidence(*[re.sub(r"^phase \S+: ", "", x) for x in lines[:3]])
            ev = self.st.compat
            if ev is not None:
                for r in ev.stages[4].rows:
                    if r.get("phase") == ph and r.get("new end-member"):
                        e.add_evidence(f"new end-member {r['new end-member']} in {r['in space of']}'s space "
                                       f"({r.get('affects subsystems', '')}): {r.get('provided by', '')}")
        return eid

    def _enters_entry(self, ph: str, side: str) -> str:
        who_from = self.a_label if side == "B" else self.b_label
        into = self.b_label if side == "B" else self.a_label
        eid = f"step{self.k}:rule:{'from-base' if side == 'B' else 'from-added'}:{ph}"
        if self.log.get(eid) is None:
            self.log.rule(eid, self.tag, ph, f"merge rule: {who_from}'s description of {ph} (phase and parameters) "
                                             f"extends into {into}'s subsystems", "from-base" if side == "B" else
                          "from-added", chosen="union of phases (merge rule)")
        return eid


def _needed_entry(n: Needed, tag: str) -> Entry:
    who = n.who or "agent"
    reason = n.reason or NOT_GIVEN
    return Entry(n.id, tag, n.id.split(":", 1)[1].replace("reject:", "REJECT_PHASE ") if n.kind == "default_command"
                 else (n.cid.split(":", 1)[-1] if ":" in n.cid else n.cid),
                 [f"{k}: {c}" for k, c in n.options], str(n.answer), who, reason, n.kind,
                 evidence=[f"question: {n.question}"])


def _delta_text(p: ts.ParamChange, X: ts.StageView, Y: ts.StageView) -> str:
    from .pwinline import evaluate
    rx, ry = X.rows.get(p.key), Y.rows.get(p.key)
    if p.kind != "value" or rx is None or ry is None or rx.segs is None or ry.segs is None:
        return f"{p.label} {p.kind}"
    try:
        d = evaluate(ry.segs, EVIDENCE_T)[0] - evaluate(rx.segs, EVIDENCE_T)[0]
    except Exception:  # noqa: BLE001 - evidence only
        return p.text()
    if p.ptype == "G":
        vals = "/".join(f"{x / 1000:+.3g}" for x in d)
        return f"{p.label}: Δ {vals} kJ/mol at {'/'.join(f'{t:g}' for t in EVIDENCE_T)} K"
    return f"{p.label}: Δ " + "/".join(f"{x:+.4g}" for x in d) + f" at {'/'.join(f'{t:g}' for t in EVIDENCE_T)} K"


def attribute_step(sc: StepContext, X: ts.StageView, Y: ts.StageView, side: str, subsystems,
                   attr: Dict[str, Dict[str, List[str]]], unexplained: List[str]) -> None:
    """Classify every change X -> Y (one step, one side) into decision-log entries; fill the attribution map
    {'params': {key: {id: [subsystems]}}, 'models': {'PHASE|kind': {id: [subsystems]}}}."""
    d = ts.diff_versions(X, Y, subsystems)
    seen_p, seen_m = {}, {}
    for s in d.subsystems:
        for p in s.params:
            if p.key not in seen_p:
                seen_p[p.key] = sc.classify_param(p, side, X, Y)
            eid = seen_p[p.key]
            if eid:
                subs = attr["params"].setdefault(p.key, {}).setdefault(eid, [])
                if s.name not in subs:
                    subs.append(s.name)
                e = sc.log.get(eid)
                e.add_affected(s.name)
                e.add_evidence(_delta_text(p, X, Y))
            else:
                unexplained.append(f"{sc.tag}: {p.text()} in {s.name} ({side} side) not attributed")
        for m in s.models:
            mk = f"{m.phase}|{m.kind}"
            if mk not in seen_m:
                seen_m[mk] = sc.classify_model(m, side)
            eid = seen_m[mk]
            if eid:
                subs = attr["models"].setdefault(mk, {}).setdefault(eid, [])
                if s.name not in subs:
                    subs.append(s.name)
                e = sc.log.get(eid)
                e.add_affected(s.name)
                e.add_evidence(f"{s.name}: {m.text()}")
            else:
                unexplained.append(f"{sc.tag}: {m.text()} in {s.name} ({side} side) not attributed")
        for note in s.notes:
            ph = note.split(":", 1)[0]
            if " read as " in note:
                eid = f"step{sc.k}:norm:{ph}:snapped"
                e = sc.log.get(eid) or sc.log.add(Entry(eid, sc.tag, ph, [], "exact fractions", "rule", SNAP_REASON,
                                                        "normalisation"))
            else:
                eid = f"step{sc.k}:rule:exact-rescale:{ph}"
                e = sc.log.rule(eid, sc.tag, ph, "an exact proportional rescale (G/L scaled alike, no magnetic terms) "
                                                 "is not a change", "rescale")
            e.add_affected(s.name)
            e.add_evidence(note)


def build_decision_log(plan: CombinePlan, results: List[Dict[str, Any]], log: DecisionLog,
                       base_given: bool, order_given: bool, reasons: Dict) -> Tuple[Dict, List[str], List[StepContext]]:
    """The decision log of the whole chain + the attribution map.  Returns (attr, unattributed changes, contexts)."""
    who_b, why_b = parse_reason(reasons.get("plan:base") or reasons.get("base"))
    who_o, why_o = parse_reason(reasons.get("plan:order") or reasons.get("order"))
    b0 = plan.infos[plan.order[0]]
    log.add(Entry("plan:base", "plan", b0["name"], [plan.infos[p]["name"] for p in plan.paths], b0["name"],
                  (who_b if base_given or order_given else "rule"),
                  (why_b or NOT_GIVEN) if (base_given or order_given) else
                  "proposed: " + plan.order_reasons[0].split(": ", 1)[1], "plan",
                  evidence=[plan.order_reasons[0]] + [w for st in plan.steps for w in st.warnings]))
    log.add(Entry("plan:order", "plan", " -> ".join(plan.infos[p]["stem"] for p in plan.order),
                  [], " -> ".join(plan.infos[p]["stem"] for p in plan.order),
                  who_o if order_given else "rule", (why_o or NOT_GIVEN) if order_given else
                  "proposed: most shared elements with the sources already in, then fewest conflicts",
                  "plan", evidence=plan.order_reasons[1:]))
    attr: Dict[str, Dict[str, List[str]]] = {"params": {}, "models": {}}
    unexplained: List[str] = []
    contexts = []
    for st, r in zip(plan.steps, results):
        k = st.k
        sc = StepContext(k, st, r, log, st.b_label, st.a_label)
        contexts.append(sc)
        tag = sc.tag
        # the caller's / policy decisions
        for n in st.needed:
            e = log.add(_needed_entry(n, tag))
            if n.kind == "phase_model":
                ph = n.cid.split(":", 1)[1]
                ren = (r.get("phase_renames") or {}).get(ph)
                if ren:
                    e.add_evidence(f"{st.b_label}'s {ph} kept as {ren} in Q{k}")
                    for sub, lab in (r.get("renamed_in") or {}).items():
                        e.add_affected(sub)
            if n.kind == "binary":
                e.add_affected(_sysname(n.cid.split(":", 1)[1].split("-")))
        # 'extend' decisions: the mapping, basis, imported parameters, collisions, missing end-members, GM checks
        for x in (r.get("extensions") or {}).get("reports", []):
            eid = f"step{k}:phase:{x['base_phase']}"
            who, why = parse_reason(reasons.get("phase_harmonization"))
            e = log.get(eid) or log.add(Entry(eid, tag, x["base_phase"], ["extend", "separate"], "extend", who,
                                              why or NOT_GIVEN, "harmonize"))
            e.kind = "harmonize"
            e.add_evidence(*[ln for ln in x["text"].split("\n") if ln], limit=30)
        # automatic (staged rule) conflicts and alias separation
        for c in sc.plan.conflicts:
            cid = c.group if (c.kind == "parameter" and c.group) else c.id
            if cid in sc.needed or not c.decision:
                continue
            eid = f"step{k}:{cid}"
            if log.get(eid) is not None:
                continue
            if c.kind == "phase_alias":
                log.add(Entry(eid, tag, cid.split(":", 1)[1], [o.key for o in c.options], c.decision, "rule",
                              "phase names are unified only through normalisation or phase_map: phases with alias "
                              "names stay separate", "alias"))
            elif c.kind == "parameter" and c.group and len(c.group.split(":", 1)[1].split("-")) == 1:
                continue                     # unary groups: the unaries entries below
            else:
                e = log.add(Entry(eid, tag, cid.split(":", 1)[1], [o.key for o in c.options], c.decision, "rule",
                                  f"staged-merge rule: the base's ({st.a_label}) {c.kind.replace('_', ' ')} is kept "
                                  f"(element data, FUNCTIONs and unaries of shared elements)", c.kind))
                e.add_evidence(*[f"{o.key}: {clip(o.detail, 110)}" for o in c.options if o.detail][:2])
        # normalisation
        for nn in r["notes"]:
            act = {ts.A_RESCALED: "magnetic-exception", ts.A_SNAPPED: "snapped"}.get(nn.action, nn.action)
            if nn.action == "candidate":
                continue
            eid = f"step{k}:norm:{nn.phase}:{act}"
            reason = {
                "renamed": f"{st.b_label}'s phase name adopted to {st.a_label}'s (same structure tokens, or "
                           f"phase_map)",
                "reordered": f"{st.b_label}'s sublattices put in {st.a_label}'s order - flipped / permuted to match "
                             f"(exact: GM identical)",
                "snapped": SNAP_REASON,
                "rescaled": f"{st.b_label}'s formula unit rescaled to {st.a_label}'s, G/L x k (exact: GM per mole "
                            f"of atoms identical)",
                "magnetic-exception": f"magnetic exception: {st.b_label}'s phase is magnetic with ~1 atom/fu, "
                                      f"{st.a_label}'s is not - {st.a_label}'s phase rescaled to {st.b_label}'s "
                                      f"site ratios (exact)",
                "not_rescaled": "not rescaled: magnetic parameters (the IHJ term is per formula unit), models "
                                "stay different",
                "incompatible": "different models: kept as is; the merge decides (phase decision)",
                "kept": "sublattice_map 'keep': B's model left unchanged",
                "unmapped": "sublattice correspondence undecidable",
            }.get(act, act)
            who = "rule"
            if "(phase_map)" in nn.detail:
                n = sc.needed.get(f"name:{nn.phase}") or next(
                    (x for x in sc.needed.values() if x.pairs and x.pairs.get(str(x.answer), ("",))[0] == "same"
                     and x.pairs[str(x.answer)][2] == nn.phase), None)
                who = n.who if n else parse_reason(reasons.get("phase_map"))[0]
            chosen = {"renamed": nn.detail.split(" (")[0], "reordered": "reordered", "rescaled": "rescaled",
                      "snapped": "exact fractions",
                      "magnetic-exception": "base rescaled", "not_rescaled": "not rescaled",
                      "incompatible": "kept as is", "kept": "kept as is", "unmapped": "kept as is"}.get(act, act)
            log.add(Entry(eid, tag, nn.phase, [], chosen, who, reason, "normalisation", evidence=[nn.detail]))
        # tentative parameters: one rule entry per term the merge added and per replacement
        for t in sc.plan.tentative:
            sc.tentative_added(t.label, t)
        for rp in sc.plan.tentative_replaced:
            if rp.kind == "both":
                e = log.get(sc.tentative_added(rp.label))
                e.add_evidence(rp.text())
            else:
                sc.tentative_replaced(rp.label, rp)
        # unary / SER replacement (rule) - evidence from the compatibility rows and the B1/B2 build logs
        for el in sorted(sc.shared):
            ser = [x for x in r["versions"]["B1"].log if x.startswith(f"{el}:")]
            uns = [x for x in r["versions"]["B2"].log if f",{el}" in x.upper() or f"({el}" in x.upper()]
            if any("now A's" in x for x in ser) or uns:
                e = log.get(sc._unary_entry(el))
                e.add_evidence(*(ser + uns)[:4])
        # attribution: B side (B_eff -> Q) and A side (A -> Q)
        Bv = parse_tdb(r["B"])
        ren = ts.stage_renames(Bv, {**(r.get("phase_renames") or {}), **(r.get("extension_renames") or {})})
        EB = sorted(Bv.real_elements)
        Q = parse_tdb(r["versions"]["Q"].path)
        XB = ts.StageView(Bv, "B", elements=EB, rename=ren)
        YB = ts.StageView(Q, "Q", elements=EB)
        attribute_step(sc, XB, YB, "B", ts.stage_subsystems(EB, (2, 3)), attr, unexplained)
        Av = parse_tdb(r["A"])
        EA = sorted(Av.real_elements)
        XA = ts.StageView(Av, "A", elements=EA)
        YA = ts.StageView(Q, "Q", elements=EA)
        attribute_step(sc, XA, YA, "A", ts.stage_subsystems(EA, (2, 3)), attr, unexplained)
        # staged evidence: invariant shifts per transition, forced replots, end-members without G
        _staged_evidence(sc, r, log)
    return attr, unexplained, contexts


def _staged_evidence(sc: StepContext, r: Dict[str, Any], log: DecisionLog) -> None:
    from .staged_merge import _fmt_shift
    k, tag = sc.k, sc.tag
    fig_dir = Path(r["files"].get("xlsx", "")).parent / "figures"
    for st in r.get("steps", []):
        kinds = STEP_KINDS.get((st.old, st.new), ())
        fig = fig_dir / f"compare_{st.old}-{st.new}.png"
        for t, row in st.moved():
            sub = t.split(" @ ")[0]
            for e in log:
                if e.step == tag and e.kind in kinds and (sub in e.affected or not e.affected and e.kind == "unaries"):
                    e.add_evidence(f"{st.title} {_fmt_shift(t, row)}", limit=24)
                    if fig.exists() and str(fig) not in e.refs:
                        e.refs.append(str(fig))
        for it in st.forced_items:
            dT, dx, ok = st.forced_shift(it.name)
            eid = f"step{k}:rule:forced:{it.tag}"
            e = log.rule(eid, tag, it.name, ("replotted although its description is identical (renamed phase) - a "
                                             "check" if "kept as" in st.forced_kind.get(it.name, "") else
                                             "force_replot: an exact-only change mapped again as evidence"),
                         "forced", chosen="replotted")
            e.add_affected(it.name)
            e.add_evidence(st.forced_label(it))
            if fig.exists():
                e.refs.append(str(fig))
        for z in st.zero_g:
            m = re.match(r"(\S+) (.+?): (.*)$", z)
            ph, ems, seen = (m.group(1), m.group(2), m.group(3)) if m else (z.split(" ")[0], "", z)
            eid = f"step{k}:rule:zero-g:{ph}:{ems.replace(', ', ',')}"
            e = log.rule(eid, tag, f"{ph} {ems}", "no database gives G for this new end-member: pycalphad uses G = 0 "
                                                  "(reported, not changed)", "zero-g", chosen="G = 0 (pycalphad)")
            e.add_evidence(seen)
            m2 = re.search(r"mapped (\S+)", seen)
            if m2:
                e.add_affected(m2.group(1))


# =============================================================================================== exactness
def gm_identity(orig: str, orig_phase: str, new: str, new_phase: str, new_from_old: Optional[List[int]] = None,
                n: int = 6, seed: int = 0) -> Dict[str, Any]:
    """Max relative |GM_orig - GM_new| (per mole of atoms) at ``n`` random T and site fractions: the check that a
    normalisation (rename / sublattice permutation / rescale) is exact on the actual data."""
    import warnings
    from pycalphad import Database, Model, variables as v
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dbo, dbn = Database(orig), Database(new)
            ph = dbo.phases[orig_phase]
            comps = sorted({str(el) for subl in ph.constituents for s in subl for el in s.constituents} | {"VA"})
            mo, mn = Model(dbo, comps, orig_phase), Model(dbn, comps, new_phase)
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        return {"max_rel": math.nan, "n": 0, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    nsub = len(dbo.phases[orig_phase].sublattices)
    perm = list(new_from_old) if new_from_old else list(range(nsub))
    new_sym = {(y.sublattice_index, y.species.name): y for y in mn.site_fractions}
    rng = np.random.default_rng(seed)
    worst, worst_abs, Ts = 0.0, 0.0, []
    for _ in range(n):
        T = float(rng.uniform(300.0, 2500.0))
        Ts.append(round(T, 1))
        so, sn = {v.T: T, v.P: 101325.0}, {v.T: T, v.P: 101325.0}
        by_sub: Dict[int, list] = {}
        for y in mo.site_fractions:
            by_sub.setdefault(y.sublattice_index, []).append(y)
        for i, ys in by_sub.items():
            w = rng.dirichlet(np.ones(len(ys))) if len(ys) > 1 else np.ones(1)
            j = perm.index(i) if i in perm else i
            for y, wi in zip(ys, w):
                so[y] = float(wi)
                ny = new_sym.get((j, y.species.name))
                if ny is None:
                    return {"max_rel": math.nan, "n": 0, "error": f"no site fraction {y.species.name} on "
                                                                  f"sublattice {j} of {new_phase} in {Path(new).name}"}
                sn[ny] = float(wi)
        go, gn = float(mo.GM.xreplace(so)), float(mn.GM.xreplace(sn))
        worst = max(worst, abs(go - gn) / max(abs(go), 1.0))
        worst_abs = max(worst_abs, abs(go - gn))
    return {"max_rel": worst, "max_abs": worst_abs, "n": n, "T": Ts}


def exactness_checks(results: List[Dict[str, Any]], log: DecisionLog) -> List[Dict[str, Any]]:
    """GM identity of every normalised / rescaled phase of every step, on the files the merge used."""
    out = []
    for r in results:
        k = r["k"]
        notes = r["notes"]
        ren = _norm_renames(notes)
        inv = {v: kk for kk, v in ren.items()}
        per: Dict[str, Dict[str, Any]] = {}
        for nn in notes:
            if nn.action in ts.B_CHANGES:
                tgt = ren.get(nn.phase, nn.phase) if nn.action == "renamed" else nn.phase
                d = per.setdefault(tgt, {"orig": inv.get(tgt, tgt), "acts": [], "perm": None, "ids": []})
                d["acts"].append(nn.action)
                d["ids"].append(f"step{k}:norm:{nn.phase}:{nn.action}")
                m = re.search(r"B's sublattice order \[([\d, ]+)\]", nn.detail)
                if m:
                    d["perm"] = [int(x) for x in m.group(1).split(",")]
        for tgt, d in per.items():
            if r["B"] == r["B_original"]:
                continue
            orig, tr = r["B_original"], "+".join(d["acts"])
            if "snapped" in d["acts"]:
                # site ratios read as exact fractions: the rest of the transformation is exact against that
                orig, snap_res = _snapped_source(r["B_original"], d["orig"], Path(r["B"]).parent / "_work")
                tr = tr.replace("snapped", "snapped (compared with the exact-fraction source)")
                e = log.get(f"step{k}:norm:{tgt}:snapped")
                if e is not None and snap_res is not None:
                    e.add_evidence(_snap_gm_text(snap_res))
            res = gm_identity(orig, d["orig"], r["B"], tgt, d["perm"])
            out.append({"step": k, "file": f"{_stem(r['B_original'])} -> {Path(r['B']).name}",
                        "phase": f"{d['orig']} -> {tgt}" if d["orig"] != tgt else tgt,
                        "transform": tr, **res})
            for eid in d["ids"]:
                e = log.get(eid)
                if e is not None:
                    e.add_evidence(_gm_text(res))
        a_acts: Dict[str, List[str]] = {}
        for nn in notes:
            if nn.action in ts.A_CHANGES and r.get("A_eff") and r["A_eff"] != r["A"]:
                a_acts.setdefault(nn.phase, []).append(nn.action)
        for ph, acts in a_acts.items():
            orig = r["A"]
            if ts.A_SNAPPED in acts:
                orig, snap_res = _snapped_source(r["A"], ph, Path(r["A_eff"]).parent / "_work")
                e = log.get(f"step{k}:norm:{ph}:snapped")
                if e is not None and snap_res is not None:
                    e.add_evidence(_snap_gm_text(snap_res))
            res = gm_identity(orig, ph, r["A_eff"], ph, None)
            out.append({"step": k, "file": f"{_stem(r['A'])} -> {Path(r['A_eff']).name}", "phase": ph,
                        "transform": " + ".join(("magnetic exception (base rescaled)" if a == ts.A_RESCALED else
                                                 "base site ratios read exactly (compared with the exact-fraction "
                                                 "source)") for a in acts), **res})
            e = log.get(f"step{k}:norm:{ph}:magnetic-exception")
            if e is not None:
                e.add_evidence(_gm_text(res))
        for x in (r.get("extensions") or {}).get("reports", []):
            eid = f"step{k}:phase:{x['base_phase']}"
            ch = x.get("checks") or {}
            rows = []
            if ch.get("base"):
                rows.append((f"extend: {x['base_phase']} on the base's elements unchanged"
                             + (f" (rescaled x{x['rescale_base']['k_text']})" if x["rescale_base"].get("k") not in
                                (None, 1.0) else ""), ch["base"]))
            for a in ch.get("add") or []:
                if a.get("expected_exact"):
                    rows.append((f"extend: B's {x['add_phase']} reproduced on {a.get('system')}", a))
            for tr, res in rows:
                out.append({"step": k, "file": f"{_stem(r['A_eff'])}", "phase": x["base_phase"], "transform": tr,
                            "max_rel": res.get("max_rel", math.nan), "n": res.get("n", 0),
                            **({"error": res["error"]} if res.get("error") else {})})
    for c in out:
        c["ok"] = bool(math.isfinite(c.get("max_rel", math.nan)) and c["max_rel"] <= GM_TOL)
    return out


def _snapped_source(path: str, phase: str, work: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(``path`` with ``phase``'s site ratios read as exact fractions, written to ``work``; GM identity of the snap
    itself) - the reference the rest of a normalisation is exact against."""
    db = parse_tdb(path)
    if phase not in db.phases or not ts.snap_phase_in(db, phase):
        return path, None
    work.mkdir(parents=True, exist_ok=True)
    out = ts.write_tdb(db, str(work / f"{_stem(path)}_{phase}_exact_ratios.tdb"),
                       [f"{Path(path).name} with {phase}'s site ratios read as exact fractions (exactness reference)"])
    return out, gm_identity(path, phase, out, phase, None)


def _snap_gm_text(res: Dict) -> str:
    if not math.isfinite(res.get("max_rel", math.nan)):
        return f"GM change of the snap not checked: {res.get('error', '?')}"
    return (f"the snap alone changes GM (per mole of atoms, {res['n']} random states) by max relative "
            f"{res['max_rel']:.1e} (max {res.get('max_abs', math.nan):.3g} J/mol of atoms): negligible, not a "
            f"question")


def _gm_text(res: Dict) -> str:
    if not math.isfinite(res.get("max_rel", math.nan)):
        return f"GM identity not checked: {res.get('error', '?')}"
    return (f"GM identity (per mole of atoms, {res['n']} random states): max relative error {res['max_rel']:.1e}"
            + (" - exact" if res["max_rel"] <= GM_TOL else " - NOT EXACT"))


# =============================================================================================== validation
def disorder_labels(db, comps, phases) -> Dict[str, Tuple[str, int, Dict[Tuple[int, str], int]]]:
    """Ordered phases with a disordered part (L12_FCC / A1_FCC, B2_BCC / A2_BCC): name a composition set of one by
    its disordered phase when its equivalent sublattices have equal site fractions - the mapping tool's rule
    (A1_FCC -> FCC_A1 when that phase exists), so equal states get equal names in the grid comparison."""
    from pycalphad import Model
    info = {}
    for name in phases:
        ph = db.phases.get(name)
        if ph is None or ph.model_hints.get("ordered_phase") != name:
            continue
        dis = ph.model_hints.get("disordered_phase")
        if not dis or dis not in db.phases:
            continue
        n_sub = len(ph.sublattices) - (len(db.phases[dis].sublattices) - 1)
        if n_sub < 2:
            continue
        m = re.fullmatch(r"([A-Z]\d+)_([A-Z]+)", dis)
        if m and f"{m.group(2)}_{m.group(1)}" in db.phases:
            dis = f"{m.group(2)}_{m.group(1)}"
        try:
            mod = Model(db, comps, name)
        except Exception:  # noqa: BLE001 - keep the plain name
            continue
        info[name] = (dis, n_sub, {(int(y.sublattice_index), str(y.species.name)): j
                                   for j, y in enumerate(mod.site_fractions)})
    return info


def _label(name: str, y, info, tol: float = 1e-4) -> str:
    inf = info.get(name)
    if inf is None:
        return name
    dis, n_sub, idx = inf
    for sp in {sp for (s, sp) in idx if s < n_sub}:
        vals = [y[idx[(s, sp)]] if (s, sp) in idx else 0.0 for s in range(n_sub)]
        if np.ptp(vals) > tol:
            return name
    return dis


def grid_task(tdb: str, elements: Sequence[str], out_json: str, T_c: Optional[float] = None,
              tmin_c: float = 300.0, tmax_c: float = 3000.0, nx: int = 25, nT: int = 24) -> Dict[str, Any]:
    """Equilibrium grid (stable phase sets, disordered states of ordered phases named by their disordered phase) +
    H, S, Cp, G of the stable assemblage at spot points; runs in a worker (diagrams.run_tasks kind 'call')."""
    import warnings
    from pycalphad import equilibrium, variables as v
    from .calc import get_db
    from .diagrams import phases_for
    t0 = time.time()
    db = get_db(tdb)
    els = [e.upper() for e in elements]
    comps = sorted(els) + ["VA"]
    phases = phases_for(db, els)
    out: Dict[str, Any] = {"tdb": str(tdb), "elements": els, "T_c": T_c, "phases": phases}

    def sets(eq, dims):
        ph = eq.Phase.transpose(*dims, "vertex").values
        Y = eq.Y.transpose(*dims, "vertex", "internal_dof").values
        gm = eq.GM.transpose(*dims).values
        lab = np.empty(ph.shape[:-1], dtype=object)
        for idx in np.ndindex(*ph.shape[:-1]):
            lab[idx] = "+".join(sorted(_label(str(p), Y[idx + (k,)], info) for k, p in enumerate(ph[idx]) if str(p)))
        return lab, gm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = disorder_labels(db, comps, phases)
        out["relabelled"] = {k: v[0] for k, v in info.items()}
        if T_c is None:
            xs = np.round(np.linspace(0.02, 0.98, nx), 6)
            Ts = np.round(np.linspace(tmin_c, tmax_c, nT), 4) + KELVIN
            eq = equilibrium(db, comps, phases, {v.X(els[1]): xs, v.T: Ts, v.P: 101325.0, v.N: 1.0})
            lab, gm = sets(eq, ("N", "P", "T", f"X_{els[1]}"))
            lab, gm = lab[0, 0], gm[0, 0]                                   # (T, x)
            pts = [{"x": [float(x)], "T_K": float(T), "set": lab[i, j], "nan": bool(not np.isfinite(gm[i, j]))}
                   for i, T in enumerate(Ts) for j, x in enumerate(xs)]
            span = tmax_c - tmin_c
            spot = [([x], tmin_c + f * span + KELVIN) for f in (0.25, 0.5, 0.75) for x in (0.25, 0.5, 0.75)]
        else:
            xs = np.round(np.linspace(0.02, 0.96, nx), 6)
            T = float(T_c) + KELVIN
            eq = equilibrium(db, comps, phases, {v.X(els[1]): xs, v.X(els[2]): xs, v.T: T, v.P: 101325.0, v.N: 1.0})
            lab, gm = sets(eq, ("N", "P", "T", f"X_{els[1]}", f"X_{els[2]}"))
            lab, gm = lab[0, 0, 0], gm[0, 0, 0]                             # (x1, x2)
            pts = []
            for i, x1 in enumerate(xs):
                for j, x2 in enumerate(xs):
                    if x1 + x2 > 0.98:
                        continue
                    pts.append({"x": [float(x1), float(x2)], "T_K": T, "set": lab[i, j],
                                "nan": bool(not np.isfinite(gm[i, j]))})
            spot = [([1 / 3, 1 / 3], T), ([0.6, 0.2], T), ([0.2, 0.6], T)]
        props = []
        for xx, T in spot:
            cond = {v.T: T, v.P: 101325.0, v.N: 1.0}
            for e, x in zip(els[1:], xx):
                cond[v.X(e)] = float(x)
            try:
                q = equilibrium(db, comps, phases, cond, output=["HM", "SM", "CPM"])
                ph = q.Phase.values.reshape(-1)
                Y = q.Y.values.reshape(len(ph), -1)
                rec = {"x": [float(x) for x in xx], "T_K": float(T),
                       "set": "+".join(sorted(_label(str(p), Y[k], info) for k, p in enumerate(ph) if str(p)))}
                for prop in ("GM", "HM", "SM", "CPM"):
                    rec[prop] = float(np.squeeze(getattr(q, prop).values))
            except Exception as e:  # noqa: BLE001 - one failed point is reported, not fatal
                rec = {"x": [float(x) for x in xx], "T_K": float(T), "error": f"{type(e).__name__}: {e}"}
            props.append(rec)
    out.update({"points": pts, "props": props, "n": len(pts), "nan": sum(p["nan"] for p in pts),
                "seconds": round(time.time() - t0, 1)})
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out))
    return {"png": out_json}


def _grid_key(tdb: str, item, g: Dict) -> str:
    h = hashlib.sha256(_content(tdb).encode())
    h.update(json.dumps([list(item.elements), item.T_c, g], sort_keys=True).encode())
    h.update(b"combine-grid-2")
    return h.hexdigest()


def _key_index(root: Path) -> Dict[str, str]:
    """{content key: png} of every diagram already mapped under ``root`` (staged steps, earlier runs)."""
    idx = {}
    for kf in sorted(root.rglob("*.png.key")):
        png = str(kf)[:-4]
        if Path(png).exists() and Path(png).with_suffix(".json").exists():
            try:
                idx.setdefault(kf.read_text().strip(), png)
            except OSError:
                pass
    return idx


def run_validation_tasks(jobs: List[Dict[str, Any]], out_v: Path, settings: Dict, grid: Dict, key_root: Path,
                         n_workers=None, progress=lambda s: None) -> Dict[Tuple[str, str, str], Dict]:
    """jobs: {'label', 'tdb', 'item', 'kind': 'map' | 'grid'} -> {(label, tag, kind): result}.  Diagrams already
    mapped from identical content anywhere under ``key_root`` are reused (content hash), not recomputed."""
    from .diagrams import run_tasks, _check_note
    from .staged_merge import _content_key, _task
    idx = _key_index(key_root)
    tasks, res = [], {}
    for j in jobs:
        it, lab = j["item"], j["label"]
        k3 = (lab, it.tag, j["kind"])
        if k3 in res:
            continue
        if j["kind"] == "map":
            png = out_v / "figures" / f"{lab}_{it.tag}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            key = _content_key(j["tdb"], it, settings)
            kf = Path(str(png) + ".key")
            if png.exists() and kf.exists() and kf.read_text() == key and png.with_suffix(".json").exists():
                res[k3] = {"ok": True, "png": str(png), "cached": True}
            elif key in idx:
                res[k3] = {"ok": True, "png": idx[key], "cached": True, "reused": idx[key]}
            else:
                t = _task(lab, j["tdb"], it, str(png), settings)
                t["name"] = f"{lab} {it.name} map"
                t["_k"] = (k3, key)
                tasks.append(t)
        else:
            js = out_v / "grid" / f"{lab}_{it.tag}.json"
            key = _grid_key(j["tdb"], it, grid)
            kf = Path(str(js) + ".key")
            if js.exists() and kf.exists() and kf.read_text() == key:
                res[k3] = {"ok": True, "json": str(js), "cached": True}
            else:
                kw = {"target": "calphad_tools.combine:grid_task", "tdb": j["tdb"], "elements": list(it.elements),
                      "out_json": str(js), "T_c": it.T_c, "tmin_c": settings["tmin_c"], "tmax_c": settings["tmax_c"],
                      "nx": grid["nx"] if it.T_c is None else grid["nx_tern"], "nT": grid["nT"]}
                tasks.append({"name": f"{lab} {it.name} grid", "kind": "call", "kwargs": kw, "_k": (k3, key)})
    if tasks:
        progress(f"combine: validation - {len(tasks)} calculation(s)")
        out = run_tasks(tasks, n_workers=n_workers, timeout=settings["timeout"], progress=progress)
        for t, r in zip(tasks, out):
            k3, key = t["_k"]
            if k3[2] == "map":
                ok = r.ok and r.png is not None and Path(r.png).exists()
                res[k3] = {"ok": ok, "png": r.png if ok else None, "error": r.error}
                if ok:
                    Path(r.png + ".key").write_text(key)
            else:
                js = t["kwargs"]["out_json"]
                ok = r.ok and Path(js).exists()
                res[k3] = {"ok": ok, "json": js if ok else None, "error": r.error}
                if ok:
                    Path(js + ".key").write_text(key)
    for k3, r in res.items():
        if k3[2] == "map" and r.get("ok"):
            try:
                data = json.loads(Path(r["png"]).with_suffix(".json").read_text())
                r["invariants"] = data.get("invariants", [])
                r["phases_found"] = [str(p) for p in data.get("phases_found") or []]
                r["hash_seed"] = data.get("hash_seed")
                r["mapping_check"] = _check_note(data.get("mapping_check"))
                r["csv"] = str(Path(r["png"]).with_name(Path(r["png"]).stem + "_boundaries.csv"))
            except (OSError, ValueError):
                r["ok"] = False
                r["error"] = "summary JSON unreadable"
        elif k3[2] == "grid" and r.get("ok"):
            r["data"] = json.loads(Path(r["json"]).read_text())
    return res


def _rn(name: str, name_map: Optional[Dict[str, str]]) -> str:
    b = _base_name(name)
    return (name_map or {}).get(b, b)


def _set_rn(s: str, name_map) -> str:
    return "+".join(sorted(_rn(p, name_map) for p in s.split("+") if p))


def line_arrays(csv: str, els: Sequence[str], ternary: bool, name_map=None) -> List[Tuple[Tuple[str, str], np.ndarray,
                                                                                         np.ndarray]]:
    """[((region, phase), composition coords for distances, plot coords)] of a mapped diagram's ZPF lines."""
    import pandas as pd
    try:
        df = pd.read_csv(csv)
    except (OSError, ValueError):
        return []
    if df.empty:
        return []
    out = []
    for (lid, ph), g in df.groupby(["line", "phase"], sort=False):
        g = g.sort_values("point")
        region = " + ".join(sorted(_rn(p, name_map) for p in str(g["region"].iloc[0]).split(" + ")))
        key = (region, _rn(str(ph), name_map))
        if ternary:
            x1, x2 = g[f"X({els[1]})"].to_numpy(float), g[f"X({els[2]})"].to_numpy(float)
            comp = np.c_[x1, x2]
            plot = np.c_[x1 + 0.5 * x2, x2 * math.sqrt(3) / 2]
        else:
            x, T = g[f"X({els[1]})"].to_numpy(float), g["T_K"].to_numpy(float)
            comp = np.c_[x, T / T_SCALE]
            plot = np.c_[x, T - KELVIN]
        out.append((key, comp, plot))
    return out


def boundary_distance(La, Lb, ternary: bool = False) -> Tuple[float, int]:
    """(max distance between corresponding boundaries (same region and phase), both directions; boundary sets
    present in one diagram only).  Binary: x and T/500 K (points at the pure-element ends and beyond the other
    diagram's traced T range skipped); ternary: composition."""
    from .mapping_check import _dist_to_polyline

    def one(P, Q):
        by: Dict = {}
        for key, comp, _ in Q:
            by.setdefault(key, []).append(comp)
        worst, miss = 0.0, set()
        for key, comp, _ in P:
            other = by.get(key)
            if not other:
                miss.add(key)
                continue
            m = np.ones(len(comp), bool)
            if not ternary:
                # pure-element ends and the parts beyond the other diagram's traced range are tracing end points
                # (the invariants and the grid cover them), not boundary shifts
                lo, hi = min(q[:, 1].min() for q in other), max(q[:, 1].max() for q in other)
                m = (comp[:, 0] > 0.002) & (comp[:, 0] < 0.998) & (comp[:, 1] >= lo - 1e-9) & (comp[:, 1] <= hi + 1e-9)
            if not m.any():
                continue
            d = np.min([_dist_to_polyline(comp[m], q) for q in other], axis=0)
            worst = max(worst, float(np.max(d)))
        return worst, miss
    a, ma = one(La, Lb)
    b, mb = one(Lb, La)
    return max(a, b), len(ma | mb)


def grid_agreement(gs: Dict, gf: Dict, name_map) -> Tuple[float, int, List[str], int]:
    """(fraction of grid points with the same stable phases, points compared, examples of differences, F points
    that did not converge)."""
    ps, pf = gs.get("points", []), gf.get("points", [])
    n, same, ex = 0, 0, []
    for a, b in zip(ps, pf):
        if a["nan"] or b["nan"]:
            continue
        n += 1
        sa, sb = _set_rn(a["set"], name_map), _set_rn(b["set"], None)
        if sa == sb:
            same += 1
        elif len(ex) < 4:
            xs = "/".join(f"{x:.2f}" for x in a["x"])
            ex.append(f"x={xs} T={a['T_K'] - KELVIN:.0f} °C: {sa or '?'} -> {sb or '?'}")
    return (same / n if n else math.nan), n, ex, int(gf.get("nan", 0))


def property_rows(gs: Dict, gf: Dict, name_map) -> List[Dict[str, Any]]:
    rows = []
    for a, b in zip(gs.get("props", []), gf.get("props", [])):
        row = {"x": a["x"], "T_K": a["T_K"], "S phases": _set_rn(a.get("set", ""), name_map),
               "F phases": b.get("set", "")}
        ok = "error" not in a and "error" not in b and row["S phases"] == row["F phases"]
        for p in ("GM", "HM", "SM", "CPM"):
            va, vb = a.get(p, math.nan), b.get(p, math.nan)
            d = vb - va if (math.isfinite(va) and math.isfinite(vb)) else math.nan
            row[f"S {p}"], row[f"F {p}"], row[f"Δ{p}"] = va, vb, d
            if not math.isfinite(d) or abs(d) > PROP_TOL[p] + 1e-7 * max(abs(va), abs(vb)):
                ok = False
        row["equal"] = ok
        rows.append(row)
    return rows


VERDICT_COLOR = {"reproduced": "#1b5e20", "changed, explained": "#e65100", "CHANGED, UNEXPLAINED": "#b71c1c",
                 "predicted, not validated": "#37474f"}
S_STYLE = dict(color="black", lw=1.5, ls="-")
F_STYLE = dict(color="#d62728", lw=1.5, ls=(0, (4, 2.5)))


def _diff_final(run_final: str, final: str) -> Dict[str, Any]:
    """Where F differs from the merge result the run produced (F edited afterwards)."""
    if not run_final or not Path(run_final).exists() or _content(run_final) == _content(final):
        return {"equal": True, "params": set(), "phases": set()}
    A, B = parse_tdb(run_final), parse_tdb(final)
    els = sorted(set(A.real_elements) | set(B.real_elements))
    X, Y = ts.StageView(A, "run"), ts.StageView(B, "F", elements=[e for e in els if e in B.elements])
    keys = {p.key for p in ts.param_changes(X, Y)}
    phases = {ph for ph in set(X.phases) | set(Y.phases) if ts.full_model(X, ph) != ts.full_model(Y, ph)}
    return {"equal": False, "params": keys, "phases": phases}


def _ids_for(m: Dict[str, List[str]], sub: str) -> List[str]:
    """Decision ids of an attribution entry {id: [subsystems]}: those seen in ``sub`` (the same parameter can be
    attributed in several steps and subsystems), else all."""
    hit = [i for i, subs in m.items() if sub in subs]
    return hit or list(m)


def _item_changes(sd: Optional[ts.SubsystemDiff], X, Y, attr, edited) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """[(change text, decision ids)], [unexplained texts] of one subsystem S -> F."""
    changes, unexplained = [], []
    if sd is None:
        return changes, unexplained
    sub = sd.name
    for p in sd.params:
        txt = _delta_text(p, X, Y) if p.kind == "value" else f"{p.label} {p.kind}"
        if p.key in edited["params"]:
            unexplained.append(f"{txt}: F differs from the merge result here (edited after the run?)")
            continue
        ids = _ids_for(attr["params"].get(p.key, {}), sub)
        if ids:
            changes.append((txt, ids))
        else:
            unexplained.append(f"{txt}: no decision explains it")
    for m in sd.models:
        if m.phase in edited["phases"]:
            unexplained.append(f"{m.text()}: F differs from the merge result here (edited after the run?)")
            continue
        ids = _ids_for(attr["models"].get(f"{m.phase}|{m.kind}", {}), sub)
        if not ids:
            ids = sorted({i for k, v in attr["models"].items() if k.split("|")[0] == m.phase
                          for i in _ids_for(v, sub)})
        if ids:
            changes.append((m.text(), ids))
        else:
            unexplained.append(f"{m.text()}: no decision explains it")
    changes.sort(key=lambda c: ": Δ" not in c[0])          # value changes first, then added / removed terms
    return changes, unexplained


RANK_KIND = {"unaries": 1, "function": 1, "binary": 1, "parameter": 1, "phase_model": 1, "constituents": 2,
             "typedef": 2, "tentative": 2, "tentative-replaced": 2, "from-base": 3, "from-added": 3}


def _id_rank(i: str, kinds: Dict[str, Tuple[str, str]]):
    who, kind = kinds.get(i, ("rule", ""))
    return (0 if who != "rule" else RANK_KIND.get(kind, 4), i)


def _inv_summary(rows: List[Dict]) -> Tuple[float, float, int]:
    dT = max([abs(r["dT_K"]) for r in rows if r["status"] in ("moved", "same") and math.isfinite(r["dT_K"])] or [0.0])
    dx = max([r["dx"] for r in rows if r["status"] in ("moved", "same") and math.isfinite(r["dx"])] or [0.0])
    un = sum(r["status"] in ("appeared", "disappeared") for r in rows)
    return dT, dx, un


def draw_overlay(ax, it, s_res, f_res, name_map, s_label: str, row: Dict, settings: Dict, f_label: str = "F") -> None:
    """S's boundaries and invariants black, F's red dashed, on one diagram."""
    from matplotlib.lines import Line2D
    els = list(it.elements)
    tern = it.T_c is not None
    if tern:
        h = math.sqrt(3) / 2
        ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="#888888", lw=0.8)
        ax.text(-0.02, -0.03, els[0].capitalize(), ha="right", va="top", fontsize=13)
        ax.text(1.02, -0.03, els[1].capitalize(), ha="left", va="top", fontsize=13)
        ax.text(0.5, h + 0.02, els[2].capitalize(), ha="center", va="bottom", fontsize=13)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, h + 0.08)
        ax.set_aspect("equal")
        ax.axis("off")
    for res, nm, kw in ((s_res, name_map, S_STYLE), (f_res, None, F_STYLE)):
        if not res or not res.get("ok"):
            continue
        for _, _, plot in line_arrays(res["csv"], els, tern, nm):
            ax.plot(plot[:, 0], plot[:, 1], **kw)
        for inv in res.get("invariants", []):
            comp = inv.get("compositions", {})
            if tern:
                pts = [(c.get(els[1], 0) + 0.5 * c.get(els[2], 0), c.get(els[2], 0) * math.sqrt(3) / 2)
                       for c in comp.values()]
                if len(pts) == 3:
                    ax.plot([p[0] for p in pts + pts[:1]], [p[1] for p in pts + pts[:1]], **{**kw, "lw": 1.0})
            else:
                xs = [c.get(els[1], math.nan) for c in comp.values()]
                if xs:
                    ax.plot([min(xs), max(xs)], [inv["T_C"], inv["T_C"]], **kw)
    if not tern:
        for r in row.get("inv_rows", []):
            if r["status"] == "moved" and math.isfinite(r["dT_K"]) and abs(r["dT_K"]) > TOL_T:
                ax.annotate(f"ΔT {r['dT_K']:+.1f} K", xy=(1.0, r["T_new_K"] - KELVIN),
                            xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                            fontsize=11, color=F_STYLE["color"], va="center", annotation_clip=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(settings["tmin_c"], settings["tmax_c"])
        ax.set_xlabel(f"X({els[1].capitalize()})", fontsize=13)
        ax.set_ylabel("T (°C)", fontsize=13)
        ax.tick_params(labelsize=12)
    v = row.get("verdict", "")
    why = ""
    if row.get("unexplained"):
        why = "\nunexplained: " + clip(row["unexplained"][0].split(": ")[0], 48)
    elif v == "changed, explained" and row.get("explained_by"):
        why = "\nby " + clip(", ".join(row["explained_by"]), 60)
    elif row.get("tentative"):
        why = "\nwith tentative " + clip(", ".join(row["tentative"]), 52)
    ax.set_title(f"{it.name}: {v}{why}", fontsize=14 if not why else 13, color=VERDICT_COLOR.get(v, "black"),
                 fontweight="bold" if v.startswith("CHANGED") else "normal")
    handles = [Line2D([], [], **S_STYLE, label=f"{s_label} (source)"),
               Line2D([], [], **F_STYLE, label=f"{f_label} (combined)")]
    ax.legend(handles=handles, fontsize=11, loc="upper right" if tern else "best", framealpha=0.85)
    if not (s_res or {}).get("ok") or not (f_res or {}).get("ok"):
        ax.text(0.5, 0.5, "diagram not available:\n" + clip(str((s_res or {}).get("error") or
                                                                (f_res or {}).get("error") or "?"), 80),
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#b71c1c")


def overlay_figure(items_rows: List[Tuple[Any, Dict]], res, src: Dict, settings: Dict, out_png: str,
                   ncols: int = 3) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items_rows)
    nc = min(ncols, n)
    nr = -(-n // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 5.0 * nr), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (it, row) in zip(axes.ravel(), items_rows):
        draw_overlay(ax, it, res.get((src["label"], it.tag, "map")), res.get(("F", it.tag, "map")), src["name_map"],
                     src["stem"], row, settings)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _items_of(src: Dict) -> List:
    from .staged_merge import Item
    return [Item(tuple(e), T) for e, T in src["items"]]


def validate(state: Dict[str, Any], final: str, *, n_workers=None, lint: bool = True, only=None,
             progress: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """The proof stage: every source's subsystems reproduced / changed (explained) / CHANGED (unexplained) in F."""
    from .staged_merge import Item, compare_invariants, rename_invariants
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings, grid = state["settings"], state["grid"]
    out = Path(state["out_dir"])
    out_v = out / "validation"
    (out_v / "overlays").mkdir(parents=True, exist_ok=True)
    srcs = [s for s in state["sources"] if not only or s["stem"] in only or s["label"] in only]
    attr = state["attr"]
    edited = _diff_final(state.get("run_final", ""), final)
    jobs = []
    for s in srcs:
        for it in _items_of(s):
            jobs += [{"label": s["label"], "tdb": s["path"], "item": it, "kind": "map"},
                     {"label": s["label"], "tdb": s["path"], "item": it, "kind": "grid"},
                     {"label": "F", "tdb": final, "item": it, "kind": "map"},
                     {"label": "F", "tdb": final, "item": it, "kind": "grid"}]
    extra = extrapolations(state, final)
    vt = [float(t) for t in state.get("validate_temperatures_c") or []]
    predicted = []
    if not only:
        predicted = [Item(tuple(b), None) for b in extra["binaries"]] + \
                    [Item(tuple(t), T) for t in extra["ternaries"] for T in vt]
        jobs += [{"label": "F", "tdb": final, "item": it, "kind": "map"} for it in predicted]
    res = run_validation_tasks(jobs, out_v, settings, grid, out, n_workers, progress)
    rows: List[Dict[str, Any]] = []
    props: List[Dict[str, Any]] = []
    overlays: Dict[str, str] = {}
    Ftdb = parse_tdb(final)
    kinds = {e["id"]: (e["who"], e["kind"]) for e in state.get("log", [])}
    for s in srcs:
        items = _items_of(s)
        if not items:
            continue
        VS = ts.StageView(parse_tdb(s["eff"]), s["stem"], elements=s["elements"], rename=s.get("eff_rename") or {})
        VF = ts.StageView(Ftdb, "F", elements=[e for e in s["elements"] if e in Ftdb.elements])
        d = ts.diff_versions(VS, VF, sorted({it.elements for it in items}))
        src_rows = []
        for it in items:
            sm, fm = res.get((s["label"], it.tag, "map")) or {}, res.get(("F", it.tag, "map")) or {}
            sg, fg = res.get((s["label"], it.tag, "grid")) or {}, res.get(("F", it.tag, "grid")) or {}
            changes, unexplained = _item_changes(d.get(it.elements), VS, VF, attr, edited)
            row: Dict[str, Any] = {"source": s["stem"], "label": s["label"], "subsystem": it.name, "tag": it.tag,
                                   "changes": [c for c, _ in changes], "unexplained": unexplained,
                                   "explained_by": sorted({i for _, ids in changes for i in ids},
                                                          key=lambda i: _id_rank(i, kinds)),
                                   "S_png": sm.get("png"), "F_png": fm.get("png"),
                                   "notes": [n for n in (d.get(it.elements).notes if d.get(it.elements) else [])]}
            diag_ok, why_not = True, []
            if sm.get("ok") and fm.get("ok"):
                inv = compare_invariants(rename_invariants(sm["invariants"], s["name_map"]), fm["invariants"],
                                         list(it.elements), it.T_c is not None, TOL_T, TOL_X)
                dT, dx, un = _inv_summary(inv)
                bd, bun = boundary_distance(line_arrays(sm["csv"], it.elements, it.T_c is not None, s["name_map"]),
                                            line_arrays(fm["csv"], it.elements, it.T_c is not None),
                                            it.T_c is not None)
                row.update({"inv_rows": inv, "inv_max_dT": dT, "inv_max_dx": dx, "inv_unmatched": un,
                            "boundary": bd, "boundary_unmatched": bun, "n_inv": len(inv)})
                if un or dT > TOL_T or dx > TOL_X:
                    diag_ok = False
                    why_not.append(f"invariants: max ΔT {dT:.2f} K, Δx {dx:.4f}, {un} appeared/disappeared")
                if bun or bd > TOL_B:
                    diag_ok = False
                    why_not.append(f"boundaries: max distance {bd:.4f}" + (f", {bun} boundary set(s) in one "
                                                                           f"diagram only" if bun else ""))
            else:
                diag_ok = False
                why_not.append("diagram not compared: " + clip(str(sm.get("error") or fm.get("error") or "?"), 80))
                row.update({"inv_rows": [], "inv_max_dT": math.nan, "inv_max_dx": math.nan, "inv_unmatched": 0,
                            "boundary": math.nan, "boundary_unmatched": 0, "n_inv": 0})
            if sg.get("ok") and fg.get("ok"):
                frac, n, ex, nan_f = grid_agreement(sg["data"], fg["data"], s["name_map"])
                nan_s = int(sg["data"].get("nan", 0))
                row.update({"grid": frac, "grid_n": n, "grid_examples": ex, "grid_nan_F": nan_f, "grid_nan_S": nan_s})
                if not (frac >= TOL_GRID):
                    diag_ok = False
                    why_not.append(f"grid: {100 * frac:.1f} % of {n} points agree")
                if nan_f > nan_s:
                    diag_ok = False
                    bad = [p for p in fg["data"]["points"] if p["nan"]]
                    Tr = (min(p["T_K"] for p in bad) - KELVIN, max(p["T_K"] for p in bad) - KELVIN)
                    why_not.append(f"equilibrium did not converge in F at {nan_f} grid point(s) (source: {nan_s}; "
                                   + (f"{Tr[0]:.0f}-{Tr[1]:.0f} °C)" if it.T_c is None else f"{it.T_c:g} °C)"))
                prs = property_rows(sg["data"], fg["data"], s["name_map"])
                for pr in prs:
                    props.append({"source": s["stem"], "subsystem": it.name, **pr})
                row["props_equal"] = all(pr["equal"] for pr in prs)
                row["props_max"] = {p: max([abs(pr[f"Δ{p}"]) for pr in prs if math.isfinite(pr[f"Δ{p}"])] or [0.0])
                                    for p in ("GM", "HM", "SM", "CPM")}
                if not row["props_equal"]:
                    diag_ok = False
                    why_not.append("properties differ: " + ", ".join(f"Δ{p} {v:.3g}" for p, v in
                                                                   row["props_max"].items() if v > PROP_TOL[p]))
            else:
                diag_ok = False
                why_not.append("grid not compared: " + clip(str(sg.get("error") or fg.get("error") or "?"), 80))
                row.update({"grid": math.nan, "grid_n": 0, "grid_examples": [], "grid_nan_F": 0,
                            "props_equal": False, "props_max": {}})
            if unexplained:
                row["verdict"] = "CHANGED, UNEXPLAINED"
                row["basis"] = "; ".join(unexplained[:3]) + (f"; +{len(unexplained) - 3} more" if len(unexplained) > 3
                                                             else "")
            elif not changes:
                row["verdict"] = "reproduced"
                row["basis"] = ("identical description in F (every parameter numerically equal, same phase models"
                                + ("; exact rescale" if row["notes"] else "") + ")"
                                + ("" if diag_ok else " - the mapping still differs numerically: "
                                   + "; ".join(why_not)))
            elif diag_ok:
                row["verdict"] = "reproduced"
                row["basis"] = (f"{len(changes)} explained change(s) with no visible effect (diagram, grid and "
                                f"properties within tolerance)")
            else:
                row["verdict"] = "changed, explained"
                row["basis"] = "; ".join(why_not)
            row["why_not"] = why_not
            tids = [i for i in row["explained_by"] if ":rule:tentative" in i]
            if tids:
                labels = sorted({i.split(":rule:", 1)[1].split(":", 1)[1] for i in tids})
                row["tentative"] = labels
                row["tentative_note"] = (
                    f"{'replaced / ' if any('tentative-replaced' in i for i in tids) else ''}tentative "
                    f"{', '.join(labels)}: with {'them' if len(labels) > 1 else 'it'} {it.name} is "
                    f"{row['verdict']} - max invariant ΔT {row['inv_max_dT']:.2f} K, Δx {row['inv_max_dx']:.4f}, "
                    f"boundaries {row['boundary']:.4f}, grid "
                    + (f"{100 * row['grid']:.1f} %" if row.get("grid") == row.get("grid") else "—")
                    + " of the source")
            src_rows.append((it, row))
            rows.append(row)
        # one overlay per subsystem + one grid per source
        for it, row in src_rows:
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            draw_overlay(ax, it, res.get((s["label"], it.tag, "map")), res.get(("F", it.tag, "map")), s["name_map"],
                         s["stem"], row, settings)
            p = out_v / "overlays" / f"{s['label']}_{it.tag}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=130, bbox_inches="tight")
            plt.close(fig)
            row["overlay"] = str(p)
        overlays[s["stem"]] = overlay_figure(src_rows, res, s, settings,
                                             str(out_v / "overlays" / f"{s['label']}_{s['stem']}_all.png"))
    rows.sort(key=lambda r: (VERDICTS.index(r["verdict"]), r["source"], r["subsystem"]))
    pred_imgs = [res[("F", it.tag, "map")]["png"] for it in predicted if res.get(("F", it.tag, "map"), {}).get("ok")]
    sanity = sanity_checks(final, state, res, out_v, lint=lint and not only, progress=progress)
    counts = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
    seeds = sorted({str(r.get("hash_seed")) for r in res.values() if r.get("hash_seed") is not None})
    return {"rows": rows, "props": props, "overlays": overlays, "counts": counts, "sanity": sanity,
            "extrapolations": extra, "predicted": pred_imgs, "edited": not edited["equal"], "final": final,
            "hash_seeds": seeds}


def extrapolations(state: Dict, final: str) -> Dict[str, Any]:
    """Subsystems of F no source describes: they are predictions of the combination, not validated."""
    F = parse_tdb(final)
    els = sorted(F.real_elements)
    bins = {tuple(sorted(b)) for s in state["inputs"] for b in s["binaries"]}
    terns = {tuple(sorted(t)) for s in state["inputs"] for t in s["ternaries"]}
    ub = [list(b) for b in ts.stage_subsystems(els, (2,)) if tuple(b) not in bins]
    ut = [list(t) for t in ts.stage_subsystems(els, (3,)) if tuple(t) not in terns]
    higher = (f"all {len(els) - 3 if len(els) > 3 else 0} quaternary and higher combinations are extrapolations"
              if len(els) >= 4 else "")
    return {"binaries": ub, "ternaries": ut, "higher": higher}


def sanity_checks(final: str, state: Dict, res, out_v: Path, lint: bool = True,
                  progress=lambda s: None) -> Dict[str, Any]:
    from .merge import validate as load_check
    out: Dict[str, Any] = {"load": load_check(final)}
    if lint:
        try:
            from .tdb_checks import run_checks
            progress("combine: check_tdb on the combined database")
            lc = run_checks(final, output_prefix=str(out_v / "F_check"), plots=False, time_limit=150.0)
            fs = lc["findings"]
            out["lint"] = {"counts": lc["meta"]["counts"], "report": lc["report_file"],
                           "top": [f"[{f.severity}] {f.check}/{f.code}: {clip(f.message, 150)}" for f in fs
                                   if f.severity in ("ERROR", "WARNING")][:8]}
        except Exception as e:  # noqa: BLE001
            out["lint"] = {"error": f"{type(e).__name__}: {e}"}
    # end-members without G that the combination created (not missing in any source already)
    VF = ts.StageView(parse_tdb(final), "F")
    miss_f = _missing_g(VF)
    miss_src = set()
    for s in state["sources"]:
        VS = ts.StageView(parse_tdb(s["path"]), s["stem"])
        miss_src |= {(s["name_map"].get(ph, ph), em) for ph, em in _missing_g(VS)}
    new = sorted(miss_f - miss_src)
    zero = []
    for ph, em in new:
        els = ts.endmember_elements(VF, em.split(":"))
        where = [(k, r) for k, r in res.items() if k[0] == "F" and k[2] == "map" and r.get("ok")
                 and set(els) <= set(k[1].split("_")[0].upper().split("-"))]
        stable = [k[1] for k, r in where if ph in [_base_name(p) for p in r.get("phases_found", [])]]
        zero.append({"phase": ph, "end-member": em, "mapped": [k[1] for k, _ in where],
                     "stable in": stable,
                     "text": f"{ph} {em}: no G (pycalphad uses 0); " + (
                         f"{ph} is STABLE in the mapped {', '.join(stable)} - check how much {em} it takes there"
                         if stable else
                         (f"not stable in the mapped {', '.join(k[1] for k, _ in where)}" if where else
                          "not in a mapped subsystem"))})
    out["zero_g"] = zero
    nan = {k[1]: r["data"].get("nan", 0) for k, r in res.items() if k[0] == "F" and k[2] == "grid" and r.get("ok")}
    out["convergence"] = {"points": sum(r["data"].get("n", 0) for k, r in res.items() if k[0] == "F"
                                        and k[2] == "grid" and r.get("ok")),
                          "not_converged": sum(nan.values()), "by_subsystem": nan}
    return out


def _missing_g(V: ts.StageView) -> set:
    out = set()
    for ph in V.phases:
        m = ts.full_model(V, ph)
        for em in ts.model_endmembers(m):
            if all(s == "VA" for s in em):
                continue
            if V.key_for("G", ph, em) not in V.rows:
                out.add((ph, ":".join(em)))
    return out


# =============================================================================================== driver
DEFAULT_SETTINGS = {"tmin_c": 300.0, "tmax_c": 3000.0, "x_step": 0.02, "t_step": 10.0, "tern_step": 0.02,
                    "timeout": 900.0}
DEFAULT_GRID = {"nx": 25, "nT": 24, "nx_tern": 17}


def default_output_dir(paths: Sequence[str]) -> str:
    name = "combine_" + "+".join(sorted(_stem(p) for p in paths))
    return name if len(name) <= 90 else "combine_" + hashlib.sha1(name.encode()).hexdigest()[:10]


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)))


def build_state(plan: CombinePlan, results: List[Dict], log: DecisionLog, attr, unattr, exact, out: Path,
                settings: Dict, grid: Dict, ternary_t, validate_t, final: str, run_final: str) -> Dict[str, Any]:
    from .hashseed import child_seed
    base = plan.order[0]
    sources = []
    i0 = plan.infos[base]
    sources.append({"label": "S0", "stem": i0["stem"], "path": base, "eff": base, "eff_rename": {}, "name_map": {},
                    "elements": i0["elements"], "sha256": i0["sha256"], "step": 0})
    for st, r in zip(plan.steps, results):
        info = plan.infos[st.b]
        Bv = parse_tdb(r["B"])
        ren = {**(r.get("phase_renames") or {}), **(r.get("extension_renames") or {})}
        eff_ren = ts.stage_renames(Bv, ren)
        norm = _norm_renames(r["notes"])
        name_map = {}
        for ph in parse_tdb(r["B_original"]).phases:
            n1 = norm.get(ph, ph)
            n2 = ren.get(n1, n1)
            if n2 != ph:
                name_map[ph] = n2
        sources.append({"label": f"S{st.k}", "stem": info["stem"], "path": st.b, "eff": r["B"],
                        "eff_rename": eff_ren, "name_map": name_map, "elements": info["elements"],
                        "sha256": info["sha256"], "step": st.k})
    temps = [float(t) for t in (ternary_t or [])]
    for s in sources:
        info = plan.infos[s["path"]]
        s["items"] = [[list(b), None] for b in info["binaries"]] + \
                     [[list(t), T] for t in info["ternaries"] for T in temps]
    steps = []
    for st, r in zip(plan.steps, results):
        steps.append({"k": st.k, "pair": st.pair, "a_label": st.a_label, "b_label": st.b_label, "A": r["A"],
                      "A_eff": r["A_eff"], "B": r["B"], "B_original": r["B_original"], "Q": r["versions"]["Q"].path,
                      "out_dir": str(Path(r["files"]["xlsx"]).parent), "decisions": st.decisions,
                      "phase_map": st.phase_map, "sublattice_map": _jsonable(st.sublattice_map),
                      "extensions": _jsonable(st.extensions),
                      "phase_renames": r.get("phase_renames") or {}, "shared": r["shared"],
                      "needed": [asdict(n) for n in st.needed], "warnings": st.warnings,
                      "deck": r["files"].get("deck", ""), "tentative": _jsonable(r.get("tentative_cfg", True)),
                      "tentative_terms": r.get("tentative") or {},
                      "xlsx": r["files"]["xlsx"], "images": list(r.get("images") or [])})
    pairs = [{"pair": f"{plan.infos[a]['stem']} + {plan.infos[b]['stem']}", "overall": ev.overall,
              "xlsx": ev.files.get("xlsx", "")} for (a, b), ev in plan.pair_eval.items()]
    return {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "out_dir": str(out),
            "inputs": [plan.infos[p] for p in plan.paths], "inputs_by_order": [plan.infos[p] for p in plan.order],
            "order": plan.order, "order_reasons": plan.order_reasons, "policy": plan.policy, "pairs": pairs,
            "settings": settings, "grid": grid, "ternary_temperatures_c": temps,
            "validate_temperatures_c": [float(t) for t in (validate_t or [])], "steps": steps,
            "sources": sources, "final": str(final), "run_final": str(run_final), "attr": attr,
            "unattributed": unattr, "exactness": _jsonable(exact), "log": [e.to_dict() for e in log],
            "tentative": _jsonable(plan.tentative), "tentative_settings": tent.settings_text(tent.settings(
                plan.tentative)), "hash_seed": child_seed()}


def _write_reports(state: Dict, V: Dict, results: Optional[List[Dict]], out: Path) -> Dict[str, str]:
    from . import combine_report as cr
    from .codegen_combine import write_notebook
    files = {"final TDB": state["final"], "deck": str(out / "combine.pptx"), "xlsx": str(out / "combine.xlsx"),
             "notebook": str(out / "combine.ipynb"), "report": str(out / "combine_report.md"),
             "state": str(out / STATE_FILE)}
    cr.write_deck(state, V, results or [], files["deck"])
    cr.write_xlsx(state, V, files["xlsx"])
    write_notebook(state, files["notebook"])
    cr.write_md(state, V, files, files["report"])
    state["validation"] = {"counts": V["counts"], "rows": [{k: v for k, v in r.items() if k != "inv_rows"}
                                                            for r in V["rows"]],
                           "extrapolations": V["extrapolations"], "sanity": _jsonable(V["sanity"]),
                           "edited": V["edited"], "final": V["final"]}
    state["files"] = files
    Path(files["state"]).write_text(json.dumps(_jsonable(state), indent=1))
    return files


def combine_databases(tdbs: Sequence[str], output_dir: str, *, base: Optional[str] = None,
                      order: Optional[Sequence[str]] = None, decisions: Optional[Dict] = None,
                      decision_reasons: Optional[Dict] = None, phase_map=None, sublattice_map=None,
                      policy: str = "ask", ternary_temperatures_c: Sequence[float] = (),
                      validate_temperatures_c: Optional[Sequence[float]] = None, n_workers: Optional[int] = None,
                      force_replot=False, unary50: Optional[str] = "auto", tmin_c: float = 300.0,
                      tmax_c: float = 3000.0, composition_step: float = 0.02, temperature_step: float = 10.0,
                      ternary_step: float = 0.02, lint: bool = True, phase_harmonization=None,
                      tentative=True, progress: Callable[[str], None] = lambda s: None,
                      on_decisions: Optional[Callable[[List[Dict]], None]] = None) -> Dict[str, Any]:
    """Plan -> (decisions) -> staged merges -> validation -> reports.  Returns {'written': bool, 'text': ...}.
    ``tentative``: True (default) | False | {'mixing_per_atom': 50000, 'vacancy': 200000} (calphad_tools.tentative)."""
    tent.settings(tentative)                    # validates the argument before anything runs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [str(Path(p).resolve()) for p in tdbs]
    settings = {**DEFAULT_SETTINGS, "tmin_c": float(tmin_c), "tmax_c": float(tmax_c), "x_step": float(composition_step),
                "t_step": float(temperature_step), "tern_step": float(ternary_step)}
    grid = dict(DEFAULT_GRID)
    reasons = dict(decision_reasons or {})
    plan = plan_combination(paths, str(out / "plan"), base=base, order=order, decisions=decisions,
                            decision_reasons=reasons, phase_map=phase_map, sublattice_map=sublattice_map,
                            policy=policy, unary50=unary50, phase_harmonization=phase_harmonization,
                            tentative=tentative, progress=progress)
    plan_json = {"order": plan.order, "order_reasons": plan.order_reasons,
                 "tentative": {f"step{st.k}": st.plan_info.get("tentative") for st in plan.steps},
                 "needed": [asdict(n) for n in plan.needed], "pairs": {f"{_stem(a)} + {_stem(b)}": ev.overall
                                                                      for (a, b), ev in plan.pair_eval.items()}}
    (out / "plan" / "combine_plan.json").write_text(json.dumps(plan_json, indent=1, default=str))
    if plan.pending:
        return {"written": False, "plan": plan, "text": pending_text(plan),
                "files": {"plan": str(out / "plan" / "combine_plan.json")}}
    results, pend = run_steps(plan, out, settings, n_workers=n_workers, force_replot=force_replot,
                              ternary_temperatures_c=ternary_temperatures_c, progress=progress)
    if pend:
        return {"written": False, "plan": plan, "text": pend, "files": {}}
    progress("combine: decision log")
    log = DecisionLog()
    attr, unattr, _ = build_decision_log(plan, results, log, base is not None, order is not None, reasons)
    progress("combine: exactness checks (GM identity of normalised phases)")
    exact = exactness_checks(results, log)
    run_final = results[-1]["versions"]["Q"].path
    stems = [plan.infos[p]["stem"] for p in plan.order]
    fname = "combined_" + "+".join(stems)
    final = str(out / ((fname if len(fname) <= 90 else "combined") + ".tdb"))
    state = build_state(plan, results, log, attr, unattr, exact, out, settings, grid, ternary_temperatures_c,
                        validate_temperatures_c, final, run_final)
    from . import combine_report as cr
    cr.write_final_tdb(run_final, final, cr.tdb_header(state, None))
    V = validate(state, final, n_workers=n_workers, lint=lint, progress=progress)
    if V["hash_seeds"]:
        state["hash_seed"] = V["hash_seeds"][0] if len(V["hash_seeds"]) == 1 else "random"
    cr.write_final_tdb(run_final, final, cr.tdb_header(state, V))
    files = _write_reports(state, V, results, out)
    if on_decisions is not None:
        try:
            on_decisions(state["log"])
        except Exception:  # noqa: BLE001 - provenance must never break the tool
            pass
    return {"written": True, "plan": plan, "results": results, "state": state, "validation": V, "files": files,
            "log": log, "text": summary_text(state, V, files)}


def validate_from_state(state_path: str, final_tdb: Optional[str] = None, *, out_dir: Optional[str] = None,
                        n_workers=None, lint: bool = True, only=None, write_reports: bool = True,
                        progress=lambda s: None) -> Dict[str, Any]:
    """Validation only (e.g. after editing the combined TDB by hand): every source against ``final_tdb`` (default
    the combined TDB of the run), with the run's decision log as the only accepted explanations."""
    state = json.loads(Path(state_path).read_text())
    final = str(final_tdb or state["final"])
    if out_dir:
        state = dict(state, out_dir=str(out_dir))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    state["final"] = final
    V = validate(state, final, n_workers=n_workers, lint=lint, only=only, progress=progress)
    files = {}
    if write_reports:
        out = Path(state["out_dir"])
        files = _write_reports(state, V, None, out)
    return {"state": state, "validation": V, "files": files, "text": summary_text(state, V, files)}


def summary_text(state: Dict, V: Dict, files: Dict[str, str], rel=lambda p: p) -> str:
    from . import combine_report as cr
    L = ["combine_databases: " + " -> ".join(Path(p).stem for p in state["order"]),
         clip(state["order_reasons"][0], 200)]
    L.append("Verdicts: " + cr.verdict_line(V["counts"]) + (" (F differs from the merge result: edited?)"
                                                            if V.get("edited") else ""))
    for st in state["steps"]:
        for w in st.get("warnings", [])[:1]:
            L.append("NOTE " + clip(w, 260))
    un = [r for r in V["rows"] if r["verdict"] == "CHANGED, UNEXPLAINED"]
    for r in un[:3]:
        L.append(f"UNEXPLAINED {r['subsystem']} ({r['source']}): {clip(r['basis'], 200)}")
    ex = [r for r in V["rows"] if r["verdict"] == "changed, explained"]
    if ex:
        L.append("Changed, explained: " + "; ".join(
            f"{r['subsystem']} ({r['source']}; {clip(r['basis'].split('; ')[0], 90)}) by "
            f"{', '.join(r['explained_by'][:3])}" for r in ex[:4]))
    L += cr.tentative_summary(state, V)
    fl = cr.flagged(state["log"])
    L.append(f"Decision log: {len(state['log'])} entries ({cr.who_counts(state['log'])})"
             + (f"; reason NOT GIVEN: {', '.join(e['id'] for e in fl[:5])}" if fl else ""))
    L.append("Exact: " + cr.exact_line(state.get("exactness", [])))
    L.append("Combined TDB: " + "; ".join(cr.sanity_lines(V["sanity"])))
    cp = cr.cannot_prove_lines(V["extrapolations"], state.get("validate_temperatures_c") or [])
    if cp:
        L.append("Not validated (no source describes them): " + ", ".join(x.split(":")[0] for x in cp[:8]))
    if files:
        L.append("Files: " + ", ".join(f"{k} {rel(v)}" for k, v in files.items() if k in ("final TDB", "report",
                                                                                         "deck", "xlsx",
                                                                                         "notebook")))
    text = "\n".join(L)
    return text if len(text) <= 2200 else text[:2150].rsplit("\n", 1)[0] + "\n(...; see the report)"


# =============================================================================================== tool
def register(tb) -> None:
    from .staged_merge import TENTATIVE_SCHEMA, tentative_arg
    S = {"type": "string"}
    N = {"type": "number"}
    NUMS = {"type": "array", "items": N}

    @tb.add("combine_databases",
            "Merge N TDBs end to end: decision log, validation against every source. Plan: pairwise "
            "compatibility, proposed base/order with reasons, EVERY decision needed listed once with stable ids "
            "and consequences - ask the user once, re-call with decisions and decision_reasons {id: 'user: why'}. "
            "Then staged merges in order, decision log (who/why/evidence, automatic rules too), validation "
            "(overlays, invariants, boundaries, grid, H/S/Cp: reproduced / explained / UNEXPLAINED), one deck + "
            "Excel + combine_report.md + notebook.",
            {"tdbs": {"type": "array", "items": S, "description": ">= 2 TDBs"},
             "base": S, "order": {"type": "array", "items": S, "description": "full merge order, base first"},
             "decisions": {"type": "object", "description": "{id from the list: option}; 'step2:phase:MU_D85', "
                                                            "global 'phase:X' too; phase options also "
                                                            "'extend' or {'extend': {phase, components, ...}}"},
             "decision_reasons": {"type": "object", "description": "{id: 'user: <their words>' | 'agent: <why>'}; "
                                                                   "also 'plan:base', 'plan:order'"},
             "policy": {"type": "string", "enum": ["ask", "first", "last"],
                        "description": "fill missing decisions (logged as policy); only if the user says so"},
             "phase_map": {"type": "object", "description": "{B phase: A phase} or {'step2': {...}}"},
             "phase_harmonization": {"type": "object", "description": "{A phase: {source_phase, components, "
                                                                    "sublattice_map, ...}}: decision 'extend'"},
             "sublattice_map": {"type": "object"},
             "ternary_temperatures_c": {**NUMS, "description": "isotherms for staged steps + validation; default []"},
             "validate_temperatures_c": {**NUMS, "description": "map F's ternaries no source describes "
                                                                "('predicted, not validated')"},
             "output_dir": S, "n_workers": {"type": "integer"},
             "force_replot": {"anyOf": [{"type": "boolean"}, {"type": "array", "items": S}]},
             "tmin_c": N, "tmax_c": N, "unary50": S,
             "validate_only": {"type": "boolean", "description": "re-validate an existing run (e.g. F edited)"},
             "tentative": TENTATIVE_SCHEMA,
             "final_tdb": {**S, "description": "with validate_only: the TDB to validate"}},
            ["tdbs"])
    def combine_databases_tool(tdbs, base=None, order=None, decisions=None, decision_reasons=None, policy="ask",
                               phase_map=None, sublattice_map=None, ternary_temperatures_c=None,
                               validate_temperatures_c=None, output_dir=None, n_workers=None, force_replot=False,
                               tmin_c: float = 300.0, tmax_c: float = 3000.0, unary50: str = "auto",
                               validate_only: bool = False, final_tdb=None, phase_harmonization=None,
                               tentative=None):
        paths = [tb._find_tdb(t) for t in tdbs or []]
        od = tb.outdir(output_dir, default_output_dir(paths))
        if validate_only:
            sp = od / STATE_FILE
            if not sp.exists():
                return f"no {STATE_FILE} in {tb.rel(str(od))}: run combine_databases first", []
            r = validate_from_state(str(sp), tb._find_tdb(final_tdb) if final_tdb else None,
                                    n_workers=n_workers, progress=tb.ui.progress)
            _record(tb, r["state"]["log"], "combine_databases (validate_only)")
            return summary_text(r["state"], r["validation"], r["files"], tb.rel), \
                list(r["validation"]["overlays"].values())[:3]
        key = "combine|" + "|".join(sorted(paths))
        dec = {**tb.merge_decisions.get(key, {}), **(decisions or {})}
        tb.merge_decisions[key] = dec
        why = {**tb.merge_decisions.get(key + "|reasons", {}), **(decision_reasons or {})}
        tb.merge_decisions[key + "|reasons"] = why
        r = combine_databases(paths, str(od), base=tb._find_tdb(base) if base else None,
                              order=[tb._find_tdb(o) for o in order] if order else None, decisions=dec,
                              decision_reasons=why, phase_map=phase_map, sublattice_map=sublattice_map, policy=policy,
                              ternary_temperatures_c=ternary_temperatures_c or [],
                              validate_temperatures_c=validate_temperatures_c, n_workers=n_workers,
                              force_replot=force_replot, unary50=unary50, tmin_c=tmin_c, tmax_c=tmax_c,
                              phase_harmonization=phase_harmonization, tentative=tentative_arg(tb, tentative),
                              progress=tb.ui.progress,
                              on_decisions=lambda entries: _record(tb, entries, "combine_databases"))
        if not r["written"]:
            return r["text"], []
        return summary_text(r["state"], r["validation"], r["files"], tb.rel), \
            list(r["validation"]["overlays"].values())[:3]


def _record(tb, entries: List[Dict], tool: str) -> None:
    """Decision-log entries into the session provenance (session log + the recorder), for package_feedback /
    show_code."""
    rec = getattr(tb, "provenance", None)
    if rec is not None and hasattr(rec, "record_decisions"):
        rec.record_decisions(entries, tool)
