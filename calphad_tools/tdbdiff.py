"""Line-level diff of two TDB versions: the commands that changed, as a presenter would paste them.

Each version's "lines" are its ELEMENT, SPECIES, FUNCTION, TYPE_DEFINITION, PHASE, CONSTITUENT and PARAMETER
commands, taken from the file's own text from the keyword to the closing ``!`` on one logical line (whitespace
normalised only: runs of blanks collapsed, none inside an expression; functions are not inlined and VV#### names stay
as written).  Comments and the other commands (TEMPERATURE_LIMITS, DEFINE_SYSTEM_DEFAULT, reference lists) are left
out.

Pairing - an old line and its counterpart in the new version share a normalised identity:

* PARAMETER: type (G = L), phase (old name mapped through the rename map), constituent array (sorted within each
  sublattice, sublattices mapped through the old phase's permutation), order and diffusing species;
* FUNCTION: name (old name mapped through the function renames, e.g. GHSERFE -> GHSERFE_2);
* PHASE / CONSTITUENT: the phase (through the rename map); TYPE_DEFINITION: kind and target phase for a
  phase's own definition (magnetic, disordered part), else its text - never the code letter, which a merge
  reassigns;
* ELEMENT / SPECIES: name.

Two lines with one identity whose content is the same apart from formatting (number format, blanks, type-code
letters, a reference key moved from a trailing comment) are unchanged and left out.  Anything else is a pair, an
old (removed) line followed by its new one, also when only a rename or a sublattice reorder changed the text.  Lines
with no counterpart are additions or removals.

Grouping (the same in every step and deck): ELEMENT / SPECIES, FUNCTION, TYPE_DEFINITION, then one group per phase
(alphabetical by the new name; "(was X in B3)" when renamed) with its PHASE and CONSTITUENT lines, then the parameters
by type - G, L of order 0, 1, 2 ..., TC, BMAGN, others - and constituent array.

Only the standard library and calphad_tools.tdbparse are used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .tdbparse import (ExpressionError, TypeDef, _split_commands_ex, parse_param_name, parse_piecewise,
                       resolve_keyword)

KINDS = ("ELEMENT", "SPECIES", "FUNCTION", "TYPE_DEFINITION", "PHASE", "CONSTITUENT", "PARAMETER")
GROUP_ELEMENTS, GROUP_FUNCTIONS, GROUP_TYPEDEFS = "ELEMENT / SPECIES", "FUNCTION", "TYPE_DEFINITION"
LEGEND = "red = removed / old; normal text = added / new; unchanged lines omitted"
NO_CHANGES = "no TDB changes (numerically identical)"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


# =============================================================================================== lines
@dataclass
class Line:
    kind: str                  # one of KINDS
    text: str                  # "PARAMETER G(C15_LAVES,CE:CO;0) 298.15 -12000; 6000 N REFB !"
    ident: Tuple               # normalised identity (new names, new sublattice order)
    canon: str                 # content apart from formatting (names as written)
    group: Tuple               # (rank, name): ordering of the groups
    order: Tuple               # ordering inside the group
    phase: str = ""            # phase as written (PHASE / CONSTITUENT / PARAMETER / a phase's TYPE_DEFINITION)
    perm: Optional[List[int]] = None     # sublattice permutation applied to the identity


def _ws(s: str) -> str:
    return " ".join(s.split())


def _nows(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _num_canon(s: str) -> str:
    def num(m):
        try:
            return repr(float(m.group(0).replace("D", "E").replace("d", "e")))
        except ValueError:
            return m.group(0)
    return re.sub(rf"(?<![A-Za-z_0-9#]){_NUM}", num, s)


def _fmt_body(rest: str) -> Optional[str]:
    """``298.15 +X ; 6000 Y +Y; 6000 N REF`` -> ``298.15 +X; 6000 Y +Y; 6000 N REF`` (no blanks inside expressions);
    None if it does not look like a piecewise body."""
    segs = rest.split(";")
    m = re.match(rf"^\s*(,+|{_NUM})\s*(.*)$", segs[0], re.S)
    if not m:
        return None
    out = [f"{m.group(1)} {_nows(m.group(2))}".rstrip()]
    for seg in segs[1:]:
        if not seg.strip():
            continue
        m = re.match(rf"^\s*(,+|{_NUM})?\s*([YN])\b\s*(.*)$", seg, re.S | re.I)
        if not m:
            m2 = re.match(rf"^\s*(,+|{_NUM})\s*(.*)$", seg, re.S)      # "6000.00 REF" (N omitted)
            if not m2:
                return None
            out.append(_ws(f"{m2.group(1)} {m2.group(2)}"))
            continue
        t = m.group(1) or ""
        if m.group(2).upper() == "Y":
            out.append(_ws(f"{t} {m.group(2)}") + " " + _nows(m.group(3)))
        else:
            out.append(_ws(f"{t} {m.group(2)} {m.group(3)}"))
    return "; ".join(out)


def _display(kw: str, cmd: str) -> str:
    words = cmd.split()
    if kw == "PARAMETER" and ")" in cmd:
        body = cmd.split(None, 1)[1]
        close = body.index(")")
        name, rest = _nows(body[:close + 1]), body[close + 1:]
        b = _fmt_body(rest)
        if b is not None:
            return f"{words[0]} {name} {b} !"
    elif kw == "FUNCTION" and len(words) > 2:
        b = _fmt_body(cmd.split(None, 2)[2])
        if b is not None:
            return f"{words[0]} {words[1]} {b} !"
    return _ws(cmd) + " !"


def renames_from_notes(notes) -> Dict[str, str]:
    """{B's phase name as given: normalised name} from tdbstage 'renamed' notes."""
    out = {}
    for n in notes or []:
        if n.action == "renamed":
            m = re.match(r"->\s*([A-Z0-9_]+)", n.detail)
            if m:
                out[n.phase] = m.group(1)
    return out


def perms_from_notes(notes) -> Dict[str, List[int]]:
    """{phase (normalised name): new_from_old} from tdbstage 'reordered' notes: new sublattice i is old
    sublattice new_from_old[i]."""
    out = {}
    for n in notes or []:
        if n.action == "reordered":
            m = re.search(r"sublattice order \[([\d,\s]+)\]", n.detail)
            if m:
                out[n.phase] = [int(x) for x in m.group(1).split(",")]
    return out


def function_renames(log: Iterable[str]) -> Dict[str, str]:
    """{old: new} of the merge's 'keep_all' function renames ("function GHSERFE: b.tdb copy renamed to GHSERFE_2")."""
    out = {}
    for x in log or []:
        m = re.match(r"function (\S+): .* copy renamed to (\S+)$", str(x))
        if m:
            out[m.group(1).upper()] = m.group(2).upper()
    return out


def _param_rank(ptype: str, cons) -> Tuple[int, str]:
    if ptype == "G":
        return (1, "") if any(len(sl) > 1 for sl in cons) else (0, "")
    return {"TC": (2, ""), "BMAGN": (3, "")}.get(ptype, (4, ptype))


def tdb_lines(path_or_text: str, *, is_text: bool = False, renames: Optional[Dict[str, str]] = None,
              perms: Optional[Dict[str, Sequence[int]]] = None,
              fn_renames: Optional[Dict[str, str]] = None) -> List[Line]:
    """The version's lines with their identities.  ``renames`` {phase as written: new name}, ``perms`` {new name:
    new_from_old} and ``fn_renames`` {function: new name} map this version onto the one it is compared with."""
    text = path_or_text if is_text else Path(path_or_text).read_text(encoding="utf-8", errors="replace")
    cmds, _, trails, _ = _split_commands_ex(text)
    ren = {k.upper(): v.upper() for k, v in (renames or {}).items()}
    fren = {k.upper(): v.upper() for k, v in (fn_renames or {}).items()}
    perms = {k.upper(): list(v) for k, v in (perms or {}).items()}
    tlo, thi = 298.15, 6000.0
    parsed = []
    for cmd, trail in zip(cmds, trails):
        words = cmd.split()
        kw = resolve_keyword(words[0]) if words else None
        if kw == "TEMPERATURE_LIMITS" and len(words) >= 3:
            try:
                tlo, thi = float(words[1]), float(words[2])
            except ValueError:
                pass
        if kw in KINDS:
            parsed.append((kw, cmd, words, trail))
    # type codes -> definition text, so a PHASE line compares by what its codes mean (a merge re-letters them)
    tdefs = {}
    for kw, cmd, words, _ in parsed:
        if kw == "TYPE_DEFINITION" and len(words) > 2:
            tdefs[words[1]] = TypeDef(words[1], " ".join(words[2:]))

    def mp(ph: str) -> str:
        ph = ph.upper()
        return ren.get(ph, ph)

    out: List[Line] = []
    seen: Dict[Tuple, int] = {}
    for kw, cmd, words, trail in parsed:
        disp = _display(kw, cmd)
        phase, perm = "", None
        try:
            if kw in ("ELEMENT", "SPECIES"):
                nm = words[1].upper()
                ident = (kw, nm)
                canon = _num_canon(" ".join(w.upper() for w in words[1:]))
                group, order = (0, GROUP_ELEMENTS), (kw != "ELEMENT", nm not in ("/-", "VA"), nm)
            elif kw == "FUNCTION":
                nm = words[1].upper().rstrip("#")
                body = cmd.split(None, 2)[2] if len(words) > 2 else ""
                pw = parse_piecewise(body, tlo, thi)
                ident = (kw, fren.get(nm, nm))
                canon = f"{nm}|{pw.canonical()}|{_ws(pw.reference).upper()}"
                group, order = (1, GROUP_FUNCTIONS), (ident[1],)
            elif kw == "TYPE_DEFINITION":
                td = TypeDef(words[1], " ".join(words[2:]))
                tp = td.target_phase
                if tp and tp != "@":
                    phase = tp
                    ident = (kw, td.kind, mp(tp))
                else:
                    ident = (kw, td.canonical())
                canon = td.canonical()
                group, order = (2, GROUP_TYPEDEFS), (ident[2] if len(ident) > 2 else "", td.kind, td.canonical())
            elif kw == "PHASE":
                full = words[1].upper()
                phase = full.split(":")[0]
                codes = words[2] if len(words) > 2 else ""
                meaning = sorted(tdefs[c].canonical(phase) if c in tdefs else c for c in codes if c != "%")
                ident = (kw, mp(phase))
                canon = _num_canon(" ".join([full] + ["/".join(meaning)] + words[3:]))
                group, order = (3, mp(phase)), (0,)
            elif kw == "CONSTITUENT":
                phase = words[1].upper().split(":")[0]
                ident = (kw, mp(phase))
                rest = cmd.split(None, 2)[2] if len(words) > 2 else ""
                subl = [sorted(x.rstrip("%") for x in re.split(r"[\s,]+", sl.upper()) if x)
                        for sl in rest.strip().strip(":").split(":")]
                canon = words[1].upper() + "|" + ":".join(",".join(sl) for sl in subl)     # majors (%) not compared
                group, order = (3, mp(phase)), (1,)
            else:                                   # PARAMETER
                body = cmd.split(None, 1)[1]
                close = body.index(")")
                raw = _nows(body[:close + 1])
                key, _, _ = parse_param_name(raw)
                pw = parse_piecewise(body[close + 1:], tlo, thi)
                if not pw.reference and trail and len(trail) <= 40:
                    pw.reference = trail                # "... 6000 N ! $1967Kau" (as tdbparse reads it)
                phase = key.phase
                ph = mp(phase)
                cons = key.constituents
                perm = perms.get(ph)
                if perm is not None and len(perm) == len(cons) and sorted(perm) == list(range(len(cons))):
                    cons = tuple(cons[o] for o in perm)
                else:
                    perm = None
                ident = (kw, key.ptype, ph, cons, key.order, key.diffusing)
                canon = f"{raw.upper()}|{pw.canonical()}|{_ws(pw.reference).upper()}"
                rank = _param_rank(key.ptype, cons)
                group = (3, ph)
                order = (2, rank, key.order if rank[0] == 1 else 0, cons, key.order, key.diffusing)
        except (ValueError, IndexError, ExpressionError):
            ident = (kw, "?", _ws(cmd).upper())
            canon = _ws(cmd).upper()
            group, order = ((3, mp(phase)) if phase else (0 if kw in ("ELEMENT", "SPECIES") else
                                                           1 if kw == "FUNCTION" else 2, kw)), (9, canon)
            if group[0] < 3:
                group = (group[0], (GROUP_ELEMENTS, GROUP_FUNCTIONS, GROUP_TYPEDEFS)[group[0]])
        n = seen.get(ident, 0)
        seen[ident] = n + 1
        if n:
            ident = ident + (("#", n),)
        out.append(Line(kw, disp, ident, canon, group, order, phase, perm))
    return out


# =============================================================================================== diff
@dataclass
class Entry:
    """One change: a removed line, an added line, or a pair (old then new)."""
    old: Optional[Line]
    new: Optional[Line]

    @property
    def change(self) -> str:
        return "changed" if self.old and self.new else "removed" if self.old else "added"

    @property
    def lines(self) -> List[Tuple[str, str]]:
        """[('removed' | 'added', text)] in display order (old first)."""
        return ([("removed", self.old.text)] if self.old else []) + ([("added", self.new.text)] if self.new else [])


@dataclass
class Group:
    key: Tuple
    title: str
    entries: List[Entry] = field(default_factory=list)


@dataclass
class TdbDiff:
    old_label: str
    new_label: str
    old_path: str
    new_path: str
    groups: List[Group] = field(default_factory=list)
    path: str = ""                  # the .diff file, once written
    note: str = ""                  # e.g. "numerically identical within tolerance"

    @property
    def title(self) -> str:
        return f"{self.old_label} → {self.new_label}"

    @property
    def entries(self) -> List[Entry]:
        return [e for g in self.groups for e in g.entries]

    @property
    def empty(self) -> bool:
        return not self.entries

    def counts(self) -> Dict[str, int]:
        es = self.entries
        return {"changed": sum(e.change == "changed" for e in es), "added": sum(e.change == "added" for e in es),
                "removed": sum(e.change == "removed" for e in es),
                "lines": sum(len(e.lines) for e in es)}

    def summary(self) -> str:
        c = self.counts()
        if self.empty:
            return NO_CHANGES
        return (f"{c['changed']} changed, {c['added']} added, {c['removed']} removed line(s)")

    def rows(self) -> List[Tuple[str, str, str]]:
        """[(group title, 'removed' | 'added', line)] - the Excel sheet."""
        return [(g.title, ch, t) for g in self.groups for e in g.entries for ch, t in e.lines]

    def text(self) -> str:
        """Unified-diff style: '@@ group', '-' removed / old, '+' added / new."""
        L = [f"# {self.title}: TDB changes ({LEGEND.replace('red', '-').replace('normal text', '+')})",
             f"--- {self.old_label}  {self.old_path}", f"+++ {self.new_label}  {self.new_path}"]
        if self.note:
            L.insert(1, f"# {self.note}")
        if self.empty:
            L.append(f"# {NO_CHANGES}")
        for g in self.groups:
            L.append(f"@@ {g.title}")
            for e in g.entries:
                L += [("-" if ch == "removed" else "+") + t for ch, t in e.lines]
        return "\n".join(L) + "\n"

    def write(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.text(), encoding="utf-8")
        self.path = str(path)
        return self.path


def read_diff(path: str) -> List[Tuple[str, str, str]]:
    """The rows of a .diff file written by TdbDiff.write: [(group, 'removed' | 'added', line)]."""
    rows, grp = [], ""
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("@@ "):
            grp = ln[3:]
        elif ln.startswith(("--- ", "+++ ", "#")):
            continue
        elif ln.startswith("-"):
            rows.append((grp, "removed", ln[1:]))
        elif ln.startswith("+"):
            rows.append((grp, "added", ln[1:]))
    return rows


def diff_tdb(old: str, new: str, old_label: str = "old", new_label: str = "new", *,
             old_renames: Optional[Dict[str, str]] = None, new_renames: Optional[Dict[str, str]] = None,
             old_perms: Optional[Dict[str, Sequence[int]]] = None,
             old_fn_renames: Optional[Dict[str, str]] = None) -> TdbDiff:
    """Line diff of the TDB files ``old`` -> ``new`` (paths).  ``old_renames`` {old phase name: new name} (a
    normalisation rename, a 'rename' decision, an alias or an extension), ``new_renames`` the same for a new version
    that still has a phase under its old name, ``old_perms`` {new phase name: new_from_old} for sublattices the
    step reordered, ``old_fn_renames`` {old function: new name}."""
    nl = tdb_lines(new, renames=new_renames)
    new_fns = {ln.ident[1] for ln in nl if ln.kind == "FUNCTION"}
    new_phases = {ln.ident[1] for ln in nl if ln.kind == "PHASE"}
    # a function rename applies when the new version has the new name and the old one does not
    ol0 = tdb_lines(old)
    old_fns = {ln.ident[1] for ln in ol0 if ln.kind == "FUNCTION"}
    fren = {o: n for o, n in (old_fn_renames or {}).items() if o.upper() in old_fns and n.upper() in new_fns
            and n.upper() not in old_fns}
    ol = tdb_lines(old, renames=old_renames, perms=old_perms, fn_renames=fren)
    old_by = {ln.ident: ln for ln in ol}
    new_by = {ln.ident: ln for ln in nl}
    entries: List[Tuple[Tuple, Tuple, Entry]] = []
    for ident in list(old_by) + [i for i in new_by if i not in old_by]:
        o, n = old_by.get(ident), new_by.get(ident)
        if o is not None and n is not None and o.canon == n.canon:
            continue
        ref = n or o
        entries.append((ref.group, ref.order + (str(ident),), Entry(o, n)))
    entries.sort(key=lambda t: (t[0], _sortable(t[1])))
    d = TdbDiff(old_label, new_label, str(old), str(new))
    for e_group, _, e in entries:
        if not d.groups or d.groups[-1].key != e_group:
            d.groups.append(Group(e_group, ""))
        d.groups[-1].entries.append(e)
    # group titles: the phase's name in the new version, and what it was called / whether it is new or gone
    old_names: Dict[str, set] = {}
    new_names: Dict[str, set] = {}
    for lines, names in ((ol, old_names), (nl, new_names)):
        for ln in lines:
            if ln.phase and ln.group[0] == 3:
                names.setdefault(ln.group[1], set()).add(ln.phase.upper())
    old_phases = {ln.ident[1] for ln in ol if ln.kind == "PHASE"}
    for g in d.groups:
        rank, name = g.key
        if rank < 3:
            g.title = name
            continue
        notes = []
        shown = name if name in new_names.get(name, {name}) else sorted(new_names[name])[0]
        was = sorted(x for x in old_names.get(name, set()) if x != shown)
        if was:
            notes.append(f"was {', '.join(was)} in {old_label}")
        perm = next((e.old.perm for e in g.entries if e.old is not None and e.old.perm), None)
        if perm:
            notes.append(f"sublattices reordered, new from old {perm}")
        if name in new_phases and name not in old_phases:
            notes.append(f"new in {new_label}")
        elif name in old_phases and name not in new_phases:
            notes.append(f"not in {new_label}")
        g.title = shown + (f" ({'; '.join(notes)})" if notes else "")
    return d


def _sortable(x):
    """Tuples mixing str / int / nested tuples, compared safely."""
    if isinstance(x, tuple):
        return tuple((0, _sortable(v)) if isinstance(v, tuple) else (1, v) if isinstance(v, (int, float))
                     else (2, str(v)) for v in x)
    return x
