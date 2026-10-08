"""Tentative parameters: temporary interaction terms added when TDBs are combined, so each source's phases behave
almost as assessed where the combined model creates a mixing no source describes.

A pair of constituents (i, j) on sublattice k of a phase is *described* when some source has both i and j on that
sublattice of the phase (whatever L terms it gives them - 0 is an assessment too); a source whose only description
of the pair is a tentative term does not count.  For a pair the combination creates and no source describes:

* **rule 1 (TENT_MIX)** - an intermetallic phase (two or more substitutional sublattices, e.g. C14 (Fe)2(Fe,Nb,W),
  sigma, mu): ``L(phase, ...:i,j:...;0) = +mixing_per_atom x atoms per formula unit`` (default 50000 J/mol-atom;
  atoms per formula unit = the sum of the site ratios of the sublattices that are not vacancy-only: 2:1 -> +150000,
  0.667:0.333 -> +50000) for each combination of the other sublattices' constituents where BOTH end-members (with
  i and with j) have G.  Ideal (entropic) mixing would otherwise give e.g. (Fe:Nb) a spurious solubility range.
  Solution phases (one substitutional sublattice: liquid, fcc, bcc, hcp) are not touched: there a pair no source
  describes is a new, unassessed binary, not a change of an assessed one.
* **rule 2 (TENT_VA)** - VA with an element X on a substitutional sublattice (e.g. bcc (Fe,Co,Cr,Ni,Va)(Va)):
  ``L(phase, ...:X,VA:...;0) = +vacancy`` (default 200000 J/mol of formula units, as written) for each combination
  of the other sublattices' constituents (for (...,VA:VA) that is :VA) where X's end-member has G, so X's phase is
  not stabilised by mixing with vacancies.  A missing vacancy end-member G(phase,VA:...) is reported, not invented.

A substitutional ("main") sublattice holds at least one element that is not an interstitial (B, C, H, N, O): in
(Fe,Cr,Ni,Va)(Va) the first sublattice is one, in (Fe)(C,Va) the second is not, so it never triggers rule 2.  Phases
with an order-disorder (DIS_PART) relation or ionic species are left alone.

Every tentative parameter carries the reference key TENT_MIX or TENT_VA, and the written TDB lists them in its
reference list ("Tentative parameter added by calphad_tools when combining <sources> (<date>): ...").
:func:`calphad_tools.tdbparse.is_tentative` recognises the keys and any citation starting with "Tentative".

Precedence: wherever two databases are combined and both have a parameter, an assessed one always replaces a
tentative one - whichever database is the base, whatever the policy or decisions; a tentative term whose pair a
source describes is removed (constituents compared sorted, so key order does not matter).  Both tentative: the larger
value is kept.
"""
from __future__ import annotations

import datetime as _dt
import itertools
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .tdbparse import TENT_MIX, TENT_VA, TENTATIVE_KEYS, Piecewise, _ref_of, is_tentative  # noqa: F401

DEFAULTS = {"mixing_per_atom": 50000.0, "vacancy": 200000.0}
INTERSTITIALS = frozenset({"B", "C", "H", "N", "O"})
RULE_TEXT = {
    TENT_MIX: "rule 1 (TENT_MIX): a pair of constituents the combination puts on a substitutional sublattice of an "
              "intermetallic phase and no source mixes there gets L0 = +{mixing_per_atom:g} J/mol-atom x atoms per "
              "formula unit, where both end-members have G (ideal mixing would otherwise dominate)",
    TENT_VA: "rule 2 (TENT_VA): an element the combination puts with VA on a substitutional sublattice, mixed with "
             "VA by no source, gets L0(X,VA) = +{vacancy:g} J/mol (per formula unit, as written), so the phase is not "
             "stabilised by mixing with vacancies",
}
PRECEDENCE = ("an assessed parameter always replaces a tentative one with the same key, whichever database is the "
              "base and whatever the policy or decisions; a tentative term goes once a source describes its pair")
VA_HINT = ("pycalphad uses 0; the usual choice is G = 0 for the vacancy end-member (e.g. G(BCC_A2,VA:VA;0) = 0 in the "
           "B2 descriptions referenced 01DUP / 01Sun in AlCoCrNi_Liu2016.tdb and CrNi-06Hal.tdb) - give it "
           "explicitly or take it from a source that has it")


def settings(tentative=True) -> Optional[Dict[str, Optional[float]]]:
    """``tentative`` -> {'mixing_per_atom': J/mol-atom or None, 'vacancy': J/mol or None}, or None (off).

    True / None = the defaults; False / 'off' = no tentative terms (precedence still applies); a dict overrides
    the values ({'mixing_per_atom': 30000}; a value of None / False switches that rule off)."""
    if tentative is None or tentative is True:
        return dict(DEFAULTS)
    if tentative is False:
        return None
    if isinstance(tentative, str):
        s = tentative.strip().lower()
        if s in ("false", "off", "no", "0", "none"):
            return None
        if s in ("true", "on", "yes", "1", "default"):
            return dict(DEFAULTS)
        raise ValueError(f"tentative={tentative!r}: true, false or {{'mixing_per_atom': 50000, 'vacancy': 200000}}")
    if isinstance(tentative, dict):
        bad = sorted(set(tentative) - set(DEFAULTS) - {"enabled"})
        if bad:
            raise ValueError(f"tentative: unknown key(s) {bad}; use 'mixing_per_atom' and 'vacancy'")
        if tentative.get("enabled", True) is False:
            return None
        out: Dict[str, Optional[float]] = dict(DEFAULTS)
        for k in DEFAULTS:
            if k in tentative:
                v = tentative[k]
                out[k] = None if v is None or v is False else float(v)
        return out
    raise ValueError(f"tentative={tentative!r}: true, false or {{'mixing_per_atom': 50000, 'vacancy': 200000}}")


def settings_text(cfg: Optional[Dict[str, Optional[float]]]) -> str:
    if cfg is None:
        return "off"
    m, v = cfg.get("mixing_per_atom"), cfg.get("vacancy")
    return (f"rule 1 {'off' if m is None else f'+{m:g} J/mol-atom'}, "
            f"rule 2 {'off' if v is None else f'+{v:g} J/mol'}")


def ref_key(param) -> str:
    """The reference key of a parameter (upper case, '' if none)."""
    r = _ref_of(param)
    return r.split()[0].upper() if r else ""


# =============================================================================================== species
def default_elements_of(species: Optional[Dict] = None) -> Callable[[str], Set[str]]:
    """species name -> its elements (VA -> {'VA'}), from SPECIES formulas when given."""
    species = species or {}

    def els(sp: str) -> Set[str]:
        s = sp.upper().strip("%")
        if s in ("VA", "*"):
            return {s}
        spc = species.get(s)
        if spc is not None:
            f = getattr(spc, "formula", str(spc))
            found = set(re.findall(r"([A-Z][A-Z]?)\d+\.?\d*", f.split("/")[0].upper()))
            if found:
                return found
        return {re.sub(r"[+-]\d*$", "", s)}
    return els


def is_charged(sp: str) -> bool:
    return sp != "/-" and bool(re.search(r"[A-Z0-9][+-]\d*$", sp.upper()))


def is_main(sublattice: Sequence[str], elements_of: Callable[[str], Set[str]]) -> bool:
    """A substitutional sublattice: holds at least one species with an element that is not an interstitial
    (B, C, H, N, O).  (Fe,Cr,Ni,Va) is one; (C,Va) and (Va) are not."""
    for s in sublattice:
        if s in ("VA", "*"):
            continue
        if not elements_of(s) <= INTERSTITIALS:
            return True
    return False


# =============================================================================================== terms
def _tcap(s: str) -> str:
    return s.capitalize() if s.isalpha() else s


@dataclass
class TentTerm:
    """One tentative L0 parameter to add."""
    ref: str                                   # TENT_MIX | TENT_VA
    phase: str
    sublattice: int                            # 0-based
    pair: Tuple[str, str]                      # sorted
    constituents: Tuple[Tuple[str, ...], ...]  # every sublattice, the pair sorted in its sublattice
    value: float
    atoms: float = 0.0
    per_atom: float = 0.0
    evidence: str = ""
    source_label: str = ""                     # the merge / extension that added it

    @property
    def key(self) -> Tuple:
        return ("G", self.phase, self.constituents, 0, "")

    @property
    def raw_name(self) -> str:
        return f"L({self.phase},{':'.join(','.join(sl) for sl in self.constituents)};0)"

    @property
    def label(self) -> str:
        return self.raw_name

    @property
    def rule(self) -> str:
        return "rule 1" if self.ref == TENT_MIX else "rule 2"

    def describe(self) -> str:
        return describe(self.ref, self.phase, self.sublattice, self.pair, self.value, self.atoms)

    def text(self) -> str:
        return f"{self.label} = {self.value:+g} ({self.ref}: {self.describe()})"

    def expression(self) -> str:
        return f"{self.value:+.10g}"

    def piecewise(self, t_lo: float = 298.15, t_hi: float = 6000.0) -> Piecewise:
        return Piecewise([(float(t_lo), float(t_hi), self.expression())], self.ref)


def describe(ref: str, phase: str, k: int, pair: Sequence[str], value: float, atoms: float = 0.0) -> str:
    p = ",".join(_tcap(x) for x in pair)
    if ref == TENT_VA:
        return f"new {p} mixing on substitutional sublattice {k + 1} of {phase}, {value:+g} J/mol (per formula unit)"
    if atoms:
        return (f"new {p} mixing on sublattice {k + 1} of {phase}, +{value / atoms:g} J/mol-atom x {atoms:g} "
                f"atoms/fu")
    return f"new {p} mixing on sublattice {k + 1} of {phase}, {value:+g} J/mol"


def pair_of(constituents: Sequence[Sequence[str]]) -> Optional[Tuple[int, frozenset]]:
    """(sublattice, pair) of an interaction with exactly one interacting sublattice holding two constituents."""
    inter = [(k, tuple(sl)) for k, sl in enumerate(constituents) if len(sl) > 1]
    if len(inter) != 1 or len(set(inter[0][1])) != 2:
        return None
    return inter[0][0], frozenset(inter[0][1])


@dataclass
class Source:
    """A source's view of one phase: its constituents on the combined phase's sublattices (None: absent), and the
    pairs it describes only through tentative terms."""
    label: str
    constituents: List[List[str]]
    tentative_pairs: Set[Tuple[int, frozenset]] = field(default_factory=set)


def described(sources: Iterable[Source]) -> Dict[Tuple[int, frozenset], List[str]]:
    """(sublattice, pair) -> labels of the sources that describe that mixing (have both constituents there)."""
    out: Dict[Tuple[int, frozenset], List[str]] = {}
    for s in sources:
        for k, sl in enumerate(s.constituents):
            for a, b in itertools.combinations(sorted(set(sl)), 2):
                kp = (k, frozenset((a, b)))
                if kp in s.tentative_pairs:
                    continue
                if s.label not in out.setdefault(kp, []):
                    out[kp].append(s.label)
    return out


def plan_phase(phase: str, site_ratios: Sequence[float], constituents: Sequence[Sequence[str]],
               sources: Sequence[Source], g_source: Callable[[Tuple[str, ...]], Optional[str]],
               exists: Callable[[Tuple[Tuple[str, ...], ...]], bool], cfg: Optional[Dict[str, Optional[float]]],
               elements_of: Callable[[str], Set[str]], ordered: bool = False) -> Tuple[List[TentTerm], List[str]]:
    """The tentative terms the combined phase needs (rules 1 and 2) and notes (missing vacancy end-members).

    ``g_source(end-member)`` names the database giving that end-member's G (None: no G); ``exists(constituents)``
    says whether the combined phase already has an L0 parameter with those constituents (tentative or not)."""
    terms: List[TentTerm] = []
    notes: List[str] = []
    if cfg is None or ordered:
        return terms, notes
    cons = [sorted(set(c)) for c in constituents]
    if any(is_charged(s) for sl in cons for s in sl):
        return terms, notes
    main = [k for k, sl in enumerate(cons) if is_main(sl, elements_of)]
    desc = described(sources)
    atoms = round(sum(r for r, sl in zip(site_ratios, cons) if sl != ["VA"]), 9)
    va_flagged: Set[Tuple[str, ...]] = set()
    for k, sl in enumerate(cons):
        if k not in main:
            continue                    # interstitial / vacancy-only sublattice: neither rule
        for a, b in itertools.combinations(sl, 2):
            pr = (k, frozenset((a, b)))
            if pr in desc or "*" in (a, b):
                continue
            others = [cons[j] if j != k else [None] for j in range(len(cons))]
            pair = tuple(sorted((a, b)))
            if "VA" in pair:
                if cfg.get("vacancy") is None:
                    continue
                x = b if a == "VA" else a
                for ctx in itertools.product(*others):
                    em_x = tuple(x if j == k else s for j, s in enumerate(ctx))
                    em_va = tuple("VA" if j == k else s for j, s in enumerate(ctx))
                    gx = g_source(em_x)
                    if gx is None:
                        continue
                    full = tuple(pair if j == k else (s,) for j, s in enumerate(ctx))
                    if exists(full):
                        continue
                    ev = (f"G({phase},{':'.join(em_x)}) from {gx}; no source mixes {_tcap(x)} with VA on sublattice "
                          f"{k + 1} of {phase} (sources with {phase}: {', '.join(s.label for s in sources)})")
                    if g_source(em_va) is None and em_va not in va_flagged:
                        va_flagged.add(em_va)
                        notes.append(f"{phase}: no G({phase},{':'.join(em_va)}) for the vacancy end-member - {VA_HINT}")
                    terms.append(TentTerm(TENT_VA, phase, k, pair, full, float(cfg["vacancy"]), evidence=ev))
            else:
                if len(main) < 2 or cfg.get("mixing_per_atom") is None:
                    continue
                for ctx in itertools.product(*others):
                    em_a = tuple(a if j == k else s for j, s in enumerate(ctx))
                    em_b = tuple(b if j == k else s for j, s in enumerate(ctx))
                    ga, gb = g_source(em_a), g_source(em_b)
                    if ga is None or gb is None:
                        continue
                    full = tuple(pair if j == k else (s,) for j, s in enumerate(ctx))
                    if exists(full):
                        continue
                    per = float(cfg["mixing_per_atom"])
                    v = round(per * atoms, 6)
                    ev = (f"end-members G({phase},{':'.join(em_a)}) from {ga} and G({phase},{':'.join(em_b)}) from "
                          f"{gb}; no source mixes {_tcap(a)},{_tcap(b)} on sublattice {k + 1} of {phase} (sources "
                          f"with {phase}: {', '.join(s.label for s in sources)})")
                    terms.append(TentTerm(TENT_MIX, phase, k, pair, full, v, atoms, per, ev))
    return terms, notes


# =============================================================================================== bookkeeping
@dataclass
class Replacement:
    label: str
    ref: str                     # the tentative term's reference key
    tentative_from: str          # database that had the tentative term
    by: str                      # database whose assessed value (or description) replaces it
    kind: str = "assessed"       # assessed (same key) | described (a source mixes the pair; term removed) | both
    value: str = ""

    def text(self) -> str:
        if self.kind == "described":
            return (f"rule:tentative-replaced: {self.label} {self.ref} ({self.tentative_from}) removed - {self.by} "
                    f"describes that mixing")
        if self.kind == "both":
            return f"tentative in both: {self.label} {self.ref}: {self.by}'s kept ({self.value})"
        return f"rule:tentative-replaced: {self.label} {self.ref} ({self.tentative_from}) -> assessed value from {self.by}"


def citation_items(params: Iterable[Tuple[str, Tuple, float, Sequence[float]]]) -> Dict[str, List[str]]:
    """{TENT_MIX / TENT_VA: [description per parameter]} from (ref, key, value, site ratios of its phase)."""
    out: Dict[str, List[str]] = {}
    for ref, key, value, ratios in params:
        pr = pair_of(key[2])
        if pr is None:
            out.setdefault(ref, []).append(f"{key[1]} {':'.join(','.join(sl) for sl in key[2])} {value:+g}")
            continue
        atoms = sum(r for r, sl in zip(ratios, key[2]) if tuple(sl) != ("VA",)) if ratios else 0.0
        out.setdefault(ref, []).append(describe(ref, key[1], pr[0], sorted(pr[1]), value, atoms if ref == TENT_MIX
                                                else 0.0))
    return out


def citation(ref: str, sources: Sequence[str], items: Sequence[str], date: Optional[str] = None,
             max_items: int = 12) -> str:
    """The reference-list text of TENT_MIX / TENT_VA (no '!' or quotes: TDB-safe)."""
    date = date or _dt.date.today().isoformat()
    shown = list(items[:max_items]) + ([f"+{len(items) - max_items} more"] if len(items) > max_items else [])
    t = (f"Tentative parameter added by calphad_tools when combining {', '.join(sources)} ({date}): "
         + "; ".join(shown) + "; not assessed - replace with an assessed value")
    return t.replace("!", ".").replace("'", "")


def reference_entries(entries: Dict[str, str], width: int = 76) -> List[str]:
    """Reference-list lines ``  KEY 'citation'`` wrapped at spaces (the parser joins the lines of a quote)."""
    out = []
    for key, text in entries.items():
        lines = textwrap.wrap(f"{key} '{text}'", width=width - 2, break_long_words=False, break_on_hyphens=False)
        out += ["  " + lines[0]] + ["    " + ln for ln in lines[1:]]
    return out


_TENT_ENTRY = re.compile(r"(?m)^[ \t]*(?:" + "|".join(TENTATIVE_KEYS) + r")[ \t]+'[^']*'[ \t]*\n?")


def strip_entries(reference_command: str) -> str:
    """A LIST_OF_REFERENCES / ADD_REFERENCES command without its TENT_MIX / TENT_VA entries (they are rewritten
    for the parameters the new file has)."""
    return _TENT_ENTRY.sub("", reference_command)


def counts_text(terms: Sequence[TentTerm]) -> str:
    m = sum(t.ref == TENT_MIX for t in terms)
    v = len(terms) - m
    return f"{m} rule 1 (TENT_MIX), {v} rule 2 (TENT_VA)"


def summary_lines(terms: Sequence[TentTerm], replaced: Sequence[Replacement] = (), notes: Sequence[str] = (),
                  max_items: int = 8) -> List[str]:
    """Report lines: the terms added per rule, replacements and missing vacancy end-members."""
    L = []
    for ref, title in ((TENT_MIX, "rule 1 (TENT_MIX, new mixing)"), (TENT_VA, "rule 2 (TENT_VA, vacancies)")):
        ts_ = [t for t in terms if t.ref == ref]
        if ts_:
            L.append(f"tentative {title}: " + "; ".join(f"{t.label} = {t.value:+g}" for t in ts_[:max_items])
                     + (f"; +{len(ts_) - max_items} more" if len(ts_) > max_items else ""))
    for r in list(replaced)[:max_items]:
        L.append(r.text())
    if len(replaced) > max_items:
        L.append(f"+{len(replaced) - max_items} more tentative replacements")
    L += list(notes)[:4]
    return L


def tdb_reference_entries(db, sources: Sequence[str]) -> Dict[str, str]:
    """{TENT_MIX / TENT_VA: citation} for the tentative parameters a parsed TDB has (keys it uses)."""
    params = []
    for p in db.parameters:
        ref = ref_key(p)
        if ref in TENTATIVE_KEYS:
            ph = db.phases.get(p.key.phase)
            try:
                v = float(p.expr.ranges[0][2]) * p.sign if len(p.expr.ranges) == 1 else float("nan")
            except ValueError:
                v = float("nan")
            params.append((ref, p.key.as_tuple(), v, ph.site_ratios if ph else []))
    items = citation_items(params)
    return {ref: citation(ref, sources, items[ref]) for ref in TENTATIVE_KEYS if ref in items}


def refresh_references(db, sources: Sequence[str]) -> None:
    """Rewrite a parsed TDB's TENT_MIX / TENT_VA reference entries (in place) for the tentative parameters it has:
    old entries removed from its reference commands, one ADD_REFERENCES command with the new ones appended."""
    entries = tdb_reference_entries(db, sources)
    refs = []
    for r in db.references:
        r2 = strip_entries(r)
        body = re.sub(r"^\s*\S+\s*(NUMBER\s+SOURCE)?", "", r2, flags=re.I).strip()
        if body:
            refs.append(r2)
    if entries:
        refs.append("ADD_REFERENCES\n" + "\n".join(reference_entries(entries)))
    db.references = refs
    for k in TENTATIVE_KEYS:
        db.reference_list.pop(k, None)
    db.reference_list.update(entries)


def tdb_phase_terms(db, phase: str, sources: Sequence[Source], cfg, ordered: bool = False,
                    origin: Optional[Dict[Tuple, str]] = None) -> Tuple[List[TentTerm], List[str]]:
    """plan_phase on a parsed TDB's phase (end-member G and existing L0 terms looked up in its parameters;
    ``origin`` {parameter key: database label} names where an end-member's G came from)."""
    ph = db.phases[phase]
    have = {}
    for p in db.parameters:
        if p.key.phase == phase:
            have[p.key.as_tuple()] = p

    def g_source(em):
        k = ("G", phase, tuple((s,) for s in em), 0, "")
        return (origin or {}).get(k, db.name) if k in have else None

    def exists(full):
        return ("G", phase, tuple(tuple(sl) for sl in full), 0, "") in have
    return plan_phase(phase, ph.site_ratios, ph.constituents, sources, g_source, exists, cfg,
                      default_elements_of(db.species), ordered)
