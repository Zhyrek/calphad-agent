"""Digitized phase-diagram points -> ESPEI ZPF datasets, with the phases read off the figure itself.

A ZPF dataset needs, for every measured point, the phases in equilibrium and which phase the measured
composition belongs to.  Often there is no database yet (the digitized data are what the database will be fitted
to), so the phases come from the figure:

1. **Region labels** printed on the diagram ("L", "(Cr)", "L+(Cr)", "γ+γ′" ...).  The model reads each label and its
   position; the tool splits the plot into the regions enclosed by the figure's own drawn lines (ink strokes,
   the frame / triangle edges and the markers; text removed), gives each region its label, infers unlabelled
   regions from their neighbours (a binary field between α and β is α+β; a field between α+β and β+γ is β ...),
   and assigns each point from the regions it borders: a point between "(Ni)" and "(Cr)+(Ni)" is FCC_A1 coexisting
   with BCC_A2; three phases around a binary point -> invariant; around a ternary point -> tie-triangle corner.
2. **The legend**, per marker group: ``{"phase": "(Ni)", "coexisting": ["(Cr)"]}`` or boundary words
   ``{"boundary": "liquidus"}`` / ``solidus`` / ``solvus`` / ``invariant`` / ``tieline``; boundary words combine with
   the region labels (legend: "solvus"; regions: which phases).
3. **Paper names -> database names** via ``name_map`` ({"(Cr)": "BCC_A2", "L": "LIQUID"}); "L"/"Liq" map to LIQUID
   automatically; other unmapped names are an error with suggestions ("(X)" = terminal solid solution of X).
4. **Optional TDB**: inference from the calculated diagram (nearest calculated boundary + equilibria on a 3x3 grid
   around the point, group majority vote) for groups nothing else assigns, and a cross-check that reports where the
   visual assignment and the database disagree (the visual assignment wins).

An assignment preview (the figure with the tinted, labelled regions and the numbered, coloured points) is written
for visual checking before anything is final; ``overrides`` / ``exclude_points`` fix single points.

Rows: tie-line ends at one T (``pair_tielines``) or joined by drawn tie-lines (digitize_points tie_lines=true) are
one row with both compositions; three-phase points at one T -> an invariant row (ternary: tie-triangle); the rest
are broken tie-lines (partner composition null).  Points sampled along a traced line (digitize_curve) are marked
as pseudo-data "digitized from a drawn line" and go to their own dataset files.

The builders at the end (:func:`to_mole_fractions`, :func:`rows_from_points`, :func:`datasets_from_rows`) are pure
Python: show_code embeds their source, so the standalone script writes the identical JSON.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KELVIN = 273.15
TOOL_TAG = "digitized_to_zpf"


# ---------------------------------------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------------------------------------
@dataclass
class DPoint:
    idx: int
    group: str
    source: str
    kind: str                        # marker | curve | inline
    raw: Tuple[float, float]         # as plotted (figure units)
    X: Dict[str, float]              # mole fractions of all elements
    T: float                         # K
    tie: str = ""
    px: Optional[Tuple[float, float]] = None
    size_px: Optional[float] = None
    raw_comp: Dict[str, float] = field(default_factory=dict)
    conv_in: Dict[str, float] = field(default_factory=dict)  # exactly what went into to_mole_fractions
    T_raw: float = 0.0                                       # temperature as plotted (binary)
    phase: str = ""
    coexisting: List[str] = field(default_factory=list)
    category: Optional[tuple] = None
    method: str = ""
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)
    dist: Optional[float] = None     # normalised distance to the evidence (region wall / calculated boundary)
    cands: Dict[tuple, float] = field(default_factory=dict)
    singles: Dict[str, float] = field(default_factory=dict)   # single-phase regions touching the point -> px
    ties: List[tuple] = field(default_factory=list)          # equally supported alternative categories
    row: int = -1
    measured: bool = True            # False: used for the temperature of a row only
    excluded: bool = False
    tdb_cat: Optional[tuple] = None


def _norm_unit(u: str) -> str:
    u = str(u or "fraction").lower().replace(".", "").replace(" ", "").replace("_", "")
    if u in ("fraction", "molefraction", "x", "mole", "mol", "atfraction", "atomicfraction"):
        return "fraction"
    if u in ("at%", "atpercent", "atomic%", "atomicpercent", "mol%", "molpercent", "%"):
        return "at%"
    if u in ("wt%", "wtpercent", "weight%", "weightpercent", "mass%", "masspercent"):
        return "wt%"
    if u in ("wt", "wtfraction", "weightfraction", "massfraction", "w"):
        return "wt_fraction"
    raise ValueError(f"unknown composition unit {u!r} (fraction, at%, wt%, wt_fraction)")


def to_mole_fractions(values, unit, elements, masses):
    """Amounts of the independent elements (as plotted) -> mole fractions of all elements.

    values: {element: amount}; unit: 'fraction' | 'at%' | 'wt%' | 'wt_fraction'; elements: all elements of the
    system (the one missing from values is the balance); masses: {element: g/mol} (only used for weight units)."""
    scale = 100.0 if unit in ("at%", "wt%") else 1.0
    f = {e: float(v) / scale for e, v in values.items()}
    balance = [e for e in elements if e not in f]
    f[balance[0]] = 1.0 - sum(f.values())
    if unit in ("wt%", "wt_fraction"):
        n = {e: max(f[e], 0.0) / masses[e] for e in elements}
        tot = sum(n.values())
        f = {e: n[e] / tot for e in elements}
    return {e: f[e] for e in elements}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _masses(db, elements: Sequence[str]) -> Dict[str, float]:
    from .expdata import ATOMIC_MASS
    out = {}
    for e in elements:
        m = float(db.refstates.get(e, {}).get("mass", 0.0) or 0.0) if db is not None else 0.0
        out[e] = m if m > 0 else float(ATOMIC_MASS[e])
    return out


@dataclass
class System:
    kind: str                        # binary | ternary
    elements: List[str]              # all elements (sorted)
    indep: List[str]                 # independent elements of the ZPF rows (binary: the composition-axis element)
    comp_unit: str
    T_unit: str
    T_fixed: Optional[float] = None  # ternary isotherm, K
    masses: Dict[str, float] = field(default_factory=dict)

    def coords(self, p: DPoint) -> Tuple[float, float]:
        """Plane coordinates used for geometry: (x, T) binary, (X(i1), X(i2)) ternary."""
        if self.kind == "binary":
            return p.X[self.indep[0]], p.T
        return p.X[self.indep[0]], p.X[self.indep[1]]


def _resample_curve(xy: np.ndarray, sampling: Dict, swap: bool, scales: Tuple[float, float]) -> np.ndarray:
    """Points along a traced line: where it crosses multiples of T_step / x_step (figure units), or every
    ``step`` (fraction of the axis ranges) along its length."""
    if len(xy) < 2:
        return xy
    ti = 0 if swap else 1                       # index of the temperature coordinate
    out = []
    if sampling.get("T_step") or sampling.get("x_step"):
        k, st = (ti, float(sampling["T_step"])) if sampling.get("T_step") else (1 - ti, float(sampling["x_step"]))
        for a, b in zip(xy[:-1], xy[1:]):
            lo, hi = sorted((a[k], b[k]))
            for lev in np.arange(math.ceil(lo / st) * st, hi + 1e-12, st):
                t = 0.0 if b[k] == a[k] else (lev - a[k]) / (b[k] - a[k])
                if 0 <= t < 1 or (t == 1 and b is xy[-1]):
                    out.append(a + t * (b - a))
        out = np.array(out) if out else xy[[0, -1]]
    else:
        step = float(sampling.get("step", 0.04))
        u = xy / np.array(scales)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(u, axis=0).T))]
        n = max(2, int(s[-1] / step) + 1)
        ss = np.linspace(0, s[-1], n)
        out = np.column_stack([np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])])
    # drop near-duplicates
    keep = [0]
    for i in range(1, len(out)):
        if np.hypot(*((out[i] - out[keep[-1]]) / np.array(scales))) > 0.004:
            keep.append(i)
    return np.asarray(out)[keep]


def load_points(system: System, csvs: Sequence[str] = (), points: Sequence = (), swap_axes: bool = False,
                ternary_axes: Optional[Sequence[str]] = None, line_sampling: Optional[Dict] = None) -> List[DPoint]:
    """Read digitize_points / digitize_curve CSVs and inline points into DPoints (converted units)."""
    pts: List[DPoint] = []
    tern_axes = [e.upper() for e in (ternary_axes or [])]

    conv: Dict[int, tuple] = {}

    def convert(x, y, comp_cols):
        if system.kind == "binary":
            c, t = (y, x) if swap_axes else (x, y)
            inp = {system.indep[0]: float(c)}
            X = to_mole_fractions(inp, system.comp_unit, system.elements, system.masses)
            T = float(t) + (KELVIN if system.T_unit == "C" else 0.0)
            conv[len(pts)] = (inp, float(t))
            return X, T, {system.indep[0]: float(c)}
        if comp_cols:
            raw = {e: float(v) for e, v in comp_cols.items()}
        else:
            ax = tern_axes or system.indep
            raw = {ax[0]: float(x), ax[1]: float(y)}
        indep = {e: v for e, v in raw.items() if e in system.indep}
        if len(indep) != 2:
            indep = dict(list(raw.items())[:2])
        X = to_mole_fractions(indep, system.comp_unit, system.elements, system.masses)
        conv[len(pts)] = (indep, float(system.T_fixed))
        return X, float(system.T_fixed), raw

    rows_all = []
    for path in csvs or []:
        rows = _read_csv(path)
        if not rows:
            continue
        cols = list(rows[0].keys())
        kind = "marker" if "x_px" in cols else "curve"
        comp_cols = [c for c in cols if re.fullmatch(r"X\(\w+\)", c.strip())]
        rows_all.append((path, rows, kind, comp_cols))
    # axis scales (for arc-length sampling of lines)
    allxy = np.array([[float(r["x"]), float(r["y"])] for _, rows, _, _ in rows_all for r in rows
                      if r.get("x") not in (None, "") and r.get("y") not in (None, "")] or [[0, 0], [1, 1]])
    scales = (max(np.ptp(allxy[:, 0]), 1e-9), max(np.ptp(allxy[:, 1]), 1e-9))
    for path, rows, kind, comp_cols in rows_all:
        if kind == "curve":
            by: Dict[str, List] = {}
            for r in rows:
                try:
                    by.setdefault((r.get("series") or Path(path).stem).strip(), []).append((float(r["x"]), float(r["y"])))
                except (KeyError, ValueError):
                    continue
            for g, xy in by.items():
                res = _resample_curve(np.array(xy), line_sampling or {}, swap_axes, scales)
                for x, y in res:
                    X, T, raw = convert(x, y, None)
                    pts.append(DPoint(idx=len(pts), group=g, source=Path(path).name, kind="curve", raw=(x, y), X=X,
                                      T=T, raw_comp=raw))
            continue
        for r in rows:
            try:
                x, y = float(r["x"]), float(r["y"])
            except (KeyError, ValueError):
                continue
            cc = {re.match(r"X\((\w+)\)", c.strip()).group(1).upper(): r[c] for c in comp_cols if r.get(c, "") != ""}
            X, T, raw = convert(x, y, cc if system.kind == "ternary" and len(cc) >= 2 else None)
            g = (r.get("series") or Path(path).stem).strip()
            px = (float(r["x_px"]), float(r["y_px"])) if r.get("x_px") not in (None, "") else None
            sz = float(r["size_px"]) if r.get("size_px") not in (None, "") else None
            pts.append(DPoint(idx=len(pts), group=g, source=Path(path).name, kind=kind, raw=(x, y), X=X, T=T,
                              tie=(r.get("tie") or "").strip(), px=px, size_px=sz, raw_comp=raw))
    for q in points or []:
        if isinstance(q, dict):
            x, y = float(q["x"]), float(q["y"])
            g, tie = str(q.get("group", "inline")), str(q.get("tie", "") or "")
        else:
            x, y = float(q[0]), float(q[1])
            g = str(q[2]) if len(q) > 2 else "inline"
            tie = str(q[3]) if len(q) > 3 and q[3] is not None else ""
        X, T, raw = convert(x, y, None)
        pts.append(DPoint(idx=len(pts), group=g, source="inline", kind="inline", raw=(x, y), X=X, T=T, tie=tie,
                          raw_comp=raw))
    if not pts:
        raise ValueError("no points: give csv (digitize_points/digitize_curve output) or inline points")
    for p in pts:
        p.conv_in, p.T_raw = conv[p.idx]
        bad = [e for e, v in p.X.items() if v < -0.02 or v > 1.02]
        if bad:
            raise ValueError(f"point {p.idx} (group {p.group!r}, raw {p.raw}) gives mole fractions "
                             f"{ {k: round(v, 3) for k, v in p.X.items()} }: check composition_unit / "
                             f"composition_element / swap_axes")
        if any(v < 0 or v > 1 for v in p.X.values()):
            p.flags.append("composition clipped to [0, 1]")
            for e in p.X:
                p.X[e] = float(min(1.0, max(0.0, p.X[e])))
    return pts


# ---------------------------------------------------------------------------------------------------------
# phase names: paper notation -> database names
# ---------------------------------------------------------------------------------------------------------
STRUCTURE = {  # common structure of the terminal solid solution "(X)" (at the temperatures diagrams usually show)
    **{e: ("BCC_A2", "bcc") for e in ("CR", "MO", "W", "V", "NB", "TA", "LI", "NA", "K", "BA", "EU", "RB", "CS")},
    **{e: ("FCC_A1", "fcc") for e in ("NI", "CU", "AL", "AG", "AU", "PT", "PD", "PB", "IR", "RH", "CA", "SR", "CE",
                                     "YB", "TH")},
    **{e: ("HCP_A3", "hcp") for e in ("TI", "ZR", "HF", "MG", "ZN", "CD", "BE", "Y", "SC", "RE", "RU", "OS", "GD",
                                     "TB", "DY", "HO", "ER", "TM", "LU")},
    **{e: ("DIAMOND_A4", "diamond") for e in ("SI", "GE")},
    "FE": ("BCC_A2", "bcc below 912 °C and above 1394 °C; FCC_A1 (γ) in between"),
    "CO": ("FCC_A1", "fcc above ~420 °C; HCP_A3 below"),
    "MN": ("CBCC_A12", "α-Mn at low T; CUB_A13, FCC_A1, BCC_A2 at higher T"),
    "SN": ("BCT_A5", "β-Sn"), "C": ("GRAPHITE", "graphite"), "B": ("BETA_RHOMBO_B", "β-rhombohedral"),
}
GREEK = {"σ": "SIGMA", "sigma": "SIGMA", "μ": "MU_PHASE", "mu": "MU_PHASE", "χ": "CHI_A12", "chi": "CHI_A12",
         "γ'": "FCC_L12 (or L12_FCC)", "γ′": "FCC_L12 (or L12_FCC)", "γ": "FCC_A1 (γ is usually the fcc phase)",
         "laves": "LAVES_C14 / LAVES_C15 / LAVES_C36", "b2": "BCC_B2", "l12": "FCC_L12"}
LIQUID_NAMES = {"L", "LIQ", "LIQUID", "MELT", "LIQUIDUS"}


def split_label(label: str) -> List[str]:
    """'L+(Cr)' -> ['L', '(Cr)'];  'γ + γ′' -> ['γ', 'γ′']."""
    return [s.strip() for s in re.split(r"\s*\+\s*", str(label).strip()) if s.strip()]


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).replace("’", "′").replace("'", "′")


def suggest_name(name: str) -> Optional[str]:
    n = _norm_name(name)
    m = re.fullmatch(r"\(([A-Za-z]{1,2})\)", n)
    if m:
        el = m.group(1).upper()
        if el in STRUCTURE:
            ph, why = STRUCTURE[el]
            return f"{ph} (terminal solid solution of {el.capitalize()}: {why})"
        return None
    low = n.lower().replace("′", "'")
    for k, v in GREEK.items():
        if low == k.lower().replace("′", "'"):
            return v
    return None


def resolve_names(names: Sequence[str], name_map: Optional[Dict[str, str]], db_phases: Optional[Sequence[str]],
                  liquid: str = "LIQUID", accept_suggestions: bool = False) -> Tuple[Dict[str, str], List[str]]:
    """Map every paper phase name to a database name.  Returns (mapping, notes); raises listing unmapped names."""
    nm = {_norm_name(k): str(v).upper() for k, v in (name_map or {}).items()}
    nm_ci = {k.lower(): v for k, v in nm.items()}
    known = {p.upper() for p in (db_phases or [])}
    out, notes, missing = {}, [], []
    for raw in dict.fromkeys(names):
        k = _norm_name(raw)
        if k in nm:
            out[raw] = nm[k]
        elif k.lower() in nm_ci:
            out[raw] = nm_ci[k.lower()]
        elif k.upper() in LIQUID_NAMES:
            out[raw] = liquid
            if k.upper() != liquid:
                notes.append(f"{raw} -> {liquid} (automatic)")
        elif known and k.upper() in known:
            out[raw] = k.upper()
        elif not known and re.fullmatch(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)+|[A-Z][A-Z0-9]{3,}", k):
            out[raw] = k                            # already a database-style name
        else:
            sug = suggest_name(raw)
            if accept_suggestions and sug:
                out[raw] = sug.split()[0]
                notes.append(f"{raw} -> {out[raw]} (suggestion accepted: {sug})")
            else:
                missing.append(f"{raw!r}" + (f" (suggestion: {sug})" if sug else " (no suggestion)"))
    if missing:
        raise ValueError("these phase names from the figure need database names - ask the user and pass "
                         "name_map, e.g. {\"(Cr)\": \"BCC_A2\"}: " + "; ".join(missing)
                         + ". Use the names the ESPEI phase models will use.")
    if known:
        bad = sorted({v for v in out.values() if v not in known})
        if bad:
            raise ValueError(f"mapped names {bad} are not phases of the database (phases: {sorted(known)})")
    return out, notes


# ---------------------------------------------------------------------------------------------------------
# legend / explicit assignments -> constraints
# ---------------------------------------------------------------------------------------------------------
def _key(s: str) -> str:
    return re.sub(r"[\s_\-:.]+", "", str(s).lower())


def match_groups(groups: Sequence[str], spec: Dict) -> Tuple[Dict[str, object], List[str]]:
    """Map assignment keys ('group 2', 'liquidus', file stems ...) to the point groups (lenient matching)."""
    out, unknown = {}, []
    gk = {_key(g): g for g in groups}
    for k, v in (spec or {}).items():
        kk = _key(k)
        hit = gk.get(kk)
        if hit is None:
            cands = [g for key_, g in gk.items() if key_.endswith(kk) or kk.endswith(key_)]
            hit = cands[0] if len(cands) == 1 else None
        if hit is None:
            unknown.append(k)
        else:
            out[hit] = v
    return out, unknown


def spec_names(v) -> List[str]:
    """Phase names used in one assignment spec (before name mapping)."""
    if v is None:
        return []
    if isinstance(v, str):
        w = v.strip().lower()
        if w in ("auto", "tdb", "infer", "?", "regions", "liquidus", "solidus", "solvus", "invariant", "tieline",
                 "tie-line", "tie_line", "boundary"):
            return []
        return split_label(v) if "+" in v else [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    out = []
    d = {str(k).lower(): val for k, val in dict(v).items()}
    for k in ("phase", "solid"):
        if isinstance(d.get(k), dict):
            out += [str(x) for x in d[k]]
        elif d.get(k):
            out.append(str(d[k]))
    for k in ("coexisting", "with", "phases"):
        w = d.get(k)
        if isinstance(w, str):
            out.append(w)
        elif isinstance(w, dict):
            out += [str(x) for x in w]
        elif w:
            out += [str(x) for x in w]
    if d.get("liquid"):
        out.append(str(d["liquid"]))
    return out


def _parse_spec(v, names: Dict[str, str], liquid: str = "LIQUID") -> Dict:
    """One group's assignment -> constraint {kind, phase, coexisting, phases, liquid_role, with_ranges, complete}.

    Complete constraints are explicit assignments; partial ones ('liquidus', {'phase': ...}) are completed from the
    figure's regions or the TDB."""
    N = lambda s: names.get(str(s), str(s).upper())      # noqa: E731
    base = {"kind": None, "phase": None, "coexisting": None, "phases": None, "liquid_role": None,
            "with_ranges": None, "phase_ranges": None, "complete": False, "words": ""}
    if v is None:
        return base
    if isinstance(v, str):
        w = v.strip().lower().replace("-", "").replace("_", "")
        if w in ("auto", "tdb", "infer", "?", "regions", "boundary"):
            return base
        if w in ("liquidus", "solidus", "solvus", "invariant", "tieline"):
            return _parse_spec({"boundary": w}, names, liquid)
        parts = split_label(v)
        if len(parts) >= 3:
            return _parse_spec({"boundary": "invariant", "phases": parts}, names, liquid)
        return {**base, "kind": "2P", "phase": N(parts[0]), "coexisting": [N(p) for p in parts[1:]] or None,
                "complete": len(parts) == 2, "words": "phase"}
    if isinstance(v, (list, tuple)):
        ph = [N(p) for p in v]
        if len(ph) >= 3:
            return {**base, "kind": "INV", "phases": ph, "complete": True, "words": "invariant"}
        return {**base, "kind": "2P", "phase": ph[0], "coexisting": ph[1:] or None, "complete": len(ph) == 2}
    d = {str(k).lower(): val for k, val in dict(v).items()}
    liq = N(d["liquid"]) if d.get("liquid") else liquid
    b = str(d.get("boundary", d.get("type", ""))).lower().replace("-", "").replace("_", "").replace(" ", "")
    with_ = d.get("with", d.get("coexisting"))
    ranges = None
    if isinstance(with_, dict):
        ranges = [(float(lo_hi[0]), float(lo_hi[1]), N(ph)) for ph, lo_hi in with_.items()]
        with_ = None
    elif isinstance(with_, str):
        with_ = [N(with_)]
    elif with_ is not None:
        with_ = [N(p) for p in with_]
    phase_raw = d.get("phase", d.get("solid"))
    phase = N(phase_raw) if phase_raw and not isinstance(phase_raw, dict) else None
    if b in ("invariant", "eutectic", "peritectic", "monotectic", "eutectoid", "peritectoid", "tietriangle",
             "threephase"):
        ph = [N(p) for p in d.get("phases", [])]
        return {**base, "kind": "INV", "phases": ph or None, "complete": len(ph) >= 3, "words": b}
    if b == "tieline":
        ph = [N(p) for p in d.get("phases", [])]
        if len(ph) != 2:
            raise ValueError(f"tieline assignment needs phases: [low-x phase, high-x phase], got {v}")
        return {**base, "kind": "TIE", "phases": ph, "complete": True, "words": b}
    if b == "liquidus":
        return {**base, "kind": "2P", "phase": liq, "coexisting": with_, "with_ranges": ranges, "liquid_role": "phase",
                "complete": bool(with_), "words": b}
    if b == "solidus":
        return {**base, "kind": "2P", "phase": phase, "coexisting": [liq], "liquid_role": "coexisting",
                "complete": bool(phase), "words": b}
    if b in ("solvus", "solid", "solidsolvus"):
        return {**base, "kind": "2P", "phase": phase, "coexisting": with_, "with_ranges": ranges, "liquid_role": "none",
                "complete": bool(phase and with_), "words": b}
    if b in ("", "boundary", "twophase"):
        if d.get("phases") and not phase:
            return _parse_spec([str(p) for p in d["phases"]], names, liquid)
        return {**base, "kind": "2P" if (phase or with_) else None, "phase": phase, "coexisting": with_,
                "with_ranges": ranges, "complete": bool(phase and (with_ or ranges)), "words": "phase" if phase else ""}
    if b in ("auto", "tdb", "regions"):
        return base
    raise ValueError(f"cannot read assignment {v!r}: use {{phase, coexisting}}, {{boundary: liquidus|solidus|solvus"
                     f"|invariant|tieline, ...}} or 'auto'")


def parse_spec(v, names: Dict[str, str], liquid: str = "LIQUID") -> Dict:
    """_parse_spec plus 'phase' given as ranges of the composition axis: {"(Cr)": [0, 45], "(Ni)": [45, 100]}."""
    c = _parse_spec(v, names, liquid)
    if isinstance(v, dict):
        d = {str(k).lower(): val for k, val in v.items()}
        pr = d.get("phase", d.get("solid"))
        if isinstance(pr, dict):
            c["phase_ranges"] = [(float(a[0]), float(a[1]), names.get(str(ph), str(ph).upper())) for ph, a in pr.items()]
            c["complete"] = False
    return c


def satisfies(cat: tuple, c: Dict, liquid: str) -> bool:
    """Does a category agree with a (partial) constraint?"""
    if not cat:
        return False
    kind = c.get("kind")
    if kind == "INV" and cat[0] not in ("INV", "TRI"):
        return False
    if kind == "2P" and cat[0] != "2P":
        return False
    if kind == "INV" and c.get("phases") and set(cat[1] if cat[0] == "INV" else (cat[1], *cat[2])) != set(c["phases"]):
        return False
    phase = cat[1] if cat[0] in ("2P", "TRI") else ""
    co = set(cat[2]) if cat[0] in ("2P", "TRI") else set()
    if c.get("phase") and cat[0] != "INV" and phase != c["phase"]:
        return False
    if c.get("coexisting") and cat[0] == "2P" and co != set(c["coexisting"]):
        return False
    role = c.get("liquid_role")
    allp = {phase} | co if cat[0] != "INV" else set(cat[1])
    if role == "phase" and cat[0] == "2P" and phase != liquid:
        return False
    if role == "coexisting" and cat[0] == "2P" and liquid not in co:
        return False
    if role == "none" and liquid in allp:
        return False
    return True


# ---------------------------------------------------------------------------------------------------------
# categories from sets of phases around a point
# ---------------------------------------------------------------------------------------------------------
INFO_FLAGS = ("phase pair set with its drawn tie-line", "tie-triangle corner (three points", "composition clipped",
              "excluded by request")


def warnings_of(p: "DPoint") -> List[str]:
    """Flags that ask for a look (informational notes excluded)."""
    return [f for f in p.flags if not f.startswith(INFO_FLAGS)]


def _cat_label(cat: Optional[tuple]) -> str:
    if not cat:
        return "?"
    if cat[0] == "2P":
        return f"{cat[1]} | {'+'.join(cat[2])}"
    if cat[0] == "INV":
        return "invariant " + "+".join(cat[1])
    if cat[0] == "TRI":
        return f"{cat[1] or '?'} | tie-triangle {'+'.join(cat[2])}"
    return str(cat)


def _apply_cat(p: DPoint, cat: tuple) -> None:
    p.category = cat
    if cat[0] == "2P":
        p.phase, p.coexisting = cat[1], list(cat[2])
    elif cat[0] == "INV":
        p.phase, p.coexisting = "", list(cat[1])
    elif cat[0] == "TRI":
        p.phase, p.coexisting = cat[1], list(cat[2])


def cats_from_sets(seen: Dict[tuple, float], kind: str) -> Dict[tuple, float]:
    """Candidate categories from the phase sets of the regions around a point (set -> ring radius px).

    Returns category -> cost (px): the 'touching' combination (all sets within 3 px of the second-nearest) costs
    its radius minus 1 (it describes the junction exactly); pair / triple combinations cost their largest radius;
    fields too thin to be drawn (two single-phase regions meeting, or only two-phase regions sharing one phase)
    cost 4 px more."""
    out: Dict[tuple, float] = {}

    def put(c, d):
        if c is not None and d < out.get(c, 9e9):
            out[c] = d
    sets = sorted(seen.items(), key=lambda kv: kv[1])
    if len(sets) >= 2:
        r2 = sets[1][1]
        touch = [set(s) for s, r in sets if r <= r2 + 3]
        put(_cat_of(touch, kind), r2 - 1)
    for (a, ra), (b, rb) in itertools.combinations(sets, 2):
        put(_cat_of([set(a), set(b)], kind), max(ra, rb))
        A, B = (set(a), set(b)) if len(a) <= len(b) else (set(b), set(a))
        if len(A) == 1 and len(B) == 1 and A != B:        # two-phase field too narrow to be drawn
            for P, Q in ((A, B), (B, A)):
                put(("2P", next(iter(P)), (next(iter(Q)),)), max(ra, rb) + 4)
    for (a, ra), (b, rb), (c, rc) in itertools.combinations(sets, 3):
        put(_cat_of([set(a), set(b), set(c)], kind), max(ra, rb, rc))
    if len(sets) == 1 and len(sets[0][0]) == 2:          # one two-phase region only: which side is unknown
        a, b = sets[0][0]
        put(("2P", a, (b,)), sets[0][1] + 8)
        put(("2P", b, (a,)), sets[0][1] + 8)
    # a single-phase field too thin to be drawn: only multi-phase regions around, sharing one phase P
    if sets and not any(len(s_) == 1 for s_, _ in sets):
        near = [(set(s_), r) for s_, r in sets if r <= sets[0][1] + 6]
        inter = set.intersection(*[s_ for s_, _ in near])
        if len(inter) == 1:
            P = next(iter(inter))
            for s_, r in near:
                if len(s_) == 2:
                    put(("2P", P, tuple(sorted(s_ - {P}))), r + 4)
                elif len(s_) == 3 and kind == "ternary":
                    put(("TRI", P, tuple(sorted(s_ - {P}))), r + 4)
    return out


def _cat_of(sets: List[set], kind: str) -> Optional[tuple]:
    union = set().union(*sets)
    inter = set.intersection(*sets) if sets else set()
    singles = [s for s in sets if len(s) == 1]
    if kind == "binary" and len(union) == 3:
        return ("INV", tuple(sorted(union)))
    if len(union) == 2 and len(inter) == 1 and singles:
        P = next(iter(inter))
        return ("2P", P, tuple(sorted(union - {P})))
    if kind == "ternary" and len(union) == 3 and len(inter) == 1 and singles and any(len(s) == 3 for s in sets):
        P = next(iter(inter))
        return ("TRI", P, tuple(sorted(union - {P})))
    return None


def tie_consistency(pts: Sequence[DPoint]) -> int:
    """Points joined by a drawn tie-line (tie-triangle) are the two (three) sides of one field: each takes a
    different one of the field's phases.  Every point's candidate phases come from its region evidence (single-phase
    regions touching it and the categories it supports); the cheapest consistent choice is applied.  Returns the
    number of tie-lines / triangles resolved."""
    by: Dict[str, List[DPoint]] = {}
    for p in pts:
        if p.tie and not p.excluded and p.method.startswith("figure"):
            by.setdefault(p.tie, []).append(p)
    fixed = 0
    for tid, ms in by.items():
        if len(ms) not in (2, 3):
            continue
        opts = []
        for p in ms:
            o: Dict[str, float] = {}
            for c, d in p.cands.items():
                if c[0] in ("2P", "TRI") and c[1]:
                    o[c[1]] = min(o.get(c[1], 9e9), d)
            for n, r in p.singles.items():
                o[n] = min(o.get(n, 9e9), 0.5 * r / 1000.0)
            opts.append(o)
        if any(not o for o in opts):
            continue
        best = None
        for combo in itertools.product(*[list(o) for o in opts]):
            if len(set(combo)) != len(combo):
                continue
            cost = sum(o[n] for o, n in zip(opts, combo))
            if best is None or cost < best[0]:
                best = (cost, combo)
        if best is None:
            continue
        phs = best[1]
        for p, n in zip(ms, phs):
            rest = tuple(sorted(set(phs) - {n}))
            c = ("2P", n, rest) if len(ms) == 2 else ("TRI", n, rest)
            if p.category != c:
                _apply_cat(p, c)
                p.flags = [f for f in p.flags if not f.startswith(("ambiguous", "only one", "regions around"))]
                p.flags.append("phase pair set with its drawn tie-line partner(s)" if len(ms) == 2 else
                               "tie-triangle corner (three points joined by drawn lines)")
            p.ties = []
            p.confidence = max(p.confidence, 0.8)
            if not p.method.endswith("tie-line"):
                p.method += "+tie-line"
        fixed += 1
    return fixed


# ---------------------------------------------------------------------------------------------------------
# the figure: regions bounded by drawn lines
# ---------------------------------------------------------------------------------------------------------
class FigureRegions:
    """Segmentation of the working image into regions, their labels (read by the model) and neighbours."""

    def __init__(self, work: str, axes: Dict, system: System, pts: Sequence[DPoint], labels: Sequence[Dict],
                 names: Dict[str, str], boundary_color=None, swap_axes: bool = False, ternary_axes=None):
        from . import digitize as D
        self.D = D
        self.work = work
        self.system = system
        self.rgb = D.load_rgb(work)
        self.tri = None
        tern = (axes or {}).get("ternary")
        if system.kind == "ternary" or tern:
            self.tri = D.detect_triangle(self.rgb)
            if tern:
                self.tcal = D.calibrate_triangle(tern, self.tri)
            else:
                raise ValueError("a ternary figure needs axes={'ternary': {...}} (the digitize_points calibration)")
            if self.tri is None:
                raise ValueError("no composition triangle found in the image")
            self.fr = self.tri.frame()
        else:
            self.fr = D.detect_frame(self.rgb)
            ticks = D.detect_ticks(self.rgb, self.fr)
            self.calx, self.caly = D._cal(axes, ticks, self.fr)
        self.swap = swap_axes
        self.ternary_axes = ternary_axes
        # pixel positions of all points
        for p in pts:
            if p.px is None:
                p.px = self.to_px(p)
        sizes = [p.size_px for p in pts if p.size_px]
        self.marker = float(np.median(sizes)) if sizes else 7.0
        rgb_det = D.mask_outside(self.rgb, self.tri, margin=8) if self.tri is not None else self.rgb
        try:
            found, info = D.detect_markers(rgb_det, self.fr)
        except Exception:  # noqa: BLE001
            found, info = [], {}
        discs = [(m.cx, m.cy, m.size_px / 2) for m in found if not m.in_legend and m.size_px < 3 * self.marker]
        discs += [(p.px[0], p.px[1], (p.size_px or self.marker) / 2) for p in pts if p.kind != "curve"]
        self.legend = [tuple(b) for b in info.get("legend_boxes", [])]
        self.reg, self.info = D.segment_regions(self.rgb, self.fr, tri=self.tri, discs=discs, boxes=self.legend,
                                                color=boundary_color)
        # labels -> regions
        self.labels: Dict[int, Dict] = {}
        self.conflicts: List[str] = []
        self.unplaced: List[str] = []
        for lab in labels or []:
            text = str(lab.get("label", lab.get("phases", "")))
            phases = tuple(sorted({names.get(n, n.upper()) for n in (split_label(text) if isinstance(
                lab.get("label"), str) else [str(x) for x in lab.get("phases", [])])}))
            if lab.get("at_px") is not None:
                q = (float(lab["at_px"][0]), float(lab["at_px"][1]))
            else:
                q = self.data_to_px(lab["at"])
            rid = D.region_at(self.reg, q[0], q[1], search=int(max(15, 1.5 * self.marker)))
            if rid < 0:
                self.unplaced.append(f"{text!r} at {lab.get('at', lab.get('at_px'))}")
                continue
            if rid in self.labels and set(self.labels[rid]["phases"]) != set(phases):
                self.conflicts.append(f"labels {self.labels[rid]['text']!r} and {text!r} fall in one region "
                                      f"(a boundary line may be missing/broken there)")
                continue
            self.labels[rid] = {"text": text, "phases": phases, "inferred": False, "at": q}
        self.inferred = self._infer_unlabelled()

    # coordinates ---------------------------------------------------------------------------------------
    def to_px(self, p: DPoint) -> Tuple[float, float]:
        if self.tri is not None:
            f = [p.raw_comp.get(e, np.nan) for e in self.tcal.elements]
            if any(np.isnan(f)):
                X = p.X
                f = [X[e] for e in self.tcal.elements]
            else:
                miss = [i for i, v in enumerate(f) if np.isnan(v)]
                if miss:
                    f[miss[0]] = 1 - np.nansum(f)
            q = self.tcal.to_px(np.array(f))[0]
            return float(q[0]), float(q[1])
        return float(self.calx.to_px(p.raw[0])), float(self.caly.to_px(p.raw[1]))

    def data_to_px(self, at) -> Tuple[float, float]:
        if self.tri is not None:
            if isinstance(at, dict):
                comp = {str(k).upper(): float(v) for k, v in at.items()}
            else:
                ax = [e.upper() for e in (self.ternary_axes or self.system.indep)]
                comp = {ax[0]: float(at[0]), ax[1]: float(at[1])}
            els = self.tcal.elements
            f = [comp.get(e, np.nan) for e in els]
            if sum(np.isnan(f)) == 1:
                i = int(np.flatnonzero(np.isnan(f))[0])
                f[i] = (100.0 if max(v for v in f if not np.isnan(v)) > 1.0 else 1.0) - np.nansum(f)
            q = self.tcal.to_px(np.array(f))[0]
            return float(q[0]), float(q[1])
        return float(self.calx.to_px(float(at[0]))), float(self.caly.to_px(float(at[1])))

    # unlabelled regions ---------------------------------------------------------------------------------
    def _infer_unlabelled(self) -> List[str]:
        D = self.D
        notes = []
        ids = self.info["regions"]
        min_big = max(200, 0.004 * sum(self.info["sizes"].values()))
        for _ in range(4):
            changed = False
            for rid in ids:
                if rid in self.labels:
                    continue
                guess = None
                if self.system.kind == "binary":        # neighbours at the same temperature (swapped: same column)
                    l, r = D.horizontal_neighbours(self.reg.T if self.swap else self.reg, rid)
                    A = set(self.labels[l]["phases"]) if l in self.labels else None
                    B = set(self.labels[r]["phases"]) if r in self.labels else None
                    if A and B:
                        if len(A) == 1 and len(B) == 1 and A != B:
                            guess = tuple(sorted(A | B))
                        elif len(A) == 2 and len(B) == 2 and len(A & B) == 1:
                            guess = tuple(sorted(A & B))
                        elif A == B and len(A) == 2:
                            guess = tuple(sorted(A))       # a strip of the same two-phase field
                else:
                    nb = D.region_neighbours(self.reg, rid)
                    sets = [set(self.labels[k]["phases"]) for k in nb if k in self.labels]
                    singles = {next(iter(s)) for s in sets if len(s) == 1}
                    twos = [s for s in sets if len(s) == 2]
                    if len(singles) == 2 and not any(len(s) == 3 for s in sets):
                        guess = tuple(sorted(singles))
                    elif not singles and twos and len(set().union(*twos)) == 3 and len(twos) >= 2:
                        guess = tuple(sorted(set().union(*twos)))
                    elif twos and all(s == twos[0] for s in twos) and singles <= twos[0] \
                            and not any(len(s) == 3 for s in sets):
                        guess = tuple(sorted(twos[0]))       # another strip of the same two-phase field
                if guess:
                    ys, xs = np.nonzero(self.reg == rid)
                    self.labels[rid] = {"text": "+".join(guess), "phases": guess, "inferred": True,
                                        "at": (float(xs.mean()), float(ys.mean()))}
                    if self.info["sizes"].get(rid, 0) >= min_big:
                        notes.append(f"unlabelled region at px ({xs.mean():.0f}, {ys.mean():.0f}) inferred as "
                                     f"{'+'.join(guess)} from its neighbours")
                    changed = True
            if not changed:
                break
        return notes

    # points -----------------------------------------------------------------------------------------------
    def around(self, p: DPoint) -> Tuple[Dict[tuple, float], Dict[str, float], float]:
        """(phase set -> ring radius px, single phase -> radius, radius of the second region)."""
        D = self.D
        r0 = (p.size_px or self.marker) / 2 + 3 if p.kind != "curve" else 3.0
        r_max = max(30.0, 0.07 * min(self.fr.right - self.fr.left, self.fr.bottom - self.fr.top))
        seen = D.regions_around(self.reg, p.px[0], p.px[1], r0, r_max, extra=max(5.0, p.size_px or self.marker))
        sets: Dict[tuple, float] = {}
        for rid, r in seen.items():
            if rid in self.labels:
                s = tuple(sorted(self.labels[rid]["phases"]))
                sets[s] = min(sets.get(s, 9e9), r)
        singles = {s[0]: r for s, r in sets.items() if len(s) == 1}
        rr = sorted(sets.values())
        return sets, singles, (rr[1] if len(rr) > 1 else (rr[0] if rr else r_max))

    def link_ties(self, pts: Sequence[DPoint], color=None) -> int:
        """Tie ids for points joined by drawn straight lines (tie-lines, tie-triangle edges)."""
        D = self.D
        cand = [p for p in pts if p.kind != "curve" and not p.excluded and p.px is not None]
        ms = [D.Marker(cx=p.px[0], cy=p.px[1], area=0.0, size_px=p.size_px or self.marker, shape="", filled=True,
                       color="", score=1.0) for p in cand]
        rgb = D.mask_outside(self.rgb, self.tri, margin=8) if self.tri is not None else self.rgb
        links = D.marker_links(rgb, self.fr, ms, color=color)
        for i, tid in links.items():
            if not cand[i].tie:
                cand[i].tie = tid
        return len(set(links.values()))

    def diag(self) -> float:
        return float(math.hypot(self.fr.right - self.fr.left, self.fr.bottom - self.fr.top))

    # preview -----------------------------------------------------------------------------------------------
    def preview(self, pts: Sequence[DPoint], path: str, title: str = "", rows: Optional[List[Dict]] = None) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        H, W = self.rgb.shape[:2]
        scale = max(1.0, 1100 / max(H, W))
        fig = plt.figure(figsize=(W * scale / 100 + 2.6, H * scale / 100), dpi=100)
        frac = W * scale / (W * scale + 260)
        ax = fig.add_axes([0, 0, frac, 1])
        base = 0.55 + 0.45 * self.rgb
        tint = np.zeros((H, W, 4))
        pal = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
               "#bab0ac", "#86bcb6", "#d37295"]
        set_col: Dict[tuple, str] = {}
        for rid in self.info["regions"]:
            lab = self.labels.get(rid)
            m = self.reg == rid
            if lab is None:
                tint[m] = (*to_rgb("#999999"), 0.35)
                continue
            key = tuple(sorted(lab["phases"]))
            col = set_col.setdefault(key, pal[len(set_col) % len(pal)])
            tint[m] = (*to_rgb(col), 0.38)
        ax.imshow(base, interpolation="nearest")
        ax.imshow(tint, interpolation="nearest")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fs = 8
        for rid, lab in self.labels.items():
            if self.info["sizes"].get(rid, 0) < 150:
                continue
            m = self.reg == rid
            from scipy import ndimage as ndi
            dt = ndi.distance_transform_edt(m)
            y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
            if not lab["inferred"]:
                x, y = lab["at"]
            ax.text(x, y, "+".join(lab["phases"]) + (" (inferred)" if lab["inferred"] else ""), fontsize=fs,
                    ha="center", va="center", color="k", style="italic" if lab["inferred"] else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=set_col.get(tuple(sorted(lab["phases"])), "k"),
                              alpha=0.85))
        cats = sorted({_cat_label(p.category) for p in pts if p.category and not p.excluded})
        pcol = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cats)}
        by_row: Dict[int, List[DPoint]] = {}
        for p in pts:
            if p.row >= 0 and p.measured and p.px is not None:
                by_row.setdefault(p.row, []).append(p)
        for rid, ms in by_row.items():                   # rows with several compositions: tie-lines etc.
            if len(ms) < 2:
                continue
            kind = (rows or [{}] * (rid + 1))[rid].get("kind", "")
            P = np.array([p.px for p in ms] + ([ms[0].px] if len(ms) == 3 and kind == "tie-triangle" else []))
            if kind == "invariant":
                P = P[np.argsort(P[:, 0])]
            ax.plot(P[:, 0], P[:, 1], color="crimson" if kind != "tie-line" else "k", lw=1.4, ls="--", zorder=3)
        for p in pts:
            if p.px is None:
                continue
            x, y = p.px
            if p.excluded:
                ax.plot(x, y, "x", color="0.4", ms=7)
                continue
            if not p.category:
                ax.add_patch(plt.Circle((x, y), 9, fill=False, ec="red", lw=2.2))
            else:
                c = pcol[_cat_label(p.category)]
                ax.add_patch(plt.Circle((x, y), 5.5, fc=c, ec="k", lw=0.6, alpha=0.95))
                if warnings_of(p):
                    ax.add_patch(plt.Circle((x, y), 9, fill=False, ec="orange", lw=1.6))
            ax.text(x + 6, y - 6, str(p.idx), fontsize=6.5, color="k",
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.6))
        handles = [plt.Line2D([], [], marker="o", ls="none", mfc=pcol[c], mec="k", label=c) for c in cats]
        handles += [plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="red", mew=2, label="unassigned"),
                    plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="orange", mew=1.6, label="flagged")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(frac + 0.005, 0.97), fontsize=7,
                   frameon=False, title="phase | coexisting", title_fontsize=8)
        if title:
            ax.text(5, 5, title, fontsize=8, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path


def assign_from_figure(fig_regions: FigureRegions, p: DPoint, cons: Dict, liquid: str) -> None:
    """Assign one point from the labelled regions it borders (respecting a legend constraint)."""
    sets, singles, r2 = fig_regions.around(p)
    p.singles = singles
    diag = fig_regions.diag()
    cands = cats_from_sets(sets, fig_regions.system.kind)
    p.cands = {c: d / diag for c, d in cands.items()}
    ok = {c: d for c, d in cands.items() if satisfies(c, cons, liquid)} if cons.get("kind") or cons.get("phase") \
        or cons.get("liquid_role") else cands
    if not sets:
        p.flags.append("no labelled region around the point")
        return
    p.method = "figure" + ("+legend" if cons.get("words") or cons.get("phase") else "")
    if not ok:
        p.flags.append(f"regions around it ({', '.join('+'.join(s) for s in sets)}) do not fit the legend "
                       f"({cons.get('words') or 'constraint'})" if cands else
                       f"only one labelled region around it ({', '.join('+'.join(s) for s in sets)})")
        return
    order = sorted(ok.items(), key=lambda kv: kv[1])
    best, d = order[0]
    tie = [c for c, dd in order[1:] if dd <= d + 0.5]
    _apply_cat(p, best)
    p.dist = d / diag
    p.method = "figure" + ("+legend" if cons.get("words") or cons.get("phase") else "")
    p.confidence = 1.0 if not tie else 0.5
    p.ties = tie
    if tie:
        p.flags.append("ambiguous: also " + ", ".join(_cat_label(c) for c in tie))
    if r2 > (p.size_px or fig_regions.marker) / 2 + 8 and p.kind != "curve":
        p.flags.append(f"not on a drawn boundary (nearest other region {r2:.0f} px away)")
        p.confidence = min(p.confidence, 0.7)
    if any(c[0] == "2P" and len(c[2]) == 1 and cands.get(c) == d and d == max(sets.get((c[1],), 0),
                                                                             sets.get(c[2], 0)) + 4
           for c in [best]):
        p.flags.append("two-phase field too narrow to see: phase from the legend" if cons.get("kind") else
                       "two-phase field too narrow to see")


# ---------------------------------------------------------------------------------------------------------
# region labels without an image (geometric fallback)
# ---------------------------------------------------------------------------------------------------------
def assign_by_label_geometry(system: System, pts: List[DPoint], regions: Sequence[Dict], names: Dict[str, str],
                             T_scale: float, swap_axes: bool, ternary_axes) -> None:
    """Without the image: the labels nearest on either side of each point's group curve."""
    labs = []
    for r in regions:
        text = r.get("label", "")
        ph = tuple(sorted({names.get(n, n.upper()) for n in (split_label(text) if text else r.get("phases", []))}))
        at = r["at"]
        q = load_points(system, points=[[float(at[0]), float(at[1])]], swap_axes=swap_axes,
                        ternary_axes=ternary_axes)[0]
        labs.append((ph, _unit_xy(system, q, T_scale)))
    by_group: Dict[str, List[DPoint]] = {}
    for p in pts:
        by_group.setdefault(p.group, []).append(p)
    for g, gp in by_group.items():
        U = np.array([_unit_xy(system, p, T_scale) for p in gp])
        for i, p in enumerate(gp):
            u = U[i]
            d = np.hypot(*(U - u).T)
            nb = [j for j in np.argsort(d)[1:4] if d[j] < 0.25]
            cat = None
            if nb:
                P = U[[i] + list(nb)]
                t = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)[2][0]
                nrm = np.array([-t[1], t[0]])
                sides: Dict[int, Tuple[float, tuple]] = {}
                for ph, q in labs:
                    s = 1 if (q - u) @ nrm > 0 else -1
                    dd = float(np.hypot(*(q - u)))
                    if s not in sides or dd < sides[s][0]:
                        sides[s] = (dd, ph)
                if len(sides) == 2:
                    cat = _cat_of([set(sides[1][1]), set(sides[-1][1])], system.kind)
            if cat is None:
                p.flags.append("labels: no adjacent labelled regions share a phase")
                continue
            _apply_cat(p, cat)
            p.method, p.confidence = "labels (no image)", 0.6


def _unit_xy(system: System, p: DPoint, T_scale: float) -> np.ndarray:
    a, b = system.coords(p)
    return np.array([a, b / T_scale]) if system.kind == "binary" else np.array([a, b])


# ---------------------------------------------------------------------------------------------------------
# optional TDB inference
# ---------------------------------------------------------------------------------------------------------
_MAP_CACHE: Dict[tuple, tuple] = {}


def _seg_dist(p: np.ndarray, poly: np.ndarray) -> float:
    if len(poly) == 1:
        return float(np.hypot(*(poly[0] - p)))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-18)
    t = np.clip(((p - a) * ab).sum(axis=1) / L2, 0, 1)
    q = a + t[:, None] * ab
    return float(np.min(np.hypot(*(q - p).T)))


class Inferencer:
    """Calculated boundaries (pycalphad mapping, cached) + local equilibrium probes for one system."""

    def __init__(self, tdb: str, system: System, pts: Sequence[DPoint], phases: Optional[Sequence[str]] = None):
        from . import pcache
        from .mapping_tools import _select_phases
        from .phase_regions import map_lines, ordered_info
        self.tdb = tdb
        self.sys = system
        self.db = pcache.get_db(tdb)
        els = system.elements
        self.comps = sorted(els) + ["VA"]
        self.phases = _select_phases(self.db, els, phases)
        self.info = ordered_info(self.db, self.comps, self.phases)
        if system.kind == "binary":
            Ts = np.array([p.T for p in pts])
            span = float(np.ptp(Ts)) if len(Ts) > 1 else 0.0
            self.T_scale = max(300.0, 1.25 * span + 100.0)
            lo = max(200.0, math.floor((Ts.min() - 0.15 * self.T_scale) / 10) * 10)
            hi = math.ceil((Ts.max() + 0.15 * self.T_scale) / 10) * 10
            recipe = {"kind": "binary", "elements": [e for e in els if e != system.indep[0]] + [system.indep[0]],
                      "phases": self.phases, "P": 101325.0, "T_range": [lo, hi], "x_range": [0.0, 1.0]}
            key = (pcache.file_key(tdb), "binary", tuple(recipe["elements"]), tuple(self.phases), lo, hi)
        else:
            self.T_scale = 1.0
            recipe = {"kind": "ternary", "elements": [e for e in els if e not in system.indep] + list(system.indep),
                      "phases": self.phases, "P": 101325.0, "T_K": float(system.T_fixed)}
            key = (pcache.file_key(tdb), "ternary", tuple(recipe["elements"]), tuple(self.phases),
                   round(float(system.T_fixed), 2))
        if key not in _MAP_CACHE:
            with pcache.use(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _MAP_CACHE[key] = map_lines(self.db, recipe, self.info, step=0.02, T_step=10.0)
            while len(_MAP_CACHE) > 24:
                _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
        self.lines, self.invariants = _MAP_CACHE[key]
        self.traces: List[Tuple[tuple, np.ndarray]] = []
        for ln in self.lines:
            nm = [re.sub(r"#\d+$", "", n) for n in ln["phases"]]
            if len(nm) != 2 or nm[0] == nm[1]:
                continue
            for k in range(2):
                poly = np.array([self.norm(self._xy(pt["phases"][k][1], pt["T"])) for pt in ln["points"]])
                self.traces.append((("2P", nm[k], (nm[1 - k],)), poly))
        self.inv: List[Dict] = []
        for inv in self.invariants:
            nm = [re.sub(r"#\d+$", "", n) for n in inv["phases"]]
            comps = {re.sub(r"#\d+$", "", n): c for n, c in inv["compositions"].items()}
            self.inv.append({"phases": nm, "T": float(inv["T_K"]),
                             "xy": {n: self.norm(self._xy(comps[n], inv["T_K"])) for n in nm}})

    def _xy(self, X: Dict[str, float], T: float) -> Tuple[float, float]:
        if self.sys.kind == "binary":
            return float(X[self.sys.indep[0]]), float(T)
        return float(X[self.sys.indep[0]]), float(X[self.sys.indep[1]])

    def norm(self, xy) -> np.ndarray:
        return np.array([xy[0], xy[1] / self.T_scale]) if self.sys.kind == "binary" else np.array(xy, float)

    def candidates(self, p: DPoint) -> Dict[tuple, float]:
        """category -> normalised distance to the nearest calculated boundary of that kind."""
        u = self.norm(self.sys.coords(p))
        out: Dict[tuple, float] = {}
        for cat, poly in self.traces:
            d = _seg_dist(u, poly)
            if d < out.get(cat, 9e9):
                out[cat] = d
        for inv in self.inv:
            names = inv["phases"]
            if self.sys.kind == "binary":
                xs = [inv["xy"][n][0] for n in names]
                y = inv["xy"][names[0]][1]
                dx = max(0.0, min(xs) - u[0], u[0] - max(xs))
                cat = ("INV", tuple(sorted(names)))
                out[cat] = min(out.get(cat, 9e9), math.hypot(dx, u[1] - y))
            else:
                for n in names:
                    cat = ("TRI", n, tuple(sorted(m for m in names if m != n)))
                    out[cat] = min(out.get(cat, 9e9), float(np.hypot(*(u - inv["xy"][n]))))
        return out

    def invariant_target(self, phases: Sequence[str]) -> Optional[Dict[str, float]]:
        want = sorted(phases)
        for inv in self.inv:
            if sorted(inv["phases"]) == want:
                return {n: float(inv["xy"][n][0]) for n in inv["phases"]}
        return None

    def probe(self, p: DPoint, dx: float = 0.012) -> Optional[tuple]:
        """Stable phase sets on a 3x3 grid around the point -> the category of the regions that meet there."""
        from pycalphad import equilibrium, variables as v
        from . import pcache
        from .phase_regions import _keys_from_eq
        conds = {v.P: 101325.0, v.N: 1}
        if self.sys.kind == "binary":
            x = p.X[self.sys.indep[0]]
            dT = 0.012 * self.T_scale
            conds[v.T] = [max(100.0, p.T - dT), p.T, p.T + dT]
            conds[v.X(self.sys.indep[0])] = sorted({float(np.clip(x + s * dx, 1e-5, 1 - 1e-5)) for s in (-1, 0, 1)})
        else:
            conds[v.T] = float(self.sys.T_fixed)
            for e in self.sys.indep:
                conds[v.X(e)] = sorted({float(np.clip(p.X[e] + s * dx, 1e-5, 0.999)) for s in (-1, 0, 1)})
        with pcache.use(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                eq = equilibrium(self.db, self.comps, self.phases, conds)
            except Exception:  # noqa: BLE001 - probes are only evidence
                return None
        ks = {tuple(sorted(set(k))) for k in _keys_from_eq(eq, self.info) if k}
        if len(ks) < 2:
            return None
        return _cat_of([set(k) for k in ks], self.sys.kind)


def infer_point(inf: Inferencer, p: DPoint, near_tol: float, far_tol: float, cons: Optional[Dict] = None,
                liquid: str = "LIQUID", inv_bias: float = 0.004, apply: bool = True) -> Optional[tuple]:
    """Category from the calculated diagram (nearest boundary, checked by local equilibria).  With apply=False
    only returns it (cross-check)."""
    cands = inf.candidates(p)
    if cons and (cons.get("kind") or cons.get("phase") or cons.get("liquid_role")):
        cands = {c: d for c, d in cands.items() if satisfies(c, cons, liquid)}
    if not cands:
        if apply:
            p.flags.append("no calculated boundary of this kind in the TDB")
        return None
    order = sorted(cands.items(), key=lambda kv: kv[1])
    best, d = order[0]
    two = [(c, dd) for c, dd in order if c[0] == "2P"]
    invs = [(c, dd) for c, dd in order if c[0] == "INV"]
    if best[0] == "2P" and invs and invs[0][1] <= d + inv_bias:
        best, d = invs[0]
    elif best[0] == "INV" and two and two[0][1] < d - inv_bias:
        best, d = two[0]
    probe_cat = inf.probe(p)
    if probe_cat is not None and probe_cat != best and probe_cat in cands and cands[probe_cat] <= d + near_tol:
        best, d = probe_cat, cands[probe_cat]
    if not apply:
        return best
    _apply_cat(p, best)
    p.dist = d
    p.cands = cands
    p.method = "tdb"
    conf = 1.0 if probe_cat == best else (0.8 if probe_cat is None else 0.6)
    if d > near_tol:
        conf *= max(0.2, 1.0 - (d - near_tol) / max(far_tol, 1e-9))
    p.confidence = round(conf, 2)
    if probe_cat is not None and probe_cat != best:
        p.flags.append(f"local equilibria suggest {_cat_label(probe_cat)}")
    if d > far_tol:
        p.flags.append(f"far from calculated boundary ({_dist_text(inf, d)})")
    return best


def _dist_text(inf: Inferencer, d: float) -> str:
    if inf.sys.kind == "binary":
        return f"{d:.3f} of the axes, ~{d * inf.T_scale:.0f} K or ~{d:.3f} in x"
    return f"{d:.3f} in mole fraction"


def _similarity(c: tuple, d: tuple) -> int:
    """How much two categories play the same role (same measured phase / same partner phases)."""
    if c == d:
        return 10
    if c[0] != d[0] or c[0] == "INV":
        return 0
    return 2 * (c[1] == d[1]) + (set(c[2]) == set(d[2]))


def group_consensus(pts: List[DPoint], near_tol: float, min_share: float = 0.25) -> Dict:
    """Majority vote over a group's categories.  Ambiguous points (equally supported alternatives, e.g. a two-phase
    field too narrow to be drawn) take the alternative that plays the group's role; points whose measured phase
    differs from the group's are reassigned when their own evidence (p.cands) supports a majority category,
    otherwise flagged."""
    live = [p for p in pts if not p.excluded]
    n = len(live)
    counts: Dict[tuple, int] = {}
    for p in live:
        if p.category and not p.ties:
            counts[p.category] = counts.get(p.category, 0) + 1
    if not counts:
        for p in live:
            if p.category:
                counts[p.category] = counts.get(p.category, 0) + 1
    if not counts:
        return {"n": n, "labels": {}, "agree": 0.0}
    dominant = [c for c, k in counts.items() if k / max(n, 1) >= min_share and (k >= 2 or n == 1)]
    if not dominant:
        dominant = [max(counts, key=counts.get)]
    dominant.sort(key=lambda c: -counts[c])
    phases: Dict[str, int] = {}
    for p in live:
        if p.category and p.category[0] != "INV" and not p.ties:
            phases[p.phase] = phases.get(p.phase, 0) + 1
    maj_phase = max(phases, key=phases.get) if phases else None
    agree = sum(counts.get(c, 0) for c in dominant) / max(n, 1)
    for p in live:
        if not p.category or p.method in ("legend", "override") or p.kind == "curve":
            continue
        if p.ties:
            opts = [p.category] + list(p.ties)
            score = lambda c: max(_similarity(c, d) * counts.get(d, 0) for d in dominant)   # noqa: E731
            best = max(opts, key=score)
            if score(best) > 0 and sum(score(c) == score(best) for c in opts) == 1:
                _apply_cat(p, best)
                p.flags = [f for f in p.flags if not f.startswith("ambiguous")]
                p.flags.append(f"ambiguous; resolved like the rest of the group ({_cat_label(dominant[0])})")
                p.method += "+group"
                p.confidence = 0.7
                p.ties = []
            continue
        if p.category in dominant or (p.category[0] == "INV" and any(c[0] == "INV" for c in dominant)):
            continue
        if maj_phase is not None and p.category[0] != "INV" and p.phase == maj_phase:
            continue                                  # same measured phase, other partner (e.g. both liquidus sides)
        # the point's own evidence: categories almost as well supported as its best one; take the one that plays
        # the group's role best (same measured phase / same partner)
        lim = max(2 * near_tol, (p.dist or 0) + 0.01)
        near = [c for c, d in p.cands.items() if d <= lim]
        role = lambda c: max(_similarity(c, d) * counts.get(d, 0) for d in dominant)   # noqa: E731
        was = _cat_label(p.category)
        best = max(near, key=role) if near else None
        if best is not None and role(best) > role(p.category):
            _apply_cat(p, best)
            p.dist = p.cands[best]
            p.method = (p.method or "") + "+group"
            p.confidence = round(min(p.confidence, 0.7), 2)
            p.flags = [f for f in p.flags if not f.startswith(("local equilibria", "ambiguous"))]
            p.flags.append(f"reassigned like the rest of the group (own best evidence: {was})")
        else:
            p.flags.append(f"group outlier (group mostly {_cat_label(dominant[0])})")
    final: Dict[str, int] = {}
    for p in live:
        k = _cat_label(p.category) if p.category else "unassigned"
        final[k] = final.get(k, 0) + 1
    return {"n": n, "labels": dict(sorted(final.items(), key=lambda kv: -kv[1])), "agree": agree}


# ---------------------------------------------------------------------------------------------------------
# rows (tie-lines, invariants, broken tie-lines)
# ---------------------------------------------------------------------------------------------------------
def build_row_assignments(system: System, pts: List[DPoint], inf: Optional[Inferencer], cons: Dict[str, Dict],
                          pair_T_tol: Optional[float], inv_T_tol: float) -> Tuple[List[Dict], Dict]:
    """Give every usable point a row id.  Returns (rows [{phases, kind, line}], stats)."""
    rows: List[Dict] = []
    stats = {"tielines": 0, "invariants": [], "broken": 0, "triangles": 0, "line_rows": 0}
    x_el = system.indep[0]
    live = [p for p in pts if not p.excluded and p.category]

    def new_row(phases, kind, members):
        rows.append({"phases": list(phases), "kind": kind, "line": all(m.kind == "curve" for m in members)})
        for p in members:
            p.row = len(rows) - 1
        return len(rows) - 1

    # --- invariants (binary) --------------------------------------------------------------------------
    inv_pts = [p for p in live if p.category[0] == "INV"]
    for phases in sorted({p.category[1] for p in inv_pts}):
        cl = sorted([p for p in inv_pts if p.category[1] == phases], key=lambda p: p.T)
        clusters: List[List[DPoint]] = []
        for p in cl:
            if clusters and abs(p.T - np.mean([q.T for q in clusters[-1]])) <= inv_T_tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        for c in clusters:
            order = None
            for p in c:
                sp = cons.get(p.group, {})
                if sp.get("kind") == "INV" and sp.get("phases") and sorted(sp["phases"]) == list(phases):
                    order = sp["phases"]
            target = inf.invariant_target(phases) if inf is not None else None
            if order is None:
                order = sorted(phases, key=lambda n: target[n]) if target else list(phases)
            slots: Dict[str, DPoint] = {}
            curve = sorted([p for p in c if p.kind == "curve"], key=lambda p: p.X[x_el])
            members = [p for p in c if p.kind != "curve"]
            if curve and target is None:
                # a drawn invariant line: its ends touch the outer single-phase regions
                for q in (curve[0], curve[-1]):
                    if q.singles:
                        n = min(q.singles, key=q.singles.get)
                        if n in phases and n not in slots:
                            slots[n] = q
            if curve and target is not None:
                slots.setdefault(order[0], curve[0])
                slots.setdefault(order[-1], curve[-1])
            # markers: each phase once; evidence = the single-phase regions touching the point (figure), the
            # calculated invariant compositions (TDB) or the explicit per-point phase
            best = None
            free_ph = [n for n in order if n not in slots]
            for k in range(min(len(members), len(free_ph)), 0, -1):
                for sel in itertools.permutations(members, k):
                    for phs in itertools.permutations(free_ph, k):
                        cost, ok = 0.0, True
                        for q, n in zip(sel, phs):
                            if q.phase and q.method in ("explicit", "override"):
                                ok &= (q.phase == n)
                            elif q.singles:
                                ok &= n in q.singles
                                cost += q.singles.get(n, 99) / 10.0
                            elif target is not None:
                                cost += abs(q.X[x_el] - target[n])
                                ok &= abs(q.X[x_el] - target[n]) <= 0.1
                            else:
                                cost += abs(sorted(members, key=lambda r: r.X[x_el]).index(q)
                                            - (order.index(n) if len(members) == 3 else
                                               (0 if order.index(n) == 0 else len(members) - 1))) * 10
                        if ok and (best is None or cost < best[0]):
                            best = (cost, sel, phs)
                if best:
                    break
            if best:
                for q, n in zip(best[1], best[2]):
                    slots[n] = q
            rid = new_row(order, "invariant", c)
            for p in c:
                hit = [n for n, q in slots.items() if q is p]
                if hit:
                    p.phase, p.measured = hit[0], True
                    p.coexisting = [n for n in order if n != hit[0]]
                else:
                    p.measured, p.phase = False, ""
                    p.flags.append("used for the invariant temperature only")
            stats["invariants"].append({"phases": order, "T": float(np.mean([p.T for p in c])),
                                        "n_comp": len(slots), "row": rid})
    # --- tie-triangles (ternary) ----------------------------------------------------------------------
    tri = [p for p in live if p.category[0] == "TRI" and p.row < 0 and p.phase]
    by: Dict[tuple, List[DPoint]] = {}
    for p in tri:
        key = (tuple(sorted([p.category[1], *p.category[2]])), p.tie if p.tie.startswith("tri") else p.group)
        by.setdefault(key, []).append(p)
    for (phases, _), ms in by.items():
        chunks: List[Dict[str, DPoint]] = [{}]
        for p in ms:
            for ch in chunks:
                if p.phase not in ch:
                    ch[p.phase] = p
                    break
            else:
                chunks.append({p.phase: p})
        for ch in chunks:
            new_row(sorted(phases), "tie-triangle", list(ch.values()))
            stats["triangles"] += 1
    # --- drawn tie-lines (tie ids) ---------------------------------------------------------------------
    two = [p for p in live if p.category[0] == "2P" and p.row < 0]
    ids: Dict[str, List[DPoint]] = {}
    for p in two:
        if p.tie:
            ids.setdefault(p.tie, []).append(p)
    for tid, ms in ids.items():
        if len(ms) == 2 and ms[0].phase in ms[1].coexisting and ms[1].phase in ms[0].coexisting:
            new_row([ms[0].phase, ms[1].phase], "tie-line", ms)
            stats["tielines"] += 1
        elif len(ms) == 2:
            for k, p in enumerate(ms):
                p.flags.append(f"drawn tie-line partner #{ms[1 - k].idx} is {_cat_label(ms[1 - k].category)} "
                               "(not the other side of the same field): kept as a broken tie-line")
    # --- explicit tie-line groups: ends at the same T -------------------------------------------------
    for g in sorted({p.group for p in pts}):
        sp = cons.get(g, {})
        if sp.get("kind") != "TIE":
            continue
        gp = sorted([p for p in pts if p.group == g and p.row < 0 and not p.excluded], key=lambda p: p.T)
        tol = pair_T_tol if pair_T_tol is not None else 2.0
        used = set()
        for i, p in enumerate(gp):
            if id(p) in used:
                continue
            q = next((q for q in gp[i + 1:] if id(q) not in used and abs(q.T - p.T) <= tol), None)
            if q is None:
                p.flags.append("tie-line end without a partner at the same T: phase unknown")
                continue
            lo, hi = sorted([p, q], key=lambda r: r.X[x_el])
            _apply_cat(lo, ("2P", sp["phases"][0], (sp["phases"][1],)))
            _apply_cat(hi, ("2P", sp["phases"][1], (sp["phases"][0],)))
            new_row(sp["phases"], "tie-line", [lo, hi])
            used |= {id(p), id(q)}
            stats["tielines"] += 1
    # --- pairing by temperature (binary) ---------------------------------------------------------------
    if pair_T_tol is not None and system.kind == "binary":
        two = [p for p in live if p.category and p.category[0] == "2P" and p.row < 0 and p.kind != "curve"]
        cands = []
        for a, b in itertools.combinations(two, 2):
            if a.phase in b.coexisting and b.phase in a.coexisting and a.phase != b.phase \
                    and abs(a.T - b.T) <= pair_T_tol:
                cands.append((abs(a.T - b.T), a.idx, b.idx, a, b))
        used = set()
        for _, _, _, a, b in sorted(cands, key=lambda t: t[:3]):
            if a.idx in used or b.idx in used:
                continue
            new_row([a.phase, b.phase], "tie-line", [a, b])
            used |= {a.idx, b.idx}
            stats["tielines"] += 1
    # --- the rest: one boundary point per row (partner composition null) ------------------------------
    for p in live:
        if p.row >= 0 or not p.phase:
            continue
        new_row([p.phase] + list(p.coexisting), "boundary", [p])
        stats["broken"] += 1
    stats["line_rows"] = sum(1 for r in rows if r["line"])
    return rows, stats


# ---------------------------------------------------------------------------------------------------------
# pure builders (their source is embedded by show_code)
# ---------------------------------------------------------------------------------------------------------
def rows_from_points(records, indep):
    """ZPF rows from per-point records.

    records: [{"row": id, "row_phases": [...], "phase": P or "", "measured": bool, "X": {el: x}, "T_K": T}, ...]
    Returns {row id: {"T": mean T of the members (K), "values": [[phase, indep, [x or None]], ...]}}."""
    rows = {}
    for r in records:
        if r["row"] < 0:
            continue
        rows.setdefault(r["row"], []).append(r)
    out = {}
    for rid in sorted(rows):
        members = rows[rid]
        T = round(sum(m["T_K"] for m in members) / len(members), 2)
        values = []
        for ph in members[0]["row_phases"]:
            src = [m for m in members if m["measured"] and m["phase"] == ph]
            if src:
                xs = [round(sum(m["X"][e] for m in src) / len(src), 6) for e in indep]
            else:
                xs = [None] * len(indep)
            values.append([ph, list(indep), xs])
        out[rid] = {"T": T, "values": values}
    return out


def datasets_from_rows(rows, row_files, components, meta):
    """ESPEI ZPF dataset dicts, one per file name: {file: dataset}.

    rows: output of rows_from_points; row_files: {row id: file name}; meta: {file: {"reference", "comment",
    optional "bibtex"}}."""
    out = {}
    for rid in sorted(rows):
        f = row_files[rid]
        d = out.setdefault(f, {"T": [], "values": []})
        d["T"].append(rows[rid]["T"])
        d["values"].append(rows[rid]["values"])
    datasets = {}
    for f, d in out.items():
        phases = sorted({v[0] for row in d["values"] for v in row})
        ds = {"components": sorted(set(components) | {"VA"}), "phases": phases,
              "conditions": {"P": 101325.0, "T": d["T"]}, "broadcast_conditions": False, "output": "ZPF",
              "values": d["values"], "reference": meta[f]["reference"], "comment": meta[f]["comment"]}
        if meta[f].get("bibtex"):
            ds["bibtex"] = meta[f]["bibtex"]
        datasets[f] = ds
    return datasets


# ---------------------------------------------------------------------------------------------------------
# the whole conversion
# ---------------------------------------------------------------------------------------------------------
def convert(components: Sequence[str], out_dir: str, csvs: Sequence[str] = (), points: Sequence = (),
            image: Optional[str] = None, axes: Optional[Dict] = None, regions: Optional[Sequence[Dict]] = None,
            name_map: Optional[Dict[str, str]] = None, accept_suggestions: bool = False,
            assignments: Optional[Dict] = None, overrides: Optional[Dict] = None, exclude_points: Sequence = (),
            tdb: Optional[str] = None, composition_element: Optional[str] = None, composition_unit: str = "fraction",
            temperature_unit: str = "K", swap_axes: bool = False, temperature: Optional[float] = None,
            ternary_axes: Optional[Sequence[str]] = None, line_sampling: Optional[Dict] = None,
            pair_tielines=None, invariant_T_tol: float = 5.0, near_tol: float = 0.012, far_tol: float = 0.03,
            reference: str = "", comment: str = "", bibtex: str = "", figure: str = "", prefix: Optional[str] = None,
            split_by: str = "phases", phases: Optional[Sequence[str]] = None, boundary_color=None,
            tie_lines=None, preview_only: bool = False, summary_csv: Optional[str] = None,
            overlay_png: Optional[str] = None, preview_png: Optional[str] = None,
            right: Optional[str] = None, corners: Optional[Dict[str, str]] = None) -> Dict:
    """Run the full conversion; returns a result dict (datasets, files, per-point table ...).

    right / corners only orient the data plot (overlay_png): binary right element, ternary {left, right, top}
    corners.  The ZPF datasets store compositions by element, so they do not depend on the orientation."""
    from .expdata import check, write_dataset
    els = sorted({str(c).upper() for c in components} - {"VA"})
    if len(els) not in (2, 3):
        raise ValueError(f"components must be a binary or ternary system, got {els}")
    kind = "binary" if len(els) == 2 else "ternary"
    T_unit = "C" if str(temperature_unit).upper().startswith("C") else "K"
    db = None
    if tdb:
        from . import pcache
        db = pcache.get_db(tdb)
        missing = [e for e in els if e not in {str(x) for x in db.elements}]
        if missing:
            raise ValueError(f"elements {missing} are not in {Path(tdb).name}")
    masses = _masses(db, els)
    if kind == "binary":
        xe = (composition_element or [str(c).upper() for c in components if str(c).upper() != "VA"][-1]).upper()
        if xe not in els:
            raise ValueError(f"composition_element {xe} not in {els}")
        indep, T_fixed = [xe], None
    else:
        if temperature is None:
            raise ValueError("a ternary isotherm needs temperature (and temperature_unit)")
        T_fixed = float(temperature) + (KELVIN if T_unit == "C" else 0.0)
        ax = [e.upper() for e in (ternary_axes or [])]
        indep = ax if len(ax) == 2 else els[1:]
    system = System(kind=kind, elements=els, indep=indep, comp_unit=_norm_unit(composition_unit), T_unit=T_unit,
                    T_fixed=T_fixed, masses=masses)
    pts = load_points(system, csvs, points, swap_axes=swap_axes, ternary_axes=ternary_axes,
                      line_sampling=line_sampling)
    excl = {int(i) for i in (exclude_points or [])}
    for p in pts:
        if p.idx in excl:
            p.excluded = True
            p.flags.append("excluded by request")
    groups = list(dict.fromkeys(p.group for p in pts))
    spec_raw, unknown = match_groups(groups, assignments or {})
    if unknown:
        raise ValueError(f"assignment keys {unknown} match no group; groups are {groups}")
    # ---- names ---------------------------------------------------------------------------------------------
    liquid = "LIQUID"
    if db is not None:
        liq = [p for p in db.phases if p.upper().startswith("LIQ") or db.phases[p].model_hints.get("liquid")]
        liquid = "LIQUID" if "LIQUID" in db.phases or not liq else sorted(liq)[0]
    used = []
    for r in regions or []:
        used += split_label(r["label"]) if r.get("label") else [str(x) for x in r.get("phases", [])]
    for v in spec_raw.values():
        used += spec_names(v)
    for v in (overrides or {}).values():
        used += spec_names(v)
    names, name_notes = resolve_names(used, name_map, sorted(db.phases) if db is not None else None, liquid,
                                      accept_suggestions)
    cons = {g: parse_spec(spec_raw.get(g), names, liquid) for g in groups}
    # ---- the figure --------------------------------------------------------------------------------------
    figr = None
    n_ties = 0
    if image:
        figr = FigureRegions(image, axes or {}, system, pts, regions or [], names, boundary_color, swap_axes,
                             ternary_axes)
        if tie_lines:
            n_ties = figr.link_ties(pts, color=tie_lines.get("color") if isinstance(tie_lines, dict) else None)
    elif tie_lines:
        raise ValueError("tie_lines needs the image (and axes) to see which points are joined by lines")
    inf = Inferencer(tdb, system, pts, phases) if db is not None else None
    T_scale = inf.T_scale if inf else (max(300.0, 1.25 * float(np.ptp([p.T for p in pts])) + 100.0)
                                       if kind == "binary" else 1.0)
    # ---- assignment ---------------------------------------------------------------------------------------
    ov = {int(k): v for k, v in (overrides or {}).items()}
    for p in pts:
        if p.excluded:
            continue
        c = cons[p.group]
        if p.idx in ov:
            oc = parse_spec(ov[p.idx], names, liquid)
            if oc["kind"] == "2P" and oc["phase"] and oc["coexisting"]:
                _apply_cat(p, ("2P", oc["phase"], tuple(oc["coexisting"])) if kind == "binary" or
                           len(oc["coexisting"]) == 1 else ("TRI", oc["phase"], tuple(sorted(oc["coexisting"]))))
            elif oc["kind"] == "INV" and oc["phases"]:
                _apply_cat(p, ("INV", tuple(sorted(oc["phases"]))) if kind == "binary"
                           else ("TRI", oc["phase"] or "", tuple(sorted(oc["phases"]))))
            else:
                raise ValueError(f"override for point {p.idx} must give phase and coexisting (or phases)")
            p.method, p.confidence = "override", 1.0
            continue
        raw_x = p.raw_comp.get(system.indep[0], p.raw[0])
        if c.get("with_ranges") and not c.get("coexisting"):
            hit = [n for lo, hi, n in c["with_ranges"] if min(lo, hi) <= raw_x <= max(lo, hi)]
            if hit:
                c = {**c, "coexisting": hit[:1]}
        if c.get("phase_ranges") and not c.get("phase"):
            hit = [n for lo, hi, n in c["phase_ranges"] if min(lo, hi) <= raw_x <= max(lo, hi)]
            if hit:
                c = {**c, "phase": hit[0]}
        if c.get("kind") == "2P" and c.get("phase") and c.get("coexisting"):
            c = {**c, "complete": True}
        if c["complete"]:
            if c["kind"] == "2P":
                co = tuple(c["coexisting"])
                _apply_cat(p, ("2P", c["phase"], co) if len(co) == 1 or kind == "binary"
                           else ("TRI", c["phase"], tuple(sorted(co))))
            elif c["kind"] == "INV":
                _apply_cat(p, ("INV", tuple(sorted(c["phases"]))) if kind == "binary"
                           else ("TRI", "", tuple(sorted(c["phases"]))))
            elif c["kind"] == "TIE":
                p.category = ("2P", "", tuple(c["phases"]))
            p.method, p.confidence = "legend", 1.0
            if figr is not None and regions:           # figure evidence for cross-checks and invariant phases
                sets, singles, _ = figr.around(p)
                p.singles = singles
            continue
        if figr is not None and regions:
            assign_from_figure(figr, p, c, liquid)
    if figr is not None and regions:
        tie_consistency(pts)
    if regions and figr is None:
        todo = [p for p in pts if not p.excluded and not p.category]
        assign_by_label_geometry(system, todo, regions, names, T_scale, swap_axes, ternary_axes)
    # ternary: coexisting phases from drawn tie-line partners (legend gives each marker's phase)
    if kind == "ternary":
        by_tie: Dict[str, List[DPoint]] = {}
        for p in pts:
            if p.tie and not p.excluded:
                by_tie.setdefault(p.tie, []).append(p)
        for tid, ms in by_tie.items():
            for p in ms:
                c = cons[p.group]
                if p.category or not c.get("phase") or c.get("coexisting"):
                    continue
                others = sorted({q_phase for q in ms if q is not p for q_phase in [cons[q.group].get("phase")
                                                                               or q.phase] if q_phase})
                if tid.startswith("tri") and len(others) == 2 and c["phase"] not in others:
                    _apply_cat(p, ("TRI", c["phase"], tuple(others)))
                elif len(ms) == 2 and len(others) == 1 and others[0] != c["phase"]:
                    _apply_cat(p, ("2P", c["phase"], tuple(others)))
                else:
                    continue
                p.method, p.confidence = "legend+tie-line", 1.0
    # ---- optional TDB -------------------------------------------------------------------------------------
    group_info: Dict[str, Dict] = {}
    todo = [p for p in pts if not p.excluded and not p.category]
    if todo and inf is not None:
        for p in todo:
            infer_point(inf, p, near_tol, far_tol, cons[p.group], liquid)
    for g in groups:
        gp = [p for p in pts if p.group == g]
        if any(p.method.startswith(("tdb", "figure", "labels")) for p in gp):
            group_info[g] = group_consensus(gp, near_tol if not figr else 0.02)
        else:
            labs: Dict[str, int] = {}
            for p in gp:
                if not p.excluded:
                    k = _cat_label(p.category) if p.category else "unassigned"
                    labs[k] = labs.get(k, 0) + 1
            group_info[g] = {"n": sum(not p.excluded for p in gp), "labels": labs, "agree": 1.0}
    disagree = []
    if inf is not None:
        for p in pts:
            if p.excluded or not p.category or p.method.startswith("tdb"):
                continue
            if (p.category[0] == "2P" and not p.category[1]) or (p.category[0] == "TRI" and not p.category[1]):
                continue
            cands = inf.candidates(p)
            own = cands.get(p.category)
            if own is not None and own > far_tol:
                p.flags.append(f"far from calculated boundary ({_dist_text(inf, own)})")
            tcat = infer_point(inf, p, near_tol, far_tol, None, liquid, apply=False)
            p.tdb_cat = tcat
            clear = own is None or (tcat in cands and own > cands[tcat] + near_tol)   # the TDB clearly prefers it
            if tcat and tcat != p.category and clear and not (tcat[0] == "INV" and p.category[0] == "INV"):
                p.flags.append(f"TDB suggests {_cat_label(tcat)} (kept the figure's assignment)")
                disagree.append(p.idx)
    # ternary tie-triangle from the legend 'invariant' without a per-point phase
    for p in pts:
        if p.category and p.category[0] == "TRI" and not p.category[1]:
            if p.singles:
                n = min(p.singles, key=p.singles.get)
                if n in p.category[2]:
                    _apply_cat(p, ("TRI", n, tuple(sorted(set(p.category[2]) - {n}))))
                    continue
            if inf is not None:
                opts = [(d, c) for c, d in inf.candidates(p).items()
                        if c[0] == "TRI" and set((c[1], *c[2])) == set(p.category[2])]
                if opts:
                    _apply_cat(p, min(opts)[1])
                    continue
            p.flags.append("tie-triangle corner: which phase? (give the phase per group or an override)")
            p.category = None
    for p in pts:
        if not p.excluded and not p.category and not any(f.startswith(("no labelled", "only one", "regions around"))
                                                         for f in p.flags):
            p.flags.append("unassigned: give its group an assignment, region labels, an override, or a tdb")
    # ---- rows ------------------------------------------------------------------------------------------------
    pair_tol = None
    if pair_tielines:
        pair_tol = float(pair_tielines.get("T_tol", 2.0)) if isinstance(pair_tielines, dict) else 2.0
    rows, stats = build_row_assignments(system, pts, inf, cons, pair_tol, float(invariant_T_tol))
    # ---- files ------------------------------------------------------------------------------------------------
    sysname = "-".join(els)
    prefix = re.sub(r"[^\w.-]+", "_", prefix or f"{sysname}_digitized")
    methods = sorted({p.method for p in pts if p.method})
    src = figure or (Path(image).name if image else "") or ", ".join(sorted({Path(c).name for c in csvs})) \
        or "inline points"
    base_comment = (f"{TOOL_TAG}: digitized from {src}; phases from: {', '.join(methods) or 'none'}"
                    + (f" (TDB {Path(tdb).name})" if tdb else "")
                    + (f"; tie-lines paired within {pair_tol:g} K" if pair_tol is not None else "")
                    + (f"; composition converted from {system.comp_unit}" if system.comp_unit != "fraction" else "")
                    + ("; T converted from °C" if T_unit == "C" and kind == "binary" else "")
                    + (f"; isotherm {T_fixed:.2f} K" if kind == "ternary" else "")
                    + (f". {comment}" if comment else ""))
    row_files: Dict[int, str] = {}
    for rid, r in enumerate(rows):
        line = "_line" if r["line"] else ""
        if split_by == "none":
            f = f"{prefix}{line}.json"
        elif split_by == "group":
            gs = sorted({p.group for p in pts if p.row == rid})
            f = f"{prefix}_{re.sub(r'[^\w.-]+', '_', gs[0])}{line}.json"
        else:
            tag = "_inv" if r["kind"] in ("invariant", "tie-triangle") else ""
            f = f"{prefix}_{'-'.join(sorted(set(r['phases'])))}{tag}{line}.json"
        row_files[rid] = f
    records = [{"row": p.row, "row_phases": rows[p.row]["phases"] if p.row >= 0 else [], "phase": p.phase,
                "measured": bool(p.measured), "X": {e: p.X[e] for e in system.indep}, "T_K": p.T} for p in pts]
    zrows = rows_from_points(records, system.indep)
    for rid in [rid for rid, r in zrows.items() if all(all(x is None for x in v[2]) for v in r["values"])]:
        for p in pts:
            if p.row == rid:
                p.flags.append("row dropped: no measured composition")
                p.row = -1
        zrows.pop(rid)
    for i, rec in enumerate(records):
        rec["row"] = pts[i].row
    meta = {}
    for f in sorted(set(row_files[r] for r in zrows)):
        rids = [r for r in zrows if row_files[r] == f]
        gs = sorted({p.group for p in pts if p.row in rids})
        line_note = ("; PSEUDO-DATA digitized from a drawn line (an assessed diagram), not experimental points"
                     if f.endswith("_line.json") else "")
        meta[f] = {"reference": reference or f"digitized {src}",
                   "comment": base_comment + f"; groups: {', '.join(gs)}" + line_note, "bibtex": bibtex}
    datasets = datasets_from_rows(zrows, {r: row_files[r] for r in zrows}, els, meta)
    for f, ds in datasets.items():
        check(json.loads(json.dumps(ds)))
    files, chk, n_loaded = [], {"valid": [], "invalid": {}}, 0
    if not preview_only and datasets:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for old in od.glob(f"{prefix}*.json"):             # stale files of an earlier run of this tool
            try:
                if json.loads(old.read_text()).get("comment", "").startswith(TOOL_TAG) and old.name not in datasets:
                    old.unlink()
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        files = [write_dataset(ds, str(od / f)) for f, ds in datasets.items()]
        from espei.datasets import load_datasets
        from .expdata import check_datasets_dir
        chk = check_datasets_dir(str(od))
        n_loaded = len(load_datasets(files).all())
    res = {"system": system, "points": pts, "rows": rows, "zrows": zrows, "row_files": row_files, "stats": stats,
           "groups": group_info, "cons": cons, "datasets": datasets, "files": files, "check": chk,
           "n_loaded": n_loaded, "inf": inf, "records": records, "meta": meta, "prefix": prefix, "names": names,
           "name_notes": name_notes, "figr": figr, "disagree": disagree, "preview_only": preview_only,
           "out_dir": out_dir, "masses": masses, "n_ties": n_ties, "regions_given": bool(regions)}
    if summary_csv:
        res["summary_csv"] = write_summary(res, summary_csv)
    if figr is not None and preview_png:
        n_un = sum(1 for p in pts if not p.excluded and not p.category)
        res["preview_png"] = figr.preview(pts, preview_png, rows=rows,
                                          title=f"regions from the drawn lines; {len(pts)} points"
                                          + (f", {n_un} unassigned" if n_un else "")
                                          + "; dashed: rows with several compositions")
    if overlay_png:
        res["overlay_png"] = plot_overlay(res, overlay_png, right=right, corners=corners)
    return res


# ---------------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------------
SUMMARY_COLUMNS = ["point", "x", "T", "group", "phase", "coexisting", "method", "confidence", "flag"]


def write_summary(res: Dict, path: str) -> str:
    sysm: System = res["system"]
    extra = (["y"] if sysm.kind == "ternary" else []) + ["x_raw", "y_raw", "x_px", "y_px", "distance", "tdb_suggests",
                                                          "zpf_row", "zpf_file"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLUMNS + extra)
        for p in res["points"]:
            ph = p.phase or ("(T only)" if p.row >= 0 and not p.measured else "")
            row = [p.idx, f"{p.X[sysm.indep[0]]:.6f}", f"{p.T:.2f}", p.group, ph, "+".join(p.coexisting),
                   p.method + (" (drawn line)" if p.kind == "curve" and p.method else ""), f"{p.confidence:.2f}",
                   "; ".join(p.flags)]
            if sysm.kind == "ternary":
                row.append(f"{p.X[sysm.indep[1]]:.6f}")
            row += [f"{p.raw[0]:.6g}", f"{p.raw[1]:.6g}", "" if p.px is None else f"{p.px[0]:.1f}",
                    "" if p.px is None else f"{p.px[1]:.1f}", "" if p.dist is None else f"{p.dist:.4f}",
                    _cat_label(p.tdb_cat) if p.tdb_cat else "", p.row if p.row >= 0 else "",
                    res["row_files"].get(p.row, "") if p.row >= 0 else ""]
            w.writerow(row)
    return path


def _tern_xy(xb, xc):
    """Gibbs-triangle projection: balance at bottom left, indep[0] bottom right, indep[1] top."""
    xb, xc = np.asarray(xb, float), np.asarray(xc, float)
    return xb + 0.5 * xc, xc * math.sqrt(3) / 2


def display_order(sysm: "System", right: Optional[str] = None, corners: Optional[Dict[str, str]] = None
                  ) -> List[str]:
    """Binary [left, right] / ternary [bottom-left, bottom-right, top] of the data plot.  Default: the figure's own
    axes (binary x = the composition-axis element; ternary x/y elements at bottom-right/top)."""
    from .orientation import binary_order, ternary_order
    if sysm.kind == "binary":
        base = [e for e in sysm.elements if e != sysm.indep[0]] + [sysm.indep[0]]
        return binary_order(base, right=right) if right else base
    base = [e for e in sysm.elements if e not in sysm.indep] + list(sysm.indep[:2])
    return ternary_order(base, corners=corners, right=right) if (corners or right) else base


def plot_overlay(res: Dict, path: str, right: Optional[str] = None, corners: Optional[Dict[str, str]] = None) -> str:
    """Data-space plot: the digitized points by assigned phase pair (+ calculated lines when a TDB was given),
    oriented by ``right`` (binary) / ``corners`` (ternary) - by default like the source figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .mapping_tools import phase_color
    sysm: System = res["system"]
    inf: Optional[Inferencer] = res["inf"]
    pts: List[DPoint] = [p for p in res["points"] if not p.excluded]
    off = KELVIN if (sysm.T_unit == "C" and sysm.kind == "binary") else 0.0
    fig, ax = plt.subplots(figsize=(8.2, 5.8) if sysm.kind == "binary" else (7.4, 6.4))
    order = display_order(sysm, right, corners)
    res["display_order"] = order

    def frac(X, e):
        if e in X:
            return X[e]
        return 1.0 - sum(v for k, v in X.items() if k != e)   # the balance element

    def xy(X, T):
        if sysm.kind == "binary":
            return frac(X, order[1]), T - off
        return _tern_xy(frac(X, order[1]), frac(X, order[2]))
    if inf is not None:
        for ln in inf.lines:
            for k in range(2):
                P = np.array([xy(pt["phases"][k][1], pt["T"]) for pt in ln["points"]])
                ax.plot(P[:, 0], P[:, 1], color="0.6", lw=1.0, zorder=1)
        for iv in inf.invariants:
            P = np.array([xy(c, iv["T_K"]) for c in iv["compositions"].values()])
            if sysm.kind == "binary":
                ax.plot([P[:, 0].min(), P[:, 0].max()], [P[0, 1]] * 2, color="0.6", lw=1.0, ls="--", zorder=1)
            else:
                ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color="0.6", lw=1.0, zorder=1)
    if sysm.kind == "ternary":
        c = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2], [0, 0]])
        ax.plot(c[:, 0], c[:, 1], color="k", lw=1.2)
        for (x0, y0), name, ha in (((0, 0), order[0], "right"), ((1, 0), order[1], "left"),
                                   ((0.5, math.sqrt(3) / 2), order[2], "center")):
            ax.text(x0, y0 + (0.03 if ha == "center" else -0.05), name, ha=ha, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
    by_row: Dict[int, List[DPoint]] = {}
    for p in pts:
        if p.row >= 0 and p.measured:
            by_row.setdefault(p.row, []).append(p)
    for rid, ms in by_row.items():
        kind = res["rows"][rid]["kind"]
        if kind in ("tie-line", "invariant", "tie-triangle") and len(ms) >= 2:
            P = np.array([xy(p.X, p.T) for p in sorted(ms, key=lambda q: q.X[sysm.indep[0]])])
            if kind == "tie-triangle":
                P = np.vstack([P, P[:1]])
            ax.plot(P[:, 0], P[:, 1], color="k" if kind == "tie-line" else "crimson", lw=1.1,
                    ls="-" if kind == "tie-line" else "--", zorder=2)
    cats: Dict[str, List[DPoint]] = {}
    for p in pts:
        lab = (p.phase + (" | " + "+".join(p.coexisting) if p.coexisting else "")) if p.phase else \
            ("T only" if p.category else "unassigned")
        if p.row >= 0 and res["rows"][p.row]["kind"] == "invariant" and p.phase:
            lab = f"{p.phase} (invariant)"
        cats.setdefault(lab, []).append(p)
    markers = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "*"]
    for i, (lab, ms) in enumerate(sorted(cats.items())):
        P = np.array([xy(p.X, p.T) for p in ms])
        col = phase_color(ms[0].phase) if ms[0].phase else "0.4"
        edge = phase_color(ms[0].coexisting[0]) if ms[0].coexisting else "k"
        ax.scatter(P[:, 0], P[:, 1], s=34, marker=markers[i % len(markers)], facecolor=col, edgecolor=edge,
                   linewidth=1.3, zorder=4, label=f"{lab} ({len(ms)})")
    fl = [p for p in pts if warnings_of(p)]
    if fl:
        P = np.array([xy(p.X, p.T) for p in fl])
        ax.scatter(P[:, 0], P[:, 1], s=150, facecolor="none", edgecolor="red", linewidth=1.5, zorder=5,
                   label=f"flagged ({len(fl)})")
        for p in fl:
            ax.annotate(str(p.idx), xy(p.X, p.T), textcoords="offset points", xytext=(6, 5), fontsize=7, color="red",
                        zorder=6)
    if sysm.kind == "binary":
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Mole fraction {order[1].capitalize()}  ({order[0].capitalize()} left, "
                      f"{order[1].capitalize()} right)")
        ax.set_ylabel("Temperature (°C)" if off else "Temperature (K)")
        ys = [p.T - off for p in pts] or [0, 1]
        pad = 0.12 * (max(ys) - min(ys) + 50)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="phase | coexisting",
              title_fontsize=8)
    t = f"{'-'.join(sysm.elements)}: digitized points by assigned phase"
    if sysm.kind == "ternary":
        t += f" ({sysm.T_fixed - KELVIN:.0f} °C)"
    ax.set_title(t + ("" if inf is None else f"\ngrey lines: {Path(inf.tdb).name} (cross-check)"), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def result_text(res: Dict, rel=str) -> str:
    sysm: System = res["system"]
    pts: List[DPoint] = res["points"]
    st = res["stats"]
    live = [p for p in pts if not p.excluded]
    head = (f"{len(pts)} points in {len(res['groups'])} group(s) -> {len(res['zrows'])} ZPF rows in "
            f"{len(res['datasets'])} dataset(s)")
    if res["preview_only"]:
        head += " - PREVIEW ONLY, nothing written (call again without preview=true to write)"
    else:
        head += (f"; ESPEI check: {len(res['check']['valid'])} valid, {len(res['check']['invalid'])} invalid; "
                 f"load_datasets: {res['n_loaded']} loaded")
    L = [head]
    if sysm.kind == "binary":
        L.append(f"system {'-'.join(sysm.elements)}: x = X({sysm.indep[0]}) (from {sysm.comp_unit}), T from "
                 f"{'°C' if sysm.T_unit == 'C' else 'K'}")
    else:
        L.append(f"ternary isotherm {'-'.join(sysm.elements)} at {sysm.T_fixed:.2f} K; independent "
                 f"X({sysm.indep[0]}), X({sysm.indep[1]}) (from {sysm.comp_unit})")
    if res["names"]:
        L.append("phase names: " + ", ".join(f"{k} -> {v}" for k, v in res["names"].items() if k != v)
                 + (f" ({'; '.join(res['name_notes'])})" if res["name_notes"] else "")
                 + ". ESPEI phase names must match the phase models you will fit.")
    figr = res.get("figr")
    if res.get("n_ties"):
        L.append(f"drawn tie-lines/tie-triangles found between the points: {res['n_ties']}")
    if figr is not None and res.get("regions_given"):
        lab = [v for v in figr.labels.values() if not v["inferred"]]
        L.append(f"figure: {len(figr.info['regions'])} regions from the drawn lines, {len(lab)} labelled"
                 + (f"; {'; '.join(figr.inferred)}" if figr.inferred else ""))
        for msg in figr.conflicts + [f"label {u} is not inside a region (on a line?)" for u in figr.unplaced]:
            L.append("  WARNING: " + msg)
    L.append("assignments per group:")
    low = []
    for g, info in res["groups"].items():
        gp = [p for p in live if p.group == g]
        meth = sorted({p.method for p in gp if p.method})
        labs = ", ".join(f"{k} x{v}" for k, v in info["labels"].items())
        agree = info.get("agree", 1.0)
        L.append(f"  {g} ({info['n']}): {labs} [{'/'.join(meth) or 'unassigned'}"
                 + (f", {agree:.0%} agree" if any(m.startswith(("tdb", "figure", "labels")) for m in meth) else "")
                 + (", drawn line (pseudo-data)" if gp and all(p.kind == "curve" for p in gp) else "") + "]")
        if gp and (agree < 0.8 or np.mean([p.confidence for p in gp]) < 0.6 or any(not p.category for p in gp)):
            low.append(g)
    inv = "; ".join(f"{'+'.join(i['phases'])} at {i['T']:.1f} K ({i['n_comp']} compositions)" for i in st["invariants"])
    L.append(f"rows: {st['tielines']} tie-line(s) with both ends" + (f", {st['triangles']} tie-triangle(s)"
                                                                     if sysm.kind == "ternary" else "")
             + f", {len(st['invariants'])} invariant(s)" + (f" [{inv}]" if inv else "")
             + f", {st['broken']} single-boundary point(s) (partner composition null)"
             + (f"; {st['line_rows']} row(s) from drawn lines (pseudo-data, own files)" if st["line_rows"] else ""))
    if low:
        L.append("CHECK these groups (low agreement / confidence or unassigned points): " + ", ".join(low))
    if res["disagree"]:
        L.append(f"figure vs TDB: {len(res['disagree'])} point(s) disagree (figure kept): "
                 + ", ".join(f"#{i}" for i in res["disagree"][:30]))
    fl = [p for p in pts if warnings_of(p)]
    if fl:
        L.append(f"flagged points ({len(fl)}):")
        for p in fl[:25]:
            L.append(f"  #{p.idx} [{p.group}] x={p.X[sysm.indep[0]]:.4f}"
                     + (f", y={p.X[sysm.indep[1]]:.4f}" if sysm.kind == "ternary" else f", T={p.T:.1f} K")
                     + f" -> {_cat_label(p.category) if p.category else 'unassigned'}: {'; '.join(warnings_of(p))}")
        if len(fl) > 25:
            L.append(f"  ... {len(fl) - 25} more in the summary CSV")
    if res["check"]["invalid"]:
        L.append("INVALID: " + json.dumps(res["check"]["invalid"])[:600])
    if res["files"]:
        L.append("datasets: " + ", ".join(rel(f) for f in res["files"]))
    if res.get("summary_csv"):
        L.append(f"summary CSV: {rel(res['summary_csv'])}")
    if res.get("preview_png"):
        L.append(f"assignment preview (LOOK at it): {rel(res['preview_png'])}")
    if res.get("provenance_png"):
        L.append(f"extracted points on the original figure: {rel(res['provenance_png'])} - show it to the user next "
                 "to the assignment preview")
    if res.get("overlay_png"):
        o = [e.capitalize() for e in res.get("display_order") or []]
        side = ("" if not o else f" ({o[0]} left, {o[1]} right)" if len(o) == 2
                else f" ({o[0]} bottom-left, {o[1]} bottom-right, {o[2]} top)")
        L.append(f"data plot{side}: {rel(res['overlay_png'])} (orientation: right / corners)")
    if any(p.method.startswith(("tdb", "figure", "labels")) or "+group" in p.method for p in live):
        L.append("Region/TDB-derived assignments are suggestions: show the preview and flagged points to the user and "
                 "fix with overrides {point: {phase, coexisting}} / exclude_points before fitting.")
    if res["files"]:
        L.append(f"Next: plot_diagram_with_data(datasets_dir={rel(str(Path(res['files'][0]).parent))!r}"
                 + (")" if sysm.kind == "binary" else ")  [binary only]") + " to check the datasets visually.")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# agent tool
# ---------------------------------------------------------------------------------------------------------
def register(tb) -> None:
    S = {"type": "string"}
    N = {"type": "number"}
    STRS = {"type": "array", "items": S}

    @tb.add("digitized_to_zpf",
            "Digitized phase-diagram points -> validated ESPEI ZPF datasets, phases read off the FIGURE: give image + "
            "axes (as for digitize_points) + regions [{label: 'L+(Cr)', at: [x, y]}] (labels printed on the diagram); "
            "the plot is split into regions bounded by its drawn lines, unlabelled fields are inferred, each point "
            "gets its phase and coexisting phase(s) from the regions it borders. And/or legend assignments per group "
            "({phase, coexisting} or {boundary: liquidus|solidus|solvus|invariant|tieline}). name_map: paper name -> "
            "database phase. Pairs tie-lines, builds invariant/tie-triangle rows, converts at%/wt%/°C. Optional tdb: "
            "cross-check. Writes an assignment preview to LOOK at (preview=true: nothing else).",
            {"components": STRS, "csv": {"type": ["string", "array"], "items": S,
                                         "description": "CSV file(s) from digitize_points / digitize_curve"},
             "points": {"type": "array", "description": "inline [[x, y, group(, tie)], ...] in figure units"},
             "image": {**S, "description": "the figure (as for digitize_points)"}, "page": {"type": "integer"},
             "crop": {"type": "array", "items": N}, "dpi": {"type": "integer"},
             "axes": {"type": "object", "description": "calibration exactly as given to digitize_points"},
             "regions": {"type": "array", "description": "[{label: 'L+(Cr)', at: [x, y]} | {..., at_px: [px, py]}]; "
                                                         "ternary at: {EL: frac, EL: frac}"},
             "name_map": {"type": "object", "additionalProperties": S,
                          "description": "paper name -> database phase, e.g. {'(Cr)': 'BCC_A2', 'σ': 'SIGMA'}"},
             "accept_suggestions": {"type": "boolean"},
             "assignments": {"type": "object", "description": "{group (CSV series): spec}"},
             "overrides": {"type": "object", "description": "{point index: {phase, coexisting}}"},
             "exclude_points": {"type": "array", "items": {"type": "integer"}},
             "tdb": {**S, "description": "optional database cross-check / inference"},
             "composition_element": S,
             "composition_unit": {"type": "string", "enum": ["fraction", "at%", "wt%", "wt_fraction"]},
             "temperature_unit": {"type": "string", "enum": ["K", "C"]},
             "swap_axes": {"type": "boolean", "description": "x is temperature, y composition"},
             "temperature": {**N, "description": "ternary isotherm temperature (temperature_unit)"},
             "ternary_axes": {**STRS, "description": "ternary CSV without X(EL) columns: elements on x and y"},
             "line_sampling": {"type": "object", "description": "digitize_curve input: {T_step} | {x_step} (figure "
                                                               "units) | {step: fraction of axes, default 0.04}"},
             "pair_tielines": {"type": ["boolean", "object"], "description": "{T_tol: K}: pair same-T ends"},
             "tie_lines": {"type": ["boolean", "object"], "description": "find points joined by drawn straight lines "
                           "(tie-lines, tie-triangles) in the image; {color: ...} to use only that line colour"},
             "invariant_T_tol": N, "boundary_color": {**S, "description": "only lines of this colour bound regions"},
             "reference": S, "bibtex": S, "comment": S, "figure": {**S, "description": "e.g. 'Smith 1990 Fig. 3'"},
             "preview": {"type": "boolean"},
             "right": {**S, "description": "data plot: binary element on the right (datasets are unaffected)"},
             "corners": {"type": "object", "description": "data plot: ternary {left, right, top} corners"},
             "split_by": {"type": "string", "enum": ["phases", "group", "none"]}, "output_dir": S, "prefix": S},
            ["components"])
    def digitized_to_zpf(**args):
        res = run(tb, args)
        imgs = [res[k] for k in ("preview_png", "provenance_png", "overlay_png") if res.get(k)]
        return result_text(res, tb.rel) + res.get("note", ""), imgs


def run(tb, args: Dict, preview: Optional[bool] = None, write_images: bool = True) -> Dict:
    """The digitized_to_zpf tool call (also used by show_code, with preview=True, to recover the assignments)."""
    a = dict(args)
    known = {"components", "csv", "points", "image", "page", "crop", "dpi", "axes", "regions", "name_map",
             "accept_suggestions", "assignments", "overrides", "exclude_points", "tdb", "composition_element",
             "composition_unit", "temperature_unit", "swap_axes", "temperature", "ternary_axes", "line_sampling",
             "pair_tielines", "tie_lines", "invariant_T_tol", "boundary_color", "reference", "bibtex", "comment",
             "figure", "preview", "split_by", "output_dir", "prefix"}
    bad = sorted(set(a) - known)
    if bad:
        raise TypeError(f"unexpected argument(s) {bad}")
    if "components" not in a:
        raise TypeError("missing required argument 'components'")
    csv_arg = a.get("csv")
    csvs = [csv_arg] if isinstance(csv_arg, str) else list(csv_arg or [])
    csvs = [tb.p(c) for c in csvs]
    for c in csvs:
        if not Path(c).exists():
            raise FileNotFoundError(f"{c} not found")
    image = a.get("image")
    work = None
    if image:
        from .digitize import prepare_image
        work = prepare_image(tb.p(image), tb.outdir(None, "digitize"), a.get("page"), a.get("crop"),
                             a.get("dpi") or 300)
    note = ("\n(no image: region labels are used by position only - give image + axes for the segmentation)"
            if a.get("regions") and not image else "")
    tdb_path = tb._find_tdb(a["tdb"]) if a.get("tdb") else None
    els = sorted({str(c).upper() for c in a["components"]} - {"VA"})
    stem = Path(image).stem if image else (Path(csvs[0]).stem if csvs else "digitized")
    stem = re.sub(r"_(points|curve)$", "", stem)
    pre = re.sub(r"[^\w.-]+", "_", a.get("prefix") or f"{'-'.join(els)}_{stem}")
    prev = bool(a.get("preview")) if preview is None else preview
    od = tb.outdir(a.get("output_dir"), f"datasets/{pre}") if not prev else tb.out
    res = convert(a["components"], str(od), csvs=csvs, points=a.get("points") or [], image=work,
                  axes=a.get("axes"), regions=a.get("regions"), name_map=a.get("name_map"),
                  accept_suggestions=bool(a.get("accept_suggestions")), assignments=a.get("assignments"),
                  overrides=a.get("overrides"), exclude_points=a.get("exclude_points") or (), tdb=tdb_path,
                  composition_element=a.get("composition_element"),
                  composition_unit=a.get("composition_unit") or "fraction",
                  temperature_unit=a.get("temperature_unit") or "K", swap_axes=bool(a.get("swap_axes")),
                  temperature=a.get("temperature"), ternary_axes=a.get("ternary_axes"),
                  line_sampling=a.get("line_sampling"), pair_tielines=a.get("pair_tielines"),
                  invariant_T_tol=float(a.get("invariant_T_tol") or 5.0), reference=a.get("reference") or "",
                  comment=a.get("comment") or "", bibtex=a.get("bibtex") or "",
                  figure=a.get("figure") or (Path(image).name if image else ""), prefix=pre,
                  split_by=a.get("split_by") or "phases", boundary_color=a.get("boundary_color"),
                  tie_lines=a.get("tie_lines"), preview_only=prev,
                  summary_csv=tb.outp(None, f"{pre}_zpf_points.csv") if write_images else None,
                  overlay_png=tb.outp(None, f"{pre}_zpf_data.png") if write_images else None,
                  preview_png=tb.outp(None, f"{pre}_assignment_preview.png") if write_images else None,
                  right=a.get("right"), corners=a.get("corners"))
    res["note"] = note
    if work and a.get("axes") and csvs and write_images:
        # the digitized values on the original figure, next to the phase-assignment preview
        from .digitize import provenance_from_csvs
        try:
            res["provenance_png"] = provenance_from_csvs(work, a["axes"], csvs, tb.outp(None, f"{pre}_provenance.png"),
                                                         source=str(image))
        except Exception as e:  # noqa: BLE001 - a check figure must not break the conversion
            res["note"] += f"\n(provenance figure not drawn: {e})"
    return res
