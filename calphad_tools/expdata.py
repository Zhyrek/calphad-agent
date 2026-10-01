"""Feature 5: experimental phase-equilibrium data -> ESPEI ZPF datasets.

Supported inputs
----------------
* Thermo-Calc PARROT ``.POP`` / ``.DOP`` files (CREATE_NEW_EQUILIBRIUM /
  CHANGE_STATUS / SET_CONDITION / EXPERIMENT, including TABLE_HEAD blocks)
  -> converted fully automatically (tie-lines, liquidus/solvus points, ...).
* Thermo-Calc DATAPLOT ``.EXP`` files (the plot files often given as
  supplementary data).  These only contain x/y points per DATASET, so each
  dataset must be mapped to phases (``dataset_map``).  The LLM agent can do
  that mapping from the dataset titles/comments and the paper text, or the CLI
  asks you.
* CSV files in a simple wide format (e.g. WebPlotDigitizer output with phase
  columns added)::

      T_K, phase_1, X(CR)_1, phase_2, X(CR)_2, reference
      1800, LIQUID, 0.30,    BCC_A2,  ,        Smith 1990

  (``T_C`` for Celsius; empty cell = composition not measured -> null)
* Papers (PDF): :func:`pdf_text` / :func:`render_pdf_page` give the text or a
  page image to the LLM, which then calls :func:`make_zpf_dataset`.

ZPF format (ESPEI)::

    {"components": ["CR","FE","VA"], "phases": ["BCC_A2","LIQUID"],
     "conditions": {"P": 101325, "T": [1800, 1850]}, "broadcast_conditions": false,
     "output": "ZPF",
     "values": [[["LIQUID", ["CR"], [0.30]], ["BCC_A2", ["CR"], [null]]], ...],
     "reference": "...", "comment": "..."}
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ATOMIC_MASS = {
    "H": 1.008, "HE": 4.0026, "LI": 6.94, "BE": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "NE": 20.180, "NA": 22.990, "MG": 24.305, "AL": 26.982, "SI": 28.085,
    "P": 30.974, "S": 32.06, "CL": 35.45, "AR": 39.948, "K": 39.098, "CA": 40.078, "SC": 44.956,
    "TI": 47.867, "V": 50.942, "CR": 51.996, "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693,
    "CU": 63.546, "ZN": 65.38, "GA": 69.723, "GE": 72.630, "AS": 74.922, "SE": 78.971, "BR": 79.904,
    "KR": 83.798, "RB": 85.468, "SR": 87.62, "Y": 88.906, "ZR": 91.224, "NB": 92.906, "MO": 95.95,
    "TC": 98.0, "RU": 101.07, "RH": 102.91, "PD": 106.42, "AG": 107.87, "CD": 112.41, "IN": 114.82,
    "SN": 118.71, "SB": 121.76, "TE": 127.60, "I": 126.90, "XE": 131.29, "CS": 132.91, "BA": 137.33,
    "LA": 138.91, "CE": 140.12, "PR": 140.91, "ND": 144.24, "PM": 145.0, "SM": 150.36, "EU": 151.96,
    "GD": 157.25, "TB": 158.93, "DY": 162.50, "HO": 164.93, "ER": 167.26, "TM": 168.93, "YB": 173.05,
    "LU": 174.97, "HF": 178.49, "TA": 180.95, "W": 183.84, "RE": 186.21, "OS": 190.23, "IR": 192.22,
    "PT": 195.08, "AU": 196.97, "HG": 200.59, "TL": 204.38, "PB": 207.2, "BI": 208.98, "PO": 209.0,
    "TH": 232.04, "PA": 231.04, "U": 238.03, "NP": 237.0, "PU": 244.0,
}


class ZPFError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Core builder + validation
# ---------------------------------------------------------------------------
def make_zpf_dataset(components: Sequence[str], temperatures: Sequence[float],
                     values: Sequence[Sequence], reference: str = "", comment: str = "",
                     pressure: float = 101325.0, temperature_unit: str = "K",
                     phases: Optional[Sequence[str]] = None, bibtex: str = "") -> Dict:
    """Build + validate an ESPEI ZPF dataset.

    ``values[i]`` is the list of phase regions in equilibrium at ``temperatures[i]``:
    ``[[phase, [dependent comps], [mole fractions or None]], ...]``.
    """
    comps = sorted({c.upper() for c in components} | {"VA"})
    if len(temperatures) != len(values):
        raise ZPFError(f"{len(temperatures)} temperatures but {len(values)} value rows")
    T = [float(t) + (273.15 if temperature_unit.upper().startswith("C") else 0.0) for t in temperatures]
    vals = []
    used = set()
    for row in values:
        new_row = []
        for region in row:
            if len(region) != 3:
                raise ZPFError(f"each phase region must be [phase, [comps], [fractions]], got {region}")
            ph, cs, xs = region
            cs = [c.upper() for c in cs]
            xs = [None if (x is None or (isinstance(x, str) and x.strip().lower() in ("", "none", "null", "nan")))
                  else float(x) for x in xs]
            if len(cs) != len(xs):
                raise ZPFError(f"components {cs} and fractions {xs} differ in length")
            if all(x is None for x in xs):
                xs = [None] * len(cs)
            new_row.append([ph.upper(), cs, xs])
            used.add(ph.upper())
        vals.append(new_row)
    ds = {"components": comps, "phases": sorted(set(phases or []) | used),
          "conditions": {"P": pressure, "T": T}, "broadcast_conditions": False,
          "output": "ZPF", "values": vals, "reference": reference, "comment": comment}
    if bibtex:
        ds["bibtex"] = bibtex
    check(ds)
    return ds


def check(ds: Dict) -> None:
    """Validate with ESPEI's own checker, plus a few extra sanity checks."""
    from espei.datasets import DatasetError, check_dataset
    for row in ds["values"]:
        if len(row) < 2:
            raise ZPFError(f"a ZPF point needs >= 2 phases in equilibrium, got {row}")
        for ph, cs, xs in row:
            if any(x is not None and not (0.0 <= x <= 1.0) for x in xs):
                raise ZPFError(f"mole fractions must be in [0,1] (convert at.%/wt.%): {ph} {xs}")
            if len(cs) != len(ds["components"]) - 2:   # minus VA and dependent component
                raise ZPFError(f"{ph}: give {len(ds['components']) - 2} independent component(s) "
                               f"(system {ds['components']}), got {cs}")
    try:
        check_dataset(ds)
    except DatasetError as e:
        raise ZPFError(f"ESPEI rejected the dataset: {e}") from None


def write_dataset(ds: Dict, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ds, indent=2))
    return str(p)


def check_datasets_dir(directory: str) -> Dict:
    """Run ESPEI's loader/checker over every JSON file in a directory."""
    from espei.datasets import check_dataset, clean_dataset
    ok, bad = [], {}
    for f in sorted(Path(directory).rglob("*.json")):
        try:
            ds = json.loads(f.read_text())
            check_dataset(clean_dataset(ds) if "values" in ds else ds)
            ok.append(str(f))
        except Exception as e:  # noqa: BLE001
            bad[str(f)] = f"{type(e).__name__}: {e}"
    return {"valid": ok, "invalid": bad}


def to_mole_fraction(value: float, unit: str, component: str, other: str) -> float:
    """Binary conversion of wt.% / wt fraction / at.% into a mole fraction of ``component``."""
    u = unit.lower()
    if u in ("x", "mole", "mole_fraction", "mol"):
        return value
    if u in ("at%", "at.%", "at_percent", "atomic_percent", "mol%"):
        return value / 100.0
    if u in ("wt", "w", "wt_fraction", "weight_fraction", "mass_fraction", "wt%", "wt.%", "weight_percent", "mass%"):
        w = value / 100.0 if "%" in u or "percent" in u else value
        ma, mb = ATOMIC_MASS[component.upper()], ATOMIC_MASS[other.upper()]
        return (w / ma) / (w / ma + (1 - w) / mb)
    raise ZPFError(f"unknown composition unit {unit!r}")


# ---------------------------------------------------------------------------
# Phase-name resolution (POP files often use LIQ, BCC, FCC ...)
# ---------------------------------------------------------------------------
def phase_resolver(tdb_phases: Optional[Sequence[str]] = None, phase_map: Optional[Dict[str, str]] = None):
    tdb_phases = [p.upper() for p in (tdb_phases or [])]
    pm = {k.upper(): v.upper() for k, v in (phase_map or {}).items()}

    def resolve(name: str) -> str:
        n = name.upper().split("#")[0]
        if n in pm:
            return pm[n]
        if not tdb_phases or n in tdb_phases:
            return n
        cands = [p for p in tdb_phases if p.startswith(n)] or [p for p in tdb_phases if p.split("_")[0] == n]
        if len(cands) == 1:
            return cands[0]
        return n
    return resolve


# ---------------------------------------------------------------------------
# POP files
# ---------------------------------------------------------------------------
POP_COMMANDS = ["CREATE_NEW_EQUILIBRIUM", "CHANGE_STATUS", "SET_CONDITION", "EXPERIMENT", "LABEL",
                "SET_WEIGHT", "SET_REFERENCE_STATE", "SET_ALL_START_VALUES", "SET_START_VALUE",
                "ENTER_SYMBOL", "COMMENT", "TABLE_HEAD", "TABLE_VALUES", "TABLE_END", "SAVE_WORKSPACES",
                "SET_NUMERICAL_LIMITS", "FLUSH_BUFFER", "DEFINE_COMPONENTS", "EXPERIMENTAL"]


def _pop_kw(word: str) -> Optional[str]:
    w = word.upper().replace("-", "_")
    parts = w.split("_")
    for kw in POP_COMMANDS:
        kp = kw.split("_")
        if len(parts) <= len(kp) and all(k.startswith(p) for p, k in zip(parts, kp)):
            return kw
    return None


@dataclass
class PopEquilibrium:
    label: str = ""
    fixed: Dict[str, float] = field(default_factory=dict)       # phase -> amount
    conditions: Dict[str, float] = field(default_factory=dict)  # "T", "P", "X(CR)" ...
    experiments: Dict[str, float] = field(default_factory=dict)  # "X(LIQ,CR)", "T", "ACR(CR)" ...
    comment: str = ""


def _expand_pop_tables(lines: List[str]) -> List[str]:
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        w = ln.split()
        if w and _pop_kw(w[0]) == "TABLE_HEAD":
            start = int(re.findall(r"\d+", ln)[0]) if re.findall(r"\d+", ln) else 1
            tmpl, i = [], i + 1
            while i < len(lines) and not (lines[i].split() and _pop_kw(lines[i].split()[0]) == "TABLE_VALUES"):
                tmpl.append(lines[i])
                i += 1
            i += 1
            rows = []
            while i < len(lines) and not (lines[i].split() and _pop_kw(lines[i].split()[0]) == "TABLE_END"):
                if lines[i].strip() and not lines[i].strip().startswith("$"):
                    rows.append(lines[i].split())
                i += 1
            for n, row in enumerate(rows):
                for t in tmpl:
                    s = t.replace("@@", str(start + n))
                    s = re.sub(r"@(\d+)", lambda m: row[int(m.group(1)) - 1] if int(m.group(1)) <= len(row) else "", s)
                    out.append(s)
            i += 1
            continue
        out.append(ln)
        i += 1
    return out


def parse_pop(path: str) -> List[PopEquilibrium]:
    raw = Path(path).read_text(errors="replace").replace("\r", "")
    lines = []
    for ln in raw.split("\n"):
        s = ln.split("$", 1)[0].strip() if not ln.strip().startswith("$") else ""
        lines.append(s)
    lines = _expand_pop_tables([l for l in lines if l])
    eqs: List[PopEquilibrium] = []
    cur: Optional[PopEquilibrium] = None
    for ln in lines:
        w = ln.split(None, 1)
        kw = _pop_kw(w[0])
        rest = w[1] if len(w) > 1 else ""
        if kw == "CREATE_NEW_EQUILIBRIUM":
            cur = PopEquilibrium()
            eqs.append(cur)
        elif cur is None:
            continue
        elif kw == "CHANGE_STATUS":
            # CHANGE_STATUS PHASE LIQ,BCC=FIX 1 | CH-ST P LIQ=F 0
            m = re.match(r"(?:PH[A-Z]*|P)\s+(.*?)=\s*(FIX|F|FIXED|ENTERED|E|DORMANT|D|SUSPENDED|S)\S*\s*([-\d.Ee+]*)",
                         rest, re.I)
            if m and m.group(2).upper().startswith("F"):
                amt = float(m.group(3)) if m.group(3) else 0.0
                for ph in re.split(r"[\s,]+", m.group(1).strip()):
                    if ph:
                        cur.fixed[ph.upper()] = amt
        elif kw == "SET_CONDITION":
            for m in re.finditer(r"([A-Za-z]+(?:\([^)]*\))?)\s*=\s*([-\d.Ee+]+)", rest):
                cur.conditions[m.group(1).upper().replace(" ", "")] = float(m.group(2))
        elif kw in ("EXPERIMENT", "EXPERIMENTAL"):
            # names may contain dots (HM.T = heat capacity, not a temperature)
            for m in re.finditer(r"(?<![A-Za-z0-9_.])([A-Za-z][A-Za-z0-9_.]*(?:\([^)]*\))?)\s*=\s*([-\d.Ee+]+)"
                                 r"(?:\s*:\s*([-\d.Ee+%]+))?", rest):
                cur.experiments[m.group(1).upper().replace(" ", "")] = float(m.group(2))
        elif kw == "LABEL":
            cur.label = rest.strip()
        elif kw == "COMMENT":
            cur.comment = rest.strip()
    return eqs


def pop_to_zpf(path: str, components: Optional[Sequence[str]] = None, tdb_phases=None, phase_map=None,
               reference: str = "", split_by_label: bool = True) -> Tuple[List[Dict], List[str]]:
    """Convert a POP file.  Returns (datasets, skipped-notes)."""
    resolve = phase_resolver(tdb_phases, phase_map)
    eqs = parse_pop(path)
    notes: List[str] = []
    groups: Dict[Tuple, List[Tuple[float, List]]] = {}
    for n, eq in enumerate(eqs, 1):
        phases = [resolve(p) for p in eq.fixed]
        x_exp = {}
        for k, v in eq.experiments.items():
            m = re.match(r"^(?:X|W)\(([^,]+),([^)]+)\)$", k)
            if m:
                if k.startswith("W"):
                    notes.append(f"eq {n}: weight-fraction experiment {k} skipped (convert to X first)")
                    continue
                x_exp.setdefault(resolve(m.group(1)), {})[m.group(2).upper()] = v
        for ph in x_exp:
            if ph not in phases:
                phases.append(ph)
        T = eq.conditions.get("T", eq.experiments.get("T"))
        if T is None or len(phases) < 2:
            others = ", ".join(eq.experiments) or "none"
            notes.append(f"eq {n} ({eq.label or 'no label'}): not a phase-boundary point "
                         f"(phases={phases}, experiments: {others}) - skipped")
            continue
        comps = sorted({c.upper() for c in (components or [])}
                       | {c for d in x_exp.values() for c in d}
                       | {re.match(r"X\((\w+)\)", k).group(1) for k in eq.conditions if re.match(r"X\((\w+)\)$", k)})
        overall = {re.match(r"X\((\w+)\)", k).group(1): v for k, v in eq.conditions.items() if re.match(r"X\((\w+)\)$", k)}
        # phase fixed at amount 1 with an overall composition condition -> it has that composition
        for ph, amt in eq.fixed.items():
            rp = resolve(ph)
            if abs(amt - 1.0) < 1e-9 and overall and rp not in x_exp:
                x_exp[rp] = dict(overall)
        key = (eq.label if split_by_label else "", tuple(comps))
        groups.setdefault(key, []).append((T, phases, x_exp, eq.conditions.get("P", 101325.0)))
    datasets = []
    for (label, comps), pts in groups.items():
        if len(comps) < 1:
            continue
        # independent components = those measured (X(phase,EL)); pad/truncate to n-1
        indep_pool = {c for _, _, xe, _ in pts for d in xe.values() for c in d}
        system = sorted({c.upper() for c in (components or [])} | indep_pool)
        system = [c for c in system if c != "VA"]
        if len(system) < 2:
            notes.append(f"label {label!r}: cannot infer all components (pass components=[...])")
            continue
        indep = [c for c in system if c in indep_pool][: len(system) - 1]
        indep += [c for c in system if c not in indep][: len(system) - 1 - len(indep)]
        vals, Ts = [], []
        for T, phases, x_exp, P in pts:
            row = []
            for ph in phases:
                xs = [x_exp.get(ph, {}).get(c) for c in indep]
                row.append([ph, list(indep), xs])
            vals.append(row)
            Ts.append(T)
        try:
            ds = make_zpf_dataset(system, Ts, vals, reference=reference or Path(path).name,
                                  comment=f"from {Path(path).name}" + (f", label {label}" if label else ""))
            datasets.append(ds)
        except ZPFError as e:
            notes.append(f"label {label!r}: {e}")
    return datasets, notes


# ---------------------------------------------------------------------------
# DATAPLOT .exp files
# ---------------------------------------------------------------------------
def parse_exp(path: str) -> Dict:
    """Parse a DATAPLOT file into axes info + datasets with points and comments."""
    text = Path(path).read_text(errors="replace").replace("\r", "")
    info = {"file": str(path), "prologue": {}, "datasets": []}
    cur = None
    pending_comments: List[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("$"):
            (cur["comments"] if cur else pending_comments).append(s.lstrip("$ ").strip())
            continue
        if "$" in s:
            s, c = s.split("$", 1)
            (cur["comments"] if cur else pending_comments).append(c.strip())
            s = s.strip()
        up = s.upper()
        w = s.split()
        if up.startswith("PROLOG"):
            cur = None
            continue
        if up.startswith("DATASET") or up.startswith("DATA_SET"):
            cur = {"id": w[1] if len(w) > 1 else str(len(info["datasets"]) + 1),
                   "title": " ".join(w[2:]), "comments": pending_comments, "points": [], "strings": []}
            pending_comments = []
            info["datasets"].append(cur)
            continue
        for key in ("XTEXT", "YTEXT", "XSCALE", "YSCALE", "TITLE", "XTYPE", "YTYPE"):
            if up.startswith(key):
                info["prologue"][key] = s[len(key):].strip()
                break
        else:
            if cur is None:
                continue
            nums = re.match(r"^\s*([-+]?[\d.]+(?:[Ee][-+]?\d+)?)[\s,]+([-+]?[\d.]+(?:[Ee][-+]?\d+)?)(.*)$", s)
            if nums:
                cur["points"].append([float(nums.group(1)), float(nums.group(2)), nums.group(3).strip()])
            elif up.startswith(("STRING", "\"", "'")):
                cur["strings"].append(s)
    info["axes_guess"] = guess_axes(info["prologue"])
    return info


def guess_axes(prologue: Dict) -> Dict:
    xt, yt = prologue.get("XTEXT", "").upper(), prologue.get("YTEXT", "").upper()
    g = {"x_unit": "mole_fraction", "x_component": None, "y_quantity": "T", "y_unit": "K"}
    if re.search(r"W\(|WEIGHT|MASS|WT", xt):
        g["x_unit"] = "wt%" if ("%" in xt or "PERCENT" in xt) else "wt_fraction"
    elif re.search(r"AT\.?%|ATOMIC|MOL.?%|PERCENT", xt):
        g["x_unit"] = "at%"
    m = re.search(r"[XW]\(\s*(?:[A-Z_]+,)?\s*([A-Z]{1,2})\s*\)", xt) or re.search(r"\b([A-Z][A-Z]?)\s*$", xt)
    if m:
        g["x_component"] = m.group(1)
    if re.search(r"CELSIUS|°C|DEG.?C|\bC\b|T_C", yt):
        g["y_unit"] = "C"
    if "T" not in yt and yt:
        g["y_quantity"] = yt
    return g


def exp_to_zpf(parsed: Dict, components: Sequence[str], dataset_map: Dict[str, Dict],
               reference: str = "", x_unit: Optional[str] = None, y_unit: Optional[str] = None) -> List[Dict]:
    """Convert DATAPLOT datasets to ZPF datasets (binary systems).

    ``dataset_map[id] = {"phases": ["LIQUID", "BCC_A2"],        # phases in equilibrium
                         "measured_phase": "LIQUID",             # whose composition x is
                         "x_component": "CR"}                    # optional
    Datasets missing from the map are skipped.
    """
    comps = [c.upper() for c in components if c.upper() != "VA"]
    if len(comps) != 2:
        raise ZPFError("DATAPLOT conversion supports binary systems (x vs T); got " + str(comps))
    ax = parsed.get("axes_guess", {})
    xu = x_unit or ax.get("x_unit", "mole_fraction")
    yu = y_unit or ax.get("y_unit", "K")
    out = []
    for dsn in parsed["datasets"]:
        m = dataset_map.get(str(dsn["id"]))
        if not m:
            continue
        xc = (m.get("x_component") or ax.get("x_component") or comps[1]).upper()
        other = [c for c in comps if c != xc][0]
        phases = [p.upper() for p in m["phases"]]
        meas = (m.get("measured_phase") or phases[0]).upper()
        Ts, vals = [], []
        for x, y, _code in dsn["points"]:
            xf = to_mole_fraction(x, xu, xc, other)
            T = y + 273.15 if yu.upper().startswith("C") else y
            row = []
            for ph in phases:
                row.append([ph, [xc], [round(xf, 6) if ph == meas else None]])
            vals.append(row)
            Ts.append(T)
        if not Ts:
            continue
        out.append(make_zpf_dataset(comps, Ts, vals, reference=reference or m.get("reference", dsn["title"]),
                                    comment=f"DATAPLOT dataset {dsn['id']} '{dsn['title']}' from "
                                            f"{Path(parsed['file']).name}"))
    return out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def csv_to_zpf(path: str, components: Sequence[str], reference: str = "", group_by_reference: bool = True,
               composition_unit: str = "mole_fraction") -> List[Dict]:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ZPFError("empty CSV")
    cols = list(rows[0].keys())
    tcol = next((c for c in cols if re.fullmatch(r"\s*T(_K|_C)?\s*", c, re.I)), None)
    if tcol is None:
        raise ZPFError("CSV needs a T, T_K or T_C column")
    celsius = tcol.strip().upper() == "T_C"
    phase_cols = sorted([c for c in cols if re.fullmatch(r"\s*phase_\d+\s*", c, re.I)],
                        key=lambda c: int(re.findall(r"\d+", c)[0]))
    comps = [c.upper() for c in components if c.upper() != "VA"]
    groups: Dict[str, Tuple[List, List]] = {}
    for r in rows:
        if not (r.get(tcol) or "").strip():
            continue
        T = float(r[tcol]) + (273.15 if celsius else 0)
        row = []
        for pc in phase_cols:
            idx = re.findall(r"\d+", pc)[0]
            ph = (r.get(pc) or "").strip()
            if not ph:
                continue
            xcols = [c for c in cols if re.fullmatch(rf"\s*X\((\w+)\)_{idx}\s*", c, re.I)]
            names = [re.match(r"\s*X\((\w+)\)", c, re.I).group(1).upper() for c in xcols]
            xs = []
            for c, nm in zip(xcols, names):
                v = (r.get(c) or "").strip()
                if not v:
                    xs.append(None)
                    continue
                other = [k for k in comps if k != nm]
                xs.append(to_mole_fraction(float(v), composition_unit, nm, other[0]) if other else float(v))
            row.append([ph, names, xs])
        ref = (r.get("reference") or reference or Path(path).name).strip()
        g = groups.setdefault(ref if group_by_reference else "all", ([], []))
        g[0].append(T)
        g[1].append(row)
    return [make_zpf_dataset(comps, Ts, vals, reference=ref, comment=f"from {Path(path).name}")
            for ref, (Ts, vals) in groups.items()]


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------
def pdf_text(path: str, pages: Optional[str] = None, max_chars: int = 60000) -> str:
    """Extract text from a PDF.  ``pages`` like "1-3,7" (1-based)."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    n = len(reader.pages)
    idx = _page_list(pages, n)
    chunks = []
    for i in idx:
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception as e:  # noqa: BLE001
            t = f"[could not extract: {e}]"
        chunks.append(f"--- page {i + 1}/{n} ---\n{t}")
    s = "\n".join(chunks)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n... [truncated at {max_chars} chars; request fewer pages]"
    return s


def render_pdf_page(path: str, page: int, out_png: str, dpi: int = 110) -> str:
    """Render one PDF page (1-based) to PNG so a vision-capable LLM can read tables/figures."""
    import shutil
    import subprocess
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        pix.save(str(out))
        return str(out)
    except ImportError:
        pass
    exe = shutil.which("pdftoppm")
    if not exe:
        raise RuntimeError("install pymupdf (pip install pymupdf) or poppler (pdftoppm) to render PDF pages")
    stem = out.with_suffix("")
    subprocess.run([exe, "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                    path, str(stem)], check=True)
    return str(stem) + ".png"


def _page_list(pages: Optional[str], n: int) -> List[int]:
    if not pages:
        return list(range(n))
    out = []
    for part in str(pages).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a) - 1, min(int(b), n)))
        elif part:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]
