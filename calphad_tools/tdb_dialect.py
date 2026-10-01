"""Write a pycalphad Database as a TDB file in the dialect of a target program.

Self-contained on purpose (standard library + pycalphad only): ``show_code`` copies this module verbatim into
standalone scripts.  Entry points: ``tdb_compatible`` (drop what TDB cannot express), ``export_tdb`` (rename
and write) and ``check_tdb_text`` (verify a file against a dialect's limits).

Dialect limits
--------------
thermocalc   FUNCTION names <= 8 chars, phase and species names <= 24 chars, lines <= 78 chars, integer
             exponents only, LN for natural logarithms, finite temperature limits, letters as type codes and
             the abbreviated ``GES A_P_D <phase> MAGNETIC|DIS_PART`` TYPE_DEFINITION forms.
opencalphad  as thermocalc but FUNCTION names <= 16 chars.
pycalphad    pycalphad's own writer; only names pycalphad cannot parse (e.g. ``(`` or ``.`` in phase names)
             are changed.
Every command ends with ``!``; FUNCTION/PARAMETER ranges end with ``Y`` (another range follows) or ``N``
(last range); names and expressions are upper case; long commands are continued on lines starting with
two spaces, split between terms.
"""
import copy
import io
import re

DIALECTS = {
    "thermocalc": {"function": 8, "phase": 24, "species": 24, "line": 78, "strict": True},
    "opencalphad": {"function": 16, "phase": 24, "species": 24, "line": 78, "strict": True},
    "pycalphad": {"function": None, "phase": None, "species": None, "line": 78, "strict": False},
}
RESERVED = {"T", "P", "R", "LN", "LOG", "EXP", "ERF", "ABS", "SQRT", "GEIN", "Y", "N"}
TDB_HINTS = {"ordered_phase", "disordered_phase", "ihj_magnetic_afm_factor", "ihj_magnetic_structure_factor",
             "ionic_liquid_2SL", "symmetry_FCC_4SL", "symmetry_BCC_4SL", "liquid", "gas", "aqueous",
             "charged_phase"}
TDB_PARAMETER_KEYS = {"phase_name", "parameter_type", "constituent_array", "parameter_order",
                      "diffusing_species", "parameter", "reference", "_generated_by_symmetry_option"}
_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_SPECIES_STRICT = re.compile(r"[A-Z][A-Z0-9_+\-]*")
_SPECIES_PYCALPHAD = re.compile(r"[A-Z0-9+\-*/_.]+")


# ---------------------------------------------------------------------------------------------------------
# 1. keep only what a TDB can express
# ---------------------------------------------------------------------------------------------------------
def tdb_compatible(db):
    """Copy of ``db`` without the model features a TDB file cannot express, and a report of what was dropped.

    Dropped: phases with non-TDB model hints (e.g. the quasichemical MQMQA/SUBQ/SUBG liquid from ChemSage
    .dat files) and phases with Kohler-Toop (QKTO) excess parameters, plus all parameters of dropped phases.
    A ``chemical_groups`` hint (only used by QKTO terms) is removed from phases that keep no QKTO term."""
    new = copy.deepcopy(db)
    dropped, notes = {}, []
    params = new._parameters.all()
    for name, ph in sorted(new.phases.items()):
        bad = sorted(set(ph.model_hints) - TDB_HINTS - {"chemical_groups"})
        if bad:
            kind = "quasichemical (MQMQA/SUBQ/SUBG) model" if "mqmqa" in bad else f"model hints {bad}"
            dropped[name] = f"{kind} cannot be written to a TDB"
            continue
        keys = {k for p in params if p["phase_name"] == name for k in p} - TDB_PARAMETER_KEYS
        types = sorted({p["parameter_type"] for p in params if p["phase_name"] == name and
                        (p["parameter_type"] == "QKT" or set(p) - TDB_PARAMETER_KEYS)})
        if types or keys:
            dropped[name] = (f"parameters {types or sorted(keys)} (Kohler-Toop / QKTO polynomial excess) "
                             "cannot be written to a TDB")
    for name, ph in list(new.phases.items()):
        o, d = ph.model_hints.get("ordered_phase"), ph.model_hints.get("disordered_phase")
        if name not in dropped and o == name and d in dropped:
            dropped[name] = f"its disordered part {d} was dropped"
    for name in dropped:
        del new.phases[name]
        getattr(new, "_structure_dict", {}).pop(name, None)
    for ph in new.phases.values():
        if "chemical_groups" in ph.model_hints:
            del ph.model_hints["chemical_groups"]
            notes.append(f"{ph.name}: chemical_groups hint removed (only used by Kohler-Toop terms, none left)")
        if ph.model_hints.get("disordered_phase") in dropped and ph.model_hints.get("ordered_phase") != ph.name:
            ph.model_hints.pop("ordered_phase", None)
            ph.model_hints.pop("disordered_phase", None)
    gone = [p.doc_id for p in params if p["phase_name"] in dropped]
    declared = {s.name for s in new.species} | {"*"}
    orphans = [p for p in params if p["phase_name"] not in dropped and
               any(s.name not in declared for subl in p["constituent_array"] for s in subl)]
    if orphans:        # e.g. .dat magnetic terms whose "constituent" is a phase name: no model ever uses them
        gone += [p.doc_id for p in orphans]
        notes.append(f"{len(orphans)} parameter(s) with undeclared constituents dropped (unused by any model): "
                     + ", ".join(sorted({f"{p['parameter_type']}({p['phase_name']})" for p in orphans})[:6]))
    if gone:
        new._parameters.remove(doc_ids=gone)
    return new, {"dropped_phases": dropped, "dropped_parameters": len(gone), "notes": notes}


# ---------------------------------------------------------------------------------------------------------
# 2. names
# ---------------------------------------------------------------------------------------------------------
def _short(name, limit, taken, prefix="F", species=False):
    """Deterministic legal name: invalid characters -> '_', truncated to ``limit`` with a numeric suffix."""
    base = name.upper()
    if species:
        base = re.sub(r"\[(\d*)([+-])\]", lambda m: m.group(2) + (m.group(1) or "1"), base)
        base = re.sub(r"(\d)\.0+\b", r"\1", base)
        base = re.sub(r"[^A-Z0-9_+\-]", "_", base)
    else:
        base = re.sub(r"[^A-Z0-9_]", "_", base)
    base = re.sub(r"_+", "_", base).strip("_") or prefix
    if not base[0].isalpha():
        base = prefix + base
    if (limit is None or len(base) <= limit) and base not in taken and base not in RESERVED:
        return base
    width = limit or (len(base) + 4)
    for k in range(1, 100000):
        suf = str(k) if not species else f"_{k}"
        cand = base[:max(1, width - len(suf))] + suf
        if cand not in taken and cand not in RESERVED:
            return cand
    raise ValueError(f"cannot find a short name for {name}")


def plan_renames(db, dialect):
    """{'functions': {old: new}, 'phases': {...}, 'species': {...}} needed for ``dialect``."""
    lim = DIALECTS[dialect]
    strict = lim["strict"]
    out = {"functions": {}, "phases": {}, "species": {}}
    taken = set(db.symbols)
    for n in sorted(db.symbols):
        if (lim["function"] and len(n) > lim["function"]) or not _NAME.fullmatch(n.upper()):
            taken.discard(n)
            out["functions"][n] = new = _short(n, lim["function"], taken | set(out["functions"].values()))
            taken.add(new)
    taken = set(db.phases)
    for n in sorted(db.phases):
        if (lim["phase"] and len(n) > lim["phase"]) or not _NAME.fullmatch(n):
            taken.discard(n)
            out["phases"][n] = new = _short(n, lim["phase"], taken, prefix="P")
            taken.add(new)
    elements = {str(e) for e in db.elements}
    names = {s.name for s in db.species}
    taken = set(names)
    pat = _SPECIES_STRICT if strict else _SPECIES_PYCALPHAD
    for n in sorted(names):
        if n in elements or n in ("VA", "/-"):
            continue
        if (lim["species"] and len(n) > lim["species"]) or not pat.fullmatch(n):
            taken.discard(n)
            out["species"][n] = new = _short(n, lim["species"], taken, prefix="S", species=True)
            taken.add(new)
    return out


def apply_renames(db, ren):
    """Copy of ``db`` with functions, phases and species renamed consistently."""
    from pycalphad.io.tdb import _apply_new_symbol_names
    from pycalphad.variables import Species
    db = copy.deepcopy(db)
    if ren["functions"]:
        _apply_new_symbol_names(db, ren["functions"])
    pm = ren["phases"]
    if pm:
        new = {}
        for name, ph in db.phases.items():
            ph.name = pm.get(name, name)
            for k in ("ordered_phase", "disordered_phase"):
                if ph.model_hints.get(k) in pm:
                    ph.model_hints[k] = pm[ph.model_hints[k]]
            new[ph.name] = ph
        db.phases = new
        sd = getattr(db, "_structure_dict", None)
        if sd:
            db._structure_dict = {pm.get(k, k): v for k, v in sd.items()}
        for rs in db.refstates.values():
            if rs.get("phase") in pm:
                rs["phase"] = pm[rs["phase"]]
        for p in db._parameters.all():
            if p["phase_name"] in pm:
                db._parameters.update({"phase_name": pm[p["phase_name"]]}, doc_ids=[p.doc_id])
    sm = ren["species"]
    if sm:
        conv = {s: Species(sm[s.name], constituents=dict(s.constituents), charge=s.charge)
                for s in db.species if s.name in sm}

        def c(s):
            return conv.get(s, s)
        db.species = {c(s) for s in db.species}
        for ph in db.phases.values():
            ph.constituents = tuple(frozenset(c(s) for s in subl) for subl in ph.constituents)
        for p in db._parameters.all():
            upd = {"constituent_array": tuple(tuple(c(s) for s in subl) for subl in p["constituent_array"])}
            if p.get("diffusing_species") in conv:
                upd["diffusing_species"] = conv[p["diffusing_species"]]
            db._parameters.update(upd, doc_ids=[p.doc_id])
    return db


# ---------------------------------------------------------------------------------------------------------
# 3. text post-processing
# ---------------------------------------------------------------------------------------------------------
def _logical_lines(text):
    """Join the writer's continuation lines (two leading spaces) back into one line per command."""
    out, cur = [], None
    for ln in text.splitlines():
        if cur is None:
            if ln.strip() and not ln.lstrip().startswith("$") and "!" not in ln:
                cur = ln
            else:
                out.append(ln)
            continue
        cur += ln[2:] if ln.startswith("  ") else ln
        if "!" in ln:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def _fix_limits(cmd):
    """Infinite temperature limits (written as ',') -> 1 K / 10000 K, as the pycalphad writer does elsewhere."""
    m = re.match(r"(\s*(?:FUNCTION\s+\S+|PARAMETER\s+\S+\([^)]*\)))\s+,\s", cmd)
    if m:
        cmd = m.group(1) + " 1 " + cmd[m.end():]
    return re.sub(r";\s*,\s*([YN])\b", r"; 10000 \1", cmd)


def _int_formula(ln):
    """SPECIES <name> <formula> !: integral stoichiometries/charges in the formula (not the name) as integers."""
    m = re.match(r"(\s*SPECIES\s+\S+\s+)(.*)$", ln, re.I)
    return m.group(1) + re.sub(r"(\d)\.0+(?![0-9])", r"\1", m.group(2)) if m else ln


def transform(text, dialect):
    """Apply a dialect's syntax rules to TDB text written by pycalphad (before line wrapping)."""
    lim = DIALECTS[dialect]
    # the writer prints EXP(x) as 2.71828182845905**(x) (15 digits of e): restore EXP
    lines = [re.sub(r"(?<![0-9.])2\.718281828459\d*\*\*\(", "EXP(", ln) if not ln.lstrip().startswith("$") else ln
             for ln in _logical_lines(text)]
    # SPECIES formulas: the writer prints stoichiometries/charges as floats (FE2.0/+2.0), which TDB readers
    # (pycalphad's included) reject; integral values are written as integers
    lines = [_int_formula(ln) if ln.strip().upper().startswith("SPECIES ") else ln for ln in lines]
    if not lim["strict"]:
        return "\n".join(lines) + "\n"
    codes, dis_owner, new_codes, out = {}, {}, iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), []
    for ln in lines:
        s = ln.strip().upper()
        m = re.match(r"TYPE_DEFINITION\s+(\S)\s+(.*)$", s)
        if m and m.group(1) != "%":
            codes[m.group(1)] = next(new_codes)
            d = re.search(r"A(?:MEND_PHASE_DESCRIPTION|_P_D)\s+(\S+)\s+DIS(?:ORDERED)?_PART\b", m.group(2))
            if d:
                dis_owner[codes[m.group(1)]] = d.group(1)   # DIS_PART code belongs to the ordered phase only
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("$"):
            out.append(ln)
            continue
        u = s.upper()
        if u.startswith("TYPE_DEFINITION"):
            m = re.match(r"TYPE_DEFINITION\s+(\S)\s+GES\s+(?:AMEND_PHASE_DESCRIPTION|A_P_D)\s+(\S+)\s+"
                         r"(MAGNETIC|DISORDERED_PART|DIS_PART)\s+(.*?)\s*!$", u)
            if m:
                ch, ph, kind, rest = m.groups()
                if kind == "MAGNETIC":
                    afm, p = rest.split()[:2]
                    rest = f"{float(afm):.1f} {float(p):.5E}"
                else:
                    kind = "DIS_PART"
                u = f"TYPE_DEFINITION {codes.get(ch, ch)} GES A_P_D {ph} {kind} {rest} !"
        elif u.startswith("PHASE "):
            m = re.match(r"PHASE\s+(\S+)\s+(\S+)\s+(.*)$", u)
            if m:
                name = m.group(1).split(":")[0]
                cc = [codes.get(c, c) for c in m.group(2)]
                cc = [c for c in cc if dis_owner.get(c, name) == name]
                u = f"PHASE {m.group(1)} {''.join(cc)} {m.group(3)}"
        elif u.startswith(("FUNCTION ", "PARAMETER ")):
            u = _fix_limits(re.sub(r"\bLOG\(", "LN(", u))
        elif u.startswith("ELEMENT "):
            f = u.split()
            if len(f) >= 3 and f[2] == "BLANK" and f[1] in ("VA", "/-"):
                f[2] = "VACUUM" if f[1] == "VA" else "ELECTRON_GAS"
                u = " ".join(f)
        out.append(" " + u if ln.startswith(" ") else u)
    return "\n".join(_disordered_first(out, dis_owner)) + "\n"


def _disordered_first(lines, dis_owner):
    """Move each disordered phase's PHASE/CONSTITUENT commands before its ordered phase (TC/OC read DIS_PART
    when the ordered phase is entered, so the disordered phase must already exist)."""
    ordered_of = {}
    for ln in lines:
        m = re.match(r"\s*TYPE_DEFINITION\s+\S+\s+GES\s+A_P_D\s+(\S+)\s+DIS_PART\s+(\S+)", ln)
        if m:
            ordered_of[m.group(2)] = m.group(1)
    if not ordered_of:
        return lines
    idx = [i for i, ln in enumerate(lines) if re.match(r"\s*(PHASE|CONSTITUENT)\s", ln)]
    lo, hi = idx[0], idx[-1]
    blocks, cur = [], []
    for ln in lines[lo:hi + 1]:
        if re.match(r"\s*PHASE\s", ln) and cur:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    blocks.append(cur)
    name = {id(b): b[0].split()[1].split(":")[0] for b in blocks}
    order = [name[id(b)] for b in blocks]
    for dis, ordered in ordered_of.items():
        if dis in order and ordered in order and order.index(dis) > order.index(ordered):
            order.remove(dis)
            order.insert(order.index(ordered), dis)
    by = {name[id(b)]: b for b in blocks}
    new = [ln for n in order for ln in by[n]]
    if new and new[-1].strip():
        new.append("")
    return lines[:lo] + new + lines[hi + 1:]


def wrap(text, width):
    """Continue long commands on lines starting with two spaces, breaking between terms."""
    from pycalphad.io.tdb import reflow_text
    return reflow_text(text, linewidth=width)


# ---------------------------------------------------------------------------------------------------------
# 4. export + check
# ---------------------------------------------------------------------------------------------------------
def export_tdb(db, path, dialect="thermocalc", groupby="subsystem"):
    """Write ``db`` to ``path`` for ``dialect``; returns {'renames': ..., 'violations': [...]}.

    ``db`` is not modified.  Raises ValueError for content the dialect cannot express (e.g. non-integer
    exponents for Thermo-Calc / OpenCalphad)."""
    from pycalphad.io.tdb import write_tdb
    if dialect not in DIALECTS:
        raise ValueError(f"dialect must be one of {sorted(DIALECTS)}")
    ren = plan_renames(db, dialect)
    work = apply_renames(db, ren) if any(ren.values()) else db
    buf = io.StringIO()
    write_tdb(work, buf, groupby=groupby, if_incompatible="ignore")
    text = wrap(transform(buf.getvalue(), dialect), DIALECTS[dialect]["line"])
    problems = check_tdb_text(text, dialect)
    fatal = [p for p in problems if p.startswith("non-integer exponent")]
    if fatal:
        raise ValueError(f"{dialect} cannot express non-integer exponents (e.g. T**(0.5) in ChemSage heat-capacity "
                         f"terms); write dialect 'pycalphad' instead. Offending: " + "; ".join(fatal[:3]))
    with open(path, "w") as fh:
        fh.write(text)
    return {"renames": ren, "violations": problems}


def check_tdb_text(text, dialect):
    """List of violations of ``dialect``'s limits in TDB text (empty = compliant)."""
    lim = DIALECTS[dialect]
    bad = []
    for i, ln in enumerate(text.splitlines(), 1):
        if len(ln) > lim["line"]:
            bad.append(f"line {i} has {len(ln)} > {lim['line']} characters")
    codes, dis = {"%"}, {}
    cmds = [c for c in _logical_lines(text) if c.strip() and not c.lstrip().startswith("$")]
    for c in cmds:
        m = re.match(r"\s*TYPE_DEFINITION\s+(\S)", c, re.I)
        if m:
            codes.add(m.group(1))
            d = re.search(r"A(?:MEND_PHASE_DESCRIPTION|_P_D)\s+(\S+)\s+DIS(?:ORDERED)?_PART\s+(\S+)", c, re.I)
            if d:
                dis[m.group(1)] = (d.group(1).upper(), d.group(2).upper())
    seen_phases = []
    for c in cmds:
        s = c.strip()
        head = s.split()[0].upper()
        if not s.endswith("!"):
            bad.append(f"command not terminated with '!': {s[:60]}")
        if lim["strict"] and s != s.upper():
            bad.append(f"lower-case text: {s[:60]}")
        if head == "FUNCTION":
            name = s.split()[1]
            if lim["function"] and len(name) > lim["function"]:
                bad.append(f"FUNCTION name {name} longer than {lim['function']} characters")
        if head in ("FUNCTION", "PARAMETER"):
            body = s[:-1] if head == "FUNCTION" else s[s.find(")") + 1:-1]
            flags = re.findall(r";\s*[-+0-9.E,]+\s+([YN])\b", body, re.I)
            if not flags or flags[-1].upper() != "N" or any(f.upper() != "Y" for f in flags[:-1]) \
                    or len(flags) != body.count(";"):
                bad.append(f"temperature ranges must end 'Y' ... 'N': {s[:70]}")
            if lim["strict"] and re.search(r"\*\*\(\s*-?\d*\.\d*[1-9]\d*(E[-+]?\d+)?\s*\)", body):
                bad.append(f"non-integer exponent: {s[:70]}")
            if lim["strict"] and re.search(r"[;\s],\s", body[len(head) if head == "FUNCTION" else 0:]):
                bad.append(f"open temperature limit ',': {s[:70]}")
        if head == "PHASE":
            f = s.split()
            name = f[1].split(":")[0]
            if lim["phase"] and len(name) > lim["phase"]:
                bad.append(f"phase name {name} longer than {lim['phase']} characters")
            if not _NAME.fullmatch(name):
                bad.append(f"illegal phase name {name}")
            missing = [ch for ch in f[2] if ch not in codes]
            if missing:
                bad.append(f"PHASE {name} uses undefined type code(s) {missing}")
            for ch in f[2]:
                if lim["strict"] and ch in dis:
                    if dis[ch][0] != name:
                        bad.append(f"PHASE {name} carries the DIS_PART type code of {dis[ch][0]}")
                    elif dis[ch][1] not in seen_phases:
                        bad.append(f"disordered phase {dis[ch][1]} must be defined before {name}")
            seen_phases.append(name)
        if head == "ELEMENT" and len(s[:-1].split()) != 6:
            bad.append(f"ELEMENT line needs symbol, reference phase, mass, H298-H0, S298: {s[:70]}")
        if head == "SPECIES":
            name = s.split()[1]
            if "." in "".join(s.split()[2:]):
                bad.append(f"non-integer stoichiometry or charge in {s[:70]}")
            pat = _SPECIES_STRICT if lim["strict"] else _SPECIES_PYCALPHAD
            if (lim["species"] and len(name) > lim["species"]) or not pat.fullmatch(name):
                bad.append(f"illegal species name {name}")
        if head == "TYPE_DEFINITION" and lim["strict"]:
            if not re.fullmatch(r"TYPE_DEFINITION \S (SEQ \*|GES A_P_D \S+ (MAGNETIC \S+ \S+|DIS_PART \S+)) ?!",
                                re.sub(r"\s+", " ", s)):
                bad.append(f"TYPE_DEFINITION not in the portable form: {s[:70]}")
    return bad
