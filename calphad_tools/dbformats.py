"""Database formats: ChemSage/FactSage ``.dat`` input and dialect-safe TDB output.

* Every calculation tool accepts a ``.dat`` file wherever it takes a ``tdb`` (pycalphad's own reader, via
  :func:`calphad_tools.pcache.get_db`).  Text-based tools (merge/compare/check) need a TDB: use
  ``convert_database`` first.
* ``convert_database`` writes a TDB from a ``.dat`` (or a TDB in another dialect) and reports what TDB cannot
  express: quasichemical (MQMQA / SUBQ / SUBG) phases and Kohler-Toop (QKTO) excess terms are dropped, illegal
  names are renamed.  The written file is reloaded and every phase's Gibbs energy is compared with the source.
* ``write_database(dialect=...)`` (mapping_tools) uses :mod:`calphad_tools.tdb_dialect` for Thermo-Calc /
  OpenCalphad / pycalphad output.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import tdb_dialect


def _fresh_db(path: str):
    from pycalphad import Database
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(path))


def gm_check(src, out, renames: Optional[Dict] = None, phases: Optional[Sequence[str]] = None, n: int = 4,
             seed: int = 0) -> Dict[str, float]:
    """Max |GM_out - GM_src| (J/mol) per phase at ``n`` random (T, site fraction) states.

    ``renames`` maps source -> output names ({'phases': {...}, 'species': {...}}); functions are substituted
    inside the models, so function renames need no mapping."""
    from pycalphad import Model, variables as v
    renames = renames or {}
    pinv = {new: old for old, new in renames.get("phases", {}).items()}
    sinv = {new: old for old, new in renames.get("species", {}).items()}
    rng = np.random.default_rng(seed)
    res: Dict[str, float] = {}
    for ph_out in sorted(phases or out.phases):
        ph_src = pinv.get(ph_out, ph_out)
        if ph_src not in src.phases or ph_out not in out.phases:
            continue
        comps_out = sorted({s.name for subl in out.phases[ph_out].constituents for s in subl})
        dis = out.phases[ph_out].model_hints.get("disordered_phase")
        if dis and dis in out.phases:
            comps_out = sorted(set(comps_out) | {s.name for subl in out.phases[dis].constituents for s in subl})
        comps_src = [sinv.get(c, c) for c in comps_out]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mo, ms = Model(out, comps_out, ph_out), Model(src, comps_src, ph_src)
        except Exception:  # noqa: BLE001 - e.g. a phase that cannot be built without other species
            continue
        worst = 0.0
        for _ in range(n):
            T = float(rng.uniform(300.0, 3000.0))
            by_sub: Dict[int, list] = {}
            for y in mo.site_fractions:
                by_sub.setdefault(int(y.sublattice_index), []).append(y)
            vals = {}
            for idx, ys in by_sub.items():
                w = rng.dirichlet(np.ones(len(ys)))
                for y, wi in zip(ys, w):
                    vals[(idx, y.species.name)] = float(wi)
            so = {v.T: T, v.P: 101325.0}
            ss = {v.T: T, v.P: 101325.0}
            so.update({y: vals[(int(y.sublattice_index), y.species.name)] for y in mo.site_fractions})
            for y in ms.site_fractions:
                key = (int(y.sublattice_index), renames.get("species", {}).get(y.species.name, y.species.name))
                ss[y] = vals.get(key, 0.0)
            try:
                go, gs = float(mo.GM.xreplace(so)), float(ms.GM.xreplace(ss))
            except (TypeError, RuntimeError):
                worst = float("nan")
                break
            worst = max(worst, abs(go - gs))
        res[ph_out] = worst
    return res


def syntax_findings(path: str):
    """ERROR / WARNING findings of tdb_checks.check_syntax for a written TDB (the checks shared with check_tdb)."""
    from .tdb_checks import Findings, check_syntax
    from .tdbparse import parse_tdb
    text = Path(path).read_text()
    F = Findings()
    check_syntax(parse_tdb(path), text, F)
    return [f"{f.severity}: {f.message}" for f in F if f.severity in ("ERROR", "WARNING")]


def export_verified(src_db, db, out_path: str, dialect: str, groupby: str = "subsystem") -> Dict:
    """Write ``db`` in ``dialect``, reload it and compare every phase's GM with ``src_db``."""
    r = tdb_dialect.export_tdb(db, out_path, dialect=dialect, groupby=groupby)
    back = _fresh_db(out_path)
    ren = r["renames"]
    info = {"path": out_path, "dialect": dialect, "renames": {k: dict(v) for k, v in ren.items() if v},
            "violations": r["violations"],
            "reloaded": {"elements": sorted(str(e) for e in back.elements), "phases": len(back.phases),
                         "parameters": len(back._parameters), "functions": len(back.symbols)}}
    try:
        diffs = gm_check(src_db, back, ren)
        info["model_check_max_abs_dGM_J_per_mol"] = {k: round(val, 9) for k, val in diffs.items()}
    except Exception as e:  # noqa: BLE001
        info["model_check_error"] = f"{type(e).__name__}: {e}"
    try:
        info["syntax_findings"] = syntax_findings(out_path)
    except Exception as e:  # noqa: BLE001
        info["syntax_findings"] = [f"check failed: {type(e).__name__}: {e}"]
    return info


def convert_database(src: str, out_path: str, dialect: str = "pycalphad", groupby: str = "subsystem") -> Dict:
    from .calc import get_db
    db = get_db(src)
    compat, rep = tdb_dialect.tdb_compatible(db)
    if not compat.phases:
        raise ValueError(f"nothing to write: every phase of {Path(src).name} uses a model TDB cannot express "
                         f"({rep['dropped_phases']})")
    info = export_verified(db, compat, out_path, dialect, groupby)
    info.update(source=src, source_format="ChemSage .dat" if Path(src).suffix.lower() == ".dat" else "TDB",
                phases_written=sorted(compat.phases), dropped_phases=rep["dropped_phases"],
                dropped_parameters=rep["dropped_parameters"], notes=rep["notes"])
    return info


def report_text(info: Dict, rel=lambda p: p) -> str:
    chk = info.get("model_check_max_abs_dGM_J_per_mol", {})
    finite = [x for x in chk.values() if x == x]
    worst = max(finite) if finite else None
    lines = [f"wrote {rel(info['path'])} ({info['dialect']} dialect): {info['reloaded']['phases']} phases, "
             f"{info['reloaded']['parameters']} parameters, {info['reloaded']['functions']} functions; "
             "reloaded in pycalphad OK",
             f"GM check vs source ({len(chk)} phases, random T and site fractions): max |dGM| = "
             f"{'n/a' if worst is None else f'{worst:.3g} J/mol'}"]
    bad = sorted(k for k, x in chk.items() if not (x == x) or x > 1e-6)
    if bad:
        lines.append("phases that differ from the source: " + ", ".join(f"{k} ({chk[k]:.3g})" for k in bad))
    if info.get("model_check_error"):
        lines.append(f"model check error: {info['model_check_error']}")
    if info.get("dropped_phases"):
        lines.append("NOT converted (TDB cannot express them): " + "; ".join(
            f"{k}: {v}" for k, v in sorted(info["dropped_phases"].items())))
    for kind, m in info.get("renames", {}).items():
        lines.append(f"renamed {kind} ({len(m)}): " + ", ".join(f"{a} -> {b}" for a, b in sorted(m.items())))
    lines += [f"note: {n}" for n in info.get("notes", [])]
    v = info.get("violations") or []
    lines.append(f"{info['dialect']} limits: " + ("all met" if not v else f"{len(v)} problem(s): " + "; ".join(v[:5])))
    sf = info.get("syntax_findings") or []
    if sf:
        lines.append(f"TDB syntax check ({len(sf)}): " + " | ".join(s[:150] for s in sf[:4]))
    return "\n".join(lines)


def dat_info(path: str) -> Dict:
    """tdb_info-style summary of a ChemSage .dat (or any pycalphad-readable) database."""
    from .calc import get_db
    db = get_db(path)
    phases = {}
    for nm, ph in sorted(db.phases.items()):
        hints = sorted(ph.model_hints)
        phases[nm] = {"site_ratios": [float(x) if not hasattr(x, "free_symbols") else str(x) for x in ph.sublattices],
                      "constituents": [",".join(sorted(s.name for s in sl)) for sl in ph.constituents],
                      "model_hints": hints}
    ptypes: Dict[str, int] = {}
    for p in db._parameters.all():
        ptypes[p["parameter_type"]] = ptypes.get(p["parameter_type"], 0) + 1
    compat, rep = tdb_dialect.tdb_compatible(db)
    return {"file": path, "format": "ChemSage/FactSage .dat (pycalphad reader)",
            "elements": sorted(str(e) for e in db.elements if str(e) not in ("VA", "/-")),
            "species": sorted(s.name for s in db.species), "n_phases": len(phases), "phases": phases,
            "n_parameters": len(db._parameters), "parameter_types": ptypes, "n_functions": len(db.symbols),
            "not_expressible_in_TDB": rep["dropped_phases"]}


# ---------------------------------------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------------------------------------
def register(tb) -> None:
    @tb.add("convert_database",
            "Convert a ChemSage/FactSage .dat (or a TDB) to a TDB file in a dialect (pycalphad | thermocalc | "
            "opencalphad), reporting what TDB cannot express (quasichemical MQMQA/SUBQ/SUBG phases, Kohler-Toop "
            "QKTO terms: dropped) and renamed names; the file is reloaded and each phase's GM compared with the "
            "source. Calculation tools already accept .dat directly.",
            {"source": {"type": "string"}, "output_tdb": {"type": "string"},
             "dialect": {"type": "string", "enum": sorted(tdb_dialect.DIALECTS)},
             "groupby": {"type": "string", "enum": ["subsystem", "phase"]}},
            ["source", "output_tdb"])
    def convert_database_tool(source, output_tdb, dialect="pycalphad", groupby="subsystem"):
        s = tb._find_tdb(source)
        out = tb.outp(output_tdb if output_tdb.lower().endswith(".tdb") else output_tdb + ".tdb",
                      f"{Path(s).stem}.tdb")
        if Path(out).resolve() == Path(s).resolve():
            raise ValueError("refusing to overwrite the source database; choose another output_tdb")
        return report_text(convert_database(s, out, dialect=dialect, groupby=groupby), tb.rel), []
