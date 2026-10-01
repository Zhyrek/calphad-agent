"""pycalphad calculations used by the chat agent (and usable directly).

All temperatures accept ``unit='K'`` or ``unit='C'``; results report both.
"""
from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .diagrams import KELVIN, phases_for, plot_binary, plot_ternary, to_celsius_axis  # noqa: E402,F401

from . import pcache  # noqa: E402


def get_db(tdb: str):
    """Load a pycalphad Database (TDB or ChemSage .dat), cached process-wide by file content (see pcache).

    The object is shared between tools: treat it as read-only."""
    return pcache.get_db(tdb)


def to_K(t: Union[float, Sequence[float]], unit: str = "K"):
    off = KELVIN if unit.upper().startswith("C") else 0.0
    if isinstance(t, (list, tuple, np.ndarray)):
        return [float(x) + off for x in t]
    return float(t) + off


def tdb_info(tdb: str) -> Dict:
    if Path(tdb).suffix.lower() == ".dat":          # ChemSage/FactSage: summarise pycalphad's reading of it
        from .dbformats import dat_info
        return dat_info(tdb)
    from .tdbparse import parse_tdb
    raw = parse_tdb(tdb)
    phases = {}
    for nm, ph in raw.phases.items():
        tds = [t.kind.lower() for t in raw.phase_typedefs(ph) if t.kind != "SEQ"]
        phases[nm] = {"site_ratios": ph.site_ratios,
                      "constituents": [":".join(sl) if False else ",".join(sl) for sl in ph.constituents],
                      "model_hints": tds}
    systems = {}
    for p in raw.parameters:
        els = sorted({s for sl in p.key.constituents for s in sl if s in raw.elements and s not in ("VA", "/-")})
        if len(els) >= 2:
            systems.setdefault("-".join(els), 0)
            systems["-".join(els)] += 1
    return {"file": tdb, "elements": raw.real_elements, "species": sorted(raw.species),
            "n_phases": len(phases), "phases": phases, "n_parameters": len(raw.parameters),
            "n_functions": len(raw.functions), "temperature_limits": raw.temp_limits,
            "assessed_subsystems (parameter count)": dict(sorted(systems.items())),
            "header_comments": [c for c in raw.header_comments if c.strip("$ -")][:25],
            "parse_warnings": raw.warnings[:20]}


def _conds(elements: Sequence[str], conditions: Dict) -> Dict:
    from pycalphad import variables as v
    out = {v.P: 101325.0, v.N: 1.0}
    for k, val in conditions.items():
        ku = k.upper().replace(" ", "")
        if isinstance(val, (list, tuple)) and len(val) == 0:
            continue
        if ku in ("T", "T_K"):
            out[v.T] = val
        elif ku == "P":
            out[v.P] = val
        elif ku == "N":
            out[v.N] = val
        else:
            m = re.fullmatch(r"(?:X\()?([A-Z]{1,2})\)?", ku)
            if not m:
                raise ValueError(f"unknown condition {k!r}; use T, P, X(EL)")
            out[v.X(m.group(1))] = val
    return out


@pcache.cached
def equilibrium_table(tdb: str, elements: Sequence[str], conditions: Dict, unit: str = "K",
                      phases: Optional[Sequence[str]] = None, max_points: int = 60) -> Dict:
    """Equilibrium at given conditions.  conditions: {"T": 1273 or [..], "X(CR)": 0.2, "P": 101325}."""
    from pycalphad import equilibrium, variables as v
    db = get_db(tdb)
    els = [e.upper() for e in elements]
    comps = els + ["VA"]
    ph = list(phases) if phases else phases_for(db, els)
    c = dict(conditions)
    for key in list(c):
        if key.upper() in ("T", "T_K") and unit.upper().startswith("C"):
            c[key] = to_K(c[key], "C")
    conds = _conds(els, c)
    n_pts = int(np.prod([len(np.atleast_1d(x)) for x in conds.values()]))
    if n_pts > max_points:
        raise ValueError(f"{n_pts} condition points requested; limit is {max_points}. Use a plot tool for maps.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, comps, ph, conds)
    rows = []
    phase_arr = eq.Phase.values
    np_arr = eq.NP.values
    x_arr = eq.X.values
    gm = eq.GM.values
    comp_names = [str(x) for x in eq.component.values]
    cond_dims = [d for d in eq.Phase.dims if d != "vertex"]
    for idx in np.ndindex(*phase_arr.shape[:-1]):
        point = {d: float(eq.coords[d].values[i]) for d, i in zip(cond_dims, idx)}
        if "T" in point:
            point["T_C"] = round(point["T"] - KELVIN, 2)
        stable = []
        for vtx in range(phase_arr.shape[-1]):
            name = str(phase_arr[idx + (vtx,)])
            if not name:
                continue
            stable.append({"phase": name, "fraction": round(float(np_arr[idx + (vtx,)]), 5),
                           "composition": {cn: round(float(x_arr[idx + (vtx, j)]), 5)
                                           for j, cn in enumerate(comp_names)}})
        rows.append({"conditions": point, "GM_J_per_mol": round(float(gm[idx]), 2), "stable_phases": stable})
    return {"tdb": tdb, "components": els, "phases_considered": ph, "points": rows}


@pcache.cached
def phase_fractions_vs_T(tdb: str, elements: Sequence[str], composition: Dict[str, float], tmin: float,
                         tmax: float, out_png: str,
                         unit: str = "K", step: float = 5.0, phases: Optional[Sequence[str]] = None,
                         title: Optional[str] = None) -> Dict:
    """Step calculation: phase fractions vs T at fixed overall composition.

    ``elements`` = all elements (incl. the balance one); ``composition`` = mole fractions of
    all but the balance element, e.g. elements=[FE,CR,NI], composition={CR: .18, NI: .08}.
    """
    from pycalphad import equilibrium, variables as v
    db = get_db(tdb)
    comp = {k.upper().replace("X(", "").replace(")", ""): float(x) for k, x in composition.items()}
    els = sorted({e.upper() for e in elements} | set(comp))
    if len(comp) != len(els) - 1:
        raise ValueError(f"give mole fractions for all elements but one (balance); elements={els}, got {comp}")
    ph = list(phases) if phases else phases_for(db, els)
    lo, hi = to_K(tmin, unit), to_K(tmax, unit)
    T = np.arange(lo, hi + step / 2, step)
    conds = {v.T: T, v.P: 101325, v.N: 1, **{v.X(k): val for k, val in comp.items()}}
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, els + ["VA"], ph, conds)
    P = eq.Phase.values.squeeze()
    NP = eq.NP.values.squeeze()
    names = sorted({str(p) for p in np.unique(P) if str(p)})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = T - KELVIN if unit.upper().startswith("C") else T
    summary = {}
    for nm in names:
        frac = np.nansum(np.where(P == nm, NP, 0.0), axis=-1)
        ax.plot(xs, frac, label=nm, lw=2)
        present = xs[frac > 1e-4]
        if present.size:
            summary[nm] = {"present_from": round(float(present.min()), 1), "present_to": round(float(present.max()), 1),
                           "max_fraction": round(float(frac.max()), 4)}
    ax.set_xlabel("Temperature (°C)" if unit.upper().startswith("C") else "Temperature (K)")
    ax.set_ylabel("Phase fraction (mol)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    comp_s = ", ".join(f"X({k})={val:g}" for k, val in comp.items())
    ax.set_title(title or f"{'-'.join(e.capitalize() for e in els)}   {comp_s}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"png": out_png, "phases": summary, "temperature_unit": "C" if unit.upper().startswith("C") else "K",
            "seconds": round(time.time() - t0, 1)}


@pcache.cached
def gibbs_energy_curves(tdb: str, elements: Sequence[str], temperature: float, out_png: str, unit: str = "K",
                        phases: Optional[Sequence[str]] = None, points: int = 2000) -> Dict:
    """Molar Gibbs energy vs X(second element) for each phase (lower envelope) at one T."""
    from pycalphad import calculate, variables as v
    db = get_db(tdb)
    a, b = sorted(e.upper() for e in elements)
    ph = list(phases) if phases else phases_for(db, [a, b])
    T = to_K(temperature, unit)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bins = np.linspace(0, 1, 201)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ph:
            try:
                r = calculate(db, [a, b, "VA"], name, T=T, P=101325, N=1, output="GM", pdens=points)
            except Exception:  # noqa: BLE001
                continue
            x = r.X.sel(component=b).values.ravel()
            g = r.GM.values.ravel()
            idx = np.digitize(x, bins)
            env_x, env_g = [], []
            for k in np.unique(idx):
                m = idx == k
                j = np.argmin(g[m])
                env_x.append(x[m][j])
                env_g.append(g[m][j])
            o = np.argsort(env_x)
            ax.plot(np.array(env_x)[o], np.array(env_g)[o] / 1000.0, label=name, lw=1.8)
    ax.set_xlabel(f"Mole fraction {b.capitalize()}")
    ax.set_ylabel("G (kJ/mol)")
    ax.set_title(f"{a.capitalize()}–{b.capitalize()}  G–x curves at {T:.0f} K ({T - KELVIN:.0f} °C)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"png": out_png, "phases": ph}


@pcache.cached
def isopleth(tdb: str, elements: Sequence[str], vary: str, fixed: Dict[str, float], tmin: float, tmax: float,
             out_png: str, unit: str = "K", x_max: float = 1.0, x_step: float = 0.01, t_step: float = 10.0,
             phases: Optional[Sequence[str]] = None) -> Dict:
    """Vertical section: T vs X(vary) with other mole fractions fixed."""
    from pycalphad import variables as v
    from pycalphad.mapping import IsoplethStrategy, plot_isopleth
    db = get_db(tdb)
    els = [e.upper() for e in elements]
    ph = list(phases) if phases else phases_for(db, els)
    lo, hi = to_K(tmin, unit), to_K(tmax, unit)
    conds = {v.T: (lo, hi, t_step), v.X(vary.upper()): (0, x_max, x_step), v.P: 101325, v.N: 1}
    for k, val in fixed.items():
        conds[v.X(k.upper().replace("X(", "").replace(")", ""))] = float(val)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = IsoplethStrategy(db, els + ["VA"], ph, conds)
        s.do_map()
        fig, ax = plt.subplots(figsize=(7, 4.8))
        plot_isopleth(s, ax=ax)
    ax.set_ylim(lo, hi)
    if unit.upper().startswith("C"):
        to_celsius_axis(ax)
    fx = ", ".join(f"X({k.upper()})={val:g}" for k, val in fixed.items())
    ax.set_title(f"{'-'.join(e.capitalize() for e in els)} isopleth, {fx}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"png": out_png, "phases": ph, "seconds": round(time.time() - t0, 1)}
