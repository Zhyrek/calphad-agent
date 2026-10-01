"""Agent tools: overlay experimental (ESPEI) data on calculations, and composition-unit conversion.

* plot_diagram_with_data      - binary phase diagram + ESPEI ZPF datasets (espei.plot.dataplot)
* plot_parameter_vs_data      - endmember / interaction property from the TDB vs ESPEI thermochemical
                                datasets (espei.plot.plot_endmember / plot_interaction)
* convert_composition         - wt% / wt fraction / at% / mole fraction conversion (pycalphad
                                variables.get_mole_fractions / get_mass_fractions)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .diagrams import KELVIN, TIELINE_COLOR, phases_for, to_celsius_axis  # noqa: E402

S = {"type": "string"}
N = {"type": "number"}
STRS = {"type": "array", "items": {"type": "string"}}


def _load_datasets(directory: str):
    from espei.datasets import load_datasets, recursive_glob
    files = recursive_glob(directory, "*.json")
    if not files:
        raise FileNotFoundError(f"no ESPEI *.json datasets under {directory}")
    return load_datasets(files), len(files)


class _PhaseColors(dict):
    """Phase -> colour; a second composition set (BCC_A2#2, miscibility gap) gets its phase's colour."""

    def __missing__(self, key):
        base = str(key).split("#")[0]
        if base != key and base in self:
            return self[base]
        raise KeyError(key)


def _consistent_legend(all_phases: List[str]):
    """legend_generator giving every phase the same colour in every plot call."""
    from pycalphad.plot.utils import phase_legend
    handles, colors = phase_legend(all_phases)
    colors = _PhaseColors(colors)

    def gen(phases):
        want = {str(p) for p in phases}
        return [h for h in handles if h.get_label() in want], colors
    return gen


def _save(fig, out: str, formats, label_regions: bool) -> List[str]:
    fmts = formats if formats is not None else (["svg", "pdf"] if label_regions else [])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    files = [out]
    for ext in fmts:
        ext = str(ext).lower().lstrip(".")
        if ext not in ("svg", "pdf", "eps", "png"):
            raise ValueError(f"unsupported format {ext}")
        if ext != "png":
            from .label_place import save_vector
            files.append(str(Path(out).with_suffix("." + ext)))
            save_vector(fig, files[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return files


def _ternary_with_data(tb, els, datasets_dir, tdb, T_K, off, phases, output_png, label_regions, phase_labels,
                       label_style, formats, T_tolerance):
    """Ternary isotherm (pycalphad mapping) + ZPF data within T_tolerance of T, placed through the corners."""
    import matplotlib.lines as mlines
    from espei.utils import bib_marker_map
    from pycalphad import ternplot, variables as v
    from pycalphad.plot import triangular  # noqa: F401 - registers the projection
    from .calc import get_db
    from .orientation import corner_labels, orientation_text, plot_zpf, zpf_equilibria
    a, b, c = els
    if label_regions and not tdb:
        raise ValueError("label_regions needs a tdb")
    datasets, nfiles = _load_datasets(tb.p(datasets_dir))
    eqs = zpf_equilibria(datasets, els, T_K, T_tolerance)
    data_phases = {ph for eq in eqs for ph, _ in eq["phases"]}
    ph = [p.upper() for p in phases] if phases else None
    shown = set(data_phases)
    fig = plt.figure(figsize=(7.4, 6.0))
    ax = fig.add_subplot(projection="triangular")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if tdb:
            db = get_db(tb._find_tdb(tdb))
            ph = ph or phases_for(db, els)
        else:
            ph = ph or sorted(data_phases)
        legend = _consistent_legend(sorted(set(ph) | data_phases))
        if tdb:
            _, strat = ternplot(db, [a, b, c, "VA"], ph, {v.X(b): (0, 1, 0.02), v.X(c): (0, 1, 0.02), v.T: T_K,
                                                          v.P: 101325, v.N: 1},
                                x=v.X(b), y=v.X(c), ax=ax, tielines=4, tieline_color=TIELINE_COLOR,
                                legend_generator=legend, return_strategy=True)
            shown |= {cs.phase_record.phase_name for z in strat.zpf_lines for p in z.points
                      for cs in p.stable_composition_sets}
    recipe = {"kind": "ternary", "tdb": str(Path(tb._find_tdb(tdb)).resolve()) if tdb else None, "elements": list(els),
              "phases": list(ph), "P": 101325.0, "rename_disordered": True, "composition_step": 0.02,
              "temperature_step": 10.0, "T_unit": "C" if off else "K", "T_K": float(T_K),
              "comp": {"scale": 1.0, "weight": False}}
    refs = sorted({eq["reference"] for eq in eqs})
    markers = bib_marker_map(refs)
    colors = legend(sorted(set(ph) | data_phases))[1]
    handles = legend(sorted(shown))[0]
    n = plot_zpf(ax, recipe, eqs, colors, markers)
    handles = list(handles) + [mlines.Line2D([], [], linestyle="", color="k", marker=markers[r]["markers"]["marker"],
                                             fillstyle=markers[r]["markers"]["fillstyle"], markersize=6,
                                             label=markers[r]["formatted"]) for r in refs]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Mole fraction {b.capitalize()}")
    ax.set_ylabel(f"Mole fraction {c.capitalize()}")
    ax.yaxis.label.set_rotation(60)
    ax.yaxis.set_label_coords(x=0.12, y=0.5)
    corner_labels(ax, els)
    ax.set_title(f"{a.capitalize()}–{b.capitalize()}–{c.capitalize()}   {T_K - KELVIN:.0f} °C ({T_K:.0f} K)"
                 + (f"  ({Path(tdb).name} + data)" if tdb else "  (data)"), fontsize=11)
    ax._calphad_diagram = {"recipe": recipe} if tdb else {"orientation": recipe}
    note = ""
    if label_regions:
        from .region_labels import label_axes
        note = "\n" + label_axes(ax, recipe, phase_labels=phase_labels, style=label_style)["text"]
    out = tb.outp(output_png, f"diagram_data_{a}-{b}-{c}_{T_K - off:g}{'C' if off else 'K'}.png")
    files = _save(fig, out, formats, label_regions)
    return (f"saved {', '.join(tb.rel(f) for f in files)}: {nfiles} dataset files loaded; {len(eqs)} ZPF "
            f"equilibria of {a}-{b}-{c} within {T_tolerance:g} K of {T_K:.2f} K ({n['tielines']} tie-lines, "
            f"{n['single']} single points of broken tie-lines, {n['triangles']} tie-triangles)"
            + ("" if tdb else " (no TDB: data only)") + "\n" + orientation_text("ternary", els) + note), [out]


def register(tb) -> None:
    @tb.add("plot_diagram_with_data",
            "Phase diagram with experimental ESPEI ZPF datasets overlaid: binary T-x (espei dataplot) or ternary "
            "isotherm (3 elements + temperature; ZPF points within T_tolerance, tie-lines as segments). Omit tdb to "
            "plot only the data. Orientation: right (binary) / corners (ternary), default alphabetical; the data "
            "follow it.",
            {"elements": STRS, "datasets_dir": S, "tdb": S, "tmin": N, "tmax": N,
             "temperature": {**N, "description": "ternary isotherm temperature"},
             "T_tolerance": {**N, "description": "ternary: ZPF points within this many K of temperature (10)"},
             "temperature_unit": {"type": "string", "enum": ["K", "C"]}, "phases": STRS, "output_png": S,
             "right": {**S, "description": "binary: element on the right (x axis = its fraction)"},
             "corners": {"type": "object", "properties": {"left": S, "right": S, "top": S},
                         "description": "ternary: element at bottom-left, bottom-right, top"},
             "label_regions": {"type": "boolean", "description": "write the phases of each region and invariant "
                                                                 "temperatures on the plot, clear of the data points"},
             "phase_labels": {"type": "object", "additionalProperties": {"type": "string"}},
             "label_style": {"type": "object", "description": "as in map_phase_diagram"},
             "formats": {**STRS, "description": "extra files svg/pdf (default svg+pdf when labelling)"}},
            ["elements", "datasets_dir"])
    def plot_diagram_with_data(elements, datasets_dir, tdb: Optional[str] = None, tmin: float = 300,
                               tmax: float = 2500, temperature_unit: str = "K", phases=None,
                               output_png: Optional[str] = None, label_regions: bool = False, phase_labels=None,
                               label_style=None, formats=None, right: Optional[str] = None,
                               corners: Optional[Dict[str, str]] = None, temperature: Optional[float] = None,
                               T_tolerance: float = 10.0):
        from .orientation import binary_order, orientation_text, ternary_order
        els = sorted(e.upper() for e in elements)
        off = KELVIN if temperature_unit.upper().startswith("C") else 0.0
        if len(els) == 3:
            if temperature is None:
                raise ValueError("a ternary (3 elements) needs the isotherm temperature")
            els = ternary_order(els, corners, right)
            return _ternary_with_data(tb, els, datasets_dir, tdb, float(temperature) + off, off, phases, output_png,
                                      label_regions, phase_labels, label_style, formats, float(T_tolerance))
        if len(els) != 2:
            raise ValueError("give 2 elements (binary T-x) or 3 elements + temperature (ternary isotherm)")
        if corners:
            raise ValueError("corners is for ternary isotherms; a binary uses right")
        from espei.plot import dataplot
        from pycalphad import binplot, variables as v
        from .calc import get_db
        a, b = binary_order(els, right)
        comps = [a, b, "VA"]
        lo, hi = tmin + off, tmax + off
        # x axis = X(b): binplot maps X(b), and ESPEI's dataplot converts each data point to X(b) by mass balance
        # (datasets store only the independent component), so the points follow the orientation
        conds = {v.P: 101325, v.T: (lo, hi, 10), v.X(b): (0, 1, 0.02), v.N: 1}
        datasets, nfiles = _load_datasets(tb.p(datasets_dir))
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ph = [p.upper() for p in phases] if phases else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_phases = {p for d in datasets.all() for p in d.get("phases", [])}
            if tdb:
                db = get_db(tb._find_tdb(tdb))
                ph = ph or phases_for(db, [a, b])
            else:
                ph = ph or sorted(data_phases)
            # one colour per phase, shared by the calculated lines and the data markers
            legend = _consistent_legend(sorted(set(ph) | data_phases))
            if tdb:
                binplot(db, comps, ph, conds, plot_kwargs=dict(ax=ax, tielines=4, tieline_color=TIELINE_COLOR,
                                                               legend_generator=legend))
            dataplot(comps, ph, conds, datasets, ax=ax, legend_generator=legend)
        n_pts = sum(len(d.get("values", [])) for d in datasets.all()
                    if d.get("output") == "ZPF" and set(d.get("components", [])) <= set(comps))
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        if off:
            to_celsius_axis(ax)
        ax.set_xlabel(f"Mole fraction {b.capitalize()}")
        ax.set_title(f"{a.capitalize()}–{b.capitalize()}" + (f"  ({Path(tdb).name} + data)" if tdb else "  (data)"))
        out = tb.outp(output_png, f"diagram_data_{a}-{b}.png")
        note = ""
        if label_regions and not tdb:
            raise ValueError("label_regions needs a tdb")
        recipe = {"kind": "binary", "tdb": str(Path(tb._find_tdb(tdb)).resolve()) if tdb else None,
                  "elements": [a, b], "phases": list(ph), "P": 101325.0, "rename_disordered": True,
                  "composition_step": 0.02, "temperature_step": 10.0, "T_unit": "C" if off else "K",
                  "tmap": [1.0, 0.0], "T_range": [lo, hi], "x_range": [0.0, 1.0],
                  "comp": {"scale": 1.0, "weight": False}}
        ax._calphad_diagram = {"recipe": recipe} if tdb else {"orientation": recipe}
        if label_regions:
            from .region_labels import label_axes
            note = "\n" + label_axes(ax, recipe, phase_labels=phase_labels, style=label_style)["text"]
        files = _save(fig, out, formats, label_regions)
        return (f"saved {', '.join(tb.rel(f) for f in files)}: {nfiles} dataset files loaded, {n_pts} ZPF points "
                f"in {a}-{b}" + ("" if tdb else " (no TDB: data only)") + "\n"
                + orientation_text("binary", [a, b], f"mole fraction {b.capitalize()}") + note), [out]

    @tb.add("plot_parameter_vs_data",
            "Compare a TDB's endmember or interaction property with ESPEI thermochemical datasets "
            "(espei plot_endmember / plot_interaction). configuration is per sublattice, e.g. [\"CR\",\"NI\"] "
            "(endmember) or [[\"CR\",\"NI\"],\"VA\"] (interaction). output e.g. HM_FORM, SM_FORM, CPM_FORM, "
            "HM_MIX, SM_MIX. Endmembers are plotted vs T; interactions vs composition.",
            {"tdb": S, "elements": STRS, "phase": S, "configuration": {"type": "array"}, "output": S,
             "datasets_dir": S, "output_png": S},
            ["tdb", "elements", "phase", "configuration", "output"])
    def plot_parameter_vs_data(tdb, elements, phase, configuration, output, datasets_dir: Optional[str] = None,
                               output_png: Optional[str] = None):
        from espei.plot import plot_endmember, plot_interaction
        from .calc import get_db
        db = get_db(tb._find_tdb(tdb))
        phase = phase.upper()
        if phase not in db.phases:
            raise ValueError(f"phase {phase} not in TDB; phases: {sorted(db.phases)}")
        comps = sorted({e.upper() for e in elements} | {"VA"})
        cfg = [[s.upper() for s in c] if isinstance(c, list) else c.upper() for c in configuration]
        cfg_t = tuple(tuple(c) if isinstance(c, list) else c for c in cfg)
        datasets = _load_datasets(tb.p(datasets_dir))[0] if datasets_dir else None
        is_interaction = any(isinstance(c, list) and len(c) > 1 for c in cfg)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_interaction:
                plot_interaction(db, comps, phase, cfg_t, output.upper(), datasets=datasets, ax=ax)
            else:
                plot_endmember(db, comps, phase, cfg_t, output.upper(), datasets=datasets, ax=ax)
        label = ":".join(",".join(c) if isinstance(c, list) else c for c in cfg)
        out = tb.outp(output_png, f"{output.upper()}_{phase}_{label.replace(':', '-').replace(',', '')}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return (f"saved {tb.rel(out)} ({'interaction' if is_interaction else 'endmember'} {phase}({label}) "
                f"{output.upper()}; {'with' if datasets else 'without'} datasets)"), [out]

    @tb.add("convert_composition",
            "Convert an alloy composition between wt%, weight fraction, at% and mole fraction (any number of "
            "elements). Give all elements except the balance element. Masses from the TDB if given.",
            {"composition": {"type": "object", "additionalProperties": {"type": "number"}},
             "balance": S, "from_unit": {"type": "string", "enum": ["wt%", "wt_fraction", "at%", "mole_fraction"]},
             "tdb": S}, ["composition", "balance", "from_unit"])
    def convert_composition(composition: Dict[str, float], balance: str, from_unit: str, tdb: Optional[str] = None):
        from pycalphad import variables as v
        from .expdata import ATOMIC_MASS
        comp = {k.upper().replace("W(", "").replace("X(", "").replace(")", ""): float(x) for k, x in composition.items()}
        bal = balance.upper()
        els = list(comp) + [bal]
        if tdb:
            from .calc import get_db
            db = get_db(tb._find_tdb(tdb))
            masses = {e: float(db.refstates[e]["mass"]) for e in els}
        else:
            masses = {e: ATOMIC_MASS[e] for e in els}
        scale = 100.0 if from_unit.endswith("%") else 1.0
        vals = {e: x / scale for e, x in comp.items()}
        if sum(vals.values()) >= 1:
            raise ValueError(f"fractions of the non-balance elements sum to {sum(vals.values()):.4f} (must be < 1)")
        if from_unit.startswith("wt"):
            mole = {str(k.name if hasattr(k, "name") else k).replace("X_", ""): float(val) for k, val in
                    v.get_mole_fractions({v.W(e): x for e, x in vals.items()}, bal, masses).items()}
            mole = {k.split("_")[-1]: val for k, val in mole.items()}
            mass = vals
        else:
            mass = {str(k).split("_")[-1]: float(val) for k, val in
                    v.get_mass_fractions({v.X(e): x for e, x in vals.items()}, bal, masses).items()}
            mole = vals
        mole[bal] = 1 - sum(x for e, x in mole.items() if e != bal)
        mass[bal] = 1 - sum(x for e, x in mass.items() if e != bal)
        rows = [f"  {e:<3s} {100 * mass[e]:9.4f} wt%   {100 * mole[e]:9.4f} at%   X={mole[e]:.6f}" for e in els]
        cond = ", ".join(f'"X({e})": {mole[e]:.6f}' for e in comp)
        return ("composition (balance " + bal + "):\n" + "\n".join(rows) +
                f"\nfor equilibrium conditions use: {{{cond}}}"), []
