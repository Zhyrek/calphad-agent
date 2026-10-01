"""Synthetic "paper figures" of phase diagrams with known ground truth, for digitized_to_zpf.

* :func:`make_binary_figure` - Cr-Ni (CrNi-06Hal.tdb) T-x diagram in °C and at% (or wt%): black boundary lines,
  noisy markers on the boundaries (one marker style per boundary type), tie-line pairs at one T, the eutectic
  points, a legend, and region labels in paper notation ("L", "(Cr)", "L+(Cr)", "(Cr)+(Ni)" ...).
* :func:`make_ternary_figure` - Co-Cr-Ni isotherm (CoCrFeNiV_Choi2019.tdb) drawn as a Gibbs triangle with
  tie-lines joining marker pairs, a tie-triangle, edge ticks and region labels.

    .venv/bin/python tests/digitize_zpf_bench.py [outdir]
"""
import io
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TDBS = ROOT / "examples" / "tdbs"
CRNI = str(TDBS / "CrNi-06Hal.tdb")
CHOI = str(TDBS / "CoCrFeNiV_Choi2019.tdb")
KELVIN = 273.15
MASS = {"CR": 51.996, "NI": 58.6934}

# marker style per boundary type (paper-like: distinct shape AND colour)
STYLES = {
    "liquidus": dict(marker="o", color="red", filled=True, label="liquidus (DTA)"),
    "solidus": dict(marker="s", color="blue", filled=True, label="solidus (DTA)"),
    "bcc_solvus": dict(marker="^", color="green", filled=True, label="(Cr)/(Cr)+(Ni) (EPMA)"),
    "fcc_solvus": dict(marker="D", color="purple", filled=False, label="(Ni)/(Cr)+(Ni) (EPMA)"),
    "eutectic": dict(marker="v", color="darkorange", filled=True, label="eutectic"),
}
PHASE_OF = {"LIQUID": "L", "BCC_A2": "(Cr)", "FCC_A1": "(Ni)"}


def crni_map(tmin_k=1000.0, tmax_k=2300.0):
    from calphad_tools import pcache
    from calphad_tools.mapping_tools import _select_phases
    from calphad_tools.phase_regions import map_lines, ordered_info
    db = pcache.get_db(CRNI)
    ph = _select_phases(db, ["CR", "NI"])
    recipe = {"kind": "binary", "elements": ["CR", "NI"], "phases": ph, "P": 101325.0,
              "T_range": [tmin_k, tmax_k], "x_range": [0.0, 1.0]}
    with pcache.use(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lines, inv = map_lines(db, recipe, ordered_info(db, ["CR", "NI", "VA"], ph), step=0.02, T_step=10.0)
    return lines, inv


def _trace(lines, phase, other):
    """(x_NI, T_K) along the boundary of ``phase`` in its two-phase field with ``other`` (sorted by T)."""
    pts = []
    for ln in lines:
        names = [n.split("#")[0] for n in ln["phases"]]
        if sorted(names) != sorted([phase, other]):
            continue
        k = names.index(phase)
        pts += [(p["phases"][k][1]["NI"], p["T"]) for p in ln["points"]]
    a = np.array(sorted(set((round(x, 6), round(t, 3)) for x, t in pts), key=lambda q: q[1]))
    return a


def _at_T(tr, T):
    """x on a trace at temperature T (linear interpolation along T; the trace is monotonic in T here)."""
    o = np.argsort(tr[:, 1])
    return float(np.interp(T, tr[o, 1], tr[o, 0]))


def x_to_wt(x):
    x = np.asarray(x, float)
    return 100 * x * MASS["NI"] / (x * MASS["NI"] + (1 - x) * MASS["CR"])


def crni_truth(seed=0, noise_x=0.002, noise_T=2.0):
    """Ground-truth points (x = X(NI), T in K) on the calculated boundaries, with small noise."""
    lines, inv = crni_map()
    rng = np.random.default_rng(seed)
    e = inv[0]
    Te = e["T_K"]
    xe = {n.split("#")[0]: c["NI"] for n, c in e["compositions"].items()}
    tr = {("LIQUID", "BCC_A2"): _trace(lines, "LIQUID", "BCC_A2"), ("BCC_A2", "LIQUID"): _trace(lines, "BCC_A2", "LIQUID"),
          ("LIQUID", "FCC_A1"): _trace(lines, "LIQUID", "FCC_A1"), ("FCC_A1", "LIQUID"): _trace(lines, "FCC_A1", "LIQUID"),
          ("BCC_A2", "FCC_A1"): _trace(lines, "BCC_A2", "FCC_A1"), ("FCC_A1", "BCC_A2"): _trace(lines, "FCC_A1", "BCC_A2")}
    pts = []

    def add(group, ph, co, T, pair=None):
        x = _at_T(tr[(ph, co)], T) + rng.normal(0, noise_x)
        pts.append({"group": group, "phase": ph, "coexisting": [co], "x": float(x),
                    "T": float(T + rng.normal(0, noise_T)), "pair": pair})
    for T in np.linspace(Te + 60, 2120, 7):                       # Cr-side liquidus
        add("liquidus", "LIQUID", "BCC_A2", T)
    for T in (Te + 30, Te + 70):                                  # Ni-side liquidus (narrow lens)
        add("liquidus", "LIQUID", "FCC_A1", T)
    for T in np.linspace(Te + 90, 2100, 6):                       # Cr-side solidus
        add("solidus", "BCC_A2", "LIQUID", T)
    add("solidus", "FCC_A1", "LIQUID", Te + 50)
    for T in (1100.0, 1200.0, 1320.0, 1400.0, 1520.0, 1570.0):  # solvus lines (T apart from the tie-line pairs)
        add("bcc_solvus", "BCC_A2", "FCC_A1", T)
    for T in (1150.0, 1280.0, 1370.0, 1500.0, 1545.0):
        add("fcc_solvus", "FCC_A1", "BCC_A2", T)
    for k, T in enumerate((1240.0, 1450.0)):                      # tie-line pairs at one T (both ends measured)
        add("bcc_solvus", "BCC_A2", "FCC_A1", T, pair=f"s{k}")
        pts[-1]["T"] = T
        add("fcc_solvus", "FCC_A1", "BCC_A2", T, pair=f"s{k}")
        pts[-1]["T"] = T
    T = Te + 150                                                  # a liquidus/solidus pair (L + (Cr))
    add("liquidus", "LIQUID", "BCC_A2", T, pair="l0")
    pts[-1]["T"] = T
    add("solidus", "BCC_A2", "LIQUID", T, pair="l0")
    pts[-1]["T"] = T
    for n in ("BCC_A2", "LIQUID", "FCC_A1"):                      # eutectic: the three phase compositions
        pts.append({"group": "eutectic", "phase": n, "coexisting": sorted({"BCC_A2", "LIQUID", "FCC_A1"} - {n}),
                    "x": float(xe[n] + rng.normal(0, noise_x)), "T": float(Te + rng.normal(0, 0.5)),
                    "pair": "inv"})
    return pts, lines, inv


# region labels in paper notation: (label, x_NI, T_C) - placed inside the regions
CRNI_LABELS = [("L", 0.55, 1750.0), ("(Cr)", 0.08, 1250.0), ("(Ni)", 0.85, 1050.0), ("L+(Cr)", 0.165, 1650.0),
               ("(Cr)+(Ni)", 0.33, 1000.0)]


def make_binary_figure(out, name="crni_paper", seed=0, wt=False, labels=CRNI_LABELS, dpi=110, draw_lines=True,
                       legend=True, jpeg=None):
    """Render the Cr-Ni 'paper figure'.  Returns (path, truth points, axis info for the simulated model)."""
    pts, lines, inv = crni_truth(seed)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    conv = (lambda x: x_to_wt(x)) if wt else (lambda x: 100 * np.asarray(x, float))
    if draw_lines:
        for ln in lines:
            for k in range(2):
                P = np.array([(p["phases"][k][1]["NI"], p["T"] - KELVIN) for p in ln["points"]])
                ax.plot(conv(P[:, 0]), P[:, 1], color="k", lw=1.0, zorder=1)
        for e in inv:
            xs = [c["NI"] for c in e["compositions"].values()]
            ax.plot(conv([min(xs), max(xs)]), [e["T_K"] - KELVIN] * 2, color="k", lw=1.0, zorder=1)
    for g, st in STYLES.items():
        gp = [p for p in pts if p["group"] == g]
        kw = dict(marker=st["marker"], linestyle="none", markersize=6.5, color=st["color"], label=st["label"],
                  zorder=3)
        if not st["filled"]:
            kw.update(markerfacecolor="white", markeredgewidth=1.3)
        ax.plot(conv([p["x"] for p in gp]), [p["T"] - KELVIN for p in gp], **kw)
    for lab, x, t in labels or []:
        ax.text(conv(x), t, lab, fontsize=10, ha="center", va="center", zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(800, 2000)
    ax.set_xlabel("Ni (wt.%)" if wt else "Ni (at.%)")
    ax.set_ylabel("Temperature (°C)")
    ax.tick_params(direction="in", top=True, right=True)
    if legend:
        ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="k", fancybox=False)
    xt = [v for v in ax.get_xticks() if 0 <= v <= 100]
    yt = [v for v in ax.get_yticks() if 800 <= v <= 2000]
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png")
    plt.close(fig)
    from PIL import Image
    im = Image.open(buf).convert("RGB")
    path = Path(out) / f"{name}.{'jpg' if jpeg else 'png'}"
    if jpeg:
        im.save(path, quality=jpeg)
    else:
        im.save(path)
    return str(path), pts, {"xticks": xt, "yticks": yt, "labels": labels}


# --------------------------------------------------------------------------------------------- ternary
def cocrni_map(T_K=1273.15):
    from calphad_tools import pcache
    from calphad_tools.mapping_tools import _select_phases
    from calphad_tools.phase_regions import map_lines, ordered_info
    db = pcache.get_db(CHOI)
    ph = _select_phases(db, ["CO", "CR", "NI"])
    recipe = {"kind": "ternary", "elements": ["CO", "CR", "NI"], "phases": ph, "P": 101325.0, "T_K": T_K}
    with pcache.use(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return map_lines(db, recipe, ordered_info(db, ["CO", "CR", "NI", "VA"], ph), step=0.02)


# corners: c0 bottom-left = NI, c1 bottom-right = CR, c2 top = CO
CORNERS = {"NI": (0.0, 0.0), "CR": (1.0, 0.0), "CO": (0.5, math.sqrt(3) / 2)}


def tern_xy(X):
    x = sum(X[e] * CORNERS[e][0] for e in CORNERS)
    y = sum(X[e] * CORNERS[e][1] for e in CORNERS)
    return x, y


TERN_STYLES = {"FCC_A1": dict(marker="o", color="blue", label="fcc"),
               "SIGMA": dict(marker="s", color="red", label="σ"),
               "BCC_A2": dict(marker="^", color="green", label="bcc")}
TERN_LABELS = [("fcc", {"CR": 0.2, "CO": 0.4}), ("σ", {"CR": 0.66, "CO": 0.22}), ("bcc", {"CR": 0.93, "CO": 0.03}),
               ("fcc+σ", {"CR": 0.5, "CO": 0.33})]


def ternary_truth(seed=0, noise=0.002, min_sep=0.055, per_line=3):
    """Tie-lines picked along the calculated two-phase fields (ends >= min_sep apart in the triangle, so markers do
    not overlap), plus the tie-triangle corners."""
    lines, inv = cocrni_map()
    rng = np.random.default_rng(seed)
    pts = []
    placed = []
    for e in inv:                                         # the tie-triangle corners first
        names = [n.split("#")[0] for n in e["phases"]]
        for n in names:
            X = {k: float(v) for k, v in e["compositions"][n].items()}
            pts.append({"phase": n, "coexisting": sorted(set(names) - {n}), "X": X, "tie": "tri"})
            placed.append(np.array(tern_xy(X)))
    tid = 0
    for ln in lines:
        names = [n.split("#")[0] for n in ln["phases"]]
        P = ln["points"]
        n_ok = 0
        for i in list(range(len(P) // 2, len(P))) + list(range(len(P) // 2 - 1, -1, -1)):
            q = P[i]
            ends = [np.array(tern_xy(q["phases"][j][1])) for j in range(2)]
            if np.hypot(*(ends[0] - ends[1])) < 0.1:
                continue
            if any(np.hypot(*(e_ - o)) < min_sep for e_ in ends for o in placed):
                continue
            for j in range(2):
                X = {e: float(v) for e, v in q["phases"][j][1].items()}
                X["CR"] += rng.normal(0, noise)
                X["CO"] += rng.normal(0, noise)
                X["NI"] = 1 - X["CR"] - X["CO"]
                pts.append({"phase": names[j], "coexisting": [names[1 - j]], "X": X, "tie": f"t{tid}"})
            placed += ends
            tid += 1
            n_ok += 1
            if n_ok >= per_line:
                break
    return pts, lines, inv


def make_ternary_figure(out, name="cocrni_paper", seed=0, dpi=110, labels=TERN_LABELS):
    pts, lines, inv = ternary_truth(seed)
    fig, ax = plt.subplots(figsize=(6.4, 5.9))
    c = np.array([tern_xy({"NI": 1, "CR": 0, "CO": 0}), tern_xy({"NI": 0, "CR": 1, "CO": 0}),
                  tern_xy({"NI": 0, "CR": 0, "CO": 1})])
    ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="k", lw=1.3, zorder=2)
    for ln in lines:                                      # calculated boundaries (thin black)
        for k in range(2):
            P = np.array([tern_xy(p["phases"][k][1]) for p in ln["points"]])
            ax.plot(P[:, 0], P[:, 1], color="k", lw=0.9, zorder=1)
    for e in inv:
        P = np.array([tern_xy(v) for v in e["compositions"].values()])
        ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color="k", lw=0.9, zorder=1)
    ties = {}
    for p in pts:
        ties.setdefault(p["tie"], []).append(tern_xy(p["X"]))
    for t, P in ties.items():                             # drawn tie-lines / tie-triangle joining the markers
        P = np.array(P + ([P[0]] if len(P) == 3 else []))
        ax.plot(P[:, 0], P[:, 1], color="0.2", lw=0.8, zorder=2)
    for ph, st in TERN_STYLES.items():
        P = np.array([tern_xy(p["X"]) for p in pts if p["phase"] == ph])
        ax.plot(P[:, 0], P[:, 1], linestyle="none", marker=st["marker"], color=st["color"], markersize=6.5,
                label=st["label"], zorder=3)
    for f in np.arange(0.1, 1.0, 0.1):                   # ticks along the edges (parallel to another edge)
        for a, b, other in (("NI", "CR", "CO"), ("CR", "CO", "NI"), ("CO", "NI", "CR")):
            X = {a: 1 - f, b: f, other: 0.0}
            x, y = tern_xy(X)
            X2 = {a: 1 - f - 0.025, b: f, other: 0.025}
            x2, y2 = tern_xy(X2)
            ax.plot([x, 2 * x - x2], [y, 2 * y - y2], color="k", lw=0.9)
    for f in (0.2, 0.4, 0.6, 0.8):
        x, y = tern_xy({"NI": 1 - f, "CR": f, "CO": 0})
        ax.text(x, y - 0.045, f"{f:.1f}", ha="center", va="top", fontsize=8)
    for lab, X in labels or []:
        XX = dict(X)
        XX["NI"] = 1 - sum(XX.values())
        ax.text(*tern_xy(XX), lab, fontsize=11, ha="center", va="center", zorder=2)
    for e, (x, y) in CORNERS.items():
        ax.text(x + (-0.04 if x < 0.3 else 0.04 if x > 0.7 else 0), y + (0.04 if y > 0.5 else -0.07),
                e.capitalize(), ha="center", fontsize=12, fontweight="bold")
    ax.text(0.02, 0.8, "1000 °C", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="k", fancybox=False)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.12, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")
    path = Path(out) / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path), pts


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "zpf_bench_out")
    out.mkdir(parents=True, exist_ok=True)
    print(make_binary_figure(out)[0])
    print(make_binary_figure(out, name="crni_paper_wt", wt=True)[0])
    print(make_ternary_figure(out)[0])
