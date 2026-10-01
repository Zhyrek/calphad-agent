"""Close / overlapping marker benchmark for the plot digitiser, run through the agent's tools
(digitize_setup -> tick calibration -> digitize_points with colour / shape filters).

    .venv/bin/python -W ignore tests/digitize_overlap_bench.py <outdir> [pairs|realistic|both]

Variants
  pairs      only the close pairs / triples (every blob of a row is a pair: no isolated reference markers)
  realistic  the same pairs plus ~15 isolated markers per series scattered in a band above them (real figures
             have mostly isolated markers)

Scoring per cell (row x separation), flag-aware:
  hits      true markers matched by a detection within 0.45 d (flagged or not)
  +N        unflagged detections that match no true marker (false / silently merged)
  fN        flagged detections that match no true marker (an honest "may be 2+ markers" flag)
  (Nf)      N of the hits carry a flag (e.g. "may be 2+ overlapping markers": the right point, flagged)
  S         silent merge: markers are missing or an unflagged detection sits between them, and no detection in
            the cell is flagged
Isolated markers (realistic variant) are scored separately: recall, false detections, flagged.
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MS = 8            # marker size (pt)
DPI = 100
SEPS = [2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.25]      # centre distance in marker diameters

# scenario: (label, [(colour, marker, filled), second marker...], layout)
ROWS = [
    ("same colour, filled circles", [("red", "o", True), ("red", "o", True)], "pair"),
    ("same colour, open circles", [("blue", "o", False), ("blue", "o", False)], "pair"),
    ("different colours (orange o / purple s)", [("darkorange", "o", True), ("purple", "s", True)], "pair"),
    ("black: filled square / open triangle", [("k", "s", True), ("k", "^", False)], "pair"),
    ("black: identical filled circles", [("k", "o", True), ("k", "o", True)], "pair"),
    ("cluster of 3 (green triangles)", [("green", "^", True)] * 3, "triple"),
]
FILTERS = {
    0: [{"color": "red"}],
    1: [{"color": "blue", "filled": False}],
    2: [{"color": "darkorange"}, {"color": "purple"}],
    3: [{"color": "black", "shape": "square", "filled": True}, {"color": "black", "shape": "triangle_up",
                                                                 "filled": False}],
    4: [{"color": "black", "shape": "circle", "filled": True}],
    5: [{"color": "green"}],
}
# isolated series (realistic variant): one per distinct marker style
ISO = [("red", "o", True), ("blue", "o", False), ("darkorange", "o", True), ("purple", "s", True),
       ("k", "s", True), ("k", "^", False), ("k", "o", True), ("green", "^", True)]
N_ISO = 15


def build(variant, OUT):
    OUT = Path(OUT)
    realistic = variant == "realistic"
    ytop = 200 if realistic else 100
    fig, ax = plt.subplots(figsize=(10, 10 if realistic else 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ytop)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, ytop + 1, 20))
    fig.canvas.draw()
    d_px = MS * DPI / 72.0
    bb = ax.get_window_extent()
    ux = 100.0 / (bb.width * DPI / fig.dpi) * d_px          # data units per marker diameter
    uy = ytop / (bb.height * DPI / fig.dpi) * d_px
    truth = []

    def draw(col, mk, filled, x, y, z=3):
        kw = dict(marker=mk, markersize=MS, linestyle="none", color=col)
        if not filled:
            kw.update(markerfacecolor="none", markeredgewidth=1.3)
        ax.plot([x], [y], zorder=z, **kw)

    for r, (label, marks, layout) in enumerate(ROWS):
        yc = 90 - r * 15
        for c, s in enumerate(SEPS):
            xc = 8 + c * 11.7
            if layout == "pair":
                pts = [(xc - s * ux / 2, yc), (xc + s * ux / 2, yc)]
            else:
                ang = np.deg2rad([90, 210, 330])
                rad = s / np.sqrt(3)
                pts = [(xc + rad * ux * np.cos(a), yc + rad * uy * np.sin(a)) for a in ang]
            for k, ((col, mk, filled), (x, y)) in enumerate(zip(marks, pts)):
                draw(col, mk, filled, x, y, 3 + k)
                truth.append({"row": r, "sep": s, "k": k, "x": x, "y": y, "color": col, "marker": mk,
                              "filled": filled})
    if realistic:
        rng = np.random.default_rng(7)
        placed = []
        for si, (col, mk, filled) in enumerate(ISO):
            n = 0
            while n < N_ISO:
                x, y = rng.uniform(4, 96), rng.uniform(108, ytop - 5)
                if all(np.hypot((x - px) / ux, (y - py) / uy) > 2.6 for px, py in placed):
                    placed.append((x, y))
                    draw(col, mk, filled, x, y)
                    truth.append({"row": -1, "series": si, "x": x, "y": y, "color": col, "marker": mk,
                                  "filled": filled})
                    n += 1
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    path = OUT / f"overlap_stress_{variant}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    (OUT / f"truth_{variant}.json").write_text(json.dumps(truth))
    return path, truth, ux, uy, ytop


def run(variant, path, truth, ux, uy, ytop, OUT):
    from calphad_tools.agent.tools import UI, ToolBox
    OUT = Path(OUT)
    od = OUT / f"out_{variant}"
    tb = ToolBox(workdir=str(OUT), output_dir=str(od), ui=UI(), python_tool="off")
    text, imgs, err = tb.call("digitize_setup", {"image": str(path)})
    assert not err, text
    nx = int(re.search(r"x0\.\.x(\d+)", text).group(1)) + 1
    ny = int(re.search(r"y0\.\.y(\d+)", text).group(1)) + 1
    axes = {"x": {"ticks": {"0": 0, str(nx - 1): 100}}, "y": {"ticks": {"0": ytop, str(ny - 1): 0}}}
    found_by_row = {}
    texts = {}
    for r, fl in FILTERS.items():
        found = []
        for i, f in enumerate(fl):
            t, im, e = tb.call("digitize_points", {"image": str(path), "axes": axes, **f,
                                                   "series": f"r{r}_{i}", "output_csv": f"r{r}_{i}.csv"})
            texts[f"r{r}_{i}"] = t
            if e:
                print("ERR", r, f, t[:300])
                continue
            with open(od / f"r{r}_{i}.csv") as fh:
                for row in csv.DictReader(fh):
                    note = (row.get("note") or "").strip()
                    found.append((float(row["x"]), float(row["y"]), i, note if "may be" in note else ""))
        found_by_row[r] = found
    tol = 0.45 * ux
    report = []
    for r, (label, marks, layout) in enumerate(ROWS):
        yc = 90 - r * 15
        line = []
        for c, s in enumerate(SEPS):
            tr = [t for t in truth if t["row"] == r and t["sep"] == s]
            xc = 8 + c * 11.7
            det = [d for d in found_by_row.get(r, []) if abs(d[1] - yc) < 5 and abs(d[0] - xc) < 5]
            used, hit, errs = set(), 0, []
            for t in tr:
                ds = [(np.hypot(d[0] - t["x"], (d[1] - t["y"]) * ux / uy), j) for j, d in enumerate(det)
                      if j not in used]
                if ds:
                    dist, j = min(ds)
                    if dist <= tol:
                        used.add(j)
                        hit += 1
                        errs.append(dist / ux)
            extra = [d for j, d in enumerate(det) if j not in used]
            flagged = [d for d in extra if d[3]]
            false = [d for d in extra if not d[3]]
            xs = [t["x"] for t in tr]
            between = [d for d in false if min(xs) - 0.1 * ux <= d[0] <= max(xs) + 0.1 * ux]
            any_flag = any(d[3] for d in det)
            # silent merge: markers missing (or an unflagged detection between them) and nothing in the cell
            # tells the user so
            silent = 0 if any_flag else max(len(tr) - hit, len(between) and 1)
            line.append({"sep": s, "true": len(tr), "found": len(det), "hit": hit, "false": len(false),
                         "flagged": len(flagged), "silent": silent,
                         "flag_on_hit": sum(1 for j in used if det[j][3]),
                         "err_diam": round(max(errs), 3) if errs else None})
        report.append((label, line))
    # every detection of every filter against the true markers of that filter's own style (whole figure)
    STYLE = {"r0_0": ("red", "o"), "r1_0": ("blue", "o"), "r2_0": ("darkorange", "o"), "r2_1": ("purple", "s"),
             "r3_0": ("k", "s"), "r3_1": ("k", "^"), "r4_0": ("k", "o"), "r5_0": ("green", "^")}
    glob = {"detections": 0, "matched": 0, "flagged_extra": 0, "false": 0}
    for r, fl in FILTERS.items():
        for i, f in enumerate(fl):
            col, mk = STYLE[f"r{r}_{i}"]
            tr = [t for t in truth if t["color"] == col and t["marker"] == mk]
            det = [d for d in found_by_row[r] if d[2] == i]
            used = set()
            for d in det:
                glob["detections"] += 1
                ds = [(np.hypot(d[0] - t["x"], (d[1] - t["y"]) * ux / uy), j) for j, t in enumerate(tr)
                      if j not in used]
                dist, j = min(ds) if ds else (1e9, -1)
                if dist <= tol:
                    used.add(j)
                    glob["matched"] += 1
                elif d[3]:
                    glob["flagged_extra"] += 1
                else:
                    glob["false"] += 1
    iso = None
    if variant == "realistic":
        itr = [t for t in truth if t["row"] == -1]
        idet = [d for r in found_by_row.values() for d in r if d[1] > 104]
        used, hit, errs = set(), 0, []
        for t in itr:
            ds = [(np.hypot(d[0] - t["x"], (d[1] - t["y"]) * ux / uy), j) for j, d in enumerate(idet)
                  if j not in used]
            if ds:
                dist, j = min(ds)
                if dist <= tol:
                    used.add(j)
                    hit += 1
                    errs.append(dist / ux)
        # each isolated marker style is extracted by the filter(s) of its row: a series may legitimately be
        # returned by one filter only, and black circles vs squares are distinguished by shape
        extra = [d for j, d in enumerate(idet) if j not in used]
        # per filter: detections in the isolated band vs the true count of that style
        per = {}
        for r, fl in FILTERS.items():
            for i, f in enumerate(fl):
                n = sum(1 for d in found_by_row[r] if d[2] == i and d[1] > 104)
                per[f"r{r}_{i}"] = n
        iso = {"per_filter": per, "true": len(itr), "hit": hit, "false": sum(1 for d in extra if not d[3]),
               "flagged": sum(1 for d in extra if d[3]) + sum(1 for j in used if idet[j][3]),
               "err_diam_max": round(max(errs), 3) if errs else None}
    return report, iso, texts, glob


def table(rep, iso, glob=None):
    L = ["separation (marker diameters):           " + "  ".join(f"{s:>8}" for s in SEPS)]
    for label, line in rep:
        cells = []
        for c in line:
            s = f"{c['hit']}/{c['true']}"
            if c["false"]:
                s += f"+{c['false']}"
            if c["flagged"]:
                s += f"f{c['flagged']}"
            if c["flag_on_hit"]:
                s += f"({c['flag_on_hit']}f)"
            if c["silent"]:
                s += "S"
            cells.append(s)
        L.append(f"{label:42s} " + "  ".join(f"{x:>8}" for x in cells))
    L.append("max centre error of hits (d):              " + "  ".join(
        f"{max([c['err_diam'] or 0 for _, ln in rep for c in ln if c['sep'] == s]):>8.2f}" for s in SEPS))
    silent = sum(c["silent"] for _, ln in rep for c in ln)
    L.append(f"silent merges: {silent}")
    if glob:
        L.append(f"all detections of all filters: {glob['detections']} = {glob['matched']} on a true marker of the "
                 f"filtered style + {glob['flagged_extra']} flagged extras + {glob['false']} unflagged false")
    if iso:
        L.append(f"isolated markers: {iso['hit']}/{iso['true']} found, {iso['false']} false, {iso['flagged']} "
                 f"flagged, max centre error {iso['err_diam_max']} d (detections per filter, {N_ISO} true each: "
                 f"{iso['per_filter']})")
    return "\n".join(L)


if __name__ == "__main__":
    OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "overlap_bench_out").resolve()
    VARIANTS = (sys.argv[2] if len(sys.argv) > 2 else "both").split(",")
    if VARIANTS == ["both"]:
        VARIANTS = ["pairs", "realistic"]
    OUT.mkdir(parents=True, exist_ok=True)
    allrep = {}
    for v in VARIANTS:
        p, truth, ux, uy, ytop = build(v, OUT)
        rep, iso, texts, glob = run(v, p, truth, ux, uy, ytop, OUT)
        print(f"=== {v} ===")
        print(table(rep, iso, glob))
        allrep[v] = {"rows": rep, "isolated": iso, "all": glob}
    json.dump(allrep, open(OUT / "report.json", "w"), indent=1)
