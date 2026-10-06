"""Phase-region labels on phase diagrams (phase_names, phase_regions, label_place, region_labels).

Run:  .venv/bin/python -m pytest -q -p no:warnings tests/test_phase_labels.py
"""
import csv
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from calphad_tools.phase_names import plain_text as name   # the name a text shows (Al₃Ni₂)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,label", [
    ("BCC_A2", "BCC"), ("A2_BCC", "BCC"), ("FCC_A1", "FCC"), ("HCP_A3", "HCP"), ("LIQUID", "L"), ("LIQ", "L"),
    ("B2_BCC", "B2"), ("BCC_B2", "B2"), ("L12_FCC", "L1₂"), ("FCC_L12", "L1₂"), ("DHCP", "DHCP"),
    ("CBCC_A12", "αMn"), ("CUB_A13", "βMn"), ("LAVES_C14", "C14"), ("C14_LAVES", "C14"), ("LAVES_C15", "C15"),
    ("C15_LAVES", "C15"), ("LAVES_C36", "C36"), ("SIGMA", "σ"), ("SIGMA_D8B", "σ"), ("SIGMA_PHASE", "σ"),
    ("SIGMA_SGTE", "σ"), ("MU_PHASE", "μ"), ("MU", "μ"), ("MU_D85", "μ"), ("CHI", "χ"), ("CHI_A12", "χ"),
    ("DELTA", "δ"), ("GAMMA_PRIME", "γ′"), ("ETA", "η"), ("R_PHASE", "R"), ("P_PHASE", "P"),
    ("BCC_A2#2", "BCC"), ("SIGMA#3", "σ"), ("FCC_A1_1", "FCC"), ("FOOBAR_XYZ", "FOOBAR_XYZ"),
])
def test_format_phase_table(raw, label):
    from calphad_tools.phase_names import format_phase
    assert format_phase(raw) == label


def test_format_phase_formulas():
    from calphad_tools.phase_names import format_phase, split_formula, strukturbericht
    assert format_phase("FE3CR4", ["FE", "CR"]) == "Fe₃Cr₄"
    assert format_phase("FE3CR4") == "Fe₃Cr₄"                       # periodic table when no element list
    assert format_phase("AL13FE4", ["AL", "FE"]) == "Al₁₃Fe₄"
    assert format_phase("M23C6", ["CR", "FE", "C"]) == "M₂₃C₆"
    assert format_phase("M7C3") == "M₇C₃"
    assert format_phase("NI3AL", ["AL", "NI"]) == "Ni₃Al"
    assert format_phase("CR3SI", ["CR", "SI"]) == "Cr₃Si"
    assert format_phase("CRNI2", ["CR", "NI"]) == "CrNi₂"
    assert format_phase("AL3NI1", ["AL", "NI"]) == "Al₃Ni"
    # CO+SI, not C+O+SI (longest symbols first, with backtracking)
    assert split_formula("COSI", ["CO", "SI", "C", "O"]) == [("CO", ""), ("SI", "")]
    assert format_phase("COSI", ["CO", "SI", "C", "O"]) == "CoSi"
    assert split_formula("CO", ["C", "O"]) == [("C", ""), ("O", "")]
    assert format_phase("NI3TI_D024", ["NI", "TI"]) == "Ni₃Ti"         # structure suffix dropped
    assert format_phase("TI3AL_D0_19", ["TI", "AL"]) == "Ti₃Al"
    assert format_phase("AL8CR5_H", ["AL", "CR"]) == "Al₈Cr₅-HT"
    assert format_phase("AL8CR5_L", ["AL", "CR"]) == "Al₈Cr₅-LT"
    assert format_phase("LAVES", ["LA", "V", "ES"]) == "Laves"          # table before formulas
    assert strukturbericht("D019") == "D0₁₉" and strukturbericht("B2") == "B2" and strukturbericht("D8B") == "D8b"


def test_region_labels_and_overrides():
    from calphad_tools.phase_names import format_phase, region_key, region_label
    assert region_label(["BCC_A2", "BCC_A2#2"]) == "BCC+BCC"
    assert region_label(["BCC_A2#1", "BCC_A2#2"]) == "BCC+BCC"
    assert region_label(["FCC_A1", "LIQUID"]) == "L+FCC"
    assert region_label(["LIQUID", "BCC_A2"]) == "L+BCC"
    assert region_label(["SIGMA", "FCC_A1", "BCC_A2"]) == "BCC+FCC+σ"
    assert region_label(["SIGMA", "BCC_A2"], overrides={"SIGMA": "σ-phase"}) == "BCC+σ-phase"
    assert format_phase("BCC_A2", overrides={"BCC": "α"}) == "α"        # override of the default label
    assert region_key(["BCC_A2#2", "BCC_A2"]) == ("BCC_A2", "BCC_A2")


# ---------------------------------------------------------------------------------------------------------
# diagrams
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    from calphad_tools.agent.tools import ToolBox, UI

    class MyUI(UI):
        def ask(self, question, options=None):
            return ""

        def confirm(self, title, body):
            return True
    out = tmp_path_factory.mktemp("labels")
    return ToolBox(workdir=str(ROOT / "examples"), output_dir=str(out / "out"), ui=MyUI(), python_tool="off")


def call(tb, name, args):
    text, images, is_error = tb.call(name, args)
    assert not is_error, text
    return text, images


def files_from(text):
    out = {}
    for line in text.split("Files:", 1)[1].splitlines():
        m = re.match(r"\s+(\w+): (.+)$", line)
        if m:
            p = Path(m.group(2).strip())
            out[m.group(1)] = p if p.is_absolute() else ROOT / "examples" / p
    return out


def stored(png):
    from calphad_tools.figures import load_figure
    fig = load_figure(str(png))
    assert fig is not None
    return fig, fig.axes[0]


def labels_of(ax, prefix=("phase_label:", "invariant_label:")):
    return [t for t in ax.texts if (t.get_gid() or "").startswith(prefix)]


MIN_FS = 8.0                                    # default label_style.min_fontsize
REACTIONS = re.compile(r"eutect|peritect|monotect|syntect|congruent|metatect|catatect", re.I)


def is_outside(t):
    return bool((getattr(t, "_calphad_label", None) or {}).get("outside"))


def is_leader(t):
    return bool((getattr(t, "_calphad_label", None) or {}).get("leader"))


def is_below(t):
    return bool((getattr(t, "_calphad_label", None) or {}).get("below"))


def in_region(t):
    """A label written inside its own region (not outside the plot, not below its field, no leader line)."""
    return not is_outside(t) and not is_leader(t) and not is_below(t)


def label_anchor(t):
    """Where a label claims its phases are: its position, or the point it was checked at for a name outside the
    plot or a leader label."""
    lab = t._calphad_label
    return tuple(lab["at"]) if (lab.get("outside") or lab.get("leader") or lab.get("below")) \
        else tuple(t.get_position())


def check_anchors(ax):
    """Every phase label lies in a region with exactly its phase set (a name outside the plot or with a leader line:
    the phase is stable there).  Labels in their region are plain texts (no leader lines)."""
    from matplotlib.text import Annotation
    from calphad_tools.calc import get_db
    from calphad_tools.phase_regions import key_at, ordered_info
    rec = ax._calphad_diagram["recipe"]
    db = get_db(rec["tdb"])
    info = ordered_info(db, sorted(rec["elements"]) + ["VA"], rec["phases"])
    n = 0
    for t in labels_of(ax, ("phase_label:",)):
        lab = t._calphad_label
        assert not in_region(t) or not isinstance(t, Annotation), name(t)
        x, y = label_anchor(t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = key_at(db, rec, info, x, y)
        if lab["kind"] == "compound" or not in_region(t):
            assert got and lab["key"][0] in got, (name(t), got)
        else:
            assert got == tuple(lab["key"]), (name(t), (x, y), got)
        n += 1
    return n


def text_rect(t, r):
    """Corners (display px) of a label's rotated glyph box, for any ha/va (rotation_mode 'anchor': turned about
    the alignment point)."""
    from matplotlib.text import Text
    if hasattr(t, "update_positions"):
        t.update_positions(r)
    rot = t.get_rotation()
    t.set_rotation(0)
    e = Text.get_window_extent(t, r)
    t.set_rotation(rot)
    cx, cy = t.get_transform().transform(t.get_unitless_position())
    a = np.radians(rot)
    c, s_ = np.cos(a), np.sin(a)
    return np.array([(cx + (u - cx) * c - (v - cy) * s_, cy + (u - cx) * s_ + (v - cy) * c)
                     for u, v in ((e.x0, e.y0), (e.x1, e.y0), (e.x1, e.y1), (e.x0, e.y1))])


def rects_overlap(p, q):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() <= b.min() + 1e-6 or b.max() <= a.min() + 1e-6:
                return False
    return True


def point_in_rect(p, u, v, pad=0.0):
    c = p.mean(axis=0)
    ex, ey = p[1] - p[0], p[3] - p[0]
    d = np.array([u, v]) - c
    return abs(d @ ex) / np.linalg.norm(ex) <= np.linalg.norm(ex) / 2 + pad and \
        abs(d @ ey) / np.linalg.norm(ey) <= np.linalg.norm(ey) / 2 + pad


def grow(q, d):
    """A rectangle (4 corners) grown by d px on every side."""
    c = q.mean(axis=0)
    u, v = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0]), (q[3] - q[0]) / np.linalg.norm(q[3] - q[0])
    return np.array([c + ((p - c) @ u + np.sign((p - c) @ u) * d) * u + ((p - c) @ v + np.sign((p - c) @ v) * d) * v
                     for p in q])


def ternary(ax):
    return (getattr(ax, "_calphad_diagram", None) or {}).get("recipe", {}).get("kind") == "ternary"


def in_plot(ax, pts, tol=0.5):
    """Display points inside the plot area (axes box / triangle), tol in pt."""
    f = ax.figure.dpi / 72
    if not ternary(ax):
        b = ax.bbox
        return (pts[:, 0] >= b.x0 - tol * f) & (pts[:, 0] <= b.x1 + tol * f) & \
            (pts[:, 1] >= b.y0 - tol * f) & (pts[:, 1] <= b.y1 + tol * f)
    d = ax.transData.inverted().transform(pts)
    eps = tol * f / ax.bbox.width
    return (d[:, 0] >= -eps) & (d[:, 1] >= -eps) & (d.sum(axis=1) <= 1 + eps)


def check_layout(fig, ax, min_fs=MIN_FS):
    """Labels do not overlap; region / invariant labels lie inside the plot, names outside it lie outside; no label
    is smaller than min_fs; no reaction type is written anywhere; leader lines only for single phases / compounds
    of a ternary.  Returns [(text, corners)]."""
    from matplotlib.text import Annotation
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    rects = [(name(t), text_rect(t, r), t) for t in labels_of(ax)]
    for s, p, t in rects:
        assert t.get_fontsize() >= min_fs - 1e-9, (s, t.get_fontsize())
        if is_outside(t):
            assert not in_plot(ax, p, tol=-0.5).any(), s
        elif is_leader(t):
            assert len(t._calphad_label["key"]) == 1 and t.arrow_patch is not None, s   # single phases only
        elif is_below(t):
            assert in_plot(ax, p).all() and t.get_rotation() == 60 and not ternary(ax), s
        else:
            assert in_plot(ax, p).all(), s
            assert not isinstance(t, Annotation), s                     # no leader lines
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects_overlap(rects[i][1], rects[j][1]), (rects[i][0], rects[j][0])
    assert not any(REACTIONS.search(name(t)) for t in ax.texts), [name(t) for t in ax.texts]
    for t in labels_of(ax, ("invariant_label:",)):
        assert re.fullmatch(r"-?\d+ (°C|K)", name(t)), name(t)
    return [(s, p) for s, p, _ in rects]


def check_inside(fig, ax):
    """The rotated box of every region label lies inside its region: inside a polygon of its own phases (two-phase
    fields, tie-triangles), and inside no polygon of other phases.  Returns the number of labels checked."""
    from matplotlib.path import Path as MPath
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    polys = [(tuple(p["key"]), MPath(ax.transData.transform(np.asarray(p["xy"], float))))
             for p in ax._calphad_diagram["regions"]["polygons"]]
    n = 0
    for t in labels_of(ax, ("phase_label:",)):
        if t._calphad_label["kind"] != "region" or not in_region(t):
            continue
        q = text_rect(t, r)
        pts = np.vstack([q, q.mean(axis=0), (q + np.roll(q, 1, axis=0)) / 2])
        key = tuple(t._calphad_label["key"])
        own = [P for k, P in polys if k == key]
        for pt in pts:
            if own:
                assert any(P.contains_point(pt) for P in own), (name(t), pt)
            assert not any(P.contains_point(pt) for k, P in polys if k != key), (name(t), pt)
        n += 1
    return n


def check_clearance(fig, ax, margin=2.5):
    """Every phase label in its region keeps >= margin - 0.5 pt from every drawn phase boundary (the boundary,
    invariant and tie-triangle lines actually drawn; not the region polygons, whose closing edges can be invisible
    tie-lines where pycalphad stopped a trace part-way).  (Invariant temperatures hug their own line:
    check_invariants.)  Returns the smallest clearance found (pt)."""
    from scipy.spatial import cKDTree
    from matplotlib.path import Path as MPath
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = fig.dpi / 72
    segs = []
    for ln in ax.lines:                 # boundaries are >= 1 pt; thin guide / leader lines of labels are not
        if not ln.get_visible() or ln.get_linewidth() < 0.9:
            continue
        q = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float))
        for a, b in zip(q[:-1], q[1:]):
            n = max(2, int(np.hypot(*(b - a)) / (0.25 * f)) + 1)
            segs.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    pts = np.vstack(segs)
    pts = pts[np.isfinite(pts).all(axis=1)]
    tree = cKDTree(pts)
    worst = np.inf
    for t in labels_of(ax, ("phase_label:",)):
        if not in_region(t):
            continue
        q = text_rect(t, r)
        ring = np.vstack([a + (b - a) * np.linspace(0, 1, 60)[:, None] for a, b in zip(q, np.roll(q, -1, axis=0))])
        assert not MPath(q).contains_points(pts).any(), name(t)
        d = tree.query(ring)[0].min() / f
        assert d >= margin - 0.5, (name(t), d)
        worst = min(worst, d)
    return worst


def check_invariants(fig, ax, min_fs=MIN_FS):
    """Invariant temperatures sit right above or right below their own line, over it, at >= min_fs; no invariant
    line has more than one temperature (also when the mapping reported it twice)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = fig.dpi / 72
    invs = ax._calphad_diagram["regions"]["invariants"]
    labs = labels_of(ax, ("invariant_label:",))
    lines_of = []
    for t in labs:                    # the reported invariants (within 0.5 K) whose line runs under this label
        x = t.get_position()[0]
        lines_of.append({i for i, inv in enumerate(invs) if abs(inv["T_K"] - t._calphad_label["T_K"]) < 0.5
                         and min(inv["x0"], inv["x1"]) - 1e-9 <= x <= max(inv["x0"], inv["x1"]) + 1e-9})
        assert lines_of[-1], name(t)
    for a in range(len(labs)):
        for b in range(a + 1, len(labs)):
            assert not lines_of[a] & lines_of[b], ("one invariant line, two temperatures", name(labs[a]))
    out = []
    for t in labs:
        inv = min(invs, key=lambda i: abs(i["T_K"] - t._calphad_label["T_K"]))
        assert abs(inv["T_K"] - t._calphad_label["T_K"]) < 1e-6
        q = text_rect(t, r)
        (xa, y), (xb, _) = ax.transData.transform([(inv["x0"], inv["y"]), (inv["x1"], inv["y"])])
        assert min(xa, xb) - f <= q[:, 0].min() and q[:, 0].max() <= max(xa, xb) + f, name(t)
        above, below = q[:, 1].min() - y, y - q[:, 1].max()
        assert max(above, below) > 0, name(t)                    # not across the line
        assert 0 < max(above, below) <= 4.0 * f, (name(t), above / f, below / f)   # right next to it
        assert t.get_fontsize() >= min_fs
        out.append((name(t), "above" if above > 0 else "below"))
    return out


def tick_state(ax, which="x"):
    """(visible tick texts, hidden tick texts) as drawn; hidden ones must touch a name outside the plot, visible
    ones must not."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    outs = [text_rect(t, r) for t in labels_of(ax, ("phase_label:",)) if is_outside(t)]
    axis = getattr(ax, which + "axis")
    locs = axis.get_majorticklocs()
    lo, hi = sorted(axis.get_view_interval())
    shown, hidden = [], []
    for tick, loc in zip(axis.get_major_ticks(len(locs)), locs):
        lab = tick.label1
        if not (lo - 1e-9 <= loc <= hi + 1e-9) or not name(lab):
            continue
        if lab.get_visible():
            assert not any(rects_overlap(text_rect(lab, r), q) for q in outs), name(lab)
            shown.append(name(lab))
        else:
            lab.set_visible(True)                                     # only colliding ones (1 pt margin)
            assert any(rects_overlap(grow(text_rect(lab, r), 1.5 * fig.dpi / 72), q) for q in outs), name(lab)
            lab.set_visible(False)
            hidden.append(name(lab))
    return shown, hidden


def omitted_line(text):
    """The one-line report of labels left out for lack of room: (count, names)."""
    m = re.search(r"^omitted (\d+) \(no room for [\d.]+ pt\): (.+)$", text.replace("; ", "\n"), re.M)
    return (int(m.group(1)), [s.strip() for s in m.group(2).split(", ")]) if m else (0, [])


def test_binary_crni(tb):
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                                  "tmax": 2200, "temperature_unit": "C", "label_regions": True,
                                                  "output_prefix": "crni_labels"})
    f = files_from(text)
    for k in ("png", "svg", "pdf"):
        assert f[k].exists() and f[k].stat().st_size > 5000, k
    assert "phase labels" in text
    fig, ax = stored(images[0])
    texts = {name(t) for t in labels_of(ax)}
    assert {"L", "BCC", "FCC", "L+BCC", "BCC+FCC"} <= texts, texts
    # the thin L+FCC field has no room for an 8 pt label: left out, and said so in one line
    n, names = omitted_line(text)
    assert "L+FCC" not in texts and "L+FCC" in names and n == len(names)
    # eutectic temperature, in °C, as in the invariants CSV; no reaction type
    with open(f["invariants_csv"], newline="") as fh:
        T_C = float(next(r for r in csv.DictReader(fh) if r["type"] == "eutectic")["T_C"])
    inv = [t for t in labels_of(ax, ("invariant_label:",))]
    assert [name(t) for t in inv] == [f"{T_C:.0f} °C"] and inv[0].get_gid() == f"invariant_label:{T_C:.0f}C"
    assert check_invariants(fig, ax)
    assert check_anchors(ax) >= 5
    assert check_inside(fig, ax) >= 5
    check_layout(fig, ax)
    pickle.dumps(fig)                                                 # figures stay picklable
    # vector output keeps the text as text
    svg = f["svg"].read_text()
    for s in ("L+BCC", "BCC+FCC", f"{T_C:.0f} °C"):
        assert re.search(r"<text[^>]*>" + re.escape(s) + "</text>", svg), s
    assert 'id="phase_label:BCC_A2.FCC_A1"' in svg and "DejaVu Sans" in svg
    assert b"/FontFile2" in f["pdf"].read_bytes()                     # TrueType (Type 42) font embedded
    assert "BCC+FCC" in _pdf_text(f["pdf"]) and f"{T_C:.0f}" in _pdf_text(f["pdf"])
    # a larger minimum size leaves out more labels, and every label drawn respects it
    text2, images2 = call(tb, "edit_figure", {"image": images[0], "operations": [
        {"op": "label_phase_regions", "min_fontsize": 9.5, "fontsize": 9.5, "invariant_fontsize": 9.5}],
        "output": "crni_big.png"})
    fig2, ax2 = stored(images2[0])
    check_layout(fig2, ax2, min_fs=9.5)
    n2, names2 = omitted_line(text2)
    assert n2 >= n and set(names) <= set(names2) and "9.5 pt" in text2
    assert not ({name(t) for t in labels_of(ax2)} & set(names2))


def check_corner_names(fig, ax, want):
    """want {name: corner index (0 bottom-left, 1 bottom-right, 2 top)}: each name is at its corner, outside the
    triangle, square to the nearest side (bottom-left -30° like the left-edge names, bottom-right and top +30°),
    never on an edge; it covers no tick label, no bold corner name, no axis title and no other
    label, and hides no tick label (the left axis's 0.0 stays).  Returns {name: text}."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = fig.dpi / 72
    c_px = ax.transData.transform([(0, 0), (1, 0), (0, 1)])
    got = {}
    for nm, k in want.items():
        hits = [t for t in labels_of(ax, ("phase_label:",)) if name(t) == nm and t._calphad_label.get("corner") == k]
        assert len(hits) == 1, (nm, k, [(name(t), t._calphad_label) for t in labels_of(ax)])
        t = hits[0]
        assert t._calphad_label["outside"] == "corner" and t.get_bbox_patch() is None
        assert t.get_rotation() == (330.0 if k == 0 else 30.0), (nm, k, t.get_rotation())   # -30° reads as 330
        q = text_rect(t, r)
        assert not in_plot(ax, q, tol=-0.5).any(), nm                   # outside the triangle
        assert np.hypot(*(q - c_px[k]).T).min() / f < 40, nm            # right next to its corner
        if k == 0:                                                      # above the left axis's 0.0, on the left
            assert q[:, 1].mean() > c_px[0][1] and q[:, 0].mean() < c_px[0][0], nm
        others = [x for x in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + list(ax.texts)
                  + [ax.xaxis.label, ax.yaxis.label] if isinstance(x, Text) and x is not t and x.get_visible()
                  and x.get_text().strip()]
        for x in others:
            assert not rects_overlap(q, text_rect(x, r)), (nm, x.get_text())
        got[nm] = t
    ylabels = {x.get_text(): x for x in ax.get_yticklabels()}
    assert ylabels["0.0"].get_visible()                                  # the 0.0 tick label is kept
    return got


def svg_has(svg, s):
    """The SVG has a <text> showing s (subscript digits as Inkscape subscript tspans)."""
    from calphad_tools.phase_names import svg_subscripts
    body = svg_subscripts(f"<text>{s}</text>")[len("<text>"):-len("</text>")]
    return re.search(r"<text[^>]*>" + re.escape(body) + "</text>", svg) is not None


def check_svg(svg):
    """Valid XML; no Unicode subscript digit left; every label one <text> whose only tspans are Inkscape subscripts
    (font-size:65%;baseline-shift:sub) - no per-glyph tspans.  Returns the number of labels."""
    import xml.etree.ElementTree as ET
    assert not re.search("[₀-₉]", svg)
    root = ET.fromstring(svg.encode("utf-8"))
    ns = "{http://www.w3.org/2000/svg}"
    n = 0
    for g in root.iter(ns + "g"):
        if not (g.get("id") or "").startswith(("phase_label:", "invariant_label:")):
            continue
        texts = [e for e in g.iter(ns + "text")]
        assert len(texts) == 1, g.get("id")
        for sp in texts[0].iter(ns + "tspan"):
            assert sp.get("style") == "font-size:65%;baseline-shift:sub" and sp.text.isdigit(), g.get("id")
            assert sp.get("x") is None and sp.get("y") is None
        n += 1
    return n


def _pdf_text(path):
    try:
        from pypdf import PdfReader
        return "".join(p.extract_text() for p in PdfReader(str(path)).pages)
    except ImportError:
        return subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True).stdout


def test_binary_fecr_sigma_and_gap(tb):
    # invariant_type is accepted but ignored: reaction types are never written
    text, images = call(tb, "map_phase_diagram", {"tdb": "CrFe-82Her.tdb", "elements": ["FE", "CR"], "tmin": 300,
                                                  "tmax": 1000, "temperature_unit": "C", "label_regions": True,
                                                  "label_style": {"invariant_type": True},
                                                  "output_prefix": "fecr_labels"})
    assert "invariant_type ignored" in text
    fig, ax = stored(images[0])
    texts = [name(t) for t in labels_of(ax)]
    assert "σ" in texts and "BCC+BCC" in texts and "FCC" in texts and "BCC" in texts
    assert texts.count("BCC+σ") == 2                                  # both sides of the σ field
    assert "431 °C" in texts and not any("eutectoid" in s for s in texts), texts
    assert not any("#" in s or "₁" in s for s in texts)               # gap is BCC+BCC, never BCC#1 / BCC₁
    assert check_invariants(fig, ax)
    assert check_anchors(ax) >= 6
    assert check_inside(fig, ax) >= 6
    check_layout(fig, ax)
    svg = files_from(text)["svg"].read_text()
    assert re.search(r"<text[^>]*>σ</text>", svg) and "eutectoid" not in svg
    assert "σ" in _pdf_text(files_from(text)["pdf"])


def test_binary_alni_compounds_outside(tb, tmp_path):
    """Line compounds Al₃Ni and Al₃Ni₅: named below the x axis at their composition (60°), never inside; the tick
    labels they cover are hidden, the others kept; show_code draws the same."""
    text, images = call(tb, "map_phase_diagram", {"tdb": "AlCoCrNi_Liu2016.tdb", "elements": ["AL", "NI"],
                                                  "tmin": 400, "tmax": 1800, "temperature_unit": "C",
                                                  "output_prefix": "alni_labels"})
    assert "stoichiometric phases named outside the plot: " in text
    fig, ax = stored(images[0])
    comps = {c["key"][0]: c for c in ax._calphad_diagram["regions"]["compounds"]}
    assert {"AL3NI1", "AL3NI5"} <= set(comps), comps
    outs = {name(t): t for t in labels_of(ax, ("phase_label:",)) if is_outside(t)}
    assert {s_ for s_, t in outs.items() if t._calphad_label["kind"] == "compound"} == {"Al₃Ni", "Al₃Ni₅"}, outs
    inside = [name(t) for t in labels_of(ax) if not is_outside(t)]
    assert "Al₃Ni" not in inside and "Al₃Ni₅" not in inside          # never in the figure
    # Al3Ni2 has a range (a region): inside if it fits, else below the axis at its mid composition
    assert ("Al₃Ni₂" in inside) != ("Al₃Ni₂" in outs)
    if "Al₃Ni₂" in outs:
        t = outs["Al₃Ni₂"]
        assert t._calphad_label["kind"] == "region" and t.get_rotation() == 60 and 0.35 < t.xy[0] < 0.42
        assert "too small inside, named at the edge instead: Al₃Ni₂" in text
    # never a terminal phase (the end-member solid solutions, the liquid) outside a binary
    assert not {tuple(t._calphad_label["key"]) for t in outs.values()} & {("FCC_A1",), ("LIQUID",)}
    assert check_clearance(fig, ax) >= 2.0
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = fig.dpi / 72
    for nm, key in (("Al₃Ni", "AL3NI1"), ("Al₃Ni₅", "AL3NI5")):
        t = outs[nm]
        assert t.get_rotation() == 60 and t.get_ha() == "right"
        assert abs(t.xy[0] - comps[key]["x"]) < 1e-9 and t.xycoords == ("data", "axes fraction") and t.xy[1] == 0
        assert t.arrow_patch is None                                  # right at its composition: no guide
        q = text_rect(t, r)
        assert q[:, 1].max() < ax.bbox.y0                             # below the x axis
        xc = ax.transData.transform((comps[key]["x"], 0))[0]
        top = q[np.argmax(q[:, 1])]
        assert abs(top[0] - xc) < 10 * f and ax.bbox.y0 - top[1] < 6 * f  # its upper end at the compound
    shown, hidden = tick_state(ax, "x")
    assert hidden and len(shown) >= 3, (shown, hidden)
    assert check_anchors(ax) >= 10
    assert check_inside(fig, ax) >= 10
    assert check_invariants(fig, ax)
    rects = check_layout(fig, ax)
    xl = text_rect(ax.xaxis.label, r)                                 # the axis title moved clear of the names
    assert not any(rects_overlap(xl, p) for s, p in rects if s in outs)
    svg = files_from(text)["svg"].read_text()
    assert svg_has(svg, "Al₃Ni₅") and 'id="phase_label:AL3NI5"' in svg
    assert check_svg(svg) >= 10
    # PNG / PDF: real subscripts (mathtext); the plain name stays on the text
    t = outs["Al₃Ni₅"]
    assert t.get_text() == r"Al$_\mathregular{3}$Ni$_\mathregular{5}$" and name(t) == "Al₃Ni₅"
    assert "Al3Ni" in re.sub(r"\s", "", _pdf_text(files_from(text)["pdf"]))
    for h in hidden:
        assert not re.search(r"<text[^>]*>" + re.escape(h) + "</text>", svg), h
    # show_code: the same labels at the same places, the same tick labels hidden
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, images[0])
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script and "TickLabelHider" in script
    d = tmp_path / "alni_code"
    d.mkdir()
    dump = '''
import json as _json
fig.canvas.draw()
_ax = fig.axes[0]
_json.dump({"ticks": [t.get_text() for t in _ax.get_xticklabels() if t.get_visible() and t.get_text()],
            "labels": {t.get_gid(): [getattr(t, "_calphad_plain", None) or t.get_text(), t.get_rotation()]
                       for t in _ax.texts
                       if (t.get_gid() or "").startswith(("phase_label:", "invariant_label:"))}}, open("art.json", "w"))
'''
    (d / "gen.py").write_text(script + dump)
    run = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True,
                         timeout=900, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-3000:]
    import json
    art = json.loads((d / "art.json").read_text())
    want = {t.get_gid(): [name(t), t.get_rotation()] for t in labels_of(ax)}
    assert art["labels"] == want, (art["labels"], want)
    assert art["ticks"] == shown
    got = {}
    for line in run.stdout.splitlines():
        m = re.match(r"label: (.+) at \(([-\d.e]+), ([-\d.e]+)\) gid (\S+)", line)
        if m:
            got[m.group(4)] = (float(m.group(2)), float(m.group(3)))
    for t in labels_of(ax):
        x, y = label_anchor(t)
        assert abs(got[t.get_gid()][0] - x) < 0.03 and abs(got[t.get_gid()][1] - y) < 40, (t.get_gid(), got, x, y)
    # orientation: Al on the right -> the names follow their compositions to the mirrored side
    _, imgs_m = call(tb, "map_phase_diagram", {"tdb": "AlCoCrNi_Liu2016.tdb", "elements": ["AL", "NI"], "right": "AL",
                                               "tmin": 400, "tmax": 1800, "temperature_unit": "C", "formats": [],
                                               "output_prefix": "nial_labels"})
    figm, axm = stored(imgs_m[0])
    outm = {name(t): t for t in labels_of(axm, ("phase_label:",)) if is_outside(t)}
    assert set(outm) == set(outs)
    assert abs(outm["Al₃Ni"].xy[0] - 0.75) < 1e-6 and abs(outm["Al₃Ni₅"].xy[0] - 0.375) < 1e-6
    assert outm["Al₃Ni"].get_rotation() == 60
    check_layout(figm, axm)
    assert tick_state(axm, "x")[1]


LABEL_DUMP = '''
import json as _json
fig.canvas.draw()
_ax = fig.axes[0]
_json.dump({"labels": {t.get_gid(): [getattr(t, "_calphad_plain", None) or t.get_text(),
                                     t._calphad_label.get("outside"), bool(t._calphad_label.get("leader"))]
                       for t in _ax.texts if (t.get_gid() or "").startswith(("phase_label:", "invariant_label:"))},
            "ticks": [t.get_text() for t in _ax.get_xticklabels() if t.get_visible() and t.get_text()]},
           open("art.json", "w"))
'''


def test_ternary_tie_triangles_and_compounds(tb, tmp_path):
    """Cr-Al-Ni at 750 °C: tie-triangles labelled inside (turned along skinny ones), those too small for 8 pt left
    out and reported in one line; the Al-Cr compounds named below the bottom edge (60°, spread apart where they
    crowd), Al₃Ni beyond the Al-Ni edge (30°); a near-pure BCC corner is not taken for a compound."""
    text, images = call(tb, "map_phase_diagram", {"tdb": "AlCoCrNi_Liu2016.tdb", "elements": ["CR", "AL", "NI"],
                                                  "temperature": 750, "temperature_unit": "C",
                                                  "output_prefix": "cralni_labels"})
    fig, ax = stored(images[0])
    import json
    summary = json.loads(files_from(text)["json"].read_text())
    tris = summary["invariants"]
    assert len(tris) >= 5
    from calphad_tools.phase_names import region_label, region_key
    labs = labels_of(ax, ("phase_label:",))
    n_omitted, omitted = omitted_line(text)
    assert n_omitted == len(omitted) >= 3 and len(text.split("omitted")[1].splitlines()[0]) < 2000
    n_in = 0
    for key in {region_key(inv["phases"]) for inv in tris}:
        nm = region_label(key, ["AL", "CR", "NI"])
        hit = [t for t in labs if tuple(t._calphad_label["key"]) == key]
        if not hit:                                                   # too small for 8 pt: left out and reported
            assert nm in omitted, (key, text)
            continue
        assert len(hit) == 1 and name(hit[0]) == nm and nm.count("+") == 2
        x, y = hit[0].get_position()                                  # inside a triangle of these phases
        lams = []
        for inv in tris:
            if region_key(inv["phases"]) == key:
                P = np.array([[inv["compositions"][n][e] for e in ("AL", "NI")] for n in inv["phases"]])
                lams.append(np.linalg.solve(np.vstack([P.T, np.ones(3)]), np.array([x, y, 1.0])))
        assert any((lam > -1e-6).all() for lam in lams), (nm, lams)
        n_in += 1
    assert n_in >= 2
    assert any(abs(t.get_rotation()) > 1 for t in labs if not is_outside(t))     # turned in a skinny triangle
    outs = {name(t): t for t in labs if is_outside(t)}
    bottom = {s_ for s_, t in outs.items() if t._calphad_label["outside"] == "bottom"}
    assert {"AlCr₂", "Al₈Cr₅-LT", "Al₉Cr₄-LT", "Al₄Cr", "Al₁₁Cr₂", "Al₁₃Cr₂"} <= bottom, outs
    assert all(outs[s_].get_rotation() == 60 for s_ in bottom)
    assert outs["Al₃Ni"]._calphad_label["outside"] == "right" and outs["Al₃Ni"].get_rotation() == 30
    # Al3Ni2 (a range, its field a sliver along the Al-Ni edge): named beyond that edge, as a region
    assert outs["Al₃Ni₂"]._calphad_label == {**outs["Al₃Ni₂"]._calphad_label, "kind": "region", "outside": "right"}
    assert outs["Al₃Ni₂"].get_rotation() == 30 and "Al₃Ni₂" not in omitted
    assert "named at the edge instead: " in text and "Al₃Ni₂" in text.split("named at the edge instead: ")[1]
    # phases stable for a pure element, too small inside: at their corner, at 30° - BCC at Cr (bottom-left, above
    # the left axis's 0.0, which stays), L at Al (bottom-right) - never on an edge
    corner = check_corner_names(fig, ax, {"BCC": 0, "L": 1})
    assert all(t._calphad_label["kind"] == "region" for t in corner.values())
    assert not any(name(t) in ("BCC", "L") and t._calphad_label["outside"] in ("bottom", "left", "right")
                   for t in labs)
    assert "named at its corner: " in text and "BCC (Cr)" in text and "L (Al)" in text
    assert not any(s_.startswith("Al") for s_ in (name(t) for t in labs if in_region(t) and "+" not in name(t)))
    assert not omitted or all("+" in s_ for s_ in omitted)           # only multi-phase fields got nowhere
    assert check_anchors(ax) >= 12
    assert check_inside(fig, ax) >= 5
    assert check_clearance(fig, ax) >= 2.0
    check_layout(fig, ax)                                             # spread apart: no two names overlap
    shown, hidden = tick_state(ax, "x")
    assert hidden and len(shown) >= 3 and "0.0" in shown and "1.0" in shown, (shown, hidden)
    # show_code: the same labels, at the same edges
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, images[0])
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    d = tmp_path / "cralni_code"
    d.mkdir()
    (d / "gen.py").write_text(script + LABEL_DUMP)
    run = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True,
                         timeout=1200, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-3000:]
    import json as _j
    art = _j.loads((d / "art.json").read_text())
    want = {t.get_gid(): [getattr(t, "_calphad_plain", None) or t.get_text(), t._calphad_label.get("outside"), bool(t._calphad_label.get("leader"))]
            for t in labels_of(ax)}
    assert art["labels"] == want, (art["labels"], want)


@pytest.fixture(scope="module")
def ternary_compounds(tb):
    out = {}
    for nm, els in (("conv", ["CO", "NI", "V"]), ("cvn", ["CO", "V", "NI"])):
        out[nm] = call(tb, "map_phase_diagram", {"tdb": "CoCrFeNiV_Choi2019.tdb", "elements": els,
                                                   "temperature": 900, "temperature_unit": "C", "formats": [],
                                                   "output_prefix": f"{nm}_labels"})
    return out


def _edge_of(ax, xy, tol=1e-6):
    x, y = xy
    return "bottom" if abs(y) < tol else "left" if abs(x) < tol else "right" if abs(x + y - 1) < tol else None


EDGE_ROT = {"bottom": 60.0, "left": -30.0, "right": 30.0}


@pytest.mark.parametrize("case,want", [("conv", {"M₃V": ("right",), "CoV₃": ("left",)}),
                                       ("cvn", {"M₃V": ("bottom", "right"), "CoV₃": ("bottom",)})])
def test_ternary_compounds_outside(ternary_compounds, case, want):
    """Line compound M₃V and point compound CoV₃ (Co-Ni-V, 900 °C): named just outside the nearest triangle edge
    (bottom 60°, slanted edges ±30°, square to the edge), never inside; covered tick labels hidden.  M₃V runs
    from edge to edge in the Co-V-Ni orientation: either end will do."""
    text, images = ternary_compounds[case]
    fig, ax = stored(images[0])
    outs = {name(t): t for t in labels_of(ax, ("phase_label:",)) if is_outside(t)}
    assert {s_ for s_, t in outs.items() if t._calphad_label["kind"] == "compound"} == set(want), (set(outs), text)
    assert not set(want) & {name(t) for t in labels_of(ax) if not is_outside(t)}
    # others outside are single-phase fields too small inside: BCC at the V corner (top / bottom-right), FCC
    assert all(len(t._calphad_label["key"]) == 1 for t in outs.values())
    check_corner_names(fig, ax, {"BCC": 2 if case == "conv" else 1})
    assert "named at its corner: BCC (V)" in text
    if case == "conv":
        # each region on its own: the thin FCC strip on the Ni-V edge is named, although the big FCC field is too
        fcc = [t for t in labels_of(ax, ("phase_label:",)) if name(t) == "FCC"]
        strip = [t for t in fcc if is_outside(t)]
        assert len(strip) == 1 and len(fcc) >= 2 and strip[0]._calphad_label["kind"] == "region", fcc
        assert strip[0]._calphad_label["outside"] == "right" and strip[0].get_rotation() == 30
        assert 0.26 < strip[0].xy[1] < 0.36 and abs(sum(strip[0].xy) - 1) < 1e-9, strip[0].xy   # X(V) of the strip
        assert "named at the edge instead: " in text and "FCC" in text.split("named at the edge instead: ")[1]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = fig.dpi / 72
    for nm, edges in want.items():
        t = outs[nm]
        edge = t._calphad_label["outside"]
        assert edge in edges and t.get_rotation() % 360 == EDGE_ROT[edge] % 360, (nm, edge, t.get_rotation())
        assert _edge_of(ax, t.xy) == edge, (nm, t.xy)               # anchored on that edge
        q = text_rect(t, r)
        d = ax.transData.inverted().transform(q)
        side = {"bottom": d[:, 1], "left": d[:, 0], "right": 1 - d.sum(axis=1)}[edge]
        assert (side < 0).all(), (nm, d)                            # entirely beyond that edge
        near = np.hypot(*(q - ax.transData.transform(t.xy)).T).min() / f    # and right next to it (pt)
        assert near < 12, (nm, near)
    if case == "cvn":                                                 # bottom edge: along the line of constant x
        comps = {c["key"][0]: c for c in ax._calphad_diagram["regions"]["compounds"]}
        assert abs(outs["CoV₃"].xy[0] - comps["COV3"]["x"]) < 1e-6
    assert check_anchors(ax) >= 5
    assert check_inside(fig, ax) >= 4
    check_layout(fig, ax)
    hidden = []
    for which in ("x", "y"):
        shown, h = tick_state(ax, which)
        assert len(shown) >= 8
        hidden += h
    assert hidden, "a name outside the triangle covers at least one tick label"


def test_obstacles_edit_ops_and_show_code(tb):
    # a diagram without labels, then data points where the L+BCC label would go, then labels on the stored figure
    _, images = call(tb, "map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 800,
                                               "tmax": 2200, "temperature_unit": "C", "output_prefix": "crni_plain",
                                               "label_regions": False})
    ref, imgs_ref = call(tb, "edit_figure", {"image": images[0], "operations": [{"op": "label_phase_regions"}],
                                             "output": "crni_ref.png"})
    assert Path(imgs_ref[0]).with_suffix(".svg").exists() and Path(imgs_ref[0]).with_suffix(".pdf").exists()
    fig0, ax0 = stored(imgs_ref[0])
    lb = next(t for t in labels_of(ax0) if name(t) == "L+BCC")
    x0, y0 = lb.get_position()
    pts_x = [x0 - 0.03, x0, x0 + 0.03, x0 - 0.015, x0 + 0.015]
    pts_y = [y0, y0 + 8, y0, y0 - 12, y0 - 12]
    ops = [{"op": "scatter", "x": pts_x, "y": pts_y, "color": "k", "s": 25, "label": "measured"},
           {"op": "label_phase_regions"}]
    text, imgs = call(tb, "edit_figure", {"image": images[0], "operations": ops, "output": "crni_data.png"})
    assert "phase labels" in text
    fig, ax = stored(imgs[0])
    rects = check_layout(fig, ax)
    sc = [c for c in ax.collections if c.get_label() == "measured"][0]
    px = sc.get_offset_transform().transform(sc.get_offsets())
    rad = np.sqrt(sc.get_sizes()[0]) / 2 * fig.dpi / 72
    for s, p in rects:
        for (u, v) in px:
            assert not point_in_rect(p, u, v, rad), (s, u, v)
    lb = [t for t in labels_of(ax) if name(t) == "L+BCC"]
    if lb:
        assert np.hypot(*(np.subtract(lb[0].get_position(), (x0, y0)) / [1.0, 1400.0])) > 0.01   # moved
    else:
        assert "L+BCC" in omitted_line(text)[1]                       # or left out (no room at 8 pt)
    check_anchors(ax)
    # edit labels by id / text
    lab = next(t for t in labels_of(ax) if t.get_gid() == "phase_label:LIQUID")
    before = ax.transData.transform(lab.get_position())
    ops = [{"op": "move_label", "id": "phase_label:LIQUID", "dx": 20, "dy": -10},
           {"op": "delete_label", "text": "BCC+FCC"},
           {"op": "relabel", "id": "FCC_A1", "new_text": "γ (FCC)", "color": "navy"}]
    text, imgs2 = call(tb, "edit_figure", {"image": imgs[0], "operations": ops, "output": "crni_edited.png"})
    fig2, ax2 = stored(imgs2[0])
    lab2 = next(t for t in labels_of(ax2) if t.get_gid() == "phase_label:LIQUID")
    after = ax2.transData.transform(lab2.get_position())
    assert np.allclose((after - before) * 72 / fig2.dpi, (20, -10), atol=0.05)
    assert "BCC+FCC" not in {name(t) for t in labels_of(ax2)}
    assert next(t for t in labels_of(ax2) if t.get_gid() == "phase_label:FCC_A1").get_text() == "γ (FCC)"
    _, _, err = tb.call("edit_figure", {"image": imgs[0], "operations": [{"op": "delete_label", "id": "NOPE"}]})
    assert err
    # standalone code reproduces the labels (same strings, nearby positions) and does not import the toolkit
    code_text, _ = call(tb, "show_code", {"target": imgs2[0], "format": "script"})
    py = Path(re.search(r"saved (\S+\.py)", code_text).group(1))
    py = py if py.is_absolute() else ROOT / "examples" / py
    src = py.read_text()
    assert "calphad_tools" not in src
    run = subprocess.run([sys.executable, str(py)], cwd=str(py.parent), capture_output=True, text=True, timeout=900,
                         env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-3000:]
    got = {}
    for line in run.stdout.splitlines():
        m = re.match(r"label: (.+) at \(([-\d.e]+), ([-\d.e]+)\) gid (\S+)", line)
        if m:
            got[m.group(4)] = (m.group(1), float(m.group(2)), float(m.group(3)))
    want = {t.get_gid(): (name(t), *t.get_position()) for t in labels_of(ax)}
    assert set(got) == set(want), (set(got) ^ set(want))
    for gid, (s, x, y) in want.items():
        assert got[gid][0] == s
        assert abs(got[gid][1] - x) < 0.08 and abs(got[gid][2] - y) < 100, (gid, got[gid], (x, y))
    out = py.parent / "standalone_outputs"
    svg = next(out.glob("crni_edited*.svg")).read_text()
    assert re.search(r"<text[^>]*>γ \(FCC\)</text>", svg) and "phase_label:LIQUID" in svg


def test_plot_diagram_with_data_labels(tb, tmp_path):
    from calphad_tools import expdata
    d = tmp_path / "ds"
    zpf, _ = expdata.pop_to_zpf(str(ROOT / "examples/expdata/CrNi_boundaries.POP"), ["CR", "NI"],
                                tdb_phases=["LIQUID", "FCC_A1", "BCC_A2"])
    for i, ds in enumerate(zpf):
        expdata.write_dataset(ds, str(d / f"z{i}.json"))
    text, imgs = call(tb, "plot_diagram_with_data", {"elements": ["CR", "NI"], "datasets_dir": str(d),
                                                     "tdb": "CrNi-06Hal.tdb", "tmin": 800, "tmax": 2000,
                                                     "temperature_unit": "C", "label_regions": True,
                                                     "phase_labels": {"LIQUID": "Liquid"},
                                                     "output_png": "pdd_labels.png"})
    assert "phase labels" in text and Path(imgs[0]).with_suffix(".svg").exists()
    fig, ax = stored(imgs[0])
    texts = {name(t) for t in labels_of(ax)}
    assert {"Liquid", "Liquid+BCC", "BCC+FCC"} <= texts, texts
    rects = check_layout(fig, ax)
    n = 0
    for ln in ax.get_lines():                          # no label covers a data marker
        if ln.get_marker() in (None, "None", "", " "):
            continue
        px = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
        rad = ln.get_markersize() / 2 * fig.dpi / 72
        for s, p in rects:
            for u, v in px:
                assert not point_in_rect(p, u, v, rad), (s, u, v)
                n += 1
    assert n > 0
    check_anchors(ax)
    check_invariants(fig, ax)


def test_label_errors(tb):
    _, _, err = tb.call("map_phase_diagram", {"tdb": "CrNi-06Hal.tdb", "elements": ["CR", "NI"], "tmin": 1000,
                                              "tmax": 1200, "label_regions": True, "label_style": {"size": 3}})
    assert err
    _, images = call(tb, "make_plot", {"panels": [{"operations": [{"op": "line", "x": [0, 1], "y": [0, 1]}]}],
                                       "output": "plain"})
    text, _, err = tb.call("edit_figure", {"image": images[0], "operations": [{"op": "label_phase_regions"}]})
    assert err and "recipe" in text


def test_label_diagram_synthetic(tmp_path):
    """Placement on a hand-made diagram: a two-phase box, a line compound, an invariant line, data points and a
    sliver too thin for any label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calphad_tools.label_place import label_diagram, save_vector, find_labels, move_label, clear_labels
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Mole fraction Cr")
    ax.plot([0.1, 0.5, 0.5, 0.1, 0.1], [100, 100, 500, 500, 100], color="C0")
    ax.plot([0.6, 0.6], [0, 800], color="C1")
    ax.plot([0.1, 0.9], [600, 600], color="red")
    ax.scatter([0.3, 0.28, 0.32], [300, 280, 320], s=30, color="k")      # data right where the label would go
    regions = {"polygons": [{"key": ("FE3CR4", "LIQUID"), "xy": [[0.1, 100], [0.5, 100], [0.5, 500], [0.1, 500]]},
                            {"key": ("BCC_A2", "SIGMA"), "xy": [[0.85, 100], [0.86, 100], [0.86, 500], [0.85, 500]]}],
               "compounds": [{"key": ("FE3CR4",), "x": 0.6, "y0": 0.0, "y1": 800.0},
                             {"key": ("FECR",), "x": 0.62, "y0": 0.0, "y1": 500.0}],
               "invariants": [{"key": ("A", "B", "C"), "T_K": 873.15, "y": 600.0, "x0": 0.1, "x1": 0.9,
                               "type": "eutectic"},           # the same line reported twice by a mapping:
                              {"key": ("A", "B", "C"), "T_K": 873.16, "y": 600.01, "x0": 0.9, "x1": 0.2,
                               "type": "eutectic"}]}
    rep = label_diagram(ax, regions, None, None, kind="binary", elements=["FE", "CR"], T_unit="C",
                        style={"invariant_type": True, "leader": True})
    assert [lab["text"] for lab in rep["labels"] if lab["kind"] == "invariant"] == ["600 °C"]   # once per line
    texts = {lab["text"]: lab for lab in rep["labels"]}
    assert {"L+Fe₃Cr₄", "Fe₃Cr₄", "FeCr", "600 °C"} <= set(texts), texts
    assert not any("eutectic" in s for s in texts)
    assert [s["label"] for s in rep["skipped"]] == ["BCC+σ"] and "8 pt" in rep["skipped"][0]["reason"]
    x, y = texts["L+Fe₃Cr₄"]["x"], texts["L+Fe₃Cr₄"]["y"]
    assert 0.1 < x < 0.5 and 100 < y < 500 and np.hypot((x - 0.3) / 0.4, (y - 300) / 400) > 0.1
    assert abs(texts["600 °C"]["y"] - 600) < 60
    # line compounds: below the x axis at 60°, two close ones spread apart (with a guide), the axis title moved clear
    a, b = (next(t for t in ax.texts if name(t) == s) for s in ("Fe₃Cr₄", "FeCr"))
    assert a.get_rotation() == b.get_rotation() == 60 and a.xy == (0.6, 0.0) and b.xy == (0.62, 0.0)
    assert texts["Fe₃Cr₄"]["outside"] == "bottom" and (texts["Fe₃Cr₄"]["guide"] or texts["FeCr"]["guide"])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    qa, qb = text_rect(a, r), text_rect(b, r)
    assert not rects_overlap(qa, qb) and max(qa[:, 1].max(), qb[:, 1].max()) < ax.bbox.y0
    assert not any(rects_overlap(text_rect(ax.xaxis.label, r), q) for q in (qa, qb))
    shown, hidden = tick_state(ax, "x")
    assert hidden == ["0.6"], (shown, hidden)
    assert "0.0" in shown and "1.0" in shown
    # moving / deleting a name outside the plot gives the tick label back
    move_label(a, dx_pt=60)
    move_label(b, dx_pt=60)
    shown2, hidden2 = tick_state(ax, "x")
    assert set(hidden) - set(hidden2) and set(hidden2) != set(hidden)
    pickle.dumps(fig)
    save_vector(fig, tmp_path / "s.svg")
    save_vector(fig, tmp_path / "s.pdf")
    svg = (tmp_path / "s.svg").read_text()
    for s in ("L+Fe₃Cr₄", "Fe₃Cr₄", "600 °C"):
        assert svg_has(svg, s), s
    assert 'id="phase_label:FE3CR4.LIQUID"' in svg and 'id="invariant_label:600C"' in svg
    assert check_svg(svg) >= 3
    assert "L+Fe3Cr4" in re.sub(r"\s", "", _pdf_text(tmp_path / "s.pdf"))
    assert a.get_text().startswith("Fe$_") and name(a) == "Fe₃Cr₄"           # the figure keeps the typeset form
    # labelling again starts from the original axis title position
    pad = ax.xaxis.labelpad
    clear_labels(ax)
    assert ax.xaxis.labelpad < pad and not find_labels(ax) and not ax.artists
    plt.close(fig)


def test_label_ternary_compounds_and_leaders_synthetic():
    """Ternary (no equilibrium needed): compounds on an edge are named beyond it; an interior ternary compound and a
    thin interior ternary solution field get a leader label (a dot at the point compound) and are never left out;
    moving a leader label keeps the line on its target, deleting it removes its dot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalphad.plot import triangular  # noqa: F401
    from calphad_tools.label_place import label_diagram, find_labels, move_label, remove_label, DOT_PREFIX
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="triangular")
    sliver = [[0.40, 0.30], [0.46, 0.30], [0.46, 0.306], [0.40, 0.306]]
    polys = [{"key": ("FCC_A1", "TAU"), "xy": [[0.2, 0.2], [0.4, 0.3], [0.4, 0.306], [0.2, 0.25]]},
             {"key": ("ETA", "FCC_A1"), "xy": [[0.46, 0.30], [0.6, 0.2], [0.6, 0.25], [0.46, 0.306]]}]
    for pl in polys + [{"xy": sliver}]:
        q = np.array(pl["xy"] + pl["xy"][:1])
        ax.plot(q[:, 0], q[:, 1], color="0.6", lw=1)
    regions = {"polygons": polys, "invariants": [], "phase_points": {"ETA": sliver},
               "compounds": [{"key": ("SIGMA",), "x": 0.5, "y": 0.0}, {"key": ("MU",), "x": 0.0, "y": 0.45},
                             {"key": ("KAPPA",), "x": 0.3, "y": 0.7}, {"key": ("TAU",), "x": 0.25, "y": 0.45}]}
    rep = label_diagram(ax, regions, None, None, kind="ternary", elements=["A", "B", "C"])
    got = {lab["text"]: lab for lab in rep["labels"]}
    assert {k: (v.get("outside"), v["rotation"]) for k, v in got.items() if v.get("outside")} == \
        {"σ": ("bottom", 60.0), "μ": ("left", -30.0), "κ": ("right", 30.0)}, got
    assert got["τ"]["leader"] and got["τ"]["kind"] == "compound" and got["η"]["leader"] and got["η"]["kind"] == "region"
    assert not {"τ", "η"} & {sk["label"] for sk in rep["skipped"]}
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    t = next(t for t in ax.texts if name(t) == "σ")
    assert np.allclose(t.xy, (0.5, 0.0))                              # along the line of constant x
    for t in ax.texts:
        if is_outside(t):
            d = ax.transData.inverted().transform(text_rect(t, r))
            assert ((d[:, 0] < 0) | (d[:, 1] < 0) | (d.sum(axis=1) > 1)).all(), name(t)
    tau = find_labels(ax, "TAU")[0]
    assert tau.arrow_patch is not None and np.allclose(tau.xy, (0.25, 0.45))
    dots = [ln for ln in ax.get_lines() if ln.get_gid() == DOT_PREFIX + tau.get_gid()]
    assert len(dots) == 1 and np.allclose(dots[0].get_xydata(), [(0.25, 0.45)])
    eta = find_labels(ax, "ETA")[0]
    assert eta.arrow_patch is not None and not any(ln.get_gid() == DOT_PREFIX + eta.get_gid() for ln in ax.get_lines())
    rects = [text_rect(t, r) for t in ax.texts if (t.get_gid() or "").startswith("phase_label:")]
    for i in range(len(rects)):                                       # no label covers another
        for j in range(i + 1, len(rects)):
            assert not rects_overlap(rects[i], rects[j])
    p0 = ax.transData.transform(tau.get_position())
    move_label(tau, 15, 10)
    assert np.allclose((ax.transData.transform(tau.get_position()) - p0) * 72 / fig.dpi, (15, 10), atol=0.05)
    assert np.allclose(tau.xy, (0.25, 0.45))                          # the line still ends at the compound
    remove_label(tau)
    assert not find_labels(ax, "TAU") and not any(ln.get_gid() == DOT_PREFIX + "phase_label:TAU"
                                                  for ln in ax.get_lines())
    plt.close(fig)


def _fenb_c14_range(tmp_path):
    """FeNb-90Hua-LB with C14 given a range (Fe on the Nb site, G(C14_LAVES,FE:FE) as the merged A1 database of
    staged_merge has it): a thin C14 field at X(Nb) 0.32-0.333 down to the bottom of the plot."""
    src = (ROOT / "examples/tdbs/FeNb-90Hua-LB.tdb").read_text()
    assert " CONST C14_LAVES : FE : NB : !" in src
    src = src.replace(" CONST C14_LAVES : FE : NB : !", " CONST C14_LAVES : FE : FE,NB : !")
    src = src.replace(" PAR  G(C14_LAVES,FE:NB),,", " PAR  G(C14_LAVES,FE:FE),,              +3*GHSERFE+44130;,,"
                      "                        N 12Liu !\n PAR  G(C14_LAVES,FE:NB),,")
    p = tmp_path / "FeNb-C14range.tdb"
    p.write_text(src)
    return p


def test_binary_thin_field_reaching_the_bottom(tb, tmp_path):
    """Fe-Nb with a non-stoichiometric C14 (the user's trigger): its field is too thin for a label inside and too
    thin to be a region of the plot near the bottom, but it reaches the bottom of the plot: named below the x axis
    at its composition (60°), like the line compound of the original database; show_code draws the same."""
    tdb = _fenb_c14_range(tmp_path)
    text, images = call(tb, "map_phase_diagram", {"tdb": str(tdb), "elements": ["FE", "NB"], "tmin": 300,
                                                  "tmax": 3000, "temperature_unit": "C", "formats": [],
                                                  "output_prefix": "fenb_c14range"})
    fig, ax = stored(images[0])
    assert not ax._calphad_diagram["regions"]["compounds"]          # no longer a stoichiometric phase
    c14 = [t for t in labels_of(ax, ("phase_label:",)) if name(t) == "C14"]
    assert len(c14) == 1, text
    t = c14[0]
    assert t._calphad_label["kind"] == "region" and "C14" not in omitted_line(text)[1]
    if is_outside(t):
        assert t._calphad_label["outside"] == "bottom" and t.get_rotation() == 60 and 0.315 < t.xy[0] < 0.34
        assert "named at the edge instead: " in text and "C14" in text.split("named at the edge instead: ")[1]
    assert check_anchors(ax) >= 8
    check_layout(fig, ax)
    assert check_clearance(fig, ax) >= 2.0
    # the original database: C14 a line compound, named the same way
    text0, images0 = call(tb, "map_phase_diagram", {"tdb": "FeNb-90Hua-LB.tdb", "elements": ["FE", "NB"],
                                                    "tmin": 300, "tmax": 3000, "temperature_unit": "C",
                                                    "formats": [], "output_prefix": "fenb_orig"})
    fig0, ax0 = stored(images0[0])
    t0 = next(t for t in labels_of(ax0, ("phase_label:",)) if name(t) == "C14")
    assert t0._calphad_label["kind"] == "compound" and t0._calphad_label["outside"] == "bottom"
    assert abs(t0.xy[0] - 1 / 3) < 1e-6 and t0.get_rotation() == 60
    # show_code
    from calphad_tools import codegen
    cells, ctx = codegen.build_cells(tb, images[0])
    script = codegen.to_script(tb, cells, ctx)
    assert "calphad_tools" not in script
    d = tmp_path / "fenb_code"
    d.mkdir()
    (d / "gen.py").write_text(script + LABEL_DUMP)
    run = subprocess.run([sys.executable, "-W", "ignore", "gen.py"], cwd=d, capture_output=True, text=True,
                         timeout=900, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-3000:]
    import json
    art = json.loads((d / "art.json").read_text())
    want = {t.get_gid(): [getattr(t, "_calphad_plain", None) or t.get_text(), t._calphad_label.get("outside"), bool(t._calphad_label.get("leader"))]
            for t in labels_of(ax)}
    assert art["labels"] == want, (art["labels"], want)


def test_binary_thin_field_ending_mid_figure_synthetic():
    """Binary single phases that end above the bottom of the plot, too thin inside: σ is named right below its
    lowest point, inside the axes (60°, starting ~2 pt below that point, clear of the lines below); μ has data points
    right below it, so it gets a leader label; a terminal phase with no room stays omitted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calphad_tools.label_place import label_diagram
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(300, 1300)
    ax.set_xlabel("Mole fraction B")
    sig = [[0.40, 600], [0.404, 600], [0.404, 1000], [0.40, 1000]]          # thin sigma field, 600-1000
    mu = [[0.70, 700], [0.704, 700], [0.704, 1100], [0.70, 1100]]           # thin mu field, 700-1100
    term = [[0.0, 900], [0.004, 900], [0.004, 1200], [0.0, 1200]]           # thin terminal field (left side)
    polys = [{"key": ("FCC_A1", "SIGMA"), "xy": [[0.2, 600], [0.40, 600], [0.40, 1000], [0.2, 1000]]},
             {"key": ("BCC_A2", "SIGMA"), "xy": [[0.404, 600], [0.6, 600], [0.6, 1000], [0.404, 1000]]},
             {"key": ("MU_PHASE", "BCC_A2"), "xy": [[0.704, 700], [0.9, 700], [0.9, 1100], [0.704, 1100]]}]
    for pl in polys + [{"xy": q} for q in (sig, mu, term)]:
        q = np.array(pl["xy"] + pl["xy"][:1])
        ax.plot(q[:, 0], q[:, 1], color="k", lw=1.25)
    ax.plot([0.2, 0.6], [600, 600], color="k", lw=1.25)                     # the line sigma ends on
    xs, ys = np.meshgrid(np.linspace(0.6, 0.72, 13), np.linspace(400, 690, 12))
    ax.scatter(xs.ravel(), ys.ravel(), s=12, color="k")                     # data right below mu
    regions = {"polygons": polys, "compounds": [], "invariants": [],
               "phase_points": {"SIGMA": sig, "MU_PHASE": mu, "HCP_A3": term}}
    rep = label_diagram(ax, regions, None, None, kind="binary", elements=["A", "B"], T_unit="C")
    got = {lab["text"]: lab for lab in rep["labels"]}
    assert got["σ"].get("below") and got["σ"]["rotation"] == 60, got
    assert got["μ"].get("leader"), got
    assert "HCP" not in got and "HCP" in [sk["label"] for sk in rep["skipped"]]     # terminal: no room, omitted
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    f = fig.dpi / 72
    t = next(t for t in ax.texts if name(t) == "σ")
    q = text_rect(t, r)
    end = ax.transData.transform((0.402, 600))
    assert q[:, 1].max() < end[1] - 1.5 * f and q[:, 1].max() > end[1] - 6 * f      # starts ~2 pt below the end
    assert np.hypot(*(q[np.argmax(q[:, 1])] - end)) < 12 * f                       # right at the end point
    assert in_plot(ax, q).all()
    assert q[:, 0].mean() < end[0] and q[:, 1].mean() < end[1]                     # runs down and away (left)
    pts = ax.collections[0].get_offset_transform().transform(ax.collections[0].get_offsets())
    for u, v in pts:
        assert not point_in_rect(q, u, v, 3 * f)
    m = next(t for t in ax.texts if name(t) == "μ")
    assert m.arrow_patch is not None and np.allclose(m.xy, m._calphad_label["at"])
    for u, v in pts:
        assert not point_in_rect(text_rect(m, r), u, v, 3 * f)
    plt.close(fig)


@pytest.mark.parametrize("plain,math", [
    ("Al₃Ni₂", r"Al$_\mathregular{3}$Ni$_\mathregular{2}$"),
    ("Ce₂₄Co₁₁", r"Ce$_\mathregular{24}$Co$_\mathregular{11}$"),
    ("L1₂", r"L1$_\mathregular{2}$"),
    ("BCC+σ", "BCC+σ"), ("L", "L"), ("FCC#2", "FCC#2"), ("1345 °C", "1345 °C"),      # nothing to typeset: as is
    ("Al₈Cr₅-LT", r"Al$_\mathregular{8}$Cr$_\mathregular{5}$-LT"),
    ("γ′+L1₂", r"γ′+L1$_\mathregular{2}$"),
    ("μ+Fe₇W₆ (HT)", r"μ+Fe$_\mathregular{7}$W$_\mathregular{6}$ (HT)"),
    ("A$B₂", r"A\$B$_\mathregular{2}$"),
])
def test_subscript_names(plain, math):
    from calphad_tools.phase_names import to_mathtext, svg_subscripts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert to_mathtext(plain) == math
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, math)
    fig.canvas.draw()                                                   # parses as mathtext
    plt.close(fig)
    svg = svg_subscripts(f'<g id="x"><text x="1" y="2" transform="rotate(-60 1 2)">{plain}</text></g>')
    assert not re.search("[₀-₉]", svg) and svg.startswith('<g id="x"><text x="1" y="2" transform="rotate(-60 1 2)">')
    if math != plain:
        assert '<tspan style="font-size:65%;baseline-shift:sub">' in svg


def test_save_vector_svg_subscripts(tmp_path):
    """save_vector: SVG with one <text> per name and Inkscape subscripts; the figure keeps its mathtext; a legend
    and a line label get the same treatment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calphad_tools.phase_names import save_vector, set_name, prettify_legend
    from calphad_tools.diagram_style import line_labels
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="AL3NI2")
    ax.legend()
    prettify_legend(ax, ["AL", "NI"])
    line_labels(ax, [("Ce₂₄Co₁₁", np.array([[0.1, 0.9], [0.9, 0.1]]))])
    t = set_name(ax.text(0.2, 0.2, "", rotation=60, gid="phase_label:CE5CO19"), "Ce₅Co₁₉")
    fig.canvas.draw()
    texts = [t] + list(ax.get_legend().get_texts()) + [x for x in ax.texts if (x.get_gid() or "").startswith("line")]
    assert len(texts) == 3 and all(x.get_text().count("$") >= 2 for x in texts)
    assert [name(x) for x in texts] == ["Ce₅Co₁₉", "Al₃Ni₂", "Ce₂₄Co₁₁"]
    save_vector(fig, tmp_path / "f.svg")
    save_vector(fig, tmp_path / "f.pdf")
    svg = (tmp_path / "f.svg").read_text()
    assert check_svg(svg) == 1
    for s in ("Ce₅Co₁₉", "Al₃Ni₂", "Ce₂₄Co₁₁"):
        assert svg_has(svg, s), s
    assert 'rotate(-60' in svg                               # the rotated label keeps its transform
    assert all(x.get_text().count("$") >= 2 for x in texts)             # restored after the SVG pass
    assert "Ce5Co19" in re.sub(r"\s", "", _pdf_text(tmp_path / "f.pdf"))
    plt.close(fig)
