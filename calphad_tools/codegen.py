"""Standalone code for an output ("show me the code behind this figure").

``show_code`` finds the tool call that produced an image / CSV / analysis (current session or, via provenance
records, earlier sessions), follows its inputs back to other generated outputs (an edited figure -> the diagram it
was drawn on), and emits readable **standalone** Python - pycalphad / numpy / pandas / scipy / matplotlib only, no
calphad_tools import - suitable for pasting into a Jupyter notebook as a provenance record.

Every tool can register a generator::

    @generator("calculate_equilibrium")
    def _gen(args: dict, ctx: Ctx) -> str: ...

Contract for generators
-----------------------
* Return plain Python source (no imports inline - request them with ``ctx.need("import numpy as np")``).
* Paths: TDBs via ``ctx.tdb(name)``, other input files via ``ctx.path(p)`` (both return code expressions relative to
  ``WORKDIR``); write outputs to ``OUT / "<name>"`` (``ctx.out_name(original_path)``).
* Figure generators must leave ``fig`` and ``ax`` (the main axes) defined, so later edit cells can continue.
* A generator may return ``[(subtitle, code), ...]`` for several cells (e.g. one per subsystem of an overview).
* Cells that run pycalphad mapping call ``ctx.mapped(summary_json)``: the notebook then checks the hash seed.
* Reproduce the *calculation* exactly (same conditions, phases, grids); plot styling may be simpler, say so in a
  comment if it differs visibly.
* If something cannot be expressed standalone, call ``ctx.fallback(reason)`` - the tool call is then reproduced
  through calphad_tools, clearly labelled.
Tools without a generator automatically get that labelled fallback.
"""
from __future__ import annotations

import json
import pprint
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

GENERATORS: Dict[str, Callable[[Dict[str, Any], "Ctx"], str]] = {}


def generator(*tools: str):
    def deco(fn):
        for t in tools:
            GENERATORS[t] = fn
        return fn
    return deco


class NoStandalone(Exception):
    """Raised by a generator (via ctx.fallback) when a call can't be expressed standalone."""


class Ctx:
    def __init__(self, tb, produced: Optional[Dict[str, str]] = None):
        self.tb = tb
        self.imports: List[str] = []
        self.notes: List[str] = []
        self.produced = produced or {}          # original output path -> name written under OUT by earlier cells
        self.record: Dict[str, Any] = {}        # the call record being translated (tool, args, images, result, ...)
        self.target: Optional[str] = None       # resolved path of the output this cell is for (None: a call number)
        self.seeds: List[str] = []              # hash seeds of the original mapping runs (see mapped / hashseed)

    # -- imports / notes -----------------------------------------------------------------------
    def need(self, *lines: str) -> None:
        for ln in lines:
            if ln not in self.imports:
                self.imports.append(ln)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def fallback(self, reason: str):
        raise NoStandalone(reason)

    def mapped(self, summary_json: Optional[str] = None) -> None:
        """This cell runs pycalphad mapping: note the hash seed the original run used (its summary JSON's
        "hash_seed", else the session's; see hashseed) - the notebook checks it."""
        seed = None
        if summary_json:
            try:
                seed = json.loads(Path(summary_json).read_text()).get("hash_seed")
            except (OSError, ValueError, AttributeError):
                seed = None
        if not seed:
            seed = self.record.get("hash_seed")
        if not seed and self.record.get("session") == "current":
            from .hashseed import current
            seed = current()
        self.seeds.append(str(seed or "unknown"))

    # -- paths ---------------------------------------------------------------------------------
    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.tb.workdir.resolve()))
        except ValueError:
            return str(p.resolve())

    def tdb(self, name: str) -> str:
        p = Path(self.tb._find_tdb(name))
        r = self._rel(p)
        return f"WORKDIR / {r!r}" if not Path(r).is_absolute() else repr(r)

    def path(self, p: str) -> str:
        """Code expression for an input file; outputs of earlier cells are read from OUT."""
        full = str(Path(self.tb.p(p)).resolve())
        if full in self.produced:
            return f"OUT / {self.produced[full]!r}"
        r = self._rel(Path(full))
        return f"WORKDIR / {r!r}" if not Path(r).is_absolute() else repr(r)

    def out_name(self, original: Optional[str], default: str) -> str:
        """File name to write under OUT for an output the tool produced (keeps the original name)."""
        name = Path(original).name if original else default
        if not original:
            return name
        full = str(Path(self.tb.p(original)).resolve())
        if full in self.produced:
            return self.produced[full]
        taken, stem, suf, k = set(self.produced.values()), Path(name).stem, Path(name).suffix, 2
        while name in taken:                    # same basename from a different folder: keep both
            name, k = f"{stem}_{k}{suf}", k + 1
        self.produced[full] = name
        return name

    @staticmethod
    def py(obj: Any, indent: int = 0) -> str:
        """Readable Python literal."""
        s = pprint.pformat(obj, width=100, sort_dicts=False)
        return s if not indent else s.replace("\n", "\n" + " " * indent)


# ---------------------------------------------------------------------------------------------------------
# finding what produced an output
# ---------------------------------------------------------------------------------------------------------
def _call_record(tb, index: int) -> Dict[str, Any]:
    if not (1 <= index <= len(tb.call_log)):
        raise ValueError(f"call #{index} does not exist in this session (1..{len(tb.call_log)})")
    r = tb.call_log[index - 1]
    return {"tool": r["tool"], "args": r["args"], "call_index": index, "images": r.get("images", []),
            "result": r.get("result_full") or r.get("result_preview", ""), "session": "current"}


# tools that describe, list or echo files but never create the thing being reproduced
NON_PRODUCERS = {"recent_results", "rerun_call", "package_feedback", "tool_help",
                 "show_code", "describe_figure", "plot_ops_help", "session_log", "provenance", "export_session",
                 "report_problem", "list_files", "find_databases", "read_file", "view_image", "pycalphad_help", "tdb_info",
                 "inspect_data", "list_jobs", "job_status", "wait_job", "stop_job", "espei_validate_input"}


def find_producer(tb, target, before: Optional[int] = None) -> Dict[str, Any]:
    """Call record (tool, args, ...) that produced ``target`` (file path or 1-based call number).  before: only
    calls of this session before that call number (the earlier version of a file a call overwrote in place)."""
    if isinstance(target, int) or (isinstance(target, str) and target.strip().lstrip("#").isdigit()):
        return _call_record(tb, int(str(target).strip().lstrip("#")))
    p = Path(tb.p(target))
    if not p.exists() and len(Path(target).parts) == 1:
        hits = list(tb.out.rglob(Path(target).name))
        p = hits[0] if hits else p
    full = str(p.resolve())
    # newest matching call of this session first: its images, then (second pass) the path in its result text.
    # Tools that only read/describe/echo files are never producers (show_code's text quotes the paths it saves).
    last = len(tb.call_log) if before is None else min(len(tb.call_log), int(before) - 1)
    recs = [(i, tb.call_log[i - 1]) for i in range(last, 0, -1)]
    recs = [(i, r) for i, r in recs if r.get("ok") and r.get("tool") not in NON_PRODUCERS]
    for i, r in recs:
        if full in {str(Path(x).resolve()) for x in r.get("images", [])}:
            return _call_record(tb, i)
    from .jobs import producer_record          # files written later by a background job -> its submitting call
    rec = producer_record(tb, full) if before is None else None
    if rec:
        return rec
    for i, r in recs:
        text = r.get("result_full") or r.get("result_preview", "")
        if p.name in text and (str(p) in text or full in text or tb.rel(full) in text.split()
                               or p.name in text.split()):
            return _call_record(tb, i)
    if before is not None:
        raise ValueError(f"no earlier call of this session produced {target}")
    try:
        from .provenance import provenance_for
        info = provenance_for(tb, str(p))
    except Exception:  # noqa: BLE001
        info = None
    if info and info.get("tool"):
        return {"tool": info["tool"], "args": info.get("args") or {}, "call_index": info.get("call_index"),
                "session": info.get("session"), "images": [str(p)], "result": "",
                "hash_seed": info.get("hash_seed")}
    raise ValueError(f"no record of a tool call that produced {target} (made outside the toolkit, or its session "
                     f"log was deleted)")


def _paths_in(obj) -> List[str]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _paths_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _paths_in(v)
    elif isinstance(obj, str) and len(obj) < 400 and re.search(r"\.(png|jpe?g|csv|tsv|json|xlsx|nc|tdb|txt)$",
                                                              obj, re.I):
        out.append(obj)
    return out


_OUTPUT_KEYS = re.compile(r"^(output|out|output_.*|.*_output|output_png|output_csv|output_name|output_prefix|"
                          r"out_png|out_dir|output_dir|file)$", re.I)


def _paths_in_inputs(args: Dict[str, Any]) -> List[str]:
    """Paths among the call's INPUT arguments (keys naming its own outputs are skipped)."""
    if not isinstance(args, dict):
        return _paths_in(args)
    out = []
    for k, v in args.items():
        if isinstance(k, str) and _OUTPUT_KEYS.match(k):
            continue
        out += _paths_in(v) if not isinstance(v, dict) else _paths_in_inputs(v)
    return out


_BASE_KEYS = ("image", "figure", "base", "base_image")


def _parents(tb, rec: Dict[str, Any]) -> List[Tuple[str, Optional[int]]]:
    """Generated files (under the output dir) that this call used as inputs, as (path, before): never its own
    outputs - except the figure it edited IN PLACE (output = image), whose earlier version comes from a call
    before this one (before = this call's number)."""
    own = {str(Path(x).resolve()) for x in rec.get("images") or []}
    args = rec.get("args") or {}
    # the figure being edited is visited LAST so its cell's `fig` is the live one when this call's cell runs
    last = [s for k in _BASE_KEYS if isinstance(args, dict) for s in _paths_in(args.get(k))]
    paths = [s for s in _paths_in_inputs(args) if s not in last] + last
    out: List[Tuple[str, Optional[int]]] = []
    for s in paths:
        full = Path(tb.p(s))
        if not full.exists() and len(Path(s).parts) == 1:
            cand = list(tb.out.rglob(Path(s).name))
            full = cand[0] if cand else full
        try:
            r = str(full.resolve())
            if not (full.exists() and full.resolve().is_relative_to(tb.out.resolve())) or r in [o for o, _ in out]:
                continue
            if r not in own:
                out.append((r, None))
            elif s in last and rec.get("session") == "current" and rec.get("call_index"):
                out.append((r, int(rec["call_index"])))         # edited in place: the version before this call
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------------------------------------
# assembling the code
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Cell:
    title: str
    code: str
    standalone: bool
    note: str = ""
    tool: str = ""


def _fallback_code(tb, rec: Dict[str, Any], reason: str, ctx: Ctx) -> str:
    ctx.need("from calphad_tools.agent.tools import ToolBox, UI")
    args = json.loads(json.dumps(rec.get("args") or {}, default=str))
    return (f"# NOTE: reproduced through calphad_tools ({reason}).\n"
            f"tb = ToolBox(workdir=str(WORKDIR), output_dir=str(OUT.resolve()), ui=UI(), python_tool='off', "
            f"save_notebooks=False)\n"
            f"text, images, failed = tb.call({rec['tool']!r}, {Ctx.py(args, 30)})\n"
            f"print(text)\n")


def build_cells(tb, target, include_parents: bool = True, max_depth: int = 4) -> Tuple[List[Cell], Ctx]:
    ctx = Ctx(tb)
    cells: List[Cell] = []
    done = set()
    in_progress = set()

    def resolve(tgt):
        if not isinstance(tgt, str) or tgt.strip().lstrip("#").isdigit():
            return None
        p = Path(tb.p(tgt))
        if not p.exists() and len(Path(tgt).parts) == 1:
            hits = list(tb.out.rglob(Path(tgt).name))
            p = hits[0] if hits else p
        return str(p.resolve())

    def visit(tgt, depth, before=None):
        rec = find_producer(tb, tgt, before=before)
        key = (rec.get("session"), rec.get("call_index"), rec["tool"], json.dumps(rec.get("args"), sort_keys=True,
                                                                                default=str))
        if key in done or key in in_progress:          # already emitted, or a cycle
            return
        in_progress.add(key)
        if include_parents and depth < max_depth:
            for parent, before_ in _parents(tb, rec):
                try:
                    visit(parent, depth + 1, before_)
                except ValueError:
                    pass                          # an input file that was not produced by a tool: just read it
        done.add(key)
        gen = GENERATORS.get(rec["tool"])
        tool = rec["tool"]
        title = f"{tool}" + (f" (call #{rec['call_index']})" if rec.get("call_index") else "")
        if gen is None:
            cells.append(Cell(title, _fallback_code(tb, rec, "no standalone generator for this tool", ctx), False,
                              tool=tool))
            return
        n_seeds = len(ctx.seeds)
        try:
            ctx.record, ctx.target = rec, resolve(tgt)
            code = gen(dict(rec.get("args") or {}), ctx)
            if isinstance(code, list):              # several cells: [(subtitle, code), ...]
                cells.extend(Cell(f"{title}: {sub}" if sub else title, c, True, tool=tool) for sub, c in code)
            else:
                cells.append(Cell(title, code, True, tool=tool))
        except NoStandalone as e:
            del ctx.seeds[n_seeds:]
            cells.append(Cell(title, _fallback_code(tb, rec, str(e), ctx), False, str(e), tool=tool))
            if tool in MAPPING_TOOLS:
                ctx.mapped()

    visit(target, 0)
    return cells, ctx


def header(tb, ctx: Ctx) -> str:
    lines = ["# Standalone reproduction (generated by show_code; needs only pycalphad, numpy, pandas, scipy, matplotlib)",
             "from pathlib import Path", *ctx.imports, "",
             f"WORKDIR = Path({str(tb.workdir)!r})   # folder with the input files (TDBs, data)",
             "OUT = Path('standalone_outputs').resolve()   # where this code writes its outputs",
             "OUT.mkdir(exist_ok=True)"]
    return "\n".join(lines) + "\n"


def recorded_seed(ctx: Ctx) -> Optional[str]:
    """Hash seed of the original mapping runs behind these cells: "0", another number, "random" (unset or
    unknown), or None when no cell maps a diagram."""
    seeds = [s for s in ctx.seeds if s]
    if not seeds:
        return None
    known = [s for s in seeds if s.isdigit()]
    if len(known) < len(seeds) or len(set(known)) != 1:
        return "random"
    return known[0]


_SEED_CELL = '''
# Reproducibility: pycalphad's mapping visits sets of phases in an order that depends on Python's hash seed
# (PYTHONHASHSEED), so a different seed can trace a few lines differently (an extra short segment, a line traced
# from its other end). The boundaries and invariants agree either way; with the same seed the files are identical.
import os

HASH_SEED = <<seed>>   # PYTHONHASHSEED of the original calculation
NOTEBOOK = <<nb>>   # this notebook (for run_with_recorded_seed)
if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
    print(f"Note: this kernel runs with PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}, the original calculation "
          f"used {HASH_SEED}. The results are valid,\\nbut lines may be traced slightly differently. For identical "
          f"files start Jupyter with\\n    PYTHONHASHSEED={HASH_SEED} jupyter lab\\nor call "
          "run_with_recorded_seed() to run this notebook's code in a fresh Python with that seed.")


def run_with_recorded_seed(notebook=NOTEBOOK):
    """Run the code cells of this notebook in a fresh Python with PYTHONHASHSEED=HASH_SEED (same folder)."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    cells = json.loads(Path(notebook).read_text())["cells"]
    script = Path(notebook).with_suffix(".seeded.py")
    script.write_text("\\n\\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"))
    subprocess.run([sys.executable, str(script)], env={**os.environ, "PYTHONHASHSEED": HASH_SEED}, check=True)
    print("done:", script)
'''

_SEED_CELL_RANDOM = '''
# Reproducibility: pycalphad's mapping visits sets of phases in an order that depends on Python's hash seed
# (PYTHONHASHSEED). The original calculation ran with a random seed (not recorded), so this run can trace a few
# lines slightly differently (an extra short segment, a line traced from its other end); the phase boundaries and
# invariants agree. (The toolkit now runs with PYTHONHASHSEED=0, which makes later runs repeatable.)
import os

print("original hash seed: random (not recorded); this kernel:", os.environ.get("PYTHONHASHSEED", "random"))
'''


def seed_code(ctx: Ctx, notebook: Optional[str] = None) -> str:
    """First notebook cell: the recorded hash seed and the check against this kernel's ("" if nothing maps)."""
    seed = recorded_seed(ctx)
    if seed is None:
        return ""
    if seed == "random":
        return _SEED_CELL_RANDOM.lstrip("\n")
    return (_SEED_CELL.lstrip("\n").replace("<<seed>>", repr(seed))
            .replace("<<nb>>", repr(notebook or "reproduction.ipynb")))


def outputs_text(ctx: Ctx) -> str:
    """The files the code writes (under OUT), for the notebook's introduction."""
    names = sorted(set(ctx.produced.values()))
    if not names:
        return ""
    shown = ", ".join(f"`{n}`" for n in names[:60]) + (f" and {len(names) - 60} more" if len(names) > 60 else "")
    return f"Running all cells writes {len(names)} file(s) to `standalone_outputs/`: {shown}.\n"


def to_script(tb, cells: List[Cell], ctx: Ctx) -> str:
    parts = [header(tb, ctx)]
    seed = recorded_seed(ctx)
    if seed is not None:
        parts.append(f"# original hash seed: PYTHONHASHSEED={seed}"
                     + (" (run with the same seed for identical files)\n" if seed != "random" else
                        " (not recorded: a few lines may be traced slightly differently)\n"))
    for c in cells:
        parts.append(f"\n# %% {c.title}" + ("" if c.standalone else "  [calphad_tools]") + "\n" + c.code.rstrip() + "\n")
    return "".join(parts)


def to_notebook(tb, cells: List[Cell], ctx: Ctx, notebook: Optional[str] = None,
                intro: str = "Generated by calphad_tools `show_code`.") -> Dict[str, Any]:
    """Jupyter notebook (nbformat 4): introduction, hash-seed check (when a cell maps a diagram), imports and
    paths, then one markdown title + one code cell per step.  notebook: the file name it is saved under."""
    def code_cell(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    def md_cell(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    nb = [md_cell("# Standalone reproduction\n" + intro + " Cells marked [calphad_tools] call the toolkit because "
                  "no standalone equivalent exists for that step.\n\n" + outputs_text(ctx))]
    seed = seed_code(ctx, notebook)
    if seed:
        nb.append(code_cell(seed))
    nb.append(code_cell(header(tb, ctx)))
    for c in cells:
        nb.append(md_cell(f"### {c.title}" + ("" if c.standalone else "  [calphad_tools]")
                          + (f"\n{c.note}" if c.note else "")))
        nb.append(code_cell(c.code.rstrip() + "\n"))
    return {"cells": nb, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                     "language": "python"},
                                      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


# ---------------------------------------------------------------------------------------------------------
# notebooks saved automatically next to every phase diagram
# ---------------------------------------------------------------------------------------------------------
MAPPING_TOOLS = {"map_phase_diagram", "plot_diagram_with_data", "plot_all_binaries", "plot_all_ternaries"}
NOTEBOOK_TOOLS = MAPPING_TOOLS | {"edit_figure"}       # edit_figure: when the edited figure is a phase diagram


def notebook_path(tb, rec: Dict[str, Any]) -> Optional[Path]:
    """Where the auto-saved notebook of a call goes: next to its main PNG (<stem>.ipynb); an overview's next to
    its deck."""
    images = rec.get("images") or []
    if rec.get("tool") in ("plot_all_binaries", "plot_all_ternaries"):
        m = re.search(r"^PowerPoint: (.+)$", rec.get("result_full") or rec.get("result_preview") or "", re.M)
        if m:
            return Path(tb.p(m.group(1).strip())).with_suffix(".ipynb")
    if not images:
        return None
    return Path(images[0]).with_suffix(".ipynb")


def save_figure_notebook(tb, index: int) -> Optional[str]:
    """Write the reproduction notebook of call ``index`` (codegen only, nothing is recomputed): the call and the
    steps it was built from; running it writes every file the call wrote.  None if the call made no phase
    diagram."""
    rec = tb.call_log[index - 1]
    if rec.get("tool") not in NOTEBOOK_TOOLS or not rec.get("ok"):
        return None
    path = notebook_path(tb, rec)
    if path is None:
        return None
    cells, ctx = build_cells(tb, str(index))
    if not any(c.tool in MAPPING_TOOLS for c in cells):
        return None                     # e.g. an edit of an ordinary plot
    intro = (f"Saved automatically by calphad_tools with `{rec['tool']}` (call #{index}): it repeats that "
             f"calculation{' and the steps it was built from' if len(cells) > 1 else ''} with pycalphad and writes "
             f"the same files.")
    nb = to_notebook(tb, cells, ctx, notebook=path.name, intro=intro)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(nb, indent=1))
    tmp.replace(path)
    return str(path)


def _notebook_hook(tb):
    import logging
    log = logging.getLogger("calphad_tools.codegen")

    def hook(rec: Dict[str, Any], index: int) -> Optional[str]:
        if not getattr(tb, "save_notebooks", True) or rec.get("tool") not in NOTEBOOK_TOOLS:
            return None
        try:
            path = save_figure_notebook(tb, index)
        except Exception as e:  # noqa: BLE001 - codegen must never break the tool that made the figure
            log.warning("reproduction notebook for %s (call #%d) not saved: %s: %s", rec.get("tool"), index,
                        type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "")
            return None
        return f"notebook: {tb.rel(path)} (regenerates all files below)" if path else None
    return hook


# ---------------------------------------------------------------------------------------------------------
# generators for a few simple tools (the rest live in codegen_calc.py / codegen_plot.py)
# ---------------------------------------------------------------------------------------------------------
@generator("run_python")
def _gen_run_python(args, ctx: Ctx) -> str:
    ctx.need("import numpy as np", "import matplotlib.pyplot as plt", "import pycalphad",
             "from pycalphad import Database, equilibrium, calculate, binplot, ternplot, variables as v")
    code = args.get("code", "")
    helpers = []
    if "load_db(" in code:
        helpers.append("def load_db(path):\n    p = Path(path)\n    return Database(str(p if p.is_absolute() "
                       "else WORKDIR / p))\n")
    if "savefig(" in code:
        helpers.append("def savefig(name, fig=None, dpi=150):\n    f = fig or plt.gcf()\n    path = OUT / (name if "
                       "name.endswith('.png') else name + '.png')\n    f.savefig(path, dpi=dpi, bbox_inches='tight')"
                       "\n    return path\n")
    return ("".join(helpers) + ("\n" if helpers else "") + "# code exactly as the assistant ran it\n" + code + "\n")


@generator("convert_composition")
def _gen_convert_composition(args, ctx: Ctx) -> str:
    ctx.need("from pycalphad import variables as v")
    comp = {k.upper().replace("W(", "").replace("X(", "").replace(")", ""): float(x)
            for k, x in args["composition"].items()}
    bal = args["balance"].upper()
    unit = args["from_unit"]
    scale = 100.0 if unit.endswith("%") else 1.0
    if args.get("tdb"):
        ctx.need("from pycalphad import Database")
        masses = (f"db = Database(str({ctx.tdb(args['tdb'])}))\n"
                  f"masses = {{el: float(db.refstates[el]['mass']) for el in {Ctx.py(list(comp) + [bal])}}}\n")
    else:
        from .expdata import ATOMIC_MASS
        masses = f"masses = {Ctx.py({e: ATOMIC_MASS[e] for e in list(comp) + [bal]})}\n"
    if unit.startswith("wt"):
        conv = (f"fractions = {{v.W(el): val / {scale} for el, val in {Ctx.py(comp)}.items()}}\n"
                f"mole = v.get_mole_fractions(fractions, {bal!r}, masses)\n"
                "for key, x in mole.items():\n    print(key, round(float(x), 6))\n")
    else:
        conv = (f"fractions = {{v.X(el): val / {scale} for el, val in {Ctx.py(comp)}.items()}}\n"
                f"mass = v.get_mass_fractions(fractions, {bal!r}, masses)\n"
                "for key, w in mass.items():\n    print(key, round(float(w), 6))\n")
    return masses + conv


@generator("calculate_equilibrium")
def _gen_calculate_equilibrium(args, ctx: Ctx) -> str:
    ctx.need("import numpy as np", "from pycalphad import Database, equilibrium, variables as v",
             "from pycalphad.core.utils import filter_phases, unpack_species")
    els = [e.upper() for e in args["elements"]]
    unit = args.get("temperature_unit", "K")
    conds = []
    for k, val in args["conditions"].items():
        ku = k.upper().replace(" ", "")
        if ku in ("T", "T_K"):
            expr = Ctx.py(val)
            if unit.upper().startswith("C"):
                expr = (f"np.asarray({expr}) + 273.15" if isinstance(val, (list, tuple)) else f"{val} + 273.15")
            conds.append(f"v.T: {expr}")
        elif ku == "P":
            conds.append(f"v.P: {Ctx.py(val)}")
        else:
            el = re.sub(r"^X\(|\)$", "", ku)
            conds.append(f"v.X({el!r}): {Ctx.py(val)}")
    if not any(c.startswith("v.P") for c in conds):
        conds.append("v.P: 101325")
    conds.append("v.N: 1")
    phases = (Ctx.py([p.upper() for p in args["phases"]]) if args.get("phases")
              else f"sorted(filter_phases(db, unpack_species(db, comps)))")
    return (f"db = Database(str({ctx.tdb(args['tdb'])}))\n"
            f"comps = {Ctx.py(els + ['VA'])}\n"
            f"phases = {phases}\n"
            f"conds = {{{', '.join(conds)}}}\n"
            "eq = equilibrium(db, comps, phases, conds)\n\n"
            "# stable phases, their fractions and compositions at every condition point\n"
            "phase, npf, x = eq.Phase.values, eq.NP.values, eq.X.values\n"
            "for idx in np.ndindex(*phase.shape[:-1]):\n"
            "    point = {d: float(eq.coords[d].values[i]) for d, i in zip(eq.Phase.dims[:-1], idx)}\n"
            "    print(point, 'GM =', round(float(eq.GM.values[idx]), 2))\n"
            "    for vtx in range(phase.shape[-1]):\n"
            "        if phase[idx + (vtx,)]:\n"
            "            comp = {str(c): round(float(x[idx + (vtx, j)]), 5) for j, c in enumerate(eq.component.values)}\n"
            "            print('   ', str(phase[idx + (vtx,)]), round(float(npf[idx + (vtx,)]), 5), comp)\n")


# ---------------------------------------------------------------------------------------------------------
# agent tool
# ---------------------------------------------------------------------------------------------------------
EXTENSION_MODULES = ["calphad_tools.codegen_calc", "calphad_tools.codegen_plot", "calphad_tools.codegen_espei",
                     "calphad_tools.codegen_params",
                     "calphad_tools.codegen_formats", "calphad_tools.codegen_compare", "calphad_tools.codegen_zpf",
                     "calphad_tools.codegen_staged", "calphad_tools.codegen_combine"]


def ensure_generators() -> None:
    import importlib
    for m in EXTENSION_MODULES:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            if e.name != m:
                raise


def register(tb) -> None:
    ensure_generators()
    if hasattr(tb, "result_hooks"):
        tb.result_hooks.append(_notebook_hook(tb))

    @tb.add("show_code", "Standalone Python (pycalphad/matplotlib, no calphad_tools) that reproduces an output - an "
            "image, CSV or analysis - including the steps it was built from (e.g. the diagram an edited figure was "
            "drawn on), as a Jupyter notebook (default; the .py script is saved too) - a provenance record. "
            "target: output file or call number (see session_log). Steps with no standalone equivalent are "
            "reproduced via calphad_tools, labelled. Phase diagrams already get <name>.ipynb next to the PNG.",
            {"target": {"type": "string"},
             "format": {"type": "string", "enum": ["notebook", "script", "cell"],
                        "description": "notebook (default: .ipynb + .py) | script / cell (.py only)"},
             "include_parents": {"type": "boolean"}, "output": {"type": "string"}}, ["target"])
    def show_code(target, format: str = "notebook", include_parents: bool = True, output: Optional[str] = None):
        cells, ctx = build_cells(tb, target, include_parents=include_parents)
        script = to_script(tb, cells, ctx)
        stem = Path(str(target)).stem if not str(target).lstrip("#").isdigit() else f"call{str(target).lstrip('#')}"
        py_path = tb.outp(output if output and output.endswith(".py") else None, f"code/{stem}_standalone.py")
        Path(py_path).write_text(script)
        files = [py_path]
        if (format or "notebook") == "notebook" or (output and output.endswith(".ipynb")):
            nb_path = tb.outp(output if output and output.endswith(".ipynb") else None,
                              f"code/{stem}_standalone.ipynb")
            Path(nb_path).write_text(json.dumps(to_notebook(tb, cells, ctx, notebook=Path(nb_path).name),
                                                indent=1))
            files.append(nb_path)
        n_fb = sum(not c.standalone for c in cells)
        head = (f"{len(cells)} step(s); {len(cells) - n_fb} standalone"
                + (f", {n_fb} via calphad_tools" if n_fb else "") + "; saved " + ", ".join(tb.rel(f) for f in files))
        notes = ("\nnotes: " + "; ".join(ctx.notes)) if ctx.notes else ""
        return f"{head}{notes}\n\n```python\n{script}```", []
