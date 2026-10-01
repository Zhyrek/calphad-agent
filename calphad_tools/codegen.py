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

    # -- imports / notes -----------------------------------------------------------------------
    def need(self, *lines: str) -> None:
        for ln in lines:
            if ln not in self.imports:
                self.imports.append(ln)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def fallback(self, reason: str):
        raise NoStandalone(reason)

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


def find_producer(tb, target) -> Dict[str, Any]:
    """Call record (tool, args, ...) that produced ``target`` (file path or 1-based call number)."""
    if isinstance(target, int) or (isinstance(target, str) and target.strip().lstrip("#").isdigit()):
        return _call_record(tb, int(str(target).strip().lstrip("#")))
    p = Path(tb.p(target))
    if not p.exists() and len(Path(target).parts) == 1:
        hits = list(tb.out.rglob(Path(target).name))
        p = hits[0] if hits else p
    full = str(p.resolve())
    # newest matching call of this session first: its images, then (second pass) the path in its result text.
    # Tools that only read/describe/echo files are never producers (show_code's text quotes the paths it saves).
    recs = [(i, tb.call_log[i - 1]) for i in range(len(tb.call_log), 0, -1)]
    recs = [(i, r) for i, r in recs if r.get("ok") and r.get("tool") not in NON_PRODUCERS]
    for i, r in recs:
        if full in {str(Path(x).resolve()) for x in r.get("images", [])}:
            return _call_record(tb, i)
    from .jobs import producer_record          # files written later by a background job -> its submitting call
    rec = producer_record(tb, full)
    if rec:
        return rec
    for i, r in recs:
        text = r.get("result_full") or r.get("result_preview", "")
        if p.name in text and (str(p) in text or full in text or tb.rel(full) in text.split()
                               or p.name in text.split()):
            return _call_record(tb, i)
    try:
        from .provenance import provenance_for
        info = provenance_for(tb, str(p))
    except Exception:  # noqa: BLE001
        info = None
    if info and info.get("tool"):
        return {"tool": info["tool"], "args": info.get("args") or {}, "call_index": info.get("call_index"),
                "session": info.get("session"), "images": [str(p)], "result": ""}
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


def _parents(tb, rec: Dict[str, Any]) -> List[str]:
    """Generated files (under the output dir) that this call used as inputs (never its own outputs)."""
    own = {str(Path(x).resolve()) for x in rec.get("images") or []}
    args = rec.get("args") or {}
    # the figure being edited is visited LAST so its cell's `fig` is the live one when this call's cell runs
    last = [s for k in _BASE_KEYS if isinstance(args, dict) for s in _paths_in(args.get(k))]
    paths = [s for s in _paths_in_inputs(args) if s not in last] + last
    out = []
    for s in paths:
        full = Path(tb.p(s))
        if not full.exists() and len(Path(s).parts) == 1:
            cand = list(tb.out.rglob(Path(s).name))
            full = cand[0] if cand else full
        try:
            if (full.exists() and full.resolve().is_relative_to(tb.out.resolve())
                    and str(full.resolve()) not in own and str(full.resolve()) not in out):
                out.append(str(full.resolve()))
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


def _fallback_code(tb, rec: Dict[str, Any], reason: str, ctx: Ctx) -> str:
    ctx.need("from calphad_tools.agent.tools import ToolBox, UI")
    args = json.loads(json.dumps(rec.get("args") or {}, default=str))
    return (f"# NOTE: reproduced through calphad_tools ({reason}).\n"
            f"tb = ToolBox(workdir=str(WORKDIR), output_dir=str(OUT.resolve()), ui=UI(), python_tool='off')\n"
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

    def visit(tgt, depth):
        rec = find_producer(tb, tgt)
        key = (rec.get("session"), rec.get("call_index"), rec["tool"], json.dumps(rec.get("args"), sort_keys=True,
                                                                                default=str))
        if key in done or key in in_progress:          # already emitted, or a cycle
            return
        in_progress.add(key)
        if include_parents and depth < max_depth:
            for parent in _parents(tb, rec):
                try:
                    visit(parent, depth + 1)
                except ValueError:
                    pass                          # an input file that was not produced by a tool: just read it
        done.add(key)
        gen = GENERATORS.get(rec["tool"])
        title = f"{rec['tool']}" + (f" (call #{rec['call_index']})" if rec.get("call_index") else "")
        if gen is None:
            cells.append(Cell(title, _fallback_code(tb, rec, "no standalone generator for this tool", ctx), False))
            return
        try:
            ctx.record, ctx.target = rec, resolve(tgt)
            code = gen(dict(rec.get("args") or {}), ctx)
            cells.append(Cell(title, code, True))
        except NoStandalone as e:
            cells.append(Cell(title, _fallback_code(tb, rec, str(e), ctx), False, str(e)))

    visit(target, 0)
    return cells, ctx


def header(tb, ctx: Ctx) -> str:
    lines = ["# Standalone reproduction (generated by show_code; needs only pycalphad, numpy, pandas, scipy, matplotlib)",
             "from pathlib import Path", *ctx.imports, "",
             f"WORKDIR = Path({str(tb.workdir)!r})   # folder with the input files (TDBs, data)",
             "OUT = Path('standalone_outputs').resolve()   # where this code writes its outputs",
             "OUT.mkdir(exist_ok=True)"]
    return "\n".join(lines) + "\n"


def to_script(tb, cells: List[Cell], ctx: Ctx) -> str:
    parts = [header(tb, ctx)]
    for c in cells:
        parts.append(f"\n# %% {c.title}" + ("" if c.standalone else "  [calphad_tools]") + "\n" + c.code.rstrip() + "\n")
    return "".join(parts)


def to_notebook(tb, cells: List[Cell], ctx: Ctx) -> Dict[str, Any]:
    def code_cell(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    def md_cell(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    nb = [md_cell("# Standalone reproduction\nGenerated by calphad_tools `show_code`. Cells marked [calphad_tools] "
                  "call the toolkit because no standalone equivalent exists for that step.\n"),
          code_cell(header(tb, ctx))]
    for c in cells:
        nb.append(md_cell(f"### {c.title}" + ("" if c.standalone else "  [calphad_tools]")
                          + (f"\n{c.note}" if c.note else "")))
        nb.append(code_cell(c.code.rstrip() + "\n"))
    return {"cells": nb, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                     "language": "python"},
                                      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


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
                     "calphad_tools.codegen_formats", "calphad_tools.codegen_compare", "calphad_tools.codegen_zpf"]


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

    @tb.add("show_code", "Standalone Python (pycalphad/matplotlib, no calphad_tools) that reproduces an output - an "
            "image, CSV or analysis - including the steps it was built from (e.g. the diagram an edited figure was "
            "drawn on), for pasting into a Jupyter notebook as provenance. target: output file or call number "
            "(see session_log). Steps with no standalone equivalent are reproduced via calphad_tools, labelled.",
            {"target": {"type": "string"}, "format": {"type": "string", "enum": ["cell", "script", "notebook"]},
             "include_parents": {"type": "boolean"}, "output": {"type": "string"}}, ["target"])
    def show_code(target, format: str = "cell", include_parents: bool = True, output: Optional[str] = None):
        cells, ctx = build_cells(tb, target, include_parents=include_parents)
        script = to_script(tb, cells, ctx)
        stem = Path(str(target)).stem if not str(target).lstrip("#").isdigit() else f"call{str(target).lstrip('#')}"
        py_path = tb.outp(output if output and output.endswith(".py") else None, f"code/{stem}_standalone.py")
        Path(py_path).write_text(script)
        files = [py_path]
        if format == "notebook" or (output and output.endswith(".ipynb")):
            nb_path = tb.outp(output if output and output.endswith(".ipynb") else None,
                              f"code/{stem}_standalone.ipynb")
            Path(nb_path).write_text(json.dumps(to_notebook(tb, cells, ctx), indent=1))
            files.append(nb_path)
        n_fb = sum(not c.standalone for c in cells)
        head = (f"{len(cells)} step(s); {len(cells) - n_fb} standalone"
                + (f", {n_fb} via calphad_tools" if n_fb else "") + "; saved " + ", ".join(tb.rel(f) for f in files))
        notes = ("\nnotes: " + "; ".join(ctx.notes)) if ctx.notes else ""
        return f"{head}{notes}\n\n```python\n{script}```", []
