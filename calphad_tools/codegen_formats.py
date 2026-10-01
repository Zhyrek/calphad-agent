"""show_code generators for database-format tools (convert_database, write_database with a dialect).

The TDB dialect rules live in :mod:`calphad_tools.tdb_dialect`, which uses only the standard library and
pycalphad; its functions are copied into the generated script so it stays standalone.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from .codegen import Ctx, generator


def dialect_helpers(ctx: Ctx) -> str:
    """Source of tdb_dialect without its module docstring and top-level imports (requested via ctx.need)."""
    from . import tdb_dialect
    src = inspect.getsource(tdb_dialect)
    tree = ast.parse(src)
    lines = src.splitlines()
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                and isinstance(node.value.value, str):
            continue                                  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                ctx.need(f"import {a.name}" if isinstance(node, ast.Import) else f"from {node.module} import {a.name}")
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        keep.append("\n".join(lines[start - 1:node.end_lineno]))
    return ("# TDB dialect exporter: renames, dialect syntax, limit checks (standard library + pycalphad)\n"
            + "\n\n".join(keep) + "\n\n\n")


def _out_name(ctx: Ctx, output: str, default: str) -> str:
    from .codegen_calc import _outp
    return ctx.out_name(_outp(ctx.tb, output, default), default)


def write_database_code(args, ctx: Ctx) -> str:
    """write_database with ``dialect`` (called from codegen_calc's write_database generator)."""
    t = ctx.tb._find_tdb(args["tdb"])
    name = _out_name(ctx, args["output_tdb"], f"{Path(t).stem}_subset.tdb")
    ctx.need("from pycalphad import Database")
    return (dialect_helpers(ctx) + f"db = Database(str({ctx.tdb(args['tdb'])}))\n"
            f"info = export_tdb(db, str(OUT / {name!r}), dialect={args['dialect']!r}, "
            f"groupby={args.get('groupby') or 'subsystem'!r})\n"
            "print('renamed:', {k: v for k, v in info['renames'].items() if v})\n"
            "print('dialect problems:', info['violations'] or 'none')\n"
            f"back = Database(str(OUT / {name!r}))   # reload check\n"
            "print(len(back.phases), 'phases,', len(back._parameters), 'parameters,', len(back.symbols), 'functions')\n")


@generator("convert_database")
def _gen_convert_database(args, ctx: Ctx) -> str:
    s = ctx.tb._find_tdb(args["source"])
    out = args["output_tdb"] if args["output_tdb"].lower().endswith(".tdb") else args["output_tdb"] + ".tdb"
    name = _out_name(ctx, out, f"{Path(s).stem}.tdb")
    ctx.need("from pycalphad import Database")
    return (dialect_helpers(ctx) + f"db = Database(str({ctx.tdb(args['source'])}))   # .dat: pycalphad's ChemSage reader\n"
            "compat, report = tdb_compatible(db)   # drop what a TDB cannot express\n"
            "print('not converted:', report['dropped_phases'] or 'nothing')\n"
            f"info = export_tdb(compat, str(OUT / {name!r}), dialect={args.get('dialect') or 'pycalphad'!r}, "
            f"groupby={args.get('groupby') or 'subsystem'!r})\n"
            "print('renamed:', {k: v for k, v in info['renames'].items() if v})\n"
            "print('dialect problems:', info['violations'] or 'none')\n"
            f"back = Database(str(OUT / {name!r}))   # reload check\n"
            "print(len(back.phases), 'phases,', len(back._parameters), 'parameters')\n")
