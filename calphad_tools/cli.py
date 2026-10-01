"""`calphad-tools` - run every feature without an LLM.

  calphad-tools info       my.tdb
  calphad-tools compare    A.tdb B.tdb -o comparison.pptx [--all]
  calphad-tools binaries   my.tdb -o binaries.pptx [--tmin 100 --tmax 3000]
  calphad-tools ternaries  my.tdb --temps 800 1000 1200 -o ternaries.pptx
  calphad-tools merge      A.tdb B.tdb [C.tdb ...] -o merged.tdb [--prefer first|last] [--decisions d.json]
  calphad-tools exp2zpf    data.POP|data.exp|data.csv --components CR NI -o datasets/ [--tdb my.tdb]
  calphad-tools pdftext    paper.pdf [--pages 3-5]
  calphad-tools phase-models my.tdb --elements CR NI -o phase_models.json
  calphad-tools espei-flux campaign.yaml -o espei_runs/
  calphad-tools chat       (same as calphad-agent)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _progress(msg: str) -> None:
    print(f"  … {msg}", flush=True)


def cmd_info(a):
    from .calc import tdb_info
    print(json.dumps(tdb_info(a.tdb), indent=1, default=str))


def cmd_compare(a):
    from .compare import run
    r = run(a.tdb_a, a.tdb_b, a.output, include_identical=a.all, csv_path=a.csv, t_min=a.tmin_k, t_max=a.tmax_k,
            rtol=a.rtol)
    print(r["summary"])
    print(f"\nPowerPoint: {r['pptx']}\nCSV: {r['csv']}")


def cmd_binaries(a):
    from .diagrams import all_binaries
    r = all_binaries(a.tdb, a.output, tmin_c=a.tmin, tmax_c=a.tmax, n_workers=a.workers, timeout=a.timeout,
                     elements=a.elements, individual_slides=a.individual, x_step=a.x_step, t_step=a.t_step,
                     exclude_phases=a.exclude, progress=_progress)
    _report_systems(r)


def cmd_ternaries(a):
    from .diagrams import all_ternaries
    r = all_ternaries(a.tdb, a.temps, a.output, n_workers=a.workers, timeout=a.timeout, elements=a.elements,
                      individual_slides=a.individual, step=a.step, exclude_phases=a.exclude,
                      units="K" if a.kelvin else "C", progress=_progress)
    _report_systems(r)


def _report_systems(r):
    bad = [s for s in r["systems"] if not s["ok"]]
    print(f"\nPowerPoint: {r['pptx']}   ({len(r['systems']) - len(bad)} ok, {len(bad)} failed)")
    for s in bad:
        print(f"  FAILED {s['name']}: {s['error']}")


def cmd_merge(a):
    from . import merge
    decisions = json.loads(Path(a.decisions).read_text()) if a.decisions else {}
    policy = a.prefer or "ask"
    ask = merge.cli_ask if (policy == "ask" and sys.stdin.isatty()) else None
    r = merge.merge(a.tdbs, a.output, decisions=decisions, policy=policy, ask=ask, rtol=a.rtol)
    print(r["report"])
    if a.save_decisions:
        merge.save_decisions(r["decisions"], a.save_decisions)
        print(f"decisions saved to {a.save_decisions} (re-use with --decisions)")
    if not r["written"]:
        print("\nNot written: conflicts remain. Re-run interactively, pass --prefer first|last, or give "
              "--decisions a JSON file mapping conflict ids (above) to option keys.")
        return 1
    print(f"\nwritten {r['output']}")
    v = r["validation"]
    print("pycalphad validation:", "OK" if v["loaded"] and not v["errors"] else v["errors"])
    return 0


def cmd_exp2zpf(a):
    from . import expdata
    src = Path(a.input)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    datasets = []
    if suffix in (".pop", ".dop"):
        phases = None
        if a.tdb:
            from .calc import get_db
            phases = sorted(get_db(a.tdb).phases)
        pm = json.loads(Path(a.phase_map).read_text()) if a.phase_map else None
        datasets, notes = expdata.pop_to_zpf(str(src), a.components, tdb_phases=phases, phase_map=pm)
        for n in notes:
            print("  note:", n)
    elif suffix == ".exp":
        parsed = expdata.parse_exp(str(src))
        print("axes:", parsed["prologue"].get("XTEXT"), "|", parsed["prologue"].get("YTEXT"), "->", parsed["axes_guess"])
        if a.map:
            dmap = json.loads(Path(a.map).read_text())
        else:
            dmap = {}
            print("Map each DATASET to the phases in equilibrium (blank = skip).")
            for d in parsed["datasets"]:
                print(f"\nDATASET {d['id']}: {d['title']}  ({len(d['points'])} points)")
                for c in d["comments"][:5]:
                    print(f"   $ {c}")
                ph = input("  phases in equilibrium (e.g. LIQUID BCC_A2): ").split()
                if not ph:
                    continue
                meas = input(f"  phase whose composition x is [{ph[0]}]: ").strip() or ph[0]
                dmap[str(d["id"])] = {"phases": ph, "measured_phase": meas}
            Path(out / f"{src.stem}_dataset_map.json").write_text(json.dumps(dmap, indent=1))
        datasets = expdata.exp_to_zpf(parsed, a.components, dmap, x_unit=a.x_unit, y_unit=a.y_unit)
    elif suffix == ".csv":
        datasets = expdata.csv_to_zpf(str(src), a.components, composition_unit=a.x_unit or "mole_fraction")
    elif suffix == ".pdf":
        print(expdata.pdf_text(str(src), a.pages))
        print("\n(PDF text shown above. Extracting tables from prose needs the LLM agent (calphad-agent), or "
              "copy the table into the CSV template and re-run with the .csv.)")
        return 0
    else:
        raise SystemExit(f"unsupported input type {suffix}")
    for i, d in enumerate(datasets, 1):
        p = expdata.write_dataset(d, str(out / f"{src.stem}_{i}_zpf.json"))
        print(f"wrote {p}  ({len(d['values'])} points, phases {d['phases']})")
    return 0


def cmd_pdftext(a):
    from .expdata import pdf_text
    print(pdf_text(a.pdf, a.pages))


def cmd_phase_models(a):
    from .espei_flux import phase_models_from_tdb
    r = phase_models_from_tdb(a.tdb, a.elements, a.output, phases=a.phases)
    print(f"wrote {a.output}: phases {list(r['phases'])}")
    if r.get("_validation_error"):
        print("WARNING:", r["_validation_error"])


def cmd_espei_flux(a):
    from .espei_flux import generate, load_campaign
    runs, fs = load_campaign(a.campaign)
    if a.mode:
        fs.mode = a.mode
    r = generate(runs, fs, a.output, base_dir=str(Path(a.campaign).resolve().parent))
    for f in r["files"]:
        print("  ", f)
    if r["problems"]:
        print("PROBLEMS:", json.dumps(r["problems"], indent=1))
        return 1
    print(f"\nSubmit with:  {r['submit']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="calphad-tools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="summarise a TDB")
    p.add_argument("tdb")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("compare", help="compare two TDBs -> pptx + csv")
    p.add_argument("tdb_a")
    p.add_argument("tdb_b")
    p.add_argument("-o", "--output", default="tdb_comparison.pptx")
    p.add_argument("--all", action="store_true", help="include identical items in the tables")
    p.add_argument("--csv")
    p.add_argument("--tmin-k", type=float, default=298.15)
    p.add_argument("--tmax-k", type=float, default=3000.0)
    p.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for 'equivalent'")
    p.set_defaults(fn=cmd_compare)

    for name, fn in (("binaries", cmd_binaries), ("ternaries", cmd_ternaries)):
        p = sub.add_parser(name, help=f"all {name} of a TDB -> pptx")
        p.add_argument("tdb")
        p.add_argument("-o", "--output", default=f"{name}.pptx")
        p.add_argument("--elements", nargs="+", help="subset of elements")
        p.add_argument("--exclude", nargs="+", help="phases to leave out (e.g. GAS)")
        p.add_argument("--workers", type=int, help="parallel processes (default: cores-1)")
        p.add_argument("--timeout", type=float, default=1800.0, help="seconds per diagram")
        p.add_argument("--individual", action="store_true", help="also one slide per diagram")
        if name == "binaries":
            p.add_argument("--tmin", type=float, default=100.0, help="°C")
            p.add_argument("--tmax", type=float, default=3000.0, help="°C")
            p.add_argument("--x-step", type=float, default=0.02)
            p.add_argument("--t-step", type=float, default=10.0)
        else:
            p.add_argument("--temps", type=float, nargs="+", required=True, help="temperatures (°C unless --kelvin)")
            p.add_argument("--kelvin", action="store_true")
            p.add_argument("--step", type=float, default=0.02)
        p.set_defaults(fn=fn)

    p = sub.add_parser("merge", help="merge TDBs (interactive conflict resolution)")
    p.add_argument("tdbs", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--prefer", choices=["first", "last"], help="resolve every real conflict automatically")
    p.add_argument("--decisions", help="JSON {conflict_id: option_key} from a previous run")
    p.add_argument("--save-decisions", help="write the decisions used to this JSON file")
    p.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for auto-resolving")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("exp2zpf", help="POP / DATAPLOT .exp / CSV -> ESPEI ZPF datasets")
    p.add_argument("input")
    p.add_argument("--components", nargs="+", required=True)
    p.add_argument("-o", "--output", default="datasets")
    p.add_argument("--tdb", help="TDB for resolving abbreviated phase names in POP files")
    p.add_argument("--phase-map", help="JSON {abbrev: PHASE_NAME} for POP files")
    p.add_argument("--map", help="JSON dataset map for .exp files (else asked interactively)")
    p.add_argument("--x-unit", choices=["mole_fraction", "at%", "wt%", "wt_fraction"])
    p.add_argument("--y-unit", choices=["K", "C"])
    p.add_argument("--pages")
    p.set_defaults(fn=cmd_exp2zpf)

    p = sub.add_parser("pdftext", help="dump text from a PDF")
    p.add_argument("pdf")
    p.add_argument("--pages")
    p.set_defaults(fn=cmd_pdftext)

    p = sub.add_parser("phase-models", help="ESPEI phase_models.json from a TDB")
    p.add_argument("tdb")
    p.add_argument("--elements", nargs="+", required=True)
    p.add_argument("--phases", nargs="+")
    p.add_argument("-o", "--output", default="phase_models.json")
    p.set_defaults(fn=cmd_phase_models)

    p = sub.add_parser("espei-flux", help="ESPEI inputs + Flux batch scripts from a campaign YAML")
    p.add_argument("campaign")
    p.add_argument("-o", "--output", default="espei_runs")
    p.add_argument("--mode", choices=["per-job", "bundle"])
    p.set_defaults(fn=cmd_espei_flux)

    p = sub.add_parser("chat", help="LLM chat agent (see calphad-agent --help)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: __import__("calphad_tools.agent.cli", fromlist=["main"]).main(a.rest))

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
