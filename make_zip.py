#!/usr/bin/env python3
"""Pack this folder into ../calphad-agent.zip for copying to another computer (run again after changes).

    python3 make_zip.py                  # -> ../calphad-agent.zip  (or ./make-zip.sh)
    python3 make_zip.py --with-config    # also your calphad_agent.toml (settings; contains a key if you pasted one)
    python3 make_zip.py --with-outputs   # also calphad_outputs/ and outputs/ (results, test outputs)
    python3 make_zip.py -o /some/where/name.zip

Left out: virtual environments (machine-specific - install.sh builds a fresh one), caches, installer-generated
files (start-agent.*, pycalphad-agent-env.sh) and, unless asked, your settings and outputs.
On the other computer: unzip, cd into the folder, ./install.sh (Windows: install.bat).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# directory names skipped anywhere in the tree
SKIP_DIRS = {".venv", "venv", "pycalphad-agent-venv", "__pycache__", ".pytest_cache", ".git", ".claude",
             ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", "node_modules"}
SKIP_DIR_GLOBS = ["*.egg-info"]
# files skipped anywhere
SKIP_FILES = [".DS_Store", "*.pyc", "*.pyo", "Thumbs.db", "*~"]
# top-level entries skipped unless asked for
TOP_GENERATED = {"start-agent.sh", "start-agent.bat", "pycalphad-agent-env.sh"}
TOP_CONFIG = {"calphad_agent.toml"}
TOP_OUTPUTS = {"calphad_outputs", "outputs"}


def wanted(rel: Path, with_config: bool, with_outputs: bool) -> bool:
    parts = rel.parts
    if any(p in SKIP_DIRS or any(fnmatch.fnmatch(p, g) for g in SKIP_DIR_GLOBS) for p in parts[:-1]):
        return False
    if any(fnmatch.fnmatch(parts[-1], g) for g in SKIP_FILES):
        return False
    top = parts[0]
    if top in TOP_GENERATED or (top in TOP_CONFIG and not with_config) or (top in TOP_OUTPUTS and not with_outputs):
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", help="zip file to write (default: ../calphad-agent.zip)")
    ap.add_argument("--with-config", action="store_true", help="include calphad_agent.toml")
    ap.add_argument("--with-outputs", action="store_true", help="include calphad_outputs/ and outputs/")
    a = ap.parse_args(argv)
    out = Path(a.output).resolve() if a.output else HERE.parent / "calphad-agent.zip"
    root_name = HERE.name
    files = []
    for dirpath, dirnames, filenames in os.walk(HERE):
        d = Path(dirpath)
        rel_d = d.relative_to(HERE)
        # prune skipped directories early (fast even with a 1 GB .venv next to the code)
        dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS and not any(fnmatch.fnmatch(n, g) for g in SKIP_DIR_GLOBS)
                       and not (rel_d == Path(".") and ((n in TOP_OUTPUTS and not a.with_outputs)))]
        for f in filenames:
            rel = (rel_d / f) if rel_d != Path(".") else Path(f)
            p = d / f
            if p.resolve() == out or p.is_symlink() and not p.exists():
                continue
            if wanted(rel, a.with_config, a.with_outputs):
                files.append((p, rel))
    files.sort(key=lambda t: str(t[1]))
    if a.with_config and (HERE / "calphad_agent.toml").exists():
        txt = (HERE / "calphad_agent.toml").read_text(errors="ignore")
        import re
        keys = re.findall(r'(?m)^\s*api_key\s*=\s*"([^"]*)"', txt) + re.findall(r"(?m)^\s*api_key\s*=\s*'([^']*)'", txt)
        if any(k.strip() for k in keys):
            print("WARNING: calphad_agent.toml contains an API key - the zip now does too. Keep it private.")
    # write to a temporary file first, then replace: a half-written zip never overwrites the previous one
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=str(out.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p, rel in files:
                z.write(p, f"{root_name}/{rel.as_posix()}")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    size = out.stat().st_size
    total = sum(p.stat().st_size for p, _ in files)
    print(f"wrote {out}  ({len(files)} files, {total / 1e6:.1f} MB -> {size / 1e6:.1f} MB zipped)")
    extras = [n for n, on in (("settings", a.with_config), ("outputs", a.with_outputs)) if on]
    print("included: code, examples, tests, docs, installers" + (", " + ", ".join(extras) if extras else ""))
    print(f"next: copy it over, then  unzip {out.name} && cd {root_name} && ./install.sh   (Windows: install.bat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
