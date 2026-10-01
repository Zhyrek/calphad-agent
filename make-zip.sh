#!/usr/bin/env bash
# Packs this folder into ../calphad-agent.zip (see make_zip.py --help for options). Rerun after changes.
cd "$(dirname "$0")" && exec python3 make_zip.py "$@"
