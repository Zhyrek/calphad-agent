#!/usr/bin/env bash
# Installer for the CALPHAD agent (macOS / Linux).
#
#   ./install.sh            # finds Python 3.11+ (newest first), creates ./pycalphad-agent-venv, installs
#                           # everything, runs the setup questions and writes ./start-agent.sh
#   ./install.sh --yes      # no questions (defaults; edit calphad_agent.toml afterwards)
#   PYTHON=/path/to/python3.13 ./install.sh     # use a specific Python
#   ./install.sh --which-python  # only show which Python / module would be used
# On a cluster without Python 3.13 on the PATH it finds a Python module (Lmod / Environment Modules: python/3.13,
# Python/3.13.1-GCCcore-..., ...) and loads it; pycalphad-agent-env.sh records the module for later use.
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"
VENV="$HERE/pycalphad-agent-venv"
YES=""
WHICH_ONLY=""
for a in "$@"; do
  case "$a" in
    --yes) YES="--yes" ;;
    --which-python) WHICH_ONLY=1 ;;       # only report which Python (and module) would be used
  esac
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
pyver() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
ok311() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; }

# 1. a Python >= 3.11: 3.13 on the PATH, else a python module (clusters: Lmod / Environment Modules),
#    else the newest 3.11+ on the PATH
PY="${PYTHON:-}"
MODULE_INIT=""      # file sourced to get the `module` command (if it was not already a shell function)
MODULE_USED=""      # module loaded for this Python
FALLBACK=""
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && ok311 "$(command -v "$c")"; then
      if [ "$(pyver "$(command -v "$c")")" = "3.13" ]; then PY="$(command -v "$c")"; break; fi
      [ -z "$FALLBACK" ] && FALLBACK="$(command -v "$c")"
    fi
  done
fi

module_init() {
  if type module >/dev/null 2>&1; then return 0; fi
  for f in "${CALPHAD_MODULE_INIT:-}" "${LMOD_PKG:+$LMOD_PKG/init/bash}" "${MODULESHOME:+$MODULESHOME/init/bash}" \
           /usr/share/lmod/lmod/init/bash /opt/lmod/lmod/init/bash /etc/profile.d/lmod.sh \
           /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [ -n "$f" ] && [ -f "$f" ]; then
      # shellcheck disable=SC1090
      . "$f" >/dev/null 2>&1 || true
      if type module >/dev/null 2>&1; then MODULE_INIT="$f"; return 0; fi
    fi
  done
  return 1
}

# best python module: 3.13, then 3.12, then 3.11 (newest patch first); 3.14+ only if nothing else
pick_module() {
  { module -t avail python 2>&1; module -t avail Python 2>&1; module -t avail python3 2>&1; } \
    | tr -d '\r' | sed -e 's/(default)//; s/(D)//; s/<[^>]*>//g; s/[[:space:]]*$//' | grep -v ':$' \
    | grep -Ei '^(cray-)?python3?[/-][^/]*3\.[0-9]+' | sort -u \
    | awk '{ n=$0; if (match(n, /3\.[0-9]+(\.[0-9]+)?/)) { v=substr(n, RSTART, RLENGTH); split(v, p, ".");
             minor=p[2]+0; patch=(p[3]==""?0:p[3]+0);
             rank = (minor==13?0 : minor==12?1 : minor==11?2 : minor>13?3 : 9);
             if (rank < 9) printf "%d %06d %s\n", rank, 999999-patch, n } }' \
    | sort | head -1 | cut -d' ' -f3-
}

if [ -z "$PY" ] && [ "$(uname -s)" = "Linux" -o -n "${CALPHAD_MODULE_INIT:-}" ] && module_init; then
  M="$(pick_module || true)"
  if [ -n "$M" ]; then
    say "Loading the module $M"
    if module load "$M" >/dev/null 2>&1; then
      for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 && ok311 "$(command -v "$c")"; then
          PY="$(command -v "$c")"; MODULE_USED="$M"; break
        fi
      done
    fi
    [ -z "$MODULE_USED" ] && echo "  (module load $M did not give a Python 3.11+; it may need a compiler module first - see 'module spider python')"
  fi
fi
[ -z "$PY" ] && PY="$FALLBACK"
if [ -z "$PY" ]; then
  echo "No Python 3.11 or newer found (on the PATH or as a module)."
  echo "  On a cluster: 'module spider python' shows the Python modules; load one (and any compiler it needs),"
  echo "  then run ./install.sh again - or run it as: PYTHON=/path/to/python3.13 ./install.sh"
  echo "  On a workstation: install Python 3.13 (https://www.python.org/downloads/, 'brew install python@3.13')."
  exit 1
fi
say "Using $("$PY" --version) at $PY${MODULE_USED:+ (module $MODULE_USED)}"
if [ -n "$WHICH_ONLY" ]; then echo "PY=$PY"; echo "MODULE=$MODULE_USED"; echo "MODULE_INIT=$MODULE_INIT"; exit 0; fi

# 2. the virtual environment
if [ -d "$VENV" ]; then
  say "Updating the existing environment pycalphad-agent-venv"
else
  say "Creating the environment pycalphad-agent-venv"
  "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python"

# 3. the packages (pycalphad, ESPEI, matplotlib, ...)
say "Installing packages (a few minutes the first time)"
"$VPY" -m pip install --upgrade pip --quiet
"$VPY" -m pip install -e "$HERE" --quiet
"$VPY" -c "import pycalphad, espei, calphad_tools; print('  pycalphad', pycalphad.__version__, '| ESPEI', espei.__version__, '| OK')"

# 3b. a Flux scheduler here? (ESPEI optimisations are then submitted as batch jobs; probe only, nothing submitted)
say "Checking for a Flux scheduler"
"$VPY" -m calphad_tools.hpc || echo "  (the check failed; calphad-agent --check repeats it)"

# 4. settings (on a Flux system the setup also asks for the Flux bank and queue)
say "Settings"
"$VENV/bin/calphad-agent" --setup $YES
# Flux found but no bank/queue in the config yet (e.g. --yes): say what to set
"$VPY" -m calphad_tools.hpc --remind-only --config "$HERE/calphad_agent.toml" || true

# 5. the environment file (module + venv; sourced by the launcher and by generated batch scripts) and a launcher
{
  echo "# Sets up the CALPHAD agent's Python: . $HERE/pycalphad-agent-env.sh  (written by install.sh)"
  if [ -n "$MODULE_USED" ]; then
    if [ -n "$MODULE_INIT" ]; then echo "type module >/dev/null 2>&1 || . \"$MODULE_INIT\""; fi
    echo "module load $MODULE_USED"
  fi
  echo ". \"$VENV/bin/activate\""
} > "$HERE/pycalphad-agent-env.sh"
cat > "$HERE/start-agent.sh" <<EOF
#!/usr/bin/env bash
# Starts the CALPHAD agent and opens it in your default browser (on a desktop; --no-browser to stop that).
# ./start-agent.sh --terminal for the terminal version.
cd "$HERE"
. "$HERE/pycalphad-agent-env.sh"
# fixed hash seed: pycalphad's mapping traces the same lines on every run (keeps a PYTHONHASHSEED you set)
export PYTHONHASHSEED="\${PYTHONHASHSEED:-0}"
if [ "\${1:-}" = "--terminal" ]; then shift; exec calphad-agent "\$@"; fi
exec calphad-agent --web "\$@"
EOF
chmod +x "$HERE/start-agent.sh"

command -v pdftoppm >/dev/null 2>&1 || echo "
  Optional: reading figures from PDF pages needs poppler ('brew install poppler' / 'sudo apt install poppler-utils')."

say "Done."
[ -n "$MODULE_USED" ] && echo "  Python comes from the module $MODULE_USED; start-agent.sh and pycalphad-agent-env.sh load it for you."
echo "  Start it:            ./start-agent.sh          (opens http://127.0.0.1:8765 in your browser)"
echo "  Terminal version:    ./start-agent.sh --terminal"
echo "  Change settings:     edit calphad_agent.toml, or: pycalphad-agent-venv/bin/calphad-agent --setup"
echo "  Test the connection: pycalphad-agent-venv/bin/calphad-agent --check   (also shows Flux / local)"
