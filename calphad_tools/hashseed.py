"""Python's hash seed and reproducible phase-diagram mapping.

pycalphad's mapping strategies iterate over sets (of phases, composition sets, ...), and the iteration order of a
set of strings depends on Python's hash seed (``PYTHONHASHSEED``), which is random per process unless it is set.
So two runs of the same diagram can trace slightly different ZPF lines (an extra short segment, a line traced from
the other end); the phase boundaries and invariants agree, but the files are byte-identical only with the same seed.

The toolkit therefore runs with ``PYTHONHASHSEED=0`` unless the user chose a seed:

* ``calphad-agent`` / ``calphad-tools`` re-execute themselves once at start-up with the variable set
  (:func:`reexec_with_seed`) - a running interpreter cannot change its own seed, and the agent maps single diagrams
  in its own process, so this is the one place that covers every tool;
* worker / job processes inherit it (:func:`for_children` around the spawn, explicit ``env`` for jobs), which also
  covers library use (tests, Jupyter) where nothing re-executes;
* the seed a figure was computed with is recorded (:func:`current`) in its JSON summary and provenance, and the
  reproduction notebook checks it.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading

DEFAULT = "0"
REEXEC_FLAG = "CALPHAD_HASHSEED_REEXEC"
_AT_IMPORT = os.environ.get("PYTHONHASHSEED")       # captured as early as possible (before anyone changes it)
_LOCK = threading.Lock()


def _given(value) -> bool:
    return bool(str(value or "").strip())


def current() -> str:
    """Hash seed of THIS process: "0", another number, or "random" (unset / "random" / not determinable)."""
    if sys.flags.hash_randomization == 0:
        return "0"                                  # PYTHONHASHSEED=0 at start-up disables randomization
    s = str(_AT_IMPORT or "").strip()
    if s.isdigit() and s != "0":                    # "0" with randomization on: set after start-up, not in effect
        return s
    return "random"


def child_seed() -> str:
    """The seed child processes started now get (the user's PYTHONHASHSEED, else the default)."""
    s = os.environ.get("PYTHONHASHSEED", "")
    return s.strip() if _given(s) else DEFAULT


def child_env(env=None) -> dict:
    """A copy of ``env`` (default os.environ) with PYTHONHASHSEED set to the default unless one is given."""
    out = dict(os.environ if env is None else env)
    if not _given(out.get("PYTHONHASHSEED")):
        out["PYTHONHASHSEED"] = DEFAULT
    return out


@contextlib.contextmanager
def for_children():
    """Processes started inside this block (multiprocessing spawn inherits os.environ) get PYTHONHASHSEED=0 unless
    the user set a seed; os.environ is restored afterwards."""
    with _LOCK:
        set_here = not _given(os.environ.get("PYTHONHASHSEED"))
        if set_here:
            os.environ["PYTHONHASHSEED"] = DEFAULT
        try:
            yield
        finally:
            if set_here:
                os.environ.pop("PYTHONHASHSEED", None)


def reexec_with_seed(module: str, argv=None) -> None:
    """At the start of a command-line program: if PYTHONHASHSEED is not set, run the same command again with
    PYTHONHASHSEED=0 (``python -m <module> <args>``) and never return; otherwise (or when called with an explicit
    argv, i.e. from code or tests) do nothing.  POSIX: os.execve (same process id, same terminal).  Windows has no
    real exec, so the command runs as a child and its exit code is passed on.  If that is impossible the program
    simply continues (the figures then record hash_seed "random")."""
    if argv is not None or _given(os.environ.get("PYTHONHASHSEED")) or os.environ.get(REEXEC_FLAG):
        return
    env = {**os.environ, "PYTHONHASHSEED": DEFAULT, REEXEC_FLAG: "1"}
    cmd = [sys.executable, "-m", module, *sys.argv[1:]]
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.flush()
    if os.name == "posix":
        with contextlib.suppress(OSError):
            os.execve(sys.executable, cmd, env)
        return
    import subprocess
    try:
        p = subprocess.Popen(cmd, env=env)
    except OSError:
        return
    while True:
        try:
            code = p.wait()
            break
        except KeyboardInterrupt:                   # Ctrl+C reaches the child too (same console): keep waiting
            continue
    sys.exit(code)


if __name__ == "__main__":          # python -m calphad_tools.hashseed: the seed this interpreter runs with
    print(current())
