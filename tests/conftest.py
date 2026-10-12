"""Test-wide settings.

Many tests run generated standalone scripts, jobs and diagram workers in fresh Python processes.  On macOS the
default matplotlib backend ("macosx") makes each of those processes flash a Python icon in the Dock, so every
child started from the test run inherits the non-GUI Agg backend instead.

Structure lookups for compound labels (structure_id) never go online in tests (CALPHAD_STRUCTURE_ONLINE=0, unless
the environment sets it).
"""
import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("CALPHAD_STRUCTURE_ONLINE", "0")
