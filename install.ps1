# Installer for the CALPHAD agent (Windows PowerShell).  Double-click install.bat, or run:
#   powershell -ExecutionPolicy Bypass -File install.ps1          (add -Yes for no questions)
# Finds Python 3.11+ (newest first), creates .\pycalphad-agent-venv, installs everything, runs the setup
# questions and writes start-agent.bat.
param([switch]$Yes)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Here = (Get-Location).Path
$Venv = Join-Path $Here "pycalphad-agent-venv"

function Say($m) { Write-Host ""; Write-Host $m -ForegroundColor Cyan }

# 1. a Python >= 3.11 (the py launcher first, newest version first)
$Py = $null
$PyArgs = @()
if ($env:PYTHON) { $Py = $env:PYTHON }
else {
    foreach ($v in @("3.13", "3.12", "3.11")) {
        try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $Py = "py"; $PyArgs = @("-$v"); break } } catch {}
    }
    if (-not $Py) {
        foreach ($c in @("python3", "python")) {
            try {
                & $c -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
                if ($LASTEXITCODE -eq 0) { $Py = $c; break }
            } catch {}
        }
    }
}
if (-not $Py) {
    Write-Host "No Python 3.11 or newer found. Install Python 3.13 from https://www.python.org/downloads/"
    Write-Host "(tick 'Add python.exe to PATH'), then run install.bat again."
    exit 1
}
Say ("Using " + (& $Py @PyArgs --version))

# 2. the virtual environment
if (Test-Path $Venv) { Say "Updating the existing environment pycalphad-agent-venv" }
else { Say "Creating the environment pycalphad-agent-venv"; & $Py @PyArgs -m venv $Venv }
$VPy = Join-Path $Venv "Scripts\python.exe"

# 3. the packages
Say "Installing packages (a few minutes the first time)"
& $VPy -m pip install --upgrade pip --quiet
& $VPy -m pip install -e $Here --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed (see above)"; exit 1 }
& $VPy -c "import pycalphad, espei, calphad_tools; print('  pycalphad', pycalphad.__version__, '| ESPEI', espei.__version__, '| OK')"

# 4. settings
Say "Settings"
$Agent = Join-Path $Venv "Scripts\calphad-agent.exe"
if ($Yes) { & $Agent --setup --yes } else { & $Agent --setup }

# 5. a launcher
@"
@echo off
rem Starts the CALPHAD agent in the browser.  start-agent.bat --terminal for the terminal version.
cd /d "$Here"
rem fixed hash seed: pycalphad's mapping traces the same lines on every run (keeps a PYTHONHASHSEED you set)
if not defined PYTHONHASHSEED set PYTHONHASHSEED=0
rem The web UI opens your default browser by itself once it is ready (--no-browser to stop that).
if "%1"=="--terminal" ( "$Agent" %2 %3 %4 %5 ) else ( "$Agent" --web %* )
"@ | Set-Content -Encoding ASCII (Join-Path $Here "start-agent.bat")

Say "Done."
Write-Host "  Start it:            start-agent.bat   (opens http://127.0.0.1:8765 in your browser)"
Write-Host "  Terminal version:    start-agent.bat --terminal"
Write-Host "  Change settings:     edit calphad_agent.toml, or: pycalphad-agent-venv\Scripts\calphad-agent --setup"
Write-Host "  Test the connection: pycalphad-agent-venv\Scripts\calphad-agent --check"
Write-Host "  Optional: reading figures from PDF pages needs poppler (https://github.com/oschwartz10612/poppler-windows)."
