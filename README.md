# calphad-tools

CALPHAD automation scripts that run **as-is** (`calphad-tools …`) or **driven by an LLM** (`calphad-agent`).

The LLM never runs anything itself. The agent loop runs *inside this Python process*: the model returns
tool-call requests and the script executes them locally. Every action therefore has exactly the
permissions of the process you started, and inherits its sandbox. The model only sees tool results.

| # | Feature | CLI | Agent tool |
|---|---------|-----|------------|
| 1 | Compare two TDBs parameter-by-parameter → PowerPoint table + CSV | `compare` | `compare_tdbs` |
| 1b | Compare several TDBs within one element subsystem (e.g. Fe–Cr): parameters as written, a differences table, one table per file → PowerPoint + Excel + CSV | | `compare_system`, `find_databases` |
| 2 | Every binary phase diagram (100–3000 °C) → PowerPoint (grid image per slide) | `binaries` | `plot_all_binaries` |
| 3 | Every ternary isothermal section of a quaternary+ TDB → PowerPoint | `ternaries` | `plot_all_ternaries` |
| 4 | Merge 2+ TDBs, auto-resolving obvious differences and asking about the rest | `merge` | `merge_tdbs` |
| 5 | Experimental data (POP / DATAPLOT .exp / CSV / paper PDF) → ESPEI ZPF JSON | `exp2zpf`, `pdftext` | `convert_*`, `read_document`, `view_pdf_page`, `write_zpf_dataset` |
| 6 | ESPEI optimisation, batched on Flux when detected (else run here), + Flux batch scripts | `espei-flux`, `phase-models` | `optimize_system`, `generate_espei_flux_batch`, `make_phase_models`, `espei_generate_parameters`, `espei_run_mcmc` |
| 7 | Natural-language pycalphad chat (terminal or browser) | `calphad-agent` | equilibria, isotherms, isopleths, step plots, G–x curves, `run_python` |

## Install (any computer)

You need **Python 3.11 or newer**; 3.13 is recommended, and every dependency has ready-made packages for it on macOS, Windows and Linux.

To move it, run `./make-zip.sh` (or `python3 make_zip.py`) in this folder. It writes `../calphad-agent.zip`, about 1 MB, and leaves out the virtual environment, caches, your settings and outputs. Copy the zip over and unzip it. Run it again whenever the code changes. `--with-config` also packs your `calphad_agent.toml` (with a warning if it contains a key), and `--with-outputs` packs your results.

Then run the installer from inside the folder:

* **macOS / Linux:** `./install.sh`
* **Windows:** double-click `install.bat` (or `powershell -ExecutionPolicy Bypass -File install.ps1`)

The installer:

1. finds the newest Python 3.11+ on the machine;
2. creates the environment `pycalphad-agent-venv/` in this folder;
3. installs pycalphad, ESPEI and the rest;
4. asks a few questions and writes `calphad_agent.toml`;
5. offers to test the connection;
6. writes a launcher, `start-agent.sh` (or `start-agent.bat`).

Add `--yes` (`-Yes` on Windows) to skip the questions and use the defaults. To pick a specific Python: `PYTHON=/path/to/python3.13 ./install.sh`.

Then set your key in the environment variable you named, and start the agent:

```bash
export SOL_API_KEY="your-key"      # Windows: setx SOL_API_KEY "your-key"  (then open a new window)
./start-agent.sh                    # browser UI at http://127.0.0.1:8765   (--terminal for the terminal version)
```

The browser UI opens in your default browser as soon as it is ready. Starting the launcher again while the agent is already running just opens another tab on it.

**A secret link per session.** Each session has its own secret link (`http://127.0.0.1:8765/?token=…`, printed in the terminal). On a shared login node, other users can reach the same port, but without the link they only get "forbidden". The link's token is swapped for a cookie, and a second launch only reuses an agent it can prove is yours.

When the browser opens:

* **A Mac, Windows or Linux desktop:** yes.
* **A VNC desktop** (`DISPLAY=:1`), even if the VNC session was started from an SSH login: yes.
* **An SSH session with X forwarding** (`ssh -X`/`-Y`, `DISPLAY=localhost:10.0`), e.g. a RealVNC desktop whose terminal SSHes into the cluster: yes. If `firefox &` on the cluster shows up in your VNC window, `./start-agent.sh` opens it there too. The browser then runs on the cluster next to the agent, so no port forwarding is needed.
* **SSH without a display:** no. It prints the exact `ssh -L …` command for the machine that has your browser, and the link to open there.
* **Choosing the browser:** `BROWSER=firefox ./start-agent.sh` picks one explicitly. If Firefox comes from a module, run `module load firefox` before starting the agent.

`--no-browser` (or `CALPHAD_NO_BROWSER=1`) turns the automatic opening off. If port 8765 is taken by another program, the next free port is used, and the terminal says which.

`pycalphad-agent-venv/bin/calphad-agent --check` tests the connection and shows whether ESPEI runs go to Flux or run here. `--setup` asks the questions again.

**On a cluster (Linux with environment modules):**

* **Python module:** if no Python 3.13 is on the PATH, `install.sh` looks for one among the modules (Lmod or Environment Modules). It recognises names like `python/3.13.1`, `Python/3.13.1-GCCcore-14.2.0` and `python3/3.12`, prefers 3.13, then 3.12, then 3.11, and loads the best one.
  * `./install.sh --which-python` shows what it would pick.
  * If the Python module needs a compiler module loaded first (hierarchical Lmod), load both yourself and run the installer again, or point it at a Python: `PYTHON=... ./install.sh`.
* **Environment file:** the installer writes `pycalphad-agent-env.sh`, which loads that module and activates the environment. `start-agent.sh` sources it. The ESPEI/Flux batch scripts the agent generates also start with it, so jobs on compute nodes use the same Python.
* **Flux:** after building the environment the installer checks for a Flux scheduler and prints what it found. If there is one, the setup also asks for your Flux bank and queue. See [Running ESPEI on a Flux cluster](#running-espei-on-a-flux-cluster).
* **Web UI:** it listens only on the machine it runs on, behind its secret link. Either:
  * start it from an SSH session with X forwarding (the cluster's Firefox opens in your VNC or X window);
  * forward the port (`ssh -L 8765:127.0.0.1:8765 you@cluster`; the agent prints the exact command) and open the printed link locally;
  * use `./start-agent.sh --terminal`.

Optional: reading figures from PDF pages needs poppler (`brew install poppler`, `apt install poppler-utils`, or the Windows build linked by the installer).

## LLM setup (GPT-5.6 Sol at a custom URL, or any OpenAI-compatible endpoint)

The setup questions fill in these fields of `calphad_agent.toml`; you can also edit them by hand:

```toml
[llm]
provider = "openai"
base_url = "https://YOUR-LLM-HOST/v1"   # your custom URL
api_key  = ""                           # your key, or leave empty and use api_key_env
api_key_env = "SOL_API_KEY"             # NAME of the environment variable that holds the key
model    = "gpt-5.6-sol"                # exactly the name your endpoint expects
api      = "responses"                  # or "chat" if your host has no /v1/responses
vision   = true                         # false if the endpoint rejects images

[agent]
workdir     = "examples"                # folder with your TDBs and data
python_tool = "auto_verify"             # ask | auto_verify | off
```

**Rate limits and token use.** On a 429 ("exceeded token rate limit") the agent resends the *same* request. By default (`rate_limit_mode = "aggressive"`) it asks again within `rate_limit_retry_cap_s` (5 s, with a little jitter; a longer Retry-After is capped too) and never slows itself down: on a quota shared by many keys, the tokens go to whoever asks first. The retries of one request share one status line, e.g. "rate limit (429) - resending within 5 s (retry ×6, 21 s so far)". It gives up after `rate_limit_max_wait_s` (300 s) in total. `rate_limit_mode = "polite"` restores the old behaviour for a quota of your own: wait as long as the endpoint asks and, with `tokens_per_minute = "auto"`, learn a pace from the first limit (about 80% of the last minute) and stay below it. A fixed `tokens_per_minute` paces in both modes; 0 disables pacing.

The conversation is append-only, and each conversation sends one `prompt_cache_key`, so repeated requests reuse the provider's prompt cache. That makes them cheaper and faster. Whether cached tokens also count toward your TPM quota is up to the provider. `/usage` and the web header show input tokens, the cached share, and the current pace.

**Tool descriptions on demand.** By default (`tool_descriptions = "compact"` under `[agent]`), only the about 15 everyday tools are sent in full: databases, equilibrium, phase diagrams, plotting, viewing images, recent results, `show_code`. Each of the other about 70 tools is sent as one line plus its argument names, ending in `[tool_help]`. Before using one for the first time, the model calls `tool_help` with the names of all the tools it is about to use, in one call. The tool list stays fixed for the whole conversation, so it stays cacheable. This cuts the fixed overhead per model call from about 23k to about 9k tokens. Wrong or missing arguments are caught before the tool runs, and the error includes that tool's full help, so one retry fixes it. `tool_descriptions = "full"` restores the old behaviour.

Most of each request is fixed overhead: about 9k tokens of tool descriptions (23k with `tool_descriptions = "full"`) plus the growing conversation. Plot images are downscaled to `image_max_px` (1600 px) when they're added. `image_detail = "low"` cuts each image to about 85 tokens, at the cost of detail. Raising the deployment's TPM quota in Azure AI Foundry (Deployments → Edit → Tokens per Minute) is the direct fix if you want full speed.

Other settings you may need:

* `api`: `"responses"` (default; /responses, not streamed, so no mid-stream disconnects) or `"chat"` (/chat/completions, streamed; for hosts without /responses). Both use the same rate pacing and retry a cut-off answer with more output room. For one run, `CALPHAD_LLM_API=chat ./start-agent.sh` overrides the file; compare with `/usage`.
* Key lookup order: `CALPHAD_LLM_API_KEY`, then `api_key`, then the variable named by `api_key_env`, then `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. The key never appears in problem reports or session logs.
* `reasoning_effort`: set it to `""` if your gateway rejects the parameter. The client also drops it automatically when the gateway returns a 400.
* `extra_headers`: for gateways that want custom auth headers.
* `ca_bundle` / `verify_ssl`: for institutional certificates.
* `vision = true`: lets the model look at the plots it makes and at rendered PDF pages. Set it to `false` if the model doesn't accept images.

To use Claude instead, set `provider = "anthropic"` and `model = "claude-opus-5"`.

**Named profiles.** You can keep several model setups in one file and switch with `--profile NAME`. Each `[profiles.NAME]` section is a complete alternative to `[llm]`: fields it leaves out take the built-in defaults, not `[llm]`'s values. So a key set in `[llm]` is never sent to a different provider.

**Claude Code CLI (`provider = "claude_cli"`).** This uses your Claude Code login through `claude -p`, so no API key is needed. How it runs:

* Claude Code's own tools are switched off, and it runs from a neutral directory with no settings or MCP servers.
* The child process gets no Claude Code session variables from its parent.
* Our tools are described in the system prompt. The model calls them with `<tool_call>{json}</tool_call>` blocks, the results come back as `<tool_result>` blocks, and the conversation continues with `--resume`.
* The code reviewer runs as a separate `claude -p` with no session history.

* Images: with `vision = true`, the images the model is meant to look at go in as image blocks of a stream-json message. These are `view_image`, the digitizer previews and the images you attach. Other plots are referenced by file path.

Limitations: each step is a whole CLI invocation, which is slower. It works as a stand-in, or for any model that lacks native tool calling.

### Arbitrary Python: `python_tool`

`python_tool` controls what happens when the model writes its own Python:

| Mode | Behaviour |
|------|-----------|
| `ask` (default) | Shows you each script and waits for your OK. |
| `auto_verify` | Runs the script automatically if it passes both checks below; otherwise asks you. |
| `auto` | Runs everything without review. |
| `off` | The Python tool is removed. |

In `auto_verify` mode, each script goes through two checks:

1. **Static scan.** A deterministic scan of the code looks for shell and subprocess calls, network access, `exec`/`eval` of built strings, sensitive paths such as `~/.ssh` or the config file, and similar. A "hard" flag here always means a human decides.
2. **Independent review.** The code goes to a fresh model call. That call has no conversation history and no tools. It sees only a fixed safety policy, the directories, the code, and any earlier snippets whose definitions the code uses (the Python namespace persists between runs). It must answer `safe`, `unsafe` or `unsure`. Overwriting or deleting files the tool itself made in the output directory counts as fine. Comments that try to talk to the reviewer are treated as a red flag.

The script runs automatically only if the reviewer says `safe` and the scan found no hard flags. Otherwise:

* In the terminal or browser, you're asked, with the reviewer's reasons shown.
* In `--once` runs, the script is declined and the reasons go back to the model so it can rewrite the code.

Every review is logged to `calphad_outputs/python_reviews.jsonl`. To turn this mode on:

* browser: tick the **auto-run Python after independent safety review** checkbox
* terminal: `/autorun on`
* command line: `calphad-agent --auto-verify`

Use `reviewer_model` under `[llm]` to run the review on a different model.

`tests/live_check.py` tests a real endpoint. It lists the model IDs your key can see, runs the reviewer on benign and malicious snippets, and runs one agent turn.

## Using it

### Chat (feature 7, and every other feature by asking)

```bash
calphad-agent                    # terminal chat
calphad-agent --web              # browser UI at http://127.0.0.1:8765 (inline plots, question cards)
calphad-agent --once "compare CrFe-82Her.tdb and Cr-Fe_her.tdb"
```

Example requests:

* "plot the ternary isotherm at 1000K for Co-Cr-Ni in CoCrFeNiV_Choi2019.tdb"
* "phase fractions vs temperature for Fe-18Cr-8Ni (mole fraction) from 400 to 1600 °C"
* "merge CrFe-82Her.tdb, CrNi-06Hal.tdb and FeNi-00Ans-2SL.tdb into CrFeNi.tdb". It will ask you about the real conflicts.
* "extract the liquidus data from Table 2 of paper.pdf into ESPEI datasets"
* "make a Flux batch for ESPEI on 4 nodes of pbatch for the Cr-Ni datasets"

### Without an LLM

```bash
calphad-tools compare   A.tdb B.tdb -o comparison.pptx            # --all to include identical rows
calphad-tools binaries  my.tdb -o binaries.pptx                   # --tmin/--tmax °C, --exclude GAS
calphad-tools ternaries my.tdb --temps 800 1000 1200 -o tern.pptx
calphad-tools merge     A.tdb B.tdb C.tdb -o merged.tdb           # interactive
calphad-tools merge     A.tdb B.tdb -o merged.tdb --prefer first  # never asks
calphad-tools exp2zpf   data.POP --components CR NI --tdb my.tdb -o datasets/
calphad-tools exp2zpf   plot.exp --components CR NI -o datasets/  # asks which phases each DATASET is
calphad-tools exp2zpf   digitized.csv --components CR NI -o datasets/
calphad-tools phase-models my.tdb --elements CR NI -o phase_models.json
calphad-tools espei-flux examples/espei_campaign.yaml -o espei_runs/ && bash espei_runs/submit_all.sh
```

## What the chat agent can do without arbitrary Python or internet

Every plotting and data-output capability of pycalphad 0.11.2 is available as a dedicated tool, so
`python_tool = "off"` loses nothing. `docs/pycalphad_coverage.md` maps each item of the 152-item
inventory (`docs/pycalphad_inventory.md`) to its tool.

| Tool | What it does |
|------|--------------|
| `map_phase_diagram` | One binary (optionally in wt% or on a T–MU axis), ternary isotherm, or isopleth. Writes the PNG plus CSV/JSON files of the phase boundaries, tie-lines and invariant reactions, and classifies the reactions (e.g. eutectic, peritectic). |
| `step_map` | Phase fractions, phase compositions or any property against T, P or X, plus the transition temperatures. |
| `equilibrium_property_plot` | Any Workspace property over one or two varying conditions. See the list below. |
| `phase_property_plot` | Single-phase G/H/S/Cp and mixing-property curves: common tangents, reference states, one phase's property against T. |
| `export_equilibrium_data` | The raw equilibrium dataset as netCDF plus a tidy CSV, and `eqplot`. |
| `plot_csv` | Re-plots or overlays columns from any CSV output. |
| `scheil_solidification` | Scheil and equilibrium solidification curves. |
| `write_database` | Normalises a TDB, or writes a subset of its elements/phases. With `dialect` (`thermocalc`, `opencalphad`, `pycalphad`) it keeps to that program's limits and lists every renamed function or phase. |
| `convert_database` | Converts a ChemSage/FactSage `.dat` file to a TDB and says what could not be converted. |
| `calc_cache` | Shows or clears the calculation cache. |
| `export_phase_model` | The symbolic Gibbs energy contributions and parameter list of a phase. |
| `plot_diagram_with_data` | A calculated binary diagram or ternary isotherm with ESPEI experimental datasets overlaid. |
| `plot_parameter_vs_data` | A TDB endmember or interaction property against ESPEI thermochemical data. |
| `convert_composition` | Converts compositions between wt% and at%. |
| `pycalphad_help` | Offline pycalphad API docs, taken from the installed package. |
| `list_calphad_capabilities` | A task → tool map for the model. |

### Editing plots and making new ones

Every figure the tools save also stores an editable matplotlib copy in `calphad_outputs/.figures`. That lets the model act on requests like "plot these specific data points on the image with these colours".

**`describe_figure`** lists a plot's axes, limits and scales as displayed. It also records display transforms: a kelvin axis labelled in °C, or fractions shown as percent. It shows the legend's phase → colour map too.

**`edit_figure`** draws onto the real axes, using the values shown on the ticks (e.g. °C):

* scatter points (per-point colours or a colourmap with a colour bar), lines, steps, error bars, bars, histograms and contours
* text, annotations with arrows, horizontal/vertical lines and bands
* rectangles, circles, ellipses and polygons
* data given inline or read from CSV columns

It can also change what's already there:

* restyle or remove artists by label, by phase name from the legend (e.g. make SIGMA magenta), or by colour
* set limits, log scales, labels, ticks, grid, title and legend (existing phase entries are kept)
* add twin axes and zoom insets
* change the figure size or suptitle
* export PDF/SVG in addition to PNG

Edits can be chained. Anything placed far outside the plot, usually a sign of the wrong units, is refused with an explanation.

For images the tools didn't produce, it draws on top of the image. Positions are either fractions of the image, or data values after you give a pixel calibration of the plot frame.

**`make_plot`** builds new single- or multi-panel figures from inline data or CSV outputs. It supports all the operations above, matplotlib styles, and PNG/PDF/SVG output.

### Which element goes on which side

You choose the sides; the data follow.

* **Binary:** `right: "CR"` puts Cr on the right. The x axis is then the fraction of Cr, running from pure Fe
  (x = 0) to pure Cr (x = 1).
* **Ternary isotherm:** `corners: {"left": "NI", "right": "FE", "top": "CR"}` puts Ni bottom-left, Fe
  bottom-right and Cr at the top. The corners are labelled on the figure.
* **Isopleth:** the x axis is the `vary` element (`right` means the same thing).
* **Default:** without these options the element order decides, as before. `map_phase_diagram`: binary
  [left, right], ternary [bottom-left, bottom-right, top]. `plot_diagram_with_data`: alphabetical.
* **Decks:** `plot_all_binaries` and `plot_all_ternaries` take `element_order`, e.g. `["FE", "CR", "NI"]`. Earlier
  elements go left (binaries) or bottom-left (ternaries) in every subsystem. `orientation` overrides single
  systems, e.g. `{"CR-NI": ["NI", "CR"]}`.

The result text states the orientation used, e.g. "x axis = mole fraction Cr; Fe on the left, Cr on the right".
The exported CSV files store compositions by element name (`X(FE)`, `X(CR)`), so they don't depend on the
orientation.

**Data follow the axes.** `plot_diagram_with_data` places the ZPF points on the chosen sides. The datasets store
only the independent components, so the balance element is computed. On a ternary, only points within
`T_tolerance` (default 10 K) of the isotherm are shown. Tie-lines are drawn as segments, and a tie-line with one
unknown end as a single point. To add your own points to any diagram, use the `edit_figure` operation
`composition_points`. You give compositions and the operation places them:

```json
{"op": "composition_points", "points": [{"X(FE)": 0.3, "X(NI)": 0.2, "T": 1273}], "unit": "mole_fraction",
 "T_unit": "K", "color": "k", "label": "measured"}
```

The missing (balance) element is computed. `unit` can also be `at%`, `wt%` or `weight_fraction`, and `T_unit`
can be `C`. `tielines: [[0, 1]]` joins points 0 and 1 with a tie-line. The points can also come from a CSV file
with element columns (`X(FE)`, `Fe`, `wt% Cr`). To turn a digitized column such as "x = at% Fe" into element
columns, use the `transform_data` step `element_fractions`:
`{"step": "element_fractions", "cols": {"x": "FE"}, "elements": ["CR", "FE"], "unit": "at%"}`.

`digitized_to_zpf` takes the same `right` / `corners` options for its data plot. The ZPF files it writes store compositions by element, so they are the same whichever way the plot is drawn.

### Diagram style: shaded (default) or color

Phase diagrams (`map_phase_diagram`, `plot_all_binaries`, `plot_all_ternaries`, `plot_diagram_with_data`) are drawn
in the **shaded** style by default:

* **Binary T–x:** black phase boundaries and invariant lines (1.25 pt), two-phase fields filled light blue
  (`aliceblue`), no tie-lines, minor ticks on and all ticks pointing in on all four sides.
* **Ternary isotherm:** grey (#999999) phase boundaries and tie-triangle edges, two-phase fields filled light blue
  (three-phase triangles stay white), no tie-lines.
* **Isopleth:** black ZPF lines and invariant lines; every region with an even number of phases (two-phase, and in a
  quaternary isopleth four-phase) filled light blue, one- and three-phase regions white - as the two-phase fields of
  a binary. The regions are not bounded by tie-lines in an isopleth, so they are found from the drawn lines (see
  *Isopleth regions* below); the fill is snapped onto the black lines.
* **Labels:** black text with no background box, so the fill shows through.
* **No legend:** the phases are named by labels inside the figure. `label_regions` is on by default in this style
  (pass `label_regions: false` to leave it out; an isopleth then names each ZPF line on the line). T–μ diagrams
  can't have region labels, so each line is named on the line instead. With experimental data, the legend lists the
  data references only, and it sits inside the axes.
* **Files:** PNG, plus SVG and PDF with editable text (`formats` changes that).

Titles, axis labels and corner labels are the same in both styles. For the coloured look (one colour per phase,
grey tie-lines, red invariants, a phase legend), pass `style: "color"` in a call, or make it the default:

```toml
[agent]
diagram_style = "color"     # shaded (default) | color
```

On the command line: `calphad-tools binaries my.tdb --style color`. `show_code` reproduces the style the tool used.
The drawing code is in `calphad_tools/diagram_style.py`; standalone scripts contain a copy of it.

### Mapping check

pycalphad's mapping can drop a boundary (Fe–Ni: the α/γ line below 1184 K) or trace a line past its end node
(Al–Cr–Ni 1200 °C: a B2+FCC line past the B2 + γ + γ′ triangle), so every binary T–x diagram and ternary isotherm
(`map_phase_diagram`, the `plot_all_*` overviews, `plot_diagram_with_data`, `staged_merge`) is checked against point
equilibria after mapping: metastable tie-lines are trimmed, duplicate traces and invariants dropped, invariants
that are not equilibria are removed (Ce–Co: BCC + Ce₅Co₁₉ + L at pure Ce's melting point), invariants that
pycalphad rejects as metastable are added when point equilibria confirm them (Fe–Ni: BCC + FCC + FCC#2 at 683 K), and
two-phase areas of an equilibrium grid that no traced line bounds are traced again; the result says what was changed or could not be
repaired in one `mapping check:` line, with details in the JSON. `check_mapping: "off"` gives pycalphad's raw lines
and `"thorough"` validates more points; the code is `calphad_tools/mapping_check.py`, which `show_code` copies.
T–μ diagrams are not checked. Isopleths are checked through their regions (next section): an area between the
traced lines that holds two phase sets is a boundary the mapping missed - short step maps through it seed new lines
(as pycalphad seeds an isopleth) and it is traced; a ZPF line with the same phases on both sides is reported as no
phase boundary. The result has one `isopleth check:` line, details under `mapping_check` in the JSON.

### Isopleth regions

In an isopleth a ZPF line separates a region with its zero-fraction phase from one without it, so the regions
(for the fill and the labels) are found from the drawn lines, in `calphad_tools/isopleth_regions.py`:

1. **Areas.** The ZPF lines and invariant lines are walls on a 0.25 pt raster of the axes; a line end within 2 pt
   of another line or the frame is joined to it. The connected areas between them are the candidate regions.
2. **Names.** A line with phase set S has S on one side and S without its zero-fraction phase on the other, so the
   set all the lines around an area allow is its name (FCC+L1₂ between the γ′ solvus and the γ solvus). Where the
   lines leave two candidates, an equilibrium grid (80 × 60 nodes, `pdens` 500) inside the area decides, weighted by
   the nodes' distance from the lines; with no node inside, the lower Gibbs energy of the two at the area's thickest
   point. Ordered/disordered states are named as on the lines (B2 disordered = BCC).
3. **Check.** Grid nodes at least one grid spacing inside an area must agree with its name. A different phase set
   on 4+ nodes counts only if it is lower in Gibbs energy there (restricted equilibria, denser sampling if needed):
   pycalphad's global minimisation misses e.g. γ′ next to γ in Al–Cr–Ni, or the better of the two σ descriptions in
   Choi 2019 Fe–Cr–Ni - those nodes are reported as corrected, not as a region. A real second phase set is a missing
   boundary (traced, see *Mapping check*); if it cannot be traced, the area is split by the nearest grid node and the
   result says so.
4. **Outlines.** Each region's outline is a contour of its raster cells with every vertex near a line moved onto
   that line and the vertices beyond the frame onto the frame, so the fill neither crosses nor falls short of the
   black lines.

The JSON has `phase_regions` (phases, number of phases, shaded, extent, area fraction, what named it). The regions
are computed only in the shaded style or with `label_regions` (an isopleth in the colour style without labels is as
before). Cost: the grid, checks and labels added 15-25 s to the test isopleths (Fe–Cr–Ni 12 → 27 s, Al–Cr–Ni 49 → 73 s).

### Phase labels on diagrams

`map_phase_diagram` (binary T–x, ternary isotherms and isopleths) and `plot_diagram_with_data` take `label_regions: true` (the default in the shaded style).
The phases of each region are then written on the diagram, and on binaries and isopleths the invariant temperatures too. To label
a diagram you already made, use the `edit_figure` operation `label_phase_regions`. That works after data points were
added, and the labels avoid those points.

* **Names.** Common names are shortened: BCC_A2 → BCC, LIQUID → L, L12_FCC → L1₂, SIGMA → σ, MU_PHASE → μ,
  GAMMA_PRIME → γ′, LAVES_C14 → C14. Formula names are split into the database's elements: FE3CR4 → Fe₃Cr₄,
  M23C6 → M₂₃C₆, COSI → CoSi. A miscibility gap is BCC+BCC. `phase_labels` overrides any name, e.g.
  `{"SIGMA": "σ-phase"}`. The rules are a table in `calphad_tools/phase_names.py`.
* **Compound names.** A stoichiometric phase's own label is its formula from the sublattice model, plus a family
  name for well-known structures: "Fe₂Nb (C14)", "CeCo₂ (C15)", "CeCo₅ (1:5)", "Ce₂Co₁₇ (2:17R)", "Fe₇Mo₆ (μ)",
  and just "Al₃Ni₂" or "CeCo₃" otherwise. Multi-phase labels use the short name so they stay compact: the family
  where there is one (BCC+C14, L+C15, 1:5+Ce₅Co₁₉, L+2:17), else the formula (C15+CeCo₃). A Laves field with a
  solubility range is "C14" too.
  * *Formula:* one constituent per sublattice, the one that gives the composition closest to where the phase sits
    in the diagram ((Fe)2(Fe,Nb) near X(Nb) = 1/3 is Fe₂Nb). Vacancies are dropped and counts reduced
    (0.6667:0.3333 → Fe₂Nb). Elements are in the order of the phase name when it is a formula of the same elements
    (AL8CR5_L → Al₈Cr₅-LT; `M` is the remaining element, so M3V is Ni₃V on the Ni–V edge), else in sublattice
    order. On a ternary edge the formula is that binary's; a ternary line compound inside the triangle is written
    with its mixed sublattice, e.g. (Co,Ni)₃V.
  * *Family:* only from evidence in the phase name, never from the ratio alone (ZrNi₅, HfNi₅ and UNi₅ are not
    CaCu5-type, WAl₁₂ is not ThMn12-type, MoSi₂, CuAl₂ and Ni₂Zr are not Laves). Laves: tokens C14, C15, C36, LAVES,
    MGZN2, MGCU2, MGNI2. Greek letters: SIGMA, MU, CHI, ETA, ZETA, TAU … (not α, β, γ). CaCu5 family: tokens CACU5 /
    D2D → 1:5, THMN12 / D2B → 1:12, TH2ZN17 / TH2NI17 → 2:17, RT5 / RT12 / RE2TM17 / CE2T17-style R–T names, and as
    the only fallback a formula name (CECO5, CE2CO17) of a rare earth (La–Lu, Y, Sc) or Th with Mn, Fe, Co, Ni or Cu
    (Zn too for 2:17) in the ratio 1:5, 1:12 or 2:17. When in doubt, the formula alone. Where
    `calphad_tools/structure_id.py` knows a compound end-member (its curated table, its cache, or AFLOW / OQMD when
    `[structure] online` allows; `CALPHAD_STRUCTURE_ONLINE=0` switches that off), its answer is used too, and it
    can add a Strukturbericht suffix to a compound's own label ("Nb₃Sn (A15)", "Ni₃Al (L1₂)", "MoSi₂ (C11b)",
    with real subscripts, letters included, in PNG / PDF and SVG). Solution phases (FCC_A1, BCC_A2, B2_BCC,
    L12_FCC …) never get one. Multi-phase labels show only the families above, never a Strukturbericht symbol.
    The result text lists the evidence for each family name ("CeCo₅ (1:5): phase name token D2D",
    "Ce₂Co₁₇ (2:17H): … end-members: Ce2Co17 2:17H (table …)").
  * *2:17R / 2:17H:* Th2Zn17-type is 2:17R, Th2Ni17-type 2:17H.
    * A phase name that says (tokens TH2ZN17, R, RHOMB, R3M / TH2NI17, H, HEX, P63MMC: CE2FE17_H) decides for the
      whole phase, on every label.
    * Otherwise the structure of each binary end-member comes from `calphad_tools/structure_id.py` when that module
      is there. A name of the end-member's binary (a ternary edge, a binary compound below the axis) uses that
      end-member's structure: one Ce2(Co,Fe)17 model gives "Ce₂Fe₁₇ (2:17R)" at the Ce–Fe edge and "Ce₂Co₁₇
      (2:17H)" at the Ce–Co edge. Labels inside the triangle and multi-phase labels say 2:17R or 2:17H only when
      all end-members agree, else "2:17" (L+2:17). The result text and the JSON (`warnings`) then say: "WARNING:
      CE2TM17 is a single CALPHAD phase representing both 2:17H (Ce2Co17) and 2:17R (Ce2Fe17) - be careful with
      these labels." Any family with variants is handled the same way.
    * Two different 2:17 phases with nothing to tell them apart: if one is known, the other is the other kind. If
      neither is, on a binary the phase whose field reaches the higher temperature is taken as 2:17H, with a
      warning that this is a guess. On an isotherm both stay "2:17" and the text says so (set `phase_labels`).
    * show_code scripts get the structures as data (the `structures` argument of `label_diagram`), so they draw
      the same labels without structure_id.
* **Practically stoichiometric phases.** A phase whose model has composition freedom but whose field stays within
  `label_style.stoich_tolerance` (default 0.01 mole fraction) of one compound end-member of its model is labelled
  like a line compound. The measure is the largest deviation of any boundary composition of the field, at any
  mapped temperature, from that end-member's ideal composition (1/3 Nb for Fe₂Nb). On a ternary it can also be a
  line between two end-members in different binaries (a line compound such as (Co,Ni)₃V); a field stretched along
  one binary edge has a range in that binary and is not one. The result text lists the phases treated so, with
  deviation and field width, and the near misses (within 3× the tolerance) with the tolerance that would include
  them. Example: the combined Fe–Nb database where C14 becomes (Fe)2(Fe,Nb). Its field reaches X(Nb) = 0.3211 at
  the eutectic, 0.0122 from Fe₂Nb, so at the default 0.01 it stays a narrow C14 field (named below the axis as
  "C14"), and the text says `stoich_tolerance 0.013` would include it. With that setting it is "Fe₂Nb (C14)". μ
  (Fe)7(Nb)2(Fe,Nb)4 of the same database is within 0.004 of Fe₇Nb₆, so it is "Fe₇Nb₆ (μ)" by default. A model
  without composition freedom is always a compound.
* **Regions.** Two-phase fields and tie-triangles come exactly from the mapped phase boundaries. The rest of the
  diagram is named from a coarse equilibrium grid. Before a label is drawn, an equilibrium calculation at its
  position must give exactly the phases it names. Any region that can't be confirmed is left out, and the result
  text says so.
* **Placement.** Each label goes inside its region, in the thickest part, away from lines, data points, the
  legend and other labels, with a `label_style.margin` (default 2.5 pt) all round. In a skinny region it is turned
  (along the region, or in 15° steps) or made smaller, down to `label_style.min_fontsize` (default 8 pt).
  Two- and three-phase fields that still don't fit are left out; the result text lists them on one line.
* **Single-phase fields that don't fit inside** each get a place of their own, one region at a time (a second field
  of a phase that is named elsewhere too). This includes fields too thin to be a region of the plot at all.
  * *Ternary, at a corner:* a field that reaches a pure-element corner (the phase is stable for that element) is
    named at the corner, outside the triangle, square to the nearest side: -30° at the bottom-left corner (like the left-edge names), +30° at the other two.
    * Bottom-left: just above the left axis's 0.0. Bottom-right: above or right of the bottom axis's 1.0. Top:
      to the right of the corner.
    * The name stays clear of tick labels (none is hidden), the bold corner names, the axis titles and the other
      labels. The result text says e.g. "named at its corner: BCC (Cr), L (Al)".
  * *Ternary, at an edge:* a field that touches an edge but no corner (the phase exists in that binary) is named
    beyond that edge, like a stoichiometric phase (below). The name sits at the middle of its contact with the
    edge. A field that reaches two (or three) edges is named on each one. The second and third edges count only
    where an equilibrium just inside them has the phase. A compound on two edges is named on both too, each with
    that binary's formula: a C15 (Ce)(Co,Fe)2 across the triangle gives "CeCo₂ (C15)" at the Ce–Co edge and
    "CeFe₂ (C15)" at the Ce–Fe edge; M3V in Co–Ni–V gives Co₃V and Ni₃V.
  * *Binary:* a field that reaches the bottom of the plot is named below the x axis at its composition there, at
    60°. Example: a C14 with a narrow range. A practically stoichiometric field is named the same way, at its mean
    composition, with its compound name; if it ends above the bottom it is named right below its lowest point
    (next item), else with a leader line.
  * *Binary:* a field that ends higher up is named right below its lowest point, inside the axes. The text is at
    60° with no box; it starts about 2 pt below that point and runs down and away, only where it crosses no line
    or other label.
  * The liquid and the end-member solid solutions are never named below the x axis.
* **Isopleths** follow the same rules for every region, whatever its number of phases: inside it (fitted at 8 pt or
  more with the margin, turned if that helps); else right below its lowest point inside the axes (60°, crossing no
  line); else in the nearest free space with a leader line to its thickest point, only where that leader crosses no
  other label; else left out (listed). Nothing is named outside the axes: the x axis of an isopleth is the alloy's
  composition, not a phase's, so there are no end-member phases or compositions to point at. Every drawn line is
  kept clear (also a line end inside a region). A label position is confirmed by an equilibrium, or - where that
  misses the stable set (γ′, σ above) - by the lower Gibbs energy of the two phase sets. An isopleth drawn with
  `label_regions: false` can be labelled later with `label_phase_regions` (from the regions stored with the figure;
  the names on the lines are removed then); `show_code` reproduces isopleth labels only through `label_regions` in
  the `map_phase_diagram` cell.
* **Leader lines** are the last resort for single phases. They are used for ternary-only phases (fields that touch
  no edge, and ternary compounds inside the triangle), and for intermediate binary phases with no room below their
  field. The label goes in the nearest free space; inside a ternary that means a free area of the triangle, else
  just outside it. A thin line runs from the label to the field, with a dot at a point compound. These are never
  left out. An end-member phase that finds no place stays omitted, and the result text says so.
* **Stoichiometric phases** (binary line compounds, ternary point or line compounds on an edge) are never named
  inside the plot. Their name goes outside, right at their composition: below the x axis of a binary (text at
  60°), or beyond the nearest edge of a ternary (60° at the bottom edge, ±30° — square to the edge — at the slanted
  edges). A tick label that such a name covers is hidden; the others stay.
* **Invariant temperatures** (binary and isopleth, `label_invariants`) go right above or right below the invariant line, inside
  the adjacent region, e.g. "1345 °C", sized to fit and kept `margin` away from other lines. Each line gets at most
  one temperature. Below the minimum size they are left out. Reaction types (eutectic …) are not written.
* **Editing.** Each label is a single text object with an id like `phase_label:LIQUID.FCC_A1` or
  `invariant_label:1345C`. The `edit_figure` operations `move_label`, `delete_label` and `relabel` change one label
  (a moved leader label keeps its line on the field; a deleted one takes its line and dot with it).
  In Inkscape, the SVG ids are the same.
* **Files.** Labelled diagrams are saved as PNG, SVG and PDF, and the text stays text. Fonts: DejaVu Sans (it comes
  with matplotlib and has the Greek letters and subscripts), then Helvetica and Arial. The PDF embeds the font. The
  SVG only names it, so the program that opens the SVG needs the font installed. On a Mac, use the PDF: SVG import
  can be unreliable there, and Illustrator, Inkscape and Preview all edit the PDF text. `formats` gives SVG/PDF for
  unlabelled diagrams too.
* **Subscripts.** Formula digits are real subscripts: Al₃Ni₂ is typeset (matplotlib mathtext) in the PNG and the
  PDF. In the SVG, each name is still one `<text>`, and the digits are subscript runs written the way Inkscape writes
  them: `<tspan style="font-size:65%;baseline-shift:sub">3</tspan>`. Editing tools and the result text use the
  Unicode name (`relabel {text: "Al₃Ni₂"}`).
* **Standalone code.** `show_code` reproduces the labels without the toolkit: the script contains copies of the
  three label modules.

### Structure identification

The suffix in a compound label ("Fe₂Nb (C14)", "CeCo₅ (1:5)", "Ce₂Fe₁₇ (2:17R)", "Fe₇Mo₆ (μ)", "Nb₃Sn (A15)",
"Ni₃Al (L1₂)") must come from the crystal structure, not from the formula's ratio: ZrNi₅, HfNi₅ and UNi₅ are
AuBe5-type, so they are "ZrNi₅ (C15b)", not 1:5; WAl₁₂ is WAl12-type, not ThMn12; MoSi₂ (C11b) and CuAl₂ (C16) are
not Laves. `calphad_tools/structure_id.py` answers this per formula (one end-member), and the `identify_structure`
tool exposes it to the agent.

The suffix of an identified structure is chosen in this order:

1. a Greek letter: σ, μ, χ, R (only for transition-metal compounds);
2. the CaCu5 family: 1:5, 1:12, 2:17R (Th2Zn17), 2:17H (Th2Ni17), or 2:17 when the variant is unknown;
3. Laves: C14, C15, C36;
4. the prototype's Strukturbericht designation. Examples: A15, B2, B8₁, L1₀, L1₂, D0₃, D0₁₁, D0₁₉, D0₂₂, D0₂₃, D0₂₄,
   D0ₐ, D8₄, D8₈, C11b, C15b, C16, C32, C40, E9₃. There are 54 designated types in all, among 61 recognised structures;
5. otherwise no suffix. This covers WAl12, PuNi3, Ce2Ni7 and any unknown structure.

The designation is returned in subscript markup that the labeller renders: `"L1_{2}"`, `"D0_{19}"`, `"C11_{b}"`.
`suffix_plain` gives the same without markup ("L12", "D019", "C11b"). Elemental designations (A1–A12) are never
used, and pure elements get no suffix, so FCC, BCC and HCP keep their names. Use the suffix for compound phases only.
Solution phases keep their short names, even ordered ones named B2_BCC or L12_FCC; the labeller makes that call.

Sources, in order (the first one that answers wins):

1. **Overrides:** `[labels] phase_labels` in `calphad_agent.toml`, keyed by phase name or formula. A value is a suffix
   (`{CECO5 = "1:5"}`), a full label (`{CECO5 = "CeCo₅ (1:5)"}`) or `""` for no suffix.
2. **Phase-name tokens:** C14 / C15 / C36 / LAVES / MGZN2, CACU5 / D2D / THMN12 / D2B / TH2ZN17 / TH2NI17 / RT5 /
   RE2TM17 / 1_5 / 2_17, SIGMA / D8B, MU / D85, CHI, and the other designations (D019 or D0_19, L12, L1_0, C11B …).
   Tokens match whole words, so MU never matches inside another name. A1–A13 tokens (CBCC_A12, CUB_A13: the Mn
   solutions) and the ambiguous B1, B3, B4, C1, C2 are not used. In a 2:17 phase, _R / _H (also RHOMB, HEX) decide the
   variant: CE2FE17_R is 2:17R.
3. **A built-in table** of 138 well-established compounds (Pearson / Strukturbericht data), so offline clusters get the
   common cases. Examples: Ni3Al L1₂, NiAl / FeAl / CoAl B2, Fe3Al D0₃, Ni3Ti D0₂₄, Ni3Nb D0ₐ, TiAl L1₀, Ti3Al D0₁₉,
   Al3Ti D0₂₂, Al3Zr D0₂₃, Nb3Sn A15, Fe3C D0₁₁, Cr23C6 D8₄, Mn5Si3 D8₈, TiB2 C32, plus the Laves, CaCu5, σ and μ
   compounds. Each entry has a short source note, and polymorphs are listed as alternatives (Sm2Co17: Th2Zn17-type
   at low T and Th2Ni17-type at high T, so the family is "2:17"; Cr2Nb: C15, with C14 at high T). σ is matched by
   composition range (Fe–Cr, Fe–V, Co–Cr, Cr–Mn, Fe–Mo, Mo–Re, Re–W).
4. **The cache:** `~/.calphad_agent/structure_cache.json` (under `CALPHAD_STATE_DIR` if set) holds earlier online
   answers for `[structure] cache_days` (default 180).
5. **AFLOW and OQMD** (keyless web APIs). AFLOW's AFLUX search returns every entry of the element system; ICSD
   entries give the experimental prototype label (`aflow_prototype_label_orig`), and computed ones give the relaxed
   label and `enthalpy_formation_atom`. OQMD's `formationenergy` gives `prototype`, `spacegroup`, `natoms`, `delta_e`
   and `stability`. The answer is the reported (ICSD) structure with the lowest formation energy. Other structures are
   listed as alternatives. Two databases that agree give high confidence; disagreement gives low. If both 2:17
   polymorphs are reported, the answer is "2:17". Timeouts are short (`[structure] timeout_s`, default 15 s), and a
   failure only adds a note.
6. **A chemistry rule**, used only as a last resort (low confidence): a rare earth (La–Lu, Y, Sc) or Th with Mn, Fe,
   Co, Ni or Cu (or Zn for 2:17) at 1:5, 1:12 or 2:17 is in the CaCu5 family. It never applies to Zr, Hf, U or Al,
   and it never gives Laves or a Greek letter.

Each answer has a suffix, prototype, space group, AFLOW label, source, confidence (high / medium / low), one line of
evidence ("AFLOW AB5_hP6_191_a_cg (CaCu5-type), 9 ICSD entries") and alternatives.

**Privacy and offline use.** Only chemical formulas (element symbols and counts) are sent to aflowlib.duke.edu and
oqmd.org; database content never is. `[structure] online = false`, or `CALPHAD_STRUCTURE_ONLINE=0` in the
environment (which beats the config), keeps everything offline: the table, the cache and the phase names are then
the only sources. A tool call can switch lookups off (`online: false`) but cannot switch them on against these
settings.

**The tool.** `identify_structure` takes `formula` (comma-separated for several), or `tdb` + `phase`, or `tdb` alone
to check every stoichiometric or near-stoichiometric phase of a database (`elements` narrows the list). It prints one
line per end-member, plus a WARNING when one CALPHAD phase covers several structures, e.g. "CE2TM17 is a single
CALPHAD phase representing both 2:17R (Ce2Fe17) and 2:17H (Ce2Co17) - be careful with these labels".

**For the labeller.** `identify(formula, phase_name=..., allow_online=...)` returns a `StructureID`. A phase is
handled with `phase_family(phase_name, endmember_formulas)`, which returns a `PhaseFamily` with these fields:

* `per_endmember`: one `StructureID` per end-member formula;
* `consensus`: the suffix when every end-member that has an answer agrees, and none lists a conflicting polymorph;
* `ambiguous_label`: the label of the whole family, "2:17" or "Laves";
* `name_suffix` / `name_evidence`: from the phase name, which decides for the whole phase;
* `warning`.

`assign_217(phases)` tells two 2:17 phases of one diagram apart. It uses name tokens first, then each phase's
structure data, then the partner's variant (the other phase is the other kind). As a last resort, the
higher-temperature phase is taken as 2:17H, with low confidence. `assign_217_detail` returns the evidence too.

### Data-science plotting: operations and tools

`edit_figure` and `make_plot` share 69 operations. The model looks up the syntax on demand with `plot_ops_help`, so the tool descriptions only list the operation names.

#### Data preparation

Any operation can read `data`:

* **Formats:** CSV/TSV (delimiter detected), Excel, JSON, netCDF, or inline records.
* **`where` filters:** equality, comparisons, `between`, `contains`, …
* **`transform` pipeline:**
  * `derive` / `filter`, using safe formulas over columns
  * `groupby` + `agg`, `pivot` / `melt`, `join` (also nearest-match joins), `concat`
  * `interpolate`, `smooth`, `derivative`, `integrate`, `normalize`, `bin`
  * `convert`: temperature, energy and pressure units, and at% ↔ wt% for binaries or multi-component alloys

Tools:

* `inspect_data` summarises a file you haven't seen yet.
* `transform_data` writes the processed table.

#### Fitting and statistics

* **`fit`** supports linear, polynomial, exponential, power, log, Arrhenius (Ea in kJ/mol; fitted in log space by default), logistic, Gaussian, and custom formulas. It draws confidence or prediction bands and an equation box, and reports the parameters ± SE and R². `residuals` plots the residuals.
* **Statistical plots:** `hist` (density, cumulative, stacked), `box`, `violin`, `kde`, `ecdf`, `qq`, `hist2d`, `hexbin`, `band` (mean ± std/SEM/CI), `corr_heatmap`, `scatter_matrix`.
* **`fit_data`** gives the fitted numbers without drawing.

#### Grids and plot types

* **Grid and field data:** `heatmap` (annotated cells), `pcolormesh`, `contour`/`contourf` (from long-format data too), `tricontour`/`tricontourf`/`tripcolor` for scattered points, including property maps on ternary triangles, and `quiver`.
* **Other plot types:** `stackplot` (phase fractions), `bars` (grouped or stacked), `pie`/donut, `radar`, and 3-D `surface`/`wireframe`/`scatter3d`/`line3d`/`view`.

#### Axes and layout

* `secondary_xaxis` / `secondary_yaxis`: K↔°C, 1000/T, log, or at%↔wt%. For binary phase diagrams the elements are read from the diagram itself.
* `twiny`, `ticks` (formats, minor ticks, number of ticks, rotation), `spines`.
* `colorbar`, shared across panels.
* `panel_labels` for (a), (b)…
* `broken_yaxis`.
* `annotate_points` and `label_lines` for inline labels.

`make_plot` adds:

* grid spans and width/height ratios;
* polar, 3-D and triangular panels;
* journal presets (`journal-single/double`, `nature`, `acs`, `slide`, `poster`);
* colour-blind-safe palettes (`okabe-ito`, `tol-bright`, …);
* fonts and allow-listed rcParams;
* transparent export.

#### Delivery tools

* `compose_images`: existing images → one figure with (a)–(d) labels.
* `figures_to_pptx`: new decks, or slides appended to an existing deck.
* `animate`: a GIF/MP4 from frames, or from sweeping any tool's parameter, e.g. isotherms against temperature.
* `image_tools`: crop, trim, resize, pad, rotate, make a background transparent, convert.
* `images_to_pdf`: a multi-page PDF; a plot made by the tools becomes a vector page.

### Assessment workflow tools

#### Plot digitiser

Extracts data from figures in papers:

1. **`digitize_setup`** takes a PDF page (or an image) plus a loose crop around the figure. It deskews scans, finds the plot frame and ticks, and writes a preview with the ticks numbered.
2. The model reads two or more tick labels on each axis from that preview.
3. **`digitize_points`** or **`digitize_curve`** extracts the data. Both return a CSV and a numbered overlay for visual checking.
4. Each call also updates a **provenance figure** for that source image, `<image>_provenance.png`. It shows every series extracted from the image so far, drawn on top of the original figure. Each point is a thin ring with a crosshair, placed from its final CSV value through the axis calibration, so a wrong point or a wrong calibration shows at a glance. Traced lines are drawn as thin lines, and flagged points get a dashed red ring with a "?". Small teal tick marks show where the calibration puts the axis tick values; if their labels don't match the printed ones, the calibration is wrong. The title records the source file, page and crop, the calibration used, and when and with which tool the data were extracted. The agent is told to show this figure to the user. `digitized_to_zpf` draws the same kind of figure from the CSVs it converts, next to its phase-assignment preview. For a digitizing step, `show_code` redraws a simpler version from the recorded values and the image.

What `digitize_points` handles:

* colour-selected markers, even when series colours are similar;
* shape and fill filters;
* automatic template matching when a shape is chosen, which finds markers sitting on lines and works in black-and-white figures;
* legend exclusion and error-bar extraction;
* log axes;
* touching and overlapping markers (see below).

**Touching and overlapping markers.** The tool learns what one marker looks like from clean, isolated markers. It splits any blob bigger than that into the markers that best explain it. A blob it cannot split with confidence is kept as one point marked "may be N+ overlapping markers". The mark appears in the tool's reply, in the CSV `note` column, and as a red ring on the overlay. So two merged markers are never reported as a single clean point. A marker partly hidden under a marker of another colour is also recovered, and marked as such. In a stress test with 15 isolated markers per series plus close pairs and triples:

* markers one diameter apart or more are all found, with centre errors below 0.1 diameter;
* at 0.6 to 0.8 diameters, pairs of identical markers are split correctly;
* heavier overlaps are split when possible and flagged otherwise;
* no merge went unflagged.

When two markers of the same colour but different shapes almost completely cover each other (0.4 diameters apart or closer), the hidden one can't be recovered; its visible neighbour is flagged.

**Asking the user how many points a blob is.** When anything is flagged or was split, `digitize_points` also writes a **check sheet**: one enlarged crop per doubtful blob, labelled `#N` with the detections found in it. The agent shows it and asks, and a reply such as "#7 is 3 points, #13 is 2, #23 is just a smudge" becomes `digitize_fix` with `counts {"7": 3, "13": 2, "23": 0}`.

* **Refitting:** each blob is refitted with exactly that many copies of a marker learned from clean markers in the same figure. Other detections in the same blob are replaced.
* **Accuracy:** knowing the count is what makes heavy overlap solvable. In the stress test, pairs 0.25 diameters apart and triangles of three at 0.4 and 0.25 come out complete, with centre errors of 0.03–0.05 diameters. Other points are left untouched.
* **Corrections:** `add` places a missed point from rough axis values, snapped onto the marker nearby; `remove` drops a spurious one.
* **Results:** the CSV notes record which points were confirmed or added by the user, and the provenance figure and an after-correction sheet are updated.

**Accuracy**, measured on a ground-truth benchmark (`tests/digitize_bench.py`) covering:

* multiple series with a legend and gridlines;
* markers on curves, touching markers and error bars;
* open markers, log axes and 72 dpi images;
* JPEG artefacts with blur;
* a rotated, noisy scan on off-white paper;
* black-and-white symbols;
* a figure inside a text PDF page.

| Case | Recall | Precision | Position error |
|------|--------|-----------|----------------|
| Colour figures (visible points) | 100% | 100% | 0.5–0.9 px RMS (0.1–0.2% of the axis range) |
| Black-and-white symbols (template route) | 87–100% | 100% | ≤ 0.9 px |

In the black-and-white case the misses are markers physically overlapped by another series' symbol. Points hidden under an opaque legend can't be recovered.

`digitize_curve`, tracing a computed liquidus, agrees with the exact boundary to within about 12 °C.

`digitize_setup` takes a PNG or JPG directly, as well as a PDF page. It also recognises a composition triangle (a Gibbs triangle or a right triangle). In that case the preview numbers the corners `c0`–`c2` and the edge ticks. The model names the element at each corner, and `digitize_points` then writes the three mole fractions of each point.

#### From a phase-diagram figure to ESPEI datasets

A ZPF dataset needs to know, for each point, which phases are in equilibrium and which of them the measured composition belongs to. Often there is no database yet, because the digitized points are what the database will be fitted to. So `digitized_to_zpf` takes the phases from the figure itself:

* **Region labels.** The model reads the labels printed on the diagram ("L", "(Cr)", "L+(Cr)", "σ", ...) and where they are. The tool splits the plot into the regions enclosed by the figure's own drawn lines, gives each region its label, and works out the unlabelled ones from their neighbours. For example, an unlabelled field between (Cr) and (Ni) is (Cr)+(Ni). Each point is then assigned from the regions it touches:
  * a point between "(Ni)" and "(Cr)+(Ni)" is FCC coexisting with BCC;
  * three phases around a binary point make an invariant;
  * in a ternary, three phases around a point that touches a tie-triangle make a tie-triangle corner.
* **The legend.** The model can say what each marker group is, either with phase names or with boundary words (`liquidus`, `solidus`, `solvus`, `invariant`, `tieline`). Boundary words combine with the region labels: the legend says "solvus", and the regions say which phases.
* **Phase names.** `name_map` translates the paper's names into database names, e.g. `{"(Cr)": "BCC_A2", "σ": "SIGMA"}`. "L" and "Liq" become LIQUID automatically. The tool suggests a name for "(X)" (the solid solution of X) and for a few Greek names, but any name that is still unmapped stops the call, so the user can decide. Use the names your ESPEI phase models will use.
* **Tie-lines and invariants.** Ends measured at the same temperature (`pair_tielines`), or joined by a drawn line (`tie_lines`), become one row with both compositions. Three-phase points at one temperature become an invariant row, or a tie-triangle row in a ternary. Any other point becomes a row where only its own composition is given; the other phase is `null`.
* **Drawn lines.** Lines traced with `digitize_curve` are sampled at chosen temperature steps and written to separate files. These are marked as pseudo-data from a drawn line, not as experimental points.
* **Units.** at%, wt% (using atomic masses) and °C are converted. Swapped axes are supported.
* **Checking the result.** An assignment preview shows the figure with the regions tinted and labelled, and the points numbered and coloured by their assigned phases. Unassigned or questionable points are circled. The model looks at it, shows it to the user, and fixes points with `overrides` or `exclude_points`. `preview=true` writes only the preview and the summary.
* **Output.** JSON files split by phase set, each checked with ESPEI's own checker and loaded with `load_datasets`. A CSV lists every point with its assignment, method, confidence and any flag.
* **A database is optional.** It is useful when you are refining an existing database. It then serves two purposes:
  * it cross-checks the figure's assignments, and reports points where they disagree (the figure's assignment is kept) and points far from any calculated boundary;
  * it infers the phases for groups that nothing else describes, using the nearest calculated boundary, equilibria just either side of the point, and a majority vote within each marker group.

`show_code` turns the result into a standalone script that rebuilds the same JSON files from the recorded per-point assignments.

On the synthetic benchmarks in `tests/test_digitize_zpf.py`:

* **Cr–Ni paper-style figure:** 36 noisy markers of five kinds, printed region labels, inward ticks, °C and at% or wt% axes. All points were recovered. Compositions were within 0.002 (mole fraction) and temperatures within about 2 K RMS. Phase assignments were 100% correct in each of these cases:
  * region labels alone;
  * a missing label;
  * the legend alone;
  * the legend plus labels;
  * TDB inference.

  In every case the tie-line pairs and the eutectic were found. With the true database, ESPEI's ZPF driving forces on these datasets average 22 J/mol, against 378 J/mol for a database with a perturbed liquid parameter.
* **Co–Cr–Ni isotherm at 1000 °C:** 13 points drawn as tie-lines and a tie-triangle. All were assigned correctly, both from the legend and from the region labels.

#### ESPEI and database tools

* **ESPEI MCMC:**
  * `analyze_mcmc`: burn-in, R̂, autocorrelation, effective sample size, credible intervals, traces, corner plot, and which TDB parameter each fitted symbol belongs to;
  * `mcmc_to_tdb`: writes a TDB from the mean, median or best-fit parameters;
  * `propagate_uncertainty`: posterior-sample phase diagrams with invariant-temperature distributions, and property bands.
* **Thermochemical datasets** (units are per mole of atoms, as verified in the ESPEI source):
  * `write_thermo_dataset` for HM/SM/CPM `_FORM`/`_MIX`, with `mixing_x` tables;
  * `write_equilibrium_dataset` for ACR, HMR, …;
  * `convert_pop_to_thermo`, `convert_csv_to_thermo`.
* **`fit_report`:** per-dataset residuals computed with ESPEI's own residual classes, a ranking of the worst datasets, parity plots, and a ZPF overlay coloured by driving force.
* **`check_tdb`:**
  * undefined or duplicate definitions, and discontinuities at T breakpoints;
  * unary melting points against SGTE values, negative Cp, a solid becoming stable again above its melting point;
  * inverted or high-temperature liquid miscibility gaps, solids stable above the liquid;
  * very large interaction parameters.

  On the bundled `CoCrFeNiV_Choi2019.tdb` it found a real defect: `GHSERVZ`'s middle range runs to 4000 K instead of 2183 K, so G jumps by 5911 J/mol there.
* **Parameter what-ifs, sensitivities and fitting** (`param_tools.py`):
  * `edit_parameters` changes parameters or symbols by name, e.g. `L(FCC_A1,CR,NI;0)` or `VV0001`. It can set, add, scale, remove or create a parameter, or add to one coefficient only (`term`: `T`, `TLNT`, ...). Edits apply to every T range unless `t_range` picks one.
    * Only the edited commands change; the rest of the file stays byte-identical, and the tool returns a diff.
    * An identical parameter of a disordered-part phase is changed too (for example `A1_FCC` of `L12_FCC`), otherwise the ordered phase takes over. `update_twins: false` turns this off.
    * `compare` overlays the original (dashed) and modified (solid) diagram and lists the shifts of invariants and of phase boundaries at given temperatures. `datasets_dir` gives the change of the ESPEI objective per data type.
    * Example: making L0 of FCC Cr-Ni 2 kJ/mol more positive moves the Cr-Ni eutectic from 1617.9 K to 1572.8 K.
  * `parameter_sensitivity` computes central finite-difference derivatives of:
    * the ESPEI objective, in total and per data type;
    * binary invariant temperatures and compositions, and boundary compositions at given temperatures;
    * equilibrium properties at one condition point.

    It writes a CSV and a tornado plot. For the objective it adds J^T J (Gauss-Newton Hessian): approximate standard deviations, correlations, and warnings for parameters the data cannot determine.
  * `optimize_parameters` minimises ESPEI's negative log-likelihood. It uses scipy `least_squares` (trf) on residual/sigma, or L-BFGS-B, with bounds and an optional Gaussian prior.
    * Free parameters are VV symbols or coefficients, e.g. `L(FCC_A1,CR,NI:VA;0)@T`.
    * It writes the optimized TDB, a before/after parameter table with approximate standard deviations, the objective history, and a before/after comparison with the data.
    * Runs with more than 3 parameters or more than 50 data points go to a background job (`job_status`, `stop_job`; stopping keeps the best point found).

  Notes from checking ESPEI 0.9.1:
  * `SciPyOptimizer` minimises the log-likelihood itself, so it moves away from the data. These tools minimise -lnL.
  * Equilibrium-property datasets with several temperatures *and* several compositions are compared in the wrong order. They are flagged.
  * In pycalphad 0.11.2 an ordered phase uses the database value of a fitted symbol that sits in its disordered part. ESPEI's objective therefore does not see the ordered phase change. This is flagged too.
  * ESPEI pull request #268 (Kunselman, Bocklund et al., "Analytical gradient-based optimization of CALPHAD model parameters", Acta Mater. 2025) adds analytic likelihood gradients. It is not released. If an ESPEI with it is installed, L-BFGS-B uses its gradients automatically.
* **Reproducibility:** every tool call is logged with input hashes, and PNGs get provenance sidecars.
  * `session_log` lists the calls.
  * `export_session` writes a replay script or Jupyter notebook that re-runs and checks the outputs.
  * `provenance` shows how a file was made.
  * The terminal commands `/log` and `/export [script|notebook|both] [steps]` do the same.

### Running ESPEI on a Flux cluster

Ask "optimise Cr-Ni with the data in X, Y and Z" and the agent turns the data into ESPEI datasets, makes the phase models, and calls `optimize_system` (parameter generation, then MCMC). Where that runs is decided automatically:

* **Detection.** Flux is used when the `flux` command is on the PATH **and** a Flux instance answers: inside an allocation (`FLUX_URI` set) `flux getattr size`, otherwise the system instance with `flux uptime`. The probes have a short timeout and never submit anything. They also read nodes, cores per node and GPUs (`flux resource info`, else `flux resource list`), the queues (`flux queue list`), and whether `flux batch` has `--bank`. Detection runs once per agent process when `[hpc] scheduler = "auto"`; the installer and `calphad-agent --check` print it, and `python -m calphad_tools.hpc` shows it at any time.
* **Batched (Flux found).** The run directory `<output>/espei_runs/<name>/` gets the ESPEI YAMLs and two batch scripts (the `per-job` layout below). They are submitted with

  ```
  flux batch -q <QUEUE> --bank=<BANK> [extra_flux_args] gen.flux
  flux batch -q <QUEUE> --bank=<BANK> [extra_flux_args] --dependency=afterok:<GEN_ID> mcmc.flux
  ```

  The node count, cores and time limit are `#flux:` lines in the scripts: generation `-N 1 -n 1 -t <gen_time_limit>`; MCMC `-N <nodes> --exclusive -t <time_limit>` with one dask worker per core. `submit.sh` in the run directory holds the exact commands. Older flux-core without `--bank` gets `--setattr=system.bank=<BANK>` instead (detected).
* **Direct (no Flux).** A background job on this machine: generation, then MCMC on a dask cluster.
* **Waiting.** The agent calls `wait_job ... until='done'` and keeps waiting with one status line (MCMC iteration from `mcmc.log`, lnprob) until the jobs finish. Ctrl-C in the terminal, or **stop waiting** in the web UI, ends the wait but not the job; `[hpc] wait_max_hours` (24) is the limit. When the run is done the agent continues: `analyze_mcmc`, and `plot_diagram_with_data` with the run's `mcmc.tdb` against the data.
* **Failures.** A failed Flux job reports its Flux status, exit code and the last lines of its batch output file (`espei-mcmc-<name>.<id>.out`). If `flux batch` refuses the job (missing bank, wrong queue, time limit), nothing runs: the agent shows Flux's message and which `[hpc]` setting to change. It never switches to a local run silently: it asks you, or follows `fallback_to_local = true`.
* **From a shell.** `flux jobs` lists the jobs (the agent prints their ids); `flux jobs -a` includes finished ones, and `flux cancel <id>` stops one. The agent's `list_jobs`, `job_status` and `stop_job` (which runs `flux cancel`) do the same. Job records live in `<output>/jobs/<id>/`, so they survive an agent restart.
* **Provenance.** The job record, `hpc_submission.json` in the run directory and the session log hold the backend, the Flux job ids, the scripts, the commands and the resources. `show_code` on the outputs of a Flux run writes the YAMLs and batch scripts and resubmits them with the same `flux batch` commands.

The `[hpc]` table of `calphad_agent.toml`:

```toml
[hpc]
scheduler = "auto"         # auto | flux | local   (or CALPHAD_HPC_SCHEDULER)
bank = ""                  # flux batch --bank=BANK (your allocation account)
queue = ""                 # flux batch -q QUEUE
nodes = 1                  # batched runs: nodes per ESPEI MCMC job
cores_per_node = 0         # 0 = from flux resource info
time_limit = "4h"          # flux batch -t, MCMC job
gen_time_limit = "1h"      # flux batch -t, 1-core generation job
local_cores = "half"       # direct runs: "half" of the available cores, "all", or a number
extra_flux_args = []       # added to every flux batch command
fallback_to_local = false  # Flux found but submission fails: run here instead (and say so)
wait_max_hours = 24        # wait_job until='done' stops waiting after this long
```

**Run-size defaults.** Direct runs use half of the available cores (CPU affinity, at least 1, never more dask workers than MCMC chains). Batched runs use 1 node per ESPEI MCMC job with all of its cores. `optimize_system` takes `cores`, `nodes`, `cores_per_node`, `time_limit`, `queue`, `bank` and `backend` (`auto`, `flux` or `local`) to override them. `optimize_parameters` (the gradient optimiser) runs on one core, so `local_cores` does not apply to it.

### Combining databases

Ask "combine these databases" and the agent calls `combine_databases` with two or more TDBs. It runs in four stages.

1. **Plan.** Every pair is checked with `evaluate_compatibility` (no mapping). The tool proposes a base and an order, with reasons: most elements, most complete unary set, SER agreement with SGTE unary50 (when a local copy exists), most binaries assessed; then the source that shares most elements with what is already in, with the fewest conflicts. An `order` (or `base`) you give wins. The chain is then simulated step by step (Q₁ = base + S₁, Q₂ = Q₁ + S₂, …) and every decision it will need is listed **once, up front**, with a stable id and the consequence of each option:
   * phase-model conflicts (`step2:phase:MU_D85`: db1 / db2 / rename);
   * which assessment of a shared binary wins (`step1:system:CR-FE`);
   * ambiguous sublattices (`step1:sublattice:C14`), name candidates (`step1:name:LAVES_C15`). Each phase pairs with at most one: a phase already paired (by name, alias, `phase_map`, an `extend` decision or an answer) is never offered again. Candidates competing for one phase are a single question, e.g. `step1:name:LAVES` with the options `LAVES_C15`, `LAVES_C14`, `extend:LAVES_C15` and `separate`. Once one pairing is answered, the questions that depended on it go away; calling again with partial decisions shows the reduced list;
   * `REJECT_PHASE` defaults another file contradicts (`step1:reject:db1:L12_FCC+A1_FCC`).

   It also says which unary data the base will overwrite, e.g. "CrFe-82Her's unary data differ from the base's (G(FCC_A1,CR:VA;0) Δ4.57 kJ/mol): its subsystems with Cr change - choose another base or order to keep them". Later steps are planned with provisional answers (phase models 'rename', binaries 'db1'); an answer that differs can raise new questions, which come back in a second round. Nothing is merged until every decision is answered. `policy="first"`/`"last"` fills the missing ones and logs them as policy, except a phase-model answer that would drop an element from a phase entirely: that one is always asked.
2. **Decisions.** The agent asks you all of them at once and calls again with `decisions` and `decision_reasons` (`{"step2:phase:MU_D85": "user: keep both mu models"}`, or `"agent: <why>"` if it chose). Ids work per step (`step2:…`), per added database (`<file stem>:…`) or globally (`phase:MU_D85`).
3. **Run.** `staged_merge` for each step, Q of step k being the base of step k+1 (its versions, per-step diagrams, invariant comparisons and content-hash cache are reused). Every parameter and phase-model change of every step is attributed to a **decision-log** entry: id, step, subject, options, chosen, who (`user` / `agent` / `policy` / `rule`), why, evidence (ΔG at 500/1000/1500 K, invariant shifts with figures, GM identity errors) and the subsystems it affects. The automatic rules are logged too: unary/SER replacement, name normalisation, site ratios read as exact fractions (one entry per phase, with the GM change), sublattice reorder, rescaling, the magnetic basis rule (which side set the basis, k, exactness, and for a magnetic A the end-member check and the interior error), kept FUNCTIONs, constituent unions (C14 gaining Fe:Fe), phases entering another database's subsystems, type definitions taken from the base, forced replots and end-members without G. A decision passed without a reason is logged as "not given" and flagged in the report.
4. **Validation and report.** For every source S and every binary it describes (ternaries at `ternary_temperatures_c`), S and the final database F are mapped with the mapping check on and compared:
   * invariants matched by phase set (ΔT, Δx);
   * ZPF boundaries (max distance between corresponding boundaries, 1 K counted as 0.002 in x);
   * a 25 × 24 equilibrium grid (fraction of points with the same stable phases; disordered states of ordered phases named by their disordered phase, as on the diagrams);
   * H, S, Cp and G of the stable assemblage at 9 points.

   Each subsystem is **reproduced** (identical description, or ΔT ≤ 0.5 K, Δx ≤ 0.001, boundaries ≤ 0.002, grid ≥ 99.5 %, equal properties), **changed, explained** (every difference is attributed to log entries) or **CHANGED, UNEXPLAINED** (listed first, in red). Renames, sublattice permutations and rescales are checked on the actual data: GM per mole of atoms identical to 1e-9 at random T and site fractions. F is loaded in pycalphad and linted with `check_tdb`. New end-members without G (pycalphad uses 0) are reported with whether the phase is stable where mapped, and grid points that do not converge are counted. Outputs: one deck (summary, plan, the staged slides of every step with its TDB changes (old line in red, then the new one; see `staged_merge` below), decision log, one overlay slide per source with black = source and red dashed = combined, property table, exactness and sanity, what cannot be proven), one Excel workbook, `combine_report.md`, the final TDB with a header block (sources with SHA-256, order, decisions with who and why, verdicts), `combine.ipynb` and `combine_state.json`. The tool's reply is about 2,000 characters.

**What "proof" means here.** For each source, every subsystem it describes either comes out of the combined database unchanged (within the tolerances above), or each parameter and phase-model difference is traced to a logged decision with its evidence. Nothing changed without a recorded reason, and the exact transformations really are exact. It does **not** prove that a changed subsystem is better: that is the decision's reason, not a test. It also does not cover what lies beyond the mapped range and grid resolution, or systems no source describes. Ternaries made of binaries from different assessments are extrapolations; they are listed as "predicted, not validated", and with `validate_temperatures_c` their isotherms are mapped and shown as such.

Examples (bundled TDBs, binaries 300–3000 °C):
* **Fe-Nb + Fe-W + Nb-W.** One decision: μ (Fe-Nb 7:2:4 vs Fe-W 1:4:2:6), answered 'rename'. Fe-W and Nb-W are reproduced. C14 gains Fe:Fe from Fe-W, so Fe-Nb's C14 becomes (Fe)2(Fe,Nb); the tentative L(C14_LAVES,FE:FE,NB;0) = +150000 (see [Tentative parameters](#tentative-parameters)) keeps it at X(Nb) = 1/3, and Fe-Nb is reproduced (max invariant ΔT 0.001 K). With `tentative=false`, C14 reaches down to X(Nb) = 0.321 and BCC + C14 + liquid moves +4.97 K: changed, explained by the constituent union. The predicted Fe-Nb-W isotherm at 1200 °C has C14 with no G for W:NB (pycalphad uses 0), and C14 is stable there, so it is flagged "check it".
* **Cr-Ni + Fe-Ni + Cr-Fe** (base Cr-Ni, proposed). Two decisions: the `REJECT_PHASE` defaults (Cr-Ni rejects L12/A1_FCC, Fe-Ni rejects FCC_A1). Cr-Ni is reproduced. Cr-Fe is changed, explained: 82Her's pre-SGTE fcc-Cr lattice stability is replaced by the base's, which widens the γ loop drastically. Either Cr-Fe or Cr-Ni has to change, because the two databases disagree on fcc Cr. Fe-Ni keeps its diagram, but 32 grid points between 2530 and 3000 °C no longer converge in F, because Cr-Ni's B2_BCC (Ni,Va) model enters Fe-Ni, so it is reported as changed. Cr-Fe-Ni is listed as not validated.

Each decision-log entry points to the TDB lines its step changed: the step's `diffs/*.diff` files for the transitions where that kind of decision acts (normalisation entries to `norm-B.diff` / `norm-A.diff`, everything else at least to `B3-Q.diff`). They appear in the Affects column of the decision-log slides, in "Files / figures" in Excel and as links in `combine_report.md`. The combine workbook's "TDB changes" sheet lists all steps' lines (step, transition, group, change, line).

`combine_databases(validate_only=true, final_tdb=…)` re-validates a hand-edited result against the run's decision log; an edited parameter shows up as CHANGED, UNEXPLAINED ("F differs from the merge result here"). Decision-log entries are also written to the session log (`{"type": "decision"}` lines), so `package_feedback` reports them and `show_code` / `combine.ipynb` reproduce the merge. The notebook rebuilds F from the inputs with standalone code (the TDB parser and merge planner are copied in) and checks that it is parameter-equivalent to the tool's file. It recomputes the grid and property tables with pycalphad, and regenerates the mapped overlays in a cell labelled [calphad_tools].

### Tentative parameters

Combining databases can create a mixing that no source assessed. (Fe)2(Fe,W) from Fe-W plus (Fe)2(Nb) from Fe-Nb gives C14 (Fe)2(Fe,Nb,W), so Fe-Nb's C14 becomes (Fe)2(Fe,Nb) with nothing but ideal mixing of Fe and Nb on the second sublattice. Entropy then gives the line compound a solubility range it never had. `merge_tdbs`, `staged_merge` (B3, Q, A1, A2), `combine_databases` and `extend_phase` therefore add **tentative** L0 terms (`calphad_tools/tentative.py`). A pair of constituents on a sublattice is *described* when some source has both on that sublattice of the phase, whatever L terms it gives them (0 is an assessment too). A source whose only description of the pair is a tentative term does not count. For a pair that no source describes:

* **Rule 1 (`TENT_MIX`), new mixing in an intermetallic.** The phase has two or more substitutional sublattices (C14, sigma, mu, …). The rule adds L(phase, …:i,j:…;0) = +50000 J/mol-atom × atoms per formula unit, for each combination of the other sublattices' constituents where both end-members (with i and with j) have G. Atoms per formula unit is the sum of the site ratios of the sublattices that are not vacancy-only, so 2:1 gives +150000 and 0.667:0.333 gives +50000. In the example: L(C14_LAVES,FE:FE,NB;0) = +150000, and L(C14_LAVES,FE:NB,W;0) = +150000 because Nb,W is new on that sublattice too. W:NB has no G, so no term is added in the W context. Fe-W's assessed Fe,W interaction is left alone. Solution phases (one substitutional sublattice: liquid, fcc, bcc, hcp) are never touched. There a pair no source describes is a new, unassessed binary, not a change to an assessed one.
* **Rule 2 (`TENT_VA`), vacancies on a substitutional sublattice.** BCC (Fe,Cr,Ni,Va)(Va) + (Co)(Va) gives (Fe,Co,Cr,Ni,Va)(Va). Fe, Cr and Ni have their assessed L(BCC_A2,X,VA:VA;0); Co gets L(BCC_A2,CO,VA:VA;0) = +200000 J/mol as written (per formula unit), with the other sublattices' constituents as context (`:VA`), so bcc-Co is not stabilised by mixing with vacancies. A missing vacancy end-member G(phase,VA:VA) is reported, not invented. The usual choice is G = 0. A substitutional ("main") sublattice holds at least one element that is not an interstitial (B, C, H, N, O). (Fe,Cr,Ni,Va) is one; the (C,Va) of (Fe)(C,Va) is not, so it never triggers rule 2.
* The ordered phase of an order-disorder (DIS_PART) pair (e.g. BCC_B2) and phases with ionic species are left alone; the disordered part (e.g. BCC_A2) gets the terms like any other phase, with a note on the ordered phase. Terms are L0, written with sorted constituents.

**Citation.** Every tentative parameter carries the reference key `TENT_MIX` or `TENT_VA`. The written TDB's reference list has one entry per key: `TENT_MIX 'Tentative parameter added by calphad_tools when combining FeNb-90Hua-LB, FeW-15Jac-static (2026-10-08): new Fe,Nb mixing on sublattice 2 of C14_LAVES, +50000 J/mol-atom x 3 atoms/fu; …; not assessed - replace with an assessed value'`. `tdbparse.is_tentative(param, tdb)` recognises both keys and any reference whose citation starts with "Tentative", so hand-written tentative terms count too. Tentative terms are flagged in `query_tdb` (`TENT_MIX · tentative`, plus a TENTATIVE line per phase), in `compare_system` tables (the reference line under the cell), and in the merge, staged and combine reports. `evaluate_compatibility` stage 5 and the combine plan list the terms each merge **will** add, rule 1 and rule 2 separately, with values.

**Precedence.** When two databases are combined and both have a parameter, an assessed one always replaces a tentative one. This holds whichever database is A, the base or B, and whatever the policy or decisions say. It applies in `merge_tdbs`, every staged step, every step of a `combine_databases` chain (a term added at step 1 goes when step 2's source assesses it, and new terms are added where still needed) and `extend_phase` collisions. A tentative term also goes when a later source describes its pair, compared with sorted constituents. Two tentative terms with one key: the larger is kept. In `combine_databases` each term added and each replacement is a rule entry in the decision log (`step1:rule:tentative:L(C14_LAVES,FE:FE,NB;0)`, `step2:rule:tentative-replaced:…` "TENT_MIX (…) -> assessed value from …"). A source subsystem whose differences come from tentative terms is "changed, explained" (or reproduced, if within tolerance), and the report says how close the terms bring it to the source (invariant ΔT, boundaries, grid).

**Turning it off or changing the values.** Pass `tentative=false`, or a dict such as `tentative={"mixing_per_atom": 30000, "vacancy": 150000}` (a value of `null` switches that rule off), to any of the four tools. The defaults come from `[merge]` in `calphad_agent.toml`: `tentative = true`, `tentative_mixing_per_atom = 50000`, `tentative_vacancy = 200000`. With tentative terms off, an assessed term still replaces an existing tentative one.

### Phase harmonisation

Two databases often describe the same phase differently: other names (C15_LAVES vs CET2, RE2TM17 vs CE2T17), another sublattice order, another formula unit (2:17 vs 2/19:17/19), other constituents, and one magnetic where the other is not. `db1` / `db2` / `rename` cannot express "keep this phase, add Co from the other one". Two tools and one merge decision can (`calphad_tools/harmonize.py`).

* **`normalize_phase_formula_unit`** changes the formula unit of phases: `target="sum_to_one"`, `"integers"`, explicit ratios, or `match_tdb` / `match_phase` (another database's site ratios, mapped onto this phase's sublattice order). Ratios written rounded (0.333:0.667) are first read as the exact fractions (see "Rounded site ratios" below), and the report says so. Site ratios are multiplied by k = new atoms / old atoms per formula unit, and every G and L parameter by k, T range by T range, so GM per mole of atoms is unchanged; TC is unchanged. Sum-to-one ratios are written as exact decimals whose sum pycalphad reads as 1 (2:17 → `0.105263157894737 0.894736842105263`). The result reports GM before vs after at random T and site fractions (relative 1e-9 for a non-magnetic phase) and runs `validate_edit`.
* **`extend_phase`** keeps `base_phase` of `base_tdb` authoritative and brings in `components` (default: the elements the add phase has and the base phase lacks) from `add_tdb`'s `add_phase`. Names may differ. It maps add's sublattices onto the base's (by their normalised, snapped ratios in every order and shared elements, or `sublattice_map`, add's sublattice i → base position; 0.667:0.333 against 1:2 is flipped and read as 2/3:1/3), harmonises the formula unit (below), and adds the new constituents where add has them (`add_constituents` restricts this; keys are add's sublattices, 0-based). It imports only add's parameters that involve a component (end-members and interactions); base parameters made of base elements are never touched. Keys present in both follow `collision` (`base`, `add` or `error`, each one reported). FUNCTIONs the imported parameters need are copied, renamed (`NAME_2`) where the base defines the name differently. A magnetic type definition is attached when the base phase had none. It lists end-members that now exist without G (pycalphad uses 0), each with whether it matters: no, if add or base already had it without G; yes, for a new combination of base and add constituents. It then checks two things. The base phase restricted to the base's elements must have the same GM per atom (1e-9). Add's GM per atom is compared on each of its subsystems with the components, e.g. Ce-Co; where base parameters govern (e.g. the base's Ce-Fe end-member in Ce-Co-Fe), the report names them and says the difference is expected. pycalphad reads interaction constituents sorted, without flipping odd Redlich-Kister orders, so imported interactions are written in sorted order with the value pycalphad uses. The note says Thermo-Calc reads the original with the opposite sign.
* **Formula-unit basis, and magnetism.** The IHJ magnetic term R·T·ln(β+1)·g(T/TC) is per formula unit and not linear in BMAGN (β is a Redlich-Kister sum, linear in the site fractions), so a magnetic phase cannot be rescaled exactly. With `magnetic="auto"` (the default) the basis follows one rule, the same everywhere a basis is chosen (`staged_merge` / `evaluate_compatibility` / `combine_databases` normalisation, `extend_phase`, the `extend` decision and `normalize_phase_formula_unit(match_tdb=…)`). For a matched pair of phases whose site ratios are proportional (after snapping), with A the base and B the added database:

  | A magnetic? | B magnetic? | Rescaled | How |
  |---|---|---|---|
  | yes | yes | A to B's basis | A's G and L × 1/k per T range, TC unchanged, A's BMAGN end-members transformed exactly, β' = (1+β)^(N_B/N_A) − 1 (N = atoms per formula unit; the AFM convention for β < 0), A's other BMAGN terms × k, no new parameters (default `magnetic_fit="endmembers"`; `"refit"` on request adds fitted interaction terms). GM per atom is exact at every end-member of A; the error between them is tolerated (directive) and reported in J/mol of atoms, max over T and site fractions of A's phase. B is untouched. |
  | no | yes | A to B's basis | exact (G and L × 1/k) |
  | yes | no | B to A's basis | exact (B has no magnetic terms) |
  | no | no | B to A's basis | exact (`extend_phase`: `normalization="base"` or `"per_atom"`) |

  In words: B's basis wins whenever B is magnetic, whatever B's atoms per formula unit. The one exception is a magnetic A against a non-magnetic B. By default (`magnetic_fit="endmembers"`, the user's directive) only the end-members are transformed exactly; A's other BMAGN terms are multiplied by k (the first-order form of ln(1+β') = k·ln(1+β)) and no parameters are added, so the error between end-members is tolerated and reported. With `magnetic_fit="refit"` (an option on `staged_merge` / `combine_databases`; the agent asks before using it) the interaction terms go through the least-squares refit (`harmonize.refit_bmagn`): the existing interaction terms, plus order 0-2 binary terms next to the magnetic end-members (reference `REFIT`), are fitted to the exact target (1+β(y))^k − 1. Example, a Fe₂Zr-type C15 (FE,MO,ZR)2(CE,FE,MO,ZR)1 with BMAGN(FE:ZR) = 3 against a magnetic (CE)1/3(CO,FE)2/3: "C15_LAVES: A (magnetic) rescaled to B's magnetic basis (sum 1), k = 1/3; BMAGN end-members exact, interior error <= 1.8 J/mol of atoms (directive: tolerated)", with BMAGN(FE:ZR) = 4^(1/3) − 1 = 0.5874 and no new parameters (with `magnetic_fit="refit"`: 15 fitted interaction terms and ≤ 0.055 J/mol of atoms). Order-disorder (DIS_PART) phases are never rescaled: they are only reordered and reported as `not_rescaled`. Explicit overrides remain: `"refuse"` (the old rule: the magnetic side's basis when only one side is magnetic, an error with the size of the other options' errors when both are), `"keep_parameters"` (TC/BMAGN as they are, the GM change reported) and `"refit"` (the transform above on whichever side `normalization` rescales). `normalize_phase_formula_unit` without `match_tdb` still refuses a magnetic phase under `"auto"`.
* **In `combine_databases` / `staged_merge`.** The plan proposes `extend` when an added phase looks like a base phase with extra constituents. That means the same formula from the names (CET2 ~ C15_LAVES ~ AB2; CE2T17 ~ RE2TM17) or shared structure tokens, proportional site ratios, a shared element, and elements the base phase lacks. Each question says which side sets the basis. Phase-model conflicts list `extend` first, marked recommended, where it is possible, then `db1` / `db2` / `rename`; each `db` option says what it drops ("A's C15_LAVES model only - drops CO from C15_LAVES (B's 5 C15_LAVES parameters lost: Ce-Co 3, Ce-Fe 1, Ce-Co-Fe 1)"). No policy or default ever resolves a phase-model conflict in a way that drops an element from a phase entirely: `policy="first"`/`"last"` leaves such a question open, and it is asked. `staged_merge` accepts `decisions={"phase:X": "extend"}` too. Answer `{"step2:phase:C15_LAVES": {"extend": {"phase": "CET2", "components": ["CO"], "sublattice_map": [1, 0], "normalization": "base", "collision": "base", "magnetic": "auto"}}}` (or just `"extend"`). Or pass `phase_harmonization={"C15_LAVES": {"source_phase": "CET2", "sublattice_map": [1, 0], "import_parameters": {"require_components": ["CO"]}, "site_ratio_normalization": "sum_to_one", "magnetic_model": "source"}}`, with optional `step` / `source`. The base phase is extended from B2, B with the base's unaries, so imported parameters use the base's unary functions, and B's source phase is left out of the merge. The decision-log entry records the mapping, the basis and k, the imported parameters, collisions, missing end-members, the GM checks and `validate_edit`. The GM checks also appear among the exactness checks. Validation views B's source phase under the base's name: B's subsystems with the new component are compared against B, and the base's subsystems are reproduced. `combine.ipynb` rebuilds the extension through `calphad_tools.harmonize` (that rebuild cell is marked [calphad_tools]).

### TDB inspection and validation

Three tools let the agent look inside databases and check its own edits without writing Python. Replies stay short (counts and the first rows, about 3,000 characters); give `output="name.xlsx"` (or `.csv`) for the full tables.

* **`query_tdb`** reads one TDB, or several for comparisons. Its `mode` is one of:
  * `phase`: the sublattice model with site ratios, constituents per sublattice, the sum of site ratios and atoms per formula unit (vacancy-only sublattices excluded; it also says where VA mixes with atoms). It shows the magnetic type definitions (AFM factor, structure factor) and the DIS_PART links in both directions. Parameters are grouped G, L0, L1, …, TC, BMAGN, others, with constituents, T ranges and expressions as written, or with every FUNCTION inlined (`expand_functions`).
  * `element`: every parameter whose constituents include all the given elements, by phase, with counts per subsystem.
  * `collisions`: across several TDBs, the parameter keys present in more than one, each identical, numerically equal (the inlined comparison `compare_system` uses) or different (with max |Δ|). It also covers phases defined in several files (same, permuted or proportional model with k, or different) and FUNCTIONs with the same name.
  * `functions`: definitions, which parameters and functions use each one, undefined references and unused functions.
  * `typedefs`: type definitions, the phases using them, and problems (undefined target or DIS_PART phase, a code no phase uses, a phase code with no definition).
  * `formula_unit`: atoms per formula unit per phase, whether it is exactly 1, and across TDBs whether the models are proportional (the factor k and the sublattice permutation).
  * `missing_endmembers`: end-members without a G parameter, optionally after adding constituents (`add_constituents={"0": ["NB"]}`, 0-based sublattices).
* **`resolve_tdb_symbol`** finds a FUNCTION such as GMOFCC in every TDB/.dat file under `search_paths`. By default it searches the working directory, `examples/tdbs` and the unary50 cache `~/.calphad_agent/sgte/`, and matches `.tdb`/`.TDB`/`.dat` in any case. It reports where the name is defined (file, line, expression, T ranges), where it is used, and whether the using file defines it. It suggests similar names: same element with another phase tag (GMOFCC ↔ GFCCMO, GHSERMO, GMOHCP), edit distance ≤ 2, and names shortened by the Thermo-Calc/OpenCalphad dialect writer. It shows the SGTE-style equivalent it finds, e.g. `G(FCC_A1,MO:VA) = GHSERMO+15200+0.63*T`, so `G(FCC_A1,MO) - GHSERMO = 15200+0.63*T`. From the referencing parameters it says whether the missing function should be that lattice stability (they add GHSERMO) or the full Gibbs energy.
* **`validate_tdb_edit`** compares an edited TDB (`after`) with the original (`before`) and gives a PASS/FAIL line plus one row per check, each with numbers:
  * pycalphad loads `after`;
  * writing it with the toolkit's TDB writer and re-parsing it gives the same parameters, functions, phases and type definitions;
  * constituents, function references (circular ones included), type definitions and duplicates: the `check_tdb` syntax findings, split into new problems (fail) and ones `before` already had;
  * odd Redlich-Kister terms newly written with non-alphabetical constituents. Thermo-Calc flips their sign; pycalphad does not;
  * a phase that had exactly 1 atom per formula unit still has;
  * phases outside `affected_phases` (default: every phase whose model or parameters changed) are identical, compared numerically with functions inlined;
  * GM per mole of atoms of each affected phase, before vs after, at `n_samples` random temperatures and site fractions. The site fractions are mapped through the sublattice permutation and proportional rescale detected from the models, or given as `sublattice_map` / `scale`. Relative 1e-9 counts as preserved; otherwise it reports the largest deviation in J/mol of atoms and where it occurs. The magnetic (IHJ) term is per formula unit and nonlinear in BMAGN, so a rescaled magnetic phase reports "non-magnetic part preserved; magnetic contribution changed by X J/mol of atoms", and GM is also compared at the phase's end-members. A magnetic basis rescale (BMAGN end-members transformed exactly) passes: "magnetic basis rescale: non-magnetic part preserved, end-members exact; magnetic contribution changed by X J/mol of atoms between them (tolerated)". Anything else magnetic is a warning, not a failure.

  `expect` can add `"only affected phases changed"` (no additions elsewhere either), `"base subsystems unchanged"`, `"energy_per_atom_preserved"` (the GM check must pass) and `"site ratios sum to 1"`. `phase_map` handles renamed phases. In Python: `from calphad_tools.tdb_validate import validate_edit; r = validate_edit("before.tdb", "after.tdb", affected_phases=["SIGMA"], expect=["energy_per_atom_preserved"])`. It returns `r["verdict"]`, `r["checks"]`, `r["gm"]` and `r["text"]`; `before`/`after` may also be parsed `tdbparse.TDB` objects.

### Problem reports and feedback (hand a bad answer straight to Claude Code)

When an answer is wrong or the agent can't do what you asked, a self-contained bundle is saved in `calphad_outputs/problem_reports/<time>_<slug>/`, with a matching `.zip` you can send to the developers. It's created in one of four ways:

* **by telling the agent**, e.g. "that's wrong, the σ field is too wide, send it to the developers". The agent asks why if you haven't said, then goes through the files with you. Databases may be proprietary, and figures or tables made from them can reveal their content, so it asks questions like "I understand Fe-Cr-Ni-internal.tdb is proprietary; may the ternary figure made from it be included?". It then packages exactly what you agreed (`package_feedback`). Before anything is written, you confirm the final list yourself, so the model can never consent for you.
* **with the button or command**: the **Report problem** button in the web UI opens a form. You write why the answer isn't good and tick the files that may be shared; each output shows which inputs it was made from. `/report [last] [why]` in the terminal does the same with a numbered list: all files, pick by number, none, or none plus no tool-result text.
* **by the model**, through the `report_problem` tool, with its own diagnosis (missing capability, tool bug, wrong result).
* **automatically**, when a turn ends in failed tool calls plus an "I couldn't" answer, or crashes. Turn this off with `auto_failure_reports = false`.

**What a bundle contains.** `REPORT.md` is written for a coding agent and contains:

* your statement of why the answer isn't good, at the top;
* every request since the last `/reset` (or only the last one), with the agent's answers and intermediate messages;
* the questions it asked you and your answers;
* safety-review verdicts on generated Python;
* every tool call with its full arguments and results (complete tracebacks for failures);
* the source file and line of each tool;
* environment and versions, with keys redacted;
* fix instructions.

Alongside it: `report.json`, the files you agreed to share (`inputs/`, `outputs/`) and `replay.py`, which replays the tool calls without an LLM.

**Sharing is opt-in.** Files you didn't agree to share are listed by name, size and SHA-256 fingerprint only, so a developer can ask for exactly those files. Reports nobody asked for (automatic ones, or model reports with nobody at the keyboard) never copy your files. The most private option also leaves out the text of tool results, which can quote database contents such as phase names and parameter values.

To use a bundle, give the zip or folder to Claude Code: *"fix the problem in calphad_outputs/problem_reports/…zip"*.

### Code behind a figure (for notebooks and provenance)

**Surveying an element system.**

* `phase_inventory` takes a list of elements and lists every phase of that system across all databases that contain them. For each phase and database it gives:
  * the sublattice model in CALPHAD notation, e.g. (Cr,Fe)₁(Va)₃;
  * the constituents of each site, and what else the full database allows;
  * the model type (substitutional or interstitial solution, stoichiometric compound, ordered with a disordered part, ionic liquid, …);
  * magnetic and order–disorder settings;
  * how many parameters the phase has in the system.

  Phases match across databases by exact name; likely matches (SIGMA vs SIGMA_D8B) are listed as suggestions but never merged. The output is an Excel file with a phase × database matrix, a CSV and a PowerPoint.
* `system_coverage` answers "what are the statistics for the Fe-Ni-Cr-Mo-Co-V-Cu system?":
  * how many binaries and ternaries (and optionally quaternaries) the system has;
  * which ones are assessed, meaning some database has parameters for exactly those elements, and in which database;
  * which ones would only be extrapolated;
  * for each ternary, how many of its binaries are assessed.

  It scans every database in the folder, including binary-only files. For example, a folder with a Fe–Cr and a Fe–Ni database gives Cr–Fe–Ni: 2 of 3 binaries and 0 of 1 ternary assessed. Subsystems assessed in different databases are not automatically compatible with each other; the report says so.

**Referring to earlier results.** Requests like "take the previous plot and add this data point" or "same diagram at 1200 K" work without naming files:

* Any file argument accepts a reference:
  * `@last`: the latest image;
  * `@last.csv`: the latest file of that type;
  * `@last:map_phase_diagram`: the latest result of that tool;
  * `@call:12`: the result of call 12.
* The reference is replaced by the real path before the tool runs, so logs, `show_code` and replays see ordinary files.
* Adding points to the last plot is `edit_figure` with `image='@last'`. It edits the stored figure, so nothing is recalculated.
* `recent_results` lists the latest outputs with their references.
* `rerun_call` repeats a call with some arguments changed.
* After a restart, both fall back to the previous session's log.

`show_code` takes any output (image, CSV, or call number from `session_log`) and returns standalone Python using only pycalphad, numpy, pandas, scipy and matplotlib, as a Jupyter notebook: a record of how the result was made.

* It follows the chain back: for an edited figure you get the phase-diagram cell, then the edit cell.
* It also works for outputs from earlier sessions, via the provenance records.
* A step with no standalone equivalent is reproduced through the toolkit and clearly labelled.
* Formats: `format: "notebook"` (default) writes `code/<name>_standalone.ipynb` and the same code as `code/<name>_standalone.py`; `"script"` / `"cell"` write only the `.py`.

**A notebook next to every phase diagram.** Each call that draws a phase diagram also saves `<name>.ipynb` next to its PNG, e.g. `map_NI-CR_CrNi-06Hal.ipynb` beside `map_NI-CR_CrNi-06Hal.png`. The reply has one extra line, `notebook: <path> (regenerates all files below)`. Run it top to bottom and it writes the same files to `standalone_outputs/` next to it: PNG, SVG and PDF (editable text, Inkscape subscripts), the boundary, tie-line and invariant CSVs (isopleths: ZPF-line and invariant CSVs), and the summary JSON (same keys; run times and paths are the new run's).

| Tool | Notebook |
|---|---|
| `map_phase_diagram` (binary, ternary, isopleth; T–μ via the toolkit) | `<png stem>.ipynb` next to the PNG |
| `plot_diagram_with_data` | next to its PNG |
| `plot_all_binaries` / `plot_all_ternaries` | `<deck>.ipynb` next to the deck: one cell per subsystem (all its files), then the grid images and the deck |
| `edit_figure` on a phase diagram (also `label_phase_regions`, `move_label`, `delete_label`, `relabel`) | next to the edited PNG, with the diagram cell first; an edit in place refreshes it |

The notebook is generated from the call log only (nothing is recomputed, about 0.05 s). If generating it fails, the tool's result is unchanged and one warning line is logged. Turn it off with `save_notebooks = false` under `[agent]`.

**Same lines on every run (hash seed).** pycalphad's mapping visits sets of phases in an order that depends on Python's hash seed (`PYTHONHASHSEED`, random per process unless set). So two runs can trace a few lines differently, for example an extra short segment or a line traced from its other end. The boundaries and invariants agree either way. The toolkit therefore runs with `PYTHONHASHSEED=0` unless you set one:

* `start-agent.sh` / `start-agent.bat` set it;
* `calphad-agent` and `calphad-tools` re-run themselves once at start-up with it when it is unset. An interpreter cannot change its own seed, and single diagrams are mapped in the agent's own process. On Windows the command runs as a child process instead;
* diagram worker processes (`plot_all_*`) and background jobs get it too, also when the toolkit is used as a library.

Each diagram's JSON and provenance record the seed (`"hash_seed": "0"`, or `"random"`). The notebook's first cell compares it with the kernel's. If they differ, it prints how to get identical files: `PYTHONHASHSEED=0 jupyter lab`, or `run_with_recorded_seed()`, which runs the notebook's code in a fresh Python with that seed. The results are valid either way.

**Tools with standalone code:**

* **Calculations:** `map_phase_diagram` (binary, ternary isotherm, isopleth), `step_map`, `equilibrium_property_plot`, `phase_property_plot`, `export_equilibrium_data`, `scheil_solidification`.
* **Models, databases and MCMC:** `export_phase_model`, `write_database`, `analyze_mcmc`, `mcmc_to_tdb`.
* **Overviews and ESPEI plots:** `plot_all_binaries`, `plot_all_ternaries`, `plot_diagram_with_data`, `plot_parameter_vs_data`.
* **Plotting:** `make_plot` and `edit_figure` with all 69 plot operations, `compose_images`, `image_tools`, `images_to_pdf`, `figures_to_pptx`, `animate`.
* **Data:** `transform_data`, `fit_data`, `convert_composition`, `calculate_equilibrium`, `run_python`.
* **Digitised data:** the extracted points are embedded as a table, so later steps run without the image analysis.

**Falls back to the toolkit:** T–μ diagrams, wildcard site fractions `Y(...*...)`, extracting a subsystem with `write_database`, animation sweeps and `.mp4`, and `digitize_setup`.

**Checks:** the tests run each generated script in a clean process and compare it with the tool's own output: CSVs to a relative tolerance of 1e-6, and figure artists (lines, collections, texts, limits, labels). Phase diagrams are compared point by point (within 0.02; invariants within 0.5 K), because pycalphad's mapping order varies with Python's hash seed. `tests/test_notebooks.py` runs the auto-saved notebooks (Cr-Ni, Co-Cr-Ni at 1000 °C, an isopleth) and checks that they write the same set of files with equivalent content. With `PYTHONHASHSEED=0` the CSVs match exactly.

`equilibrium_property_plot` accepts all of these quantities:

* `NP(*)`, `X(PH,EL)`, `MU`, `ACR(EL,REFPH)`
* `HM`, `SM`, `CPM`, and `HM.T` for the apparent Cp
* `DOO`, `TC`, `VM`, and site fractions `Y`
* units written in brackets, e.g. `HM[kJ/mol]`
* `DF(PH)` for driving forces, `DORMANT(PH):X(EL)` for the nucleus composition, `ISOLATED(PH):PROP`, and `T0`

It draws a line plot for one varying condition and a 2-D map for two. `scheil` is now a dependency.

## How the pieces behave

**Compare.** Each item is classified as one of:

* **identical**: same text, ignoring whitespace, number formatting and a leading `+`.
* **equivalent**: written differently but numerically equal over 298–3000 K, within `rtol = 1e-4`.
* **different**, **only A** or **only B**.

Matching and alignment rules:

* Constituents are sorted before matching, and odd Redlich–Kister orders get their sign flipped when a pair is reordered. Ternary orders are remapped.
* `L(...)` is treated as the same parameter as `G(...)`, and `BM` as `BMAGN`.
* Phases whose models differ only by a vacancy-only sublattice are aligned. For example, `(CR,FE)1(VA)3` matches `(CR,FE)1`.

**Comparing several databases for one subsystem (`compare_system`).** Use `compare_tdbs` for a full diff of two
files (every element, phase, function and parameter). Use `compare_system` to put two or more databases side by side
for one subsystem, for example "compare the Fe-Cr part of these three files".

* **Which files.** Give the files, or a folder (and a name pattern) to take every database there that contains the
  elements. `find_databases` lists the databases in a folder with their exact paths, elements and how many
  parameters each has for the subsystem. File names are matched exactly: a name that does not exist gives an error
  with the closest names ("did you mean"), and no other file is ever used in its place. A name without its
  extension is accepted when only one `.tdb`/`.dat` file has that name.
* **What is compared.** Only the part of each database inside the subsystem: the phases that can exist with those
  elements (plus vacancies), the parameters whose constituents are all in the subsystem (pure elements, end-members
  of sublattice phases such as `G(SIGMA,FE:CR:CR)`, interactions), and the functions they use. Parameters with Ni,
  Co and so on are left out.
* **What the tables show.** Each parameter as it is written in each file, for example `GHSERFE+5000` or
  `-14190 + 2.29997*T`. Function names such as GHSERFE are kept. A parameter with several temperature ranges shows
  one line per range. ESPEI's `VV0001`-style names are only handles for the fitted numbers, so they are replaced by
  their values, written with the precision the file uses; a negative value becomes a minus sign on the term.
  Rows are sorted by phase (LIQUID first), then G terms (pure elements first), then L0, L1, L2 for each pair of
  constituents, then ternary terms, then TC, BMAGN and the rest.
* **Three kinds of table.** "Differences" comes first in the deck: every parameter whose value differs between the
  files (checked numerically over 298–3000 K with all functions substituted, relative tolerance 1e-4 of the largest
  value plus 1e-3 J/mol, or 1e-6 for TC/BMAGN), and every parameter missing from some file. Parameters written
  differently but numerically equal (`-311.50` and `-311.5`, terms in another order, `GHSERFE + 5000` and the same
  polynomial typed out) are not differences, and neither are rows that differ only in their reference. The rows
  written differently get their own table right after "Differences" ("Numerically equal within tolerance (written
  differently)", with the largest difference); "All parameters" still lists everything. When the files of a row do not use the same functions (one writes `GHSERFE + 5000`, another the
  polynomial or `GBCCFE`), or a shared function such as GHSERFE is defined differently, that row shows every cell
  with the functions substituted (one polynomial in T per range, terms collected) so the cells compare term by term;
  the grey note on the left says "functions expanded for comparison: GHSERFE, GBCCFE". The Excel sheets keep the
  as-written text in extra columns. Then each file gets its own table of parameters (as written) and of the
  functions they use. With more than four files the columns are split across slides.
* **References.** Each parameter's reference is the key after its last temperature range (`... 6000 N 91Din !`,
  `N REF283 !`; a comment right after the `!`, as in `N ! $1967Kau`, is used when there is none). If the file has a
  reference list (`LIST_OF_REFERENCES` / `ADD_REFERENCES`, or a `$`-comment list such as `$ [1967Kau] ...` or
  `$ REF7: ...`), the key is looked up there. On the slides each cell has a small grey line with the key and, when it
  was found, the first author and year (`82Her · Hertzman & Sundman 1982`); the per-file tables have a reference
  column. The Excel sheets have the key and the full citation for every database, and the long CSV has `reference`,
  `reference_short` and `reference_citation` for every parameter.
* **Numbers.** The Excel file also has the values at the chosen temperatures, each parameter with all its functions
  substituted (terms added up: constant, T, T·ln T, T², 1/T, T⁷, T⁻⁹ ...), and a term-by-term sheet. The tool also
  notes when a database uses element references that are not SER (for example Kaufman lattice stabilities with
  fcc Fe = 0), because its Gibbs energies are then not directly comparable.
* **Phase names.** Phases are matched by exact name only. Phases that might be the same under another name
  (`SIGMA` and `SIGMA_D8B`, `BCC` and `BCC_A2`) are listed as candidates, with the evidence: site ratios,
  constituents, magnetic model, disordered part. Nothing is merged automatically. The agent decides, asks you if it
  is unsure, and runs the comparison again with a `phase_map`, for example `{"SIGMA_D8B": "SIGMA"}` for every file,
  or `{"CrFe-82Her.tdb": {"SIGMA_D8B": "SIGMA"}}` for one file.
* **Derived quantities.** `expressions` take TDB-style parameter names, function names, T and numbers, for example
  `G(SIGMA,FE:CR:CR;0) - 10*GHSERFE - 20*GHSERCR`. `formation` does the subtraction for you:
  `{"phase": "LAVES_C15", "constituents": "FE:CR", "per": "mole_of_atoms"}` gives G of the end-member minus the
  site-ratio-weighted reference of each element, using each database's own reference (the element's G in its
  reference phase, whatever the function is called), and says which references it used. If no phase is given, as
  in `G(FE,CR)`, the tool lists the matching end-members in each database instead of guessing.
* **Output.** One PowerPoint (title, databases, which phases each has, possible phase correspondences,
  differences, numerically equal rows, all parameters, one table per file, derived expressions), an Excel workbook with the same tables plus
  the values and substituted expressions, and a long CSV. `plot_diagrams: true` adds one slide with the binary
  diagrams of all databases drawn on top of each other (slower). `show_code` gives a standalone script that rebuilds
  the tables, CSVs, Excel file and a plain PowerPoint.

**Merge.** These are resolved automatically and logged in the merged file's header:

* identical or numerically equivalent definitions (e.g. SGTE data typed with different decimals); the earlier file wins
* vacancy-sublattice model differences; the parameters are converted
* constituent unions
* type-definition re-lettering, so codes never clash

Everything else becomes a conflict with a stable ID for you to decide:

* different functions (you can also keep both, with the second one renamed)
* incompatible sublattice models
* phase-name aliases (`SIGMA` vs `SIGMA_D8B`)
* differing parameters, grouped by subsystem so one answer (`system:CR-FE`) can cover a whole binary
* `REJECT_PHASE` defaults that would disable a phase another file uses

The merged file is validated by loading it in pycalphad and building every phase model.

**Can two TDBs be merged? (`evaluate_compatibility`, then `staged_merge`).** A is the base, B the database to add.

First, B adopts A's conventions; this is reported:

* **Phase names.** Clear aliases are renamed with their parameters, type definitions and DIS_PART references: `C15_LAVES`/`LAVES_C15`/`C15`, `SIGMA`/`SIGMA_D8B`, `A2_BCC`/`BCC_A2`. Weaker matches are only listed; apply them with `phase_map`. Two phases of B are never merged. Pairing is **one-to-one**: the confident pairings come first (same names, `phase_map`, clear aliases, `extend` decisions, answers already given), and those phases leave the candidate pool. Weaker candidates are then sought only among the phases still unpaired. If A's LAVES is paired with B's LAVES_C15, nobody asks whether LAVES is also LAVES_C14; LAVES_C14 can still be offered for another unpaired phase of A.
* **Rounded site ratios.** Ratios written as a rounded form of simple fractions are read as those fractions, silently: 0.333:0.667 is 1/3:2/3, 0.105:0.895 is 2/19:17/19, 0.538:0.154:0.308 is 7/13:2/13:4/13 (`tdbparse.snap_ratios`). The rule: the smallest integer formula with at most 30 sites in total whose proportions match every ratio within 2·10⁻³ relative, or as the ratio's own rounded decimal (at least 3 significant digits). 0.4:0.6 is 2:3 exactly and stays as written; 0.41:0.59 matches no simple formula and stays a model difference. The merged phase is written with the exact ratios of whichever basis the rules pick: `1 2` for an integer basis, `0.333333333333333 0.666666666666667` for a sum-to-one basis (pycalphad's sum is exactly 1). This is a rule note (`snapped`, or `A snapped` when the base's ratios are the rounded ones, then written to `A0_…`), never a question or a warning. The note gives the GM change, which is tiny because only the weights of the configurational entropy move (at most a few J/mol of atoms at 3000 K). `validate_edit` and the exactness checks compare against the exact-fraction version, so a snap does not count as a GM change.
* **Sublattice order.** B's sublattices are matched to A's by their normalised ratios (each divided by its sum), in every order (brute force, at most 5 sublattices in practice), as written or else snapped. 0.333:0.667 against 2:1 is therefore flipped automatically (k = 3): no question, and a rule note ("flipped sublattices to match A's order"). Equal sizes are decided by shared elements: Fe on the 2-site sublattice of both a Fe₂Mo and a W:Fe 1:2 C14. If that is undecidable, the tool stops and asks for `sublattice_map {'C14': [1, 0]}`. This happens with no shared element (Na:Cl vs K:Br), with overlap pointing both ways, or with a different number of sublattices. The magnetic basis rule, `extend_phase` and `normalize_phase_formula_unit(match_tdb=…)` use the same matcher.
* **Formula unit.** B's site ratios are rescaled to A's when proportional (4:2 → 2:1; 0.333:0.667 → 1:2 with k = 3 exactly, from the snapped values), with G and L × k, when B's phase has no magnetic terms (whether or not A's has). The note says which side set the basis and why, k, and that it is exact.
* **How rescaled parameters are written.** The factor is folded into each term, as for VV#### values, and checked numerically against the exact product: `(GHSERWW+2*GHSERFE-30000+6*T)/3` × 3 is written `GHSERWW+2*GHSERFE-30000+6*T`, and × 1/3 `GHSERWW/3+2*GHSERFE/3-10000+2*T` (a coefficient with no short decimal stays a fraction). Each T range is scaled separately; a plain number stays a number; an expression that cannot be parsed falls back to `k*(...)`.
* **The magnetic basis rule.** If B's phase has TC/BMAGN parameters, B's basis wins, whatever B's atoms per formula unit (0.75:0.25, 4:2, 1/3:2/3). A's phase takes B's site ratios, in A's sublattice order, and A's G and L are scaled by 1/k per T range. When A has no magnetic terms this is exact. When A is magnetic too, TC stays, A's BMAGN end-members are transformed exactly (β' = (1+β)^(N_B/N_A) − 1), its other BMAGN terms × k, no new parameters unless `magnetic_fit="refit"` (see "Formula-unit basis, and magnetism" above); GM per atom is then exact at A's end-members and changes slightly between them, which the directive tolerates. The note reports it, e.g. "A rescaled to B's basis - A (magnetic) rescaled to B's magnetic basis (sum 1), k = 1/3; BMAGN end-members exact, interior error <= 1.8 J/mol of atoms (directive: tolerated)". The rescaled A (`A0_…`) is used for B2, B3, Q, A1, A2 and the replot decision. B's parameters, including its magnetic ones, are unchanged (only reordered). Order-disorder phases (A's own DIS_PART link or a magnetic disordered part) are not rescaled: the phase is only reordered and reported as "not_rescaled", with the reason. A magnetic A against a non-magnetic B is the one case where B is rescaled to A (exact).
* **Checks.** The tests check that GM per mole of atoms is unchanged (< 1e-9) after every normalisation (against the exact-fraction version where ratios were snapped). An exact proportional rescale (G/L scaled alike, no magnetic terms) is not counted as a change in the replot decision or the version comparisons; it is named in one line instead. A magnetic basis rescale is compared the same way for G/L and TC, while its non-zero BMAGN terms count as changes, so the subsystems where the magnetic term moves are replotted. `combine_databases` attributes those changes to the rescale's decision-log entry (changed, explained, or reproduced within the tolerances), and its exactness check for that phase is GM at A's end-members.

`evaluate_compatibility` then checks in stages, with one short verdict each:

1. **SGTE unaries.** The SER Gibbs energies are compared over 298.15–6000 K. If they differ, each is compared with SGTE's unary database 5.0, which says which file matches it.
2. **Other unary terms.**
3. **Shared binaries.**
4. **Ternary and higher terms.** Stages 2–4 are compared numerically over 298.15–3000 K, with functions inlined.
5. **Phase models.** For each shared phase: the merged (union) model, and the new end-members it creates in each database's subsystems. For example, A's C14 (Fe)₂(Mo)₁ plus B's (Fe,W)₂(Fe,W)₁ puts Fe:Fe into A's Fe-Mo. The stage also says which file supplies their G, or that none does (pycalphad would use 0). It also reports magnetic/ordering differences and phases that newly appear in the other database's space.
6. **Replot decision.** The A-priority merge is built, and every binary and ternary of A and of B is compared with it. A subsystem is provably unchanged when all its parameters are numerically equal and its phase models are identical (site ratios, constituents, magnetic and order-disorder settings). Otherwise it is listed with its reasons.

Details go to an Excel file and a 4-slide PowerPoint.

`staged_merge` writes the intermediate databases to the output folder:

* **B1:** B with A's SER functions.
* **B2:** + A's other unaries of the shared phases.
* **B3:** + A's description of the shared binaries.
* **Q:** the full merge. A wins what the stages settle. Ternary conflicts, incompatible models and type definitions are asked as in `merge_tdbs`.
* **A1, A2:** if B has unary or binary terms A lacks, A + B's unary terms, then A + B's unary and binary terms.

Each version is checked in pycalphad. For B…Q, every binary and ternary isotherm of B's system is mapped in the shaded style. A subsystem that is provably unchanged from the previous version is reused, not recomputed. For A→A1→A2, only A's subsystems that change are mapped.

Only consecutive versions are compared: parameters, phase models, and invariant reactions that moved (ΔT, Δx). `force_replot` (`true`, or a list such as `["Fe-Mo"]`) maps subsystems whose change is exact (an exact rescale, or numerically identical parameters) again instead of reusing them. Each forced figure is labelled with its largest invariant shift, e.g. "Fe-Mo: exact rescale, replotted on request – max invariant ΔT 0.00 K, Δx 0.0000". A shift above 0.5 K or 0.001 is flagged in the text, the slide and the summary row as a failed exactness claim. Ternary isotherms compare tie-triangles. The deck has one summary slide, then per step its **TDB changes** and one slide with the changed diagrams of both versions.

**TDB changes** (`calphad_tools/tdbdiff.py`) show, at a glance, which lines of the TDB a step changed. Each line is the full command as the version file has it, from the keyword to the `!` (`PARAMETER G(C15_LAVES,CE:CO;0) 298.15 -12000; 6000 N REFB !`), in Courier New, wrapped inside the row; only whitespace is normalised, functions are not inlined and VV#### names stay as written. A removed or old line is in red text with a "−" gutter, and the line that replaces it follows immediately. Added and new lines are in the slide's normal text colour (black on a white slide, white on a dark one) with a "+" gutter. Unchanged lines are left out, including lines that differ only in formatting (number format, blanks, `PAR` vs `PARAMETER`, type-code letters a merge re-assigns, a reference key moved out of a trailing comment). An old line is paired with its new one by identity: the parameter type (G = L), the phase through the rename map (normalisation, the `rename` decision, aliases, `extend`), the constituent array through the sublattice permutation, the order and the diffusing species; FUNCTIONs by name (and through `keep_all` function renames such as `GHSERFE_2`); PHASE, CONSTITUENT and a phase's TYPE_DEFINITION by the phase. A line whose text changed only because of a rename or a reorder is still shown as a pair. Groups are always in the same order: ELEMENT / SPECIES, FUNCTION, TYPE_DEFINITION, then one group per phase, alphabetical by its new name, with a grey header ("MU_D85_2 (was MU_D85 in B3)", "C15 (sublattices reordered, new from old [1, 0])", "MU_D85 (new in Q)"). Inside a group come PHASE and CONSTITUENT, then G, L of order 0, 1, 2…, TC, BMAGN and others, by constituent array. Long steps continue over several slides ("B3 → Q: TDB changes (2/3)"). A pair is never split and a header never ends a slide. After 8 slides per step (`DIFF_MAX_SLIDES`) the last row says "… N more lines: see diffs/B3-Q.diff". A step without changes says "no TDB changes (numerically identical)". When B or A was normalised (renamed, flipped, rescaled, snapped: B as given → B, A → A0), those changes come first, with the same view.

Every step's full diff is also written to `diffs/<old>-<new>.diff` (`norm-B.diff`, `norm-A.diff` for the normalisation) in unified-diff style: `@@ group`, then `-` old and `+` new lines, grouped as on the slides. The Excel workbook has one sheet per step, `TDB B2-B3`, with the columns group, change (removed / added) and line; removed lines are in red.

For example, A1 of a stoichiometric Fe₂Mo C14 plus a Fe-W C14 with Fe:Fe turns the Fe₂Mo line compound into a wide C14 field: the Fe-Mo eutectic moves +105 K. Merging CrFe-82Her into Choi's Cr-Fe-Ni (B2) swaps the fcc-Cr lattice stability (82Her vs SGTE). That moves the Cr-Ni eutectic by −85 K.

`unary50.tdb` is downloaded once from SGTE (<https://www.sgte.net/en/free-pure-elements-database>) to `~/.calphad_agent/sgte/`, only when needed and never into the repository. You can also give `unary50='path'`, or set `$CALPHAD_UNARY50` on offline clusters. SGTE's terms: "This database can only be used for extracting data for assessment work or to tabulate or plot data for the pure elements". Cite A.T. Dinsdale, Calphad 15 (1991) 317–425.

`show_code` gives standalone scripts:

* **`evaluate_compatibility`:** the same stages and Excel file.
* **`staged_merge`:** the same TDBs, the version differences, and pycalphad binplot/ternplot figures of the changed subsystems.

**Experimental data.**

* **POP files** convert automatically. This includes `TABLE_HEAD` blocks and abbreviated commands and phase names (`CH-ST PH LIQ=FIX 1`, `LIQ` → `LIQUID` via the TDB). Thermochemical points such as ACR or HMR are reported as skipped.
* **DATAPLOT .exp files** carry no phase information, so each DATASET must be mapped to phases. The LLM does this from the dataset titles and comments or the paper text; the CLI asks you.
* **All datasets** are checked with ESPEI's own `check_dataset` before they are written.

**ESPEI + Flux.** There are two layouts:

* **`per-job`**: a 1-core generation job, then an N-node MCMC job, chained with `flux batch --dependency=afterok`. The MCMC job starts a dask scheduler plus one `dask worker` per core via `flux submit`, and points ESPEI at the scheduler file.
* **`bundle`**: one allocation in which each ESPEI run is dispatched with `flux submit`.

The generated YAML is validated against ESPEI's schema, and the scripts are checked with `bash -n`.

**Local ESPEI runs and background jobs.** The usual workflow is a short local run to check the inputs, then the Flux batch for production. The local tools write the same YAML as the Flux generator, so both runs use one configuration; only the scheduler differs.

* `espei_validate_input` checks the YAML against ESPEI's schema, the phase models, every dataset, and whether the datasets match the phase models. It also flags datasets hit by a known ESPEI 0.9.1 bug: an equilibrium-property dataset with several temperatures *and* several compositions is compared in the wrong order. Split such datasets so only one of pressure, temperature or composition varies.
* `espei_generate_parameters` runs parameter generation (ref state SGTE91 or SR2016, AICc penalty factors, fitting description). It writes `dft.tdb` in `<output>/espei_runs/<name>/` and lists the chosen parameters of each phase with their values. Small inputs run straight away; large ones become a background job.
* `espei_input` (on `espei_generate_parameters`, `espei_run_mcmc` and `optimize_system`) takes an existing ESPEI input YAML, such as a tutorial's `run_param_gen.yaml`. Its phase models, datasets, `tags` and generation or MCMC settings become the defaults, and explicit arguments override them. `tags` can also be given directly, for example to exclude `idmix` and `mag` from DFT data. MCMC runs inherit the tags of the generation run with the same name.
* **Live end-to-end check:** `tests/espei_cumg_live.py --profile claude --scenario files|image` runs the agent on the ESPEI Cu-Mg tutorial (`examples/espei/cu-mg/`). The `files` scenario uses the tutorial's datasets; the generated parameters must equal the tutorial's. The `image` scenario gives the agent only a plot of the ZPF points, which it digitizes into ZPF datasets. Both then run a short MCMC, and the results are scored against the tutorial's data.
* `espei_run_mcmc` starts MCMC as a background job on a dask cluster on this machine: half of the available cores by default (`[hpc] local_cores`; never more workers than chains), `cores` to choose, `cores: 1` for a serial run. It supports restarting from a trace, a subset of symbols, priors and data weights. `trace.npy`, `lnprob.npy` and `mcmc.tdb` land in the run folder, where `analyze_mcmc` and `mcmc_to_tdb` find them. A folder that already holds a trace is never overwritten; the new run goes to `<name>_2`.
* Background jobs run in their own process, so they keep going if the agent exits. `list_jobs`, `job_status` (progress, ETA, log tail, results and plots; for MCMC also an lnprob plot at any time), `wait_job` and `stop_job` follow them. Stopping an MCMC run ends it after the current iteration and keeps the chains so far. Job files are in `<output>/jobs/<id>/`.
* `show_code` on a file made by one of these runs gives standalone code that writes the ESPEI YAML inline and calls `run_espei`.

On the Cr-Ni test system (15 parameters, 30 chains), generation takes under a second and each serial MCMC iteration 1.3 to 2 s.

## Notes and limitations

* **Speed and caching:** the tools keep parsed databases, phase models and compiled code in memory, so a second calculation on the same system is much faster (for example 0.9 s to 0.07 s for a 4-component equilibrium). Editing the TDB file clears its entries automatically. The code from `show_code` does not use this cache and still runs on its own.
* **FactSage/ChemSage `.dat` files:** the calculation tools accept a `.dat` file wherever they take a TDB, using pycalphad's reader. `convert_database` writes a TDB from it. Quasichemical liquids (MQMQA/SUBQ/SUBG) and Kohler-Toop (QKTO) solution terms have no TDB form, so those phases are left out and listed. Thermo-Calc and OpenCalphad output also refuses non-integer powers of T, which are common in `.dat` heat capacities; use the `pycalphad` dialect for those. The text-based tools (merge, compare, check) need the converted TDB.
* **Phase diagram maps:** pycalphad's mapper can occasionally miss a region in complicated systems. Check critical diagrams, and use `--exclude GAS` when a gas phase slows things down. Each diagram runs in its own process with a timeout, so one bad system can't hang the deck. Timing examples: ternaries of the 23-phase Al-Co-Cr-Ni database take about 20 s to 2.5 min each; the 10 binaries of Co-Cr-Fe-Ni-V take about 30 s in total.
* **Scanned papers:** reading tables out of scanned papers depends on the model's vision ability (`view_pdf_page`). Data digitized from figures (e.g. WebPlotDigitizer) goes through the CSV path: `T_C, phase_1, X(CR)_1, phase_2, X(CR)_2, reference`.
* **Offline testing:** `tests/fake_openai_server.py` is a scripted OpenAI-compatible server for testing without a key.
  * `python tests/demo_web_offline.py 8791` runs the web UI against it.
  * `python -m pytest -q tests` runs the 16 tests.
* **Example TDBs** in `examples/tdbs` come from phasediagrams.org and the TDBDB (Brown University, van de Walle group) Elsevier supplementary files. The files in `examples/expdata` are synthetic, for testing the converters.
