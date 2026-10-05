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
| 6 | ESPEI inputs + Flux batch scripts (multi-node dask MCMC, or many runs in one allocation) | `espei-flux`, `phase-models` | `generate_espei_flux_batch`, `make_phase_models`, `espei_generate_parameters`, `espei_run_mcmc` |
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

`pycalphad-agent-venv/bin/calphad-agent --check` tests the connection. `--setup` asks the questions again.

**On a cluster (Linux with environment modules):**

* **Python module:** if no Python 3.13 is on the PATH, `install.sh` looks for one among the modules (Lmod or Environment Modules). It recognises names like `python/3.13.1`, `Python/3.13.1-GCCcore-14.2.0` and `python3/3.12`, prefers 3.13, then 3.12, then 3.11, and loads the best one.
  * `./install.sh --which-python` shows what it would pick.
  * If the Python module needs a compiler module loaded first (hierarchical Lmod), load both yourself and run the installer again, or point it at a Python: `PYTHON=... ./install.sh`.
* **Environment file:** the installer writes `pycalphad-agent-env.sh`, which loads that module and activates the environment. `start-agent.sh` sources it. The ESPEI/Flux batch scripts the agent generates also start with it, so jobs on compute nodes use the same Python.
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
api      = "chat"                       # or "responses" if your host serves /v1/responses
vision   = true                         # false if the endpoint rejects images

[agent]
workdir     = "examples"                # folder with your TDBs and data
python_tool = "auto_verify"             # ask | auto_verify | off
```

**Rate limits and token use.** On a 429 ("exceeded token rate limit"), the agent waits as long as the endpoint asks, shows it in the UI, and resends the *same* request. By default (`tokens_per_minute = "auto"`) it also sets its own pace from that first limit: about 80% of what it sent in the last minute. After that it spaces its requests to stay below the limit for the rest of the session, tightening further if it's still limited. Set `tokens_per_minute` to your deployment's quota to pace from the start, or to 0 to disable pacing.

The conversation is append-only, and each conversation sends one `prompt_cache_key`, so repeated requests reuse the provider's prompt cache. That makes them cheaper and faster. Whether cached tokens also count toward your TPM quota is up to the provider. `/usage` and the web header show input tokens, the cached share, and the current pace.

**Tool descriptions on demand.** By default (`tool_descriptions = "compact"` under `[agent]`), only the about 15 everyday tools are sent in full: databases, equilibrium, phase diagrams, plotting, viewing images, recent results, `show_code`. Each of the other about 70 tools is sent as one line plus its argument names, ending in `[tool_help]`. Before using one for the first time, the model calls `tool_help` with the names of all the tools it is about to use, in one call. The tool list stays fixed for the whole conversation, so it stays cacheable. This cuts the fixed overhead per model call from about 23k to about 9k tokens. Wrong or missing arguments are caught before the tool runs, and the error includes that tool's full help, so one retry fixes it. `tool_descriptions = "full"` restores the old behaviour.

Most of each request is fixed overhead: about 9k tokens of tool descriptions (23k with `tool_descriptions = "full"`) plus the growing conversation. Plot images are downscaled to `image_max_px` (1600 px) when they're added. `image_detail = "low"` cuts each image to about 85 tokens, at the cost of detail. Raising the deployment's TPM quota in Azure AI Foundry (Deployments → Edit → Tokens per Minute) is the direct fix if you want full speed.

Other settings you may need:

* `api`: `"chat"` (/chat/completions, streamed) or `"responses"` (/responses, not streamed, so no mid-stream disconnects). Both use the same rate pacing and retry a cut-off answer with more output room. For one run, `CALPHAD_LLM_API=responses ./start-agent.sh` overrides the file; compare with `/usage`.
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

Limitations: there's no image input, so plots are referenced by file path, and each step is a whole CLI invocation, which is slower. It works as a stand-in, or for any model that lacks native tool calling.

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
* **Labels:** black text with no background box, so the fill shows through.
* **No legend:** the phases are named by labels inside the figure. `label_regions` is on by default in this style
  (pass `label_regions: false` to leave it out). Isopleths and T–μ diagrams can't have region labels, so each line is
  named on the line instead. With experimental data, the legend lists the data references only, and it sits inside
  the axes.
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
and `"thorough"` validates more points; the code is `calphad_tools/mapping_check.py`, which `show_code` copies, and
isopleths and T–μ diagrams are not checked.

### Phase labels on diagrams

`map_phase_diagram` (binary T–x and ternary isotherms) and `plot_diagram_with_data` take `label_regions: true` (the default in the shaded style).
The phases of each region are then written on the diagram, and on binaries the invariant temperatures too. To label
a diagram you already made, use the `edit_figure` operation `label_phase_regions`. That works after data points were
added, and the labels avoid those points.

* **Names.** Common names are shortened: BCC_A2 → BCC, LIQUID → L, L12_FCC → L1₂, SIGMA → σ, MU_PHASE → μ,
  GAMMA_PRIME → γ′, LAVES_C14 → C14. Formula names are split into the database's elements: FE3CR4 → Fe₃Cr₄,
  M23C6 → M₂₃C₆, COSI → CoSi. A miscibility gap is BCC+BCC. `phase_labels` overrides any name, e.g.
  `{"SIGMA": "σ-phase"}`. The rules are a table in `calphad_tools/phase_names.py`.
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
  * *Ternary:* a field that touches an edge (the phase exists in that binary) is named beyond that edge, like a
    stoichiometric phase (below). The name sits at the middle of its contact with the edge, or of the longer
    contact if it touches two.
  * *Binary:* a field that reaches the bottom of the plot is named below the x axis at its composition there, at
    60°. Example: a C14 with a narrow range.
  * *Binary:* a field that ends higher up is named right below its lowest point, inside the axes. The text is at
    60° with no box; it starts about 2 pt below that point and runs down and away, only where it crosses no line
    or other label.
  * The liquid and the end-member solid solutions are never named below the x axis.
* **Leader lines** are the last resort for single phases. They are used for ternary-only phases (fields that touch
  no edge, and ternary compounds inside the triangle), and for intermediate binary phases with no room below their
  field. The label goes in the nearest free space; inside a ternary that means a free area of the triangle, else
  just outside it. A thin line runs from the label to the field, with a dot at a point compound. These are never
  left out. An end-member phase that finds no place stays omitted, and the result text says so.
* **Stoichiometric phases** (binary line compounds, ternary point or line compounds on an edge) are never named
  inside the plot. Their name goes outside, right at their composition: below the x axis of a binary (text at
  60°), or beyond the nearest edge of a ternary (60° at the bottom edge, ±30° — square to the edge — at the slanted
  edges). A tick label that such a name covers is hidden; the others stay.
* **Invariant temperatures** (binary, `label_invariants`) go right above or right below the invariant line, inside
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
* **Standalone code.** `show_code` reproduces the labels without the toolkit: the script contains copies of the
  three label modules.

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

`show_code` takes any output (image, CSV, or call number from `session_log`) and returns standalone Python using only pycalphad, numpy, pandas, scipy and matplotlib. Paste it into Jupyter as a record of how the result was made.

* It follows the chain back: for an edited figure you get the phase-diagram cell, then the edit cell.
* It also works for outputs from earlier sessions, via the provenance records.
* A step with no standalone equivalent is reproduced through the toolkit and clearly labelled.
* Formats: a copy-paste cell, a `.py`, or `format: "notebook"` for an `.ipynb`.

**Tools with standalone code:**

* **Calculations:** `map_phase_diagram` (binary, ternary isotherm, isopleth), `step_map`, `equilibrium_property_plot`, `phase_property_plot`, `export_equilibrium_data`, `scheil_solidification`.
* **Models, databases and MCMC:** `export_phase_model`, `write_database`, `analyze_mcmc`, `mcmc_to_tdb`.
* **Overviews and ESPEI plots:** `plot_all_binaries`, `plot_all_ternaries`, `plot_diagram_with_data`, `plot_parameter_vs_data`.
* **Plotting:** `make_plot` and `edit_figure` with all 69 plot operations, `compose_images`, `image_tools`, `images_to_pdf`, `figures_to_pptx`, `animate`.
* **Data:** `transform_data`, `fit_data`, `convert_composition`, `calculate_equilibrium`, `run_python`.
* **Digitised data:** the extracted points are embedded as a table, so later steps run without the image analysis.

**Falls back to the toolkit:** T–μ diagrams, wildcard site fractions `Y(...*...)`, extracting a subsystem with `write_database`, animation sweeps and `.mp4`, and `digitize_setup`.

**Checks:** the tests run each generated script in a clean process and compare it with the tool's own output: CSVs to a relative tolerance of 1e-6, and figure artists (lines, collections, texts, limits, labels). Phase diagrams are compared point by point (within 0.02; invariants within 0.5 K), because pycalphad's mapping order varies with Python's hash seed.

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

* **Phase names.** Clear aliases are renamed with their parameters, type definitions and DIS_PART references: `C15_LAVES`/`LAVES_C15`/`C15`, `SIGMA`/`SIGMA_D8B`, `A2_BCC`/`BCC_A2`. Weaker matches are only listed; apply them with `phase_map`. Two phases of B are never merged.
* **Sublattice order.** B's sublattices are matched to A's by size. Equal sizes are decided by shared elements: Fe on the 2-site sublattice of both a Fe₂Mo and a W:Fe 1:2 C14. If that is undecidable, the tool stops and asks for `sublattice_map {'C14': [1, 0]}`. This happens with no shared element (Na:Cl vs K:Br), with overlap pointing both ways, or with a different number of sublattices.
* **Formula unit.** B's site ratios are rescaled to A's when proportional (4:2 → 2:1), with G and L × k. A phase with TC/BMAGN parameters is never rescaled. pycalphad's magnetic term R·T·ln(β+1)·f(τ) is per formula unit and not linear in BMAGN, so no exact rescaling exists.
* **The magnetic exception.** If B's phase has TC/BMAGN parameters and about 1 atom per formula unit (0.75:0.25), and A's phase has no magnetic terms (2:1, 3:1), the rescale goes the other way. A's phase takes B's site ratios, in A's sublattice order, and A's G and L are scaled by 1/k. This is reported as "A rescaled to B (magnetic exception)". The rescaled A (`A0_…`) is used for B2, B3, Q, A1, A2 and the replot decision. B's parameters, including its magnetic ones, are unchanged. In every other magnetic case (A magnetic too, B not about 1 atom/fu, VA on a substitutional sublattice, A order-disorder) the phase is only reordered and reported as "not_rescaled", with the reason.
* **Checks.** The tests check that GM per mole of atoms is unchanged (< 1e-9) after every normalisation. An exact proportional rescale (G/L scaled alike, no magnetic terms) is not counted as a change in the replot decision or the version comparisons; it is named in one line instead.

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

Only consecutive versions are compared: parameters, phase models, and invariant reactions that moved (ΔT, Δx). `force_replot` (`true`, or a list such as `["Fe-Mo"]`) maps subsystems whose change is exact (an exact rescale, or numerically identical parameters) again instead of reusing them. Each forced figure is labelled with its largest invariant shift, e.g. "Fe-Mo: exact rescale, replotted on request – max invariant ΔT 0.00 K, Δx 0.0000". A shift above 0.5 K or 0.001 is flagged in the text, the slide and the summary row as a failed exactness claim. Ternary isotherms compare tie-triangles. The deck has one summary slide, then two slides per step: a differences-only table and the changed diagrams of both versions.

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
* `espei_run_mcmc` starts MCMC as a background job: serial by default, or a dask cluster on this machine with `cores`. It supports restarting from a trace, a subset of symbols, priors and data weights. `trace.npy`, `lnprob.npy` and `mcmc.tdb` land in the run folder, where `analyze_mcmc` and `mcmc_to_tdb` find them. A folder that already holds a trace is never overwritten; the new run goes to `<name>_2`.
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
