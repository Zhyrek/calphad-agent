# Figure operation extension API (calphad_tools/figures.py)

The chat agent draws and edits plots through two tools, `edit_figure` and `make_plot`, plus `plot_ops_help`. Both
tools take a list of **operations** (JSON objects `{"op": NAME, ...}`), executed by `figures.apply_ops(fig, ops,
resolve)`. Extension modules add operations and data features **without editing figures.py**.

## Registering an operation
```python
from calphad_tools.figures import register_op, OpContext, make_norm, main_axes, find_col

@register_op("stackplot", "stackplot {x, ys: [[...], ...] | y_cols: [..], labels, colors, alpha}", group="draw")
def _stackplot(ctx: OpContext) -> None:
    ...
```
- `name`: lower-case op name. Registered ops take precedence over built-ins of the same name, so an extension
  may *extend* a built-in (e.g. `hist` with density/cumulative), as long as the old syntax still works.
- `help`: the one-line syntax shown by `plot_ops_help` and appended to error messages. Keep it compact but
  complete (every key the op understands).
- `needs_axes=False` for figure-level operations (e.g. panel labels, shared colour bars). `ctx.ax` may then be
  None or the axes chosen by the op's `axes` key.
- `group`: one of "draw", "statistics", "fit", "grid & fields", "annotate", "axes & layout", "existing artists",
  "data", "other". It is used in the op list shown in the tool descriptions.
- Pseudo-topics: add `figures.OP_HELP["transform"] = "..."` to document things that are not ops (such as the
  data transform pipeline). `plot_ops_help(["transform"])` returns them.

## OpContext (what an op receives)
- `ctx.fig`, `ctx.ax`: the matplotlib Figure and target Axes.
- `ctx.op`: the operation dict. `ctx.n` is its 1-based index. `ctx.resolve(path)` gives an absolute path.
- `ctx.coords` ("data" | "axes" | "figure") and `ctx.tr`, the matching transform. Pass `transform=ctx.tr` when
  drawing in data coordinates, so the coords key works.
- Data:
  - `ctx.has_source`: True if the op has a `csv` or `data` key.
  - `ctx.frame()`: pandas DataFrame after loaders, filters and transforms.
  - `ctx.colname(key)`: the resolved column for `<key>_col`.
  - `ctx.raw(key, dtype=float)`: values from `<key>_col`, else the inline `op[key]` as a 1-D array. Pass
    `dtype=None` for non-numeric values such as labels.
- Coordinates: `ctx.X(key="x")`, `ctx.Y(key="y")` convert values given "as displayed" into data coordinates.
  For example, a kelvin axis shown in °C: 1300 → 1573.15. Use these for positions. `ctx.dX(key)` and
  `ctx.dY(key)` do the same for differences such as widths and error bars (scale only, no offset). For values
  that are not on the x/y axes (z, colours, weights), use `ctx.raw`.
- Helpers:
  - `ctx.pick(allowed_keys)` filters the op dict to matplotlib style kwargs and validates colours.
  - `ctx.colorbar(mappable)` honours `op["colorbar"]` (true or {label, ticks, extend, orientation, shrink,
    location}) and `colorbar_label`.
  - `make_norm(op["norm"])` builds log, symlog, power, diverging and bounds norms.
  - `ctx.note(text)` adds a line to the tool result.
  - `ctx.new_axes(ax2, "what")` reports a newly created axes index.
- Raise `ValueError` with an actionable message on bad input. The engine adds the op number and the op's
  syntax to the message.

## Data sources and loaders
- Ops read data inline or from a source: `"csv": path` (legacy) or `"data": path | inline records | {col: [..]}`,
  with `"<key>_col"` column names (exact, or an unambiguous prefix or substring via `find_col`) and
  `"where": {col: value|[values]}` for equality filters.
- `figures.add_data_loader(fn)` registers `fn(op, resolve) -> DataFrame | None`. The most recently added loader
  that returns a frame wins, and the built-in CSV loader is the fallback. A loader must implement at least the
  same semantics as the built-in one (`csv`/`data` path, inline records/columns, `where`).
- Column keys that `ctx.cols()` extracts: x y z yerr xerr c s y1 y2 values height width labels weights. Other
  `<key>_col` keys work through `ctx.raw(key)`.

## Hard rules
1. **Figures must stay picklable.** Every saved figure is pickled so it can be edited later. Never attach
   lambdas or local functions to artists, formatters, locators or secondary-axis transforms. Use module-level
   classes or functions: `figures.ScaledFormatter` is the model, and `secondary_xaxis(functions=(f, g))` needs
   picklable callables. Test this with `pickle.dumps(fig)` after the op.
2. Deterministic, offline, no subprocesses or network. Only numpy, scipy, pandas, matplotlib and pillow (all
   installed). scikit-learn 1.9 is also installed.
3. The "units as displayed" convention: positions go through `ctx.X`/`ctx.Y`.
4. Figures may have triangular (pycalphad ternary), polar or 3-D axes. An op that cannot work on a projection
   must raise a clear error rather than draw garbage.
5. Keep help strings short. Their total size is paid in prompt tokens on every model request.

## Tools
A module may also define `register(tb)` to add agent tools (see calphad_tools/agent/tools.py: `@tb.add(name,
description, properties, required)` and a function returning `(text, [image_paths])`). Helpers: `tb.p(path)`,
`tb.outp(name, default)`, `tb.outdir(dir, default)`, `tb.rel(path)`, `tb.j(obj)`. It is already wired: these module
names are imported by the ToolBox plugin loader and by `figures.ensure_extensions()`:
`calphad_tools.figdata`, `calphad_tools.figfit`, `calphad_tools.figops_extra`, `calphad_tools.figcompose`,
`calphad_tools.region_labels` (phase-region label ops).

## Testing
Build a ToolBox the way tests/test_figures.py does (`python_tool="off"`) and call `make_plot` / `edit_figure`
with your ops. Assert on the numbers (read back through `figures.load_figure(png)` and inspect the artists),
look at a few PNGs, and check that pickling still works. Run `.venv/bin/python -m pytest -q -p no:warnings
tests/<your_test_file>.py`.
