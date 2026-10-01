# pycalphad 0.11.2 plotting & data-output inventory

Generated offline from the installed packages by `python -m calphad_tools.pycalphad_docs docs`.
Kinds: **plotting** (makes figures), **data-output** (produces numbers/files/text), **both**, **support** (inputs/infrastructure needed to use the others).

## top-level

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| equilibrium | `pycalphad.equilibrium` | data-output | `(dbf, comps, phases, conditions, output=None, model=None, verbose=False, calc_opts=None, to_xarray=True, parameters=None, solver=None, phase_records=None, **kwa ...` | Global equilibrium over a grid of conditions; returns xarray Dataset (NP, GM, MU, X, Y, Phase + `output` props). |
| calculate | `pycalphad.calculate` | data-output | `(dbf, comps, phases, mode=None, output='GM', fake_points=False, broadcast=True, parameters=None, to_xarray=True, phase_records=None, conditions=None, **kwargs)` | Sample a Model property (output='GM','HM',...) over site-fraction points for phases; returns xarray Dataset (X, Y, Phase, <output>). |
| Workspace | `pycalphad.Workspace` | data-output | `(database, components, phases, conditions, *, models=None, parameters=None, calc_opts=None, solver=None, verbose=False, constituents=None, phase_record_factory= ...` | Reactive equilibrium calculator (db, components, phases, conditions); wks.get(*props) returns arrays of any computable property. |
| binplot | `pycalphad.binplot` | plotting | `(database, components, phases, conditions, return_strategy=False, plot_kwargs=None, **map_kwargs)` | Binary isobaric phase diagram via BinaryStrategy + plot_binary; return_strategy=True also returns the strategy (data). |
| ternplot | `pycalphad.ternplot` | plotting | `(dbf, comps, phases, conds, x=None, y=None, return_strategy=False, map_kwargs=None, **plot_kwargs)` | Ternary isothermal section via TernaryStrategy + plot_ternary; return_strategy=True also returns the strategy (data). |
| eqplot | `pycalphad.eqplot` | plotting | `(eq, ax=None, x=None, y=None, z=None, tielines=True, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), legend_generator=<function phase_legend>, **kw ...` | Plot an equilibrium() result (binary T-x or ternary isothermal) with tielines / tie-triangles. |
| as_property | `pycalphad.as_property` | support | `(inp: Union[str, symengine.lib.symengine_wrapper.Basic, pycalphad.property_framework.types.ComputableProperty]) -> pycalphad.property_framework.types.Computable ...` | Parse a property string ('GM', 'NP(FCC_A1)', 'X(LIQUID,CR)', 'HM.T', ...) into a ComputableProperty. |
| Database | `pycalphad.Database` | data-output | `(fname_or_tdb_string=None)` | Thermodynamic database: read TDB/DAT, write TDB (to_file/to_string), search parameters. |
| Model | `pycalphad.Model` | data-output | `(dbe, comps, phase_name, parameters=None)` | Symbolic CALPHAD model of one phase: GM, HM, SM, CPM, mixing props, DOO, TC, BMAG, VM, contributions. |
| ReferenceState (Model) | `pycalphad.ReferenceState` | support | `(species, reference_phase, fixed_statevars=None)` | Pure-element reference state (species, phase, fixed_statevars) for Model.shift_reference_state. |
| variables (v) | `pycalphad.variables` | support | `(module)` | State variables / conditions / properties: T, P, N, X, W, MU, NP, Y, Species, Component. |

## plot

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| eqplot | `pycalphad.plot.eqplot.eqplot` | plotting | `(eq, ax=None, x=None, y=None, z=None, tielines=True, tieline_color=(0, 1, 0, 1), tie_triangle_color=(1, 0, 0, 1), legend_generator=<function phase_legend>, **kw ...` | Scatter-plot phase boundaries from an equilibrium Dataset (x/y/z StateVariables auto-detected). |
| phase_legend | `pycalphad.plot.utils.phase_legend` | plotting | `(phases)` | Build matplotlib legend handles and a phase->color dict for a list of phases. |
| TriangularAxes | `pycalphad.plot.triangular.TriangularAxes` | plotting | `(*args, **kwargs)` | matplotlib projection 'triangular' for Gibbs-triangle ternary plots (fig.add_subplot(projection='triangular')). |
| ternplot (legacy) | `pycalphad.plot.ternary.ternplot` | plotting | `(dbf, comps, phases, conds, x=None, y=None, eq_kwargs=None, **plot_kwargs)` | Legacy ternary isotherm: equilibrium on a composition grid then eqplot. |
| binplot (legacy) | `pycalphad.plot.binary.plot.binplot` | plotting | `(database, components, phases, conditions, plot_kwargs=None, **map_kwargs)` | Legacy binary mapper (map_binary + plot_boundaries). |
| plot_boundaries | `pycalphad.plot.binary.plot.plot_boundaries` | plotting | `(zpf_boundary_sets, tielines=True, tieline_color=(0, 1, 0, 1), scatter=True, legend_generator=<function phase_legend>, ax=None, gridlines=False)` | Plot ZPFBoundarySets (legacy binary mapper) as lines/scatter with tielines. |
| map_binary | `pycalphad.plot.binary.map.map_binary` | data-output | `(dbf, comps, phases, conds, eq_kwargs=None, calc_kwargs=None, boundary_sets=None, verbose=False, summary=False)` | Legacy binary T-X mapping; returns ZPFBoundarySets (phase-boundary data). |
| ZPFBoundarySets | `pycalphad.plot.binary.zpf_boundary_sets.ZPFBoundarySets` | both | `(comps, indep_composition_condition)` | Legacy container of two-phase regions; get_line_plot_boundaries/get_scatter_plot_boundaries/get_phases. |
| ZPFBoundarySets.get_line_plot_boundaries | `pycalphad.plot.binary.zpf_boundary_sets.ZPFBoundarySets.get_line_plot_boundaries` | both | `(self, close_miscibility_gaps=0.05, tieline_color=(0, 1, 0, 1), legend_generator=<function phase_legend>)` | Line collections + legend for each two-phase region boundary (data for custom plots). |
| ZPFBoundarySets.get_scatter_plot_boundaries | `pycalphad.plot.binary.zpf_boundary_sets.ZPFBoundarySets.get_scatter_plot_boundaries` | both | `(self, tieline_color=(0, 1, 0, 1), legend_generator=<function phase_legend>)` | Scatter x/y/colors + tieline collection for boundaries. |
| get_compsets | `pycalphad.plot.binary.compsets.get_compsets` | data-output | `(eq_dataset, indep_comp=None, indep_comp_index=None)` | Extract a pair of binary composition sets (phase, T, x, site fractions) from an equilibrium Dataset point. |
| find_two_phase_region_compsets | `pycalphad.plot.binary.compsets.find_two_phase_region_compsets` | data-output | `(hull, temperature, indep_comp, indep_comp_idx, misc_gap_tol=0.1, minimum_composition=None)` | Two-phase composition sets from a 1D convex hull at fixed T, P. |

## mapping

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| binplot (mapping) | `pycalphad.mapping.compat_api.binplot` | both | `(database, components, phases, conditions, return_strategy=False, plot_kwargs=None, **map_kwargs)` | Map + plot binary diagram; return_strategy=True returns (ax, BinaryStrategy) for data export. |
| ternplot (mapping) | `pycalphad.mapping.compat_api.ternplot` | both | `(dbf, comps, phases, conds, x=None, y=None, return_strategy=False, map_kwargs=None, **plot_kwargs)` | Map + plot ternary isotherm; return_strategy=True returns (ax, TernaryStrategy). |
| MapStrategy | `pycalphad.mapping.strategy.strategy_base.MapStrategy` | data-output | `(dbf: pycalphad.io.database.Database, components: list[str], phases: list[str], conditions: dict[pycalphad.variables.StateVariable, typing.Union[float, tuple[fl ...` | Base mapping strategy: do_map(), zpf_lines, node_queue, get_all_phases(); kwargs GLOBAL_MIN_PDENS etc. |
| MapStrategy.do_map | `pycalphad.mapping.strategy.strategy_base.MapStrategy.do_map` | support | `(self, max_iter: int = -1)` | Run the mapping until all nodes are exhausted (max_iter). |
| MapStrategy.get_all_phases | `pycalphad.mapping.strategy.strategy_base.MapStrategy.get_all_phases` | data-output | `(self)` | All phases found along ZPF lines (miscibility gaps as PHASE#2). |
| MapStrategy.add_nodes_from_conditions | `pycalphad.mapping.strategy.strategy_base.MapStrategy.add_nodes_from_conditions` | support | `(self, conditions: dict[pycalphad.variables.StateVariable, float], direction: pycalphad.mapping.primitives.Direction = None, force_add: bool = True)` | Add a starting point computed from explicit conditions (optionally direction). |
| BinaryStrategy | `pycalphad.mapping.BinaryStrategy` | data-output | `(dbf: pycalphad.io.database.Database, components: list[str], phases: list[str], conditions: dict[pycalphad.variables.StateVariable, typing.Union[float, tuple[fl ...` | Binary mapping (1 composition + 1 potential axis); data via get_tieline_data/get_invariant_data. |
| BinaryStrategy.get_tieline_data | `pycalphad.mapping.strategy.binary_strategy.BinaryStrategy.get_tieline_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x: bool = False, global_y: bool = False) -> list[pycalphad.mapping.str ...` | List[StrategyData]: per ZPF line, phase-labelled x/y arrays (tieline end points) for any x,y variables. |
| BinaryStrategy.get_invariant_data | `pycalphad.mapping.strategy.binary_strategy.BinaryStrategy.get_invariant_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x: bool = False, global_y: bool = False) -> list[pycalphad.mapping.str ...` | List[StrategyData]: three-phase invariant nodes (phase, x, y). |
| TernaryStrategy | `pycalphad.mapping.TernaryStrategy` | data-output | `(dbf: pycalphad.io.database.Database, components: list[str], phases: list[str], conditions: dict[pycalphad.variables.StateVariable, typing.Union[float, tuple[fl ...` | Ternary isothermal mapping (2 composition axes); get_tieline_data/get_invariant_data (tie-triangles). |
| TernaryStrategy.get_tieline_data | `pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy.get_tieline_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x: bool = False, global_y: bool = False) -> list[pycalphad.mapping.str ...` | List[StrategyData] of two-phase tieline end points. |
| TernaryStrategy.get_invariant_data | `pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy.get_invariant_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x: bool = False, global_y: bool = False) -> list[pycalphad.mapping.str ...` | List[StrategyData] of three-phase tie-triangles. |
| IsoplethStrategy | `pycalphad.mapping.IsoplethStrategy` | data-output | `(dbf: pycalphad.io.database.Database, components: list[str], phases: list[str], conditions: dict[pycalphad.variables.StateVariable, typing.Union[float, tuple[fl ...` | Isopleth (vertical section) mapping; get_zpf_data/get_invariant_data. |
| IsoplethStrategy.get_zpf_data | `pycalphad.mapping.strategy.isopleth_strategy.IsoplethStrategy.get_zpf_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable) -> pycalphad.mapping.strategy.strategy_data.StrategyData` | StrategyData of zero-phase-fraction boundary lines (phase, x, y) for the isopleth. |
| IsoplethStrategy.get_invariant_data | `pycalphad.mapping.strategy.isopleth_strategy.IsoplethStrategy.get_invariant_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable) -> list[pycalphad.mapping.strategy.strategy_data.StrategyData]` | List[StrategyData] of invariant lines/nodes in the isopleth. |
| StepStrategy | `pycalphad.mapping.StepStrategy` | data-output | `(dbf: pycalphad.io.database.Database, components: list[str], phases: list[str], conditions: dict[pycalphad.variables.StateVariable, typing.Union[float, tuple[fl ...` | One-axis step calculation (e.g. T or X); get_data(x, y) for any property vs axis. |
| StepStrategy.get_data | `pycalphad.mapping.strategy.step_strategy.StepStrategy.get_data` | data-output | `(self, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x: bool = False, global_y: bool = False, set_nan_to_zero: bool = False ...` | StrategyData of y vs x per stable phase (or 'SYSTEM' for global props); global_x/global_y, set_nan_to_zero. |
| add_starting_points_from_step | `pycalphad.mapping.strategy.binary_strategy.BinaryStrategy.add_starting_points_from_step` | support | `(self, step: pycalphad.mapping.strategy.step_strategy.StepStrategy)` | Seed a Binary/Ternary/Isopleth map from a StepStrategy result. |
| generate_automatic_starting_points | `pycalphad.mapping.strategy.strategy_base.MapStrategy.generate_automatic_starting_points` | support | `(self)` | Automatic starting-point search along axis limits. |
| StrategyData | `pycalphad.mapping.strategy.strategy_data.StrategyData` | data-output | `(data: list[pycalphad.mapping.strategy.strategy_data.SinglePhaseData]) -> None` | Container: .data (list of SinglePhaseData), .phases, .x, .y. |
| SinglePhaseData | `pycalphad.mapping.strategy.strategy_data.SinglePhaseData` | data-output | `(phase: str, x: float \| list[float], y: float \| list[float]) -> None` | phase name + x + y arrays for one phase / line. |
| PhaseRegionData | `pycalphad.mapping.strategy.strategy_data.PhaseRegionData` | data-output | `(data: list[pycalphad.mapping.strategy.strategy_data.SinglePhaseData]) -> None` | StrategyData subclass for one phase region (tieline / invariant). |
| get_tieline_data_from_tieline_strategy | `pycalphad.mapping.strategy.strategy_data.get_tieline_data_from_tieline_strategy` | data-output | `(strategy, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x=False, global_y=False) -> list[pycalphad.mapping.strategy.strate ...` | Shared implementation of get_tieline_data for Binary/Ternary strategies. |
| get_invariant_data_from_tieline_strategy | `pycalphad.mapping.strategy.strategy_data.get_invariant_data_from_tieline_strategy` | data-output | `(strategy, x: pycalphad.variables.StateVariable, y: pycalphad.variables.StateVariable, global_x=False, global_y=False) -> list[pycalphad.mapping.strategy.strate ...` | Shared implementation of get_invariant_data for Binary/Ternary strategies. |
| plot_binary | `pycalphad.mapping.plot_binary` | plotting | `(strategy: pycalphad.mapping.strategy.binary_strategy.BinaryStrategy, x: pycalphad.variables.StateVariable = None, y: pycalphad.variables.StateVariable = None, ...` | Plot a finished BinaryStrategy (x/y vars, tielines every n, label_nodes, colors). |
| plot_ternary | `pycalphad.mapping.plot_ternary` | plotting | `(strategy: pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy, x: pycalphad.variables.StateVariable = None, y: pycalphad.variables.StateVariable = None ...` | Plot a finished TernaryStrategy on triangular axes. |
| plot_isopleth | `pycalphad.mapping.plot_isopleth` | plotting | `(strategy: pycalphad.mapping.strategy.isopleth_strategy.IsoplethStrategy, x: pycalphad.variables.StateVariable = None, y: pycalphad.variables.StateVariable = No ...` | Plot a finished IsoplethStrategy (ZPF lines + invariants). |
| plot_step | `pycalphad.mapping.plot_step` | plotting | `(strategy: pycalphad.mapping.strategy.step_strategy.StepStrategy, x: pycalphad.variables.StateVariable = None, y: pycalphad.variables.StateVariable = None, ax=N ...` | Plot a finished StepStrategy (any y property vs step axis, per phase). |
| plot_tielines | `pycalphad.mapping.plotting.plot_tielines` | plotting | `(ax, strategy: Union[pycalphad.mapping.strategy.binary_strategy.BinaryStrategy, pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy], x: pycalphad.varia ...` | Draw tielines of a Binary/Ternary strategy on an axis. |
| plot_invariants | `pycalphad.mapping.plotting.plot_invariants` | plotting | `(ax, strategy: Union[pycalphad.mapping.strategy.binary_strategy.BinaryStrategy, pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy], x: pycalphad.varia ...` | Draw invariant nodes / tie-triangles of a Binary/Ternary strategy on an axis. |
| get_label | `pycalphad.mapping.plotting.get_label` | plotting | `(var: pycalphad.variables.StateVariable)` | Axis label text for a StateVariable. |
| ZPFLine | `pycalphad.mapping.primitives.ZPFLine` | data-output | `(fixed_phases: List[str], free_phases: List[str], points: Optional[List[pycalphad.mapping.primitives.Point]] = None, status: pycalphad.mapping.primitives.ZPFSta ...` | One zero-phase-fraction line: fixed/free phases, points, status; get_var_list(var). |
| ZPFLine.get_var_list | `pycalphad.mapping.primitives.ZPFLine.get_var_list` | data-output | `(self, var: pycalphad.variables.StateVariable)` | Values of a variable along a ZPF line. |
| Point | `pycalphad.mapping.primitives.Point` | data-output | `(global_conditions: Mapping[pycalphad.variables.StateVariable, float], chemical_potentials: List[float], _fixed_composition_sets: List[pycalphad.core.compositio ...` | Map point: global_conditions, chemical_potentials, stable phases/composition sets; get_property(var), get_local_property. |
| Node | `pycalphad.mapping.primitives.Node` | data-output | `(global_conditions: Mapping[pycalphad.variables.StateVariable, float], chemical_potentials: List[float], _fixed_composition_sets: List[pycalphad.core.compositio ...` | Special Point where phases change (invariant / start of new ZPF lines). |
| point_from_equilibrium | `pycalphad.mapping.starting_points.point_from_equilibrium` | data-output | `(dbf: pycalphad.io.database.Database, components: list[str], phases: list[str], conditions: dict[pycalphad.variables.StateVariable, float], **eq_kwargs)` | Compute a mapping Point from an equilibrium at given conditions. |
| degrees_of_freedom | `pycalphad.mapping.utils.degrees_of_freedom` | data-output | `(point: Union[pycalphad.mapping.primitives.Point, pycalphad.mapping.primitives.Node], components: list[str], num_potential_conditions: int)` | Gibbs-phase-rule degrees of freedom of a Point/Node. |

## workspace

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| Workspace.get | `pycalphad.core.workspace.Workspace.get` | data-output | `(self, *args: Tuple[pycalphad.property_framework.types.ComputableProperty])` | Compute one or more properties over the condition grid; arrays shaped like the conditions (display units). |
| Workspace.get_dict | `pycalphad.core.workspace.Workspace.get_dict` | data-output | `(self, *args: Tuple[pycalphad.property_framework.types.ComputableProperty])` | Like get() but returns {property: array} (wildcards like 'NP(*)' expand to one key per phase). |
| Workspace.eq | `pycalphad.core.workspace.Workspace.eq` | data-output | `(attribute: LightDataset \| None)` | Underlying LightDataset equilibrium result (NP, GM, MU, X, Y, Phase); .get_dataset() -> xarray. |
| Workspace.enumerate_composition_sets | `pycalphad.core.workspace.Workspace.enumerate_composition_sets` | data-output | `(self)` | Yield (index, [CompositionSet]) for every grid point (stable phases, site fractions, NP). |
| Workspace.get_composition_sets | `pycalphad.core.workspace.Workspace.get_composition_sets` | data-output | `(self)` | Composition sets of a point calculation (ndim == 0). |
| Workspace.copy | `pycalphad.core.workspace.Workspace.copy` | support | `(self)` | Copy a workspace (then change conditions/phases). |
| Workspace.ndim / condition_axis_order | `pycalphad.core.workspace.Workspace.condition_axis_order` | support | `(property)` | Number of varying conditions and their axis order in returned arrays. |
| Conditions | `pycalphad.core.conditions.Conditions` | support | `(wks: Optional[ForwardRef('Workspace')])` | Workspace conditions mapping (keys, str_keys, items(units=...)). |

## property framework

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| ModelComputedProperty | `pycalphad.property_framework.ModelComputedProperty` | data-output | `(model_attr_name: str, phase_name: Optional[str] = None)` | Any Model attribute as a property: 'GM', 'HM(FCC_A1)', 'DOO(BCC_B2)', 'TC', 'VM'...; ['units'] changes display units. |
| JanssonDerivative | `pycalphad.property_framework.JanssonDerivative` | data-output | `(numerator: pycalphad.property_framework.types.DifferentiableComputableProperty, denominator: pycalphad.property_framework.types.ConditionableComputableProperty ...` | Derivative of an equilibrium property w.r.t. a condition: 'HM.T' (=Cp), 'X(LIQUID,CR).T', 'MU(CR).X(CR)'. |
| LinearCombination | `pycalphad.property_framework.computed_property.LinearCombination` | support | `(expr: symengine.lib.symengine_wrapper.Basic)` | Symbolic linear combination of properties usable as condition (e.g. v.X('A') - v.X('B')). |
| IsolatedPhase | `pycalphad.property_framework.IsolatedPhase` | data-output | `(phase: Union[pycalphad.core.composition_set.CompositionSet, str], wks: Optional[ForwardRef('Workspace')])` | Meta-property: property of one phase minimized alone under the same conditions (metastable single-phase, e.g. GM curves). |
| DormantPhase | `pycalphad.property_framework.DormantPhase` | data-output | `(phase: Union[pycalphad.core.composition_set.CompositionSet, str], wks: Optional[ForwardRef('Workspace')])` | Meta-property: phase minimized at the equilibrium potentials without becoming stable; .driving_force. |
| DormantPhase.driving_force | `pycalphad.property_framework.metaproperties.DormantPhase.driving_force` | data-output | `(property)` | Driving force (J/mol) for precipitation of a dormant phase; >0 means it wants to form. |
| DrivingForce | `pycalphad.property_framework.DrivingForce` | data-output | `(phase_name)` | Driving force property class used by DormantPhase(...).driving_force. |
| ReferenceState (property) | `pycalphad.property_framework.ReferenceState` | data-output | `(reference_conditions: List[Tuple[str, Dict]], wks: 'Workspace')` | Meta-property re-referencing a property to phases at given conditions: ReferenceState([('FCC_A1',{v.X('CR'):0}),...], wks)('HM'). |
| T0 | `pycalphad.property_framework.T0` | data-output | `(phase_one: Union[pycalphad.core.composition_set.CompositionSet, str], phase_two: Union[pycalphad.core.composition_set.CompositionSet, str], wks: Optional[Forwa ...` | Property: T0 temperature where GM(phase1) == GM(phase2) at the current composition. |
| as_quantity | `pycalphad.property_framework.as_quantity` | data-output | `(prop: 'ComputableProperty', qt: ArrayLike)` | Wrap a property value as a pint Quantity in the property's display units. |
| to_display_units | `pycalphad.property_framework.units.to_display_units` | data-output | `(value: ArrayLike, comp_sets: Sequence[pycalphad.core.composition_set.CompositionSet], prop: 'ComputableProperty') -> ArrayLike` | Convert implementation units -> display units (J/mol -> J/g etc. using phase molar masses). |
| DimensionalityError | `pycalphad.property_framework.DimensionalityError` | support | `(units1: 'ty.Any', units2: 'ty.Any', dim1: "'str \| UnitsContainerT'" = '', dim2: "'str \| UnitsContainerT'" = '', extra_msg: 'str' = '') -> 'None'` | pint error raised by an impossible unit conversion. |
| ComputableProperty | `pycalphad.property_framework.ComputableProperty` | support | `(*args, **kwargs)` | Protocol any property must satisfy (compute_property, shape, units, display_name). |
| ConditionableComputableProperty | `pycalphad.property_framework.ConditionableComputableProperty` | support | `(*args, **kwargs)` | Protocol: may appear in the denominator of a Jansson derivative (T, P, X, MU, N ...). |
| DifferentiableComputableProperty | `pycalphad.property_framework.DifferentiableComputableProperty` | support | `(*args, **kwargs)` | Protocol: may appear in the numerator of a Jansson derivative. |

## variables

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| T | `pycalphad.variables.T` | support | `(TemperatureType instance; v.T['degC'] for display units)` | Temperature (K). v.T['degC'] gives display in Celsius. |
| P | `pycalphad.variables.P` | support | `(PressureType instance)` | Pressure (Pa); usual condition v.P: 101325. |
| N | `pycalphad.variables.N` | support | `(SystemMolesType instance)` | Total system moles (usually v.N: 1). |
| X | `pycalphad.variables.X` | data-output | `(*args)` | Mole fraction: v.X('CR') overall or v.X('FCC_A1','CR') in a phase; condition and property. |
| W | `pycalphad.variables.W` | data-output | `(*args)` | Mass fraction: v.W('CR') / v.W(phase, 'CR'); condition and property. |
| MU | `pycalphad.variables.MU` | data-output | `(species)` | Chemical potential v.MU('CR') (J/mol); property and condition. |
| NP | `pycalphad.variables.NP` | data-output | `(phase_name)` | Phase fraction v.NP('FCC_A1') (moles of atoms); 'NP(*)' for all phases. |
| Y (site fraction) | `pycalphad.variables.Y` | data-output | `(phase_name, subl_index, species)` | Site fraction v.Y('BCC_B2', sublattice_index, 'AL'); string 'Y(BCC_B2,0,AL)'. |
| Species | `pycalphad.variables.Species` | support | `(name, constituents=None, charge=0)` | Chemical species (name, constituents, charge); .weight. |
| Component | `pycalphad.variables.Component` | support | `(name, constituents=None)` | Component (pure element or linear combination). |
| StateVariable | `pycalphad.variables.StateVariable` | support | `(*args, **kwargs)` | Base class of all variables; prop['units'] sets display units; display_name. |
| get_mass_fractions | `pycalphad.variables.get_mass_fractions` | data-output | `(mole_fractions, dependent_species, pure_element_mass_dict)` | Convert mole fractions -> mass fractions for a point composition. |
| get_mole_fractions | `pycalphad.variables.get_mole_fractions` | data-output | `(mass_fractions, dependent_species, pure_element_mass_dict)` | Convert mass fractions -> mole fractions (e.g. wt% alloy spec -> conditions). |
| unpack_components | `pycalphad.variables.unpack_components` | support | `(components: 'Sequence[Component \| Species \| str]', dbf: Optional[ForwardRef('Database')] = None)` | Normalize component list (strings/Species) to Components. |

## model

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| Model.GM | `pycalphad.Model.GM` | data-output | `(property)` | Molar Gibbs energy per mole of atoms (J/mol) (alias energy). |
| Model.G | `pycalphad.Model.G` | data-output | `(property)` | Gibbs energy per formula unit (J) (alias formulaenergy). |
| Model.HM | `pycalphad.Model.HM` | data-output | `(property)` | Molar enthalpy (alias enthalpy). |
| Model.H | `pycalphad.Model.H` | data-output | `(property)` | Enthalpy per formula unit (alias formulaenthalpy). |
| Model.SM | `pycalphad.Model.SM` | data-output | `(property)` | Molar entropy (alias entropy). |
| Model.CPM | `pycalphad.Model.CPM` | data-output | `(property)` | Molar heat capacity of the phase (alias heat_capacity); apparent equilibrium Cp incl. transformations = 'HM.T'. |
| Model.GM_MIX | `pycalphad.Model.GM_MIX` | data-output | `(property)` | Gibbs energy of mixing relative to endmembers (alias mixing_energy). |
| Model.HM_MIX | `pycalphad.Model.HM_MIX` | data-output | `(property)` | Enthalpy of mixing (alias mixing_enthalpy). |
| Model.SM_MIX | `pycalphad.Model.SM_MIX` | data-output | `(property)` | Entropy of mixing (alias mixing_entropy). |
| Model.CPM_MIX | `pycalphad.Model.CPM_MIX` | data-output | `(property)` | Heat capacity of mixing (alias mixing_heat_capacity). |
| Model.DOO | `pycalphad.Model.degree_of_ordering` | data-output | `(property)` | Degree of ordering (0 disordered .. 1), property 'DOO(PHASE)'. |
| Model.TC / BMAG / NT | `pycalphad.Model.TC` | data-output | `(instance attribute, symbolic; also BMAG, NT)` | Curie temperature, Bohr magneton number (BMAG), Neel temperature of the phase (symbolic). |
| Model.VM | `pycalphad.Model.volume_energy` | data-output | `(self, dbe)` | Molar volume VM (and V0, VA, VC, VK) from volume parameters. |
| Model.models | `pycalphad.Model.build_phase` | data-output | `(self, dbe)` | Dict of energy contributions: ref, idmix, xsmix, mag, 2st, ein, vol, ord (each a symbolic expression). |
| Model.ast | `pycalphad.Model.ast` | data-output | `(property)` | Full symbolic Gibbs-energy expression (SymEngine). |
| Model.variables | `pycalphad.Model.variables` | data-output | `(property)` | State variables + site fractions in the model. |
| Model.moles | `pycalphad.Model.moles` | data-output | `(self, species, per_formula_unit=False)` | Symbolic moles of a species/element in the phase. |
| Model.endmember_reference_model | `pycalphad.Model.endmember_reference_model` | data-output | `(property)` | Model with only endmember contributions (reference for *_MIX properties). |
| Model.shift_reference_state | `pycalphad.Model.shift_reference_state` | data-output | `(self, reference_states, dbe, contrib_mods=None, output=('GM', 'HM', 'SM', 'CPM'), fmt_str='{}R')` | Adds GMR/HMR/SMR/CPMR attributes: properties relative to given pure-element ReferenceStates. |
| Model energy contributions | `pycalphad.Model.excess_mixing_energy` | data-output | `(self, dbe)` | reference_energy, ideal_mixing_energy, excess_mixing_energy, magnetic_energy, atomic_ordering_energy, twostate_energy, einstein_energy, volume_energy (symbolic pieces). |
| ModelMQMQA | `pycalphad.models.model_mqmqa.ModelMQMQA` | data-output | `(dbe, comps, phase_name, parameters=None)` | Modified quasichemical (MQMQA) model used for ChemSage DAT SUBQ phases. |

## database

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| Database.from_file | `pycalphad.Database.from_file` | support | `(fname, fmt=None)` | Read database (fmt 'tdb' or 'dat', autodetected). |
| Database.from_string | `pycalphad.Database.from_string` | support | `(cls, data, **kwargs)` | Read database from a string (fmt=...). |
| Database.to_file | `pycalphad.Database.to_file` | data-output | `(self, fname, fmt=None, if_exists='raise', **write_kwargs)` | Write database (only 'tdb' writable); if_exists raise/rename/overwrite; write kwargs groupby, if_incompatible. |
| Database.to_string | `pycalphad.Database.to_string` | data-output | `(self, **kwargs)` | Database as TDB text (fmt='tdb'). |
| Database.search | `pycalphad.Database.search` | data-output | `(self, query)` | Query parameters (tinydb where('phase_name')==...) -> list of parameter dicts. |
| Database.register_format | `pycalphad.Database.register_format` | support | `(fmt, read=None, write=None)` | Register read/write functions for a format. |
| Database.add_parameter / add_phase | `pycalphad.Database.add_parameter` | support | `(self, param_type, phase_name, constituent_array, param_order, param, ref=None, diffusing_species=None, force_insert=True, **kwargs)` | Build/modify databases programmatically. |
| Database attributes | `pycalphad.io.database.Database` | data-output | `(fname_or_tdb_string=None)` | elements, species, phases (Phase: sublattices, constituents, model_hints), symbols (FUNCTIONs), refstates, _parameters. |
| Phase | `pycalphad.io.database.Phase` | data-output | `()` | Phase record: name, constituents, sublattices, model_hints. |
| write_tdb | `pycalphad.io.tdb.write_tdb` | data-output | `(dbf, fd, groupby='subsystem', if_incompatible='warn')` | TDB writer: groupby 'subsystem'/'phase'; if_incompatible 'warn'/'raise'/'ignore'/'fix' (name mangling, line length). |
| read_tdb | `pycalphad.io.tdb.read_tdb` | support | `(dbf, fd)` | TDB parser used by Database.from_file. |
| TCPrinter | `pycalphad.io.tdb.TCPrinter` | data-output | `(if_incompatible='warn')` | Print SymEngine expressions in Thermo-Calc syntax. |
| reflow_text | `pycalphad.io.tdb.reflow_text` | data-output | `(text, linewidth=80)` | Wrap TDB text lines to <= linewidth. |
| read_cs_dat | `pycalphad.io.cs_dat.read_cs_dat` | support | `(dbf: pycalphad.io.database.Database, fd)` | ChemSage/FactSage DAT reader (read-only format). |
| format_registry | `pycalphad.io.database.format_registry` | support | `(dict: fmt -> DatabaseFormat(read, write))` | Registered formats: tdb (read+write), dat (read only). |

## results

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| equilibrium Dataset | `xarray.Dataset` | data-output | `(data_vars: 'DataVars \| None' = None, coords: 'Mapping[Any, Any] \| None' = None, attrs: 'Mapping[Any, Any] \| None' = None) -> 'None'` | equilibrium()/calculate() return xarray.Dataset; dims = condition axes (N,P,T,X_..), vertex, component, internal_dof. |
| Dataset.to_netcdf | `xarray.Dataset.to_netcdf` | data-output | `(self, path: 'str \| PathLike \| io.IOBase \| None' = None, mode: 'NetcdfWriteModes' = 'w', format: 'T_NetcdfTypes \| None' = None, group: 'str \| None' = None, engi ...` | Save a full equilibrium/calculate result to NetCDF. |
| Dataset.to_dataframe | `xarray.Dataset.to_dataframe` | data-output | `(self, dim_order: 'Sequence[Hashable] \| None' = None) -> 'pd.DataFrame'` | Flatten a result to a pandas DataFrame (then .to_csv). |
| Dataset.sel / squeeze / where | `xarray.Dataset.sel` | data-output | `(self, indexers: 'Mapping[Any, Any] \| None' = None, method: 'str \| None' = None, tolerance: 'int \| float \| Iterable[int \| float] \| None' = None, drop: 'bool' = ...` | Slice results, e.g. eq.NP.sel(T=1000).squeeze(). |
| LightDataset.get_dataset | `pycalphad.core.light_dataset.LightDataset.get_dataset` | data-output | `(self)` | Convert equilibrium(to_xarray=False) / Workspace.eq result to an xarray Dataset. |

## utils

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| filter_phases | `pycalphad.core.utils.filter_phases` | support | `(dbf, comps, candidate_phases=None)` | Phases valid for the given components. |
| instantiate_models | `pycalphad.core.utils.instantiate_models` | support | `(dbf, comps, phases, model=None, parameters=None, symbols_only=True)` | Build {phase: Model} dict (custom Model classes). |
| get_pure_elements | `pycalphad.core.utils.get_pure_elements` | support | `(dbf, comps)` | Pure elements of a component list. |
| unpack_species | `pycalphad.core.utils.unpack_species` | support | `(dbf, comps)` | Component names -> Species set. |
| generate_dof | `pycalphad.core.utils.generate_dof` | data-output | `(phase, active_comps)` | Site-fraction variable list of a phase for given components. |
| endmember_matrix | `pycalphad.core.utils.endmember_matrix` | data-output | `(dof, vacancy_indices=None)` | Endmember site-fraction matrix (for sampling/plots). |
| point_sample | `pycalphad.core.utils.point_sample` | support | `(comp_count, pdof=10)` | Random site-fraction sampling used by calculate. |
| PhaseRecordFactory | `pycalphad.codegen.phase_record_factory.PhaseRecordFactory` | support | `(dbf, comps, state_variables, models, parameters=None)` | Compiled phase records (fast property evaluation) used by Workspace/mapping. |

## related packages

| name | import path | kind | signature | purpose |
|---|---|---|---|---|
| simulate_scheil_solidification | `scheil.simulate_scheil_solidification` | data-output | `(dbf, comps, phases, composition, start_temperature, step_temperature=1.0, liquid_phase_name='LIQUID', eq_kwargs=None, stop=0.0001, verbose=False, adaptive=True ...` | Scheil-Gulliver solidification path (scheil 0.3): returns SolidificationResult. |
| simulate_equilibrium_solidification | `scheil.simulate_equilibrium_solidification` | data-output | `(dbf, comps, phases, composition, start_temperature, step_temperature=1.0, liquid_phase_name='LIQUID', adaptive=True, eq_kwargs=None, binary_search_tol=0.1, ver ...` | Equilibrium (lever-rule) solidification path: returns SolidificationResult. |
| SolidificationResult | `scheil.SolidificationResult` | data-output | `(phase_compositions, fraction_solid, temperatures, phase_amounts, converged, method, output: dict[str, ArrayLike] \| None = None, validate_invariants=True)` | fraction_solid, temperatures, phase_amounts, phase_compositions; to_dict/from_dict/to_dataframe. |
| espei dataplot | `espei.plot.dataplot` | plotting | `(comps, phases, conds, datasets, tielines=True, ax=None, legend_generator=<function phase_legend>, plot_kwargs=None, tieline_plot_kwargs=None) -> matplotlib.axe ...` | Overlay ESPEI ZPF datasets (experimental phase boundaries) on a phase-diagram axis. |
| espei plot_endmember | `espei.plot.plot_endmember` | plotting | `(dbf, comps, phase_name, configuration, output, datasets=None, symmetry=None, x='T', ax=None, plot_kwargs=None, dataplot_kwargs=None) -> matplotlib.axes._axes.A ...` | Plot an endmember property vs T with thermochemical datasets. |
| espei plot_interaction | `espei.plot.plot_interaction` | plotting | `(dbf, comps, phase_name, configuration, output, datasets=None, symmetry=None, ax=None, plot_kwargs=None, dataplot_kwargs=None) -> matplotlib.axes._axes.Axes` | Plot an interaction (mixing) property vs composition with datasets. |

## Usage notes (offline guides)

### conditions

```
Conditions (dict) for equilibrium / Workspace / strategies:
  {v.T: 1273.15, v.P: 101325, v.N: 1, v.X('CR'): 0.3}          point
  {v.T: (300, 2000, 10), ...}                                  range (start, stop, step) -> an axis
Temperatures are always in KELVIN, pressure in Pa, compositions as mole fractions (independent ones
only: n_elements-1 X conditions; the balance element is implicit). Mass fractions: v.W('CR'): 0.2.
Components must include 'VA' when phases have vacancies: ['CR', 'NI', 'VA'].
Phases: list of names from the database (e.g. sorted(db.phases)); pycalphad drops invalid ones.
```

### property strings

```
Workspace.get / as_property strings (also usable as StepStrategy.get_data variables):
  'T' 'P' 'N'                 state variables
  'GM' 'HM' 'SM' 'CPM'        system (equilibrium) molar properties; any Model attribute works
  'GM(FCC_A1)'                property of one phase (Model attribute with phase argument)
  'NP(FCC_A1)'  'NP(*)'       phase fraction (wildcard -> every phase)
  'X(CR)'  'X(FCC_A1,CR)'  'X(*,CR)'   overall / phase mole fraction
  'W(CR)'  'W(FCC_A1,CR)'     mass fraction
  'MU(CR)'                    chemical potential (J/mol)
  'Y(BCC_B2,0,AL)'            site fraction (phase, sublattice index, species)
  'DOO(BCC_B2)'  'TC(BCC_A2)'  'BMAG(BCC_A2)'  'VM'  'GM_MIX'  'HM_MIX'
  'HM.T'                      Jansson derivative d(HM)/dT at equilibrium = apparent Cp (includes transformations)
  'X(LIQUID,CR).T'  'MU(CR).X(CR)'   any numerator.denominator derivative
Units: NOT in the string. Use objects: v.T['degC'], as_property('GM')['kJ/mol'], as_property('HM')['J/g'].
Values for phases that are not stable at a point are NaN. MU for an element absent from a condition set may be NaN.
```

### workspace

```
Workspace usage:
  from pycalphad import Workspace, variables as v
  wks = Workspace(db, ['CR','NI','VA'], phases, {v.T:(1000,1600,50), v.X('CR'):0.3, v.P:101325, v.N:1})
  T, npf = wks.get('T', 'NP(FCC_A1)')          # arrays over the condition grid
  d = wks.get_dict('NP(*)')                    # {prop: array}
  wks.conditions[v.T] = 1200                   # reactive: recomputes on next get
  wks.eq.get_dataset()                         # xarray Dataset of the raw result
Arrays are shaped by wks.condition_axis_order (varying conditions only).
```

### metastable / driving force

```
Metastable and dormant phases (pycalphad.property_framework):
  IsolatedPhase('BCC_A2', wks)('GM')           phase minimized alone (metastable single-phase property)
  DormantPhase('BCC_A2', wks).driving_force    J/mol; >0 means the phase would precipitate
  DormantPhase('BCC_A2', wks)('X(BCC_A2,CR)')  composition of the most favourable nucleus
  T0('FCC_A1', 'BCC_A2', wks)                  T0 temperature (GM equal); needs both phases present in wks
The phase must have been computed in wks (start point) or pass a CompositionSet.
```

### reference states

```
Reference states:
  1) Workspace meta-property (any property, e.g. HM relative to pure FCC Ni and pure BCC Cr):
     from pycalphad.property_framework import ReferenceState
     ref = ReferenceState([('FCC_A1', {v.X('CR'): 0}), ('BCC_A2', {v.X('CR'): 1})], wks)
     wks.get(ref('HM'))          # mixing enthalpy vs those references (T,P float with wks)
     Fixed conditions (e.g. {v.X('CR'):0, v.T:298.15}) pin the reference; one entry per component.
  2) Model level: from pycalphad import Model, ReferenceState
     m = Model(db, comps, 'LIQUID'); m.shift_reference_state([ReferenceState('CR','BCC_A2'),
         ReferenceState('NI','FCC_A1')], db)   -> m.GMR, m.HMR, m.SMR, m.CPMR
         calculate(db, comps, 'LIQUID', T=1500, P=101325, N=1, output='HMR', model={'LIQUID': m})
     ReferenceState('CR', 'BCC_A2', fixed_statevars={v.T: 298.15}) for a fixed-T reference (SER-like).
  Activity: a_i = exp((MU_i - MU_i_ref)/(R T)), with MU_i_ref from a Workspace of the reference phase.
```

### mapping

```
Mapping (pycalphad.mapping) - modern phase-diagram engine:
  from pycalphad.mapping import BinaryStrategy, TernaryStrategy, IsoplethStrategy, StepStrategy, plot_binary
  s = BinaryStrategy(db, ['CR','NI','VA'], phases, {v.X('CR'):(0,1,0.02), v.T:(500,2200,10), v.P:101325, v.N:1})
  s.do_map()                        # finds starting points automatically (binplot/ternplot do this for you)
  data = s.get_tieline_data(v.X('CR'), v.T)   # list[StrategyData]; each .data -> [SinglePhaseData(phase,x,y)]
  inv = s.get_invariant_data(v.X('CR'), v.T)   # three-phase invariants
  Isopleth: IsoplethStrategy(... {v.X('CR'):(0,0.5,0.01), v.X('FE'):0.2, v.T:(...)}) -> get_zpf_data / get_invariant_data
  Step:     StepStrategy(... {v.T:(300,2000,10), v.X('CR'):0.3, ...}) -> get_data(v.T, v.NP('*')) or any property
  s.zpf_lines -> ZPFLine(fixed_phases, free_phases, points); ZPFLine.get_var_list(v.T)
  Plot: plot_binary(s), plot_ternary(s), plot_isopleth(s), plot_step(s, x, y)
```

### results

```
equilibrium() result (xarray.Dataset):
  dims: N, P, T, X_<EL>... (conditions), vertex (phase slot), component, internal_dof
  vars: NP (phase fraction per vertex), Phase (name per vertex, '' if empty), GM, MU(component),
        X(vertex, component) phase composition, Y site fractions, + any `output=` property (e.g. output='HM').
  Stable phases at a point: set(eq.Phase.values.ravel()) - {''}
  Export: eq.to_netcdf('x.nc'), eq.to_dataframe().to_csv(...), eq.NP.sel(T=1000).squeeze()
calculate() result: dims N,P,T,points; vars X, Y, Phase, <output>; points = sampled configurations.
```

### units

```
Display units (pint strings) are attached to property OBJECTS, not strings:
  v.T['degC'], v.P['bar'], as_property('GM')['kJ/mol'], as_property('HM')['J/g'] (per mass, uses phase molar masses),
  as_property('HM.T')['J/g/K'].  Defaults: energies J/mol, entropy/Cp J/mol/K, T kelvin, P pascal, fractions.
  wks.get(...) returns values in the display units; as_quantity(prop, value) wraps them as pint Quantities.
```

### database output

```
Database output:
  db.to_file('out.tdb', if_exists='overwrite')      # TDB is the only writable format ('dat' is read-only)
  db.to_file('out.tdb', groupby='phase', if_incompatible='fix')   # 'warn'|'raise'|'ignore'|'fix'
  txt = db.to_string(fmt='tdb')
  db.elements, db.species, db.phases[name].constituents / .sublattices / .model_hints, db.symbols
  from tinydb import where; db.search(where('phase_name') == 'LIQUID')
  Model(db, comps, phase).GM etc. are SymEngine expressions (str() gives text).
```
