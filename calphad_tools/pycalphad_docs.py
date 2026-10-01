"""Offline pycalphad documentation tools for the chat agent.

The agent may run without internet access and without permission to execute
arbitrary code, so it cannot "just look up" pycalphad's API.  This module
provides:

* ``INVENTORY`` - a curated inventory of pycalphad 0.11's public plotting and
  data-output functionality (the single source of truth for
  ``docs/pycalphad_inventory.{md,json}``; regenerate them with
  ``python -m calphad_tools.pycalphad_docs [docs_dir]``);
* ``GUIDES`` - short offline notes on pycalphad conventions;
* ``register(tb)`` - adds the ``pycalphad_help`` and
  ``list_calphad_capabilities`` tools to a :class:`~calphad_tools.agent.tools.ToolBox`.

Everything here works from the *installed* packages (``inspect`` on live
objects); nothing touches the network.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# kind codes used in the inventory
KINDS = {"P": "plotting", "D": "data-output", "B": "both", "S": "support"}

# (area, name, import path, kind code, one-line purpose)
_INV: List[Tuple[str, str, str, str, str]] = [
    # ---------------------------------------------------------------- top level
    ("top-level", "equilibrium", "pycalphad.equilibrium", "D",
     "Global equilibrium over a grid of conditions; returns xarray Dataset (NP, GM, MU, X, Y, Phase + `output` props)."),
    ("top-level", "calculate", "pycalphad.calculate", "D",
     "Sample a Model property (output='GM','HM',...) over site-fraction points for phases; returns xarray Dataset (X, Y, Phase, <output>)."),
    ("top-level", "Workspace", "pycalphad.Workspace", "D",
     "Reactive equilibrium calculator (db, components, phases, conditions); wks.get(*props) returns arrays of any computable property."),
    ("top-level", "binplot", "pycalphad.binplot", "P",
     "Binary isobaric phase diagram via BinaryStrategy + plot_binary; return_strategy=True also returns the strategy (data)."),
    ("top-level", "ternplot", "pycalphad.ternplot", "P",
     "Ternary isothermal section via TernaryStrategy + plot_ternary; return_strategy=True also returns the strategy (data)."),
    ("top-level", "eqplot", "pycalphad.eqplot", "P",
     "Plot an equilibrium() result (binary T-x or ternary isothermal) with tielines / tie-triangles."),
    ("top-level", "as_property", "pycalphad.as_property", "S",
     "Parse a property string ('GM', 'NP(FCC_A1)', 'X(LIQUID,CR)', 'HM.T', ...) into a ComputableProperty."),
    ("top-level", "Database", "pycalphad.Database", "D",
     "Thermodynamic database: read TDB/DAT, write TDB (to_file/to_string), search parameters."),
    ("top-level", "Model", "pycalphad.Model", "D",
     "Symbolic CALPHAD model of one phase: GM, HM, SM, CPM, mixing props, DOO, TC, BMAG, VM, contributions."),
    ("top-level", "ReferenceState (Model)", "pycalphad.ReferenceState", "S",
     "Pure-element reference state (species, phase, fixed_statevars) for Model.shift_reference_state."),
    ("top-level", "variables (v)", "pycalphad.variables", "S",
     "State variables / conditions / properties: T, P, N, X, W, MU, NP, Y, Species, Component."),
    # ---------------------------------------------------------------- pycalphad.plot
    ("plot", "eqplot", "pycalphad.plot.eqplot.eqplot", "P",
     "Scatter-plot phase boundaries from an equilibrium Dataset (x/y/z StateVariables auto-detected)."),
    ("plot", "phase_legend", "pycalphad.plot.utils.phase_legend", "P",
     "Build matplotlib legend handles and a phase->color dict for a list of phases."),
    ("plot", "TriangularAxes", "pycalphad.plot.triangular.TriangularAxes", "P",
     "matplotlib projection 'triangular' for Gibbs-triangle ternary plots (fig.add_subplot(projection='triangular'))."),
    ("plot", "ternplot (legacy)", "pycalphad.plot.ternary.ternplot", "P",
     "Legacy ternary isotherm: equilibrium on a composition grid then eqplot."),
    ("plot", "binplot (legacy)", "pycalphad.plot.binary.plot.binplot", "P",
     "Legacy binary mapper (map_binary + plot_boundaries)."),
    ("plot", "plot_boundaries", "pycalphad.plot.binary.plot.plot_boundaries", "P",
     "Plot ZPFBoundarySets (legacy binary mapper) as lines/scatter with tielines."),
    ("plot", "map_binary", "pycalphad.plot.binary.map.map_binary", "D",
     "Legacy binary T-X mapping; returns ZPFBoundarySets (phase-boundary data)."),
    ("plot", "ZPFBoundarySets", "pycalphad.plot.binary.zpf_boundary_sets.ZPFBoundarySets", "B",
     "Legacy container of two-phase regions; get_line_plot_boundaries/get_scatter_plot_boundaries/get_phases."),
    ("plot", "ZPFBoundarySets.get_line_plot_boundaries", "pycalphad.plot.binary.zpf_boundary_sets.ZPFBoundarySets.get_line_plot_boundaries", "B",
     "Line collections + legend for each two-phase region boundary (data for custom plots)."),
    ("plot", "ZPFBoundarySets.get_scatter_plot_boundaries", "pycalphad.plot.binary.zpf_boundary_sets.ZPFBoundarySets.get_scatter_plot_boundaries", "B",
     "Scatter x/y/colors + tieline collection for boundaries."),
    ("plot", "get_compsets", "pycalphad.plot.binary.compsets.get_compsets", "D",
     "Extract a pair of binary composition sets (phase, T, x, site fractions) from an equilibrium Dataset point."),
    ("plot", "find_two_phase_region_compsets", "pycalphad.plot.binary.compsets.find_two_phase_region_compsets", "D",
     "Two-phase composition sets from a 1D convex hull at fixed T, P."),
    # ---------------------------------------------------------------- pycalphad.mapping
    ("mapping", "binplot (mapping)", "pycalphad.mapping.compat_api.binplot", "B",
     "Map + plot binary diagram; return_strategy=True returns (ax, BinaryStrategy) for data export."),
    ("mapping", "ternplot (mapping)", "pycalphad.mapping.compat_api.ternplot", "B",
     "Map + plot ternary isotherm; return_strategy=True returns (ax, TernaryStrategy)."),
    ("mapping", "MapStrategy", "pycalphad.mapping.strategy.strategy_base.MapStrategy", "D",
     "Base mapping strategy: do_map(), zpf_lines, node_queue, get_all_phases(); kwargs GLOBAL_MIN_PDENS etc."),
    ("mapping", "MapStrategy.do_map", "pycalphad.mapping.strategy.strategy_base.MapStrategy.do_map", "S",
     "Run the mapping until all nodes are exhausted (max_iter)."),
    ("mapping", "MapStrategy.get_all_phases", "pycalphad.mapping.strategy.strategy_base.MapStrategy.get_all_phases", "D",
     "All phases found along ZPF lines (miscibility gaps as PHASE#2)."),
    ("mapping", "MapStrategy.add_nodes_from_conditions", "pycalphad.mapping.strategy.strategy_base.MapStrategy.add_nodes_from_conditions", "S",
     "Add a starting point computed from explicit conditions (optionally direction)."),
    ("mapping", "BinaryStrategy", "pycalphad.mapping.BinaryStrategy", "D",
     "Binary mapping (1 composition + 1 potential axis); data via get_tieline_data/get_invariant_data."),
    ("mapping", "BinaryStrategy.get_tieline_data", "pycalphad.mapping.strategy.binary_strategy.BinaryStrategy.get_tieline_data", "D",
     "List[StrategyData]: per ZPF line, phase-labelled x/y arrays (tieline end points) for any x,y variables."),
    ("mapping", "BinaryStrategy.get_invariant_data", "pycalphad.mapping.strategy.binary_strategy.BinaryStrategy.get_invariant_data", "D",
     "List[StrategyData]: three-phase invariant nodes (phase, x, y)."),
    ("mapping", "TernaryStrategy", "pycalphad.mapping.TernaryStrategy", "D",
     "Ternary isothermal mapping (2 composition axes); get_tieline_data/get_invariant_data (tie-triangles)."),
    ("mapping", "TernaryStrategy.get_tieline_data", "pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy.get_tieline_data", "D",
     "List[StrategyData] of two-phase tieline end points."),
    ("mapping", "TernaryStrategy.get_invariant_data", "pycalphad.mapping.strategy.ternary_strategy.TernaryStrategy.get_invariant_data", "D",
     "List[StrategyData] of three-phase tie-triangles."),
    ("mapping", "IsoplethStrategy", "pycalphad.mapping.IsoplethStrategy", "D",
     "Isopleth (vertical section) mapping; get_zpf_data/get_invariant_data."),
    ("mapping", "IsoplethStrategy.get_zpf_data", "pycalphad.mapping.strategy.isopleth_strategy.IsoplethStrategy.get_zpf_data", "D",
     "StrategyData of zero-phase-fraction boundary lines (phase, x, y) for the isopleth."),
    ("mapping", "IsoplethStrategy.get_invariant_data", "pycalphad.mapping.strategy.isopleth_strategy.IsoplethStrategy.get_invariant_data", "D",
     "List[StrategyData] of invariant lines/nodes in the isopleth."),
    ("mapping", "StepStrategy", "pycalphad.mapping.StepStrategy", "D",
     "One-axis step calculation (e.g. T or X); get_data(x, y) for any property vs axis."),
    ("mapping", "StepStrategy.get_data", "pycalphad.mapping.strategy.step_strategy.StepStrategy.get_data", "D",
     "StrategyData of y vs x per stable phase (or 'SYSTEM' for global props); global_x/global_y, set_nan_to_zero."),
    ("mapping", "add_starting_points_from_step", "pycalphad.mapping.strategy.binary_strategy.BinaryStrategy.add_starting_points_from_step", "S",
     "Seed a Binary/Ternary/Isopleth map from a StepStrategy result."),
    ("mapping", "generate_automatic_starting_points", "pycalphad.mapping.strategy.strategy_base.MapStrategy.generate_automatic_starting_points", "S",
     "Automatic starting-point search along axis limits."),
    ("mapping", "StrategyData", "pycalphad.mapping.strategy.strategy_data.StrategyData", "D",
     "Container: .data (list of SinglePhaseData), .phases, .x, .y."),
    ("mapping", "SinglePhaseData", "pycalphad.mapping.strategy.strategy_data.SinglePhaseData", "D",
     "phase name + x + y arrays for one phase / line."),
    ("mapping", "PhaseRegionData", "pycalphad.mapping.strategy.strategy_data.PhaseRegionData", "D",
     "StrategyData subclass for one phase region (tieline / invariant)."),
    ("mapping", "get_tieline_data_from_tieline_strategy", "pycalphad.mapping.strategy.strategy_data.get_tieline_data_from_tieline_strategy", "D",
     "Shared implementation of get_tieline_data for Binary/Ternary strategies."),
    ("mapping", "get_invariant_data_from_tieline_strategy", "pycalphad.mapping.strategy.strategy_data.get_invariant_data_from_tieline_strategy", "D",
     "Shared implementation of get_invariant_data for Binary/Ternary strategies."),
    ("mapping", "plot_binary", "pycalphad.mapping.plot_binary", "P",
     "Plot a finished BinaryStrategy (x/y vars, tielines every n, label_nodes, colors)."),
    ("mapping", "plot_ternary", "pycalphad.mapping.plot_ternary", "P",
     "Plot a finished TernaryStrategy on triangular axes."),
    ("mapping", "plot_isopleth", "pycalphad.mapping.plot_isopleth", "P",
     "Plot a finished IsoplethStrategy (ZPF lines + invariants)."),
    ("mapping", "plot_step", "pycalphad.mapping.plot_step", "P",
     "Plot a finished StepStrategy (any y property vs step axis, per phase)."),
    ("mapping", "plot_tielines", "pycalphad.mapping.plotting.plot_tielines", "P",
     "Draw tielines of a Binary/Ternary strategy on an axis."),
    ("mapping", "plot_invariants", "pycalphad.mapping.plotting.plot_invariants", "P",
     "Draw invariant nodes / tie-triangles of a Binary/Ternary strategy on an axis."),
    ("mapping", "get_label", "pycalphad.mapping.plotting.get_label", "P",
     "Axis label text for a StateVariable."),
    ("mapping", "ZPFLine", "pycalphad.mapping.primitives.ZPFLine", "D",
     "One zero-phase-fraction line: fixed/free phases, points, status; get_var_list(var)."),
    ("mapping", "ZPFLine.get_var_list", "pycalphad.mapping.primitives.ZPFLine.get_var_list", "D",
     "Values of a variable along a ZPF line."),
    ("mapping", "Point", "pycalphad.mapping.primitives.Point", "D",
     "Map point: global_conditions, chemical_potentials, stable phases/composition sets; get_property(var), get_local_property."),
    ("mapping", "Node", "pycalphad.mapping.primitives.Node", "D",
     "Special Point where phases change (invariant / start of new ZPF lines)."),
    ("mapping", "point_from_equilibrium", "pycalphad.mapping.starting_points.point_from_equilibrium", "D",
     "Compute a mapping Point from an equilibrium at given conditions."),
    ("mapping", "degrees_of_freedom", "pycalphad.mapping.utils.degrees_of_freedom", "D",
     "Gibbs-phase-rule degrees of freedom of a Point/Node."),
    # ---------------------------------------------------------------- Workspace / property framework
    ("workspace", "Workspace.get", "pycalphad.core.workspace.Workspace.get", "D",
     "Compute one or more properties over the condition grid; arrays shaped like the conditions (display units)."),
    ("workspace", "Workspace.get_dict", "pycalphad.core.workspace.Workspace.get_dict", "D",
     "Like get() but returns {property: array} (wildcards like 'NP(*)' expand to one key per phase)."),
    ("workspace", "Workspace.eq", "pycalphad.core.workspace.Workspace.eq", "D",
     "Underlying LightDataset equilibrium result (NP, GM, MU, X, Y, Phase); .get_dataset() -> xarray."),
    ("workspace", "Workspace.enumerate_composition_sets", "pycalphad.core.workspace.Workspace.enumerate_composition_sets", "D",
     "Yield (index, [CompositionSet]) for every grid point (stable phases, site fractions, NP)."),
    ("workspace", "Workspace.get_composition_sets", "pycalphad.core.workspace.Workspace.get_composition_sets", "D",
     "Composition sets of a point calculation (ndim == 0)."),
    ("workspace", "Workspace.copy", "pycalphad.core.workspace.Workspace.copy", "S",
     "Copy a workspace (then change conditions/phases)."),
    ("workspace", "Workspace.ndim / condition_axis_order", "pycalphad.core.workspace.Workspace.condition_axis_order", "S",
     "Number of varying conditions and their axis order in returned arrays."),
    ("workspace", "Conditions", "pycalphad.core.conditions.Conditions", "S",
     "Workspace conditions mapping (keys, str_keys, items(units=...))."),
    ("property framework", "ModelComputedProperty", "pycalphad.property_framework.ModelComputedProperty", "D",
     "Any Model attribute as a property: 'GM', 'HM(FCC_A1)', 'DOO(BCC_B2)', 'TC', 'VM'...; ['units'] changes display units."),
    ("property framework", "JanssonDerivative", "pycalphad.property_framework.JanssonDerivative", "D",
     "Derivative of an equilibrium property w.r.t. a condition: 'HM.T' (=Cp), 'X(LIQUID,CR).T', 'MU(CR).X(CR)'."),
    ("property framework", "LinearCombination", "pycalphad.property_framework.computed_property.LinearCombination", "S",
     "Symbolic linear combination of properties usable as condition (e.g. v.X('A') - v.X('B'))."),
    ("property framework", "IsolatedPhase", "pycalphad.property_framework.IsolatedPhase", "D",
     "Meta-property: property of one phase minimized alone under the same conditions (metastable single-phase, e.g. GM curves)."),
    ("property framework", "DormantPhase", "pycalphad.property_framework.DormantPhase", "D",
     "Meta-property: phase minimized at the equilibrium potentials without becoming stable; .driving_force."),
    ("property framework", "DormantPhase.driving_force", "pycalphad.property_framework.metaproperties.DormantPhase.driving_force", "D",
     "Driving force (J/mol) for precipitation of a dormant phase; >0 means it wants to form."),
    ("property framework", "DrivingForce", "pycalphad.property_framework.DrivingForce", "D",
     "Driving force property class used by DormantPhase(...).driving_force."),
    ("property framework", "ReferenceState (property)", "pycalphad.property_framework.ReferenceState", "D",
     "Meta-property re-referencing a property to phases at given conditions: ReferenceState([('FCC_A1',{v.X('CR'):0}),...], wks)('HM')."),
    ("property framework", "T0", "pycalphad.property_framework.T0", "D",
     "Property: T0 temperature where GM(phase1) == GM(phase2) at the current composition."),
    ("property framework", "as_quantity", "pycalphad.property_framework.as_quantity", "D",
     "Wrap a property value as a pint Quantity in the property's display units."),
    ("property framework", "to_display_units", "pycalphad.property_framework.units.to_display_units", "D",
     "Convert implementation units -> display units (J/mol -> J/g etc. using phase molar masses)."),
    ("property framework", "DimensionalityError", "pycalphad.property_framework.DimensionalityError", "S",
     "pint error raised by an impossible unit conversion."),
    ("property framework", "ComputableProperty", "pycalphad.property_framework.ComputableProperty", "S",
     "Protocol any property must satisfy (compute_property, shape, units, display_name)."),
    ("property framework", "ConditionableComputableProperty", "pycalphad.property_framework.ConditionableComputableProperty", "S",
     "Protocol: may appear in the denominator of a Jansson derivative (T, P, X, MU, N ...)."),
    ("property framework", "DifferentiableComputableProperty", "pycalphad.property_framework.DifferentiableComputableProperty", "S",
     "Protocol: may appear in the numerator of a Jansson derivative."),
    # ---------------------------------------------------------------- variables
    ("variables", "T", "pycalphad.variables.T", "S", "Temperature (K). v.T['degC'] gives display in Celsius."),
    ("variables", "P", "pycalphad.variables.P", "S", "Pressure (Pa); usual condition v.P: 101325."),
    ("variables", "N", "pycalphad.variables.N", "S", "Total system moles (usually v.N: 1)."),
    ("variables", "X", "pycalphad.variables.X", "D",
     "Mole fraction: v.X('CR') overall or v.X('FCC_A1','CR') in a phase; condition and property."),
    ("variables", "W", "pycalphad.variables.W", "D", "Mass fraction: v.W('CR') / v.W(phase, 'CR'); condition and property."),
    ("variables", "MU", "pycalphad.variables.MU", "D", "Chemical potential v.MU('CR') (J/mol); property and condition."),
    ("variables", "NP", "pycalphad.variables.NP", "D", "Phase fraction v.NP('FCC_A1') (moles of atoms); 'NP(*)' for all phases."),
    ("variables", "Y (site fraction)", "pycalphad.variables.Y", "D",
     "Site fraction v.Y('BCC_B2', sublattice_index, 'AL'); string 'Y(BCC_B2,0,AL)'."),
    ("variables", "Species", "pycalphad.variables.Species", "S", "Chemical species (name, constituents, charge); .weight."),
    ("variables", "Component", "pycalphad.variables.Component", "S", "Component (pure element or linear combination)."),
    ("variables", "StateVariable", "pycalphad.variables.StateVariable", "S",
     "Base class of all variables; prop['units'] sets display units; display_name."),
    ("variables", "get_mass_fractions", "pycalphad.variables.get_mass_fractions", "D",
     "Convert mole fractions -> mass fractions for a point composition."),
    ("variables", "get_mole_fractions", "pycalphad.variables.get_mole_fractions", "D",
     "Convert mass fractions -> mole fractions (e.g. wt% alloy spec -> conditions)."),
    ("variables", "unpack_components", "pycalphad.variables.unpack_components", "S",
     "Normalize component list (strings/Species) to Components."),
    # ---------------------------------------------------------------- Model
    ("model", "Model.GM", "pycalphad.Model.GM", "D", "Molar Gibbs energy per mole of atoms (J/mol) (alias energy)."),
    ("model", "Model.G", "pycalphad.Model.G", "D", "Gibbs energy per formula unit (J) (alias formulaenergy)."),
    ("model", "Model.HM", "pycalphad.Model.HM", "D", "Molar enthalpy (alias enthalpy)."),
    ("model", "Model.H", "pycalphad.Model.H", "D", "Enthalpy per formula unit (alias formulaenthalpy)."),
    ("model", "Model.SM", "pycalphad.Model.SM", "D", "Molar entropy (alias entropy)."),
    ("model", "Model.CPM", "pycalphad.Model.CPM", "D", "Molar heat capacity of the phase (alias heat_capacity); apparent equilibrium Cp incl. transformations = 'HM.T'."),
    ("model", "Model.GM_MIX", "pycalphad.Model.GM_MIX", "D", "Gibbs energy of mixing relative to endmembers (alias mixing_energy)."),
    ("model", "Model.HM_MIX", "pycalphad.Model.HM_MIX", "D", "Enthalpy of mixing (alias mixing_enthalpy)."),
    ("model", "Model.SM_MIX", "pycalphad.Model.SM_MIX", "D", "Entropy of mixing (alias mixing_entropy)."),
    ("model", "Model.CPM_MIX", "pycalphad.Model.CPM_MIX", "D", "Heat capacity of mixing (alias mixing_heat_capacity)."),
    ("model", "Model.DOO", "pycalphad.Model.degree_of_ordering", "D", "Degree of ordering (0 disordered .. 1), property 'DOO(PHASE)'."),
    ("model", "Model.TC / BMAG / NT", "pycalphad.Model.TC", "D",
     "Curie temperature, Bohr magneton number (BMAG), Neel temperature of the phase (symbolic)."),
    ("model", "Model.VM", "pycalphad.Model.volume_energy", "D", "Molar volume VM (and V0, VA, VC, VK) from volume parameters."),
    ("model", "Model.models", "pycalphad.Model.build_phase", "D",
     "Dict of energy contributions: ref, idmix, xsmix, mag, 2st, ein, vol, ord (each a symbolic expression)."),
    ("model", "Model.ast", "pycalphad.Model.ast", "D", "Full symbolic Gibbs-energy expression (SymEngine)."),
    ("model", "Model.variables", "pycalphad.Model.variables", "D", "State variables + site fractions in the model."),
    ("model", "Model.moles", "pycalphad.Model.moles", "D", "Symbolic moles of a species/element in the phase."),
    ("model", "Model.endmember_reference_model", "pycalphad.Model.endmember_reference_model", "D",
     "Model with only endmember contributions (reference for *_MIX properties)."),
    ("model", "Model.shift_reference_state", "pycalphad.Model.shift_reference_state", "D",
     "Adds GMR/HMR/SMR/CPMR attributes: properties relative to given pure-element ReferenceStates."),
    ("model", "Model energy contributions", "pycalphad.Model.excess_mixing_energy", "D",
     "reference_energy, ideal_mixing_energy, excess_mixing_energy, magnetic_energy, atomic_ordering_energy, "
     "twostate_energy, einstein_energy, volume_energy (symbolic pieces)."),
    ("model", "ModelMQMQA", "pycalphad.models.model_mqmqa.ModelMQMQA", "D",
     "Modified quasichemical (MQMQA) model used for ChemSage DAT SUBQ phases."),
    # ---------------------------------------------------------------- Database / IO
    ("database", "Database.from_file", "pycalphad.Database.from_file", "S", "Read database (fmt 'tdb' or 'dat', autodetected)."),
    ("database", "Database.from_string", "pycalphad.Database.from_string", "S", "Read database from a string (fmt=...)."),
    ("database", "Database.to_file", "pycalphad.Database.to_file", "D",
     "Write database (only 'tdb' writable); if_exists raise|rename|overwrite; write kwargs groupby, if_incompatible."),
    ("database", "Database.to_string", "pycalphad.Database.to_string", "D", "Database as TDB text (fmt='tdb')."),
    ("database", "Database.search", "pycalphad.Database.search", "D",
     "Query parameters (tinydb where('phase_name')==...) -> list of parameter dicts."),
    ("database", "Database.register_format", "pycalphad.Database.register_format", "S", "Register read/write functions for a format."),
    ("database", "Database.add_parameter / add_phase", "pycalphad.Database.add_parameter", "S", "Build/modify databases programmatically."),
    ("database", "Database attributes", "pycalphad.io.database.Database", "D",
     "elements, species, phases (Phase: sublattices, constituents, model_hints), symbols (FUNCTIONs), refstates, _parameters."),
    ("database", "Phase", "pycalphad.io.database.Phase", "D", "Phase record: name, constituents, sublattices, model_hints."),
    ("database", "write_tdb", "pycalphad.io.tdb.write_tdb", "D",
     "TDB writer: groupby 'subsystem'|'phase'; if_incompatible 'warn'|'raise'|'ignore'|'fix' (name mangling, line length)."),
    ("database", "read_tdb", "pycalphad.io.tdb.read_tdb", "S", "TDB parser used by Database.from_file."),
    ("database", "TCPrinter", "pycalphad.io.tdb.TCPrinter", "D", "Print SymEngine expressions in Thermo-Calc syntax."),
    ("database", "reflow_text", "pycalphad.io.tdb.reflow_text", "D", "Wrap TDB text lines to <= linewidth."),
    ("database", "read_cs_dat", "pycalphad.io.cs_dat.read_cs_dat", "S", "ChemSage/FactSage DAT reader (read-only format)."),
    ("database", "format_registry", "pycalphad.io.database.format_registry", "S",
     "Registered formats: tdb (read+write), dat (read only)."),
    # ---------------------------------------------------------------- results / xarray
    ("results", "equilibrium Dataset", "xarray.Dataset", "D",
     "equilibrium()/calculate() return xarray.Dataset; dims = condition axes (N,P,T,X_..), vertex, component, internal_dof."),
    ("results", "Dataset.to_netcdf", "xarray.Dataset.to_netcdf", "D", "Save a full equilibrium/calculate result to NetCDF."),
    ("results", "Dataset.to_dataframe", "xarray.Dataset.to_dataframe", "D", "Flatten a result to a pandas DataFrame (then .to_csv)."),
    ("results", "Dataset.sel / squeeze / where", "xarray.Dataset.sel", "D", "Slice results, e.g. eq.NP.sel(T=1000).squeeze()."),
    ("results", "LightDataset.get_dataset", "pycalphad.core.light_dataset.LightDataset.get_dataset", "D",
     "Convert equilibrium(to_xarray=False) / Workspace.eq result to an xarray Dataset."),
    # ---------------------------------------------------------------- core utils
    ("utils", "filter_phases", "pycalphad.core.utils.filter_phases", "S", "Phases valid for the given components."),
    ("utils", "instantiate_models", "pycalphad.core.utils.instantiate_models", "S", "Build {phase: Model} dict (custom Model classes)."),
    ("utils", "get_pure_elements", "pycalphad.core.utils.get_pure_elements", "S", "Pure elements of a component list."),
    ("utils", "unpack_species", "pycalphad.core.utils.unpack_species", "S", "Component names -> Species set."),
    ("utils", "generate_dof", "pycalphad.core.utils.generate_dof", "D", "Site-fraction variable list of a phase for given components."),
    ("utils", "endmember_matrix", "pycalphad.core.utils.endmember_matrix", "D", "Endmember site-fraction matrix (for sampling/plots)."),
    ("utils", "point_sample", "pycalphad.core.utils.point_sample", "S", "Random site-fraction sampling used by calculate."),
    ("utils", "PhaseRecordFactory", "pycalphad.codegen.phase_record_factory.PhaseRecordFactory", "S",
     "Compiled phase records (fast property evaluation) used by Workspace/mapping."),
    # ---------------------------------------------------------------- related installed packages
    ("related packages", "simulate_scheil_solidification", "scheil.simulate_scheil_solidification", "D",
     "Scheil-Gulliver solidification path (scheil 0.3): returns SolidificationResult."),
    ("related packages", "simulate_equilibrium_solidification", "scheil.simulate_equilibrium_solidification", "D",
     "Equilibrium (lever-rule) solidification path: returns SolidificationResult."),
    ("related packages", "SolidificationResult", "scheil.SolidificationResult", "D",
     "fraction_solid, temperatures, phase_amounts, phase_compositions; to_dict/from_dict/to_dataframe."),
    ("related packages", "espei dataplot", "espei.plot.dataplot", "P",
     "Overlay ESPEI ZPF datasets (experimental phase boundaries) on a phase-diagram axis."),
    ("related packages", "espei plot_endmember", "espei.plot.plot_endmember", "P",
     "Plot an endmember property vs T with thermochemical datasets."),
    ("related packages", "espei plot_interaction", "espei.plot.plot_interaction", "P",
     "Plot an interaction (mixing) property vs composition with datasets."),
]

# Conventions / how-to notes that are searchable by pycalphad_help.
GUIDES: Dict[str, str] = {
    "conditions": """Conditions (dict) for equilibrium / Workspace / strategies:
  {v.T: 1273.15, v.P: 101325, v.N: 1, v.X('CR'): 0.3}          point
  {v.T: (300, 2000, 10), ...}                                  range (start, stop, step) -> an axis
Temperatures are always in KELVIN, pressure in Pa, compositions as mole fractions (independent ones
only: n_elements-1 X conditions; the balance element is implicit). Mass fractions: v.W('CR'): 0.2.
Components must include 'VA' when phases have vacancies: ['CR', 'NI', 'VA'].
Phases: list of names from the database (e.g. sorted(db.phases)); pycalphad drops invalid ones.""",
    "property strings": """Workspace.get / as_property strings (also usable as StepStrategy.get_data variables):
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
Values for phases that are not stable at a point are NaN. MU for an element absent from a condition set may be NaN.""",
    "workspace": """Workspace usage:
  from pycalphad import Workspace, variables as v
  wks = Workspace(db, ['CR','NI','VA'], phases, {v.T:(1000,1600,50), v.X('CR'):0.3, v.P:101325, v.N:1})
  T, npf = wks.get('T', 'NP(FCC_A1)')          # arrays over the condition grid
  d = wks.get_dict('NP(*)')                    # {prop: array}
  wks.conditions[v.T] = 1200                   # reactive: recomputes on next get
  wks.eq.get_dataset()                         # xarray Dataset of the raw result
Arrays are shaped by wks.condition_axis_order (varying conditions only).""",
    "metastable / driving force": """Metastable and dormant phases (pycalphad.property_framework):
  IsolatedPhase('BCC_A2', wks)('GM')           phase minimized alone (metastable single-phase property)
  DormantPhase('BCC_A2', wks).driving_force    J/mol; >0 means the phase would precipitate
  DormantPhase('BCC_A2', wks)('X(BCC_A2,CR)')  composition of the most favourable nucleus
  T0('FCC_A1', 'BCC_A2', wks)                  T0 temperature (GM equal); needs both phases present in wks
The phase must have been computed in wks (start point) or pass a CompositionSet.""",
    "reference states": """Reference states:
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
  Activity: a_i = exp((MU_i - MU_i_ref)/(R T)), with MU_i_ref from a Workspace of the reference phase.""",
    "mapping": """Mapping (pycalphad.mapping) - modern phase-diagram engine:
  from pycalphad.mapping import BinaryStrategy, TernaryStrategy, IsoplethStrategy, StepStrategy, plot_binary
  s = BinaryStrategy(db, ['CR','NI','VA'], phases, {v.X('CR'):(0,1,0.02), v.T:(500,2200,10), v.P:101325, v.N:1})
  s.do_map()                        # finds starting points automatically (binplot/ternplot do this for you)
  data = s.get_tieline_data(v.X('CR'), v.T)   # list[StrategyData]; each .data -> [SinglePhaseData(phase,x,y)]
  inv = s.get_invariant_data(v.X('CR'), v.T)   # three-phase invariants
  Isopleth: IsoplethStrategy(... {v.X('CR'):(0,0.5,0.01), v.X('FE'):0.2, v.T:(...)}) -> get_zpf_data / get_invariant_data
  Step:     StepStrategy(... {v.T:(300,2000,10), v.X('CR'):0.3, ...}) -> get_data(v.T, v.NP('*')) or any property
  s.zpf_lines -> ZPFLine(fixed_phases, free_phases, points); ZPFLine.get_var_list(v.T)
  Plot: plot_binary(s), plot_ternary(s), plot_isopleth(s), plot_step(s, x, y)""",
    "results": """equilibrium() result (xarray.Dataset):
  dims: N, P, T, X_<EL>... (conditions), vertex (phase slot), component, internal_dof
  vars: NP (phase fraction per vertex), Phase (name per vertex, '' if empty), GM, MU(component),
        X(vertex, component) phase composition, Y site fractions, + any `output=` property (e.g. output='HM').
  Stable phases at a point: set(eq.Phase.values.ravel()) - {''}
  Export: eq.to_netcdf('x.nc'), eq.to_dataframe().to_csv(...), eq.NP.sel(T=1000).squeeze()
calculate() result: dims N,P,T,points; vars X, Y, Phase, <output>; points = sampled configurations.""",
    "units": """Display units (pint strings) are attached to property OBJECTS, not strings:
  v.T['degC'], v.P['bar'], as_property('GM')['kJ/mol'], as_property('HM')['J/g'] (per mass, uses phase molar masses),
  as_property('HM.T')['J/g/K'].  Defaults: energies J/mol, entropy/Cp J/mol/K, T kelvin, P pascal, fractions.
  wks.get(...) returns values in the display units; as_quantity(prop, value) wraps them as pint Quantities.""",
    "database output": """Database output:
  db.to_file('out.tdb', if_exists='overwrite')      # TDB is the only writable format ('dat' is read-only)
  db.to_file('out.tdb', groupby='phase', if_incompatible='fix')   # 'warn'|'raise'|'ignore'|'fix'
  txt = db.to_string(fmt='tdb')
  db.elements, db.species, db.phases[name].constituents / .sublattices / .model_hints, db.symbols
  from tinydb import where; db.search(where('phase_name') == 'LIQUID')
  Model(db, comps, phase).GM etc. are SymEngine expressions (str() gives text).""",
}

_STOP = {"the", "a", "an", "to", "of", "how", "do", "i", "in", "for", "and", "with", "is", "are", "what", "does",
         "can", "get", "use", "using", "pycalphad", "on", "by", "from", "my", "me", "it", "or", "as", "be"}

# phrase -> extra search tokens
_SYNONYMS: Dict[str, str] = {
    "vertical section": "isopleth isoplethstrategy",
    "isopleth": "isoplethstrategy plot_isopleth get_zpf_data",
    "driving force": "driving_force drivingforce dormantphase",
    "reference state": "referencestate shift_reference_state reference",
    "reference": "referencestate",
    "phase fraction": "np phasefraction",
    "chemical potential": "mu chemicalpotential",
    "activity": "mu chemicalpotential referencestate",
    "heat capacity": "cpm heat_capacity jansson",
    "cp": "cpm heat_capacity",
    "enthalpy": "hm",
    "entropy": "sm",
    "gibbs": "gm",
    "mixing": "gm_mix hm_mix mixing",
    "export": "to_file to_string to_netcdf to_dataframe get_data get_tieline_data",
    "save": "to_file to_netcdf to_dataframe",
    "write": "to_file write_tdb to_string",
    "tdb": "database write_tdb to_file read_tdb",
    "tieline": "get_tieline_data plot_tielines",
    "tie line": "get_tieline_data plot_tielines",
    "invariant": "get_invariant_data plot_invariants",
    "eutectic": "get_invariant_data invariant",
    "step": "stepstrategy plot_step get_data",
    "ternary": "ternarystrategy ternplot plot_ternary triangular",
    "isothermal": "ternarystrategy ternplot",
    "binary": "binarystrategy binplot plot_binary",
    "phase diagram": "binplot ternplot mapping strategy",
    "scheil": "simulate_scheil_solidification solidificationresult",
    "solidification": "simulate_scheil_solidification simulate_equilibrium_solidification",
    "site fraction": "y sitefraction",
    "mole fraction": "x molefraction",
    "mass fraction": "w massfraction get_mass_fractions",
    "weight": "w massfraction get_mole_fractions",
    "ordering": "doo degree_of_ordering",
    "metastable": "isolatedphase dormantphase",
    "units": "as_quantity to_display_units display_units",
    "legend": "phase_legend",
    "triangular": "triangularaxes",
    "magnetic": "tc bmag magnetic_energy",
    "curie": "tc",
    "volume": "vm volume_energy",
    "xarray": "dataset to_netcdf to_dataframe",
    "netcdf": "to_netcdf",
    "csv": "to_dataframe get_data",
    "zpf": "zpfline get_zpf_data",
    "condition": "conditions",
    "derivative": "janssonderivative",
    "t0": "t0",
    "experimental": "dataplot",
    "vs": "stepstrategy get_data plot_step",
    "versus": "stepstrategy get_data plot_step",
    "unit": "as_quantity to_display_units display_units",
}


# signatures that introspection cannot recover (``*args`` constructors, descriptors, instances)
_WS_SIG = ("(database, components, phases, conditions, *, models=None, parameters=None, calc_opts=None, "
           "solver=None, verbose=False, constituents=None, phase_record_factory=None)")
_SIG_OVERRIDES: Dict[str, str] = {
    "pycalphad.Workspace": _WS_SIG,
    "pycalphad.core.workspace.Workspace": _WS_SIG,
    "pycalphad.Model": "(dbe, comps, phase_name, parameters=None)",
    "pycalphad.model.Model": "(dbe, comps, phase_name, parameters=None)",
    "pycalphad.Database": "(fname_or_tdb_string=None)",
    "pycalphad.io.database.Database": "(fname_or_tdb_string=None)",
    "pycalphad.core.workspace.Workspace.eq": "(attribute: LightDataset | None)",
    "pycalphad.variables.T": "(TemperatureType instance; v.T['degC'] for display units)",
    "pycalphad.variables.P": "(PressureType instance)",
    "pycalphad.variables.N": "(SystemMolesType instance)",
    "pycalphad.Model.TC": "(instance attribute, symbolic; also BMAG, NT)",
    "pycalphad.io.database.format_registry": "(dict: fmt -> DatabaseFormat(read, write))",
}


def _inventory_raw() -> List[Dict[str, str]]:
    return [{"area": a, "name": n, "path": p, "kind": KINDS[k], "summary": s} for a, n, p, k, s in _INV]


# ------------------------------------------------------------------ introspection helpers
def _quiet_import(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module(name)


def resolve(path: str) -> Any:
    """Resolve a dotted path to a live object (module, class, function, attribute). Raises on failure."""
    parts = path.split(".")
    obj, i = None, len(parts)
    while i > 0:
        try:
            obj = _quiet_import(".".join(parts[:i]))
            break
        except Exception:  # noqa: BLE001 - try a shorter prefix
            i -= 1
    if obj is None:
        raise ImportError(path)
    for attr in parts[i:]:
        if inspect.isclass(obj) and attr in getattr(obj, "__dict__", {}):
            obj = obj.__dict__[attr]            # keep property / staticmethod objects (not their values)
            continue
        found = False
        if inspect.isclass(obj):
            for base in obj.__mro__:
                if attr in base.__dict__:
                    obj, found = base.__dict__[attr], True
                    break
        if not found:
            obj = getattr(obj, attr)
    return obj


def _clean(s: str) -> str:
    return re.sub(r" at 0x[0-9a-f]+>", ">", s)


def signature_of(obj: Any) -> str:
    if isinstance(obj, property):
        return "(property)"
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if inspect.ismodule(obj):
        return "(module)"
    try:
        if inspect.isclass(obj):
            init = obj.__dict__.get("__init__")
            sig = inspect.signature(obj)
            if str(sig) in ("(*args, **kwargs)",) and init is None:
                return "(*args, **kwargs)"
            return _clean(str(sig))
        if callable(obj):
            return _clean(str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "(...)"
    return ""


def doc_of(obj: Any) -> str:
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        return inspect.getdoc(obj.fget) or inspect.getdoc(obj) or ""
    if not (inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj)):
        return ""  # plain values (e.g. symbols) would only show their type's docstring
    return inspect.getdoc(obj) or ""


@lru_cache(maxsize=1)
def build_inventory() -> List[Dict[str, str]]:
    """Inventory entries with live signatures (from the installed packages)."""
    out = []
    for e in _inventory_raw():
        e = dict(e)
        try:
            e["signature"] = _SIG_OVERRIDES.get(e["path"]) or signature_of(resolve(e["path"]))
        except Exception:  # noqa: BLE001 - package missing or attribute moved
            e["signature"] = "(unavailable)"
        out.append(e)
    return out


def _walk_modules(pkg_name: str) -> List[Any]:
    try:
        pkg = _quiet_import(pkg_name)
    except Exception:  # noqa: BLE001
        return []
    mods = [pkg]
    if hasattr(pkg, "__path__"):
        for m in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
            if ".tests" in m.name or m.name.endswith("._dev"):
                continue
            try:
                mods.append(_quiet_import(m.name))
            except Exception:  # noqa: BLE001
                continue
    return mods


@lru_cache(maxsize=1)
def live_index() -> List[Dict[str, str]]:
    """Every public function/class/method/property defined in pycalphad (+ scheil, espei.plot)."""
    entries: List[Dict[str, str]] = []
    seen = set()
    mods = _walk_modules("pycalphad") + _walk_modules("scheil") + _walk_modules("espei.plot")
    for mod in mods:
        mname = mod.__name__
        mdoc = inspect.getdoc(mod) or ""
        if mdoc:
            entries.append({"name": mname.rsplit(".", 1)[-1], "path": mname, "kind": "module",
                            "summary": mdoc.split("\n")[0], "doc": mdoc, "signature": "(module)"})
        for n, o in vars(mod).items():
            if n.startswith("_") or not (inspect.isfunction(o) or inspect.isclass(o) or inspect.isbuiltin(o)
                                         or type(o).__name__ == "cython_function_or_method"):
                continue
            if getattr(o, "__module__", None) != mname:
                continue
            path = f"{mname}.{n}"
            if id(o) in seen:
                continue
            seen.add(id(o))
            d = doc_of(o)
            entries.append({"name": n, "path": path, "kind": "api", "summary": d.split("\n")[0] if d else "",
                            "doc": d, "signature": signature_of(o), "oid": id(o)})
            if inspect.isclass(o):
                for an, ao in vars(o).items():
                    if an.startswith("_"):
                        continue
                    if not (isinstance(ao, (property, staticmethod, classmethod)) or inspect.isroutine(ao)):
                        continue
                    ad = doc_of(ao)
                    entries.append({"name": f"{n}.{an}", "path": f"{path}.{an}", "kind": "api",
                                    "summary": ad.split("\n")[0] if ad else "", "doc": ad,
                                    "signature": signature_of(ao), "oid": id(ao)})
    return entries


@lru_cache(maxsize=1)
def search_corpus() -> List[Dict[str, str]]:
    """Guides + curated inventory (with live docstrings) + the rest of the live index."""
    corpus: List[Dict[str, str]] = []
    for name, text in GUIDES.items():
        corpus.append({"name": f"guide: {name}", "path": f"guide:{name}", "kind": "guide", "summary": text.split("\n")[0],
                       "doc": text, "signature": "", "curated": "1"})
    used_paths, used_ids = set(), set()
    for e in build_inventory():
        try:
            obj = resolve(e["path"])
            doc = doc_of(obj)
            if inspect.isclass(obj) or inspect.isroutine(obj) or isinstance(obj, (property, staticmethod, classmethod)):
                used_ids.add(id(obj))
        except Exception:  # noqa: BLE001
            doc = ""
        used_paths.add(e["path"])
        corpus.append({**e, "doc": doc, "curated": "1"})
    for e in live_index():   # the same object is often re-exported under several paths: keep one
        if e["path"] in used_paths or e.get("oid") in used_ids:
            continue
        used_ids.add(e.get("oid"))
        corpus.append(e)
    return corpus


def _tokens(q: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_*#]+", q.lower()) if t not in _STOP]


def _expand(q: str) -> List[str]:
    ql = q.lower()
    toks = _tokens(q)
    extra: List[str] = []
    for phrase, syn in _SYNONYMS.items():
        if (" " in phrase and phrase in ql) or phrase in toks:
            extra += syn.split()
    out: List[str] = []
    for t in toks + extra:
        if t not in out:
            out.append(t)
    return out


def _score(e: Dict[str, str], toks: List[str], orig: List[str], phrase: str) -> float:
    name = e["name"].lower()
    last = name.rsplit(".", 1)[-1]
    name_n = name.replace("_", "")
    path = e["path"].lower()
    summ = e.get("summary", "").lower()
    doc = e.get("doc", "").lower()
    s = 0.0
    matched = 0
    for t in toks:
        w = 1.0 if t in orig else 0.6          # synonyms count a bit less
        hit = 0.0
        tn = t.replace("_", "")
        if t == last or tn == last.replace("_", ""):
            hit += 10
        elif len(t) > 1 and (t in name or tn in name_n):
            hit += 5
        if len(t) > 1 and t in path:
            hit += 2
        if len(t) > 2 and t in summ:
            hit += 2
        if len(t) > 2:
            c = doc.count(t)
            hit += min(c, 3) * 0.7
        if hit:
            matched += 1 if t in orig else 0
        s += w * hit
    if phrase and len(phrase) > 3 and (phrase in summ or phrase in doc or phrase in name):
        s += 6
    if orig:
        s *= 0.5 + 0.5 * matched / len(orig)     # prefer entries matching all the user's words
    if e.get("curated"):
        s *= 1.5
    if e.get("kind") == "guide":
        s *= 1.2
    return s


def search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    orig = _tokens(query)
    toks = _expand(query)
    if not toks:
        return []
    phrase = " ".join(orig)
    scored = [(_score(e, toks, orig, phrase), i, e) for i, e in enumerate(search_corpus())]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [e for _, _, e in scored[:limit]]


def _trim(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def _members(obj: Any, limit: int = 80) -> List[str]:
    lines = []
    if inspect.ismodule(obj):
        items = [(n, o) for n, o in vars(obj).items() if not n.startswith("_")
                 and (inspect.isclass(o) or inspect.isroutine(o) or inspect.ismodule(o))]
    else:
        items = []
        for klass in reversed(obj.__mro__[:-1]):          # own methods override inherited ones
            for n, o in vars(klass).items():
                if n.startswith("_"):
                    continue
                if isinstance(o, (property, staticmethod, classmethod)) or inspect.isroutine(o):
                    items = [x for x in items if x[0] != n] + [(n, o)]
    for n, o in sorted(items, key=lambda x: x[0])[:limit]:
        tag = "property" if isinstance(o, property) else ("module" if inspect.ismodule(o) else "")
        sig = "" if tag else signature_of(o)
        first = (doc_of(o) or "").split("\n")[0]
        lines.append(f"  .{n}{_trim(sig, 140)}{' [' + tag + ']' if tag else ''}" + (f" - {_trim(first, 110)}" if first else ""))
    return lines


def describe(path: str) -> Optional[str]:
    """Full signature + docstring + public members for an exact dotted path, or None if unresolvable."""
    candidates = [path] if path.startswith(("pycalphad", "scheil", "espei", "xarray")) else [path, "pycalphad." + path]
    for cand in candidates:
        try:
            obj = resolve(cand)
        except Exception:  # noqa: BLE001
            continue
        inv = [e for e in build_inventory() if e["path"] == cand]
        head = [f"{cand}  [{inv[0]['kind'] if inv else 'api'}]"]
        if inv:
            head.append(f"purpose: {inv[0]['summary']}")
        sig = _SIG_OVERRIDES.get(cand) or signature_of(obj)
        if sig:
            head.append(f"signature: {cand.rsplit('.', 1)[-1]}{sig}")
        d = doc_of(obj)
        if not d and not (inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj)
                          or isinstance(obj, property)):
            d = f"value: {_trim(repr(obj), 300)} (type {type(obj).__name__})"
        head.append("\n" + (_trim(d, 6000) if d else "(no docstring)"))
        if inspect.isclass(obj) or inspect.ismodule(obj):
            mem = _members(obj)
            if mem:
                head.append("\npublic members:\n" + "\n".join(mem))
        return "\n".join(head)
    return None


def _fmt_hit(e: Dict[str, str], n_doc: int = 1500) -> str:
    kind = e.get("kind", "api")
    lines = [f"### {e['name']}  [{kind}]", f"path: {e['path']}" if kind != "guide" else ""]
    if e.get("signature") and e["signature"] not in ("(module)",):
        lines.append(f"signature: {e['name'].rsplit('.', 1)[-1]}{_trim(e['signature'], 400)}")
    if e.get("curated") and kind != "guide" and e.get("summary"):
        lines.append(f"purpose: {e['summary']}")
    doc = e.get("doc", "")
    if doc and doc.strip() != e.get("summary", "").strip():
        lines.append(_trim(doc, n_doc))
    return "\n".join(x for x in lines if x)


def help_text(query: str, limit: int = 5) -> str:
    q = (query or "").strip()
    if not q:
        return ("Give a topic (e.g. 'isopleth', 'driving force', 'reference state', 'property strings', 'to_file') "
                "or an exact dotted path (e.g. 'pycalphad.mapping.BinaryStrategy'). Guides: "
                + ", ".join(GUIDES))
    if q.lower().startswith("guide:") and q[6:].strip().lower() in GUIDES:
        return GUIDES[q[6:].strip().lower()]
    if re.fullmatch(r"[A-Za-z_][\w]*(\.[A-Za-z_]\w*)+", q) or re.fullmatch(r"[A-Z]\w*", q):
        d = describe(q)
        if d is not None:
            related = [e["path"] for e in search(q.rsplit(".", 1)[-1], limit=6) if e["path"] != q][:5]
            return d + ("\n\nsee also: " + ", ".join(related) if related else "")
    hits = search(q, limit=limit)
    if not hits:
        return f"No offline pycalphad documentation matched {q!r}. Try other words or a dotted path."
    out = [f"Top {len(hits)} offline matches for {q!r} (pycalphad {_version()}); "
           "call again with an exact dotted path for the full docstring and members."]
    out += [_fmt_hit(e) for e in hits]
    return "\n\n".join(out)


def _version() -> str:
    try:
        return _quiet_import("pycalphad").__version__
    except Exception:  # noqa: BLE001
        return "?"


# ------------------------------------------------------------------ capabilities map
PLUGIN_MODULES = ["calphad_tools.figures", "calphad_tools.props", "calphad_tools.mapping_tools",
                  "calphad_tools.espei_plot_tools", "calphad_tools.pycalphad_docs"]

# task -> tool names (only tools actually registered in the session are shown)
TASKS: List[Tuple[str, List[str]]] = [
    ("find files / read PDFs & text / look at images", ["list_files", "read_document", "view_pdf_page", "view_image"]),
    ("find databases (TDB/.dat) by folder, name pattern and elements; resolve user file names exactly",
     ["find_databases"]),
    ("inspect a TDB (elements, phases, sublattice models, assessed subsystems)", ["tdb_info"]),
    ("ONE phase diagram with data export - binary T-x (also wt%, T-MU axis), ternary isotherm, isopleth; "
     "boundaries/tie-lines/invariant reactions as CSV/JSON", ["map_phase_diagram"]),
    ("phase names on a diagram (each region / tie-triangle, e.g. L+FCC, BCC+σ; binary invariant temperatures), "
     "editable SVG/PDF text: label_regions=true, or edit_figure label_phase_regions / move_label / relabel",
     ["map_phase_diagram", "plot_diagram_with_data", "edit_figure"]),
    ("PowerPoint decks of ALL binaries / ALL ternary sections of a TDB", ["plot_all_binaries", "plot_all_ternaries"]),
    ("calculated binary diagram or ternary isotherm with experimental ESPEI datasets overlaid",
     ["plot_diagram_with_data"]),
    ("which element goes on which side: right (binary, e.g. Cr on the right) / corners (ternary, e.g. Fe "
     "bottom-right); element_order for decks; data follow: edit_figure composition_points places points by "
     "composition; transform_data element_fractions turns digitized x columns into X(EL) columns",
     ["map_phase_diagram", "plot_diagram_with_data", "plot_all_binaries", "edit_figure", "transform_data"]),
    ("step calculation: phase fractions/compositions/properties vs T, P or X; transition temperatures",
     ["step_map", "equilibrium_property_plot"]),
    ("equilibrium at a few specific conditions (table of stable phases/compositions)", ["calculate_equilibrium"]),
    ("any equilibrium property over 1-2 varying conditions (NP, X(phase,el), MU, ACR activity, HM/SM/CPM, HM.T, "
     "DOO, site fractions Y, units like J/g) incl. 2-D maps", ["equilibrium_property_plot"]),
    ("driving force / nucleus composition / metastable (isolated or dormant) phases / T0",
     ["equilibrium_property_plot"]),
    ("single-phase properties without equilibrium: G-x curves, common tangent, mixing (GM_MIX...) properties, "
     "reference states, one phase's property vs T", ["phase_property_plot"]),
    ("raw equilibrium dataset (netCDF + tidy CSV), eqplot", ["export_equilibrium_data"]),
    ("add your own data points / lines / labels / arrows / shaded regions to an existing plot, recolour or restyle "
     "phases, change limits/scales/legend, export PDF/SVG (describe first, then edit)",
     ["describe_figure", "edit_figure"]),
    ("new custom figure (multi-panel, any plot type: fits, statistics, heatmaps, ternary tricontour maps, stack "
     "plots, bars, 3-D, secondary axes, journal presets) from inline data or files; op syntax: plot_ops_help",
     ["make_plot", "plot_ops_help"]),
    ("look at / reshape a data file (CSV/Excel/JSON/netCDF): columns, stats, group means, pivots, joins, unit "
     "conversions -> new table", ["inspect_data", "transform_data"]),
    ("curve fitting numbers only (Arrhenius, polynomial, custom expression; params ± SE, R²)", ["fit_data"]),
    ("combine images into (a)-(d) panels, PowerPoint decks from figures, animations (GIF/MP4, e.g. isotherms vs T), "
     "image crop/resize/convert, multipage PDF",
     ["compose_images", "figures_to_pptx", "animate", "image_tools", "images_to_pdf"]),
    ("Scheil-Gulliver and equilibrium solidification", ["scheil_solidification"]),
    ("write / normalise / subset a TDB (elements, phases); export for Thermo-Calc / OpenCalphad (dialect: name "
     "and line limits, listed renames)", ["write_database"]),
    ("use or convert a ChemSage/FactSage .dat database (calculation tools read .dat directly; convert to TDB, "
     "report what TDB cannot express)", ["convert_database", "tdb_info"]),
    ("calculations feel slow / cache statistics / clear the model & compiled-code cache", ["calc_cache"]),
    ("symbolic Gibbs energy (ref/ideal/excess/magnetic/ordering) and parameter list of a phase",
     ["export_phase_model"]),
    ("TDB endmember/interaction property vs ESPEI thermochemical data", ["plot_parameter_vs_data"]),
    ("composition unit conversion (wt% <-> at% / mole fraction)", ["convert_composition"]),
    ("compare / merge TDBs (2-way full diff; N-way table for one element subsystem with inlined functions, "
     "formation energies, phase candidates -> phase_map on a second call)",
     ["compare_tdbs", "compare_system", "merge_tdbs"]),
    ("experimental data -> ESPEI datasets; ESPEI + Flux jobs",
     ["parse_dataplot_exp", "convert_dataplot_exp_to_zpf", "convert_pop_to_zpf", "convert_csv_to_zpf",
      "write_zpf_dataset", "check_espei_datasets", "make_phase_models", "generate_espei_flux_batch"]),
    ("extract data points / curves from a figure in a paper (PDF page or image; x-y axes or a composition "
     "triangle): calibrate from ticks/corners, then markers by colour/shape or lines by colour",
     ["digitize_setup", "digitize_points", "digitize_curve", "digitize_fix"]),
    ("extract phase-diagram data from a figure into ESPEI ZPF datasets: phases from the region labels / legend on "
     "the figure (paper names -> database names), tie-lines, invariants, binary and ternary; optional TDB check",
     ["digitize_setup", "digitize_points", "digitize_curve", "digitized_to_zpf", "check_espei_datasets",
      "plot_diagram_with_data"]),
    ("run ESPEI locally (check inputs, parameter generation, quick MCMC before a Flux run)",
     ["espei_validate_input", "espei_generate_parameters", "espei_run_mcmc"]),
    ("background jobs (long ESPEI/optimizer runs): progress, ETA, plots, results, stop",
     ["list_jobs", "job_status", "wait_job", "stop_job"]),
    ("ESPEI MCMC results: convergence, burn-in, parameter statistics, corner plots, best-fit TDB, uncertainty "
     "bands on diagrams/properties", ["analyze_mcmc", "mcmc_to_tdb", "propagate_uncertainty"]),
    ("thermochemical ESPEI datasets (HM_MIX, HM_FORM, activities ...) and goodness-of-fit of a TDB vs all data",
     ["write_thermo_dataset", "write_equilibrium_dataset", "convert_pop_to_thermo", "convert_csv_to_thermo",
      "fit_report"]),
    ("what-if: change TDB parameters or symbols (set / add to a coefficient / scale / remove / add), with the "
     "diagram overlay, invariant and boundary shifts, and the change of the ESPEI objective", ["edit_parameters"]),
    ("parameter sensitivities: d(ESPEI objective, invariant T/X, boundary X, property)/d(parameter), tornado plot, "
     "correlations and identifiability", ["parameter_sensitivity"]),
    ("fit TDB parameters to ESPEI datasets with a gradient-based optimizer (least squares or L-BFGS-B; long runs "
     "as a background job)", ["optimize_parameters"]),
    ("database sanity checks (discontinuities, unary Tm/Cp, inverted liquid gaps, high-T solid stability)",
     ["check_tdb"]),
    ("session history, provenance of an output, reproducible script / notebook export",
     ["session_log", "provenance", "export_session"]),
    ("standalone pycalphad/matplotlib code behind a specific figure or result (paste into Jupyter)",
     ["show_code"]),
    ("every phase of an element system across databases: sublattice models, constituents per site, model "
     "type, magnetic/order-disorder settings (phase x database table)", ["phase_inventory"]),
    ("statistics of an element system: how many binaries/ternaries, which are assessed in which database and "
     "which are only extrapolated", ["system_coverage"]),
    ("the previous result: add data to the last plot (edit_figure image='@last'), list recent outputs, or "
     "repeat a call with changed settings", ["recent_results", "rerun_call", "edit_figure"]),
    ("cannot satisfy a request / tool bug / wrong result -> save a problem report for the developers",
     ["report_problem"]),
    ("the user is not satisfied with an answer: package the conversation, decisions, files they agree to share and "
     "their statement into a zip for the developers", ["package_feedback"]),
    ("custom Python (only if enabled)", ["run_python"]),
    ("ask the user", ["ask_user"]),
    ("offline pycalphad docs", ["pycalphad_help", "list_calphad_capabilities"]),
]

CONVENTIONS = """pycalphad conventions (offline notes):
- Temperatures are Kelvin inside pycalphad (tools may take temperature_unit 'K'/'C'); P in Pa (101325); v.N = 1.
- Components: element names in UPPERCASE plus 'VA' when phases have vacancies, e.g. ['CR','NI','VA'].
- Conditions: {v.T: 1200, v.P: 101325, v.N: 1, v.X('CR'): 0.3}; a range is (start, stop, step);
  give n_elements-1 composition conditions (balance element implicit); v.W('CR') for mass fraction.
- Variables: v.T v.P v.N; v.X('CR') / v.X('FCC_A1','CR'); v.W(...); v.MU('CR'); v.NP('FCC_A1');
  v.Y('BCC_B2', 0, 'AL') site fraction (phase, sublattice index, species).
- Workspace property strings: 'GM' 'HM' 'SM' 'CPM' 'NP(FCC_A1)' 'NP(*)' 'X(CR)' 'X(FCC_A1,CR)' 'W(CR)'
  'MU(CR)' 'Y(PHASE,subl,SPECIES)' 'DOO(PHASE)' 'TC(PHASE)' 'GM_MIX'; derivatives 'HM.T' (apparent Cp),
  'X(LIQUID,CR).T'. Units via objects: v.T['degC'], as_property('GM')['kJ/mol'] / ['J/g'] (not in strings).
- Values for unstable phases are NaN. Miscibility-gap second composition sets are named PHASE#2.
- Metastable: IsolatedPhase('BCC_A2', wks)('GM'); driving force DormantPhase('BCC_A2', wks).driving_force (J/mol,
  >0 = would precipitate); T0('FCC_A1','BCC_A2', wks).
- Reference states: ReferenceState([('FCC_A1',{v.X('CR'):0}),('BCC_A2',{v.X('CR'):1})], wks)('HM');
  Model-level: Model.shift_reference_state([ReferenceState('CR','BCC_A2'), ...], db) -> GMR/HMR/SMR/CPMR.
- Mapping: BinaryStrategy / TernaryStrategy / IsoplethStrategy / StepStrategy (pycalphad.mapping); data via
  get_tieline_data / get_invariant_data / get_zpf_data / get_data -> StrategyData(phase, x, y).
- Database output: only TDB is writable (db.to_file(path, if_exists='overwrite', groupby='subsystem'|'phase',
  if_incompatible='warn'|'fix')); ChemSage .dat is read-only.
- Use pycalphad_help(query) for signatures/docstrings of any pycalphad object (offline)."""


class _Recorder:
    """Stand-in ToolBox that records tool registrations without touching the real one."""

    def __init__(self, tb):
        self._tb = tb
        self.tools: Dict[str, str] = {}

    def add(self, name, description, params, required):
        def deco(fn):
            self.tools[name] = description
            return fn
        return deco

    def __getattr__(self, item):
        return getattr(self._tb, item)


def _first_sentence(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    m = re.match(r"(.+?(?<!e\.g)(?<!i\.e)(?<!etc)\.)(\s+[A-Z(]|$)", text)
    s = m.group(1) if m else text
    return _trim(s, n)


def capabilities_text(tb) -> str:
    tools = getattr(tb, "tools", {}) or {}
    desc: Dict[str, str] = {}
    source: Dict[str, str] = {}
    for name, t in tools.items():
        spec = getattr(t, "spec", None)
        desc[name] = getattr(spec, "description", "") or ""
        fn = getattr(t, "fn", None)
        source[name] = getattr(fn, "__module__", "") or ""
    notes = []
    # plugin modules: import defensively; list tools that exist in a module but are not loaded in this session
    for mod in PLUGIN_MODULES:
        try:
            m = _quiet_import(mod)
        except Exception as e:  # noqa: BLE001 - module missing or mid-edit
            notes.append(f"{mod}: not available ({type(e).__name__})")
            continue
        loaded = [n for n, s in source.items() if s == mod]
        if loaded or not hasattr(m, "register"):
            continue
        rec = _Recorder(tb)
        try:
            m.register(rec)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{mod}: register failed ({type(e).__name__}: {_trim(str(e), 80)})")
            continue
        missing = [n for n in rec.tools if n not in tools]
        if missing:
            notes.append(f"{mod}: importable but not loaded in this session (restart to use): {', '.join(missing)}")

    lines = ["TASK -> TOOL(S) registered in this session:"]
    covered = set()
    for label, names in TASKS:
        hits = [n for n in names if n in tools]
        if hits:
            covered.update(hits)
            lines.append(f"- {label}: {', '.join(hits)}")
    other = [n for n in tools if n not in covered]
    if other:
        lines.append(f"- other tools: {', '.join(other)}")
    lines.append("\nALL TOOLS (name: purpose):")
    groups: Dict[str, List[str]] = {}
    for n in tools:
        groups.setdefault(source.get(n, "") or "?", []).append(n)
    for mod, names in groups.items():
        short = mod.replace("calphad_tools.", "") or "?"
        lines.append(f"[{short}]")
        for n in names:
            lines.append(f"  {n}: {_first_sentence(desc.get(n, ''))}")
    if notes:
        lines.append("\nNOTES:\n" + "\n".join(f"- {x}" for x in notes))
    lines.append("\n" + CONVENTIONS)
    return "\n".join(lines)


# ------------------------------------------------------------------ registration
def register(tb) -> None:
    S = {"type": "string"}

    @tb.add("pycalphad_help",
            "Offline pycalphad docs: search API/docstrings/usage notes by topic (e.g. 'isopleth', 'driving force', "
            "'property strings', 'to_file'), or give an exact dotted path (e.g. 'pycalphad.mapping.BinaryStrategy') "
            "for full signature, docstring and members.",
            {"query": {**S, "description": "topic words or dotted path"}}, ["query"])
    def pycalphad_help(query: str):
        return help_text(query), []

    @tb.add("list_calphad_capabilities",
            "Map of tasks -> which tool to call (from the tools actually loaded), plus pycalphad conventions "
            "(variables, conditions, property strings, units, reference states).", {}, [])
    def list_calphad_capabilities():
        return capabilities_text(tb), []


# ------------------------------------------------------------------ docs generation
def write_inventory_files(docs_dir: str) -> Tuple[str, str]:
    """Write docs/pycalphad_inventory.json and .md from INVENTORY (live signatures)."""
    d = Path(docs_dir)
    d.mkdir(parents=True, exist_ok=True)
    inv = build_inventory()
    jpath = d / "pycalphad_inventory.json"
    jpath.write_text(json.dumps([{k: e[k] for k in ("name", "path", "kind", "signature", "summary", "area")}
                                 for e in inv], indent=1) + "\n")
    md = [f"# pycalphad {_version()} plotting & data-output inventory", "",
          "Generated offline from the installed packages by `python -m calphad_tools.pycalphad_docs docs`.",
          "Kinds: **plotting** (makes figures), **data-output** (produces numbers/files/text), "
          "**both**, **support** (inputs/infrastructure needed to use the others).", ""]
    areas: Dict[str, List[Dict[str, str]]] = {}
    for e in inv:
        areas.setdefault(e["area"], []).append(e)
    for area, es in areas.items():
        md += [f"## {area}", "", "| name | import path | kind | signature | purpose |", "|---|---|---|---|---|"]
        for e in es:
            sig = _trim(e["signature"], 160).replace("|", "\\|")
            md.append(f"| {e['name']} | `{e['path']}` | {e['kind']} | `{sig}` | {e['summary'].replace('|', '/')} |")
        md.append("")
    md += ["## Usage notes (offline guides)", ""]
    for name, text in GUIDES.items():
        md += [f"### {name}", "", "```", text, "```", ""]
    mpath = d / "pycalphad_inventory.md"
    mpath.write_text("\n".join(md))
    return str(jpath), str(mpath)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "docs")
    print(*write_inventory_files(target), sep="\n")
