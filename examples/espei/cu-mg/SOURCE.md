# ESPEI Cu-Mg example

Input files of the ESPEI Cu-Mg tutorial (https://espei.org/0.9.1/tutorials/cu-mg-example/cu-mg-example.html),
copied from https://github.com/PhasesResearchLab/ESPEI/tree/master/docs/tutorials/cu-mg-example
(MIT License, Copyright (c) 2015-2018 Richard Otis; datasets prepared by Aleksei Egorov).

* `phase_models.json`, `input-data/`: phase models and datasets (thermochemical, activity, ZPF)
* `run_param_gen.yaml`, `run_mcmc.yaml`: the tutorial's ESPEI inputs (2000 MCMC iterations, one core)

The tutorial's own result TDBs are kept out of the agent's view in `tests/data/espei_cu_mg/` for comparison.
