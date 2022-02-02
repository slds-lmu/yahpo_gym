
Scenarios \& Instances
************************

The following table provides an overview over **Scenarios** included in **YAHPO GYM** and included instances:

* `scenario`: The name of the scenario.
* `n_targets`: Numer of objectives (if multi-objective)
* `n_cat`, `n_num`: Number of categorical and continuous hyperparameters.
* `fidelity`: Available fidelity parameters
* `n_instances`: Number of instances (global).
* `min(task)`, `max(task)`: Statistics on the number of samples used to fit surrogate models for each task.

.. csv-table:: Scenario Overview
   :file: _static/scenario_stats.csv
   :header-rows: 1
   :stub-columns: 1

In `yahpo_gym`, there is a `Configuration` object for each **scenario**. 
A list of all available scenarios can be obtained as follows:

.. code-block:: python

   from yahpo_gym.configuration import cfg, cfgs()
   print(cfg())


Instances
=======================

YAHPO GYM **instances** are evaluations of an ML algorithm (**scenario**) with a given set of hyperparameters on a specific task. 
We list **OpenML Task IDs** from openml.org as instance names to obtain a cleary defined setting that allows for reproducibility.
To provide an example, instance `168868` defines the `Task` a supervised classification task on the `APSFailure` dataset using `10-fold CV` (https://www.openml.org/t/168868).

.. csv-table:: Scenario Overview
   :file: _static/instances.csv
   :header-rows: 1
   :stub-columns: 1

A list of available instances can be obtained from the `instances` slot after instantiating the `BenchmarkSet`.

.. code-block:: python

   from yahpo_gym import BenchmarkSet
   BenchmarkSet("lcbench").instances

Users can now instantiate a `Benchmark` or `config` with this `<ID>`, e.g. using `cfg("<ID>")`.
