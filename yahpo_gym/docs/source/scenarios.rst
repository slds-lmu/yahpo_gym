Scenarios \& Instances
************************


Scenarios
=======================

The following table provides an overview over **Scenarios** included in **YAHPO Gym** and included **Instances**:

.. csv-table:: Scenario Overview
   :file: _static/scenario_stats.csv
   :header-rows: 1
   :stub-columns: 1

* mixed = numeric and categorical hyperparameters
* perf = performance measures
* rt = train/predict time
* mem = memory consumption
* inp = interpretability measures
* H = Hierarchical search space

Note that the fidelity parameter is not included in the search space dimensionality.

In `yahpo_gym`, there is a `Configuration` object for each **scenario**. 

A list of all available scenarios can be obtained as follows:

.. code-block:: python

   from yahpo_gym.configuration import cfg
   print(cfg())

Instances
=======================

YAHPO Gym **instances** are evaluations of an ML algorithm (**scenario**) with a given set of hyperparameters on a specific dataset. 
Currently, the `rbv2_*`, `lcbench`, and `iaml_*` scenarios contain instances based on OpenML datasets.
For `rbv2_*` and `iaml_*` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the `lcbench` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>.

.. csv-table:: Scenario Overview
   :file: _static/instances.csv
   :header-rows: 1
   :stub-columns: 1

A list of available instances can be obtained from the `instances` slot after instantiating the `BenchmarkSet`.

.. code-block:: python

   from yahpo_gym import BenchmarkSet
   BenchmarkSet("lcbench").instances

Users can now instantiate a `Benchmark` or `config` with this `<ID>`, e.g. using `cfg("<ID>")`.
