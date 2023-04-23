Scenarios \& Instances
************************


Scenarios
=======================

The following table provides an overview over **Scenarios** included in `YAHPO Gym` and included **Instances**:

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

Original data sources are given by:

* [1] Binder M., Pfisterer F. & Bischl B. (2020). Collecting Empirical Data About Hyperparameters for Data Driven AutoML. 7th ICML Workshop on Automated Machine Learning.
* [2] Siems, J., Zimmer, L., Zela, A., Lukasik, J., Keuper, M., & Hutter, F. (2020). NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search. arXiv preprint arXiv:2008.09777, 11.
* [3] Zimmer, L. (2020). nasbench301_full_data. figshare. Dataset. https://doi.org/10.6084/m9.figshare.13286105.v1, Apache License, Version 2.0.
* [4] Zimmer, L., Lindauer, M., & Hutter, F. (2021). Auto-Pytorch: Multi-Fidelity Metalearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(9), 3079-3090.
* [5] Zimmer, L. (2020). data_2k_lw.zip. figshare. Dataset. https://doi.org/10.6084/m9.figshare.11662422.v1, Apache License, Version 2.0.
* [6] None, simply cite Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M., & Bischl, B. (2022). YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. In International Conference on Automated Machine Learning.

Please make sure to always also cite the original data sources as YAHPO Gym would not have been possible without them!

In `yahpo_gym`, there is a `Configuration` object for each **scenario**. 

A list of all available scenarios can be obtained as follows:

.. code-block:: python

   from yahpo_gym.configuration import cfg
   print(cfg())

Instances
=======================

`YAHPO Gym` **instances** are evaluations of an ML algorithm (**scenario**) with a given set of hyperparameters on a specific dataset. 
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
