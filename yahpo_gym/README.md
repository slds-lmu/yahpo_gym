# YAHPO Gym (python)
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (R)](https://img.shields.io/badge/Software-R-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym_r)

### What is YAHPO Gym?

**YAHPO Gym** (Yet Another Hyperparameter Optimization Gym) is a collection of interesting problems to benchmark hyperparameter optimization (HPO) methods described in [our paper](https://arxiv.org/abs/2109.03670).
YAHPO Gym consists of several `scenarios`. A scenario (e.g. `lcbench`) is a collection of benchmark instances with the same underlying hyperparameter optimization task (e.g., optimizing the hyperparameters of a neural network) on different datasets (usually taken from [OpenML](https://www.openml.org/)).

**Overview over benchmark instances**

|Scenario    |Search Space    |# Instances|Target Metrics                       |Fidelity| H|     Source|
|:-----------|---------------:|----------:|------------------------------------:|:-------|-:|----------:|
|rbv2_super  |38D: Mixed      |        103| 9: perf(6) + rt(2) + mem            |fraction| ✓|        [1]|
|rbv2_svm    | 6D: Mixed      |        106| 9: perf(6) + rt(2) + mem            |fraction| ✓|        [1]|
|rbv2_rpart  | 5D: Mixed      |        117| 9: perf(6) + rt(2) + mem            |fraction|  |        [1]|
|rbv2_aknn   | 6D: Mixed      |        118| 9: perf(6) + rt(2) + mem            |fraction|  |        [1]|
|rbv2_glmnet | 3D: Mixed      |        115| 9: perf(6) + rt(2) + mem            |fraction|  |        [1]|
|rbv2_ranger | 8D: Mixed      |        119| 9: perf(6) + rt(2) + mem            |fraction| ✓|        [1]|
|rbv2_xgboost|14D: Mixed      |        119| 9: perf(6) + rt(2) + mem            |fraction| ✓|        [1]|
|nb301       |34D: Categorical|          1| 2: perf(1) + rt(1)                  |epoch   | ✓|   [2], [3]|
|lcbench     | 7D: Numeric    |         34| 6: perf(5) + rt(1)                  |epoch   |  |   [4], [5]|
|iaml_super  |28D: Mixed      |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction| ✓|        [6]|
|iaml_rpart  | 4D: Numeric    |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction|  |        [6]|
|iaml_glmnet | 2D: Numeric    |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction|  |        [6]|
|iaml_ranger | 8D: Mixed      |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction| ✓|        [6]|
|iaml_xgboost|13D: Mixed      |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction| ✓|        [6]|

The **full, up-to-date overview** can be obtained from the [Documentation](https://slds-lmu.github.io/yahpo_gym/scenarios.html).
The fidelity is given either as the dataset fraction `fraction` or the number of epochs `epoch`.
Search spaces can be numeric, mixed and have dependencies (as indicated in the `H` column).

Original data sources are given by:

- [1] Binder M., Pfisterer F. & Bischl B. (2020). Collecting Empirical Data About Hyperparameters for Data Driven AutoML. 7th ICML Workshop on Automated Machine Learning.
- [2] Siems, J., Zimmer, L., Zela, A., Lukasik, J., Keuper, M., & Hutter, F. (2020). NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search. arXiv preprint arXiv:2008.09777, 11.
- [3] Zimmer, L. (2020). nasbench301_full_data. figshare. Dataset. https://doi.org/10.6084/m9.figshare.13286105.v1, Apache License, Version 2.0.
- [4] Zimmer, L., Lindauer, M., & Hutter, F. (2021). Auto-Pytorch: Multi-Fidelity Metalearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(9), 3079-3090.
- [5] Zimmer, L. (2020). data_2k_lw.zip. figshare. Dataset. https://doi.org/10.6084/m9.figshare.11662422.v1, Apache License, Version 2.0.
- [6] None, simply cite Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M., & Bischl, B. (2022). YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. In International Conference on Automated Machine Learning.

**Please make sure to always also cite the original data sources as YAHPO Gym would not have been possible without them!**

### Installation

```console
pip install yahpo-gym
```

### Setup

To run a benchmark you need to obatin the ONNX model (`new_model.onnx`), [ConfigSpace](https://automl.github.io/ConfigSpace/) (`config_space.json`) and some encoding info (`encoding.json`).

You can download these [here (Github)](https://github.com/slds-lmu/yahpo_data) or [here (Syncshare)](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

You should pertain the folder structure as on the hosting site (i.e., create a `"path-to-data"` directory, for example named `"multifidelity_data"`, containing the individual, e.g., `"lcbench"`, directories).

```py
# Initialize the local config & set path for surrogates and metadata
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```
You can test whether the setup was successful by instantiating the object as documented in the **Usage** section below.

### Usage

This example showcases the simplicity of YAHPO Gym's API.
A longer introduction is given in the accompanying [jupyter notebook](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb).

```py
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench
# Select a Benchmark
bench = benchmark_set.BenchmarkSet("lcbench")
# List available instances
bench.instances
# Set an instance
bench.set_instance("3945")
# Sample a point from the configspace (containing parameters for the instance and budget)
config = bench.get_opt_space().sample_configuration(1).get_dictionary()
# Evaluate
print(bench.objective_function(config))
```

### A note on OpenML Task IDs

Currently, the `rbv2_*`, `lcbench`, and `iaml_*` scenarios contain instances based on OpenML datasets.
For `rbv2_*` and `iaml_*` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the `lcbench` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>.

### Example: Tuning an instance using HPBandSter

We include a full example for optimization using **BOHB** on a YAHPO Gym instance in a [jupyter notebook](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb).

### All Examples

- [General Usage](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Code Samples](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/code_sample.ipynb)
- [Tuning with HpBandSter on YAHPO Gym](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb)
- [Transfer HPO](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Paper Experiments](https://github.com/slds-lmu/yahpo_exps/tree/main/paper)

### Citation

If you use YAHPO Gym, please cite the following paper:

- Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M., & Bischl, B. (2022). YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. In International Conference on Automated Machine Learning.

Moreover, certain `scenarios` built upon previous work, e.g., the `lcbench` scenario uses data from:

- Zimmer, L., Lindauer, M., & Hutter, F. (2021). Auto-Pytorch: Multi-Fidelity Metalearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(9), 3079-3090.
- Zimmer, L. (2020). data_2k_lw.zip. figshare. Dataset. https://doi.org/10.6084/m9.figshare.11662422.v1, Apache License, Version 2.0.

**Please make sure to always also cite the original data sources as YAHPO Gym would not have been possible without them!**

Original data sources of a scenario that should also be cited are provided via the `"citation"` key within the `config` dictionary of a scenario, e.g.:

```py
from yahpo_gym.configuration import cfg
lcbench = cfg("lcbench")
lcbench.config.get("citation")
```
