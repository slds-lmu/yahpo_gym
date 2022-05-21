# YAHPO Gym (python)
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (R)](https://img.shields.io/badge/Software-R-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym_r)

### What is YAHPO Gym?

**YAHPO Gym** (Yet Another Hyperparameter Optimization Gym) is a collection of interesting problems to benchmark hyperparameter optimization (HPO) methods described in [our paper](https://arxiv.org/abs/2109.03670).

YAHPO Gym consists of several `scenarios`. A scenario (e.g. `lcbench`) is a collection of benchmark instances with the same underlying hyperparameter optimization task (e.g., optimizing the hyperparameters of a neural network) on different datasets (usually taken from [OpenML](https://www.openml.org/)).

### A note on OpenML IDs

Currently, the `rbv2_*`, `lcbench`, and `iaml_*` scenarios contain instances based on OpenML datasets.
For `rbv2_*` and `iaml_*` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the `lcbench` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>.

### Why should I use it?

**YAHPO Gym** provides blazing fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks across a large variety of problems.
Our library makes use of [ConfigSpace](https://automl.github.io/ConfigSpace/) to describe the hyperparameter space and can thus be seamlessly integrated into many existing projects (e.g. [HpBandSter](https://github.com/automl/HpBandSter)).

![image](https://github.com/slds-lmu/yahpo_gym/blob/main/assets/anytime_average_rank_mf.jpg?raw=true)

**Overview over benchmark instances**

|Scenario    |Search Space    |# Instances|Target Metrics                       |Fidelity| H|
|:-----------|---------------:|----------:|------------------------------------:|:-------|:-|
|rbv2_super  |38D: Mixed      |        103| 9: perf(6) + rt(2) + mem            |fraction| ✓|
|rbv2_svm    | 6D: Mixed      |        106| 9: perf(6) + rt(2) + mem            |fraction| ✓|
|rbv2_rpart  | 5D: Mixed      |        117| 9: perf(6) + rt(2) + mem            |fraction|  |
|rbv2_aknn   | 6D: Mixed      |        118| 9: perf(6) + rt(2) + mem            |fraction|  |
|rbv2_glmnet | 3D: Mixed      |        115| 9: perf(6) + rt(2) + mem            |fraction|  |
|rbv2_ranger | 8D: Mixed      |        119| 9: perf(6) + rt(2) + mem            |fraction| ✓|
|rbv2_xgboost|14D: Mixed      |        119| 9: perf(6) + rt(2) + mem            |fraction| ✓|
|nb301       |34D: Categorical|          1| 2: perf(1) + rt(1)                  |epoch   | ✓|
|lcbench     | 7D: Numeric    |         34| 6: perf(5) + rt(1)                  |epoch   |  |
|iaml_super  |28D: Mixed      |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction| ✓|
|iaml_rpart  | 4D: Numeric    |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction|  |
|iaml_glmnet | 2D: Numeric    |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction|  |
|iaml_ranger | 8D: Mixed      |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction| ✓|
|iaml_xgboost|13D: Mixed      |          4|12: perf(4) + inp(3) + rt(2) + mem(3)|fraction| ✓|

The fidelity is given either as the dataset fraction `fraction` or the number of epochs `epoch`.
Search spaces can be numeric, mixed and have dependencies (as indicated in the `H` column). 

The **full, up-to-date overview** can be obtained from the [Documentation](https://slds-lmu.github.io/yahpo_gym/scenarios.html).

### Installation

```console
pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
```

### Setup

To run a benchmark you need to obatin the ONNX model (`new_model.onnx`), [ConfigSpace](https://automl.github.io/ConfigSpace/) (`config_space.json`) and some encoding info (`encoding.json`).

You can download these [here (Github)](https://github.com/slds-lmu/yahpo_data) or [here (Syncshare)](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

You should pertain the folder structure as on the hosting site (i.e., create a `"path-to-data"` directory, for example named `"multifidelity_data"`, containing the individual, e.g., `"lcench"`, directories).

```py
# Initialize the local config & set path for surrogates and metadata
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

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

### Example: Tuning an instance using HPBandSter

We include a full example for optimization using **BOHB** on a YAHPO Gym instance in a [jupyter notebook](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb).

### All Examples

- [General Usage](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Code Samples](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/code_sample.ipynb)
- [Tuning with HpBandSter on Yahpo Gym](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb)
- [Transfer HPO](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Paper experiments](https://github.com/slds-lmu/yahpo_exps/tree/main/paper)
