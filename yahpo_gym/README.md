# YAHPO Gym (python)
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (R)](https://img.shields.io/badge/Software-R-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym)


### What is YAHPO GYM?

---

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) is a collection of interesting problems to benchmark hyperparameter optimization (HPO) methods described in [our paper](https://arxiv.org/abs/2109.03670).

YAHPO GYM consists of several `scenarios`. A scenario (e.g. `lcbench`) is a collection of benchmark instances with the same underlying hyperparameter optimization task (e.g., optimizing the hyperparameters of a neural network) on different datasets taken from [OpenML](https://www.openml.org/). 


### Why should I use it?

**YAHPO GYM** provides blazingly fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks
across a large variety of problems.
Our library makes use of [ConfigSpace](https://automl.github.io/ConfigSpace/) to describe the hyperparameter space and can thus be seamlessly integrated into many existing projects (e.g. [HpBandSter](https://github.com/automl/HpBandSter)).

![image](https://github.com/slds-lmu/yahpo_gym/blob/main/assets/results.png?raw=true)


**Overview over benchmark instances**

|Scenario     | #HPs| #Targets| #Instances|Space      |Fidelity |
|:------------|----:|--------:|----------:|:----------|:--------|
|rbv2_svm     |    6|        9|        106|Mixed      |frac     |
|rbv2_ranger  |    8|        9|        119|Mixed      |frac     |
|rbv2_rpart   |    5|        9|        117|Mixed      |frac     |
|rbv2_glmnet  |    3|        9|        115|Mixed      |frac     |
|rbv2_xgboost |   14|        9|        119|Mixed      |frac     |
|rbv2_aknn    |    6|        9|        118|Mixed      |frac     |
|rbv2_super   |   38|        9|        103|Mixed      |frac     |
|nb301        |   33|        2|          1|Mixed+Deps |epoch    |
|lcbench      |    7|        6|         34|Continuous |epoch    |
|iaml_ranger  |    8|       12|          4|Mixed+Deps |frac     |
|iaml_rpart   |    4|       12|          4|Continuous |frac     |
|iaml_glmnet  |    2|       12|          4|Continuous |frac     |
|iaml_xgboost |   13|       12|          4|Mixed+Deps |frac     |
|iaml_super   |   28|       12|          4|Mixed+Deps |frac     |

with "#HPs" hyperparameter, "#Targets" output metrics available across "#Instances" different instances.
The fidelity is given either as the dataset fraction `frac` or the number of epochs `epoch`.
Search spaces can be continuous, mixed and have dependencies (Deps).


The **full, up-to-date overview** can be obtained from the [Documentation](https://slds-lmu.github.io/yahpo_gym/scenarios.html).

---

### Installation

```console
pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
```

### Setup

To run a benchmark you need to obatin the ONNX model (`new_model.onnx`), [ConfigSpace](https://automl.github.io/ConfigSpace/) (`config_space.json`) and some encoding info (`encoding.json`).

You can download these [here (Github)](https://github.com/slds-lmu/yahpo_data) or [here (Synchshare)](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

You should pertain the folder structure as on the hosting site (i.e., create a `"path-to-data"` directory, for example named `"multifidelity_data"`, containing the individual, e.g., `"lcench"`, directories).

```py
# Initialize the local config & set path for surrogates and metadata
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

### Usage

This example showcases the simplicity of YAHPO GYM's API.
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

We include a full example for optimization using **BOHB** on a YAHPO GYM instance in a [jupyter notebook](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb).

### All Examples

- [General Usage](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Code Samples](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/code_sample.ipynb)
- [Tuning with HpBandSter on Yahpo Gym](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb)
- [Transfer HPO](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Paper experiments](https://github.com/slds-lmu/yahpo_exps/tree/main/paper)
