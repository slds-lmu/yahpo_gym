# YAHPO Gym (python)
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (R)](https://img.shields.io/badge/Software-R-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym)


### What is YAHPO GYM?

---

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) is a collection of interesting problem sets for benchmark hyperparameter optimization / black-box optimization methods described in [our paper](https://arxiv.org/abs/2109.03670).

YAHPO GYM consists of several `scenarios`, e.g. the collection of all benchmark instances in `lcbench` is a `scenario`.
Here, an `instance` is the concrete task of optimizing hyperparameters of a neural network on a given dataset from OpenML.

### Why should I use it?

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) provides blazingly fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks
across a large variety of problems.
Our library makes use of [ConfigSpace](https://automl.github.io/ConfigSpace/) to describe the hyperparameter space to optimize and can thus be seamlessly integrated into many existing projects e.g. [HpBandSter](https://github.com/automl/HpBandSter).

![image](https://github.com/slds-lmu/yahpo_gym/blob/main/assets/results.png?raw=true)


**Overview over problems**

|Scenario     | #HPs| #Targets| #Instances|Space      |Fidelity |
|:------------|----:|--------:|----------:|:----------|:--------|
|lcbench      |    9|        6|         35|Numeric    |epoch    |
|fcnet        |   12|        4|          4|Mixed      |epoch    |
|nb301        |   35|        2|          1|Mixed+Deps |epoch    |
|rbv2_svm     |    9|        6|         96|Mixed+Deps |frac     |
|rbv2_ranger  |   11|        6|        114|Mixed+Deps |frac     |
|rbv2_rpart   |    8|        6|        101|Mixed      |frac     |
|rbv2_glmnet  |    6|        6|         98|Mixed      |frac     |
|rbv2_xgboost |   17|        6|        109|Mixed+Deps |frac     |
|rbv2_aknn    |    9|        6|         99|Mixed      |frac     |
|rbv2_super   |   41|        6|         89|Mixed+Deps |frac     |
|iaml_ranger  |   10|       12|          4|Mixed+Deps |frac     |
|iaml_rpart   |    6|       12|          4|Numeric    |frac     |
|iaml_glmnet  |    4|       12|          4|Numeric    |frac     |
|iaml_xgboost |   15|       12|          4|Mixed+Deps |frac     |
|iaml_super   |   30|       12|          4|Mixed+Deps |frac     |

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
value = bench.config_space.sample_configuration(1).get_dictionary()
# Evaluate
print(bench.objective_function(value))
```

The `BenchmarkSet` has the following important functions and fields (with relevant args):

```
- `__init__`:
  args:
    scenario: str, "Name of the scenario"
    instance: str (optional), "A valid instance"
  "Instantiate the benchmark."

- `objective_function`, configuration: Dict, "A dictionary of HP values to evaluate"
  "Evaluate the objective function."

- `get_opt_space`:
  "Get the Opt. Space (A `ConfigSpace.ConfigSpace`)."

- `set_instance`: value: str, "A valid instance"
  "Set an instance. A list of available instances can be obtained via the `instances` field."

- `set_session`: session: str, "A onnx session"
  "Set an onnx session."
```

### Example: Tuning an instance using HPBandSter

We include a full example for optimization using **BOHB** on a yahpo_gym instance in a [jupyter notebook](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb).

### All Examples

- [General Usage](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Code Samples](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/code_sample.ipynb)
- [Tuning with HpBandSter on Yahpo Gym](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/tuning_hpandster_on_yahpo.ipynb)
- [Transfer HPO](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)
- [Paper experiments](https://github.com/slds-lmu/yahpo_exps/tree/main/paper)
