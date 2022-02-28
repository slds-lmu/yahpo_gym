# YAHPO GYM
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Docs](https://github.com/slds-lmu/yahpo_gym/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/) 
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (Python)](https://img.shields.io/badge/Software-Python-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym)
[![Software (R)](https://img.shields.io/badge/Software-R-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym)

### What is YAHPO GYM? 

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) is a collection of interesting problem sets for benchmark hyperparameter optimization / black-box optimization methods described in [this paper](https://arxiv.org/abs/2109.03670).
The underlying software with additional documentation and background can be found [here](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym).
See the [module Documentation](https://slds-lmu.github.io/yahpo_gym/) for more info.

- **Problem Variety**: Optimization problems in YAHPO Gym stem from diverse Hyperparameter Optimization scenarios on tabular as well as image data
- **Multi-Fidelity**: Allows for simulating low-fidelity approximations to the real target values to simulate multi-fidelity HPO.
- **Multi-Objective**: Benchmarks usually contain multiple objectives: performance metrics, runtime and memory consumption allowing for multi-objective and ressource aware HPO.

YAHPO Gym distinguishes between `scenarios` and `instances`.
A `scenario` is a collection of `instances` that share the same hyperparameter space. In practice, a `scenario` usually consists of a single algorithm fitted on a variety of datasets (= `instances`).

This repository contains three modules/packages:

- `yahpo_gym` (python): The core package allowing for  inference on the surrogates
- `yahpo_train` (python): Module for training surrogate models used in `yahpo_gym`.
- `yahpo_gym_r`(R): An R wrapper for  `yahpo gym`.


### Why should I use it?

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) provides blazingly fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks 
across a large variety of problems.

![image](https://github.com/slds-lmu/yahpo_gym/blob/main/assets/results.png?raw=true)

<br><br>
---

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


<br><br>
### What does this repository contain?

This repository contains two modules: `yahpo_gym` and `yahpo_train`. 
While we mainly focus on `yahpo_gym`, as it is provides an interface to the benchmark described in our [paper](https://arxiv.org/abs/2109.03670),
we also provide the full reproducible codebase used to generate the underlying surrogate neural networks in `yahpo_train`.

#### YAHPO GYM

YAHPO GYM is the module for inference and allows for evaluating a HPC configuration on a given benchmark instance.

Surrogate models (ONNX files), configspaces and metadata (encoding) can be obtained [here](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

An example for evaluation and running HPO methods is given in the README of the [YAHPO GYM Module](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym).

A quick introduction is given in the accompanying [jupyter notebook](https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)

#### YAHPO Train

YAHPO Train is the module for training new surrogate models.

YAHPO Train is still in a preliminary state but can already be used to reproduce and refit models introduced in our [paper](https://arxiv.org/abs/2109.03670).


#### Roadmap

We want to add several features to **yahpo gym** in future versions:

- **Asynchronous Evaluation**
  We would like to allow for faster-than-realtime asynchronous evaluation in future versions. This is currently available as an experimental feature via `objective_function_timed`, but requires additional (experimental) evaluation for release. 
- **Noisy Surrogate Models**
  We would like to allow for surrogates that more closely reflect the underlying (noisy) nature of real HPO experiments. Currently, noisy evaluation are available using `noisy = True` during instantiation, but this feature is considered experimental and
  requires additional evaluation for release.
- **Integration with HPO-Bench**
  HPO-Bench is a robust and mature library for benchmarking HPO Problems. Due to similarity in structure and scope, it would make sense to integrate YAHPO Gym with HPO, extending the number of scenarios available in HPO-Bench. 
- **Additional Scenarios**
  We are always happy to include additional (interesting) scenarios. If you know of (or want to add) an additional scenario, get in touch!

  We welcome input, discussion or additions by the broader community. Get in touch via issues or emails if you have questions, comments or would like to collaborate!

#### Related Software:

- [rbv2](https://github.com/slds-lmu/rbv2) (R-Package) can be used to reproduce runs from all `rbv2_***` in a real setting.
- [HPOBench](https://github.com/automl/HPOBench/tree/master/hpobench): can be used to reproduce several other scenarios in a real setting. Furthermroe, we soon hope to integrate our surrogates with **HPOBench** in order to provide a single, common API.
