# YAHPO GYM

### What is YAHPO GYM? 

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) is a collection of interesting problem sets for benchmark hyperparameter optimization / black-box optimization methods described in [this paper](https://arxiv.org/abs/2109.03670).
The underlying software with additional documentation and background can be found [here](https://github.com/pfistfl/yahpo_gym/tree/main/yahpo_gym).
See the [module Documentation](https://pfistfl.github.io/yahpo_gym/) for more info.


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

![image](https://github.com/pfistfl/yahpo_gym/blob/main/assets/results.png?raw=true)

<br><br>
---

**Overview over problems**

|     | scenario     | space   | n_dims | n_targets        | fidelity       | n_problems | status |
|:----|:-------------|:--------|-------:|:-----------------|:---------------|-----------:|:-------|
| 1   | rbv2_super   | Mix+Dep |     38 | 6:perf(4)+rt+pt  | trainsize+repl |         89 |        |
| 2   | rbv2_svm     | Mix+Dep |      6 | 6:perf(4)+rt+pt  | trainsize+repl |         96 |        |
| 3   | rbv2_rpart   | Mix     |      5 | 6:perf(4)+rt+pt  | trainsize+repl |        101 |        |
| 4   | rbv2_aknn    | Mix     |      6 | 6:perf(4)+rt+pt  | trainsize+repl |         99 |        |
| 5   | rbv2_glmnet  | Mix     |      3 | 6:perf(4)+rt+pt  | trainsize+repl |         98 |        |
| 6   | rbv2_ranger  | Mix+Dep |      8 | 6:perf(4)+rt+pt  | trainsize+repl |        114 |        |
| 7   | rbv2_xgboost | Mix+Dep |     14 | 6:perf(4)+rt+pt  | trainsize+repl |        109 |        |
| 8   | lcbench      | Numeric |      7 | 6:perf(5)+rt     | epoch          |         35 |        |
| 9   | nb301        | Cat+Dep |     34 | 2:perf(1)+rt     | epoch          |          1 |        |

where for **n\_targets** (\#number):

-   perf = performance measure
-   ms = model\_size
-   rt = runtime
-   pt = predicttime

<br><br>
### What does this repository contain?

This repository contains two modules: `yahpo_gym` and `yahpo_train`. 
While we mainly focus on `yahpo_gym`, as it is provides an interface to the benchmark described in our [paper](https://arxiv.org/abs/2109.03670),
we also provide the full reproducible codebase used to generate the underlying surrogate neural networks in `yahpo_train`.

#### YAHPO GYM

YAHPO GYM is the module for inference and allows for evaluating a HPC configuration on a given benchmark instance.

Surrogate models (ONNX files), configspaces and metadata (encoding) can be obtained [here](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

An example for evaluation and running HPO methods is given in the README of the [YAHPO GYM Module](https://github.com/pfistfl/yahpo_gym/tree/main/yahpo_gym).

A quick introduction is given in the accompanying [jupyter notebook](https://github.com/pfistfl/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb)

#### YAHPO Train

YAHPO Train is the module for training new surrogate models.

YAHPO Train is still in a preliminary state but can already be used to reproduce and refit models introduced in our [paper](https://arxiv.org/abs/2109.03670).
