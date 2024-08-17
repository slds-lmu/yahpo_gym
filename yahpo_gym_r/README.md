# YAHPO Gym (R)
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/) 
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (Python)](https://img.shields.io/badge/Software-Python-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym)

R Interface for the YAHPO Gym python module. Documentation for the python module is available via the [module handbook](https://slds-lmu.github.io/yahpo_gym/) while the R module builds its own documentation with the package.

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

The package can be installed from GitHub via

```r
remotes::install_github("slds-lmu/yahpo_gym/yahpo_gym_r")
```

### Setup

YAHPO Gym requires a one-time setup to install the required python dependencies.
Here we install all packages into the `yahpo_gym` conda environment.

```r
reticulate::conda_create(
  envname = "yahpo_gym",
  packages = c("onnxruntime", "pip", "pyyaml", "pandas"),
  channel = "conda-forge",
  python_version = "3.8"
)
reticulate::conda_install(envname = "yahpo_gym", packages="configspace", channel="conda-forge")
reticulate::conda_install(envname = "yahpo_gym", pip=TRUE,
  packages="'git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym'")
```

Now we can instantiate a local config that sets up the path files are installed to:

```r
reticulate::use_condaenv("yahpo_gym", required=TRUE)
library("yahpogym")
init_local_config(data_path = "~/multifidelity_data")
```
you can download the multifidelity data using 

```sh
git clone https://github.com/slds-lmu/yahpo_data.git
```

or from this URL: (https://github.com/slds-lmu/yahpo_data.git).

### Usage

We first load the package and the required conda environment:

```r
reticulate::use_condaenv("yahpo_gym", required=TRUE)
library("yahpogym")
```

and subsequently instantiate the benchmark (random search, full fidelity) to obtain our objective.

```r
b = BenchmarkSet$new("iaml_glmnet")
obj = b$get_objective("40981", multifidelity = FALSE)
```

and run our search procedure.

```r
library(bbotk)
p = opt("random_search")
ois = OptimInstanceBatchMultiCrit$new(obj, search_space = b$get_search_space(drop_fidelity_params = TRUE), terminator = trm("evals", n_evals = 10))
p$optimize(ois)
```

Note that specifying `multifidelity = FALSE` in the `$get_objective()` method of a `BenchmarkSet` always also requires
to specify `drop_fidelity_params = TRUE` when getting the search space via `$get_search_space()`.

#### or with Hyperband using (`mlr3hyperband`)

```r
library(mlr3hyperband)
obj = b$get_objective("40981", multifidelity = TRUE)
ois = OptimInstanceBatchMultiCrit$new(obj, search_space = b$get_search_space(), terminator = trm("none"))
p = opt("hyperband")
p$optimize(ois)
```

We can list all available benchmark problems:

```r
str(list_benchmarks())
```

and available instances in a `Benchmark`:

```r
b$instances
```

### A note on OpenML IDs

Currently, the `rbv2_*`, `lcbench`, and `iaml_*` scenarios contain instances based on OpenML datasets.
For `rbv2_*` and `iaml_*` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the `lcbench` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>.

### Technical Questions:

#### Single-Crit Optimization

We can use `subset_codomain` to obtain a single-crit optimization instance by specifying the target to keep:

```r
b$subset_codomain("auc")
obj = b$get_objective("40981", multifidelity = FALSE)
```

#### Using yahpogym with `future`

Parallelization with `future` and `reticulate` does not always work out of the box.
The following configurations allow to use `yahpogym` together with `future`.

1. If `yahpogym` requires a conda env / virtual env set up the `.Renvirion` file by adding 
  `RETICULATE_PYTHON=path_to_conda_python_bin`. This path can be obtained through `reticulate::py_discover_config()`.

2. Silence `future` warnings using `options(future.globals.onReference = "string")`.
  Note: `future`s check will still find unresolved references, but `yahpogym` constructs those on the child process via active bindings.

3. Run the evaluation using `future`:
  ```r
    b = BenchmarkSet$new("lcbench")
    objective = b$get_objective("3945", check_values = FALSE)

    xdt = generate_design_random(b$get_search_space(), 1)$data
    xss_trafoed = transform_xdt_to_xss(xdt, b$get_search_space())

    future::plan("multisession")
    promise = future::future(objective$eval_many(xss_trafoed), packages = "yahpogym", seed = NULL)
    future::value(promise)
  ```

#### Radian

Since `yahpogym` relies on `reticulate`, interoperability with e.g. `radian` does sometimes not work. 
See [here](https://github.com/randy3k/radian#i-cant-specify-python-runtime-in-reticulate) for more information.

### Citation

If you use YAHPO Gym, please cite the following paper:

- Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M., & Bischl, B. (2022). YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. In International Conference on Automated Machine Learning.

Moreover, certain `scenarios` built upon previous work, e.g., the `lcbench` scenario uses data from:

- Zimmer, L., Lindauer, M., & Hutter, F. (2021). Auto-Pytorch: Multi-Fidelity Metalearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(9), 3079-3090.
- Zimmer, L. (2020). data_2k_lw.zip. figshare. Dataset. https://doi.org/10.6084/m9.figshare.11662422.v1, Apache License, Version 2.0.

**Please make sure to always also cite the original data sources as YAHPO Gym would not have been possible without them!**

Original data sources of a scenario that should also be cited are provided via the `"citation"` key within the `config` dictionary of a scenario, which can be accessed from a `BenchmarkSet` via the following:

```r
reticulate::use_condaenv("yahpo_gym", required=TRUE)
library("yahpogym")
b = BenchmarkSet$new("lcbench")
b$py_instance$config$config$citation
```
