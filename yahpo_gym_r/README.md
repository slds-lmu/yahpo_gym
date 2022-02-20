# YAHPO GYM (R)
[![Unittests](https://github.com/slds-lmu/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/slds-lmu/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://slds-lmu.github.io/yahpo_gym/) 
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (Python)](https://img.shields.io/badge/Software-Python-green)](https://github.com/slds-lmu/yahpo_gym/tree/main/yahpo_gym)

R Interface for the YAHPO GYM python module. Documentation for the python module is available via the [module handbook](https://slds-lmu.github.io/yahpo_gym/)
while the R module builds its own documentation with the package.
## Installation

The package can be installed from GitHub via


```r
remotes::install_github("slds-lmu/yahpo_gym/yahpo_gym_r")
```

### Setup

YAHPO GYM requires a one-time setup to install the required python dependencies.
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


## Usage

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
library("bbotk")
p = opt("random_search")
ois = OptimInstanceMultiCrit$new(obj, search_space = b$get_search_space(drop_fidelity_params = TRUE), terminator = trm("evals", n_evals = 10))
p$optimize(ois)
```



### or with Hyperband using (`mlr3hyperband`)

```r
library(mlr3hyperband)
obj = b$get_objective("40981", multifidelity = TRUE)
ois = OptimInstanceMultiCrit$new(obj, search_space = b$get_search_space(), terminator = trm("none"))
p = opt("hyperband")
p$optimize(ois)
```


### Available Problems

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

We can list all available benchmark problems

```r
list_benchmarks()
```

and available instances in a `Benchmark`:

```r
b$instances
```
  

## Technical Questions:

### Single-Crit Optimization

We can use `subset_codomain` to obtain a single-crit optimization instance by
specifying the target to keep:

```r
b$subset_codomain("auc")
obj = b$get_objective("40981", multifidelity = FALSE)
```

### Using yahpogym with `future`:

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

### Radian

Since `yahpogym` relies on `reticulate`, interoperability with e.g. radian does sometimes not work. 
See [here](https://github.com/randy3k/radian#i-cant-specify-python-runtime-in-reticulate) for more information.
