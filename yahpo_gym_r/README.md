# YAHPO GYM (R Package)

R Interface for the YAHPO GYM python module


## Installation

The package can be installed from Github via



### Setup

YAHPO GYM requires a one-time setup to install the required python dependencies.
Here we install all packages into the `yahpo_gym` conda environment.

```r
reticulate::conda_create(
  envname = "yahpo_gym",
  packages = c("onnxruntime"),
  channel = "conda-forge"
)
reticulate::conda_install(envname = "yahpo_gym", packages="fastdownload", pip=TRUE)
```

Now we can instantiate a local config that sets up the path files are downloaded to:

```r
reticulate::use_condaenv("yahpo_gym", required=TRUE)
library("yahpo_gym")
init_local_config(path = "~/LRZ Sync+Share/multifidelity_data")
```


## Usage

We first load the package and the required conda environment:

```r
reticulate::use_condaenv("yahpo_gym", required=TRUE)
library("yahpo_gym")
```

and subsequently instantiate the benchmark to obtain our objective.

```r
b = BenchmarkSet$new("lcbench")
obj = b$get_objective()
```

and run our search procedure.

```r
p = opt("random_search")
ois = OptimInstanceMultiCrit$new(obj, terminator = trm("evals"), check_values = FALSE)
p$optimize(ois)
```

### Available Problems

```r
list_benchmarks()
```