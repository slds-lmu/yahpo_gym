# YAHPO GYM (R Package)

R Interface for the YAHPO GYM python module


## Installation

The package can be installed from Github via


```r
remotes::install_github("pfistfl/yahpo_gym/yahpo_gym_r")
```

### Setup

YAHPO GYM requires a one-time setup to install the required python dependencies.
Here we install all packages into the `yahpo_gym` conda environment.

```{r, eval = FALSE}
reticulate::conda_create(
  envname = "yahpo_gym",
  packages = c("onnxruntime", "pip", "pyyaml"),
  channel = "conda-forge",
  python_version = "3.8"
)
reticulate::conda_install(envname = "yahpo_gym", packages="configspace", channel="conda-forge")
reticulate::conda_install(envname = "yahpo_gym", packages="fastdownload", channel="fastai")
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
devtools::load_all()
# library("yahpo_gym")
```

and subsequently instantiate the benchmark to obtain our objective.

```r
b = BenchmarkSet$new("rbv2_super")
obj = b$get_objective("1040")
```

and run our search procedure.

```r
p = opt("random_search")
ois = OptimInstanceMultiCrit$new(obj, terminator = trm("evals"), check_values = FALSE)
p$optimize(ois)
```

### Available Problems

We can list all available benchmark problems

```r
list_benchmarks()
```

and available instances in a `Benchmark`:

```r
b$instances
```

```r
reticulate::conda_create(
  envname = "yahpo_gym",
  packages = c("onnxruntime", "pip", "pyyaml", "pandas"),
  channel = "conda-forge",
  python_version = "3.8"
)
reticulate::conda_install(envname = "yahpo_gym",
  packages="configspace", channel="conda-forge")
reticulate::conda_install(envname = "yahpo_gym",
  packages="pandas", channel="conda-forge")
reticulate::conda_install(envname = "yahpo_gym",
  packages="fastdownload", channel="fastai")
```