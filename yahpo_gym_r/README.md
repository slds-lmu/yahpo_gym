# YAHPO GYM (R Package)

R Interface for the YAHPO GYM python module


## Installation

The package can be installed from Github via



Get the objective
```r
library("yahpo_gym")
b = BenchmarkSet$new("lcbench")
obj = b$get_objective()
```

and run random_search!
```r
p = opt("random_search")
ois = OptimInstanceMultiCrit$new(obj, terminator = trm("evals"), check_values = FALSE)
p$optimize(ois)
```