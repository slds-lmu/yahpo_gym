# YAHPO GYM

Modules for training and inference of surrogate based HPO benchmarks.

For a pre-alpha version of this project relying on the v1 surrogate models, please go [here](https://github.com/compstat-lmu/paper_2021_multi_fidelity_surrogates).

### Overview

|     | instance     | space   | n_dims | n_targets        | fidelity       | n_problems | status |
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

### YAHPO GYM

YAHPO GYM is the module for inference and allows for evaluating a HPC configuration on a given benchmark instance.

Surrogate models (ONNX files), configspaces and metadata (encoding) can be obtained [here](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

An example for evaluation and running HPO methods is given in the README of YAHPO GYM itself.

### YAHPO Train

YAHPO Train is the module for training new surrogate models.

YAHPO Train is still in a very preliminary state.

