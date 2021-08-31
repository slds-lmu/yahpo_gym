# YAHPO-GYM

Surrogate based benchmarks for HPO problems.

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
| 8   | lcbench      | Mix     |      7 | 6:perf(5)+rt     | epoch          |         35 |        |
| 9   | nb301        | Cat+Dep |     34 | 2:perf(1)+rt     | epoch          |          1 |        |

where for **n\_targets** (\#number):

-   perf = performance measure
-   ms = model\_size
-   rt = runtime
-   pt = predicttime

### Installation

```console
pip install -e .
```

### Setup

To run a benchmark you need to obatin the ONNX model (`new_model.onnx`), ConfigSpace (`config_space.json`) and some encoding info (`encoding.json`).

You can download these [here](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

You should pertain the folder structure as on the hosting site (i.e., create a `"path-to-data"` directory, for example named `"multifidelity_data"`, containing the individual, e.g., `"lcench"`, directories).

```py
# Initialize the local config & set path for surrogates and metadata
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

### Run Inference

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

### BOHB example

```py
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench
import time
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB

bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance("3945")

class lcbench(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench = bench
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of epochs the model can use to train

        Returns:
            dictionary with mandatory fields:
                "loss" (scalar)
                "info" (dict)
        """

        config.update({"epoch": int(np.round(budget))})  # update epoch
        result = bench.objective_function(config)  # evaluate

        time.sleep(self.sleep_interval)

        return({
                    "loss": - result.get("val_accuracy"),  # we want to maximize validation accuracy
                    "info": "empty"
                })
    
    @staticmethod
    def get_configspace():
        hps = bench.config_space.get_hyperparameters()
        oml_idx = bench.config_space.get_hyperparameter_names().index("OpenML_task_id")
        hps[oml_idx] = CSH.Constant("OpenML_task_id", "3945")  # we additionally fix the instance here for BOHB
        epoch_idx = bench.config_space.get_hyperparameter_names().index("epoch")
        del hps[epoch_idx]  # drop budget parameter
        cnds = bench.config_space.get_conditions()
        fbds = bench.config_space.get_forbiddens()
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        cs.add_conditions(cnds)
        cs.add_forbidden_clauses(fbds)
        return(cs)

NS = hpns.NameServer(run_id="lcbench", host="127.0.0.1", port=None)
NS.start()

w = lcbench(sleep_interval=0, nameserver="127.0.0.1", run_id ="lcbench")
w.run(background=True)

bohb = BOHB(configspace=w.get_configspace(),
            run_id="lcbench", nameserver="127.0.0.1",
            min_budget=1, max_budget=52)

res = bohb.run(n_iterations=1)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print("Best found configuration:", id2config[incumbent]["config"])
print("A total of %i unique configurations where sampled." % len(id2config.keys()))
print("A total of %i runs where executed." % len(res.get_all_runs()))
print("Total budget corresponds to %.1f full function evaluations."%(sum([r.budget for r in res.get_all_runs()])/1))
```

