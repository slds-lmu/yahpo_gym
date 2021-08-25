# YAHPO-GYM

Surrogate based benchmarks for HPO problems


### Setup 

```py
# Initialize the local config & set save path for surrogates and metadata
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
# Sample a point from the configspace
value = bench.config_space.sample_configuration(1).get_dictionary()
# Add a budget value for the fidelity parameter(s)
value.update(epoch = 1)
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
        hps[0] = CSH.Constant("OpenML_task_id", "3945")  # we additionally fix the instance here for BOHB
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

