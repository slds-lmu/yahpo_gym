# YAHPO Gym (python)
[![Unittests](https://github.com/pfistfl/yahpo_gym/actions/workflows/unittests_gym_py.yml/badge.svg?branch=main)](https://github.com/pfistfl/yahpo_gym/actions)
[![Module Handbook](https://img.shields.io/badge/Website-Documentation-blue)](https://pfistfl.github.io/yahpo_gym/) 
[![Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/abs/2109.03670)
[![Software (R)](https://img.shields.io/badge/Software-R-green)](https://github.com/pfistfl/yahpo_gym/tree/main/yahpo_gym)


### What is YAHPO GYM? 

---

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) is a collection of interesting problem sets for benchmark hyperparameter optimization / black-box optimization methods described in [our paper](https://arxiv.org/abs/2109.03670).

YAHPO GYM consists of several `scenarios`, e.g. the collection of all benchmark instances in `lcbench` is a `scenario`.
Here, an `instance` is the concrete task of optimizing hyperparameters of a neural network on a given dataset from OpenML.

### Why should I use it?

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) provides blazingly fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks 
across a large variety of problems.
Our library makes use of [ConfigSpace](https://automl.github.io/ConfigSpace/) to describe the hyperparameter space to optimize and can thus be seamlessly integrated into many existing projects e.g. [HpBandSter](https://github.com/automl/HpBandSter).

![image](https://github.com/pfistfl/yahpo_gym/blob/main/assets/results.png?raw=true)


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


The **full, up-to-date overview** can be obtained from the [Documentation](https://pfistfl.github.io/yahpo_gym/scenarios.html).

The **full, up-to-date overview** can be obtained from the [Documentation](https://pfistfl.github.io/yahpo_gym/scenarios.html).

---

### Installation

```console
pip install "git+https://github.com/pfistfl/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
```

### Setup

To run a benchmark you need to obatin the ONNX model (`new_model.onnx`), [ConfigSpace](https://automl.github.io/ConfigSpace/) (`config_space.json`) and some encoding info (`encoding.json`).

You can download these [here (Github)](https://github.com/pfistfl/yahpo_data) or [here (Synchshare)](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

You should pertain the folder structure as on the hosting site (i.e., create a `"path-to-data"` directory, for example named `"multifidelity_data"`, containing the individual, e.g., `"lcench"`, directories).

```py
# Initialize the local config & set path for surrogates and metadata
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

### Usage

This example showcases the simplicity of YAHPO GYM's API. 
A longer introduction is given in the accompanying [jupyter notebook](https://github.com/pfistfl/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb).


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

The `BenchmarkSet` has the following important functions and fields (with relevant args):

```
- `__init__`: 
  args:
    config_id: str, "Name of the scenario"
    instance: str (optional), "A valid instance"
  "Instantiate the benchmark."

- `objective_function`, configuration: Dict, "A dictionary of HP values to evaluate"
  "Evaluate the objective function."

- `get_opt_space`:
  "Get the Opt. Space (A `ConfigSpace.ConfigSpace`)."

- `set_instance`: value: str, "A valid instance"
  "Set an instance. A list of available instances can be obtained via the `instances` field."

- `set_session`: session: str, "A onnx session"
  "Set an onnx session."
```

### BOHB example

```py
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench
import time
import numpy as np
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB

bench = benchmark_set.BenchmarkSet("lcbench", instance = "3945")

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
        result = bench.objective_function(config)[0]  # evaluate

        time.sleep(self.sleep_interval)

        return({
                    "loss": - float(result.get("val_accuracy")),  # we want to maximize validation accuracy
                    "info": "empty"
                })
    
    @staticmethod
    def get_configspace():
        # sets OpenML_task_id constant to "3945" and removes the epoch fidelity parameter
        cs = bench.get_opt_space(drop_fidelity_params = True)
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

