# export PYTHONPATH="$PYTHONPATH:/home/lps/Phd/DEHB" must be on path

import time
import warnings
import numpy as np
import ConfigSpace
from typing import Dict, Union, List
warnings.filterwarnings('ignore')

from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from dehb import DEHB

bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance("3945")
opt_space = bench.get_opt_space("3945")
dimensions = len(opt_space.get_hyperparameters())
fidelity_space = bench.get_fidelity_space()
fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper

def dehb_target_function(configuration, budget, **kwargs):
    fidelity_param_id = kwargs["fidelity_param_id"]
    bench = kwargs["bench"]
    X = configuration.get_dictionary()
    X.update({fidelity_param_id: budget})
    y = bench.objective_function(X)

    result = {
        "fitness": - y.get("val_accuracy"),  # FIXME: should be changed, see #21, #20
        "cost": y.get("time"),
        "info": {
            "budget": budget
        }
    }

    return result

dehb = DEHB(
    f=dehb_target_function,
    cs=opt_space,
    dimensions=dimensions, 
    min_budget=min_budget, 
    max_budget=max_budget,
    n_workers=1,
    output_path="./temp"
)

trajectory, runtime, history = dehb.run(
    fevals=100,
    verbose=False,
    save_intermediate=False,
    # parameters expected as **kwargs in dehb_target_function are passed here
    fidelity_param_id=fidelity_param_id,
    bench=bench
)

dehb.reset()

