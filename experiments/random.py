from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import *

import random
import pandas as pd
import numpy as np

def run_random(scenario, instance, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario)
    bench.set_instance(instance)
    opt_space = bench.get_opt_space(instance)
    opt_space.seed(seed)
    fidelity_space = bench.get_fidelity_space()
    fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper

    for i in range(n_trials):
        x = opt_space.sample_configuration(1).get_dictionary()
        x.update({fidelity_param_id: max_budget})
        y = bench.objective_function(x, logging=True)[0]

    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis = 1)
    bench.archive = []
    return data

