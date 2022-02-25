from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import *
import random
import time
import pandas as pd
import numpy as np
import shutil

from dehb import DEHB
import ConfigSpace as CS

def dehb_target_function(configuration, budget, **kwargs):
    fidelity_param_id = kwargs["fidelity_param_id"]
    bench = kwargs["bench"]
    instance = kwargs["instance"]
    target = kwargs["target"]
    factor = kwargs["factor"]
    on_integer_scale = kwargs["on_integer_scale"]

    X = configuration.get_dictionary()
    if "rbv2_" in bench.config.config_id:
        X.update({"repl":10})  # manual fix required for rbv2_
    X.update({bench.config.instance_names: instance})
    X.update({fidelity_param_id: int(round(budget)) if on_integer_scale else budget})
    y = bench.objective_function(X, logging=True, multithread=False)[0]

    result = {
        "fitness": factor * float(y.get(target)),
        "cost": 0,
        "info": {
            "budget": int(round(budget)) if on_integer_scale else budget
        }
    }

    return result

def run_dehb(scenario, instance, target, minimize, on_integer_scale, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    opt_space = bench.get_opt_space(instance)
    opt_space_fixed = CS.ConfigurationSpace(seed=seed)
    hps = opt_space.get_hyperparameter_names()
    for hp in hps:
        if hp != bench.config.instance_names:
            opt_space_fixed.add_hyperparameter(opt_space.get_hyperparameter(hp))
    conditions = opt_space.get_conditions()
    for condition in conditions:
        opt_space_fixed.add_condition(condition)
    forbiddens = opt_space.get_forbiddens()
    for forbidden in forbiddens:
        opt_space_fixed.add_forbidden(forbidden)
    dimensions = len(opt_space_fixed.get_hyperparameters())
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:  # manual fix required for rbv2_
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    factor = 1 if minimize else -1
    path = "dehb_tmp_" + str(seed) + "_" + str(random.randrange(49152, 65535 + 1))

    dehb = DEHB(
        f=dehb_target_function,
        cs=opt_space_fixed,
        dimensions=dimensions, 
        min_budget=min_budget, 
        max_budget=max_budget,
        n_workers=1,
        output_path=path
    )

    trajectory, runtime, history = dehb.run(
        fevals=n_trials,
        verbose=False,
        save_intermediate=False,
        # parameters expected as **kwargs in dehb_target_function are passed here
        fidelity_param_id=fidelity_param_id,
        bench=bench,
        instance=instance,
        target=target,
        factor=factor,
        on_integer_scale=on_integer_scale
    )

    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis = 1)
    bench.archive = []
    dehb.reset()
    shutil.rmtree(path)
    return data

