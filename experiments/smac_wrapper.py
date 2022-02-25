from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from functools import partial
import random
import pandas as pd
import numpy as np
import shutil

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.intensification.hyperband import Hyperband

def tae_runner(configuration, budget, bench, fidelity_param_id, target, factor, on_integer_scale):
    X = configuration.get_dictionary()
    if "rbv2_" in bench.config.config_id:
        X.update({"repl":10})  # manual fix required for rbv2_
    X.update({fidelity_param_id: int(round(budget)) if on_integer_scale else budget})
    y = bench.objective_function(X, logging=False, multithread=False)[0]

    return factor * float(y.get(target))

def tae_runner_max_budget(configuration, max_budget, bench, fidelity_param_id, target, factor, on_integer_scale):
    X = configuration.get_dictionary()
    if "rbv2_" in bench.config.config_id:
        X.update({"repl":10})  # manual fix required for rbv2_
    X.update({fidelity_param_id: int(round(max_budget)) if on_integer_scale else max_budget})
    y = bench.objective_function(X, logging=False, multithread=False)[0]

    return factor * float(y.get(target))

def run_smac4mf(scenario, instance, target, minimize, on_integer_scale, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    opt_space = bench.get_opt_space(instance)
    opt_space.seed(seed)
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:  # manual fix required for rbv2_
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    factor = 1 if minimize else -1
    path = "smac4mf_tmp_" + str(seed) + "_" + str(random.randrange(49152, 65535 + 1))

    sscenario = Scenario({
        "run_obj": "quality",
        "ta_run_limit": n_trials,  # limit given in terms of evaluations
        "cs": opt_space,
        "deterministic": "true",
        "output_dir": path
    })
    
    intensifier_kwargs = {"initial_budget": min_budget, "max_budget": max_budget, "eta": 3}
    
    smac4mf = SMAC4MF(
        scenario=sscenario,
        rng=np.random.RandomState(seed),
        tae_runner=partial(tae_runner, bench=bench, fidelity_param_id=fidelity_param_id, target=target, factor=factor, on_integer_scale=on_integer_scale),
        intensifier=Hyperband,
        intensifier_kwargs=intensifier_kwargs
    )
    
    smac4mf.optimize()
    results = smac4mf.get_runhistory()
    values = pd.DataFrame.from_records([[int(round(k.budget)) if on_integer_scale else k.budget, k.config_id] for k, v in results.data.items()])
    values = values.rename(columns = {0: fidelity_param_id, 1: "config_id"})
    for index, row in values.iterrows():
        X = results.get_all_configs()[int(row["config_id"] - 1)].get_dictionary()
        if "rbv2_" in scenario:
            X.update({"repl":10})  # manual fix required for rbv2_
        X.update({fidelity_param_id:row[fidelity_param_id]})
        bench.objective_function(X, logging=True, multithread=False)
    
    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis = 1)
    bench.archive = []
    shutil.rmtree(path)
    return data

def run_smac4hpo(scenario, instance, target, minimize, on_integer_scale, n_trials, seed):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    opt_space = bench.get_opt_space(instance)
    opt_space.seed(seed)
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:  # manual fix required for rbv2_
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    factor = 1 if minimize else -1
    path = "smac4mf_tmp_" + str(seed) + "_" + str(random.randrange(49152, 65535 + 1))

    sscenario = Scenario({
        "run_obj": "quality",
        "ta_run_limit": n_trials,  # limit given in terms of evaluations
        "cs": opt_space,
        "deterministic": "true",
        "output_dir": path
    })

    smac4hpo = SMAC4HPO(
        scenario=sscenario,
        rng=np.random.RandomState(seed),
        tae_runner=partial(tae_runner_max_budget, max_budget=max_budget, bench=bench, fidelity_param_id=fidelity_param_id, target=target, factor=factor, on_integer_scale=on_integer_scale)
    )
    
    smac4hpo.optimize()
    results = smac4hpo.get_runhistory()
    values = pd.DataFrame.from_records([[int(round(k.budget)) if on_integer_scale else k.budget, k.config_id] for k, v in results.data.items()])
    values = values.rename(columns = {0: fidelity_param_id, 1: "config_id"})
    for index, row in values.iterrows():
        X = results.get_all_configs()[int(row["config_id"] - 1)].get_dictionary()
        if "rbv2_" in scenario:  # manual fix required for rbv2_
            X.update({"repl":10})
            X.update({"trainsize":1})
        else:
            X.update({fidelity_param_id: max_budget})
        bench.objective_function(X, logging=True, multithread=False)
    
    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis = 1)
    bench.archive = []
    shutil.rmtree(path)
    return data

