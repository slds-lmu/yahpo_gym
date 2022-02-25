from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import *
import random
import time
import pandas as pd
import numpy as np

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import HyperBand as HyperBand

class worker(Worker):

    def __init__(self, fidelity_param_id, bench, instance, target, factor, on_integer_scale, **kwargs):
        super().__init__(**kwargs)
        self.sleep_interval = 1e-2  # minor sleeping so that logging cannot be messed up
        self.fidelity_param_id = fidelity_param_id
        self.bench = bench
        self.instance = instance
        self.target = target
        self.factor = factor
        self.on_integer_scale = on_integer_scale


    def compute(self, config, budget, **kwargs):
        fidelity_param_id = self.fidelity_param_id
        bench = self.bench
        target = self.target
        on_integer_scale = self.on_integer_scale
        if "rbv2_" in bench.config.config_id:
            config.update({"repl":10})  # manual fix required for rbv2_
        config.update({fidelity_param_id: int(round(budget)) if self.on_integer_scale else budget})
        y = bench.objective_function(config, logging=True, multithread=False)[0]

        result = {
            "loss": self.factor * float(y.get(self.target)),
            "info": {
                "budget": int(round(budget)) if self.on_integer_scale else budget
            }
        }
        time.sleep(self.sleep_interval)
    
        return result
    
    @staticmethod
    def get_configspace():
        opt_space = self.bench.get_opt_space(self.instance)
        return(opt_space)

def run_bohb(scenario, instance, target, minimize, on_integer_scale, n_iterations, seed):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:  # manual fix required for rbv2_
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    factor = 1 if minimize else -1
    randport = random.randrange(49152, 65535 + 1)

    configspace = bench.get_opt_space(instance)
    configspace.seed(seed)
    
    NS = hpns.NameServer(run_id=scenario, host="127.0.0.1", port=randport)
    NS.start()
    
    w = worker(
        nameserver="127.0.0.1",
        nameserver_port=randport,
        run_id =scenario,
        fidelity_param_id=fidelity_param_id,
        bench=bench,
        instance=instance,
        target=target,
        factor=factor,
        on_integer_scale=on_integer_scale
    )
    w.run(background=True)
    
    bohb = BOHB(
        configspace=configspace,
        eta = 3,
        run_id=scenario,
        nameserver="127.0.0.1",
        nameserver_port=randport,
        min_budget=min_budget,
        max_budget=max_budget,
    )
    
    results_bohb = bohb.run(n_iterations=n_iterations)
    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis = 1)
    bench.archive = []
    bohb.shutdown()
    w.shutdown()
    NS.shutdown()
    return data

def run_hb(scenario, instance, target, minimize, on_integer_scale, n_iterations, seed):
    random.seed(seed)
    np.random.seed(seed)

    bench = benchmark_set.BenchmarkSet(scenario, instance=instance, multithread=False)
    fidelity_space = bench.get_fidelity_space()
    if "rbv2_" in scenario:  # manual fix required for rbv2_
        fidelity_param_id = "trainsize"
    else:
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
    min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
    max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
    factor = 1 if minimize else -1
    randport = random.randrange(49152, 65535 + 1)

    configspace = bench.get_opt_space(instance)
    configspace.seed(seed)
    
    NS = hpns.NameServer(run_id=scenario, host="127.0.0.1", port=randport)
    NS.start()
    
    w = worker(
        nameserver="127.0.0.1",
        nameserver_port=randport,
        run_id =scenario,
        fidelity_param_id=fidelity_param_id,
        bench=bench,
        instance=instance,
        target=target,
        factor=factor,
        on_integer_scale=on_integer_scale
    )
    w.run(background=True)

    hb = HyperBand(
        configspace=configspace,
        eta = 3,
        run_id=scenario,
        nameserver="127.0.0.1",
        nameserver_port=randport,
        min_budget=min_budget,
        max_budget=max_budget
    )
    
    results_hb = hb.run(n_iterations=n_iterations)
    time = pd.DataFrame.from_dict([x.get("time") for x in bench.archive])
    X = pd.DataFrame.from_dict([x.get("x") for x in bench.archive])
    Y = pd.DataFrame.from_dict([x.get("y") for x in bench.archive])
    data = pd.concat([time, X, Y], axis = 1)
    bench.archive = []
    hb.shutdown()
    w.shutdown()
    NS.shutdown()
    return data

