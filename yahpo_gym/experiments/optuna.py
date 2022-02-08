from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

import ConfigSpace as CS
import optuna
from ConfigSpace import ConfigurationSpace
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import Trial

from functools import partial
import numpy as np

bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance("3945")
opt_space = bench.get_opt_space("3945")
fidelity_space = bench.get_fidelity_space()
fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper

def sample_config_from_optuna(trial, cs):

    config = {}
    for hp_name in cs:
        hp = cs.get_hyperparameter(hp_name)

        if isinstance(hp, CS.UniformFloatHyperparameter):
            value = float(trial.suggest_float(name=hp_name, low=hp.lower, high=hp.upper, log=hp.log))

        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            value = int(trial.suggest_int(name=hp_name, low=hp.lower, high=hp.upper, log=hp.log))

        elif isinstance(hp, CS.CategoricalHyperparameter):
            hp_type = type(hp.default_value)
            value = hp_type(trial.suggest_categorical(name=hp_name, choices=hp.choices))

        elif isinstance(hp, CS.OrdinalHyperparameter):
            num_vars = len(hp.sequence)
            index = trial.suggest_int(hp_name, low=0, high=num_vars - 1, log=False)
            hp_type = type(hp.default_value)
            value = hp.sequence[index]
            value = hp_type(value)

        elif isinstance(hp, CS.Constant):
            value = hp.value

        else:
            raise ValueError(f'Please implement the support for hps of type {type(hp)}')

        config[hp.name] = value
    return config

def objective_mf(trial, bench, opt_space, fidelity_param_id, valid_budgets):
    X = sample_config_from_optuna(trial, opt_space)

    # FIXME: this follows sh brackets
    # we could also evaluate all budgets in a seq from min_budget to max_budget 
    for budget in valid_budgets:
        X.update({fidelity_param_id: budget})
        y = bench.objective_function(X, logging=True)[0]
        trial.report(- float(y.get("val_accuracy")), step=budget)

        if trial.should_prune():
                raise optuna.TrialPruned()

    return - float(y.get("val_accuracy"))

def precompute_sh_iters(min_budget, max_budget, eta):
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    return max_SH_iter

def precompute_budgets(max_budget, eta, max_SH_iter, on_integer_scale=False):
    s0 = -np.linspace(start=max_SH_iter - 1, stop=0, num=max_SH_iter)
    budgets = max_budget * np.power(eta, s0)
    if on_integer_scale:
        budgets = budgets.round().astype(int)
    return budgets

# TPEsampler with median pruning checked at sh brackets below
study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=MedianPruner())
reduction_factor = 3
sh_iters = precompute_sh_iters(min_budget, max_budget, reduction_factor)
valid_budgets = precompute_budgets(max_budget, reduction_factor, sh_iters, on_integer_scale=True)
study.optimize(
    func=partial(
        objective_mf,
        bench=bench,
        opt_space=opt_space,
        fidelity_param_id=fidelity_param_id,
        valid_budgets=valid_budgets
    ),
    n_trials=100
)

