from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance("3945")
opt_space = bench.get_opt_space("3945")
dimensions = len(opt_space.get_hyperparameters())
fidelity_space = bench.get_fidelity_space()
fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper

def tae_runner(configuration, budget):
    X = configuration.get_dictionary()
    X.update({fidelity_param_id: int(round(budget))})
    y = bench.objective_function(X, logging=True)[0]

    return - float(y.get("val_accuracy"))

scenario = Scenario({
    'run_obj': 'quality',
    'ta_run_limit': 3 * 52,  # limit given in terms of times full fidelity
    'cs': opt_space,
    'deterministic': 'true',
    'num_workers': 1
})

intensifier_kwargs = {'initial_budget': min_budget, 'max_budget': max_budget, 'eta': 3}

smac4mf = SMAC4MF(
    scenario=scenario,
    tae_runner=tae_runner,
    intensifier_kwargs=intensifier_kwargs
)

smac4mf.optimize()
results_mf = smac4mf.get_trajectory()

def tae_runner_max_budget(configuration):
    X = configuration.get_dictionary()
    # FIXME: rounding of budget?
    X.update({fidelity_param_id: int(round(max_budget))})
    y = bench.objective_function(X, logging=True)[0]

    return - float(y.get("val_accuracy"))

scenario = Scenario({
    'run_obj': 'quality',
    'ta_run_limit': 10,  # limit given in terms of evals
    'cs': opt_space,
    'deterministic': 'true',
    'num_workers': 1
})

smac4hpo = SMAC4HPO(
    scenario=scenario,
    tae_runner=tae_runner_max_budget,
)

smac4hpo.optimize()
results_hpo = smac4hpo.get_trajectory()
