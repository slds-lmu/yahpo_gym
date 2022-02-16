from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import *

# ax_platform == 0.1.18

from ax.core import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig

from baselines import MultiObjectiveSimpleExperiment

bench = benchmark_set.BenchmarkSet("iaml_glmnet", multithread=False)

def get_yahpo(name=None):
    metric_1 = Metric("mmce", True)
    metric_2 = Metric("nf", True)

    objective = MultiObjective([metric_1, metric_2])
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective
    )

    alpha = RangeParameter(
        name="alpha", lower=0, upper=1, parameter_type=ParameterType.FLOAT
    )
    s = RangeParameter(
        name="s", lower=0.0001, upper=100, parameter_type=ParameterType.FLOAT, log_scale=True
    )

    search_space = SearchSpace(
        parameters=[alpha, s],
    )

    fun = YahpoFunction()

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=search_space,
        eval_function=fun,
        optimization_config=optimization_config,
    )

class YahpoFunction:
    def __init__(self):
        None
    def __call__(self, x):
        seed = 0
        bench = benchmark_set.BenchmarkSet("iaml_glmnet", multithread=False)
        bench.set_instance("40981")
        opt_space = bench.get_opt_space("40981")
        opt_space.seed(seed)
        fidelity_space = bench.get_fidelity_space()
        fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
        min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
        max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper
        x.update({"task_id": "40981"})
        if not fidelity_param_id in x:
            x.update({fidelity_param_id: max_budget})
        y = bench.objective_function(x, logging=True, multithread=False)[0]
        return dict((k, (float(y[k]), 0.0)) for k in ("mmce", "nf"))

