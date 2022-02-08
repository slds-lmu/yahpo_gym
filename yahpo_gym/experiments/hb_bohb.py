from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import HyperBand as HyperBand
from hpbandster.optimizers import BOHB as BOHB

bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance("3945")
fidelity_space = bench.get_fidelity_space()
fidelity_param_id = fidelity_space.get_hyperparameter_names()[0]
min_budget = fidelity_space.get_hyperparameter(fidelity_param_id).lower
max_budget = fidelity_space.get_hyperparameter(fidelity_param_id).upper

class lcbench(Worker):

    def __init__(self, fidelity_param_id, bench, **kwargs):
        super().__init__(**kwargs)
        self.sleep_interval = 0
        self.fidelity_param_id = fidelity_param_id
        self.bench = bench

    def compute(self, config, budget, **kwargs):
        fidelity_param_id = self.fidelity_param_id
        bench = self.bench
        config.update({fidelity_param_id: int(round(budget))})
        y = bench.objective_function(config, logging=True)[0]

        result = {
            "loss": - float(y.get("val_accuracy")),
            "info": {
                "cost": float(y.get("time")),
                "budget": int(round(budget))
            }
        }
        
        return result
    
    @staticmethod
    def get_configspace():
        opt_space = bench.get_opt_space("3945")
        return(opt_space)

NS = hpns.NameServer(run_id="lcbench", host="127.0.0.1", port=None)
NS.start()

w = lcbench(
    nameserver="127.0.0.1",
    run_id ="lcbench",
    fidelity_param_id=fidelity_param_id,
    bench=bench
)
w.run(background=True)

bohb = BOHB(
    configspace=w.get_configspace(),
    run_id="lcbench",
    nameserver="127.0.0.1",
    min_budget=min_budget,
    max_budget=max_budget,
)

results_bohb = bohb.run()
bohb.shutdown()

hb = HyperBand(
    configspace=w.get_configspace(),
    run_id="lcbench",
    nameserver="127.0.0.1",
    min_budget=min_budget,
    max_budget=max_budget,
)
results_hb = hb.run()
hb.shutdown()

w.shutdown()
NS.shutdown()

