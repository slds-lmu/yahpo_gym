import pytest
import time
import ConfigSpace

import yahpo_gym
from yahpo_gym.benchmark_set import BenchmarkSet
from yahpo_gym.benchmarks import *


def test_objective_timed_lcbench():
    b = BenchmarkSet("lcbench", download = False)
    fidelity_config = {"epoch" : 50}
    test_instance = "3945"

    optspace =  b.get_opt_space(instance = test_instance)
    assert type(optspace) == ConfigSpace.configuration_space.ConfigurationSpace

    for i in range(3):
        xs = optspace.sample_configuration()
        with pytest.raises(Exception) as info:
            out = b.objective_function(xs)

        xs = optspace.sample_configuration()
        xs = xs.get_dictionary()
        xs.update(fidelity_config)
        
        # Learn quantization:
        assert b.quant is None
        out = b.objective_function_timed(xs.copy())
        assert b.quant is not None
        assert b.quant > 1e-5
        assert b.quant < 1
        
        # Predicted runtime:
        rth = (out[b.config.runtime_name])
        assert rth > 0
        
        # Runtime:
        start_time = time.time()
        out = b.objective_function_timed(xs.copy())
        tt = time.time() - start_time  

        # Sped up runtime
        rtt = out[b.config.runtime_name] * b.quant
        assert rtt > 0
        # Sped up runtime and actual runtime match
        assert rtt / tt > 0.85
        
        b.quant = None
        

test_objective_timed_lcbench()