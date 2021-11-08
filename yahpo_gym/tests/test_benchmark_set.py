import pytest
import yahpo_gym
import ConfigSpace
import random
from yahpo_gym.benchmark_set import BenchmarkSet
from yahpo_gym.benchmarks import *

# Abstract test function
def test_benchmarkset_abstract(key, test_instance, fidelity_config):

  b = BenchmarkSet(key)

  # Instance Fields
  assert type(b.config) == yahpo_gym.configuration.Configuration
  assert type(b.encoding) == dict
  assert type(b.config_space) == ConfigSpace.configuration_space.ConfigurationSpace
  assert b.active_session == False
  assert type(b.constants) == dict
  assert b.session is None
  assert len(b.archive) == 0
  assert b.check == True

  # Properties
  assert (type(b.instances) == list) & (len(b.instances) > 0)

  # Setters
  b.set_constant(b.config.instance_names, test_instance)
  assert b.constants == {b.config.instance_names : test_instance}
  with pytest.raises(Exception) as info:
    b.set_constant("foo", "bar")

  # Getters
  assert type(b.get_fidelity_space()) == ConfigSpace.configuration_space.ConfigurationSpace

  optspace =  b.get_opt_space(instance = test_instance, drop_fidelity_params = False)
  assert type(optspace) == ConfigSpace.configuration_space.ConfigurationSpace
  assert len(optspace)  == len(b.config.hp_names)

  optspace =  b.get_opt_space(instance = test_instance)
  assert type(optspace) == ConfigSpace.configuration_space.ConfigurationSpace
  assert len(optspace)  == len(b.config.hp_names) - 1

  xs = optspace.sample_configuration()
  with pytest.raises(Exception) as info:
    out = b.objective_function(xs)

  xs = xs.get_dictionary()
  xs.update(fidelity_config)
  out = b.objective_function(xs)
  assert type(out) == dict
  assert [k for k in out.keys()] == b.config.y_names

  # Invariant to dict order
  tmp = list(xs)
  random.shuffle(tmp)
  xs = {x:xs.get(x) for x in tmp}
  assert b.objective_function(xs) == out

  # timed predict
  out = b.objective_function_timed(xs)
  assert type(out) == dict
  assert [k for k in out.keys()] == b.config.y_names

  # Unexported functions
  out = b._eval_random()
  assert type(out) == dict
  assert [k for k in out.keys()] == b.config.y_names

  return b


def test_benchmarkset_lcbench():
  fidelity_config = {"epoch" : 50}
  test_instance = "3945"
  b = test_benchmarkset_abstract("lcbench", test_instance, fidelity_config)

def test_benchmarkset_rbv2_svm():
  fidelity_config = {"trainsize" : .5}
  test_instance = "15"
  # b = test_benchmarkset_abstract("rbv2_svm", test_instance, fidelity_config)

