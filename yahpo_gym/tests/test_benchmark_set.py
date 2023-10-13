import random
import time
import pytest

import ConfigSpace

import yahpo_gym
from yahpo_gym.benchmark_set import BenchmarkSet
from yahpo_gym.benchmarks import *


# Abstract test function
def test_benchmarkset_abstract(
    key: str = None, test_instance: str = None, fidelity_config: dict = {}
):
    if key is None or test_instance is None:
        return None

    b = BenchmarkSet(key, instance=test_instance)

    # Instance Fields
    assert type(b.config) == yahpo_gym.configuration.Configuration
    assert type(b.encoding) == dict
    assert type(b.config_space) == ConfigSpace.configuration_space.ConfigurationSpace
    assert b.active_session == True
    assert type(b.constants) == dict
    assert b.session is not None
    assert len(b.archive) == 0
    assert b.check == True

    # Properties
    assert (type(b.instances) == list) & (len(b.instances) > 0)
    b.properties

    if b.instance is not None:
        assert b.instance == test_instance

    # Setters
    b.set_constant(b.config.instance_names, test_instance)
    if b.config.instance_names is not None:
        assert b.constants == {b.config.instance_names: test_instance}
    with pytest.raises(Exception) as info:
        b.set_constant("foo", "bar")

    # Getters
    assert (
        type(b.get_fidelity_space())
        == ConfigSpace.configuration_space.ConfigurationSpace
    )

    optspace = b.get_opt_space(drop_fidelity_params=False)
    assert type(optspace) == ConfigSpace.configuration_space.ConfigurationSpace
    assert len(optspace.get_hyperparameter_names()) == len(b.config.hp_names)

    optspace = b.get_opt_space(drop_fidelity_params=True)
    assert type(optspace) == ConfigSpace.configuration_space.ConfigurationSpace
    assert len(optspace.get_hyperparameter_names()) == len(b.config.hp_names) - len(
        b.get_fidelity_space().get_hyperparameter_names()
    )

    xs = optspace.sample_configuration()
    with pytest.raises(Exception) as info:
        out = b.objective_function(xs)

    xs = optspace.sample_configuration()
    xs = xs.get_dictionary()
    xs.update(fidelity_config)
    out = b.objective_function(xs.copy())[0]
    assert type(out) == dict
    assert [k for k in out.keys()] == b.config.y_names

    # Invariant to dict order
    tmp = list(xs)
    random.shuffle(tmp)
    xs2 = {hp: xs.get(hp) for hp in tmp}
    assert b.objective_function(xs2)[0] == out

    # timed predict
    b.quant = max(0, 0.5 / out[b.config.runtime_name]) + 0.000001
    start = time.time()
    out = b.objective_function_timed(xs.copy())[0]
    end = time.time()
    assert (end - start) < 0.6

    assert type(out) == dict
    assert [k for k in out.keys()] == b.config.y_names

    # Unexported functions
    out = b._eval_random()
    assert type(out) == dict
    assert [k for k in out.keys()] == b.config.y_names

    # Statistics
    statdf = b.target_stats
    assert len(statdf.columns) == 5
    assert len(statdf) > 0

    return b


def test_benchmarkset_lcbench():
    fidelity_config = {"epoch": 50}
    test_instance = "3945"
    b = test_benchmarkset_abstract("lcbench", test_instance, fidelity_config)


def test_benchmarkset_nb301():
    fidelity_config = {"epoch": 50}
    test_instance = "CIFAR10"
    key = "nb301"
    b = test_benchmarkset_abstract("nb301", test_instance, fidelity_config)


def test_benchmarkset_rbv2_super():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_super", test_instance, fidelity_config)


def test_benchmarkset_rbv2_svm():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_svm", test_instance, fidelity_config)


def test_benchmarkset_rbv2_glmnet():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_glmnet", test_instance, fidelity_config)


def test_benchmarkset_rbv2_ranger():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_ranger", test_instance, fidelity_config)


def test_benchmarkset_rbv2_aknn():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_aknn", test_instance, fidelity_config)


def test_benchmarkset_rbv2_rpart():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_rpart", test_instance, fidelity_config)


def test_benchmarkset_rbv2_xgboost():
    fidelity_config = {"trainsize": 0.5, "repl": 9}
    test_instance = "15"
    b = test_benchmarkset_abstract("rbv2_xgboost", test_instance, fidelity_config)


def test_benchmarkset_iaml_rpart():
    fidelity_config = {"trainsize": 0.5}
    test_instance = "40981"
    b = test_benchmarkset_abstract("iaml_rpart", test_instance, fidelity_config)


def test_benchmarkset_iaml_ranger():
    fidelity_config = {"trainsize": 0.5}
    test_instance = "40981"
    b = test_benchmarkset_abstract("iaml_ranger", test_instance, fidelity_config)


def test_benchmarkset_iaml_glmnet():
    fidelity_config = {"trainsize": 0.5}
    test_instance = "40981"
    b = test_benchmarkset_abstract("iaml_glmnet", test_instance, fidelity_config)


def test_benchmarkset_iaml_xgboost():
    fidelity_config = {"trainsize": 0.5}
    test_instance = "40981"
    b = test_benchmarkset_abstract("iaml_xgboost", test_instance, fidelity_config)


def test_benchmarkset_iaml_super():
    fidelity_config = {"trainsize": 0.5}
    test_instance = "40981"
    b = test_benchmarkset_abstract("iaml_super", test_instance, fidelity_config)
