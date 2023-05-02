import pytest
import os


def test_config():
    import yahpo_gym.configuration as cfg

    _test_dict = {
        "config_id": "TEST_BENCHMARK",
        "y_names": ["valid_loss", "runtime", "n_params"],
        "y_minimize": [True, False, True, False, True, False],
        "cont_names": ["epoch", "batch_size", "dropout_1"],
        "cat_names": ["task", "lr_schedule"],
        "instance_names": "task",
        "fidelity_params": ["epoch", "replication"],
        "runtime_name": "runtime",
    }

    cc = cfg.Configuration(_test_dict)

    assert cc.config_id == "TEST_BENCHMARK"
    assert cc.y_names == ["valid_loss", "runtime", "n_params"]
    assert cc.cont_names == ["epoch", "batch_size", "dropout_1"]
    assert cc.cat_names == ["task", "lr_schedule"]
    assert cc.instance_names == "task"
    assert cc.fidelity_params == ["epoch", "replication"]
    assert cc.runtime_name == "runtime"

    # properties
    assert cc.hp_names == (
        ["task", "lr_schedule"] + ["epoch", "batch_size", "dropout_1"]
    )


def test_print_cfg_table():
    from yahpo_gym.configuration import cfg

    print(cfg())


def test_local_config():
    tmp_local_config_path = os.environ.get("YAHPO_LOCAL_CONFIG", None)
    os.environ["YAHPO_LOCAL_CONFIG"] = "/tmp/yahpo_conf"

    from yahpo_gym.local_config import LocalConfiguration

    local_config = LocalConfiguration()

    assert str(local_config.settings_path) == os.environ["YAHPO_LOCAL_CONFIG"]

    if tmp_local_config_path is not None:
        os.environ["YAHPO_LOCAL_CONFIG"] = tmp_local_config_path
