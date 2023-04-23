import pytest
from yahpo_gym.get_suite import get_suite


def test_get_tasks_single():
    df = get_suite("single", version=1.0)
    assert list(df.columns.values) == ["scenario", "instance", "target"]
    assert len(df) == 20

    with pytest.raises(Exception) as info:
        get_suite("single", version=3)
        assert info == "version must coincide with version in `local_config.data_path`"


def test_get_tasks_multi():
    df = get_suite("multi", version=1.0)
    assert list(df.columns.values) == ["scenario", "instance", "target"]
    assert len(df) == 25

    with pytest.raises(Exception) as info:
        get_suite("single", version=3)
        assert info == "version must coincide with version in `local_config.data_path`"
