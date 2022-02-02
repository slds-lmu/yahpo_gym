import pytest
from yahpo_gym.get_tasks import get_tasks

def test_get_tasks_single():

    df = get_tasks('single', version=0)
    assert list(df.columns.values) == ['scenario', 'instance', 'target']
    assert len(df) == 21

    with pytest.raises(Exception) as info:
        get_tasks('single', version=3)
        assert info == "version must coincide with version in `local_config.data_path`"
    
def test_get_tasks_multi():

    df = get_tasks('multi', version=0)
    assert list(df.columns.values) == ['scenario', 'instance', 'target']
    assert len(df) == 21

    with pytest.raises(Exception) as info:
        get_tasks('single', version=3)
        assert info == "version must coincide with version in `local_config.data_path`"