import pytest

from openmodelica_microgrid_gym.net import Network


def test_load():
    Network.load('tests/net/net_valid.yaml')
    assert True

def test_load_dup_inputs():
    with pytest.raises(ValueError):
        Network.load('tests/net/net_dupinputs.yaml')
