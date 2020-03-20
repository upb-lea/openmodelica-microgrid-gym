import gym
import pytest


@pytest.fixture
def env():
    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   viz_mode=None,
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={
                       'lc1': [
                           ['inductor1.i', 'inductor2.i', 'inductor3.i'],
                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                       'lcl1':
                           [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                            ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})
    return env


def test_modelica_env(env):
    assert hasattr(env, 'history')
