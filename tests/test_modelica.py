"""
Testing the Environment using simple action inputs and verifying the simulation result.
"""

import gym
import numpy as np
import pytest
from pytest import approx


@pytest.fixture
def env():
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode=None,
                   model_path='omg_grid/test.fmu',
                   net='net/net_test.yaml')
    return env


def test_reset(env):
    assert env.reset() == approx(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 325.10861867, -153.70643068, -171.40218799,
         0., 0., 0., 0., 0., 0., 0., 0., 0.])


def test_step(env):
    np.random.seed(1)
    env.reset()
    obs, r, done, _ = env.step(np.random.random(6))
    assert obs == approx([172.9788, 285.5633, 3.2967, 35.4845, 61.5703, -0.0585,
                          0.0000, 0.0000, 0.0000, 324.6152, -144.6233, -179.9919,
                          142.6508, 87.4849, 39.0866, 25.3610, 11.9275, 7.8399,
                          0.0000, 0.0000, 0.0000], 1e-3)
    assert r == 1
    assert not done


def test_proper_reset(env):
    np.random.seed(1)
    actions = np.random.random((100, 6))
    env.reset()
    for a in actions:
        env.step(a)
    state = str(env) + str(env.history.df)

    env.reset()
    for a in actions:
        env.step(a)
    assert state == str(env) + str(env.history.df)
