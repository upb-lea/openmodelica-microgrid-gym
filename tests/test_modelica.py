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
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


def test_step(env):
    np.random.seed(1)
    env.reset()
    obs, r, done, _ = env.step(np.random.random(6))
    assert obs == approx([3.54844812e+01, 6.15702788e+01, -5.85093688e-02, 1.72978825e+02,
                          2.85563336e+02, 3.29671269e+00, 2.53610151e+01, 1.19274553e+01,
                          7.83990524e+00, 1.42650770e+02, 8.74848857e+01, 3.90866066e+01])
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