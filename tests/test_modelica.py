import gym
import pytest
import numpy as np
from pytest import approx


@pytest.fixture
def env():
    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   viz_mode=None,
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={'lc1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                 'lcl1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})
    return env


def test_reset(env):
    assert np.array_equal(np.zeros(12), env.reset())


def test_step(env):
    np.random.seed(1)
    env.reset()
    obs, r, done = env.step(np.random.random(6))
    assert obs == approx(
        [3.548958e-02, 6.157956e-02, -5.862613e-05, 1.727147e-01, 2.851154e-01, 3.294574e-03, 2.536135e-02,
         1.192695e-02, 7.840170e-03, 1.426348e-01, 8.746552e-02, 3.908457e-02])
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


def test_params_simple():
    np.random.seed(1)
    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   viz_mode=None,
                   model_params=dict(i1p1=lambda t: np.sin(t), i1p2=3),
                   model_input=['i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={'lc1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                 'lcl1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})
    env.reset()
    obs, r, done = env.step(np.random.random(4))
    assert obs == approx([-5.32812108e-04, 2.56906733e-01, 3.54883017e-02, 2.53605852e-02,
                          1.16595698e+00, 1.72720287e-01, 6.11590476e-02, -2.20933938e-03,
                          2.53610394e-02, 3.04877618e-01, 1.05669912e-01, 1.42644541e-01])
    assert r == 1
    assert not done
