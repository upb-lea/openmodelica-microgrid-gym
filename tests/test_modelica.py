import gym
import pytest
import numpy as np
from pytest import approx


@pytest.fixture
def env():
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode=None,
                   model_path='fmu/test.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={'lc1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                 'lcl1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})
    return env


def test_reset(env):
    assert np.array_equal(np.zeros(12), env.reset().to_numpy())


def test_step(env):
    np.random.seed(1)
    env.reset()
    obs, r, done, _ = env.step(np.random.random(6))
    assert obs.to_numpy() == approx([3.54844812e+01, 6.15702788e+01, -5.85093688e-02, 1.72978825e+02,
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


def test_params_simple():
    def fun(t):
        return t + 1

    np.random.seed(1)
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode=None,
                   max_episode_steps=100,
                   model_path='fmu/test.fmu',
                   model_params=dict(i1p1=lambda t: np.sin(t), i1p2=3, i1p3=fun),
                   model_input=['i2p1', 'i2p2', 'i2p3'],
                   model_output={'lc1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                 'lcl1': [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})
    env.reset()
    obs, r, done, _ = env.step(np.random.random(3))
    assert obs.to_numpy() == approx([-3.08472072e-01, 2.56346548e+02, 8.56263731e+01, 1.46876784e+01,
                                     1.19315365e+03, 3.89265210e+02, 3.54071687e+01, 5.89398985e+01,
                                     -7.29991175e-01, 1.76505718e+02, 4.10540511e+02, 3.52688013e+01])
    assert r == 1
    assert not done
