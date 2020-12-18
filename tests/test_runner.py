"""
The Tests in this file are high level integration tests.
Based on a pre-packed and fixed application example it is verified that the simulated state and action trajectories match expected values. 
If the simulated system behavior changes against baseline a problem within the software toolchain is likely. 
"""

import gym
import numpy as np
import pandas as pd
import pytest
from pytest import approx

from openmodelica_microgrid_gym import Runner, Agent
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import *
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import flatten


@pytest.fixture
def agent():
    delta_t = 1e-4
    nomFreq = 50
    nomVoltPeak = 230 * 1.414
    iLimit = 30
    DroopGain = 40000.0  # W/Hz
    QDroopGain = 1000.0  # VAR/V

    mutable_params = dict(voltP=MutableFloat(25e-3), voltI=MutableFloat(60))

    ctrl = []
    # Voltage PI parameters for the current sourcing inverter
    voltage_dqp_iparams = PI_params(kP=mutable_params['voltP'], kI=mutable_params['voltI'], limits=(-iLimit, iLimit))
    # Current PI parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param, qdroop_param,
                                            ts_sim=delta_t, name='master'))

    # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
    # Current PI parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI params for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop of the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, 0, nomFreq, tau_filt=0.04)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, iLimit, droop_param, qdroop_param,
                                              ts_sim=delta_t, name='slave'))

    # validate that parameters can be changed later on
    agent = StaticControlAgent(ctrl, {'master': [[f'lc1.inductor{i + 1}.i' for i in range(3)],
                                                 [f'lc1.capacitor{i + 1}.v' for i in range(3)]],
                                      'slave': [[f'lcl1.inductor{i + 1}.i' for i in range(3)],
                                                [f'lcl1.capacitor{i + 1}.v' for i in range(3)],
                                                np.zeros(3)]})
    return mutable_params, agent


@pytest.fixture
def agent_undersample():
    delta_t = 1e-4
    nomFreq = 50
    nomVoltPeak = 230 * 1.414
    iLimit = 30
    DroopGain = 40000.0  # W/Hz
    QDroopGain = 1000.0  # VAR/V

    mutable_params = dict(voltP=MutableFloat(25e-3), voltI=MutableFloat(60))

    ctrl = []
    # Voltage PI parameters for the current sourcing inverter
    voltage_dqp_iparams = PI_params(kP=mutable_params['voltP'], kI=mutable_params['voltI'], limits=(-iLimit, iLimit))
    # Current PI parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param, qdroop_param,
                                            ts_sim=delta_t, ts_ctrl=2 * delta_t, name='master'))

    # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
    # Current PI parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI params for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop of the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, 0, nomFreq, tau_filt=0.04)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, iLimit, droop_param, qdroop_param,
                                              ts_sim=delta_t, ts_ctrl=5 * delta_t, name='slave'))

    # validate that parameters can be changed later on
    agent = StaticControlAgent(ctrl, {'master': [[f'lc1.inductor{i + 1}.i' for i in range(3)],
                                                 [f'lc1.capacitor{i + 1}.v' for i in range(3)]],
                                      'slave': [[f'lcl1.inductor{i + 1}.i' for i in range(3)],
                                                [f'lcl1.capacitor{i + 1}.v' for i in range(3)],
                                                np.zeros(3)]})
    return agent


@pytest.fixture()
def env():
    net = Network.load('net/net_test.yaml')
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode=None,
                   model_path='omg_grid/test.fmu',
                   max_episode_steps=100,
                   net=net)

    return env, net.in_vars(), flatten(net.out_vars())


def test_main(agent, env):
    env, _, out_params = env
    runner = Runner(agent[1], env)
    runner.run(1)
    # env.history.df.to_hdf('tests/test_main.hd5', 'hist')
    df = env.history.df.head(100)
    df = df.reindex(sorted(df.columns), axis=1)
    df2 = pd.read_hdf('tests/test_main.hd5', 'hist').head(100)  # noqa
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    assert df[out_params].to_numpy() == approx(df2[out_params].to_numpy(), 5e-2)


def test_main_paramchange(agent, env):
    params, agent = agent
    env, _, out_params = env
    runner = Runner(agent, env)
    params['voltP'].val = 4
    runner.run(1)

    # env.history.df.to_hdf('tests/test_main2.hd5', 'hist')
    df = env.history.df.head(50)
    df = df.reindex(sorted(df.columns), axis=1)
    df2 = pd.read_hdf('tests/test_main.hd5', 'hist').head(50)  # noqa
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    assert df[out_params].to_numpy() != approx(df2[out_params].to_numpy(), 5e-3)

    df2 = pd.read_hdf('tests/test_main2.hd5', 'hist').head(50)  # noqa
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    assert df[out_params].to_numpy() == approx(df2[out_params].to_numpy(), 5e-3)


def test_main_undersample(agent_undersample, env):
    env, _, out_params = env
    runner = Runner(agent_undersample, env)
    runner.run(1)

    # env.history.df.to_hdf('tests/test_main4.hd5', 'hist')
    df = env.history.df.head(100)
    df = df.reindex(sorted(df.columns), axis=1)
    df2 = pd.read_hdf('tests/test_main4.hd5', 'hist').head(100)  # noqa
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    assert df[out_params].to_numpy() == approx(df2[out_params].to_numpy(), 5e-2)


def test_simpleagent(env):
    np.random.seed(1)
    env, inputs, out_params = env

    class RndAgent(Agent):
        def act(self, obs: pd.Series) -> np.ndarray:
            return np.random.random(len(inputs))

    agent = RndAgent()
    runner = Runner(agent, env)
    runner.run(1)

    # env.history.df.to_hdf('tests/test_main3.hd5', 'hist')
    df = env.history.df.head(40)
    df = df.reindex(sorted(df.columns), axis=1)
    df2 = pd.read_hdf('tests/test_main3.hd5', 'hist').head(40)  # noqa
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    assert df[out_params].to_numpy() == approx(df2[out_params].to_numpy(), 5e-4)
