import logging

import GPy

from gym_microgrid.env import FullHistory

delta_t = 1e-4
V_dc = 1000
nomFreq = 50
nomVoltPeak = 230 * 1.414
iLimit = 30
DroopGain = 40000.0  # W/Hz
QDroopGain = 1000.0  # VAR/V

import gym
from gym_microgrid.auxiliaries import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams, MutableFloat
from gym_microgrid.agents import StaticControlAgent, SafeOptAgent
from gym_microgrid import Runner
from gym_microgrid.common import *

import numpy as np


def rew_fun(obs):
    # Vabc_master = [obs['lc1.capacitor1.v'][0], obs['lc1.capacitor2.v'][0], obs['lc1.capacitor3.v'][0]]
    # print(agent.controllers['master'].history['SPVdq0'].iloc[-1][0])

    # Measurements
    Vabc_master = obs[[f'lc1.capacitor{i + 1}.v' for i in range(3)]].to_numpy()[0]
    Iabc_master = obs[[f'lc1.inductor{i + 1}.i' for i in range(3)]].to_numpy()[0]

    phase = obs[['master.phase']].to_numpy()[0, 0]
    # = agent.controllers['master'].history['phase'].iloc[-1]
    Vdq0_master = abc_to_dq0(Vabc_master, phase)
    Idq0_master = abc_to_dq0(Iabc_master, phase)

    # Setpoints
    VSPdq0_master = obs[['master.'f'SPV{k}' for k in 'dq0']].to_numpy()[0]
    ISPdq0_master = obs[['master.'f'SPI{k}' for k in 'dq0']].to_numpy()[0]
    # np.array(agent.controllers['master'].history['SPVdq0'].iloc[-1])
    # ISPdq0_master = np.array(agent.controllers['master'].history['SPIdq0'].iloc[-1])

    VSPabc_master = dq0_to_abc(VSPdq0_master, phase)
    ISPabc_master = dq0_to_abc(ISPdq0_master, phase)

    # error = np.sum((VSPabc_master - Vabc_master) ** 2, axis=0) + np.sum((ISPabc_master - Iabc_master) ** 2, axis=0) \
    #        + 0  #np.sum((Idq0_master - agent.controllers['master']._voltagePI.controllers)**4)

    error = np.sum((ISPabc_master - Iabc_master) ** 2, axis=0) \
            + 0  # np.sum((Idq0_master - agent.controllers['master']._voltagePI.controllers)**4)

    return -error


if __name__ == '__main__':
    # Bounds on the inputs variable
    bounds = [(-0.1, 0.2)]

    # Define Kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), lengthscale=.01)

    # Define mutable parameters
    # mutable_params = dict(voltP=MutableFloat(25e-3))  #, voltI=MutableFloat(60))

    mutable_params = dict(currentP=MutableFloat(5e-3))
    # mutable_params = dict( currentI=MutableFloat(90))
    # mutable_params = dict(currentP=MutableFloat(12e-3), currentI=MutableFloat(90))

    ctrl = dict()
    # Voltage PI parameters for the current sourcing inverter
    # voltage_dqp_iparams = PI_params(kP=mutable_params['voltP'], kI=mutable_params['voltI'], limits=(-iLimit, iLimit))
    voltage_dqp_iparams = PI_params(kP=25e-3, kI=60, limits=(-iLimit, iLimit))

    # Current PI parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=90, limits=(-1, 1))
    # current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI= mutable_params['currentI'], limits=(-1, 1))
    # current_dqp_iparams = PI_params(kP=0.008, kI=mutable_params['currentI'], limits=(-1, 1))

    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    ctrl['master'] = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
                                                 qdroop_param, history=FullHistory())

    # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
    # Current PI parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI params for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop of the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, 0, nomFreq, tau_filt=0.04)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
    ctrl['slave'] = MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                                  droop_param, qdroop_param)

    # TODO add V_dc into the controllers as output gain
    agent = SafeOptAgent(mutable_params, kernel, ctrl,
                         dict(master=[np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                      np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)])],
                              slave=[np.array([f'lcl1.inductor{i + 1}.i' for i in range(3)]),
                                     np.array([f'lcl1.capacitor{i + 1}.v' for i in range(3)]),
                                     np.zeros(3)]), history=FullHistory())



    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   reward_fun=rew_fun,
                   viz_cols=['master.freq', 'lc1.*'],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=400,
                   model_path='../fmu/grid.network_singleController.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                     lcl1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]))

    runner = Runner(agent, env)
    runner.run(10, visualize=True)
