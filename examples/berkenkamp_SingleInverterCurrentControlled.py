import logging

import GPy

from gym_microgrid.env import FullHistory

delta_t = 1e-4
V_dc = 1000
nomFreq = 50
nomVoltPeak = 230 * 1.414
iLimit = 30
mu = 0.05
DroopGain = 40000.0  # W/Hz
QDroopGain = 1000.0  # VAR/V

import gym
from gym_microgrid.auxiliaries import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    InverseDroopParams, PLLParams, MutableFloat, MultiPhaseDQCurrentSourcingController
from gym_microgrid.agents import StaticControlAgent, SafeOptAgent
from gym_microgrid import Runner
from gym_microgrid.common import *

import numpy as np
import pandas as pd


def rew_fun(obs: pd.Series) -> float:
    """
    Defines the reward function for the enviroment. Uses the observations and setpoints to tell the quality of the
    used parameters.
    Takes Current measurements and setpoints so calculate the MSE and uses a logarithmic barrier function in case of the
    boundary of the current limit. Barrier adjustable using parameter mu.

    :param obs: Observation from the enviroment (ControlVariables, e.g. Currents and voltages)
    :return: Error as negative Reward
    """


    # Measurements
    Iabc_master = obs[[f'lc1.inductor{i + 1}.i' for i in range(3)]].to_numpy()
    phase = obs[['master.phase']].to_numpy()[0]
    # Idq0_master = abc_to_dq0(Iabc_master, phase)

    # Setpoints
    ISPdq0_master = obs[['master.'f'SPI{k}' for k in 'dq0']].to_numpy()
    ISPabc_master = dq0_to_abc(ISPdq0_master, phase)

    error = np.sum((ISPabc_master - Iabc_master) ** 2, axis=0) \
            + -np.sum(mu * np.log(iLimit - np.abs(Iabc_master)), axis=0)

    return -error.squeeze()


if __name__ == '__main__':

    # Bounds on the inputs variable
    bounds = [(-0.1, 0.2)]

    # Define Kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), lengthscale=.01)

    # Define mutable parametersnp.sum((ISPabc_master - Iabc_master) ** 2)
    # mutable_params = dict(voltP=MutableFloat(25e-3))  #, voltI=MutableFloat(60))

    # mutable_params = dict(currentP=MutableFloat(10e-3))
    # mutable_params = dict( currentI=MutableFloat(90))
    mutable_params = dict(currentP=MutableFloat(7e-3), currentI=MutableFloat(90))

    ctrl = dict()

    # PI parameters for the current Controller of the inverter
    # mutable_params define which parameter is adjusted via optimaization (kp, ki or kp&ki)
    # current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=90, limits=(-1, 1))
    # current_dqp_iparams = PI_params(kP=0.01027, kI=mutable_params['currentI'], limits=(-1, 1))
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    ctrl['master'] = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param)

    agent = SafeOptAgent(mutable_params, kernel, ctrl,
                         dict(master=[np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                      np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)]), np.array([15, 0, 0])],
                              ),
                         history=FullHistory()
                         )

    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   reward_fun=rew_fun,
                   # viz_cols=['master.freq', 'master.CVI*', 'lc1.ind*'],
                   viz_cols=['lc1.ind*'],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=200,
                   model_path='../fmu/grid.network_singleInverter.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']])
                   )

    runner = Runner(agent, env)
    runner.run(30, visualise_env=True)
