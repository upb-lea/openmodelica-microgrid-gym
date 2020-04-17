#####################################
# Example using a FMU by OpenModelica and safeopt algorithm to find "optimal" controller parameters.
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller which integral gain is fixed and the proportional gain is optimized by SafeOpt
# (1D example)


import logging

import GPy

import gym
from openmodelica_microgrid_gym.env import FullHistory
from openmodelica_microgrid_gym.auxiliaries import PI_params, DroopParams, MutableFloat, \
    MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.common import *

import numpy as np
import pandas as pd

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 300  # number of simulation steps per episode
num_episodes = 15  # number of simulation episodes (i.e. SafeOpt iterations)
V_dc = 1000  # DC-link voltage / V
nomFreq = 50  # grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # current limit / A
iNominal = 20  # nominal current / A
mu = 1  # Factor for barrier function (see below)
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V

i_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A


def rew_fun(obs: pd.Series) -> float:
    """
    Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
    used parameters.
    Takes current measurements and setpoints so calculate the MSE control error and uses a logarithmic barrier function
    in case of violating the current limit. Barrier adjustable using parameter mu.

    :param obs: Observation from the environment (ControlVariables, e.g. currents and voltages)
    :return: Error as negative reward
    """

    # Measurements
    Iabc_master = obs[[f'lc1.inductor{i + 1}.i' for i in range(3)]].to_numpy()  # 3 phase currents at LC inductors
    phase = obs[['master.phase']].to_numpy()[0]

    # Setpoints
    ISPdq0_master = obs[['master.'f'SPI{k}' for k in 'dq0']].to_numpy()  # setting dq reference
    ISPabc_master = dq0_to_abc(ISPdq0_master, phase)  # convert dq setpoints into three-phase abc coordinates

    # control error = mean-root-error of reference minus measurements
    # plus barrier penalty for violating the current constraint
    error = np.sum((np.abs((ISPabc_master - Iabc_master)) / iLimit) ** 0.5, axis=0) \
            + -np.sum(mu * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
            * max_episode_steps

    return -error.squeeze()


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    # Bounds on the inputs variable Kp, Ki - For single parameter adaption use only one dimension
    bounds = [(0.001, 0.015)]  # , (50, 155)]
    noise_var = 0.001 ** 2  # Measurement noise sigma_omega

    # Length scale for the parameter variation [Kp, Ki] for the GP
    lengthscale = [.004]  # , 20.]

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=1, variance=0.025, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers
    # mutable_params = Parameters (Kp & Ki of the current controller of the inverter) to change using safeopt
    # algorithms in the safeopt agent to find an "optimal" PI controller parameters Kp and Ki
    ctrl = dict()
    mutable_params = dict(currentP=MutableFloat(7e-3))  # , currentI=MutableFloat(90))
    # PI parameters for the current Controller of the inverter
    # current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=90, limits=(-1, 1))
    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    ctrl['master'] = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param,
                                                           undersampling=2)

    #####################################
    # Definition of the agent
    # Agent using the safeopt algorithm by Berkenkamp
    # (https://arxiv.org/abs/1509.01066)
    # as example to
    agent = SafeOptAgent(mutable_params, kernel, dict(bounds=bounds, noise_var=noise_var), ctrl,
                         dict(master=[np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                      np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)]), i_ref],
                              ),
                         history=FullHistory()
                         )

    #####################################
    # Definition of the enviroment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=rew_fun,
                   time_step=delta_t,
                   # viz_cols=['master.freq', 'master.CVI*', 'lc1.ind*'],
                   viz_cols=['lc1.ind*'],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   model_path='../fmu/grid.network_singleInverter.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']])
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 10 different episodes
    runner = Runner(agent, env)
    runner.run(num_episodes, visualise_env=True)
