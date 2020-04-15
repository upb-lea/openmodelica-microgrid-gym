#####################################
# Example using a FMU by OpenModelica and safeopt algorithm to find "optimal" controller parameters.
# Simulationsetup: Inverter supplying 15 A d-current to an RL-load via 2 LC filters
# Controller: PI current controller



import logging

import GPy

from gym_microgrid.env import FullHistory
import gym
from gym_microgrid.auxiliaries import PI_params, DroopParams, MutableFloat, MultiPhaseDQCurrentSourcingController
from gym_microgrid.agents import SafeOptAgent
from gym_microgrid import Runner
from gym_microgrid.common import *

import numpy as np
import pandas as pd

# Simulation definitons
delta_t = 1e-4  # time step size / s
V_dc = 1000  # / V
nomFreq = 50  # / Hz
nomVoltPeak = 230 * 1.414  # / V
iLimit = 30  # / A
mu = 0.05  # Factor for barrier function (see below)
DroopGain = 40000.0  # / W/Hz
QDroopGain = 1000.0  # / VAR/V

i_ref = np.array([15, 0, 0])  # / A

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
    #####################################
    # Definitions for the GP
    # Bounds on the inputs variable Kp, Ki - For single parameter adaption use only one dimension
    bounds = [(0.001, 0.015), (50, 155)]
    noise_var = 0.05 ** 2  # Measurement noise sigma_omega

    # Length scale for the parameter variation [Kp, Ki] for the GP
    lengthscale = [.004, 20.]

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=2, variance=2., lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers
    # mutable_params = Parameters (Kp & Ki of the current controller of the inverter) to change using safeopt
    # algorithms in the safeopt agent to find an "optimal" PI controller parameters Kp and Ki
    ctrl = dict()
    mutable_params = dict(currentP=MutableFloat(7e-3), currentI=MutableFloat(90))
    # PI parameters for the current Controller of the inverter
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))
    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    ctrl['master'] = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param)

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

    #####################################
    # Execution of the experiment
    # Using a runner to execute 10 different episodes
    runner = Runner(agent, env)
    runner.run(10, visualise_env=True)
