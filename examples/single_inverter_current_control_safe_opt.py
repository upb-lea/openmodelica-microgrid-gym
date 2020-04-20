#####################################
# Example using a FMU by OpenModelica and safeopt algorithm to find "optimal" controller parameters.
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller which integral gain is fixed and the proportional gain is optimized by SafeOpt
# (1D example)
# toDo: Change doku from Kp to Kp&Ki

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

# Choose which controller parameters should be adjusted.
# - adjust_Kp_only == True: 1D example: Only Kp ot the PI parameter is adjusted
# - adjust_Ki_only == True: 1D example: Only Ki ot the PI parameter is adjusted
# - adjust_Kp_and_Ki == True: 2D example: Kp and Ki are adjusted
# ATTENTION! Choose only one parameter as TRUE!
adjust_Kp_only = True
adjust_Ki_only = False
adjust_Kp_and_Ki = False
# summe aus 3 mal true = 1, sonst Fehler
if sum([adjust_Kp_only, adjust_Ki_only, adjust_Kp_and_Ki]) is not 1:
    raise ValueError("ATTENTION! Choose only one of the adjustable controller parameter as TRUE (l.30ff.)!")

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 300  # number of simulation steps per episode
num_episodes = 15  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 700  # DC-link voltage / V; will be set as model parameter in the fmu
nomFreq = 50  # grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # current limit / A
iNominal = 20  # nominal current / A
mu = 2  # Factor for barrier function (see below)
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

    # Measurements: 3 Currents through the inductors of the inverter -> Control variables (CV)
    Iabc_master = obs[[f'lc1.inductor{i + 1}.i' for i in range(3)]].to_numpy()  # 3 phase currents at LC inductors
    phase = obs[['master.phase']].to_numpy()[0]  # Phase from the master controller needed for transformation

    # Setpoints
    ISPdq0_master = obs[['master.'f'SPI{k}' for k in 'dq0']].to_numpy()  # setting dq reference
    ISPabc_master = dq0_to_abc(ISPdq0_master, phase)  # convert dq set-points into three-phase abc coordinates

    # control error = mean-root-error of reference minus measurements
    # plus barrier penalty for violating the current constraint
    error = np.sum((np.abs((ISPabc_master - Iabc_master)) / iLimit) ** 0.5, axis=0) \
            + -np.sum(mu * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
            * max_episode_steps

    return -error.squeeze()


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    prior_mean = 2  # Mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001 ** 2  # Measurement noise sigma_omega
    prior_var = 0.1  # Prior variance of the GP

    if adjust_Kp_only:
        bounds = [(0.00, 0.03)]  # Bounds on the input variable Kp
        lengthscale = [.01]  # Length scale for the parameter variation [Kp] for the GP

    # For 1D example, if Ki should be adjusted instead of Kp, choose adjust_Kp == False
    if adjust_Ki_only:
        bounds = [(0, 300)]  # Bounds on the input variable Ki
        lengthscale = [50.]  # Length scale for the parameter variation [Ki] for the GP

    # For 2D example, choose Kp and Ki as mutable parameter (below) and define bounds and lengthscale for both of them
    if adjust_Kp_and_Ki:
        bounds = [(0.0, 0.03), (0, 300)]
        lengthscale = [.01, 50.]

    # If the performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the inital Performance: safe_threshold = 1.2 means, performance measurements for parameter-change are seen as
    # unsafe, if the new measured performance drops below 20% of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 2

    # The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
    # expanding points eventually.
    # is multiplied with the first performance of the initial set by the factor below:
    explore_threshold = 2

    # Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
    # limit exceeded
    abort_reward = 10

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers

    ctrl = dict()  # empty dictionary for the the controller(s)

    if adjust_Kp_only:
        # mutable_params = Parameter (Kp of the current controller of the inverter) to change using safeopt algorithms in
        # the safeopt agent to find an "optimal" PI controller parameter Kp using the safe inital parameter
        mutable_params = dict(currentP=MutableFloat(10e-3))

        # Define the PI parameters for the current Controller of the inverter
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=115, limits=(-1, 1))

    # For 1D example, if Ki should be adjusted instead of Kp, choose adjust_Kp == False
    if adjust_Ki_only:
        mutable_params = dict(currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=10e-3, kI=mutable_params['currentI'], limits=(-1, 1))

    # For 2D example, choose Kp and Ki as mutable parameter
    if adjust_Kp_and_Ki:
        mutable_params = dict(currentP=MutableFloat(10e-3), currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal Frequenzy
    # Droop controller used to calculate the virtual frequency drop due to load
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    # ToDo: needed? Due to class definition still inside
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from abouve
    ctrl['master'] = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param,
                                                           undersampling=2)

    #####################################
    # Definition of the agent
    # Agent using the safeopt algorithm by Berkenkamp
    # (https://arxiv.org/abs/1509.01066)
    # as example
    # Arguments described above
    # history is used to store results
    agent = SafeOptAgent(mutable_params,
                         abort_reward,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold
                              ),
                         ctrl,
                         dict(master=[np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                      np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)]), i_ref],
                              ),
                         history=FullHistory()
                         )

    #####################################
    # Definition of the enviroment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed (here only the 3 currents across the
    #   inductors of the inverters (alternatively also the grid frequency and the currents transformed by the agent to
    #   dq0
    # - inputs to the models are the connection points to the inverters (see user guide for more)
    # - model outputs are the the 3 currents through the inductors and the 3 voltages across the capacitors
    # ToDo: v_caps needed?
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=rew_fun,
                   time_step=delta_t,
                   # viz_cols=['master.freq', 'master.CVI*', 'lc1.ind*'],
                   viz_cols=['lc1.ind*'],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   # model_params={'inverter1.v_DC': v_DC}, # if v_DC should not be default = 1000 V
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
