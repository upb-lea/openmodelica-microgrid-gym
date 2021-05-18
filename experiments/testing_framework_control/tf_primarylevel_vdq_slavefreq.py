#####################################
# Example using an FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt
# Testing Framework for primary level (frequency, voltage)
# Definition of Benchmark, Implementation of load steps, Measurement of control metrics


import logging
import os
from random import random
from time import strftime, gmtime
from typing import List

import GPy
import gym
import numpy as np
import pandas as pd
from scipy.stats import linregress

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, \
    MultiPhaseDQ0PIPIController, PLLParams, InverseDroopParams, MultiPhaseDQCurrentController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.execution import Callback
from openmodelica_microgrid_gym.util import nested_map, FullHistory

pd.options.mode.chained_assignment = None  # default='warn'

# Simulation definitions
ts = .5e-4  # simulation time step size / s
max_episode_steps = 3000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 1000  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VA/V
vd_ref = np.array([325, 0, 0])  # exemplary set point i.e. vd = 325, vq = 0, v0 = 0 / A
vq_ref = 0  # vq = 0
R = 20  # resistance value / Ohm
L = 0.001  # resistance value / Henry
load_step_resistance = 7.5  # load step / Ohm
load_step_inductance = 0.015  # sets the inductive load step / H
movement_1_resistor = load_step_resistance  # first load step is negative
movement_2_resistor = (load_step_resistance if random() < 0.5
                       else -1 * load_step_resistance) + movement_1_resistor  # randomly chosen load step
movement_1_inductance = load_step_inductance  # first load step is negative
movement_2_inductance = (load_step_inductance if movement_2_resistor >= 0
                         else -load_step_inductance) + movement_1_inductance  # same sign as resistive load step

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, r'saves_droop_control_safeopt')
os.makedirs(save_folder, exist_ok=True)


#####################################
# Definition of Random Walk
# Starting Value: R = 20 Ohm, L= 1 mH
# Constant noise is created

def load_step_random_walk_resistor():
    random_walk = [R]
    for i in range(1, max_episode_steps):
        movement = -0.01 if random() < 0.5 else 0.01  # constant slight and random fluctuation of load
        value = random_walk[i - 1] + movement
        random_walk.append(value)
    return random_walk


def load_step_inductance_random_walk():
    random_walk_inductance = [L]
    for i in range(1, max_episode_steps):
        movement = -0.000001 if random() < 0.5 else 0.000001  # constant slight and random fluctuation of load
        value = random_walk_inductance[i - 1] + movement
        random_walk_inductance.append(value)
    return random_walk_inductance


list_resistor = load_step_random_walk_resistor()
list_inductance = load_step_inductance_random_walk()


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'slave.freq'],
                 [f'master.CVV{s}' for s in 'dq0']])

    def rew_fun(self, cols: List[str], data: np.ndarray, risk) -> float:
        """
        Defines the reward function for the environment. Uses the observations and set-points to evaluate the quality of
        the used parameters.
        Takes current measurement and set-points so calculate the mean-root control error

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx
        freq = data[idx[0]]

        vdq0_master = data[idx[1]]  # 3 phase voltages at LC capacitor

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        error = np.sum((np.abs((nomFreq - freq)) / nomFreq) ** 0.5, axis=0) + \
                np.sum((np.abs(([nomVoltPeak, 0, 0] - vdq0_master)) / nomVoltPeak) ** 0.5, axis=0)

        return -error.squeeze()


#########################################################################################
# Implementation of Load Steps
# It is checked, if quantity reaches steady state
# Values are stored in a databuffer
# If the databuffer fulfills several conditions, the first load steps will be implemented
# Programming 'on the fly'

class LoadstepCallback(Callback):
    def __init__(self):
        self.steady_state_reached = False
        self.settling_time = None
        self.databuffer_droop = None
        self._idx = None
        self.interval_check = False
        self.steady_state_check = False
        self.list_data = []
        self.list_abcissa = []
        self.upper_bound = 1.02 * nomFreq
        self.lower_bound = 0.98 * nomFreq
        self.databuffer_settling_time = 0
        self.databuffer_length = 150
        self.slope_info = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'slave.freq'],
                 [f'master.CVV{s}' for s in 'dq0']])

    def reset(self):
        self.databuffer_droop = np.empty(self.databuffer_length)  # creates a databuffer with a length of ten

    def __call__(self, cols, obs):
        self.set_idx(cols)
        self.databuffer_droop[-1] = obs[self._idx[0][0]]
        self.list_abcissa.append(len(self.list_data))
        self.list_data.append(obs[self._idx[0][0]])  # the current value is appended to the list

        # only when a certain number of values is available, slopeinfo is created --> used for conditions below
        if len(self.list_data) > self.databuffer_length:
            del self.list_abcissa[0]
            self.slope_info = linregress(self.list_abcissa, self.databuffer_droop)

        self.databuffer_droop = np.roll(self.databuffer_droop, -1)  # all values are shifted to the left by one

        # condition: it is checked whether the current is outside of the interval --> settling time may not be reached
        if (obs[self._idx[0][0]] < self.lower_bound or obs[
            self._idx[0][0]] > self.upper_bound) and not self.steady_state_reached:
            self.interval_check = False

        # pre-condition: checks if observation is within the specified interval --> settling time may be reached
        if self.lower_bound < obs[self._idx[0][0]] < self.upper_bound:
            if not self.interval_check:
                global position_settling_time
                position_settling_time = len(self.list_data)
                self.interval_check = True
            if self.databuffer_droop.std() < 0.003 and np.abs(self.slope_info[0]) < 0.00001 and np.round(
                    self.databuffer_droop.mean()) == nomFreq:  # if databuffer fulfills the conditions-->steady state
                self.steady_state_reached = True
                if not self.steady_state_check:
                    global position_steady_state
                    position_steady_state = len(self.list_data)  # stores the steady state position
                    self.steady_state_check = True

    # After steady state is reached, a fixed load step of + 7.5 Ohm is implemented
    # After 3/4 of the simulation time, another random load step of +/- 7.5 Ohm is implemented
    def load_step_resistance(self, t):
        for i in range(1, len(list_resistor)):
            if not self.steady_state_reached and t <= ts * i + ts:  # no steady state, no load step
                return list_resistor[i]  # for every step, it contains values that fluctuate a little bit
            elif self.steady_state_reached and t <= ts * i + ts and i <= max_episode_steps * 0.75:
                return list_resistor[i] + movement_1_resistor  # when steady state reached, first load step
            elif self.steady_state_reached and t <= ts * i + ts and i > max_episode_steps * 0.75:  # 75 % ts
                return list_resistor[i] + movement_2_resistor

    # After steady state is reached, a fixed load step of + 150 mH is implemented
    # After 3/4 of the simulation time, another random load step of +/- 150 mH is implemented
    def load_step_inductance(self, t):
        for i in range(1, len(list_inductance)):
            if not self.steady_state_reached and t <= ts * i + ts:  # no load step, no steady state
                return list_inductance[i]  # for every step, it contains values that fluctuate a little bit
            elif self.steady_state_reached and t <= ts * i + ts and i <= max_episode_steps * 0.75:
                return list_inductance[i] + movement_1_inductance  # when steady state reached, first load step
            elif self.steady_state_reached and t <= ts * i + ts and i > max_episode_steps * 0.75:  # 75 % ts
                return list_inductance[i] + movement_2_inductance


if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    #####################################
    # Definitions for the GP
    prior_mean = 2  # mean factor of the GP prior mean which is multiplied with the first performance of the init set
    noise_var = 0.001 ** 2  # measurement noise sigma_omega
    prior_var = 0.2  # prior variance of the GP

    # Choose all droop params as mutable parameters (below) and define bounds and lengthscale for both of them
    bounds = [(0, 100000), (0, 100000), (0, 3000), (0, 100)]  # bounds on the input variable
    lengthscale = [10000, 10000, 300., 10.]  # length scale for the parameter variation for the GP

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means. Performance measurement for optimization are seen as
    # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 0.5
    # minimal allowed performance (depending on episode return)
    j_min = -2
    # The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
    # expanding points eventually.
    # The following variable is multiplied with the first performance of the initial set by the factor below:
    explore_threshold = 0

    # Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
    # limit exceeded
    abort_reward = -10 * j_min

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers

    # Choose all droop parameter as mutable parameters
    mutable_params = dict(pDroop_master=MutableFloat(30000.0), pDroop_slave=MutableFloat(30000.0),
                          qDroop_master=MutableFloat(QDroopGain), qDroop_slave=MutableFloat(10))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for
    # the filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param_master = DroopParams(mutable_params['pDroop_master'], 0.005, nomFreq)
    # droop parameter used to react on frequency drop
    droop_param_slave = InverseDroopParams(mutable_params['pDroop_slave'], ts, nomFreq, tau_filt=0.04)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    # qDroop parameter used for virtual voltage drop
    qdroop_param_master = DroopParams(mutable_params['qDroop_master'], 0.002, nomVoltPeak)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param_slave = InverseDroopParams(mutable_params['qDroop_slave'], ts, nomVoltPeak, tau_filt=0.01)

    ###############
    # define Master
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))  # kp=0.025
    # Current control PI gain parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))  # kp=0.012

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param_master,
                                            qdroop_param_master, ts_sim=ts, ts_ctrl=2 * ts, name='master'))

    ###############
    # define slave
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))  #
    # PI gain parameters for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop characteristic for the active power Watts/Hz, W.s/Hz

    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, iLimit, droop_param_slave,
                                              qdroop_param_slave, ts_sim=ts, name='slave'))

    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    # Arguments described above
    # History is used to store results
    agent = SafeOptAgent(mutable_params,
                         abort_reward,
                         j_min,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold),
                         ctrl,
                         {'master': [[f'lc1.inductor{k}.i' for k in '123'],
                                     [f'lc1.capacitor{k}.v' for k in '123'],
                                     ],
                          'slave': [[f'lcl1.inductor{k}.i' for k in '123'],
                                    [f'lcl1.capacitor{k}.v' for k in '123'],
                                    np.zeros(3)]},
                         history=FullHistory()
                         )


    #####################################
    # Definition of the environment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using two inverters supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed
    #   Labels and grid is adjusted using the PlotTmpl (For more information, see UserGuide)
    #   figures are stored to folder
    # - inputs to the models are the connection points to the inverters (see user guide for more details)
    # - model outputs are the the 3 currents through the inductors and the 3 voltages across the capacitors for each
    #   inverter and the 3 currents through the load

    def xylables_i(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/Inductor_currents' + time + '.pdf')


    def xylables_v_abc(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/abc_voltage' + time + '.pdf')


    def xylables_v_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/dq0_voltage' + time + '.pdf')


    def xylables_P_master(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$P_{\mathrm{master}}\,/\,\mathrm{W}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/P_master' + time + '.pdf')


    def xylables_P_slave(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$P_{\mathrm{slave}}\,/\,\mathrm{W}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/P_slave' + time + '.pdf')


    def xylables_freq(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$f_{\mathrm{slave}}\,/\,\mathrm{Hz}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/f_slave' + time + '.pdf')
        fig.show()


    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{\u03A9}$')
        ax.grid(which='both')


    def xylables_L(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')  # zeit
        ax.set_ylabel('$L_{\mathrm{123}}\,/\,\mathrm{H}$')
        ax.grid(which='both')


    callback = LoadstepCallback()

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   viz_cols=[
                       PlotTmpl([f'slave.freq'],
                                callback=xylables_freq
                                ),
                       PlotTmpl([f'master.CVV{i}' for i in 'dq0'],
                                callback=xylables_v_dq0
                                ),
                       PlotTmpl([f'rl1.resistor{i}.R' for i in '123'],  # Plot Widerstand RL
                                callback=xylables_R
                                ),
                       PlotTmpl([f'rl1.inductor{i}.L' for i in '123'],  # Plot Widerstand RL
                                callback=xylables_L
                                ),
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   model_params={'rl1.resistor1.R': callback.load_step_resistance,
                                 'rl1.resistor2.R': callback.load_step_resistance,
                                 'rl1.resistor3.R': callback.load_step_resistance,
                                 'rl1.inductor1.L': callback.load_step_inductance,
                                 'rl1.inductor2.L': callback.load_step_inductance,
                                 'rl1.inductor3.L': callback.load_step_inductance
                                 },
                   model_path='../../omg_grid/grid.network.fmu',
                   net='net_RL_load.yaml',
                   history=FullHistory()
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)
    runner = Runner(agent, env, callback)

    runner.run(num_episodes, visualise=True)

    #####################################
    # Performance results and Parameters as well as plots are stored in folder saves_droop
    agent.history.df.to_csv(save_folder + '/result.csv')

    print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df.round({'J': 4, 'Params': 4})))
    print('\n Experiment finished with best set: \n')
    print('\n  [pDroop_master, pDroop_slave, qDroop_master, qDroop_slave]= {}'.format(
        agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
    print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
    print('\n\nBest experiment results are plotted in the following:')
    print()

#####################################
# Calculation of Metrics
# Here, the d- term of vd is analysed

df_master_CVVd = env.history.df[['master.CVVd']]
from metrics import Metrics

voltage_controller_metrics_vd = Metrics(df_master_CVVd, vd_ref[0], ts, max_episode_steps,
                                        position_steady_state=position_steady_state,
                                        position_settling_time=position_settling_time)

d = {'Overshoot': [voltage_controller_metrics_vd.overshoot()],
     'Rise Time/s ': [voltage_controller_metrics_vd.rise_time()],
     'Settling Time/s ': [voltage_controller_metrics_vd.settling_time_vd_droop()],
     'Root Mean Squared Error/V': [voltage_controller_metrics_vd.RMSE()],
     'Steady State Error/V': [voltage_controller_metrics_vd.steady_state_error()]}

print("\n")
print()
df_metrics_vd = pd.DataFrame(data=d).T
df_metrics_vd.columns = ['Value']
print('Metrics of Vd')
print(df_metrics_vd)

######IMPORTANT FOR THE SCORING MODEL PRIMARY LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_vd.to_pickle("./df_metrics_vd_controller1_droop.pkl")
# Maybe you need to replace 'controller1_droop.pkl' with 'controller2_droop.pkl'


#####################################
# Calculation of Metrics
# Here, the q-term of vdq is analysed
df_master_CVVq = env.history.df[['master.CVVq']]

from metrics import Metrics

voltage_controller_metrics_vq = Metrics(df_master_CVVq, vq_ref, ts, max_episode_steps,
                                        position_steady_state=position_steady_state,
                                        position_settling_time=position_settling_time)

d = {'Root Mean Squared Error/V': [voltage_controller_metrics_vq.RMSE()],
     'Steady State Error/V': [voltage_controller_metrics_vq.steady_state_error()],
     'Peak Value/V': [voltage_controller_metrics_vq.absolute_peak()]}

print("\n")
print()
df_metrics_vq = pd.DataFrame(data=d).T
df_metrics_vq.columns = ['Value']
print('Metrics of Vq')
print(df_metrics_vq)

######IMPORTANT FOR THE SCORING MODEL PRIMARY LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_vq.to_pickle("./df_metrics_vq_controller1_droop.pkl")
# Maybe you need to replace 'controller1_droop.pkl' with 'controller2_droop.pkl'


#####################################
# Calculation of Metrics
# Here, the slave frequency is analysed

df_slave_frequency = env.history.df[['slave.freq']]
from metrics import Metrics

frequency_controller_metrics = Metrics(df_slave_frequency, nomFreq, ts, max_episode_steps,
                                       position_steady_state=position_steady_state,
                                       position_settling_time=position_settling_time)

d = {'Overshoot': [frequency_controller_metrics.overshoot()],
     'Rise Time/s ': [frequency_controller_metrics.rise_time()],
     'Settling Time/s ': [frequency_controller_metrics.settling_time()],
     'Root Mean Squared Error/Hz': [frequency_controller_metrics.RMSE()],
     'Steady State Error/Hz': [frequency_controller_metrics.steady_state_error()]}

print("\n")
print()
df_metrics_slave_f = pd.DataFrame(data=d).T
df_metrics_slave_f.columns = ['Value']
print('Metrics of Slave Frequency')
print(df_metrics_slave_f)

######IMPORTANT FOR THE SCORING MODEL PRIMARY LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_slave_f.to_pickle("./df_metrics_slave_f_controller1_droop.pkl")
# Maybe you need to replace 'controller1.pkl' with 'controller2.pkl'
