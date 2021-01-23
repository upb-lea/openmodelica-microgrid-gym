#####################################
# Example using a FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller gain parameters are optimized by SafeOpt
# Testing Framework for current control
# Definition of Benchmark, Implementation of load steps, Measurement of control metrics


import logging
from typing import List
from math import sqrt
from random import random
import GPy
import gym
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import scipy
from gym.envs.tests.test_envs_semantics import steps
from sklearn.metrics import mean_squared_error
from scipy.signal import argrelextrema
from statistics import mean

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.execution import Callback
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

from random import seed
from random import random

# Choose which controller parameters should be adjusted by SafeOpt.
# - Kp: 1D example: Only the proportional gain Kp of the PI controller is adjusted
# - Ki: 1D example: Only the integral gain Ki of the PI controller is adjusted
# - Kpi: 2D example: Kp and Ki are adjusted simultaneously

adjust = 'Kpi'

# Check if really only one simulation scenario was selected
if adjust not in {'Kp', 'Ki', 'Kpi'}:
    raise ValueError("Please set 'adjust' to one of the following values: 'Kp', 'Ki', 'Kpi'")

# Simulation definitions
net = Network.load('../net/net_single-inv-curr.yaml')
max_episode_steps = 1200  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
id_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A
iq_ref = 0 # iq = 0
R = 20  # resistance value / Ohm
L = 0.001  # resistance value / Henry
ts = 1e-4  # duration of episode / s
position_settling_time = 0  # initial value
position_steady_state = 0  # initial value
load_jump_resistance = 5  # load jump
load_jump_inductance = 0.0004  # sets the inductive load_jump
movement_1_resistor = -load_jump_resistance  # first load jump is negative
movement_2_resistor = (load_jump_resistance if random() < 0.5
else -1 * load_jump_resistance) + movement_1_resistor  # randomly chosen load jump
movement_1_inductance = -load_jump_inductance  # first load jump is negative
movement_2_inductance = ((load_jump_inductance + movement_1_inductance) if movement_2_resistor > 0
else -load_jump_inductance) + movement_1_inductance  # same sign as resistive load jump


#####################################
# Definition of Random Walk
# Starting Value: R = 20 Ohm, L= 1 mH
# Constant noise is created

def load_step_random_walk_resistor():
    random_walk = []
    random_walk.append(R)  # R = 20 Ohm
    for i in range(1, max_episode_steps):
        movement = -0.01 if random() < 0.5 else 0.01  # constant slight and random fluctuation of load
        value = random_walk[i - 1] + movement
        random_walk.append(value)
    return random_walk


def load_step_inductance_random_walk():
    random_walk_inductance = []
    random_walk_inductance.append(L)  # L = 1 mH
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
                [[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                 [f'master.CVI{k}' for k in 'dq0']])

    def rew_fun(self, cols: List[str], data: np.ndarray) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the
        quality of the used parameters.
        Takes current measurement and setpoints so calculate the mean-root-error control error and uses a logarithmic
        barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        Iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        phase = data[idx[1]]  # phase from the master controller needed for transformation

        # setpoints
        ISPdq0_master = data[idx[2]]  # setting dq reference
        ISPabc_master = dq0_to_abc(ISPdq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = np.sum((np.abs((ISPabc_master - Iabc_master)) / iLimit) ** 0.5, axis=0) \
                + -np.sum(mu * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
                * max_episode_steps

        return -error.squeeze()


#########################################################################################
# Implementation of Load Steps
# It is checked, if quantity reaches steady state
# Values are stored in a Databuffer
# If the Databuffer fulfills several conditions, the first load steps will be implemented
# Programming 'on the fly'

class LoadstepCallback(Callback):
    def __init__(self):
        self.steady_state_reached = False
        self.settling_time = None
        self.databuffer = None
        self._idx = None
        self.interval_check = False
        self.steady_state_check = False
        self.list_data = []
        self.upper_bound = 1.02 * id_ref[0]
        self.lower_bound = 0.98 * id_ref[0]
        self.databuffer_settling_time = []

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                 [f'master.CVI{k}' for k in 'dq0']])

    def reset(self):
        self.databuffer = np.empty(20) # creates a databuffer with a length of twenty

    def __call__(self, cols, obs):
        self.set_idx(cols)  # all values are shifted to the left by one
        self.databuffer[-1] = obs[self._idx[3][0]] # last element is replaced with current value
        self.list_data.append(obs[self._idx[3][
            0]])
        self.databuffer = np.roll(self.databuffer, -1)

        # condition: it is checked whether the current is outside of the interval
        if (obs[self._idx[3][0]] < self.lower_bound or obs[
            self._idx[3][0]] > self.upper_bound) and self.steady_state_reached == False:
            self.interval_check = False

        # pre-condition: checks if observation is within the specified bound --> settling time may be reached
        if obs[self._idx[3][0]] > self.lower_bound and obs[
        self._idx[3][0]] < self.upper_bound:
            if not self.interval_check:
                global position_settling_time
                position_settling_time = len(self.list_data)
                self.interval_check = True
            if self.databuffer.std() < 0.05 and np.round(self.databuffer.mean(), decimals=1) == id_ref[
                0]:  # if the databuffer fulfills the conditions, steady state is reached
                self.steady_state_reached = True
                if not self.steady_state_check:
                    global position_steady_state
                    position_steady_state = len(self.list_data) # position steady state is returned
                    self.steady_state_check = True

    # After steady state is reached, a load jump of + 5 Ohm is implemented
    # After 3/4 of the simulation time, another random load jump of +/- 5 Ohm is implemented
    def load_step_resistance(self, t):
        for i in range(1, len(list_resistor)):
            if self.steady_state_reached == False and t <= ts * i + ts:  # no steady state, no load step
                return list_resistor[i]  # contains values for every step that fluctuate a little bit
            elif self.steady_state_reached == True and t <= ts * i + ts and i <= max_episode_steps * 0.75:
                return list_resistor[i] + movement_1_resistor  # when steady state reached, first load jump
            elif self.steady_state_reached == True and t <= ts * i + ts and i > max_episode_steps * 0.75: #75 % ts
                return list_resistor[i] + movement_2_resistor

    # After steady state is reached, a random load jump of + 5 Ohm is implemented
    # After 3/4 of the simulation time, another random load jump of +/- 0.4 mH is implemented
    def load_step_inductance(self, t):
       for i in range(1, len(list_inductance)):
            if self.steady_state_reached == False and t <= ts * i + ts:  # no load step, no steady state
                return list_inductance[i]  # contains for every step values that fluctuate a little bit
            elif self.steady_state_reached == True and t <= ts * i + ts and i <= max_episode_steps * 0.75:
                return list_inductance[i] + movement_1_inductance  # when steady state reached, first load jump
            elif self.steady_state_reached == True and t <= ts * i + ts and i > max_episode_steps * 0.75: #75 % ts
                return list_inductance[i] + movement_2_inductance


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    prior_mean = 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001 ** 2  # measurement noise sigma_omega
    prior_var = 0.1  # prior variance of the GP

    bounds = None
    lengthscale = None
    if adjust == 'Kp':
        bounds = [(0.00, 0.03)]  # bounds on the input variable Kp
        lengthscale = [.01]  # length scale for the parameter variation [Kp] for the GP

    # For 1D example, if Ki should be adjusted
    if adjust == 'Ki':
        bounds = [(0, 300)]  # bounds on the input variable Ki
        lengthscale = [50.]  # length scale for the parameter variation [Ki] for the GP

    # For 2D example, choose Kp and Ki as mutable parameters (below) and define bounds and lengthscale for both of them
    if adjust == 'Kpi':
        bounds = [(0.0, 0.03), (0, 300)]
        lengthscale = [.01, 50.]

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means. Performance measurement for optimization are seen as
    # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 2

    # The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
    # expanding points eventually.
    # The following variable is multiplied with the first performance of the initial set by the factor below:
    explore_threshold = 2

    # Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
    # limit exceeded
    abort_reward = 10

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers
    mutable_params = None
    current_dqp_iparams = None
    if adjust == 'Kp':
        # mutable_params = parameter (Kp gain of the current controller of the inverter) to be optimized using
        # the SafeOpt algorithm
        mutable_params = dict(currentP=MutableFloat(5e-2))

        # Define the PI parameters for the current controller of the inverter
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=115, limits=(-1, 1))

    # For 1D example, if Ki should be adjusted
    elif adjust == 'Ki':
        mutable_params = dict(currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=10e-3, kI=mutable_params['currentI'], limits=(-1, 1))

    # For 2D example, choose Kp and Ki as mutable parameters
    elif adjust == 'Kpi':
        mutable_params = dict(currentP=MutableFloat(5e-3), currentI=MutableFloat(10))  # setP= 10 e^-3 and setQ=10
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, net.ts, f_nom=net.freq_nom,
                                                 undersampling=2, name='master')

    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    # Arguments described above
    # History is used to store results
    agent = SafeOptAgent(mutable_params,
                         abort_reward,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold),
                         [ctrl],
                         dict(master=[[f'lc1.inductor{k}.i' for k in '123'],
                                      id_ref]),
                         history=FullHistory()
                         )


    #####################################
    # Definition of the environment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed (here, only the 3 currents across the
    #   inductors of the inverters are plotted. Labels and grid is adjusted using the PlotTmpl (For more information,
    #   see UserGuide)
    # - inputs to the models are the connection points to the inverters (see user guide for more details)
    # - model outputs are the the 3 currents through the inductors and the 3 voltages across the capacitors

    def xylables(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')


    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{\u03A9}$')
        ax.grid(which='both')

    def xylables_L(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$L_{\mathrm{123}}\,/\,\mathrm{H}$')
        ax.grid(which='both')

    def xylables_i_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{i}$')
        ax.grid(which='both')



    callback = LoadstepCallback()

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],
                                callback=xylables
                                ),
                       PlotTmpl([f'rl1.resistor{i}.R' for i in '123'],
                                callback=xylables_R
                                ),
                       PlotTmpl([f'master.CVI{s}' for s in 'dq0'],
                                ),
                       PlotTmpl([f'rl1.inductor{i}.L' for i in '123'],
                                callback=xylables_L
                                )
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   model_params={'rl1.resistor1.R': callback.load_step_resistance, # see LoadstepCallback(Callback)
                                 'rl1.resistor2.R': callback.load_step_resistance,
                                 'rl1.resistor3.R': callback.load_step_resistance,
                                 'rl1.inductor1.L': callback.load_step_inductance,
                                 'rl1.inductor2.L': callback.load_step_inductance,
                                 'rl1.inductor3.L': callback.load_step_inductance
                                 },

                   max_episode_steps=max_episode_steps,
                   net=net,
                   model_path='../omg_grid/grid.network_singleInverter.fmu',
                   history=FullHistory()
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)
    runner = Runner(agent, env, callback)

    runner.run(num_episodes, visualise=True)

    print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df[:]))

    print('\n Experiment finished with best set: \n')
    print('\n  {} = {}'.format(adjust, agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
    print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
    print('\n\nBest experiment results are plotted in the following:')

    # Show best episode measurment (current) plot
    best_env_plt = runner.run_data['best_env_plt']
    if best_env_plt:
        ax = best_env_plt[0].axes[0]
        ax.set_title('Best Episode')
        best_env_plt[0].show()
        best_env_plt[0].savefig('best_env_plt.png')

    # Show last performance plot
    best_agent_plt = runner.run_data['last_agent_plt']
    if best_agent_plt:
        ax = best_agent_plt.axes[0]
        ax.grid(which='both')
        ax.set_axisbelow(True)

        if adjust == 'Ki':
            ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
            ax.set_ylabel(r'$J$')
        elif adjust == 'Kp':
            ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
            ax.set_ylabel(r'$J$')
        elif adjust == 'Kpi':
            agent.params.reset()
            ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
            ax.set_ylabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
            ax.get_figure().axes[1].set_ylabel(r'$J$')
            plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-', zorder=1, lw=4,
                     alpha=.5)
        best_agent_plt.show()
        best_agent_plt.savefig('agent_plt.png')

df_master_CVId = env.history.df[['master.CVId']]


#####################################
# Calculation of Metrics
# Here, the d- term of id is analysed


class Metrics_id:
    ref_value = id_ref[0]  # set value of inverter that serves as a set value for the inner current controller
    ts = 1e-4  # duration of the episodes in seconds
    n = 5  # number of points to be checked before and after --> important for command 'agrelextrema'
    overshoot_available = True
    max_quantity = 0

    def __init__(self, quantity):
        self.quantity = quantity  # here the class is given any quantity to calculate the metrics
        self.test = None
        self.interval_before_load_steps = self.quantity.iloc[0:position_steady_state]

    def overshoot(self): # calculation of overshoot
        self.interval_before_load_steps['max'] = \
            self.interval_before_load_steps.iloc[
                argrelextrema(self.interval_before_load_steps['master.CVId'].values, np.greater_equal, order=self.n)[
                    0]][
                'master.CVId'] # all maxima are identified
        self.max_quantity = self.interval_before_load_steps['max'].max()
        overshoot = (self.max_quantity / self.ref_value) - 1
        if self.max_quantity > self.ref_value:
            return round(overshoot, 4)
        else:
            self.overshoot_available = False
            sentence_error1 = "No"
            return sentence_error1

    def rise_time(self): # calculation of rise time
        position_start_rise_time = self.quantity[self.quantity['master.CVId'] >= 0.1 * self.ref_value].index[
            0]  # 10% of its final value
        position_end_rise_time = self.quantity[self.quantity['master.CVId'] >= 0.9 * self.ref_value].index[
            0]  # 90% of its final value
        position_rise_time = position_end_rise_time - position_start_rise_time
        rise_time_in_seconds = position_rise_time * ts
        return round(rise_time_in_seconds, 4)

    def settling_time(self): # calculation of settling time
        if position_settling_time == 0:
            sys.exit("Steady State could not be reached. The controller need to be improved. PROGRAM EXECUTION STOP")
        settling_time_value = position_settling_time * ts  # is calculated in the class LoadstepCallback
        return settling_time_value

    def RMSE(self):
        Y_true = self.quantity.to_numpy().reshape(
            -1)  # converts and reshapes it into an array
        Y_true = Y_true[~np.isnan(Y_true)]  # drops nan
        Y_pred = [self.ref_value] * (len(
            Y_true))  # creates an list with the set value of the voltage and the length of the real voltages (Y_true)
        Y_pred = np.array(Y_pred)  # converts this list into an array
        return round(sqrt(mean_squared_error(Y_true, Y_pred)), 4)  # returns the RMSE from sklearn

    def steady_state_error(self):  # for a controller with an integral part, steady_state_error may be zero
        last_value_quantity = self.quantity.iloc[max_episode_steps - 1]  # the last value of the quantity is stored
        steady_state_error = np.abs(self.ref_value - last_value_quantity[0])  # calculation of the steady-state-error
        return round(steady_state_error, 4)


current_controller_metrics_id = Metrics_id(df_master_CVId)

d = {'Overshoot': [current_controller_metrics_id.overshoot()],
     'Rise Time/s ': [current_controller_metrics_id.rise_time()],
     'Settling Time/s ': [current_controller_metrics_id.settling_time()],
     'Root Mean Squared Error/A': [current_controller_metrics_id.RMSE()],
     'Steady State Error/A': [current_controller_metrics_id.steady_state_error()]}

df_metrics_id = pd.DataFrame(data=d).T
df_metrics_id.columns = ['Value']
print()
print('Metrics of id')
print(df_metrics_id)

######IMPORTANT FOR THE SCORING MODEL INNER LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_id0.to_pickle("./df_metrics_id_controller1.pkl")
# Maybe you need to replace 'controller1.pkl' with 'controller2.pkl'


#####################################
# Calculation of Metrics
# Here, the d- term of id is analysed

df_master_CVIq = env.history.df[['master.CVIq']]

class Metrics_Master_iq:
    ref_value = iq_ref  # set value of inverter that serves as a set value for the outer voltage controller
    ts = .1e-4  # duration of the episodes in seconds
    n = 5  # number of points to be checked before and after

    def __init__(self, quantity):
        self.quantity = quantity  # here the class is given any quantity to calculate the metrics
        self.test = None

    def RMSE(self):
        Y_true = self.quantity.to_numpy().reshape(
            -1)  # converts and reshapes it into an array with size
        Y_true = Y_true[~np.isnan(Y_true)]  # drops nan
        Y_pred = [self.ref_value] * (len(
            Y_true))  # creates an list with the set value of the voltage and the length of the real voltages (Y_true)
        Y_pred = np.array(Y_pred)  # converts this list into an array
        return round(sqrt(mean_squared_error(Y_true, Y_pred)), 4)  # returns the RMSE from sklearn

    def steady_state_error(self):
        last_value_quantity = self.quantity.iloc[max_episode_steps - 1]  # the last value of the quantity is stored
        steady_state_error = np.abs(self.ref_value - last_value_quantity[0])  # calculation of the steady-state-error
        return round(steady_state_error, 4)

    def peak(self): # absolute peak is calculated
        max_quantity = self.quantity.abs().max()
        return round(max_quantity[0], 4)


current_controller_metrics_id = Metrics_Master_iq(df_master_CVIq)

d = {'Root Mean Squared Error/A': [current_controller_metrics_id.RMSE()],
     'Steady State Error/A': [current_controller_metrics_id.steady_state_error()],
     'absolute Peak Value/A': [current_controller_metrics_id.peak()]}

df_metrics_iq = pd.DataFrame(data=d).T
df_metrics_iq.columns = ['Value']
print()
print('Metrics of iq')
print(df_metrics_iq)

######IMPORTANT FOR THE SCORING MODEL INNER LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_iqq.to_pickle("./df_metrics_iq_controller1.pkl")
# Maybe you need to replace 'controller1.pkl' with 'controller2.pkl'











# #####################################
# # Creation of plots for documentation
#
# import plot_quantities
# import matplotlib.pyplot as plt
#
# print()
# print("Hallo")
# ax = plt.axes()
# ax.arrow(0.06, 7.5, -0.033, 6.8, head_width=0.005, head_length=0.5, fc='r', ec='r')
# ax.arrow(0.06, 17.5, 0, -1.9, head_width=0.005, head_length=0.5, fc='black', ec='black')
# # plt.arrow(0.06, 7.5,-0.04,6.8, color='r', length_includes_head=True)
# plt.plot(plot_quantities.test(df_master_CVId['master.CVId'], ts)[0],
#          plot_quantities.test(df_master_CVId['master.CVId'], ts)[1])
# plt.axhline(y=id_ref[0], color='black', linestyle='-')
# plt.xlabel(r'$t\,/\,\mathrm{s}$')
# plt.ylabel('$i_{\mathrm{d}}\,/\,\mathrm{A}$')
# plt.text(0.045, 6.5, 'Steady State reached')
# plt.text(0.058, 18.0, '$i_{\mathrm{d}}^*$')
# # plt.arrow(0.06,7.5,-(0.06-0.02),14.3-7.5,color='red',arrowprops={'arrowstyle': '->'})
# #
# # plt.grid()
# # plt.arrow(0.06, 7.5,-0.04,6.8,  head_width=0.05, head_length=0.03, linewidth=4, color='r', length_includes_head=False)
# # p1 = patches.FancyArrowPatch((0.06, 7.5), (0.02, 14.3), arrowstyle='->', mutation_scale=20)
# plt.show()
#
# ax = plt.axes()
# ax.arrow(0.06, 7.5, -0.04, 6.8, head_width=0.005, head_length=0.5, fc='r', ec='r')
# # ax.arrow(0.00, 16.6,0,-0.5, head_width=0.005, head_length=0.5, fc='r', ec='r')
# ax.arrow(0.06, 17.5, 0, -1.9, head_width=0.005, head_length=0.5, fc='black', ec='black')
# ax.arrow(0.02, 5, -(0.009), (13.47 - 5.6), head_width=0.005, head_length=0.5, fc='red', ec='red')
# # plt.arrow(0.06, 7.5,-0.04,6.8, color='r', length_includes_head=True)
# plt.plot(plot_quantities.test(df_master_CVId['master.CVId'], ts)[0],
#          plot_quantities.test(df_master_CVId['master.CVId'], ts)[1], label='$i_{\mathrm{d}}$')
# plt.axhline(y=id_ref[0], color='black', linestyle='-')
# plt.xlabel(r'$t\,/\,\mathrm{s}$')
# plt.ylabel('$i_{\mathrm{dq}}\,/\,\mathrm{A}$')
# plt.text(0.045, 6.7, 'Settling Time')
# plt.text(0.058, 18.0, '$i_{\mathrm{d}}^*$')
# plt.text(0.011, 4.2, 'Rise Time')
# plt.text(0.057, 0.2, '$i_{\mathrm{q}}^*$')
# plt.plot(plot_quantities.test(df_master_CVIq['master.CVIq'], ts)[0],
#          plot_quantities.test(df_master_CVIq['master.CVIq'], ts)[1], label='$i_{\mathrm{q}}$')
# ax.legend()
# # plt.text(-0.006, 17.1, 'Overshoot')
# # plt.arrow(0.06,7.5,-(0.06-0.02),14.3-7.5,color='red',arrowprops={'arrowstyle': '->'})
# #
# # plt.grid()
# # plt.arrow(0.06, 7.5,-0.04,6.8,  head_width=0.05, head_length=0.03, linewidth=4, color='r', length_includes_head=False)
# # p1 = patches.FancyArrowPatch((0.06, 7.5), (0.02, 14.3), arrowstyle='->', mutation_scale=20)
# plt.show()
