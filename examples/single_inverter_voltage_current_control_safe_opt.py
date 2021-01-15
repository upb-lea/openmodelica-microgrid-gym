#####################################
# Example using an FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt


import logging
import os
from math import sqrt
from random import random
from time import strftime, gmtime
from typing import List

import GPy
import gym
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import scipy
from gym.envs.tests.test_envs_semantics import steps
from sklearn.metrics import mean_squared_error
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt


from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.execution import Callback
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

# Simulation definitions
net = Network.load('../net/net_single-inv-curr.yaml')
max_episode_steps = 1200  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 0  # virtual droop gain for active power / W/Hz 40000
QDroopGain = 0  # virtual droop gain for reactive power / VAR/V 1000
R = 20  # sets the resistance value of RL
load_jump = 5  # defines the load jump that is implemented at every quarter of the simulation time
ts = 1e-4
v_ref_d = np.array([325, 0, 0])
v_ref_q=np.array([0, 0, 0])
movement_1 = load_jump
movement_2 = (load_jump if random() < 0.5 else -1 * load_jump) + movement_1 # randomly chosen load jump
movement_2=10
position_settling_time=0
position_steady_state=0


# The starting value of the the resistance is 20 Ohm.
# Every step and therefore the whole simulation time, it randomly changes +/- 0.01 Ohm to generate noise.


def load_step_random_walk():
    random_walk = []
    random_walk.append(R)  # R=20 Ohm
    for i in range(1, max_episode_steps):
        movement = -0.01 if random() < 0.5 else 0.01  # constant slight and random fluctuation of load
        value = random_walk[i - 1] + movement
        if value < 0:
            value = 1
        random_walk.append(value)
    return random_walk


list_resistor = load_step_random_walk()

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, r'saves_VI_control_safeopt')
os.makedirs(save_folder, exist_ok=True)


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):  # [f'lc1.inductor{k}.i' for k in '123']
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                 [f'lc1.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0']])

    def rew_fun(self, cols: List[str], data: np.ndarray) -> float:
        """
        Defines the reward function for the environment. Uses the observations and set-points to evaluate the quality of
        the used parameters.
        Takes current and voltage measurements and set-points to calculate the mean-root control error and uses a
        logarithmic barrier function in case of violating the current limit. Barrier function is adjustable using
        parameter mu.
        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        phase = data[idx[1]]  # phase from the master controller needed for transformation
        vabc_master = data[idx[3]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_dq0_master = data[idx[2]]  # setting dq current reference
        isp_abc_master = dq0_to_abc(isp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates
        vsp_dq0_master = data[idx[4]]  # setting dq voltage reference
        vsp_abc_master = dq0_to_abc(vsp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = np.sum((np.abs((isp_abc_master - iabc_master)) / iLimit) ** 0.5, axis=0) \
                + -np.sum(mu * np.log(1 - np.maximum(np.abs(iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
                + np.sum((np.abs((vsp_abc_master - vabc_master)) / net.v_nom) ** 0.5, axis=0)

        return -error.squeeze()

###########################################
#This class first identifies the steady state
#Then two load jumps are implemented to check the controller's performance

class LoadstepCallback(Callback):
    def __init__(self):
        self.steady_state_reached=False
        self.settling_time=0
        self.databuffer = None
        self._idx = None
        self.counter2 = 0
        self.counter1 = 0
        self.list_data = []
        self.upper_bound = 1.02 * v_ref_d[0]
        self.lower_bound = 0.98 * v_ref_d[0]

    def set_idx(self, obs):  # [f'lc1.inductor{k}.i' for k in '123']
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                 [f'lc1.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0'], [f'master.CVV{i}' for i in 'dq0']])

    def reset(self):
        self.databuffer = np.empty(10)

    def __call__(self, cols, obs):
        self.set_idx(cols)
        self.databuffer[-1]=obs[self._idx[5][0]] # the last element of the array is replaced with the current value of the inverter's voltage
        self.databuffer=np.roll(self.databuffer, -1) # all values are shifted to the left by one
        self.list_data.append(obs[self._idx[5][0]])
        if (obs[self._idx[5][0]] < self.lower_bound or obs[self._idx[5][0]] > self.upper_bound) and self.steady_state_reached == False:
            self.counter2 = 0

        if obs[self._idx[5][0]]>self.lower_bound and obs[self._idx[5][0]]<self.upper_bound:
            if self.counter2 <1:
                global position_settling_time
                position_settling_time=len(self.list_data)
                self.counter2 = +1
            if self.databuffer.std() < 0.1 and np.round(self.databuffer.mean())==v_ref_d[0]:  # i_ref[0]= 15 A
                self.steady_state_reached=True
                if self.counter1<1:
                    global position_steady_state
                    position_steady_state=len(self.list_data)
                    self.counter1=+1
        # if the ten values in the databuffer fulfill the conditions (std <0.1 and
            # rounded mean == 325 V), the the steady state is reached

    # After steady state is reached, a bigger random load jump of +/- 5 Ohm is implemented
    # After 3/4 of the simulation time, another random load jump of +/- 5 Ohm is implemented
    def load_step(self, t):
        for i in range(1, len(list_resistor)):
            if self.steady_state_reached==False and t <= ts * i + ts: # while steady state is not reached, no load jump
                return list_resistor[i] # contains for every step values that fluctuate a little bit
            elif self.steady_state_reached==True and t<=ts*i+ts and i<=max_episode_steps*0.75:
                return list_resistor[i] + movement_1 # when steady state reached, a load jump of +5/-5 Ohm is implemented (movement 1)
            elif self.steady_state_reached==True and t<=ts * i+ts and i>max_episode_steps*0.75:
               # after 75% of the simulation time, another load jump is implemented
                return list_resistor[i] + movement_2 # the load jump of +/- 5 Ohm is stored in movement_2


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    prior_mean = 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001 ** 2  # measurement noise sigma_omega
    prior_var = 2  # prior variance of the GP

    # Choose Kp and Ki (current and voltage controller) as mutable parameters (below) and define bounds and lengthscale
    # for both of them
    bounds = [(0.0, 0.03), (0, 300), (0.0, 0.03),
              (0, 300)]  # bounds on the input variable current-Ki&Kp and voltage-Ki&Kp
    lengthscale = [.005, 50., .005,
                   50.]  # length scale for the parameter variation [current-Ki&Kp and voltage-Ki&Kp] for the GP

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means: performance measurement for optimization are seen as
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
    # Choose Kp and Ki for the current and voltage controller as mutable parameters
    mutable_params = dict(currentP=MutableFloat(10e-3), currentI=MutableFloat(10), voltageP=MutableFloat(25e-3),
                          voltageI=MutableFloat(60))

    voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                    limits=(-iLimit, iLimit))
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param = DroopParams(DroopGain, 0.005, net.freq_nom)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, net.v_nom)

    # Define a voltage forming inverter using the PIPI and droop parameters from above
    ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, net.ts, droop_param, qdroop_param,
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
                                      [f'lc1.capacitor{k}.v' for k in '123'],
                                      ]),
                         history=FullHistory()
                         )


    #####################################
    # Definition of the environment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed.
    #   Labels and grid is adjusted using the PlotTmpl (For more information, see UserGuide)
    #   generated figures are stored to file
    # - inputs to the models are the connection points to the inverters (see user guide for more details)
    # - model outputs are the 3 currents through the inductors and the 3 voltages across the capacitors

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


    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')  # zeit
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{\u03A9}$')  # widerstände definieren , ohm angucken backslash omega
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/Resistor_Course_Load_Step' + time + '.pdf')


    def xylables_i_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{i}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/dq0_current' + time + '.pdf')


    def xylables_v_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        fig.savefig(save_folder + '/dq0_voltage' + time + '.pdf')


    callback = LoadstepCallback()

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],  # 123
                                callback=xylables_i
                                ),
                       PlotTmpl([f'lc1.capacitor{i}.v' for i in '123'],  # 123
                                callback=xylables_v_abc
                                ),
                       PlotTmpl([f'rl1.resistor{i}.R' for i in '123'],  # Plot Widerstand RL
                                callback=xylables_R
                                ),
                       PlotTmpl([f'master.CVi{s}' for s in 'dq0'],  # Plot Widerstand RL I=s
                                callback=xylables_i_dq0
                                ),
                       PlotTmpl([f'master.CVV{i}' for i in 'dq0'],
                                callback=xylables_v_dq0
                                )
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   model_params={'rl1.resistor1.R': callback.load_step,
                                 'rl1.resistor2.R': callback.load_step,
                                 'rl1.resistor3.R': callback.load_step,
                                 'rl1.inductor1.L': 0.001,
                                 'rl1.inductor2.L': 0.001,
                                 'rl1.inductor3.L': 0.001
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

    #####################################
    # Performance results and parameters as well as plots are stored in folder pipi_signleInvALT
    agent.history.df.to_csv(save_folder + '/result.csv')

    print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df.round({'J': 4, 'Params': 4})))
    print('\n Experiment finished with best set: \n')
    print('\n  Current-Ki&Kp and voltage-Ki&Kp = {}'.format(
        agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
    print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
    print('\n\nBest experiment results are plotted in the following:')

    # Show best episode measurment (current) plot
    best_env_plt = runner.run_data['best_env_plt']
    for ii in range(len(best_env_plt)):
        ax = best_env_plt[ii].axes[0]
        ax.set_title('Best Episode')
        best_env_plt[ii].show()
        # best_env_plt[0].savefig('best_env_plt.png')

# Data of the currents/voltages to be analyzed with the help of Pandas/Numpy

df_inductor_i = env.history.df[['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i']]
pd.set_option('display.max_rows', None)
df_master_CVVd = env.history.df[
    ['master.CVVd']]  # voltage for the Controller, it is the measurement of the Master-Inverter's Voltage7


##############################################
# Implementation of the class Metrics

class Metrics:
    ref_value = v_ref_d[0]  # set value of inverter that serves as a set value for the outer voltage controller
    ts = 1e-4  # duration of the episodes in seconds
    n=5 # number of points to be checked before and after
    overshoot_available=True



    def __init__(self, quantity):
        self.quantity = quantity  # here the class is given any quantity to calculate the metrics
        self.test=None
        self.interval_before_load_steps = self.quantity.iloc[0:position_steady_state]

    def overshoot(self):
        self.interval_before_load_steps['max'] = \
            self.interval_before_load_steps.iloc[
                argrelextrema(self.interval_before_load_steps['master.CVVd'].values, np.greater_equal, order=self.n)[
                    0]][
                'master.CVVd']
        self.max_quantity = self.interval_before_load_steps['max'].max()
        overshoot = (self.max_quantity / self.ref_value) - 1
        if self.max_quantity > self.ref_value:
            return round(overshoot, 4)
        else:
            self.overshoot_available = False
            sentence_error1 = "No"
            return sentence_error1

    def rise_time(self):
        if self.overshoot_available: # if overdamped system
            position_rise_time = self.quantity[self.quantity['master.CVVd'] >= self.ref_value].index[0]  # the number of episode steps where the real value = set_value is stored
            rise_time_in_seconds = position_rise_time * ts
        else: #if underdamped/critically damped
            position_start_rise_time=self.quantity[self.quantity['master.CVVd'] >= 0.1*self.ref_value].index[0] # 5% of its final value
            position_end_rise_time= self.quantity[self.quantity['master.CVVd'] <= 0.9*self.ref_value].index[0] # 95% of its final value
            position_rise_time=position_end_rise_time-position_start_rise_time
            rise_time_in_seconds=position_rise_time *ts


        return round(rise_time_in_seconds, 4)

    def settling_time(self):
        settling_time_value = position_settling_time * ts
        return settling_time_value
        # if self.overshoot_available:
        #     self.quantity['max'] = \
        #     self.quantity.iloc[argrelextrema(self.quantity['master.CVVd'].values, np.greater_equal, order=self.n)[0]][
        #     'master.CVVd']
        #     index_first_max = self.quantity['max'].first_valid_index() #returns the index of the maximum that is used for the calculation
        #     self.quantity.drop(columns=['max'])
        #     array_settling_time = self.quantity.iloc[int(index_first_max):]
        # else:
        #     array_settling_time=self.quantity['master.CVVd']# stores the values after the second load jump (a little delay +6) in an array
        # upper_bound = 1.02 * self.ref_value  # the settling time is defined as the time when the inverter's voltage is
        # lower_bound = 0.98 * self.ref_value  # within the upper and lower bound of the set value for the first time at Steady State
        # position_settling_time = array_settling_time[(array_settling_time['master.CVVd'] >= lower_bound) & (
        #         array_settling_time['master.CVVd'] <= upper_bound)].index[
        #     0]  # returns the position of the settling time
        # settling_time_in_seconds = position_settling_time * ts
        # settling_time_df= self.quantity['master.CVVd'].iloc[position_settling_time:(position_settling_time + 20)] #array
        # if (settling_time_df.mean() < upper_bound) & (settling_time_df.mean() > lower_bound): #checks if the quantity is still in steady state after the settling_time
        #     return round(settling_time_in_seconds, 4)
        # else:
        #     sentence_error2="The settling time could not be identified"
        #     return sentence_error2


    def RMSE(self):
        Y_true = self.quantity.to_numpy().reshape(-1)  # converts and reshapes it into an array with size [max_epsiodes,]
        Y_true = Y_true[~np.isnan(Y_true)]  # drops nan
        Y_pred = [self.ref_value] * (len(Y_true))  # creates an list with the set value of the voltage and the length of the real voltages (Y_true)
        Y_pred = np.array(Y_pred)  # converts this list into an array
        return round(sqrt(mean_squared_error(Y_true, Y_pred)), 4)  # returns the MSE from sklearn

    def steady_state_error(self):  # for a Controller with an integral part, steady_state_error is almost equal to zero
        last_value_quantity = self.quantity.iloc[max_episode_steps-1]  # the last value of the quantity is stored
        steady_state_error = np.abs(self.ref_value - last_value_quantity[0])  # calculation of the steady-state-error
        return round(steady_state_error, 4)


voltage_controller_metrics_vd0 = Metrics(df_master_CVVd)


d = {'Overshoot': [voltage_controller_metrics_vd0.overshoot()],
     'Rise Time/s ': [voltage_controller_metrics_vd0.rise_time()],
     'Settling Time/s ': [voltage_controller_metrics_vd0.settling_time()],
     'Root Mean Squared Error/V': [voltage_controller_metrics_vd0.RMSE()],
     'Steady State Error/V': [voltage_controller_metrics_vd0.steady_state_error()]}

print()
df_metrics_vd0 = pd.DataFrame(data=d).T
df_metrics_vd0.columns = ['Value']
print('Metrics of Vd0')
print(df_metrics_vd0)
################################################################################
###q-component of vdq0

df_master_CVVq = env.history.df[['master.CVVq']]
class Metrics_Master_Vq0:
    ref_value = v_ref_q[0]  # set value of inverter that serves as a set value for the outer voltage controller
    ts = .5e-4  # duration of the episodes in seconds
    n = 5  # number of points to be checked before and after

    def __init__(self, quantity):
        self.quantity = quantity  # here the class is given any quantity to calculate the metrics
        self.test = None


    def RMSE(self):
        Y_true = self.quantity.to_numpy().reshape(
            -1)  # converts and reshapes it into an array with size [max_epsiodes,]
        Y_true = Y_true[~np.isnan(Y_true)]  # drops nan
        Y_pred = [self.ref_value] * (len(
            Y_true))  # creates an list with the set value of the voltage and the length of the real voltages (Y_true)
        Y_pred = np.array(Y_pred)  # converts this list into an array
        return round(sqrt(mean_squared_error(Y_true, Y_pred)), 4)  # returns the MSE from sklearn

    def steady_state_error(self):  # for a Controller with an integral part, steady_state_error is almost equal to zero
        last_value_quantity = self.quantity.iloc[max_episode_steps - 1]  # the last value of the quantity is stored
        steady_state_error = np.abs(self.ref_value - last_value_quantity[0])  # calculation of the steady-state-error
        return round(steady_state_error, 4)

    def peak(self):
        max_quantity=self.quantity.abs().max()
        return round(max_quantity[0],4)


voltage_controller_metrics_vq0 = Metrics_Master_Vq0(df_master_CVVq)

d = {'Root Mean Squared Error/V': [voltage_controller_metrics_vq0.RMSE()],
     'Steady State Error/V': [voltage_controller_metrics_vq0.steady_state_error()],
     'Peak Value/V': [voltage_controller_metrics_vq0.peak()]}

print()
df_metrics_vq0 = pd.DataFrame(data=d).T
df_metrics_vq0.columns = ['Value']
print('Metrics of Vq0')
print(df_metrics_vq0)


#####################################
# Creation of plots for documentation

import plot_quantities
import matplotlib.pyplot as plt
ax = plt.axes()
#ax.arrow(0.00, 16.6,0,-0.5, head_width=0.005, head_length=0.5, fc='r', ec='r')
#ax.arrow(0.06, 17.5,0,-1.9, head_width=0.005, head_length=0.5, fc='black', ec='black')
#plt.arrow(0.06, 7.5,-0.04,6.8, color='r', length_includes_head=True)
plt.plot(plot_quantities.test(df_master_CVVd['master.CVVd'], ts)[0], plot_quantities.test(df_master_CVVd['master.CVVd'], ts)[1], label='$V_{\mathrm{d0}}$')
plt.plot(plot_quantities.test(df_master_CVVq['master.CVVq'], ts)[0], plot_quantities.test(df_master_CVVq['master.CVVq'], ts)[1], label='$V_{\mathrm{q0}}$')
plt.axhline(y=v_ref_d[0], color='black', linestyle='-')
plt.xlabel(r'$t\,/\,\mathrm{s}$')
plt.ylabel('$V_{\mathrm{dq0}}\,/\,\mathrm{V}$')
ax.arrow(0.02, 150,-0.015,170, head_width=0.005, head_length=5, fc='r', ec='r')
plt.text(0.01,130,'Rise Time')
ax.arrow(0.06, 150,-0.043,162, head_width=0.005, head_length=7, fc='r', ec='r')
plt.text(0.05,130,'Settling Time')
ax.arrow(0.06, 350,0,-17, head_width=0.005, head_length=7, fc='black', ec='black')
plt.text(0.057,360,'$V_{\mathrm{d0}}^*$')
plt.text(0.057,5,'$V_{\mathrm{q0}}^*$')

plt.text(-0.002, 360, 'Overshoot')
ax.legend()
#plt.arrow(0.06,7.5,-(0.06-0.02),14.3-7.5,color='red',arrowprops={'arrowstyle': '->'})
#
#plt.grid()
#plt.arrow(0.06, 7.5,-0.04,6.8,  head_width=0.05, head_length=0.03, linewidth=4, color='r', length_includes_head=False)
#p1 = patches.FancyArrowPatch((0.06, 7.5), (0.02, 14.3), arrowstyle='->', mutation_scale=20)
plt.show()