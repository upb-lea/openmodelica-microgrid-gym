#####################################
# Example using an FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt
# Testing Framework for voltage control
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

pd.options.mode.chained_assignment = None  # default='warn'

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.execution import Callback
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

# Simulation definitions
net = Network.load('net_single-inv-curr.yaml')
max_episode_steps = 1200  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 0  # virtual droop gain for active power / W/Hz 40000
QDroopGain = 0  # virtual droop gain for reactive power / VAR/V 1000
vd_ref = np.array([325, 0, 0])  # exemplary set point i.e. vd = 325, vq = 0, v0 = 0 / A
vq_ref = 0  # vq = 0
R = 20  # resistance value / Ohm
L = 0.001  # resistance value / Henry
ts = 1e-4  # duration of episode / s
load_step_resistance = 5  # load step / Ohm
load_step_inductance = 0.0004  # sets the inductive load step / H
movement_1_resistor = load_step_resistance  # first load step is negative
movement_2_resistor = (load_step_resistance if random() < 0.5
                       else -1 * load_step_resistance) + movement_1_resistor  # randomly chosen load step
movement_1_inductance = load_step_inductance  # first load step is negative
movement_2_inductance = (load_step_inductance if movement_2_resistor >= 0
                         else -load_step_inductance) + movement_1_inductance  # same sign as resistive load step


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

    def rew_fun(self, cols: List[str], data: np.ndarray, risk) -> float:
        """
        Defines the reward function for the environment. Uses the observations and set-points to evaluate the quality
        of the used parameters.
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


#########################################################################################
# Implementation of Load Steps
# It is checked, if quantity reaches steady state
# Values are stored in a databuffer
# If the databuffer fulfills several conditions, the first load steps will be implemented
# Programming 'on the fly'

class LoadstepCallback(Callback):
    def __init__(self):
        self.steady_state_reached = False
        self.settling_time = 0
        self.databuffer = None
        self._idx = None
        self.interval_check = False
        self.steady_state_check = False
        self.list_data = []
        self.upper_bound = 1.02 * vd_ref[0]
        self.lower_bound = 0.98 * vd_ref[0]

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                 [f'lc1.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0'],
                 [f'master.CVV{i}' for i in 'dq0']])

    def reset(self):
        self.databuffer = np.empty(10)  # creates a databuffer with a length of ten

    def __call__(self, cols, obs):
        self.set_idx(cols)
        self.databuffer[-1] = obs[self._idx[5][0]]  # all values are shifted to the left by one
        self.databuffer = np.roll(self.databuffer, -1)  # all values are shifted to the left by one
        self.list_data.append(obs[self._idx[5][0]])  # current value is appended to the list

        # condition: it is checked whether the current is outside of the interval --> settling time may not be reached
        if (obs[self._idx[5][0]] < self.lower_bound or obs[
            self._idx[5][0]] > self.upper_bound) and not self.steady_state_reached:
            self.interval_check = False

        # pre-condition: checks if observation is within the specified interval --> settling time may be reached
        if self.lower_bound < obs[self._idx[5][0]] < self.upper_bound:
            if not self.interval_check:
                global position_settling_time
                position_settling_time = len(self.list_data)
                self.interval_check = True
            if self.databuffer.std() < 0.1 and np.round(self.databuffer.mean()) == \
                    vd_ref[0]:  # if the databuffer fulfills the conditions, steady state is reached
                self.steady_state_reached = True
                if not self.steady_state_check:
                    global position_steady_state
                    position_steady_state = len(self.list_data)  # stores the steady state position
                    self.steady_state_check = True

    # After steady state is reached, a fixed load step of + 5 Ohm is implemented
    # After 3/4 of the simulation time, another random load step of +/- 5 Ohm is implemented
    def load_step_resistance(self, t):
        for i in range(1, len(list_resistor)):
            if self.steady_state_reached == False and t <= ts * i + ts:  # no steady state, no load step
                return list_resistor[i]  # for every step, it contains values that fluctuate a little bit
            elif self.steady_state_reached == True and t <= ts * i + ts and i <= max_episode_steps * 0.75:
                return list_resistor[i] + movement_1_resistor  # when steady state reached, first load step
            elif self.steady_state_reached == True and t <= ts * i + ts and i > max_episode_steps * 0.75:  # 75 % ts
                return list_resistor[i] + movement_2_resistor

    # After steady state is reached, a fixed load step of + 0.4 mH is implemented
    # After 3/4 of the simulation time, another random load step of +/- 0.4 mH is implemented
    def load_step_inductance(self, t):
        for i in range(1, len(list_inductance)):
            if self.steady_state_reached == False and t <= ts * i + ts:  # no load step, no steady state
                return list_inductance[i]  # for every step, it contains values that fluctuate a little bit
            elif self.steady_state_reached == True and t <= ts * i + ts and i <= max_episode_steps * 0.75:
                return list_inductance[i] + movement_1_inductance  # when steady state reached, first load step
            elif self.steady_state_reached == True and t <= ts * i + ts and i > max_episode_steps * 0.75:  # 75 % ts
                return list_inductance[i] + movement_2_inductance


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    prior_mean = 0  # mean factor of the GP prior mean which is multiplied with the first performance of the init set
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
    # Choose Kp and Ki for the current and voltage controller as mutable parameters
    mutable_params = dict(currentP=MutableFloat(10e-3), currentI=MutableFloat(10), voltageP=MutableFloat(25e-3),
                          voltageI=MutableFloat(60))

    voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                    limits=(-iLimit, iLimit))
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for
    # the filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param = DroopParams(DroopGain, 0.005, net.freq_nom)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, net.v_nom)

    # Define a voltage forming inverter using the PIPI and droop parameters from above
    # ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, net.ts, droop_param, qdroop_param,
    # undersampling=2, name='master')
    ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param, qdroop_param,
                                       ts_sim=net.ts,
                                       ts_ctrl=2 * net.ts, name='master')
    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    agent = SafeOptAgent(mutable_params,
                         abort_reward,
                         j_min,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold),
                         ctrls=[ctrl],
                         obs_template=dict(master=[[f'lc1.inductor{k}.i' for k in '123'],
                                                   [f'lc1.capacitor{k}.v' for k in '123'],
                                                   ]),
                         history=FullHistory()
                         )


    # Arguments described above
    # History is used to store results

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
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{\u03A9}$')
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


    def xylables_L(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$L_{\mathrm{123}}\,/\,\mathrm{H}$')
        ax.grid(which='both')


    callback = LoadstepCallback()

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],
                                callback=xylables_i
                                ),
                       PlotTmpl([f'lc1.capacitor{i}.v' for i in '123'],
                                callback=xylables_v_abc
                                ),
                       PlotTmpl([f'rl1.resistor{i}.R' for i in '123'],
                                callback=xylables_R
                                ),
                       PlotTmpl([f'master.CVi{s}' for s in 'dq0'],
                                callback=xylables_i_dq0
                                ),
                       PlotTmpl([f'master.CVV{i}' for i in 'dq0'],
                                callback=xylables_v_dq0
                                ),
                       PlotTmpl([f'rl1.inductor{i}.L' for i in '123'],
                                callback=xylables_L
                                )
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   model_params={'rl1.resistor1.R': callback.load_step_resistance,
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

#####################################
# Calculation of Metrics
# Here, the d- term of vdq is analysed

df_master_CVVd = env.history.df[['master.CVVd']]

from metrics import Metrics

voltage_controller_metrics_vd = Metrics(df_master_CVVd, vd_ref[0], ts, max_episode_steps)

d = {'Overshoot': [voltage_controller_metrics_vd.overshoot()],
     'Rise Time/s ': [voltage_controller_metrics_vd.rise_time()],
     'Settling Time/s ': [voltage_controller_metrics_vd.settling_time()],
     'Root Mean Squared Error/V': [voltage_controller_metrics_vd.RMSE()],
     'Steady State Error/V': [voltage_controller_metrics_vd.steady_state_error()]}

print()
df_metrics_vd = pd.DataFrame(data=d).T
df_metrics_vd.columns = ['Value']
print('Metrics of Vd')
print(df_metrics_vd)

######IMPORTANT FOR THE SCORING MODEL INNER LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_vd.to_pickle("./df_metrics_vd_controller1.pkl")
# Maybe you need to replace 'controller1.pkl' with 'controller2.pkl'


#####################################
# Calculation of Metrics
# Here, the q- term of vdq is analysed

df_master_CVVq = env.history.df[['master.CVVq']]

from metrics import Metrics

voltage_controller_metrics_vq = Metrics(df_master_CVVq, vq_ref, ts, max_episode_steps)

d = {'Root Mean Squared Error/V': [voltage_controller_metrics_vq.RMSE()],
     'Steady State Error/V': [voltage_controller_metrics_vq.steady_state_error()],
     'absolute Peak Value/V': [voltage_controller_metrics_vq.absolute_peak()]}

print()
df_metrics_vq = pd.DataFrame(data=d).T
df_metrics_vq.columns = ['Value']
print('Metrics of Vq')
print(df_metrics_vq)

######IMPORTANT FOR THE SCORING MODEL INNER LEVEL##############################
# Use the following code, to create a pkl-File in which the Dataframe is stored
# df_metrics_vq.to_pickle("./df_metrics_vq_controller1.pkl")
# Maybe you need to replace 'controller1.pkl' with 'controller2.pkl'
