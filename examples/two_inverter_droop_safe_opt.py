#####################################
# Example using an FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Two inverters (one voltage forming, one current sourcing) supplying an RL-load via an LC-filter
# Controller: droop controller gains are optimized by SafeOpt


import logging
import os
from functools import partial
from time import strftime, gmtime
from typing import List

import GPy
import gym
import numpy as np

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, \
    MultiPhaseDQ0PIPIController, PLLParams, InverseDroopParams, MultiPhaseDQCurrentController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import nested_map, FullHistory

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 6000  # number of simulation steps per episode
num_episodes = 10  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 1000  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VA/V

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, r'saves_droop_control_safeopt')
os.makedirs(save_folder, exist_ok=True)


def load_step(t, gain):
    """
    Defines a load step after 0.15 s
    Increases the load parameters by a factor of 5
    :param t: time
    :param gain: device parameter
    :return: Dictionary with load parameters
    """
    return 1 * gain if t < .15 else 5 * gain


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


if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    #####################################
    # Definitions for the GP
    prior_mean = 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001 ** 2  # measurement noise sigma_omega
    prior_var = 0.2  # prior variance of the GP

    # Choose all droop params as mutable parameters (below) and define bounds and lengthscale for both of them
    bounds = [(0, 100000), (0, 100000), (0, 3000), (0, 100)]  # bounds on the input variable
    lengthscale = [10000, 10000, 300., 10.]  # length scale for the parameter variation for the GP

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

    # Choose all droop parameter as mutable parameters
    mutable_params = dict(pDroop_master=MutableFloat(30000.0), pDroop_slave=MutableFloat(30000.0), \
                          qDroop_master=MutableFloat(QDroopGain), qDroop_slave=MutableFloat(10))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for
    # the filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param_master = DroopParams(mutable_params['pDroop_master'], 0.005, nomFreq)
    # droop parameter used to react on frequency drop
    droop_param_slave = InverseDroopParams(mutable_params['pDroop_slave'], delta_t, nomFreq, tau_filt=0.04)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    # qDroop parameter used for virtual voltage drop
    qdroop_param_master = DroopParams(mutable_params['qDroop_master'], 0.002, nomVoltPeak)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param_slave = InverseDroopParams(mutable_params['qDroop_slave'], delta_t, nomVoltPeak, tau_filt=0.01)

    ###############
    # define Master
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    # Current control PI gain parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param_master,
                                            qdroop_param_master,
                                            undersampling=2, name='master'))

    ###############
    # define slave
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI gain parameters for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop characteristic for the active power Watts/Hz, W.s/Hz

    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit, droop_param_slave,
                                              qdroop_param_slave, name='slave'))

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


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],
                                callback=xylables_i
                                ),
                       PlotTmpl([f'lc1.capacitor{i}.v' for i in '123'],
                                callback=xylables_v_abc
                                ),
                       PlotTmpl([f'master.instPow'],
                                callback=xylables_P_master
                                ),
                       PlotTmpl([f'slave.instPow'],
                                callback=xylables_P_slave
                                ),
                       PlotTmpl([f'slave.freq'],
                                callback=xylables_freq
                                ),
                       PlotTmpl([f'master.CVV{i}' for i in 'dq0'],
                                callback=xylables_v_dq0
                                )
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   model_params={'rl1.resistor1.R': partial(load_step, gain=20),
                                 'rl1.resistor2.R': partial(load_step, gain=20),
                                 'rl1.resistor3.R': partial(load_step, gain=20),
                                 'rl1.inductor1.L': partial(load_step, gain=0.001),  # 0.001,
                                 'rl1.inductor2.L': partial(load_step, gain=0.001),  # 0.001,
                                 'rl1.inductor3.L': partial(load_step, gain=0.001)  # 0.001
                                 },
                   model_path='../omg_grid/grid.network.fmu',
                   net='../net/net.yaml',
                   history=FullHistory()
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)
    runner = Runner(agent, env)

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
