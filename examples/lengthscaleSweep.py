#####################################
# Example using a FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller gain parameters are optimized by SafeOpt


import logging
from time import strftime, gmtime
from typing import List
from tqdm import tqdm

import GPy
import gym
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.physical_testbench import TestbenchEnv
from openmodelica_microgrid_gym.execution.runner_hardware import RunnerHardware
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

# Choose which controller parameters should be adjusted by SafeOpt.
# - Kp: 1D example: Only the proportional gain Kp of the PI controller is adjusted
# - Ki: 1D example: Only the integral gain Ki of the PI controller is adjusted
# - Kpi: 2D example: Kp and Ki are adjusted simultaneously

adjust = 'Kpi'

# Check if really only one simulation scenario was selected
if adjust not in {'Kp', 'Ki', 'Kpi'}:
    raise ValueError("Please set 'adjust' to one of the following values: 'Kp', 'Ki', 'Kpi'")

include_simulate = True
do_measurement = False

ii_steps = 18
kk_steps = 20

lengthscale_vec_kI = np.linspace(5, 200, ii_steps)
lengthscale_vec_kP = np.linspace(0.1, 6, kk_steps)
unsafe_vec = np.zeros([kk_steps, ii_steps])

# Simulation definitions
delta_t = 1e-4  # simulation time step size / s
max_episode_steps = 1000  # number of simulation steps per episode
num_episodes = 200  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 60  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
i_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A
i_noise = np.array([[0.0, 0.0822], [0.0, 0.103], [0.0, 0.136]])

# Controller layout due to magniitude optimum:
L = 2.2e-3  # / H
R = 585e-3  # / Ohm
tau_plant = L / R
gain_plant = 1 / R

# take inverter into account uning s&h (exp(-s*delta_T/2))

Tn = tau_plant  # Due to compensate
Kp_init = tau_plant / (2 * delta_t * gain_plant * v_DC)
Ki_init = Kp_init / (Tn)


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'rl.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0']])

    def rew_fun(self, cols: List[str], data: np.ndarray) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
        used parameters.
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


if __name__ == '__main__':

    for kk in range(len(lengthscale_vec_kP)):  # tqdm(range(len(lengthscale_vec_kP)), desc='episodes', unit='epoch'):

        for ii in range(
                len(lengthscale_vec_kI)):  # tqdm(range(len(lengthscale_vec_kI)), desc='episodes', unit='epoch'):
            #####################################
            # Definitions for the GP
            prior_mean = 0  # 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
            noise_var = 0.001  # 0.001 ** 2  # measurement noise sigma_omega
            prior_var = 2  # 0.1  # prior variance of the GP

            bounds = None
            lengthscale = None
            if adjust == 'Kp':
                bounds = [(0, 10)]  # bounds on the input variable Kp
                lengthscale = [.75]  # length scale for the parameter variation [Kp] for the GP

            # For 1D example, if Ki should be adjusted
            if adjust == 'Ki':
                bounds = [(0, 30)]  # bounds on the input variable Ki
                lengthscale = [5.]  # length scale for the parameter variation [Ki] for the GP

            # For 2D example, choose Kp and Ki as mutable parameters (below) and define bounds and lengthscale for both of them
            if adjust == 'Kpi':
                bounds = [(0.0, 8), (0, 100)]
                # lengthscale = [lengthscale_vec[ll], 20.]
                lengthscale = [lengthscale_vec_kP[kk], lengthscale_vec_kI[ii]]

            # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
            # the initial performance: safe_threshold = 1.2 means. Performance measurement for optimization are seen as
            # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
            # parameter set
            safe_threshold = 0.5

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
                mutable_params = dict(currentP=MutableFloat(0.01))  # 5e-3))

                # Define the PI parameters for the current controller of the inverter
                current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=115, limits=(-1, 1))

            # For 1D example, if Ki should be adjusted
            elif adjust == 'Ki':
                mutable_params = dict(currentI=MutableFloat(20))
                current_dqp_iparams = PI_params(kP=0.01, kI=mutable_params['currentI'], limits=(-1, 1))

            # For 2D example, choose Kp and Ki as mutable parameters
            elif adjust == 'Kpi':
                mutable_params = dict(currentP=MutableFloat(Kp_init), currentI=MutableFloat(Ki_init))
                current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'],
                                                limits=(-1, 1))

            # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
            # filter and the nominal frequency
            # Droop controller used to calculate the virtual frequency drop due to load changes
            droop_param = DroopParams(DroopGain, 0.005, nomFreq)

            # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
            # filter and the nominal voltage
            qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

            # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
            ctrl = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param,
                                                         undersampling=1, name='master')

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
                                 dict(master=[[f'rl.inductor{k}.i' for k in '123'],
                                              [f'rl.inductor{k}.i' for k in '123'],
                                              i_ref]),
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

            if include_simulate:

                def xylables(fig):
                    ax = fig.gca()
                    ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
                    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
                    plt.title('Simulation')
                    ax.grid(which='both')
                    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    fig.savefig('len_search/abc_current' + time + '.pdf')


                def xylables_dq0(fig):
                    ax = fig.gca()
                    ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
                    ax.set_ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
                    ax.grid(which='both')
                    plt.title('Simulation')
                    plt.ylim(0, 36)
                    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    fig.savefig('len_search/dq0_current' + time + '.pdf')


                def xylables_mdq0(fig):
                    ax = fig.gca()
                    ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
                    ax.set_ylabel('$m_{\mathrm{dq0}}\,/\,\mathrm{}$')
                    plt.title('Simulation')
                    ax.grid(which='both')
                    # plt.ylim(0,36)


                env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                               reward_fun=Reward().rew_fun,
                               time_step=delta_t,
                               viz_cols=[
                                   PlotTmpl([f'rl.inductor{i}.i' for i in '123'],
                                            callback=xylables
                                            ),
                                   PlotTmpl([f'master.CVI{i}' for i in 'dq0'],
                                            callback=xylables_dq0
                                            )
                                   # PlotTmpl([f'master.m{i}' for i in 'dq0'],
                                   #         callback=xylables_mdq0
                                   #         ),
                                   # PlotTmpl([f'master.m{i}' for i in 'abc'],
                                   #         #callback=xylables_dq0
                                   #         )
                               ],
                               # viz_cols = ['inverter1.*', 'rl.inductor1.i'],
                               log_level=logging.INFO,
                               viz_mode=None,
                               max_episode_steps=max_episode_steps,
                               # model_params={'inverter1.gain.u': v_DC},
                               model_path='../fmu/grid.testbench_SC.fmu',
                               model_input=['i1p1', 'i1p2', 'i1p3'],
                               model_output=dict(rl=[['inductor1.i', 'inductor2.i', 'inductor3.i']],
                                                 # inverter1=['inductor1.i', 'inductor2.i', 'inductor3.i']
                                                 ),
                               history=FullHistory(),
                                measurement_noise = i_noise

                               )

                runner = Runner(agent, env)

                runner.run(num_episodes, visualise=True)

                print(agent.unsafe)

                if agent.unsafe:
                    unsafe_vec[kk, ii] = 1
                else:
                    unsafe_vec[kk, ii] = 0

                df = pd.DataFrame([[unsafe_vec],
                                   [lengthscale_vec_kP],
                                   [lengthscale_vec_kI]])  # ,
                # 'lengthscale_vec_kP': lengthscale_vec_kP,
                # 'lengthscale_vec_kI': lengthscale_vec_kI})
                df.to_pickle('Unsafe_matrix')

                # print(lengthscale_vec[ll])
                # print(' ')
                print(unsafe_vec)
                """
                #####################################
                # Performance results and parameters as well as plots are stored in folder pipi_signleInv
                agent.history.df.to_csv('len_search/result.csv')

                env.history.df.to_pickle('Simulation')

                # Show best episode measurment (current) plot
                best_env_plt = runner.run_data['best_env_plt']
                ax = best_env_plt[0].axes[0]
                ax.set_title('Best Episode')
                best_env_plt[0].show()
                #best_env_plt[0].savefig('best_env_plt.png')

                # Show worst episode measurment (current) plot
                best_env_plt = runner.run_data['worst_env_plt']
                ax = best_env_plt[0].axes[0]
                ax.set_title('Worst Episode')
                best_env_plt[0].show()
                #best_env_plt[0].savefig('worst_env_plt.png')

                """
                best_agent_plt = runner.run_data['last_agent_plt']
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
                    ax.set_ylabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
                    ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
                    ax.get_figure().axes[1].set_ylabel(r'$J$')
                    ax.set_title('lengthscale {}'.format([lengthscale_vec_kP[kk],lengthscale_vec_kI[ii]]))
                    plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-',
                             zorder=1, lw=4,
                             alpha=.5)

                best_agent_plt.show()
                best_agent_plt.savefig('len_search/agent_plt{}_{}.png'.format(kk, ii))

            if do_measurement:
                #####################################
                # Execution of the experiment
                # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)

                env = TestbenchEnv(num_steps=max_episode_steps, DT=1 / 20000, i_ref=i_ref[0])
                runner = RunnerHardware(agent, env)

                runner.run(num_episodes, visualise=True)

                print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df[:]))

                print('\n Experiment finished with best set: \n')
                print('\n  {} = {}'.format(adjust, agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
                print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
                print('\n\nBest experiment results are plotted in the following:')

                # # Show best episode measurment (current) plot
                # best_env_plt = runner.run_data['best_env_plt']
                # ax = best_env_plt[0].axes[0]
                # ax.set_title('Best Episode')
                # best_env_plt[0].show()
                # best_env_plt[0].savefig('best_env_plt.png')

                agent.history.df.to_csv('hardwareTest_plt/result.csv')

                # Show last performance plot
                best_agent_plt = runner.run_data['last_agent_plt']
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
                    ax.set_ylabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
                    ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
                    ax.get_figure().axes[1].set_ylabel(r'$J$')
                    plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-',
                             zorder=1, lw=4,
                             alpha=.5)
                best_agent_plt.show()
                best_agent_plt.savefig('hardwareTest_plt/agent_plt.png')

