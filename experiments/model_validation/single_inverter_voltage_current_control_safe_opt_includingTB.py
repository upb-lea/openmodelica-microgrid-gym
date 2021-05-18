#####################################
# Experiment : Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt
# a) FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# b) connecting via ssh to a testbench to perform real-world measurement

import logging
import os
from functools import partial
from itertools import tee

import GPy
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.model_validation.env.testbench_voltage_ctrl import TestbenchEnvVoltage
from experiments.model_validation.execution.monte_carlo_runner import MonteCarloRunner
from experiments.model_validation.execution.runner_hardware import RunnerHardwareGradient
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, \
    MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.plotmanager import PlotManager
from experiments.model_validation.env.rewards import Reward
from experiments.model_validation.env.stochastic_components import Load
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import FullHistory

# Plot setting
params = {'backend': 'ps',
          'text.latex.preamble': [r'\usepackage{gensymb}'
                                  r'\usepackage{amsmath,amssymb,mathtools}'
                                  r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                  r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
          'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
          'axes.titlesize': 8,
          'font.size': 8,  # was 10
          'legend.fontsize': 8,  # was 10
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': [3.9, 3.1],
          'font.family': 'serif',
          'lines.linewidth': 1
          }
matplotlib.rcParams.update(params)

include_simulate = True
show_plots = True
balanced_load = False
do_measurement = False
save_results = False

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, r'VSim_rebase2_MC3')
os.makedirs(save_folder, exist_ok=True)

np.random.seed(1)

# Simulation definitions
net = Network.load('../../net/net_single-inv-Paper_Loadstep.yaml')
delta_t = 1e-4  # simulation time step size / s
undersample = 1
max_episode_steps = 2000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
n_MC = 1  # number of Monte-Carlo samples for simulation - samples device parameters (e.g. L,R, noise) from
v_DC = 600  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 60  # nominal grid frequency / Hz
nomVoltPeak = 169.7  # 230 * 1.414  # nominal grid voltage / V
iLimit = 16  # inverter current limit / A
iNominal = 12  # nominal inverter current / A
vNominal = 190  # nominal inverter current / A
vLimit = vNominal * 1.5  # inverter current limit / A
funnelFactor = 0.02
vFunnel = np.array([vNominal * funnelFactor, vNominal * funnelFactor, vNominal * funnelFactor])
mu = 400  # factor for barrier function (see below)
DroopGain = 0.0  # virtual droop gain for active power / W/Hz
QDroopGain = 0.0  # virtual droop gain for reactive power / VAR/V

# plant
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F
R = 28  # nomVoltPeak / 7.5   # / Ohm

phase_shift = 5
amp_dev = 1.1

# Observer matrices
A = np.array([[-R_filter, -1 / L_filter, 0],
              [1 / C_filter, 0, -1 / C_filter],
              [0, 0, 0]])

B = np.array([[1 / L_filter, 0, 0]]).T

C = np.array([[1, 0, 0],
              [0, 1, 0]])

# Observer values
L_iL_iL = 2e3
L_vc_iL = -435  # influence from delta_y_vc onto xdot = iL
L_iL_vc = 100229
L_vc_vc = 4000
L_iL_io = -13.22
L_vc_io = -80

L = np.array([[L_iL_iL, L_vc_iL],
              [L_iL_vc, L_vc_vc],
              [L_iL_io, L_vc_io]])


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def cal_J_min(phase_shift, amp_dev):
    """
    Calulated the miminum performance for safeopt
    Best case error of all safe boundary scenarios is used (max) to indicate which typ of error tears
    the safe boarder first (the weakest link in the chain)
    """

    ph_list = [phase_shift, 0]
    amp_list = [1, amp_dev]
    return_Jmin = np.empty(len(ph_list))
    error_Jmin = np.empty(3)
    ph_shift = [0, 120, 240]
    t = np.linspace(0, max_episode_steps * delta_t, max_episode_steps)

    grad = 0.3
    irefs = [0, nomVoltPeak, nomVoltPeak]
    ts = [0, max_episode_steps // 2, max_episode_steps]

    noiseH = 0.02 * np.sin(2 * np.pi * 1500 * t)  # add noise for dev return

    for l in range(len(ph_list)):
        for p in range(3):
            amplitudeSP = np.concatenate([np.full(t1 - t0, r1)
                                          for (r0, t0), (r1, t1) in pairwise(zip(irefs, ts))])
            amplitude = np.concatenate(
                [np.minimum(
                    r0 + grad * np.arange(0, t1 - t0),  # ramp up phase
                    np.full(t1 - t0, r1)  # max amplitude
                ) for (r0, t0), (r1, t1) in pairwise(zip(irefs, ts))])
            Mess = noiseH * amplitude + amp_list[l] * amplitude * np.cos(
                2 * np.pi * 60 * t + (ph_list[l] * np.pi / 180) + (ph_shift[p] * np.pi / 180))
            SP = amplitudeSP * np.cos(2 * np.pi * 60 * t + (ph_shift[p] * np.pi / 180))
            error_Jmin[p] = -np.sum((np.abs((SP - Mess)) / vLimit) ** 0.5, axis=0) / max_episode_steps
            w2 = noiseH * amplitude + amplitude
            dw2 = np.gradient(w2)
            SP_sattle = (amplitude > amplitudeSP * (1 - 0.12)).astype(int)
            error2 = -np.mean(abs(SP_sattle * dw2))
            error_Jmin[p] += error2 * .5  # add gradient error
        return_Jmin[l] = np.sum(error_Jmin)  # Sum all 3 phases

    return max(return_Jmin)


if __name__ == '__main__':

    rew = Reward(i_limit=iLimit, i_nominal=iNominal, mu_v=mu, max_episode_steps=max_episode_steps,
                 v_limit=vLimit, v_nominal=vNominal,
                 obs_dict=[[f'lc.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                           [f'lc.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0'],
                           [f'master.CVV{k}' for k in 'dq0']])

    #####################################
    # Definitions for the GP
    prior_mean = 0  # 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001  # ** 2  # measurement noise sigma_omega
    prior_var = 2  # prior variance of the GP

    # Choose Kp and Ki (current and voltage controller) as mutable parameters (below) and define bounds and lengthscale
    # for both of them
    bounds = [(0.000, 0.045), (4, 450)]  # bounds on the input variable current-Ki&Kp and voltage-Ki&Kp
    lengthscale = [.003, 50.]  # length scale for the parameter variation [current-Ki&Kp and voltage-Ki&Kp] for the GP

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means: performance measurement for optimization are seen as
    # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 0
    j_min = cal_J_min(phase_shift, amp_dev)  # cal min allowed performance

    # The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
    # expanding points eventually.
    # The following variable is multiplied with the first performance of the initial set by the factor below:
    explore_threshold = 0

    # Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
    # limit exceeded
    abort_reward = 100 * j_min

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers
    # Choose Kp and Ki for the current and voltage controller as mutable parameters
    mutable_params = dict(voltageP=MutableFloat(0.0175), voltageI=MutableFloat(12))  # 300Hz
    voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                    limits=(-iLimit, iLimit))

    kp_c = 0.04
    ki_c = 11.8
    current_dqp_iparams = PI_params(kP=kp_c, kI=ki_c, limits=(-1, 1))  # Current controller values

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param = DroopParams(DroopGain, 0.005, net.freq_nom)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, net.v_nom)

    # Define a voltage forming inverter using the PIPI and droop parameters from above

    # Controller with observer
    # ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param, qdroop_param,
    #                                   observer=[Lueneberger(*params) for params in
    #                                             repeat((A, B, C, L, delta_t * undersample, v_DC / 2), 3)], undersampling=undersample,
    #                                   name='master')

    # Controller without observer
    ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param, qdroop_param,
                                       ts_sim=delta_t,
                                       ts_ctrl=undersample * delta_t,
                                       name='master')

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
                         [ctrl],
                         dict(master=[[f'lc.inductor{k}.i' for k in '123'],
                                      [f'lc.capacitor{k}.v' for k in '123']
                                      ]),
                         history=FullHistory(),
                         )

    if include_simulate:
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

        # If without noise:
        # r_filt = Load(R_filt, 0 * R_filt, balanced=balanced_load)
        # l_filt = Load(L_filt, 0 * L_filt, balanced=balanced_load)
        # c_filt = Load(C_filt, 0 * C_filt, balanced=balanced_load)
        # r_load = Load(R, 0 * R, balanced=balanced_load)
        # meas_noise = Noise([0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0)

        r_filt = Load(R_filter, 0.1 * R_filter, balanced=balanced_load)
        l_filt = Load(L_filter, 0.1 * L_filter, balanced=balanced_load)
        c_filt = Load(C_filter, 0.1 * C_filter, balanced=balanced_load)
        r_load = Load(R, 0.1 * R, balanced=balanced_load)


        # meas_noise = Noise([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                   [0.45, 0.39, 0.42, 0.0023, 0.0015, 0.0018, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, 0.5)

        def reset_loads():
            r_load.reset()
            r_filt.reset()
            l_filt.reset()
            c_filt.reset()


        plotter = PlotManager(agent, save_results=save_results, save_folder=save_folder,
                              show_plots=show_plots)

        env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                       reward_fun=rew.rew_fun_v,
                       viz_cols=[
                           PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'master.SPV{i}' for i in 'abc']],
                                    callback=plotter.xylables_v_abc,
                                    color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                    style=[[None], ['--']]
                                    ),
                           PlotTmpl([[f'master.CVV{i}' for i in 'dq0'], [f'master.SPV{i}' for i in 'dq0']],
                                    callback=plotter.xylables_v_dq0,
                                    color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                    style=[[None], ['--']]
                                    ),
                           PlotTmpl([[f'lc.inductor{i}.i' for i in '123'], [f'master.SPI{i}' for i in 'abc']],
                                    callback=plotter.xylables_i_abc,
                                    color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                    style=[[None], ['--']]
                                    ),
                           # PlotTmpl([[f'master.I_hat{i}' for i in 'abc'], [f'r_load.resistor{i}.i' for i in '123'], ],
                           #         callback=lambda fig: plotter.update_axes(fig, title='Simulation',
                           #                                                  ylabel='$i_{\mathrm{o estimate,abc}}\,/\,\mathrm{A}$'),
                           #         color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                           #         style=[['-*'], ['--*']]
                           #         ),
                           # PlotTmpl([[f'master.m{i}' for i in 'dq0']],
                           #         callback=lambda fig: plotter.update_axes(fig, title='Simulation',
                           #                                                  ylabel='$m_{\mathrm{dq0}}\,/\,\mathrm{}$',
                           #                                                  filename='Sim_m_dq0')
                           #         ),
                           PlotTmpl([[f'master.CVi{i}' for i in 'dq0'], [f'master.SPI{i}' for i in 'dq0']],
                                    callback=plotter.xylables_i_dq0,
                                    color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                    style=[[None], ['--']]
                                    )
                       ],
                       log_level=logging.INFO,
                       viz_mode='episode',
                       max_episode_steps=max_episode_steps,
                       model_params={'lc.resistor1.R': partial(r_filt.give_value, n=0),
                                     'lc.resistor2.R': partial(r_filt.give_value, n=1),
                                     'lc.resistor2.R': partial(r_filt.give_value, n=1),
                                     'lc.resistor3.R': partial(r_filt.give_value, n=2),
                                     'lc.resistor4.R': 0.0000001,
                                     'lc.resistor5.R': 0.0000001,
                                     'lc.resistor6.R': 0.0000001,
                                     'lc.inductor1.L': partial(l_filt.give_value, n=0),
                                     'lc.inductor2.L': partial(l_filt.give_value, n=1),
                                     'lc.inductor3.L': partial(l_filt.give_value, n=2),
                                     'lc.capacitor1.C': partial(c_filt.give_value, n=0),
                                     'lc.capacitor2.C': partial(c_filt.give_value, n=1),
                                     'lc.capacitor3.C': partial(c_filt.give_value, n=2),
                                     'r_load.resistor1.R': partial(r_load.load_step, n=0),
                                     'r_load.resistor2.R': partial(r_load.load_step, n=1),
                                     'r_load.resistor3.R': partial(r_load.load_step, n=2),
                                     },
                       net=net,
                       model_path='../../omg_grid/grid.paper_loadstep.fmu',
                       history=FullHistory(),
                       action_time_delay=1 * undersample
                       )

        runner = MonteCarloRunner(agent, env)

        runner.run(num_episodes, n_mc=n_MC, visualise=True, prepare_mc_experiment=reset_loads,
                   return_gradient_extend=True)

        df_len = pd.DataFrame({'lengthscale': lengthscale,
                               'bounds': bounds,
                               'balanced_load': balanced_load,
                               'barrier_param_mu': mu,
                               'J_min': j_min})

        if save_results:
            agent.history.df.to_csv(save_folder + '/_result.csv')
            df_len.to_csv(save_folder + '/_params.csv')

        best_agent_plt = runner.run_data['last_agent_plt']
        ax = best_agent_plt.axes[0]
        ax.grid(which='both')
        ax.set_axisbelow(True)

        agent.params.reset()
        ax.set_ylabel(r'$K_\mathrm{i}\,/\,\mathrm{(AV^{-1}s^{-1})}$')
        ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(AV^{-1})}$')
        ax.get_figure().axes[1].set_ylabel(r'$J$')
        plt.title('Lengthscale = {}; balanced = '.format(lengthscale, balanced_load))
        # ax.plot([0.01, 0.01], [0, 250], 'k')
        # ax.plot([mutable_params['currentP'].val, mutable_params['currentP'].val], bounds[1], 'k-', zorder=1,
        #         lw=4,
        #         alpha=.5)
        best_agent_plt.show()
        if save_results:
            best_agent_plt.savefig(save_folder + '/_agent_plt.pdf')
            best_agent_plt.savefig(save_folder + '/_agent_plt.pgf')
            agent.history.df.to_csv(save_folder + '/_result.csv')

        print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df.round({'J': 4, 'Params': 4})))
        print('\n Experiment finished with best set: \n')
        print('\n  Current-Ki&Kp and voltage-Ki&Kp = {}'.format(
            agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
        print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
        print('\n\nBest experiment results are plotted in the following:')

    if do_measurement:
        #####################################
        # Execution of the experiment
        # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)

        env = TestbenchEnvVoltage(num_steps=max_episode_steps, DT=1 / 10000, v_nominal=nomVoltPeak,
                                  kP=kp_c, kI=ki_c, v_limit=vLimit, f_nom=nomFreq, mu=mu)
        # runner = RunnerHardware(agent, env)
        runner = RunnerHardwareGradient(agent, env)

        runner.run(num_episodes, visualise=True, save_folder=save_folder)

        print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df[:]))

        print('\n Experiment finished with best set: \n')
        print('\n  Current-Kp&Ki and voltage-Kp&Ki = {}'.format(
            agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
        print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
        print('\n\nBest experiment results are plotted in the following:')

        df_len = pd.DataFrame({'lengthscale': lengthscale,
                               'bounds': bounds,
                               'balanced_load': balanced_load,
                               'barrier_param_mu': mu,
                               'J_min': j_min})

        if save_results:
            agent.history.df.to_csv(save_folder + '/_meas_result.csv')
            df_len.to_csv(save_folder + '/_meas_params.csv')

        # Show last performance plot
        best_agent_plt = runner.run_data['last_agent_plt']
        ax = best_agent_plt.axes[0]
        ax.grid(which='both')
        ax.set_axisbelow(True)

        agent.params.reset()
        ax.set_ylabel(r'$K_\mathrm{i}\,/\,\mathrm{(AV^{-1}s^{-1})}$')
        ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(AV^{-1})}$')
        ax.get_figure().axes[1].set_ylabel(r'$J$')
        # plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-', zorder=1,
        #         lw=4,
        #         alpha=.5)
        best_agent_plt.show()
        if save_results:
            best_agent_plt.savefig(save_folder + '/_meas_agent_plt.png')
            best_agent_plt.savefig(save_folder + '/_meas_agent_plt.pdf')
            best_agent_plt.savefig(save_folder + '/_meas_agent_plt.pgf')
