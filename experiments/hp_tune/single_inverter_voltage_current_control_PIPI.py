#####################################
# Experiment : Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt
# a) FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# b) connecting via ssh to a testbench to perform real-world measurement
import time
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
from pymongo import MongoClient
from stochastic.processes import VasicekProcess
from tqdm import tqdm

from experiments.hp_tune.env.random_load import RandomLoad
from experiments.model_validation.env.testbench_voltage_ctrl import TestbenchEnvVoltage
from experiments.model_validation.execution.monte_carlo_runner import MonteCarloRunner
from experiments.model_validation.execution.runner_hardware import RunnerHardwareGradient
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, \
    MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.plotmanager import PlotManager
from experiments.hp_tune.env.rewards import Reward
from experiments.model_validation.env.stochastic_components import Load
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import FullHistory, RandProcess

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
save_results = True

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
folder_name = 'Pipi_safeopt_vc_cc_analytical_MRE'
save_folder = os.path.join(current_directory, folder_name)
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


class Recorder:

    def __init__(self, URI: str = 'mongodb://localhost:27017/', database_name: str = 'OMG', ):
        self.client = MongoClient(URI)
        self.db = self.client[database_name]

    def save_to_mongodb(self, col: str = ' trails', data=None):
        trial_coll = self.db[col]  # get collection named col
        if data is None:
            raise ValueError('No data given to store in database!')
        trial_coll.insert_one(data)


def run_experiment():
    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma=0,
                 use_gamma_normalization=1, error_exponent=0.5, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

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
    j_min = -5  # cal min allowed performance

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
    # mutable_params = dict(voltageP=MutableFloat(0.0175), voltageI=MutableFloat(12))  # 300Hz
    mutable_params = dict(voltageP=MutableFloat(0.016), voltageI=MutableFloat(105))  # 300Hz
    voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                    limits=(-iLimit, iLimit))

    kp_c = 0.04
    ki_c = 27  # 11.8
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

        i_lim = net['inverter1'].i_lim  # inverter current limit / A
        i_nom = net['inverter1'].i_nom  # nominal inverter current / A
        v_nom = net.v_nom
        v_lim = net['inverter1'].v_lim
        v_DC = net['inverter1'].v_DC
        # plant

        # toDo: shift this to net?!
        L_filter = 2.3e-3  # / H
        R_filter = 400e-3  # / Ohm
        C_filter = 10e-6  # / F

        lower_bound_load = 11  # to allow maximal load that draws i_limit (toDo: let exceed?)
        upper_bound_load = 160  # to apply symmetrical load bounds

        R = np.random.uniform(low=lower_bound_load, high=upper_bound_load)

        loadstep_timestep = max_episode_steps / 2

        gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=1000, vol=10, mean=R), initial=R,
                          bounds=(lower_bound_load, upper_bound_load))
        plotter = PlotManager(agent, save_results=save_results, save_folder=save_folder,
                              show_plots=show_plots)

        rand_load_test = RandomLoad(max_episode_steps, net.ts, gen,
                                    load_curve=pd.read_pickle('R_load_test_case_2_seconds'))

        def xylables_R(fig):
            ax = fig.gca()
            ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
            ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
            ax.grid(which='both')
            # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
            ts = time.gmtime()
            fig.savefig(f'{save_folder}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
            plt.close()

        env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                       reward_fun=rew.rew_fun_PIPI,
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
                           PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                                    callback=xylables_R,
                                    color=[['b', 'r', 'g']],
                                    style=[[None]]
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
                       max_episode_steps=20000,
                       model_params={'lc.resistor1.R': R_filter,
                                     'lc.resistor2.R': R_filter,
                                     'lc.resistor3.R': R_filter,
                                     'lc.resistor4.R': 0.0000001,
                                     'lc.resistor5.R': 0.0000001,
                                     'lc.resistor6.R': 0.0000001,
                                     'lc.inductor1.L': L_filter,
                                     'lc.inductor2.L': L_filter,
                                     'lc.inductor3.L': L_filter,
                                     'lc.capacitor1.C': C_filter,
                                     'lc.capacitor2.C': C_filter,
                                     'lc.capacitor3.C': C_filter,
                                     'r_load.resistor1.R': partial(rand_load_test.give_dataframe_value,
                                                                   col='r_load.resistor1.R'),
                                     'r_load.resistor2.R': partial(rand_load_test.give_dataframe_value,
                                                                   col='r_load.resistor2.R'),
                                     'r_load.resistor3.R': partial(rand_load_test.give_dataframe_value,
                                                                   col='r_load.resistor3.R')
                                     },
                       net=net,
                       model_path='../../omg_grid/grid.paper_loadstep.fmu',
                       history=FullHistory(),
                       action_time_delay=1 * undersample
                       )

        return_sum = 0.0

        rew.gamma = 0
        # episodes will not abort, if limit is exceeded reward = -1
        rew.det_run = True
        rew.exponent = 0.5
        limit_exceeded_in_test = False
        limit_exceeded_penalty = 0


        """
        # toDo: - Use other Test-episode
        #       - Rückgabewert = (Summe der üblichen Rewards) / (Anzahl steps Validierung) + (Penalty i.H.v. -1)
        while True:
            action = agent.act(obs)
            obs, rewards, done, info = env_test.step(action)

            if rewards == -1 and not limit_exceeded_in_test:
                # Set addidional penalty of -1 if limit is exceeded once in the test case
                limit_exceeded_in_test = True
                limit_exceeded_penalty = -1
            env_test.render()
            return_sum += rewards
            rew_list.append(rewards)
            # print(rewards)
            if done:
                env_test.close()
                # print(limit_exceeded_in_test)
                break
        """

        agent.reset()
        agent.obs_varnames = env.history.cols
        env.history.cols = env.history.structured_cols(None) + agent.measurement_cols
        env.measure = agent.measure

        reward_list = []

        agent_fig = None
        obs = env.reset()
        for _ in tqdm(range(env.max_episode_steps), desc='steps', unit='step', leave=False):

            done, r = False, None

            agent.observe(r, done)
            act = agent.act(obs)
            obs, r, done, info = env.step(act)
            reward_list.append(r)
            env.render()
            return_sum += r
            if r == -1 and not limit_exceeded_in_test:
                # Set addidional penalty of -1 if limit is exceeded once in the test case
                limit_exceeded_in_test = True
                limit_exceeded_penalty = -1

            # if done:
            #    break
            # close env before calling final agent observe to see plots even if agent crashes
        _, env_fig = env.close()
        agent.observe(r, done)
        print(limit_exceeded_in_test)
        ret = return_sum / env.max_episode_steps + limit_exceeded_penalty

        ts = time.gmtime()
        test_after_training = {"Name": "Test",
                               "time": ts,
                               "Reward": reward_list}

        # Add v-measurements
        test_after_training.update({env.viz_col_tmpls[j].vars[i].replace(".", "_"): env.history[
            env.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                    })

        test_after_training.update({env.viz_col_tmpls[2].vars[i].replace(".", "_"): env.history[
            env.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                    })

        # va = self.env.env.history[self.env.env.viz_col_tmpls[0].vars[0]].copy()
        mongo_recorder = Recorder(database_name=folder_name)
        mongo_recorder.save_to_mongodb('Trail_number_0', test_after_training)

        return (return_sum / env.max_episode_steps + limit_exceeded_penalty)


ret = run_experiment()

print(ret)
