import logging
import os
import platform
import time
from functools import partial

import GPy
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
from stochastic.processes import VasicekProcess
from tqdm import tqdm
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# from agents.my_ddpg import myDDPG
from experiments.hp_tune.env.env_wrapper import FeatureWrapper
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net  # , folder_name
from experiments.hp_tune.util.config import cfg
from experiments.hp_tune.util.recorder import Recorder

# imports for PIPI
from experiments.hp_tune.env.random_load import RandomLoad
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, \
    MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.plotmanager import PlotManager
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import FullHistory, RandProcess

import pandas as pd

from openmodelica_microgrid_gym.util import abc_to_dq0


class CallbackList(list):
    def fire(self, *args, **kwargs):
        for listener in self:
            listener(*args, **kwargs)


import gym

# np.random.seed(0)

show_plots = False
balanced_load = False
save_results = False

folder_name = 'Comparison_PI_DDPG'  # cfg['STUDY_NAME']
node = platform.uname().node

# mongo_recorder = Recorder(database_name=folder_name)
mongo_recorder = Recorder(node=node,
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne or locally as json to cfg[meas_data_folder]

num_average = 1
max_episode_steps_list = [1000, 100000]  # [1000, 5000, 10000, 20000, 50000, 100000]

result_list = []
ret_list = []
mean_list = []
std_list = []
ret_array = np.zeros(num_average)

df = pd.DataFrame()
ret_dict = dict()

#################PI Config stuff##############################################################################

current_directory = os.getcwd()
# folder_name = 'Pipi_safeopt_best_run4d'
save_folder = os.path.join(current_directory, folder_name)
os.makedirs(save_folder, exist_ok=True)

# Simulation definitions
# net = Network.load('../../net/net_single-inv-Paper_Loadstep.yaml')
net = Network.load('net/net_vctrl_single_inv.yaml')
delta_t = 1e-4  # simulation time step size / s
undersample = 1
max_episode_steps = 1002  # number of simulation steps per episode
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
# kp_v = 0.002
# ki_v = 143
kp_v = 0.0095  # 0.0
ki_v = 173.22  # 200
# Choose Kp and Ki for the current and voltage controller as mutable parameters
mutable_params = dict(voltageP=MutableFloat(kp_v), voltageI=MutableFloat(ki_v))  # 300Hz
# mutable_params = dict(voltageP=MutableFloat(0.016), voltageI=MutableFloat(105))  # 300Hz
voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                limits=(-iLimit, iLimit))

# kp_c = 0.033
# ki_c = 17.4  # 11.8

kp_c = 0.0404  # 0.04
ki_c = 4.065  # 11.8
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

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F

lower_bound_load = -10  # to allow maximal load that draws i_limit
upper_bound_load = 200  # to apply symmetrical load bounds
lower_bound_load_clip = 14  # to allow maximal load that draws i_limit (let exceed?)
upper_bound_load_clip = 200  # to apply symmetrical load bounds
lower_bound_load_clip_std = 2
upper_bound_load_clip_std = 0

################DDPG Config Stuff#########################################################################
gamma = 0.946218
integrator_weight = 0.311135
antiwindup_weight = 0.660818
model_path = 'experiments/hp_tune/trained_models/study_22_run_11534/'
error_exponent = 0.5
use_gamma_in_rew = 1
n_trail = 50001
actor_number_layers = 2
critic_number_layers = 4
alpha_relu_actor = 0.208098
alpha_relu_critic = 0.00678497

for max_eps_steps in tqdm(range(len(max_episode_steps_list)), desc='steps', unit='step', leave=False):

    for ave_run in tqdm(range(num_average), desc='steps', unit='step', leave=False):

        rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                     use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent,
                     i_lim=net['inverter1'].i_lim,
                     i_nom=net['inverter1'].i_nom)

        ####################################PI Stuff################################################
        R = np.random.uniform(low=lower_bound_load, high=upper_bound_load)

        gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=800, vol=40, mean=R), initial=R,
                          bounds=(lower_bound_load, upper_bound_load))

        rand_load_train = RandomLoad(max_episode_steps, net.ts, gen,
                                     bounds=(lower_bound_load_clip, upper_bound_load_clip),
                                     bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))

        cb = CallbackList()
        # set initial = None to reset load random in range of bounds
        cb.append(partial(gen.reset))  # , initial=np.random.uniform(low=lower_bound_load, high=upper_bound_load)))
        cb.append(rand_load_train.reset)

        plotter = PlotManager(agent, save_results=save_results, save_folder=save_folder,
                              show_plots=show_plots)

        rand_load_test = RandomLoad(max_episode_steps, net.ts, gen,
                                    load_curve=pd.read_pickle(
                                        'experiments/hp_tune/data/R_load_test_case_2_seconds.pkl'))


        def xylables_R(fig):
            ax = fig.gca()
            ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
            ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
            ax.grid(which='both')
            # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
            ts = time.gmtime()
            fig.savefig(f'{save_folder}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
            if show_plots:
                plt.show()
            else:
                plt.close()


        def xylables_i(fig):
            ax = fig.gca()
            ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
            ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
            ax.grid(which='both')
            # fig.savefig(f'{folder_name + experiment_name + n_trail}/Inductor_currents.pdf')
            if show_plots:
                plt.show()
            else:
                plt.close()


        def xylables_v(fig):
            ax = fig.gca()
            ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
            ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
            ax.grid(which='both')
            # ax.set_xlim([0, 0.005])
            ts = time.gmtime()
            # fig.savefig(
            #    f'{folder_name + experiment_name}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
            if show_plots:
                plt.show()
            else:
                plt.close()


        env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                       reward_fun=rew.rew_fun_PIPI_MRE,
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
                       # log_level=logging.INFO,
                       viz_mode='episode',
                       max_episode_steps=max_episode_steps_list[max_eps_steps],
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
                                     # 'r_load.resistor1.R': partial(rand_load_train.load_step, gain=R),
                                     # 'r_load.resistor2.R': partial(rand_load_train.load_step, gain=R),
                                     # 'r_load.resistor3.R': partial(rand_load_train.load_step, gain=R),
                                     'r_load.resistor1.R': rand_load_train.random_load_step,
                                     'r_load.resistor2.R': rand_load_train.random_load_step,
                                     'r_load.resistor3.R': rand_load_train.random_load_step,
                                     'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_nom,
                                                                                    high=v_nom) if t == -1 else None,
                                     'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_nom,
                                                                                    high=v_nom) if t == -1 else None,
                                     'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_nom,
                                                                                    high=v_nom) if t == -1 else None,
                                     'lc.inductor1.i': lambda t: np.random.uniform(low=-i_nom,
                                                                                   high=i_nom) if t == -1 else None,
                                     'lc.inductor2.i': lambda t: np.random.uniform(low=-i_nom,
                                                                                   high=i_nom) if t == -1 else None,
                                     'lc.inductor3.i': lambda t: np.random.uniform(low=-i_nom,
                                                                                   high=i_nom) if t == -1 else None,
                                     },
                       net=net,
                       model_path='omg_grid/grid.paper_loadstep.fmu',
                       history=FullHistory(),
                       # on_episode_reset_callback=cb.fire,
                       action_time_delay=1 * undersample
                       )

        return_sum_PI = 0.0
        rew_list_PI = []
        v_d_PI = []
        v_q_PI = []
        v_0_PI = []
        R_load_PI = []
        limit_exceeded_in_test_PI = False
        limit_exceeded_penalty_PI = 0

        ####################################DDPG Stuff##############################################
        return_sum = 0.0

        rew.gamma = 0
        # episodes will not abort, if limit is exceeded reward = -1
        rew.det_run = True
        rew.exponent = 0.5  # 1
        limit_exceeded_in_test = False
        limit_exceeded_penalty = 0

        env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v1',
                            reward_fun=rew.rew_fun_dq0,
                            abort_reward=-1,  # no needed if in rew no None is given back
                            # on_episode_reset_callback=cb.fire  # needed?
                            obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                        'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                        'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'],
                            max_episode_steps=max_episode_steps_list[max_eps_steps],

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
                                          # 'r_load.resistor1.R': partial(rand_load_train.load_step, gain=R),
                                          # 'r_load.resistor2.R': partial(rand_load_train.load_step, gain=R),
                                          # 'r_load.resistor3.R': partial(rand_load_train.load_step, gain=R),
                                          'r_load.resistor1.R': rand_load_train.random_load_step,
                                          'r_load.resistor2.R': rand_load_train.random_load_step,
                                          'r_load.resistor3.R': rand_load_train.random_load_step,
                                          'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_nom,
                                                                                         high=v_nom) if t == -1 else None,
                                          'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_nom,
                                                                                         high=v_nom) if t == -1 else None,
                                          'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_nom,
                                                                                         high=v_nom) if t == -1 else None,
                                          'lc.inductor1.i': lambda t: np.random.uniform(low=-i_nom,
                                                                                        high=i_nom) if t == -1 else None,
                                          'lc.inductor2.i': lambda t: np.random.uniform(low=-i_nom,
                                                                                        high=i_nom) if t == -1 else None,
                                          'lc.inductor3.i': lambda t: np.random.uniform(low=-i_nom,
                                                                                        high=i_nom) if t == -1 else None,
                                          },
                            on_episode_reset_callback=cb.fire
                            )

        env_test = FeatureWrapper(env_test, number_of_features=11, integrator_weight=integrator_weight,
                                  recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                                  gamma=1, penalty_I_weight=0, penalty_P_weight=0)
        # using gamma=1 and rew_weigth=3 we get the original reward from the env without penalties

        env_test.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

        model = DDPG.load(model_path + f'model.zip')  # , env=env_test)

        count = 0
        for kk in range(actor_number_layers + 1):

            if kk < actor_number_layers:
                model.actor.mu._modules[str(count + 1)].negative_slope = alpha_relu_actor
                model.actor_target.mu._modules[str(count + 1)].negative_slope = alpha_relu_actor

            count = count + 2

        count = 0

        for kk in range(critic_number_layers + 1):

            if kk < critic_number_layers:
                model.critic.qf0._modules[str(count + 1)].negative_slope = alpha_relu_critic
                model.critic_target.qf0._modules[str(count + 1)].negative_slope = alpha_relu_critic

            count = count + 2

        env_test.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))
        rew_list = []
        v_d = []
        v_q = []
        v_0 = []
        R_load = []

        ####### Run Test #########
        # agent ~ PI Controllerv using env
        # model ~ RL Controller using env_test
        # Both run in the same loop

        agent.reset()
        agent.obs_varnames = env.history.cols
        env.history.cols = env.history.structured_cols(None) + agent.measurement_cols
        env.measure = agent.measure
        agent_fig = None

        obs = env_test.reset()
        obs_PI = env.reset()

        for step in range(env_test.max_episode_steps):

            agent.observe(None, False)
            act_PI = agent.act(obs_PI)
            obs_PI, r_PI, done_PI, info_PI = env.step(act_PI)
            rew_list_PI.append(r_PI)
            env.render()
            return_sum_PI += r_PI
            if r_PI == -1 and not limit_exceeded_in_test_PI:
                # Set addidional penalty of -1 if limit is exceeded once in the test case
                limit_exceeded_in_test_PI = True
                limit_exceeded_penalty_PI = -1

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env_test.step(action)

            if rewards == -1 and not limit_exceeded_in_test:
                # Set addidional penalty of -1 if limit is exceeded once in the test case
                limit_exceeded_in_test = True
                limit_exceeded_penalty = -1
            env_test.render()
            return_sum += rewards
            rew_list.append(rewards)

            v_a = env_test.history.df['lc.capacitor1.v'].iloc[-1]
            v_b = env_test.history.df['lc.capacitor2.v'].iloc[-1]
            v_c = env_test.history.df['lc.capacitor3.v'].iloc[-1]
            R_load.append(env_test.history.df['r_load.resistor1.R'].iloc[-1])

            v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), env_test.env.net.components[0].phase)

            v_d.append(v_dq0[0])
            v_q.append(v_dq0[1])
            v_0.append(v_dq0[2])

            v_a_PI = env.history.df['lc.capacitor1.v'].iloc[-1]
            v_b_PI = env.history.df['lc.capacitor2.v'].iloc[-1]
            v_c_PI = env.history.df['lc.capacitor3.v'].iloc[-1]
            R_load_PI.append(env.history.df['r_load.resistor1.R'].iloc[-1])

            v_dq0_PI = abc_to_dq0(np.array([v_a_PI, v_b_PI, v_c_PI]), env.net.components[0].phase)

            v_d_PI.append(v_dq0_PI[0])
            v_q_PI.append(v_dq0_PI[1])
            v_0_PI.append(v_dq0_PI[2])

            if step % 1000 == 0 and step != 0:
                env_test.close()
                obs = env_test.reset()

                env.close()
                agent.reset()
                obs_PI = env.reset()

            # print(rewards)
            if done:
                env_test.close()
                # print(limit_exceeded_in_test)
                break

        _, env_fig = env.close()
        agent.observe(r_PI, done_PI)

        plt.plot(v_d, 'b')
        plt.plot(v_q, 'r')
        plt.plot(v_0, 'g')
        plt.xlabel("")
        plt.grid()
        plt.ylabel("v_dq0")
        plt.title('Test')
        plt.show()

        plt.plot(R_load, 'g')
        plt.xlabel("")
        plt.grid()
        plt.ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
        plt.title('Test')
        plt.show()

        plt.plot(v_d_PI, 'b')
        plt.plot(v_q_PI, 'r')
        plt.plot(v_0_PI, 'g')
        plt.xlabel("")
        plt.grid()
        plt.ylabel("v_dq0")
        plt.title('Test')
        plt.show()

        plt.plot(R_load_PI, 'g')
        plt.xlabel("")
        plt.grid()
        plt.ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
        plt.title('Test')
        plt.show()

        # return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)

        print(f'RL: {(return_sum / env_test.max_episode_steps + limit_exceeded_penalty)}')
        print(f'PI: {(return_sum_PI / env.max_episode_steps + limit_exceeded_penalty_PI)}')

        ts = time.gmtime()
        compare_result = {"Name": "comparison_PI_DDPG",
                          "time": ts,
                          "PI_Kp_c": kp_c,
                          "PI_Ki_c": ki_c,
                          "PI_Kp_v": kp_v,
                          "PI_Ki_v": ki_v,
                          "DDPG_model_path": model_path,
                          "Return PI": (return_sum_PI / env.max_episode_steps + limit_exceeded_penalty_PI),
                          "Return DDPG": (return_sum / env_test.max_episode_steps + limit_exceeded_penalty),
                          "v_d_PI": v_d_PI,
                          "v_q_PI": v_q_PI,
                          "v_0_PI": v_0_PI,
                          "v_d_DDPG": v_d,
                          "v_q_DDPG": v_q,
                          "v_0_DDPG": v_0,
                          "R_load": R_load,
                          "max_episode_steps": str(max_episode_steps_list[max_eps_steps]),
                          "number of averages per run": num_average
                          }
        node = platform.uname().node

        # mongo_recorder = Recorder(database_name=folder_name)

        # mongo_recorder.save_to_mongodb('Comparison1' + n_trail, compare_result)
        mongo_recorder.save_to_mongodb('Comparison_4D_optimizedPIPI', compare_result)
        # mongo_recorder.save_to_mongodb('Comparison_2D_optimizedPIPI', compare_result)

        ret_list.append((return_sum / env_test.max_episode_steps + limit_exceeded_penalty))
        ret_array[ave_run] = (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)

        # ret_dict[str(ave_run)] = (return_sum / env.max_episode_steps + limit_exceeded_penalty)

        # zipped = zip(max_episode_steps_list[max_eps_steps], ret_list)
        # temp_dict = dict(zipped)
    temp_dict = {str(max_episode_steps_list[max_eps_steps]): ret_list}
    result_list.append(temp_dict)
    # ret_dict.append(zipped)
    # df = df.append(ret_dict)

    mean_list.append(np.mean(ret_array))
    std_list.append(np.std(ret_array))

# df = df.append(temp_list, True)
print(mean_list)
print(std_list)
print(result_list)

results = {
    'Mean': mean_list,
    'Std': std_list,
    'All results': result_list,
    'max_episode_steps_list': max_episode_steps_list
}

df = pd.DataFrame(results)
# df.to_pickle("DDPG_study18_best_test_varianz.pkl")
asd = 1
"""
m = np.array(df['Mean'])
s = np.array(df['Std'])
max_episode_steps_list = np.array(df['max_episode_steps_list'])

plt.plot(max_episode_steps_list, m)
plt.fill_between(max_episode_steps_list, m - s, m + s, facecolor='r')
plt.ylabel('Average return +- sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title('DDPG')
plt.show()

# plt.plot(max_episode_steps_list, m)
# plt.fill_between(max_episode_steps_list, m - s, m + s, facecolor='r')
plt.errorbar(max_episode_steps_list, m, s, fmt='-o')
plt.ylabel('Average return +- sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title('DDPG')
plt.show()

plt.plot(max_episode_steps_list, s)
plt.ylabel('std')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title('DDPG')
plt.show()
"""
