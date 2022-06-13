print('Start script')
import gym
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
from experiments.P10.env.env_wrapper_P10 import FeatureWrapper, FeatureWrapper_pastVals, BaseWrapper
from experiments.P10.env.rewards_P10 import Reward
from experiments.P10.env.vctrl_single_inv_P10 import net  # , folder_name
from experiments.P10.util.config import cfg
from experiments.P10.util.recorder_P10 import Recorder

# imports for PIPI
from experiments.P10.env.random_load_P10 import RandomLoad
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


show_plots = True
save_results = False
# 2128 ->0; 3125 -> -1, 956-> best
trial = '956'

folder_name = 'saves/P10_SEC_R_load'
os.makedirs(folder_name, exist_ok=True)
wrapper = ['past']
model_path = 'experiments/P10/viz/data/'
model_path = 'experiments/P10/retrain/'
node = platform.uname().node
model_name = ['model_'+trial+'.zip']
model_name = ['model'+trial+'_retrained']

################DDPG Config Stuff#########################################################################
print('Using model_'+trial+' setting')
if trial == '956':
    actor_number_layers = 2
    alpha_relu_actor = 0.0225049
    alpha_relu_critic = 0.00861825
    antiwindup_weight = 0.350646
    critic_number_layers = 4
    error_exponent = 0.5
    gamma = 0.794337
    integrator_weight = 0.214138
    use_gamma_in_rew = 1
    n_trail = 50001
    number_past_vals = [18]

if trial == '3125':
    actor_number_layers = 1
    alpha_relu_actor = 0.305758
    alpha_relu_critic = 0.0119687
    antiwindup_weight = 0.767766
    critic_number_layers = 4
    error_exponent = 0.5
    gamma = 0.922121
    integrator_weight = 0.237488
    use_gamma_in_rew = 1
    n_trail = 50001
    number_past_vals = [2]

if trial == '2128':
    actor_number_layers = 1
    alpha_relu_actor = 0.334101
    alpha_relu_critic = 0.0729528
    antiwindup_weight = 0.648373
    critic_number_layers = 4
    error_exponent = 0.5
    gamma = 0.798319
    integrator_weight = 0.122662
    use_gamma_in_rew = 1
    n_trail = 50001
    number_past_vals = [7]

mongo_recorder = Recorder(node=node, database_name=folder_name)

num_average = 1
max_episode_steps_list = [1000]

data_str = 'experiments/hp_tune/data/R_load_deterministic_test_case2_1_seconds.pkl'
#data_str = 'experiments/hp_tune/data/R_load_hard_test_case_10_seconds.pkl'
# data_str = 'experiments/hp_tune/data/R_load_hard_test_case_60_seconds_noReset.pkl'

result_list = []
ret_list = []
mean_list = []
std_list = []
ret_array = np.zeros(num_average)

df = pd.DataFrame()
ret_dict = dict()

#################PI Config stuff##############################################################################
current_directory = os.getcwd()

save_folder = os.path.join(current_directory, folder_name)
os.makedirs(save_folder, exist_ok=True)

# Simulation definitions
# net = Network.load('../../net/net_single-inv-Paper_Loadstep.yaml')
net = Network.load('net/net_p10.yaml')
delta_t = 1e-4  # simulation time step size / s
undersample = 1
# max_episode_steps = 1002  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
n_MC = 1  # number of Monte-Carlo samples for simulation - samples device parameters (e.g. L,R, noise) from
DroopGain = 0.0  # virtual droop gain for active power / W/Hz
QDroopGain = 0.0  # virtual droop gain for reactive power / VAR/V

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC

print("P10 stuff!")
L_filter = 70e-6  # / H
R_filter = 1.1e-3  # / Ohm
C_filter = 250e-6  # / F

lower_bound_load = -10  # to allow maximal load that draws i_limit
upper_bound_load = 200  # to apply symmetrical load bounds
lower_bound_load_clip = 1  # to allow maximal load that draws i_limit (let exceed?)
upper_bound_load_clip = 200  # to apply symmetrical load bounds
lower_bound_load_clip_std = 1
upper_bound_load_clip_std = 0
#####################################
# Definitions for the GP
prior_mean = 0  # 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
noise_var = 0.001  # ** 2  # measurement noise sigma_omega
prior_var = 2  # prior variance of the GP

bounds = [(0.000, 0.045), (4, 450)]  # bounds on the input variable current-Ki&Kp and voltage-Ki&Kp
lengthscale = [.003, 50.]  # length scale for the parameter variation [current-Ki&Kp and voltage-Ki&Kp] for the GP

safe_threshold = 0
j_min = -5  # cal min allowed performance

explore_threshold = 0

# Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
# limit exceeded
abort_reward = 100 * j_min

# Definition of the kernel
kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

#####################################
# Definition of the controllers
#From MATLAB:
#################################################################################
#Layout using mangtude optimum for inverter with LC-filter using L = 7e-05 H, C = 0.00025 F, R = 0.001 Ohm.
#Current controller:
#Kp = 0.0009773 A/V and Ki = 0.13159 A/(Vs)
#Voltage controller:
#Kp = 0.45052 V/A and Ki = 305.6655 V/(As)
# P10:
print('using p10 setting')
kp_v = 0.45052
ki_v = 305.6655
kp_c = 0.0009773
ki_c = 0.13159

# Choose Kp and Ki for the current and voltage controller as mutable parameters
mutable_params = dict(voltageP=MutableFloat(kp_v), voltageI=MutableFloat(ki_v))  # 300Hz
# mutable_params = dict(voltageP=MutableFloat(0.016), voltageI=MutableFloat(105))  # 300Hz
voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                limits=(-i_lim * 10, i_lim * 10))

current_dqp_iparams = PI_params(kP=kp_c, kI=ki_c, limits=(-1, 1))  # Current controller values
droop_param = DroopParams(DroopGain, 0.005, net.freq_nom)
qdroop_param = DroopParams(QDroopGain, 0.002, net.v_nom)

ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param, qdroop_param,
                                   ts_sim=delta_t,
                                   ts_ctrl=undersample * delta_t,
                                   name='master')

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

""""""
for max_eps_steps in tqdm(range(len(max_episode_steps_list)), desc='steps', unit='step', leave=False):

    for ave_run in tqdm(range(num_average), desc='steps', unit='step', leave=False):

        rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                     use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent,
                     i_lim=net['inverter1'].i_lim,
                     i_nom=net['inverter1'].i_nom, det_run=True)

        ####################################PI Stuff################################################
        R = np.random.uniform(low=lower_bound_load, high=upper_bound_load)

        gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=800, vol=40, mean=R), initial=R,
                          bounds=(lower_bound_load, upper_bound_load))

        rand_load_train = RandomLoad(max_episode_steps_list[max_eps_steps], net.ts, gen,
                                     bounds=(lower_bound_load_clip, upper_bound_load_clip),
                                     bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))
        rand_load_test = RandomLoad(max_episode_steps_list[max_eps_steps], net.ts, gen,
                                    load_curve=pd.read_pickle(
                                        # 'experiments/hp_tune/data/R_load_tenLoadstepPerEpisode2881Len_test_case_10_seconds.pkl'))
                                        # 'experiments/hp_tune/data/R_load_hard_test_case_10_seconds.pkl'))
                                        # 'experiments/hp_tune/data/R_load_hard_test_case_60_seconds_noReset.pkl'))
                                        # 'experiments/hp_tune/data/R_load_deterministic_test_case_25_ohm_1_seconds.pkl'))
                                        data_str))

        cb = CallbackList()
        # set initial = None to reset load random in range of bounds
        cb.append(partial(gen.reset))  # , initial=np.random.uniform(low=lower_bound_load, high=upper_bound_load)))
        cb.append(rand_load_train.reset)

        plotter = PlotManager(agent, save_results=save_results, save_folder=save_folder,
                              show_plots=show_plots)


        # rand_load_test = RandomLoad(max_episode_steps_list[max_eps_steps], net.ts, gen,
        #                            load_curve=pd.read_pickle(
        #                                'experiments/hp_tune/data/R_load_test_case_2_seconds.pkl'))

        def xylables_R(fig):
            ax = fig.gca()
            ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
            ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
            ax.grid(which='both')
            # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
            ts = time.gmtime()
            # fig.savefig(f'{save_folder}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
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
                           PlotTmpl([[f'master.CVi{i}' for i in 'dq0'], [f'master.SPI{i}' for i in 'dq0']],
                                    callback=plotter.xylables_i_dq0,
                                    color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                    style=[[None], ['--']]
                                    )
                       ],
                       viz_mode='episode',
                       max_episode_steps=max_episode_steps_list[max_eps_steps],
                       model_params={'lc.resistor1.R': R_filter,
                                     'lc.resistor2.R': R_filter,
                                     'lc.resistor3.R': R_filter,
                                     'lc.resistor4.R': R_filter,
                                     'lc.resistor5.R': R_filter,
                                     'lc.resistor6.R': R_filter,
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
                                                                   col='r_load.resistor3.R'),
                                     # 'lc.capacitor1.v': 0,
                                     # 'lc.capacitor2.v': 0,
                                     # 'lc.capacitor3.v': 0,
                                     # 'lc.inductor1.i': 0,
                                     # 'lc.inductor2.i': 0,
                                     # 'lc.inductor3.i': 0,
                                     },
                       net=net,
                       model_path='omg_grid/grid.paper_loadstep.fmu',
                       history=FullHistory(),
                       # on_episode_reset_callback=cb.fire,
                       action_time_delay=1 * undersample
                       )

        rew.gamma = 0
        return_sum_PI = 0.0
        rew_list_PI = []
        v_d_PI = []
        v_q_PI = []
        v_0_PI = []
        R_load_PI = []
        limit_exceeded_in_test_PI = False
        limit_exceeded_penalty_PI = 0
        """
        agent.reset()
        agent.obs_varnames = env.history.cols
        env.history.cols = env.history.structured_cols(None) + agent.measurement_cols
        env.measure = agent.measure
        agent_fig = None
        obs_PI = env.reset()

        for step in tqdm(range(env.max_episode_steps), desc='steps', unit='step', leave=False):
            # for max_eps_steps in tqdm(range(len(max_episode_steps_list)), desc='steps', unit='step', leave=False):

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

        # _, env_fig = env.close()
        agent.observe(r_PI, done_PI)

        v_a_PI = env.history.df['lc.capacitor1.v']
        v_b_PI = env.history.df['lc.capacitor2.v']
        v_c_PI = env.history.df['lc.capacitor3.v']
        i_a_PI = env.history.df['lc.inductor1.i']
        i_b_PI = env.history.df['lc.inductor2.i']
        i_c_PI = env.history.df['lc.inductor3.i']
        R_load_PI = (env.history.df['r_load.resistor1.R'].tolist())
        phase_PI = env.history.df['inverter1.phase.0']  # env.net.components[0].phase

        i_dq0_PI = abc_to_dq0(np.array([i_a_PI, i_b_PI, i_c_PI]), phase_PI)
        v_dq0_PI = abc_to_dq0(np.array([v_a_PI, v_b_PI, v_c_PI]), phase_PI)

        i_d_PI = i_dq0_PI[0].tolist()
        i_q_PI = i_dq0_PI[1].tolist()
        i_0_PI = i_dq0_PI[2].tolist()
        v_d_PI = (v_dq0_PI[0].tolist())
        v_q_PI = (v_dq0_PI[1].tolist())
        v_0_PI = (v_dq0_PI[2].tolist())

        ts = time.gmtime()
        compare_result = {"Name": "comparison_PI_DDPG",
                          "time": ts,
                          "PI_Kp_c": kp_c,
                          "PI_Ki_c": ki_c,
                          "PI_Kp_v": kp_v,
                          "PI_Ki_v": ki_v,
                          "DDPG_model_path": model_path,
                          "Return PI": (return_sum_PI / env.max_episode_steps + limit_exceeded_penalty_PI),
                          "Reward PI": rew_list_PI,
                          "env_hist_PI": env.history.df,
                          "max_episode_steps": str(max_episode_steps_list[max_eps_steps]),
                          "number of averages per run": num_average,
                          "info": "PI result for comparison with RL agent",
                          "optimization node": 'Thinkpad',
                          "optimization folder name": 'Pipi_new_testcase_opt_4d_undsafe_2'
                          }
        store_df = pd.DataFrame([compare_result])
        store_df.to_pickle(f'{folder_name}/PI_{max_episode_steps_list[max_eps_steps]}steps')
        """
        ####################################DDPG Stuff##############################################

        rew.gamma = 0
        # episodes will not abort, if limit is exceeded reward = -1
        rew.det_run = True
        rew.exponent = 0.5  # 1

        #net = Network.load('net/net_vctrl_single_inv_dq0.yaml')  # is used from vctrl_single_env, not needed here

        for used_model, wrapper_mode, used_number_past_vales in zip(model_name, wrapper, number_past_vals):

            if wrapper_mode == 'i_load':
                env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v0',
                                    reward_fun=rew.rew_fun_dq0,
                                    abort_reward=-1,  # no needed if in rew no None is given back
                                    # on_episode_reset_callback=cb.fire  # needed?
                                    obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                                'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                                'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'  # ],
                                        , 'r_load.resistor1.i', 'r_load.resistor2.i', 'r_load.resistor3.i'],
                                    max_episode_steps=max_episode_steps_list[max_eps_steps],
                                    model_params={'lc.resistor1.R': R_filter,
                                                  'lc.resistor2.R': R_filter,
                                                  'lc.resistor3.R': R_filter,
                                                  'lc.resistor4.R': R_filter,
                                                  'lc.resistor5.R': R_filter,
                                                  'lc.resistor6.R': R_filter,
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
                                                                                col='r_load.resistor3.R'),
                                                  # 'lc.capacitor1.v': 0,
                                                  # 'lc.capacitor2.v': 0,
                                                  # 'lc.capacitor3.v': 0,
                                                  # 'lc.inductor1.i': 0,
                                                  # 'lc.inductor2.i': 0,
                                                  # 'lc.inductor3.i': 0,
                                                  },
                                    )
            else:
                env_test = gym.make('experiments.P10.env:vctrl_single_inv_test-v0',
                                    reward_fun=rew.rew_fun_dq0,
                                    abort_reward=-1,  # no needed if in rew no None is given back
                                    # on_episode_reset_callback=cb.fire  # needed?
                                    obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                                'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                                'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'],
                                    # , 'r_load.resistor1.i', 'r_load.resistor2.i', 'r_load.resistor3.i'],
                                    max_episode_steps=max_episode_steps_list[max_eps_steps],
                                    model_params={'lc.resistor1.R': R_filter,
                                                  'lc.resistor2.R': R_filter,
                                                  'lc.resistor3.R': R_filter,
                                                  'lc.resistor4.R': R_filter,
                                                  'lc.resistor5.R': R_filter,
                                                  'lc.resistor6.R': R_filter,
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
                                                                                col='r_load.resistor3.R'),
                                                  # 'lc.capacitor1.v': 0,
                                                  # 'lc.capacitor2.v': 0,
                                                  # 'lc.capacitor3.v': 0,
                                                  # 'lc.inductor1.i': 0,
                                                  # 'lc.inductor2.i': 0,
                                                  # 'lc.inductor3.i': 0,
                                                  },
                                    )

            if wrapper_mode in ['past', 'i_load']:
                env_test = FeatureWrapper_pastVals(env_test, number_of_features=9 + used_number_past_vales * 3,
                                                   # training_episode_length=training_episode_length, (da aus pickle!)
                                                   recorder=mongo_recorder, n_trail=n_trail,
                                                   integrator_weight=integrator_weight,
                                                   antiwindup_weight=antiwindup_weight, gamma=1,
                                                   penalty_I_weight=0, penalty_P_weight=0,
                                                   number_past_vals=used_number_past_vales)


            elif wrapper_mode == 'no-I-term':
                env_test = BaseWrapper(env_test, number_of_features=6 + used_number_past_vales * 3,
                                       recorder=mongo_recorder, n_trail=n_trail, gamma=gamma,
                                       number_past_vals=used_number_past_vales)

            else:
                env_test = FeatureWrapper(env_test, number_of_features=11,
                                          recorder=mongo_recorder, integrator_weight=integrator_weight,
                                          antiwindup_weight=antiwindup_weight, gamma=1,
                                          penalty_I_weight=0,
                                          penalty_P_weight=0)  # , use_past_vals=True, number_past_vals=30)

            if wrapper_mode not in ['no-I-term', 'I-controller']:
                env_test.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

            # model2 = DDPG.load(model_path + f'model.zip')  # , env=env_test)
            print('Before load')

            model = DDPG.load(model_path + f'{used_model}', env=env_test)

            print('After load')

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

            if wrapper_mode not in ['no-I-term', 'I-controller']:
                env_test.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

            return_sum = 0.0
            limit_exceeded_in_test = False
            limit_exceeded_penalty = 0

            rew_list = []
            v_d = []
            v_q = []
            v_0 = []
            action_P0 = []
            action_P1 = []
            action_P2 = []
            action_I0 = []
            action_I1 = []
            action_I2 = []
            integrator_sum0 = []
            integrator_sum1 = []
            integrator_sum2 = []
            R_load = []

            ####### Run Test #########
            # agent ~ PI Controllerv using env
            # model ~ RL Controller using env_test
            # Both run in the same loop

            obs = env_test.reset()

            for step in tqdm(range(env_test.max_episode_steps), desc='steps', unit='step', leave=False):
                # for max_eps_steps in tqdm(range(len(max_episode_steps_list)), desc='steps', unit='step', leave=False):

                action, _states = model.predict(obs, deterministic=True)
                if step == 988:
                    asd = 1
                obs, rewards, done, info = env_test.step(action)
                action_P0.append(np.float64(action[0]))
                action_P1.append(np.float64(action[1]))
                action_P2.append(np.float64(action[2]))
                if wrapper_mode not in ['no-I-term', 'I-controller']:
                    action_I0.append(np.float64(action[3]))
                    action_I1.append(np.float64(action[4]))
                    action_I2.append(np.float64(action[5]))
                    integrator_sum0.append(np.float64(env_test.integrator_sum[0]))
                    integrator_sum1.append(np.float64(env_test.integrator_sum[1]))
                    integrator_sum2.append(np.float64(env_test.integrator_sum[2]))

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

            env_test.close()

            v_a = env_test.history.df['lc.capacitor1.v']
            v_b = env_test.history.df['lc.capacitor2.v']
            v_c = env_test.history.df['lc.capacitor3.v']
            i_a = env_test.history.df['lc.inductor1.i']
            i_b = env_test.history.df['lc.inductor2.i']
            i_c = env_test.history.df['lc.inductor3.i']
            R_load = (env_test.history.df['r_load.resistor1.R'].tolist())
            phase = env_test.history.df['inverter1.phase.0']  # env_test.env.net.components[0].phase
            v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
            i_dq0 = abc_to_dq0(np.array([i_a, i_b, i_c]), phase)

            i_d = i_dq0[0].tolist()
            i_q = i_dq0[1].tolist()
            i_0 = i_dq0[2].tolist()
            v_d = (v_dq0[0].tolist())
            v_q = (v_dq0[1].tolist())
            v_0 = (v_dq0[2].tolist())


            plt.plot(v_d)
            plt.show()
            plt.plot(v_q)
            plt.show()
            plt.plot(v_0)
            plt.show()

            # return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)

            print(f'RL: {(return_sum / env_test.max_episode_steps + limit_exceeded_penalty)}')
            # print(f'PI: {(return_sum_PI / env.max_episode_steps + limit_exceeded_penalty_PI)}')

            ts = time.gmtime()
            compare_result = {"Name": "comparison_PI_DDPG",
                              "model name": model_name,
                              "Wrapper": wrapper,
                              "used_number_past_vales": used_number_past_vales,
                              "time": ts,
                              "ActionP0": action_P0,
                              "ActionP1": action_P1,
                              "ActionP2": action_P2,
                              "ActionI0": action_I0,
                              "ActionI1": action_I1,
                              "ActionI2": action_I2,
                              "integrator_sum0": integrator_sum0,
                              "integrator_sum1": integrator_sum1,
                              "integrator_sum2": integrator_sum2,
                              "DDPG_model_path": model_path,
                              "Return DDPG": (return_sum / env_test.max_episode_steps + limit_exceeded_penalty),
                              "Reward DDPG": rew_list,
                              "env_hist_DDPG": env_test.env.history.df,
                              "max_episode_steps": str(max_episode_steps_list[max_eps_steps]),
                              "number of averages per run": num_average,
                              "info": "execution of RL agent on 10 s test case-loading values",
                              "optimization node": 'Thinkpad',
                              }
            store_df = pd.DataFrame([compare_result])
            store_df.to_pickle(f'{folder_name}/' + used_model + f'_{max_episode_steps_list[max_eps_steps]}steps')

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
