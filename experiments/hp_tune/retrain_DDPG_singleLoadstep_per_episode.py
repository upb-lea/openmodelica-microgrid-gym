import json
import platform
import time
from functools import partial

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# from agents.my_ddpg import myDDPG
from stochastic.processes import VasicekProcess

from experiments.hp_tune.env.env_wrapper import FeatureWrapper
from experiments.hp_tune.env.random_load import RandomLoad
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net, CallbackList  # , folder_name
from experiments.hp_tune.util.config import cfg
from experiments.hp_tune.util.recorder import Recorder

# np.random.seed(0)
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import RandProcess

model_path = 'experiments/hp_tune/trained_models/study_22_run_11534/'

folder_name = cfg['STUDY_NAME'] + '_retrain'
node = platform.uname().node
file_congfig = open(model_path +
                    'PC2_DDPG_Vctrl_single_inv_22_newTestcase_Trial_number_11534_0.json', )
trial_config = json.load(file_congfig)
print('Config-Params:')
print(*trial_config.items(), sep='\n')
# mongo_recorder = Recorder(database_name=folder_name)
mongo_recorder = Recorder(node=node,
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne or locally as json to cfg[meas_data_folder]

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC


def xylables_v(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
    ax.grid(which='both')
    # ax.set_xlim([0, 0.005])
    ts = time.gmtime()
    # fig.savefig(
    #    f'{folder_name + experiment_name}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.show()


def xylables_i(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    ax.grid(which='both')
    # fig.savefig(f'{folder_name + experiment_name + n_trail}/Inductor_currents.pdf')
    plt.close()


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    # ts = time.gmtime()
    # fig.savefig(f'{folder_name + experiment_name}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.show()


# plant
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F
# R = 40  # nomVoltPeak / 7.5   # / Ohm
lower_bound_load = -10  # to allow maximal load that draws i_limit
upper_bound_load = 200  # to apply symmetrical load bounds
lower_bound_load_clip = 14  # to allow maximal load that draws i_limit (let exceed?)
upper_bound_load_clip = 200  # to apply symmetrical load bounds
lower_bound_load_clip_std = 2
upper_bound_load_clip_std = 0
R = np.random.uniform(low=lower_bound_load, high=upper_bound_load)

gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=800, vol=40, mean=R), initial=R,
                  bounds=(lower_bound_load, upper_bound_load))


def experiment_fit_DDPG(number_learning_steps,
                        gamma=trial_config['gamma'], n_trail=trial_config['Trial number'],
                        training_episode_length=trial_config['training_episode_length'],
                        integrator_weight=trial_config['integrator_weight'],
                        antiwindup_weight=trial_config['antiwindup_weight'],
                        penalty_I_weight=trial_config['penalty_I_weight'],
                        penalty_P_weight=trial_config['penalty_P_weight'],
                        t_start_penalty_I=trial_config['penalty_I_decay_start'],
                        t_start_penalty_P=trial_config['penalty_P_decay_start'],
                        actor_number_layers=trial_config['actor_number_layers'],
                        alpha_relu_actor=trial_config['alpha_relu_actor'],
                        critic_number_layers=trial_config['critic_number_layers'],
                        alpha_relu_critic=trial_config['alpha_relu_critic'],

                        ):
    rand_load_train = RandomLoad(training_episode_length, net.ts, gen,
                                 bounds=(lower_bound_load_clip, upper_bound_load_clip),
                                 bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))

    cb = CallbackList()
    # set initial = None to reset load random in range of bounds
    cb.append(partial(gen.reset))  # , initial=np.random.uniform(low=lower_bound_load, high=upper_bound_load)))
    cb.append(rand_load_train.reset)

    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=1, error_exponent=0.5, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

    """
    env = gym.make('experiments.hp_tune.env:vctrl_single_inv_train-v1',
                   reward_fun=rew.rew_fun_dq0,
                   max_episode_steps=training_episode_length,
                   abort_reward=-1,
                   obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                               'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                               'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                   )
    """

    env = gym.make('experiments.hp_tune.env:vctrl_single_inv_train-v1',
                   reward_fun=rew.rew_fun_dq0,
                   abort_reward=-1,
                   obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                               'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                               'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'],
                   max_episode_steps=training_episode_length,

                   viz_mode='episode',
                   viz_cols=[
                       PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'inverter1.v_ref.{k}' for k in '012']],
                                callback=xylables_v,
                                color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                style=[[None], ['--']]
                                ),
                       PlotTmpl([[f'lc.inductor{i}.i' for i in '123'], [f'inverter1.i_ref.{k}' for k in '012']],
                                callback=xylables_i,
                                color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                style=[[None], ['--']]
                                ),
                       PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                                callback=xylables_R,
                                color=[['b', 'r', 'g']],
                                style=[[None]]
                                )
                   ],
                   # max_episode_steps=max_episode_steps,
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
                                 'r_load.resistor1.R': rand_load_train.one_random_loadstep_per_episode,
                                 'r_load.resistor2.R': rand_load_train.clipped_step,
                                 'r_load.resistor3.R': rand_load_train.clipped_step,
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
                   on_episode_reset_callback=cb.fire,
                   is_normalized=True,
                   action_time_delay=1
                   )

    env = FeatureWrapper(env, number_of_features=11, training_episode_length=training_episode_length,
                         recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                         antiwindup_weight=antiwindup_weight, gamma=gamma,
                         penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                         t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                         number_learing_steps=number_learning_steps)

    # todo: Upwnscale actionspace - lessulgy possible? Interaction pytorch...
    env.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

    model = DDPG.load(model_path + f'model.zip', env=env, tensorboard_log=model_path)

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

    # todo: Downscale actionspace - lessulgy possible? Interaction pytorch...
    env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

    # start training
    model.learn(total_timesteps=number_learning_steps)

    # Log Train-info data
    train_data = {"Name": "After_Training",
                  "Mean_eps_reward": env.reward_episode_mean,
                  "Trial number": n_trail,
                  "Database name": folder_name,
                  "Sum_eps_reward": env.get_episode_rewards()
                  }
    mongo_recorder.save_to_json('Trial_number_' + n_trail, train_data)

    model.save(model_path + f'model_retrained.zip')

    ####### Run Test #########
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
                                    'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                        )
    env_test = FeatureWrapper(env_test, number_of_features=11, integrator_weight=integrator_weight,
                              recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                              gamma=1, penalty_I_weight=0, penalty_P_weight=0)
    # using gamma=1 and rew_weigth=3 we get the original reward from the env without penalties
    obs = env_test.reset()
    phase_list = []
    phase_list.append(env_test.env.net.components[0].phase)

    rew_list = []
    aP0 = []
    aP1 = []
    aP2 = []
    aI0 = []
    aI1 = []
    aI2 = []
    integrator_sum0 = []
    integrator_sum1 = []
    integrator_sum2 = []
    va = []
    vb = []
    vc = []
    v_ref0 = []
    v_ref1 = []
    v_ref2 = []
    ia = []
    ib = []
    ic = []
    R_load = []
    for step in range(env_test.max_episode_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)
        phase_list.append(env_test.env.net.components[0].phase)
        aP0.append(np.float64(action[0]))
        aP1.append(np.float64(action[1]))
        aP2.append(np.float64(action[2]))
        aI0.append(np.float64(action[3]))
        aI1.append(np.float64(action[4]))
        aI2.append(np.float64(action[5]))
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

        if step % 1000 == 0 and step != 0:
            va.extend(env_test.history[env_test.viz_col_tmpls[0].vars[0]].copy().values.tolist())
            vb.extend(env_test.history[env_test.viz_col_tmpls[0].vars[1]].copy().values.tolist())
            vc.extend(env_test.history[env_test.viz_col_tmpls[0].vars[2]].copy().values.tolist())
            v_ref0.extend(env_test.history[env_test.viz_col_tmpls[0].vars[3]].copy().values.tolist())
            v_ref1.extend(env_test.history[env_test.viz_col_tmpls[0].vars[4]].copy().values.tolist())
            v_ref2.extend(env_test.history[env_test.viz_col_tmpls[0].vars[5]].copy().values.tolist())
            ia.extend(env_test.history[env_test.viz_col_tmpls[1].vars[0]].copy().values.tolist())
            ib.extend(env_test.history[env_test.viz_col_tmpls[1].vars[1]].copy().values.tolist())
            ic.extend(env_test.history[env_test.viz_col_tmpls[1].vars[2]].copy().values.tolist())
            R_load.extend(env_test.history[env_test.viz_col_tmpls[2].vars[1]].copy().values.tolist())

            env_test.close()
            obs = env_test.reset()
            phase_list.append(env_test.env.net.components[0].phase)

        if done:
            env_test.close()
            # print(limit_exceeded_in_test)
            break

    ts = time.gmtime()
    test_after_training = {"Name": "Test",
                           "time": ts,
                           "Reward": rew_list,
                           "lc_capacitor1_v": va,
                           "lc_capacitor2_v": vb,
                           "lc_capacitor3_v": vc,
                           "inverter1_v_ref_0": v_ref0,
                           "inverter1_v_ref_1": v_ref1,
                           "inverter1_v_ref_2": v_ref2,
                           "lc_inductor1_i": ia,
                           "lc_inductor2_i": ib,
                           "lc_inductor3_i": ic,
                           "r_load_resistor1_R": R_load,
                           "ActionP0": aP0,
                           "ActionP1": aP1,
                           "ActionP2": aP2,
                           "ActionI0": aI0,
                           "ActionI1": aI1,
                           "ActionI2": aI2,
                           "integrator_sum0": integrator_sum0,
                           "integrator_sum1": integrator_sum1,
                           "integrator_sum2": integrator_sum2,
                           "Phase": phase_list,
                           "Node": platform.uname().node,
                           "End time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime()),
                           "Reward function": 'rew.rew_fun_dq0',
                           "Trial number": n_trail,
                           "Database name": folder_name,
                           "Info": "Delay, obs=[v_mess,sp_dq0, i_mess_dq0, error_mess_sp, last_action, sin/cos(phase),"
                                   "integrator_zustand(delayed!), genutzte Aktion (P-anteil)]; "
                                   "Reward = MRE, PI-Approch using AntiWindUp"
                                   "without abort! (risk=0 manullay in env); only voltage taken into account in reward!"}

    """
    In new testenv not used, because then only the last episode is stored
    # Add v-&i-measurements
    test_after_training.update({env_test.viz_col_tmpls[j].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                })
    test_after_training.update({env_test.viz_col_tmpls[2].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                })
    """

    mongo_recorder.save_to_json('Trial_number_' + n_trail, test_after_training)

    print(return_sum / env_test.max_episode_steps + limit_exceeded_penalty)

    return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)


experiment_fit_DDPG(10000)
