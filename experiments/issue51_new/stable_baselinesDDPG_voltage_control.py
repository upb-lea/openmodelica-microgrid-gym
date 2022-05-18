from datetime import datetime
from functools import partial
from itertools import accumulate
from os import makedirs

import time

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import numpy as np
import pandas as pd
import optuna

import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stochastic.processes import VasicekProcess

from experiments.issue51_new.env.rewards import Reward
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import nested_map, RandProcess

np.random.seed(0)

folder_name = 'DDPG_VC_randLoad_exploringStarts/'
# experiment_name = 'DDPG_VC_Reward_MRE_reward_NOT_NORMED'
experiment_name = 'DDPG_VC_bestParamsTest'
timestamp = datetime.now().strftime(f'_%Y.%b.%d_%X')



makedirs(folder_name, exist_ok=True)

# Simulation definitions
net = Network.load('../../net/net_single-inv-Paper_Loadstep.yaml')
max_episode_steps = 1000  # number of simulation steps per episode
# num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
# iLimit = 30  # inverter current limit / A
# iNominal = 20  # nominal inverter current / A
mu_c = 2  # factor for barrier function (see below)
mu_v = 2  # factor for barrier function (see below)
i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC
# plant
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F
R = 28  # nomVoltPeak / 7.5   # / Ohm
lower_bound_load = 11  # to allow maximal load that draws i_limit (toDo: let exceed?)
upper_bound_load = 45  # to apply symmetrical load bounds

loadstep_timestep = max_episode_steps / 2

gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=1000, vol=10, mean=R), initial=R,
                  bounds=(lower_bound_load, upper_bound_load))


class RandomLoad:
    def __init__(self, max_episode_steps, ts, loadstep_time=None):
        self.max_episode_steps = max_episode_steps
        self.ts = ts
        if loadstep_time is None:
            self.loadstep_time = np.random.randint(0, self.max_episode_steps)
        else:
            self.loadstep_time = loadstep_time

    def reset(self, loadstep_time=None):
        if loadstep_time is None:
            self.loadstep_time = np.random.randint(0, self.max_episode_steps)
        else:
            self.loadstep_time = loadstep_time

    def load_step(self, t, gain):
        """
        Changes the load parameters
        :param t:
        :param gain: device parameter
        :return: Sample from SP
        """
        # Defines a load step after 0.01 s
        if self.loadstep_time * self.ts < t <= self.loadstep_time * self.ts + self.ts:
            gen.proc.mean = gain * 0.55
            gen.reserve = gain * 0.55
        elif t <= self.ts:
            gen.proc.mean = gain

        return gen.sample(t)


class CallbackList(list):
    def fire(self, *args, **kwargs):
        for listener in self:
            listener(*args, **kwargs)


# def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, n_trail):
def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                        actor_hidden_size, critic_hidden_size, n_trail):
    makedirs(folder_name + experiment_name + n_trail, exist_ok=True)

    def xylables_i(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        fig.savefig(f'{folder_name + experiment_name + n_trail}/Inductor_currents.pdf')
        plt.close()

    def xylables_v(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        #ax.set_xlim([0, 0.005])
        ts = time.gmtime()
        fig.savefig(f'{folder_name + experiment_name + n_trail}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
        plt.close()

    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
        fig.savefig(f'{folder_name + experiment_name + n_trail}/Load.pdf')
        plt.close()

    rew = Reward(v_nom, v_lim, v_DC, gamma, use_gamma_normalization=use_gamma_in_rew)
    rand_load = RandomLoad(max_episode_steps, net.ts)

    cb = CallbackList()
    cb.append(partial(gen.reset, initial=R))
    cb.append(rand_load.reset)

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=rew.rew_fun,
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
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
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
                                 'r_load.resistor1.R': partial(rand_load.load_step, gain=R),
                                 'r_load.resistor2.R': partial(rand_load.load_step, gain=R),
                                 'r_load.resistor3.R': partial(rand_load.load_step, gain=R),
                                 'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_lim,
                                                                                high=v_lim) if t == 0 else None,
                                 'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_lim,
                                                                                high=v_lim) if t == 0 else None,
                                 'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_lim,
                                                                                high=v_lim) if t == 0 else None,
                                 'lc.inductor1.i': lambda t: np.random.uniform(low=-i_lim,
                                                                               high=i_lim) if t == 0 else None,
                                 'lc.inductor2.i': lambda t: np.random.uniform(low=-i_lim,
                                                                               high=i_lim) if t == 0 else None,
                                 'lc.inductor3.i': lambda t: np.random.uniform(low=-i_lim,
                                                                               high=i_lim) if t == 0 else None,
                                 },
                   net=net,
                   model_path='../../omg_grid/grid.paper_loadstep.fmu',
                   # on_episode_reset_callback=partial(gen.reset, initial=R),
                   # on_episode_reset_callback=[partial(gen.reset, initial=R), partial(rand_load.reset)],
                   # on_episode_reset_callback=rand_load.reset,
                   on_episode_reset_callback=cb.fire,
                   is_normalized=True
                   )

    with open(f'{folder_name + experiment_name + n_trail}/env.txt', 'w') as f:
        print(str(env), file=f)
    env = Monitor(env)

    # DDPG learning parameters
    # gamma = 0.9  # discount factor
    batch_size = batch_size
    memory_interval = 1
    # alpha_actor = 5e-6
    # learning_rate = 5e-3

    # learning_rate = trail.suggest_loguniform("lr", 1e-5, 1)

    noise_var = 0.2
    noise_theta = 5  # stiffness of OU
    #alpha_lRelu = alpha_lRelu
    weigth_regularizer = 0.5

    memory_lim = 5000  # = buffersize?
    warm_up_steps_actor = 2048
    warm_up_steps_critic = 1024
    target_model_update = 1000

    # NN architecture
    actor_hidden_size = actor_hidden_size  # Using LeakyReLU
    # output linear
    critic_hidden_size_1 = critic_hidden_size  # Using LeakyReLU
    critic_hidden_size_2 = critic_hidden_size  # Using LeakyReLU
    critic_hidden_size_3 = critic_hidden_size  # Using LeakyReLU

    # output linear

    class RecordEnvCallback(BaseCallback):
        def _on_step(self) -> bool:
            rewards = []
            obs = env.reset()
            for _ in range(max_episode_steps):
                env.render()
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                if done:
                    break

            acc_Reward = list(accumulate(rewards))

            plt.plot(rewards)
            plt.xlabel(r'$t\,/\,\mathrm{s}$')
            plt.ylabel('$Reward$')
            plt.grid(which='both')
            ts = time.gmtime()
            plt.savefig(f'{folder_name + experiment_name + n_trail}/reward{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
            plt.close()

            # plt.plot(acc_Reward)
            # plt.xlabel(r'$t\,/\,\mathrm{s}$')
            # plt.ylabel('$Reward_sum$')
            # plt.grid(which='both')
            # plt.savefig(f'{folder_name + experiment_name + timestamp}/reward_sum_{datetime.now()}.pdf')
            # plt.show()

            env.close()
            env.reset()
            return True

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                sigma=noise_var * np.ones(n_actions), dt=net.ts)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size], qf=[critic_hidden_size_1,
                                                                                                  critic_hidden_size_2,
                                                                                                  critic_hidden_size_3]))

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name + experiment_name + n_trail}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=memory_lim, learning_starts=warm_up_steps_critic,
                 batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
                 train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    model.actor.mu._modules['0'].weight.data = model.actor.mu._modules['0'].weight.data * weight_scale
    model.actor.mu._modules['2'].weight.data = model.actor.mu._modules['2'].weight.data * weight_scale
    model.actor_target.mu._modules['0'].weight.data = model.actor_target.mu._modules['0'].weight.data * weight_scale
    model.actor_target.mu._modules['2'].weight.data = model.actor_target.mu._modules['2'].weight.data * weight_scale
    # model.actor.mu._modules['0'].bias.data = model.actor.mu._modules['0'].bias.data * weight_bias_scale
    # model.actor.mu._modules['2'].bias.data = model.actor.mu._modules['2'].bias.data * weight_bias_scale

    checkpoint_on_event = CheckpointCallback(save_freq=10000,
                                             save_path=f'{folder_name + experiment_name + n_trail}/checkpoints/')
    record_env = RecordEnvCallback()
    plot_callback = EveryNTimesteps(n_steps=10000, callback=record_env)
    model.learn(total_timesteps=200000, callback=[checkpoint_on_event, plot_callback])

    model.save(f'{folder_name + experiment_name + n_trail}/model.zip')

    return_sum = 0.0
    obs = env.reset()
    rew.gamma = 0
    while True:

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        return_sum += rewards
        if done:
            break

    return return_sum


def objective(trail):
    learning_rate = 0.0004  # trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = 0.7  # trail.suggest_loguniform("gamma", 0.5, 0.99)
    weight_scale = 0.02  # trail.suggest_loguniform("weight_scale", 5e-4, 1)  # 0.005
    batch_size = 128  # trail.suggest_int("batch_size", 32, 1024)  # 128
    # alpha_lRelu = trail.suggest_loguniform("alpha_lRelu", 0.0001, 0.5)  #0.1
    actor_hidden_size = 100  # trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
    # output linear
    critic_hidden_size = 100  # trail.suggest_int("critic_hidden_size", 10, 500)  # # Using LeakyReLU

    # memory_interval = 1
    # noise_var = 0.2
    # noise_theta = 5  # stiffness of OU
    # weigth_regularizer = 0.5

    memory_lim = 5000  # = buffersize?
    warm_up_steps_actor = 2048
    warm_up_steps_critic = 1024
    target_model_update = 1000

    #

    # n_trail = str(0)#str(trail.number)
    # gamma = 0.75
    use_gamma_in_rew = 1

    return experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                               actor_hidden_size, critic_hidden_size, str(trail.number))


# study = optuna.load_study(study_name="V-crtl_learn_use_hyperopt_params", storage="sqlite:///Hyperotp_visualization/test.sqlite3")

study = optuna.create_study(study_name="V-crtl_stochLoad_single_Loadstep_exploring_starts",
                            direction='maximize', storage=f'sqlite:///{folder_name}optuna_data.sqlite3')

study.optimize(objective, n_trials=1)
print(study.best_params, study.best_value)

# pd.Series(index=[trail.params['lr'] for trail in study.trials], data=[trail.value for trail in study.trials]).scatter()
