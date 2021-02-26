import time
from typing import Union

import matplotlib.pyplot as plt

import gym
import numpy as np
import optuna
import torch as th
from pymongo import MongoClient
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net, folder_name, max_episode_steps
from experiments.hp_tune.util.record_env import RecordEnvCallback
from openmodelica_microgrid_gym.env import PlotTmpl

np.random.seed(0)


def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                        actor_hidden_size, critic_hidden_size, noise_var, noise_theta, error_exponent, n_trail):
    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent)

    def xylables_v(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        # ax.set_xlim([0, 0.005])
        ts = time.gmtime()
        fig.savefig(
            f'{folder_name}/{n_trail}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
        plt.show()

    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
        ts = time.gmtime()
        fig.savefig(f'{folder_name}/{n_trail}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
        plt.show()

    env = gym.make('experiments.hp_tune.env:vctrl_single_inv_train-v0',
                   reward_fun=rew.rew_fun,
                   abort_reward=-(1 - gamma),
                   viz_cols=[
                       PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'inverter1.v_ref.{k}' for k in '012']],
                                callback=xylables_v,
                                color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                style=[[None], ['--']]
                                ),
                       PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                                callback=xylables_R,
                                color=[['b', 'r', 'g']],
                                style=[[None]]
                                )
                   ],
                   # on_episode_reset_callback=cb.fire  # needed?
                   )

    # toDo: Whatfore?
    env = Monitor(env)

    n_actions = env.action_space.shape[-1]
    noise_var = noise_var  # 20#0.2
    noise_theta = noise_theta  # 50 # stiffness of OU
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                sigma=noise_var * np.ones(n_actions), dt=net.ts)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size], qf=[critic_hidden_size,
                                                                                                  critic_hidden_size,
                                                                                                  critic_hidden_size]))

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=5000, learning_starts=1024,
                 batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
                 train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    model.actor.mu._modules['0'].weight.data = model.actor.mu._modules['0'].weight.data * weight_scale
    model.actor.mu._modules['2'].weight.data = model.actor.mu._modules['2'].weight.data * weight_scale
    model.actor_target.mu._modules['0'].weight.data = model.actor_target.mu._modules['0'].weight.data * weight_scale
    model.actor_target.mu._modules['2'].weight.data = model.actor_target.mu._modules['2'].weight.data * weight_scale
    # model.actor.mu._modules['0'].bias.data = model.actor.mu._modules['0'].bias.data * weight_bias_scale
    # model.actor.mu._modules['2'].bias.data = model.actor.mu._modules['2'].bias.data * weight_bias_scale

    # todo: instead /here? store reward per step?!
    plot_callback = EveryNTimesteps(n_steps=1000, callback=RecordEnvCallback(env, model, max_episode_steps))
    model.learn(total_timesteps=2000, callback=[plot_callback])

    # todo: instead: store model(/weights+bias?) to database
    model.save(f'{folder_name}/{n_trail}/model.zip')

    return_sum = 0.0

    rew.gamma = 0
    # episodes will not abort, if limit is exceeded reward = -1
    rew.det_run = True
    limit_exceeded_in_test = False
    limit_exceeded_penalty = 0
    env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v0',
                        reward_fun=rew.rew_fun,
                        abort_reward=-1,  # no needed if in rew no None is given back
                        # on_episode_reset_callback=cb.fire  # needed?
                        viz_cols=[
                            PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'inverter1.v_ref.{k}' for k in '012']],
                                     callback=xylables_v,
                                     color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                                     style=[[None], ['--']]
                                     ),
                            PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                                     callback=xylables_R,
                                     color=[['b', 'r', 'g']],
                                     style=[[None]]
                                     )
                        ],
                        )

    obs = env_test.reset()

    # toDo: - Use other Test-episode
    #       - Rückgabewert = (Summe der üblichen Rewards) / (Anzahl steps Validierung) + (Penalty i.H.v. -1)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)

        if rewards == -1 and not limit_exceeded_in_test:
            # Set addidional penalty of -1 if limit is exceeded once in the test case
            limit_exceeded_in_test = True
            limit_exceeded_penalty = -1
        env_test.render()
        return_sum += rewards
        print(rewards)
        if done:
            env_test.close()
            print(limit_exceeded_in_test)
            break

    return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)


def objective(trail):
    learning_rate = trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = trail.suggest_loguniform("gamma", 0.1, 0.99)
    weight_scale = trail.suggest_loguniform("weight_scale", 5e-4, 1)  # 0.005
    batch_size = trail.suggest_int("batch_size", 32, 1024)  # 128
    actor_hidden_size = trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
    critic_hidden_size = trail.suggest_int("actor_hidden_size", 10, 500)  # 100
    n_trail = str(trail.number)
    use_gamma_in_rew = 1
    noise_var = trail.suggest_loguniform("lr", 0.01, 10)  # 2
    noise_theta = trail.suggest_loguniform("lr", 1, 50)  # 25  # stiffness of OU

    # toDo:
    error_exponent = trail.suggest_loguniform("error_exponent", 0.01, 4)
    # alpha_lRelu = trail.suggest_loguniform("alpha_lRelu", 0.0001, 0.5)  #0.1
    # memory_interval = 1
    # weigth_regularizer = 0.5
    # memory_lim = 5000  # = buffersize?
    # warm_up_steps_actor = 2048
    # warm_up_steps_critic = 1024  # learning starts?
    # target_model_update = 1000

    return experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                               actor_hidden_size, critic_hidden_size, noise_var, noise_theta, error_exponent, n_trail)


# toDo: postgresql instead of sqlite
study = optuna.create_study(study_name="V-crtl_stochLoad_single_Loadstep_exploring_starts",
                            direction='maximize', storage=f'sqlite:///{folder_name}optuna_data.sqlite3',
                            load_if_exists=True)

study.optimize(objective, n_trials=200)
