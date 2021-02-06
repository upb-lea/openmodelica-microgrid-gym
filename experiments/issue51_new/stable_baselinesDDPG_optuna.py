import logging
from datetime import datetime
from os import makedirs
from typing import List

import pandas as pd
import gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch as th
import torch.nn as nn
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import nested_map

np.random.seed(0)

folder_name = 'DDPG/'
experiment_name = 'DDPG_CC_Reward_MRE_randsearch'
timestamp = datetime.now().strftime(f'%Y.%b.%d_%X')

makedirs(folder_name + experiment_name + timestamp)

train_steps = 5000

# Simulation definitions
net = Network.load('../../net/net_single-inv-curr.yaml')
max_episode_steps = 300  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc1.inductor{k}.i' for k in '123'], [f'inverter1.i_ref.{k}' for k in '012']])

    def rew_fun(self, cols: List[str], data: np.ndarray, risk) -> float:
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
        ISPabc_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = np.sum((np.abs((ISPabc_master - Iabc_master)) / iLimit) ** 0.5, axis=0) \
                + -np.sum(mu * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0)
        error /= max_episode_steps

        return -np.clip(error.squeeze(), 0, 1e5)


def xylables(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    ax.grid(which='both')
    fig.savefig(f'{folder_name + experiment_name + timestamp}/Inductor_currents{datetime.now()}.pdf')
    plt.show()


env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
               reward_fun=Reward().rew_fun,
               viz_cols=[
                   PlotTmpl([[f'lc1.inductor{i}.i' for i in '123'], [f'inverter1.i_ref.{k}' for k in '012']],
                            callback=xylables,
                            color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                            style=[[None], ['--']]
                            ),
               ],
               viz_mode='episode',
               max_episode_steps=max_episode_steps,
               net=net,
               model_path='../../omg_grid/grid.network_singleInverter.fmu',
               is_normalized=True,
               log_level=logging.WARNING)

with open(f'{folder_name + experiment_name + timestamp}/env.txt', 'w') as f:
    print(str(env), file=f)
env = Monitor(env)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))



def objective(trail):
    # DDPG learning parameters
    gamma = 0.9  # discount factor
    batch_size = 128
    memory_interval = 1

    noise_var = 0.2
    noise_theta = 5  # stiffness of OU
    alpha_lRelu = 0.1
    weigth_regularizer = 0.01

    memory_lim = 5000  # = buffersize?
    warm_up_steps_actor = 2048
    warm_up_steps_critic = 1024
    target_model_update = 1000

    # NN architecture
    actor_hidden_size = 100  # Using LeakyReLU
    # output linear
    critic_hidden_size_1 = 75  # Using LeakyReLU
    critic_hidden_size_2 = 75  # Using LeakyReLU
    critic_hidden_size_3 = 75  # Using LeakyReLU
    # output linear

    n_actions = env.action_space.shape[-1]


    max_learning_steps = train_steps

    learning_rate = trail.suggest_loguniform("lr", 1e-5, 1)

    class RecordEnvCallback(BaseCallback):
        def _on_step(self) -> bool:
            obs = env.reset()
            for _ in range(max_episode_steps):
                env.render()
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    break
            env.close()
            env.reset()
            return True

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                sigma=noise_var * np.ones(n_actions), dt=net.ts)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size], qf=[critic_hidden_size_1,
                                                                                                  critic_hidden_size_2,
                                                                                                  critic_hidden_size_3]))

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name + experiment_name + timestamp}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=memory_lim, learning_starts=warm_up_steps_critic,
                 batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
                 train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    checkpoint_on_event = CheckpointCallback(save_freq=1000,
                                             save_path=f'{folder_name + experiment_name + timestamp}/checkpoints/')
    record_env = RecordEnvCallback()
    plot_callback = EveryNTimesteps(n_steps=1000, callback=record_env)
    model.learn(total_timesteps=max_learning_steps, callback=[checkpoint_on_event, plot_callback])

    model.save(f'{folder_name + experiment_name + timestamp}/model.zip')

    return_sum = 0.0
    obs = env.reset()
    while True:

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        return_sum += rewards
        if done:
            break

    return return_sum


study = optuna.create_study(direction='maximize', storage='sqlite:///db.sqlite3')
# change to MAXIMIZE
study.optimize(objective, n_trials=2)

pd.Series(index=[trail.params['lr'] for trail in study.trials], data=[trail.value for trail in study.trials]).scatter()
