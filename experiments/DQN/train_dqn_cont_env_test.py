from multiprocessing import Pool
from typing import Union

import gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import DQN

import matplotlib.pyplot as plt

# env = gym.make("CartPole-v0")
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn, GymObs

from experiments.DQN.env.Custom_Cartpole import CartPoleEnv


# return_all_agents = []

class FeatureWrapper(Monitor):

    def __init__(self, env, training_episode_length):
        """
        :
        """
        super().__init__(env)
        self.training_episode_length = training_episode_length
        self._n_training_steps = 0
        self.episode_return = []

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        """

        obs, reward, done, info = super().step(action)

        self._n_training_steps += 1

        if self._n_training_steps == self.training_episode_length:
            done = True
            # info["timelimit_reached"] = True

        if self._n_training_steps == self.training_episode_length or done:
            self.episode_return.append(sum(self.rewards))

        return obs, reward, done, info

    def reset(self, **kwargs) -> GymObs:
        """
                state_constraints = [[  - 2.4,    2.4],
                                     [     -7,      7],
                                     [ -np.pi, +np.pi],
                                     [    -10,     10]]
        """
        obs = super().reset()

        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[0] = self.np_random.uniform(low=-2.4, high=2.4, size=(1,))
        self.state[1] = self.np_random.uniform(low=-7, high=7, size=(1,))
        self.state[2] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(1,))
        self.state[3] = self.np_random.uniform(low=-10, high=10, size=(1,))
        self.steps_beyond_done = None

        self._n_training_steps = 0

        return np.array(self.state)


# for i in range(2):
def bla(idx):
    env = FeatureWrapper(CartPoleEnv(), training_episode_length=200)
    # env = FeatureWrapper(gym.make("CartPole-v1"))

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[200, 200, 200])

    model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=100000, learning_starts=1000, batch_size=32,
                tau=0.001,
                gamma=0.99, train_freq=(1, "step"), gradient_steps=-1, optimize_memory_usage=False,
                target_update_interval=1,
                exploration_fraction=0.4, exploration_initial_eps=0.5, exploration_final_eps=0.01, max_grad_norm=1000,
                tensorboard_log='TB_log/', create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, seed=None,
                device='auto',
                _init_setup_model=True)

    model.q_net.q_net._modules['1'].negative_slope = 0.1
    model.q_net.q_net._modules['3'].negative_slope = 0.1
    model.q_net.q_net._modules['5'].negative_slope = 0.1
    model.q_net_target.q_net._modules['1'].negative_slope = 0.1
    model.q_net_target.q_net._modules['3'].negative_slope = 0.1
    model.q_net_target.q_net._modules['5'].negative_slope = 0.1

    # model = DQN("MlpPolicy", env, verbose=1)

    # learn(total_timesteps, callback=None, log_interval=4, eval_env=None, eval_freq=- 1, n_eval_episodes=5,
    #      tb_log_name='DQN', eval_log_path=None, reset_n um_timesteps=True)
    model.learn(total_timesteps=100000, log_interval=4)
    return np.array(env.episode_return)


with Pool(1) as p:
    return_all_agents = p.map(bla, range(25))

    pvc = 1

# return_all_agents.append(np.array(env.episode_return))


# asd = min(map(len,return_all_agents))
# np_return_all_agents = np.array([l[:asd] for l in return_all_agents])

df = pd.DataFrame(return_all_agents)

df.to_pickle("DQN_without_fix")

m = df.mean()
s = df.std()

episode = pd.Series(range(0, df.shape[1]))

plt.plot(episode, m)
plt.fill_between(episode, m - s, m + s, facecolor='r')
plt.ylabel('Average return')
plt.xlabel('Episode')
plt.show()
"""
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
"""
