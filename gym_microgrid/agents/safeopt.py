from typing import Dict, Union

import GPy
from GPy.kern import Kern
from safeopt import SafeOptSwarm

from gym_microgrid.agents.staticctrl import StaticControlAgent
from gym_microgrid.agents.util import MutableParams
from gym_microgrid.controllers import Controller
from gym_microgrid.env import EmptyHistory

import matplotlib.pyplot as plt

import numpy as np


class SafeOptAgent(StaticControlAgent):
    def __init__(self, mutable_params: Union[dict, list], kernel: Kern, ctrls: Dict[str, Controller],
                 observation_action_mapping: dict, history=EmptyHistory()):
        self.params = MutableParams(
            list(mutable_params.values()) if isinstance(mutable_params, dict) else mutable_params)
        self.kernel = kernel

        self.episode_reward = None
        self.optimizer = None
        self.inital_reward = None
        super().__init__(ctrls, observation_action_mapping, history)
        self.history.cols = ['J']

    def reset(self):
        # reinstantiate kernel
        # self.kernel = type(self.kernel)(**self.kernel.to_dict())
        self.kernel = GPy.kern.Matern32(input_dim=1, lengthscale=.01)

        self.params.reset()
        self.optimizer = None
        self.episode_reward = 0
        self.inital_reward = None
        return super().reset()

    def observe(self, reward, terminated):
        self.episode_reward += reward or 0
        if terminated:
            # calculate MSE, divide Summed error by length of measurement
            self.episode_reward = self.episode_reward / len(self.controllers['master'].history.df)
            # safeopt update step
            self.update_params()
            # reset for new episode
            self.prepare_episode()
        # on other steps we don't need to do anything

    def update_params(self):
        if self.optimizer is None:
            # First Iteration
            self.inital_reward = self.episode_reward
            # Norm for Safe-point
            # y_0 = y_meas = initial_reward
            J = 1 / (self.episode_reward / self.inital_reward)

            bounds = [(-0.005, 0.035)]
            noise_var = 0.05 ** 2
            gp = GPy.models.GPRegression(np.array([self.params[:]]),
                                         np.array([[J]]), self.kernel,
                                         noise_var=noise_var)
            self.optimizer = SafeOptSwarm(gp, 0., bounds=bounds, threshold=1)
        else:

            # y_safe = y_meas/y_norm
            # y = y_meas/y_norm - 0.95*y_safe
            # J_0 = y_meas - 0.95 * y_0
            # J.append(2 - J_0 / J_0)
            J = 1 / (self.episode_reward / self.inital_reward)

            self.optimizer.add_new_data_point(self.params[:], J)

        self.history.append([J])
        self.params[:] = self.optimizer.optimize()
        print(self.episode_reward / self.inital_reward)

    def render(self):
        plt.figure
        self.optimizer.plot(1000)
        plt.show()
