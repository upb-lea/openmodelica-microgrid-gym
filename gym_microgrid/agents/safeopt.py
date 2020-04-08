from typing import Dict, Union

import GPy
from GPy.kern import Kern
from safeopt import SafeOptSwarm

from gym_microgrid.agents.staticctrl import StaticControlAgent
from gym_microgrid.agents.util import MutableParams
from gym_microgrid.auxiliaries import Controller
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
        self.inital_Performance = None
        self._iterations = 0
        super().__init__(ctrls, observation_action_mapping, history)
        self.history.cols = ['J', 'Params']

    def reset(self):
        # reinstantiate kernel
        # self.kernel = type(self.kernel)(**self.kernel.to_dict())

        # Kp
        self.kernel = GPy.kern.Matern32(input_dim=1, variance=2., lengthscale=.005)

        # Ki
        # self.kernel = GPy.kern.Matern32(input_dim=1, variance=2., lengthscale=100.)

        # Kp, Ki
        # self.kernel = GPy.kern.Matern32(input_dim=1, variance=2., lengthscale=.005)

        self.params.reset()
        self.optimizer = None
        self.episode_reward = 0
        self.inital_Performance = None
        self._iterations = 0
        return super().reset()

    def observe(self, reward, terminated):
        self._iterations += 1
        self.episode_reward += reward or 0
        if terminated:
            # calculate MSE, divide Summed error by length of measurement
            self.episode_reward = self.episode_reward / self._iterations
            # safeopt update step
            self.update_params()
            # reset for new episode
            self.prepare_episode()
        # on other steps we don't need to do anything

    def update_params(self):
        if self.optimizer is None:
            # First Iteration
            self.inital_Performance = 1 / self.episode_reward

            # Norm for Safe-point
            J = 1 / self.episode_reward / self.inital_Performance

            # Kp
            bounds = [(-0.005, 0.02)]
            noise_var = 0.0005 ** 2  # Measurement noise sigma_omega

            # Ki
            # bounds = [(10, 200)]
            # noise_var = 0.05 ** 2  # Measurement noise sigma_omega

            # Kp, Ki
            # bounds = [(-0.005, 0.02), (10, 200)]
            # noise_var = 0.05 ** 2  # Measurement noise sigma_omega

            gp = GPy.models.GPRegression(np.array([self.params[:]]),
                                         np.array([[J]]), self.kernel,
                                         noise_var=noise_var)
            self.optimizer = SafeOptSwarm(gp, 0.5, bounds=bounds, threshold=1)

        else:

            # J = self.episode_reward - self.inital_reward + 0.5
            J = 1 / self.episode_reward / self.inital_Performance

            self.optimizer.add_new_data_point(self.params[:], J)

        self.history.append([J, self.params[:]])
        self.params[:] = self.optimizer.optimize()


    def render(self):
        plt.figure()
        self.optimizer.plot(1000)
        plt.show()
