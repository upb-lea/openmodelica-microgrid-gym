from typing import Dict, Union, Tuple

import GPy
import pandas as pd
from GPy import Parameterized
from GPy.kern import Kern
from safeopt import SafeOptSwarm

from gym_microgrid.agents.staticctrl import StaticControlAgent
from gym_microgrid.controllers import Controller

import numpy as np


class SafeOptAgent(StaticControlAgent):
    def __init__(self, mutable_params: Union[dict, list], kernel: Kern, ctrls: Dict[str, Controller],
                 observation_action_mapping: dict):
        self.default_params = list(mutable_params.values()) if isinstance(mutable_params, dict) else mutable_params
        self.kernel = kernel

        self.params = None
        self.score = None
        self.optimizer = None
        super().__init__(ctrls, observation_action_mapping)

    def reset(self):
        # reinstantiate kernel
        # self.kernel = type(self.kernel)(**self.kernel.to_dict())
        self.kernel = GPy.kern.Matern32(input_dim=1, lengthscale=.01)

        print(np.array(self.default_params))
        # TODO: correctly copy the parameters
        self.params = self.default_params.copy()
        self.score = None
        self.optimizer = None
        return super().reset()

    def observe(self, reward, terminated):
        self.episode_reward += reward or 0
        if terminated:
            self.score = self.episode_reward
            # reset episode reward
            self.episode_reward = 0
            # safeopt update step
            self.update_params()
        # on other steps we don't need to do anything

    def update_params(self):
        if self.optimizer is None:
            bounds = [(-0.1, 0.2)]
            noise_var = 0.05 ** 2
            gp = GPy.models.GPRegression(np.array([[self.params]]), np.array([[self.score]]), self.kernel,
                                         noise_var=noise_var)
            self.optimizer = SafeOptSwarm(gp, 0., bounds=bounds, threshold=1)
        else:
            # TODO correctly change the parameters
            self.optimizer.add_new_data_point(self.params, self.score)
        self.params = self.optimizer.optimize()

    def render(self):
        # TODO plot the GP
        pass
