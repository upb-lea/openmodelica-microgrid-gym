from typing import Dict

import pandas as pd

from gym_microgrid.agents import Agent
from gym_microgrid.common.pd_convert import fill_params
from gym_microgrid.controllers import Controller

import numpy as np


class StaticControlAgent(Agent):
    def __init__(self, ctrls: Dict[str, Controller], observation_action_mapping: Dict):
        super().__init__()
        self.episode_reward = 0
        self.controllers = ctrls
        self.obs_mapping = observation_action_mapping

    def reset(self):
        for ctrl in self.controllers.values():
            ctrl.reset()

    def act(self, state: pd.DataFrame):
        """

        :param state: the agent is stateless. the state is stored in the controllers.
        Therefore we simply pass the observation from the environment into the controllers.
        """
        parameters = fill_params(state, self.obs_mapping)
        controls = list()
        for key, params in parameters:
            controls.append(self.controllers[key].step(*params)[0])

        return np.append(*controls)

    def observe(self, reward, terminated):
        self.episode_reward += reward or 0
        if terminated:
            # safeopt update step
            # TODO
            # reset episode reward
            self.episode_reward = 0
        # on other steps we don't need to do anything

    def render(self):
        # TODO plot the GP
        pass
