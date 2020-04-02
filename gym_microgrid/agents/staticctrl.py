from typing import Dict

from gym_microgrid.agents.agent import Agent
from gym_microgrid.common.itertools_ import fill_params
from gym_microgrid.controllers import Controller

from gym_microgrid.env import EmptyHistory


import pandas as pd
import numpy as np


class StaticControlAgent(Agent):
    def __init__(self, ctrls: Dict[str, Controller], observation_action_mapping: Dict, history=EmptyHistory()):
        super().__init__(history)
        self.episode_reward = 0
        self.controllers = ctrls
        self.obs_template = observation_action_mapping

    def reset(self):
        super().reset()
        self.prepare_episode()

    def act(self, state: pd.DataFrame):
        """

        :param state: the agent is stateless. the state is stored in the controllers.
        Therefore we simply pass the observation from the environment into the controllers.
        """
        obs = fill_params(self.obs_template, state)
        controls = list()
        for key, params in obs.items():
            controls.append(self.controllers[key].step(*params))

        # TODO: Remove this constant 1000. it should actually be in the fmu!
        return np.append(*controls) * 1000

    def observe(self, reward, terminated):
        self.episode_reward += reward or 0
        if terminated:
            # reset episode reward
            self.prepare_episode()
        # on other steps we don't need to do anything

    def prepare_episode(self):
        for ctrl in self.controllers.values():
            ctrl.reset()
        self.episode_reward = 0
