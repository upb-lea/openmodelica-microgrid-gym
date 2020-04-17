from typing import Union, List, Mapping

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.common.itertools_ import fill_params, nested_map
from openmodelica_microgrid_gym.auxiliaries import Controller

from openmodelica_microgrid_gym.env import EmptyHistory

import pandas as pd
import numpy as np


class StaticControlAgent(Agent):
    def __init__(self, ctrls: Mapping[str, Controller], ctrl_to_obs: Mapping,
                 history: EmptyHistory = EmptyHistory()):
        """
        Simple agent that controls the environment by using auxiliary controllers that are fully configured.
        The Agent does not learn anything.

        :param ctrls: Controllers that are feed with the observations and exert actions on the environment
        :param ctrl_to_obs: form controller keys to observation keys, whose observation values will be passed to the controller
        :param history: Storage of internal data
        """
        super().__init__(history)
        self.episode_reward = 0
        self.controllers = ctrls
        self.obs_template = ctrl_to_obs

    def act(self, state: pd.Series):
        """
        Executes the actions with the observations as parameters.

        :param state: the agent is stateless. the state is stored in the controllers.
        Therefore we simply pass the observation from the environment into the controllers.
        """
        obs = fill_params(self.obs_template, state)
        controls = list()
        for key, params in obs.items():
            controls.append(self.controllers[key].step(*params))

        if len(controls) == 1:
            return controls[0]
        return np.append(*controls)

    def observe(self, reward: float, terminated: bool):
        """
        The observe function is might be called after the act function.
        It might trigger the learning in some implementations.

        :param reward: reward from the environment after the last action
        :param terminated: whether the episode is finished
        """
        self.episode_reward += reward or 0
        if terminated:
            # reset episode reward
            self.prepare_episode()
        # on other steps we don't need to do anything

    @property
    def measurement(self) -> Union[pd.Series, List]:
        measurements = []
        for name, ctrl in self.controllers.items():
            def prepend(col): return '.'.join([name, col])

            measurements.append((nested_map(prepend, ctrl.history.structured_cols(None)),
                                 ctrl.history.df.tail(1).rename(columns=prepend).squeeze()))

        return measurements

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (initial value of integrators...)
        """
        for ctrl in self.controllers.values():
            ctrl.reset()
        self.episode_reward = 0
