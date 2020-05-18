from typing import List, Mapping

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.common.itertools_ import fill_params
from openmodelica_microgrid_gym.auxiliaries import Controller

import pandas as pd
import numpy as np

from openmodelica_microgrid_gym.env import EmptyHistory, StructuredMapping, ModelicaEnv


class StaticControlAgent(Agent):
    def __init__(self, ctrls: List[Controller], ctrl_to_obs: Mapping,
                 history: EmptyHistory = EmptyHistory(), env: ModelicaEnv = None):
        """
        Simple agent that controls the environment by using auxiliary controllers that are fully configured.
        The Agent does not learn anything.

        :param ctrls: Controllers that are feed with the observations and exert actions on the environment
        :param ctrl_to_obs: form controller keys to observation keys, whose observation values will be passed to the controller
        :param history: Storage of internal data
        :param env: reference to the environment (only needed when used in internal act function)
        """
        super().__init__(history, env)
        self.episode_reward = 0
        self.controllers = {ctrl.name: ctrl for ctrl in ctrls}
        self.obs_template = ctrl_to_obs

        self.idx = None

    def act(self, state):
        """
        Executes the actions with the observations as parameters.

        :param state: the agent is stateless. the state is stored in the controllers.
        Therefore we simply pass the observation from the environment into the controllers.
        """
        obs = fill_params(self.obs_template, state.df.iloc[0].to_dict())
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
    def measurement(self) -> StructuredMapping:
        cols, vals = [], []
        for ctrl in self.controllers.values():
            cols.append(ctrl.history.structured_cols(None))
            vals.extend(ctrl.history.last())

        return StructuredMapping(cols, vals)

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (initial value of integrators...)
        """
        for ctrl in self.controllers.values():
            ctrl.reset()
        self.episode_reward = 0
