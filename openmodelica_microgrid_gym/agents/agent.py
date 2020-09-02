from typing import List, Union

import numpy as np
from matplotlib.figure import Figure

from openmodelica_microgrid_gym.env import ModelicaEnv
from openmodelica_microgrid_gym.util import EmptyHistory


class Agent:
    def __init__(self, obs_varnames: List[str] = None, history: EmptyHistory = None, env: ModelicaEnv = None):
        """
        Abstract base class for all Agents. The agent can act on the environment and observe its result.
        This class is aims to wrap the whole learning process into a class to simplify the implementation.

        This class is strongly inspired by the tensorforce library.

        :param obs_varnames: list of variable names that match the values of the observations
         passed in the act function. Will be automatically set by the Runner class
        :param history: used to store agent internal data for monitoring
        :param env: reference to the environment (only needed when used in internal act function)
        """
        self.env = env
        self.history = history or EmptyHistory()
        self.obs_varnames = obs_varnames

    def reset(self):
        """
        Resets all agent buffers and discards unfinished episodes.
        """
        self.history.reset()
        self.prepare_episode()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Select an action with respect to the state this might update the internal state with respect to the history.

        :param obs: Observation
        :return: Actions the agent takes with respect to the observation
        """
        pass

    def observe(self, reward: float, terminated: bool):
        """
        The observe function is might be called after the act function.
        It might trigger the learning in some implementations.

        :param reward: reward from the environment after the last action
        :param terminated: whether the episode is finished
        """
        pass

    @property
    def measurement_cols(self) -> List[Union[List, str]]:
        """
        Structured columns of the measurement. Used in the Runner to setup the history columns of the Environment.

        :return: structured columns of measurement
        """
        return []

    @property
    def measurement(self) -> np.ndarray:
        """
        Measurements the agent takes on the environment. This data is passed to the environment.
        The values returned by this property should be fully determined by the environment.
        This is a workaround to provide data measurement like PLL controllers in the environment even though
        they are functionally part of the Agent.

        :return: current measurement
        """
        return np.empty(0)

    def render(self) -> Figure:
        """
        Visualisation of the agent, e.g. its learning state or similar
        """
        pass

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (initial value of integrators...)
        """
        pass

    @property
    def has_improved(self) -> bool:
        """
        Defines if the performance increased or stays constant
        Does not learn, can never improve

        """
        return False

    @property
    def has_worsened(self) -> bool:
        """
        Defines if the performance decreased or stays constant
        Does not learn, can never improve

        """
        return False
