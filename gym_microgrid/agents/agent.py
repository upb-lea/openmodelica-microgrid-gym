"""
agent and Environment are
"""

import numpy as np
import pandas as pd

from gym_microgrid.env import EmptyHistory


class Agent:
    def __init__(self, history=EmptyHistory()):
        self.history = history

    def reset(self):
        """
        Resets all agent buffers and discards unfinished episodes.
        TODO: Also set up the agent with respect to the environment (#actions, ...)
        """
        self.history.reset()
        self.prepare_episode()

    def act(self, obs: pd.DataFrame) -> np.ndarray:
        """
        select an action with respect to the state
        this might update the internal state with respect to the history.
        :param obs:
        :return:
        """
        pass

    def observe(self, reward: float, terminated: bool):
        """
        :param reward:
        :param terminated:
        :return:
        """
        pass

    def measure(self) -> pd.DataFrame:
        """

        :return: DataFrame or nested list of DataFrames
        """
        return pd.DataFrame()

    def render(self):
        pass

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (inital value of integrators...)
        """
        pass
