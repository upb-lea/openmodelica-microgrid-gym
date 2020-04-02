import numpy as np
import pandas as pd


class Agent:
    def __init__(self):
        pass

    def reset(self):
        """
        Resets all agent buffers and discards unfinished episodes.
        TODO: Also set up the agent with respect to the environment (#actions, ...)
        """
        pass

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

        :return:
        """
        return pd.DataFrame()
