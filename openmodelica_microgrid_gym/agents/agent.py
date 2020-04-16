import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.env import EmptyHistory


class Agent:
    def __init__(self, history: EmptyHistory = EmptyHistory()):
        """
        Abstract base class for all Agents. The agent can act on the environment and observe its result.
        This class is aims to wrap the whole learning process into a class to simplify the implementation.

        This class is strongly inspired by the tensorforce library.

        :param history: used to store agent internal data for monitoring
        """
        self.history = history

    def reset(self):
        """
        Resets all agent buffers and discards unfinished episodes.
        """
        self.history.reset()
        self.prepare_episode()

    def act(self, obs: pd.Series) -> np.ndarray:
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
    def measurement(self) -> pd.Series:
        """
        Measurements the agent takes on the environment. This data is passed to the environment.
        The values returned by this property should be fully determined by the environment.
        This is a workaround to provide data measurements like PLL controllers in the environment even though
        they are functionally part of the Agent.

        :return: pd.Series or list of Tuples of columns and pd.Series. The columns can be provided as nested list
        """
        return pd.Series()

    def render(self):
        """
        Visualisation of the agent, e.g. its learning state or similar
        """
        pass

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (initial value of integrators...)
        """
        pass
