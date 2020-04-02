import numpy as np

from gym_microgrid.env import EmptyHistory


class Agent:
    def __init__(self, history=EmptyHistory()):
        self.history = history

    def reset(self, ):
        """
        Resets all agent buffers and discards unfinished episodes.
        TODO: Also set up the agent with respect to the environment (#actions, ...)
        """
        self.history.reset()

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        select an action with respect to the state
        :param state:
        :return:
        """
        pass

    def observe(self, reward, terminated):
        pass

    def render(self):
        pass
