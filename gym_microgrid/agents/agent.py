import numpy as np


class Agent:
    def __init__(self):
        pass

    def reset(self, ):
        """
        Resets all agent buffers and discards unfinished episodes.
        TODO: Also set up the agent with respect to the environment (#actions, ...)
        """
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        select an action with respect to the state
        :param state:
        :return:
        """
        pass

    def observe(self, reward, terminated):
        pass
