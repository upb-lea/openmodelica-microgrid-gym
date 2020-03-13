import logging
import math
import numpy as np
from gym import spaces
from .me_env import FMI2MEEnv

logger = logging.getLogger(__name__)

NINETY_DEGREES_IN_RAD = (90 / 180) * math.pi
TWELVE_DEGREES_IN_RAD = (12 / 180) * math.pi


class ConvEnv:

    def _is_done(self):
        """
        #TODO: Add an proper is_done policy
        Internal logic that is utilized by parent classes.
        Checks if the experiment is finished using a time limit

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        True, if the simulation time is larger than the time threshold.
        """
        time = self.stop
        logger.debug("t: {0}, ".format(self.stop))

        if abs(time) > self.time_threshold:
            done = True
        else:
            done = False

        return done

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size 3
        """
        return spaces.Discrete(3)

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements

        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([self.time_threshold, np.inf])
        return spaces.Box(-high, high)

    # OpenAI Gym API implementation
    def step(self, action):
        """
        OpenAI Gym API. Executes one step in the environment:
        in the current state perform given action to move to the next action.
        
        :param action: action to be performed.
        :return: next (resulting) state
        """

        return super().step(action)

    def render(self, mode='human', close=False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Does nothing at the moment

        :param mode: rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        :return: rendering result
        """
        return True

    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return self.render(close=True)


class JModelicaConvEnv(ConvEnv, FMI2MEEnv):
    """
    Wrapper class 
    
    Defining the inputs and outputs to the FMU as well as the FMU file.
    Attributes:
        time_step (float): the delta T to be used during the experiment for simulation
        positive_reward (int): positive reward for RL agent.
        negative_reward (int): negative reward for RL agent.
    """

    def __init__(self,
                 time_step,
                 positive_reward,
                 negative_reward,
                 log_level,
                 solver_method):
        logger.setLevel(log_level)
        # TODO time.threshhold needed? I would delete it completely, no one knows about it, just leads to confusion if exceeded.
        # Right now still there until we defined an other stop-criteria according to safeness
        self.time_threshold = 10000.0

        self.viewer = None
        self.display = None

        # Define the interface between the software to the FMU
        # Defines the order of inputs and outputs.
        config = {
            'model_input_names': ['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
            'model_output_names': ['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i',
                                   'lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v',
                                   'lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i',
                                   'lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v', ],
            'model_parameters': {},
            'initial_state': (),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward,
            'solver_method': solver_method
        }
        super().__init__("grid.network.fmu", config, log_level, simulation_start_time=time_step)
