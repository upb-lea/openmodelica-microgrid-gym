import logging
import math

import numpy as np
from gym import spaces
from gym_microgrid.env.me_env import ModelicaMEEnv

logger = logging.getLogger(__name__)

NINETY_DEGREES_IN_RAD = (90 / 180) * math.pi
TWELVE_DEGREES_IN_RAD = (12 / 180) * math.pi


class JModelicaConvEnv(ModelicaMEEnv):
    """
    Wrapper class 
    
    Defining the inputs and outputs to the FMU as well as the FMU file.
    Attributes:
        time_step (float): the delta T to be used during the experiment for simulation
        positive_reward (int): positive reward for RL agent.
        negative_reward (int): negative reward for RL agent.
    """
    viz_modes = {'episode', 'step'}

    def __init__(self, time_step=1e-4, positive_reward=1, negative_reward=-100,
                 log_level=2, solver_method='LSODA', max_episode_steps=10000.0, model_input=None, model_output=None,
                 viz_mode='episode'):
        logger.setLevel(log_level)
        # TODO time.threshhold needed? I would delete it completely, no one knows about it, just leads to confusion if exceeded.
        # Right now still there until we defined an other stop-criteria according to safeness
        self.time_threshold = max_episode_steps * time_step
        if model_input is None:
            raise ValueError('Please specify model_input variables')
        if model_output is None:
            raise ValueError('Please specify model_output variables')
        if viz_mode not in self.viz_modes:
            raise ValueError(f'Please select one of the following viz_modes: {self.viz_modes}')

        self.viz_mode = viz_mode

        # Define the interface between the software to the FMU
        # Defines the order of inputs and outputs.
        config = {
            'model_input_names': model_input,
            'model_output_names': model_output,
            'model_parameters': {},
            'initial_state': (),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward,
            'solver_method': solver_method
        }
        super().__init__("grid.network.fmu", config, log_level, simulation_start_time=time_step)

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
        if close:
            if self.viz_mode == 'episode':
                # TODO create the plot
                pass
            else:
                # TODO close plot
                pass
        elif self.viz_mode == 'step':
            # TODO update plot
            pass
        return True

    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return self.render(close=True)
