import importlib
import logging
from typing import Dict, Union, Any, List, Mapping

import GPy
import matplotlib.pyplot as plt
import numpy as np
from GPy.kern import Kern
from matplotlib.figure import Figure
from safeopt import SafeOptSwarm

from openmodelica_microgrid_gym.agents.staticctrl import StaticControlAgent
from openmodelica_microgrid_gym.agents.util import MutableParams
from openmodelica_microgrid_gym.aux_ctl import Controller

logger = logging.getLogger(__name__)


class simRewardMapAgent(StaticControlAgent):
    def __init__(self, kMatrix, mutable_params: Union[dict, list],
                 ctrls: List[Controller],
                 obs_template: Mapping[str, List[Union[List[str], np.ndarray]]], obs_varnames: List[str] = None,
                 **kwargs):
        """
        Agent to execute safeopt algorithm (https://arxiv.org/abs/1509.01066) to control the environment by using
        auxiliary controllers and Gaussian process to adopt the controller parameters (mutable_params) to safely
        increase the performance.

        :param mutable_params: safe inital controller parameters to adopt
        :param abort_reward: factor to multiply with the initial reward to give back an abort_reward-times higher
               negative reward in case of limit exceeded
        :param kernel: kernel for the Gaussian process unsing GPy
        :param gp_params: kernel parameters like bounds and lengthscale
        :param ctrls: Controllers that are feed with the observations and exert actions on the environment
        :param obs_template:
            Template describing how the observation array should be transformed and passed to the internal controllers.
            The key must match the name field of an internal controller.
            The values are a list of:
                - list of strings
                    - matching variable names of the state
                    - must match self.obs_varnames
                    - will be substituted by the values on runtime
                    - will be passed as an np.array of floats to the controller
                - np.ndarray of floats (to be passed statically to the controller)
                - a mixture of static and dynamic values in one parameter is not supported for performance reasons.
            The values will be passed as parameters to the controllers step function.
        :param obs_varnames: list of variable names that match the values of the observations
         passed in the act function. Will be automatically set by the Runner class
        """
        self.params = MutableParams(
            list(mutable_params.values()) if isinstance(mutable_params, dict) else mutable_params)

        self.episode_reward = None


        self.kMatrix = kMatrix #np.array([np.linspace(0, 8, 20), np.linspace(0, 100, 20)])

        self._iterations = 0

        super().__init__(ctrls, obs_template, obs_varnames, **kwargs)
        self.history.cols = ['J', 'Params']

    def reset(self):

        self.params.reset()
        self.episode_reward = 0

        self._iterations = 0

        return super().reset()

    def observe(self, reward, terminated, kp, ki):
        """
        Makes an observation of the enviroment.
        If terminated, then caclulates the performance and the next values for the parameters using safeopt

        :param reward: reward of the simulation step
        :param terminated: True if episode is over or aborted
        :return:
        """
        self._iterations += 1
        self.episode_reward += reward or 0
        if terminated:
            # calculate MSE, divide Summed error by length of measurement
            self.performance = 1/ (self.episode_reward / self._iterations)
            # safeopt update step
            self.update_params(kp, ki)
            # reset for new episode
            #self.prepare_episode()
        # on other steps we don't need to do anything

    def update_params(self, kp, ki):

        self.params[:] = [self.kMatrix[0][kp], self.kMatrix[1][ki]]


    def render(self) -> Figure:
        pass

    def prepare_episode(self):
        """
        Prepares the next episode; reset iteration counting variable and call superclass to reset controllers
        """
        self._iterations = 0

        super().prepare_episode()


