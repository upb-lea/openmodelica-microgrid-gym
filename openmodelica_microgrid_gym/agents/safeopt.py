import importlib
from typing import Dict, Union, Any

import GPy
from GPy.kern import Kern
from safeopt import SafeOptSwarm

import logging

from openmodelica_microgrid_gym.agents.staticctrl import StaticControlAgent
from openmodelica_microgrid_gym.agents.util import MutableParams
from openmodelica_microgrid_gym.auxiliaries import Controller
from openmodelica_microgrid_gym.env import EmptyHistory

import matplotlib.pyplot as plt

import numpy as np

logger = logging.getLogger(__name__)

class SafeOptAgent(StaticControlAgent):
    def __init__(self, mutable_params: Union[dict, list], abort_reward: int, kernel: Kern, gp_params: Dict[str, Any],
                 ctrls: Dict[str, Controller],
                 observation_action_mapping: dict, history=EmptyHistory()):
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
        :param observation_action_mapping: form controller keys to observation keys, whose observation values will be
         passed to the controller
        :param history:Storage of internal data
        """
        self.params = MutableParams(
            list(mutable_params.values()) if isinstance(mutable_params, dict) else mutable_params)
        self.kernel = kernel
        self.bounds = gp_params['bounds']
        self.noise_var = gp_params['noise_var']
        self.prior_mean = gp_params['prior_mean']
        self.safe_threshold = gp_params['safe_threshold']
        self.explore_threshold = gp_params['explore_threshold']

        self.abort_reward = abort_reward
        self.episode_reward = None
        self.optimizer = None
        self.inital_Performance = None
        self._iterations = 0
        super().__init__(ctrls, observation_action_mapping, history)
        self.history.cols = ['J', 'Params']

    def reset(self):
        """
        Resets the kernel, episodic reward and the optimizer
        """
        # reinstantiate kernel

        kernel_params = self.kernel.to_dict()
        cls_name = kernel_params['class']
        mod = importlib.import_module('.'.join(cls_name.split('.')[:-1]))
        cls = getattr(mod, cls_name.split('.')[-1])
        remaining_params = {k: v for k, v in kernel_params.items() if k not in {'class', 'useGPU'}}
        self.kernel = cls(**remaining_params)

        self.params.reset()
        self.optimizer = None
        self.episode_reward = 0
        self.inital_Performance = None
        self._iterations = 0
        return super().reset()

    def observe(self, reward, terminated):
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
            self.episode_reward = self.episode_reward / self._iterations
            # safeopt update step
            self.update_params()
            # reset for new episode
            self.prepare_episode()
        # on other steps we don't need to do anything

    def update_params(self):
        """
        Sets up the Gaussian process in the first episodes, updates the parameters in the following.
        """
        if self.optimizer is None:
            # First Iteration
            # self.inital_Performance = 1 / self.episode_reward
            self.inital_Performance = self.episode_reward
            # Norm for Safe-point
            # J = 1 / self.episode_reward / self.inital_Performance

            J = self.inital_Performance


            # Define Mean "Offset": Like BK: Assume Mean = Threshold (BK = 0, now = 20% below first (safe) J: means: if
            # new Performance is 20 % lower than the inital we assume as unsafe)
            mf = GPy.core.Mapping(len(self.bounds), 1)
            mf.f = lambda x: self.prior_mean * J
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0

            gp = GPy.models.GPRegression(np.array([self.params[:]]),
                                         np.array([[J]]), self.kernel,
                                         noise_var=self.noise_var, mean_function=mf)
            self.optimizer = SafeOptSwarm(gp, self.safe_threshold * J, bounds=self.bounds,
                                          threshold=self.explore_threshold * J)

        else:

            if np.isnan(self.episode_reward):
                # set r to doubled (negative!) initial reward
                self.episode_reward = self.abort_reward * self.inital_Performance
                # toDo: set reward to -inf and stop agent?
                # warning mit logger
                logger.warning('UNSAFE! Limit exceeded, epsiode abort, give a reward of {} times the'
                               'initial reward'.format(self.abort_reward))

            J = self.episode_reward

            self.optimizer.add_new_data_point(self.params[:], J)

        self.history.append([J, self.params[:]])
        self.params[:] = self.optimizer.optimize()

    def render(self):
        """
        Renders the results for the performance
        """
        plt.figure()
        self.optimizer.plot(1000)
        plt.show()

    def prepare_episode(self):
        """
        Prepares the next episode; reset iteration counting variable and call superclass to reset controllers
        """
        self._iterations = 0
        super().prepare_episode()