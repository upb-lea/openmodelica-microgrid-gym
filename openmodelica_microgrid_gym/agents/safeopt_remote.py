import importlib
import logging
from typing import Dict, Union, Any, List, Mapping

import GPy
import matplotlib.pyplot as plt
import numpy as np
from GPy.kern import Kern
from matplotlib.figure import Figure
from safeopt import SafeOptSwarm

from openmodelica_microgrid_gym import Agent
from openmodelica_microgrid_gym.agents.util import MutableParams
from openmodelica_microgrid_gym.aux_ctl import Controller


logger = logging.getLogger(__name__)


class remote_SafeOptAgent(Agent):
    def __init__(self, mutable_params: Union[dict, list], abort_reward: int, kernel: Kern, gp_params: Dict[str, Any],
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
        self.kernel = kernel
        self.bounds = gp_params['bounds']
        self.noise_var = gp_params['noise_var']
        self.prior_mean = gp_params['prior_mean']
        self.safe_threshold = gp_params['safe_threshold']
        self.explore_threshold = gp_params['explore_threshold']

        self.abort_reward = abort_reward
        self.episode_reward = None
        self.optimizer = None
        self.inital_performance = None
        self.last_best_performance = None
        self.performance = None

        self._iterations = 0

        super().__init__(ctrls, obs_template, obs_varnames, **kwargs)
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
        self.inital_performance = None
        self.last_best_performance = None
        self.performance = None

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
            self.performance = self.episode_reward / self._iterations
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

            # Norm for Safe-point
            # J = 1 / self.episode_reward / self.inital_Performance

            self.last_best_performance = self.performance

            # Define Mean "Offset": Like BK: Assume Mean = Threshold (BK = 0, now = 20% below first (safe) J: means: if
            # new Performance is 20 % lower than the inital we assume as unsafe)
            mf = GPy.core.Mapping(len(self.bounds), 1)
            mf.f = lambda x: self.prior_mean * self.performance
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0

            gp = GPy.models.GPRegression(np.array([self.params[:]]),  # noqa
                                         np.array([[self.performance]]), self.kernel,
                                         noise_var=self.noise_var, mean_function=mf)
            self.optimizer = SafeOptSwarm(gp, self.safe_threshold * self.performance, bounds=self.bounds,
                                          threshold=self.explore_threshold * self.performance)

        else:
            if np.isnan(self.episode_reward):
                # set r to doubled (negative!) initial reward
                self.episode_reward = self.abort_reward * self.inital_performance
                # toDo: set reward to -inf and stop agent?
                # warning mit logger
                logger.warning('UNSAFE! Limit exceeded, epsiode abort, give a reward of {} times the'
                               'initial reward'.format(self.abort_reward))

            self.optimizer.add_new_data_point(self.params[:], self.performance)

        self.history.append([self.performance, self.params[:]])
        self.params[:] = self.optimizer.optimize()

        if self.has_improved:
            # if performance has improved store the current last index of the df
            self.best_episode = self.history.df.shape[0] - 1

            self.last_best_performance = self.performance

    def render(self) -> Figure:
        """
        Renders the results for the performance
        """
        figure, ax = plt.subplots()
        self.optimizer.plot(1000, figure=figure)

        # mark best performance in green
        y, x = self.history.df.loc[self.best_episode, ['J', 'Params']]

        if len(x) == 1:
            ax.scatter([x], [y], s=20 * 10, marker='x', linewidths=3, color='g')
        elif len(x) == 2:
            ax.plot(x[0], x[1], 'og')
        else:
            logger.warning('Choose appropriate numer of control parameters')

        plt.show()
        return figure

    def prepare_episode(self):
        """
        Prepares the next episode; reset iteration counting variable and call superclass to reset controllers
        """
        self._iterations = 0

        super().prepare_episode()

    @property
    def has_improved(self) -> bool:
        """
        Defines if the performance increased or stays constant

        :return: True, if performance was increased or equal, else False
        """
        return self.performance >= self.last_best_performance
