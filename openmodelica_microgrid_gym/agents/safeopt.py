import importlib
import logging
from typing import Dict, Union, Any, List, Mapping

import GPy
import matplotlib.pyplot as plt
import numpy as np
from GPy.kern import Kern
from matplotlib.figure import Figure
from safeopt import SafeOptSwarm

from openmodelica_microgrid_gym.agents.episodic import EpisodicLearnerAgent
from openmodelica_microgrid_gym.agents.staticctrl import StaticControlAgent
from openmodelica_microgrid_gym.agents.util import MutableParams
from openmodelica_microgrid_gym.aux_ctl import Controller, DDS

logger = logging.getLogger(__name__)


class SafeOptAgent(StaticControlAgent, EpisodicLearnerAgent):
    def __init__(self, mutable_params: Union[dict, list], abort_reward: int, kernel: Kern, gp_params: Dict[str, Any],
                 ctrls: List[Controller],
                 obs_template: Mapping[str, List[Union[List[str], np.ndarray]]], obs_varnames: List[str] = None,
                 min_performance: float = None,
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
         :param min_performance: Minial allowed performance to define parameters as safe
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
        self.episode_return = None
        self.optimizer = None
        self._performance = None
        self._min_performance = min_performance
        self.initial_performance = min_performance/2
        self.last_best_performance = 0      # set to 0 due to in MC otherwise we do not get the best/worst before the
        # first update_params after the MC loop
        self.last_worst_performance = 0
        self.unsafe = False

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
        self.episode_return = 0
        self.initial_performance = 1
        self._performance = None
        self.last_best_performance = 0
        self.last_worst_performance = 0
        self.unsafe = False

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
        self.episode_return += reward or 0
        if terminated:

            if self.optimizer is None:
                self.initial_performance = self.episode_return

            #self.performance = self._iterations / (self.episode_return * self.initial_performance)
            self.performance = (self.episode_return - self._min_performance) / (self.initial_performance - self.min_performance)
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
            self.last_best_performance = self.performance
            self.last_worst_performance = self.performance

            # Define Mean "Offset": Like BK: Assume Mean = Threshold (BK = 0, now = 20% below first (safe) J: means: if
            # new Performance is 20 % lower than the inital we assume as unsafe)
            mf = GPy.core.Mapping(len(self.bounds), 1)
            mf.f = lambda x: self.prior_mean * self.performance
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0

            gp = GPy.models.GPRegression(np.array([self.params[:]]),  # noqa
                                         np.array([[self.performance]]), self.kernel,
                                         noise_var=self.noise_var)#, mean_function=mf)
            self.optimizer = SafeOptSwarm(gp, self.safe_threshold * self.performance, bounds=self.bounds,
                                          threshold=self.explore_threshold * self.performance)

        else:
            self.optimizer.add_new_data_point(self.params[:], self.performance)

        if self.performance < self.safe_threshold:  # Due to nromalization tp 1 safe_threshold directly enough
            self.unsafe = True
        self.history.append([self.performance, self.params[:]])
        self.params[:] = self.optimizer.optimize()

        if self.has_improved:
            # if performance has improved store the current last index of the df
            self.best_episode = self.history.df.shape[0] - 1

            self.last_best_performance = self.performance

        if self.has_worsened:
            # if performance has improved store the current last index of the df
            self.worst_episode = self.history.df.shape[0] - 1

            self.last_worst_performance = self.performance

    def render(self) -> Figure:
        """
        Renders the results for the performance
        """
        figure, ax = plt.subplots()
        if self.optimizer.x.shape[1] > 3:
            # check if the dimensionality is less then 4 dimension
            logger.info('Plotting of GP landscape not possible for then 3 dimensions')
            return figure
        self.optimizer.plot(1000, figure=figure, ms = 1)#, color = 'k')

        # mark best performance in green
        y, x = self.history.df.loc[self.best_episode, ['J', 'Params']]

        y0, x0 = self.history.df.loc[0, ['J', 'Params']]
        #ax.scatter([x0], [y0], s=20, marker='x', linewidths=3, color='m')

        if len(x) == 1:
            ax.scatter([x], [y], s=20, marker='x', linewidths=3, color='g')
            ax.scatter([x0], [y0], s=20, marker='x', linewidths=3, color='m')
            # ax.scatter([x], [y], s=20 * 10, marker='x', linewidths=3, color='g')


        elif len(x) == 2:
            ax.plot(x[0], x[1], 'og')
            ax.plot(x0[0], x0[1], 'om')
        else:
            logger.warning('Choose appropriate number of control parameters')

        plt.show()             # only comment for lengthscale sweep
        #plt.close(figure)       # only needed for lengthscale sweep
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

    @property
    def has_worsened(self) -> bool:
        """
        Defines if the performance increased or stays constant

        :return: True, if performance was increased or equal, else False
        """
        return self.performance <= self.last_worst_performance

    @property
    def performance(self):

        if np.isnan(self.episode_return):
            # toDo: set reward to -inf and stop agent?
            # warning mit logger

            logger.warning('UNSAFE! Limit exceeded, epsiode abort, give a reward of {} times the '
                           'initial reward'.format(self.abort_reward))
            # set r to doubled (negative!) initial reward
            self._performance = self.abort_reward

            """print('Reward exceeded bad bound! Continue?(Type Y/N)')
            variable = input('Does this spike make sence?! (Type Y/N): ')
            if variable == 'N':
                self.episode_return = -300
            elif variable == 'Y':
                logger.warning('UNSAFE! Limit exceeded, epsiode abort, give a reward of {} times the '
                               'initial reward'.format(self.abort_reward))
                # set r to doubled (negative!) initial reward
                self._performance = self.abort_reward
            else:
                print('Choose appropiate answer! Programm will run with J =1')
                self.episode_return = -300
            """

        if self._performance is None:
            # Performance = inverse average return (return/iterations)^⁻1 normalized by initial performance
            #self._performance = self._iterations / (self.episode_return * self.initial_performance)
            self._performance = (self.episode_return - self._min_performance) / (self.initial_performance - self.min_performance)

        return self._performance


    @performance.setter
    def performance(self, new_performance):
        self._performance = new_performance
        #self.episode_return = self._iterations / (new_performance*self.initial_performance)

    @property
    def min_performance(self):
        if self._min_performance is None:
            self._min_performance = 0.95 * self.initial_performance
        return self._min_performance
