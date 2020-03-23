from datetime import datetime
import logging
from os.path import basename
from typing import Sequence

import gym
import numpy as np
import pandas as pd
import scipy
from pyfmi import load_fmu
from pyfmi.fmi import FMUModelME2
from scipy import integrate
import matplotlib.pyplot as plt

from gym_microgrid.common.flattendict import flatten
from gym_microgrid.env.recorder import FullHistory, EmptyHistory

logger = logging.getLogger(__name__)


class ModelicaEnv(gym.Env):
    """
    Superclass for all environments simulated with Modelica FMU exported.
    Implements abstract logic, common to all such environments.

    In addition, allows usage of simple reward function out of box:
        positive reward is assigned if experiment goes on,
        negative reward - when experiment is done.

    To create your own environment inherit this class and implement:
        _get_action_space(), _get_observation_space(), _is_done()
    and provide required config, that describes your model.

    Note:
        Here is an example of config for the inverted pendulum problem:
        config = {
            'model_input_names': 'u',
            'model_output_names': ['s', 'v', 'phi1', 'w'],
            'model_parameters': {'m_trolley': 1, 'm_load': 0.1,
                                 'phi1_start': 85/180*math.pi, 'w1_start': 0},
            'time_step': 0.01,
            'positive_reward': 1, # (optional)
            'negative_reward': -100 # (optional)
        }

    If you need more sophisticated logic to be implemented - override methods of OpenAI Gym API
    in the child-class: reset(), step(), render(), close(), seed(), _reward_policy().
    You can reuse get_state() and do_simulation() for faster development.

    All methods called on model are from implemented PyFMI API.
    """

    viz_modes = {'episode', 'step', None}

    def __init__(self, time_step: float = 1e-4, positive_reward: float = 1, negative_reward: float = -100,
                 log_level: int = logging.WARNING, solver_method='LSODA', max_episode_steps: int = 10000.0,
                 model_params: dict = None,
                 model_input: Sequence[str] = None, model_output: Sequence[str] = None, model_path='grid.network.fmu',
                 time_start=0,
                 viz_mode: str = 'episode', history: EmptyHistory = FullHistory()):
        """
        :type record_states: EmptyHistory
        :type viz_mode: str
        """
        logger.setLevel(log_level)
        # Right now still there until we defined an other stop-criteria according to safeness
        if model_input is None:
            raise ValueError('Please specify model_input variables from your OM FMU.')
        if model_output is None:
            raise ValueError('Please specify model_output variables from your OM FMU.')
        if viz_mode not in self.viz_modes:
            raise ValueError(f'Please select one of the following viz_modes: {self.viz_modes}')

        self.viz_mode = viz_mode
        logger.setLevel(log_level)
        self.solver_method = solver_method

        # load model from fmu
        model_name = basename(model_path)
        logger.debug("Loading model {}".format(model_name))
        self.model: FMUModelME2 = load_fmu(model_path,
                                           log_file_name=datetime.now().strftime(f'%Y-%m-%d_{model_name}.txt'))
        logger.debug("Successfully loaded model {}".format(model_name))

        # if you reward policy is different from just reward/penalty - implement custom step method
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        # Parameters required by this implementation
        self.time_start = time_start
        self.time_step_size = time_step
        self.time_end = np.inf if max_episode_steps is None else self.time_start + max_episode_steps * self.time_step_size

        self.model_parameters = model_params
        self.model_input_names = model_input
        self.model_output_names = model_output
        self.sim_time_interval = None
        self.state = None
        self.record_states = viz_mode == 'episode'
        self.history = history
        self.history.cols = flatten(self.model_output_names)

        # OpenAI Gym requirements
        self.action_space = gym.spaces.Discrete(3)
        high = np.array([self.time_end, np.inf])
        self.observation_space = gym.spaces.Box(-high, high)

    def _set_init_parameter(self):
        """
        Sets initial parameters of a model.

        :return: environment
        """

        self.model.setup_experiment(start_time=self.time_start)
        self.model.enter_initialization_mode()
        self.model.exit_initialization_mode()

        e_info = self.model.get_event_info()
        e_info.newDiscreteStatesNeeded = True
        # Event iteration
        while e_info.newDiscreteStatesNeeded:
            self.model.enter_event_mode()
            self.model.event_update()
            e_info = self.model.get_event_info()

        self.model.enter_continuous_time_mode()
        # list of keys and list of values

        if self.model_parameters is not None:
            self.model.set(*zip(*self.model_parameters.items()))

        # precalculating indices for more efficient lookup
        statename_index_map = {v: i for i, v in enumerate(list(self.model.get_states_list()))}
        self.model_output_index = np.array([statename_index_map[k] for k in flatten(self.model_output_names)])

        return self

    def _calc_jac(self, t, x):
        # t and x are indirectly retrieved from the model
        states = self.model.get_states_list()
        states_references = [s.value_reference for s in states.values()]
        derivatives = self.model.get_derivatives_list()
        derivatives_references = [d.value_reference for d in derivatives.values()]
        self.jacobian = np.identity(len(states))
        np.apply_along_axis(
            lambda col: self.model.get_directional_derivative(states_references, derivatives_references, col), 0,
            self.jacobian)
        return self.jacobian

    def _get_deriv(self, t, x):
        self.model.time = t
        self.model.continuous_states = x.copy(order='C')

        # Compute the derivative
        dx = self.model.get_derivatives()
        return dx

    def _simulate(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        logger.debug(f'Simulation started for time interval {self.sim_time_interval[0]}-{self.sim_time_interval[1]}')

        # Advance
        x_0 = self.model.continuous_states

        # Get the output from a step of the solver

        sol_out = scipy.integrate.solve_ivp(
            self._get_deriv, self.sim_time_interval, x_0, method=self.solver_method, jac=self._calc_jac)
        # get the last solution of the solver
        self.model.continuous_states = sol_out.y[:, -1]

        obs = np.array(self.model.continuous_states)[self.model_output_index]
        self.history.append(obs)
        return obs

    @property
    def reward(self):
        """
        Determines reward based on the current environment state.
        By default, implements simple logic of penalizing for experiment end and rewarding each step.

        :return: reward associated with the current state
        """
        return self.negative_reward or -100 if self.is_done else self.positive_reward or 1

    @property
    def is_done(self) -> bool:
        """
        Checks if the experiment is finished using a time limit

        :return: True if simulation time exceeded
        """
        logger.debug(f't: {self.sim_time_interval[1]}, ')
        return abs(self.sim_time_interval[1]) > self.time_end

    def reset(self):
        """
        OpenAI Gym API. Restarts environment and sets it ready for experiments.
        In particular, does the following:
            * resets model
            * sets simulation start time to 0
            * sets initial parameters of the model
            * initializes the model
            * sets environment class attributes, e.g. start and stop time.
        :return: state of the environment after resetting
        """
        logger.debug("Experiment reset was called. Resetting the model.")

        self.model.reset()
        self.model.setup_experiment(start_time=0)

        self._set_init_parameter()
        self.sim_time_interval = np.array([self.time_start, self.time_start + self.time_step_size])
        self.history.reset()
        self.state = self._simulate()

        return self.state

    def step(self, action):
        """
        OpenAI Gym API. Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.
        :param action: action to be executed.
        :return: resulting state
        """
        logger.debug("Experiment next step was called.")
        if self.is_done:
            logging.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return self.state, self.negative_reward, self.is_done

        # check if action is a list. If not - create list of length 1
        try:
            iter(action)
        except TypeError:
            action = [action]
            logging.warning("Model input values (action) should be passed as a list")

        # Check if number of model inputs equals number of values passed
        if len(action) != len(list(self.model_input_names)):
            message = "List of values for model inputs should be of the length {}," \
                      "equal to the number of model inputs. Actual length {}".format(
                len(list(self.model_input_names)), len(action))
            logging.error(message)
            raise ValueError(message)

        # Set input values of the model
        logger.debug("model input: {}, values: {}".format(self.model_input_names, action))
        self.model.set(list(self.model_input_names), list(action))

        # Simulate and observe result state
        self.state = self._simulate()

        logger.debug("model output: {}, values: {}".format(flatten(self.model_output_names), self.state))
        # print("model output: {}, values: {}".format(self.model_output_names, self.state))
        # Check if experiment has finished

        # Move simulation time interval if experiment continues
        if not self.is_done:
            logger.debug("Experiment step done, experiment continues.")
            self.sim_time_interval += self.time_step_size
        else:
            logger.debug("Experiment step done, experiment done.")

        return self.state, self.reward, self.is_done

    def render(self, mode='human', close=False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Does nothing at the moment

        :param mode: rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        :return: rendering result
        """
        if self.viz_mode is None:
            return True
        elif close:
            if self.viz_mode == 'step':
                # TODO close plot
                pass
            else:
                # TODO create the plot
                for cols in flatten(self.model_output_names, 1):
                    self.history.df[cols].plot()
                    plt.show()
                # print(self.history)
                pass
        elif self.viz_mode == 'step':
            # TODO update plot
            pass
        return True

    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True on success
        """
        return self.render(close=True)
