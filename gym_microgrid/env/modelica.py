# -*- coding: utf-8 -*-


import datetime
import logging
import os
import gym
import pandas as pd
import scipy
from scipy import integrate
from pyfmi import load_fmu
import numpy as np

from gym_microgrid.common.flattendict import flatten

import math
from itertools import groupby
from typing import Sequence
import re

import numpy as np
from gym import spaces

from gym_microgrid.common.flattendict import flatten

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


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

    viz_modes = {'episode', 'step'}

    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return self.render(close=True)

    def __init__(self, time_step: float = 1e-4, positive_reward: float = 1, negative_reward: float = -100,
                 log_level: int = logging.WARNING, solver_method='LSODA', max_episode_steps: int = 10000.0,
                 model_input: Sequence[str] = None, model_output: Sequence[str] = None, model_path='grid.network.fmu',
                 simulation_start_time=0,
                 viz_mode='episode'):
        logger.setLevel(log_level)
        # TODO time.threshold needed? I would delete it completely, no one knows about it, just leads to confusion if exceeded.
        # Right now still there until we defined an other stop-criteria according to safeness
        if model_input is None:
            raise ValueError('Please specify model_input variables from your OM FMU.')
        if model_output is None:
            raise ValueError('Please specify model_output variables from your OM FMU.')
        if viz_mode not in self.viz_modes:
            raise ValueError(f'Please select one of the following viz_modes: {self.viz_modes}')

        self.time_threshold = max_episode_steps * time_step
        self.viz_mode = viz_mode

        # Define the interface between the software to the FMU
        # Defines the order of inputs and outputs.
        config = dict(model_input_names=model_input, model_output_names=model_output, model_parameters={},
                      initial_state=(), time_step=time_step, positive_reward=positive_reward,
                      negative_reward=negative_reward, solver_method=solver_method)
        self.simulation_start_time = simulation_start_time

        logger.setLevel(log_level)

        self.solver_method = config.get('solver_method')
        # load model from fmu
        self.model_name = model_path.split(os.path.sep)[-1]

        logger.debug("Loading model {}".format(self.model_name))

        # self.model = load_fmu(model_path, kind=mode, log_file_name=self.get_log_file_name())
        self.model = load_fmu(model_path, log_file_name=self.get_log_file_name())

        logger.debug("Successfully loaded model {}".format(self.model_name))
        # if you reward policy is different from just reward/penalty - implement custom step method
        self.positive_reward = config.get('positive_reward')
        self.negative_reward = config.get('negative_reward')

        # Parameters required by this implementation
        self.tau = config.get('time_step')
        self.model_input_names = config.get('model_input_names')
        self.model_output_names = config.get('model_output_names')
        self.model_parameters = config.get('model_parameters')

        # initialize the model time and state
        self.start = 0
        self.count_iter = 0
        self.stop = self.tau
        self.done = False
        self.state = self.reset()

        # OpenAI Gym requirements
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.history = pd.DataFrame([], columns=flatten(self.model_output_names))

    def calc_jac(self, t, x):
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

    def deriv_func(self, t, x):
        self.model.time = t
        self.model.continuous_states = x.copy(order='C')

        # Compute the derivative
        dx = self.model.get_derivatives()
        return dx

        pass

    def reset(self):
        """
        OpenAI Gym API. Determines restart procedure of the environment
        :return: environment state after restart
        """
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

        self.start = 0
        self.stop = self.simulation_start_time

        self.state = self.simulate()

        self.start = self.simulation_start_time
        self.stop = self.start + self.tau
        self.done = self._is_done()
        return self.state

    def step(self, action):
        """
        OpenAI Gym API. Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.
        :param action: action to be executed.
        :return: resulting state
        """
        logger.debug("Experiment next step was called.")
        if self.done:
            logging.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return self.state, self.negative_reward, self.done

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
        self.state = self.simulate()
        self.history = self.history.append(pd.DataFrame([self.state], columns=flatten(self.model_output_names)),
                                           ignore_index=True)

        logger.debug("model output: {}, values: {}".format(flatten(self.model_output_names), self.state))
        # print("model output: {}, values: {}".format(self.model_output_names, self.state))
        # Check if experiment has finished
        self.done = self._is_done()

        # Move simulation time interval if experiment continues
        if not self.done:
            logger.debug("Experiment step done, experiment continues.")
            self.start = self.stop
            self.stop += self.tau
        else:
            logger.debug("Experiment step done, experiment done.")

        return self.state, self._reward_policy(), self.done

    # logging
    def get_log_file_name(self):
        log_date = datetime.datetime.utcnow()
        log_file_name = "{}-{}-{}_{}.txt".format(log_date.year, log_date.month, log_date.day, self.model_name)
        return log_file_name

    # part of the step() method extracted for convenience
    def simulate(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        logger.debug(f'Simulation started for time interval {self.start}-{self.stop}')

        # Advance
        x_0 = self.model.continuous_states

        t_span = [self.start, self.stop]

        # Get the output from a step of the solver
        sol_out = scipy.integrate.solve_ivp(self.deriv_func, t_span, x_0, method=self.solver_method, jac=self.calc_jac)
        # Unpack the solver output
        size, n_sols = sol_out.y.shape
        # get the last solution of the solver
        self.x = sol_out.y[:, n_sols - 1]

        # Count the number of solutions needed by the solver
        self.count_iter = n_sols

        # Not strictly necessary, the last call of the solver to get the derivative
        # should have set this.
        self.model.continuous_states = self.x

        return self.get_state()

    def _reward_policy(self):
        """
        Determines reward based on the current environment state.
        By default, implements simple logic of penalizing for experiment end and rewarding each step.

        :return: reward associated with the current state
        """
        return self.negative_reward or -100 if self.done else self.positive_reward or 1

    def get_state(self):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute
                 as detetermined during initialization
        """

        return np.array([self.x[k] for k in self.model_output_index])

    def _set_init_parameter(self):
        """
        Sets initial parameters of a model.

        :return: environment
        """
        if self.model_parameters is not None:
            self.model.setup_experiment(start_time=self.start)
            self.model.enter_initialization_mode()
            self.model.exit_initialization_mode()

            eInfo = self.model.get_event_info()
            eInfo.newDiscreteStatesNeeded = True
            # Event iteration
            while eInfo.newDiscreteStatesNeeded == True:
                self.model.enter_event_mode()
                self.model.event_update()
                eInfo = self.model.get_event_info()

            self.model.enter_continuous_time_mode()

            self.x = self.model.continuous_states
            self.event_ind = self.model.get_event_indicators()
            # print(self.model.get_states_list())
            self.model.set(list(self.model_parameters),
                           list(self.model_parameters.values()))

            """
            gets the indices of the model output states, as they exist within the
            model state space. This is an optimisation to reduce simulation time
            when rearranging the output (instead of using the string names every iteration)
            """
            states = self.model.get_states_list()
            self.model_output_index = [list(states).index(k) for k in flatten(self.model_output_names)]

        return self

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
        return abs(time) > self.time_threshold

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size 3
        """
        return gym.spaces.Discrete(3)

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements

        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([self.time_threshold, np.inf])
        return gym.spaces.Box(-high, high)

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
            if self.viz_mode == 'step':
                # TODO close plot
                pass
            else:
                # TODO create the plot
                for cols in flatten(self.model_output_names, 1):
                    self.history[cols].plot()
                    plt.show()
                # print(self.history)
                pass
        elif self.viz_mode == 'step':
            # TODO update plot
            pass
        return True
