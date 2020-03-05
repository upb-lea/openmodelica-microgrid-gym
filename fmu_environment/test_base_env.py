# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:53:59 2020

@author: jarren
"""

import datetime
import logging
import os
import gym
import scipy
from scipy import integrate
from pyfmi import load_fmu
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class ModelicaType(Enum):
    Dymola = 1
    JModelica = 2
    OpenModelica = 3


class FMIStandardVersion(Enum):
    first = 1
    second = 2


class ModelicaBaseEnv(gym.Env):
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

    def __init__(self, model_path, mode, config, log_level):
        """

        :param model_path: path to the model FMU. Absolute path is advised.
        :param mode: FMU exporting mode "CS" or "ME"
        :param config: dictionary with model specifications:
            model_input_names - names of parameters to be used as action.
            model_output_names - names of parameters to be used as state descriptors.
            model_parameters - dictionary of initial parameters of the model
            time_step - time difference between simulation steps

            positive_reward - (optional) positive reward for default reward policy. Is returned when episode goes on.
            negative_reward - (optional) negative reward for default reward policy. Is returned when episode is ended

        :param log_level: level of logging to be used
        """

        logger.setLevel(log_level)
        if mode != 'CS' and mode != 'ME':
            logger.warning("Mode should be either CS or ME. Actual value {}. Trying to load in CS mode".format(mode))
            mode = 'CS'

        self.solver_method = config.get('solver_method')
        # load model from fmu
        self.model_name = model_path.split(os.path.sep)[-1]

        logger.debug("Loading model {}".format(self.model_name))

        # self.model = load_fmu(model_path, kind=mode, log_file_name=self.get_log_file_name())
        self.model = load_fmu(model_path, log_file_name=self.get_log_file_name())

        #        states = self.model.get_states_list()
        #        states_references = [s.value_reference for s in states.values()]
        #        derivatives = self.model.get_derivatives_list()
        #        derivatives_references = [d.value_reference for d in derivatives.values()]
        #        n_states = len(states)
        #        self.jacobian = np.zeros([n_states, n_states])
        #        for n in range(0, n_states):
        #            v = np.zeros(n_states)
        #            v[n] = 1
        #            dx = self.model.get_directional_derivative(states_references, derivatives_references, v)
        #            self.jacobian[:,n] = dx
        #

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

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50
        }

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

    # OpenAI Gym API
    def render(self, **kwargs):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        :param kwargs:
        :return: implementation should return rendering result
        """
        pass

    def reset(self):
        """
        OpenAI Gym API. Determines restart procedure of the environment
        :return: environment state after restart
        """
        pass

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
            return np.array(self.state), self.negative_reward, self.done, {}

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
        self.state = self.do_simulation()

        logger.debug("model output: {}, values: {}".format(self.model_output_names, self.state))
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

        return self.state, self._reward_policy(), self.done, self.count_iter, {}

    # logging
    def get_log_file_name(self):
        log_date = datetime.datetime.utcnow()
        log_file_name = "{}-{}-{}_{}.txt".format(log_date.year, log_date.month, log_date.day, self.model_name)
        return log_file_name

    # internal logic
    def _get_action_space(self):
        """
        Returns action space according to OpenAI Gym API requirements.

        :return: one of gym.spaces classes that describes action space according to environment specifications.
        """
        pass

    def _get_observation_space(self):
        """
        Returns state space according to OpenAI Gym API requirements.

        :return: one of gym.spaces classes that describes state space according to environment specifications.
        """
        pass

    def _is_done(self):
        """
        Determines logic when experiment is considered to be done.

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        """
        pass

    # part of the step() method extracted for convenience
    def do_simulation(self):
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

        return tuple([self.x[k] for k in self.model_output_index])

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
            model_outputs = self.model_output_names
            states = self.model.get_states_list()
            self.model_output_index = [list(states).index(k) for k in model_outputs]

        return self


class ModelicaMEEnv(ModelicaBaseEnv):
    """
    Test version.
    Wrapper class of ModelicaBaseEnv for convenient creation of environments that utilize
    FMU exported in a model-exchange mode.
    Should be used as a superclass for all such environments.
    """

    def __init__(self, model_path, config, log_level):
        super().__init__(model_path, "ME", config, log_level)
