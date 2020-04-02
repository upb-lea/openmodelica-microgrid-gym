from datetime import datetime
import logging
from os.path import basename
from typing import Sequence, Callable, List, Union

import gym
import numpy as np
import pandas as pd
import scipy
from pyfmi import load_fmu
from pyfmi.fmi import FMUModelME2
from scipy import integrate
import matplotlib.pyplot as plt
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


    If you need more sophisticated logic to be implemented - override methods of OpenAI Gym API
    in the child-class: reset(), step(), render(), close()

    All methods called on model are from implemented PyFMI API.
    """

    viz_modes = {'episode', 'step', None}

    def __init__(self, time_step: float = 1e-4, reward_fun: callable = lambda obs: 1,
                 log_level: int = logging.WARNING, solver_method='LSODA', max_episode_steps: int = None,
                 model_params: dict = None,
                 model_input: Sequence[str] = None, model_output: Sequence[str] = None, model_path='grid.network.fmu',
                 time_start=0,
                 viz_mode: str = 'episode', history: EmptyHistory = FullHistory()):
        """
        :type model_params: dict
        :param model_params: key is variable name, value is a scalar or a callable with parameter time
        :type history: EmptyHistory
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
        self.reward = reward_fun

        # Parameters required by this implementation
        self.time_start = time_start
        self.time_step_size = time_step
        self.time_end = np.inf if max_episode_steps is None else self.time_start + max_episode_steps * self.time_step_size

        # if there are parameters, we will convert all scalars to constant functions.
        self.model_parameters = model_params and {var: (val if isinstance(val, Callable) else lambda t: val) for
                                                  var, val in model_params.items()}

        self.sim_time_interval = None
        self.__state = None
        self.__measurements = None
        self.record_states = viz_mode == 'episode'
        self.history = history
        self.history.cols = model_output
        self.model_input_names = model_input
        self.model_output_names = self.history.cols

        # OpenAI Gym requirements
        self.action_space = gym.spaces.Discrete(3)
        high = np.array([self.time_end, np.inf])
        self.observation_space = gym.spaces.Box(-high, high)

    def _setup_fmu(self):
        """
        Setup FMU
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

        # precalculating indices for more efficient lookup
        self.model_output_idx = np.array(
            [self.model.get_variable_valueref(k) for k in self.model_output_names])

    def _calc_jac(self, t, x):
        # get state and derivative value reference lists
        refs = [[s.value_reference for s in getattr(self.model, attr)().values()]
                for attr in
                ['get_states_list', 'get_derivatives_list']]
        jacobian = np.identity(len(refs[1]))
        np.apply_along_axis(lambda col: self.model.get_directional_derivative(*refs, col), 0, jacobian)
        return jacobian

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

        obs = self.model.get_real(self.model_output_idx)
        return pd.DataFrame([obs], columns=self.model_output_names)

    @property
    def is_done(self) -> bool:
        """
        Checks if the experiment is finished using a time limit

        :return: True if simulation time exceeded
        """
        logger.debug(f't: {self.sim_time_interval[1]}, ')
        return abs(self.sim_time_interval[1]) > self.time_end

    def update_measurements(self, measurements: Union[pd.DataFrame, List]):
        """
        records measurements
        """
        if isinstance(measurements, pd.DataFrame):
            measurements = [measurements]
        for df in measurements:
            self.history.cols = self.history.structured_cols(None) + [col for col in df.columns if
                                                                      col not in set(self.history.cols)]

        self.__measurements = pd.concat(measurements)

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

        self._setup_fmu()
        self.sim_time_interval = np.array([self.time_start, self.time_start + self.time_step_size])
        self.history.reset()
        self.__state = self._simulate()
        self.__measurements = pd.DataFrame()
        self.history.append(self.__state.join(self.__measurements))

        return self.__state

    def step(self, action):
        """
        OpenAI Gym API. Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.
        :param action: action to be executed.
        :return: state, reward, is done, info
        """
        logger.debug("Experiment next step was called.")
        if self.is_done:
            logging.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return self.__state, None, self.is_done

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
        if self.model_parameters:
            # list of keys and list of values
            self.model.set(*zip(*[(var, f(self.sim_time_interval[0])) for var, f in self.model_parameters.items()]))

        # Simulate and observe result state
        self.__state = self._simulate()
        obs = self.__state.join(self.__measurements)
        self.history.append(obs)

        logger.debug("model output: {}, values: {}".format(self.model_output_names, self.__state))

        # Check if experiment has finished
        # Move simulation time interval if experiment continues
        if not self.is_done:
            logger.debug("Experiment step done, experiment continues.")
            self.sim_time_interval += self.time_step_size
        else:
            logger.debug("Experiment step done, experiment done.")

        return obs, self.reward(self.__state), self.is_done, {}

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
                for cols in self.history.structured_cols():
                    df = self.history.df[cols].copy()
                    df.index = self.history.df.index * self.time_step_size
                    df.plot()
                    plt.show()

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
