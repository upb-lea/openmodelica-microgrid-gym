from datetime import datetime
import re
import logging
from os.path import basename
from typing import Sequence, Callable, List, Union, Tuple, Optional, Mapping

import gym
import numpy as np
import pandas as pd
import scipy
from pyfmi import load_fmu
from pyfmi.fmi import FMUModelME2
from scipy import integrate
from fnmatch import translate
import matplotlib.pyplot as plt

from openmodelica_microgrid_gym.common.itertools_ import flatten
from openmodelica_microgrid_gym.env.recorder import FullHistory, EmptyHistory

logger = logging.getLogger(__name__)


class ModelicaEnv(gym.Env):
    """
    OpenAI gym Environment encapsulating an FMU model.
    """

    viz_modes = {'episode', 'step', None}
    """Set of all valid visualisation modes"""

    def __init__(self, time_step: float = 1e-4, time_start: float = 0,
                 reward_fun: Callable[[pd.Series], float] = lambda obs: 1,
                 log_level: int = logging.WARNING, solver_method: str = 'LSODA', max_episode_steps: int = None,
                 model_params: Optional[dict] = None, model_input: Optional[Sequence[str]] = None,
                 model_output: Optional[Union[dict, Sequence[str]]] = None, model_path: str = '../fmu/grid.network.fmu',
                 viz_mode: Optional[str] = 'episode', viz_cols: Optional[Union[str, List[str]]] = None,
                 history: EmptyHistory = FullHistory()):
        """
        Initialize the Environment.
        The environment can only be used after reset() is called.

        :param time_step: step size of the simulation in seconds
        :param time_start: offset of the time in seconds

        :param reward_fun:
            The function receives the observation as a pd.Series and must return the reward of this timestep as float.
            In case of a failure this function should return np.nan or -np.inf or None.
            Should have no side-effects
        :param log_level: logging granularity. see logging in stdlib
        :param solver_method: solver of the scipy.integrate.solve_ivp function
        :param max_episode_steps: maximum number of episode steps.
            The end time of the episode is calculated by the time resolution and the number of steps.

        :param model_params: parameters of the FMU.

            dictionary of variable names and scalars or callables.
            If a callable is provided it is called every time step with the current time.
        :param model_input: list of strings. Each string representing a FMU input variable.
        :param model_output: nested dictionaries containing nested lists of strings.
         The keys of the nested dictionaries will be flattened down and appended to their children and finally prepended
         to the strings in the nested lists. The strings final strings represent variables from the FMU and the nesting
         of the lists conveys structure used in the visualisation

         >>> {'inverter': {'condensator': ['i', 'v']}}

         results in

         >>> ['inverter.condensator.i', 'inverter.condensator.v']
        :param model_path: Path to the FMU
        :param viz_mode: specifies how and if to render

            - 'episode': render after the episode is finished
            - 'step': render after each time step
            - None: disable visualization
        :param viz_cols: enables specific columns while plotting
             - None: all columns will be used for vizualization (default)
             - string: will be interpret as regex. all fully matched columns names will be enabled
             - list of strings: Each string might be a unix-shell style wildcard like "*.i"
                                to match all data series ending with ".i".
        :param history: history to store observations and measurements (from the agent) after each step
        """
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
        self._failed = False

        # Parameters required by this implementation
        self.time_start = time_start
        self.time_step_size = time_step
        self.time_end = np.inf if max_episode_steps is None \
            else self.time_start + max_episode_steps * self.time_step_size

        # if there are parameters, we will convert all scalars to constant functions.
        self.model_parameters = model_params and {var: (val if isinstance(val, Callable) else lambda t: val) for
                                                  var, val in model_params.items()}

        self.sim_time_interval = None
        self.__state = pd.Series()
        self.__measurements = pd.Series()
        self.record_states = viz_mode == 'episode'
        self.history = history
        self.history.cols = model_output
        self.model_input_names = model_input
        # variable names are flattened to a list if they have specified in the nested dict manner
        self.model_output_names = self.history.cols
        if viz_cols is None:
            logger.info('Provide the option "viz_cols" if you wish to select only specific plots. '
                        'The default behaviour is to plot all data series')
            self.viz_col_regex = '.*'
        elif isinstance(viz_cols, list):
            self.viz_col_regex = '|'.join([translate(glob) for glob in viz_cols])
        elif isinstance(viz_cols, str):
            # is directly interpret as regex
            self.viz_col_regex = viz_cols
        else:
            raise ValueError('"selected_vis_series" must be one of the following:'
                             ' None, str(regex), list of strings (list of shell like globbing patterns) '
                             f'and not {type(viz_cols)}')

        # OpenAI Gym requirements
        self.action_space = gym.spaces.Discrete(3)
        high = np.array([self.time_end, np.inf])
        self.observation_space = gym.spaces.Box(-high, high)

    def _setup_fmu(self):
        """
        Initialize fmu model in self.model
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

    def _calc_jac(self, t, x) -> np.ndarray:  # noqa
        """
        Compose Jacobian matrix from the directional derivatives of the FMU model.
        This function will be called by the scipy.integrate.solve_ivp solver,
        therefore we have to obey the expected signature.

        :param t: time (ignored)
        :param x: state (ignored)
        :return: the Jacobian matrix
        """
        # get state and derivative value reference lists
        refs = [[s.value_reference for s in getattr(self.model, attr)().values()]
                for attr in
                ['get_states_list', 'get_derivatives_list']]
        jacobian = np.identity(len(refs[1]))
        np.apply_along_axis(lambda col: self.model.get_directional_derivative(*refs, col), 0, jacobian)
        return jacobian

    def _get_deriv(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Retrieve derivatives at given time and with given state from the FMU model

        :param t: time
        :param x: 1d float array of continuous states
        :return: 1d float array of derivatives
        """
        self.model.time = t
        self.model.continuous_states = x.copy(order='C')

        # Compute the derivative
        dx = self.model.get_derivatives()
        return dx

    def _simulate(self) -> pd.Series:
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment
        """
        logger.debug(f'Simulation started for time interval {self.sim_time_interval[0]}-{self.sim_time_interval[1]}')

        # Advance
        x_0 = self.model.continuous_states

        # Get the output from a step of the solver
        sol_out = scipy.integrate.solve_ivp(
            self._get_deriv, self.sim_time_interval, x_0, method=self.solver_method, jac=self._calc_jac)
        # get the last solution of the solver
        self.model.continuous_states = sol_out.y[:, -1]  # noqa

        obs = self.model.get_real(self.model_output_idx)
        return pd.Series(obs, index=self.model_output_names)

    @property
    def is_done(self) -> bool:
        """
        Checks if the experiment is finished using a time limit

        :return: True if simulation time exceeded
        """
        if self._failed:
            logger.info(f'reward was extreme, episode terminated')
            return True
        # TODO allow for other stopping criteria
        logger.debug(f't: {self.sim_time_interval[1]}, ')
        return abs(self.sim_time_interval[1]) > self.time_end

    def update_measurements(self, measurements: Union[pd.Series, List[Tuple[List, pd.Series]]]):
        """
        Store measurements into a internal field and update the columns of the history

        :param measurements:
         If provided as a list of tuples the first parameter of each tuple is a nested list of strings.
         Each string is a column name of the pd.Series.
         The nesting structure provides grouping that is used for visualization.
        """
        if isinstance(measurements, pd.Series):
            measurements = [(measurements.colums, measurements)]
        for cols, _ in measurements:
            miss_col_count = len(set(flatten(cols)) - set(self.history.cols))
            if 0 < miss_col_count < len(flatten(cols)):
                raise ValueError('some of the columns are already added, this should not happen: '
                                 f'cols:"{cols}"; self.history.cols:"{self.history.cols}"')
            elif miss_col_count:
                self.history.cols = self.history.structured_cols() + cols

        self.__measurements = pd.concat(list(map(lambda ser: ser[1], measurements)))  # type: pd.Series

    def reset(self) -> pd.Series:
        """
        OpenAI Gym API. Restarts environment and sets it ready for experiments.
        In particular, does the following:
            * resets model
            * sets simulation start time to 0
            * sets initial parameters of the model
            * initializes the model
        :return: state of the environment after resetting.
        """
        logger.debug("Experiment reset was called. Resetting the model.")

        self.model.reset()
        self.model.setup_experiment(start_time=0)

        self._setup_fmu()
        self.sim_time_interval = np.array([self.time_start, self.time_start + self.time_step_size])
        self.history.reset()
        self.__state = self._simulate()
        self.__measurements = pd.Series()
        self.history.append(self.__state)
        self._failed = False

        return self.__state

    def step(self, action: Sequence) -> Tuple[pd.Series, float, bool, Mapping]:
        """
        OpenAI Gym API. Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.

        :param action: action to be executed.
        :return: state, reward, is done, info

         The state also contains the measurements passed through by update_measurements
        """
        logger.debug("Experiment next step was called.")
        if self.is_done:
            logger.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return self.__state, -np.inf, True, {}

        # check if action is a list. If not - create list of length 1
        try:
            iter(action)
        except TypeError:
            action = [action]
            logger.warning("Model input values (action) should be passed as a list")

        # Check if number of model inputs equals number of values passed
        if len(action) != len(list(self.model_input_names)):
            message = f'List of values for model inputs should be of the length {len(list(self.model_input_names))},'
            f'equal to the number of model inputs. Actual length {len(action)}'
            logger.error(message)
            raise ValueError(message)

        # Set input values of the model
        logger.debug("model input: {}, values: {}".format(self.model_input_names, action))
        self.model.set(list(self.model_input_names), list(action))
        if self.model_parameters:
            # list of keys and list of values
            self.model.set(*zip(*[(var, f(self.sim_time_interval[0])) for var, f in self.model_parameters.items()]))

        # Simulate and observe result state
        self.__state = self._simulate()
        obs = self.__state.append(self.__measurements)
        self.history.append(obs)

        logger.debug("model output: {}, values: {}".format(self.model_output_names, self.__state))

        # Check if experiment has finished
        # Move simulation time interval if experiment continues
        if not self.is_done:
            logger.debug("Experiment step done, experiment continues.")
            self.sim_time_interval += self.time_step_size
        else:
            logger.debug("Experiment step done, experiment done.")

        reward = self.reward(obs)
        self._failed = np.isnan(reward) or np.isinf(reward) and reward < 0 or reward is None

        # only return the state, the agent does not need the measurements
        return self.__state, reward, self.is_done, dict(self.__measurements)

    def render(self, mode: str = 'human', close: bool = False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Does nothing at the moment

        :param mode: (ignored) rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        """
        if self.viz_mode is None:
            return True
        elif close:
            if self.viz_mode == 'step':
                # TODO close plot
                pass
            else:
                for cols in self.history.structured_cols():
                    if not isinstance(cols, list):
                        cols = [cols]
                    cols = [col for col in cols if re.fullmatch(self.viz_col_regex, col)]
                    if not cols:
                        continue
                    df = self.history.df[cols].copy()
                    df.index = self.history.df.index * self.time_step_size
                    df.plot(legend=True)
                    plt.show()

        elif self.viz_mode == 'step':
            # TODO update plot
            pass

    def close(self) -> bool:
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.

        :return: True on success
        """
        self.render(close=True)
        return True
