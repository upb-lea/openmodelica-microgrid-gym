import logging
import re
from fnmatch import translate
from functools import partial
from typing import Sequence, Callable, List, Union, Tuple, Optional, Mapping, Dict, Any

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.figure import Figure
from scipy import integrate

from openmodelica_microgrid_gym.env.plot import PlotTmpl
from openmodelica_microgrid_gym.env.pyfmi import PyFMI_Wrapper
from openmodelica_microgrid_gym.net.base import Network
from openmodelica_microgrid_gym.util import FullHistory, EmptyHistory, Fastqueue

logger = logging.getLogger(__name__)


class ModelicaEnv(gym.Env):
    """
    OpenAI gym Environment encapsulating an FMU model.
    """

    viz_modes = {'episode', 'step', None}
    """Set of all valid visualisation modes"""

    def __init__(self, net: Union[str, Network], time_start: float = 0,
                 reward_fun: Callable[[List[str], np.ndarray, float], float] = lambda cols, obs, risk: 1,
                 is_normalized=True,
                 log_level: int = logging.WARNING, solver_method: str = 'LSODA', max_episode_steps: Optional[int] = 200,
                 model_params: Optional[Dict[str, Union[Callable[[float], float], float]]] = None,
                 model_path: str = '../omg_grid/grid.network.fmu',
                 viz_mode: Optional[str] = 'episode', viz_cols: Optional[Union[str, List[Union[str, PlotTmpl]]]] = None,
                 history: EmptyHistory = FullHistory(),
                 action_time_delay: int = 0):
        """
        Initialize the Environment.
        The environment can only be used after reset() is called.

        :param time_start: offset of the time in seconds
        :param reward_fun:
            The function receives as a list of variable names and a np.ndarray of the values
            of the current observation as well as a risk value between 0 and 1.
            The separation is mainly for performance reasons, such that the resolution of data indices can be cached.
            It must return the reward of this timestep as float.
            It should return np.nan or -np.inf or None in case of a failiure.
            It should have no side-effects
        :param log_level: logging granularity. see logging in stdlib
        :param solver_method: solver of the scipy.integrate.solve_ivp function
        :param max_episode_steps: maximum number of episode steps.
            The end time of the episode is calculated by the time resolution and the number of steps.

            If set to None, the environment will never finish because of step sizes, but it might still stop because of
            system failure (-inf reward)
        :param model_params: parameters of the FMU.
            dictionary of variable names and scalars or callables.
            If a callable is provided it is called every time step with the current time.
            This callable must return a float that is passed to the fmu.
        :param net: Path to the network configuration file passed to the net.Network.load() function
        :param model_path: Path to the FMU
        :param viz_mode: specifies how and if to render

            - 'episode': render after the episode is finished
            - 'step': render after each time step
            - None: disable visualization
        :param viz_cols: enables specific columns while plotting

             - None: all columns will be used for visualization (default)
             - string: will be interpret as regex. all fully matched columns names will be enabled
             - list of strings: Each string might be a unix-shell style wildcard like "*.i"
                                to match all data series ending with ".i".
             - list of PlotTmpl: Each template will result in a plot
        :param history: history to store observations and measurement (from the agent) after each step
        :param action_time_delay: Defines how many time steps the controller needs before the action is applied; action is buffered in an array
        """
        if viz_mode not in self.viz_modes:
            raise ValueError(f'Please select one of the following viz_modes: {self.viz_modes}')

        self.viz_mode = viz_mode
        self._register_render = False
        logger.setLevel(log_level)
        self.solver_method = solver_method

        # load model from fmu
        self.model = PyFMI_Wrapper.load(model_path)

        # if you reward policy is different from just reward/penalty - implement custom step method
        self.reward = reward_fun
        self._failed = False

        # Parameters required by this implementation
        self.max_episode_steps = max_episode_steps
        self.time_start = time_start
        if isinstance(net, Network):
            self.net = net
        else:
            self.net = Network.load(net)
        self.time_step_size = self.net.ts
        self.time_end = np.inf if max_episode_steps is None \
            else self.time_start + max_episode_steps * self.time_step_size

        # if there are parameters, we will convert all scalars to constant functions.
        model_params = model_params or dict()
        # the "partial" is needed because of some absurd python behaviour https://stackoverflow.com/a/34021333/13310191
        self.model_parameters = {var: (val if callable(val) else partial(lambda t, val_: val_, val_=val)) for var, val
                                 in
                                 model_params.items()}

        self.sim_time_interval = None
        self._state = np.empty(0)
        self.measure = lambda obs: np.empty(0)  # type : Callable([np.ndarray],np.ndarray)
        self.record_states = viz_mode == 'episode'
        self.history = history
        self.is_normalized = is_normalized
        # also add the augmented values to the history
        self.history.cols = self.net.out_vars(True, False)
        self.model_input_names = self.net.in_vars()
        # variable names are flattened to a list if they have specified in the nested dict manner)
        self.model_output_names = self.net.out_vars(False, True)

        self.viz_col_tmpls = []
        if viz_cols is None:
            logger.info('Provide the option "viz_cols" if you wish to select only specific plots. '
                        'The default behaviour is to plot all data series')
            self.viz_col_regex = '.*'
        elif isinstance(viz_cols, list):
            # strings are glob patterns that can be used in the regex
            patterns, tmpls = [], []
            for elem in viz_cols:
                if isinstance(elem, str):
                    patterns.append(translate(elem))
                elif isinstance(elem, PlotTmpl):
                    tmpls.append(elem)
                else:
                    raise ValueError('"viz_cols" list must contain only strings or PlotTmpl objects not'
                                     f' {type(viz_cols)}')

            self.viz_col_regex = '|'.join(patterns)
            self.viz_col_tmpls = tmpls
        elif isinstance(viz_cols, str):
            # is directly interpret as regex
            self.viz_col_regex = viz_cols
        else:
            raise ValueError('"viz_cols" must be one type Optional[Union[str, List[Union[str, PlotTmpl]]]]'
                             f'and not {type(viz_cols)}')

        # OpenAI Gym requirements
        d_i, d_o = len(self.model_input_names), len(self.history.cols)
        self.action_space = gym.spaces.Box(low=np.full(d_i, -1), high=np.full(d_i, 1))
        self.observation_space = gym.spaces.Box(low=np.full(d_o, -np.inf), high=np.full(d_o, np.inf))

        self.action_time_delay = action_time_delay
        self.delay_buffer = Fastqueue(self.action_time_delay + 1, self.action_space.shape[0])

    def _calc_jac(self, t, x) -> np.ndarray:  # noqa
        """
        Compose Jacobian matrix from the directional derivatives of the FMU model.
        This function will be called by the scipy.integrate.solve_ivp solver,
        therefore we have to obey the expected signature.

        :param t: time (ignored)
        :param x: state (ignored)
        :return: the Jacobian matrix
        """
        return self.model.jacc()

    def _get_deriv(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Retrieve derivatives at given time and with given state from the FMU model

        :param t: time
        :param x: 1d float array of continuous states
        :return: 1d float array of derivatives
        """
        self.model.time = t
        self.model.states = x.copy(order='C')

        # Compute the derivative
        dx = self.model.deriv
        return dx

    def _simulate(self) -> np.ndarray:
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment
        """
        logger.debug(f'Simulation started for time interval {self.sim_time_interval[0]}-{self.sim_time_interval[1]}')

        # Advance
        x_0 = self.model.states

        # Get the output from a step of the solver
        sol_out = scipy.integrate.solve_ivp(
            self._get_deriv, self.sim_time_interval, x_0, method=self.solver_method, jac=self._calc_jac)
        # get the last solution of the solver
        self.model.states = sol_out.y[:, -1]  # noqa

        obs = self.model.obs
        return obs

    @property
    def is_done(self) -> bool:
        """
        Checks if the experiment is finished using a time limit

        :return: True if simulation time exceeded
        """
        if self._failed:
            logger.info(f'risk level exceeded')
            return True
        logger.debug(f't: {self.sim_time_interval[1]}, ')
        return abs(self.sim_time_interval[1]) > self.time_end

    def reset(self) -> np.ndarray:
        """
        OpenAI Gym API. Restarts environment and sets it ready for experiments.
        In particular, does the following:

            * resets model
            * sets simulation start time to 0
            * sets initial parameters of the model
                - Using the parameters defined in self.model_parameters
            * initializes the model

        :return: state of the environment after resetting.
        """
        self.net.reset()
        logger.debug("Experiment reset was called. Resetting the model.")

        self.sim_time_interval = np.array([self.time_start, self.time_start + self.time_step_size])
        self.model.setup(self.time_start, self.model_output_names, self.model_parameters)

        self.history.reset()
        self._failed = False
        self._register_render = False
        self.delay_buffer.clear()
        outputs = self._create_state()
        return outputs

    def step(self, action: Sequence) -> Tuple[np.ndarray, float, bool, Mapping]:
        """
        OpenAI Gym API. Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.

        The state also contains the measurement.

        :param action: action to be executed.
        :return: state, reward, is done, info
        """
        logger.debug("Experiment next step was called.")
        if self.is_done:
            logger.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return self._state, -np.inf, True, {}

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

        # enqueue action and get delayed/last action
        delayed_action = self.delay_buffer.shift(action)

        # Set input values of the model
        logger.debug('model input: %s, values: %s', self.model_input_names, delayed_action)
        self.model.set(**dict(zip(self.model_input_names, delayed_action)))

        if self.model_parameters:
            values = {var: f(self.sim_time_interval[0]) for var, f in self.model_parameters.items()}
        else:
            values = {}

        params = {**values, **self.net.params(delayed_action)}
        if params:
            self.model.set_params(**params)
        risk = self.net.risk()

        # Simulate and observe result state
        outputs = self._create_state()

        logger.debug("model output: %s, values: %s", self.model_output_names, self._state)

        # Check if experiment has finished
        # Move simulation time interval if experiment continues
        if not self.is_done:
            logger.debug("Experiment step done, experiment continues.")
            self.sim_time_interval += self.time_step_size
        else:
            logger.debug("Experiment step done, experiment done.")

        reward = self.reward(self.history.cols, outputs, risk)
        self._failed = risk >= 1 or np.isnan(reward) or (np.isinf(reward) and reward < 0) or reward is None

        # only return the state, the agent does not need the measurement
        return outputs, reward, self.is_done, dict(risk=risk)

    def _create_state(self):
        # Simulate and observe result state
        self._state = self._simulate()
        outputs = self.net.augment(self._state, self.is_normalized)
        outputs = np.hstack([outputs, self.measure(outputs)])
        self.history.append(outputs)
        return outputs

    def render(self, mode: str = 'human', close: bool = False) -> List[Figure]:
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Does nothing at the moment

        :param mode: (ignored) rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned. Used, when environment is closed.
        """
        if self.viz_mode is None:
            return []
        elif self.viz_mode == 'step':
            if close:
                # TODO close plot
                pass
            pass

        elif self.viz_mode == 'episode':
            # TODO update plot
            if not close:
                self._register_render = True
            elif self._register_render:
                figs = []

                # plot cols by theirs structure filtered by the vis_cols param
                for cols in self.history.structured_cols():
                    if not isinstance(cols, list):
                        cols = [cols]
                    cols = [col for col in cols if re.fullmatch(self.viz_col_regex, col)]
                    if not cols:
                        continue
                    df = self.history.df[cols].copy()
                    df.index = self.history.df.index * self.time_step_size + self.time_start

                    fig, ax = plt.subplots()
                    df.plot(legend=True, figure=fig, ax=ax)
                    plt.show()
                    figs.append(fig)

                # plot all templates
                for tmpl in self.viz_col_tmpls:
                    fig, ax = plt.subplots()

                    for series, kwargs in tmpl:
                        ser = self.history.df[series].copy()
                        ser.index = self.history.df.index * self.time_step_size + self.time_start
                        ser.plot(figure=fig, ax=ax, **kwargs)
                    tmpl.callback(fig)
                    figs.append(fig)

                return figs

    def close(self) -> Tuple[bool, Any]:
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.

        :return: True on success
        """
        figs = self.render(close=True)
        return True, figs
