from itertools import chain
from typing import List, Mapping, Union

import numpy as np

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.aux_ctl import Controller


class ObsTempl:
    def __init__(self, varnames: List[str], simple_tmpl: List[Union[List[str], np.ndarray]]):
        """
        Internal dataclass to handle the conversion of dynamic observation templates for the StaticControlAgent

        :param varnames: list of variable names
        :param simple_tmpl: list of:
                - list of strings
                    - matching variable names of the state
                    - must match self.obs_varnames
                    - will be substituted by the values on runtime
                    - will be passed as an np.array of floats to the controller
                - np.array of floats (to be passed statically to the controller)
                - a mixture of static and dynamic values in one parameter is not supported for performance reasons.
        """
        idx = {v: i for i, v in enumerate(varnames)}
        self._static_params = set()
        self._data = []

        for i, tmpl in enumerate(simple_tmpl):
            if isinstance(tmpl, np.ndarray):
                # all np.ndarrays are considered static parameters
                self._static_params.add(i)
                self._data.append(tmpl)
            else:
                # else we save the indices of the variables into an indexarray
                self._data.append(np.array([idx[varname] for varname in tmpl]))

    def fill(self, obs: np.ndarray) -> List[np.ndarray]:
        """
        generates a list of parameters by filling the dynamic values and passing the static values

        :param obs: np.ndarray of values
        :return: list of parameters
        """
        params = []
        for i, arr in enumerate(self._data):
            # append static data or use dynamic data with indexing
            params.append(arr if i in self._static_params else obs[arr])
        return params


class StaticControlAgent(Agent):
    def __init__(self, ctrls: List[Controller], obs_template: Mapping[str, List[Union[List[str], np.ndarray]]],
                 obs_varnames: List[str] = None, **kwargs):
        """
        Simple agent that controls the environment by using auxiliary controllers that are fully configured.
        The Agent does not learn anything.

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
                - np.array of floats (to be passed statically to the controller)
                - a mixture of static and dynamic values in one parameter is not supported for performance reasons.
            The values will be passed as parameters to the controllers step function.
        :param obs_varnames: list of variable names that match the values of the observations
         passed in the act function. Will be automatically set by the Runner class
        """
        super().__init__(obs_varnames, **kwargs)
        self.episode_return = 0
        self.controllers = {ctrl.name: ctrl for ctrl in ctrls}
        self.obs_template_param = obs_template
        self._obs_template = None

    @property
    def obs_template(self):
        """
        lazy fill convert the observation template as the runner will set the

        :return:
        """
        if self._obs_template is None:
            self._obs_template = {ctrl: ObsTempl(self.obs_varnames, tmpl)
                                  for ctrl, tmpl in self.obs_template_param.items()}
        return self._obs_template

    def act(self, state: np.ndarray):
        """
        Executes the actions with the observations as parameters.

        :param state: the agent itself is stateless. the state is stored in the controllers.
        Therefore we simply pass the observation from the environment into the controllers.
        """
        controls = []
        for key, tmpl in self.obs_template.items():
            params = tmpl.fill(state)
            controls.append(self.controllers[key].step(*params))
        return np.concatenate(controls)

    def observe(self, reward: float, terminated: bool):
        """
        The observe function is might be called after the act function.
        It might trigger the learning in some implementations.

        :param reward: reward from the environment after the last action
        :param terminated: whether the episode is finished
        """
        self.episode_return += reward or 0
        if terminated:
            # reset episode reward
            self.prepare_episode()
        # on other steps we don't need to do anything

    @property
    def measurement_cols(self) -> List[Union[List, str]]:
        """
        Structured columns of the measurement. Used in the Runner to setup the history columns of the Environment.

        :return: structured columns of measurement
        """
        return [ctrl.history.structured_cols(None) for ctrl in self.controllers.values()]

    @property
    def measurement(self) -> np.ndarray:
        """
        Measurements the agent takes on the environment. This data is passed to the environment.
        The values returned by this property should be fully determined by the environment.
        This is a workaround to provide data measurement like PLL controllers in the environment even though
        they are functionally part of the Agent.

        :return: current measurement
        """
        return np.array(list(chain.from_iterable([ctrl.history.last() for ctrl in self.controllers.values()])))

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (initial value of integrators...)
        """
        for ctrl in self.controllers.values():
            ctrl.reset()
        self.episode_return = 0

    @property
    def has_improved(self) -> bool:
        """
        Defines if the performance increased or stays constant
        Does not learn, can never improve

        :return: False
        """
        return False
