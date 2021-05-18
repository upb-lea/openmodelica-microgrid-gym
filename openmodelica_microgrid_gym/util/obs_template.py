from typing import List, Union, Optional
import numpy as np
from openmodelica_microgrid_gym.agents.util import MutableParams


class ObsTempl:
    def __init__(self, varnames: List[str], simple_tmpl: Optional[List[Union[List[str], np.ndarray]]]):
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
                If None: self.fill() will not filter and return its input wrapped into a list
        """
        idx = {v: i for i, v in enumerate(varnames)}
        self._static_params = set()
        self._data = []
        self.is_tmpl_empty = simple_tmpl is None

        if not self.is_tmpl_empty:
            for i, tmpl in enumerate(simple_tmpl):
                if isinstance(tmpl, np.ndarray) or isinstance(tmpl, MutableParams):
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
        if self.is_tmpl_empty:
            return [obs]
        params = []
        for i, arr in enumerate(self._data):
            # append static data or use dynamic data with indexing
            params.append(arr if i in self._static_params else obs[arr])
        return params
