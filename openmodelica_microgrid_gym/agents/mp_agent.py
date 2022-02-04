from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import PLLParams, PLL
from typing import List, Mapping, Union

import numpy as np

from openmodelica_microgrid_gym.aux_ctl import Controller
from openmodelica_microgrid_gym.util import ObsTempl


class PllAgent(StaticControlAgent):

    def __init__(self, pllPIParams: PLLParams, ts, ctrls: List[Controller],
                 obs_template: Mapping[str, List[Union[List[str], np.ndarray]]],
                 obs_varnames: List[str] = None, **kwargs):

        self._ts = ts

        super().__init__(ctrls, obs_template, obs_varnames, **kwargs)

        self._pll = PLL(pllPIParams, self._ts)


    def measure(self, state) -> np.ndarray:

        obs = super().measure(state)

        asd = 1
        #self._pll.step()

        return obs


