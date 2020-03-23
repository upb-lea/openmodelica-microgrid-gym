"""
Created on Tue Jan  7 09:01:39 2020

@author: jarren
"""
from gym_microgrid.controllers import PI_params
import numpy as np


class PIController:
    """
    Implements a basic PI controller.
    Uses back calculation for anti-windup
    """

    def __init__(self, PI_param: PI_params, ts):
        """
        :param PI_param: The PI_Parameters object with the PI controller
        parameters
        """
        self._params = PI_param
        self.integralSum = 0
        self.windup_compensation = 0
        self._ts = ts

    def reset(self):
        self.integralSum = 0
        self._subsample_count = 0

    def step(self, error):
        """
        implements a step of a basic PI controller with anti-windup by back-calculation

        :param error: control error to act on
        :return: the calculated PI controller response to the error, using the
                PI_Parameters provided during initialisation.
        """

        self.integralSum = self.integralSum + (self._params.kI * error + self.windup_compensation) * self._ts
        output = self._params.kP * error + self.integralSum
        clipped = np.clip(output, *self._params.limits)
        self.windup_compensation = (output - clipped) * self._params.kB
        return clipped
