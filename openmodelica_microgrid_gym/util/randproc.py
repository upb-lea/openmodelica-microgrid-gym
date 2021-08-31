from typing import Type

import numpy as np
from stochastic.processes import DiffusionProcess
from stochastic.processes.base import BaseProcess


class RandProcess:
    def __init__(self, process_cls: Type[BaseProcess], proc_kwargs=None, bounds=None, initial=0):
        """
        wrapper around stochastic processes to allow easier integration

        :param process_cls: class of the stochastic process
        :param proc_kwargs: arguments passed to the class on initialization
        :param bounds: boundaries of admissible values
        :param initial: starting value of the process
        :param gain: scale of the process output (of not set, results are between the clipped bounds)
        """
        self.proc = process_cls(**(proc_kwargs or {}))
        if bounds is None:
            self.bounds = (-np.inf, np.inf)
        else:
            self.bounds = bounds

        # will contain the previous value, hence initialized accordingly
        self._last = initial
        self._last_t = 0
        self._reserve = None

    def reset(self, initial=None):
        """
        Resets the process, if initial is None, it is set randomly in the range of bounds
        """
        if initial is None:
            self._last = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
            self.proc.mean = self._last
            # self.reserve = self._last
        else:
            self._last = initial
            self.proc.mean = self._last
            #self.reserve = self._last
        self._last_t = 0
        self._reserve = None

    def sample(self, t):
        """
        calculates time differential and calculates the change in the outputs of the process
        :param t: timestep
        :return: value at the timestep
        """
        # if not initial actually sample from processes otherwise return initial value
        if t != self._last_t and t >= 0:
            if self.reserve is not None:
                self._last = self.reserve
                self.reserve = None
            self.proc.t = t - self._last_t
            self._last_t = t
            if isinstance(self.proc, DiffusionProcess):
                self._last = np.clip(self.proc.sample(1, initial=self._last)[-1], *self.bounds).squeeze()
            else:
                self._last = np.clip(self._last + self.proc.sample(1)[-1], *self.bounds).squeeze()

        return self._last

    @property
    def reserve(self):
        """
        This variable is used to prepare for external loadsteps or other abrupt changes in the process' variables.
        """
        return self._reserve

    @reserve.setter
    def reserve(self, v):
        self._reserve = v
