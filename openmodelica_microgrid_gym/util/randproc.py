from typing import Type

import numpy as np
from stochastic.processes import DiffusionProcess
from stochastic.processes.base import BaseProcess


class RandProcess:
    def __init__(self, process_cls=Type[BaseProcess], proc_kwargs=None, bounds=(0, 1), initial=0, gain=1):
        """
        wrapper around stochastic processes to allow easier integration

        :param process_cls: class of the stochastic process
        :param proc_kwargs: arguments passed to the class on initialization
        :param bounds: boundaries of admissible values
        :param initial: starting value of the process
        :param gain: scale of the process output (of not set, results are between the clipped bounds)
        """
        self.proc = process_cls(**(proc_kwargs or {}))
        self.bounds = bounds
        self.gain = gain

        # will contain the previous value, hence initialized accordingly
        self.last = initial
        self.last_t = 0

    def sample(self, t):
        """
        calculates time differential and calculates the change in the outputs of the process
        :param t: timestep
        :return: value at the timestep
        """
        if t != self.last_t:
            # if not initial actually sample from processes otherwise return initial value
            self.proc.t = t - self.last_t
            self.last_t = t
            if isinstance(self.proc, DiffusionProcess):
                self.last = np.clip(self.proc.sample(1, initial=self.last)[-1], *self.bounds).squeeze()
            else:
                self.last = np.clip(self.last + self.proc.sample(1)[-1], *self.bounds).squeeze()

        return self.last * self.gain
