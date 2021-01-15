from typing import Type

import numpy as np
from stochastic.processes import DiffusionProcess
from stochastic.processes.base import BaseProcess


class RandProcess:
    def __init__(self, process_cls=Type[BaseProcess], proc_kwargs=None, bounds=(0, 1), initial=0):
        """
        wrapper around stochastic processes to allow easier integration
        """
        self.proc = process_cls(**(proc_kwargs or {}))
        self.bounds = bounds

        # will contain the previous value, hence initialized accordingly
        self.last = initial
        self.last_t = 0

    def sample(self, t):
        self.proc.t = t - self.last_t
        self.last_t = t
        if isinstance(self.proc, DiffusionProcess):
            self.last = np.clip(self.proc.sample(1, initial=self.last)[-1], *self.bounds).squeeze()
        else:
            self.last = np.clip(self.last + self.proc.sample(1)[-1], *self.bounds).squeeze()
        return self.last
