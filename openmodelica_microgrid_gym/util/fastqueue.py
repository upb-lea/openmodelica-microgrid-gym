from typing import Optional

import numpy as np


class Fastqueue:
    def __init__(self, size: int, dim: Optional[int] = 1):
        """
        Efficient numpy implementation of constant sized queue without queue-shifting (indices-based value selection).
        Queue size of n leads to a delay of n-1.
        :param size: Size of queue
        """
        self._buffer = None
        self._size, self._dim = size, dim
        self._idx = 0

    def shift(self, val):
        """
        Pushes val into buffer and returns popped last element
        """
        if self._buffer is None:
            raise RuntimeError('please call clear() before using the object')
        self._idx = self.wrap_index(self._idx + 1)
        last = self._buffer[self._idx, :].copy()
        self._buffer[self._idx, :] = val
        return last

    def __len__(self):
        return self._size

    def clear(self):
        self._buffer = np.zeros((self._size, self._dim))

    def wrap_index(self, i):
        # ringbuffer implementation with index calculated using np.ravel... -> no shifting
        return np.ravel_multi_index([i], (len(self),), mode='wrap')
