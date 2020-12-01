from typing import Optional

import numpy as np


class Fastqueue:
    def __init__(self, size: int, dim: Optional[int] = 1):
        """
        Efficient numpy implementation of constant sized queue
        :param size: Size of queue
        """
        self._buffer = np.empty((size, dim))
        self._idx = 0

    def shift(self, val):
        """
        Pushes val into buffer and returns popped last element
        """
        last = self._buffer[np.ravel_multi_index([self._idx - 1], (len(self._buffer),), mode='wrap'), :]
        self._idx = np.ravel_multi_index([self._idx + 1], (len(self._buffer),), mode='wrap')
        self._buffer[self._idx, :] = val
        return last

    def __len__(self):
        return self._buffer.shape[0]

    def clear(self):
        self._buffer.fill(0)
