from pytest import approx

from openmodelica_microgrid_gym.util import Fastqueue
import numpy as np


def test_fastqueue():
    np.random.seed(1)
    test_queue1d = Fastqueue(3)
    first_val = np.random.uniform()
    test_queue1d.shift(first_val)
    test_queue1d.shift(np.random.uniform())
    assert first_val == test_queue1d.shift(np.random.uniform())


def test_fastqueue2d():
    np.random.seed(1)
    test_queue2d = Fastqueue(3, 2)
    first_val = np.random.uniform(size=2)
    test_queue2d.shift(first_val)
    test_queue2d.shift(np.random.uniform(size=2))
    assert first_val == approx(test_queue2d.shift(np.random.uniform(size=2)))
