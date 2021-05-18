import numpy as np
import pytest
from pytest import approx

from openmodelica_microgrid_gym.util import Fastqueue


def test_fastqueue_1_element():
    np.random.seed(1)
    test_queue1d = Fastqueue(1)
    test_queue1d.clear()
    first_val = np.random.uniform()
    test_queue1d.shift(first_val)
    assert first_val == test_queue1d.shift(np.random.uniform())


def test_fastqueue_2_element():
    np.random.seed(1)
    test_queue1d = Fastqueue(2)
    test_queue1d.clear()
    first_val = np.random.uniform()
    test_queue1d.shift(first_val)
    test_queue1d.shift(np.random.uniform())
    assert first_val == test_queue1d.shift(np.random.uniform())


def test_fastqueue_3_elem():
    np.random.seed(1)
    test_queue1d = Fastqueue(3)
    test_queue1d.clear()
    first_val = np.random.uniform()
    test_queue1d.shift(first_val)
    test_queue1d.shift(np.random.uniform())
    test_queue1d.shift(np.random.uniform())
    assert first_val == test_queue1d.shift(np.random.uniform())


def test_fastqueue2d():
    np.random.seed(1)
    test_queue2d = Fastqueue(3, 2)
    test_queue2d.clear()
    first_val = np.random.uniform(size=2)
    test_queue2d.shift(first_val)
    test_queue2d.shift(np.random.uniform(size=2))
    test_queue2d.shift(np.random.uniform(size=2))
    assert first_val == approx(test_queue2d.shift(np.random.uniform(size=2)))


def test_fastqueue_initialize():
    q = Fastqueue(5)
    with pytest.raises(RuntimeError):
        q.shift(3)


def test_fastqueue2d_not_random():
    np.random.seed(1)
    test_queue2d = Fastqueue(3, 2)
    test_queue2d.clear()
    first_val = np.array([1, 2])
    test_queue2d.shift(first_val)
    test_queue2d.shift(np.random.uniform(size=2))
    test_queue2d.shift(np.random.uniform(size=2))
    assert np.array([1, 2]) == approx(test_queue2d.shift(np.random.uniform(size=2)))
