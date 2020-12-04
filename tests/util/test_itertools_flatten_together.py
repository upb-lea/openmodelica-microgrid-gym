import pytest

from openmodelica_microgrid_gym.util.itertools_ import flatten_together


@pytest.mark.parametrize('i,o', [[([1, 2, 3], [4, 5, 6]), [4, 5, 6]],
                                 [([1, 1], [0]), [0, 0]],
                                 [([[1, 1]], [0]), [0, 0]],
                                 [([[3, 2], [4]], [2, 3]), [2, 2, 3]],
                                 [([[3, 2], 4], [2, 3]), [2, 2, 3]],
                                 [([[3, 2], 4], 13), [13, 13, 13]],
                                 ])
def test_flatten_together(i, o):
    assert flatten_together(*i) == o


def test_flatten_together_negative():
    with pytest.raises(ValueError):
        # params don't match up
        flatten_together([[3, 3], [3, 3], [3, 3]], [[1], [2]])


def test_flatten_together_negative2():
    with pytest.raises(ValueError):
        # to many nestings in values
        flatten_together([4, 4], [[1], [3]])
