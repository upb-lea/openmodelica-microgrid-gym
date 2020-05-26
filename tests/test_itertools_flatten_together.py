import pytest

from openmodelica_microgrid_gym.util.itertools_ import flatten_together


@pytest.mark.parametrize('i,o', [[([1, 2, 3], [4, 5, 6]), [4, 5, 6]],
                                 [([1, 1], [0]), [0, 0]],
                                 [([[1, 1]], [0]), [0, 0]],
                                 [([[3, 2], [4]], [2, 3]), [2, 2, 3]]
                                 ])
def test_flatten_together(i, o):
    assert flatten_together(*i) == o
