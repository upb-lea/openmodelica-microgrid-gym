import numpy as np
import pytest

from openmodelica_microgrid_gym.util import ObsTempl
from tests.helpers import nested_arrays_equal


@pytest.mark.parametrize('i,o', [[list('ab'), [np.array([1]), np.array([2])]],
                                 [list('a'), [np.array([1])]],
                                 [[['a', 'b']], [np.array([1, 2])]],
                                 [[['a', 'b'], ['c']], [np.array([1, 2]), np.array([3])]],
                                 [None, [np.array([1, 2, 3])]]
                                 ])
def test_obs_templ(i, o):
    tmpl = ObsTempl(list('abc'), i)
    assert nested_arrays_equal(o, tmpl.fill(np.array([1, 2, 3])))
