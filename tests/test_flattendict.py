import pytest
import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.common.itertools_ import fill_params, flatten, nested_map, nested_depth

conf = {
    'lc1': [
        ['inductor1.i', 'inductor2.i', 'inductor3.i'],
        ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
    'lcl1':
        [['inductor1.i', 'inductor2.i', 'inductor3.i'],
         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]}
conf2 = {
    'lc1': [
        ['inductor1.i', 'inductor2.i', 'inductor3.i'],
        ['capacitor1.v', 'capacitor2.v'],
        ['capacitor3.v']],
    'lcl1': conf['lcl1']}

conf3 = {
    'lc1': [
        ['inductor1.i', 'inductor2.i', 'inductor3.i'],
        ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
    'lcl1':
        [['inductor1.i', 'inductor2.i', 'inductor3.i'],
         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
    'pll':
        ['add_freq_nom_delta_f.y']}

result_1 = [['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i'],
            ['lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v'],
            ['lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i'],
            ['lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v']]
result_1_2 = [['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i'],
              ['lc1.capacitor1.v', 'lc1.capacitor2.v'],
              ['lc1.capacitor3.v'],
              ['lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i'],
              ['lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v']]
result_0 = ['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i',
            'lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v',
            'lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i',
            'lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v']
result_3_0 = result_0 + ['pll.add_freq_nom_delta_f.y']


@pytest.mark.parametrize('i,o', [[[conf], result_0], [[result_1], result_0], [[conf2], result_0], [[conf3], result_3_0],
                                 [[conf, 1], result_1], [[result_1, 1], result_1], [[conf2, 1], result_1_2],
                                 [[result_1_2, None], result_1_2]])
def test_flatten(i, o):
    assert flatten(*i) == o


def test_nested_map():
    assert nested_map(lambda x: 'p' + x, ['a', 'b', 'c']) == ['pa', 'pb', 'pc']


def test_nested_map1():
    assert np.array_equal(nested_map(len, np.array(['a', 'b', 'c'])), np.array([1, 1, 1]))


@pytest.mark.parametrize('i,o', [[1, 0], [[1], 1], [[], 1], [[[], 1], 2], [result_1, 2], [result_1_2, 2]])
def test_nested_depth(i, o):
    assert nested_depth(i) == o


@pytest.mark.parametrize('tmpl,data,result',
                         [[dict(a=['a', 'b', 'c']), pd.Series(dict(a=1, b=2, c=3)), dict(a=[1, 2, 3])],
                          [dict(a=[np.array(['a', 'b', 'c']), np.array(['d', 'b', 'c']), 1]),
                           pd.Series(dict(a=1, b=2, c=3, d=4)),
                           dict(a=[np.array([1., 2., 3.]), np.array([4., 2., 3.]), 1])]])
def test_fill_params(tmpl, data, result):
    # properly testing datastructures with nested numpy arrays is complicated because of "ambiguous truthness"
    assert str(fill_params(tmpl, data)) == str(result)
