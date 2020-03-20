import pytest

from gym_microgrid.common.flattendict import flatten, nested_map, nested_depth

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


@pytest.mark.parametrize('i,o', [[[conf], result_0], [[result_1], result_0], [[conf2], result_0],
                                 [[conf, 1], result_1], [[result_1, 1], result_1], [[conf2, 1], result_1_2]])
def test_flatten(i, o):
    assert flatten(*i) == o


def test_nested_map():
    assert nested_map(['a', 'b', 'c'], lambda x: 'p' + x) == ['pa', 'pb', 'pc']


@pytest.mark.parametrize('i,o', [[1, 0], [[1], 1], [[], 1], [[[], 1], 2], [result_1, 2], [result_1_2, 2]])
def test_nested_depth(i, o):
    assert nested_depth(i) == o
