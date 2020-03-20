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
    'lcl1':
        [['inductor1.i', 'inductor2.i', 'inductor3.i'],
         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]}

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


def test_flatten():
    assert flatten(conf) == result_0


def test_flatten2():
    assert flatten(conf, 1) == result_1


def test_flatten3():
    assert flatten(result_1, 1) == result_1


def test_flatten4():
    assert flatten(result_1) == result_0


def test_flatten5():
    assert flatten(conf2) == result_0


def test_flatten6():
    assert flatten(conf2, 1) == result_1_2


def test_nested_map():
    assert nested_map(['a', 'b', 'c'], lambda x: 'p' + x) == ['pa', 'pb', 'pc']


def test_nested_depth():
    assert nested_depth(1) == 0


def test_nested_depth2():
    assert nested_depth(result_0) == 1


def test_nested_depth3():
    assert nested_depth(result_1) == 2


def test_nested_depth4():
    assert nested_depth(result_1_2) == 2
