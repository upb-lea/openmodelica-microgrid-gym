from gym_microgrid.common.flattendict import flatten

conf = {
    'lc1': [
        ['inductor1.i', 'inductor2.i', 'inductor3.i'],
        ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
    'lcl1':
        [['inductor1.i', 'inductor2.i', 'inductor3.i'],
         ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]}

result_1 = [['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i'],
            ['lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v'],
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
