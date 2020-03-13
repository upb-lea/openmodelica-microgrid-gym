from collections import namedtuple

import numpy as np
from pytest import approx

from controllers.DiscreteInverterControl import *

droop_params = namedtuple('params', ['gain', 'tau', 'nom_val', 'derivativeFiltParams'])
pi_parms = namedtuple('params',
                      ['kI', 'kP', 'kB', 'lower_limit', 'upper_limit', 'f_nom', 'theta_0', 'derivativeFiltParams'])


def test_step():
    np.random.seed(1)
    dparams = droop_params(2, 3, 2, 4)
    piparams = pi_parms(2, 3, 2, 3, 5, 1, 4, 4)
    ctl = MultiPhaseABCPIPIController(piparams, piparams, 3, dparams, dparams)
    assert ctl.step(np.random.random(3), np.random.random(3), 4, 4) == [5, 5, 5]


def test_step2():
    np.random.seed(1)
    dparams = droop_params(2, 3, 2, 4)
    piparams = pi_parms(2, 3, 2, 3, 5, 1, 4, 4)
    ctl = MultiPhaseDQCurrentController(piparams, piparams, 1, 2, 3, dparams, dparams)
    mv, freq, idq, mvdq0 = ctl.step(np.random.random(3), np.random.random(3), np.random.random(3))
    assert mv == approx([6.0000000000000036, 4.097999999999995, -1.097999999999999])
    assert freq == approx(4)
    assert idq == approx([0.03786838038188123, 0.4158013084860589, 0.37915362432069233])
    assert mvdq0 == [3, 3, 3]


def test_step3():
    np.random.seed(1)
    dparams = droop_params(2, 3, 2, 4)
    piparams = pi_parms(2, 3, 2, 3, 5, 1, 4, 4)
    ctl = MultiPhaseDQ0PIPIController(piparams, piparams, 3, dparams, dparams)
    mv, cv = ctl.step(np.random.random(3), np.random.random(3), 4, 4)
    assert mv == approx([4.29871026, -0.74270939, 11.44399913])
    assert cv == approx([-0.29270728309561767, -0.2977367776984352, 0.37915362432069233])
