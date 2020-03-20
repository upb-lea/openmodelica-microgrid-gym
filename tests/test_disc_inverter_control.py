from pytest import approx

from gym_microgrid.controllers import *


def test_step():
    np.random.seed(1)
    dparams = DroopParams(2, 3)
    piparams = PIParams(2, 3, 2, 3)
    ctl = MultiPhaseABCPIPIController(piparams, piparams, 3, dparams, dparams)
    assert ctl.step(np.random.random(3), np.random.random(3), 4, 4) == [2, 2, 2]


def test_step2():
    np.random.seed(1)
    dparams = InverseDroopParams(2, 3)
    piparams = PLLParams(2, 3, 2, 3)
    ctl = MultiPhaseDQCurrentController(piparams, piparams, 1, 2, 3, dparams, dparams)
    mv, freq, idq, mvdq0 = ctl.step(np.random.random(3), np.random.random(3), np.random.random(3))
    assert mv == approx([6.0000000000000036, 4.097999999999995, -1.097999999999999])
    assert freq == approx(3)
    assert idq == approx([0.03786838038188123, 0.4158013084860589, 0.37915362432069233])
    assert mvdq0 == [3, 3, 3]


def test_step3():
    np.random.seed(1)
    dparams = DroopParams(2, 3)
    piparams = PIParams(2, 3, 2, 3)
    ctl = MultiPhaseDQ0PIPIController(piparams, piparams, 3, dparams, dparams)
    mv, cv = ctl.step(np.random.random(3), np.random.random(3), 4, 4)
    assert mv == approx([2.48274095, -0.65484911, 4.17210816])
    assert cv == approx([-0.36169698, -0.20856662, 0.37915362])
