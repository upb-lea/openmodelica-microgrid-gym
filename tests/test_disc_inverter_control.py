import pytest
from pytest import approx

from gym_microgrid.controllers import *


@pytest.fixture
def seed():
    np.random.seed(1)


@pytest.fixture(scope='module')
def droop_par():
    return InverseDroopParams(1, 4)


@pytest.fixture(scope='module')
def pll_par():
    return PLLParams(2, 3, (-1, 1))


def test_step(seed, pll_par, droop_par):
    ctl = MultiPhaseABCPIPIController(pll_par, pll_par, 1, droop_par, droop_par)
    assert ctl.step(np.random.random(3), np.random.random(3)) == approx([-1.0, -1.0, -1.0])


def test_step2(seed, droop_par, pll_par):
    ctl = MultiPhaseDQCurrentController(pll_par, pll_par, 1, 1, droop_par, droop_par)
    mv, freq, idq, mvdq0 = ctl.step(np.random.random(3), np.random.random(3), np.random.random(3))
    assert mv == approx([1.2610252, -1.18258947, 0.18577202])
    assert freq == approx(0.7806602122646245)
    assert idq == approx([-0.40085954, 0.11677488, 0.37915362])
    assert mvdq0 == approx([1.0, 1.0, 0.08806924954988804])


def test_step3(seed, droop_par, pll_par):
    ctl = MultiPhaseDQ0PIPIController(pll_par, pll_par, 3, droop_par, droop_par)
    mv, cv = ctl.step(np.random.random(3), np.random.random(3))
    assert mv == approx([-2.12585763, -1.17821977, 0.3040774])
    assert cv == approx([0.01861732, -0.41710687, 0.37915362])
