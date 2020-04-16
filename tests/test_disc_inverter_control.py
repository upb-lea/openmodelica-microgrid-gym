import pytest
from pytest import approx

from openmodelica_microgrid_gym.auxiliaries import *
import numpy as np


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
    ctl.reset()
    assert ctl.step(np.random.random(3), np.random.random(3)) == approx([-1.0, -1.0, -1.0])


def test_step2(seed, droop_par, pll_par):
    ctl = MultiPhaseDQCurrentController(pll_par, pll_par, 1, 1, droop_par, droop_par)
    ctl.reset()
    mv = ctl.step(np.random.random(3), np.random.random(3), np.random.random(3))
    assert mv == approx([1.2610252, -1.18258947, 0.18577202])



def test_step3(seed, droop_par, pll_par):
    ctl = MultiPhaseDQ0PIPIController(pll_par, pll_par, 3, droop_par, droop_par)
    ctl.reset()
    mv = ctl.step(np.random.random(3), np.random.random(3))
    assert mv == approx([-2.12585763, -1.17821977, 0.3040774])
