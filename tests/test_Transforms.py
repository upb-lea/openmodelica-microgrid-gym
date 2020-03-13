from pytest import approx
import numpy as np
from common.Transforms import *


def test_inst_reactive():
    a = np.array([0.23240663328276812, 0.6399024956598093, 0.018417432066713868])
    b = np.array([0.9806739244887752, 0.9851769442672816, 0.15935207738969825])
    assert inst_reactive(a, b) == approx(-0.38373764942576977)


def test_inst_power():
    a = np.array([0.23240663328276812, 0.6399024956598093, 0.018417432066713868])
    b = np.array([0.9806739244887752, 0.9851769442672816, 0.15935207738969825])
    assert inst_power(a, b) == approx(0.8612671665017887)


def test_inst_rms():
    a = np.array([0.23240663328276812, 0.6399024956598093, 0.018417432066713868])
    assert inst_rms(a) == approx(0.3932036151704845)


def test_dq0_to_abc_cos_sin():
    a = np.array([0.23240663328276812, 0.6399024956598093, 0.018417432066713868])
    assert dq0_to_abc_cos_sin(a, (.5, .6)) == approx(np.array([-0.24932075, 0.55012279, -0.24554974]))
