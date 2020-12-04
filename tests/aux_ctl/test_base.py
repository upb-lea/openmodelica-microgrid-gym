import numpy as np
from pytest import approx

from openmodelica_microgrid_gym.aux_ctl.base import LimitLoadIntegral


def test_limit_load_integral():
    dt = .05
    freq = 2
    i_lim = 5
    i2t = LimitLoadIntegral(dt, freq, i_lim=i_lim, i_nom=i_lim / 5)
    size = int(1 / freq / dt / 2)
    assert len(i2t._buffer) == size

    i2t.reset()
    seq = [5, 4]
    for i in seq:
        i2t.step(i)
    integral = i2t.integral
    assert integral == (np.power(seq, 2) * dt).sum()
    assert i2t.risk() == approx(.3)
    for i in [5, 4, 5, 5, 5, 5, 5, 5] + [0] * size + seq:
        i2t.step(i)
    integral = i2t.integral
    assert integral == (np.power(seq, 2) * dt).sum()
