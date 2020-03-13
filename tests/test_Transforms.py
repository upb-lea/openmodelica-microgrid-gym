from pytest import approx
import numpy as np
from common.Transforms import *


def test_inst_reactive():
    np.random.seed(1)
    assert inst_reactive(np.random.random(3), np.random.random(3)) == approx(-0.14795581512942466)


def test_inst_power():
    np.random.seed(1)
    assert inst_power(np.random.random(3), np.random.random(3)) == approx(0.23180175944821654)


def test_inst_rms():
    np.random.seed(1)
    assert inst_rms(np.random.random(3)) == approx(0.4805464741106228)


def test_dq0_to_abc_cos_sin():
    np.random.seed(1)
    assert dq0_to_abc_cos_sin(np.random.random(3), *np.random.random(2)) == approx([0.02048185, 0.23152558, -0.2516643])
