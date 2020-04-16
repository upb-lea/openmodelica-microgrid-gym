import pytest
from pytest import approx
from openmodelica_microgrid_gym.common import *


@pytest.fixture
def seed():
    np.random.seed(1)


def test_inst_reactive(seed):
    assert inst_reactive(np.random.random(3), np.random.random(3)) == approx(-0.07421999471678885)


def test_inst_power(seed):
    assert inst_power(np.random.random(3), np.random.random(3)) == approx(0.23180175944821654)


def test_inst_rms(seed):
    assert inst_rms(np.random.random(3)) == approx(0.4805464741106228)


def test_dq0_to_abc(seed):
    assert dq0_to_abc(np.random.random(3), np.random.random(1)) == approx([0.18374714, 0.61133519, -0.79473921])


def test_dq0_to_abc_cos_sin(seed):
    assert dq0_to_abc_cos_sin(np.random.random(3), *np.random.random(2)) == approx([0.02048185, 0.23152558, -0.2516643])


def test_dq0_to_abc_cos_sin_power_inv(seed):
    assert dq0_to_abc_cos_sin_power_inv(np.random.random(3), *np.random.random(2)) == approx(
        [0.025085037842289073, 0.28355976715193565, -0.3082245650813462])


def test_abc_to_dq0(seed):
    assert abc_to_dq0(np.random.random(3), np.random.random(1)) == approx([0.15995477, 0.38566723, 0.37915362432069233])


def test_abc_to_dq0_cos_sin(seed):
    assert abc_to_dq0_cos_sin(np.random.random(3), *np.random.random(2)) == approx([0.07247014, 0.12015287, 0.37915362])


def test_abc_to_alpha_beta(seed):
    assert abc_to_alpha_beta(np.random.random(3)) == approx([0.03786838, 0.41580131])
