from typing import Tuple

import numpy as np

from openmodelica_microgrid_gym.aux_ctl.params import PLLParams
from openmodelica_microgrid_gym.aux_ctl.pi_controllers import PIController
from openmodelica_microgrid_gym.util import abc_to_alpha_beta, cos_sin, normalise_abc


class DDS:
    """
    Implements a basic Direct Digital Synthesizer (DDS) controller.
    Basically is a resetting integrator to provide a theta reference at a given
    frequency.
    """

    def __init__(self, ts: float, dds_max: float = 1, theta_0: float = 0):
        """
        :param ts: Sample time
        :param dds_max: The value at which the DDS resets the integrator
        :param theta_0: The initial value of the DDS upon initialisation (not reset)
        """
        self._integralSum = 0
        self._ts = ts
        self._max = dds_max
        self._integralSum = theta_0

    def reset(self):
        """
        Resets the DDS integrator to 0
        """
        self._integralSum = 0

    def step(self, freq: float):
        """
        Advances the Oscilator

        :param freq: Absolute frequency to oscillate at for the next time step

        :return theta: The angle in RADIANS [0:2pi]
        """
        self._integralSum += self._ts * freq

        # reset oscilator one phase
        if self._integralSum > self._max:
            self._integralSum -= self._max

        return self._integralSum * 2 * np.pi


class LimitLoadIntegral:
    def __init__(self, dt, freq, i_lim, i_nom=None):
        """

        :param dt: time resolution in s
        :param freq: nominal frequency, used to calculate timewindow of interest
        :param i_lim: maximum energy allowed over one half frequency time in J
        :param i_nom: nominal energy allowed over one half frequency time in J
        """
        self.dt = dt
        # number of samples for a half freq
        self._buffer = np.empty(int(1/freq / 2 / dt))  # type:np.ndarray
        self._buff_idx = 0
        self.integral = 0
        self.lim_integral = self._buffer.size * self.dt * i_lim ** 2
        if i_nom:
            self.nom_integral = self._buffer.size * self.dt * i_nom
        else:
            self.nom_integral = self._buffer.size * self.dt * i_lim * .9

    def reset(self):
        self.integral = 0
        self._buffer.fill(0)
        self._buff_idx = 0

    def step(self, value):
        self.integral += self.dt * value ** 2
        last = self._buffer[np.ravel_multi_index([self._buff_idx - 1], self._buffer.size, mode='wrap')]
        self.integral -= self.dt * last ** 2
        self._buff_idx = np.ravel_multi_index([self._buff_idx + 1], self._buffer.size, mode='wrap')
        self._buffer[self._buff_idx] = value

    def risk(self):
        return np.clip((self.integral - self.nom_integral) / (self.lim_integral - self.nom_integral), 0, 1)


class PLL:
    """
    Implements a basic PI controller based PLL to track the angle of a three-phase
    ABC voltage
    """

    def __init__(self, params: PLLParams, ts: float):
        """
        :param params: PI Params for controller (kP, kI, limits, kB, f_nom, theta_0)
        :param ts: absolute sampling time for the controller
        """
        self._params = params
        self._controller = PIController(params, ts)

        # Uses a DDS oscillator to keep track of the internal angle
        self._dds = DDS(ts=ts, theta_0=params.theta_0)

        self._prev_cossin = cos_sin(params.theta_0)
        self._sqrt2 = np.sqrt(2)

    def step(self, v_abc: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Performs a discrete set of calculations for the PLL

        :param v_abc: Voltages in the abc frame to track

        :return _prev_cossin: The internal cos-sin of the current phase angle
        :return freq: The frequency of the internal oscillator
        :return theta: The internal phase angle
        """
        v_abc = normalise_abc(v_abc)
        cossin_x = abc_to_alpha_beta(v_abc)
        dphi = self.__phase_comp(cossin_x, self._prev_cossin)
        freq = self._controller.step(dphi) + self._params.f_nom

        theta = self._dds.step(freq)
        self._prev_cossin = cos_sin(theta)

        return self._prev_cossin, freq, theta

    def reset(self):
        self._dds.reset()

    @staticmethod
    def __phase_comp(cos_sin_x: np.ndarray, cos_sin_i: np.ndarray):
        """
        The phase comparison calculation uses sin(A-B)= sin(A)cos(B)-cos(A)sin(B) =  A-B (approximates for small A-B)

        :param cos_sin_x: Alpha-beta components of the external angle to track, should be normalised [-1,1]
        :param cos_sin_i: Alpha-beta components of the internal angle to compare to, should be normalised [-1,1]

        :return dphi: The approximate error between the two phases
        """
        dphi = (cos_sin_x[1] * cos_sin_i[0]) - (cos_sin_x[0] * cos_sin_i[1])

        return dphi
