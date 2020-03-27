import math

from gym_microgrid.common import inst_rms, abc_to_alpha_beta, cos_sin
from gym_microgrid.controllers import PLLParams
from gym_microgrid.controllers.pi import PIController


class DDS:
    """
    Implements a basic Direct Digital Synthesizer (DDS) controller.
    Basically is a resetting integrator to provide a theta reference at a given
    frequency
    """

    def __init__(self, ts, DDSMax: float = 1, theta_0: float = 0):
        """
        :param tau: the constant timestep at which the DDS is called
        :param DDSMax: The value at which the DDS resets the integrator
        :param theta_0: The initial value of the DDS upon initialisation (not reset)
        """
        self._integralSum = 0
        self._ts = ts
        self._max = DDSMax
        self._integralSum = theta_0

    def reset(self):
        """
        Resets the DDS integrator to 0
        """
        self._integralSum = 0

    def step(self, freq):
        """
        advances the Oscilator

        :param freq: absolute frequency to oscilate at for the next time step

        :return theta: the angle in RADIANS [0:2pi]
        """
        self._integralSum = self._integralSum + self._ts * freq

        # Limit output to exactly the limit
        if self._integralSum > self._max:
            self._integralSum = self._integralSum - self._max

        return self._integralSum * 2 * math.pi


class PLL:
    """
    Implements a basic PI controller based PLL to track the angle of a threephase
    ABC voltage
    """

    def __init__(self, params: PLLParams, ts):
        """
        :param params:PI Params for controller
        :param tau: absolute sampling time for the controller
        """
        self._params = params
        self._controller = PIController(params, ts)

        # Uses a DDS oscillator to keep track of the internal angle
        self._dds = DDS(ts, params.theta_0)

        self._prev_cossin = cos_sin(params.theta_0)
        self._sqrt2 = math.sqrt(2)

    def step(self, v_abc):
        """
        Performs a discrete set of calculations for the PLL
        :param v_abc: Voltages in the abc frame to track

        :return _prev_cossin: the internal cos-sin of the current phase angle
        :return freq: the frequency of the internal oscillator
        :return theta: the internal phase angle

        """
        v_abc = self.__normalise_abc(v_abc)
        cossin_x = abc_to_alpha_beta(v_abc)
        dphi = self.__phase_comp(cossin_x, self._prev_cossin)
        freq = self._controller.step(dphi) + self._params.f_nom

        theta = self._dds.step(freq)
        self._prev_cossin = cos_sin(theta)

        # debug vector that can be returned for debugging purposes
        debug = [*self._prev_cossin, *cossin_x, theta]

        return self._prev_cossin, freq, theta, debug

    @staticmethod
    def __phase_comp(cossin_x, cossin_i):
        """
        The phase comparison calculation
        uses sin(A-B)= sinAcosB-cosAsinB =  A-B (approximates for small A-B)
        :param cossin_x: Alpha-beta components of the external angle to track,
                    should be normalised [-1,1]
        :param cossin_i: Alpha-beta components of the internal angle to compare
                    to, should be normalised [-1,1]

        :return dphi: The approximate error between the two phases
        """
        dphi = (cossin_x[1] * cossin_i[0]) - (cossin_x[0] * cossin_i[1])

        return dphi

    def __normalise_abc(self, abc):
        """
        Normalises the abc magnitudes to the RMS of the 3 magnitudes
        Determines the instantaneous RMS value of the 3 waveforms
        :param abc: three phase magnitudes input

        :return abc_norm: abc result normalised to [-1,1]
        """
        # Get the magnitude of the waveforms to normalise the PLL calcs
        mag = inst_rms(abc)
        if mag != 0:
            abc = abc / mag

        return abc
