"""
The parameter classes wrap controller parameters.
The fields are wrapped into properties in order to allow transparent usage of the MutableFloat wrapper

"""

from typing import Tuple, Union

from openmodelica_microgrid_gym.agents.util import MutableFloat


class FilterParams:

    def __init__(self, gain: Union[MutableFloat, float], tau: Union[MutableFloat, float]):
        """
        Defines Filter Parameters

        :param gain: Filter gain
        :param tau: Filter time constant
        """
        self._gain = gain
        self._tau = tau

    @property
    def gain(self):
        return float(self._gain)

    @property
    def tau(self):
        return float(self._tau)


class DroopParams(FilterParams):
    """
    Defines droop parameters needed for the droop-controller for a voltage forming inverter
    """

    def __init__(self, gain: Union[MutableFloat, float], tau: Union[MutableFloat, float], nom_value: float = 0):
        """
        e.g. for a P-f droop controller (for voltage forming inverter)
            Inverter of 10 kW, droop of 10% , tau of 1 sec, 50 Hz
            Droop = gain = 1000 [W/Hz]
            tau = 1
            nomValue = 50 [Hz]

        :param gain: The droop gain [W/Hz or VA/V], gets inverted
        :param tau: The first order time constant [s]
        :param nom_value: An offset to add to the output of the droop (e.g. f = 50 Hz)
        """
        super().__init__(gain, tau)
        self.nom_val = nom_value

    @property
    def gain(self):
        if float(self._gain) != 0:
            return 1 / float(self._gain)
        return 0


class InverseDroopParams(DroopParams):
    """
    Defines droop parameters needed for the droop-controller for a current sourcing inverter
    """

    def __init__(self, droop: Union[MutableFloat, float], tau: Union[MutableFloat, float], nom_value: float = 0,
                 tau_filt: Union[MutableFloat, float] = 0):
        """
        e.g. for a f-P droop controller (for current sourcing inverter)
            Inverter of 10 kW, droop of 10% , tau of 1 sec, 50 Hz
            Droop = gain = 1000 [W/Hz]
            tau = 1
            nomValue = 50 [Hz]
        :param droop: The droop gain [W/Hz or VA/V] - Defines the power output due to the frequency/voltage change from
                      nom_val
        :param tau: The first order time constant [s]
        :param nom_value: An offset to add to the output of the droop (e.g. f = 50 Hz)
        :param tau_filt: timeresolution for filter
        """

        super().__init__(droop, tau, nom_value)
        self.derivativeFiltParams = FilterParams(1, tau_filt)


class PI_params:
    """
    The params for a basic PI Controller
    All fields are represented by properties to allow passing MutableFloats
    """

    def __init__(self, kP: Union[MutableFloat, float], kI: Union[MutableFloat, float],
                 limits: Union[Tuple[MutableFloat, MutableFloat], Tuple[float, float]], kB: float = 1):
        """
        :param kP: Proportional gain
        :param kI: Intergral gain
        :param limits: Controller limits
        :param kB: Anti-windup (back calculation)
        """
        self._kP = kP
        self._kI = kI
        self._limits = limits
        self._kB = kB

    @property
    def kP(self):
        return float(self._kP)

    @property
    def kI(self):
        return float(self._kI)

    @property
    def limits(self):
        return [float(limit) for limit in self._limits]

    @property
    def kB(self):
        return float(self._kB)


class PLLParams(PI_params):
    """
    The params for a Phase Lock Loop (PLL) to measure the frequency
    """

    def __init__(self, kP: Union[MutableFloat, float], kI: Union[MutableFloat, float],
                 limits: Union[Tuple[MutableFloat, MutableFloat], Tuple[float, float]],
                 kB: Union[MutableFloat, float] = 1, f_nom: float = 0, theta_0: float = 0):
        """
        :param kP: Proportional gain
        :param kI: Intergral gain
        :param limits: Controller limits
        :param kB: Anti-windup (back calculation)
        :param f_nom: Nominal grid frequency to track (e.g. 50 Hz)
        :param theta_0: Inital angle
        """
        super().__init__(kP, kI, limits, kB)
        self._f_nom = f_nom
        self._theta_0 = theta_0

    @property
    def f_nom(self):
        return float(self._f_nom)

    @property
    def theta_0(self):
        return float(self._theta_0)
