from openmodelica_microgrid_gym.auxiliaries import InverseDroopParams


class Filter:
    """
    An empty Filter defining a base interface for any inherenting classes
    """

    def step(self, value):
        pass


class PT1Filter(Filter):
    """
    A PT1 Filter implementation
    """

    def __init__(self, filtParams, ts):
        """
        :type filtParams: DroopParams or InverseDroopParams
        :param filtParams: The filter params
        :type ts: float
        :param ts: Sample time
        """
        self._params = filtParams
        self._integral = 0
        self._ts = ts

    def reset(self):
        """
        Resets the filter Integrator
        """
        self._integral = 0

    def step(self, val_in):
        """
        Implements a first order PT1 filter on the input

        :type val_in: float
        :param val_in: Filter input
        :return: Filtered output
        """

        output = val_in * self._params.gain - self._integral

        if self._params.tau != 0:
            intIn = output / self._params.tau
            self._integral = (self._integral + intIn * self._ts)
            output = self._integral
        elif self._params.gain != 0:
            self._integral = 0
        else:
            output = 0

        return output


class DroopController(PT1Filter):
    """
    Implements a basic first order filter with gain and time constant.
    Uses the PT1 to implement the droop but modifies the gains and outputs as
    required to implement inverter droop

    Ignores the first order element if gain is set to 0, providing a linear gain
    """

    def __init__(self, DroopParams, ts):
        """
        :type DroopParams: DroopParams
        :param DroopParams: The droop parameters (gain, tau, nom_value)
        :type ts: float
        :param ts: Sample time
        """
        self._droopParams = DroopParams
        super().__init__(DroopParams, ts)

    def step(self, val_in):
        """
        Implements a first order response on the input, using the initialised params

        :type val_in: float
        :param val_in: Input - instantaneous power/reactive power
        :return f/V: frequency or voltage, depending on the load and nominal value
        """

        return super().step(val_in) + self._droopParams.nom_val


class InverseDroopController(DroopController):
    """
    Implements an inverse Droop controller. For the use in grid following inverters
    as opposed to grid forming inverters
    Uses the frequency to determine the power output.
    Contains a derivative elements and an input filter.

    Ignores the first order element if gain is set to 0, providing a linear gain
    """

    def __init__(self, DroopParams: InverseDroopParams, ts: float):
        """

        :param DroopParams: The InverseDroopControllerParams for the droop controller
        :param ts: Sample step size
        """
        super().__init__(DroopParams, ts)
        self._params = DroopParams
        self._prev_val = 0
        self._ts = ts
        self._droop_filt = PT1Filter(DroopParams.derivativeFiltParams, ts)

    def step(self, val_in: float):
        """
        Implements a inverse of the first order system

        :param val_in: The result of a first order response to be reversed
        :return f/V: frequency or voltage, depending on the load and nominal value
        """
        val_in = self._droop_filt.step(val_in - self._params.nom_val)

        derivative = (val_in - self._prev_val) / self._ts
        derivative = derivative * self._params.tau

        self._prev_val = val_in
        if self._params.gain != 0:
            output = (val_in / self._params.gain + derivative)
            # print("Inverse val: {}, nom: {}, output: {}".format(val_in,self._params.gain, output))
            return output
        else:
            return 0
