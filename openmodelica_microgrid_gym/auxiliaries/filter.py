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


