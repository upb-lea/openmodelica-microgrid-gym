class Filter:
    """
    An empty Filter defining a base interface for any inherenting classes
    Mightnot be needed, but my use of Java suggests it may be useful.
    """

    def step(self, value):
        pass


class PT1Filter(Filter):
    """
    A PT1 Filter implementation
    """

    def __init__(self, filtParams, ts):
        """
        :param filtParams: The filter params
        """
        self._params = filtParams
        self._integral = 0
        self._ts = ts

    def step(self, val_in):
        """
        Implements a first order PT1 filter on the input

        :param val_in: new input
        :return omega: The new output
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
        :param Droopparams: The droop params
        """
        self._droopParams = DroopParams
        super().__init__(DroopParams, ts)

    def step(self, val_in):
        """
        Implements a first order response on the input, using the initialised params

        :param val_in: new input
        :return omega: The new setpoint
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

    def __init__(self, DroopParams, ts):
        """
        :param Droopparams: The InverseDroopControllerParams for the droop
        controller
        """
        super().__init__(DroopParams, ts)
        self._params = DroopParams
        self._prev_val = 0
        self._ts = ts
        self._droop_filt = PT1Filter(DroopParams.derivativeFiltParams, ts)

    def step(self, val_in):
        """
        Implements a inverse of the first order system
        :param val_in: The result of a first order response to be reversed

        :return: The new setpoint
        """
        val_in = self._droop_filt.step(val_in - self._params.nom_val)

        derivative = (val_in - self._prev_val) / (self._ts)
        derivative = derivative * self._params.tau

        self._prev_val = val_in
        if self._params.gain != 0:
            output = (val_in / self._params.gain + derivative)
            # print("Inverse val: {}, nom: {}, output: {}".format(val_in,self._params.gain, output))
            return output
        else:
            return 0
