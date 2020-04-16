import logging
from openmodelica_microgrid_gym.auxiliaries import PI_params
import numpy as np

N_phase = 3


class PIController:
    """
    Implements a basic, discrete PI controller.
    Uses back calculation for anti-windup.
    """

    def __init__(self, PI_param: PI_params, ts: float):
        """
        :param PI_param: The PI_Parameters object with the PI controller parameters (kP, kI, kB for the gains of the
            proportional, integral and anti-windup part and the limits of the output)
        :param ts: Sample time
        """
        self._params = PI_param
        self.integralSum = 0
        self.windup_compensation = 0
        self._ts = ts

    def reset(self):
        """
        Resets the Integrator
        """
        self.integralSum = 0

    def step(self, error: float) -> float:
        """
        Implements a step of a basic PI controller with anti-windup by back-calculation

        :param error: Control error to act on
        :return: The calculated PI controller response to the error, using the
                PI_Parameters provided during initialisation, clipped due to the defined limits
        """

        self.integralSum = self.integralSum + (self._params.kI * error + self.windup_compensation) * self._ts
        output = self._params.kP * error + self.integralSum
        clipped = np.clip(output, *self._params.limits)
        self.windup_compensation = (output - clipped) * self._params.kB
        return clipped.squeeze()


class MultiPhasePIController:
    """
    Implements a number of PI controllers for use in multiphase systems
    Number of phases is set to N_phase = 3
    """

    def __init__(self, PI_param: PI_params, ts: float):
        """

        :param PI_param: The PI_Parameters object with the PI controller parameters (kP, kI, kB for the gains of the
            proportional, integral and anti-windup part and the limits of the output)
        :param ts: Sample time
        """
        self.controllers = [PIController(PI_param, ts) for _ in range(N_phase)]

    def reset(self):
        """
        Resets all controllers
        """
        for ctl in self.controllers:
            ctl.reset()

    def step(self, SP: np.ndarray, CV: np.ndarray) -> np.ndarray:
        """
        Performs a controller step calculating the error itself using the array of
        Setpoints (SP) and Controlled Variables (CV, feedback)

        :param SP: Floats of setpoints
        :param CV: Floats of system state to be controlled (feedback)
        
        :return output: An array of the controller outputs.
        """

        error = SP - CV

        if len(error) != len(self.controllers):
            message = f'List of values for error inputs should be of the length {len(self.controllers)}, '
            f'equal to the number of model inputs. Actual length {len(error)}'
            logging.error(message)
            raise ValueError(message)

        # perform all the steps for each phase
        return np.array([ctl.step(error[i]) for i, ctl in enumerate(self.controllers)])
