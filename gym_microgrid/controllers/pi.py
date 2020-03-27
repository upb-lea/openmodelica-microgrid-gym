import logging
from gym_microgrid.controllers import PI_params
import numpy as np

N_phase = 3


class PIController:
    """
    Implements a basic PI controller.
    Uses back calculation for anti-windup
    """

    def __init__(self, PI_param: PI_params, ts):
        """
        :param PI_param: The PI_Parameters object with the PI controller
        parameters
        """
        self._params = PI_param
        self.integralSum = 0
        self.windup_compensation = 0
        self._ts = ts

    def reset(self):
        self.integralSum = 0
        self._subsample_count = 0

    def step(self, error):
        """
        implements a step of a basic PI controller with anti-windup by back-calculation

        :param error: control error to act on
        :return: the calculated PI controller response to the error, using the
                PI_Parameters provided during initialisation.
        """

        self.integralSum = self.integralSum + (self._params.kI * error + self.windup_compensation) * self._ts
        output = self._params.kP * error + self.integralSum
        clipped = np.clip(output, *self._params.limits)
        self.windup_compensation = (output - clipped) * self._params.kB
        return clipped


class MultiPhasePIController:
    """
    Implements a number of PI controllers for use in multiphase systems
    Essentially a number of identical PI controllers to use in parrallel
    
    """

    def __init__(self, piParams, ts):
        """
        :params piParams:PI_Parameter object for the PI controllers
        :params n_phase: the number of phases to be controlled
        """
        self.controllers = [PIController(piParams, ts) for _ in range(N_phase)]

    def reset(self):
        # Reset all controllers
        for ctl in self.controllers:
            ctl.reset()

    def step(self, error):
        """
        :params error: List of n_phase errors to be calculated by the controllers
        :returns output: the controller outputs in a list
        """
        # Check if number of error inputs equals number of phases
        if len(error) != len(self.controllers):
            message = 'List of values for error inputs should be of the length {},' \
                      'equal to the number of model inputs. Actual length {}'.format(
                len(self.controllers), len(error))
            logging.error(message)
            raise ValueError(message)

        # perform all the steps for each phase
        return [ctl.step(error[i]) for i, ctl in enumerate(self.controllers)]

    def stepSPCV(self, SP: np.ndarray, CV: np.ndarray):
        """
        Performs a controller step calculating the error itself using the lists
        Setpoints (SP) and Controlled Variables (CV, feedback)
        
        :params SP: A list of the setpoints
        :params CV: A list of the system state to be controlled (feedback)
        
        :return output: A list of the controller outputs.
        """
        return self.step(SP - CV)
