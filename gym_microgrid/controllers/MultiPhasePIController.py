# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:42:23 2020

@author: jarren
"""
import logging

from .pi_controller import *


class MultiPhasePIController:
    """
    Implements a number of PI controllers for use in multiphase systems
    Essentially a number of identical PI controllers to use in parrallel
    
    """

    def __init__(self, piParams, ts, n_phase=3):
        """
        :params piParams:PI_Parameter object for the PI controllers
        :params n_phase: the number of phases to be controlled
        """

        self.n_phase = n_phase
        self.controllers = []
        for phase in range(self.n_phase):
            self.controllers.append(PIController(piParams, ts))

    def reset(self):
        # Reset all controllers
        for phase in range(self.n_phase):
            self.controllers[phase].reset()

    def step(self, error):
        """
        :params error: List of n_phase errors to be calculated by the controllers
        :returns output: the controller outputs in a list
        """
        # Check if number of error inputs equals number of phases
        if len(error) != self.n_phase:
            message = "List of values for error inputs should be of the length {}," \
                      "equal to the number of model inputs. Actual length {}".format(
                self.n_phase, len(error))
            logging.error(message)
            raise ValueError(message)

        output = []
        # perform all the steps for each phase
        for phase in range(self.n_phase):
            output.append(self.controllers[phase].step(error[phase]))

        return output

    def stepSPCV(self, SP, CV):
        """
        Performs a controller step calculating the error itself using the lists
        Setpoints (SP) and Controlled Variables (CV, feedback)
        
        :params SP: A list of the setpoints
        :params CV: A list of the system state to be controlled (feedback)
        
        :return output: A list of the controller outputs.
        """

        # Check if number of error inputs equals number of phases
        if len(SP) != self.n_phase:
            message = "List of values for SP inputs should be of the length {}," \
                      "equal to the number of model inputs. Actual length {}".format(
                self.n_phase, len(SP))
            logging.error(message)
            raise ValueError(message)
        if len(CV) != self.n_phase:
            message = "List of values for CV inputs should be of the length {}," \
                      "equal to the number of model inputs. Actual length {}".format(
                self.n_phase, len(CV))
            logging.error(message)
            raise ValueError(message)

        error = []

        # calculate the error for each phase
        for phase in range(self.n_phase):
            error.append(SP[phase] - CV[phase])

        # Use the defined step function
        return self.step(error=error)
