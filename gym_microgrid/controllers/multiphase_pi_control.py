# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:42:23 2020

@author: jarren
"""
import logging
from itertools import repeat

from .pi_control import *


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
        self.controllers = [PIController(piParams, ts) for _ in range(n_phase)]

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
