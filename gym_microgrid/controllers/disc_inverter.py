# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:29:23 2020

@author: jarren
"""

import math

from gym_microgrid.env import EmptyHistory
from .filter import DroopController, InverseDroopController
from .base import DDS, PLL
from .pi import MultiPhasePIController
from .params import *
from gym_microgrid.common import *

import numpy as np

N_PHASE = 3


class Controller:
    def __init__(self, IPIParams, tau, undersampling=1, history=EmptyHistory()):
        self.history = history

        self._ts = tau * undersampling
        self._undersample = undersampling

        self._currentPI = MultiPhasePIController(IPIParams, self._ts)

        # defining memory variables
        self._undersampling_count = None
        self._prev_MV = None

    def reset(self):
        self.history.reset()
        self._undersampling_count = 0
        self._prev_MV = np.zeros(N_PHASE)

        self._currentPI.reset()

    def handle_undersampling(self):
        pass

    def step(self, *args):
        pass


class VoltageCtl(Controller):
    def __init__(self, VPIParams, IPIParams, tau, PdroopParams: DroopParams, QdroopParams: DroopParams, undersampling=1,
                 history=EmptyHistory()):
        super().__init__(IPIParams, tau, undersampling, history)
        self._integralSum = 0

        self._droopController = DroopController(PdroopParams, self._ts)
        self._droopQController = DroopController(QdroopParams, self._ts)

        self._voltagePI = MultiPhasePIController(VPIParams, self._ts)
        self._phaseDDS = DDS(self._ts)
        # Populate the previous MV with n_phase 0's

    def reset(self):
        super().reset()
        self._voltagePI.reset()


class CurrentCtl(Controller):
    def __init__(self, IPIParams, tau, i_limit, Pdroop_param: InverseDroopParams,
                 Qdroop_param: InverseDroopParams, undersampling=1, history=EmptyHistory()):
        super().__init__(IPIParams, tau, undersampling, history)

        self._i_limit = i_limit

        self._droop_control = InverseDroopController(Pdroop_param, self._ts)
        self._Qdroop_control = InverseDroopController(Qdroop_param, self._ts)


class MultiPhaseABCPIPIController(VoltageCtl):
    """
    Implements a discrete multiphase PIPI voltage forming control with current 
    limiting. Has its own internal oscillator to keep track of the internal angle
    
    Controls each phase individualy in the abc axis.
    """

    def step(self, currentCV: np.ndarray, voltageCV: np.ndarray):
        """
        Performs the calculations for a discrete step of the controller
        
        :param currentCV: The feedback values for current
        :param voltageCV: The feedback values for voltage
        
        :param VSP: The peak voltage setpoint
        :param freqSP: the frequency setpoint
        
        :param MV: The controller output for the current calculation in the ABC 
                    frame
        """
        if self._undersampling_count == (self._undersample - 1):

            instPow = -inst_power(voltageCV, currentCV)
            freq = self._droopController.step(instPow)
            # Get the next phase rotation angle to implement
            phase = self._phaseDDS.step(freq)

            instQ = -inst_reactive(voltageCV, currentCV)
            voltage = self._droopQController.step(instQ)

            VSP = voltage * 1.732050807568877
            # Voltage SP in dq0 (static for the moment)
            SPVdq0 = np.array([VSP, 0, 0])

            # Get the voltage SPs in abc vector
            # print("SPVdq0: {}, phase: {}".format(SPVdq0,phase))
            SPV = dq0_to_abc(SPVdq0, phase)

            # print("QInst: {}, Volt {}".format(instQ,VSP))
            SPI = self._voltagePI.stepSPCV(SPV, voltageCV)

            # Average voltages from modulation indices created by current controller
            MV = self._currentPI.stepSPCV(SPI, currentCV)

            # print("SPi: {}, MV: {}".format(SPI,MV))
            self._prev_MV = MV
            self._undersampling_count = 0
        else:
            self._undersampling_count = self._undersampling_count + 1

        return self._prev_MV


class MultiPhaseDQ0PIPIController(VoltageCtl):
    """
    Implements a discrete multiphase PIPI voltage forming control with current 
    limiting. Has its own internal oscillator to keep track of the internal angle
    
    Controls each phase individualy in the dq0 axis.
    """

    def __init__(self, VPIParams, IPIParams, tau, PdroopParams, QdroopParams, undersampling=1,
                 history=EmptyHistory()):
        """
        :param VPIParams: PI parameters for the voltage control loop
        :param IPIParams: PI parameters for the current control loop
        
        :param tau: absolute sampling time for the controller
        :param undersampling: reduces the actual sampling time of the controller,
                    for example if set to 10, the controller will only calculate 
                    the setpoint every 10th controller call
        :param n_phase: The number of individual axis of voltage and current to
                    control
        
        """

        super().__init__(VPIParams, IPIParams, tau, PdroopParams, QdroopParams, undersampling,
                         history)
        self.history.cols = ['phase', 'SPVdq0', 'SPIdq0', 'M_dq0', 'SPIabc']
        self._prev_CV = np.zeros(N_PHASE)

    def step(self, currentCV, voltageCV):
        """
        Performs the calculations for a discrete step of the controller
        
        :param currentCV: The feedback values for current
        :param voltageCV: The feedback values for voltage
        
        :param MV: The controller output for the current calculation in the ABC 
                    frame
        """
        if self._undersampling_count == (self._undersample - 1):

            instPow = -inst_power(voltageCV, currentCV)
            freq = self._droopController.step(instPow)
            # Get the next phase rotation angle to implement
            phase = self._phaseDDS.step(freq)

            instQ = -inst_reactive(voltageCV, currentCV)
            voltage = self._droopQController.step(instQ)

            # Transform the feedback to the dq0 frame
            CVIdq0 = abc_to_dq0(currentCV, phase)
            CVVdq0 = abc_to_dq0(voltageCV, phase)

            # Voltage controller calculations
            VSP = voltage * 1.732050807568877
            # Voltage SP in dq0 (static for the moment)
            SPVdq0 = [VSP, 0, 0]
            SPIdq0 = self._voltagePI.stepSPCV(SPVdq0, CVVdq0)

            SPIdq0 = [15, 0, 0]

            # Current controller calculations
            MVdq0 = self._currentPI.stepSPCV(SPIdq0, CVIdq0)

            # Transform the MVs back to the abc frame
            self._prev_MV = dq0_to_abc(MVdq0, phase)

            # print("SPi: {}, MV: {}".format(SPI,MV))
            self._prev_CV = CVIdq0
            self._undersampling_count = 0

            # Add intern measurment
            self.history.append([phase, SPVdq0, SPIdq0, MVdq0, dq0_to_abc(SPIdq0, phase)])


        else:
            self._undersampling_count = self._undersampling_count + 1

        return self._prev_MV, self._prev_CV


class MultiPhaseDQCurrentController(CurrentCtl):
    """
    Implements a discrete 3-phase current sourcing inverter, using a PLL to 
    keep track of the external phase angle
    
    Controls the currents dq0 axis, aligned to the external voltage vector,
    d-axis is aligned with the A phase. Rotating frame aligned with A axis at #
    t = 0, that is, at t = 0, the d-axis is aligned with the a-axis. 
    
    DOES NOT wait for PLL lock before activating
    """

    def __init__(self, IPIParams, pllPIParams, tau, i_limit, Pdroop_param: InverseDroopParams,
                 Qdroop_param: InverseDroopParams, undersampling=1, history=EmptyHistory()):
        """
        :param IPIParams: PI parameters for the current control loops along the
                        dq0 axes
        :param pllPIParams: PI parameters for the PLL controller
        :param tau: absolute sampling time for the controller
        :param droop_perc: The percentage [0,1] for the droop controller per Hz
        :param undersampling: reduces the actual sampling time of the controller,
                    for example if set to 10, the controller will only calculate 
                    the setpoint every 10th controller call
        """
        super().__init__(IPIParams, tau, i_limit, Pdroop_param, Qdroop_param, undersampling, history)
        self.history.cols = ['phase']

        # Three controllers  for each axis (d,q,0)
        self._pll = PLL(pllPIParams, self._ts)

        # Populate the previous values with 0's
        self._prev_cossine = np.zeros(2)
        self._lastIDQ = np.zeros(N_PHASE)
        self._prev_theta = 0
        self._prev_freq = 0

    def step(self, currentCV, voltageCV, idq0SP: np.ndarray):
        """
        Performs the calculations for a discrete step of the controller
        
        :param currentCV: The feedback values for current
        :param voltageCV: The feedback values for voltage
        
        :param idq0SP: The peak current setpoints in the dq0 frame
        
        :return MV: the controller outputs calculated for the current step
                transformed to the abc reference frame
        :return freq: the PLL determined frequency
        :return IDQ: the feedback currents transformed to the DQ0 axis
        :return MVdq0: the controller outputs in the dq0 axis
        """

        self._undersampling_count += 1
        if self._undersampling_count == self._undersample:
            self._undersampling_count = 0
            Vinst = inst_rms(voltageCV)
            # Get current phase information from the voltage measurements
            self._prev_cossine, self._prev_freq, self._prev_theta, debug = self._pll.step(voltageCV)

            # Transform the current feedback to the DQ0 frame
            self._lastIDQ = abc_to_dq0_cos_sin(currentCV, *self._prev_cossine)

            droopPI = 0
            droopQI = 0
            if Vinst > 200:
                # Determine the droop power setpoints
                droopPI = self._droop_control.step(self._prev_freq) / inst_rms(voltageCV)

                droopPI = droopPI / 1.732050807568877  # * 1.4142135623730951      # RMS to Peak

                droopPI = np.clip(droopPI, -self._i_limit, self._i_limit)

                # Determine the droop reactive power setpoints
                droopQI = self._Qdroop_control.step(Vinst) / Vinst

                # print("droop: {}, Vinst: {}".format(droopModification,Vinst))
                droopQI = droopQI / 1.732050807568877  # * 1.4142135623730951      # RMS to Peak

                droopQI = np.clip(droopQI, -self._i_limit, self._i_limit)

            idq0SP = idq0SP + np.array([-droopPI, +droopQI, 0])
            # Calculate the control applied to the DQ0 currents
            # action space is limited to [-1,1]

            # print("Freq: {}, Volt: {}, idq0sp {}".format(self._prev_freq,Vinst,idq0SP))
            MVdq0 = self._currentPI.stepSPCV(idq0SP, self._lastIDQ)
            # print("SP: {}, act: {}, fb {}".format(idq0SP,MVdq0,self._lastIDQ))
            # Transform the outputs from the controllers (dq0) to abc
            # also divide by SQRT(2) to ensure the transform is limited to [-1,1]
            self._prev_MVdq0 = MVdq0
            self._prev_MV = dq0_to_abc_cos_sin(MVdq0, *self._prev_cossine)
            # print("SP: {}, act: {}, actabc {}".format(idq0SP,MVdq0,self._prev_MV))

        return self._prev_MV, self._prev_freq, self._lastIDQ, self._prev_MVdq0
