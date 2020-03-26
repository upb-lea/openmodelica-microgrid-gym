# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:29:23 2020

@author: jarren
"""

import math

from .multiphase_pi_control import MultiPhasePIController
from .pi_control import PIController
from .control_params import *
from gym_microgrid.common import *

import numpy as np


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


class MultiPhaseABCPIPIController:
    """
    Implements a discrete multiphase PIPI voltage forming control with current 
    limiting. Has its own internal oscillator to keep track of the internal angle
    
    Controls each phase individualy in the abc axis.
    """

    def __init__(self, VPIParams, IPIParams, tau, PdroopParams, QdroopParams, undersampling=1, n_phase=3):
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
        self._integralSum = 0
        self._ts = tau * undersampling
        self._undersample_count = 0
        self._undersample = undersampling

        self._droopController = DroopController(PdroopParams, self._ts)
        self._droopQController = DroopController(QdroopParams, self._ts)

        self._currentPI = MultiPhasePIController(IPIParams, self._ts)
        self._voltagePI = MultiPhasePIController(VPIParams, self._ts)
        self._phaseDDS = DDS(self._ts)
        self._undersampling_count = 0
        # Populate the previous MV with n_phase 0's
        self._prev_MV = ([0 for k in range(n_phase)])

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


class MultiPhaseDQ0PIPIController:
    """
    Implements a discrete multiphase PIPI voltage forming control with current 
    limiting. Has its own internal oscillator to keep track of the internal angle
    
    Controls each phase individualy in the dq0 axis.
    """

    def __init__(self, VPIParams, IPIParams, tau, PdroopParams, QdroopParams, undersampling=1, n_phase=3):
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
        self._integralSum = 0
        self._ts = tau * undersampling
        self._undersample = undersampling

        self._droopController = DroopController(PdroopParams, self._ts)
        self._droopQController = DroopController(QdroopParams, self._ts)

        self._currentPI = MultiPhasePIController(IPIParams, self._ts)
        self._voltagePI = MultiPhasePIController(VPIParams, self._ts)
        self._phaseDDS = DDS(self._ts)
        self._undersampling_count = 0
        # Populate the previous MV with n_phase 0's
        self._prev_MV = np.zeros(n_phase)
        self._prev_CV = np.zeros(n_phase)

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
            SPI = self._voltagePI.stepSPCV(SPVdq0, CVVdq0)

            # Current controller calculations
            MVdq0 = self._currentPI.stepSPCV(SPI, CVIdq0)

            # Transform the MVs back to the abc frame
            self._prev_MV = dq0_to_abc(MVdq0, phase)

            # print("SPi: {}, MV: {}".format(SPI,MV))
            self._prev_CV = CVIdq0
            self._undersampling_count = 0
        else:
            self._undersampling_count = self._undersampling_count + 1

        return self._prev_MV, self._prev_CV


class MultiPhaseDQCurrentController:
    """
    Implements a discrete 3-phase current sourcing inverter, using a PLL to 
    keep track of the external phase angle
    
    Controls the currents dq0 axis, aligned to the external voltage vector,
    d-axis is aligned with the A phase. Rotating frame aligned with A axis at #
    t = 0, that is, at t = 0, the d-axis is aligned with the a-axis. 
    
    DOES NOT wait for PLL lock before activating
    """

    def __init__(self, IPIParams, pllPIParams, tau, i_limit, Pdroop_param: InverseDroopParams,
                 Qdroop_param: InverseDroopParams, undersampling=1):
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
        self._ts = tau * undersampling
        self._undersampling_count = 0
        self._undersample = undersampling
        self._i_limit = i_limit

        # Three controllers  for each axis (d,q,0)
        self._currentPI = MultiPhasePIController(IPIParams, self._ts)
        self._pll = PLL(pllPIParams, self._ts)

        # Populate the previous values with 0's
        self._prev_MV = np.zeros(3)
        self._prev_MVdq0 = np.zeros(3)
        self._prev_cossine = np.zeros(2)
        self._lastIDQ = np.zeros(3)
        self._prev_theta = 0
        self._prev_freq = 0
        self._droop_control = InverseDroopController(Pdroop_param, self._ts)
        self._Qdroop_control = InverseDroopController(Qdroop_param, self._ts)

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
                droopPI = droopPI / 1.732050807568877

                droopPI = np.clip(droopPI, -self._i_limit, self._i_limit)

                # Determine the droop reactive power setpoints
                droopQI = self._Qdroop_control.step(Vinst) / Vinst

                # print("droop: {}, Vinst: {}".format(droopModification,Vinst))
                droopQI = droopQI / 1.732050807568877

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
