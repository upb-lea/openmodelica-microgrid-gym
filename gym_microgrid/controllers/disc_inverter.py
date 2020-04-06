from gym_microgrid.env.recorder import SingleHistory
from .filter import DroopController, InverseDroopController
from .base import DDS, PLL
from .pi import MultiPhasePIController
from .params import *
from gym_microgrid.common import *

import numpy as np

N_PHASE = 3


class Controller:
    """
    Base class for all voltage and current controllers
    """

    def __init__(self, IPIParams: PI_params, tau: float, undersampling=1, history=SingleHistory()):
        """

        :type IPIParams: PI_params
        :param IPIParams: PI parameters for the current control loop

        :type tau: float
        :param tau: positive float. absolute sampling time for the controller

        :type undersampling: int
        :param undersampling: reduces the actual sampling time of the controller,
                    for example if set to 10, the controller will only calculate
                    the setpoint every 10th controller call
        :param history:
        """
        self.history = history

        self._ts = tau * undersampling
        self._undersample = undersampling

        self._currentPI = MultiPhasePIController(IPIParams, self._ts)

        # defining memory variables
        self._undersampling_count = None
        self._stored_control = None

    def reset(self):
        """
        Resets the controller to initialization state. Must be called before usage.
        """
        self.history.reset()
        # enforce the first step call to calculate the set point
        self._undersampling_count = self._undersample
        self._stored_control = np.zeros(N_PHASE)

        self._currentPI.reset()

    def step(self, currentCV, voltageCV, *args, **kwargs):
        """
        Will call self.control() with the given *args and **kwargs and handle undersampling.
        The function will replay the last control action for the duration of the undersampling.

        :type currentCV: np.ndarray
        :param currentCV: 1d-array with 3 entries, one for each phase. The feedback values for current

        :type voltageCV: np.ndarray
        :param voltageCV:  1d-array with 3 entries, one for each phase. The feedback values for voltage

        :return: most up to date control action
        """
        self._undersampling_count += 1
        if self._undersampling_count >= self._undersample:
            self._undersampling_count = 0
            self._stored_control = self.control(currentCV, voltageCV, *args, **kwargs)

        return self._stored_control

    def control(self, currentCV: np.ndarray, voltageCV: np.ndarray, idq0SP: np.ndarray = None, *args, **kwargs):
        """
        Performs the calculations for a discrete step of the controller

        :type currentCV: np.ndarray
        :param currentCV: 1d-array with 3 entries, one for each phase. The feedback values for current

        :type voltageCV: np.ndarray
        :param voltageCV:  1d-array with 3 entries, one for each phase. The feedback values for voltage

        :return: The controller output for the current calculation in the ABC
                    frame
        """
        pass


class VoltageCtl(Controller):
    def __init__(self, VPIParams: PI_params, IPIParams: PI_params, tau: float,
                 PdroopParams: DroopParams, QdroopParams: DroopParams,
                 undersampling=1, history=SingleHistory()):
        """

        :type VPIParams: PI_params
        :param VPIParams: PI parameters for the voltage control loop
        """
        super().__init__(IPIParams, tau, undersampling, history)
        self._integralSum = 0

        self._droopController = DroopController(PdroopParams, self._ts)
        self._droopQController = DroopController(QdroopParams, self._ts)

        self._voltagePI = MultiPhasePIController(VPIParams, self._ts)
        self._phaseDDS = DDS(self._ts)

    def reset(self):
        super().reset()
        self._voltagePI.reset()


class CurrentCtl(Controller):
    def __init__(self, IPIParams: PI_params, tau: float, i_limit, Pdroop_param: InverseDroopParams,
                 Qdroop_param: InverseDroopParams, undersampling=1, history=SingleHistory()):
        """

        :param i_limit:
        """
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

    def control(self, currentCV: np.ndarray, voltageCV: np.ndarray, **kwargs):
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
        return self._currentPI.stepSPCV(SPI, currentCV)


class MultiPhaseDQ0PIPIController(VoltageCtl):
    """
    Implements a discrete multiphase PIPI voltage forming control with current 
    limiting. Has its own internal oscillator to keep track of the internal angle
    
    Controls each phase individualy in the dq0 axis.
    """

    def __init__(self, VPIParams, IPIParams, tau, PdroopParams, QdroopParams, undersampling=1,
                 history=SingleHistory()):
        """

        :param VPIParams:
        :param IPIParams:
        :param PdroopParams:
        :param QdroopParams:
        :param history:
        """
        super().__init__(VPIParams, IPIParams, tau, PdroopParams, QdroopParams, undersampling,
                         history)
        self.history.cols = ['phase', [f'SPV{s}' for s in 'dq0'], [f'SPI{s}' for s in 'dq0'], [f'M{s}' for s in 'dq0'],
                             [f'SPI{s}' for s in 'abc']]
        self._prev_CV = np.zeros(N_PHASE)

    def control(self, currentCV, voltageCV, **kwargs):
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
        VSP = voltage
        # Voltage SP in dq0 (static for the moment)
        SPVdq0 = [VSP, 0, 0]
        SPIdq0 = self._voltagePI.stepSPCV(SPVdq0, CVVdq0)

        # SPIdq0 = [15, 0, 0]

        # Current controller calculations
        MVdq0 = self._currentPI.stepSPCV(SPIdq0, CVIdq0)

        # Add intern measurment
        self.history.append([phase, *SPVdq0, *SPIdq0, *MVdq0, *dq0_to_abc(SPIdq0, phase)])

        # Transform the MVs back to the abc frame
        return dq0_to_abc(MVdq0, phase)


class MultiPhaseDQCurrentController(CurrentCtl):
    """
    Implements a discrete 3-phase current sourcing inverter, using a PLL to 
    keep track of the external phase angle
    
    Controls the currents dq0 axis, aligned to the external voltage vector,
    d-axis is aligned with the A phase. Rotating frame aligned with A axis at #
    t = 0, that is, at t = 0, the d-axis is aligned with the a-axis. 
    
    DOES NOT wait for PLL lock before activating
    """

    def __init__(self, IPIParams: PI_params, pllPIParams: PLLParams, tau: float, i_limit,
                 Pdroop_param: InverseDroopParams, Qdroop_param: InverseDroopParams,
                 undersampling=1, history=SingleHistory()):
        """
        :param pllPIParams: PI parameters for the PLL controller
        """
        super().__init__(IPIParams, tau, i_limit, Pdroop_param, Qdroop_param, undersampling, history)
        self.history.cols = ['freq']

        # Three controllers  for each axis (d,q,0)
        self._pll = PLL(pllPIParams, self._ts)

        # Populate the previous values with 0's
        self._prev_cossine = np.zeros(2)
        self._lastIDQ = np.zeros(N_PHASE)
        self._prev_theta = 0
        self._prev_freq = 0

    def control(self, currentCV: np.ndarray, voltageCV: np.ndarray, idq0SP: np.ndarray = np.zeros(3), **kwargs):
        """
        Performs the calculations for a discrete step of the controller

        :param idq0SP: The peak current setpoints in the dq0 frame
        """

        Vinst = inst_rms(voltageCV)
        # Get current phase information from the voltage measurements
        self._prev_cossine, self._prev_freq, self._prev_theta, debug = self._pll.step(voltageCV)

        # Transform the current feedback to the DQ0 frame
        self._lastIDQ = abc_to_dq0_cos_sin(currentCV, *self._prev_cossine)

        droop = np.zeros(2)
        if Vinst > 200:
            # Determine the droop power setpoints
            droopPI = self._droop_control.step(self._prev_freq) / inst_rms(voltageCV)

            # Determine the droop reactive power set points
            droopQI = self._Qdroop_control.step(Vinst) / Vinst
            droop = np.array([droopPI, droopQI])

            droop = droop * 1.4142135623730951  # RMS to Peak
            droop = np.clip(droop, -self._i_limit, self._i_limit)

        idq0SP = idq0SP + np.array([-droop[0], +droop[1], 0])
        # Calculate the control applied to the DQ0 currents
        # action space is limited to [-1,1]

        MVdq0 = self._currentPI.stepSPCV(idq0SP, self._lastIDQ)
        # Transform the outputs from the controllers (dq0) to abc
        # also divide by SQRT(2) to ensure the transform is limited to [-1,1]

        control = dq0_to_abc_cos_sin(MVdq0, *self._prev_cossine)
        self.history.append([self._prev_freq])
        return control
