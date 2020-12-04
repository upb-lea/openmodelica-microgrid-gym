from typing import List

import numpy as np

from openmodelica_microgrid_gym.util import SingleHistory, EmptyHistory, nested_map, inst_power, inst_reactive, \
    dq0_to_abc, abc_to_dq0, abc_to_dq0_cos_sin, inst_rms, dq0_to_abc_cos_sin
from .base import DDS, PLL
from .droop_controllers import DroopController, InverseDroopController
from .observers import Observer
from .params import *
from .pi_controllers import MultiPhasePIController

N_PHASE = 3


class Controller:
    """
    Base class for all voltage and current controllers
    """

    def __init__(self, IPIParams: PI_params, tau: float, undersampling: int = 1,
                 history: EmptyHistory = None, name=''):
        """

        :param IPIParams: PI parameters for the current control loop
        :param tau: positive float. absolute sampling time for the controller
        :param undersampling: reduces the actual sampling time of the controller,
                    for example if set to 10, the controller will only calculate
                    the setpoint every 10th controller call
        :param history: Dataframe to store internal data
        """
        self.history = history or SingleHistory()
        self._last_meas = []

        self._ts = tau * undersampling
        self._undersample = undersampling

        self._currentPI = MultiPhasePIController(IPIParams, self._ts)

        # defining memory variables
        self._undersampling_count = None
        self._stored_control = None
        self.name = name

    def _set_hist_cols(self, cols):
        """
        prefixes column names with controller name and append a entry for the clipped values
        :param cols:
        :return:
        """
        self.history.cols = nested_map(lambda col: '.'.join([self.name, col]), cols + [f'm{s}_clipped' for s in 'abc'])

    def reset(self):
        """
        Resets the controller to initialization state. Must be called before usage.
        """
        self.history.reset()
        # enforce the first step call to calculate the set point
        self._undersampling_count = 0
        self._stored_control = np.zeros(N_PHASE)

        self._currentPI.reset()

    def step(self):
        """
        Will use precalculated action and handle undersampling.
        The function will replay the last control action for the duration of the undersampling.

        :return: most up to date control action
        """

        return self._stored_control

    def prepare(self, *args, **kwargs):
        """
        Performs the calculations for a discrete step of the controller and stores control response
        """

        if self._undersampling_count == 0:
            abc_action = self.control(*args, **kwargs)
            abc_action = np.clip(abc_action, -1, 1)
            self._last_meas.extend(abc_action)
            self.history.append(self._last_meas)
            self._stored_control = abc_action
        self._undersampling_count = (self._undersampling_count + 1) % self._undersample

    def control(self, *args, **kwargs) -> np.ndarray:
        """
        Performs the calculations for a discrete step of the controller

        :param currentCV: 1d-array with 3 entries, one for each phase. The feedback values for current
        :param voltageCV:  1d-array with 3 entries, one for each phase. The feedback values for voltage
        :param idq0SP:

        :return: The controller output for the current calculation in the ABC frame
        """
        pass


class VoltageCtl(Controller):
    def __init__(self, VPIParams: PI_params, IPIParams: PI_params, tau: float,
                 Pdroop_param: DroopParams, Qdroop_param: DroopParams,
                 undersampling: int = 1, *args, **kwargs):
        """
        Defines the controller for a voltage forming inverter (Master)

        :param VPIParams: PI parameters for the voltage control loop
        :param IPIParams: PI_parameter for the current control loop
        :param tau: sampling time
        :param Pdroop_param: Droop parameters for P-droop
        :param Qdroop_param: Droop parameters for Q-droop
        :param undersampling: Control is applied every undersampling-th cycle
        """
        super().__init__(IPIParams, tau, undersampling, *args, **kwargs)
        self._integralSum = 0

        self._PdroopController = DroopController(Pdroop_param, self._ts)
        self._QdroopController = DroopController(Qdroop_param, self._ts)

        self._voltagePI = MultiPhasePIController(VPIParams, self._ts)
        self._phaseDDS = DDS(self._ts)

    def reset(self):
        super().reset()
        self._voltagePI.reset()
        self._phaseDDS.reset()
        self._PdroopController.reset()
        self._QdroopController.reset()


class CurrentCtl(Controller):
    def __init__(self, IPIParams: PI_params, pllPIParams: PLLParams, tau: float, i_limit: float,
                 Pdroop_param: InverseDroopParams, Qdroop_param: InverseDroopParams,
                 undersampling: int = 1, *args, **kwargs):
        """
        Defines the controller for a current sourcing inverter (Slave)

        :param IPIParams: PI_parameter for the current control loop
        :param pllPIParams: PI parameters for the PLL controller
        :param tau: sampling time
        :param i_limit: Current limit
        :param Pdroop_param: Droop parameters for P-droop
        :param Qdroop_param: Droop parameters for Q-droop
        :param undersampling: Control is applied every undersampling-th cycle
        """
        super().__init__(IPIParams, tau, undersampling, *args, **kwargs)

        self._i_limit = i_limit

        self._PdroopController = InverseDroopController(Pdroop_param, self._ts)
        self._QdroopController = InverseDroopController(Qdroop_param, self._ts)

        # Three controllers  for each axis (d,q,0)
        self._pll = PLL(pllPIParams, self._ts)

    def reset(self):
        super().reset()
        self._pll.reset()
        self._PdroopController.reset()
        self._QdroopController.reset()


class MultiPhaseABCPIPIController(VoltageCtl):
    """
    Implements a discrete multiphase PI-PI voltage forming control with current
    limiting in ABC. Has its own internal oscillator to keep track of the internal angle
    
    Controls each phase individually in the abc axis.
    """

    def control(self, currentCV: np.ndarray, voltageCV: np.ndarray, **kwargs):
        instPow = -inst_power(voltageCV, currentCV)
        freq = self._PdroopController.step(instPow)
        # Get the next phase rotation angle to implement
        phase = self._phaseDDS.step(freq)

        instQ = -inst_reactive(voltageCV, currentCV)
        voltage = self._QdroopController.step(instQ)

        VSP = voltage * 1.732050807568877
        # Voltage SP in dq0 (static for the moment)
        SPVdq0 = np.array([VSP, 0, 0])

        # Get the voltage SPs in abc vector
        # print("SPVdq0: {}, phase: {}".format(SPVdq0,phase))
        SPV = dq0_to_abc(SPVdq0, phase)

        # print("QInst: {}, Volt {}".format(instQ,VSP))
        SPI = self._voltagePI.step(SPV, voltageCV)

        # Average voltages from modulation indices created by current controller
        return self._currentPI.step(SPI, currentCV)


class MultiPhaseDQ0PIPIController(VoltageCtl):
    """
    Implements a discrete multiphase PI-PI voltage forming control with current
    limiting. Has its own internal oscillator to keep track of the internal angle.
    
    Controls each phase individualy in the dq0 axis.
    """

    def __init__(self, VPIParams: PI_params, IPIParams: PI_params, tau: float,
                 Pdroop_param: DroopParams, Qdroop_param: DroopParams, observer: Optional[List[Observer]] = None,
                 undersampling: int = 1, *args, **kwargs):
        """

        :param VPIParams: PI parameters for the voltage control loop
        :param IPIParams: PI_parameter for the current control loop
        :param tau: sampling time
        :param Pdroop_param: Droop parameters for P-droop
        :param Qdroop_param: Droop parameters for Q-droop
        :param undersampling: Control is applied every undersampling-th cycle
        """
        super().__init__(VPIParams, IPIParams, tau, Pdroop_param, Qdroop_param, undersampling,
                         *args, **kwargs)
        self._set_hist_cols(['phase', 'instPow', 'instQ', 'freq',
                             [f'SPV{s}' for s in 'dq0'], [f'SPV{s}' for s in 'abc'],
                             [f'SPI{s}' for s in 'dq0'], [f'SPI{s}' for s in 'abc'],
                             [f'CVV{s}' for s in 'dq0'],
                             [f'CVi{s}' for s in 'dq0'],
                             [f'I_hat{s}' for s in 'dq0'], [f'I_hat{s}' for s in 'abc'],
                             [f'm{s}' for s in 'dq0'], [f'm{s}' for s in 'abc']])

        self._prev_CV = np.zeros(N_PHASE)
        self._lastMabc = np.zeros(N_PHASE)
        self._observer = observer

    def reset(self):
        self._prev_CV = np.zeros(N_PHASE)
        self._lastMabc = np.zeros(N_PHASE)
        super().reset()

    def control(self, currentCV: np.ndarray, voltageCV: np.ndarray, **kwargs):
        """
        Performs the calculations for a discrete step of the controller

        :param currentCV: 1d-array with 3 entries, one for each phase in abc. The feedback values for current
        :param voltageCV:  1d-array with 3 entries, one for each phase in abc. The feedback values for voltage

        :return: Modulation index for the inverter in abc
        """

        instPow = -inst_power(voltageCV, currentCV)
        freq = self._PdroopController.step(instPow)
        # Get the next phase rotation angle to implement
        self.phase = phase = self._phaseDDS.step(freq)

        instQ = -inst_reactive(voltageCV, currentCV)
        voltage = self._QdroopController.step(instQ)

        # Transform the feedback to the dq0 frame
        CVIdq0 = abc_to_dq0(currentCV, phase)
        CVVdq0 = abc_to_dq0(voltageCV, phase)

        # If available, calulate load current using observer
        iabc_out_etimate = np.empty(N_PHASE)
        if self._observer is None:
            iabc_out_etimate = np.zeros(N_PHASE)
        else:
            for j in range(N_PHASE):
                iabc_out_etimate[j] = self._observer[j].cal_estimate(y=[currentCV[j], voltageCV[j]],
                                                                     u=self._lastMabc[j])

        i_dq0_out_estimat = abc_to_dq0(iabc_out_etimate, phase)

        # Voltage controller calculations
        VSP = voltage
        # Voltage SP in dq0 (static for the moment)
        SPVdq0 = np.array([VSP, 0, 0])
        SPVabc = dq0_to_abc(SPVdq0, phase)
        SPIdq0 = self._voltagePI.step(SPVdq0, CVVdq0, i_dq0_out_estimat)

        SPIabc = dq0_to_abc(SPIdq0, phase)

        # Current controller calculations
        MVdq0 = self._currentPI.step(SPIdq0, CVIdq0)
        MVabc = dq0_to_abc(MVdq0, phase)  # Transform the MVs back to the abc frame
        self._lastMabc = MVabc
        # Add intern measurment
        self._last_meas = [phase, instPow, instQ, freq,
                           *SPVdq0, *SPVabc,
                           *SPIdq0, *SPIabc,
                           *CVVdq0,
                           *CVIdq0,
                           *i_dq0_out_estimat, *iabc_out_etimate,
                           *MVdq0, *MVabc]

        return MVabc


class MultiPhaseDQCurrentController(CurrentCtl):
    """
    Implements a discrete 3-phase current sourcing inverter, using a PLL to 
    keep track of the external phase angle
    
    Controls the currents dq0 axis, aligned to the external voltage vector,
    d-axis is aligned with the A phase. Rotating frame aligned with A axis at #
    t = 0, that is, at t = 0, the d-axis is aligned with the a-axis. 
    
    DOES NOT wait for PLL lock before activating
    """

    def __init__(self, IPIParams: PI_params, pllPIParams: PLLParams, tau: float, i_limit: float,
                 Pdroop_param: InverseDroopParams, Qdroop_param: InverseDroopParams,
                 lower_droop_voltage_threshold: float = 150., undersampling: int = 1,
                 *args, **kwargs):
        """
        :param IPIParams: PI_parameter for the current control loop
        :param pllPIParams: PI parameters for the PLL controller
        :param tau: sampling time
        :param i_limit: Current limit
        :param Pdroop_param: Droop parameters for P-droop
        :param Qdroop_param: Droop parameters for Q-droop
        :param lower_droop_voltage_threshold: Grid voltage threshold from where the controller starts to react on the
               voltage and frequency in the grid
        :param undersampling: Control is applied every undersmpling* cycles
                :param history: Dataframe to store internal data
        """
        super().__init__(IPIParams, pllPIParams, tau, i_limit, Pdroop_param, Qdroop_param, undersampling, *args,
                         **kwargs)
        self._set_hist_cols(['phase', 'instPow', 'instQ', 'freq',
                             [f'CVI{s}' for s in 'dq0'],
                             [f'SPI{s}' for s in 'dq0'],
                             [f'm{s}' for s in 'dq0'], [f'm{s}' for s in 'abc']])

        # Populate the previous values with 0's
        self._prev_cossine = np.zeros(2)
        self._lastIDQ = np.zeros(N_PHASE)
        self._prev_theta = 0
        self._prev_freq = 0
        self.lower_droop_voltage_threshold = lower_droop_voltage_threshold

    def control(self, currentCV: np.ndarray, voltageCV: np.ndarray, idq0SP: np.ndarray = np.zeros(3), **kwargs):
        """
        Performs the calculations for a discrete step of the controller
        Droop-control is started if Vgrid_rms exceeds 200 V, to avoid oscillation with the grid forming inverter

        :param currentCV: 1d-array with 3 entries, one for each phase. The feedback values for current
        :param voltageCV:  1d-array with 3 entries, one for each phase. The feedback values for voltage
        :param idq0SP: The peak current setpoints in the dq0 frame (Additional power output to droop, if == 0, than
            only droop is applied

        :return: Modulation indices for the current sourcing inverter in ABC
        """
        # Calulate P&Q for slave
        instPow = -inst_power(voltageCV, currentCV)
        instQ = -inst_reactive(voltageCV, currentCV)

        Vinst = inst_rms(voltageCV)
        # Get current phase information from the voltage measurement
        self._prev_cossine, self._prev_freq, self._prev_theta = self._pll.step(voltageCV)

        # Transform the current feedback to the DQ0 frame
        self._lastIDQ = abc_to_dq0_cos_sin(currentCV, *self._prev_cossine)

        droop = np.zeros(2)
        if Vinst > self.lower_droop_voltage_threshold:
            # Determine the droop power setpoints
            droopPI = (self._PdroopController.step(self._prev_freq) / inst_rms(voltageCV))

            # Determine the droop reactive power set points
            droopQI = (self._QdroopController.step(Vinst) / Vinst)
            droop = np.array([droopPI, droopQI]) / 3  # devide by 3 due to here dq, but we want to set the e.g.
            # d-setpoint in the sum of the phases abc

            droop = droop * 1.4142135623730951  # RMS to Peak
            droop = np.clip(droop, -self._i_limit, self._i_limit)

        idq0SP = idq0SP + np.array([-droop[0], +droop[1], 0])
        # Calculate the control applied to the DQ0 currents
        # action space is limited to [-1,1]

        MVdq0 = self._currentPI.step(idq0SP, self._lastIDQ)
        # Transform the outputs from the controllers (dq0) to abc
        # also divide by SQRT(2) to ensure the transform is limited to [-1,1]

        MVabc = dq0_to_abc_cos_sin(MVdq0, *self._prev_cossine)
        self._last_meas = [self._prev_theta, instPow, instQ, self._prev_freq,
                           *self._lastIDQ,
                           *idq0SP,
                           *MVdq0, *MVabc]
        return MVabc


class MultiPhaseDQCurrentSourcingController(Controller):
    """
    Implements a discrete multiphase PI current controller to supply an amount of current to a load / the grid which can
    be chosen via state"extension" using idq0SP at a frequency of f_nom.
    Has its own internal oscillator to keep track of the internal angle.

    Controls each phase individually in the dq0 axis.
    """

    def __init__(self, IPIParams: PI_params, tau: float, f_nom: float = 50.0,
                 undersampling: int = 1, *args, **kwargs):
        """
        :param IPIParams: PI_parameter for the current control loop
        :param tau: sampling time
        :param f_nom: nominal frequency for current sourcing ctrl
        :param undersampling: Control is applied every undersmpling* cycles
        """
        super().__init__(IPIParams, tau, undersampling, *args, **kwargs)
        self._phaseDDS = DDS(self._ts)
        self.f_nom = f_nom
        self._set_hist_cols(['phase',
                             [f'CVI{s}' for s in 'dq0'],
                             [f'SPI{s}' for s in 'dq0'], [f'SPI{s}' for s in 'abc'],
                             [f'm{s}' for s in 'dq0'], [f'm{s}' for s in 'abc']])

        self._prev_CV = np.zeros(N_PHASE)

    def reset(self):
        super().reset()
        self._phaseDDS.reset()

    def control(self, currentCV: np.ndarray, idq0SP: np.ndarray = np.zeros(3), **kwargs):
        """
        Performs the calculations for a discrete step of the controller

        :param currentCV: 1d-array with 3 entries, one for each phase in abc. The feedback values for current
        :param idq0SP: 1d-array with 3 entries, one for each phase in dq0. Current set points for current sourcing ctrl

        :return: Modulation index for the inverter in abc
        """

        # Get the next phase rotation angle to implement
        phase = self._phaseDDS.step(self.f_nom)

        # Transform the feedback to the dq0 frame
        CVIdq0 = abc_to_dq0(currentCV, phase)

        # current setpoint
        SPIdq0 = np.array(idq0SP[:])
        SPIabc = dq0_to_abc(SPIdq0, phase)
        # Current controller calculations
        MVdq0 = self._currentPI.step(SPIdq0, CVIdq0)
        MVabc = dq0_to_abc(MVdq0, phase)

        # Add intern measurment
        self._last_meas = [phase,
                           *CVIdq0,
                           *SPIdq0, *SPIabc,
                           *MVdq0, *MVabc]

        # Transform the MVs back to the abc frame
        return MVabc
