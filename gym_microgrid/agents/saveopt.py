from gym_microgrid.agents import Agent
from gym_microgrid.controllers import PIParams, MultiPhaseDQ0PIPIController, DroopParams, InverseDroopParams, PLLParams, \
    MultiPhaseDQCurrentController

import numpy as np

fcontrol = 1e4
delta_t = 1 / fcontrol
V_dc = 1000
nPhase = 3
nomFreq = 50
nomVoltPeak = 230 * 1.414
iLimit = 30
DroopGain = 40000.0  # W/Hz
QDroopGain = 1000.0  # VAR/V


class SafeOptAgent(Agent):
    def __init__(self):
        super().__init__()
        self.episode_reward = 0

    def reset(self):
        # TODO setup GP model
        # TODO setup the controllers

        # Droop of the active power Watt/Hz, delta_t
        droopParam = DroopParams(DroopGain, 0.005, nomFreq)
        # droopParam= DroopParams(0,0,nomFreq)
        # Droop of the reactive power VAR/Volt Var.s/Volt
        qdroopParam = DroopParams(QDroopGain, 0.002, nomVoltPeak)
        # qdroopParam= DroopParams(0,0,nomVoltPeak)

        # Current PI parameters for the voltage sourcing inverter
        currentDQPIparams = PIParams(kP=0.012, kI=90, uL=1, lL=-1, kB=1)
        # Voltage PI parameters for the current sourcing inverter
        voltageDQPIparams = PIParams(kP=0.025, kI=60, uL=iLimit, lL=-iLimit, kB=1)

        self.controller = MultiPhaseDQ0PIPIController(voltageDQPIparams, currentDQPIparams,
                                                      delta_t, droopParam, qdroopParam,
                                                      undersampling=1, n_phase=3)

        # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
        # Droop of the active power Watts/Hz, W.s/Hz
        droopParam = InverseDroopParams(DroopGain, 0, nomValue=nomFreq, tau_filt=0.04)
        # Droop of the reactive power VAR/Volt Var.s/Volt
        qdroopParam = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
        # qdroopParam=InverseDroopParams(0,0,nomVoltPeak)

        # PI params for the PLL in the current forming inverter
        pllparams = PLLParams(kP=10, kI=200, uL=10000, lL=-10000, kB=1, f_nom=50)

        # Current PI parameters for the current sourcing inverter
        currentDQPIparams = PIParams(kP=0.005, kI=200, uL=1, lL=-1, kB=1)

        self.slave_controller = MultiPhaseDQCurrentController(currentDQPIparams, pllparams,
                                                              delta_t, nomFreq, iLimit, droopParam,
                                                              qdroopParam, undersampling=1)

    def act(self, state: np.ndarray):
        CVI1 = state[0:3]
        CVV1 = state[3:6]
        CVI2 = state[6:9]
        CVV2 = state[9:12]

        mod_indSlave, freq, Idq0, mod_dq0 = self.slave_controller.step(CVI2, CVV2, [0, 0, 0])

        # Perform controller calculations
        mod_ind, CVI1dq = self.controller.step(CVI1, CVV1, nomVoltPeak, nomFreq)
        return np.append(mod_ind, mod_indSlave) * V_dc

    def observe(self, reward, terminated):
        self.episode_reward += reward or 0
        if terminated:
            # safeopt update step
            # TODO
            # reset episode reward
            self.episode_reward = 0
        # on other steps we don't need to do anything

    def render(self):
        # TODO plot the GP
        pass
