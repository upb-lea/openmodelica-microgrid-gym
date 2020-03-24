from gym_microgrid.agents import Agent
from gym_microgrid.controllers import PI_params, MultiPhaseDQ0PIPIController, DroopParams, InverseDroopParams, \
    PLLParams, \
    MultiPhaseDQCurrentController

import numpy as np

delta_t = 1e-4
V_dc = 1000
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

        # Voltage PI parameters for the current sourcing inverter
        voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
        # Current PI parameters for the voltage sourcing inverter
        current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
        # Droop of the active power Watt/Hz, delta_t
        droop_param = DroopParams(DroopGain, 0.005, nomFreq)
        # Droop of the reactive power VAR/Volt Var.s/Volt
        qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
        self.controller = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
                                                      qdroop_param)

        # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
        # Current PI parameters for the current sourcing inverter
        current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
        # PI params for the PLL in the current forming inverter
        pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
        # Droop of the active power Watts/Hz, W.s/Hz
        droop_param = InverseDroopParams(DroopGain, 0, nomFreq, tau_filt=0.04)
        # Droop of the reactive power VAR/Volt Var.s/Volt
        qdroop_param = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
        self.slave_controller = MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                                              droop_param, qdroop_param)

    def act(self, state: np.ndarray):
        state = state.reshape((-1, 3))
        mod_ind = self.controller.step(*state[0:2])[0]
        mod_indSlave = self.slave_controller.step(*state[2:4], np.zeros(3))[0]
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
