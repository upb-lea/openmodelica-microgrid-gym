import logging

delta_t = 1e-4
nomFreq = 50
nomVoltPeak = 230 * 1.414
iLimit = 30
DroopGain = 40000.0  # W/Hz
QDroopGain = 1000.0  # VAR/V

import gym
from gym_microgrid.auxiliaries import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams
from gym_microgrid.agents import StaticControlAgent
from gym_microgrid import Runner

import numpy as np

logging.basicConfig()

if __name__ == '__main__':
    ctrl = dict()

    # Voltage PI parameters for the current sourcing inverter
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    # Current PI parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    ctrl['master'] = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
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
    ctrl['slave'] = MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                                  droop_param, qdroop_param)

    agent = StaticControlAgent(ctrl, {'master': [np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                                 np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)])],
                                      'slave': [np.array([f'lcl1.inductor{i + 1}.i' for i in range(3)]),
                                                np.array([f'lcl1.capacitor{i + 1}.v' for i in range(3)]),
                                                np.zeros(3)]})

    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   viz_cols=['*.m[dq0]', 'slave.freq', 'lcl1.*'],
                   log_level=logging.INFO,
                   model_path='../fmu/grid.network.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={
                       'lc1': [
                           ['inductor1.i', 'inductor2.i', 'inductor3.i'],
                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                       'rl1': [f'inductor{i}.i' for i in range(1, 4)],
                       'lcl1':
                           [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                            ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]},
                   )

    runner = Runner(agent, env)
    runner.run(1, visualise_env=True)
