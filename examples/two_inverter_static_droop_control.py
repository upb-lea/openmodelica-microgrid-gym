#####################################
# Example using a FMU by OpenModelica as gym environment containing two inverters, each connected via an LC-filter to
# supply in parallel a rc load.
# This example uses the in axillaries available controllers. One inverter is set up as voltage formin inverter with a
# direct droop controller which e.g. frequency drops due to the applied power. The other controller is used as current
# sourcing inverter with an inverse droop controller which reacts on the frequency and voltage change due to its droop
# control parameters by a power/reactive power change.

import logging
import gym
import numpy as np
from openmodelica_microgrid_gym.auxiliaries import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym import Runner

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 300  # number of simulation steps per episode
num_episodes = 11  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 700  # DC-link voltage / V; will be set as model parameter in the fmu
nomFreq = 50  # grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # current limit / A
iNominal = 20  # nominal current / A
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V


logging.basicConfig()

if __name__ == '__main__':
    ctrl = dict()  # Empty dict which shall include all controllers

    #####################################
    # Define voltage forming inverter as master
    # Voltage PI parameters for the current sourcing inverter
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    # Current PI parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    # Droop of the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    # Add to dict
    ctrl['master'] = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
                                                 qdroop_param)

    #####################################
    # Define current sourcing inverter as slave
    # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
    # Current PI parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI params for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop of the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, 0, nomFreq, tau_filt=0.04)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
    # Add to dict
    ctrl['slave'] = MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                                  droop_param, qdroop_param)

    # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
    agent = StaticControlAgent(ctrl, {'master': [np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                                 np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)])],
                                      'slave': [np.array([f'lcl1.inductor{i + 1}.i' for i in range(3)]),
                                                np.array([f'lcl1.capacitor{i + 1}.v' for i in range(3)]),
                                                np.zeros(3)]})

    # Define the environment
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   viz_cols=['*.m[dq0]', 'slave.freq', 'lcl1.*'],
                   log_level=logging.INFO,
                   max_episode_steps=max_episode_steps,
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

    # User runner to execute num_episodes-times episodes of the env controlled by the agent
    runner = Runner(agent, env)
    runner.run(num_episodes, visualise_env=True)
