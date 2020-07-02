#####################################
# Example using a FMU by OpenModelica as gym environment containing two inverters, each connected via an LC-filter to
# supply in parallel a RC load.
# This example uses the available standard controllers as defined in the 'auxiliaries' folder.
# One inverter is set up as voltage forming inverter with a direct droop controller.
# The other controller is used as current sourcing inverter with an inverse droop controller which reacts on the
# frequency and voltage change due to its droop control parameters by a power/reactive power change.

import logging

import gym
import numpy as np

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 3000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes
# (here, only 1 episode makes sense since simulation conditions don't change in this example)
v_DC = 1000  # DC-link voltage / V; will be set as model parameter in the fmu
nomFreq = 50  # grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # current limit / A
iNominal = 20  # nominal current / A
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V

logging.basicConfig()

if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    #####################################
    # Define the voltage forming inverter as master
    # Voltage control PI gain parameters for the voltage sourcing inverter
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    # Current control PI gain parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    # Droop characteristic for the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    # Add to dict
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
                                            qdroop_param, name='master'))

    #####################################
    # Define the current sourcing inverter as slave
    # Current control PI gain parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI gain parameters for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=None, f_nom=nomFreq)
    # Droop characteristic for the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, delta_t, nomFreq, tau_filt=0.04)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(50, delta_t, nomVoltPeak, tau_filt=0.01)
    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                              droop_param, qdroop_param, name='slave'))

    # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
    agent = StaticControlAgent(ctrl, {'master': [[f'lc1.inductor{k}.i' for k in '123'],
                                                 [f'lc1.capacitor{k}.v' for k in '123']],
                                      'slave': [[f'lcl1.inductor{k}.i' for k in '123'],
                                                [f'lcl1.capacitor{k}.v' for k in '123'],
                                                np.zeros(3)]})

    # Define the environment
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   # viz_cols=['*.m[dq0]', 'slave.freq', 'lcl1.*'],
                   log_level=logging.INFO,
                   max_episode_steps=max_episode_steps,
                   model_params={'inverter1.v_DC': v_DC},
                   model_path='../fmu/grid.network.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                     rl1=[f'inductor{i}.i' for i in range(1, 4)],
                                     lcl1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]),
                   )

    # User runner to execute num_episodes-times episodes of the env controlled by the agent
    runner = Runner(agent, env)
    runner.run(num_episodes, visualise=True)
