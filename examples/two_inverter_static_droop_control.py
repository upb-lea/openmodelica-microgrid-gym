#####################################
# Example using a FMU by OpenModelica as gym environment containing two inverters, each connected via an LC-filter to
# supply in parallel a RL load.
# This example uses the available standard controllers as defined in the 'auxiliaries' folder.
# One inverter is set up as voltage forming inverter with a direct droop controller.
# The other controller is used as current sourcing inverter with an inverse droop controller which reacts on the
# frequency and voltage change due to its droop control parameters by a power/reactive power change.

import logging
from functools import partial

import gym
import numpy as np

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams
from openmodelica_microgrid_gym.net import Network

# Simulation definitions
max_episode_steps = 10000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes
# (here, only 1 episode makes sense since simulation conditions don't change in this example)
DroopGain = 8000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 2000.0  # virtual droop gain for reactive power / VAR/V
net = Network.load('../experiments/swing_equation/microgrid.yaml')
delta_t = net.ts  # simulation time step size / s
freq_nom = net.freq_nom  # nominal grid frequency / Hz
v_nom = net.v_nom  # nominal grid voltage / V
v_DC = net['inverter1'].v_DC  # DC-link voltage / V; will be set as model parameter in the FMU
i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A

logging.basicConfig()


def load_step(t, gain):
    """
    Defines a load step after 0.25 s
    Doubles the load parameters
    :param t:
    :param gain: device parameter
    :return: Dictionary with load parameters
    """
    return 1 * gain if t < .25 else gain / 2


if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    #####################################
    # Define the voltage forming inverter as master
    # Voltage control PI gain parameters for the voltage sourcing inverter
    voltage_dqp_iparams = PI_params(kP=0.025, kI=40, limits=(-i_lim, i_lim))
    # Current control PI gain parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=10, limits=(-1, 1))
    # Droop characteristic for the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, delta_t, freq_nom)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, delta_t, v_nom)
    # Add to dict
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param,
                                            qdroop_param, ts_sim=delta_t, name='master'))

    #####################################
    # Define the current sourcing inverter as slave
    # Current control PI gain parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.1, kI=200, limits=(-1, 1))
    # PI gain parameters for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=None, f_nom=freq_nom)
    # Droop characteristic for the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, delta_t, freq_nom, tau_filt=0.04)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(QDroopGain, delta_t, v_nom, tau_filt=0.01)
    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, i_lim,
                                              droop_param, qdroop_param, ts_sim=delta_t, name='slave'))

    # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
    agent = StaticControlAgent(ctrl, {'master': [[f'lc1.inductor{k}.i' for k in '123'],
                                                 [f'lc1.capacitor{k}.v' for k in '123']],
                                      'slave': [[f'lc2.inductor{k}.i' for k in '123'],
                                                [f'lc2.capacitor{k}.v' for k in '123'],
                                                np.zeros(3)]})

    # Define the environment
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   # viz_cols=['*.m[dq0]', 'slave.freq', 'lcl1.*'],
                   viz_cols=['master.inst*', 'slave.inst*', 'lc1.*', 'slave.freq', 'lc2.*', 'rl1.*', 'master.freq'],
                   log_level=logging.INFO,
                   max_episode_steps=max_episode_steps,
                   model_params={'rl1.resistor1.R': partial(load_step, gain=6.22),
                                 'rl1.resistor2.R': partial(load_step, gain=6.22),
                                 'rl1.resistor3.R': partial(load_step, gain=6.22),
                                 'rl1.inductor1.L': 0.00495,
                                 'rl1.inductor2.L': 0.00495,
                                 'rl1.inductor3.L': 0.00495
                                 },
                   model_path='../omg_grid/grid.microgrid.fmu',
                   net=net
                   )

    # User runner to execute num_episodes-times episodes of the env controlled by the agent
    runner = Runner(agent, env)
    runner.run(num_episodes, visualise=True)
