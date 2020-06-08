import logging
from typing import List, Mapping, Union

import gym
import numpy as np
from openmodelica_microgrid_gym.util import flatten
from tqdm import trange

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.agents.staticctrl import ObsTempl
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams, Controller

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 3000  # number of simulation steps per episode
# (here, only 1 episode makes sense since simulation conditions don't change in this example)
iLimit = 30  # current limit / A
iNominal = 20  # nominal current / A
nom_freq = 50
nomVoltPeak = 230 * 1.414
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V

logging.basicConfig()

np.random.seed(1)


if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    droop_param = DroopParams(DroopGain, 0.005, nom_freq)
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    # Add to dict
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
                                            qdroop_param, name='master'))

    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000))
    droop_param = InverseDroopParams(DroopGain, delta_t, nom_freq, tau_filt=0.04)
    qdroop_param = InverseDroopParams(50, delta_t, nomVoltPeak, tau_filt=0.01)
    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                              droop_param, qdroop_param, name='slave'))

    # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
    tmpl = dict(master=[[f'lc1.inductor{k}.i' for k in '123'],
                        [f'lc1.capacitor{k}.v' for k in '123']],
                slave=[[f'lcl1.inductor{k}.i' for k in '123'],
                       [f'lcl1.capacitor{k}.v' for k in '123'],
                       ['SPd', 'SPq', 'SP0']])

    agent = StaticControlAgent(ctrl, tmpl, obs_varnames=list(flatten(list(tmpl.values()))))
    agent.reset()

    N = 10 ** 6

    # inputs that are later of the NN. This includes the real state, but also the augmentation data that the agent class would generate
    # e.g. integration error
    X = np.random.normal(0, 10, (N, 15))
    y = []
    for i in trange(N):
        y.append(agent.act(X[i]))
    y = np.vstack(y)

    # shuffle
    perm = np.random.permutation(X.shape[0])
    np.take(X, perm, axis=0, out=X)
    np.take(y, perm, axis=0, out=y)

    # save
    np.save("seqX", X)
    np.save("seqy", y)
