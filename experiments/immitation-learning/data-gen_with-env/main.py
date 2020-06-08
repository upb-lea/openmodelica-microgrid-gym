import logging
from typing import List, Mapping, Union

import gym
import numpy as np

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


class PerturbStaticAgent(StaticControlAgent):
    """
    This class aims to make modifications in order to improve state-action-space exploration
    """

    def __init__(self, ctrls: List[Controller], obs_template: Mapping[str, List[Union[List[str], np.ndarray]]],
                 **kwargs):
        self.i = 0
        self.setpoint = np.array([0, 0, 0])
        super().__init__(ctrls, obs_template, **kwargs)

    def act(self, state: np.ndarray):
        self.i = (self.i + 1) % 500
        if self.i == 0:
            self.setpoint = np.full(3, np.random.uniform(0, 5))
        return super().act(np.hstack((self.setpoint, state)))

    @property
    def obs_template(self):
        if self._obs_template is None:
            self._obs_template = {ctrl: ObsTempl(['SPd', 'SPq', 'SP0'] + self.obs_varnames, tmpl)
                                  for ctrl, tmpl in self.obs_template_param.items()}
        return self._obs_template


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
    agent = PerturbStaticAgent(ctrl, {'master': [[f'lc1.inductor{k}.i' for k in '123'],
                                                 [f'lc1.capacitor{k}.v' for k in '123']],
                                      'slave': [[f'lcl1.inductor{k}.i' for k in '123'],
                                                [f'lcl1.capacitor{k}.v' for k in '123'],
                                                ['SPd', 'SPq', 'SP0']]})

    # Define the environment
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   viz_cols='slave.*',
                   log_level=logging.INFO,
                   max_episode_steps=max_episode_steps,
                   model_path='../../../fmu/grid.network.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                     rl1=[f'inductor{i}.i' for i in range(1, 4)],
                                     lcl1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]))

    # User runner to execute num_episodes-times episodes of the env controlled by the agent
    runner = Runner(agent, env)
    runner.run(n_episodes=1, visualise=True)
