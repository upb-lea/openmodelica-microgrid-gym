import logging
from typing import List, Mapping, Union

import gym
import numpy as np
from openmodelica_microgrid_gym.util import flatten
from tqdm import trange

from openmodelica_microgrid_gym import Runner, Agent
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.agents.staticctrl import ObsTempl
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams, Controller

logging.basicConfig()

np.random.seed(1)


class NNAgent(Agent):
    """
    This class aims to make modifications in order to improve state-action-space exploration
    """

    def __init__(self, **kwargs):
        self.i = 0
        self.setpoint = np.array([0, 0, 0])
        super().__init__( **kwargs)

    def act(self, state: np.ndarray):
        # set internal integration error state
        for i, ctrl in enumerate(self.controllers.values()):
            for j, pictl in enumerate(ctrl._internalPI.controllers):
                pictl.integralSum = state[i * 3 + j]
        return super().act(state[6:])

    @property
    def obs_template(self):
        if self._obs_template is None:
            self._obs_template = {ctrl: ObsTempl(self.obs_varnames, tmpl)
                                  for ctrl, tmpl in self.obs_template_param.items()}
        return self._obs_template


if __name__ == '__main__':



    # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
    tmpl = dict(master=[[f'lc1.inductor{k}.i' for k in '123'],
                        [f'lc1.capacitor{k}.v' for k in '123']],
                slave=[[f'lcl1.inductor{k}.i' for k in '123'],
                       [f'lcl1.capacitor{k}.v' for k in '123'],
                       ['SPd', 'SPq', 'SP0']])
    agent = NNAgent( obs_varnames=list(flatten(list(tmpl.values()))))
    agent.reset()


    # Define the environment
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   # viz_cols=['*.m[dq0]', 'slave.freq', 'lcl1.*'],
                   log_level=logging.INFO,
                   max_episode_steps=2000,
                   model_params={'inverter1.v_DC': 1000},
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
    runner.run(1, visualise=True)
