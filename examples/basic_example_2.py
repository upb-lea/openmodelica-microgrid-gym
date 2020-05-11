import gym
import numpy as np
import pandas as pd
from openmodelica_microgrid_gym import Agent, Runner

inputs = ['i1p1', 'i1p2', 'i1p3']

class RndAgent(Agent):
    def act(self, obs: pd.Series) -> np.ndarray:
        return np.random.random(len(inputs))


if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   model_input=inputs,
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../fmu/grid.network.fmu')

    agent = RndAgent()
    runner = Runner(agent, env)

    runner.run(1)