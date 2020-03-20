import gym
from pytest import approx

from gym_microgrid import Runner
from gym_microgrid.agents import SafeOptAgent
import pandas as pd


def test_main():
    agent = SafeOptAgent()
    env = gym.make('gym_microgrid:JModelicaConvEnv_test-v1',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={
                       'lc1': [
                           ['inductor1.i', 'inductor2.i', 'inductor3.i'],
                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                       'lcl1':
                           [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                            ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})

    runner = Runner(agent, env)
    runner.run(1)
    # env.history.to_hdf('test_main.hd5','hist')
    assert env.history.to_numpy() == approx(pd.read_hdf('test_main.hd5', 'hist').to_numpy())
