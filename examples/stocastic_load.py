import gym
from stochastic.processes import VasicekProcess

from openmodelica_microgrid_gym.util import RandProcess

if __name__ == '__main__':
    gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=50, vol=10, mean=.5), initial=.5, gain=100)

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   net='../net/net.yaml',
                   model_params={'rl1.resistor1.R': gen.sample,
                                 'rl1.resistor2.R': gen.sample,
                                 'rl1.resistor3.R': gen.sample},
                   model_path='../omg_grid/grid.network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        if done:
            break
    env.close()
