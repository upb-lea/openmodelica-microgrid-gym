import gym
from stochastic.processes import OrnsteinUhlenbeckProcess

from stochastic.processes.continuous import BrownianMotion
import matplotlib.pyplot as plt
from stochastic.processes.continuous import WienerProcess

proc = OrnsteinUhlenbeckProcess()
s = proc.sample(32)
times = proc.times(32)

plt.plot(times, s)
plt.show()

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   is_normalized=True,
                   net='../net/net.yaml',
                   model_path='../omg_grid/grid.network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        if done:
            break
    env.close()
