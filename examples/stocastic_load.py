from functools import partial

import gym
import matplotlib.pyplot as plt
from stochastic.processes import VasicekProcess

from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import RandProcess

load = 28
upper_bound_load = 45
lower_bound_load = 11
net = Network.load('../net/net.yaml')


def load_step(t):
    """
    Doubles the load parameters
    :param t:
    :param gain: device parameter
    :return: Dictionary with load parameters
    """
    # Defines a load step after 0.01 s
    if .01 < t <= .01 + net.ts:
        gen.proc.mean = load * 0.55
        gen.reserve = load * 0.55

    return gen.sample(t)


if __name__ == '__main__':
    gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=100, vol=70, mean=load), initial=load,
                      bounds=(lower_bound_load, upper_bound_load))


    def xylables(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{load}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
        plt.title('Load example drawn from Ornstein-Uhlenbeck process \n- Clipping outside the shown y-range')
        plt.legend()
        fig.show()


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   net=net,
                   model_params={'rl1.resistor1.R': load_step,
                                 'rl1.resistor2.R': load_step,
                                 'rl1.resistor3.R': load_step},
                   viz_cols=[
                       PlotTmpl([f'rl1.resistor{i}.R' for i in '123'],
                                callback=xylables
                                )],
                   model_path='../omg_grid/grid.network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        if done:
            break
    env.close()
