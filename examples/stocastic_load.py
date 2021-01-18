import gym
import matplotlib.pyplot as plt
from stochastic.processes import VasicekProcess

from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import RandProcess

load = 20
upper_bound_load = 40
lower_bound_load = 0

if __name__ == '__main__':
    gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=10, vol=10, mean=load), initial=load,
                      bounds=(lower_bound_load, upper_bound_load))


    def xylables(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{load}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        ax.set_ylim([lower_bound_load, upper_bound_load])
        plt.title('Load example drawn from Ornstein-Uhlenbeck process \n- Clipping outside the shown y-range')
        fig.show()


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   net='../net/net.yaml',
                   model_params={'rl1.resistor1.R': gen.sample,
                                 'rl1.resistor2.R': gen.sample,
                                 'rl1.resistor3.R': gen.sample},
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
