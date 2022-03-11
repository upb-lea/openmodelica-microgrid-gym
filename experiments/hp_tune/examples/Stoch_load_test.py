from functools import partial

import gym
import matplotlib.pyplot as plt
import pandas as pd
from stochastic.processes import VasicekProcess

from experiments.hp_tune.env.random_load import RandomLoad
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import RandProcess

load = 1e-6  # 28
upper_bound_load = 1e-3
lower_bound_load = 1e-9
net = Network.load('net/net_vctrl_single_inv.yaml')
max_episode_steps = 10000  # int(2 / net.ts)

if __name__ == '__main__':
    gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=10, vol=1e-40, mean=load), initial=load,
                      bounds=(lower_bound_load, upper_bound_load))

    rand_load = RandomLoad(max_episode_steps, net.ts, gen, bounds=(1e-9, 1e-3), bounds_std=(0, 0))

    R_load = []
    t_vec = []
    t = 0

    for ii in range(200):
        # if ii % 1000 == 0:
        #    gen.reset()

        R_load.append(rand_load.random_load_step(t))

        t += net.ts

        t_vec.append(t)

    plt.plot(t_vec, R_load)
    # plt.ylim([5,20])
    plt.show()

    df = pd.DataFrame(R_load)

    hist = df.hist(bins=100)
    plt.show()
