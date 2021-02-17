from datetime import datetime, time
from functools import partial
from itertools import accumulate
from os import makedirs

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from stochastic.processes import VasicekProcess

from experiments.hp_tune.env.random_load import RandomLoad
from experiments.hp_tune.env.rewards import Reward
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network

from openmodelica_microgrid_gym.util import RandProcess
from gym.envs.registration import register

folder_name = 'DDPG_VC_test1/'
# experiment_name = 'DDPG_VC_Reward_MRE_reward_NOT_NORMED'
experiment_name = 'plots'
timestamp = datetime.now().strftime(f'_%Y.%b.%d_%X')

makedirs(folder_name, exist_ok=True)
makedirs(folder_name + experiment_name, exist_ok=True)

# toDo: give net and params via config from mail script

# Simulation definitions
net = Network.load('net/net_vctrl_single_inv.yaml')
max_episode_steps = 1000  # net.max_episode_steps  # number of simulation steps per episode

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC
# plant

# toDo: shift this to net?!
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F
R = 28  # nomVoltPeak / 7.5   # / Ohm
lower_bound_load = 11  # to allow maximal load that draws i_limit (toDo: let exceed?)
upper_bound_load = 45  # to apply symmetrical load bounds

loadstep_timestep = max_episode_steps / 2

gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=1000, vol=10, mean=R), initial=R,
                  bounds=(lower_bound_load, upper_bound_load))


class CallbackList(list):
    def fire(self, *args, **kwargs):
        for listener in self:
            listener(*args, **kwargs)


def xylables_i(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    ax.grid(which='both')
    # fig.savefig(f'{folder_name + experiment_name + n_trail}/Inductor_currents.pdf')
    plt.close()


def xylables_v(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
    ax.grid(which='both')
    # ax.set_xlim([0, 0.005])
    ts = time.gmtime()
    fig.savefig(
        f'{folder_name + experiment_name}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    ts = time.gmtime()
    fig.savefig(f'{folder_name + experiment_name}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


# rew = Reward(v_nom, v_lim, v_DC, gamma, use_gamma_in_rew)
rand_load = RandomLoad(max_episode_steps, net.ts, gen)

cb = CallbackList()
cb.append(partial(gen.reset, initial=R))
cb.append(rand_load.reset)

register(id='vctrl_single_inv-v0',
         entry_point='openmodelica_microgrid_gym.env:ModelicaEnv',
         kwargs=dict(  # reward_fun=rew.rew_fun,
             viz_cols=[
                 PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'inverter1.v_ref.{k}' for k in '012']],
                          callback=xylables_v,
                          color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                          style=[[None], ['--']]
                          ),
                 PlotTmpl([[f'lc.inductor{i}.i' for i in '123'], [f'inverter1.i_ref.{k}' for k in '012']],
                          callback=xylables_i,
                          color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                          style=[[None], ['--']]
                          ),
                 PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                          callback=xylables_R,
                          color=[['b', 'r', 'g']],
                          style=[[None]]
                          )
             ],
             viz_mode='episode',
             max_episode_steps=max_episode_steps,
             model_params={'lc.resistor1.R': R_filter,
                           'lc.resistor2.R': R_filter,
                           'lc.resistor3.R': R_filter,
                           'lc.resistor4.R': 0.0000001,
                           'lc.resistor5.R': 0.0000001,
                           'lc.resistor6.R': 0.0000001,
                           'lc.inductor1.L': L_filter,
                           'lc.inductor2.L': L_filter,
                           'lc.inductor3.L': L_filter,
                           'lc.capacitor1.C': C_filter,
                           'lc.capacitor2.C': C_filter,
                           'lc.capacitor3.C': C_filter,
                           'r_load.resistor1.R': partial(rand_load.load_step, gain=R),
                           'r_load.resistor2.R': partial(rand_load.load_step, gain=R),
                           'r_load.resistor3.R': partial(rand_load.load_step, gain=R),
                           'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_lim,
                                                                          high=v_lim) if t == 0 else None,
                           'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_lim,
                                                                          high=v_lim) if t == 0 else None,
                           'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_lim,
                                                                          high=v_lim) if t == 0 else None,
                           'lc.inductor1.i': lambda t: np.random.uniform(low=-i_lim,
                                                                         high=i_lim) if t == 0 else None,
                           'lc.inductor2.i': lambda t: np.random.uniform(low=-i_lim,
                                                                         high=i_lim) if t == 0 else None,
                           'lc.inductor3.i': lambda t: np.random.uniform(low=-i_lim,
                                                                         high=i_lim) if t == 0 else None,
                           },
             net=net,
             model_path='../../omg_grid/grid.paper_loadstep.fmu',
             on_episode_reset_callback=cb.fire,
             is_normalized=True
         )
         )
