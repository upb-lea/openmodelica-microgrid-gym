from datetime import datetime, time
from functools import partial
from itertools import accumulate
from os import makedirs

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stochastic.processes import VasicekProcess

from experiments.hp_tune.env.random_load import RandomLoad
from experiments.hp_tune.env.rewards import Reward
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network

from openmodelica_microgrid_gym.util import RandProcess
from gym.envs.registration import register
from experiments.hp_tune.util.config import cfg

folder_name = cfg['STUDY_NAME']  # 'DDPG_MRE_sqlite_PC2'
# experiment_name = 'DDPG_VC_Reward_MRE_reward_NOT_NORMED'
experiment_name = 'plots'
timestamp = datetime.now().strftime(f'_%Y.%b.%d_%X')

makedirs(folder_name, exist_ok=True)
# makedirs(folder_name + experiment_name, exist_ok=True)


# Simulation definitions
if not cfg['is_dq0']:
    # load net using abc reference values
    net = Network.load('net/net_vctrl_single_inv.yaml')
else:
    # load net using dq0 reference values
    net = Network.load('net/net_vctrl_single_inv_dq0.yaml')

# set high to not terminate env! Termination should be done in wrapper by env after episode-length-HP
max_episode_steps = 1500000  # net.max_episode_steps  # number of simulation steps per episode

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC

# plant
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F
# R = 40  # nomVoltPeak / 7.5   # / Ohm
lower_bound_load = -10  # to allow maximal load that draws i_limit
upper_bound_load = 200  # to apply symmetrical load bounds
lower_bound_load_clip = 14  # to allow maximal load that draws i_limit (let exceed?)
upper_bound_load_clip = 200  # to apply symmetrical load bounds
lower_bound_load_clip_std = 2
upper_bound_load_clip_std = 0
R = np.random.uniform(low=lower_bound_load, high=upper_bound_load)

gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=800, vol=40, mean=R), initial=R,
                  bounds=(lower_bound_load, upper_bound_load))

class CallbackList(list):
    def fire(self, *args, **kwargs):
        for listener in self:
            listener(*args, **kwargs)


# if save needed in dependence of trial ( -> foldername) shift to executive file?
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
    # fig.savefig(
    #    f'{folder_name + experiment_name}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    # ts = time.gmtime()
    # fig.savefig(f'{folder_name + experiment_name}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


rand_load_train = RandomLoad(cfg['train_episode_length'], net.ts, gen,
                             bounds=(lower_bound_load_clip, upper_bound_load_clip),
                             bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))

cb = CallbackList()
# set initial = None to reset load random in range of bounds
cb.append(partial(gen.reset))  # , initial=np.random.uniform(low=lower_bound_load, high=upper_bound_load)))
cb.append(rand_load_train.reset)

register(id='vctrl_single_inv_train-v0',
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
                           # 'r_load.resistor1.R': partial(rand_load_train.load_step, gain=R),
                           # 'r_load.resistor2.R': partial(rand_load_train.load_step, gain=R),
                           # 'r_load.resistor3.R': partial(rand_load_train.load_step, gain=R),
                           'r_load.resistor1.R': rand_load_train.random_load_step,
                           'r_load.resistor2.R': rand_load_train.random_load_step,
                           'r_load.resistor3.R': rand_load_train.random_load_step,
                           'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.inductor1.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           'lc.inductor2.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           'lc.inductor3.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           },
             net=net,
             model_path='omg_grid/grid.paper_loadstep.fmu',
             on_episode_reset_callback=cb.fire,
             is_normalized=True,
             action_time_delay=1
         )
         )

register(id='vctrl_single_inv_train-v1',
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
                           # 'r_load.resistor1.R': partial(rand_load_train.load_step, gain=R),
                           # 'r_load.resistor2.R': partial(rand_load_train.load_step, gain=R),
                           # 'r_load.resistor3.R': partial(rand_load_train.load_step, gain=R),
                           'r_load.resistor1.R': rand_load_train.one_random_loadstep_per_episode,
                           'r_load.resistor2.R': rand_load_train.clipped_step,
                           'r_load.resistor3.R': rand_load_train.clipped_step,
                           'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.inductor1.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           'lc.inductor2.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           'lc.inductor3.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           },
             net=net,
             model_path='omg_grid/grid.paper_loadstep.fmu',
             on_episode_reset_callback=cb.fire,
             is_normalized=True,
             action_time_delay=1
         )
         )

rand_load_test = RandomLoad(max_episode_steps, net.ts, gen,
                            load_curve=pd.read_pickle('experiments/hp_tune/data/R_load_hard_test_case_10_seconds.pkl'))

register(id='vctrl_single_inv_test-v0',
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
             max_episode_steps=20000,
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
                           'r_load.resistor1.R': partial(rand_load_test.give_dataframe_value, col='r_load.resistor1.R'),
                           'r_load.resistor2.R': partial(rand_load_test.give_dataframe_value, col='r_load.resistor2.R'),
                           'r_load.resistor3.R': partial(rand_load_test.give_dataframe_value, col='r_load.resistor3.R')
                           },
             net=net,
             model_path='omg_grid/grid.paper_loadstep.fmu',
             on_episode_reset_callback=cb.fire,
             is_normalized=True,
             action_time_delay=1
         )
         )

register(id='vctrl_single_inv_test-v1',
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
             max_episode_steps=100001,
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
                           # 'r_load.resistor1.R': partial(rand_load_train.load_step, gain=R),
                           # 'r_load.resistor2.R': partial(rand_load_train.load_step, gain=R),
                           # 'r_load.resistor3.R': partial(rand_load_train.load_step, gain=R),
                           'r_load.resistor1.R': rand_load_train.one_random_loadstep_per_episode,
                           'r_load.resistor2.R': rand_load_train.clipped_step,
                           'r_load.resistor3.R': rand_load_train.clipped_step,
                           'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_nom,
                                                                          high=v_nom) if t == -1 else None,
                           'lc.inductor1.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           'lc.inductor2.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           'lc.inductor3.i': lambda t: np.random.uniform(low=-i_nom,
                                                                         high=i_nom) if t == -1 else None,
                           },
             net=net,
             model_path='omg_grid/grid.paper_loadstep.fmu',
             on_episode_reset_callback=cb.fire,
             is_normalized=True,
             action_time_delay=1
         )
         )
