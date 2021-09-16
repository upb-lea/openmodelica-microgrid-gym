from functools import partial
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
from stochastic.processes import VasicekProcess
from tqdm import tqdm

from experiments.hp_tune.env.random_load import RandomLoad
from experiments.hp_tune.env.vctrl_single_inv import CallbackList
from experiments.hp_tune.util.config import cfg
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import RandProcess

# load = 55  # 28
# net = Network.load('net/net_vctrl_single_inv.yaml')
# max_episode_steps = int(2 / net.ts)


# Simulation definitions
if not cfg['is_dq0']:
    # load net using abc reference values
    net = Network.load('net/net_vctrl_single_inv.yaml')
else:
    # load net using dq0 reference values
    net = Network.load('net/net_vctrl_single_inv_dq0.yaml')

# set high to not terminate env! Termination should be done in wrapper by env after episode-length-HP
max_episode_steps = 10000  # net.max_episode_steps  # number of simulation steps per episode

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC

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

"""
 Tescases need to have:
  - Full load
  - (nearly) No load
  - Step up/down
  - Drift up/down
1 second, start at nominal power
"""
time_to_nomPower = 0.1
time_nomPower_drift = 0.32
time_loadshading = 0.587
time_power_ramp_up = 0.741
time_power_ramp_down = 0.985
time_power_Ramp_stop = 1.3
time_drift_down2 = 1.52
time_step_up2 = 1.66
time_drift_down3 = 1.72

R_load = []


def load_step_deterministic(t):
    if -2 < t <= 0.1:
        return 100.0
    if 0.1 < t <= 0.2:
        return 50.0
    if 0.2 < t <= 0.3:
        return 100.0
    if 0.3 < t <= 0.4:
        return 50.0
    if 0.4 < t <= 0.5:
        return 200.0
    if 0.5 < t <= 0.6:
        return 50.0
    if 0.7 < t <= 0.7:
        return 14.0
    if 0.7 < t <= 0.8:
        return 200.0
    else:
        return 14


def load_step(t):
    """
    Doubles the load parameters
    :param t:
    :param gain: device parameter
    :return: Dictionary with load parameters
    """
    # Defines a load step after 0.01 s
    if time_to_nomPower < t <= time_to_nomPower + net.ts:
        # step to p_nom
        gen.proc.mean = 14
        gen.reserve = 14

    elif time_nomPower_drift < t <= time_nomPower_drift + net.ts:
        # drift
        gen.proc.mean = 40
        gen.proc.speed = 40
        # gen.reserve = 40


    elif time_loadshading < t <= time_loadshading + net.ts:
        # loadshading
        gen.proc.mean = upper_bound_load
        gen.reserve = upper_bound_load
        gen.proc.vol = 25

    elif time_power_ramp_up < t <= time_power_ramp_up + net.ts:
        # drift
        gen.proc.mean = 80
        gen.proc.speed = 10
        # gen.reserve = 40


    elif time_power_ramp_down < t <= time_power_ramp_down + net.ts:
        gen.proc.mean = 30
        gen.proc.speed = 80
        gen.proc.vol = 10
        # gen.reserve = 40

    elif time_power_Ramp_stop < t <= time_power_Ramp_stop + net.ts:
        gen.proc.mean = 30
        gen.proc.speed = 1000
        gen.proc.vol = 100
        # gen.reserve = 40

    elif time_drift_down2 < t <= time_drift_down2 + net.ts:
        gen.proc.mean = 100
        gen.proc.speed = 100
        # gen.reserve = 40

    elif time_step_up2 < t <= time_step_up2 + net.ts:
        gen.proc.mean = 20
        gen.proc.speed = 1000
        gen.reserve = 20

    elif time_drift_down3 < t <= time_drift_down3 + net.ts:
        gen.proc.mean = 50
        gen.proc.speed = 60
        gen.proc.vol = 2
        # gen.reserve = 40

    R_load_sample = gen.sample(t)
    R_load.append(R_load_sample)

    return R_load_sample


if __name__ == '__main__':
    # gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=1000, vol=10, mean=load), initial=load,
    #                  bounds=(lower_bound_load, upper_bound_load))

    # rand_load = RandomLoad(max_episode_steps, net.ts, gen)

    rand_load = RandomLoad(round(cfg['train_episode_length'] / 10), net.ts, gen,
                           bounds=(lower_bound_load_clip, upper_bound_load_clip),
                           bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))

    rand_load_train = RandomLoad(cfg['train_episode_length'], net.ts, gen,
                                 bounds=(lower_bound_load_clip, upper_bound_load_clip),
                                 bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))

    cb = CallbackList()
    # set initial = None to reset load random in range of bounds
    cb.append(partial(gen.reset))  # , initial=np.random.uniform(low=lower_bound_load, high=upper_bound_load)))
    cb.append(rand_load_train.reset)


    def xylables(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$R_{\mathrm{load}}\,/\,\mathrm{\Omega}$')
        ax.grid(which='both')
        ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
        # plt.title('Load example drawn from Ornstein-Uhlenbeck process \n- Clipping outside the shown y-range')
        plt.legend()
        fig.show()


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   net=net,
                   # model_params={'r_load.resistor1.R': rand_load.random_load_step,  # For use upper function
                   # model_params={'r_load.resistor1.R': rand_load_train.one_random_loadstep_per_episode,
                   # model_params={'r_load.resistor1.R': rand_load_train.random_load_step,
                   #              # For use upper function
                   #              'r_load.resistor2.R': rand_load.clipped_step,
                   #              'r_load.resistor3.R': rand_load.clipped_step},
                   model_params={'r_load.resistor1.R': load_step_deterministic,  # for check train-random
                                 'r_load.resistor2.R': load_step_deterministic,  # loadstep
                                 'r_load.resistor3.R': load_step_deterministic},
                   viz_cols=[
                       PlotTmpl([f'r_load.resistor{i}.R' for i in '123'],
                                callback=xylables
                                )],
                   model_path='omg_grid/grid.paper_loadstep.fmu',
                   max_episode_steps=max_episode_steps,
                   on_episode_reset_callback=cb.fire, )

    env.reset()
    R_load1 = []
    R_load2 = []
    R_load3 = []
    # for _ in range(max_episode_steps):
    for current_step in tqdm(range(max_episode_steps), desc='steps', unit='step', leave=False):
        env.render()

        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action

        # If env is reset for several loadsteps, store env.df
        """
        if current_step % round(cfg['train_episode_length'] / 10) == 0 and current_step != 0:
            R_load1.extend(env.history.df['r_load.resistor1.R'].copy().values.tolist())
            R_load2.extend(env.history.df['r_load.resistor2.R'].copy().values.tolist())
            R_load3.extend(env.history.df['r_load.resistor3.R'].copy().values.tolist())

            # obs = env.reset()
            env.on_episode_reset_callback()
        """
        if done:
            break
    env.close()
    R_load1.extend(env.history.df['r_load.resistor1.R'].copy().values.tolist())
    R_load2.extend(env.history.df['r_load.resistor2.R'].copy().values.tolist())
    R_load3.extend(env.history.df['r_load.resistor3.R'].copy().values.tolist())

    df_store = pd.DataFrame(list(zip(R_load1, R_load2, R_load3)),
                            columns=['r_load.resistor1.R', 'r_load.resistor2.R', 'r_load.resistor3.R'])

    # df_store = env.history.df[['r_load.resistor1.R', 'r_load.resistor2.R', 'r_load.resistor3.R']]
    # df_store.to_pickle('R_load_tenLoadstepPerEpisode2881Len_test_case_10_seconds.pkl')
    df_store.to_pickle('R_load_deterministic_test_case2_1_seconds.pkl')
