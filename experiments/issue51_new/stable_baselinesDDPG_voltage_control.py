from datetime import datetime
from functools import partial
from os import makedirs
from typing import List

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stochastic.processes import VasicekProcess

from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import nested_map, RandProcess

np.random.seed(0)

timestamp = datetime.now().strftime(f'%Y.%b.%d %X ')
makedirs(timestamp)

# Simulation definitions
net = Network.load('../../net/net_single-inv-Paper_Loadstep.yaml')
max_episode_steps = 1000  # number of simulation steps per episode
# num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
# iLimit = 30  # inverter current limit / A
# iNominal = 20  # nominal inverter current / A
mu_c = 2  # factor for barrier function (see below)
mu_v = 2  # factor for barrier function (see below)
i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
# plant
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F
R = 28  # nomVoltPeak / 7.5   # / Ohm
lower_bound_load = 11  # to allow maximal load that draws i_limit (toDo: let exceed?)
upper_bound_load = 45  # to apply symmetrical load bounds

gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=1000, vol=70, mean=R), initial=R,
                  bounds=(lower_bound_load, upper_bound_load))


def load_step(t, gain):
    """
    Changes the load parameters
    :param t:
    :param gain: device parameter
    :return: Sample from SP
    """
    # Defines a load step after 0.01 s
    if .05 < t <= .05 + net.ts:
        gen.proc.mean = gain * 0.55
        gen.reserve = gain * 0.55

    return gen.sample(t)


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc.inductor{k}.i' for k in '123'], [f'inverter1.i_ref.{k}' for k in '012'],
                 [f'lc.capacitor{k}.v' for k in '123'], [f'inverter1.v_ref.{k}' for k in '012']])

    def rew_fun(self, cols: List[str], data: np.ndarray, risk) -> float:
        """
        Defines the reward function for the environment. Uses the observations and set-points to evaluate the quality of
        the used parameters.
        Takes current and voltage measurements and set-points to calculate the mean-root control error and uses a
        logarithmic barrier function in case of violating the current limit. Barrier function is adjustable using
        parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        vabc_master = data[idx[2]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_abc_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates
        vsp_abc_master = data[idx[3]]  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint

        error = (  # np.sum((np.abs((isp_abc_master - iabc_master)) / i_lim) ** 0.5, axis=0)
            # toDo: Es gibt kein iREF!!! Nur Grenze überprüfen
            # + -np.sum(mu_c * np.log(1 - np.maximum(np.abs(iabc_master) - i_nom, 0) /
            #                             (i_lim - i_nom)), axis=0)
            np.sum((np.abs((vsp_abc_master - vabc_master)) / v_nom) ** 0.5, axis=0) \
            # + -np.sum(mu_v * np.log(1 - np.maximum(np.abs(vabc_master) - v_nom, 0) /
            #                             (v_lim - v_nom)), axis=0)\
        )  # / max_episode_steps

        return -np.clip(error.squeeze(), 0, 1e5)


def xylables_i(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    ax.grid(which='both')
    fig.savefig(f'{timestamp}/Inductor_currents.pdf')
    fig.show()


def xylables_v(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
    ax.grid(which='both')
    fig.savefig(f'{timestamp}/Capacitor_voltages.pdf')
    fig.show()


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    fig.savefig(f'{timestamp}/Load.pdf')
    fig.show()


env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
               reward_fun=Reward().rew_fun,
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
                             'r_load.resistor1.R': partial(load_step, gain=R),
                             'r_load.resistor2.R': partial(load_step, gain=R),
                             'r_load.resistor3.R': partial(load_step, gain=R),
                             # 'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_lim,
                             #                                              high=v_lim) if t == 0 else None,
                             # 'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_lim,
                             #                                              high=v_lim) if t == 0 else None,
                             # 'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_lim,
                             #                                              high=v_lim) if t == 0 else None,
                             # 'lc.inductor1.i': lambda t: np.random.uniform(low=-i_lim,
                             #                                              high=i_lim) if t == 0 else None,
                             # 'lc.inductor2.i': lambda t: np.random.uniform(low=-i_lim,
                             #                                              high=i_lim) if t == 0 else None,
                             # 'lc.inductor3.i': lambda t: np.random.uniform(low=-i_lim,
                             #                                              high=i_lim) if t == 0 else None,
                             },
               net=net,
               model_path='../../omg_grid/grid.paper_loadstep.fmu',
               on_episode_reset_callback=partial(gen.reset, initial=R)
               )

with open(f'{timestamp}/env.txt', 'w') as f:
    print(str(env), file=f)
env = Monitor(env)


class RecordEnvCallback(BaseCallback):
    def _on_step(self) -> bool:
        obs = env.reset()
        for _ in range(max_episode_steps):
            env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                break
        env.close()
        env.reset()
        return True


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# DDPG learning parameters
gamma = 0.9  # discount factor
batch_size = 128
memory_interval = 1
alpha_actor = 5e-6
alpha_critic = 5e-4
noise_var = 0.2
noise_theta = 5  # stiffness of OU
alpha_lRelu = 0.1
weigth_regularizer = 0.01

memory_lim = 5000  # = buffersize?
warm_up_steps_actor = 2048
warm_up_steps_critic = 1024
target_model_update = 1000

# NN architecture
actor_hidden_size = 100  # Using LeakyReLU
# output linear
critic_hidden_size_1 = 75  # Using LeakyReLU
critic_hidden_size_2 = 75  # Using LeakyReLU
critic_hidden_size_3 = 75  # Using LeakyReLU
# output linear

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                            sigma=noise_var * np.ones(n_actions), dt=net.ts)

policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size], qf=[critic_hidden_size_1,
                                                                                              critic_hidden_size_2,
                                                                                              critic_hidden_size_3]))

model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{timestamp}/', policy_kwargs=policy_kwargs,
             learning_rate=alpha_critic, buffer_size=memory_lim, learning_starts=warm_up_steps_critic,
             batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
             train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
             create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

checkpoint_on_event = CheckpointCallback(save_freq=10000, save_path=f'{timestamp}/checkpoints/')
record_env = RecordEnvCallback()
plot_callback = EveryNTimesteps(n_steps=1000, callback=record_env)
model.learn(total_timesteps=500000, callback=[checkpoint_on_event, plot_callback])

# model.save('ddpg_CC2')

# del model  # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_CC")

# obs = env.reset()
# while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()
