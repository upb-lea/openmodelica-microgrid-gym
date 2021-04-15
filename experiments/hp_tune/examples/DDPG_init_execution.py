import itertools
import time
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from pymongo import MongoClient
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net, folder_name
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import nested_map

np.random.seed(0)

# toDo: what to store:
"""
Alle importieren vom Recorder der in DB speichert und interagieren an den richtungen stellen mit dem env/agent...

after training: -> like: SaveOnBestTrainingRewardCallback(BaseCallback): after training
    hyperopt-data 
    weights
    model / net-architecture

Each step: -> StepRecorder (ggf. StepMonitor?)
    training_reward
    messdaten? (aus der net.yaml die outs?)

    training_return -> if episode done: store return(-> sollte der Monitor kennen)

config    
skriptname
start- und endzeit stempel
Computername
Architektur des Netzes (mit model.to_json() )
Gewichte des Netzes (mit model.get_layer('layer_name').weights)
Prädiktion (für jede Zielgröße eine längere Liste)
Testset (profilnummern von den messschrieben die prädiziert wurden)

"""


class Recorder:

    def __init__(self, URI: str = 'mongodb://localhost:27017/', database_name: str = 'OMG', ):
        self.client = MongoClient(URI)
        self.db = self.client[database_name]

    def save_to_mongodb(self, col: str = ' trails', data=None):
        trial_coll = self.db[col]  # get collection named col
        if data is None:
            raise ValueError('No data given to store in database!')
        trial_coll.insert_one(data)


class StepRecorder(Monitor):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=np.full(3 * 3, -np.inf), high=np.full(3 * 3, np.inf))
        self.obs_idx = None

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        # print(reward)
        obs = observation[list(itertools.chain.from_iterable(self.obs_idx))]

        # hier vll noch die Messung loggen? aus der obs die richtigen suchen? wie figure out die augmented states?

        # todo: hier unnötige states rauswerfen?

        return obs, reward, done, info

    def set_idx(self, obs):
        if self.obs_idx is None:
            self.obs_idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc.inductor{k}.i' for k in '123'],
                 [f'lc.capacitor{k}.v' for k in '123'], [f'inverter1.v_ref.{k}' for k in '012']])

    def reset(self, **kwargs):
        observation = super().reset()
        self.set_idx(self.env.history.cols)
        obs = observation[list(itertools.chain.from_iterable(self.obs_idx))]
        return obs


class TrainRecorder(BaseCallback):

    def __init__(self, verbose=1):
        super(TrainRecorder, self).__init__(verbose)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # asd = 1
        # ads = 2
        pass

    def _on_step(self) -> bool:
        # asd = 1
        return True

    def _on_rollout_end(self) -> None:
        # asd = 1
        pass


gamma = 0.28  # trail.suggest_loguniform("gamma", 0.1, 0.99)
error_exponent = 0.03  # trail.suggest_loguniform("error_exponent", 0.01, 4)
rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
             use_gamma_normalization=1, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
             i_nom=net['inverter1'].i_nom, det_run=True)


def xylables_v(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
    ax.grid(which='both')
    # ax.set_xlim([0, 0.005])
    ts = time.gmtime()
    # fig.savefig(
    #    f'{folder_name}/{n_trail}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.show()


def xylables_i(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    ax.grid(which='both')
    ts = time.gmtime()
    # fig.savefig(
    #    f'{folder_name}/{n_trail}/Inductor_currents{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.show()


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    ts = time.gmtime()
    # fig.savefig(f'{folder_name}/{n_trail}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.show()


env = gym.make('experiments.hp_tune.env:vctrl_single_inv_train-v0',
               # reward_fun=rew.rew_fun,
               reward_fun=rew.rew_fun_include_current,
               abort_reward=-(1 - rew.gamma),
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
               # obs_output = ['v1,v2']
               # on_episode_reset_callback=cb.fire  # needed?
               )

env = StepRecorder(env)

learning_rate = 0.00027  # trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#

weight_scale = 0.03  # trail.suggest_loguniform("weight_scale", 5e-4, 1)  # 0.005
batch_size = 900  # trail.suggest_int("batch_size", 32, 1024)  # 128
actor_hidden_size = 223  # trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
critic_hidden_size = 413  # trail.suggest_int("critic_hidden_size", 10, 500)  # 100

use_gamma_in_rew = 1
noise_var = 0.4  # trail.suggest_loguniform("noise_var", 0.01, 10)  # 2
noise_theta = 13.8  # trail.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU

n_actions = env.action_space.shape[-1]
noise_var = noise_var  # 20#0.2
noise_theta = noise_theta  # 50 # stiffness of OU
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                            sigma=noise_var * np.ones(n_actions), dt=net.ts)

policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size], qf=[critic_hidden_size,
                                                                                              critic_hidden_size,
                                                                                              critic_hidden_size]))

callback = TrainRecorder()

# model = myDDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
model = DDPG('MlpPolicy', env, verbose=1,
             policy_kwargs=policy_kwargs,
             learning_rate=learning_rate, buffer_size=5000, learning_starts=100,
             batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
             train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
             create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

for ex_run in range(10):

    model = DDPG('MlpPolicy', env, verbose=1,
                 policy_kwargs=policy_kwargs,
                 learning_rate=0, buffer_size=5000, learning_starts=100,
                 batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
                 train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    print(model.actor.mu._modules['2'].weight.T)
    model.actor.mu._modules['2'].weight.data = model.actor.mu._modules['2'].weight.data * 1
    model.actor.mu._modules['0'].weight.data = model.actor.mu._modules['0'].weight.data * 1
    model.actor.mu._modules['0'].bias.data = model.actor.mu._modules['0'].bias.data * 0.1
    model.actor.mu._modules['2'].bias.data = model.actor.mu._modules['2'].bias.data * 0.1
    # model.actor.mu._modules['0'].weight.T = model.actor.mu._modules['0'].weight.T * 0.1
    # print(model.actor.mu._modules['2'].weight.T)

    return_sum = 0.0
    obs = env.reset()
    while True:

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        return_sum += rewards
        if done:
            break
    env.close()
