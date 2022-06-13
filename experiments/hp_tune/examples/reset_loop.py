import time
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net, folder_name
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import abc_to_alpha_beta

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


class FeatureWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = np.inf):
        """
        Env Wrapper to add features to the env-observations and adds information to env.step output which can be used in
        case of an continuing (non-episodic) task to reset the environment without being terminated by done
        :param env: Gym environment to wrap
        :param number_of_features: Number of features added to the env observations in the wrapped step method
        :param training_episode_length: (For non-episodic environments) number of training steps after the env is reset
            by the agent for training purpose (Set to inf in test env!)
        :
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape[0] + number_of_features, -np.inf),
            high=np.full(env.observation_space.shape[0] + number_of_features, np.inf))
        self.training_episode_length = training_episode_length
        self._n_training_steps = 0

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """

        obs, reward, done, info = super().step(action)

        self._n_training_steps += 1

        if self._n_training_steps % self.training_episode_length == 0:
            info["timelimit_reached"] = True

        # log measurement here?

        # add wanted features here (add appropriate self.observation in init!!)
        # calculate magnitude of current phasor abc
        feature_diff_imax_iphasor = self.cal_phasor_magnitude(obs[0:3])

        obs = np.append(obs, feature_diff_imax_iphasor)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """
        obs = super().reset()
        self._n_training_steps = 0

        # reset timelimit_reached flag
        # self.info["timelimit_reached"] = False

        feature_diff_imax_iphasor = self.cal_phasor_magnitude(obs[0:3])
        obs = np.append(obs, feature_diff_imax_iphasor)

        return obs

    def cal_phasor_magnitude(self, abc: np.array) -> float:
        """
        Calculated the magnitude of a phasor in a three phase system. Maps the return to the range of [-0.5, 0.5] in
        case of magnitude = [-lim, lim] using (1 - phasor_mag) - 0.5. -0.5 can be exceeded in case of the magnitude
        exceeds the limit (no extra env interruption here!, all phases should be validated separately)

        :param abc: Due to limit normed currents or voltages in abc frame
        :return: magnitude of the current or voltage phasor
        """
        # calculate magnitude of current phasor abc-> alpha,beta ->|sqrt(alpha² + beta²)|
        i_alpha_beta = abc_to_alpha_beta(abc)
        i_phasor_mag = np.sqrt(i_alpha_beta[0] ** 2 + i_alpha_beta[1] ** 2)

        # mapping [0,1+]
        # feature_diff_imax_iphasor = 1 - (1 - i_phasor_mag)

        # mapping [-0.5 -,0.5] (can be < 0.5 if phasor exceeds lim)
        feature_diff_imax_iphasor = (1 - i_phasor_mag) - 0.5

        return feature_diff_imax_iphasor


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
        asd = 1

        # nach env.step()

        return True

    def _on_rollout_end(self) -> None:
        # asd = 1
        pass


mongo_recorder = Recorder(database_name=folder_name)

rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, 1,
             use_gamma_normalization=1, error_exponent=1, i_lim=net['inverter1'].i_lim,
             i_nom=net['inverter1'].i_nom)


def xylables_v(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
    ax.grid(which='both')
    # ax.set_xlim([0, 0.005])
    ts = time.gmtime()
    fig.savefig(
        f'{folder_name}/{n_trail}/Capacitor_voltages{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


def xylables_i(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    ax.grid(which='both')
    ts = time.gmtime()
    fig.savefig(
        f'{folder_name}/{n_trail}/Inductor_currents{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    ts = time.gmtime()
    fig.savefig(f'{folder_name}/{n_trail}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    plt.close()


env = gym.make('experiments.hp_tune.env:vctrl_single_inv_train-v0',
               # reward_fun=rew.rew_fun,
               reward_fun=rew.rew_fun_include_current,
               # reward_fun=rew.rew_fun,
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
               obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                           'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                           'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
               )

env = FeatureWrapper(env, number_of_features=1, training_episode_length=1000)

while True:
    obs = env.reset()

    asd = 1
