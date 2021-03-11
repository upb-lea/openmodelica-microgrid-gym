import itertools
import time
from typing import Union
from gym.spaces import Discrete, Box

import matplotlib.pyplot as plt

import gym
import numpy as np
import optuna
import torch as th
from pymongo import MongoClient
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net, folder_name, max_episode_steps
from experiments.hp_tune.util.record_env import RecordEnvCallback
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import nested_map, abc_to_alpha_beta

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
        self.observation_space = gym.spaces.Box(low=np.full(3 * 3 + 1, -np.inf), high=np.full(3 * 3 + 1, np.inf))
        self.obs_idx = None

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        # print(reward)
        obs = observation[list(itertools.chain.from_iterable(self.obs_idx))]

        # hier vll noch die Messung loggen? aus der obs die richtigen suchen? wie figure out die augmented states?

        # add wanted features here (add appropriate self.observation in init!!)

        # calculate magnitude of current phasor abc-> alpha,beta ->|sqrt(alpha² + beta²)|
        i_alpha_beta = abc_to_alpha_beta(obs[0:3])
        i_phasor_mag = np.sqrt(i_alpha_beta[0] ** 2 + i_alpha_beta[1] ** 2)

        # feature_diff_imax_iphasor = 1 - (1 - i_phasor_mag)  # mapping [0,1+]
        feature_diff_imax_iphasor = (
                                                1 - i_phasor_mag) - 0.5  # mapping [-0.5 -,0.5]  (can be < 0.5 if phasor exceeds lim)

        obs = np.append(obs, feature_diff_imax_iphasor)

        return obs, reward, done, info

    def set_idx(self, obs):
        if self.obs_idx is None:
            self.obs_idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc.inductor{k}.i' for k in '123'],
                 [f'lc.capacitor{k}.v' for k in '123'], [f'inverter1.v_ref.{k}' for k in '012']])

    def reset(self, **kwargs):
        obs = super().reset()
        # self.set_idx(self.env.history.cols)
        # obs = observation[list(itertools.chain.from_iterable(self.obs_idx))]
        # calculate magnitude of current phasor abc-> alpha,beta ->|sqrt(alpha² + beta²)|
        i_alpha_beta = abc_to_alpha_beta(obs[0:3])
        i_phasor_mag = np.sqrt(i_alpha_beta[0] ** 2 + i_alpha_beta[1] ** 2)

        # feature_diff_imax_iphasor = 1 - (1 - i_phasor_mag)  # mapping [0,1+]
        feature_diff_imax_iphasor = (
                                            1 - i_phasor_mag) - 0.5  # mapping [-0.5 -,0.5]  (can be < 0.5 if phasor exceeds lim)

        obs = np.append(obs, feature_diff_imax_iphasor)
        return obs


class TrainRecorder(BaseCallback):

    def __init__(self, verbose=1):
        super(TrainRecorder, self).__init__(verbose)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        #asd = 1
        #ads = 2
        pass

    def _on_step(self) -> bool:
        # asd = 1
        return True

    def _on_rollout_end(self) -> None:
        # asd = 1
        pass


mongo_recorder = Recorder(database_name=folder_name)


def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                        actor_hidden_size, critic_hidden_size, noise_var, noise_theta, error_exponent, n_trail):
    # rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
    #             use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent)
    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
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
    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=5000, learning_starts=100,
                 batch_size=batch_size, tau=0.005, gamma=gamma, action_noise=action_noise,
                 train_freq=- 1, gradient_steps=- 1, n_episodes_rollout=1, optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    model.actor.mu._modules['0'].weight.data = model.actor.mu._modules['0'].weight.data * weight_scale
    model.actor.mu._modules['2'].weight.data = model.actor.mu._modules['2'].weight.data * weight_scale
    model.actor_target.mu._modules['0'].weight.data = model.actor_target.mu._modules['0'].weight.data * weight_scale
    model.actor_target.mu._modules['2'].weight.data = model.actor_target.mu._modules['2'].weight.data * weight_scale
    # model.actor.mu._modules['0'].bias.data = model.actor.mu._modules['0'].bias.data * weight_bias_scale
    # model.actor.mu._modules['2'].bias.data = model.actor.mu._modules['2'].bias.data * weight_bias_scale

    # todo: instead /here? store reward per step?!
    plot_callback = EveryNTimesteps(n_steps=50000,
                                    callback=RecordEnvCallback(env, model, max_episode_steps, mongo_recorder, n_trail))
    model.learn(total_timesteps=200000, callback=[callback, plot_callback])
    # model.learn(total_timesteps=1000, callback=callback)

    monitor_rewards = env.get_episode_rewards()
    # print(monitor_rewards)
    # todo: instead: store model(/weights+bias?) to database
    model.save(f'{folder_name}/{n_trail}/model.zip')

    return_sum = 0.0

    rew.gamma = 0
    # episodes will not abort, if limit is exceeded reward = -1
    rew.det_run = True
    rew.exponent = 1
    limit_exceeded_in_test = False
    limit_exceeded_penalty = 0
    env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v0',
                        reward_fun=rew.rew_fun_include_current,
                        abort_reward=-1,  # no needed if in rew no None is given back
                        # on_episode_reset_callback=cb.fire  # needed?
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
                                     )],
                        obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                    'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                    'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                        )
    env_test = StepRecorder(env_test)
    obs = env_test.reset()

    rew_list = []

    # toDo: - Use other Test-episode
    #       - Rückgabewert = (Summe der üblichen Rewards) / (Anzahl steps Validierung) + (Penalty i.H.v. -1)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)

        if rewards == -1 and not limit_exceeded_in_test:
            # Set addidional penalty of -1 if limit is exceeded once in the test case
            limit_exceeded_in_test = True
            limit_exceeded_penalty = -1
        env_test.render()
        return_sum += rewards
        rew_list.append(rewards)
        # print(rewards)
        if done:
            env_test.close()
            # print(limit_exceeded_in_test)
            break

    ts = time.gmtime()
    test_after_training = {"Name": "Test",
                           "time": ts,
                           "Reward": rew_list}

    # Add v-measurements
    test_after_training.update({env_test.viz_col_tmpls[j].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                })

    test_after_training.update({env_test.viz_col_tmpls[2].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                })

    # va = self.env.env.history[self.env.env.viz_col_tmpls[0].vars[0]].copy()

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, test_after_training)

    return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)


def objective(trail):
    learning_rate = 0.0001  # trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = 0.68  # trail.suggest_loguniform("gamma", 0.1, 0.99)
    weight_scale = 0.08  # trail.suggest_loguniform("weight_scale", 5e-4, 1)  # 0.005
    batch_size = 124  # trail.suggest_int("batch_size", 32, 1024)  # 128
    actor_hidden_size = 100  # trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
    critic_hidden_size = 400  # trail.suggest_int("critic_hidden_size", 10, 500)  # 100
    n_trail = str(trail.number)
    use_gamma_in_rew = 1
    noise_var = 0.2  # trail.suggest_loguniform("noise_var", 0.01, 10)  # 2
    noise_theta = 20  # trail.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU
    error_exponent = 0.08  # trail.suggest_loguniform("error_exponent", 0.01, 4)

    """

    learning_rate = trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = trail.suggest_loguniform("gamma", 0.1, 0.99)
    weight_scale = trail.suggest_loguniform("weight_scale", 5e-4, 0.1)  # 0.005
    batch_size = trail.suggest_int("batch_size", 32, 1024)  # 128
    actor_hidden_size = trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
    critic_hidden_size = trail.suggest_int("critic_hidden_size", 10, 500)  # 100
    n_trail = str(trail.number)
    use_gamma_in_rew = 1
    noise_var = trail.suggest_loguniform("noise_var", 0.01, 4)  # 2
    noise_theta = trail.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU
    error_exponent = trail.suggest_loguniform("error_exponent", 0.01, 0.5)

    # toDo:
    # alpha_lRelu = trail.suggest_loguniform("alpha_lRelu", 0.0001, 0.5)  #0.1
    # memory_interval = 1
    # weigth_regularizer = 0.5
    # memory_lim = 5000  # = buffersize?
    # warm_up_steps_actor = 2048
    # warm_up_steps_critic = 1024  # learning starts?
    # target_model_update = 1000

    trail_config_mongo = {"Name": "Config"}
    trail_config_mongo.update(trail.params)

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, trail_config_mongo)

    return experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                               actor_hidden_size, critic_hidden_size, noise_var, noise_theta, error_exponent, n_trail)


# toDo: postgresql instead of sqlite
study = optuna.create_study(study_name=folder_name,
                            direction='maximize', storage=f'sqlite:///{folder_name}/optuna_data.sqlite',
                            load_if_exists=True)

study.optimize(objective, n_trials=50)
