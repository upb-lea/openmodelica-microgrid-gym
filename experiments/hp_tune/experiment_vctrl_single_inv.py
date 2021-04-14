import time
from collections import OrderedDict
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch as th
from pymongo import MongoClient
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.agents.my_ddpg import myDDPG
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net, folder_name, max_episode_steps
from experiments.hp_tune.util.action_noise_wrapper import myOrnsteinUhlenbeckActionNoise
from experiments.hp_tune.util.record_env import RecordEnvCallback
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import abc_to_alpha_beta

np.random.seed(0)

number_learning_steps = 20000
number_plotting_steps = 50000
number_trails = 1

# reward_episode_mean = []
# R_training = []  # np.zeros(shape=(number_learning_steps+5,number_trails))
# i_phasor_training = []  # np.zeros(shape=(number_learning_steps,number_trails))
# v_phasor_training = []  # np.zeros(shape=(number_learning_steps,number_trails))
# i_a = []
# i_b = []
# i_c = []
# v_a = []
# v_b = []
# v_c = []
# params_change = OrderedDict()   # für schichtweise
params_change = []
# asd
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

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = np.inf, recorder=None,
                 n_trail=""):
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
        self.recorder = recorder
        self._n_training_steps = 0
        self._i_phasor = 0.0
        self.i_a = []
        self.i_b = []
        self.i_c = []
        self.v_a = []
        self.v_b = []
        self.v_c = []
        self._v_pahsor = 0.0
        self.n_episode = 0
        self.R_training = []
        self.i_phasor_training = []
        self.v_phasor_training = []
        self.reward_episode_mean = []
        self.n_trail = n_trail

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """

        obs, reward, done, info = super().step(action)

        self._n_training_steps += 1

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True

        # log measurement here?

        # add wanted features here (add appropriate self.observation in init!!)
        # calculate magnitude of current phasor abc
        self.i_phasor = self.cal_phasor_magnitude(obs[0:3])
        self.v_phasor = self.cal_phasor_magnitude(obs[3:6])

        self.R_training.append(self.env.history.df['r_load.resistor1.R'].iloc[-1])
        self.i_phasor_training.append((self.i_phasor) * self.env.net['inverter1'].i_lim)
        self.v_phasor_training.append((self.v_phasor) * self.env.net['inverter1'].v_lim)

        self.i_a.append(self.env.history.df['lc.inductor1.i'].iloc[-1])
        self.i_b.append(self.env.history.df['lc.inductor2.i'].iloc[-1])
        self.i_c.append(self.env.history.df['lc.inductor3.i'].iloc[-1])

        self.v_a.append(self.env.history.df['lc.capacitor1.v'].iloc[-1])
        self.v_b.append(self.env.history.df['lc.capacitor2.v'].iloc[-1])
        self.v_c.append(self.env.history.df['lc.capacitor3.v'].iloc[-1])

        if done:
            self.reward_episode_mean.append(np.mean(self.rewards))
            episode_data = {"Name": "On_Training",
                            "Epsisode_number": self.n_episode,
                            "Episode_length": self._n_training_steps,
                            "R_load_training": self.R_training,
                            "i_phasor_training": self.i_phasor_training,
                            "i_a_training": self.i_a,
                            "i_b_training": self.i_b,
                            "i_c_training": self.i_c,
                            "v_a_training": self.v_a,
                            "v_b_training": self.v_b,
                            "v_c_training": self.v_c,
                            "v_phasor_training": self.v_phasor_training,
                            "Rewards": self.rewards
                            }

            mongo_recorder.save_to_mongodb('Trail_number_' + self.n_trail, episode_data)

            # clear lists
            self.R_training = []
            self.i_phasor_training = []
            self.v_phasor_training = []
            self.i_a = []
            self.i_b = []
            self.i_c = []
            self.v_a = []
            self.v_b = []
            self.v_c = []
            self.n_episode += 1


        obs = np.append(obs, self.i_phasor - 0.5)

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

        self.i_phasor = self.cal_phasor_magnitude(obs[0:3])
        """
        self.v_phasor = self.cal_phasor_magnitude(obs[3:6])

        R_training.append(self.env.history.df['r_load.resistor1.R'].iloc[-1])
        i_phasor_training.append((self.i_phasor) * self.env.net['inverter1'].i_lim)
        v_phasor_training.append((self.v_phasor) * self.env.net['inverter1'].v_lim)

        i_a.append(self.env.history.df['lc.inductor1.i'].iloc[-1])
        i_b.append(self.env.history.df['lc.inductor2.i'].iloc[-1])
        i_c.append(self.env.history.df['lc.inductor3.i'].iloc[-1])

        v_a.append(self.env.history.df['lc.capacitor1.v'].iloc[-1])
        v_b.append(self.env.history.df['lc.capacitor2.v'].iloc[-1])
        v_c.append(self.env.history.df['lc.capacitor3.v'].iloc[-1])
        """
        obs = np.append(obs, self.i_phasor - 0.5)

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
        feature_diff_imax_iphasor = (i_phasor_mag) - 0.5

        if (i_phasor_mag) * 16 > 15:
            asd = 1

        return i_phasor_mag


class TrainRecorder(BaseCallback):

    def __init__(self, verbose=1):
        super(TrainRecorder, self).__init__(verbose)
        self.last_model_params = None  #self.model.policy.state_dict()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        #asd = 1
        #ads = 2
        pass

    def _on_step(self) -> bool:
        asd = 1
        # R_training[self.n_calls, number_trails] = self.training_env.envs[0].env.history.df['r_load.resistor1.R'].iloc[-1]
        # R_training[self.n_calls-1, 0] = self.training_env.envs[0].env.history.df['r_load.resistor1.R'].iloc[-1]
        """
        R_training.append(self.training_env.envs[0].env.history.df['r_load.resistor1.R'].iloc[-1])
        i_phasor_training.append((self.training_env.envs[0].i_phasor+0.5)*net['inverter1'].i_lim)
        v_phasor_training.append((self.training_env.envs[0].v_phasor+0.5)*net['inverter1'].v_lim)

        if (self.training_env.envs[0].i_phasor)*net['inverter1'].i_lim > 15:
            asd = 1

        i_a.append(self.training_env.envs[0].env.history.df['lc.inductor1.i'].iloc[-1])
        i_b.append(self.training_env.envs[0].env.history.df['lc.inductor2.i'].iloc[-1])
        i_c.append(self.training_env.envs[0].env.history.df['lc.inductor3.i'].iloc[-1])

        v_a.append(self.training_env.envs[0].env.history.df['lc.capacitor1.v'].iloc[-1])
        v_b.append(self.training_env.envs[0].env.history.df['lc.capacitor2.v'].iloc[-1])
        v_c.append(self.training_env.envs[0].env.history.df['lc.capacitor3.v'].iloc[-1])
        # nach env.step()
        """
        return True

    def _on_rollout_end(self) -> None:
        # asd = 1

        model_params = self.model.policy.parameters_to_vector()

        if self.last_model_params is None:
            self.last_model_params = model_params
        else:
            params_change.append(np.float64(np.mean(self.last_model_params - model_params)))

        """model_params = self.model.policy.state_dict()
        if self.last_model_params is None:
            for key, value in model_params.items():
                params_change[key.replace(".", "_")] = []
        else:
            for key, value in model_params.items():
                #print(key)
                params_change[key.replace(".", "_")].append(th.mean((model_params[key]-self.last_model_params[key])).tolist())
        """

        self.last_model_params = model_params

        # self.model.actor.mu._modules # alle :)
        pass


mongo_recorder = Recorder(database_name=folder_name)




def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale, alpha_relu_actor,
                        batch_size,
                        actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                        alpha_relu_critic,
                        noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                        training_episode_length, buffer_size,
                        learning_starts, tau, n_trail):
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

    env = FeatureWrapper(env, number_of_features=1, training_episode_length=training_episode_length,
                         recorder=mongo_recorder, n_trail=n_trail)

    n_actions = env.action_space.shape[-1]
    noise_var = noise_var  # 20#0.2
    noise_theta = noise_theta  # 50 # stiffness of OU
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
    #                                            sigma=noise_var * np.ones(n_actions), dt=net.ts)

    action_noise = myOrnsteinUhlenbeckActionNoise(n_steps_annealing=noise_steps_annealing,
                                                  sigma_min=noise_var * np.ones(n_actions) * noise_var_min,
                                                  mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                  sigma=noise_var * np.ones(n_actions), dt=net.ts)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size] * actor_number_layers
                                                                      , qf=[critic_hidden_size] * critic_number_layers))

    callback = TrainRecorder()

    # model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
    model = myDDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                   policy_kwargs=policy_kwargs,
                   learning_rate=learning_rate, buffer_size=buffer_size,
                   learning_starts=int(learning_starts * training_episode_length),
                   batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
                   train_freq=(1, "episode"), gradient_steps=- 1,
                   # n_episodes_rollout=1,
                   optimize_memory_usage=False,
                   create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    count = 0

    for kk in range(actor_number_layers + 1):

        model.actor.mu._modules[str(count)].weight.data = model.actor.mu._modules[str(count)].weight.data * weight_scale
        model.actor_target.mu._modules[str(count)].weight.data = model.actor_target.mu._modules[
                                                                     str(count)].weight.data * weight_scale

        model.actor.mu._modules[str(count)].bias.data = model.actor.mu._modules[str(count)].bias.data * bias_scale
        model.actor_target.mu._modules[str(count)].bias.data = model.actor.mu._modules[
                                                                   str(count)].bias.data * bias_scale

        if kk < actor_number_layers:
            model.actor.mu._modules[str(count + 1)].negative_slope = alpha_relu_actor
            model.actor_target.mu._modules[str(count + 1)].negative_slope = alpha_relu_actor

        count = count + 2

    count = 0

    for kk in range(critic_number_layers + 1):

        if kk < critic_number_layers:
            model.critic.qf0._modules[str(count + 1)].negative_slope = alpha_relu_critic
            model.critic_target.qf0._modules[str(count + 1)].negative_slope = alpha_relu_critic

        count = count + 2

    # todo: instead /here? store reward per step?!
    plot_callback = EveryNTimesteps(n_steps=number_plotting_steps,
                                    callback=RecordEnvCallback(env, model, 1000, mongo_recorder,
                                                               n_trail))
    model.learn(total_timesteps=number_learning_steps, callback=[callback, plot_callback])
    # model.learn(total_timesteps=1000, callback=callback)

    train_data = {"Name": "After_Training",
                  "Mean_eps_reward": env.reward_episode_mean,
                  "Sum_eps_rewad": env.get_episode_rewards()
                  }

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, train_data)
    # monitor_rewards = env.get_episode_rewards()
    # print(monitor_rewards)
    # todo: instead: store model(/weights+bias?) to database
    model.save(f'{folder_name}/{n_trail}/model.zip')

    return_sum = 0.0

    rew.gamma = 0
    # episodes will not abort, if limit is exceeded reward = -1
    rew.det_run = True
    rew.exponent = 0.5#1
    limit_exceeded_in_test = False
    limit_exceeded_penalty = 0
    env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v0',
                        reward_fun=rew.rew_fun_include_current,
                        # reward_fun=rew.rew_fun,
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
    env_test = FeatureWrapper(env_test, number_of_features=1)
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

    """
    train_R = {"Name": "Training",
               "R_load_training": R_training,
               "i_phasor_training": i_phasor_training,
               "v_phasor_training": v_phasor_training,
               "i_a": i_a,
               "i_b": i_b,
               "i_c": i_c,
               "v_a": v_a,
               "v_b": v_b,
               "v_c": v_c,
               "model_params_change": params_change,
               "actor_loss_batch_mean": model.actor_loss_batch_mean,
               "critic_loss_batch_mean": model.critic_loss_batch_mean
               }

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, train_R)
    """

    # train_data = pd.DataFrame(R_training)#, i_phasor_training, v_phasor_training, params_change,
    # model.actor_loss_batch_mean,model.critic_loss_batch_mean)

    # train_data.to_pickle(f'{folder_name}/Trail_number_' + n_trail)

    return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)


def objective(trail):
    """
    learning_rate = 0.0001  # trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = 0.5  # trail.suggest_loguniform("gamma", 0.1, 0.99)

    # gamma =  trail.suggest_float("gamma", 0.001, 0.99)

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

    learning_rate = 70e-6  # trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = 0.65  # trail.suggest_loguniform("gamma", 0.1, 0.99)
    weight_scale = 0.012  # trail.suggest_loguniform("weight_scale", 5e-4, 0.1)  # 0.005

    bias_scale = 0.01  # trail.suggest_loguniform("bias_scale", 5e-4, 0.1)  # 0.005
    alpha_relu_actor = 0.4  # trail.suggest_loguniform("alpha_relu_actor", 0.0001, 0.5)  # 0.005
    alpha_relu_critic = 0.04  # trail.suggest_loguniform("alpha_relu_critic", 0.0001, 0.5)  # 0.005

    batch_size = 256  # trail.suggest_int("batch_size", 32, 1024)  # 128
    buffer_size = 100000  # trail.suggest_int("buffer_size", 10, 20000)  # 128

    actor_hidden_size = 400  # trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
    actor_number_layers = 1  # trail.suggest_int("actor_number_layers", 1, 3)

    critic_hidden_size = 200  # trail.suggest_int("critic_hidden_size", 10, 500)  # 100
    critic_number_layers = 3  # trail.suggest_int("critic_number_layers", 1, 4)

    n_trail = str(trail.number)
    use_gamma_in_rew = 1
    noise_var = 0.04  # trail.suggest_loguniform("noise_var", 0.01, 4)  # 2
    # min var, action noise is reduced to (depends on noise_var)
    noise_var_min = 360e-9  # trail.suggest_loguniform("noise_var_min", 0.0000001, 2)
    # min var, action noise is reduced to (depends on training_episode_length)
    noise_steps_annealing = 0.8 * number_learning_steps  # trail.suggest_int("noise_steps_annealing", int(0.1 * number_learning_steps),
    #             number_learning_steps)
    noise_theta = 3.7  # trail.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU
    error_exponent = 0.04  # trail.suggest_loguniform("error_exponent", 0.01, 0.5)

    training_episode_length = 800  # trail.suggest_int("training_episode_length", 200, 5000)  # 128
    learning_starts = 0.1  # 0.32# trail.suggest_loguniform("learning_starts", 0.1, 2)  # 128
    tau = 0.05  # trail.suggest_loguniform("tau", 0.0001, 0.2)  # 2

    # toDo:
    # memory_interval = 1
    # weigth_regularizer = 0.5
    # memory_lim = 5000  # = buffersize?
    # warm_up_steps_actor = 2048
    # warm_up_steps_critic = 1024  # learning starts?
    # target_model_update = 1000

    trail_config_mongo = {"Name": "Config"}
    trail_config_mongo.update(trail.params)

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, trail_config_mongo)





    return experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale, alpha_relu_actor,
                               batch_size,
                               actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                               alpha_relu_critic,
                               noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                               training_episode_length, buffer_size,
                               learning_starts, tau, n_trail)


# for gamma grid search:
# gamma_list = list(itertools.chain(*[[0.001]*5, [0.25]*5, [0.5]*5, [0.75]*5, [0.99]*5]))
# search_space = {'gamma': gamma_list}

# toDo: postgresql instead of sqlite
study = optuna.create_study(study_name=folder_name,
                            direction='maximize',
                            storage=f'sqlite:///{folder_name}/R_load_log_test.sqlite',
                            load_if_exists=True,
                            # sampler=optuna.samplers.GridSampler(search_space)
                            )

study.optimize(objective, n_trials=number_trails)
