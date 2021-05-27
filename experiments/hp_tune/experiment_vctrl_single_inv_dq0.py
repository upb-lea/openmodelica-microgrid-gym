import itertools
import platform
import time
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import sshtunnel
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.agents.my_ddpg import myDDPG
# from agents.my_ddpg import myDDPG
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_dq0 import net, folder_name
from experiments.hp_tune.util.action_noise_wrapper import myOrnsteinUhlenbeckActionNoise
from experiments.hp_tune.util.record_env import RecordEnvCallback
from experiments.hp_tune.util.recorder import Recorder
from experiments.hp_tune.util.training_recorder import TrainRecorder
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import abc_to_alpha_beta, dq0_to_abc, abc_to_dq0
import experiments.hp_tune.util.config as cfg
np.random.seed(0)

number_learning_steps1 = 600000
number_plotting_steps = 100000
number_trails = 200

params_change = []

# mongo_recorder = Recorder(database_name=folder_name)


# mongo_recorder = Recorder(URI='mongodb://localhost:12001/',
#                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne
# mongo_recorder = Recorder(database_name=folder_name)


node = platform.uname().node

if node in cfg['lea_vpn_nodes']:
    server_name = 'lea38'
    tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                       12001)}

else:
    # assume we are on a node of pc2 -> connect to frontend and put data on prt 12001
    # from there they can be grep via permanent tunnel from cyberdyne
    server_name = 'fe.pc2.uni-paderborn.de'
    tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                       12001),
               'ssh_username': 'webbah'}

mongo_tunnel = sshtunnel.open_tunnel(server_name, **tun_cfg)

# mongo_tunnel = sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001))
mongo_tunnel.start()
mongo_recorder = Recorder(URI=f'mongodb://localhost:{mongo_tunnel.local_bind_port}/',
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne


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
        self.phase = []

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """

        action_abc = dq0_to_abc(action, self.env.net.components[0].phase)

        # clipping?
        obs, reward, done, info = super().step(action_abc)

        self._n_training_steps += 1

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True

        # log measurement here?

        # add wanted features here (add appropriate self.observation in init!!)
        # calculate magnitude of current phasor abc
        self.i_phasor = self.cal_phasor_magnitude(obs[0:3])
        self.v_phasor = self.cal_phasor_magnitude(obs[3:6])

        # todo: delta (ref-mess), letzte aktion, beides, delta i_phasor zur stromgrenze

        self.R_training.append(self.env.history.df['r_load.resistor1.R'].iloc[-1])
        self.i_phasor_training.append((self.i_phasor) * self.env.net['inverter1'].i_lim)
        self.v_phasor_training.append((self.v_phasor) * self.env.net['inverter1'].v_lim)

        self.i_a.append(self.env.history.df['lc.inductor1.i'].iloc[-1])
        self.i_b.append(self.env.history.df['lc.inductor2.i'].iloc[-1])
        self.i_c.append(self.env.history.df['lc.inductor3.i'].iloc[-1])

        self.v_a.append(self.env.history.df['lc.capacitor1.v'].iloc[-1])
        self.v_b.append(self.env.history.df['lc.capacitor2.v'].iloc[-1])
        self.v_c.append(self.env.history.df['lc.capacitor3.v'].iloc[-1])
        self.phase.append(self.env.net.components[0].phase)

        if done:
            self.reward_episode_mean.append(np.mean(self.rewards))
            episode_data = {"Name": "On_Training",
                            "Episode_number": self.n_episode,
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
                            "Rewards": self.rewards,
                            "Phase": self.phase
                            }

            """
            add here "model_params_change": callback.params_change, from training_recorder?
            """

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
            self.phase = []
            self.n_episode += 1

        # if setpoint in dq: Transform measurement to dq0!!!!
        obs[3:6] = abc_to_dq0(obs[3:6], self.env.net.components[0].phase)
        obs[0:3] = abc_to_dq0(obs[0:3], self.env.net.components[0].phase)

        """
        Feature control error: v_setpoint - v_mess
        """
        error = obs[6:9] - obs[3:6]

        """
        Feature delta to current limit
        """
        #delta_i_lim_i_phasor = 1 - self.i_phasor

        """
        Following maps the return to the range of [-0.5, 0.5] in
        case of magnitude = [-lim, lim] using (phasor_mag) - 0.5. 0.5 can be exceeded in case of the magnitude
        exceeds the limit (no extra env interruption here!, all phases should be validated separately)
        """
        #obs = np.append(obs, self.i_phasor - 0.5)
        obs = np.append(obs, error)
        #obs = np.append(obs, delta_i_lim_i_phasor)

        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_action)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """
        obs = super().reset()
        self._n_training_steps = 0

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
        self.phase.append(self.env.net.components[0].phase)

        # if setpoint in dq: Transform measurement to dq0!!!!
        obs[3:6] = abc_to_dq0(obs[3:6], self.env.net.components[0].phase)
        obs[0:3] = abc_to_dq0(obs[0:3], self.env.net.components[0].phase)
        """
        Feature control error: v_setpoint - v_mess
        """
        error = obs[6:9] - obs[3:6]

        """
        Feature delta to current limit
        """
        delta_i_lim_i_phasor = 1 - self.i_phasor

        #obs = np.append(obs, self.i_phasor - 0.5)
        obs = np.append(obs, error)
        #obs = np.append(obs, delta_i_lim_i_phasor)

        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_action)

        return obs

    def cal_phasor_magnitude(self, abc: np.array) -> float:
        """
        Calculated the magnitude of a phasor in a three phase system. M

        :param abc: Due to limit normed currents or voltages in abc frame
        :return: magnitude of the current or voltage phasor
        """
        # calculate magnitude of current phasor abc-> alpha,beta ->|sqrt(alpha² + beta²)|
        i_alpha_beta = abc_to_alpha_beta(abc)
        i_phasor_mag = np.sqrt(i_alpha_beta[0] ** 2 + i_alpha_beta[1] ** 2)

        return i_phasor_mag


def experiment_fit_DDPG_dq0(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale, alpha_relu_actor,
                            batch_size,
                            actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                            alpha_relu_critic,
                            noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                            training_episode_length, buffer_size,
                            learning_starts, tau, number_learning_steps, n_trail):
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
                   reward_fun=rew.rew_fun_dq0,
                   abort_reward=-1,
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

    env = FeatureWrapper(env, number_of_features=6, training_episode_length=training_episode_length,
                         recorder=mongo_recorder, n_trail=n_trail)

    n_actions = env.action_space.shape[-1]
    noise_var = noise_var  # 20#0.2
    noise_theta = noise_theta  # 50 # stiffness of OU
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                sigma=noise_var * np.ones(n_actions), dt=net.ts)

    # action_noise = myOrnsteinUhlenbeckActionNoise(n_steps_annealing=noise_steps_annealing,
    #                                              sigma_min=noise_var * np.ones(n_actions) * noise_var_min,
    #                                              mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
    #                                              sigma=noise_var * np.ones(n_actions), dt=net.ts)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size] * actor_number_layers
                                                                      , qf=[critic_hidden_size] * critic_number_layers))

    callback = TrainRecorder()

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                 # model = myDDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=buffer_size,
                 learning_starts=int(learning_starts * training_episode_length),
                 batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
                 train_freq=(1, "episode"), gradient_steps=- 1,
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

    # Plotting tests during Training (no training in here!)
    plot_callback = EveryNTimesteps(n_steps=number_plotting_steps,
                                    callback=RecordEnvCallback(env, model, 1000, mongo_recorder,
                                                               n_trail))
    model.learn(
        total_timesteps=number_learning_steps)  # , callback=[callback])#  Ohne Plotting zwischendurch, plot_callback])

    train_data = {"Name": "After_Training",
                  "Mean_eps_reward": env.reward_episode_mean,
                  "Sum_eps_reward": env.get_episode_rewards()
                  }

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, train_data)

    model.save(f'{folder_name}/{n_trail}/model.zip')  # Hier woanders speichern!
    # model.save(f'/scratch/hpc-prf-reinfl/weber/OMG/{folder_name}/{n_trail}/model.zip')  #Hier woanders speichern!

    return_sum = 0.0
    rew.gamma = 0
    # episodes will not abort, if limit is exceeded reward = -1
    rew.det_run = True
    rew.exponent = 0.5  # 1
    limit_exceeded_in_test = False
    limit_exceeded_penalty = 0
    env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v0',
                        reward_fun=rew.rew_fun_dq0,
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
    env_test = FeatureWrapper(env_test, number_of_features=6)
    obs = env_test.reset()
    phase_list = []
    phase_list.append(env_test.env.net.components[0].phase)

    rew_list = []

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)
        phase_list.append(env_test.env.net.components[0].phase)

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
                           "Reward": rew_list,
                           "Phase": phase_list,
                           "Node": platform.uname().node,
                           "Info": "No delay, obs=[v_mess,sp_dq0, i_mess_dq0, error_mess_sp, last_action]"}

    # Add v-&i-measurements
    test_after_training.update({env_test.viz_col_tmpls[j].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                })

    test_after_training.update({env_test.viz_col_tmpls[2].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                })

    mongo_recorder.save_to_mongodb('Trail_number_' + n_trail, test_after_training)

    return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)
