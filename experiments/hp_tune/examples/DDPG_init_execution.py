import platform
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.type_aliases import GymStepReturn

# from agents.my_ddpg import myDDPG
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net  # , folder_name
from experiments.hp_tune.util.config import cfg
from experiments.hp_tune.util.recorder import Recorder
from experiments.hp_tune.util.scheduler import linear_schedule
from openmodelica_microgrid_gym.util import abc_to_alpha_beta, dq0_to_abc, abc_to_dq0

# np.random.seed(0)

folder_name = cfg['STUDY_NAME']
node = platform.uname().node

# mongo_recorder = Recorder(database_name=folder_name)
mongo_recorder = Recorder(node=node,
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne or locally as json to cfg[meas_data_folder]


class FeatureWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = np.inf,
                 recorder=None, n_trail="", integrator_weight=net.ts, antiwindup_weight=net.ts):
        """
        Env Wrapper to add features to the env-observations and adds information to env.step output which can be used in
        case of an continuing (non-episodic) task to reset the environment without being terminated by done

        Hint: is_dq0: if the control is done in dq0; if True, the action is tranfered to abc-system using env-phase and
            the observation is tranfered back to dq using the next phase

        :param env: Gym environment to wrap
        :param number_of_features: Number of features added to the env observations in the wrapped step method
        :param training_episode_length: (For non-episodic environments) number of training steps after the env is reset
            by the agent for training purpose (Set to inf in test env!)

        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape[0] + number_of_features, -np.inf),
            high=np.full(env.observation_space.shape[0] + number_of_features, np.inf))

        # increase action-space for PI-seperation
        # self.action_space=gym.spaces.Box(low=np.full(d_i, -1), high=np.full(d_i, 1))

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
        self.integrator_sum = np.zeros(self.action_space.shape)
        self.integrator_weight = integrator_weight
        self.antiwindup_weight = antiwindup_weight
        self.used_P = np.zeros(self.action_space.shape)
        self.used_I = np.zeros(self.action_space.shape)
        self.used_I_action = np.zeros(self.action_space.shape)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """
        action_P = action[0:3]
        action_I = action[3:6]

        self.used_I_action = np.copy(action_I)

        self.integrator_sum += action_I * self.integrator_weight

        action_PI = action_P + self.integrator_sum

        if cfg['is_dq0']:
            # Action: dq0 -> abc
            action_abc = dq0_to_abc(action_PI, self.env.net.components[0].phase)

        # check if m_abc will be clipped
        if np.any(abs(action_abc) > 1):
            # if, reduce integrator by clipped delta
            action_delta = abc_to_dq0(np.clip(action_abc, -1, 1) - action_abc, self.env.net.components[0].phase)
            self.integrator_sum += action_delta * self.antiwindup_weight

        obs, reward, done, info = super().step(action_abc)
        self._n_training_steps += 1

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True

        # add wanted features here (add appropriate self.observation in init!!)
        # calculate magnitude of current phasor abc
        self.i_phasor = self.cal_phasor_magnitude(obs[0:3])
        self.v_phasor = self.cal_phasor_magnitude(obs[3:6])

        if cfg['loglevel'] == 'train':
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
            self.n_episode += 1

            if cfg['loglevel'] == 'train':
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
                                "Phase": self.phase,
                                "Node": platform.uname().node,
                                "Trial number": self.n_trail,
                                "Database name": folder_name,
                                "Reward function": 'rew.rew_fun_dq0',
                                }

                """
                add here "model_params_change": callback.params_change, from training_recorder?
                """

                # stores data locally to cfg['meas_data_folder'], needs to be grept / transfered via reporter to mongodc
                mongo_recorder.save_to_json('Trial_number_' + self.n_trail, episode_data)

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

        if cfg['is_dq0']:
            # if setpoint in dq: Transform measurement to dq0!!!!
            obs[3:6] = abc_to_dq0(obs[3:6], self.env.net.components[0].phase)
            obs[0:3] = abc_to_dq0(obs[0:3], self.env.net.components[0].phase)

        """
        Features
        """
        error = obs[6:9] - obs[3:6]  # control error: v_setpoint - v_mess
        # delta_i_lim_i_phasor = 1 - self.i_phasor  # delta to current limit

        """
        Following maps the return to the range of [-0.5, 0.5] in
        case of magnitude = [-lim, lim] using (phasor_mag) - 0.5. 0.5 can be exceeded in case of the magnitude
        exceeds the limit (no extra env interruption here!, all phases should be validated separately)
        """
        # obs = np.append(obs, self.i_phasor - 0.5)
        obs = np.append(obs, error)
        obs = np.append(obs, np.sin(self.env.net.components[0].phase))
        obs = np.append(obs, np.cos(self.env.net.components[0].phase))
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)
        # obs = np.append(obs, delta_i_lim_i_phasor)

        """
        Add used action to the NN input to learn delay
        """
        # obs = np.append(obs, self.used_action)

        # todo efficiency?
        self.used_P = np.copy(action_P)
        self.used_I = np.copy(self.integrator_sum)
        # self.used_P = action_P
        # self.used_I = self.integrator_sum

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """
        obs = super().reset()
        self._n_training_steps = 0
        self.integrator_sum = np.zeros(self.action_space.shape)
        self.used_P = np.zeros(self.action_space.shape)
        self.used_I = np.zeros(self.action_space.shape)

        self.i_phasor = self.cal_phasor_magnitude(obs[0:3])
        self.v_phasor = self.cal_phasor_magnitude(obs[3:6])

        if cfg['loglevel'] == 'train':
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

        if cfg['is_dq0']:
            # if setpoint in dq: Transform measurement to dq0!!!!
            obs[3:6] = abc_to_dq0(obs[3:6], self.env.net.components[0].phase)
            obs[0:3] = abc_to_dq0(obs[0:3], self.env.net.components[0].phase)
        """
        Features
        """
        error = obs[6:9] - obs[3:6]  # control error: v_setpoint - v_mess
        # delta_i_lim_i_phasor = 1 - self.i_phasor  # delta to current limit

        """
        Following maps the return to the range of [-0.5, 0.5] in
        case of magnitude = [-lim, lim] using (phasor_mag) - 0.5. 0.5 can be exceeded in case of the magnitude
        exceeds the limit (no extra env interruption here!, all phases should be validated separately)
        """
        # obs = np.append(obs, self.i_phasor - 0.5)
        obs = np.append(obs, error)
        obs = np.append(obs, np.sin(self.env.net.components[0].phase))
        obs = np.append(obs, np.cos(self.env.net.components[0].phase))
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)
        # obs = np.append(obs, delta_i_lim_i_phasor)
        """
        Add used action to the NN input to learn delay
        """
        # obs = np.append(obs, self.used_action)

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


number_learning_steps = 10000
integrator_weight = 0.216  # trial.suggest_loguniform("integrator_weight", 1 / 20, 20)
# integrator_weight = trial.suggest_loguniform("integrator_weight", 1e-6, 1e-0)
# antiwindup_weight = trial.suggest_loguniform("antiwindup_weight", 50e-6, 50e-3)
antiwindup_weight = 0.96  # trial.suggest_float("antiwindup_weight", 0.00001, 1)

learning_rate = 1.42e-5  # trial.suggest_loguniform("learning_rate", 100e-9, 100e-6)  # 0.0002#

lr_decay_start = 0.88  # trial.suggest_float("lr_decay_start", 0.00001, 1)  # 3000  # 0.2 * number_learning_steps?
lr_decay_duration = 0.064  # trial.suggest_float("lr_decay_duration", 0.00001,
#              1)  # 3000  # 0.2 * number_learning_steps?
t_start = int(lr_decay_start * number_learning_steps)
t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                       number_learning_steps))
final_lr = 0.3  # trial.suggest_float("final_lr", 0.00001, 1)

gamma = 0.927  # trial.suggest_float("gamma", 0.8, 0.99)
weight_scale = 0.000132  # trial.suggest_loguniform("weight_scale", 5e-5, 0.2)  # 0.005

bias_scale = 0.1  # trial.suggest_loguniform("bias_scale", 5e-4, 0.1)  # 0.005
alpha_relu_actor = 0.1  # trial.suggest_loguniform("alpha_relu_actor", 0.0001, 0.5)  # 0.005
alpha_relu_critic = 0.1  # trial.suggest_loguniform("alpha_relu_critic", 0.0001, 0.5)  # 0.005

batch_size = 1024  # trial.suggest_int("batch_size", 32, 1024)  # 128
buffer_size = int(1e6)  # trial.suggest_int("buffer_size", 10, 1000000)  # 128

actor_hidden_size = 131  # trial.suggest_int("actor_hidden_size", 10, 200)  # 100  # Using LeakyReLU
actor_number_layers = 3  # trial.suggest_int("actor_number_layers", 1, 5)

critic_hidden_size = 324  # trial.suggest_int("critic_hidden_size", 10, 500)  # 100
critic_number_layers = 3  # trial.suggest_int("critic_number_layers", 1, 4)

n_trail = str(9999999)
use_gamma_in_rew = 1
noise_var = 0.012  # trial.suggest_loguniform("noise_var", 0.01, 1)  # 2
# min var, action noise is reduced to (depends on noise_var)
noise_var_min = 0.0013  # trial.suggest_loguniform("noise_var_min", 0.0000001, 2)
# min var, action noise is reduced to (depends on training_episode_length)
noise_steps_annealing = int(
    0.25 * number_learning_steps)  # trail.suggest_int("noise_steps_annealing", int(0.1 * number_learning_steps),
# number_learning_steps)
noise_theta = 1.74  # trial.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU
error_exponent = 0.5  # trial.suggest_loguniform("error_exponent", 0.01, 4)

training_episode_length = 2000  # trial.suggest_int("training_episode_length", 200, 5000)  # 128
learning_starts = 0.32  # trial.suggest_loguniform("learning_starts", 0.1, 2)  # 128
tau = 0.005  # trial.suggest_loguniform("tau", 0.0001, 0.2)  # 2

learning_rate = linear_schedule(initial_value=learning_rate, final_value=learning_rate * final_lr,
                                t_start=t_start,
                                t_end=t_end,
                                total_timesteps=number_learning_steps)

rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
             use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
             i_nom=net['inverter1'].i_nom)

env = gym.make('experiments.hp_tune.env:vctrl_single_inv_train-v0',
               reward_fun=rew.rew_fun_dq0,
               abort_reward=-1,
               obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                           'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                           'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
               )

env = FeatureWrapper(env, number_of_features=11, training_episode_length=training_episode_length,
                     recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                     antiwindup_weight=antiwindup_weight)

# todo: Upwnscale actionspace - lessulgy possible? Interaction pytorch...
env.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

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

model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log='test',
             # model = myDDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
             policy_kwargs=policy_kwargs,
             learning_rate=learning_rate, buffer_size=buffer_size,
             learning_starts=int(learning_starts * training_episode_length),
             batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
             train_freq=(1, "episode"), gradient_steps=- 1,
             optimize_memory_usage=False,
             create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

# Adjust network -> maybe change to Costume net like https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# adn scale weights and biases
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

# todo: Downscale actionspace - lessulgy possible? Interaction pytorch...
env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

for ex_run in range(10):
    # todo: Upwnscale actionspace - lessulgy possible? Interaction pytorch...
    env.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log='test',
                 # model = myDDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=buffer_size,
                 learning_starts=int(learning_starts * training_episode_length),
                 batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
                 train_freq=(1, "episode"), gradient_steps=- 1,
                 optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    # Adjust network -> maybe change to Costume net like https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    # adn scale weights and biases
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

    # todo: Downscale actionspace - lessulgy possible? Interaction pytorch...
    env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

    I_list0 = []
    I_list1 = []
    I_list2 = []
    I_action_list0 = []
    I_action_list1 = []
    I_action_list2 = []
    P_list0 = []
    P_list1 = []
    P_list2 = []
    return_sum = 0.0
    obs = env.reset()
    while True:

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # I_list.append(env.used_I)
        I_list0.append(env.used_I[0])
        I_list1.append(env.used_I[1])
        I_list2.append(env.used_I[2])
        P_list0.append(env.used_P[0])
        P_list1.append(env.used_P[1])
        P_list2.append(env.used_P[2])
        I_action_list0.append(env.used_I_action[0])
        I_action_list1.append(env.used_I_action[1])
        I_action_list2.append(env.used_I_action[2])
        env.render()
        return_sum += rewards
        if done:
            break
    env.close()

    plt.plot(P_list0, 'b')
    plt.plot(P_list1, 'r')
    plt.plot(P_list2, 'g')
    # plt.xlim([0, 0.01])
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("action_P")
    plt.title('Test')
    plt.show()

    plt.plot(I_list0, 'b')
    plt.plot(I_list1, 'r')
    plt.plot(I_list2, 'g')
    # plt.xlim([0, 0.01])
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Integratorzustand")
    plt.title('Test')
    plt.show()

    plt.plot(I_action_list0, 'b')
    plt.plot(I_action_list1, 'r')
    plt.plot(I_action_list2, 'g')
    # plt.xlim([0, 0.01])
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("action_I")
    plt.title('Test')
    plt.show()
