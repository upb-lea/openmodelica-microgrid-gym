import platform
from typing import Union

import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.GEM.util.config import cfg
from experiments.hp_tune.env.vctrl_single_inv import net
from openmodelica_microgrid_gym.util import Fastqueue


class BaseWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = 5000000,
                 recorder=None, n_trail="", gamma=0,
                 number_learing_steps=500000, number_past_vals=0):
        """
        Base Env Wrapper to add features to the env-observations and adds information to env.step output which can be
        used in case of an continuing (non-episodic) task to reset the environment without being terminated by done

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
        self.n_episode = 0
        self.reward_episode_mean = []
        self.i_d_mess = []
        self.i_q_mess = []
        self.i_d_ref = []
        self.i_q_ref = []
        self.action_d = []
        self.action_q = []
        self.n_trail = n_trail
        self.used_P = np.zeros(self.action_space.shape)
        self.gamma = gamma
        self.number_learing_steps = number_learing_steps
        self.delay_queues = [Fastqueue(1, 2) for _ in range(number_past_vals)]

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """

        obs, reward, done, info = super().step(action)
        reward = reward * (1 - self.gamma)
        # super().render()

        self._n_training_steps += 1

        if cfg['loglevel'] == 'train':
            self.i_d_mess.append(np.float64(obs[0]))
            self.i_q_mess.append(np.float64(obs[1]))
            self.i_d_ref.append(np.float64(obs[2]))
            self.i_q_ref.append(np.float64(obs[3]))
            self.action_d.append(np.float64(action[0]))
            self.action_q.append(np.float64(action[1]))

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True
            super().close()

        if done:
            self.reward_episode_mean.append(np.mean(self.rewards))
            self.n_episode += 1

            if cfg['loglevel'] == 'train':
                episode_data = {"Name": "On_Training",
                                "Episode_number": self.n_episode,
                                "Episode_length": self._n_training_steps,
                                "i_d_mess": self.i_d_mess,
                                "i_q_mess": self.i_q_mess,
                                "i_d_ref": self.i_d_ref,
                                "i_q_ref": self.i_q_ref,
                                'action_d': self.action_d,
                                'action_q': self.action_q,
                                "Rewards": self.rewards,
                                "Node": platform.uname().node,
                                "Trial number": self.n_trail,
                                "Database name": cfg['STUDY_NAME'],
                                "Reward function": 'MRE'
                                }

                """
                add here "model_params_change": callback.params_change, from training_recorder?
                """

                # stores data locally to cfg['meas_data_folder'], needs to be grept / transfered via reporter to mongodc
                self.recorder.save_to_json('Trial_number_' + self.n_trail, episode_data)

                # clear lists
                self.i_d_mess = []
                self.i_q_mess = []
                self.i_d_ref = []
                self.i_q_ref = []
                self.action_d = []
                self.action_q = []
        """
        Features
        """
        error = (obs[2:4] - obs[0:2]) / 2  # control error: v_setpoint - v_mess
        obs = np.append(obs, error)
        obs = np.append(obs, self.used_P)
        obs_delay_array = self.shift_and_append(obs[0:2])
        obs = np.append(obs, obs_delay_array)

        # todo efficiency?
        self.used_P = np.copy(action)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """

        [x.clear() for x in self.delay_queues]
        obs = super().reset()

        if cfg['loglevel'] == 'train':
            self.i_d_mess.append(np.float64(obs[0]))
            self.i_q_mess.append(np.float64(obs[1]))
            self.i_d_ref.append(np.float64(obs[2]))
            self.i_q_ref.append(np.float64(obs[3]))
            self.action_d.append(np.float64(0))
            self.action_q.append(np.float64(0))

        self._n_training_steps = 0
        self.used_P = np.zeros(self.action_space.shape)

        # if cfg['loglevel'] == 'train':
        # print("Log Data for Taining in Basewrapper L2xx?")

        """
        Features
        """
        # SP wir an den State gehangen!
        error = (obs[2:4] - obs[0:2]) / 2  # control error: v_setpoint - v_mess
        obs = np.append(obs, error)
        obs = np.append(obs, self.used_P)
        obs_delay_array = self.shift_and_append(obs[0:2])
        obs = np.append(obs, obs_delay_array)

        return obs

    def shift_and_append(self, obs):
        """
        Takes the observation and shifts throught the queue
        every queue output is added to total obs
        """
        obs_delay_array = np.array([])
        obs_temp = obs
        for queue in self.delay_queues:
            obs_temp = queue.shift(obs_temp)
            obs_delay_array = np.append(obs_delay_array, obs_temp)

        return obs_delay_array


class FeatureWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = 5000000,
                 recorder=None, n_trail="", integrator_weight=net.ts, antiwindup_weight=net.ts, gamma=0,
                 penalty_I_weight=1, penalty_P_weight=1, t_start_penalty_I=0, t_start_penalty_P=0,
                 number_learing_steps=500000):  # , use_past_vals=False, number_past_vals=0):
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
        self.i_d_mess = []
        self.i_q_mess = []
        self.i_d_ref = []
        self.i_q_ref = []
        self.action_d = []
        self.action_q = []
        self.n_episode = 0
        self.reward_episode_mean = []
        self.n_trail = n_trail
        self.integrator_sum = np.zeros(self.action_space.shape)
        self.integrator_weight = integrator_weight
        self.antiwindup_weight = antiwindup_weight
        self.used_P = np.zeros(self.action_space.shape)
        self.used_I = np.zeros(self.action_space.shape)
        self.gamma = gamma
        self.penalty_I_weight = penalty_I_weight
        self.penalty_P_weight = penalty_P_weight
        self.t_start_penalty_I = t_start_penalty_I
        self.t_start_penalty_P = t_start_penalty_P
        self.number_learing_steps = number_learing_steps
        self.integrator_sum_list0 = []
        self.integrator_sum_list1 = []
        self.action_P0 = []
        self.action_P1 = []
        self.action_I0 = []
        self.action_I1 = []


    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """
        action_P = action[0:2]
        action_I = action[2:4]

        self.integrator_sum += action_I * self.integrator_weight

        action_PI = action_P + self.integrator_sum

        # check if m_abc will be clipped
        if np.any(abs(action_PI) > 1):

            clipped_action = np.clip(action_PI, -1, 1)

            delta_action = clipped_action - action_PI
            # if, reduce integrator by clipped delta
            # action_delta = abc_to_dq0(delta_action, self.env.net.components[0].phase)
            self.integrator_sum += delta_action * self.antiwindup_weight

            """
            clip_reward = np.clip(np.sum(np.abs(delta_action) * \
                                         (-1 / (self.env.net.components[0].v_lim / self.env.net.components[
                                             0].v_DC))) / 3 * (1 - self.gamma),
                                  -1, 0)
            """
            clip_reward = 0

            action_PI = clipped_action

        else:
            clip_reward = 0

        obs, reward, done, info = super().step(action_PI)
        reward = reward + clip_reward
        reward = reward * (1 - self.gamma)

        #super().render()

        integrator_penalty = np.sum(-((np.abs(action_I)) ** 0.5)) * (1 - self.gamma) / 3
        # action_P_penalty = - np.sum((np.abs(action_P - self.used_P)) ** 0.5) * (1 - self.gamma) / 3
        action_P_penalty = np.sum(-((np.abs(action_P)) ** 0.5)) * (1 - self.gamma) / 3

        # reward_weight is = 1

        if self.total_steps > self.t_start_penalty_I:
            penalty_I_weight_scale = 1 / (self.t_start_penalty_I - self.number_learing_steps) * self.total_steps - \
                                     self.number_learing_steps / (self.t_start_penalty_I - self.number_learing_steps)

        else:
            penalty_I_weight_scale = 1

        if self.total_steps > self.t_start_penalty_P:
            penalty_P_weight_scale = 1 / (self.t_start_penalty_P - self.number_learing_steps) * self.total_steps - \
                                     self.number_learing_steps / (self.t_start_penalty_P - self.number_learing_steps)

        else:

            penalty_P_weight_scale = 1

        reward = (reward + (self.penalty_I_weight * penalty_I_weight_scale) * integrator_penalty
                  + self.penalty_P_weight * penalty_P_weight_scale * action_P_penalty) \
                 / (1 + self.penalty_I_weight * penalty_I_weight_scale + self.penalty_P_weight * penalty_P_weight_scale)

        self._n_training_steps += 1

        if cfg['loglevel'] == 'train':
            self.i_d_mess.append(np.float64(obs[0]))
            self.i_q_mess.append(np.float64(obs[1]))
            self.i_d_ref.append(np.float64(obs[2]))
            self.i_q_ref.append(np.float64(obs[3]))
            self.action_d.append(np.float64(action[0]))
            self.action_q.append(np.float64(action[1]))
            self.integrator_sum_list0.append(self.integrator_sum[0])
            self.integrator_sum_list1.append(self.integrator_sum[1])
            self.action_P0.append(np.float64(action_P[0]))
            self.action_P1.append(np.float64(action_P[1]))
            self.action_I0.append(np.float64(action_I[0]))
            self.action_I1.append(np.float64(action_I[1]))

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True
            super().close()

        if done:
            self.reward_episode_mean.append(np.mean(self.rewards))
            self.n_episode += 1

            if cfg['loglevel'] == 'train':
                episode_data = {"Name": "On_Training",
                                "Episode_number": self.n_episode,
                                "Episode_length": self._n_training_steps,
                                "i_d_mess": self.i_d_mess,
                                "i_q_mess": self.i_q_mess,
                                "i_d_ref": self.i_d_ref,
                                "i_q_ref": self.i_q_ref,
                                'action_d': self.action_d,
                                'action_q': self.action_q,
                                "Rewards": self.rewards,
                                "Node": platform.uname().node,
                                "Trial number": self.n_trail,
                                "Database name": cfg['STUDY_NAME'],
                                "Reward function": 'rew.rew_fun_dq0',
                                'Integrator0': self.integrator_sum_list0,
                                'Integrator1': self.integrator_sum_list1,
                                'actionP0': self.action_P0,
                                'actionP1': self.action_P1,
                                'actionI0': self.action_I0,
                                'actionI1': self.action_I1,
                                }

                """
                add here "model_params_change": callback.params_change, from training_recorder?
                """

                # stores data locally to cfg['meas_data_folder'], needs to be grept / transfered via reporter to mongodc
                self.recorder.save_to_json('Trial_number_' + self.n_trail, episode_data)

                # clear lists
                self.i_d_mess = []
                self.i_q_mess = []
                self.i_d_ref = []
                self.i_q_ref = []
                self.action_d = []
                self.action_q = []

            # if self._n_training_steps > 500:
            # super().close()

        """
        Features
        """
        error = (obs[2:4] - obs[0:2]) / 2  # control error: v_setpoint - v_mess
        obs = np.append(obs, error)
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)

        self.used_P = np.copy(action_P)
        self.used_I = np.copy(self.integrator_sum)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """
        obs = super().reset()

        if cfg['loglevel'] == 'train':
            self.i_d_mess.append(np.float64(obs[0]))
            self.i_q_mess.append(np.float64(obs[1]))
            self.i_d_ref.append(np.float64(obs[2]))
            self.i_q_ref.append(np.float64(obs[3]))
            self.action_d.append(np.float64(0))
            self.action_q.append(np.float64(0))

        self._n_training_steps = 0
        self.integrator_sum = np.zeros(self.action_space.shape)
        self.used_P = np.zeros(self.action_space.shape)
        self.used_I = np.zeros(self.action_space.shape)

        """"
        Features
        """
        # SP wir an den State gehangen!
        error = (obs[2:4] - obs[0:2]) / 2  # control error: v_setpoint - v_mess
        obs = np.append(obs, error)
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)
        # obs = np.append(obs, self.used_action)

        return obs




class FeatureWrapper_pastVals(FeatureWrapper):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = 500000,
                 recorder=None, n_trail="", integrator_weight=net.ts, antiwindup_weight=net.ts, gamma=0,
                 penalty_I_weight=1, penalty_P_weight=1, t_start_penalty_I=0, t_start_penalty_P=0,
                 number_learing_steps=500000, number_past_vals=10):
        """
        Env Wrapper which adds the number_past_vals voltage ([3:6]!!!) observations to the observations.
        Initialized with zeros!
        """
        super().__init__(env, number_of_features, training_episode_length,
                         recorder, n_trail, integrator_weight, antiwindup_weight, gamma,
                         penalty_I_weight, penalty_P_weight, t_start_penalty_I, t_start_penalty_P,
                         number_learing_steps)

        # self.observation_space = gym.spaces.Box(
        #    low=np.full(env.observation_space.shape[0] + number_of_features, -np.inf),
        #    high=np.full(env.observation_space.shape[0] + number_of_features, np.inf))

        self.delay_queues = [Fastqueue(1, 2) for _ in range(number_past_vals)]

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        obs, reward, done, info = super().step(action)
        obs_delay_array = self.shift_and_append(obs[0:2])
        obs = np.append(obs, obs_delay_array)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """

        [x.clear() for x in self.delay_queues]
        obs = super().reset()
        obs_delay_array = self.shift_and_append(obs[0:2])
        obs = np.append(obs, obs_delay_array)

        return obs

    def shift_and_append(self, obs):
        """
        Takes the observation and shifts throught the queue
        every queue output is added to total obs
        """
        obs_delay_array = np.array([])
        obs_temp = obs
        for queue in self.delay_queues:
            obs_temp = queue.shift(obs_temp)
            obs_delay_array = np.append(obs_delay_array, obs_temp)

        return obs_delay_array




