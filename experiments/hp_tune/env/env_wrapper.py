import platform
from typing import Union

import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.hp_tune.env.vctrl_single_inv import net
from experiments.hp_tune.util.config import cfg
from openmodelica_microgrid_gym.util import abc_to_alpha_beta, dq0_to_abc, abc_to_dq0, Fastqueue


class FeatureWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = np.inf,
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
        self.gamma = gamma
        self.penalty_I_weight = penalty_I_weight
        self.penalty_P_weight = penalty_P_weight
        self.t_start_penalty_I = t_start_penalty_I
        self.t_start_penalty_P = t_start_penalty_P
        self.number_learing_steps = number_learing_steps


    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """
        action_P = action[0:3]
        action_I = action[3:6]

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

        if len(obs) > 9:
            # ASSUME  THAT LOADCURRENT is included!
            obs[9:12] = obs[9:12] / net['inverter1'].i_lim

        super().render()

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

        if self._n_training_steps % round(self.training_episode_length / 10) == 0:
            self.env.on_episode_reset_callback()

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True
            super().close()

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
                                "Database name": cfg['STUDY_NAME'],
                                "Reward function": 'rew.rew_fun_dq0',
                                }

                """
                add here "model_params_change": callback.params_change, from training_recorder?
                """

                # stores data locally to cfg['meas_data_folder'], needs to be grept / transfered via reporter to mongodc
                self.recorder.save_to_json('Trial_number_' + self.n_trail, episode_data)

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

        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)
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

        if len(obs) > 9:
            # ASSUME  THAT LOADCURRENT is included!
            obs[9:12] = obs[9:12] / net['inverter1'].i_lim

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

        # obs = np.append(obs, delta_i_lim_i_phasor)
        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)
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


class FeatureWrapper_pastVals(FeatureWrapper):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = np.inf,
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

        self.delay_queues = [Fastqueue(1, 3) for _ in range(number_past_vals)]

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        obs, reward, done, info = super().step(action)
        obs_delay_array = self.shift_and_append(obs[3:6])
        obs = np.append(obs, obs_delay_array)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """

        [x.clear() for x in self.delay_queues]
        obs = super().reset()
        obs_delay_array = self.shift_and_append(obs[3:6])
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


class FeatureWrapper_futureVals(FeatureWrapper):
    pass
