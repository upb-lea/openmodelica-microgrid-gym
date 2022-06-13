import platform
import time
from typing import Union

import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.type_aliases import GymStepReturn

# from agents.my_ddpg import myDDPG
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.util.config import cfg
from experiments.hp_tune.env.vctrl_single_inv import net  # , folder_name
from experiments.hp_tune.util.recorder import Recorder
from openmodelica_microgrid_gym.env import PlotTmpl
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
        self.action_P0 = []
        self.action_P1 = []
        self.action_P2 = []
        self.action_I0 = []
        self.action_I1 = []
        self.action_I2 = []

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """
        self.integrator_sum += action * self.integrator_weight

        self.action_P0.append(action[0])
        self.action_P1.append(action[1])
        self.action_P2.append(action[2])
        self.action_I0.append(self.integrator_sum[0])
        self.action_I1.append(self.integrator_sum[1])
        self.action_I2.append(self.integrator_sum[2])

        action_PI = action  # + self.integrator_sum

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
                # mongo_recorder.save_to_json('Trial_number_' + self.n_trail, episode_data)

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
        # obs = np.append(obs, delta_i_lim_i_phasor)

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
        self.integrator_sum = np.zeros(self.action_space.shape)

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


def run_testcase_DDPG(gamma, integrator_weight, antiwindup_weight, model_path, error_exponent=0.5, use_gamma_in_rew=1,
                      n_trail=50000):
    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

    model = DDPG.load(model_path + f'model.zip')

    ####### Run Test #########
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
                        obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                    'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                    'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                        )
    env_test = FeatureWrapper(env_test, number_of_features=8, integrator_weight=integrator_weight,
                              antiwindup_weight=antiwindup_weight)
    obs = env_test.reset()
    phase_list = []
    phase_list.append(env_test.env.net.components[0].phase)

    rew_list = []
    a0 = []
    a1 = []
    a2 = []
    v_d = []
    v_q = []
    v_0 = []

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)
        phase_list.append(env_test.env.net.components[0].phase)
        a0.append(np.float64(action[0]))
        a1.append(np.float64(action[1]))
        a2.append(np.float64(action[2]))
        v_a = env_test.history.df['lc.capacitor1.v'].iloc[-1]
        v_b = env_test.history.df['lc.capacitor2.v'].iloc[-1]
        v_c = env_test.history.df['lc.capacitor3.v'].iloc[-1]

        v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), env_test.env.net.components[0].phase)

        v_d.append(v_dq0[0])
        v_q.append(v_dq0[1])
        v_0.append(v_dq0[2])

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
                           "Action0": a0,
                           "Action1": a1,
                           "Action2": a2,
                           "Phase": phase_list,
                           "Node": platform.uname().node,
                           "End time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime()),
                           "Reward function": 'rew.rew_fun_dq0',
                           "Trial number": n_trail,
                           "Database name": folder_name,
                           "Info": "Delay, obs=[v_mess,sp_dq0, i_mess_dq0, error_mess_sp, last_action, sin/cos(phase)]; "
                                   "Reward = MRE, PI-Approch using AntiWindUp"
                                   "without abort! (risk=0 manullay in env); only voltage taken into account in reward!"}

    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    # ax1, ax2 = ax.flatten()

    plt.plot(env_test.action_P0)
    plt.plot(env_test.action_P1)
    plt.plot(env_test.action_P2)
    plt.xlabel("")
    plt.ylabel("action_P")
    plt.title('Test')
    plt.show()

    plt.plot(env_test.action_I0)
    plt.plot(env_test.action_I1)
    plt.plot(env_test.action_I2)
    plt.xlabel("")
    plt.ylabel("action_I")
    plt.title('Test')
    plt.show()

    plt.plot(v_d)
    plt.plot(v_q)
    plt.plot(v_0)
    plt.xlabel("")
    plt.ylabel("v_dq0")
    plt.title('Test')
    plt.show()
    # Add v-&i-measurements
    test_after_training.update({env_test.viz_col_tmpls[j].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                })
    test_after_training.update({env_test.viz_col_tmpls[2].vars[i].replace(".", "_"): env_test.history[
        env_test.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                })

    # mongo_recorder.save_to_json('Trial_number_' + n_trail, test_after_training)

    return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)


run_testcase_DDPG(gamma=0.94713819942184, integrator_weight=2.3448684965845657e-06,
                  antiwindup_weight=0.0035792826486838116,
                  model_path='experiments/hp_tune/trained_models/study_10_run_954/',
                  error_exponent=0.5, use_gamma_in_rew=1, n_trail=50000)
