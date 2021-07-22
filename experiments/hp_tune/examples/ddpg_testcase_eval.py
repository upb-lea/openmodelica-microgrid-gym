import platform
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
from tqdm import tqdm
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# from agents.my_ddpg import myDDPG
from experiments.hp_tune.env.env_wrapper import FeatureWrapper
from experiments.hp_tune.env.rewards import Reward
from experiments.hp_tune.env.vctrl_single_inv import net  # , folder_name
from experiments.hp_tune.util.config import cfg
from experiments.hp_tune.util.recorder import Recorder

import pandas as pd

# np.random.seed(0)
from openmodelica_microgrid_gym.util import abc_to_dq0

folder_name = cfg['STUDY_NAME']
node = platform.uname().node

# mongo_recorder = Recorder(database_name=folder_name)
mongo_recorder = Recorder(node=node,
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne or locally as json to cfg[meas_data_folder]

num_average = 25
max_episode_steps_list = [1000, 5000, 10000, 20000, 50000, 100000]

result_list = []
ret_list = []
mean_list = []
std_list = []
ret_array = np.zeros(num_average)

df = pd.DataFrame()
ret_dict = dict()

# def run_testcase_DDPG(gamma=0.8003175741091463, integrator_weight=0.6618979905182214,
#                  antiwindup_weight=0.9197062574269099,
#                  model_path='experiments/hp_tune/trained_models/study_18_run_6462/',
#                  error_exponent=0.3663140388100587, use_gamma_in_rew=1, n_trail=50000,
#                      actor_number_layers=2, critic_number_layers=3,
#                      alpha_relu_actor=0.04768952563400553,
#                      alpha_relu_critic=0.00019026593928712137
#                      ):
gamma = 0.8003175741091463
integrator_weight = 0.6618979905182214
antiwindup_weight = 0.9197062574269099
model_path = 'experiments/hp_tune/trained_models/study_18_run_6462_new/'
error_exponent = 0.3663140388100587
use_gamma_in_rew = 1
n_trail = 50000
actor_number_layers = 2
critic_number_layers = 3
alpha_relu_actor = 0.04768952563400553
alpha_relu_critic = 0.00019026593928712137

for max_eps_steps in tqdm(range(len(max_episode_steps_list)), desc='steps', unit='step', leave=False):

    for ave_run in tqdm(range(num_average), desc='steps', unit='step', leave=False):

        rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                     use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent,
                     i_lim=net['inverter1'].i_lim,
                     i_nom=net['inverter1'].i_nom)

        ####### Run Test #########
        return_sum = 0.0
        rew.gamma = 0
        # episodes will not abort, if limit is exceeded reward = -1
        rew.det_run = True
        rew.exponent = 0.5  # 1
        limit_exceeded_in_test = False
        limit_exceeded_penalty = 0
        env_test = gym.make('experiments.hp_tune.env:vctrl_single_inv_test-v1',
                            reward_fun=rew.rew_fun_dq0,
                            abort_reward=-1,  # no needed if in rew no None is given back
                            # on_episode_reset_callback=cb.fire  # needed?
                            obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                        'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                        'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'],
                            max_episode_steps=max_episode_steps_list[max_eps_steps]
                            )
        env_test = FeatureWrapper(env_test, number_of_features=11, integrator_weight=integrator_weight,
                                  recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                                  gamma=1, penalty_I_weight=0, penalty_P_weight=0)
        # using gamma=1 and rew_weigth=3 we get the original reward from the env without penalties

        env_test.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

        model = DDPG.load(model_path + f'model.zip')  # , env=env_test)

        count = 0
        for kk in range(actor_number_layers + 1):

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

        env_test.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

        obs = env_test.reset()
        phase_list = []
        phase_list.append(env_test.env.net.components[0].phase)

        rew_list = []
        aP0 = []
        aP1 = []
        aP2 = []
        aI0 = []
        aI1 = []
        aI2 = []
        integrator_sum0 = []
        integrator_sum1 = []
        integrator_sum2 = []
        v_d = []
        v_q = []
        v_0 = []

        for step in range(env_test.max_episode_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env_test.step(action)
            phase_list.append(env_test.env.net.components[0].phase)
            aP0.append(np.float64(action[0]))
            aP1.append(np.float64(action[1]))
            aP2.append(np.float64(action[2]))
            aI0.append(np.float64(action[3]))
            aI1.append(np.float64(action[4]))
            aI2.append(np.float64(action[5]))
            integrator_sum0.append(np.float64(env_test.integrator_sum[0]))
            integrator_sum1.append(np.float64(env_test.integrator_sum[1]))
            integrator_sum2.append(np.float64(env_test.integrator_sum[2]))

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

            if step % 1000 == 0 and step != 0:
                env_test.close()

                obs = env_test.reset()

            # print(rewards)
            if done:
                env_test.close()
                # print(limit_exceeded_in_test)
                break

        ts = time.gmtime()
        test_after_training = {"Name": "Test",
                               "time": ts,
                               "Reward": rew_list,
                               "ActionP0": aP0,
                               "ActionP1": aP1,
                               "ActionP2": aP2,
                               "ActionI0": aI0,
                               "ActionI1": aI1,
                               "ActionI2": aI2,
                               "integrator_sum0": integrator_sum0,
                               "integrator_sum1": integrator_sum1,
                               "integrator_sum2": integrator_sum2,
                               "Phase": phase_list,
                               "Node": platform.uname().node,
                               "End time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime()),
                               "Reward function": 'rew.rew_fun_dq0',
                               "Trial number": n_trail,
                               "Database name": folder_name,
                               "Info": "Delay, obs=[v_mess,sp_dq0, i_mess_dq0, error_mess_sp, last_action, sin/cos(phase),"
                                       "integrator_zustand(delayed!), genutzte Aktion (P-anteil)]; "
                                       "Reward = MRE, PI-Approch using AntiWindUp"
                                       "without abort! (risk=0 manullay in env); only voltage taken into account in reward!"}

        # Add v-&i-measurements
        test_after_training.update({env_test.viz_col_tmpls[j].vars[i].replace(".", "_"): env_test.history[
            env_test.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                    })
        test_after_training.update({env_test.viz_col_tmpls[2].vars[i].replace(".", "_"): env_test.history[
            env_test.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                    })

        # mongo_recorder.save_to_json('Trial_number_' + n_trail, test_after_training)

        plt.plot(v_d)
        plt.plot(v_q)
        plt.plot(v_0)
        plt.xlabel("")
        plt.ylabel("v_dq0")
        plt.title('Test')
        plt.show()

        # return (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)

        ret_list.append((return_sum / env_test.max_episode_steps + limit_exceeded_penalty))
        ret_array[ave_run] = (return_sum / env_test.max_episode_steps + limit_exceeded_penalty)

        # ret_dict[str(ave_run)] = (return_sum / env.max_episode_steps + limit_exceeded_penalty)

        # zipped = zip(max_episode_steps_list[max_eps_steps], ret_list)
        # temp_dict = dict(zipped)
    temp_dict = {str(max_episode_steps_list[max_eps_steps]): ret_list}
    result_list.append(temp_dict)
    # ret_dict.append(zipped)
    # df = df.append(ret_dict)

    mean_list.append(np.mean(ret_array))
    std_list.append(np.std(ret_array))

# df = df.append(temp_list, True)
print(mean_list)
print(std_list)
print(result_list)

results = {
    'Mean': mean_list,
    'Std': std_list,
    'All results': result_list,
    'max_episode_steps_list': max_episode_steps_list
}

df = pd.DataFrame(results)
df.to_pickle("DDPG_study18_best_test_varianz.pkl")
asd = 1

m = np.array(df['Mean'])
s = np.array(df['Std'])
max_episode_steps_list = np.array(df['max_episode_steps_list'])

plt.plot(max_episode_steps_list, m)
plt.fill_between(max_episode_steps_list, m - s, m + s, facecolor='r')
plt.ylabel('Average return +- sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title('')
plt.show()

# plt.plot(max_episode_steps_list, m)
# plt.fill_between(max_episode_steps_list, m - s, m + s, facecolor='r')
plt.errorbar(max_episode_steps_list, m, s, fmt='-o')
plt.ylabel('Average return +- sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title('')
plt.show()

plt.plot(max_episode_steps_list, s)
plt.ylabel('std')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title('')
plt.show()
