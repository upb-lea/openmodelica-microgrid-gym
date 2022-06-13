print('Start script')
import gym
import logging
import os
import platform
import time
from functools import partial

import GPy
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
from stochastic.processes import VasicekProcess
from tqdm import tqdm
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# from agents.my_ddpg import myDDPG
from experiments.P10.env.env_wrapper_P10 import FeatureWrapper, FeatureWrapper_pastVals, BaseWrapper
from experiments.P10.env.rewards_P10 import Reward
from experiments.P10.env.vctrl_single_inv_P10 import net  # , folder_name
from experiments.P10.util.config import cfg
from experiments.P10.util.recorder_P10 import Recorder
import pandas as pd


class CallbackList(list):
    def fire(self, *args, **kwargs):
        for listener in self:
            listener(*args, **kwargs)


show_plots = True
save_results = False
# 2128 ->0; 3125 -> -1, 956-> best
trial = '956'

number_learning_steps = 200000

folder_name = 'experiments/P10/retrain/'
os.makedirs(folder_name, exist_ok=True)
model_path = 'experiments/P10/viz/data/'
node = platform.uname().node
model_name = 'model_'+trial+'.zip'

################DDPG Config Stuff#########################################################################
print('Using model_'+trial+' setting')
if trial == '956':
    actor_number_layers = 2
    alpha_relu_actor = 0.0225049
    alpha_relu_critic = 0.00861825
    antiwindup_weight = 0.350646
    critic_number_layers = 4
    error_exponent = 0.5
    gamma = 0.794337
    integrator_weight = 0.214138
    use_gamma_in_rew = 1
    n_trail = 50001
    number_past_vals = 18
    training_episode_length = 2577
    penalty_I_weight = 1.46321
    penalty_P_weight = 0.662572
    t_start_penalty_I = 100000
    t_start_penalty_P = 100000



mongo_recorder = Recorder(node=node, database_name=folder_name)

current_directory = os.getcwd()

save_folder = os.path.join(current_directory, folder_name)
os.makedirs(save_folder, exist_ok=True)

rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
             use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent,
             i_lim=net['inverter1'].i_lim,
             i_nom=net['inverter1'].i_nom, det_run=True)

####################################DDPG Stuff##############################################

rew.gamma = 0
# episodes will not abort, if limit is exceeded reward = -1
rew.det_run = True
rew.exponent = 0.5  # 1

env = gym.make('experiments.P10.env:vctrl_single_inv_train-v0',
               reward_fun=rew.rew_fun_dq0,
               abort_reward=-1,
               obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                           'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                           'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
               )

env = FeatureWrapper_pastVals(env, number_of_features=9 + number_past_vals * 3,
                              training_episode_length=training_episode_length,
                              recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                              antiwindup_weight=antiwindup_weight, gamma=gamma,
                              penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                              t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                              number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)

env.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

# model2 = DDPG.load(model_path + f'model.zip')  # , env=env_test)
print('Before load')

model = DDPG.load(model_path + f'{model_name}', env=env)

print('After load')

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

env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

###################################################################################
# retrain!
print('Train ' +str(number_learning_steps) + ' steps')
model.learn(total_timesteps=number_learning_steps)
print('Finished ' +str(number_learning_steps) + 'training steps')
model.save(folder_name + f'model' + trial + '_retrained.zip')

# Log Train-info data
train_data = {  # "Name": "After_Training",
    "Mean_eps_env_reward_raw": env.reward_episode_mean,
    "Mean_eps_reward_sum": env.reward_plus_addon_episode_mean,
}
df = pd.DataFrame(train_data)
df.to_pickle(f'{folder_name}/' + 'trainRewards_model' + trial + '_retrained' + ".pkl.bz2")

"""
####### Run Test #########
return_sum = 0.0
rew.gamma = 0
# episodes will not abort, if limit is exceeded reward = -1
rew.det_run = True
rew.exponent = 0.5  # 1
limit_exceeded_in_test = False

env_test = gym.make('experiments.P10.env:vctrl_single_inv_test-v1',
                    reward_fun=rew.rew_fun_dq0,
                    abort_reward=-1,  # no needed if in rew no None is given back
                    # on_episode_reset_callback=cb.fire  # needed?
                    obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                    )

env_test = FeatureWrapper_pastVals(env_test, number_of_features=9 + number_past_vals * 3,
                                   integrator_weight=integrator_weight,
                                   recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                                   gamma=1, penalty_I_weight=0,
                                   penalty_P_weight=0, number_past_vals=number_past_vals,
                                   training_episode_length=training_episode_length, )

obs = env_test.reset()
rew_list = []

for step in range(env_test.max_episode_steps):

    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env_test.step(action)

    if rewards == -1 and not limit_exceeded_in_test:  # and env_test.rew[-1]:
        # Set addidional penalty of -1 if limit is exceeded once in the test case
        limit_exceeded_in_test = True

    if limit_exceeded_in_test:
        # if limit was exceeded once, reward will be kept to -1 till the end of the episode,
        # nevertheless what the agent does
        rewards = -1

    env_test.render()
    return_sum += rewards
    rew_list.append(rewards)
    # print(rewards)

    if step % 1000 == 0 and step != 0:
        env_test.close()
        obs = env_test.reset()

    if done:
        env_test.close()
        break

ts = time.gmtime()

reward_test_after_training = {"Reward": rew_list}

df = pd.DataFrame(reward_test_after_training)
df.to_pickle(f'{folder_name}/' + 'model' + trial + '_retrained' + ".pkl.bz2")

print((return_sum / env_test.max_episode_steps))
"""
