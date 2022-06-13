import platform
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from experiments.P10.env.env_wrapper_P10 import FeatureWrapper, FeatureWrapper_pastVals, BaseWrapper
from experiments.P10.env.rewards_P10 import Reward
from experiments.P10.env.vctrl_single_inv_P10 import net  # , folder_name
from experiments.P10.util.config import cfg
from experiments.P10.util.recorder_P10 import Recorder

# np.random.seed(0)

folder_name = cfg['STUDY_NAME']
node = platform.uname().node

# mongo_recorder = Recorder(database_name=folder_name)
mongo_recorder = Recorder(node=node,
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne or locally as json to cfg[meas_data_folder]


def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale, alpha_relu_actor,
                        batch_size,
                        actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                        alpha_relu_critic,
                        noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                        training_episode_length, buffer_size,  # learning_starts,
                        tau, number_learning_steps, integrator_weight, antiwindup_weight,
                        penalty_I_weight, penalty_P_weight,
                        train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer, n_trail,
                        number_past_vals=0):
    if node not in cfg['lea_vpn_nodes']:
        # assume we are on pc2
        pc2_log_path = cfg['pc2_logpath']
        log_path = f'{pc2_log_path}/{folder_name}/{n_trail}/'
    else:
        log_path = f'{folder_name}/{n_trail}/'

    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

    env = gym.make('experiments.P10.env:vctrl_single_inv_train-v0',
                   reward_fun=rew.rew_fun_dq0,
                   abort_reward=-1,
                   obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                               'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                               'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                   # , 'r_load.resistor1.i', 'r_load.resistor2.i', 'r_load.resistor3.i']
                   )

    if cfg['env_wrapper'] == 'past':
        env = FeatureWrapper_pastVals(env, number_of_features=9 + number_past_vals * 3,
                                      training_episode_length=training_episode_length,
                                      recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                                      antiwindup_weight=antiwindup_weight, gamma=gamma,
                                      penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                      t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                      number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)


    elif cfg['env_wrapper'] == 'no-I-term':
        env = BaseWrapper(env, number_of_features=6 + number_past_vals * 3,
                          training_episode_length=training_episode_length,
                          recorder=mongo_recorder, n_trail=n_trail, gamma=gamma,
                          number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)

    else:
        env = FeatureWrapper(env, number_of_features=11, training_episode_length=training_episode_length,
                             recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                             antiwindup_weight=antiwindup_weight, gamma=gamma,
                             penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                             t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                             number_learing_steps=number_learning_steps)  # , use_past_vals=True, number_past_vals=30)

    # todo: Upwnscale actionspace - lessulgy possible? Interaction pytorch...
    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
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
    print(optimizer)
    if optimizer == 'SGD':
        used_optimzer = th.optim.SGD
    elif optimizer == 'RMSprop':
        used_optimzer = th.optim.RMSprop
    # elif optimizer == 'LBFGS':
    # needs in step additional argument
    #    used_optimzer = th.optim.LBFGS
    else:
        used_optimzer = th.optim.Adam

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size] * actor_number_layers
                                                                      , qf=[critic_hidden_size] * critic_number_layers),
                         optimizer_class=used_optimzer)

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=log_path,
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=buffer_size,
                 # learning_starts=int(learning_starts * training_episode_length),
                 batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
                 train_freq=(train_freq, train_freq_type), gradient_steps=- 1,
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
    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
        env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

    # start training
    model.learn(total_timesteps=number_learning_steps)

    # Log Train-info data
    train_data = {#"Name": "After_Training",
                  "Mean_eps_env_reward_raw": env.reward_episode_mean,
                  "Mean_eps_reward_sum": env.reward_plus_addon_episode_mean,
                  #"Trial number": n_trail,
                  #"Database name": folder_name,
                  #"Sum_eps_reward": env.get_episode_rewards()
                  }
    #mongo_recorder.save_to_json('Trial_number_' + n_trail, train_data)
    mongo_recorder.save_local_to_pkl('traing_rewards_Trial_number_' + n_trail, train_data, n_trail)

    model.save(log_path + f'model.zip')

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
                        # , 'r_load.resistor1.i', 'r_load.resistor2.i', 'r_load.resistor3.i']
                        )

    if cfg['env_wrapper'] == 'past':
        env_test = FeatureWrapper_pastVals(env_test, number_of_features=9 + number_past_vals * 3,
                                           integrator_weight=integrator_weight,
                                           recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                                           gamma=1, penalty_I_weight=0,
                                           penalty_P_weight=0, number_past_vals=number_past_vals,
                                           training_episode_length=training_episode_length, )

    elif cfg['env_wrapper'] == 'no-I-term':
        env_test = BaseWrapper(env_test, number_of_features=6 + number_past_vals * 3,
                               training_episode_length=training_episode_length,
                               recorder=mongo_recorder, n_trail=n_trail, gamma=gamma,
                               number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)

    else:
        env_test = FeatureWrapper(env_test, number_of_features=11, integrator_weight=integrator_weight,
                                  recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                                  gamma=1, penalty_I_weight=0,
                                  penalty_P_weight=0,
                                  training_episode_length=training_episode_length, )  # , use_past_vals=True, number_past_vals=30)
    # using gamma=1 and rew_weigth=3 we get the original reward from the env without penalties
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
    va = []
    vb = []
    vc = []
    v_ref0 = []
    v_ref1 = []
    v_ref2 = []
    ia = []
    ib = []
    ic = []
    R_load = []

    for step in range(env_test.max_episode_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)
        if cfg['loglevel'] in ['train', 'test']:
            phase_list.append(env_test.env.net.components[0].phase)
            aP0.append(np.float64(action[0]))
            aP1.append(np.float64(action[1]))
            aP2.append(np.float64(action[2]))
            if cfg['env_wrapper'] not in ['no-I-term']:
                aI0.append(np.float64(action[3]))
                aI1.append(np.float64(action[4]))
                aI2.append(np.float64(action[5]))
                integrator_sum0.append(np.float64(env_test.integrator_sum[0]))
                integrator_sum1.append(np.float64(env_test.integrator_sum[1]))
                integrator_sum2.append(np.float64(env_test.integrator_sum[2]))

        if rewards == -1 and not limit_exceeded_in_test:# and env_test.rew[-1]:
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
            # if step % cfg['train_episode_length'] == 0 and step != 0:
            if cfg['loglevel'] in ['train', 'test']:
                va.extend(env_test.history[env_test.viz_col_tmpls[0].vars[0]].copy().values.tolist())
                vb.extend(env_test.history[env_test.viz_col_tmpls[0].vars[1]].copy().values.tolist())
                vc.extend(env_test.history[env_test.viz_col_tmpls[0].vars[2]].copy().values.tolist())
                v_ref0.extend(env_test.history[env_test.viz_col_tmpls[0].vars[3]].copy().values.tolist())
                v_ref1.extend(env_test.history[env_test.viz_col_tmpls[0].vars[4]].copy().values.tolist())
                v_ref2.extend(env_test.history[env_test.viz_col_tmpls[0].vars[5]].copy().values.tolist())
                ia.extend(env_test.history[env_test.viz_col_tmpls[1].vars[0]].copy().values.tolist())
                ib.extend(env_test.history[env_test.viz_col_tmpls[1].vars[1]].copy().values.tolist())
                ic.extend(env_test.history[env_test.viz_col_tmpls[1].vars[2]].copy().values.tolist())
                R_load.extend(env_test.history[env_test.viz_col_tmpls[2].vars[1]].copy().values.tolist())

            env_test.close()
            obs = env_test.reset()
            if cfg['loglevel'] in ['train', 'test']:
                phase_list.append(env_test.env.net.components[0].phase)

        if done:
            env_test.close()
            # print(limit_exceeded_in_test)
            break

    ts = time.gmtime()
    """
    test_after_training_config = {"Name": "Test_Reward",
                                  "time": ts,
                                  #"Reward": rew_list,
                                  "Return": (return_sum / env_test.max_episode_steps),
                                  "Trial number": n_trail,
                                  "Database name": folder_name,
                                  "Node": platform.uname().node,
                                  "End time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())}
    """

    reward_test_after_training = {"Reward": rew_list}

    if cfg['loglevel'] in ['train', 'test', 'setting']:
        #mongo_recorder.save_to_json('Test_setting_Trial_number_' + n_trail, test_after_training_config, n_trail)
        mongo_recorder.save_local_to_pkl('Test_reward_Trial_number_' + n_trail, reward_test_after_training, n_trail)

    if cfg['loglevel'] in ['train', 'test']:
        test_after_training = {"Name": "Test",
                               "time": ts,
                               "Reward": rew_list,
                               "lc_capacitor1_v": va,
                               "lc_capacitor2_v": vb,
                               "lc_capacitor3_v": vc,
                               "inverter1_v_ref_0": v_ref0,
                               "inverter1_v_ref_1": v_ref1,
                               "inverter1_v_ref_2": v_ref2,
                               "lc_inductor1_i": ia,
                               "lc_inductor2_i": ib,
                               "lc_inductor3_i": ic,
                               "r_load_resistor1_R": R_load,
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

        """
        In new testenv not used, because then only the last episode is stored
        """
        # Add v-&i-measurements
        test_after_training.update({env_test.viz_col_tmpls[j].vars[i].replace(".", "_"): env_test.history[
            env_test.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
                                    })
        test_after_training.update({env_test.viz_col_tmpls[2].vars[i].replace(".", "_"): env_test.history[
            env_test.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
                                    })

        mongo_recorder.save_to_json('Trial_number_' + n_trail, test_after_training, n_trail)

    return (return_sum / env_test.max_episode_steps)
