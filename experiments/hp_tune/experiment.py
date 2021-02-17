import gym
import numpy as np
import torch as th
import optuna

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# imports net to define reward and executes script to register experiment
from experiments.hp_tune.env.vctrl_single_inv import net, folder_name, max_episode_steps
from experiments.hp_tune.util.record_env import RecordEnvCallback
from experiments.issue51_new.env.rewards import Reward

np.random.seed(0)


def experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                        actor_hidden_size, critic_hidden_size, n_trail):
    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew)

    env = gym.make('experiments.hp_tune.env:vctrl_single_inv-v0',
                   reward_fun=rew.rew_fun,
                   # on_episode_reset_callback=cb.fire  # needed?
                   )

    # toDo: Whatfore?
    env = Monitor(env)

    n_actions = env.action_space.shape[-1]
    noise_var = 0.2
    noise_theta = 5  # stiffness of OU
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                sigma=noise_var * np.ones(n_actions), dt=net.ts)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size], qf=[critic_hidden_size,
                                                                                                  critic_hidden_size,
                                                                                                  critic_hidden_size]))

    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=5000, learning_starts=1024,
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
    plot_callback = EveryNTimesteps(n_steps=1000, callback=RecordEnvCallback(env, model, max_episode_steps))
    model.learn(total_timesteps=2000, callback=[plot_callback])

    # todo: instead: store model(/weights+bias?) to database
    # model.save(f'{folder_name + experiment_name + n_trail}/model.zip')

    return_sum = 0.0
    obs = env.reset()
    rew.gamma = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        return_sum += rewards
        if done:
            env.close()
            break

    return return_sum


def objective(trail):
    learning_rate = 0.0004  # trail.suggest_loguniform("lr", 1e-5, 5e-3)  # 0.0002#
    gamma = 0.7  # trail.suggest_loguniform("gamma", 0.5, 0.99)
    weight_scale = 0.02  # trail.suggest_loguniform("weight_scale", 5e-4, 1)  # 0.005
    batch_size = 128  # trail.suggest_int("batch_size", 32, 1024)  # 128
    actor_hidden_size = 100  # trail.suggest_int("actor_hidden_size", 10, 500)  # 100  # Using LeakyReLU
    n_trail = str(trail.number)
    use_gamma_in_rew = 1
    critic_hidden_size = 100  # trail.suggest_int("critic_hidden_size", 10, 500)  # # Using LeakyReLU

    # toDo:
    # alpha_lRelu = trail.suggest_loguniform("alpha_lRelu", 0.0001, 0.5)  #0.1
    # memory_interval = 1
    # noise_var = 0.2
    # noise_theta = 5  # stiffness of OU
    # weigth_regularizer = 0.5
    # memory_lim = 5000  # = buffersize?
    # warm_up_steps_actor = 2048
    # warm_up_steps_critic = 1024  # learning starts?
    # target_model_update = 1000

    return experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, batch_size,
                               actor_hidden_size, critic_hidden_size, n_trail)


# toDo: postgresql instead of sqlite
study = optuna.create_study(study_name="V-crtl_stochLoad_single_Loadstep_exploring_starts",
                            direction='maximize', storage=f'sqlite:///{folder_name}optuna_data.sqlite3',
                            load_if_exists=True)

study.optimize(objective, n_trials=1)
