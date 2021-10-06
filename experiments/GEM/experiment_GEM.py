import platform
import time
import gym_electric_motor as gem
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import DDPG
# imports net to define reward and executes script to register experiment
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from experiments.GEM.env.env_wrapper_GEM import FeatureWrapper, FeatureWrapper_pastVals, FeatureWrapper_futureVals, \
    FeatureWrapper_I_controller, BaseWrapper
# from experiments.GEM.env.GEM_env import AppendLastActionWrapper
from experiments.GEM.util.config import cfg
from experiments.GEM.util.recorder_GEM import Recorder

from gym.wrappers import FlattenObservation
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, ConstReferenceGenerator, \
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.motor_dashboard_plots import MeanEpisodeRewardPlot
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gym.core import Wrapper
from gym.spaces import Box, Tuple
from gym_electric_motor.constraints import SquaredConstraint

test_length = 10000
folder_name = cfg['STUDY_NAME']
node = platform.uname().node

# mongo_recorder = Recorder(database_name=folder_name)
mongo_recorder = Recorder(node=node,
                          database_name=folder_name)  # store to port 12001 for ssh data to cyberdyne or locally as json to cfg[meas_data_folder]
Ki_ddpg_combi = 182


class AppendLastActionWrapper(Wrapper):
    """
    The following environment considers the dead time in the real-world motor control systems.
    The real-world system changes its state, while the agent simultaneously calculates the next action based on a
    previously measured observation.
    Therefore, for the agents it seems as if the applied action affects the environment with one step delay
    (with a dead time of one time step).
    As a measure of feature engineering we append the last selected action to the observation of each time step,
    because this action will be the one that is active while the agent has to make the next decision.
    """

    def __init__(self, environment):
        super().__init__(environment)
        # append the action space dimensions to the observation space dimensions
        self.observation_space = Tuple((Box(
            np.concatenate((environment.observation_space[0].low, environment.action_space.low)),
            np.concatenate((environment.observation_space[0].high, environment.action_space.high))
        ), environment.observation_space[1]))

    def step(self, action):
        (state, ref), rew, term, info = self.env.step(action)

        # extend the output state by the selected action
        # state = np.concatenate((state, action))

        return (state, ref), rew, term, info

    def reset(self, **kwargs):
        state, ref = self.env.reset()

        # extend the output state by zeros after reset
        # no action can be appended yet, but the dimension must fit
        # state = np.concatenate((state, np.zeros(self.env.action_space.shape)))

        # set random reference values
        self.env.reference_generator._sub_generators[0]._reference_value = np.random.uniform(-1, 0)
        self.env.reference_generator._sub_generators[1]._reference_value = np.random.uniform(-1, 1)

        return state, ref


class AppendLastActionWrapper_testsetting(AppendLastActionWrapper):

    def reset(self, **kwargs):
        state, ref = super().reset()

        self.env.reference_generator._sub_generators[0]._reference_value = 0.5  # np.random.uniform(-1, 0)
        self.env.reference_generator._sub_generators[1]._reference_value = -0.2  # np.random.uniform(-1, 1)

        return state, ref


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
        log_path = f'/scratch/hpc-prf-reinfl/weber/OMG/{folder_name}/{n_trail}/'
    else:
        log_path = f'{folder_name}/{n_trail}/'

    ####################################################################################################################
    # GEM
    # Define reference generators for both currents of the flux oriented dq frame
    # d current reference is chosen to be constantly at zero to simplify this showcase scenario
    d_generator = ConstReferenceGenerator('i_sd', 0)
    # q current changes dynamically
    q_generator = ConstReferenceGenerator('i_sq', 0)

    # The MultipleReferenceGenerator allows to apply these references simultaneously
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    # Set the electric parameters of the motor
    motor_parameter = dict(
        r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, p=3, j_rotor=0.06
    )

    # Change the motor operational limits (important when limit violations can terminate and reset the environment)
    limit_values = dict(
        i=160 * 1.41,
        omega=12000 * np.pi / 30,
        u=450
    )

    # Change the motor nominal values
    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}

    # Create the environment
    env_train = gem.make(
        # Choose the permanent magnet synchronous motor with continuous-control-set
        'DqCont-CC-PMSM-v0',
        # Pass a class with extra parameters
        visualization=MotorDashboard(
            state_plots=['i_sq', 'i_sd'],
            action_plots='all',
            reward_plot=True,
            additional_plots=[MeanEpisodeRewardPlot()]
        ),
        # Set the mechanical load to have constant speed
        load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),

        # Define which numerical solver is to be used for the simulation
        ode_solver='scipy.solve_ivp',

        # Pass the previously defined reference generator
        reference_generator=rg,

        reward_function=dict(
            # Set weighting of different addends of the reward function
            reward_weights={'i_sq': 1, 'i_sd': 1},
            # Exponent of the reward function
            # Here we use a square root function
            reward_power=0.5,
        ),

        # Define which state variables are to be monitored concerning limit violations
        # Here, only overcurrent will lead to termination
        constraints=(),

        # Consider converter dead time within the simulation
        # This means that a given action will show effect only with one step delay
        # This is realistic behavior of drive applications
        converter=dict(
            dead_time=True,
        ),
        # Set the DC-link supply voltage
        supply=dict(
            u_nominal=400
        ),

        motor=dict(
            # Pass the previously defined motor parameters
            motor_parameter=motor_parameter,

            # Pass the updated motor limits and nominal values
            limit_values=limit_values,
            nominal_values=nominal_values,
        ),
        # Define which states will be shown in the state observation (what we can "measure")
        state_filter=['i_sd', 'i_sq'],  # , 'epsilon'],
    )

    # Now we apply the wrapper defined at the beginning of this script
    env_train = AppendLastActionWrapper(env_train)

    # We flatten the observation (append the reference vector to the state vector such that
    # the environment will output just a single vector with both information)
    # This is necessary for compatibility with kerasRL2
    env_train = FlattenObservation(env_train)

    ####################################################################################################################

    if cfg['env_wrapper'] == 'past':
        env = FeatureWrapper_pastVals(env_train, number_of_features=9 + number_past_vals * 3,
                                      training_episode_length=training_episode_length,
                                      recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                                      antiwindup_weight=antiwindup_weight, gamma=gamma,
                                      penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                      t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                      number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)

    elif cfg['env_wrapper'] == 'future':
        env = FeatureWrapper_futureVals(env_train, number_of_features=9,
                                        training_episode_length=training_episode_length,
                                        recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                                        antiwindup_weight=antiwindup_weight, gamma=gamma,
                                        penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                        t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                        number_learing_steps=number_learning_steps, number_future_vals=10)

    elif cfg['env_wrapper'] == 'I-controller':
        env = FeatureWrapper_I_controller(env_train, number_of_features=12 + number_past_vals * 3,
                                          # including integrator_sum
                                          training_episode_length=training_episode_length,
                                          recorder=mongo_recorder, n_trail=n_trail, integrator_weight=integrator_weight,
                                          antiwindup_weight=antiwindup_weight, gamma=gamma,
                                          penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                          t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                          number_learing_steps=number_learning_steps, Ki=Ki_ddpg_combi,
                                          number_past_vals=number_past_vals)

    elif cfg['env_wrapper'] == 'no-I-term':
        env = BaseWrapper(env_train, number_of_features=2 + number_past_vals * 2,
                          training_episode_length=training_episode_length,
                          recorder=mongo_recorder, n_trail=n_trail, gamma=gamma,
                          number_learing_steps=number_learning_steps, number_past_vals=number_past_vals)

    else:
        env = FeatureWrapper(env_train, number_of_features=11, training_episode_length=training_episode_length,
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
                                                sigma=noise_var * np.ones(n_actions), dt=1e-4)

    print('SCHRITTWEITE DES ACTIONNOISE?!?!?!?Passt laut standard 1e-4')

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
                 # model = myDDPG('MlpPolicy', env, verbose=1, tensorboard_log=f'{folder_name}/{n_trail}/',
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

    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
        env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

    # start training
    model.learn(total_timesteps=number_learning_steps)

    # Log Train-info data
    train_data = {"Name": "After_Training",
                  "Mean_eps_reward": env.reward_episode_mean,
                  "Trial number": n_trail,
                  "Database name": folder_name,
                  "Sum_eps_reward": env.get_episode_rewards()
                  }
    mongo_recorder.save_to_json('Trial_number_' + n_trail, train_data)

    model.save(log_path + f'model.zip')

    ####### Run Test #########
    return_sum = 0.0
    limit_exceeded_in_test = False
    limit_exceeded_penalty = 0
    env_test = env_train

    if cfg['env_wrapper'] == 'past':
        env_test = FeatureWrapper_pastVals(env_test, number_of_features=9 + number_past_vals * 3,
                                           integrator_weight=integrator_weight,
                                           recorder=mongo_recorder, antiwindup_weight=antiwindup_weight,
                                           gamma=1, penalty_I_weight=0,
                                           penalty_P_weight=0, number_past_vals=number_past_vals,
                                           training_episode_length=training_episode_length, )
    elif cfg['env_wrapper'] == 'future':
        env_test = FeatureWrapper_futureVals(env_test, number_of_features=9,
                                             training_episode_length=training_episode_length,
                                             recorder=mongo_recorder, n_trail=n_trail,
                                             integrator_weight=integrator_weight,
                                             antiwindup_weight=antiwindup_weight, gamma=gamma,
                                             penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                             t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                             number_learing_steps=number_learning_steps, number_future_vals=10)
    elif cfg['env_wrapper'] == 'I-controller':
        env_test = FeatureWrapper_I_controller(env_test, number_of_features=12 + number_past_vals * 3,
                                               # including integrator_sum
                                               training_episode_length=training_episode_length,
                                               recorder=mongo_recorder, n_trail=n_trail,
                                               integrator_weight=integrator_weight,
                                               antiwindup_weight=antiwindup_weight, gamma=gamma,
                                               penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                               t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                               number_learing_steps=number_learning_steps, Ki=Ki_ddpg_combi,
                                               number_past_vals=number_past_vals)

    elif cfg['env_wrapper'] == 'no-I-term':
        env_test = BaseWrapper(env_test, number_of_features=2 + number_past_vals * 2,
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

    rew_list = []

    aP0 = []
    aP1 = []
    aI0 = []
    aI1 = []
    integrator_sum0 = []
    integrator_sum1 = []
    i_d_mess = []
    i_q_mess = []
    i_d_ref = []
    i_q_ref = []
    action_d = []
    action_q = []

    for step in range(test_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env_test.step(action)

        aP0.append(np.float64(action[0]))
        aP1.append(np.float64(action[1]))
        if action.shape[0] > 2:
            aI0.append(np.float64(action[3]))
            aI1.append(np.float64(action[4]))
            integrator_sum0.append(np.float64(env_test.integrator_sum[0]))
            integrator_sum1.append(np.float64(env_test.integrator_sum[1]))

        env_test.render()
        return_sum += rewards
        rew_list.append(rewards)
        i_d_mess.append(np.float64(obs[0]))
        i_q_mess.append(np.float64(obs[1]))
        i_d_ref.append(np.float64(obs[2]))
        i_q_ref.append(np.float64(obs[3]))
        action_d.append(np.float64(action[0]))
        action_q.append(np.float64(action[1]))

        if done:
            env_test.close()
            # print(limit_exceeded_in_test)
            break

    ts = time.gmtime()
    test_after_training = {"Name": "Test",
                           "time": ts,
                           "Reward": rew_list,
                           "i_d_mess": i_d_mess,
                           "i_q_mess": i_q_mess,
                           "i_d_ref": i_d_ref,
                           "i_q_ref": i_q_ref,
                           'action_d': action_d,
                           'action_q': action_q,
                           "ActionP0": aP0,
                           "ActionP1": aP1,
                           "ActionI0": aI0,
                           "ActionI1": aI1,
                           "integrator_sum0": integrator_sum0,
                           "integrator_sum1": integrator_sum1,
                           "Node": platform.uname().node,
                           "End time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime()),
                           "Reward function": 'rew.rew_fun_dq0',
                           "Trial number": n_trail,
                           "Database name": folder_name,
                           "Info": "GEM; features: error, past_vals, used_action"}

    mongo_recorder.save_to_json('Trial_number_' + n_trail, test_after_training)

    return (return_sum / test_length)
