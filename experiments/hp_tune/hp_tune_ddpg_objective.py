import itertools
import time

import optuna
import platform
import argparse
import sshtunnel
import os
import numpy as np
# import experiments.hp_tune.util.config as cfg
from experiments.hp_tune.util.config import cfg

# from experiments.hp_tune.experiment_vctrl_single_inv import experiment_fit_DDPG, mongo_recorder
from experiments.hp_tune.experiment_vctrl_single_inv_dq0 import experiment_fit_DDPG_dq0, mongo_recorder
from experiments.hp_tune.util.scheduler import linear_schedule

PC2_LOCAL_PORT2PSQL = 11999
DB_NAME = 'optuna'
SERVER_LOCAL_PORT2PSQL = 6432
STUDY_NAME = 'DDPG_Lr_gamma_Anoise'


# cfg = dict(lea_vpn_nodes=['lea-skynet', 'lea-picard', 'lea-barclay',
#                          'lea-cyberdyne', 'webbah-ThinkPad-L380'])


def ddpg_objective(trial):
    number_learning_steps = 500000  # trial.suggest_int("number_learning_steps", 100000, 1000000)

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 5e-2)  # 0.0002#

    lr_decay_start = trial.suggest_float("lr_decay_start", 0.00001, 1)  # 3000  # 0.2 * number_learning_steps?
    lr_decay_duration = trial.suggest_float("lr_decay_duration", 0.00001,
                                            1)  # 3000  # 0.2 * number_learning_steps?
    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))
    final_lr = trial.suggest_float("final_lr", 0.00001, 1)

    gamma = trial.suggest_float("gamma", 0.5, 0.99)
    weight_scale = 0.1  # trial.suggest_loguniform("weight_scale", 5e-4, 0.1)  # 0.005

    bias_scale = 0.1  # trial.suggest_loguniform("bias_scale", 5e-4, 0.1)  # 0.005
    alpha_relu_actor = 0.1  # trial.suggest_loguniform("alpha_relu_actor", 0.0001, 0.5)  # 0.005
    alpha_relu_critic = 0.1  # trial.suggest_loguniform("alpha_relu_critic", 0.0001, 0.5)  # 0.005

    batch_size = 1024  # trial.suggest_int("batch_size", 32, 1024)  # 128
    buffer_size = int(1e6)  # trial.suggest_int("buffer_size", 10, 1000000)  # 128

    actor_hidden_size = 150  # trial.suggest_int("actor_hidden_size", 10, 200)  # 100  # Using LeakyReLU
    actor_number_layers = 1  # trial.suggest_int("actor_number_layers", 1, 3)

    critic_hidden_size = 150  # trial.suggest_int("critic_hidden_size", 10, 300)  # 100
    critic_number_layers = 2  # trial.suggest_int("critic_number_layers", 1, 4)

    n_trail = str(trial.number)
    use_gamma_in_rew = 1
    noise_var = trial.suggest_loguniform("noise_var", 0.01, 4)  # 2
    # min var, action noise is reduced to (depends on noise_var)
    noise_var_min = 0.0013  # trial.suggest_loguniform("noise_var_min", 0.0000001, 2)
    # min var, action noise is reduced to (depends on training_episode_length)
    noise_steps_annealing = int(
        0.25 * number_learning_steps)  # trail.suggest_int("noise_steps_annealing", int(0.1 * number_learning_steps),
    # number_learning_steps)
    noise_theta = trial.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU
    error_exponent = 0.5  # trial.suggest_loguniform("error_exponent", 0.01, 4)

    training_episode_length = 2000  # trial.suggest_int("training_episode_length", 200, 5000)  # 128
    learning_starts = 0.32  # trial.suggest_loguniform("learning_starts", 0.1, 2)  # 128
    tau = 0.005  # trial.suggest_loguniform("tau", 0.0001, 0.2)  # 2

    learning_rate = linear_schedule(initial_value=learning_rate, final_value=learning_rate * final_lr,
                                    t_start=t_start,
                                    t_end=t_end,
                                    total_timesteps=number_learning_steps)

    trail_config_mongo = {"Name": "Config",
                          "Node": platform.uname().node,
                          "Number_learning_Steps": number_learning_steps,
                          "Start time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())}
    trail_config_mongo.update(trial.params)
    mongo_recorder.save_to_mongodb('Trial_number_' + n_trail, trail_config_mongo)

    loss = experiment_fit_DDPG_dq0(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale, alpha_relu_actor,
                                   batch_size,
                                   actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                                   alpha_relu_critic,
                                   noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                                   training_episode_length, buffer_size,
                                   learning_starts, tau, number_learning_steps, n_trail)

    return loss


def optuna_optimize(objective, sampler=None, study_name='dummy'):
    parser = argparse.ArgumentParser(description='Train DDPG Single Inverter V-ctrl')
    parser.add_argument('-n', '--n_trials', default=50, required=False,
                        help='number of trials to execute', type=int)
    args = parser.parse_args()
    n_trials = args.n_trials or 10

    print(n_trials)

    node = platform.uname().node

    if node in ('lea-picard', 'lea-barclay'):
        creds_path = 'C:\\Users\\webbah\\Documents\\creds\\optuna_psql.txt'
    else:
        # read db credentials
        creds_path = f'{os.getenv("HOME")}/creds/optuna_psql'
    with open(creds_path, 'r') as f:
        optuna_creds = ':'.join([s.strip(' \n') for s in f.readlines()])
    # set trial to failed if it seems dead for 20 minutes
    storage_kws = dict(engine_kwargs={"connect_args": {"timeout": 600}})
    if node in ('lea-cyberdyne', 'fe1'):
        if node == 'fe1':
            port = PC2_LOCAL_PORT2PSQL
        else:
            port = SERVER_LOCAL_PORT2PSQL
        storage = optuna.storages.RDBStorage(
            url=f'postgresql://{optuna_creds}@localhost:{port}/{DB_NAME}',
            **storage_kws)
        study = optuna.create_study(
            storage=storage,
            # storage=f'postgresql://{optuna_creds}@localhost:{port}/{DB_NAME}',
            sampler=sampler, study_name=study_name,
            load_if_exists=True,
            direction='maximize')
        study.optimize(objective, n_trials=n_trials)
    else:
        if node in cfg['lea_vpn_nodes']:
            # we are in LEA VPN
            server_name = 'lea38'
            tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                               SERVER_LOCAL_PORT2PSQL)}
        else:
            # assume we are on a PC2 compute node
            server_name = 'fe.pc2.uni-paderborn.de'
            tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                               PC2_LOCAL_PORT2PSQL),
                       'ssh_username': 'webbah'}
        with sshtunnel.open_tunnel(server_name, **tun_cfg) as tun:
            storage = optuna.storages.RDBStorage(
                url=f'postgresql://{optuna_creds}'
                    f'@localhost:{tun.local_bind_port}/{DB_NAME}',
                **storage_kws)
            study = optuna.create_study(
                storage=storage,
                # storage=f'postgresql://{optuna_creds}'
                #        f'@localhost:{tun.local_bind_port}/{DB_NAME}',
                sampler=sampler, study_name=study_name,
                load_if_exists=True,
                direction='maximize')
            study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    # with tf.device('/cpu:0'):
    #    optuna_optimize(ddpg_objective, study_name=STUDY_NAME)

    learning_rate = list(itertools.chain(*[[1e-9] * 1]))
    # number_learning_steps = list(itertools.chain(*[[1100000] * 1]))
    # learning_rate = list(itertools.chain(*[[1e-3]*1, [1e-4]*1, [1e-5]*1, [1e-6]*1, [1e-7]*1, [1e-8]*1, [1e-9]*1]))
    search_space = {'learning_rate': learning_rate}  # , 'number_learning_steps': number_learning_steps}

    optuna_optimize(ddpg_objective, study_name=STUDY_NAME)#, sampler=optuna.samplers.GridSampler(search_space))
