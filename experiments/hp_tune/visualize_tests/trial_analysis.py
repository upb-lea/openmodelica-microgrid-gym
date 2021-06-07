import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import sshtunnel
from bson import ObjectId
from plotly import tools
from pymongo import MongoClient

from openmodelica_microgrid_gym.util import dq0_to_abc, abc_to_dq0

db_name = 'PC2_DDPG_MRE_V_only'
trial = '834'
show_episode_number = 10

with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
    with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
        db = client[db_name]

        # trial = db.Trail_number_232
        trial = db.Trial_number_7

        trial_config = trial.find_one({"Name": "Config"})
        trial_test = trial.find_one({"Name": "Test"})
        train_data = trial.find_one({"Name": "After_Training"})
        train_episode_data = trial.find_one({"Episode_number": show_episode_number})

        print(f'Starttime = {trial_config["Start time"]}')
        print(f'Starttime = {trial_test["End time"]}')
        print(' ')
        print(f'Node = {trial_config["Node"]}')
        print(' ')

        print('Config-Params:')
        print(*trial_config.items(), sep='\n')

        ts = 1e-4  # if ts stored: take from db
        t_test = np.arange(0, len(trial_test['lc_capacitor1_v']) * ts, ts).tolist()
        v_a_test = trial_test['lc_capacitor1_v']
        v_b_test = trial_test['lc_capacitor2_v']
        v_c_test = trial_test['lc_capacitor3_v']
        i_a_test = trial_test['lc_inductor1_i']
        i_b_test = trial_test['lc_inductor2_i']
        i_c_test = trial_test['lc_inductor3_i']

        v_sp_d_test = trial_test['inverter1_v_ref_0']
        v_sp_q_test = trial_test['inverter1_v_ref_1']
        v_sp_0_test = trial_test['inverter1_v_ref_2']

        phase_test = trial_test['Phase']

        v_sp_abc = dq0_to_abc(np.array([v_sp_d_test, v_sp_q_test, v_sp_0_test]), np.array(phase_test))

        v_mess_dq0 = abc_to_dq0(np.array([v_a_test, v_b_test, v_c_test]), np.array(phase_test))

        plt.plot(t_test, v_a_test)
        plt.plot(t_test, v_b_test)
        plt.plot(t_test, v_c_test)
        plt.plot(t_test, v_sp_abc[0, :])
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("v_abc")
        plt.title('Test')
        plt.show()

        plt.plot(t_test, v_mess_dq0[0, :])
        plt.plot(t_test, v_mess_dq0[1, :])
        plt.plot(t_test, v_mess_dq0[2, :])
        plt.plot(t_test, v_sp_d_test)
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("v_dq0")
        plt.title('Test')
        plt.show()

        plt.plot(t_test, i_a_test)
        plt.plot(t_test, i_b_test)
        plt.plot(t_test, i_c_test)
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("v_abc")
        plt.title('Test')
        plt.show()

        # pyplot v_abc
        plot = px.Figure()
        plot.add_trace(
            px.Scatter(x=t_test, y=v_a_test))
        # px.Scatter(x=x, y=v_mess_dq0[0][:]))

        plot.add_trace(
            px.Scatter(x=t_test, y=v_b_test))
        # px.Scatter(x=x, y=v_mess_dq0[1][:]))
        plot.add_trace(
            px.Scatter(x=t_test, y=v_c_test))
        # px.Scatter(x=x, y=v_mess_dq0[2][:]))

        plot.add_trace(
            px.Scatter(x=t_test, y=v_sp_abc[1, :]))
        # px.Scatter(x=x, y=df2['v_1_SP']))

        plot.add_trace(
            px.Scatter(x=t_test, y=v_sp_abc[2, :]))
        # px.Scatter(x=x, y=df2['v_2_SP']))

        plot.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
            )
        )

        plot.show()

    ##############################################################
    # After Training

    train_reward_per_episode = train_data['Mean_eps_reward']
    number_learning_steps = trial_config['Number_learning_Steps']
    episode_len = 2000  # trial_config['training_episode_length']
    learning_rate = trial_config['learning_rate']
    lr_decay_start = trial_config['lr_decay_start']
    lr_decay_duration = trial_config['lr_decay_duration']
    final_lr = trial_config['final_lr'] * learning_rate

    ax = plt.plot(train_reward_per_episode)
    plt.grid()
    plt.xlabel("Episodes")
    # plt.yscale('log')
    plt.ylabel("Mean episode Reward")
    # plt.title("1.000.000")
    plt.show()

    plot = px.Figure()
    plot.add_trace(
        px.Scatter(y=train_reward_per_episode))

    plot.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )

    plot.show()

    t = np.arange(number_learning_steps)

    progress_remaining = 1.0 - (t / number_learning_steps)

    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))

    lr_curve = np.maximum(
        np.minimum(learning_rate, learning_rate + (t_start * (learning_rate - final_lr)) / (t_end - t_start) \
                   - (learning_rate - final_lr) / (t_end - t_start) * ((1.0 - progress_remaining) \
                                                                       * number_learning_steps)), final_lr)

    # no step-vise MA needed but episode-vise!
    # lr_ma = np.convolve(lr_curve, np.ones(episode_len), 'valid') / episode_len
    num_episodes = int(number_learning_steps / episode_len)
    lr_ma = np.zeros(num_episodes)
    count = 0

    for i in range(num_episodes):
        lr_ma[i] = np.mean(lr_curve[count:count + episode_len])
        count += episode_len

    # plt.plot(lr_curve)
    # plt.show()

    fig = plt.figure()  # a new figure window
    ax = fig.add_subplot(2, 1, 1)  # a new axes
    ax = plt.plot(train_reward_per_episode)
    plt.grid()
    # plt.xlabel("Episodes")
    # plt.yscale('log')
    plt.ylabel("Mean episode Reward")

    ax2 = fig.add_subplot(2, 1, 2)  # a new axes
    ax2 = plt.plot(lr_ma)
    plt.grid()
    plt.xlabel("Episodes")
    plt.ylabel("Mean episode LR")
    plt.show()

    plt.show()

    if train_episode_data is not None:
        # only available if loglevel == 'train'
        ##############################################################
        # Plot example Training Episode
        R_load = train_episode_data['R_load_training']
        i_a = train_episode_data['i_a_training']
        i_b = train_episode_data['i_b_training']
        i_c = train_episode_data['i_c_training']
        v_a = train_episode_data['v_a_training']
        v_b = train_episode_data['v_b_training']
        v_c = train_episode_data['v_c_training']
        reward = train_episode_data['Rewards']
        phase = train_episode_data['Phase']

        plt.plot(R_load)
        plt.grid()
        plt.xlabel("steps")
        plt.ylabel("R_load")
        plt.title(f"Trainingepisode {show_episode_number}")
        plt.show()

        plt.plot(i_a)
        plt.plot(i_b)
        plt.plot(i_c)
        plt.grid()
        plt.xlabel("steps")
        plt.ylabel("i_abc")
        plt.title(f"Trainingepisode {show_episode_number}")
        plt.show()

        plt.plot(v_a)
        plt.plot(v_b)
        plt.plot(v_c)
        plt.grid()
        plt.xlabel("steps")
        plt.ylabel("v_abc")
        plt.title(f"Trainingepisode {show_episode_number}")
        plt.show()

        plt.plot(reward)
        plt.grid()
        plt.xlabel("steps")
        plt.ylabel("Reward")
        plt.title(f"Trainingepisode {show_episode_number}")
        plt.show()

        df = pd.DataFrame()
        df['R_load'] = R_load

        hist = df['R_load'].hist(bins=50)
        plt.title(f"Trainingepisode {show_episode_number}")
        plt.show()
        """
        plot = px.Figure()
        plot.add_trace(
            px.Scatter(y=R_load)
        """
        # df2['v_0_SP'] = pd.DataFrame(test_data['inverter1_v_ref_0'])
        # df2['v_1_SP'] = pd.DataFrame(test_data['inverter1_v_ref_1'])
        # df2['v_2_SP'] = pd.DataFrame(test_data['inverter1_v_ref_2'])

        # df2['phase'] = pd.DataFrame(test_data['Phase'])

        # v_sp_abc = dq0_to_abc(np.array([df2['v_0_SP'], df2['v_1_SP'], df2['v_2_SP']]), np.array(df2['phase']))

        v_mess_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), np.array(phase))

        # x = df2['t']
        v_d = v_mess_dq0[0][:]  # df2['v_a']
        v_q = v_mess_dq0[1][:]  # df2['v_b']
        v_0 = v_mess_dq0[2][:]  # df2['v_c']

        plt.plot(v_d)
        plt.plot(v_q)
        plt.plot(v_0)
        plt.grid()
        plt.xlabel("steps")
        plt.ylabel("v_dq0")
        plt.title(f"Trainingepisode {show_episode_number}")
        plt.show()

        # v_a_SP = df2['v_0_SP']#v_sp_abc[0,:]
        # v_b_SP = df2['v_1_SP']#v_sp_abc[1,:]
        # v_c_SP = df2['v_2_SP']#v_sp_abc[2,:]

        plot = px.Figure()
        plot.add_trace(
            px.Scatter(y=v_a))

        plot.add_trace(
            px.Scatter(y=v_b))

        plot.add_trace(
            px.Scatter(y=v_c))

        plot.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
            )
        )

        plot.show()
