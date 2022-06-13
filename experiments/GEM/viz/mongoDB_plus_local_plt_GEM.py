import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import sshtunnel
from bson import ObjectId
from plotly import tools
from pymongo import MongoClient

from openmodelica_microgrid_gym.util import dq0_to_abc, abc_to_dq0

plt_train = True
plotly = False
local = True

# db_name = 'PC2_DDGP_Vctrl_single_inv_18_penalties'
# db_name = 'DDPG_SplitActor_Best_study18_6462'
db_name = 'GEM_I_term_3'  # 15
# db_name = 'GEM_past' # 17
trial = '0'
show_episode_number = 19

with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
    with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
        db = client[db_name]

        trial = db.Trial_number_23

        if local:
            file_train = open('I_term_3_23/GEM_I_term_3_Trial_number_23_178.json', )
            train_data = json.load(file_train)
            file_test = open('I_term_3_23/GEM_I_term_3_Trial_number_23_179.json', )
            trial_test = json.load(file_test)
        else:
            train_data = trial.find_one({"Name": "After_Training"})
            train_episode_data = trial.find_one({"Episode_number": show_episode_number})
            trial_test = trial.find_one({"Name": "Test"})
        trial_config = trial.find_one({"Name": "Config"})

        train_reward_per_episode = train_data['Mean_eps_reward']

        ax = plt.plot(train_reward_per_episode)  # [::2])
        plt.grid()
        plt.xlabel("Episodes")
        plt.ylabel("Mean episode Reward")
        plt.title(f"Test {db_name}")
        plt.show()

        ts = 1e-4  # if ts stored: take from db
        t_test = np.arange(0, len(trial_test['i_d_mess']) * ts, ts).tolist()

        plt.plot(t_test, trial_test['i_d_mess'])
        plt.plot(t_test, trial_test['i_d_ref'], 'r')
        plt.plot(t_test, trial_test['i_d_ref'], 'r')
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("i_d")
        plt.title(f"Test{db_name}")
        plt.show()

        plt.plot(t_test, trial_test['i_q_mess'])
        plt.plot(t_test, trial_test['i_q_ref'], 'r')
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("i_q")
        plt.title(f"Test {db_name}")
        plt.show()

        plt.plot(t_test, trial_test['action_d'])
        if len(trial_test['integrator_sum0']) > 0:
            plt.plot(t_test, trial_test['integrator_sum0'], 'g')
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("action_d (integrator_sum-g)")
        plt.title(f"Test {db_name}")
        plt.show()

        plt.plot(t_test, trial_test['action_q'])
        if len(trial_test['integrator_sum1']) > 0:
            plt.plot(t_test, trial_test['integrator_sum1'], 'g')
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("action_q (integrator_sum-g)")
        plt.title(f"Test {db_name}")
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

            plt.plot(train_episode_data['Integrator0'])
            plt.plot(train_episode_data['Integrator1'])
            plt.plot(train_episode_data['Integrator2'])
            plt.grid()
            plt.xlabel("steps")
            plt.ylabel("Int Zustand")
            plt.title(f"Trainingepisode {show_episode_number}")
            plt.show()

            plt.plot(train_episode_data['actionP0'])
            plt.plot(train_episode_data['actionP1'])
            plt.plot(train_episode_data['actionP2'])
            plt.grid()
            plt.xlabel("steps")
            plt.ylabel("actionP")
            plt.title(f"Trainingepisode {show_episode_number}")
            plt.show()

            plt.plot(train_episode_data['actionI0'])
            plt.plot(train_episode_data['actionI1'])
            plt.plot(train_episode_data['actionI2'])
            plt.grid()
            plt.xlabel("steps")
            plt.ylabel("actionI")
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

        ##############################################################
        # After Training

        if plotly:
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
