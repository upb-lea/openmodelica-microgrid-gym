import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import sshtunnel
from pymongo import MongoClient

plt_train = True
plotly = False

folder_name = 'saves/compare_I_noI_4/'

# db_name = 'PC2_DDGP_Vctrl_single_inv_18_penalties'
# db_name = 'DDPG_SplitActor_Best_study18_6462'
db_name = 'GEM_I_term_4'  # 770
db_name = 'GEM_no_I_term_4'  # 1192
trial = '0'
show_episode_number = 19

with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
    with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
        db = client[db_name]

        # trial = db.Trial_number_770
        trial = db.Trial_number_1192

        train_data = trial.find_one({"Name": "After_Training"})
        train_episode_data = trial.find_one({"Episode_number": show_episode_number})
        trial_config = trial.find_one({"Name": "Config"})
        trial_test = trial.find_one({"Name": "Test"})

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
        # plt.plot(t_test, trial_test['i_d_ref'], 'r')
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

        plt.plot(t_test, trial_test['v_d_mess'][:-1])
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("v_d")
        plt.title(f"Test{db_name}")
        plt.show()

        plt.plot(t_test, trial_test['v_q_mess'][:-1])
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("v_q")
        plt.title(f"Test{db_name}")
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

        plt.plot(t_test, trial_test['Reward'])
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("reward")
        plt.title(f"Test {db_name}")
        plt.show()

        ts = time.gmtime()
        compare_result = {"Name": db_name,
                          "time": ts,
                          "i_q_mess": trial_test['i_q_mess'],
                          "i_q_ref": trial_test['i_q_ref'],
                          "i_d_mess": trial_test['i_d_mess'],
                          "i_d_ref": trial_test['i_d_ref'],
                          "v_d_mess": trial_test['v_d_mess'],
                          "v_q_mess": trial_test['v_q_mess'],
                          "Reward_test": trial_test['Reward'],
                          "train_reward_per_episode": train_reward_per_episode,
                          "info": "GEM results from testcase",
                          }
        store_df = pd.DataFrame([compare_result])
        store_df.to_pickle(f'{folder_name}/_{db_name}_trial770')

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
