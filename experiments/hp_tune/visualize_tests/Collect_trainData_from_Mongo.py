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

db_name = 'OMG_DDPG_Actor'  # 15
db_name = 'OMG_DDPG_Integrator_no_pastVals'  # 15

trial = '0'
show_episode_number = 19

ret_mean_list_test = []
ret_std_list_test = []
i_q_delta_mean_list_test = []
i_q_delta_std_list_test = []
i_d_delta_mean_list_test = []
i_d_delta_std_list_test = []
reward_list = []

reward_df = pd.DataFrame()

with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
    with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
        db = client[db_name]

        idx = 0
        for coll_name in db.list_collection_names():
            trial = db[coll_name]
            # trial = db.Trial_number_23

            train_data = trial.find_one({"Name": "After_Training"})
            # trial_test = trial.find_one({"Name": "Test"})

            if train_data is not None:  # if trial not finished (was in actor_Ddpg > 550)

                if idx == 0:
                    reward_df = pd.DataFrame({str(idx): train_data['Mean_eps_reward']})
                else:

                    df_tmp = pd.DataFrame({str(idx): train_data['Mean_eps_reward']})
                    reward_df = reward_df.join(df_tmp)
                idx += 1

            # reward_list.append(train_data['Mean_eps_reward'])

            # reward_df = reward_df.join(df_tmp)

            # if len(reward_list) >= 550:
            #   break

reward_df.to_pickle(db_name + "_8XX_agents_train_data.pkl")
# print(ret_mean_list_test)
# print(ret_std_list_test)
# asd = 1
# results = {
#    'reward': reward_list,
#    'study_name': db_name}

# df = pd.DataFrame(results)
# df.to_pickle(db_name+"_1250_agents_train_data.pkl")


plt.plot(reward_list)
# plt.fill_between( m - s, m + s, facecolor='r')
plt.ylabel('Average return +- sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title(db_name)
plt.show()
