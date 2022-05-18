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

# db_name = 'PC2_DDGP_Vctrl_single_inv_18_penalties'
# db_name = 'DDPG_SplitActor_Best_study18_6462'
db_name = 'GEM_no_I_term_4'  # 15
# db_name = 'GEM_past' # 17
trial = '0'
show_episode_number = 19

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

reward_df.to_pickle(db_name + "_1250_agents_train_data.pkl")
