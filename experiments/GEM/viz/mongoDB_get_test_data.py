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
db_name = 'GEM_I_term_4'  # 15
# db_name = 'GEM_past' # 17
trial = '0'
show_episode_number = 19

ret_mean_list_test = []
ret_std_list_test = []
i_q_delta_mean_list_test = []
i_q_delta_std_list_test = []
i_d_delta_mean_list_test = []
i_d_delta_std_list_test = []

with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
    with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
        db = client[db_name]

        for coll_name in db.list_collection_names():
            trial = db[coll_name]
            # trial = db.Trial_number_23

            # train_data = trial.find_one({"Name": "After_Training"})
            trial_test = trial.find_one({"Name": "Test"})

            """
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

            plt.plot(t_test, trial_test['Reward'])
            plt.grid()
            plt.xlabel("t")
            plt.ylabel("reward")
            plt.title(f"Test {db_name}")
            plt.show()

            print(np.mean(trial_test['Reward']))
            print(np.std(trial_test['Reward']))
            """

            ret_mean_list_test.append(np.mean(trial_test['Reward']))
            ret_std_list_test.append(np.std(trial_test['Reward']))
            i_q_delta_mean_list_test.append(np.mean(np.array(trial_test['i_q_ref']) - np.array(trial_test['i_q_mess'])))
            i_q_delta_std_list_test.append(np.std(np.array(trial_test['i_q_ref']) - np.array(trial_test['i_q_mess'])))
            i_d_delta_mean_list_test.append(np.mean(np.array(trial_test['i_d_ref']) - np.array(trial_test['i_d_mess'])))
            i_d_delta_std_list_test.append(np.std(np.array(trial_test['i_d_ref']) - np.array(trial_test['i_d_mess'])))

print(ret_mean_list_test)
print(ret_std_list_test)
asd = 1
results = {
    'return_Mean': ret_mean_list_test,
    'return_Std': ret_std_list_test,
    'i_q_delta_Mean': i_q_delta_mean_list_test,
    'i_q_delta_Std': i_q_delta_std_list_test,
    'i_d_delta_Mean': i_d_delta_mean_list_test,
    'i_d_delta_Std': i_d_delta_std_list_test,
    'study_name': db_name}

df = pd.DataFrame(results)
df.to_pickle(db_name + "mean_over_1250_agents.pkl")

m = np.array(ret_mean_list_test)
s = np.array(ret_std_list_test)

plt.plot(m)
plt.fill_between(m - s, m + s, facecolor='r')
plt.ylabel('Average return +- sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([0, 200])
plt.grid()
plt.title(db_name)
plt.show()
