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

# db_name = 'OMG_DDPG_Actor' # 17
# db_name = 'OMG_DDPG_Integrator_no_pastVals'
db_names = ['OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr']  # 15
# db_name = 'OMG_DDPG_Integrator_no_pastVals_corr'

# db_names = ['OMG_DDPG_Actor', 'OMG_DDPG_Integrator_no_pastVals', 'OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr',
#            'OMG_DDPG_Integrator_no_pastVals_corr']

for db_name in db_names:
    ret_list_test = []
    ret_mean_list_test = []
    ret_std_list_test = []

    with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
        with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
            db = client[db_name]

            for coll_name in db.list_collection_names():
                trial = db[coll_name]

                trial_test = trial.find_one({"Name": "Test_Reward"})

                ret_list_test.append(trial_test['Return'])
                ret_std_list_test.append(np.std(trial_test['Reward']))
                ret_mean_list_test.append(np.mean(trial_test['Reward']))

    print(ret_mean_list_test)
    print(ret_std_list_test)
    asd = 1
    results = {
        'return': ret_list_test,
        'return_Mean': ret_mean_list_test,
        'return_Std': ret_std_list_test,
        'study_name': db_name}

    df = pd.DataFrame(results)
    df.to_pickle(db_name + "return_500_agents.pkl")
