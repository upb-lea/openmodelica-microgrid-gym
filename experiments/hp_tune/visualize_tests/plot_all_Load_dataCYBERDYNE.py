import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import sshtunnel
from plotly import tools
from pymongo import MongoClient

from openmodelica_microgrid_gym.util import abc_to_dq0

mongo_tunnel = sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001))
mongo_tunnel.start()
mongo_recorder = MongoClient(
    f'mongodb://localhost:{mongo_tunnel.local_bind_port}/')  # store to port 12001 for ssh data to cyberdyne

# client = MongoClient('mongodb://localhost:27017/')

db = mongo_recorder['DDPG_V_ctrl_dq0_no_delay_using_iO']

trail = db.Trail_number_2

R_load = []
i_a = []
i_b = []
i_c = []
v_a = []
v_b = []
v_c = []
reward = []
phase = []

for ii in range(1):
    # test_data = trail.find_one({"Name": "On_Training"})
    test_data = trail.find_one({"Episode_number": 5})

    # R_load.append(test_data['R_load_training'][:])
    R_load = R_load + test_data['R_load_training']
    i_a = i_a + test_data['i_a_training']
    i_b = i_b + test_data['i_b_training']
    i_c = i_c + test_data['i_c_training']
    v_a = v_a + test_data['v_a_training']
    v_b = v_b + test_data['v_b_training']
    v_c = v_c + test_data['v_c_training']
    reward = reward + test_data['Rewards']
    phase = phase + test_data['Phase'][8658 - 204:]

plt.plot(R_load)
plt.grid()
plt.xlabel("steps")
plt.ylabel("R_load")
plt.show()

plt.plot(i_a)
plt.plot(i_b)
plt.plot(i_c)
plt.grid()
plt.xlabel("steps")
plt.ylabel("i_abc")
plt.show()

plt.plot(v_a)
plt.plot(v_b)
plt.plot(v_c)
plt.grid()
plt.xlabel("steps")
plt.ylabel("v_abc")
plt.show()

plt.plot(reward)
plt.grid()
plt.xlabel("steps")
plt.ylabel("Reward")
plt.show()

df = pd.DataFrame()
df['R_load'] = R_load

hist = df['R_load'].hist(bins=50)
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
