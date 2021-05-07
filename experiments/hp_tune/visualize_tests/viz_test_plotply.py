import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
from bson import ObjectId
from plotly import tools
from pymongo import MongoClient

from openmodelica_microgrid_gym.util import dq0_to_abc, abc_to_dq0

client = MongoClient('mongodb://localhost:27017/')
db = client['Master_V_ctrl_dq0_noDelay_rewNew']

trail = db.Trail_number_1

#test_data = trail.find_one({"Name": "Test"})
#test_data = trail.find({"Name": "Test"})
#test_data = trail.find_one({"_id": ObjectId("608b147223c108820cf2b664")})

for post in trail.find({"Name": "Test"}):
    test_data = post

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax1, ax2 = ax.flatten()
ts = 1e-4  # if ts stored: take from db
t = np.arange(0, len(test_data['lc_capacitor1_v']) * ts, ts).tolist()

df2 = pd.DataFrame(t)
df2['t'] = pd.DataFrame(t)
df2['v_a'] = pd.DataFrame(test_data['lc_capacitor1_v'])
df2['v_b'] = pd.DataFrame(test_data['lc_capacitor2_v'])
df2['v_c'] = pd.DataFrame(test_data['lc_capacitor3_v'])
#df2['v_a'] = pd.DataFrame(test_data['lc_inductor1_i'])
#df2['v_b'] = pd.DataFrame(test_data['lc_inductor2_i'])
#df2['v_c'] = pd.DataFrame(test_data['lc_inductor3_i'])

df2['v_0_SP'] = pd.DataFrame(test_data['inverter1_v_ref_0'])
df2['v_1_SP'] = pd.DataFrame(test_data['inverter1_v_ref_1'])
df2['v_2_SP'] = pd.DataFrame(test_data['inverter1_v_ref_2'])

df2['phase'] = pd.DataFrame(test_data['Phase'])

v_sp_abc = dq0_to_abc(np.array([df2['v_0_SP'], df2['v_1_SP'], df2['v_2_SP']]), np.array(df2['phase']))

v_mess_dq0 = abc_to_dq0(np.array([df2['v_a'], df2['v_b'], df2['v_c']]), np.array(df2['phase']))

#plt.plot(t, v_sp_abc[1,:])
#plt.show()

x = df2['t']
v_a = df2['v_a']
v_b = df2['v_b']
v_c = df2['v_c']
#v_a_SP = df2['v_0_SP']
#v_b_SP = df2['v_1_SP']
#v_c_SP = df2['v_2_SP']

# df2['v_a'] = pd.DataFrame(test_data['lc_inductor1_i'])
# df2['v_b'] = pd.DataFrame(test_data['lc_inductor2_i'])
# df2['v_c'] = pd.DataFrame(test_data['lc_inductor3_i'])
# df2['v_a_SP'] = pd.DataFrame(test_data['master_SPVa'])
# df2['v_b_SP'] = pd.DataFrame(test_data['master_SPVb'])
# df2['v_c_SP'] = pd.DataFrame(test_data['master_SPVc'])
# #
# v_b_SP = df2['v_b_SP']
# v_c_SP = df2['v_c_SP']

# plot = px.Figure(data=[px.Scatter(
#    x=x,
#    y=y,
#    mode='lines', )
# ])


"""
plot = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

plot.add_trace(
    px.Scatter(x=x, y=v_a),1,1)


plot.add_trace(
    px.Scatter(x=x, y=v_b),1,2)

plot.add_trace(
    px.Scatter(x=x, y=v_c),1,2)

plot.add_trace(
    px.Scatter(x=x, y=v_a_SP),1,1)
"""

plot = px.Figure()
plot.add_trace(
    px.Scatter(x=x, y=v_a))
    #px.Scatter(x=x, y=v_mess_dq0[0][:]))

plot.add_trace(
    px.Scatter(x=x, y=v_b))
    #px.Scatter(x=x, y=v_mess_dq0[1][:]))

plot.add_trace(
    px.Scatter(x=x, y=v_c))
    #px.Scatter(x=x, y=v_mess_dq0[2][:]))

plot.add_trace(
    #px.Scatter(x=x, y=v_sp_abc[0,:]))
    px.Scatter(x=x, y=df2['v_0_SP']))

plot.add_trace(
    #px.Scatter(x=x, y=v_sp_abc[1,:]))
    px.Scatter(x=x, y=df2['v_1_SP']))

plot.add_trace(
    #px.Scatter(x=x, y=v_sp_abc[2,:]))
    px.Scatter(x=x, y=df2['v_2_SP']))


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
