import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
from plotly import tools
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['Load_clipp']

trail = db.Trail_number_1

R_load = []
i_a = []
i_b = []
i_c = []
v_a = []
v_b = []
v_c = []
reward = []

for ii in range(287):
    # test_data = trail.find_one({"Name": "On_Training"})
    test_data = trail.find_one({"Epsisode_number": ii})

    # R_load.append(test_data['R_load_training'][:])
    R_load = R_load + test_data['R_load_training']
    i_a = i_a + test_data['i_a_training']
    i_b = i_b + test_data['i_b_training']
    i_c = i_c + test_data['i_c_training']
    v_a = v_a + test_data['v_a_training']
    v_b = v_b + test_data['v_b_training']
    v_c = v_c + test_data['v_c_training']
    reward = reward + test_data['Rewards']

plt.plot(R_load)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Mean episode Reward")
plt.show()

plt.plot(i_a)
plt.plot(i_b)
plt.plot(i_c)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Mean episode Reward")
plt.show()

plt.plot(v_a)
plt.plot(v_b)
plt.plot(v_c)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Mean episode Reward")
plt.show()

plt.plot(reward)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Mean episode Reward")
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
