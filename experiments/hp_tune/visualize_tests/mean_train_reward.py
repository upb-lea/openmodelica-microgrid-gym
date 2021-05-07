import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
from plotly import tools
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['Master_V_ctrl_dq0_Delay_rewNew']

trail = db.Trail_number_5

for post in trail.find({"Name": "After_Training"}):
    test_data = post

#test_data = trail.find_one({"Name": "After_Training"})
#fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
#ax1, ax2 = ax.flatten()
ts = 1e-4  # if ts stored: take from db
t = np.arange(0, len(test_data['Mean_eps_reward']) * ts, ts).tolist()

df2 = pd.DataFrame(t)
df2['t'] = pd.DataFrame(t)
df2['r'] = pd.DataFrame(test_data['Mean_eps_reward'])

x = df2['t']
y = df2['r']

ax = plt.plot(y)
plt.grid()
plt.xlabel("Episodes")
plt.yscale('log')
plt.ylabel("Mean episode Reward")
#plt.title("1.000.000")
plt.show()


plot = px.Figure()
plot.add_trace(
    px.Scatter(y=y))

# plot.add_trace(
#    px.Scatter(x=x, y=v_a_SP))

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
