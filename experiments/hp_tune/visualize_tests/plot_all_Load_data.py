import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
from plotly import tools
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['Train_logging_hyperopt']

trail = db.Trail_number_5

R_load = []

for ii in range(295):

#test_data = trail.find_one({"Name": "On_Training"})
    test_data = trail.find_one({"Epsisode_number": ii})

    #R_load.append(test_data['R_load_training'][:])
    R_load = R_load+test_data['R_load_training']




plt.plot(R_load)
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
"""
