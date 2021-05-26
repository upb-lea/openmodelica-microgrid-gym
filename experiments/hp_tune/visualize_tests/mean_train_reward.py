import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import sshtunnel
from plotly import tools
from pymongo import MongoClient

# client = MongoClient('mongodb://localhost:27017/')

mongo_tunnel = sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001))
mongo_tunnel.start()
mongo_recorder = MongoClient(
    f'mongodb://localhost:{mongo_tunnel.local_bind_port}/')  # store to port 12001 for ssh data to cyberdyne

db = mongo_recorder['DDPG_Lr_gamma_Anoise']

"""
trail = db.Trail_number_119
number_learning_steps = 500000
episode_len = 2000
learning_rate = 1.6e-6
lr_decay_start = 0.18
lr_decay_duration = 0.43
final_lr = 0.23 * learning_rate
"""

trial = db.Trail_number_101

trial_config = trial.find_one({"Name": "Config"})

number_learning_steps = trial_config['Number_learning_Steps']
episode_len = 2000  # trial_config['training_episode_length']
learning_rate = trial_config['learning_rate']
lr_decay_start = trial_config['lr_decay_start']
lr_decay_duration = trial_config['lr_decay_duration']
final_lr = trial_config['final_lr'] * learning_rate

for post in trial.find({"Name": "After_Training"}):
    test_data = post

# test_data = trail.find_one({"Name": "After_Training"})
# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
# ax1, ax2 = ax.flatten()
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
# plt.yscale('log')
plt.ylabel("Mean episode Reward")
#plt.title("1.000.000")
plt.show()


plot = px.Figure()
plot.add_trace(
    px.Scatter(y=y))


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
 Lernrate plotten - hier: Konstruiert aus den HP-Daten, kein Log - ggf. prüfen via Tensorboard 
"""

t = np.arange(number_learning_steps)

progress_remaining = 1.0 - (t / number_learning_steps)

t_start = int(lr_decay_start * number_learning_steps)
t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                       number_learning_steps))

lr_curve = np.maximum(
    np.minimum(learning_rate, learning_rate + (t_start * (learning_rate - final_lr)) / (t_end - t_start) \
               - (learning_rate - final_lr) / (t_end - t_start) * ((1.0 - progress_remaining) \
                                                                   * number_learning_steps)), final_lr)

# no step-vise MA needed but episode-vise!
# lr_ma = np.convolve(lr_curve, np.ones(episode_len), 'valid') / episode_len
num_episodes = int(number_learning_steps / episode_len)
lr_ma = np.zeros(num_episodes)
count = 0

for i in range(num_episodes):
    lr_ma[i] = np.mean(lr_curve[count:count + episode_len])
    count += episode_len

# plt.plot(lr_curve)
# plt.show()

fig = plt.figure()  # a new figure window
ax = fig.add_subplot(2, 1, 1)  # a new axes
ax = plt.plot(y)
plt.grid()
# plt.xlabel("Episodes")
# plt.yscale('log')
plt.ylabel("Mean episode Reward")

ax2 = fig.add_subplot(2, 1, 2)  # a new axes
ax2 = plt.plot(lr_ma)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Mean episode LR")
plt.show()

plt.show()
