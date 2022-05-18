import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px

from openmodelica_microgrid_gym.util import abc_to_dq0

make_pyplot = False
show_load = True
interval_plt = True

# interval_list_x = [[0, 0.01], [0.01, 1.0], [0.78, 0.9]]
# interval_list_y = [[-25, 210], [-40, 210], [165, 175]]


# Fuer den Detzerministc case
interval_list_x = [[0, 0.01], [0.105, 0.2], [0.695, 0.71], [0.85, 0.88]]
interval_list_y = [[-25, 400], [-25, 400], [-25, 400], [-25, 400]]

# Fuer den 10s Fall
# interval_list_x = [[0, 0.02], [2.09, 2.1], [2.11, 2.12], [7.08, 7.16], [7.145, 7.16]]
# interval_list_y = [[-50, 400], [-25, 340], [160, 190], [-25, 340], [125, 340]]

run = '204 Return: -2 '
# run = '374 Return: 0 '
# run = '213 Return: 0.8 '
run = '1080 Return: 0.9 '

folder_names = ['saves/P10_SEC_R_load']  # _deterministic'


number_of_steps = '_1000steps'
"""
df = pd.read_pickle('saves/P10_I_term_2/204_deterministic/PI' + number_of_steps)

env_hist_PI = df['env_hist_PI']
v_a_PI = env_hist_PI[0]['lc.capacitor1.v'].tolist()
v_b_PI = env_hist_PI[0]['lc.capacitor2.v'].tolist()
v_c_PI = env_hist_PI[0]['lc.capacitor3.v'].tolist()
R_load_PI = (env_hist_PI[0]['r_load.resistor1.R'].tolist())
phase_PI = env_hist_PI[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
v_dq0_PI = abc_to_dq0(np.array([v_a_PI, v_b_PI, v_c_PI]), phase_PI)
v_d_PI = (v_dq0_PI[0].tolist())
v_q_PI = (v_dq0_PI[1].tolist())
v_0_PI = (v_dq0_PI[2].tolist())

reward_PI = df['Reward PI'][0]
return_PI = df['Return PI'][0]
kp_c = df['PI_Kp_c'][0]
ki_c = df['PI_Ki_c'][0]
kp_v = df['PI_Kp_v'][0]
ki_v = df['PI_Ki_v'][0]
"""
model_names = ['model.zip']
ylabels = ['DDPG-I+pastVals']

pastVals = ['16']
return_list_DDPG = []
reward_list_DDPG = []

ts = 1e-4  # if ts stored: take from db

# t_test_R = np.arange(ts, (len(testcase_100k['v_d_PI'])) * ts, ts).tolist()

t_test = np.arange(0, round((len(v_0_PI)) * ts, 4), ts).tolist()
t_reward = np.arange(0, round((len(reward_PI)) * ts, 4), ts).tolist()

# fig, axs = plt.subplots(len(model_names)+2, len(interval_list_y), figsize=(16, 12))  # , sharex=True)  # a new figure window
fig, axs = plt.subplots(len(model_names) + 3, len(interval_list_y),
                        figsize=(12, 10))  # , sharex=True)  # a new figure window

for i in range(len(interval_list_y)):
    plt_count = 3
    ############## Subplots
    # fig = plt.figure(figsize=(10,12))  # a new figure window

    for model_name, pV, folder_name, ylabel_use in zip(model_names, pastVals, folder_names, ylabels):

        df_DDPG = pd.read_pickle(folder_name + '/' + model_name + number_of_steps)
        # df_DDPG = pd.read_pickle(folder_name + '/' 'model_5_pastVals.zip_100000steps_NoPhaseFeature_1427')

        if i == 0:
            return_list_DDPG.append(round(df_DDPG['Return DDPG'][0], 7))
        #    reward_list_DDPG.append(df_DDPG['Reward DDPG'][0])

        env_hist_DDPG = df_DDPG['env_hist_DDPG']

        v_a = env_hist_DDPG[0]['lc.capacitor1.v'].tolist()
        v_b = env_hist_DDPG[0]['lc.capacitor2.v'].tolist()
        v_c = env_hist_DDPG[0]['lc.capacitor3.v'].tolist()
        phase = env_hist_DDPG[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
        v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
        v_d_DDPG = (v_dq0[0].tolist())
        v_q_DDPG = (v_dq0[1].tolist())
        v_0_DDPG = (v_dq0[2].tolist())

        axs[0, i].plot(t_test, R_load_PI)
        axs[0, i].grid()
        axs[0, i].set_xlim(interval_list_x[i])
        # axs[0, i].set_ylim([15, 75])
        if i == 0:
            axs[0, i].set_ylabel("$R_{\mathrm{load}}\,/\,\mathrm{\Omega}$")
        # ax.set_xlabel(r'$t\,/\,\mathrm{s}$')

        DDPG_reward = df_DDPG['Reward DDPG'][0]
        if plt_count == 3:
            axs[1, i].plot(t_reward, reward_PI, 'b', label=f'      PI: '
                                                           f'{round(sum(reward_PI[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')
        axs[1, i].plot(t_reward, DDPG_reward, 'r', label=f'DDPG: '
                                                         f'{round(sum(DDPG_reward[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')
        axs[1, i].grid()
        axs[1, i].set_xlim(interval_list_x[i])
        # axs[1, i].set_ylim(interval_list_y[i])
        axs[1, i].legend()
        if i == 0:
            axs[1, i].set_ylabel("Reward")

        axs[2, i].plot(t_test, v_d_PI, 'b', label='v_d')
        axs[2, i].plot(t_test, v_q_PI, 'r', label='v_q')
        axs[2, i].plot(t_test, v_0_PI, 'g', label='v_0')
        axs[2, i].grid()
        axs[2, i].set_xlim(interval_list_x[i])
        axs[2, i].set_ylim(interval_list_y[i])
        if i == 0:
            axs[2, i].set_ylabel("$v_{\mathrm{dq0, PI}}\,/\,\mathrm{V}$")
        # else:
        #    axs[1, i].set_ylabel("$v_{\mathrm{q0, PI}}\,/\,\mathrm{V}$")

        axs[plt_count, i].plot(t_test, v_d_DDPG, 'b')
        axs[plt_count, i].plot(t_test, v_q_DDPG, 'r')
        axs[plt_count, i].plot(t_test, v_0_DDPG, 'g')
        axs[plt_count, i].grid()
        axs[plt_count, i].set_xlim(interval_list_x[i])
        axs[plt_count, i].set_ylim(interval_list_y[i])
        axs[plt_count, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
        if i == 0:
            # axs[plt_count, i].set_ylabel(pV)
            axs[plt_count, i].set_ylabel(ylabel_use)
            # axs[plt_count, i].set_ylabel("$v_{\mathrm{dq0, DDPG}}\,/\,\mathrm{V}$")
        # else:
        #    axs[plt_count, i].set_ylabel("$v_{\mathrm{q0, DDPG}}\,/\,\mathrm{V}$")
        plt_count += 1

fig.suptitle(run)

fig.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

fig.savefig(f'{folder_name}/Ausschnitt_q0' + number_of_steps + '.pdf')

if make_pyplot:
    # pyplot Load
    plot = px.Figure()
    plot.add_trace(
        px.Scatter(x=t_test, y=R_load_PI))  # , title='R_load')

    plot.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
        dict(count=1, step="day", stepmode="backward"), ])),
        rangeslider=dict(visible=True), ))
    plot.show()

    # pyplot PI
    plot = px.Figure()
    plot.add_trace(
        px.Scatter(x=t_reward, y=DDPG_reward))
    plot.add_trace(
        px.Scatter(x=t_reward, y=reward_PI))
    # plot.add_trace(
    #    px.Scatter(x=t_test, y=v_sp_abc[1, :]))

    plot.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
        dict(count=1, step="day", stepmode="backward"), ])),
        rangeslider=dict(visible=True), ))
    plot.show()

    for model_name in model_names:
        df_DDPG = pd.read_pickle(folder_name + '/' + model_name + number_of_steps)

        env_hist_DDPG = df_DDPG['env_hist_DDPG']

        v_a = env_hist_DDPG[0]['lc.capacitor1.v'].tolist()
        v_b = env_hist_DDPG[0]['lc.capacitor2.v'].tolist()
        v_c = env_hist_DDPG[0]['lc.capacitor3.v'].tolist()
        phase = env_hist_DDPG[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
        v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
        v_d_DDPG = (v_dq0[0].tolist())
        v_q_DDPG = (v_dq0[1].tolist())
        v_0_DDPG = (v_dq0[2].tolist())
        # pyplot ddpg
        plot = px.Figure()
        plot.add_trace(
            px.Scatter(x=t_test, y=v_d_DDPG))
        plot.add_trace(
            px.Scatter(x=t_test, y=v_q_DDPG))
        plot.add_trace(
            px.Scatter(x=t_test, y=v_0_DDPG))
        # plot.add_trace(
        #    px.Scatter(x=t_test, y=v_sp_abc[1, :]))
        plot.add_trace(
            px.Scatter(x=t_test, y=v_d_PI))
        plot.add_trace(
            px.Scatter(x=t_test, y=v_q_PI))
        plot.add_trace(
            px.Scatter(x=t_test, y=v_0_PI))

        plot.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
            dict(count=1, step="day", stepmode="backward"), ])),
            rangeslider=dict(visible=True), ))
        plot.show()

plt.plot(t_test, v_d_DDPG, 'b')
plt.plot(t_test, v_q_DDPG, 'r')
plt.plot(t_test, v_0_DDPG, 'g')
# plt.plot(t_test, v_d_PI, 'r')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
# plt.xlim([0.1, 0.11])
# plt.ylim([290, 360])
plt.xlabel("time")
plt.ylabel("v_dq0_DDPG")
plt.title(f'DDPG' + run)
plt.show()

plt.plot(t_test, v_d_PI, 'b')
plt.plot(t_test, v_q_PI, 'r')
plt.plot(t_test, v_q_PI, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
# plt.xlim([0.1, 0.2])
# plt.ylim([290, 360])
plt.xlabel("time")
plt.ylabel("v_dq0_PI")
plt.title(f'PI')
plt.show()
