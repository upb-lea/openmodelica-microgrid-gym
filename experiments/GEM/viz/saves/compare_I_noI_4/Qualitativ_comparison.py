import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px

from openmodelica_microgrid_gym.util import abc_to_dq0

make_pyplot = False
show_load = True
interval_plt = True

# Fuer den Detzerministc case
interval_list_x = [[0.197, 0.206], [0.795, 0.81], [0.8025, 0.81], [0.849, 0.855]]
interval_list_x = [[0.699, 0.705], [0.499, 0.505]]  # ,[0.8025, 0.81], [0.849, 0.855]]
interval_list_y_q = [[-0.9, 0.9], [-1.2, 0.7], [0.55, 0.725], [-0.2, 0.67]]
interval_list_y_d = [[-1.5, -0.1], [-1.2, 0.2], [-0.3, -0.1], [-1.1, -0.21]]
interval_list_y_q = [[-1, 1], [-1, 1]]  # , [-1, 1], [-1, 1]]
interval_list_y_d = [[-1, 1], [-1, 1]]  # , [-1, 1], [-1, 1]]

file_names = ['_GEM_I_term_4_trial770', '_GEM_no_I_term_4_trial1192']
ylabels = ['I', 'no-I']

reward_list_DDPG = []

# fig, axs = plt.subplots(len(model_names)+2, len(interval_list_y), figsize=(16, 12))  # , sharex=True)  # a new figure window
fig, axs = plt.subplots(len(file_names) * 2 + 1, len(interval_list_y_q),
                        figsize=(14, 10))  # , sharex=True)  # a new figure window

for i in range(len(interval_list_y_q)):
    plt_count = 1
    ############## Subplots
    # fig = plt.figure(figsize=(10,12))  # a new figure window

    for file_name, ylabel_use in zip(file_names, ylabels):

        df_DDPG = pd.read_pickle(file_name)
        # df_DDPG = pd.read_pickle(folder_name + '/' 'model_5_pastVals.zip_100000steps_NoPhaseFeature_1427')

        ts = 1e-4  # if ts stored: take from db
        t_test = np.arange(0, len(df_DDPG['i_d_mess'][0]) * ts, ts).tolist()

        Name = df_DDPG['Name'].tolist()[0]
        reward = df_DDPG['Reward_test'].tolist()[0]

        if plt_count == 1:
            axs[0, i].plot(t_test, reward, 'b', label=f'      {Name}: '
                                                      f'{round(sum(reward[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')
        else:
            axs[0, i].plot(t_test, reward, 'r', label=f'{Name}: '
                                                      f'{round(sum(reward[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')
        axs[0, i].grid()
        axs[0, i].set_xlim(interval_list_x[i])
        # axs[0, i].set_ylim(interval_list_y[i])
        axs[0, i].legend()
        if 0 == 0:
            axs[0, i].set_ylabel("Reward")

        axs[plt_count, i].plot(t_test, df_DDPG['i_q_mess'].tolist()[0], 'b', label='i_q')
        axs[plt_count, i].plot(t_test, df_DDPG['i_q_ref'].tolist()[0], 'r', label='i_q_ref')
        axs[plt_count, i].grid()
        axs[plt_count, i].legend()
        axs[plt_count, i].set_xlim(interval_list_x[i])
        axs[plt_count, i].set_ylim(interval_list_y_q[i])
        axs[plt_count, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
        if i == 0:
            # axs[plt_count, i].set_ylabel(pV)
            axs[plt_count, i].set_ylabel(Name)
            # axs[plt_count, i].set_ylabel("$v_{\mathrm{dq0, DDPG}}\,/\,\mathrm{V}$")
        # else:
        #    axs[plt_count, i].set_ylabel("$v_{\mathrm{q0, DDPG}}\,/\,\mathrm{V}$")
        plt_count += 1

        axs[plt_count, i].plot(t_test, df_DDPG['i_d_mess'].tolist()[0], 'b', label='i_d')
        axs[plt_count, i].plot(t_test, df_DDPG['i_d_ref'].tolist()[0], 'r', label='i_d_ref')
        axs[plt_count, i].grid()
        axs[plt_count, i].legend()
        axs[plt_count, i].set_xlim(interval_list_x[i])
        axs[plt_count, i].set_ylim(interval_list_y_d[i])
        axs[plt_count, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
        if i == 0:
            # axs[plt_count, i].set_ylabel(pV)
            axs[plt_count, i].set_ylabel(Name)
            # axs[plt_count, i].set_ylabel("$v_{\mathrm{dq0, DDPG}}\,/\,\mathrm{V}$")
        # else:
        #    axs[plt_count, i].set_ylabel("$v_{\mathrm{q0, DDPG}}\,/\,\mathrm{V}$")
        plt_count += 1
        """
        axs[plt_count, i].plot(t_test, df_DDPG['v_d_mess'].tolist()[0][:-1], 'b', label='v_d')
        axs[plt_count, i].plot(t_test, df_DDPG['v_q_mess'].tolist()[0][:-1], 'r', label='v_q')
        axs[plt_count, i].grid()
        axs[plt_count, i].legend()
        axs[plt_count, i].set_xlim(interval_list_x[i])
        #axs[plt_count, i].set_ylim(interval_list_y_d[i])
        axs[plt_count, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
        if i == 0:
            # axs[plt_count, i].set_ylabel(pV)
            axs[plt_count, i].set_ylabel(Name)
            # axs[plt_count, i].set_ylabel("$v_{\mathrm{dq0, DDPG}}\,/\,\mathrm{V}$")
        # else:
        #    axs[plt_count, i].set_ylabel("$v_{\mathrm{q0, DDPG}}\,/\,\mathrm{V}$")
        plt_count += 1
        """

"""
fig.suptitle(f'Model using pastVals:' + str(pastVals) + ' \n '
                                                        f'Model-return(MRE)' + str(return_list_DDPG) + ' \n'
                                                                                                          f'  PI-return(MRE):     {round(return_PI, 7)} \n '
                                                                                                          f'PI: Kp_i = {kp_c}, Ki_i = {ki_c}, Kp_v = {kp_v}, Ki_v = {ki_v}',
             fontsize=14)
"""
fig.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

fig.savefig(f'/Ausschnitt.pdf')

"""
plt.plot(t_test, trial_test['i_d_mess'])
plt.plot(t_test, trial_test['i_d_ref'], 'r')
plt.plot(t_test, trial_test['i_d_ref'], 'r')
plt.grid()
plt.xlabel("t")
plt.ylabel("i_d")
plt.title(f"Test{db_name}")
plt.show()

plt.plot(t_test, df_DDPG['i_q_mess'].tolist())
plt.plot(t_test, df_DDPG['i_q_ref'].tolist(), 'r')
plt.grid()
plt.xlabel("t")
plt.ylabel("i_q")
plt.title(f"Test {Name}")
plt.show()

plt.plot(t_test, trial_test['Reward'])
plt.grid()
plt.xlabel("t")
plt.ylabel("reward")
plt.title(f"Test {db_name}")
plt.show()
"""
