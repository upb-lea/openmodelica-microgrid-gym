import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import matplotlib

from openmodelica_microgrid_gym.util import abc_to_dq0

save_results = True

# Fuer den 10s Fall
interval_list_x = [[0, 0.005], [7.145, 7.155]]
interval_list_y = [[-5, 202], [80, 345]]

folder_name = 'saves/paper_new2'  # _deterministic'

number_of_steps = '_100000steps'

df = pd.read_pickle(folder_name + '/PI' + number_of_steps)

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

i_a_PI = env_hist_PI[0]['lc.inductor1.i'].tolist()
i_b_PI = env_hist_PI[0]['lc.inductor2.i'].tolist()
i_c_PI = env_hist_PI[0]['lc.inductor3.i'].tolist()
i_dq0_PI = abc_to_dq0(np.array([i_a_PI, i_b_PI, i_c_PI]), phase_PI)
i_d_PI = (i_dq0_PI[0].tolist())
i_q_PI = (i_dq0_PI[1].tolist())
i_0_PI = (i_dq0_PI[2].tolist())

reward_PI = df['Reward PI'][0]
return_PI = df['Return PI'][0]
kp_c = df['PI_Kp_c'][0]
ki_c = df['PI_Ki_c'][0]
kp_v = df['PI_Kp_v'][0]
ki_v = df['PI_Ki_v'][0]

model_names = ['model_OMG_DDPG_Actor.zip',
               'model_OMG_DDPG_Integrator_no_pastVals.zip']
ylabels = ['DDPG', 'DDPG-I']

return_list_DDPG = []
reward_list_DDPG = []

ts = 1e-4  # if ts stored: take from db

v_d_ref = [169.7] * len(v_0_PI)

t_test = np.arange(0, round((len(v_0_PI)) * ts, 4), ts).tolist()
t_reward = np.arange(0, round((len(reward_PI)) * ts, 4), ts).tolist()

# fig, axs = plt.subplots(len(model_names) + 4, len(interval_list_y),
fig, axs = plt.subplots(3, len(interval_list_y),
                        figsize=(9, 7))  # , sharex=True)  # a new figure window

for i in range(len(interval_list_y)):
    plt_count = 4
    ############## Subplots
    # fig = plt.figure(figsize=(10,12))  # a new figure window

    df_DDPG = pd.read_pickle(folder_name + '/' + model_names[0] + number_of_steps)

    if i == 0:
        return_list_DDPG.append(round(df_DDPG['Return DDPG'][0], 7))
    #    reward_list_DDPG.append(df_DDPG['Reward DDPG'][0])

    env_hist_DDPG = df_DDPG['env_hist_DDPG']

    v_a = env_hist_DDPG[0]['lc.capacitor1.v'].tolist()
    v_b = env_hist_DDPG[0]['lc.capacitor2.v'].tolist()
    v_c = env_hist_DDPG[0]['lc.capacitor3.v'].tolist()
    i_a = env_hist_DDPG[0]['lc.inductor1.i'].tolist()
    i_b = env_hist_DDPG[0]['lc.inductor2.i'].tolist()
    i_c = env_hist_DDPG[0]['lc.inductor3.i'].tolist()
    phase = env_hist_DDPG[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
    v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
    i_dq0 = abc_to_dq0(np.array([i_a, i_b, i_c]), phase)
    v_d_DDPG = (v_dq0[0].tolist())
    v_q_DDPG = (v_dq0[1].tolist())
    v_0_DDPG = (v_dq0[2].tolist())
    i_d_DDPG = (i_dq0[0].tolist())
    i_q_DDPG = (i_dq0[1].tolist())
    i_0_DDPG = (i_dq0[2].tolist())

    DDPG_reward = df_DDPG['Reward DDPG'][0]

    df_DDPG_I = pd.read_pickle(folder_name + '/' + model_names[1] + number_of_steps)

    if i == 0:
        return_list_DDPG.append(round(df_DDPG_I['Return DDPG'][0], 7))
    #    reward_list_DDPG.append(df_DDPG['Reward DDPG'][0])

    env_hist_DDPG_I = df_DDPG_I['env_hist_DDPG']

    v_a_I = env_hist_DDPG_I[0]['lc.capacitor1.v'].tolist()
    v_b_I = env_hist_DDPG_I[0]['lc.capacitor2.v'].tolist()
    v_c_I = env_hist_DDPG_I[0]['lc.capacitor3.v'].tolist()
    i_a_I = env_hist_DDPG_I[0]['lc.inductor1.i'].tolist()
    i_b_I = env_hist_DDPG_I[0]['lc.inductor2.i'].tolist()
    i_c_I = env_hist_DDPG_I[0]['lc.inductor3.i'].tolist()
    phase_I = env_hist_DDPG_I[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
    v_dq0_I = abc_to_dq0(np.array([v_a_I, v_b_I, v_c_I]), phase_I)
    i_dq0_I = abc_to_dq0(np.array([i_a_I, i_b_I, i_c_I]), phase_I)
    v_d_DDPG_I = (v_dq0_I[0].tolist())
    v_q_DDPG_I = (v_dq0_I[1].tolist())
    v_0_DDPG_I = (v_dq0_I[2].tolist())
    i_d_DDPG_I = (i_dq0_I[0].tolist())
    i_q_DDPG_I = (i_dq0_I[1].tolist())
    i_0_DDPG_I = (i_dq0_I[2].tolist())

    DDPG_reward_I = df_DDPG_I['Reward DDPG'][0]

    axs[2, i].plot(t_reward, reward_PI, 'b', label=f'          PI: '
                                                   f'{round(sum(reward_PI[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')
    axs[2, i].plot(t_reward, DDPG_reward, 'r', label=f'    DDPG: '
                                                     f'{round(sum(DDPG_reward[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')
    axs[2, i].plot(t_reward, DDPG_reward_I, 'g', label=f'DDPG+I: '
                                                       f'{round(sum(DDPG_reward_I[int(interval_list_x[i][0] / ts):int(interval_list_x[i][1] / ts)]) / ((interval_list_x[i][1] - interval_list_x[i][0]) / ts), 4)}')

    axs[2, i].grid()
    axs[2, i].set_xlim(interval_list_x[i])
    # axs[1, i].set_ylim(interval_list_y[i])
    axs[2, i].legend()
    axs[2, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
    if i == 0:
        axs[2, i].set_ylabel("Reward")

    axs[0, i].plot(t_test, v_d_ref, '--', color='gray')
    axs[0, i].plot(t_test, v_d_PI, 'b', label='PI')
    axs[0, i].plot(t_test, v_d_DDPG, 'r', label='DDPG')
    axs[0, i].plot(t_test, v_d_DDPG_I, 'g', label='DDPG+I')
    # axs[2, i].plot(t_test, v_q_PI, 'r', label='v_q')
    # axs[2, i].plot(t_test, v_0_PI, 'g', label='v_0')
    axs[0, i].grid()
    axs[0, i].legend()
    axs[0, i].set_xlim(interval_list_x[i])
    axs[0, i].set_ylim(interval_list_y[i])
    if i == 0:
        axs[0, i].set_ylabel("$v_{\mathrm{d}}\,/\,\mathrm{V}$")
    # else:
    #    axs[1, i].set_ylabel("$v_{\mathrm{q0, PI}}\,/\,\mathrm{V}$")

    axs[1, i].plot(t_test, i_d_PI, 'b', label='PI')
    axs[1, i].plot(t_test, i_d_DDPG, 'r', label='DDPG')
    axs[1, i].plot(t_test, i_d_DDPG_I, 'g', label='DDPG+I')
    # axs[1, i].plot(t_test, i_q_PI, 'r', label='v_q')
    # axs[1, i].plot(t_test, i_0_PI, 'g', label='v_0')
    axs[1, i].grid()
    # axs[1, i].legend()
    axs[1, i].set_xlim(interval_list_x[i])
    # axs[3, i].set_ylim(interval_list_y[i])
    if i == 0:
        axs[1, i].set_ylim([0, 15])
        axs[1, i].set_ylabel("$i_{\mathrm{d}}\,/\,\mathrm{A}$")

    """
    axs[plt_count, i].plot(t_test, v_d_DDPG, 'b')
    axs[plt_count, i].plot(t_test, v_q_DDPG, 'r')
    axs[plt_count, i].plot(t_test, v_0_DDPG, 'g')
    axs[plt_count, i].grid()
    axs[plt_count, i].set_xlim(interval_list_x[i])
    axs[plt_count, i].set_ylim(interval_list_y[i])
    axs[plt_count, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
    if i == 0:

        axs[plt_count, i].set_ylabel("$v_{\mathrm{dq0, DDPG}}\,/\,\mathrm{V}$")
    # else:
    #    axs[plt_count, i].set_ylabel("$v_{\mathrm{q0, DDPG}}\,/\,\mathrm{V}$")
    plt_count += 1

    axs[plt_count, i].plot(t_test, i_d_DDPG, 'b')
    axs[plt_count, i].plot(t_test, i_q_DDPG, 'r')
    axs[plt_count, i].plot(t_test, i_0_DDPG, 'g')
    axs[plt_count, i].grid()
    axs[plt_count, i].set_xlim(interval_list_x[i])
    #axs[plt_count, i].set_ylim(interval_list_y[i])
    axs[plt_count, i].set_xlabel(r'$t\,/\,\mathrm{s}$')
    if i == 0:
        axs[plt_count, i].set_ylabel("$i_{\mathrm{dq0, DDPG}}\,/\,\mathrm{A}$")
    """

# fig.suptitle(f'Model using pastVals:' + str(14) + ' \n '
#                                                        f'Model-return(MRE)' + str(return_list_DDPG) + ' \n'
#                                                                                                       f'  PI-return(MRE):     {round(return_PI, 7)} \n '
#                                                                                                       f'PI: Kp_i = {kp_c}, Ki_i = {ki_c}, Kp_v = {kp_v}, Ki_v = {ki_v}',
#             fontsize=14)

fig.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

fig.savefig(f'{folder_name}/Ausschnitt_2pV_q0.pdf')
if save_results:
    # Plot setting
    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'
                                      r'\usepackage{amsmath,amssymb,mathtools}'
                                      r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                      r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [3.9, 3.1],
              'font.family': 'serif',
              'lines.linewidth': 1
              }
    matplotlib.rcParams.update(params)

    fig.savefig(f'{folder_name}/_OMG_U_I.png')
    fig.savefig(f'{folder_name}/_OMG_U_I.pdf')
    fig.savefig(f'{folder_name}/_OMG_U_I.pgf')
