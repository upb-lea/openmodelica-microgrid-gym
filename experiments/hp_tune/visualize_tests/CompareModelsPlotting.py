import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px

from openmodelica_microgrid_gym.util import abc_to_dq0

make_pyplot = False
show_load = True
interval_plt = True
# interval_list_x = [[0, 0.015], [6.32, 6.42], [6.4, 6.42]]#, [6.4, 6.42]]#[0.993, 0.997], [0.993, 0.997]]
# interval_list_y = [[-20, 310], [-20, 330], [50, 330]]#, [50, 325]]
interval_list_x = [[0, 0.015], [0.173, 0.177], [0.173, 0.177]]  # [0.993, 0.997], [0.993, 0.997]]
interval_list_y = [[-20, 310], [-5, 210], [150, 201]]  # , [50, 325]]
folder_name = 'saves/Comparison_PI_DDPG_WandWO_pastVals_HPO'
# folder_name = 'saves/Comparison_PI_DDPGs_oneLoadstepPerEpisode'
# name = 'DDPG_PI_local_PastVals_10000steps'
df = pd.read_pickle(folder_name + '/PI_10000steps')
df_DDPG_past_vals = pd.read_pickle(folder_name + '/model_pastVals.zip_10000steps')
# df_DDPG_past_vals = pd.read_pickle('saves/Comparison_PI_DDPGs_oneLoadstepPerEpisode'+'/model_pastVals.zip_100000steps')
df_DDPG = pd.read_pickle(folder_name + '/model_noPastVals_10000steps')
# df_PI = pd.read_pickle(folder_name+'/PI_10000steps')
# df = pd.read_pickle('DDPG_PI_local_10000steps')

env_hist_DDPG = df_DDPG['env_hist_DDPG']

v_a = env_hist_DDPG[0]['lc.capacitor1.v'].tolist()
v_b = env_hist_DDPG[0]['lc.capacitor2.v'].tolist()
v_c = env_hist_DDPG[0]['lc.capacitor3.v'].tolist()
i_a = env_hist_DDPG[0]['lc.inductor1.i'].tolist()
i_b = env_hist_DDPG[0]['lc.inductor2.i'].tolist()
i_c = env_hist_DDPG[0]['lc.inductor3.i'].tolist()
R_load = (env_hist_DDPG[0]['r_load.resistor1.R'].tolist())
phase = env_hist_DDPG[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
i_dq0 = abc_to_dq0(np.array([i_a, i_b, i_c]), phase)

i_d_DDPG = i_dq0[0].tolist()
i_q_DDPG = i_dq0[1].tolist()
i_0_DDPG = i_dq0[2].tolist()
v_d_DDPG = (v_dq0[0].tolist())
v_q_DDPG = (v_dq0[1].tolist())
v_0_DDPG = (v_dq0[2].tolist())

env_hist_DDPG_pastVals = df_DDPG_past_vals['env_hist_DDPG']

v_a_pastVals = env_hist_DDPG_pastVals[0]['lc.capacitor1.v'].tolist()
v_b_pastVals = env_hist_DDPG_pastVals[0]['lc.capacitor2.v'].tolist()
v_c_pastVals = env_hist_DDPG_pastVals[0]['lc.capacitor3.v'].tolist()
i_a_pastVals = env_hist_DDPG_pastVals[0]['lc.inductor1.i'].tolist()
i_b_pastVals = env_hist_DDPG_pastVals[0]['lc.inductor2.i'].tolist()
i_c_pastVals = env_hist_DDPG_pastVals[0]['lc.inductor3.i'].tolist()
R_load_DDPG_pastVals = (env_hist_DDPG_pastVals[0]['r_load.resistor1.R'].tolist())
phase_pastVals = env_hist_DDPG_pastVals[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
v_dq0_pastVals = abc_to_dq0(np.array([v_a_pastVals, v_b_pastVals, v_c_pastVals]), phase_pastVals)
i_dq0_pastVals = abc_to_dq0(np.array([i_a_pastVals, i_b_pastVals, i_c_pastVals]), phase_pastVals)

i_d_DDPG_pastVals = i_dq0_pastVals[0].tolist()
i_q_DDPG_pastVals = i_dq0_pastVals[1].tolist()
i_0_DDPG_pastVals = i_dq0_pastVals[2].tolist()
v_d_DDPG_pastVals = (v_dq0_pastVals[0].tolist())
v_q_DDPG_pastVals = (v_dq0_pastVals[1].tolist())
v_0_DDPG_pastVals = (v_dq0_pastVals[2].tolist())

env_hist_PI = df['env_hist_PI']
v_a_PI = env_hist_PI[0]['lc.capacitor1.v'].tolist()
v_b_PI = env_hist_PI[0]['lc.capacitor2.v'].tolist()
v_c_PI = env_hist_PI[0]['lc.capacitor3.v'].tolist()
i_a_PI = env_hist_PI[0]['lc.inductor1.i'].tolist()
i_b_PI = env_hist_PI[0]['lc.inductor2.i'].tolist()
i_c_PI = env_hist_PI[0]['lc.inductor3.i'].tolist()
R_load_PI = (env_hist_PI[0]['r_load.resistor1.R'].tolist())
phase_PI = env_hist_PI[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
v_dq0_PI = abc_to_dq0(np.array([v_a_PI, v_b_PI, v_c_PI]), phase_PI)
i_dq0_PI = abc_to_dq0(np.array([i_a_PI, i_b_PI, i_c_PI]), phase_PI)
i_d_PI = i_dq0_PI[0].tolist()
i_q_PI = i_dq0_PI[1].tolist()
i_0_PI = i_dq0_PI[2].tolist()
v_d_PI = (v_dq0_PI[0].tolist())
v_q_PI = (v_dq0_PI[1].tolist())
v_0_PI = (v_dq0_PI[2].tolist())

m_d_PI = env_hist_PI[0]['master.md'].tolist()
m_q_PI = env_hist_PI[0]['master.mq'].tolist()
m_0_PI = env_hist_PI[0]['master.m0'].tolist()

m_a_PI = env_hist_PI[0]['master.ma'].tolist()
m_b_PI = env_hist_PI[0]['master.mb'].tolist()
m_c_PI = env_hist_PI[0]['master.mc'].tolist()

return_PR = df['Return PI'][0]
return_DDPG = df_DDPG['Return DDPG'][0]
return_DDPG_pastVals = df_DDPG_past_vals['Return DDPG'][0]

kp_c = df['PI_Kp_c'][0]
ki_c = df['PI_Ki_c'][0]
kp_v = df['PI_Kp_v'][0]
ki_v = df['PI_Ki_v'][0]

ts = 1e-4  # if ts stored: take from db

# t_test_R = np.arange(ts, (len(testcase_100k['v_d_PI'])) * ts, ts).tolist()

t_test = np.arange(0, round((len(v_0_PI)) * ts, 4), ts).tolist()

# plot first interval once
plt.plot(t_test, R_load)
plt.grid()
# plt.xlim([0, 0.025])
plt.xlabel("time")
plt.ylabel("R_load")
plt.title('')
plt.show()

plt.plot(t_test, R_load_DDPG_pastVals)
plt.grid()
# plt.xlim([0, 0.025])
plt.xlabel("time")
plt.ylabel("R_load_DDPG_pastVals")
plt.title('')
plt.show()

plt.plot(t_test, R_load_PI)
plt.grid()
# plt.xlim([0, 0.025])
plt.xlabel("time")
plt.ylabel("R_load_PI")
plt.title('')
plt.show()

plt.plot(t_test, v_d_PI, 'b', label='v_d')
plt.plot(t_test, v_q_PI, 'r', label='v_q')
plt.plot(t_test, v_0_PI, 'g', label='v_0')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
plt.xlim([0, 0.025])
plt.ylim([-50, 250])
plt.xlabel("time")
plt.ylabel("v_dq0_PI")
plt.title(f' PI-return(MRE): {return_PR}')
plt.show()

plt.plot(t_test, v_d_DDPG, 'b')
plt.plot(t_test, v_q_DDPG, 'r')
plt.plot(t_test, v_0_DDPG, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
plt.xlim([0, 0.025])
plt.ylim([-50, 250])
plt.xlabel("time")
plt.ylabel("v_dq0_DDPG")
plt.title(f'DDPG-return(MRE): {return_DDPG}')
plt.show()

plt.plot(t_test, v_d_DDPG_pastVals, 'b')
plt.plot(t_test, v_q_DDPG_pastVals, 'r')
plt.plot(t_test, v_0_DDPG_pastVals, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
plt.xlim([0, 0.025])
plt.ylim([-50, 250])
plt.xlabel("time")
plt.ylabel("v_dq0_DDPG_pV")
plt.title(f'DDPG-return(MRE) using past observations: {return_DDPG_pastVals}')
plt.show()

plt.plot(t_test, v_d_PI, 'b', label='v_d')
plt.plot(t_test, v_q_PI, 'r', label='v_q')
plt.plot(t_test, v_0_PI, 'g', label='v_0')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
# plt.xlim([0, 0.025])
plt.ylim([-50, 300])
plt.xlabel("time")
plt.ylabel("v_dq0_PI")
plt.title(f' PI-return(MSE): {return_PR}')
plt.show()

plt.plot(t_test, v_d_DDPG, 'b')
plt.plot(t_test, v_q_DDPG, 'r')
plt.plot(t_test, v_0_DDPG, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
# plt.xlim([0, 0.025])
plt.ylim([-50, 300])
plt.xlabel("time")
plt.ylabel("v_dq0_DDPG")
plt.title(f'DDPG-return(MSE): {return_DDPG}')
plt.show()

plt.plot(t_test, v_d_DDPG_pastVals, 'b')
plt.plot(t_test, v_q_DDPG_pastVals, 'r')
plt.plot(t_test, v_0_DDPG_pastVals, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
plt.grid()
# plt.xlim([0, 0.025])
plt.ylim([-50, 300])
plt.xlabel("time")
plt.ylabel("v_dq0_DDPG_pV")
plt.title(f'DDPG-return(MRE) using past observations: {return_DDPG_pastVals}')
plt.show()
#############################
############## Subplots
# fig = plt.figure(figsize=(10,12))  # a new figure window
fig, axs = plt.subplots(4, 3, figsize=(16, 12))  # , sharex=True)  # a new figure window
fig.suptitle(f'DDPG-return using i_load-feature(MRE): {return_DDPG} \n '
             f'DDPG-return(MRE) using past observations: {return_DDPG_pastVals}  \n'
             f'PI-return(MRE): {return_PR} \n '
             f'PI: Kp_i = {kp_c}, Ki_i = {ki_c}, Kp_v = {kp_v}, Ki_v = {ki_v}', fontsize=14)

plt_count = 1

i = 0

# ax = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[0, 0].plot(t_test, R_load)
axs[0, 0].grid()
axs[0, 0].set_xlim(interval_list_x[i])
# axs[0, i].set_xlabel("time")
axs[0, 0].set_ylabel("R_load")
# axs[0, i].set_title(f'#{plt_count}')
# plt.show()
plt_count += 1

# ax2 = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[1, 0].plot(t_test, v_d_PI, 'b', label='v_d')
axs[1, 0].plot(t_test, v_q_PI, 'r', label='v_q')
axs[1, 0].plot(t_test, v_0_PI, 'g', label='v_0')
# plt.plot(t_test, v_sp_abc[0, :])
axs[1, 0].grid()
axs[1, 0].set_xlim(interval_list_x[i])
axs[1, 0].set_ylim(interval_list_y[i])

axs[1, 0].set_ylabel("v_dq0_PI")

# ax3 = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[2, 0].plot(t_test, v_d_DDPG, 'b')
axs[2, 0].plot(t_test, v_q_DDPG, 'r')
axs[2, 0].plot(t_test, v_0_DDPG, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
axs[2, 0].grid()
axs[2, 0].set_xlim(interval_list_x[i])
axs[2, 0].set_ylim(interval_list_y[i])
axs[2, 0].set_xlabel("time")
axs[2, 0].set_ylabel("v_dq0_DDPG")

#######
# ax = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[0, 1].plot(t_test, m_d_PI, 'b')
axs[0, 1].plot(t_test, m_q_PI, 'r')
axs[0, 1].plot(t_test, m_0_PI, 'g')
axs[0, 1].grid()
axs[0, 1].set_xlim(interval_list_x[i])
axs[0, 1].set_ylim([-0.2, 0.8])
axs[0, 1].set_ylabel("m_dq0_PI")

# ax2 = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[1, 1].plot(t_test, i_d_PI, 'b', label='i_d')
axs[1, 1].plot(t_test, i_q_PI, 'r', label='i_q')
axs[1, 1].plot(t_test, i_0_PI, 'g', label='i_0')
# plt.plot(t_test, v_sp_abc[0, :])
axs[1, 1].grid()
axs[1, 1].set_xlim(interval_list_x[i])
axs[1, 1].set_ylim([-10, 14])
# axs[1, i].set_xlabel("time")

axs[1, 1].set_ylabel("i_dq0_PI")

axs[2, 1].plot(t_test, i_d_DDPG, 'b')
axs[2, 1].plot(t_test, i_q_DDPG, 'r')
axs[2, 1].plot(t_test, i_0_DDPG, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
axs[2, 1].grid()
axs[2, 1].set_xlim(interval_list_x[i])
axs[2, 1].set_ylim([-10, 14])
axs[2, 1].set_xlabel("time")
axs[2, 1].set_ylabel("i_dq0_DDPG")
# axs[2, i].set_title(f'#{plt_count}')
# plt.show()


#######
actionP0 = df_DDPG['ActionP0'][0]
t_action = np.arange(0, round((len(actionP0)) * ts, 4), ts).tolist()

axs[2, 2].plot(t_action, actionP0, 'b')
axs[2, 2].plot(t_action, df_DDPG['ActionP1'][0], 'r')
axs[2, 2].plot(t_action, df_DDPG['ActionP2'][0], 'g')
axs[2, 2].grid()
axs[2, 2].set_xlim(interval_list_x[i])
axs[2, 2].set_ylabel("action_P_012_DDPG")

"""
axs[1, 2].plot(t_action, df['ActionI0'][0], 'b')
axs[1, 2].plot(t_action, df['ActionI1'][0], 'r')
axs[1, 2].plot(t_action, df['ActionI2'][0], 'g')
axs[1, 2].grid()
axs[1, 2].set_xlim(interval_list_x[i])
#axs[1, 2].set_ylim(interval_list_x[i])
axs[1, 2].set_ylabel("action_I_012_DDPG")
"""
axs[0, 2].plot(t_action, [sum(x) for x in zip(df_DDPG['integrator_sum0'][0], actionP0)], 'b')
axs[0, 2].plot(t_action, [sum(x) for x in zip(df_DDPG['integrator_sum1'][0], df_DDPG['ActionP1'][0])], 'r')
axs[0, 2].plot(t_action, [sum(x) for x in zip(df_DDPG['integrator_sum2'][0], df_DDPG['ActionP2'][0])], 'g')
axs[0, 2].grid()
axs[0, 2].set_xlim(interval_list_x[i])
axs[0, 2].set_ylim([-0.2, 0.8])
axs[0, 2].set_ylabel("m_dq0_DDPG")

axs[1, 2].plot(t_action, df_DDPG['integrator_sum0'][0], 'b')
axs[1, 2].plot(t_action, df_DDPG['integrator_sum1'][0], 'r')
axs[1, 2].plot(t_action, df_DDPG['integrator_sum2'][0], 'g')
axs[1, 2].grid()
axs[1, 2].set_xlim(interval_list_x[i])
axs[1, 2].set_ylim([-0.2, 0.8])
axs[1, 2].set_ylabel("Intergrator_sum_dq0_DDPG")
"""
# ax = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[0, 2].plot(t_test, m_a_PI, 'b')
axs[0, 2].plot(t_test, m_b_PI, 'r')
axs[0, 2].plot(t_test, m_c_PI, 'g')
axs[0, 2].grid()
axs[0, 2].set_xlim(interval_list_x[i])
axs[0, 2].set_ylabel("m_abc_PI")



# ax2 = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
axs[1, 1].plot(t_test, i_d_PI, 'b', label='i_d')
axs[1, 1].plot(t_test, i_q_PI, 'r', label='i_q')
axs[1, 1].plot(t_test, i_0_PI, 'g', label='i_0')
# plt.plot(t_test, v_sp_abc[0, :])
axs[1, 1].grid()
axs[1, 1].set_xlim(interval_list_x[i])
axs[1, 1].set_ylim(interval_list_y[i])
# axs[1, i].set_xlabel("time")

axs[1, 1].set_ylabel("i_dq0_PI")


axs[2, 1].plot(t_test, i_d_DDPG, 'b')
axs[2, 1].plot(t_test, i_q_DDPG, 'r')
axs[2, 1].plot(t_test, i_0_DDPG, 'g')
# plt.plot(t_test, v_sp_abc[0, :])
axs[2, 1].grid()
axs[2, 1].set_xlim(interval_list_x[i])
axs[2, 1].set_ylim(interval_list_y[i])
axs[2, 1].set_xlabel("time")
axs[2, 1].set_ylabel("i_dq0_DDPG")
# axs[2, i].set_title(f'#{plt_count}')
# plt.show()



"""

fig.savefig(f'{folder_name}/overview.pdf')

fig.subplots_adjust(wspace=0.4, hspace=0.2)
plt.show()

if interval_plt:
    ############## Subplots
    # fig = plt.figure(figsize=(10,12))  # a new figure window
    fig, axs = plt.subplots(4, len(interval_list_y), figsize=(16, 12))  # , sharex=True)  # a new figure window
    fig.suptitle(f'DDPG-return(MRE): {return_DDPG} \n '
                 f'DDPG-return(MRE) using past observations: {return_DDPG_pastVals}  \n'
                 f'  PI-return(MRE): {return_PR} \n '
                 f'PI: Kp_i = {kp_c}, Ki_i = {ki_c}, Kp_v = {kp_v}, Ki_v = {ki_v}', fontsize=14)

    plt_count = 1

    for i in range(len(interval_list_y)):

        # ax = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
        axs[0, i].plot(t_test, R_load)
        axs[0, i].grid()
        axs[0, i].set_xlim(interval_list_x[i])
        # axs[0, i].set_xlabel("time")
        if i == 0:
            axs[0, i].set_ylabel("R_load")
        # axs[0, i].set_title(f'#{plt_count}')
        # plt.show()
        plt_count += 1

        # ax2 = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
        axs[1, i].plot(t_test, v_d_PI, 'b', label='v_d')
        axs[1, i].plot(t_test, v_q_PI, 'r', label='v_q')
        axs[1, i].plot(t_test, v_0_PI, 'g', label='v_0')
        # plt.plot(t_test, v_sp_abc[0, :])
        axs[1, i].grid()
        axs[1, i].set_xlim(interval_list_x[i])
        axs[1, i].set_ylim(interval_list_y[i])
        # axs[1, i].set_xlabel("time")
        if i == 0:
            axs[1, i].set_ylabel("v_dq0_PI")
        # axs[1, i].set_title(f'#{plt_count}')
        # plt.show()
        plt_count += 1

        # ax3 = fig.add_subplot(3, len(interval_list_y), plt_count)  # a new axes
        axs[2, i].plot(t_test, v_d_DDPG, 'b')
        axs[2, i].plot(t_test, v_q_DDPG, 'r')
        axs[2, i].plot(t_test, v_0_DDPG, 'g')
        # plt.plot(t_test, v_sp_abc[0, :])
        axs[2, i].grid()
        axs[2, i].set_xlim(interval_list_x[i])
        axs[2, i].set_ylim(interval_list_y[i])
        axs[2, i].set_xlabel("time")
        if i == 0:
            axs[2, i].set_ylabel("v_dq0_DDPG")
        # axs[2, i].set_title(f'#{plt_count}')
        # plt.show()
        plt_count += 1

        axs[3, i].plot(t_test, v_d_DDPG_pastVals, 'b')
        axs[3, i].plot(t_test, v_q_DDPG_pastVals, 'r')
        axs[3, i].plot(t_test, v_0_DDPG_pastVals, 'g')
        axs[3, i].grid()
        axs[3, i].set_xlim(interval_list_x[i])
        axs[3, i].set_ylim(interval_list_y[i])
        axs[3, i].set_xlabel("time")
        if i == 0:
            axs[3, i].set_ylabel("v_dq0_DDPG_pastVals")
        # axs[2, i].set_title(f'#{plt_count}')
        # plt.show()
        plt_count += 1

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

    fig.savefig(f'{folder_name}/Ausschnitt.pdf')

if make_pyplot:
    # pyplot PI
    plot = px.Figure()
    plot.add_trace(
        px.Scatter(x=t_test, y=v_d_PI))
    plot.add_trace(
        px.Scatter(x=t_test, y=v_q_PI))
    plot.add_trace(
        px.Scatter(x=t_test, y=v_0_PI))
    # plot.add_trace(
    #    px.Scatter(x=t_test, y=v_sp_abc[1, :]))

    plot.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
        dict(count=1, step="day", stepmode="backward"), ])),
        rangeslider=dict(visible=True), ))
    plot.show()

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

    plot.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
        dict(count=1, step="day", stepmode="backward"), ])),
        rangeslider=dict(visible=True), ))
    plot.show()

    # pyplot Load
    plot = px.Figure()
    plot.add_trace(
        px.Scatter(x=t_test, y=R_load))

    plot.update_layout(xaxis=dict(rangeselector=dict(buttons=list([
        dict(count=1, step="day", stepmode="backward"), ])),
        rangeslider=dict(visible=True), ))
    plot.show()
