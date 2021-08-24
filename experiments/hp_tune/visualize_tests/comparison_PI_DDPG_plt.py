import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import sshtunnel
from bson import ObjectId
from plotly import tools
from pymongo import MongoClient

from openmodelica_microgrid_gym.util import dq0_to_abc, abc_to_dq0

db_name = 'Comparison_PI_DDPG'

interval_list_x = [[0, 0.025], [0.49, 0.62], [0.498, 0.51], [0.57, 0.587]]
interval_list_y = [[-100, 280], [-120, 375], [-100, 225], [-50, 375]]

make_pyplot = False
show_load = True
# trial_name = 'Comparison_4D_optimizedPIPI__unsafe2'
trial_name = 'Comparison_4D_optimizedPIPI_reset_100'

with sshtunnel.open_tunnel('lea38', remote_bind_address=('127.0.0.1', 12001)) as tun:
    with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
        db = client[db_name]

        # get data from db
        # trial = db['Comparison_numSteps_[1000, 5000, 10000, 20000, 50000, 100000]_average_1']
        trial = db[trial_name]

        # find last trial/collection
        # for trials in trial.find():
        #    testcase_100k = trials

        testcase_100k = trial.find_one({"max_episode_steps": "100000"})

        return_PR = testcase_100k['Return PI']
        return_DDPG = testcase_100k['Return DDPG']

        kp_c = testcase_100k['PI_Kp_c']
        ki_c = testcase_100k['PI_Ki_c']
        kp_v = testcase_100k['PI_Kp_v']
        ki_v = testcase_100k['PI_Ki_v']

        ts = 1e-4  # if ts stored: take from db
        t_test = np.arange(0, len(testcase_100k['v_d_PI']) * ts, ts).tolist()

        v_d_PI = testcase_100k['v_d_PI']
        v_q_PI = testcase_100k['v_q_PI']
        v_0_PI = testcase_100k['v_0_PI']

        v_d_DDPG = testcase_100k['v_d_DDPG']
        v_q_DDPG = testcase_100k['v_q_DDPG']
        v_0_DDPG = testcase_100k['v_0_DDPG']

        R_load = testcase_100k['R_load']

        # plot first interval once
        plt.plot(t_test, R_load)
        plt.grid()
        plt.xlim([0, 0.025])
        plt.xlabel("time")
        plt.ylabel("R_load")
        plt.title('')
        plt.show()

        plt.plot(t_test, v_d_PI, 'b', label='v_d')
        plt.plot(t_test, v_q_PI, 'r', label='v_q')
        plt.plot(t_test, v_0_PI, 'g', label='v_0')
        # plt.plot(t_test, v_sp_abc[0, :])
        plt.grid()
        plt.xlim([0, 0.025])
        plt.ylim([-150, 250])
        plt.xlabel("time")
        plt.ylabel("v_dq0_PI")
        plt.title(f' PI-return(MSE): {return_PR}')
        plt.show()

        plt.plot(t_test, v_d_DDPG, 'b')
        plt.plot(t_test, v_q_DDPG, 'r')
        plt.plot(t_test, v_0_DDPG, 'g')
        # plt.plot(t_test, v_sp_abc[0, :])
        plt.grid()
        plt.xlim([0, 0.025])
        plt.ylim([-150, 250])
        plt.xlabel("time")
        plt.ylabel("v_dq0_DDPG")
        plt.title(f'DDPG-return(MSE): {return_DDPG}')
        plt.show()

        ############## Subplots
        # fig = plt.figure(figsize=(10,12))  # a new figure window
        fig, axs = plt.subplots(3, len(interval_list_y), figsize=(16, 12))  # , sharex=True)  # a new figure window
        fig.suptitle(f'DDPG-return(MSE): {return_DDPG} \n   PI-return(MSE): {return_PR} \n '
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

        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.show()

        fig.savefig(f'{trial_name}.pdf')

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
