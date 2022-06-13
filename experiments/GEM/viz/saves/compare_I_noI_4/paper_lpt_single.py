import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.util import abc_to_dq0

save_results = False

# Fuer den 10s Fall
interval_list_x = [0.498, 0.505]
interval_list_y = [80, 345]

if save_results:
    # Plot setting
    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'
                                      r'\usepackage{amsmath,amssymb,mathtools}'
                                      r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                      r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
              'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'font.size': 10,  # was 10
              'legend.fontsize': 10,  # was 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [4.5, 4.6],  # [5.4, 6],#[3.9, 3.1],
              'font.family': 'serif',
              'lines.linewidth': 1
              }
    matplotlib.rcParams.update(params)

folder_name = 'saves/'  # _deterministic'

df_DDPG = pd.read_pickle('_GEM_no_I_term_4_trial1192')
df_DDPG_I = pd.read_pickle('_GEM_I_term_4_trial770')

ts = 1e-4
t_test = np.arange(0, len(df_DDPG['i_d_mess'][0]) * ts, ts).tolist()

# fig, axs = plt.subplots(len(model_names) + 4, len(interval_list_y),
fig = plt.figure()

iq_I = df_DDPG_I['i_q_mess']

fig, axs = plt.subplots(2, 1)
axs[1].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_q_mess'].tolist()[0]], 'r', label='$\mathrm{SEC}$')
axs[1].plot(t_test, [i * 160 * 1.41 for i in df_DDPG['i_q_mess'].tolist()[0]], '--r', label='$\mathrm{DDPG}_\mathrm{}$')
axs[1].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_q_ref'].tolist()[0]], ':', color='gray',
            label='$\mathrm{i}_\mathrm{q}^*$')
axs[1].grid()
# axs[1].legend()
axs[1].set_xlim(interval_list_x)
axs[1].set_ylim([-0.5 * 160 * 1.41, 0.55 * 160 * 1.41])
# axs[0].set_xlabel(r'$t\,/\,\mathrm{s}$')
axs[1].set_xlabel(r'$t\,/\,\mathrm{s}$')
axs[1].set_ylabel("$i_{\mathrm{q}}\,/\,{\mathrm{A}}$")
axs[1].tick_params(direction='in')

axs[0].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_d_mess'].tolist()[0]], 'b', label='$\mathrm{SEC}_\mathrm{}$')
axs[0].plot(t_test, [i * 160 * 1.41 for i in df_DDPG['i_d_mess'].tolist()[0]], '--b', label='$\mathrm{DDPG}_\mathrm{}$')
axs[0].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_d_ref'].tolist()[0]], ':', color='gray',
            label='$i_\mathrm{}^*$')
axs[0].grid()
axs[0].legend()
axs[0].set_xlim(interval_list_x)
axs[0].set_ylim([-0.78 * 160 * 1.41, 0.05 * 160 * 1.41])
axs[0].tick_params(axis='x', colors='w')
axs[0].set_ylabel("$i_{\mathrm{d}}\,/\,{\mathrm{A}}$")
axs[0].tick_params(direction='in')
fig.subplots_adjust(wspace=0, hspace=0.05)
plt.show()

if save_results:
    fig.savefig(f'{folder_name}/GEM_DDPG_I_noI_idq.pgf')
    fig.savefig(f'{folder_name}/GEM_DDPG_I_noI_idq.png')
    fig.savefig(f'{folder_name}/GEM_DDPG_I_noI_idq.pdf')
