import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

save_results = True
folder_name = 'errorbar_plots/'

# Plot setting
params = {'backend': 'ps',
          'text.latex.preamble': [r'\usepackage{gensymb}'
                                  r'\usepackage{amsmath,amssymb,mathtools}'
                                  r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                  r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
          'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
          'axes.titlesize': 8,
          'font.size': 10,  # was 10
          'legend.fontsize': 10,  # was 10
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': [5.8, 3.8],  # [3.9, 3.1],
          'font.family': 'serif',
          'lines.linewidth': 1
          }

SEC = pd.read_pickle('GEM_I_term_4_1250_agents_data_with_rewards.pkl')
DDPG = pd.read_pickle('GEM_no_I_term_4_1250_agents_data_with_rewards.pkl')

m_sec = np.array(SEC['return_Mean'])
s_sec = np.array(SEC['return_Std'])

idx_SEC_sort = np.argsort(m_sec)

agents = np.arange(0, 1250)

# take the best 50 and the worst 50 and and 450 random

idxs = np.random.randint(low=50, high=1200, size=450)
m_sort = np.sort(m_sec)
m_sec_550 = np.concatenate([m_sort[0:50], m_sort[1200:1250], np.take(m_sort, idxs)])

m_ddgp = np.array(DDPG['return_Mean'])
s_ddgp = np.array(DDPG['return_Std'])

idx_DDPG_sort = np.argsort(m_ddgp)

# take the best 50 and the worst 50 and and 450 random
m_sort = np.sort(m_ddgp)
m_ddpg_550 = np.concatenate([m_sort[0:50], m_sort[1200:1250], np.take(m_sort, idxs)])

if save_results:
    matplotlib.rcParams.update(params)

fig = plt.figure()
plt.boxplot((m_sec_550, m_ddpg_550))
plt.grid()
plt.ylim([-1, 0])
plt.xticks([1, 2], ['$\mathrm{SEC}$', '$\mathrm{DDPG}$'])
plt.ylabel('$\overline{\sum{r_k}}$')
plt.tick_params(direction='in')
plt.show()

if save_results:
    fig.savefig(f'{folder_name}/GEM_Errorbar_lim.pgf')
    fig.savefig(f'{folder_name}/GEM_Errorbar_lim.png')
    fig.savefig(f'{folder_name}/GEM_Errorbar_lim.pdf')

##########################LearningCurve###############

params = {'backend': 'ps',
          'text.latex.preamble': [r'\usepackage{gensymb}'
                                  r'\usepackage{amsmath,amssymb,mathtools}'
                                  r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                  r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
          'axes.labelsize': 12,  # fontsize for x and y labels (was 10)
          'axes.titlesize': 12,
          'font.size': 12,  # was 10
          'legend.fontsize': 12,  # was 10
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'text.usetex': True,
          'figure.figsize': [5.5, 3.7],  # [3.9, 3.1],
          'font.family': 'serif',
          'lines.linewidth': 1
          }

matplotlib.rcParams.update(params)

SEC_train_data = pd.read_pickle('GEM_I_term_4_1250_agents_train_data.pkl')
DDPG_train_data = pd.read_pickle('GEM_no_I_term_4_1250_agents_train_data.pkl')

# sort df based on test return top down
# sort df by idx (return of test case from above) - not needed, just for doublecheck
df_sort_sec = SEC_train_data.iloc[:, idx_SEC_sort]
df_sort_ddpg = SEC_train_data.iloc[:, idx_SEC_sort]

# get the best/worst idx2 out ouf sort_idx and snip the df to 550 based on that idx2
idx2_ddpg = np.concatenate([idx_DDPG_sort[0:50], idx_DDPG_sort[idxs], idx_DDPG_sort[747:798]])
ddpg550 = DDPG_train_data.iloc[:, idx2_ddpg]

idx2_sec = np.concatenate([idx_SEC_sort[0:50], idx_SEC_sort[idxs], idx_SEC_sort[747:798]])
sec550 = SEC_train_data.iloc[:, idx2_sec]

DDPG_mean_learningCurve_550 = ddpg550.mean(axis=1)
DDPG_std_learningCurve_550 = ddpg550.std(axis=1)

SEC_mean_learningCurve_550 = sec550.mean(axis=1)
SEC_std_learningCurve_550 = sec550.std(axis=1)

low = (SEC_mean_learningCurve_550 - SEC_std_learningCurve_550).to_numpy()
up = (SEC_mean_learningCurve_550 + SEC_std_learningCurve_550).to_numpy()
SEC = SEC_mean_learningCurve_550.to_numpy()
DDPG = DDPG_mean_learningCurve_550.to_numpy()
episode = np.array([list(range(0, 177))]).squeeze()

fig, ax = plt.subplots()
plt.fill_between(episode, up, low, facecolor='b', alpha=0.25)
plt.fill_between(episode, (DDPG_mean_learningCurve_550 + DDPG_std_learningCurve_550).to_numpy(),
                 (DDPG_mean_learningCurve_550 - DDPG_std_learningCurve_550).to_numpy(), facecolor='r', alpha=0.25)
plt.plot(episode, SEC, 'b', label='$\mathrm{SEC}$', linewidth=2)
plt.plot(episode, low, '--b', linewidth=0.5)
plt.plot(episode, up, '--b', linewidth=0.5)
plt.plot(episode, DDPG, 'r', label='$\mathrm{DDPG}$', linewidth=2)
plt.plot(episode, (DDPG_mean_learningCurve_550 + DDPG_std_learningCurve_550).to_numpy(), '--r', linewidth=0.5)
plt.plot(episode, (DDPG_mean_learningCurve_550 - DDPG_std_learningCurve_550).to_numpy(), '--r', linewidth=0.5)
plt.grid()
plt.tick_params(direction='in')
plt.legend()
plt.xlim([0, 176])
# plt.set_xlim([0, 10])
plt.ylabel('$\overline{{r}}$')
plt.xlabel(r'$\mathrm{Episode}$')
plt.show()

if save_results:
    matplotlib.rcParams.update(params)

    fig.savefig(f'{folder_name}/GEM_learning_curve.pgf')
    fig.savefig(f'{folder_name}/GEM_learning_curve.png')
    fig.savefig(f'{folder_name}/GEM_learning_curve.pdf')
