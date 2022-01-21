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

"""
I_term = pd.read_pickle('GEM_I_term_4_1250_agents_data_with_rewards.pkl')
no_I_term = pd.read_pickle('GEM_no_I_term_4_1250_agents_data_with_rewards.pkl')

asd = 1

m = np.array(I_term['return_Mean'])
s = np.array(I_term['return_Std'])
agents = np.arange(0,1250)

# take the best 50 and the worst 50 and and 450 random

idxs = np.random.randint(low=50, high=1200, size=450)
m_sort = np.sort(m)
m550 = np.concatenate([m_sort[0:50],m_sort[1200:1250], np.take(m_sort, idxs)])
"""

# typo! das sind die mit 5 pastVals!
OMG_DDPG_Integrator_no_pastVals = [-0.0566483, -0.177257, -0.22384, -0.0566379, -0.0613575,
                                   -0.866927, -0.0591551, -0.0409672, -0.0410715, -0.0405743,
                                   -0.0481607, -1.00176, -0.0398449, -0.0584291, -0.0428567,
                                   -0.754902, -0.0499666, -0.346553, -0.0448563, -0.0424514,
                                   -0.19927, -0.0424081, -0.0613121, -0.0501086, -0.287048,
                                   -0.214733, -0.0421697, -0.0474572, -0.0464294, -0.0467267,
                                   -0.0483718, -0.0584424, -0.354886, -0.0451979, -0.04627,
                                   -0.047793, -0.0471481, -0.0846913, -0.0446951, -0.0500306,
                                   -0.043155, -0.0718899, -0.039992, -0.0453119, -0.0673279,
                                   -0.0408377, -0.047179, -0.0438636, -0.0430013, -0.0595805]

OMG_DDPG_Integrator_no_pastVals_500 = \
    pd.read_pickle('OMG_DDPG_Integrator_no_pastValsreturn_500_agents.pkl')['return'].tolist()
OMG_SEC_return = OMG_DDPG_Integrator_no_pastVals + OMG_DDPG_Integrator_no_pastVals_500

# OMG_DDPG_return_798 = pd.read_pickle('OMG_DDPG_Actorreturn_8XX_agents.pkl')['return'].tolist()
OMG_DDPG_return_798 = pd.read_pickle('OMG_DDPG_Actorreturn_8XX_agents.pkl')['return_Mean'].tolist()

idxs = np.random.randint(low=50, high=748, size=450)
m_sort = np.sort(OMG_DDPG_return_798)
OMG_DDPG_return = np.concatenate([m_sort[0:50], m_sort[747:798], np.take(m_sort, idxs)])

idx_DDPG_sort = np.argsort(OMG_DDPG_return_798)

# OMG_DDPG_return = OMG_DDPG_return_798

if save_results:
    matplotlib.rcParams.update(params)

fig, ax = plt.subplots()  # figsize =(6, 5))
# plt.boxplot((OMG_DDPG_Actor, OMG_DDPG_Integrator_no_pastVals_corr,
#             OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr, OMG_DDPG_Integrator_no_pastVals))
ax.boxplot((OMG_SEC_return, OMG_DDPG_return))
# ax.plot( 3, 0.0332, marker='o' )
plt.grid()
plt.ylim([-0.4, 0])
plt.xticks([1, 2], ['$\mathrm{SEC}$', '$\mathrm{DDPG}$'])
plt.ylabel('$\overline{\sum{r_k}}$')
plt.tick_params(direction='in')
plt.show()

# if save_results:
#    fig.savefig(f'{folder_name}/OMG_Errorbar_lim.pgf')
#    fig.savefig(f'{folder_name}/OMG_Errorbar_lim.png')
#    fig.savefig(f'{folder_name}/OMG_Errorbar_lim.pdf')


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

SEC_train_data = pd.read_pickle('OMG_DDPG_Integrator_no_pastVals_8XX_agents_train_data.pkl')
DDPG_train_data = pd.read_pickle('OMG_DDPG_Actor_8XX_agents_train_data.pkl')

# DDPG data to long -> sort by mean -> take best/worst 50 and 450 random

SEC_mean_learningCurve_550 = SEC_train_data.mean(axis=1)
SEC_std_learningCurve_550 = SEC_train_data.std(axis=1)

# sort df by idx (return of test case from above) - not needed, just for doublecheck
df3 = DDPG_train_data.iloc[:, idx_DDPG_sort]

# get the best/worst idx2 out ouf sort_idx and snip the df to 550 based on that idx2
idx2 = np.concatenate([idx_DDPG_sort[0:50], idx_DDPG_sort[idxs], idx_DDPG_sort[747:798]])
df550 = DDPG_train_data.iloc[:, idx2]

DDPG_mean_learningCurve_550 = df3.mean(axis=1)
DDPG_std_learningCurve_550 = df3.std(axis=1)

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
plt.xlim([0, 176])
plt.tick_params(direction='in')
plt.legend()
# plt.set_xlim([0, 10])
plt.ylabel('$\overline{{r}}$')
plt.xlabel(r'$\mathrm{Episode}$')
plt.show()

if save_results:
    matplotlib.rcParams.update(params)

    fig.savefig(f'{folder_name}/OMG_learning_curve.pgf')
    fig.savefig(f'{folder_name}/OMG_learning_curve.png')
    fig.savefig(f'{folder_name}/OMG_learning_curve.pdf')

plt.plot(SEC_mean_learningCurve_550, 'b', label='$\mathrm{SEC}$')
plt.plot(DDPG_mean_learningCurve_550, '-.b', label='$\mathrm{DDPG}$')
plt.fill_between(SEC_mean_learningCurve_550 - SEC_std_learningCurve_550,
                 SEC_mean_learningCurve_550 + SEC_std_learningCurve_550, facecolor='r')
plt.fill_between(DDPG_mean_learningCurve_550 - DDPG_std_learningCurve_550,
                 DDPG_mean_learningCurve_550 + DDPG_std_learningCurve_550, facecolor='r')
plt.grid()
plt.legend()
plt.xlim([0, 177])
# plt.set_xlim([0, 10])
plt.ylabel('$\overline{\sum{r}}$')
plt.xlabel(r'$\mathrm{Episode}$')
plt.show()
asd = 1

# not needed, but maybe interesting for futuer to reorder df:
# df2 = DDPG_train_data.iloc[:,idx_DDPG_sort]
# idx2 = np.concatenate([np.array([list(range(0,50))]).squeeze(), np.array([list(range(748,798))]).squeeze(), idxs])
