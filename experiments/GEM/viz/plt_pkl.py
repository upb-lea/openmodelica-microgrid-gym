import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

save_results = False
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

I_term = pd.read_pickle('GEM_I_term_4mean_over_1250_agents.pkl')
no_I_term = pd.read_pickle('GEM_no_I_term_4mean_over_1250_agents.pkl')

asd = 1

m = np.array(I_term['return_Mean'])
s = np.array(I_term['return_Std'])
agents = np.arange(0, 1250)

plt.plot(agents, m)
plt.fill_between(agents, m - s, m + s, facecolor='r')
plt.ylabel('Average return ')
plt.xlabel('Agents')
plt.ylim([-0.6, 0.2])
plt.grid()
plt.title('I_term')
plt.show()

plt.plot(s)
# plt.fill_between( m - s, m + s, facecolor='r')
plt.ylabel('Average return  sdt')
plt.xlabel('agents')
# plt.ylim([-0.4, 0])
plt.grid()
plt.title('I_term')
plt.show()

m_no_I = np.array(no_I_term['return_Mean'])
s_no_I = np.array(no_I_term['return_Std'])
agents = np.arange(0, 1250)

plt.plot(agents, m_no_I)
plt.fill_between(agents, m_no_I - s_no_I, m_no_I + s_no_I, facecolor='r')
plt.ylabel('Average return +- sdt')
plt.xlabel('Agents')
plt.ylim([-0.6, 0.2])
plt.grid()
plt.title('no_I_term')
plt.show()

plt.plot(s_no_I)
# plt.fill_between( m - s, m + s, facecolor='r')
plt.ylabel('Average return  sdt')
plt.xlabel('Max_episode steps')
# plt.ylim([-0.4, 0])
plt.grid()
plt.title('no_I_term')
plt.show()

if save_results:
    matplotlib.rcParams.update(params)

fig = plt.figure()
plt.boxplot((m_no_I, m))
plt.grid()
plt.ylim([-1, 0])
plt.xticks([1, 2], ['$\mathrm{DDPG}$', '$\mathrm{SEC}$'])
plt.ylabel('$\overline{\sum{r_k}}$')
plt.tick_params(direction='in')
plt.show()

if save_results:
    fig.savefig(f'{folder_name}/GEM_Errorbar_lim.pgf')
    fig.savefig(f'{folder_name}/GEM_Errorbar_lim.png')
    fig.savefig(f'{folder_name}/GEM_Errorbar_lim.pdf')

plt.boxplot((m, m_no_I))
plt.grid()
plt.xticks([1, 2], ['I', 'no-I'])
plt.show()
