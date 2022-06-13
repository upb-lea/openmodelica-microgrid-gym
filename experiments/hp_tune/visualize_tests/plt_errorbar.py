import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# I_term = pd.read_pickle('GEM_I_term_3mean_over_50_agents.pkl')
# no_I_term = pd.read_pickle('GEM_no_I_term_3mean_over_50_agents.pkl')

asd = 1

save_results = False
folder_name = 'errorbar_plots/'

# da json zu groß, kopie aus dashboard....

OMG_DDPG_Actor = [-0.226037, -0.128363, -0.139432, -0.121386, -0.137367,
                  -0.216827, -0.116579, -0.0831927, -0.112777, -0.127669,
                  -0.185162, -0.128747, -0.113952, -0.122981, -0.114832,
                  -0.120671, -0.226531, -0.118882, -0.134699, -0.118027,
                  -0.149192, -0.121207, -0.253065, -0.219944, -0.1244,
                  -0.0993589, -0.12237, -0.143523, -0.244333, -0.124357,
                  -0.152193, -0.118973, -0.0955573, -0.114242, -0.111534,
                  -0.127907, -0.102504, -0.225466, -0.219972, -0.120333,
                  -0.134156, -0.116749, -0.122513, -0.167896, -0.062778,
                  -0.239305, -0.110423, -0.103946, -0.160686, -0.127362]

OMG_DDPG_Actor_500 = pd.read_pickle('OMG_DDPG_Actorreturn_500_agents.pkl')['return'].tolist()[
                     :-1]  # einen zu viel geladen!

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

OMG_DDPG_Integrator_no_pastVals_corr = [-0.048334, -0.251245, -0.0688722, -0.0565136, -0.202199,
                                        -0.042535, -0.0408258, -0.0480982, -0.0423354, -0.0461098,
                                        -0.543109, -0.0444726, -0.134507, -0.101061, -0.0410615,
                                        -0.0423758, -0.0732737, -0.0531188, -0.0451057, -0.0557529,
                                        -0.0516102, -0.272256, -0.0494411, -0.0453498, -0.049296,
                                        -0.0524428, -0.0417263, -0.0453462, -0.0466777, -0.0772813,
                                        -0.217484, -0.0407658, -0.0403833, -0.0795559, -0.0393357,
                                        -0.0526313, -0.0443727, -0.0455981, -0.049839, -0.046536,
                                        -0.0453199, -0.0421393, -0.0469275, -0.0441136, -0.0426031,
                                        -0.162181, -0.0523912, -0.0403753, -0.0412137, -0.770299]

OMG_DDPG_Integrator_no_pastVals_corr_500 = \
    pd.read_pickle('OMG_DDPG_Integrator_no_pastVals_corrreturn_500_agents.pkl')['return'].tolist()[
    :-1]  # einen zu viel geladen!

OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr = [-0.0387997, -0.0409335, -0.0685522, -0.164238, -0.0409236,
                                                       -0.0410673, -0.039469, -0.0399732, -0.13207, -0.0415697,
                                                       -0.122869, -0.0611268, -0.306491, -0.0992046, -0.044661,
                                                       -0.0458972, -0.043849, -0.0500543, -0.0531591, -0.0679286,
                                                       -0.20993, -0.0497402, -0.0405819, -0.0746702, -0.203728,
                                                       -0.0408563, -0.0708935, -0.0409779, -0.0438561, -0.0432274,
                                                       -0.0395637, -0.0404426, -0.0377221, -0.0404959, -0.0465647,
                                                       -0.0612425, -0.0409127, -0.0416884, -0.198034, -0.0523231,
                                                       -0.2017, -0.0414555, -0.0422072, -0.0398287, -0.0400683,
                                                       -0.0461625, -0.264055, -0.0453719, -0.0396692, -0.0411879]

OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr_500 = \
    pd.read_pickle('OMG_DDPG_Integrator_no_pastVals_i_load_feature_corrreturn_500_agents.pkl')['return'].tolist()
# m = np.array(I_term['return_Mean'])
# s = np.array(I_term['return_Std'])
agents = np.arange(0, 550)
agents = np.arange(0, 500)

OMG_DDPG_Actor = OMG_DDPG_Actor + OMG_DDPG_Actor_500
OMG_DDPG_Integrator_no_pastVals = OMG_DDPG_Integrator_no_pastVals + OMG_DDPG_Integrator_no_pastVals_500
OMG_DDPG_Integrator_no_pastVals_corr = OMG_DDPG_Integrator_no_pastVals_corr + OMG_DDPG_Integrator_no_pastVals_corr_500
OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr = OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr + \
                                                      OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr_500

OMG_DDPG_Actor = OMG_DDPG_Actor_500
OMG_DDPG_Integrator_no_pastVals = OMG_DDPG_Integrator_no_pastVals_500
OMG_DDPG_Integrator_no_pastVals_corr = OMG_DDPG_Integrator_no_pastVals_corr_500
OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr = OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr_500

plt.plot(agents, OMG_DDPG_Actor)
plt.plot(agents, OMG_DDPG_Integrator_no_pastVals, 'r')
plt.plot(agents, OMG_DDPG_Integrator_no_pastVals_corr, 'g')
plt.plot(agents, OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr, 'm')
# plt.fill_between(agents, m - s, m + s, facecolor='r')
plt.ylabel('Average return ')
plt.xlabel('Agents')
plt.ylim([-0.6, 0.2])
plt.grid()
plt.title('I_term')
plt.show()

if save_results:
    matplotlib.rcParams.update(params)

fig, ax = plt.subplots()  # figsize =(6, 5))
# plt.boxplot((OMG_DDPG_Actor, OMG_DDPG_Integrator_no_pastVals_corr,
#             OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr, OMG_DDPG_Integrator_no_pastVals))
ax.boxplot((OMG_DDPG_Integrator_no_pastVals, OMG_DDPG_Actor))
# ax.plot( 3, 0.0332, marker='o' )
plt.grid()
plt.ylim([-0.4, 0])
plt.xticks([1, 2], ['$\mathrm{SEC}$', '$\mathrm{DDPG}$'])
plt.ylabel('$\overline{\sum{r_k}}$')
plt.tick_params(direction='in')

if save_results:
    fig.savefig(f'{folder_name}/OMG_Errorbar_lim.pgf')
    fig.savefig(f'{folder_name}/OMG_Errorbar_lim.png')
    fig.savefig(f'{folder_name}/OMG_Errorbar_lim.pdf')

fig = plt.figure()  # figsize =(6, 5))
plt.boxplot((OMG_DDPG_Actor, OMG_DDPG_Integrator_no_pastVals_corr,
             OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr, OMG_DDPG_Integrator_no_pastVals))
plt.grid()
# plt.ylim([-0.75, 0])
plt.xticks([1, 2, 3, 4], ['$\mathrm{DDPG}$', '$\mathrm{DDPG}_\mathrm{I}$',
                          '$\mathrm{DDPG}_\mathrm{I,i_{load}}$', '$\mathrm{DDPG}_\mathrm{I,pv}$'])
plt.ylabel('$\overline{\sum{r_k}}$')
plt.show()
if save_results:
    fig.savefig(f'{folder_name}/OMG_Errorbar.png')
    fig.savefig(f'{folder_name}/OMG_Errorbar.pdf')
    fig.savefig(f'{folder_name}/OMG_Errorbar.pgf')

plt.boxplot((OMG_DDPG_Actor, OMG_DDPG_Integrator_no_pastVals_corr,
             OMG_DDPG_Integrator_no_pastVals_i_load_feature_corr, OMG_DDPG_Integrator_no_pastVals))
plt.grid()
plt.ylim([-0.06, 0])
plt.xticks([1, 2, 3, 4], ['$\mathrm{DDPG}$', '$\mathrm{DDPG}_\mathrm{I}$',
                          '$\mathrm{DDPG}_\mathrm{I,i_{load}}$', '$\mathrm{DDPG}_\mathrm{I,pv}$'])
plt.show()
