#####################################
# Experiment : Single voltage forming inverter supplying an RL-load via an LC-filter
# Controller: Cascaded PI-PI voltage and current controller gain parameters are optimized by SafeOpt
# a) FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# b) connecting via ssh to a testbench to perform real-world measurement
import logging
import os
import time
from functools import partial

import GPy
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from stochastic.processes import VasicekProcess

from experiments.hp_tune.env.random_load import RandomLoad
from experiments.hp_tune.env.rewards import Reward
# from experiments.model_validation.execution.monte_carlo_runner import MonteCarloRunner
from experiments.hp_tune.execution.reset_runner import MonteCarloRunner
from experiments.hp_tune.execution.runner import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, \
    MultiPhaseDQ0PIPIController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.plotmanager import PlotManager
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import FullHistory, RandProcess


class CallbackList(list):
    def fire(self, *args, **kwargs):
        for listener in self:
            listener(*args, **kwargs)


show_plots = False
balanced_load = False
save_results = True
PIPI = True

num_average = 25
max_episode_steps_list = [1000, 5000, 10000, 20000, 50000, 100000]

result_list = []
ret_list = []
mean_list = []
std_list = []
ret_array = np.zeros(num_average)

df = pd.DataFrame()
ret_dict = dict()

# Files saves results and  resulting plots to the folder saves_VI_control_safeopt in the current directory
current_directory = os.getcwd()
folder_name = 'Pipi_new_testcase_opt_4D_reset2'
save_folder = os.path.join(current_directory, folder_name)
os.makedirs(save_folder, exist_ok=True)

np.random.seed(1)

# Simulation definitions
# net = Network.load('../../net/net_single-inv-Paper_Loadstep.yaml')
net = Network.load('net/net_vctrl_single_inv.yaml')
delta_t = 1e-4  # simulation time step size / s
undersample = 1
max_episode_steps = 99999#10000  # number of simulation steps per episode
num_episodes = 100  # number of simulation episodes (i.e. SafeOpt iterations)
n_MC = 1  # number of Monte-Carlo samples for simulation - samples device parameters (e.g. L,R, noise) from
v_DC = 600  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 60  # nominal grid frequency / Hz
nomVoltPeak = 169.7  # 230 * 1.414  # nominal grid voltage / V
iLimit = 16  # inverter current limit / A
iNominal = 12  # nominal inverter current / A
vNominal = 190  # nominal inverter current / A
vLimit = vNominal * 1.5  # inverter current limit / A
funnelFactor = 0.02
vFunnel = np.array([vNominal * funnelFactor, vNominal * funnelFactor, vNominal * funnelFactor])
mu = 400  # factor for barrier function (see below)
DroopGain = 0.0  # virtual droop gain for active power / W/Hz
QDroopGain = 0.0  # virtual droop gain for reactive power / VAR/V


class Recorder:

    def __init__(self, URI: str = 'mongodb://localhost:27017/', database_name: str = 'OMG', ):
        self.client = MongoClient(URI)
        self.db = self.client[database_name]

    def save_to_mongodb(self, col: str = ' trails', data=None):
        trial_coll = self.db[col]  # get collection named col
        if data is None:
            raise ValueError('No data given to store in database!')
        trial_coll.insert_one(data)


rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma=0,
             use_gamma_normalization=1, error_exponent=0.5, i_lim=net['inverter1'].i_lim,
             i_nom=net['inverter1'].i_nom)

#####################################
# Definitions for the GP
prior_mean = 0  # 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
noise_var = 0.001  # ** 2  # measurement noise sigma_omega
prior_var = 2  # prior variance of the GP

# Choose Kp and Ki (current and voltage controller) as mutable parameters (below) and define bounds and lengthscale
# for both of them
if PIPI:
    # bounds = [(0.001, 0.07), (2, 150), (0.000, 0.045), (4, 450)]
    # lengthscale = [0.005, 25., 0.008, 150] # .003, 50.]
    bounds = [(0.001, 0.07), (2, 150), (0.000, 0.05), (4, 600)]
    lengthscale = [0.01, 35., 0.01, 175]  # .003, 50.]
    mutable_params = dict(currentP=MutableFloat(0.04), currentI=MutableFloat(11.8),
                          voltageP=MutableFloat(0.0175), voltageI=MutableFloat(12))
    current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'],
                                    limits=(-1, 1))  # Best set from paper III-D

else:
    bounds = [(0.000, 0.045), (4, 450)]
    lengthscale = [0.01, 150]  # [0.003, 50]
    mutable_params = dict(voltageP=MutableFloat(0.0175), voltageI=MutableFloat(12))  # 300Hz
    kp_c = 0.04
    ki_c = 11.8  # 11.8
    current_dqp_iparams = PI_params(kP=kp_c, kI=ki_c, limits=(-1, 1))

# The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
# the initial performance: safe_threshold = 1.2 means: performance measurement for optimization are seen as
# unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
# parameter set
safe_threshold = 0
j_min = -50000  # -5  # 15000? # cal min allowed performance

# The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
# expanding points eventually.
# The following variable is multiplied with the first performance of the initial set by the factor below:
explore_threshold = -200000

# Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
# limit exceeded
abort_reward = 100 * j_min

# Definition of the kernel
kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

#####################################
# Definition of the controllers
# Choose Kp and Ki for the current and voltage controller as mutable parameters

# mutable_params = dict(voltageP=MutableFloat(0.016), voltageI=MutableFloat(105))  # 300Hz
voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'],
                                limits=(-iLimit, iLimit))

# Current controller values

# Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
# filter and the nominal frequency
# Droop controller used to calculate the virtual frequency drop due to load changes
droop_param = DroopParams(DroopGain, 0.005, net.freq_nom)

# Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
# filter and the nominal voltage
qdroop_param = DroopParams(QDroopGain, 0.002, net.v_nom)

# Define a voltage forming inverter using the PIPI and droop parameters from above

# Controller with observer
# ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param, qdroop_param,
#                                   observer=[Lueneberger(*params) for params in
#                                             repeat((A, B, C, L, delta_t * undersample, v_DC / 2), 3)], undersampling=undersample,
#                                   name='master')

# Controller without observer
ctrl = MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param, qdroop_param,
                                   ts_sim=delta_t,
                                   ts_ctrl=undersample * delta_t,
                                   name='master')

#####################################
# Definition of the optimization agent
# The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
# Arguments described above
# History is used to store results
agent = SafeOptAgent(mutable_params,
                     abort_reward,
                     j_min,
                     kernel,
                     dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                          safe_threshold=safe_threshold, explore_threshold=explore_threshold),
                     [ctrl],
                     dict(master=[[f'lc.inductor{k}.i' for k in '123'],
                                  [f'lc.capacitor{k}.v' for k in '123']
                                  ]),
                     history=FullHistory(),
                     )

i_lim = net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = net['inverter1'].v_lim
v_DC = net['inverter1'].v_DC
# plant

# toDo: shift this to net?!
L_filter = 2.3e-3  # / H
R_filter = 400e-3  # / Ohm
C_filter = 10e-6  # / F

lower_bound_load = -10  # to allow maximal load that draws i_limit
upper_bound_load = 200  # to apply symmetrical load bounds
lower_bound_load_clip = 14  # to allow maximal load that draws i_limit (let exceed?)
upper_bound_load_clip = 200  # to apply symmetrical load bounds
lower_bound_load_clip_std = 2
upper_bound_load_clip_std = 0

R = np.random.uniform(low=lower_bound_load, high=upper_bound_load)

gen = RandProcess(VasicekProcess, proc_kwargs=dict(speed=800, vol=40, mean=R), initial=R,
                  bounds=(lower_bound_load, upper_bound_load))

rand_load_train = RandomLoad(max_episode_steps, net.ts, gen,
                             bounds=(lower_bound_load_clip, upper_bound_load_clip),
                             bounds_std=(lower_bound_load_clip_std, upper_bound_load_clip_std))

cb = CallbackList()
# set initial = None to reset load random in range of bounds
cb.append(partial(gen.reset))  # , initial=np.random.uniform(low=lower_bound_load, high=upper_bound_load)))
cb.append(rand_load_train.reset)

plotter = PlotManager(agent, save_results=save_results, save_folder=save_folder,
                      show_plots=show_plots)

rand_load_test = RandomLoad(max_episode_steps, net.ts, gen,
                            load_curve=pd.read_pickle(
                                'experiments/hp_tune/data/R_load_test_case_2_seconds.pkl'))


def xylables_R(fig):
    ax = fig.gca()
    ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
    ax.set_ylabel('$R_{\mathrm{abc}}\,/\,\mathrm{\Omega}$')
    ax.grid(which='both')
    # ax.set_ylim([lower_bound_load - 2, upper_bound_load + 2])
    ts = time.gmtime()
    fig.savefig(f'{save_folder}/Load{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
    if show_plots:
        plt.show()
    else:
        plt.close()


env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
               reward_fun=rew.rew_fun_PIPI_MRE,
               viz_cols=[
                   PlotTmpl([[f'lc.capacitor{i}.v' for i in '123'], [f'master.SPV{i}' for i in 'abc']],
                            callback=plotter.xylables_v_abc,
                            color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                            style=[[None], ['--']]
                            ),
                   PlotTmpl([[f'master.CVV{i}' for i in 'dq0'], [f'master.SPV{i}' for i in 'dq0']],
                            callback=plotter.xylables_v_dq0,
                            color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                            style=[[None], ['--']]
                            ),
                   PlotTmpl([[f'lc.inductor{i}.i' for i in '123'], [f'master.SPI{i}' for i in 'abc']],
                            callback=plotter.xylables_i_abc,
                            color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                            style=[[None], ['--']]
                            ),
                   PlotTmpl([[f'r_load.resistor{i}.R' for i in '123']],
                            callback=xylables_R,
                            color=[['b', 'r', 'g']],
                            style=[[None]]
                            ),
                   # PlotTmpl([[f'master.I_hat{i}' for i in 'abc'], [f'r_load.resistor{i}.i' for i in '123'], ],
                   #         callback=lambda fig: plotter.update_axes(fig, title='Simulation',
                   #                                                  ylabel='$i_{\mathrm{o estimate,abc}}\,/\,\mathrm{A}$'),
                   #         color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                   #         style=[['-*'], ['--*']]
                   #         ),
                   # PlotTmpl([[f'master.m{i}' for i in 'dq0']],
                   #         callback=lambda fig: plotter.update_axes(fig, title='Simulation',
                   #                                                  ylabel='$m_{\mathrm{dq0}}\,/\,\mathrm{}$',
                   #                                                  filename='Sim_m_dq0')
                   #         ),
                   PlotTmpl([[f'master.CVi{i}' for i in 'dq0'], [f'master.SPI{i}' for i in 'dq0']],
                            callback=plotter.xylables_i_dq0,
                            color=[['b', 'r', 'g'], ['b', 'r', 'g']],
                            style=[[None], ['--']]
                            )
               ],
               log_level=logging.INFO,
               viz_mode='episode',
               max_episode_steps=max_episode_steps,
               model_params={'lc.resistor1.R': R_filter,
                             'lc.resistor2.R': R_filter,
                             'lc.resistor3.R': R_filter,
                             'lc.resistor4.R': 0.0000001,
                             'lc.resistor5.R': 0.0000001,
                             'lc.resistor6.R': 0.0000001,
                             'lc.inductor1.L': L_filter,
                             'lc.inductor2.L': L_filter,
                             'lc.inductor3.L': L_filter,
                             'lc.capacitor1.C': C_filter,
                             'lc.capacitor2.C': C_filter,
                             'lc.capacitor3.C': C_filter,
                             # 'r_load.resistor1.R': partial(rand_load_test.give_dataframe_value, col='r_load.resistor1.R'),
                             # 'r_load.resistor2.R': partial(rand_load_test.give_dataframe_value, col='r_load.resistor2.R'),
                             # 'r_load.resistor3.R': partial(rand_load_test.give_dataframe_value, col='r_load.resistor3.R'),
                             'r_load.resistor1.R': rand_load_train.random_load_step,
                             'r_load.resistor2.R': rand_load_train.random_load_step,
                             'r_load.resistor3.R': rand_load_train.random_load_step,
                             'lc.capacitor1.v': lambda t: np.random.uniform(low=-v_nom,
                                                                            high=v_nom) if t == -1 else None,
                             'lc.capacitor2.v': lambda t: np.random.uniform(low=-v_nom,
                                                                            high=v_nom) if t == -1 else None,
                             'lc.capacitor3.v': lambda t: np.random.uniform(low=-v_nom,
                                                                            high=v_nom) if t == -1 else None,
                             'lc.inductor1.i': lambda t: np.random.uniform(low=-i_nom,
                                                                           high=i_nom) if t == -1 else None,
                             'lc.inductor2.i': lambda t: np.random.uniform(low=-i_nom,
                                                                           high=i_nom) if t == -1 else None,
                             'lc.inductor3.i': lambda t: np.random.uniform(low=-i_nom,
                                                                           high=i_nom) if t == -1 else None,
                             },
               net=net,
               model_path='omg_grid/grid.paper_loadstepWIN.fmu',
               history=FullHistory(),
               on_episode_reset_callback=cb.fire,
               action_time_delay=1 * undersample
               )

runner = MonteCarloRunner(agent, env)
runner.run(num_episodes, n_mc=n_MC, visualise=True,  # prepare_mc_experiment=reset_loads,
           return_gradient_extend=False)

df_len = pd.DataFrame({'lengthscale': lengthscale,
                       'bounds': bounds,
                       'balanced_load': balanced_load,
                       'barrier_param_mu': mu,
                       'J_min': j_min})

if save_results:
    agent.history.df.to_csv(save_folder + '/_result.csv')
    df_len.to_csv(save_folder + '/_params.csv')
if not PIPI:
    best_agent_plt = runner.run_data['last_agent_plt']
    ax = best_agent_plt.axes[0]
    ax.grid(which='both')
    ax.set_axisbelow(True)

    agent.params.reset()
    ax.set_ylabel(r'$K_\mathrm{i}\,/\,\mathrm{(AV^{-1}s^{-1})}$')
    ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(AV^{-1})}$')
    ax.get_figure().axes[1].set_ylabel(r'$J$')
    plt.title('Lengthscale = {}; balanced = '.format(lengthscale, balanced_load))
    # ax.plot([0.01, 0.01], [0, 250], 'k')
    # ax.plot([mutable_params['currentP'].val, mutable_params['currentP'].val], bounds[1], 'k-', zorder=1,
    #         lw=4,
    #         alpha=.5)
    best_agent_plt.show()
    if save_results:
        best_agent_plt.savefig(save_folder + '/_agent_plt.pdf')
        #best_agent_plt.savefig(save_folder + '/_agent_plt.pgf')
        agent.history.df.to_csv(save_folder + '/_result.csv')

print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df.round({'J': 4, 'Params': 4})))
print('\n Experiment finished with best set: \n')
print('\n  Current-Ki&Kp and voltage-Ki&Kp = {}'.format(
    agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
print('\n\nBest experiment results are plotted in the following:')
