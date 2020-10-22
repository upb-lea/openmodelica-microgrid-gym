#####################################
# Example using a FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller gain parameters are optimized by SafeOpt


import logging
from typing import List

import GPy
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

from random import seed
from random import random

# Choose which controller parameters should be adjusted by SafeOpt.
# - Kp: 1D example: Only the proportional gain Kp of the PI controller is adjusted
# - Ki: 1D example: Only the integral gain Ki of the PI controller is adjusted
# - Kpi: 2D example: Kp and Ki are adjusted simultaneously

adjust = 'Kpi'

# Check if really only one simulation scenario was selected
if adjust not in {'Kp', 'Ki', 'Kpi'}:
    raise ValueError("Please set 'adjust' to one of the following values: 'Kp', 'Ki', 'Kpi'")

# Simulation definitions
net = Network.load('../net/net_single-inv-curr.yaml')
max_episode_steps = 600  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
i_ref = np.array([10, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A
R=20  #sets the resistance value of RL
load_jump=5 #defines the load jump that is implemented at every quarter of the simulation time
ts= 1e-4


#The starting value of the the resistance is 20 Ohm.
#Every step and therefore the whole simulationt time, it randomly changes +/- 0.01 Ohm to generate noise.
#Every quarter of the simulation time, a bigger load jump of 5  Ohms is implemented.
#After 3/4 of the simulation time, the controller is given time to control the current to its setpoint.

def load_step_random_walk():
    random_walk = []
    random_walk.append(R)
    load_step_t1 = max_episode_steps * 0.25
    load_step_t2 = max_episode_steps * 0.5
    load_step_t3 = max_episode_steps * 0.75
    for i in range(1, max_episode_steps):
        movement = -0.01 if random() < 0.5 else 0.01
        value=random_walk[i-1] + movement
        if value < 0:
            value = 1
        if i == load_step_t1 or i == load_step_t2 or i == load_step_t3:
            movement = load_jump if random() < 0.5 else -1*load_jump
            value = random_walk[i - 1] + movement
        random_walk.append(value)
    return random_walk

#print(load_step_random_walk())
#print(len(load_step_random_walk()))
#list_resistor=load_step_random_walk()

def load_step(t):
    for i in range(1,len(list_resistor)):
        if t <= ts * i + ts:
            return list_resistor[i]






class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0']])

    def rew_fun(self, cols: List[str], data: np.ndarray) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
        used parameters.
        Takes current measurement and setpoints so calculate the mean-root-error control error and uses a logarithmic
        barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        Iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        phase = data[idx[1]]  # phase from the master controller needed for transformation

        # setpoints
        ISPdq0_master = data[idx[2]]  # setting dq reference
        ISPabc_master = dq0_to_abc(ISPdq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = np.sum((np.abs((ISPabc_master - Iabc_master)) / iLimit) ** 0.5, axis=0) \
                + -np.sum(mu * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
                * max_episode_steps

        return -error.squeeze()


if __name__ == '__main__':
    #####################################
    # Definitions for the GP
    prior_mean = 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001 ** 2  # measurement noise sigma_omega
    prior_var = 0.1  # prior variance of the GP

    bounds = None
    lengthscale = None
    if adjust == 'Kp':
        bounds = [(0.00, 0.03)]  # bounds on the input variable Kp
        lengthscale = [.01]  # length scale for the parameter variation [Kp] for the GP

    # For 1D example, if Ki should be adjusted
    if adjust == 'Ki':
        bounds = [(0, 300)]  # bounds on the input variable Ki
        lengthscale = [50.]  # length scale for the parameter variation [Ki] for the GP

    # For 2D example, choose Kp and Ki as mutable parameters (below) and define bounds and lengthscale for both of them
    if adjust == 'Kpi':
        bounds = [(0.0, 0.03), (0, 300)]
        lengthscale = [.01, 50.]

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means. Performance measurement for optimization are seen as
    # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 2

    # The algorithm will not try to expand any points that are below this threshold. This makes the algorithm stop
    # expanding points eventually.
    # The following variable is multiplied with the first performance of the initial set by the factor below:
    explore_threshold = 2

    # Factor to multiply with the initial reward to give back an abort_reward-times higher negative reward in case of
    # limit exceeded
    abort_reward = 10

    # Definition of the kernel
    kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=prior_var, lengthscale=lengthscale, ARD=True)

    #####################################
    # Definition of the controllers
    mutable_params = None
    current_dqp_iparams = None
    if adjust == 'Kp':
        # mutable_params = parameter (Kp gain of the current controller of the inverter) to be optimized using
        # the SafeOpt algorithm
        mutable_params = dict(currentP=MutableFloat(5e-3))

        # Define the PI parameters for the current controller of the inverter
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=115, limits=(-1, 1))

    # For 1D example, if Ki should be adjusted
    elif adjust == 'Ki':
        mutable_params = dict(currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=10e-3, kI=mutable_params['currentI'], limits=(-1, 1))

    # For 2D example, choose Kp and Ki as mutable parameters
    elif adjust == 'Kpi':
        mutable_params = dict(currentP=MutableFloat(10e-3), currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, net.ts, f_nom=net.freq_nom,
                                                 undersampling=2, name='master')

    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    # Arguments described above
    # History is used to store results
    agent = SafeOptAgent(mutable_params,
                         abort_reward,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold),
                         [ctrl],
                         dict(master=[[f'lc1.inductor{k}.i' for k in '123'],
                                      i_ref]),
                         history=FullHistory()
                         )


    #####################################
    # Definition of the environment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed (here, only the 3 currents across the
    #   inductors of the inverters are plotted. Labels and grid is adjusted using the PlotTmpl (For more information,
    #   see UserGuide)
    # - inputs to the models are the connection points to the inverters (see user guide for more details)
    # - model outputs are the the 3 currents through the inductors and the 3 voltages across the capacitors

    def xylables(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        # fig.savefig('Inductor_currents.pdf')

    def xylables_R(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{s}$') #zeit
        ax.set_ylabel('$R_{\mathrm{123}}\,/\,\mathrm{Ohm}$') #widerstände definieren , ohm angucken backslash omega
        ax.grid(which='both')
        # fig.savefig('Inductor_currents.pdf')


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'], #Plot Strom durch Filterinduktivität
                                callback=xylables
                                ),
                       PlotTmpl([f'rl1.resistor{i}.R' for i in '123'],  #Plot Widerstand RL
                                callback=xylables_R
                                 )
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   model_params={'rl1.resistor1.R': load_step,
                                 'rl1.resistor2.R': load_step,
                                 'rl1.resistor3.R': load_step,
                                 'rl1.inductor1.L': 0.001,
                                 'rl1.inductor2.L': 0.001,
                                 'rl1.inductor3.L': 0.001
                                 },
                   max_episode_steps=max_episode_steps,
                   net=net,
                   model_path='../omg_grid/grid.network_singleInverter.fmu',

                   # model_input=['i1p1', 'i1p2', 'i1p3'],
                   # model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                   #                        ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                   #                   ),
                   history=FullHistory()
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)
    runner = Runner(agent, env)

    runner.run(num_episodes, visualise=True)

    print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df[:]))

    print('\n Experiment finished with best set: \n')
    print('\n  {} = {}'.format(adjust, agent.history.df.at[np.argmax(agent.history.df['J']), 'Params']))
    print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
    print('\n\nBest experiment results are plotted in the following:')

    # Show best episode measurment (current) plot
    best_env_plt = runner.run_data['best_env_plt']
    if best_env_plt:
        ax = best_env_plt[0].axes[0]
        ax.set_title('Best Episode')
        best_env_plt[0].show()
        best_env_plt[0].savefig('best_env_plt.png')

    # Show last performance plot
    best_agent_plt = runner.run_data['last_agent_plt']
    if best_agent_plt:
        ax = best_agent_plt.axes[0]
        ax.grid(which='both')
        ax.set_axisbelow(True)

        if adjust == 'Ki':
            ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
            ax.set_ylabel(r'$J$')
        elif adjust == 'Kp':
            ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
            ax.set_ylabel(r'$J$')
        elif adjust == 'Kpi':
            agent.params.reset()
            ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
            ax.set_ylabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
            ax.get_figure().axes[1].set_ylabel(r'$J$')
            plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-', zorder=1, lw=4,
                     alpha=.5)
        best_agent_plt.show()
        best_agent_plt.savefig('agent_plt.png')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
#df = env.history.df[['lc1.inductor1.i', 'lc1.inductor2.i','rl1.resistor1.R']]
#print(df)
#mean_lc1 = df["lc1.inductor1.i"].mean()
#mean_lc2 = df["lc1.inductor2.i"].mean()