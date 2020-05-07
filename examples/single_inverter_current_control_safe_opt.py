#####################################
# Example using a FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller gain parameters are optimized by SafeOpt


import logging
from datetime import datetime
from operator import itemgetter

import GPy

import gym
import matplotlib

from openmodelica_microgrid_gym.env import FullHistory
from openmodelica_microgrid_gym.auxiliaries import PI_params, DroopParams, MutableFloat, \
    MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.common import *

import numpy as np
import pandas as pd

# Choose which controller parameters should be adjusted by SafeOpt.
# - Kp: 1D example: Only the proportional gain Kp of the PI controller is adjusted
# - Ki: 1D example: Only the integral gain Ki of the PI controller is adjusted
# - Kpi: 2D example: Kp and Ki are adjusted simultaneously
adjust = 'Ki'

# Check if really only one simulation scenario was selected
if adjust not in {'Kp', 'Ki', 'Kpi'}:
    raise ValueError("Please set 'adjust' to one of the following values: 'Kp', 'Ki', 'Kpi'")

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 300  # number of simulation steps per episode
num_episodes = 15  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 1000  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
i_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A

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
          'figure.figsize': [3.39, 2.5],
          'font.family': 'serif',
          'lines.linewidth': 1
          }

matplotlib.rcParams.update(params)


def rew_fun(obs: pd.Series) -> float:
    """
    Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
    used parameters.
    Takes current measurements and setpoints so calculate the mean-root-error control error and uses a logarithmic
    barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

    :param obs: Observation from the environment (ControlVariables, e.g. currents and voltages)
    :return: Error as negative reward
    """

    # measurements: 3 currents through the inductors of the inverter -> control variables (CV)
    Iabc_master = obs[[f'lc1.inductor{i + 1}.i' for i in range(3)]].to_numpy()  # 3 phase currents at LC inductors
    phase = obs[['master.phase']].to_numpy()[0]  # phase from the master controller needed for transformation

    # setpoints
    ISPdq0_master = obs[['master.'f'SPI{k}' for k in 'dq0']].to_numpy()  # setting dq reference
    ISPabc_master = dq0_to_abc(ISPdq0_master, phase)  # convert dq set-points into three-phase abc coordinates

    # control error = mean-root-error (MRE) of reference minus measurements
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
        bounds = [(0, 300), (0.0, 0.03)]
        lengthscale = [50., .01]

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means. Performance measurements for optimization are seen as
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

    ctrl = dict()  # empty dictionary for the controller(s)

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
        current_dqp_iparams = PI_params(kP=5e-3, kI=mutable_params['currentI'], limits=(-1, 1))

    # For 2D example, choose Kp and Ki as mutable parameters
    elif adjust == 'Kpi':
        mutable_params = dict(currentP=MutableFloat(5e-3), currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl['master'] = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param,
                                                           undersampling=2)

    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    # Arguments described above
    # History is used to store results
    agent = SafeOptAgent([v for k, v in sorted(mutable_params.items(), key=itemgetter(0))],
                         abort_reward,
                         kernel,
                         dict(bounds=bounds, noise_var=noise_var, prior_mean=prior_mean,
                              safe_threshold=safe_threshold, explore_threshold=explore_threshold
                              ),
                         ctrl,
                         dict(master=[np.array([f'lc1.inductor{i + 1}.i' for i in range(3)]),
                                      np.array([f'lc1.capacitor{i + 1}.v' for i in range(3)]), i_ref],
                              ),
                         history=FullHistory()
                         )

    #####################################
    # Definition of the environment using a FMU created by OpenModelica
    # (https://www.openmodelica.org/)
    # Using an inverter supplying a load
    # - using the reward function described above as callable in the env
    # - viz_cols used to choose which measurement values should be displayed (here, only the 3 currents across the
    #   inductors of the inverters are plotted. Alternatively, the grid frequency and the currents transformed by
    #   the agent to the dq0 system can be plotted if requested.)
    # - inputs to the models are the connection points to the inverters (see user guide for more details)
    # - model outputs are the the 3 currents through the inductors and the 3 voltages across the capacitors
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=rew_fun,
                   time_step=delta_t,
                   # viz_cols=['master.freq', 'master.CVI*', 'lc1.ind*'],
                   viz_cols=['master.SPI[abc]', 'lc1.ind*'],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   model_params={'inverter1.v_DC': v_DC},
                   model_path='../fmu/grid.network_singleInverter.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']])
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)
    runner = Runner(agent, env)
    runner.run(num_episodes, visualise_env=True)

    import matplotlib.pyplot as plt

    agent.optimizer.plot(1000, ms=1)
    ax = plt.gca()

    ax.grid(which='both')
    ax.set_axisbelow(True)

    if adjust == 'Ki':
        ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
        ax.set_ylabel(r'$J$')
    elif adjust == 'Kpi':
        agent.params.reset()
        ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
        ax.set_ylabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
        ax.get_figure().axes[1].set_ylabel(r'$J$')
        plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-', zorder=1, lw=4,
                 alpha=.5)

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(f'GP_model{datetime.now().isoformat()}.pgf')
    fig.savefig(f'GP_model{datetime.now().isoformat()}.pdf')
