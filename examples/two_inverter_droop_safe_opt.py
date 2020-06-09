#####################################
# Example using a FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller gain parameters are optimized by SafeOpt


import logging
from functools import partial
from time import strftime, gmtime
from typing import List

import GPy
import gym
import numpy as np

import matplotlib.pyplot as plt

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQCurrentSourcingController, \
    MultiPhaseDQ0PIPIController, PLLParams, InverseDroopParams, MultiPhaseDQCurrentController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

# Choose which controller parameters should be adjusted by SafeOpt.
# - Kp: 1D example: Only the proportional gain Kp of the PI controller is adjusted
# - Ki: 1D example: Only the integral gain Ki of the PI controller is adjusted
# - Kpi: 2D example: Kp and Ki are adjusted simultaneously

adjust = 'pDroop'

# Check if really only one simulation scenario was selected
if adjust not in {'pDroop', 'Ki', 'all'}:
    raise ValueError("Please set 'adjust' to one of the following values: 'pDroop', 'Ki', 'all'")

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 6000  # number of simulation steps per episode
num_episodes = 20  # number of simulation episodes (i.e. SafeOpt iterations)
v_DC = 1000  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
mu = 2  # factor for barrier function (see below)
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
i_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A

def load_step(t, gain):
    """
    Defines a load step after 0.2 s
    Doubles the load parameters
    :param t:
    :param gain: device parameter
    :return: Dictionary with load parameters
    """
    return 1*gain if t < .15 else 5*gain


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [f'slave.freq'])
                #[[f'lc1.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
                # [f'lc1.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0']])

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
        freq = data[idx[0]]

        """
        Iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        phase = data[idx[1]]  # phase from the master controller needed for transformation
        Vabc_master = data[idx[3]]  # 3 phase currents at LC inductors

        # setpoints
        ISPdq0_master = data[idx[2]]  # setting dq reference
        ISPabc_master = dq0_to_abc(ISPdq0_master, phase)  # convert dq set-points into three-phase abc coordinates
        VSPdq0_master = data[idx[4]]  # setting dq reference
        VSPabc_master = dq0_to_abc(VSPdq0_master, phase)  # convert dq set-points into three-phase abc coordinates
        """

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = np.sum((np.abs((nomFreq- freq)) / nomFreq) ** 0.5, axis=0)

        #error = np.sum((np.abs((ISPabc_master - Iabc_master)) / iLimit) ** 0.5, axis=0) \
        #        + -np.sum(mu * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
        #        * max_episode_steps \
        #        + np.sum((np.abs((VSPabc_master - Vabc_master)) / nomVoltPeak) ** 0.5, axis=0)

        if np.isnan(error):
            asd=1
        return -error.squeeze()


if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    #####################################
    # Definitions for the GP
    prior_mean = 2  # mean factor of the GP prior mean which is multiplied with the first performance of the initial set
    noise_var = 0.001 ** 2  # measurement noise sigma_omega
    prior_var = 0.1  # prior variance of the GP

    bounds = None
    lengthscale = None
    if adjust == 'pDroop':
        bounds = [(0, 100000), (0, 100000)]  # bounds on the input variable Kp
        lengthscale = [50000, 50000]  # length scale for the parameter variation [Kp] for the GP

    # For 1D example, if Ki should be adjusted
    if adjust == 'Ki':
        bounds = [(0, 300)]  # bounds on the input variable Ki
        lengthscale = [50.]  # length scale for the parameter variation [Ki] for the GP

    # For 2D example, choose Kp and Ki as mutable parameters (below) and define bounds and lengthscale for both of them
    if adjust == 'all':
        bounds = [(0.0, 0.03), (0, 300),(0.0, 0.03), (0, 300)]
        lengthscale = [.005, 10.,.005, 10.]

    # The performance should not drop below the safe threshold, which is defined by the factor safe_threshold times
    # the initial performance: safe_threshold = 1.2 means. Performance measurement for optimization are seen as
    # unsafe, if the new measured performance drops below 20 % of the initial performance of the initial safe (!)
    # parameter set
    safe_threshold = 5

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
    if adjust == 'pDroop':
        # mutable_params = parameter (Kp gain of the current controller of the inverter) to be optimized using
        # the SafeOpt algorithm
        mutable_params = dict(pDroop_master=MutableFloat(20000.0), pDroop_slave=MutableFloat(20000.0))

        droop_param_master = DroopParams(mutable_params['pDroop_master'], 0.005, nomFreq)
        droop_param_slave = InverseDroopParams(mutable_params['pDroop_slave'], delta_t, nomFreq, tau_filt = 0.04)

    # For 1D example, if Ki should be adjusted
    elif adjust == 'Ki':
        mutable_params = dict(currentI=MutableFloat(10))
        current_dqp_iparams = PI_params(kP=10e-3, kI=mutable_params['currentI'], limits=(-1, 1))

    # For 2D example, choose Kp and Ki as mutable parameters
    elif adjust == 'all':
        mutable_params = dict(currentP=MutableFloat(10e-3), currentI=MutableFloat(10), voltageP=MutableFloat(25e-3),
                              voltageI=MutableFloat(60), Pdroop=40000)
        #mutable_params = dict(currentP=MutableFloat(0e-3), currentI=MutableFloat(0), voltageP=MutableFloat(0e-3),
         #                     voltageI=MutableFloat(0))

        voltage_dqp_iparams = PI_params(kP=mutable_params['voltageP'], kI=mutable_params['voltageI'], limits=(-iLimit, iLimit))
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))
        droop_param = DroopParams(mutable_params['Pdroop'], 0.005, nomFreq)
    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    # droop_param = DroopParams(DroopGain, 0.005, nomFreq)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    # define Master
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    # Current control PI gain parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param_master, qdroop_param,
                                                 undersampling=2, name='master'))


    #######
    # slave
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI gain parameters for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=(-10000, 10000), f_nom=nomFreq)
    # Droop characteristic for the active power Watts/Hz, W.s/Hz

    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(50, delta_t, nomVoltPeak, tau_filt=0.01)
    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                              droop_param_slave, qdroop_param, name='slave'))

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
                         ctrl,
                         {'master':[[f'lc1.inductor{k}.i' for k in '123'],
                                      [f'lc1.capacitor{k}.v' for k in '123'],
                                      ],
                          'slave': [[f'lcl1.inductor{k}.i' for k in '123'],
                                    [f'lcl1.capacitor{k}.v' for k in '123'],
                                    np.zeros(3)]},
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

    def xylables_i(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        #timestamps = df.index.strftime("%Y-%m-%d %H:%M:%S")
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig('saves_droop/Inductor_currents'+time+'.pdf')

    def xylables_v_abc(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig('saves_droop/abc_voltage' + time + '.pdf')


    def xylables_v_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig('saves_droop/dq0_voltage' + time + '.pdf')

    def xylables_P_master(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$P_{\mathrm{master}}\,/\,\mathrm{W}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig('saves_droop/P_master' + time + '.pdf')

    def xylables_P_slave(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$P_{\mathrm{slave}}\,/\,\mathrm{W}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig('saves_droop/P_slave' + time + '.pdf')

    def xylables_freq(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$f_{\mathrm{slave}}\,/\,\mathrm{Hz}$')
        ax.grid(which='both')
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig('saves_droop/f_slave' + time + '.pdf')

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   reward_fun=Reward().rew_fun,
                   time_step=delta_t,
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],
                                callback=xylables_i
                                ),
                       PlotTmpl([f'lc1.capacitor{i}.v' for i in '123'],
                                callback=xylables_v_abc
                                ),
                       PlotTmpl([f'master.instPow'],
                                callback=xylables_P_master
                                ),
                       PlotTmpl([f'slave.instPow'],
                                callback=xylables_P_slave
                                ),
                       PlotTmpl([f'slave.freq'],
                                callback=xylables_freq
                                )
                   ],
                   log_level=logging.INFO,
                   viz_mode='episode',
                   max_episode_steps=max_episode_steps,
                   model_params={'rl1.resistor1.R': partial(load_step, gain=20),
                                 'rl1.resistor2.R': partial(load_step, gain=20),
                                 'rl1.resistor3.R': partial(load_step, gain=20),
                                 'rl1.inductor1.L': partial(load_step, gain=0.001),#0.001,
                                 'rl1.inductor2.L': partial(load_step, gain=0.001),#0.001,
                                 'rl1.inductor3.L': partial(load_step, gain=0.001)#0.001
                                 },
                   model_path='../fmu/grid.network.fmu',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output=dict(lc1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                          ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                                     rl1=[f'inductor{i}.i' for i in range(1, 4)],
                                     lcl1=[['inductor1.i', 'inductor2.i', 'inductor3.i'],
                                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]),
                   history=FullHistory()
                   )

    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)
    runner = Runner(agent, env)

    runner.run(num_episodes, visualise=True)

    agent.history.df.to_csv('saves_droop/result.csv')

    print('\n Experiment finished with best set: \n\n {}'.format(agent.history.df.round({'J': 4, 'Params': 4})))

    print('\n Experiment finished with best set: \n')
    print('\n  {} = {}' .format(adjust, agent.history.df.at[np.argmax(agent.history.df['J']),'Params']))
    print('  Resulting in a performance of J = {}'.format(np.max(agent.history.df['J'])))
    print('\n\nBest experiment results are plotted in the following:')


    # Show best episode measurment (current) plot
    best_env_plt = runner.run_data['best_env_plt']
    for ii in range(len(best_env_plt)):
        ax = best_env_plt[ii].axes[0]
        ax.set_title('Best Episode')
        best_env_plt[ii].show()
        #best_env_plt[0].savefig('best_env_plt.png')

    asd = 1

    # Show last performance plot
    best_agent_plt = runner.run_data['last_agent_plt']
    ax = best_agent_plt.axes[0]
    ax.grid(which='both')
    ax.set_axisbelow(True)

    if adjust == 'pDroop':
        ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(VA^{-1}s^{-1})}$')
        ax.set_ylabel(r'$J$')
    elif adjust == 'Kp':
        ax.set_xlabel(r'$K_\mathrm{p}\,/\,\mathrm{(VA^{-1})}$')
        ax.set_ylabel(r'$J$')
    elif adjust == 'pDroop':
        agent.params.reset()
        ax.set_xlabel(r'$K_\mathrm{Pdroop,master}\,/\,\mathrm{(WHz^{-1})}$')
        ax.set_ylabel(r'$K_\mathrm{Pdroop,master}\,/\,\mathrm{(WHz^{-1})}$')
        ax.get_figure().axes[1].set_ylabel(r'$J$')
        plt.plot(bounds[0], [mutable_params['currentP'].val, mutable_params['currentP'].val], 'k-', zorder=1, lw=4,
                 alpha=.5)
    best_agent_plt.show()
    best_agent_plt.savefig('saves_droop/agent_plt.png')


