#####################################
# Example using a FMU by OpenModelica and SafeOpt algorithm to find optimal controller parameters
# Simulation setup: Single inverter supplying 15 A d-current to an RL-load via a LC filter
# Controller: PI current controller gain parameters are optimized by SafeOpt


import logging
from typing import List

import GPy
import gym
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from openmodelica_microgrid_gym import Runner, Agent
from openmodelica_microgrid_gym.agents import SafeOptAgent, StaticControlAgent
from openmodelica_microgrid_gym.agents.simRewardMapAgent import simRewardMapAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.physical_testbench import TestbenchEnv
from openmodelica_microgrid_gym.execution.runnerRewardMap import RunnerRewardMap
from openmodelica_microgrid_gym.execution.runnerRewardMapSim import RunnerRewardMapSim
from openmodelica_microgrid_gym.execution.runner_hardware import RunnerHardware
from openmodelica_microgrid_gym.util import dq0_to_abc, nested_map, FullHistory

# Choose which controller parameters should be adjusted by SafeOpt.
# - Kp: 1D example: Only the proportional gain Kp of the PI controller is adjusted
# - Ki: 1D example: Only the integral gain Ki of the PI controller is adjusted
# - Kpi: 2D example: Kp and Ki are adjusted simultaneously

adjust = 'Kpi'

# Check if really only one simulation scenario was selected
if adjust not in {'Kp', 'Ki', 'Kpi'}:
    raise ValueError("Please set 'adjust' to one of the following values: 'Kp', 'Ki', 'Kpi'")


include_simulate = False




# Simulation definitions
delta_t = 1e-4  # simulation time step size / s
max_episode_steps = 1000  # number of simulation steps per episode
num_episodes = 200  # number of simulation episodes (i.e. SafeOpt iterations)
#v_DC = 40  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
i_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A

# Controller layout due to magniitude optimum:
L = 2.2e-3  # / H
R = 585e-3  # / Ohm
tau_plant = L/R
gain_plant = 1/R

# take inverter into account uning s&h (exp(-s*delta_T/2))

Tn = tau_plant   # Due to compensate
Kp_init = 0.1#tau_plant/(2*delta_t*gain_plant*v_DC)
Ki_init = 50#Kp_init/(Tn)


class Reward:
    def __init__(self):
        self._idx = None

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'rl.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0']])

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
                + -np.sum(2 * np.log(1 - np.maximum(np.abs(Iabc_master) - iNominal, 0) / (iLimit - iNominal)), axis=0) \
                * max_episode_steps

        return -error.squeeze()




if __name__ == '__main__':
    #####################################
    # Definition of the controllers

    kMatrix = np.array([np.linspace(0, 8, 20), np.linspace(0, 100, 20)])



    mutable_params = None
    current_dqp_iparams = None
    if adjust == 'Kp':
        # mutable_params = parameter (Kp gain of the current controller of the inverter) to be optimized using
        # the SafeOpt algorithm
        mutable_params = dict(currentP=MutableFloat(0.01))  # 5e-3))

        # Define the PI parameters for the current controller of the inverter
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=115, limits=(-1, 1))

    # For 1D example, if Ki should be adjusted
    elif adjust == 'Ki':
        mutable_params = dict(currentI=MutableFloat(20))
        current_dqp_iparams = PI_params(kP=0.01, kI=mutable_params['currentI'], limits=(-1, 1))

    # For 2D example, choose Kp and Ki as mutable parameters
    elif adjust == 'Kpi':
        mutable_params = dict(currentP=MutableFloat(0), currentI=MutableFloat(0))
        current_dqp_iparams = PI_params(kP=mutable_params['currentP'], kI=mutable_params['currentI'], limits=(-1, 1))

    # Define the droop parameters for the inverter of the active power Watt/Hz (DroopGain), delta_t (0.005) used for the
    # filter and the nominal frequency
    # Droop controller used to calculate the virtual frequency drop due to load changes
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)

    # Define the Q-droop parameters for the inverter of the reactive power VAR/Volt, delta_t (0.002) used for the
    # filter and the nominal voltage
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

    # Define a current sourcing inverter as master inverter using the pi and droop parameters from above
    ctrl = MultiPhaseDQCurrentSourcingController(current_dqp_iparams, delta_t, droop_param, qdroop_param,
                                                 undersampling=1, name='master')

    #####################################
    # Definition of the optimization agent
    # The agent is using the SafeOpt algorithm by F. Berkenkamp (https://arxiv.org/abs/1509.01066) in this example
    # Arguments described above
    # History is used to store results
    agent = simRewardMapAgent(kMatrix, mutable_params,
                             [ctrl],
                             dict(master=[[f'rl.inductor{k}.i' for k in '123'],
                                          [f'rl.inductor{k}.i' for k in '123'],
                                          i_ref]),
                            history=FullHistory()
                            )



    #env
    def xylables(fig):
            ax = fig.gca()
            ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
            ax.set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
            plt.title('Simulation')
            ax.grid(which='both')
            #time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            #fig.savefig('len_search/abc_current' + time + '.pdf')

    def xylables_dq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
        ax.grid(which='both')
        plt.title('Simulation')
        plt.ylim(0,36)
        #time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        #fig.savefig('len_search/dq0_current' + time + '.pdf')

    def xylables_mdq0(fig):
        ax = fig.gca()
        ax.set_xlabel(r'$t\,/\,\mathrm{ms}$')
        ax.set_ylabel('$m_{\mathrm{dq0}}\,/\,\mathrm{}$')
        plt.title('Simulation')
        ax.grid(which='both')
                #plt.ylim(0,36)

    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                        reward_fun=Reward().rew_fun,
                        time_step=delta_t,
                        viz_cols=[
                            PlotTmpl([f'rl.inductor{i}.i' for i in '123'],
                                        callback=xylables
                                        ),
                            PlotTmpl([f'master.CVI{i}' for i in 'dq0'],
                                        callback=xylables_dq0
                                        )
                               #PlotTmpl([f'master.m{i}' for i in 'dq0'],
                               #         callback=xylables_mdq0
                               #         ),
                               #PlotTmpl([f'master.m{i}' for i in 'abc'],
                               #         #callback=xylables_dq0
                               #         )
                        ],
                        #viz_cols = ['inverter1.*', 'rl.inductor1.i'],
                        log_level=logging.INFO,
                        viz_mode='episode',
                        max_episode_steps=max_episode_steps,
                        #model_params={'inverter1.gain.u': v_DC},
                        model_path='../fmu/grid.testbench_SC.fmu',
                        model_input=['i1p1', 'i1p2', 'i1p3'],
                        model_output=dict(rl=[['inductor1.i', 'inductor2.i', 'inductor3.i']],
                                             #inverter1=['inductor1.i', 'inductor2.i', 'inductor3.i']
                                           ),
                        history=FullHistory()
                    )
    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)



    runner = RunnerRewardMapSim(agent, env)

    runner.run(num_episodes, visualise=True)

    df = pd.DataFrame([[runner.rewardMatrix],
                       [agent.kMatrix[0]],
                       [agent.kMatrix[1]],
                       [num_episodes]])

    df.to_pickle('rewardMatrix_sim_test')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(agent.kMatrix[0], agent.kMatrix[1], runner.rewardMatrix, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.xlabel(r'$t\,/\,\mathrm{s}$')
    #plt.ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    plt.show()

    print(df)








