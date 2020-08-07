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
from openmodelica_microgrid_gym.agents import SafeOptAgent
from openmodelica_microgrid_gym.agents.util import MutableFloat
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQCurrentSourcingController
from openmodelica_microgrid_gym.env import PlotTmpl
from openmodelica_microgrid_gym.env.physical_testbench import TestbenchEnv
from openmodelica_microgrid_gym.execution.runnerRewardMap import RunnerRewardMap
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
num_episodes = 1  # number of simulation episodes (i.e. SafeOpt iterations)
#v_DC = 40  # DC-link voltage / V; will be set as model parameter in the FMU
nomFreq = 50  # nominal grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # inverter current limit / A
iNominal = 20  # nominal inverter current / A
i_ref = np.array([15, 0, 0])  # exemplary set point i.e. id = 15, iq = 0, i0 = 0 / A



class measureRewardAgent(Agent):

    def __init__(self):
        super().__init__([])
        self.kMatrix = np.array([np.linspace(0, 8, 20),np.linspace(0,100, 20)])
        self.episode_reward = 0

    def reset(self):
        self.episode_reward = 0

    def observe(self, reward: float, terminated: bool):
        """
        The observe function is might be called after the act function.
        It might trigger the learning in some implementations.

        :param reward: reward from the environment after the last action
        :param terminated: whether the episode is finished
        """
        self.episode_reward += reward or 0

    def act(self, obs) -> np.ndarray:
        return self.env.action_space.sample()


if __name__ == '__main__':
    #####################################
    # Dummy agent, not needed for measurement, kp&ki get set by martix, fixed
    agent = measureRewardAgent()



    #####################################
    # Execution of the experiment
    # Using a runner to execute 'num_episodes' different episodes (i.e. SafeOpt iterations)

    env = TestbenchEnv(num_steps= max_episode_steps, DT = 1/20000, i_ref = i_ref[0])
    runner = RunnerRewardMap(agent, env)

    runner.run(num_episodes, visualise=True)

    df = pd.DataFrame([[runner.rewardMatrix],
                       [agent.kMatrix[0]],
                       [agent.kMatrix[1]]])

    df.to_pickle('rewardMatrix')

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








