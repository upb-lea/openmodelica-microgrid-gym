from typing import Dict, Any

from tqdm import tqdm
import numpy as np

from openmodelica_microgrid_gym.agents import Agent
from experiments.model_validation.env.physical_testbench import TestbenchEnv
from experiments.model_validation.env.testbench_voltage_ctrl import TestbenchEnvVoltage


class RunnerHardware:
    """
    This class will execute an agent on the environment.
    It handles communication between agent and environment and handles the execution of multiple epochs
    """

    def __init__(self, agent: Agent, env: TestbenchEnv):
        """

        :param agent: Agent that acts on the environment
        :param env: Environment that Agent acts on
        """
        self.env = env
        self.agent = agent
        self.agent.env = env
        self.run_data = dict()  # type: Dict[str,Any]
        """
        :type dict:
        
        Stores information about the experiment.
        best_env_plt - environment best plots
        best_episode_idx - index of best episode
        agent_plt - last agent plot
        
        """

    def run(self, n_episodes: int = 10, visualise: bool = False, save_folder: str = 'Mess'):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param visualise: turns on visualization of the environment
        :param save_folder: string with save folder name
        """
        self.agent.reset()

        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            self.env.reset(self.agent.params[0], self.agent.params[1])
            #self.env.render(0)
            done, r = False, None
            for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                self.agent.observe(r, done)
                #act = self.agent.act(obs)
                #self.env.measurement = self.agent.measurement
                obs, r, done, info = self.env.step()
                #self.env.render()
                if done:
                    break
            self.env.render(0, save_folder)
            self.agent.observe(r, done)

            self.env.render(self.agent.history.df.J.iloc[-1], save_folder)


            print(self.agent.unsafe)

            if visualise:
                agent_fig = self.agent.render()

            self.run_data['last_agent_plt'] = agent_fig

            if i == 0 or self.agent.has_improved:
                #self.run_data['best_env_plt'] = env_fig
                self.run_data['best_episode_idx'] = i


class RunnerHardwareGradient:
    """
    This class will execute an agent on the environment.
    It handles communication between agent and environment and handles the execution of multiple epochs
    adds gradient depending return
    """

    def __init__(self, agent: Agent, env: TestbenchEnvVoltage):
        """

        :param agent: Agent that acts on the environment
        :param env: Environment that Agent acts on
        """
        self.env = env
        self.agent = agent
        self.agent.env = env
        self.run_data = dict()  # type: Dict[str,Any]
        """
        :type dict:

        Stores information about the experiment.
        best_env_plt - environment best plots
        best_episode_idx - index of best episode
        agent_plt - last agent plot

        """

    def run(self, n_episodes: int = 10, visualise: bool = False, save_folder: str = 'Mess'):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param visualise: turns on visualization of the environment
        :param save_folder: string with save folder name
        """
        self.agent.reset()
        # self.env.history.cols = self.env.history.structured_cols(None) + self.agent.measurement_cols
        # self.agent.obs_varnames = self.env.history.cols

        # if not visualise:
        #    self.env.viz_mode = None
        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            self.env.reset4D(self.agent.params[0], self.agent.params[1], self.agent.params[2], self.agent.params[3])
            #self.env.reset(self.agent.params[0], self.agent.params[1])
            # self.env.reset(0.01, self.agent.params[0])
            # self.env.render(0)
            print(self.agent.params[0])
            print(self.agent.params[1])
            done, r = False, None
            for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                self.agent.observe(r, False)
                # act = self.agent.act(obs)
                # self.env.measurement = self.agent.measurement
                obs, r, done, info = self.env.step()
                # self.env.render()
                if done:
                    self.agent.observe(r, False)

                    w = self.env.data[:, 27]
                    w1 = self.env.data[:, 28]
                    w2 = self.env.data[:, 29]

                    v = np.ones(len(w)) * 169.7

                    SP_sattle = (abs(w - v) < v * 0.12).astype(int)  # 0.12 -> +-20V setpoint

                    dw = np.gradient(w)
                    dw1 = np.gradient(w1)
                    dw2 = np.gradient(w2)

                    dev_return = (np.mean(abs(SP_sattle * dw)) + np.mean(abs(SP_sattle * dw1)) + np.mean(
                        abs(SP_sattle * dw2)))

                    dev_fac = 2.5


                    if i == 0:
                        self.agent.initial_performance = self.agent.episode_return - dev_return *dev_fac
                        self.agent.performance = ((
                                                              self.agent.episode_return - dev_return *dev_fac) - self.agent.min_performance) \
                                                 / (self.agent.initial_performance - self.agent.min_performance)
                        self.agent.last_best_performance = self.agent.performance
                        self.agent.last_worst_performance = self.agent.performance

                    self.agent.performance = ((self.agent.episode_return - dev_return *dev_fac) - self.agent.min_performance) \
                                             / (self.agent.initial_performance - self.agent.min_performance)

                    self.agent.prepare_episode()
            #self.env.render(0, save_folder)
            self.agent.update_params()
            # self.agent.observe(r, done)

            self.env.render(self.agent.history.df.J.iloc[-1], save_folder)

            print(self.agent.unsafe)

            if visualise:
                agent_fig = self.agent.render()

            self.run_data['last_agent_plt'] = agent_fig

            if i == 0 or self.agent.has_improved:
                # self.run_data['best_env_plt'] = env_fig
                self.run_data['best_episode_idx'] = i
