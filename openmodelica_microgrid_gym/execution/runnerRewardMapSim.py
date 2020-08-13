from typing import Dict, Any

from tqdm import tqdm
import numpy as np

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv


class RunnerRewardMapSim:
    """
    This class will execute an agent on the environment.
    It handles communication between agent and environment and handles the execution of multiple epochs
    """

    def __init__(self, agent: Agent, env: ModelicaEnv):
        """

        :param agent: Agent that acts on the environment
        :param env: Environment tha Agent acts on
        """
        self.env = env
        self.agent = agent
        self.agent.env = env
        self.run_data = dict()  # type: Dict[str,Any]

        self.rewardMatrix = np.zeros([len(self.agent.kMatrix[0]), len(self.agent.kMatrix[0])])


        """
        Dictionary storing information about the experiment.

        - "best_env_plt": environment best plots
        - "best_episode_idx": index of best episode
        - "agent_plt": last agent plot
        """

    def run(self, n_episodes: int = 10, visualise: bool = False):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param visualise: turns on visualization of the environment
        """
        self.agent.reset()
        self.env.history.cols = self.env.history.structured_cols(None) + self.agent.measurement_cols
        self.agent.obs_varnames = self.env.history.cols

        if not visualise:
            self.env.viz_mode = None
        #agent_fig = None

        for i in tqdm(range(len(self.agent.kMatrix[0])), desc='episodes', unit='epoch'):
            for j in tqdm(range(len(self.agent.kMatrix[0])), desc='episodes', unit='epoch'):
                obs = self.env.reset()
                done, r = False, None
                for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                    self.agent.observe(r, done, i+3, j+3)
                    act = self.agent.act(obs)
                    self.env.measurement = self.agent.measurement
                    obs, r, done, info = self.env.step(act)
                    self.env.render()
                    if done:
                        break
                self.agent.observe(r, done, i+18, j+18)

                _, env_fig = self.env.close()

                self.rewardMatrix[i,j] = self.agent.episode_reward

                self.agent.prepare_episode()  # not in observe due to sets episode reward to 0


