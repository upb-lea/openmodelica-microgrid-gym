from typing import Dict, Any

from tqdm import tqdm

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv


class Runner:
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
        self.best_episode = dict()  # type: Dict[str,Any]
        """
        :type dict:
        
        blabla
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
        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            obs = self.env.reset()
            done, r = False, None
            for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                self.agent.observe(r, done)
                act = self.agent.act(obs)
                self.env.measurement = self.agent.measurement
                obs, r, done, info = self.env.step(act)
                self.env.render()
                if done:
                    break
            self.agent.observe(r, done)
            _, env_fig = self.env.close()

            if visualise:
                agent_fig = self.agent.render()

            if i == 0 or self.agent.has_improved:
                # self.best_episode['best_agent_plt'] = self.agent.figure
                self.best_episode['best_env_plt'] = env_fig
                self.best_episode['best_episode_idx'] = agent_fig
                self.best_episode['i'] = i
