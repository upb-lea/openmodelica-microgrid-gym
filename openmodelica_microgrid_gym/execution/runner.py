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

    def run(self, n_episodes: int = 10, visualise_env: bool = False):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param visualise_env: turns on visualization of the environment
        """
        self.agent.reset()

        for _ in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            obs = self.env.reset()
            done, r = False, None
            for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                self.agent.observe(r, done)
                act = self.agent.act(obs)
                self.env.update_measurements(self.agent.measurement)
                obs, r, done, info = self.env.step(act)
                if visualise_env:
                    self.env.render()
                if done:
                    break
            self.agent.observe(r, done)
            self.env.close()
            self.agent.render()
