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
        self.best_episode = dict()

        # toDo: Var: best_episode als dict: plots - best agent plot (ueber return von agend.render(->figure)); best env plot; best J_wert?

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

        for _ in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
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
            self.env.close()
            if visualise:
                self.agent.render()

            if self.agent.has_improved:
                self.best_episode['best_agent_plt'] = self.agent.figure
                self.best_episode['best_episode_idx'] = self.agent.best_episode

            # toDo: if self.agenthas_improved: save best_episde dict
