from typing import Dict, Any, Optional

from tqdm import tqdm

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv
from openmodelica_microgrid_gym.execution.callbacks import Callback


class Runner:
    """
    This class will execute an agent on the environment.
    It handles communication between agent and environment and handles the execution of multiple epochs
    """

    def __init__(self, agent: Agent, env: ModelicaEnv, callback: Optional[Callback] = None):
        """

        :param agent: Agent that acts on the environment
        :param env: Environment tha Agent acts on
        """
        self.env = env
        self.agent = agent
        self.agent.env = env
        self.run_data = dict()  # type: Dict[str,Any]
        self.callback = callback
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
        self.agent.obs_varnames = self.env.history.cols
        self.env.history.cols = self.env.history.structured_cols(None) + self.agent.measurement_cols
        self.env.measure=self.agent.measure

        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            obs = self.env.reset()
            if self.callback is not None:
                self.callback.reset()
            done, r = False, None
            for _ in tqdm(range(self.env.max_episode_steps), desc='steps', unit='step', leave=False):
                self.agent.observe(r, done)
                act = self.agent.act(obs)
                obs, r, done, info = self.env.step(act)
                if self.callback is not None:
                    self.callback(self.env.history.cols, self.env.history.last())
                if visualise:
                    self.env.render()
                if done:
                    break
            # close env before calling final agent observe to see plots even if agent crashes
            _, env_fig = self.env.close()
            self.agent.observe(r, done)

            if visualise:
                agent_fig = self.agent.render()

            self.run_data['last_agent_plt'] = agent_fig

            if i == 0 or self.agent.has_improved:
                self.run_data['best_env_plt'] = env_fig
                self.run_data['best_episode_idx'] = i

            if i == 0 or self.agent.has_worsened:
                self.run_data['worst_env_plt'] = env_fig
                self.run_data['worst_episode_idx'] = i
