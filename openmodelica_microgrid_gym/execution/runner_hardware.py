from typing import Dict, Any

from tqdm import tqdm

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv


class RunnerHardware:
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
        """
        :type dict:
        
        Stores information about the experiment.
        best_env_plt - environment best plots
        best_episode_idx - index of best episode
        agent_plt - last agent plot
        
        """

    def run(self, n_episodes: int = 10, visualise: bool = False):
        """
        Trains/executes the agent on the environment for a number of epochs

        :param n_episodes: number of epochs to play
        :param visualise: turns on visualization of the environment
        """
        self.agent.reset()
        #self.env.history.cols = self.env.history.structured_cols(None) + self.agent.measurement_cols
        #self.agent.obs_varnames = self.env.history.cols

        #if not visualise:
        #    self.env.viz_mode = None
        agent_fig = None

        for i in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            #self.env.reset(self.agent.params[0], self.agent.params[1])
            self.env.reset(self.agent.params[0], 5)
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
            self.agent.observe(r, done)
            self.env.render(self.agent.history.df.J.iloc[-1])

            if visualise:
                agent_fig = self.agent.render()

            self.run_data['last_agent_plt'] = agent_fig

            if i == 0 or self.agent.has_improved:
                #self.run_data['best_env_plt'] = env_fig
                self.run_data['best_episode_idx'] = i
