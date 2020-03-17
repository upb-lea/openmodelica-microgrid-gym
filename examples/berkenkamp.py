import gym

from gym_microgrid.agents import SafeOptAgent
from gym_microgrid import Runner

agent = SafeOptAgent()
env = gym.make('JModelicaConvEnv-v1')

runner = Runner(agent, env)
runner.run(10)
