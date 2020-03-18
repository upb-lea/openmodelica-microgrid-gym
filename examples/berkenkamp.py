import gym
from gym_microgrid.agents import SafeOptAgent
from gym_microgrid import Runner

if __name__ == '__main__':
    agent = SafeOptAgent()
    env = gym.make('gym_microgrid:JModelicaConvEnv-v1', max_episode_steps=1000)

    runner = Runner(agent, env)
    runner.run(1)
