import gym
from gym_microgrid.agents import SafeOptAgent
from gym_microgrid import Runner

if __name__ == '__main__':
    agent = SafeOptAgent()
    env = gym.make('gym_microgrid:JModelicaConvEnv_test-v1',
                   viz_mode='step',
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output=['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i',
                                 'lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v',
                                 'lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i',
                                 'lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v'])

    runner = Runner(agent, env)
    runner.run(1, visualize=True)
