import gym
from gym_microgrid.agents import SafeOptAgent
from gym_microgrid import Runner

def f(t):
    return 0 if t < 0.2 else 1

if __name__ == '__main__':
    agent = SafeOptAgent()
    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   model_params={'rl1.switch1.state': f,'rl1.switch2.state': f, 'rl1.switch3.state': f},
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={
                       'lc1': [
                           ['inductor1.i', 'inductor2.i', 'inductor3.i'],
                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                       'lcl1':
                           [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                            ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]})

    runner = Runner(agent, env)
    runner.run(1, visualize=True)
