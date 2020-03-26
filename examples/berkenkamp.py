import gym
from gym_microgrid.agents import SafeOptAgent
from gym_microgrid import Runner

if __name__ == '__main__':
    agent = SafeOptAgent()
    env = gym.make('gym_microgrid:ModelicaEnv_test-v1',
                   viz_mode='episode',
                   model_path='grid.pll.fmu',
                   model_params={'pll.pi.k': 0.00001, 'pll.pi.T': 0.02},
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output={
                       'lc1': [
                           ['inductor1.i', 'inductor2.i', 'inductor3.i'],
                           ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                       'lcl1':
                           [['inductor1.i', 'inductor2.i', 'inductor3.i'],
                            ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
                       'pll': ['add_freq_nom_delta_f.y']}
                   )

    runner = Runner(agent, env)
    runner.run(1, visualize=True)
