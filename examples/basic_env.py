import gym

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   max_episode_steps=None,
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../OpenModelica_Microgrids/OpenModelica_Microgrids.Grids.Network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
