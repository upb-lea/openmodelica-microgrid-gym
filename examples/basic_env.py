import gym

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   max_episode_steps=None,
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../omg_grid/omg_grid.Grids.Network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
