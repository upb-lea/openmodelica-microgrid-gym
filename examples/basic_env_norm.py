import gym

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   is_normalized=True,
                   net='net.yaml',
                   model_path='../omg_grid/omg_grid.Grids.Network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        if done:
            break
    env.close()
