import gym

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:NormalizedEnv_test-v1',
                   net='net.yaml',
                   model_path='../fmu/grid.network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
