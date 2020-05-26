import gym

from openmodelica_microgrid_gym.env import PlotTmpl

if __name__ == '__main__':
    def second_plot(fig):
        ax = fig.gca()
        ax.legend(['line'])
        fig.savefig('plot2.pdf')


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   viz_mode='episode',
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],
                                callback=lambda fig: fig.savefig('plot.pdf'),
                                linewidth=4,
                                style=[None, '--', '*'],
                                marker=[r'$\heartsuit$', None, None],
                                c=['pink', None, None],
                                linestyle=['None', None, None]),
                       PlotTmpl(['lc1.inductor1.i'], callback=second_plot)
                   ],
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   max_episode_steps=None,
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../fmu/grid.network.fmu')

    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
