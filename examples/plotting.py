import gym

from openmodelica_microgrid_gym.env import PlotTmpl

if __name__ == '__main__':
    def second_plot(fig):
        ax = fig.gca()
        ax.set_ylabel('y label!')
        ax.set_xlabel('$t\,/\,\mathrm{ms}$')
        fig.savefig('plot2.pdf')


    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   viz_mode='episode',
                   viz_cols=[
                       PlotTmpl([f'lc1.inductor{i}.i' for i in '123'],
                                callback=lambda fig: fig.savefig('plot.pdf'),
                                linewidth=4,
                                style=[None, '--', '*'],
                                linestyle=['None', None, None],
                                marker=[r'$\heartsuit$', None, None],
                                c=['pink', None, None],
                                title='test'),
                       PlotTmpl(['lc1.inductor1.i', 'lc1.inductor2.i'], callback=second_plot,
                                legend=[False, True],
                                label=[None, 'something'])
                   ],
                   max_episode_steps=None,
                   net='../net/net.yaml',
                   model_path='../omg_grid/grid.network.fmu')

    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
