from gym_microgrid.execution import Runner
from gym_microgrid.agents import Agent

from gym.envs.registration import register

__all__ = ['Agent', 'Runner']

register(
    id='JModelicaConvEnv-v1',
    entry_point='gym_microgrid.env:JModelicaConvEnv',
    kwargs=dict(time_step=1e-4, positive_reward=1, negative_reward=-100, log_level=4, solver_method='LSODA')
)
