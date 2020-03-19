import logging

from gym_microgrid.execution import Runner
from gym_microgrid.agents import Agent

from gym.envs.registration import register

__all__ = ['Agent', 'Runner']

register(
    id='JModelicaConvEnv_test-v1',
    entry_point='gym_microgrid.env:JModelicaConvEnv',
    kwargs=dict(log_level=logging.DEBUG, max_episode_steps=100, viz_mode='step')
)

register(id='JModelicaConvEnv-v1', entry_point='gym_microgrid.env:JModelicaConvEnv')
