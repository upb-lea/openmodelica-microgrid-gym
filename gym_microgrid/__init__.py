import logging

from gym_microgrid.execution import Runner
from gym_microgrid.agents import Agent
from gym_microgrid.env import ModelicaEnv

from gym.envs.registration import register

__all__ = ['Agent', 'ModelicaEnv', 'Runner']

register(
    id='ModelicaEnv_test-v1',
    entry_point='gym_microgrid.env:ModelicaEnv',
    kwargs=dict(log_level=logging.DEBUG, max_episode_steps=500, viz_mode='step')
)

register(id='ModelicaEnv-v1', entry_point='gym_microgrid.env:ModelicaEnv')
