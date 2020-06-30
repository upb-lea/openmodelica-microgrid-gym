import logging

from openmodelica_microgrid_gym.execution import Runner
from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv, NormalizedEnv

from gym.envs.registration import register

__all__ = ['Agent', 'Runner']
__version__ = '0.2.0'

register(
    id='ModelicaEnv_test-v1',
    entry_point='openmodelica_microgrid_gym.env:ModelicaEnv',
    kwargs=dict(log_level=logging.DEBUG, max_episode_steps=500, viz_mode='step')
)

register(id='ModelicaEnv-v1', entry_point='openmodelica_microgrid_gym.env:ModelicaEnv')

register(
    id='NormalizedEnv_test-v1',
    entry_point='openmodelica_microgrid_gym.env:NormalizedEnv',
    kwargs=dict(log_level=logging.DEBUG, max_episode_steps=500, viz_mode='step')
)