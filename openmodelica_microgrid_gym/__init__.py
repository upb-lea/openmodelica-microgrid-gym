import logging

from gym.envs.registration import register

from openmodelica_microgrid_gym.agents import Agent
from openmodelica_microgrid_gym.env import ModelicaEnv
from openmodelica_microgrid_gym.execution import Runner

__all__ = ['Agent', 'Runner']
__version__ = '0.3.0'

register(
    id='ModelicaEnv_test-v1',
    entry_point='openmodelica_microgrid_gym.env:ModelicaEnv',
    kwargs=dict(log_level=logging.DEBUG, max_episode_steps=500, viz_mode='episode', is_normalized=False)
)

register(
    id='ModelicaEnv-v1',
    entry_point='openmodelica_microgrid_gym.env:ModelicaEnv',
    kwargs=dict(is_normalized=False)
)
