from typing import Union, Callable

import numpy as np


def linear_schedule(initial_value, final_value, t_start, t_end, total_timesteps: int = 1000) -> Callable[
    [float], float]:
    """
    Linear learning rate schedule from t_start to t_end in between initial -> final value.
    :param initial_value: (float or str) start value
    :param final_value: final value
    :param t_start: timestep (int!) at which the linear decay starts
    :param t_ends: timestep (int!) at which the linear decay ends
    :param total_timesteps: number of learning steps
    :return: (function)
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)  progress_remaining = 1.0 - (num_timesteps / total_timesteps)
        :return: (float)
        """
        # Original: return= initial_value *  progress_remaining

        return np.maximum(
            np.minimum(initial_value, initial_value + (t_start * (initial_value - final_value)) / (t_end - t_start) \
                       - (initial_value - final_value) / (t_end - t_start) * ((1.0 - progress_remaining) \
                                                                              * total_timesteps)), final_value)

        # return  np.maximum(final_value, np.minimum(initial_value,b+ m *(1.0 - progress_remaining) * total_timesteps))

    return func


def exopnential_schedule(initial_value: Union[float, str], final_value: float = 0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str) start value
    :param final_value: final value as percentage of initial value (e.g. 0.1 -> final value is 10 % of initial value
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float) 1.0 - (num_timesteps / total_timesteps)
                                            Y - X *  M
        :return: (float)
        https://www.jeremyjordan.me/nn-learning-rate/
        """
        # return (progress_remaining * initial_value)*(1-(1-progress_remaining))# + final_value * initial_value
        # return ( initial_value)**(1/progress_remaining)# + final_value * initial_value
        raise NotImplementedError
        return (initial_value) * (progress_remaining - 1)

    return func
