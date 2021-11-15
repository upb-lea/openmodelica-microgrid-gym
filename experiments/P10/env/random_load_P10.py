import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.util import RandProcess


class RandomLoad:
    def __init__(self, train_episode_length: int, ts: float, rand_process: RandProcess, loadstep_time: int = None,
                 load_curve: pd.DataFrame = None, bounds=None, bounds_std=None):
        """

        :param max_episode_steps: number of steps per training episode (can differ from env.max_episode_steps)
        :param ts: sampletime of env
        :param rand_pocess: Instance of random process defines noise added to load
        :param loadstep_time: number of env step where load step should happen
        :param load_curve: Stored load data to sample from instead of smaple from distribution
        :param bounds: Bounds to clip the sampled load data
        :param bounds_std: Chosen bounds are sampled from a distribution with std=bounds_std and mean=bounds

        """
        self.train_episode_length = train_episode_length
        self.ts = ts
        self.rand_process = rand_process
        if loadstep_time is None:
            self.loadstep_time = np.random.randint(0, self.train_episode_length)
        else:
            self.loadstep_time = loadstep_time
        self.load_curve = load_curve
        if bounds is None:
            self.bounds = (-np.inf, np.inf)
        else:
            self.bounds = bounds
        if bounds_std is None:
            self.bounds_std = (0, 0)
        else:
            self.bounds_std = bounds_std

        self.lowerbound_std = 0
        self.upperbound_std = 0

    def reset(self, loadstep_time=None):
        if loadstep_time is None:
            self.loadstep_time = np.random.randint(0, self.train_episode_length)
        else:
            self.loadstep_time = loadstep_time

    def clipped_step(self, t):
        return np.clip(self.rand_process.sample(t),
                       self.bounds[0] + self.lowerbound_std,
                       self.bounds[1] + self.upperbound_std
                       )

    def give_dataframe_value(self, t, col):
        """
        Gives load values from a stored dataframe (self.load_curve)
        :parma t: time - represents here the row of the dataframe
        :param col: colon name of the dataframe (typically str)
        """
        if t < 0:
            # return None
            return self.load_curve[col][0]
        if self.load_curve is None:
            raise ValueError('No dataframe given! Please feed load class (.load_curve) with data')
        return self.load_curve[col][int(t / self.ts)]

    def random_load_step(self, t, event_prob: int = 2, step_prob: int = 50):
        """
        Changes the load parameters applying a loadstep with 0.2% probability which is a pure step with 50 %
        probability otherwise a drift. In every event the random process variance is drawn randomly [1, 150].
        :param t: time
        :param event_prob: probability (in pre mill) that the step event is triggered in the current step
        :param step_prob: probability (in pre cent) that event is a abrupt step (drift otherwise!, random process speed
                          not adjustable yet
        :return: Sample from SP
        """
        # Changes rand process data with probability of 5% and sets new value randomly
        if np.random.randint(0, 1001) < 2:

            gain = np.random.randint(self.rand_process.bounds[0], self.rand_process.bounds[1])

            self.rand_process.proc.mean = gain
            self.rand_process.proc.vol = np.random.randint(1, 150)
            self.rand_process.proc.speed = np.random.randint(10, 1200)
            # define sdt for clipping once every event
            # np.maximum to not allow negative values
            self.lowerbound_std = np.maximum(np.random.normal(scale=self.bounds_std[0]), 0.0001)
            self.upperbound_std = np.random.normal(scale=self.bounds_std[1])

            # With 50% probability do a step or a drift
            if np.random.randint(0, 101) < 50:
                # step
                self.rand_process.reserve = gain

            else:
                # drift -> Lower speed to allow
                self.rand_process.proc.speed = np.random.randint(10, 100)

        return np.clip(self.rand_process.sample(t),
                       self.bounds[0] + self.lowerbound_std,
                       self.bounds[1] + self.upperbound_std
                       )

    def do_change(self, event_prob_permill=2, step_prob_percent=50):
        if np.random.randint(0, 1001) < event_prob_permill:

            gain = np.random.randint(self.rand_process.bounds[0], self.rand_process.bounds[1])

            self.rand_process.proc.mean = gain
            self.rand_process.proc.vol = np.random.randint(1, 150)
            self.rand_process.proc.speed = np.random.randint(10, 1200)
            # define sdt for clipping once every event
            self.lowerbound_std = np.random.normal(scale=self.bounds_std[0])
            self.upperbound_std = np.random.normal(scale=self.bounds_std[1])

            # With 50% probability do a step or a drift
            if np.random.randint(0, 101) < step_prob_percent:
                # step
                self.rand_process.reserve = gain

            else:
                # drift -> Lower speed to allow
                self.rand_process.proc.speed = np.random.randint(10, 100)
