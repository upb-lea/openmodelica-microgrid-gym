import numpy as np

from openmodelica_microgrid_gym.util import RandProcess


class RandomLoad:
    def __init__(self, max_episode_steps: int, ts: float, rand_process: RandProcess, loadstep_time: int = None):
        """

        :param max_episode_steps: number of steps per episode
        :param ts: sampletime of env
        :param rand_pocess: Instance of random process defines noise added to load
        :param loadstep_time: number of env step where load step should happen
        toDo: choose number of loadsteps; distribute them randomly in the episode and let rand choose if up or down
        """
        self.max_episode_steps = max_episode_steps
        self.ts = ts
        self.rand_process = rand_process
        if loadstep_time is None:
            self.loadstep_time = np.random.randint(0, self.max_episode_steps)
        else:
            self.loadstep_time = loadstep_time

    def reset(self, loadstep_time=None):
        if loadstep_time is None:
            self.loadstep_time = np.random.randint(0, self.max_episode_steps)
        else:
            self.loadstep_time = loadstep_time

    def load_step(self, t, gain):
        """
        Changes the load parameters
        :param t:
        :param gain: device parameter
        :return: Sample from SP
        """
        # Defines a load step after 0.01 s
        if self.loadstep_time * self.ts < t <= self.loadstep_time * self.ts + self.ts:
            self.rand_process.proc.mean = gain * 0.55
            self.rand_process.reserve = gain * 0.55
        elif t <= self.ts:
            self.rand_process.proc.mean = gain

        return self.rand_process.sample(t)
