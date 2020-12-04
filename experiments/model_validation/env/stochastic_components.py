import numpy as np


class Load:

    def __init__(self, load_mean: float, load_std: float = None, balanced: bool = True, tolerance: float = 0.05):
        """
        Load class which defines stochastic load. Samples load value from normal Gaussian distribution (GD) using given
        mean and standard deviation
        :param load_mean: Mean of the GD the load is sampled from
        :param load_std: Standard deviation of the GD the load is sampled from
        :param balanced: If True: all 3 phases are takes as equal; if False symmetrical load is applied
        :param tolerance: Device manufacturing tolerance (with reference to the mean value)
        """

        self.balanced = balanced
        self.load_mean = load_mean
        self.tolerance = tolerance
        if load_std is None:
            self.load_std = self.load_mean * 0.1
        else:
            self.load_std = load_std
        self.gains = np.clip(
            [np.random.normal(self.load_mean, self.load_std) for _ in range(1 if self.balanced else 3)],
            (self.load_mean - self.load_mean * self.tolerance),
            (self.load_mean + self.load_mean * self.tolerance)).tolist()

    def load_step(self, t, n: int):
        """
        Defines a load step after 0.2 s
        Doubles the load parameters
        :param t: t :D
        :param n: Index referring to the current phase
        :return: Dictionary with load parameters
        """

        if n > 2:
            raise ValueError('Choose between single or three phase!')

        if len(self.gains) == 1:
            return 1 * self.gains[0] if (t < .05 + 0.023  or t > 0.1 + 0.023) else 0.55 * self.gains[0]
        else:
            return 1 * self.gains[n] if (t < .05 + 0.023  or t > 0.1 + 0.023) else 0.55 * self.gains[n]

    def give_value(self, t, n: int):
        """
        Defines a load step after 0.2 s
        Doubles the load parameters
        :param t: t :D
        :param n: Index referring to the current phase
        :return: Dictionary with load parameters
        """

        if n > 2:
            raise ValueError('Choose between single or three phase!')

        if len(self.gains) == 1:
            return self.gains[0]
        else:
            return self.gains[n]

    def reset(self):
        self.gains = np.clip(
            [np.random.normal(self.load_mean, self.load_std) for _ in range(1 if self.balanced else 3)],
            (self.load_mean - self.load_mean * self.tolerance),
            (self.load_mean + self.load_mean * self.tolerance)).tolist()


