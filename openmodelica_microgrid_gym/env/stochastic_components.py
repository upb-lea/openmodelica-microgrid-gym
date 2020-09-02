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
            return 1 * self.gains[0] if t < .05 else 1 * self.gains[0]
        else:
            return 1 * self.gains[n] if t < .05 else 1 * self.gains[n]

    def reset(self):
        self.gains = np.clip(
            [np.random.normal(self.load_mean, self.load_std) for _ in range(1 if self.balanced else 3)],
            (self.load_mean - self.load_mean * self.tolerance),
            (self.load_mean + self.load_mean * self.tolerance)).tolist()



class Noise:

    def __init__(self, noise_mean, noise_std, std_min, std_max):
        """
        Load class which defines stochastic load. Samples load value from normal Gaussian distribution (GD) using given
        mean and standard deviation
        :param load_mean: Mean of the GD the load is sampled from
        :param load_std: Standard deviation of the GD the load is sampled from
        :param balanced: If True: all 3 phases are takes as equal; if False symmetrical load is applied
        :param tolerance: Device manufacturing tolerance (with reference to the mean value)
        """

        self.noise_mean = noise_mean
        self.std_min = std_min
        self.std_max = std_max
        if noise_std is None:
            if any(x == 0 for x in self.noise_mean):
                self.noise_std = 0.1
            else:
                self.noise_std = self.noise_mean * 0.1
        else:
            self.noise_std = noise_std
        #self.gains = np.clip(
         #   [np.random.normal(self.noise_mean[n], self.noise_std[n]) for n in range(len(self.noise_std))],
          #  (self.noise_mean - self.noise_mean * self.tolerance),
           # (self.noise_mean + self.noise_mean * self.tolerance)).tolist()
        self.gains = [np.clip(np.random.normal(self.noise_mean[n], self.noise_std[n]),
                              self.std_min,
                              self.std_max).tolist()
                      for n in range(len(self.noise_std))]


    def reset(self):
        self.gains = [np.clip(np.random.normal(self.noise_mean[n], self.noise_std[n]),
                              self.std_min,
                              self.std_max).tolist()
                      for n in range(len(self.noise_std))]

