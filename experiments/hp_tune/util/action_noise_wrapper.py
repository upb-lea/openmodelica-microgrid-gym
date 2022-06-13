from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np


class myOrnsteinUhlenbeckActionNoise(OrnsteinUhlenbeckActionNoise):
    """
    Wraps the OU-noise from sb3 to give the possibility to reduce the action noise over training time
    Implementation similar to kerasRL2 (https://github.com/wau/keras-rl2/blob/master/rl/random.py)
    """

    def __init__(self, n_steps_annealing=1000, sigma_min=None, *args, **kwargs):
        super(myOrnsteinUhlenbeckActionNoise, self).__init__(*args, **kwargs)
        self.n_steps_annealing = n_steps_annealing
        self.sigma_min = sigma_min
        self.n_steps = 0

        if sigma_min is not None:
            # self.m = -float(self._sigma - sigma_min) / float(n_steps_annealing)
            self.m = -(self._sigma - sigma_min) / (n_steps_annealing)
            self.c = self._sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = self._sigma
            self.sigma_min = self._sigma

    @property
    def current_sigma(self):
        sigma = np.maximum(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

    def __call__(self) -> np.ndarray:
        noise = (
                self.noise_prev
                + self._theta * (self._mu - self.noise_prev) * self._dt
                + self.current_sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        self.n_steps += 1
        return noise

    def reset(self) -> None:
        super().reset()

        # should not be reset because action_noise is reset after episode, but noise reduction over learning-length
        # does not reset the noise reduction! Reduction not per episode but per learing, since action noise
        # is redifiend then, no reset of annealing needed
        # self.n_steps = 0
