from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import matplotlib.pyplot as plt

from experiments.hp_tune.util.action_noise_wrapper import myOrnsteinUhlenbeckActionNoise

noise_var = 2.
noise_theta = 25  # stiffness of OU

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(3), theta=noise_theta * np.ones(3),
                                            sigma=noise_var * np.ones(3), dt=1e-4)

action_noise = myOrnsteinUhlenbeckActionNoise(n_steps_annealing=1000, sigma_min=np.zeros(3), mean=np.zeros(3),
                                              theta=noise_theta * np.ones(3),
                                              sigma=noise_var * np.ones(3), dt=1e-4)

noise = np.zeros([3, 1000])
noise2 = np.zeros([3, 1000])
noise3 = np.zeros([3, 1000])

for i in range(1000):
    noise[:, i] = action_noise.__call__()

action_noise.reset()  # does not reset the noise reduction! Reduction not per episode but per learing, since action noise
# is redifiend then, no reset of annealing needed
for i in range(1000):
    noise2[:, i] = action_noise.__call__()

action_noise3 = myOrnsteinUhlenbeckActionNoise(n_steps_annealing=1000, sigma_min=np.zeros(3), mean=np.zeros(3),
                                               theta=noise_theta * np.ones(3),
                                               sigma=noise_var * np.ones(3), dt=1e-4)
for i in range(1000):
    noise3[:, i] = action_noise3.__call__()

plt.plot(noise[0, :])
plt.plot(noise[1, :])
plt.plot(noise[2, :])
plt.title(f'Stiffness theta = {noise_theta}, Varianz = {noise_var}')
plt.show()

plt.plot(noise2[0, :])
plt.plot(noise2[1, :])
plt.plot(noise2[2, :])
plt.title(f'Stiffness theta = {noise_theta}, Varianz = {noise_var}')
plt.show()

plt.plot(noise3[0, :])
plt.plot(noise3[1, :])
plt.plot(noise3[2, :])
plt.title(f'Stiffness theta = {noise_theta}, Varianz = {noise_var}')
plt.show()
