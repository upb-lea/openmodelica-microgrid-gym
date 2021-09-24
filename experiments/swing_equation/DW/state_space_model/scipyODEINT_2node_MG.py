from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# from util import config

ts = 1e-3
t_end = 0.1
steps = int(1 / ts)
f0 = 50
V_eff = 230 * np.sqrt(2)

R = 0.4
L = 2.3e-3
C = 10e-6
LT = 2.3e-3
RLoad = 14

R1 = R
L1 = L
C1 = C
R2 = R
L2 = L
C2 = C
LT1 = LT
LT2 = LT
RT1 = R
RT2 = R

t = np.linspace(0, t_end, steps)

num_episodes = 1  # number of simulation episodes


def env_model_ode(x, t):  # , arg):

    # y = array([i1, v1, iT1, i2, v2, iT2])
    i1 = x[0]
    v1 = x[1]
    iT1 = x[2]
    i2 = x[3]
    v2 = x[4]
    iT2 = x[5]

    # vi1 = V_eff * np.sin(2 * np.pi * f0 * t)
    # vi2 = V_eff * np.sin(2 * np.pi * f0 * t + 0.5)
    vi1 = 230
    vi2 = 230

    iLoad = iT1 + iT2

    di1 = (vi1 - v1) / L1 - R1 / L1 * i1
    dv1 = (i1 - iT1) / C1
    diT1 = v1 / LT1 - RT1 / LT1 * iT1 - RLoad / LT1 * iLoad

    di2 = (vi2 - v2) / L2 - R2 / L2 * i2
    dv2 = (i2 - iT2) / C2
    diT2 = v2 / LT2 - RT2 / LT2 * iT2 - RLoad / LT2 * iLoad

    return np.array([di1, dv1, diT1, di2, dv2, diT2])


if __name__ == '__main__':
    i10 = 0
    v10 = 0
    iT10 = 0
    i20 = 0
    v20 = 0
    iT20 = 0
    t0 = 0

    # x0 = np.array([i10, v10, iT10, i20, v20, iT20])
    x0 = [i10, v10, iT10, i20, v20, iT20]

    f_list = []
    u_list = []

    result = odeint(env_model_ode, x0, t)
    # ys = np.array(ys).flatten()

    plt.plot(t, result[:steps, 1], label='v1')
    # plt.plot(t,result[:steps,0], label = 'i1')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$v_{\mathrm{1}}\,/\,\mathrm{V}$')
    # plt.title('{}'.format())
    plt.legend()
    plt.grid()
    plt.xlim([0, 0.005])
    # plt.ylim([49.25,50.1])
    plt.show()
