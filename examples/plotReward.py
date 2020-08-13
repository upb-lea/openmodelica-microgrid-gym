import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

rew = pd.read_pickle('rewardMatrix')

delta_t = 0.5e-4  # simulation time step size / s

if __name__ == '__main__':

    kP = rew[0][1]
    kI = rew[0][2]
    reward = rew[0][0]
    # reward[0][0] = -1616 # correktion for sim - initial kp/i have to be adjusted
    xx, yy = np.meshgrid(kI, kP, sparse=True)


    #alles größer 500 auf 500 setzen??
    reward[reward<-550] = -500

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(xx, yy, reward, cmap=cm.viridis)
    #ax.contour(kI, kP, reward)
    # ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(50, 30)
    ax.set_xlabel(r'$K_\mathrm{i}\,/\,\mathrm{(A^{-1}s^{-1})}$')
    ax.set_ylabel(r'$K_\mathrm{p}\,/\,\mathrm{(A^{-1})}$')
    ax.set_zlabel(r'$i_\mathrm{RME+Barrier}$')
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
