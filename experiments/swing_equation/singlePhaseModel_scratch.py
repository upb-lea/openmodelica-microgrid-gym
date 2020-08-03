from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

from openmodelica_microgrid_gym.aux_ctl import DroopParams, InverseDroopParams, DroopController, InverseDroopController

sim_time = 1.5#5e-3#30e-3  # /s

delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = sim_time/delta_t  # number of simulation steps per episode

t = np.linspace(0, sim_time, int(max_episode_steps)+1, endpoint=False)

num_episodes = 1  # number of simulation episodes

nomFreq = 50  # grid frequency / Hz
nomVolt = 230
nomVoltPeak = nomVolt * 1.414  # nominal grid voltage / V
#DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
#QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
R = 100
L = 0.001





B_load = 0#1/0.001
B12 = 1/0.01
B = np.array([[B_load+B12, -B12], [-B12, B12]])  # Susceptance matrix

G_load = 1/R
#G12 = 0


G = np.array([[G_load, 0], [0, 0]])

P_offset = np.array([0, 0])
droop_linear = np.array([1000, 1000])     # W/Hz

def env_model_ode(t, y):#, arg):

    # y = array([theta1, theta2,  f1,     f2])

    thetas = y[0:2]#len(y)/2]
    freqs = y[2:]
    # theta in rad! correct? Yes

    num_nodes = len(freqs)

    p = np.zeros(num_nodes)
    for k in range(num_nodes):
        for j in range(num_nodes):  # l works, due to B is symmetric
            # Assume Voltage as constant
            p[k] += nomVolt * nomVolt * (-G[k][j]*np.cos(thetas[k] - thetas[j]) + \
                                         B[k][j]*np.sin(thetas[k] - thetas[j]))

            #print('Pk = {} for k,j = {}{}'.format(p[k],k,j))

    p = p+P_offset

    J= [2, 2]

    df = (p-droop_linear*(freqs-nomFreq))/(J*freqs)
    dtheta = freqs * 2 * np.pi

    # d theta_k / dt = f_k
    # d f_k / dt = sum(P_k)/J*f
    return np.array([dtheta[0], dtheta[1], df[0], df[1]])



if __name__ == '__main__':

    f = nomFreq
    theta1_0 = 0
    theta2_0 = 0
    t0 = 0

    x = np.array([theta1_0, theta2_0, nomFreq, nomFreq])

    u = nomVoltPeak

    f_list = []
    u_list = []

    ode_solver = ode(env_model_ode)

    ode_solver.set_initial_value(x, t0)#.set_f_params(2.0)



    count = 0

    result = np.zeros([int(max_episode_steps)+1,4])
    theta = np.zeros([int(max_episode_steps)+1,2])
    freq = np.zeros([int(max_episode_steps)+1,2])


    while ode_solver.successful() and ode_solver.t < max_episode_steps*delta_t:

        if ode_solver.t > (max_episode_steps*delta_t)-1*delta_t:
            asd = 1
        #print(count)
        #f_list.append(ode_solver.integrate(ode_solver.t+delta_t))
        result[count] = ode_solver.integrate(ode_solver.t+delta_t)
        #result[count] = ode_solver.integrate(t[count])
        theta[count] = result[count][0:2]
        freq[count] = result[count][2:]


        count+=1

    print(result)
    #plt.plot(theta)
    #plt.show()


    plt.plot(t,freq[:,0], label = 'f1')
    plt.plot(t, freq[:,1], label = 'f2')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$f_{\mathrm{k}}\,/\,\mathrm{Hz}$')
    #plt.title('{}'.format())
    plt.legend()
    plt.grid()
    #plt.xlim([1.21,1.351])
    #plt.ylim([49.25,50.1])
    plt.show()



