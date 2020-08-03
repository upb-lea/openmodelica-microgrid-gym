from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

from openmodelica_microgrid_gym.aux_ctl import DroopParams, InverseDroopParams, DroopController, InverseDroopController

sim_time = 5#5e-3#30e-3  # /s

delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = sim_time/delta_t  # number of simulation steps per episode

t = np.linspace(0, sim_time, int(max_episode_steps)+1, endpoint=False)

num_episodes = 1  # number of simulation episodes

nomFreq = 50  # grid frequency / Hz
nomVolt = 230
nomVoltPeak = nomVolt * 1.414  # nominal grid voltage / V
#DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
#QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V

omega = 2*np.pi*nomFreq

R_load = 100
L_load = 0.001
L_lv_line_10km = 0.083*10/(omega)      # nach MG book chapter 5, table 5.1

B_L_lv_line_10km = -1/(omega*L_lv_line_10km)

G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B_load = 0#1/0.001
B12 = 1/0.01
B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km+B_RL_load]])  # Susceptance matrix

G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])

P_offset = np.array([0, 0, 0])
droop_linear = np.array([10000, 1000, 0])     # W/Hz

def env_model_ode(t, y):#, arg):

    # y = array([theta1, theta2,  f1,     f2])

    thetas = y[0:3]#len(y)/2]
    freqs = y[3:]
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

    J= [2, 2, 2]

    df = (p-droop_linear*(freqs-nomFreq))/(J*freqs)
    dtheta = freqs * 2 * np.pi

    # d theta_k / dt = f_k
    # d f_k / dt = sum(P_k)/J*f
    return np.array([dtheta[0], dtheta[1], dtheta[2], df[0], df[1], df[2]])



if __name__ == '__main__':

    f = nomFreq
    theta1_0 = 0
    theta2_0 = 0
    theta3_0 = 0
    t0 = 0

    x = np.array([theta1_0, theta2_0, theta3_0, nomFreq, nomFreq, nomFreq])

    u = nomVoltPeak

    f_list = []
    u_list = []

    ode_solver = ode(env_model_ode)

    ode_solver.set_initial_value(x, t0)#.set_f_params(2.0)



    count = 0

    result = np.zeros([int(max_episode_steps)+1,6])
    theta = np.zeros([int(max_episode_steps)+1,3])
    freq = np.zeros([int(max_episode_steps)+1,3])


    while ode_solver.successful() and ode_solver.t < max_episode_steps*delta_t:

        if ode_solver.t > (max_episode_steps*delta_t)-1*delta_t:
            asd = 1
        #print(count)
        #f_list.append(ode_solver.integrate(ode_solver.t+delta_t))
        result[count] = ode_solver.integrate(ode_solver.t+delta_t)
        #result[count] = ode_solver.integrate(t[count])
        theta[count] = result[count][0:3]
        freq[count] = result[count][3:]


        count+=1

    print(result)
    #plt.plot(theta)
    #plt.show()


    plt.plot(t,freq[:,0], label = 'f1')
    plt.plot(t, freq[:,1], label = 'f2')
    plt.plot(t, freq[:,2], label = 'f3')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$f_{\mathrm{k}}\,/\,\mathrm{Hz}$')
    #plt.title('{}'.format())
    plt.legend()
    plt.grid()
    #plt.xlim([1.21,1.351])
    #plt.ylim([49.25,50.1])
    plt.show()



