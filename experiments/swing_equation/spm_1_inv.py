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

R_load2 = 0.02
R_load = 0.01
R_lv_line_10km = 0 #
#R_lv_line_10km = 31
L_load = 0
L_lv_line_10km = 0.00083*10     # nach MG book chapter 5, table 5.1
#L_lv_line_10km = 0
#B_L_lv_line_10km = -1/(omega*L_lv_line_10km)

#QUESTION: Assume fixed omega, or use the real frequencies?
B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)
G_L_lv_line_10km = R_lv_line_10km/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)

G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = (omega * L_load)/(R_load**2 + (omega * L_load)**2)
G_RL_load2 = 0

B_load = 0#1/0.001

B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km+0, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km+B_RL_load]])  # Susceptance matrix

G = np.array([[2*G_L_lv_line_10km, G_L_lv_line_10km, G_L_lv_line_10km],
                   [G_L_lv_line_10km, 2*G_L_lv_line_10km+0, G_L_lv_line_10km],
                   [G_L_lv_line_10km, G_L_lv_line_10km, 2*G_L_lv_line_10km+G_RL_load]])
print(B)
print(G)
P_offset = np.array([10000, 0, 0])
Q_offset = np.array([0, 0, 0])


droop_linear = np.array([1000, 0, 0])     # W/Hz
q_droop_linear = np.array([100, 0, 0])


def env_model_ode(t, y):#, arg):

    # y = array([theta1, theta2,  f1,     f2])

    thetas = y[0:3]#len(y)/2]
    freqs = y[3:6]
    voltages = y[6:]
    # theta in rad! correct? Yes
 #   print(voltages)
    num_nodes = len(freqs)

    p = np.zeros(num_nodes)
    q = np.zeros(num_nodes)

    for k in range(num_nodes):
        for j in range(num_nodes):  # l works, due to B is symmetric
            # Assume Voltage as constant
            p[k] += voltages[k] * voltages[j] * (G[k][j]*np.cos(thetas[k] - thetas[j]) + \
                                         B[k][j]*np.sin(thetas[k] - thetas[j]))
            q[k] += voltages[k] * voltages[j] * (G[k][j]*np.sin(thetas[k] - thetas[j]) + \
                                         B[k][j]*np.cos(thetas[k] - thetas[j]))

            #print('Pk = {} for k,j = {}{}'.format(p[k],k,j))
   # q = np.zeros(3)
 #   p_tr[1] = voltages[1] * cos(thetas[1]) -* (-G[k][j]*np.cos(thetas[k] - thetas[j]) + \
 #                                        B[k][j]*np.sin(thetas[k] - thetas[j]))
    p = p+P_offset
    q = q+Q_offset

    J= [2, 2, 2]
    J_voltage = [20, 20, 20]
    df = (p-droop_linear*(freqs-nomFreq))/(J*freqs)
    dv = (q + q_droop_linear * (voltages - nomVolt)) / (J_voltage * voltages)
    dtheta = freqs * 2 * np.pi

    # d theta_k / dt = f_k
    # d f_k / dt = sum(P_k)/J*f

    # @ DANIEL: voltage or dv (calculated 4 lines above) as the return? I would have said dv,
    # I would take dv, but this causes calculation issues
 #   return np.array([dtheta[0], dtheta[1], dtheta[2], df[0], df[1], df[2], dv[0], dv[1], dv[2]])
    return np.array([dtheta[0], dtheta[1], dtheta[2], df[0], df[1], df[2], dv[0], dv[1], dv[2]])

if __name__ == '__main__':

    f = nomFreq
    voltage1_0 = 230
    voltage2_0 = 15
    voltage3_0 = 10

    theta1_0 = 0
    theta2_0 = 0
    theta3_0 = 0
    t0 = 0

    x = np.array([theta1_0, theta2_0, theta3_0, nomFreq, nomFreq, nomFreq, voltage1_0, voltage2_0, voltage3_0])


    f_list = []
    u_list = []

    ode_solver = ode(env_model_ode)

    ode_solver.set_initial_value(x, t0)#.set_f_params(2.0)



    count = 0

    result = np.zeros([int(max_episode_steps)+1,9])
    theta = np.zeros([int(max_episode_steps)+1,3])
    freq = np.zeros([int(max_episode_steps)+1,3])
    volt = np.zeros([int(max_episode_steps) + 1, 3])
#    p_transfer = np.zeros([int(max_episode_steps) + 1, 3])
   # q_transfer = np.zeros([int(max_episode_steps) + 1, 3])



    while ode_solver.successful() and ode_solver.t < max_episode_steps*delta_t:

        if ode_solver.t > (max_episode_steps*delta_t)-1*delta_t:
            asd = 1
        #print(count)
        #f_list.append(ode_solver.integrate(ode_solver.t+delta_t))
        result[count] = ode_solver.integrate(ode_solver.t+delta_t)
        #result[count] = ode_solver.integrate(t[count])
        theta[count] = result[count][0:3] #% (2*np.pi)
        freq[count] = result[count][3:6]
        volt[count] = result[count][6:]
  #      p_transfer[count] = result[count][9:12]
  #      q_transfer[count] = result[count][12:15]

        count+=1

    plt.plot(t,theta[:,0], label = 'theta1')
    plt.plot(t, theta[:,1], label = 'theta2')
    plt.plot(t, theta[:,2], label = 'theta3')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$theta_{\mathrm{k}}\,/\,\mathrm{Hz}$')
    #plt.title('{}'.format())
    plt.legend()
    plt.grid()
    #plt.xlim([1.21,1.351])
    #plt.ylim([49.25,50.1])
    plt.show()



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

    plt.plot(t, volt[:,0], label = 'v1')
    plt.plot(t, volt[:,1], label = 'v2')
    plt.plot(t, volt[:,2], label = 'v3')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$Spannung_{\mathrm{k}}\,/\,\mathrm{V}$')
    #plt.title('{}'.format())
    plt.legend()
    plt.grid()
    #plt.xlim([1.21,1.351])
    #plt.ylim([0,25])
    plt.show()
