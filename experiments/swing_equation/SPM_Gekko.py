from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
#Initialize Model
m = GEKKO()

#define parameter

nomFreq = 50  # grid frequency / Hz
nomVolt = 230

u1 = nomVolt
u2 = nomVolt
u3 = nomVolt

omega = 2*np.pi*nomFreq

R_lv_line_10km = 0
L_lv_line_10km = 0.083*10

R_load = 100
L_load = 0.001

B_L_lv_line_10km = -1/(omega*L_lv_line_10km)
G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km+B_RL_load]])  # Susceptance matrix

G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])

P_offset = np.array([0, 0, 0])
droop_linear = np.array([10000, 1000, 1000])     # W/Hz


print(B)
print(G)

P1 = m.Var(value=0)
P2 = m.Var(value=0)
P3 = m.Var(value=0)
#Q1 = m.Param(value=100)
#Q2 = m.Param(value=100)
#Q3 = m.Param(value=-200)
#initialize variables
#P1, P2, P3 = [m.Var() for i in range(3)]
freq1, freq2, freq3 = [m.Var(lb=-1000, ub=1000) for i in range(3)]
theta1, theta2, theta3 = [m.Var(lb=0, ub = 100000) for i in range(3)]

#initial values
#P1.value = 230
#P2.value = 200
#P3.value = 300
freq1.value = 50
freq2.value = 50
freq3.value = 50
theta1.value = 0
theta2.value = 0
theta3.value = 0

    #Equations

#constraints

m.Equation(P1 == u1 * u1 * (-G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
           u1 * u2 * (-G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
           u1 * u3 * (-G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)))
m.Equation(P2 == u2 * u1 * (-G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
           u2 * u2 * (-G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
           u2 * u3 * (-G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)))
m.Equation(P3 == u3 * u1 * (-G[0][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
           u3 * u2 * (-G[0][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
           u3 * u3 * (-G[0][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)))

#Q1 = u1 * u1 * (G[0][0] * m.sin(theta1 - theta1) + B[0][0] * m.cos(theta1 - theta1)) + \
#           u1 * u2 * (G[0][1] * m.sin(theta1 - theta2) + B[0][1] * m.cos(theta1 - theta2)) + \
#           u1 * u3 * (G[0][2] * m.sin(theta1 - theta3) + B[0][2] * m.cos(theta1 - theta3))
#Q2 = u2 * u1 * (G[1][0] * m.sin(theta2 - theta1) + B[1][0] * m.cos(theta2 - theta1)) + \
#           u2 * u2 * (G[1][1] * m.sin(theta2 - theta2) + B[1][1] * m.cos(theta2 - theta2)) + \
#           u2 * u3 * (G[1][2] * m.sin(theta2 - theta3) + B[1][2] * m.cos(theta2 - theta3))
#Q3 = u3 * u1 * (G[0][0] * m.sin(theta3 - theta1) + B[2][0] * m.cos(theta3 - theta1)) + \
#           u3 * u2 * (G[0][1] * m.sin(theta3 - theta2) + B[2][1] * m.cos(theta3 - theta2)) + \
#           u3 * u3 * (G[0][2] * m.sin(theta3 - theta3) + B[2][2] * m.cos(theta3 - theta3))
       #    q[k] += voltages[k] * voltages[j] * (G[k][j]*np.sin(thetas[k] - thetas[j]) + \
       #                                  B[k][j]*np.cos(thetas[k] - thetas[j]))

# define omega
m.Equation(theta1.dt()==freq1)
m.Equation(theta2.dt()==freq2)
m.Equation(theta3.dt()==freq3)

#Power ODE

J = 2

m.Equation(freq1.dt()== (P1+droop_linear[0]*(freq1-nomFreq))/(J*freq1))
m.Equation(freq2.dt()== (P2+droop_linear[1]*(freq2-nomFreq))/(J*freq2))
m.Equation(freq3.dt()== (P3+droop_linear[2]*(freq3-nomFreq))/(J*freq3))

# m.Equation(freq1.dt()== ((u1 * u1 * (-G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#            u1 * u2 * (-G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#            u1 * u3 * (-G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)))+droop_linear[0]*(freq1-nomFreq))/(J*freq1))
# m.Equation(freq2.dt()== ((u2 * u1 * (-G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#            u2 * u2 * (-G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#            u2 * u3 * (-G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)))+droop_linear[1]*(freq2-nomFreq))/(J*freq2))
# m.Equation(freq3.dt()== ((u3 * u1 * (-G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#            u3 * u2 * (-G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#            u3 * u3 * (-G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)))+droop_linear[2]*(freq3-nomFreq))/(J*freq3))

#m.Equation(J*w1*w1.dt()==u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3))+ \
#           u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3))+  \
#           u3 * u1 * (G[0][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#           u3 * u2 * (G[0][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#           u3 * u3 * (G[0][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)))
#m.Equation(J*w2*w3.dt()==u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3))+ \
#           u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3))+  \
#           u3 * u1 * (G[0][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#           u3 * u2 * (G[0][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#           u3 * u3 * (G[0][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)))
#m.Equation(J*w3*w3.dt()==u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3))+ \
#           u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3))+  \
#           u3 * u1 * (G[0][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#           u3 * u2 * (G[0][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#           u3 * u3 * (G[0][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)))#


#m.Equation(x1*x2*x3*x4>=25)
#m.Equation(x1**2+x2**2+x3**2+x4**2==eq)


#Set global options
m.options.IMODE = 7
m.options.SOLVER = 1
m.time = np.linspace(0, 5, 50) # time points



#Solve simulation
m.solve()

#Results

plt.plot(m.time,freq1,'b')
plt.plot(m.time,freq2,'r')
plt.plot(m.time,freq3,'g')
plt.xlabel('time')
plt.ylabel('w1(t)')
plt.show()

plt.plot(m.time,P1,'b')
plt.plot(m.time,P2,'r')
plt.plot(m.time,P3,'g')
plt.xlabel('time')
plt.ylabel('w1(t)')
plt.show()


plt.plot(m.time,theta1)
plt.xlabel('time')
plt.ylabel('theta1(t)')
plt.show()