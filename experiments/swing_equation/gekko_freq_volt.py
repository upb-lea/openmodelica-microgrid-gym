from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import csv
#Initialize Model
m = GEKKO(remote=False)

#define parameter
Pdroop = 8000
Qdroop = 2000
t_end = 1
steps = 1000
nomFreq = 50  # grid frequency / Hz
nomVolt = value=230
omega = 2*np.pi*nomFreq

J = 0.0005
J_Q = 0.00005

R_lv_line_10km = 0.0
L_lv_line_10km = 0.000589
B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)

step = np.zeros(steps)
step[0:499] = 6.22
step[500:]  = 6.22/2

step_l = np.zeros(steps)
step_l[0:499] = 0.00495 # in Henry
step_l[500:]  = 0.00495/2 # in Henry
#step_l[0:499] = 3.5267/omega # in Henry
#step_l[500:]  = 3.5267/(omega*2) # in Henry


R_load = m.Param(value=step)
L_load = m.Param(value=step_l)
G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km+0, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km+B_RL_load]])

#B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
#              [-B_L_lv_line_10km, 2*B_L_lv_line_10km+0, -B_L_lv_line_10km],
#              [-B_L_lv_line_10km, -B_L_lv_line_10km, -10.8463]])

G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])
#G = np.array([[0, 0, 0],
#                   [0, 0, 0],
#                   [0, 0, 0.1512]])

#constants

p_offset = [00, 00, 0]
q_offset = [50, 50, 0]

#variables

u1 = m.Var(value=10)
u2 = m.Var(value=10)
u3 = m.Var(value=10)
P1 = m.Var(value=0)
P2 = m.Var(value=0)
P3 = m.Var(value=0)
Q1 = m.Var(value=0)
Q2 = m.Var(value=0)
Q3 = m.Var(value=0)
w1 = m.Var(value=2)
w2 = m.Var(value=2)
w3 = m.Var(value=2)
theta1 = m.Var(value=1)
theta2 = m.Var(value=1)
theta3 = m.Var(value=1)
#theta1, theta2, theta3 = [m.Var() for i in range(3)]

#initialize variables

droop_linear=[Pdroop,Pdroop,0]
q_droop_linear=[Qdroop,Qdroop,0]
#initial values

theta1.value = 0
theta2.value = 0
theta3.value = 0
#w1.value = 50
#w2.value = 50
#w3.value = 50

    #Equations

#constraints

#m.Equation(Q3 == 0)

m.Equation(u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P1)
m.Equation(u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P2)
m.Equation(u3 * u1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
           u3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
           u3 * u3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P3)

m.Equation(u1 * u1 * (G[0][0] * m.sin(theta1 - theta1) - B[0][0] * m.cos(theta1 - theta1)) + \
           u1 * u2 * (G[0][1] * m.sin(theta1 - theta2) - B[0][1] * m.cos(theta1 - theta2)) + \
           u1 * u3 * (G[0][2] * m.sin(theta1 - theta3) - B[0][2] * m.cos(theta1 - theta3)) == Q1)
m.Equation(u2 * u1 * (G[1][0] * m.sin(theta2 - theta1) - B[1][0] * m.cos(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.sin(theta2 - theta2) - B[1][1] * m.cos(theta2 - theta2)) + \
           u2 * u3 * (G[1][2] * m.sin(theta2 - theta3) - B[1][2] * m.cos(theta2 - theta3)) == Q2)
m.Equation(u3 * u1 * (G[2][0] * m.sin(theta3 - theta1) - B[2][0] * m.cos(theta3 - theta1)) + \
           u3 * u2 * (G[2][1] * m.sin(theta3 - theta2) - B[2][1] * m.cos(theta3 - theta2)) + \
           u3 * u3 * (G[2][2] * m.sin(theta3 - theta3) - B[2][2] * m.cos(theta3 - theta3)) == Q3)

# Equations

# define omega
m.Equation(theta1.dt()==w1)
m.Equation(theta2.dt()==w2)
m.Equation(theta3.dt()==w3)

#Power ODE

#m.Equation(J*w1*w1.dt()==(u1 * u1 * -(G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#           u1 * u2 * -(G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#           u1 * u3 * -(G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3))))
#m.Equation(J*w2*w3.dt()==(u2 * u1 * -(G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#           u2 * u2 * -(G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#           u2 * u3 * -(G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3))))
#m.Equation(J*w3*w3.dt()==(u3 * u1 * -(G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#           u3 * u2 * -(G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#           u3 * u3 * -(G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3))))

m.Equation(w1.dt()==((-P1+p_offset[0])-(droop_linear[0]*(w1-omega)))/(J*w1))
m.Equation(w2.dt()==((-P2+p_offset[1])-(droop_linear[1]*(w2-omega)))/(J*w2))
m.Equation(w3.dt()==((-P3+p_offset[2])-(droop_linear[2]*(w3-omega)))/(J*w3))

#m.Equation(w1.dt()==((-P1+p_offset[0])-(droop_linear[0]*(w1-nomFreq)))/(J*w1))
#m.Equation(w2.dt()==((-P2+p_offset[1])-(droop_linear[1]*(w2-nomFreq)))/(J*w2))
#m.Equation(w3.dt()==((-P3+p_offset[2])-(droop_linear[2]*(w3-nomFreq)))/(J*w3))
#m.Equation(P3==0)
#Q_ODE


#m.Equation(J_Q*w1*w1.dt()==(u1 * u1 * -(G[0][0] * m.sin(theta1 - theta1) + B[0][0] * m.cos(theta1 - theta1)) + \
#           u1 * u2 * -(G[0][1] * m.sin(theta1 - theta2) + B[0][1] * m.cos(theta1 - theta2)) + \
#           u1 * u3 * -(G[0][2] * m.sin(theta1 - theta3) + B[0][2] * m.cos(theta1 - theta3))))
#m.Equation(J_Q*w2*w3.dt()==(u2 * u1 * -(G[1][0] * m.sin(theta2 - theta1) + B[1][0] * m.cos(theta2 - theta1)) + \
#           u2 * u2 * -(G[1][1] * m.sin(theta2 - theta2) + B[1][1] * m.cos(theta2 - theta2)) + \
#           u2 * u3 * -(G[1][2] * m.sin(theta2 - theta3) + B[1][2] * m.cos(theta2 - theta3))))
#m.Equation(J_Q*w3*w3.dt()==(u3 * u1 * -(G[2][0] * m.sin(theta3 - theta1) + B[2][0] * m.cos(theta3 - theta1)) + \
#           u3 * u2 * -(G[2][1] * m.sin(theta3 - theta2) + B[2][1] * m.cos(theta3 - theta2)) + \
#           u3 * u3 * -(G[2][2] * m.sin(theta3 - theta3) + B[2][2] * m.cos(theta3 - theta3))))



m.Equation(u1.dt()==((-Q1+q_offset[0])-(q_droop_linear[0]*(u1-nomVolt)))/(J_Q*u1))
m.Equation(u2.dt()==((-Q2+q_offset[1])-(q_droop_linear[1]*(u2-nomVolt)))/(J_Q*u2))
m.Equation(u3.dt()==((-Q3+q_offset[2])-(q_droop_linear[2]*(u3-nomVolt)))/(J_Q*u3))
#m.Equation(u3.dt()==0)
#m.Equation(Q3==0)
#m.Equation(J_Q*u1*u1.dt()==(-Q1))
#m.Equation(J_Q*u2*u2.dt()==(-Q2))
#m.Equation(J_Q*u3*u3.dt()==(-Q3))


#Set global options
m.options.IMODE = 7
#m.options.RTOL = 1.0e-9
#Set number of iterations
#m.options.MAX_ITER = 1
#steady state optimization
#m.options.NODES = 1

m.time = np.linspace(0,t_end,steps) # time points


#Solve simulation
m.solve()

#print(B)
#print(G)

#Results

f1 = np.divide(w1,(2*np.pi))
f2 = np.divide(w2,(2*np.pi))
f3 = np.divide(w3,(2*np.pi))

print(f1[400])

plt.plot(m.time,f1,'r')
plt.plot(m.time,f2,'b')
plt.plot(m.time,f3,'--g')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(49.5, 50.5)
plt.show()

plt.plot(m.time,theta1,'r')
plt.plot(m.time,theta2,'b')
plt.plot(m.time,theta3,'--g')
plt.xlabel('Time (s)')
plt.ylabel('Theta')
plt.show()

plt.plot(m.time,u1, 'r')
plt.plot(m.time,u2, 'b')
plt.plot(m.time,u3,'--g')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.ylim(220, 240)
plt.legend()
plt.show()


plt.plot(m.time,P1, 'r', label='P1')
plt.plot(m.time,P2, '--b', label='P2')
plt.plot(m.time,P3, 'g', label='P3')
plt.xlabel('Time (s)')
plt.ylabel('Active Power (W)')
plt.legend()
plt.show()


plt.plot(m.time,Q1,'r', label='Q1')
plt.plot(m.time,Q2,'--b', label='Q2')
plt.plot(m.time,Q3,'g', label='Q3')
plt.xlabel('Time (s)')
plt.ylabel('Reactive Power (VAr)')
#plt.ylim(-100, 100)
plt.legend()
plt.show()

a = w1
np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")

np.savetxt("PyCharm_value_rev1.csv", np.column_stack((m.time, f1, f2, f3, u1, u2, u3, P1, P2, P3, Q1, Q2, Q3)), delimiter=",", fmt='%s') #, header=header