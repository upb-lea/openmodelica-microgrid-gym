from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
#Initialize Model
m = GEKKO(remote=False)

#define parameter
Pdroop = 4000
Qdroop = 500
t_end = 0.4
steps = 400
nomFreq = 50  # grid frequency / Hz
nomVolt = value=230
omega = 2*np.pi*nomFreq

J = 0.05
J_Q = 0.0005

R_lv_line_10km = 0.0
L_lv_line_10km = 0.00083/3
B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)

step = np.zeros(steps)
step[0:200] = 10/3
step[200:]  = 20/3

step_l = np.zeros(steps)
step_l[0:250] = 0.0
step_l[250:]  = 0

R_load = m.Param(value=step)
L_load = m.Param(value=step_l)
G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km+0, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km+B_RL_load]])

G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])

#constants

p_offset = [100, 100, 0]
q_offset = [20, 20, 0]

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
w1 = m.Var(value=50)
w2 = m.Var(value=50)
w3 = m.Var(value=50)
theta1, theta2, theta3 = [m.Var() for i in range(3)]
#initialize variables

droop_linear=[-Pdroop,-Pdroop,0]
q_droop_linear=[-Qdroop,-Qdroop,0]
#initial values

theta1.value = 0
theta2.value = 0
theta3.value = 0
#w1.value = 50
#w2.value = 50
#w3.value = 50

    #Equations

#constraints

m.Equation(u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P1)
m.Equation(u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P2)
m.Equation(u3 * u1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
           u3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
           u3 * u3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P3)

m.Equation(u1 * u1 * (G[0][0] * m.sin(theta1 - theta1) + B[0][0] * m.cos(theta1 - theta1)) + \
           u1 * u2 * (G[0][1] * m.sin(theta1 - theta2) + B[0][1] * m.cos(theta1 - theta2)) + \
           u1 * u3 * (G[0][2] * m.sin(theta1 - theta3) + B[0][2] * m.cos(theta1 - theta3)) == Q1)
m.Equation(u2 * u1 * (G[1][0] * m.sin(theta2 - theta1) + B[1][0] * m.cos(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.sin(theta2 - theta2) + B[1][1] * m.cos(theta2 - theta2)) + \
           u2 * u3 * (G[1][2] * m.sin(theta2 - theta3) + B[1][2] * m.cos(theta2 - theta3)) == Q2)
m.Equation(u3 * u1 * (G[2][0] * m.sin(theta3 - theta1) + B[2][0] * m.cos(theta3 - theta1)) + \
           u3 * u2 * (G[2][1] * m.sin(theta3 - theta2) + B[2][1] * m.cos(theta3 - theta2)) + \
           u3 * u3 * (G[2][2] * m.sin(theta3 - theta3) + B[2][2] * m.cos(theta3 - theta3)) == Q3)

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

m.Equation(w1.dt()==((-P1+p_offset[0])+(droop_linear[0]*(w1-nomFreq)))/(J*w1))
m.Equation(w2.dt()==((-P2+p_offset[1])+(droop_linear[1]*(w2-nomFreq)))/(J*w2))
m.Equation(w3.dt()==((-P3+p_offset[2])+(droop_linear[2]*(w3-nomFreq)))/(J*w3))

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



m.Equation(u1.dt()==((Q1+q_offset[0])+(q_droop_linear[0]*(u1-nomVolt)))/(J_Q*u1))
m.Equation(u2.dt()==((Q2+q_offset[1])+(q_droop_linear[1]*(u2-nomVolt)))/(J_Q*u2))
m.Equation(u3.dt()==((Q3+q_offset[2])+(q_droop_linear[2]*(u3-nomVolt)))/(J_Q*u3))

#m.Equation(J_Q*u1*u1.dt()==(-Q1))
#m.Equation(J_Q*u2*u2.dt()==(-Q2))
#m.Equation(J_Q*u3*u3.dt()==(-Q3))


#Set global options
m.options.IMODE = 7
#steady state optimization

m.time = np.linspace(0,t_end,steps) # time points



#Solve simulation
m.solve()

#Results

plt.plot(m.time,w1)
plt.xlabel('time')
plt.ylabel('w1(t)')


plt.plot(m.time,w2)
plt.xlabel('time')
plt.ylabel('w2(t)')


plt.plot(m.time,w3,'--')
plt.xlabel('time')
plt.ylabel('w3(t)')
#plt.ylim(48, 52)
plt.show()



plt.plot(m.time,u1, 'b')
plt.xlabel('time')
plt.ylabel('u1(t)')


plt.plot(m.time,u2, 'r')
plt.xlabel('time')
plt.ylabel('u2(t)')


plt.plot(m.time,u3,'--g')
plt.xlabel('time')
plt.ylabel('u3(t)')
plt.ylim(200, 240)
plt.show()




plt.plot(m.time,P1, 'b', label='P1')
plt.plot(m.time,P2, '--r', label='P2')
#plt.plot(m.time,P3, 'g')
plt.xlabel('time')
plt.ylabel('P(t)')
plt.legend()
plt.show()


plt.plot(m.time,Q1,'b')
plt.plot(m.time,Q2,'r')
plt.plot(m.time,Q3,'g')
plt.xlabel('time')
plt.ylabel('Q(t)')
plt.ylim(-100, 100)
plt.show()


plt.plot(m.time,(np.array(u1.value)-np.array(u2.value)))
#plt.plot(m.time,(np.array(theta1.value)-np.array(theta3.value)))
#plt.plot(m.time,(np.array(theta2.value)-np.array(theta3.value)))
#plt.legend()
plt.xlabel('time')
plt.ylabel('diff_u(t)')
plt.show()

a = w1
np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")