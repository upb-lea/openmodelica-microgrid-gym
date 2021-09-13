from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
#   import csv
#   Initialize Model
m = GEKKO(remote=False)

#define parameter
Pdroop = 8000
Qdroop = 2000
t_end = 0.3
steps = 1000
nomFreq = 50  # grid frequency / Hz
nomVolt = value = 230
omega = 2*np.pi*nomFreq

L_lcl_11 = 0.002
L_lcl_12 = 0.002
L_lcl_21 = 0.002
L_lcl_22 = 0.002

C_lcl_1 = 0.00003
C_lcl_2 = 0.00003

#Filter Calculations

Zc1 = 1/(C_lcl_1*omega)
Zc2 = 1/(C_lcl_2*omega)
Zl11 = L_lcl_11*omega
Zl21 = L_lcl_21*omega

B_lcl1 = -(omega * L_lcl_12)/(0**2 + (omega*L_lcl_12)**2)
B_lcl2 = -(omega * L_lcl_22)/(0**2 + (omega*L_lcl_22)**2)

star_connection = False

J = 0.0005
J_Q = 0.00005

R_lv_line_10km = 0.0
#L_lv_line_10km = 0.000589
L_lv_line_10km = 0.0002

if star_connection:
    R_line_delta = 0    # (R_lv_line_10km * 6) / R_lv_line_10km
    X_line_star = L_lv_line_10km * omega
    X_line_delta = (X_line_star**2 + X_line_star**2 + X_line_star**2) / X_line_star     # Equation adapted due to singular value
    B_L_lv_line_10km = -X_line_delta / (R_line_delta**2 + X_line_delta**2)
    print(X_line_star)
    print(X_line_delta)
else:
    B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)


step = np.zeros(steps)
step[0:499] = 6.22
step[500:] = 6.22/2

step_l = np.zeros(steps)
step_l[0:499] = 0.00495    # in Henry
step_l[500:] = 0.00495/2    # in Henry


R_load = m.Param(value=step)
L_load = m.Param(value=step_l)
G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km + B_RL_load]])

#B = np.array([[2*B_L_lv_line_10km + B_lcl1, -B_L_lv_line_10km, -B_L_lv_line_10km],
#              [-B_L_lv_line_10km, 2*B_L_lv_line_10km + B_lcl2, -B_L_lv_line_10km],
#              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km + B_RL_load]])

#B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
#              [-B_L_lv_line_10km, 2*B_L_lv_line_10km+0, -B_L_lv_line_10km],
#              [-B_L_lv_line_10km, -B_L_lv_line_10km, -10.8463]])

G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])
#G = np.array([[0, 0, 0],
#                   [0, 0, 0],
#                   [0, 0, 0.1512]])
print(B)

#constants

p_offset = [00, 00, 0]
q_offset = [50, 50, 50]

#variables

e1 = m.Var(value=10)
e2 = m.Var(value=10)
e3 = m.Var(value=10)
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

droop_linear = [Pdroop, Pdroop, 0]
q_droop_linear = [Qdroop, Qdroop, 0]
#initial values

theta1.value = 0
theta2.value = 0
theta3.value = 0

    #Equations

    #Inverter

#m.Equation(e1 == u1/abs(Zc1/(Zc1+Zl11)))
#m.Equation(e2 == u2/abs(Zc2/(Zc2+Zl21)))
m.Equation(e1 == m.sqrt(u1**2 + ((L_lcl_11+L_lcl_12)*omega*(m.sqrt(P1**2 + Q1**2)/u1))**2))
m.Equation(e2 == m.sqrt(u2**2 + ((L_lcl_21+L_lcl_22)*omega*(m.sqrt(P2**2 + Q2**2)/u2))**2))
m.Equation(e3 == 0)


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
m.Equation(theta1.dt() == w1)
m.Equation(theta2.dt() == w2)
m.Equation(theta3.dt() == w3)

#Power ODE


m.Equation(w1.dt() == ((-P1+p_offset[0])-(droop_linear[0]*(w1-omega)))/(J*w1))
m.Equation(w2.dt() == ((-P2+p_offset[1])-(droop_linear[1]*(w2-omega)))/(J*w2))
m.Equation(w3.dt() == ((-P3+p_offset[2])-(droop_linear[2]*(w3-omega)))/(J*w3))


m.Equation(u1.dt() == ((-Q1+q_offset[0])-(q_droop_linear[0]*(u1-nomVolt)))/(J_Q*u1))
m.Equation(u2.dt() == ((-Q2+q_offset[1])-(q_droop_linear[1]*(u2-nomVolt)))/(J_Q*u2))
m.Equation(u3.dt() == ((-Q3+q_offset[2])-(q_droop_linear[2]*(u3-nomVolt)))/(J_Q*u3))
#m.Equation(u3.dt()==0)

#Set global options
m.options.IMODE = 7
#m.options.RTOL = 1.0e-9
#Set number of iterations
#m.options.MAX_ITER = 1
#steady state optimization
#m.options.NODES = 1

m.time = np.linspace(0, t_end, steps) # time points


#Solve simulation
m.solve()

#print(B)
#print(G)

#Results

f1 = np.divide(w1, (2*np.pi))
f2 = np.divide(w2, (2*np.pi))
f3 = np.divide(w3, (2*np.pi))

#Voltage drop

Vd_1 = np.subtract(e1,u1)
Vd_2 = np.subtract(e2,u2)

print(f1[400])
print(e1[400])
print(e1[500])
print(u1[400])
print(u1[400])
print(B)

plt.title('Frequency')
plt.plot(m.time, f1, 'r', label='f1')
plt.plot(m.time, f2, 'b', label='f2')
plt.plot(m.time, f3, '--g', label='f3')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(49.5, 50.2)
plt.legend()
plt.show()

plt.title('Phase Angle')
plt.plot(m.time, theta1, 'r', label='theta 1')
plt.plot(m.time, theta2, 'b', label='theta 2')
plt.plot(m.time, theta3, '--g', label='theta 3')
plt.xlabel('Time (s)')
plt.ylabel('Theta')
plt.legend()
plt.show()

plt.plot(m.time, u1, 'r', label='V1')
plt.plot(m.time, u2, 'b', label='V2')
plt.plot(m.time, u3, '--g', label='V3')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.ylim(220, 230)
plt.legend()
plt.show()

plt.title('Voltage')
plt.plot(m.time, e1, 'c', label='Inverter 1 Voltage')
plt.plot(m.time, e2, 'm', label='Inverter 2 Voltage')
plt.plot(m.time, u1, 'r', label='V1')
plt.plot(m.time, u2, 'b', label='V2')
plt.plot(m.time, u3, '--g', label='V3')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.ylim(200, 240)
plt.legend()
plt.show()

plt.title('Voltage drop over LCL filter')
plt.plot(m.time, Vd_1, 'r', label='V1')
plt.plot(m.time, Vd_2, 'b', label='V2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()

plt.title('Active Power Flow')
plt.plot(m.time, P1, 'r', label='P1')
plt.plot(m.time, P2, '--b', label='P2')
plt.plot(m.time, P3, 'g', label='P3')
plt.xlabel('Time (s)')
plt.ylabel('Active Power (W)')
plt.ylim(-100, 10000)
plt.legend()
plt.show()

plt.title('Reactive Power Flow')
plt.plot(m.time, Q1, 'r', label='Q1')
plt.plot(m.time, Q2, '--b', label='Q2')
plt.plot(m.time, Q3, 'g', label='Q3')
plt.xlabel('Time (s)')
plt.ylabel('Reactive Power (VAr)')
#plt.ylim(-100, 2000)
plt.legend()
plt.show()

a = w1
np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")

np.savetxt("PyCharm_value_rev1.csv", np.column_stack((m.time, f1, f2, f3, u1, u2, u3, P1, P2, P3, Q1, Q2, Q3)), delimiter=",", fmt='%s') #, header=header