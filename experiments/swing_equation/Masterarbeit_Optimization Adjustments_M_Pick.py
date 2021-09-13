from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Initialize Model empty model as m
m = GEKKO(remote=False)

#Load OMG Values, only for comparision between two models

B1_V = pd.read_pickle("B1_V.pkl")
B2_V = pd.read_pickle("B2_V.pkl")
B3_V = pd.read_pickle("B3_V.pkl")

B1_F = pd.read_pickle("B1_F.pkl")
B2_F = pd.read_pickle("B2_F.pkl")

B1_P = pd.read_pickle("B1_P.pkl")
B2_P = pd.read_pickle("B2_P.pkl")
B3_P = pd.read_pickle("B3_P.pkl")

B1_Q = pd.read_pickle("B1_Q.pkl")
B2_Q = pd.read_pickle("B2_Q.pkl")
B3_Q = pd.read_pickle("B3_Q.pkl")



#define parameter
Pdroop = 24000
Qdroop = 10000
t_end = 0.5
steps = 1001        # 1 more step than expected, starting and endpoint are included
nomFreq = 50  # grid frequency / Hz
nomVolt = value = 230 # Voltage / Volt
omega = 2*np.pi*nomFreq
tau = 0.0011 # Filter constant of pt1 filter for P and Q, inverse of cut-off frequency


# virtual inertias are only blind guesses, especially J might be way larger, e.g. something around 0.5, but not sure

J = 1.00000000e-04
J_Q = 1.000000e-05

R_lv_line_10km = 0.0 # this indicates that there is no ohmic loss within the lines
L_lv_line_10km = 0.0002 # line assumed to have 0.2 mH, check values with the "bible book"
#B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)
B_L_lv_line_10km = -1.52405526e+01 # only hardcoding, chose otherwise the line above for a proper calculation
print('B_LV ist')
print(B_L_lv_line_10km)


# LCL Filter Calculation, the bible only speaks of a L filter, probably need to reformulate the equations slightly, otherwise you might run in numerical issues

L_lcl_11 = 0.001
L_lcl_12 = 0.001
L_lcl_21 = 0.001
L_lcl_22 = 0.001

C_lcl_1 = 0.00001
C_lcl_2 = 0.00001





#Filter Calculations, only LC in this case, change depending on what you want
Zc1 = 1/(C_lcl_1*omega)
Zc2 = 1/(C_lcl_2*omega)
Zl11 = L_lcl_11*omega
Zl21 = L_lcl_21*omega


#hard-coded load-steps of R, L doesnt get changed

step = np.zeros(steps)
step[0:500] = 6.22
step[500:] = 6.22/2

step_l = np.zeros(steps)
step_l[0:500] = 0.00495    # in Henry
step_l[500:] = 0.00495    # in Henry


#Calculation of Admittance matrices B and G
# The real and complex part of the Y in Equation 3 in Olivers Pader

R_load = m.Param(value=step)
L_load = m.Param(value=step_l)
G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km + B_RL_load]])


G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])

print(B)

#constants


# Offsets for P and Q which you could use for a steady state error compensation, e.g. via an I-Controller (or guessing)
p_offset = [00, 00, 0]
q_offset = [0, 0, 0]

droop_linear = [Pdroop, Pdroop, 0]
q_droop_linear = [Qdroop, Qdroop, 0]

#variables

e1 = m.Var(value=10) #(Votlage BEHIND the inverter filter, just to calculate the voltage drop over the filter)
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
theta1 = m.Var(value=0)
theta2 = m.Var(value=0)
theta3 = m.Var(value=0)
P1f = m.Var(value=0) #
P2f = m.Var(value=0)
P3f = m.Var(value=0)
Q1f = m.Var(value=0)
Q2f = m.Var(value=0)
Q3f = m.Var(value=0)
p1f = m.Var(value=0)
p2f = m.Var(value=0)
p3f = m.Var(value=0)
q1f = m.Var(value=0)
q2f = m.Var(value=0)
q3f = m.Var(value=0)




    #Equations

    #calcualte the voltage at the inverter

#m.Equation(e1 == u1/abs(Zc1/(Zc1+Zl11)))
#m.Equation(e2 == u2/abs(Zc2/(Zc2+Zl21)))
m.Equation(e1 == m.sqrt(u1**2 + ((L_lcl_11+L_lcl_12)*omega*(m.sqrt(P1**2 + Q1**2)/u1))**2))
m.Equation(e2 == m.sqrt(u2**2 + ((L_lcl_21+L_lcl_22)*omega*(m.sqrt(P2**2 + Q2**2)/u2))**2))
m.Equation(e3 == 0)


# Equation 8 in Ollis paper, powerflow equations


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

# define omega as the derivation of the phase angle
m.Equation(theta1.dt() == w1)
m.Equation(theta2.dt() == w2)
m.Equation(theta3.dt() == w3)


#PT1 Filtering of the powers

m.Equation(P1f.dt() == p1f)
m.Equation(P2f.dt() == p2f)
m.Equation(P3f.dt() == p3f)

m.Equation(Q1f.dt() == q1f)
m.Equation(Q2f.dt() == q2f)
m.Equation(Q3f.dt() == q3f)

m.Equation(P1f + tau * p1f == P1)
m.Equation(P2f + tau * p2f == P2)
m.Equation(P3f + tau * p3f == P3)

m.Equation(Q1f + tau * q1f == Q1)
m.Equation(Q2f + tau * q2f == Q2)
m.Equation(Q3f + tau * q3f == Q3)

#Power ODE, inklusive control. Mixture of VSM and linear droop control, compare equation 7.1 from the bible. Damping is neglected, and droop is applied to an other part fo the eq. Here will be your biggest changes.

m.Equation(w1.dt() == ((-P1f+p_offset[0])-(droop_linear[0]*(w1-omega)))/(J*w1))
m.Equation(w2.dt() == ((-P2f+p_offset[1])-(droop_linear[1]*(w2-omega)))/(J*w2))
m.Equation(w3.dt() == ((-P3f+p_offset[2])-(droop_linear[2]*(w3-omega)))/(J*w3))


m.Equation(u1.dt() == ((-Q1f+q_offset[0])-(q_droop_linear[0]*(u1-nomVolt)))/(J_Q*u1))
m.Equation(u2.dt() == ((-Q2f+q_offset[1])-(q_droop_linear[1]*(u2-nomVolt)))/(J_Q*u2))
m.Equation(u3.dt() == ((-Q3f+q_offset[2])-(q_droop_linear[2]*(u3-nomVolt)))/(J_Q*u3))
#m.Equation(u3.dt()==0)
#here you assume to have some inertia and delay also at bus 3, where is no real pendant in reality. To be discussed if those equations also aqlso valid for bus 3 where is no real VSM


#Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
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

# Since P3 is mostly equal to 0 (balaned powerflows), the real transferred power can be calculated with the following lines:
#Calculate bus 3 Power
Z_eff = np.sqrt(np.array(step)**2 + (np.array(step_l)*w3)**2)
powerfactor = np.array(step) / Z_eff
phi = np.arccos(powerfactor)
S3_real = np.array(u3)**2/Z_eff
P3_real = S3_real * powerfactor
Q3_real = np.sin(phi) * S3_real

#gekko writes its variables in a strange format, sometimes workarounds like this are neccesary, this only get the frequency from the omega

f1 = np.divide(w1, (2*np.pi))
f2 = np.divide(w2, (2*np.pi))
f3 = np.divide(w3, (2*np.pi))

# Calculate reward function: the following part is onyl for an automated parameter tuning (see swing_eq_param_opti.py)

#Delta_V1_start = np.mean(abs(np.array(u1)-B1_V))
#Delta_V2_start = np.mean(abs(np.array(u2)-B2_V))
#Delta_V3_start = np.mean(abs(np.array(u3)-B3_V))
#Delta_F1_start = np.mean(abs(np.array(f1)-B1_F))
#Delta_F2_start = np.mean(abs(np.array(f2)-B2_F))
#Delta_P1_start = np.mean(abs(np.array(P1)+B1_P))
#Delta_P2_start = np.mean(abs(np.array(P2)+B2_P))
#Delta_P3_start = np.mean(abs(np.array(P3_real)-B3_P))

#print(Delta_V1_start)
#print(Delta_V2_start)
#print(Delta_V3_start)
#print(Delta_F1_start)
#print(Delta_F2_start)
#print(Delta_P1_start)
#print(Delta_P2_start)
#print(Delta_P3_start)
#optimization




#Voltage drop over the filter, just for plotting reasons

Vd_1 = np.subtract(e1,u1)
Vd_2 = np.subtract(e2,u2)




plt.title('Voltage drop over LCL filter')
plt.plot(m.time, e1, 'r', label='V1')
plt.plot(m.time, e2, 'b', label='V2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.ylim(225, 235)
plt.legend()
plt.show()

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
#plt.plot(m.time, e1, 'c', label='Inverter 1 Voltage')
#plt.plot(m.time, e2, 'm', label='Inverter 2 Voltage')
plt.plot(m.time, u1, 'r', label='V1')
plt.plot(m.time, u2, 'b', label='V2')
plt.plot(m.time, u3, '--g', label='V3')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.ylim(200, 240)
plt.legend()
plt.show()


plt.title('Voltage comparison')
#plt.plot(m.time, e1, 'c', label='Inverter 1 Voltage')
#plt.plot(m.time, e2, 'm', label='Inverter 2 Voltage')
plt.plot(m.time, u1, 'r', label='V1')
plt.plot(m.time, B1_V, 'b', label='V1_OMG')
plt.plot(m.time, u2, 'y', label='V2')
plt.plot(m.time, B2_V, label='V2_OMG')
plt.plot(m.time, u3, 'g', label='V3')
plt.plot(m.time, B3_V, label='V3_OMG')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.ylim(200, 240)
plt.legend()
plt.show()


plt.title('Frequency comparison')
plt.plot(m.time, f1, 'r', label='f1')
plt.plot(m.time, f2, 'b', label='f2')
plt.plot(m.time, f3, '--g', label='f3')
plt.plot(m.time, B1_F, label='f1_OMG')
plt.plot(m.time, B2_F, label='f2_OMG')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(46, 52)
plt.legend()
plt.show()


#plt.title('Voltage drop over LCL filter')
#plt.plot(m.time, Vd_1, 'r', label='V1')
#plt.plot(m.time, Vd_2, 'b', label='V2')
#plt.xlabel('Time (s)')
#plt.ylabel('Voltage (V)')
#plt.legend()
#plt.show()



#Z_eff = (np.array(step))
plt.title('Z_Eff')
plt.plot(m.time, Z_eff, 'r', label='Z')
plt.legend()
plt.show()

plt.title('Powerfactor')
plt.plot(m.time, powerfactor, 'r', label='Z')
plt.legend()
plt.show()

print(Z_eff)
print('powerfactor')
print(powerfactor)
print(S3_real)


plt.title('Bus 3 Powers')
plt.plot(m.time, S3_real , 'r', label='S3')
plt.plot(m.time, P3_real, '--b', label='P3')
plt.plot(m.time, Q3_real, 'g', label='Q3')
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

plt.title('Active Power Flow Comparison')
plt.plot(m.time, P1, 'r', label='P1')
plt.plot(m.time, P2, '--b', label='P2')
plt.plot(m.time, P3_real, 'g', label='P3')
plt.plot(m.time, -B1_P, label='P1_OMG')
plt.plot(m.time, -B2_P, label='P2_OMG')
plt.plot(m.time, B3_P, label='P3_OMG')
plt.xlabel('Time (s)')
plt.ylabel('Active Power (W)')
#plt.ylim(-100, 16000)
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


plt.title('Reactive Power Flow Comparison')
plt.plot(m.time, Q1, 'r', label='Q1')
plt.plot(m.time, Q2, '--b', label='Q2')
plt.plot(m.time, Q3_real, 'g', label='Q3')
plt.plot(m.time, B1_Q, label='Q1_OMG')
plt.plot(m.time, B2_Q, label='Q2_OMG')
plt.plot(m.time, B3_Q, label='Q3_OMG')
plt.xlabel('Time (s)')
plt.ylabel('Reactive Power (VAr)')
#plt.ylim(-100, 2000)
plt.legend()
plt.show()

#doubled, last one often does not get plotted

plt.title('Reactive Power Flow Comparison')
plt.plot(m.time, Q1, 'r', label='Q1')
plt.plot(m.time, Q2, '--b', label='Q2')
plt.plot(m.time, Q3_real, 'g', label='Q3')
plt.plot(m.time, B1_Q, label='Q1_OMG')
plt.plot(m.time, B2_Q, label='Q2_OMG')
plt.plot(m.time, B3_Q, label='Q3_OMG')
plt.xlabel('Time (s)')
plt.ylabel('Reactive Power (VAr)')
#plt.ylim(-100, 2000)
plt.legend()
plt.show()



#np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")

#np.savetxt("PyCharm_value_rev1.csv", np.column_stack((m.time, f1, f2, f3, u1, u2, u3, P1, P2, P3, Q1, Q2, Q3)), delimiter=",", fmt='%s') #, header=header