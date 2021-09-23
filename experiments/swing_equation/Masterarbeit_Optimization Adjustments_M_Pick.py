from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Initialize Model empty model as m
m = GEKKO(remote=False)

#define parameter
Pdroop = 0.05
Qdroop = 0.3
t_end = 0.5
steps = 1001 # 1 more step than expected, starting and endpoint are included
nomFreq = 50  # Grid frequency / Hz
nomVolt = value = 6000 # Voltage / Volt
omega = 2*np.pi*nomFreq
tau = 0.005
e1=6600 #V
u2=6570 #V
P2_nom=509900 #Watt
P2_step=9500
P1_nom=(e1/u2)*P2_nom# Filter constant of pt1 filter for P and Q, inverse of cut-off frequency

# Inertia, Damping Factor, Line Parameters, Filters, Load steps
J = 0.2815 # Inertia / kgm²
D= 0.11  #[?, unfortunately given in per unit]
X_L_lv_line_1km=0.329 #Line inductance / Henry
B_L_lv_line_1km=-1/X_L_lv_line_1km # Line suszeptance / (1/Henry)
R_load=u2**2/P2_nom
R_load_step=u2**2/(P2_nom+P2_step)
Pbase=1000000 #MW


step = np.zeros(steps)
step[0:500] = R_load # Resistance load / Ohm
step[500:] = R_load_step# Resistance load / Ohm #calculated through R=U²/R

# Offsets for P and Q which you could use for a steady state error compensation, e.g. via an I-Controller (or guessing)
p_offset = [0, 00, 0]
q_offset = [0, 0, 0]

droop_linear = [Pdroop, Pdroop, 0]
q_droop_linear = [Qdroop, Qdroop, 0]

#variables/parameters
e1=m.Const(e1)
u2=m.Const(u2)
# e1 = m.Param(value=6600) # Output voltage DG / Volt
# u2 = m.Param(value=6570) # Load voltage / Volt
# P1 = m.Var(value=1000000) # DG Power / Watt
# P2 = m.Var(value=996600) #Load Power / Watt
P1 = m.Var(value=0)
P2= m.Var(value=0)
w1 = m.Var(value=2)
w2 = m.Var(value=2)
theta1 = m.Var(value=0)
theta2 = m.Var(value=0)
P1f = m.Var(value=0)
p1f = m.Var(value=0)



#Matrices for node admittance matrix
R_load_variable=m.Param(value=step)
G_load=1/R_load_variable
B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km],
              [-B_L_lv_line_1km, -B_L_lv_line_1km]])

G = np.array([[0, 0],
              [0, G_load]])

#Power flow equations
m.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) == P1)
m.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) == P2)

# define omega as the derivation of the phase angle
m.Equation(theta1.dt() == w1)
m.Equation(theta2.dt() == w2)


#PT1 Filtering of the powers
m.Equation(P1f.dt() == p1f)
m.Equation(P1f + tau * p1f == P1)

#Swing Equation
# m.Equation(w1.dt() == ((-P1f+p_offset[0])-(droop_linear[0]*(w1-omega)))/(J*w1))
# m.Equation(w2.dt() == ((-P2+p_offset[1])-(droop_linear[1]*(w2-omega)))/(J*w2))
#m.Equation(P1 == P0
m.Equation(P1 == P1_nom-(Pdroop*Pbase*((w1-omega)/omega)))
m.Equation((P1-P2) == J*w1*w1.dt()+D*((w1-omega)/omega))





#Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
m.options.IMODE = 7


m.time = np.linspace(0, t_end, steps) # time points


#Solve simulation
m.solve()

#print(B)
#print(G)

# Since P3 is mostly equal to 0 (balaned powerflows), the real transferred power can be calculated with the following lines:
#Calculate bus 3 Power
# Z_eff = np.sqrt(np.array(step)**2 + (np.array(step_l)*w3)**2)
# powerfactor = np.array(step) / Z_eff
# phi = np.arccos(powerfactor)
# S3_real = np.array(u3)**2/Z_eff #
# P3_real = S3_real * powerfactor
# Q3_real = np.sin(phi) * S3_real

#gekko writes its variables in a strange format, sometimes workarounds like this are neccesary, this only get the frequency from the omega

f1 = np.divide(w1, (2*np.pi))

print(f1)

plt.title('Frequency')
plt.plot(m.time, f1, 'r', label='f1')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(40, 60)
plt.legend()
plt.show()

plt.title('Phase Angle')
plt.plot(m.time, theta1, 'r', label='theta 1')
plt.plot(m.time, theta2, 'b', label='theta 2')
plt.xlabel('Time (s)')
plt.ylabel('Theta')
plt.legend()
plt.show()

# plt.title('Voltage')
# plt.plot(m.time, e1, 'r', label='V1')
# plt.plot(m.time, u2, 'b', label='V2')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.ylim(220, 230)
# plt.legend()
# plt.show()

plt.title('Active Power Flow')
plt.plot(m.time, P1, 'r', label='P1')
plt.plot(m.time, P2, '--b', label='P2')
plt.xlabel('Time (s)')
plt.ylabel('Active Power (W)')
plt.ylim(100000, 1000000)
plt.legend()
plt.show()












#np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")

#np.savetxt("PyCharm_value_rev1.csv", np.column_stack((m.time, f1, f2, f3, u1, u2, u3, P1, P2, P3, Q1, Q2, Q3)), delimiter=",", fmt='%s') #, header=header