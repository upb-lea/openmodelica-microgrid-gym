from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Initialize Model empty model as m
m = GEKKO(remote=False)

#define parameter
Pdroop = 0.05  #1000
Qdroop = 0.3
t_end = 0.5
steps = 1001 # 1 more step than expected, starting and endpoint are included
nomFreq = 50  # Grid frequency / Hz
nomVolt = value = 6000 # Voltage / Volt
omega = 2*np.pi*nomFreq
tau = 0.005 # Filter constant of pt1 filter for P and Q, inverse of cut-off frequency
e1=6600 #V
u2=6570 #V
P2_nom=509900 #Watt
Delta_P2_step=500000      #9500
P1_nom=(e1/u2)*P2_nom

# Inertia, Damping Factor, Line Parameters, Filters, Load steps
J = 0.2815 # Inertia / kgm²
D= 17  #[?, unfortunately given in per unit]
X_L_lv_line_1km=0.329 #Line inductance / Henry
B_L_lv_line_1km=-1/X_L_lv_line_1km # Line suszeptance / (1/Henry) #
R_load=u2**2/P2_nom
R_load_step=u2**2/(P2_nom + Delta_P2_step)
Pbase=1000000 #W


step_ohmic_load = np.zeros(steps)
step_ohmic_load[0:500] = R_load # Resistance load / Ohm
step_ohmic_load[500:] = R_load_step# Resistance load / Ohm #calculated through R=U²/R

step_load_P2= np.zeros(steps)
step_load_P2[0:500] = P2_nom
step_load_P2[500:]=P2_nom+Delta_P2_step

# Offsets for P and Q which you could use for a steady state error compensation, e.g. via an I-Controller (or guessing)
p_offset = [0, 00, 0]
q_offset = [0, 0, 0]

#variables/parameters
e1=m.Const(e1)
u2=m.Const(u2)
P_in = m.Var(value=0) #Input Voltage governor
P_out= m.Var(value=0) #Output Voltage behind filte
#P_2=m.Param(value=step_load_P2)
P_2=m.Var(value=0)
w1 = m.Var(value=2)
w2 = m.value(value=2)
theta1 = m.Var(value=0)
theta2 = m.Var(value=0)
# P1f = m.Var(value=0)
# p1f = m.Var(value=0)

#Matrices for node admittance matrix
R_load_variable=m.Param(value=step_ohmic_load)
G_load=1/R_load_variable
B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km],     #nochmal angucken!
              [-B_L_lv_line_1km, B_L_lv_line_1km]])

G = np.array([[0, 0],
              [0, G_load]])

#Power flow equations
m.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) == P_out) #modell
m.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) == P_2) #

# define omega as the derivation of the phase angle
m.Equation(theta1.dt() == w1)
m.Equation(theta2.dt() == w2)


m.Equation(P_in == P1_nom-(Pdroop*Pbase*((w1-omega)/omega))) #7.2
m.Equation((P_in-P_out) == J*w1*w1.dt()+D*Pbase*((w1-omega)/omega))
m.Equation(w2.dt()==-P_2/(J*w2)) #'diskussionsfähig' #7.1 #stellgröße #mit welcher variable du einfluss auf die regelstrecke nimmst, regelgröße w1 omega
m.Equation(w2==w1)



 #Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
m.options.IMODE = 7


m.time = np.linspace(0, t_end, steps) # time points


#Solve simulation
m.solve()

#print(B)
#print(G)


#gekko writes its variables in a strange format, sometimes workarounds like this are neccesary, this only get the frequency from the omega

f1 = np.divide(w1, (2*np.pi))
f2= np.divide(w2, (2*np.pi))

print(f1)

plt.title('Frequency')
plt.plot(m.time, f1, 'r', label='f1')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(48, 52)
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

# plt.title('Active Power Flow')
# plt.plot(m.time, P1, 'r', label='P1')
# plt.plot(m.time, P2, '--b', label='P2')
# plt.xlabel('Time (s)')
# plt.ylabel('Active Power (W)')
# plt.ylim(100000, 1000000)
# plt.legend()
# plt.show()












#np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")

#np.savetxt("PyCharm_value_rev1.csv", np.column_stack((m.time, f1, f2, f3, u1, u2, u3, P1, P2, P3, Q1, Q2, Q3)), delimiter=",", fmt='%s') #, header=header