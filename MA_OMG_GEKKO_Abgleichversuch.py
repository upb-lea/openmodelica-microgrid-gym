from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds


# Variables
t_end = 0.5  # Simulation time / sec
steps = 1001  # 1 more step than expected, starting and endpoint are included
nomFreq = 49.55  # Grid frequency / Hz

nomVolt = 230  # Voltage / Volt
omega = 2 * np.pi * nomFreq
e1 = 230  # Voltage inverter/V
u2 = 230  # Voltage load/V
e3 = 230


P1_nom = 4000
Q1_nom = 8200
P3_nom = 4000
Q3_nom = 10000
Pbase = 8000

all_f1 = []  # list to store f1, f2, f3 and the voltages e1, e2,e3
all_f2 = []
all_f3 = []
all_e1 = []  # list where all voltages can be found
all_e2 = []  # list where all voltages can be found
all_e3 = []  # list where all voltages can be found

omega = 2 * 3.14 * nomFreq  # np.pi cant be used, gekko does not found a solution in this case if it has lots of decimal places
L_L_lv_line = omega * 0.0002  # Line inductance/Henry
X_L_lv_line = L_L_lv_line
B_L_lv_line = -1 / X_L_lv_line  # Line suszeptance/(1/Henry)
Pdroop1 = 8000  # virtual droop gain for active power / W/Hz
Pdroop3 = 6000  # virtual droop gain for active power / W/Hz
Qdroop1 = 200  # virtual droop gain for reactive power / VAR/V
Qdroop3 = 100  # virtual droop gain for reactive power / VAR/V
q_offset = [0, 0, 0]

# Similarities between Droop Parameter and Parameter of Swing Equation (see Master Thesis)
tau_Droop = 0.005
k_p_1 = (Pdroop1 * omega) / P1_nom
T_a = tau_Droop * k_p_1
J1 = T_a * P1_nom / (omega ** 2)
tau_Droop = 0.005
k_p_3 = (Pdroop3 * omega) / P3_nom
T_a = tau_Droop * k_p_3
J3 = T_a * P3_nom / (omega ** 2)
D_1 = k_p_1
D_3 = k_p_3

#Definition of the load
J_Q = 0.00005
R_load = 6.22
R_load_step = 6.22 / 2
L_load = 0.00495
L_load_step = 0.00495 / 2

step_ohmic_r_load = np.zeros(steps)
step_ohmic_r_load[0:500] = R_load  # Resistance load/Ohm
step_ohmic_r_load[500:] = R_load_step  # Resistance load/Ohm (calculated through P=U²/R)

step_ohmic_l_load = np.zeros(steps)
step_ohmic_l_load[0:500] = L_load  # Resistance load/Ohm
step_ohmic_l_load[500:] = L_load_step  # Resistance load/Ohm (calculated through P=U²/R)

##### Model using Swing Equation #####################################
n = GEKKO(remote=False)
e1 = n.Var(value=10)  # Output voltage first Inverter/Volt
u2 = n.Var(value=10)  # Load voltage/Volt
e3 = n.Var(value=10)  # Output voltage second Inverter/Volt
P_1 = n.Var(value=0)  # Power at Bus 1/W
P_2 = n.Var(value=0)  # Power at Bus 2/W
P_3 = n.Var(value=0)  # Power at Bus 3/W
Q_1 = n.Var(value=0)  # Power at Bus 1/W
Q_2 = n.Var(value=0)  # Power at Bus 2/W
Q_3 = n.Var(value=0)
P_in = n.Const(value=P1_nom)  # Input power first inverter/W
P_in_2 = n.Const(value=P3_nom)  # Input power second Inverter7W
Q_in = n.Const(value=Q1_nom)  # Input power first inverter/W
Q_in_2 = n.Const(value=Q3_nom)  # Input power second Inverter7W
w1 = n.Var(value=300)  # Frequency at Bus 1/(1/s) (behind filter)
w2 = n.Var(value=300)  # Frequency at Bus 2/(1/s)
w3 = n.Var(value=300)  # Frequency at Bus 3/(1/s) (behind filter)
theta1 = n.Var(value=0)  # Phase angle at Bus 1/rad
theta2 = n.Var(value=0)  # Phase angle at Bus 2/rad
theta3 = n.Var(value=0)  # Phase angle at Bus 3/rad

# Matrices for node admittance matrix
R_load = n.Param(value=step_ohmic_r_load)
L_load = n.Param(value=step_ohmic_l_load)
G_RL_load = R_load / (R_load ** 2 + (omega * L_load) ** 2)
B_RL_load = -(omega * L_load) / (R_load ** 2 + (omega * L_load) ** 2)

B = np.array([[2 * B_L_lv_line, -B_L_lv_line, -B_L_lv_line],
              [-B_L_lv_line, 2 * B_L_lv_line + B_RL_load, -B_L_lv_line],
              [-B_L_lv_line, -B_L_lv_line, 2 * B_L_lv_line]])

G = np.array([[0, 0, 0],
              [0, G_RL_load, 0],
              [0, 0, 0]])

# Power flow equations
n.Equation(e1 * e1 * (G[0][0] * n.cos(theta1 - theta1) + B[0][0] * n.sin(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * n.cos(theta1 - theta2) + B[0][1] * n.sin(theta1 - theta2)) + \
           e1 * e3 * (G[0][2] * n.cos(theta1 - theta3) + B[0][2] * n.sin(theta1 - theta3)) == P_1)
n.Equation(u2 * e1 * (G[1][0] * n.cos(theta2 - theta1) + B[1][0] * n.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * n.cos(theta2 - theta2) + B[1][1] * n.sin(theta2 - theta2)) + \
           u2 * e3 * (G[1][2] * n.cos(theta2 - theta3) + B[1][2] * n.sin(theta2 - theta3)) == P_2)
n.Equation(e3 * e1 * (G[2][0] * n.cos(theta3 - theta1) + B[2][0] * n.sin(theta3 - theta1)) + \
           e3 * u2 * (G[2][1] * n.cos(theta3 - theta2) + B[2][1] * n.sin(theta3 - theta2)) + \
           e3 * e3 * (G[2][2] * n.cos(theta3 - theta3) + B[2][2] * n.sin(theta3 - theta3)) == P_3)

n.Equation(e1 * e1 * (G[0][0] * n.sin(theta1 - theta1) - B[0][0] * n.cos(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * n.sin(theta1 - theta2) - B[0][1] * n.cos(theta1 - theta2)) + \
           e1 * e3 * (G[0][2] * n.sin(theta1 - theta3) - B[0][2] * n.cos(theta1 - theta3)) == Q_1)
n.Equation(u2 * e1 * (G[1][0] * n.sin(theta2 - theta1) - B[1][0] * n.cos(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * n.sin(theta2 - theta2) - B[1][1] * n.cos(theta2 - theta2)) + \
           u2 * e3 * (G[1][2] * n.sin(theta2 - theta3) - B[1][2] * n.cos(theta2 - theta3)) == Q_2)
n.Equation(e3 * e1 * (G[2][0] * n.sin(theta3 - theta1) - B[2][0] * n.cos(theta3 - theta1)) + \
           e3 * u2 * (G[2][1] * n.sin(theta3 - theta2) - B[2][1] * n.cos(theta3 - theta2)) + \
           e3 * e3 * (G[2][2] * n.sin(theta3 - theta3) - B[2][2] * n.cos(theta3 - theta3)) == Q_3)

# define omega as the derivation of the phase angle
n.Equation(theta1.dt() == w1)
n.Equation(theta2.dt() == w2)
n.Equation(theta3.dt() == w3)

# General VSM Equations
n.Equation((P_in - P_1) == J1 * w1 * w1.dt() + D_1 * P1_nom * ((w1 - omega) / omega))
n.Equation(w2.dt() == -P_2 / (J1 * w2))
n.Equation((P_in_2 - P_3) == J3 * w3 * w3.dt() + D_3 * P3_nom * ((w3 - omega) / omega))

n.Equation(Q_1 == Qdroop1 * (e1 - nomVolt))
n.Equation(Q_2 == 0)
n.Equation(Q_3 == Qdroop3 * (e3 - nomVolt))

n.options.IMODE = 7
n.time = np.linspace(0, t_end, steps)  # time points

# Solve simulation
n.solve()
f1 = np.divide(w1, (2 * np.pi))
f3 = np.divide(w3, (2 * np.pi))

all_f1.append(f1)
all_e1.append(e1)

all_f3.append(f3)
all_e3.append(e3)
n.cleanup()



########GEKKO Model - Droop mit Filter###############################################
m = GEKKO(remote=False)
e1 = m.Var(10)  # Output voltage first inverter/Volt
u2 = m.Var(10)  # Output voltage first inverter/Volt
e3 = m.Var(10)  # Output voltage second inverter/Volt
P_1 = m.Var(value=0)  # Power at Bus 1/W
P_2 = m.Var(value=0)  # Power at Bus 2/W
P_3 = m.Var(value=0)  # Power at Bus 3/W
Q_1 = m.Var(value=0)  # Power at Bus 1/W
Q_2 = m.Var(value=0)  # Power at Bus 2/W
Q_3 = m.Var(value=0)
P_in = m.Const(value=P1_nom)  # Input Power first Inverter/W
P_in_2 = m.Const(value=P3_nom)  # Input Power second Inverter/W
Q_in = m.Const(value=Q1_nom)  # Input Power first Inverter/W
Q_in_2 = m.Const(value=Q3_nom)
w1 = m.Var(value=300)  # Frequency at Bus 1/(1/s) (behind filter)
w2 = m.Var(value=300)  # Frequency at Bus 2/(1/s)
w3 = m.Var(value=300)  # Frequency at Bus 3/(1/s) (behind filter)
theta1 = m.Var(value=0)  # Phase angle at Bus 1/rad
theta2 = m.Var(value=0)  # Phase angle at Bus 2/rad
theta3 = m.Var(value=0)  # Phase angle at Bus 3/rad
omega_Droop_w1 = m.Var(value=0)  # Frequency at Bus 1/(1/s) (before filter)
omega_Droop_w3 = m.Var(value=0)  # Frequency at Bus 3/(1/s) (before filter)

# Matrices for node admittance matrix
R_load = m.Param(value=step_ohmic_r_load)
L_load = m.Param(value=step_ohmic_l_load)
# u2 = m.Param(value=step_u2)
G_RL_load = R_load / (R_load ** 2 + (omega * L_load) ** 2)
# B_RL_load = -(omega * L_load) / (R_load ** 2 + (omega * L_load) ** 2)
B_RL_load = -(omega * L_load) / (R_load ** 2 + (omega * L_load) ** 2)

B = np.array([[2 * B_L_lv_line, -B_L_lv_line, -B_L_lv_line],
              [-B_L_lv_line, 2 * B_L_lv_line + B_RL_load, -B_L_lv_line],
              [-B_L_lv_line, -B_L_lv_line, 2 * B_L_lv_line]])

G = np.array([[0, 0, 0],
              [0, G_RL_load, 0],
              [0, 0, 0]])

# Power flow equations
m.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
           e1 * e3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P_1)
m.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
           u2 * e3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P_2)
m.Equation(e3 * e1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
           e3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
           e3 * e3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P_3)

m.Equation(e1 * e1 * (G[0][0] * m.sin(theta1 - theta1) - B[0][0] * m.cos(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * m.sin(theta1 - theta2) - B[0][1] * m.cos(theta1 - theta2)) + \
           e1 * e3 * (G[0][2] * m.sin(theta1 - theta3) - B[0][2] * m.cos(theta1 - theta3)) == Q_1)
m.Equation(u2 * e1 * (G[1][0] * m.sin(theta2 - theta1) - B[1][0] * m.cos(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.sin(theta2 - theta2) - B[1][1] * m.cos(theta2 - theta2)) + \
           u2 * e3 * (G[1][2] * m.sin(theta2 - theta3) - B[1][2] * m.cos(theta2 - theta3)) == Q_2)
m.Equation(e3 * e1 * (G[2][0] * m.sin(theta3 - theta1) - B[2][0] * m.cos(theta3 - theta1)) + \
           e3 * u2 * (G[2][1] * m.sin(theta3 - theta2) - B[2][1] * m.cos(theta3 - theta2)) + \
           e3 * e3 * (G[2][2] * m.sin(theta3 - theta3) - B[2][2] * m.cos(theta3 - theta3)) == Q_3)

# define omega as the derivation of the phase angle
m.Equation(theta1.dt() == w1)
m.Equation(theta2.dt() == w2)
m.Equation(theta3.dt() == w3)

# General Equations Droop
m.Equation(w1 + (tau_Droop * w1.dt()) == omega_Droop_w1)
m.Equation(omega_Droop_w1 == (-(P_1 - P1_nom) / Pdroop1) + omega)
m.Equation(w2.dt() == -P_2 / (J1 * w2))
m.Equation(w3 + (tau_Droop * w3.dt()) == omega_Droop_w3)
m.Equation(omega_Droop_w3 == (-(P_3 - P3_nom) / Pdroop3) + omega)

m.Equation(Q_1 == Qdroop1 * (e1 - nomVolt))
m.Equation(Q_2 == 0)
m.Equation(Q_3 == Qdroop3 * (e3 - nomVolt))

m.options.IMODE = 7
m.time = np.linspace(0, t_end, steps)  # time points

m.options.IMODE = 7
m.time = np.linspace(0, t_end, steps)  # time points

# Solve simulation
m.solve()
f1 = np.divide(w1, (2 * np.pi))
f2 = np.divide(w2, (2 * np.pi))
f3 = np.divide(w3, (2 * np.pi))

m.cleanup()

all_f1.append(f1)
all_f3.append(f3)
all_e1.append(e1)
all_e3.append(e3)
m.cleanup()


### Loading of the OMG-Simulation ###
with open("B1_V.pkl", "rb") as fh:
    B1_V = pickle.load(fh)

with open("B2_V.pkl", "rb") as fh:
    B2_V = pickle.load(fh)

with open("B1_F.pkl", "rb") as fh:
    B1_F = pickle.load(fh)

with open("B2_F.pkl", "rb") as fh:
    B2_F = pickle.load(fh)

plt.grid(visible=True, which='major', axis='both')
plt.rcParams.update({'font.size': 12})
all_f1 = np.asarray(all_f1)

# Plots - Comparison between OMG and the results of this GEKKO model ###
all_f1 = np.asarray(all_f1)
plt.title('Frequenz $f$: Neues Modell u. OMG-Simulation')
plt.plot(n.time, all_f1[0], c='xkcd:light red', label=r'$f_\mathrm{1}$ (Neues Modell)', linestyle='dashed')
plt.plot(n.time, all_f3[0], c='xkcd:blue', label=r'$f_\mathrm{2}$ (Neues Modell)', linestyle='dashed')
plt.plot(n.time, B1_F, c='xkcd:dark red', label=r'$f_\mathrm{1}$ (Simulation OMG)')
plt.plot(n.time, B2_F, c='xkcd:dark blue', label=r'$f_\mathrm{2}$ (Simulation OMG)')
plt.axvline(x=0.249, color='black')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$f\,/\,\mathrm{Hz}$')
plt.ylim(47, 53)
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.show()

plt.title('Busspannung $U$: Neues Modell u. OMG-Simulation')
plt.plot(n.time, all_e1[0], c='xkcd:light red', label=r'$U_\mathrm{1}$ (Neues Modell)', linestyle='dashed')
plt.plot(n.time, all_e3[0], c='xkcd:blue', label=r'$U_\mathrm{2}$ (Neues Modell)', linestyle='dashed')
plt.plot(n.time, B1_V, c='xkcd:dark red', label=r'$U_\mathrm{1}$ (Simulation OMG)')
plt.plot(n.time, B2_V, c='xkcd:dark blue', label=r'$U_\mathrm{2}$ (Simulation OMG)')
plt.ylim(100, 300)
plt.grid(visible=True, which='major', axis='both')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$U\,/\,\mathrm{V}$')
plt.axvline(x=0.249, color='black')
plt.legend()
plt.show()

x = 1
