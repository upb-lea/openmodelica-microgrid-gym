from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




#define general parameter
# Inertia, Damping Factor, Line Parameters, Filters, Load steps
k_p = 20 # P-Droop Factor [dimensionless], it's actually inverse
t_end = 0.5 # Simulation time / sec
steps = 1001 # 1 more step than expected, starting and endpoint are included
nomFreq = 50  # Grid frequency / Hz
nomVolt = value = 6000 # Voltage / Volt
omega = 2*np.pi*nomFreq
tau_VSG = 0.0000 # Filter constant of pt1 filter for P and Q, inverse of cut-off frequency
e1=6600 # Voltage inverter/V
u2=6570 # Voltage load/V
P2_nom=500000 # Nominal frequency/Hz
Delta_P2_step=100000 # Load step/W
J=np.ones(3)
J[0]=0.5 # Inertia J_nom/kgm² J_nom
J[1]=J[0]*0.5
J[2]=J[0]*0.1
Pbase = 500000  # P_base/W
P1_nom=500000 # Nominal active power of the first inverter/W
P2_nom=P1_nom ## Nominal active power of the Second Bus/W

all_f1=[] #list where all frequencies of the different J are included
D = 20  # Damping factor
X_L_lv_line_1km = 0.329  # Line inductance/Henry
X_L_lv_line_two = 0.329  # Line inductance/Henry
B_L_lv_line_1km = -1 / X_L_lv_line_1km  # Line suszeptance/(1/Henry)
B_L_lv_line_two_inverters= -1 / X_L_lv_line_two
R_load = u2 ** 2 / P2_nom # Load resistance/Ohm
R_load_step = u2 ** 2 / (P2_nom + Delta_P2_step) # Load step/Ohm
Pdroop_normal= (k_p * Pbase) / omega # Droop factor

step_ohmic_load = np.zeros(steps)
step_ohmic_load[0:500] = R_load # Resistance load/Ohm
step_ohmic_load[500:] = R_load_step # Resistance load/Ohm (calculated through P=U²/R)

T_a=(J[0]*(omega**2))/Pbase # inertia Constant/s
tau_Droop= (1 / k_p) * T_a # Tau for Droop control/s




#######VSG-Control###########################

#Loop through model in order to apply different J for the solver
for i in range(1):
    m = GEKKO(remote=False)
    e1=m.Const(e1) # Output voltage inverter/Volt
    u2=m.Const(u2) # Load voltage / Volt
    P_out= m.Var(value=0) #Output Voltage behind filter
    P_2=m.Var(value=0) # Power at second Bus/Watt
    w1 = m.Var(value=50) # Frequency at Bus 1/(1/s) (behind filter)
    w2 = m.Var(value=50) # Frequency at Bus 2/(1/s) (behind filter)
    theta1 = m.Var(value=0) #Phase Angle at Bus 1/rad
    theta2 = m.Var(value=0) #Phase Angle at Bus 2/rad
    P_out_dt=m.Var(value=0)
    P_in=m.Param(value=Pbase) #Input Power inverter/W

    #Matrices for node admittance matrix
    R_load_variable=m.Param(value=step_ohmic_load)
    G_load=1/R_load_variable
    B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km],     #nochmal angucken!
                  [-B_L_lv_line_1km, B_L_lv_line_1km]])
    G = np.array([[0, 0],
                  [0, G_load]])

    #Power flow equations
    m.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
               e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) == P_out)
    m.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
               u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) == P_2)

    # define omega as the derivation of the phase angle
    m.Equation(theta1.dt() == w1)
    m.Equation(theta2.dt() == w2)
    m.Equation(P_out.dt() == P_out_dt)

    #General VSM equations
    m.Equation((P_in-P_out) == J[i]*w1*w1.dt()+D*Pbase*((w1-omega)/omega))
    m.Equation(w2.dt()==-P_2/(J[i]*w2))

    #Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
    m.options.IMODE = 7
    m.time = np.linspace(0, t_end, steps) # time points


    #Solve simulation
    m.solve()
    f1 = np.divide(w1, (2*np.pi))
    all_f1.append(f1)
    m.cleanup()

######Droop Control Model###############################################################################################
n = GEKKO(remote=False)
#variables/parameters
e1=n.Const(e1) # Output voltage / Volt
u2=n.Const(u2) # Load voltage / Volt
P_1= n.Var(value=0)
w1 = n.Var(value=50) # Frequency at Bus 1/(1/s)
w2 = n.Var(value=50) # Frequency at Bus 2/(1/s)
theta1 = n.Var(value=0) # Phase angle/rad
theta2 = n.Var(value=0) # Phase angle/rad
P_2=n.Var(value=0) # Power at second Bus/W
omega_Droop=n.Var(value=0) #Frequency before filter/Hz

#Matrices for node admittance matrix
R_load_variable=n.Param(value=step_ohmic_load)
G_load=1/R_load_variable
B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km],
              [-B_L_lv_line_1km, B_L_lv_line_1km]])


G = np.array([[0, 0],
              [0, G_load]])


#Power flow equations
n.Equation(e1 * e1 * (G[0][0] * n.cos(theta1 - theta1) + B[0][0] * n.sin(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * n.cos(theta1 - theta2) + B[0][1] * n.sin(theta1 - theta2)) == P_1)
n.Equation(u2 * e1 * (G[1][0] * n.cos(theta2 - theta1) + B[1][0] * n.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * n.cos(theta2 - theta2) + B[1][1] * n.sin(theta2 - theta2)) == P_2)

#General Droop Equations
n.Equation(w1 + (tau_Droop * w1.dt()) == omega_Droop)
n.Equation(theta1.dt() == w1)
n.Equation(theta2.dt() == w2)
n.Equation(omega_Droop == (-(P_1 - P1_nom)/Pdroop_normal)+omega)
n.Equation(w2.dt() == -P_2 /(J[0]*w2))

#Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
n.options.IMODE = 7


n.time = np.linspace(0, t_end, steps) # time points


#Solve simulation
n.solve()
f1 = np.divide(w1, (2*np.pi))
all_f1.append(f1)
n.cleanup()


####Plots#####
all_f1=np.asarray(all_f1)
deviation_f1=all_f1-nomFreq
plt.title('Frequenzabweichung $f_1$ von $f_0$')
plt.plot(m.time, deviation_f1[0], 'r', label=r'VSG: $\Delta_{\mathrm{f_1}}\:(J=J_\mathrm{nom}, D=D_\mathrm{nom})$')
plt.plot(m.time, deviation_f1[1], 'g', label=r'Droop Control: $\Delta_{\mathrm{f_1}}$')
plt.axvline(x=0.249, color='black')
plt.xlabel('Time (s)')
plt.ylabel(r'$\Delta_{\mathrm{f_1}}\,/\,\mathrm{Hz}$')
plt.ylim(-2, 1)
plt.legend()
plt.show()



# plt.title('Phase Angle')
# plt.plot(m.time, theta1, 'r', label='theta 1')
# plt.plot(m.time, theta2, 'b', label='theta 2')
# plt.xlabel('Time (s)')
# plt.ylabel('Theta')
# plt.legend()
# plt.show()

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


#  Loop through model
# for i in range(2):
#     #   Initialize Model empty model as m
#     m = GEKKO(remote=False)
#     #variables/parameters
#     e1=m.Const(e1)
#     u2=m.Const(u2)
#     P_in = m.Var(value=0) #Input Voltage governor
#     P_out= m.Var(value=0) #Output Voltage behind filte
#     #P_2=m.Param(value=step_load_P2)
#     P_2=m.Var(value=0)
#     w1 = m.Var(value=2)
#     w2 = m.Var(value=2)
#     theta1 = m.Var(value=0)
#     theta2 = m.Var(value=0)
#     # P1f = m.Var(value=0)
#     # p1f = m.Var(value=0)
#
#     #Matrices for node admittance matrix
#     R_load_variable=m.Param(value=step_ohmic_load)
#     G_load=1/R_load_variable
#     B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km],     #nochmal angucken!
#                   [-B_L_lv_line_1km, B_L_lv_line_1km]])
#
#     G = np.array([[0, 0],
#                   [0, G_load]])
#
#     #Power flow equations
#     m.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#                e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) == P_out) #modell
#     m.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#                u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) == P_2) #
#
#     # define omega as the derivation of the phase angle
#     m.Equation(theta1.dt() == w1)
#     m.Equation(theta2.dt() == w2)
#
#
#     m.Equation(P_in == P1_nom-(Pdroop*Pbase*((w1-omega)/omega))) #7.2
#     m.Equation((P_in-P_out) == J[i]*w1*w1.dt()+D*Pbase*((w1-omega)/omega))
#     m.Equation(w2.dt()==-P_2/(J[i]*w2)) #'diskussionsfähig' #7.1 #stellgröße #mit welcher variable du einfluss auf die regelstrecke nimmst, regelgröße w1 omega
#     #m.Equation(w2==w1)
#
#
#
#      #Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
#     m.options.IMODE = 7
#
#
#     m.time = np.linspace(0, t_end, steps) # time points
#
#
#     #Solve simulation
#     m.solve()










#np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")

#np.savetxt("PyCharm_value_rev1.csv", np.column_stack((m.time, f1, f2, f3, u1, u2, u3, P1, P2, P3, Q1, Q2, Q3)), delimiter=",", fmt='%s') #, header=header