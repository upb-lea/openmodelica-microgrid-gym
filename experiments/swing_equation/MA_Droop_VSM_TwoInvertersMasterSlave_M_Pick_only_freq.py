from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




#Definition of general parameters
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
P1_nom=400000 # Nominal active power of the first inverter/W
P3_nom=100000 # Nominal active power of the first inverter/W
P2_nom=P1_nom+P3_nom

all_f=[] # list where all frequencies are included
D = 20  # Damping factor
X_L_lv_line_1km = 0.329  # Line inductance/Henry
X_L_lv_line_two = 0.329/20  # Line inductance/Henry
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



#######Droop-Control-Model#######

#Loop through model in order to apply different J for the solver
for i in range(1):
    m = GEKKO(remote=False)
    e1=m.Const(e1) # Output voltage first inverter/Volt
    u2=m.Const(u2) # Load voltage/Volt
    e3=m.Const(e1) # Output voltage second inverter/Volt
    P_1=m.Var(value=0) # Power at Bus 1/W
    P_2=m.Var(value=0) # Power at Bus 2/W
    P_3=m.Var(value=0) # Power at Bus 3/W
    P_in=m.Const(value=P1_nom) # Input Power first Inverter/W
    P_in_2=m.Const(value=P3_nom) # Input Power secons Inverter/W
    w1 = m.Var(value=50) # Frequency at Bus 1/(1/s) (behind filter)
    w2 = m.Var(value=50) # Frequency at Bus 2/(1/s)
    w3 = m.Var(value=50) # Frequency at Bus 3/(1/s) (behind filter)
    theta1 = m.Var(value=0) # Phase angle at Bus 1/rad
    theta2 = m.Var(value=0) # Phase angle at Bus 2/rad
    theta3=m.Var(value=0) # Phase angle at Bus 3/rad
    omega_Droop_w1=m.Var(value=0) # Frequency at Bus 1/(1/s) (before filter)
    omega_Droop_w3=m.Var(value=0) # Frequency at Bus 3/(1/s) (before filter)


    #Matrices for node admittance matrix
    R_load_variable=m.Param(value=step_ohmic_load)
    G_load=1/R_load_variable

    B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km, 0],
                  [-B_L_lv_line_1km, 2 * B_L_lv_line_1km, -B_L_lv_line_1km],
                  [0, -B_L_lv_line_1km, B_L_lv_line_1km]])


    G = np.array([[0, 0, 0],
                  [0, G_load, 0],
                  [0, 0, 0]])

    #Power flow equations
    m.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
               e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
               e1 * e3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P_1)
    m.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
               u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
               u2 * e3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P_2)
    m.Equation(e3 * e1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
               e3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
               e3 * e3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P_3)

    # define omega as the derivation of the phase angle
    m.Equation(theta1.dt() == w1)
    m.Equation(theta2.dt() == w2)
    m.Equation(theta3.dt()== w3)

    #General Equations Droop
    m.Equation(w1 + (tau_Droop * w1.dt()) == omega_Droop_w1)
    m.Equation(omega_Droop_w1== (-(P_1 - P1_nom) / Pdroop_normal) + omega)
    m.Equation(w2.dt() == -P_2 / (J[0] * w2))
    m.Equation(w3 + (tau_Droop * w3.dt()) == omega_Droop_w3)
    m.Equation(P_3 == P3_nom-(k_p*Pbase*((omega_Droop_w3-omega)/omega)))


    #Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
    m.options.IMODE = 7
    m.time = np.linspace(0, t_end, steps) # time points


    #Solve simulation
    m.solve()
    f1 = np.divide(w1, (2*np.pi))
    #f2=np.divide(w2, (2*np.pi))
    f3=np.divide(w3, (2*np.pi))

    all_f.append(f1)
    all_f.append(f3)
    m.cleanup()

######VSM Model###############################################################################################
for i in range(1):
    n = GEKKO(remote=False)
    e1=n.Const(e1) # Output voltage first Inverter/Volt
    u2=n.Const(u2) # Load voltage/Volt
    e3=n.Const(e1) # Output voltage second Inverter/Volt
    P_1=n.Var(value=0) # Power at Bus 1/W
    P_2=n.Var(value=0) # Power at Bus 2/W
    P_3=n.Var(value=0) # Power at Bus 3/W
    P_in=n.Const(value=P1_nom) # Input power first inverter/W
    P_in_2=n.Const(value=P3_nom) # Input power second Inverter7W
    w1 = n.Var(value=50) # Frequency at Bus 1/(1/s) (behind filter)
    w2 = n.Var(value=50) # Frequency at Bus 2/(1/s)
    w3 = n.Var(value=50) # Frequency at Bus 3/(1/s) (behind filter)
    theta1 = n.Var(value=0) # Phase angle at Bus 1/rad
    theta2 = n.Var(value=0) # Phase angle at Bus 2/rad
    theta3=n.Var(value=0) # Phase angle at Bus 3/rad

    #Matrices for node admittance matrix
    R_load_variable=n.Param(value=step_ohmic_load)
    G_load=1/R_load_variable
    B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km, 0],
                  [-B_L_lv_line_1km, 2 * B_L_lv_line_1km, -B_L_lv_line_1km],
                  [0, -B_L_lv_line_1km, B_L_lv_line_1km]])

    G = np.array([[0, 0, 0],
                  [0, G_load, 0],
                  [0, 0, 0]])

    #Power flow equations
    n.Equation(e1 * e1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
               e1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
               e1 * e3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P_1)
    n.Equation(u2 * e1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
               u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
               u2 * e3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P_2)
    n.Equation(e3 * e1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
               e3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
               e3 * e3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P_3)

    # define omega as the derivation of the phase angle
    n.Equation(theta1.dt() == w1)
    n.Equation(theta2.dt() == w2)
    n.Equation(theta3.dt()== w3)
    #m.Equation(theta3.dt() == w3)

    #General VSM Equations
    n.Equation((P_in-P_1) == J[i]*w1*w1.dt()+D*Pbase*((w1-omega)/omega))
    n.Equation(w2.dt() == -P_2 / (J[i] * w2))
    n.Equation((P_in_2-P_3) == J[i]*w3*w3.dt()+D*Pbase*((w3-omega)/omega))



    #Set global options, 7 is the solving of DAE. You can check the GEKKO handbook, but i think, mode 7 and the default stuff of the rest should be fine.
    n.options.IMODE = 7
    n.time = np.linspace(0, t_end, steps) # time points


    #Solve simulation
    n.solve()
    f1 =np.divide(w1, (2*np.pi))
    f3=np.divide(w3, (2*np.pi))

    all_f.append(f1)
    all_f.append(f3)
    n.cleanup()

###Droop without Slave###
k_p = 20 # P-Droop Factor [dimensionless], it's actually inverse
t_end = 0.5 # Simulation time / sec
steps = 1001 # 1 more step than expected, starting and endpoint are included
nomFreq = 50  # Grid frequency / Hz
nomVolt = value = 6000 # Voltage / Volt
omega = 2*np.pi*nomFreq
tau_VSG = 0.0000# Filter constant of pt1 filter for P and Q, inverse of cut-off frequency
e1=6600 #V
u2=6570 #V
P2_nom=500000 #Watt
Delta_P2_step=100000
J=np.ones(3)
J[0]=0.5 # Inertia J_nom  / kgm² J_nom
J[1]=J[0]*0.5 # Inertia J_1 / kgm²
J[2]=J[0]*0.1 # Inertia J_2/ kgm²
D = 20  # [?, unfortunately given in per unit]
X_L_lv_line_1km = 0.329  # Line inductance / Henry
B_L_lv_line_1km = -1 / X_L_lv_line_1km  # Line suszeptance / (1/Henry)
R_load = u2 ** 2 / P2_nom # Load resistance / Ohm
R_load_step = u2 ** 2 / (P2_nom + Delta_P2_step) # Load step / Ohm
Pbase = 500000  # P_base / W
P1_nom=Pbase # Nominal active power of the inverter / W
Pdroop_normal= (k_p * Pbase) / omega

step_ohmic_load = np.zeros(steps)
step_ohmic_load[0:500] = R_load # Resistance load / Ohm
step_ohmic_load[500:] = R_load_step # Resistance load / Ohm (calculated through P=U²/R)

T_a=(J[0]*(omega**2))/Pbase # inertia Constant / s
tau_Droop= (1 / k_p) * T_a # tau for Droop control / s

n = GEKKO(remote=False)
#variables/parameters
e1=n.Const(e1) # Output voltage / Volt
u2=n.Const(u2) # Load voltage / Volt
P_1= n.Var(value=0) #Output Voltage behind filter
w1 = n.Var(value=50)
w2 = n.Var(value=50)
theta1 = n.Var(value=0)
theta2 = n.Var(value=0)
P_2=n.Var(value=0)
omega_Droop=n.Var(value=0)

#Matrices for node admittance matrix
R_load_variable=n.Param(value=step_ohmic_load)
G_load=1/R_load_variable
B = np.array([[B_L_lv_line_1km, -B_L_lv_line_1km],     #nochmal angucken!
              [-B_L_lv_line_1km, B_L_lv_line_1km]])


G = np.array([[0, 0],
              [0, G_load]])


#Power flow equations
n.Equation(e1 * e1 * (G[0][0] * n.cos(theta1 - theta1) + B[0][0] * n.sin(theta1 - theta1)) + \
           e1 * u2 * (G[0][1] * n.cos(theta1 - theta2) + B[0][1] * n.sin(theta1 - theta2)) == P_1)
n.Equation(u2 * e1 * (G[1][0] * n.cos(theta2 - theta1) + B[1][0] * n.sin(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * n.cos(theta2 - theta2) + B[1][1] * n.sin(theta2 - theta2)) == P_2)


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
all_f.append(f1)
n.cleanup()


####Plots####
all_f=np.asarray(all_f)
deviation_f1= all_f - 50
plt.title('Frequenzabweichung $f$ von $f_0$')
plt.plot(m.time, deviation_f1[0], 'r', label=r'f1 (Droop - mit Slave')
plt.plot(m.time, deviation_f1[1], 'black', label=r'f2 (Droop - mit Slave)')
plt.plot(n.time, deviation_f1[2], 'orange', label=r'f1 (VSM)')
plt.plot(n.time, deviation_f1[3], 'g', label=r'f2 (VSM)')
plt.plot(n.time, deviation_f1[4], 'yellow', label=r'f1 (Droop, ein Inverter ohne Slave)')
plt.axvline(x=0.249, color='black')
plt.xlabel('Time (s)')
plt.ylabel(r'$\Delta_{\mathrm{f}}\,/\,\mathrm{Hz}$')
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