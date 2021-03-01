from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
#Initialize Model
m = GEKKO(remote=False)

#User system property input

S_base_kVA = 100               #Enter maximum expected load of system (kVA)
V_base_ll = 400                #Nominal Voltage Line-Line
Freq_base = 50                 #System frequency


#p.u. calculations

S_base_VA = S_base_kVA*1000
I_base = S_base_VA/(np.sqrt(3)*V_base_ll)
Z_base = (V_base_ll**2)/S_base_VA

print(I_base)
print(Z_base)
#define parameter
Pdroop = 8000
Qdroop = 50
t_end = 0.4
steps = 400
Freq_base = 50                        # grid frequency / Hz
nomVolt = 230
omega = 2*np.pi*Freq_base

J = 0.05
J_Q = 0.0005

R_lv_line_10km = 0.0
#L_lv_line_10km = 0.00083/3         # Splitting Inductance into per phase value
L_lv_line_10km = 0.00026
B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)

# Resistors kan not be devided into three

step = np.zeros(steps)
step[0:50] = 10000000
step[50:200] = 20
step[200:] = 40

step_l = np.zeros(steps)
step_l[0:50] = 0
step_l[50:200] = 0.002
step_l[200:] = 0.002

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
q_offset = [0, 0, 0]

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

droop_linear=[-Pdroop,-Pdroop,0]                  # Droop?
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

m.Equation(u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) +
           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) +
           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P1)
m.Equation(u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) +
           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) +
           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P2)
m.Equation(u3 * u1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) +
           u3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) +
           u3 * u3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P3)

m.Equation(u1 * u1 * (G[0][0] * m.sin(theta1 - theta1) + B[0][0] * m.cos(theta1 - theta1)) +
           u1 * u2 * (G[0][1] * m.sin(theta1 - theta2) + B[0][1] * m.cos(theta1 - theta2)) +
           u1 * u3 * (G[0][2] * m.sin(theta1 - theta3) + B[0][2] * m.cos(theta1 - theta3)) == Q1)
m.Equation(u2 * u1 * (G[1][0] * m.sin(theta2 - theta1) + B[1][0] * m.cos(theta2 - theta1)) +
           u2 * u2 * (G[1][1] * m.sin(theta2 - theta2) + B[1][1] * m.cos(theta2 - theta2)) +
           u2 * u3 * (G[1][2] * m.sin(theta2 - theta3) + B[1][2] * m.cos(theta2 - theta3)) == Q2)
m.Equation(u3 * u1 * (G[2][0] * m.sin(theta3 - theta1) + B[2][0] * m.cos(theta3 - theta1)) +
           u3 * u2 * (G[2][1] * m.sin(theta3 - theta2) + B[2][1] * m.cos(theta3 - theta2)) +
           u3 * u3 * (G[2][2] * m.sin(theta3 - theta3) + B[2][2] * m.cos(theta3 - theta3)) == Q3)

# Equations

# Define angular frequency as a function of phase angle
m.Equation(theta1.dt() == w1)
m.Equation(theta2.dt() == w2)
m.Equation(theta3.dt() == w3)

#Power ODE


m.Equation(w1.dt() == ((p_offset[0]-P1)+(droop_linear[0]*(w1-Freq_base)))/(J*w1))
m.Equation(w2.dt() == ((p_offset[1]-P2)+(droop_linear[1]*(w2-Freq_base)))/(J*w2))
m.Equation(w3.dt() == ((p_offset[2]-P3)+(droop_linear[2]*(w3-Freq_base)))/(J*w3))

#Q_ODE


m.Equation(u1.dt() == ((q_offset[0]-Q1)+(q_droop_linear[0]*(u1-nomVolt)))/(J_Q*u1))
m.Equation(u2.dt() == ((q_offset[1]-Q2)+(q_droop_linear[1]*(u2-nomVolt)))/(J_Q*u2))
m.Equation(u3.dt() == ((q_offset[2]-Q3)+(q_droop_linear[2]*(u3-nomVolt)))/(J_Q*u3))

#m.Equation(J_Q*u1*u1.dt()==(-Q1))
#m.Equation(J_Q*u2*u2.dt()==(-Q2))
#m.Equation(J_Q*u3*u3.dt()==(-Q3))

m.Obj(Freq_base-w3)
m.Obj(nomVolt-u3)

#Set global options
m.options.IMODE = 7                 # Setting Sequential method of dynamic simulation https://gekko.readthedocs.io/en/latest/global.html

#steady state optimization

m.time = np.linspace(0,t_end,steps) # time points



#Solve simulation
m.solve()

#Results

# plt.plot(m.time,w1)
# plt.xlabel('time (s)')
# plt.ylabel('Frequency (Hz)')
#
#
# plt.plot(m.time,w2)
# plt.xlabel('time')
# plt.ylabel('w2(t)')


plt.plot(m.time,w3,'--', label='f at load')
plt.xlabel('time')
plt.ylabel('w3(t)')
#plt.ylim(48, 52)
plt.legend()
plt.show()



# plt.plot(m.time,u1, 'b')
# plt.xlabel('time')
# plt.ylabel('u1(t)')
#
#
# plt.plot(m.time,u2, 'r')
# plt.xlabel('time')
# plt.ylabel('u2(t)')
#
#
plt.plot(m.time,u3,'g')
plt.xlabel('time')
plt.ylabel('u3(t)')
plt.ylim(0, 400)
plt.show()




plt.plot(m.time,P1, 'b', label='P1')
plt.plot(m.time,P2, '--r', label='P2')
#plt.plot(m.time,P3, 'g')
plt.xlabel('time')
plt.ylabel('P(t)')
plt.legend()
plt.show()


plt.plot(m.time,np.multiply(1,np.add(Q1,Q2)),'b', label="Total Supply")
plt.plot(m.time,np.multiply(1,Q1),'m', label = 'Inverter 1')
plt.plot(m.time,np.multiply(1,Q2),'r', label = 'Inverter 2')
plt.plot(m.time,Q3,'g', label='load')
#plt.plot(m.time,np.add(np.add(Q1,Q2),Q3),'k', label='Difference')
#plt.plot(m.time,u3,'--g')
plt.xlabel('time')
plt.ylabel('Q(t)')
#plt.ylim(-40, 40)
plt.legend()
plt.show()


# plt.plot(m.time,(np.array(u1.value)-np.array(u2.value)))
# #plt.plot(m.time,(np.array(theta1.value)-np.array(theta3.value)))
# #plt.plot(m.time,(np.array(theta2.value)-np.array(theta3.value)))
# #plt.legend()
# plt.xlabel('time')
# plt.ylabel('diff_u(t)')
# plt.show()
#
# R_load
#
# plt.plot(m.time,R_load,'b')
# plt.xlabel('time')
# plt.ylabel('Resistance (Ohm)')
# #plt.ylim(48, 52)
# plt.show()

a = w1
np.savetxt("Swing_4000Q50j0_5jq0_0005.csv", a, delimiter=",")