import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#Voltage 1
output = pd.read_pickle("1_V.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
A = output['lc1.capacitor1.v']
B = output['lc1.capacitor2.v']
C = output['lc1.capacitor3.v']
Voltage_1 = np.sqrt(A*A+B*B+C*C)/np.sqrt(3)
Voltage_1.plot()
plt.title('B1_V')
plt.show()
Voltage_1 = Voltage_1[::10]
Voltage_1.to_pickle("B1_V.pkl")

#Voltage2

output = pd.read_pickle("2_V.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
A = output['lc2.capacitor1.v']
B = output['lc2.capacitor2.v']
C = output['lc2.capacitor3.v']
Voltage_1 = np.sqrt(A*A+B*B+C*C)/np.sqrt(3)
Voltage_1.plot()
plt.title('B2_V')
plt.show()
Voltage_1 = Voltage_1[::10]
Voltage_1.to_pickle("B2_V.pkl")


#Voltage 3
output = pd.read_pickle("3_complete.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
A = output['rl1.resistor1.v']
B = output['rl1.resistor2.v']
C = output['rl1.resistor3.v']
D = output['rl1.inductor1.v']
E = output['rl1.inductor2.v']
F = output['rl1.inductor3.v']
A_1 = np.sqrt(A*A+D*D)
A_2 = np.sqrt(B*B+E*E)
A_3 = np.sqrt(C*C+F*F)
Voltage_3 = np.sqrt(A_1*A_1+A_2*A_2+A_3*A_3)/np.sqrt(3)
Voltage_3.plot()
plt.title('B3_V')
plt.show()
Voltage_3 = Voltage_3[::10]
Voltage_3.to_pickle("B3_V.pkl")

#P3
G = output['rl1.resistor1.i']
H = output['rl1.resistor2.i']
I = output['rl1.resistor3.i']

I_eff = np.sqrt(G*G+H*H+I*I)/np.sqrt(3)
U_eff_R = np.sqrt(A*A+B*B+C*C)/np.sqrt(3)
P_3 = I_eff * U_eff_R

P_3.plot()
plt.title('B3_P')
plt.show()
P_3 = P_3[::10]
P_3.to_pickle("B3_P.pkl")

#Q3

U_eff_L = np.sqrt(D*D+E*E+F*F)/np.sqrt(3)
Q_3 = I_eff * U_eff_L

Q_3.plot()
plt.title('B3_Q')
plt.show()
Q_3 = Q_3[::10]
Q_3.to_pickle("B3_Q.pkl")





#F1
output = pd.read_pickle("1_F.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
Freq_1 = output['master.freq']
Freq_1 = Freq_1[::10]
Freq_1.to_pickle("B1_F.pkl")


#F2
output = pd.read_pickle("2_f.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
Freq_2 = output['slave.freq']
Freq_2 = Freq_2[::10]
Freq_2.to_pickle("B2_F.pkl")


#1P

output = pd.read_pickle("1_P.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
P_1 = output['master.instPow']/3
P_1 = P_1[::10]
P_1.to_pickle("B1_P.pkl")


#Q1

output = pd.read_pickle("1_Q.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
Q_1 = output['master.instQ']/3
Q_1 = Q_1[::10]
Q_1.to_pickle("B1_Q.pkl")

#P2

output = pd.read_pickle("2_P.pkl")
plt.figure()
#print(output)
output.plot()
plt.show()
P_2 = output['slave.instPow']/3
P_2 = P_2[::10]
P_2.to_pickle("B2_P.pkl")


#Q2

output = pd.read_pickle("2_Q.pkl")
plt.figure()
print(output)
output.plot()
plt.show()
Q_2 = output['slave.instQ']/3
Q_2 = Q_2[::10]
Q_2.to_pickle("B2_Q.pkl")