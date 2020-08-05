# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:11:49 2020

@author: Jarren
"""
import paramiko
import numpy as np
import matplotlib.pyplot as plt

ssh = paramiko.SSHClient()

ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect('lea-jde10', username='root', password='')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('./my_first_hps 1000  0.01 10.0 15.0 50.0 ')

DT = 1 / 10000

result = list()
lastm = 0
laste = 0
count = 0
for line in ssh_stdout.read().splitlines():

    temp = line.decode("utf-8").split(",")

    # print(temp)
    if len(temp) == 20:
        temp.pop(-1)  # Drop the last item

        floats = [float(i) for i in temp]
        # print(floats)
        result.append(floats)
        count = count + 1
    elif len(temp) != 1:
        print(len(temp))
        print(temp)

N = (len(result))
y = np.array(result)
t = np.linspace(0, N * DT, N)

V_A = y[:, 0];
V_B = y[:, 1];
V_C = y[:, 2];
I_A = y[:, 3];
I_B = y[:, 4];
I_C = y[:, 5];
I_D = y[:, 6];
I_Q = y[:, 7];
I_0 = y[:, 8];
Ph = y[:, 9];
SP_A = y[:, 10];
SP_B = y[:, 11];
SP_C = y[:, 12];
m_A = y[:, 13];
m_B = y[:, 14];
m_C = y[:, 15];
m_D = y[:, 16];
m_Q = y[:, 17];
m_0 = y[:, 18];

plt.plot(t, V_A, t, V_B, t, V_C)
plt.ylabel('Voltages (V)')
plt.grid()
plt.show()

plt.plot(t, SP_A, t, SP_B, t, SP_C)
plt.ylabel('SPs (A)')
plt.grid()
plt.show()

plt.plot(t, SP_A, t, I_A)
plt.ylabel('ERROR_PhA(A)')
plt.grid()
plt.show()

plt.plot(t, m_A, t, m_B, t, m_C)
plt.ylabel('ms (#)')
plt.grid()
plt.show()

plt.plot(t, I_A, t, I_B, t, I_C)
plt.ylabel('Currents (A)')
plt.grid()
plt.show()

plt.plot(t, I_D, t, I_Q, t, I_0)
plt.ylabel('Currents DQ0(A)')
plt.grid()
plt.ylim(-25, 25)
plt.show(),

# plt.plot(t,V_A,t,I_A/36)
# plt.ylabel('DT Comp TESTS (V)')
# plt.ylim(-1, 1)
# plt.grid()
# plt.show()

# plt.plot(KI)
# plt.ylabel('KI (V)')
# plt.show()


ssh.close()
