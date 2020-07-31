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
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('./my_first_hps 2000  0.01 5.0 10.0 50.0')

DT = 1/20000



result = list()
for line in ssh_stdout.read().splitlines():
    
    temp = line.decode("utf-8").split(",")
    
    if len(temp) == 11  :
        temp.pop(-1)    #Drop the last item

        floats = [float(i) for i in temp]
        #print(floats)
        result.append(floats)
    elif len (temp)!=1:
        print(temp)

N = (len(result))
y = np.array(result)
t = np.linspace(0, N*DT, N)

V_A = y[:,0];
V_B = y[:,1];
V_C = y[:,2];
I_A = y[:,3];
I_B = y[:,4];
I_C = y[:,5];
I_D = y[:,6];
I_Q = y[:,7];
I_0 = y[:,8];

print(sum(KI) / len(KI) )

plt.plot(t,V_A,t,V_B,t,V_C)
plt.ylabel('Voltages (V)')
plt.show()

plt.plot(t,I_A,t,I_B,t,I_C)
plt.ylabel('Currents (A)')
plt.show()

plt.plot(t,I_D,t,I_Q,t,I_0)
plt.ylabel('Currents DQ0(A)')
plt.show()

#plt.plot(KI)
#plt.ylabel('KI (V)')
#plt.show()


ssh.close()
