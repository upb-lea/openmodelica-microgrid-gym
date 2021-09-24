import time

from gekko import GEKKO
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ts = 1e-3
t_end = 0.1
steps = int(1 / ts)
t = np.linspace(0, t_end, steps)

f0 = 50
V_eff = 230 * np.sqrt(2)

v_sin1 = V_eff * np.sin(2 * np.pi * f0 * t)
v_sin2 = V_eff * np.sin(2 * np.pi * f0 * t + 0.5)
plt.plot(t, v_sin1)
plt.plot(t, v_sin2)
plt.xlabel('Time (s)')
plt.grid()
# plt.ylim(225, 235)
# plt.legend()
plt.show()

R = 0.4
L = 2.3e-3
C = 10e-6

LT = 2.3e-3

R_load = 14

#   Initialize Model empty model as m
m = GEKKO(remote=False)

# Input
vi1 = m.Param(value=230)
vi2 = m.Param(value=230)
# vi1 = m.Param(value=v_sin1)#230)
# vi2 = m.Param(value=v_sin2)#230)

# Mdoel params
R1 = m.Param(value=R)
L1 = m.Param(value=L)
C1 = m.Param(value=C)
RT1 = m.Param(value=R)
LT1 = m.Param(value=LT)

R2 = m.Param(value=R)
L2 = m.Param(value=L)
C2 = m.Param(value=C)
RT2 = m.Param(value=R)
LT2 = m.Param(value=LT)

RLoad = m.Param(value=R_load)
iLoad = m.Var(value=0)

v1 = m.Var(value=0)
i1 = m.Var(value=0)
v2 = m.Var(value=0)
i2 = m.Var(value=0)
iT1 = m.Var(value=0)
iT2 = m.Var(value=0)

# DGLs node1
m.Equation(i1.dt() == (vi1 - v1) / L1 - R1 / L1 * i1)
m.Equation(v1.dt() == (i1 - iT1) / C1)
m.Equation(iT1.dt() == v1 / LT1 - RT1 / LT1 * iT1 - RLoad / LT1 * iLoad)

# DGLs node2
m.Equation(i2.dt() == (vi2 - v2) / L2 - R2 / L2 * i2)
m.Equation(v2.dt() == (i2 - iT2) / C1)
m.Equation(iT2.dt() == v2 / LT2 - RT2 / LT2 * iT2 - RLoad / LT2 * iLoad)

# constraints
m.Equation(iLoad == iT1 + iT2)

m.options.IMODE = 7  # oder 4?
m.time = t  # time points
# t = time.process_time()
start = time.time()
m.solve()
end = time.time()
elapsed_time = end - start

# elapsed_time = time.process_time() - t

print(f'Time for solving: {elapsed_time}')
# plt.title('Voltage drop over LCL filter')#
plt.plot(m.time, v1, 'r', label='V1')
plt.plot(m.time, v2, 'b', label='V2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
# plt.ylim(225, 235)
plt.legend()
plt.show()

plt.plot(m.time, iLoad)
plt.xlabel('Time (s)')
plt.ylabel('load Current (A)')
plt.grid()
# plt.ylim(225, 235)
plt.legend()
plt.show()
