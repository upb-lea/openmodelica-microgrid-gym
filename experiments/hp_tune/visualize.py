from pymongo import MongoClient

import matplotlib.pyplot as plt
import numpy as np

client = MongoClient('mongodb://localhost:27017/')
db = client['Plotting_test_db']

trail = db.Trail_number_0

test_data = trail.find_one({"Name": "Test"})
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax1, ax2 = ax.flatten()
ts = 1e-4  # if ts stored: take from db
t = np.arange(0, len(test_data['lc_capacitor1_v']) * ts, ts)
ax1.plot(t, test_data['lc_capacitor1_v'], label='v_a', color='b')
ax1.plot(t, test_data['lc_capacitor2_v'], color='r')
ax1.plot(t, test_data['lc_capacitor3_v'], color='g')
ax1.plot(t, test_data['inverter1_v_ref_0'], '--b', label='SP_a')
# plt.plot(t, test_data['lc_capacitor1_v'], label='v_a' ,color='--r')
# plt.plot(t, test_data['lc_capacitor1_v'], label='v_a' ,color='--gb')
ax1.legend()
ax1.set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
ax1.grid(which='both')

ax2.plot(t, test_data['r_load_resistor1_R'], 'g')
ax2.set_xlabel(r'$t\,/\,\mathrm{s}$')
ax2.set_ylabel('$R_{\mathrm{a}}\,/\,\mathrm{\Omega}$')
ax2.grid(which='both')
plt.show()

asd = 1
