import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sim = pd.read_pickle('Simulation')
mes = pd.read_pickle('Measurement')

delta_t = 0.5e-4  # simulation time step size / s

N_sim = (len(sim['rl.inductor1.i']))
t_sim = np.linspace(0, N_sim * delta_t, N_sim)

N_mes = (len(mes['I_A']))
t_mes = np.linspace(0, N_mes * delta_t, N_mes)

if __name__ == '__main__':
    print(sim['rl.inductor1.i'])

    plt.plot(t_sim, sim['rl.inductor1.i'], label='sim')
    plt.plot(t_mes, mes['I_A'],'--', label='mes')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    plt.title('')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_sim, sim['rl.inductor1.i'], 'b', label=r'$i_{\mathrm{a}}$')
    plt.plot(t_sim, sim['rl.inductor2.i'], 'g')
    plt.plot(t_sim, sim['rl.inductor3.i'], 'r')
    #plt.plot(t_sim, sim['rl.inductor1.i'], 'b--', label=r'$i_{\mathrm{a}}$')
    #plt.plot(t_sim, sim['rl.inductor1.i'], 'g--')
    #plt.plot(t_sim, sim['rl.inductor1.i'], 'r--')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    plt.title('Simulation')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_mes, mes['I_A'], 'b', label=r'$i_{\mathrm{a}}$')
    plt.plot(t_mes, mes['I_B'], 'g')
    plt.plot(t_mes, mes['I_C'], 'r')
    plt.plot(t_mes, mes['SP_A'], 'b--', label=r'$i_{\mathrm{a}}$')
    plt.plot(t_mes, mes['SP_B'], 'g--')
    plt.plot(t_mes, mes['SP_C'], 'r--')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
    plt.title('Measurement')
    plt.grid()
    plt.legend()
    plt.show()


