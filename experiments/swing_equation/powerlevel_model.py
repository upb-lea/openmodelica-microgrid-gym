
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

from openmodelica_microgrid_gym.aux_ctl import DroopParams, InverseDroopParams, DroopController, InverseDroopController, \
    DDS

delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 1000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes

nomFreq = 50  # grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
R = 20
L = 0.001

# Master
pdroop_param = DroopParams(DroopGain, 0.005, nomFreq)
qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)

master_PdroopController = DroopController(pdroop_param, delta_t)
master_QdroopController = DroopController(qdroop_param, delta_t)

# Slave
pdroop_param = InverseDroopParams(DroopGain, delta_t, nomFreq, tau_filt=0.04)
qdroop_param = InverseDroopParams(50, delta_t, nomVoltPeak, tau_filt=0.01)
slave_PdroopController = InverseDroopController(pdroop_param, delta_t)
slave_QdroopController = InverseDroopController(qdroop_param, delta_t)


class net_env:

    def __init__(self, G: np.ndarray, B: np.ndarray):
        """
        :param B: Susceptance matrix of the network; Bij connects node i to j with Bij
        :param G: Conductance matrix of the network; Bij connects node i to j with Bij
        """

        self.B = B
        self.G = G


class inverter:

    def __init__(self, id: int, tau: float = 0.5e-4, theta: float = 0, f: float = 50, v: float = 230, J:float = 2.0):
        """
        :param theta: Initial angle
        :param f: Inverter frequency in Hz
        :param v: Inverter voltage in V
        """
        self.id = id
        self._ts = tau
        self.theta = theta
        self.f = f
        self.v = v
        self.P = 0  # active power + -> from grid
        self.Q = 0  # reactive power
        self.J = J

        self._phaseDDS = DDS(self._ts)
        self.ode_solver = ode(self.env_model_ode)

        self.ode_solver.set_initial_value([self.f], 0).set_f_params(self.J)

    def step(self, net: net_env, inv_list: list):
        """
        Calculates the active power flow between inverter and inverter2 while the latter's
        angle is theta2.

        :param net: Network defining the environment
        :param inv_list: list with other inverters

        :return: active power transfer to inverter - negative?!
        """

        self.theta = self._phaseDDS.step(self.f)

        for inv2 in inv_list:
            self.P += self.v * inv2.v * (net.G[self.id][inv2.id] * np.cos(self.theta - inv2.theta) + \
                                         net.B[self.id][inv2.id] * np.sin(self.theta - inv2.theta))

            self.Q += self.v * inv2.v * (net.G[self.id][inv2.id] * np.sin(self.theta - inv2.theta) + \
                                         net.B[self.id][inv2.id] * np.cos(self.theta - inv2.theta))

        self.f = self.ode_solver.integrate(self.ode_solver.t+delta_t)

    def env_model_ode(self, t, y, arg):
        f = y[0]
        # u = y[1]

        J = arg

        return self.P / (J * f)




B_load = 2*np.pi*nomFreq * L #0.001
B12 = 2*np.pi*nomFreq * L
# toDo: f changes from step to step + different for each inverter
B = np.array([[B_load + B12, -B12], [-B12, B_load + B12]])  # Susceptance matrix







if __name__ == '__main__':

    inv1 = inverter(id = 0, theta=0)
    inv2 = inverter(id = 1)

    network = net_env(np.zeros(([2,2])), B)



    for step in range(max_episode_steps):
        inv1.step(network, [inv2])
        inv2.step(network, [inv1])
        print([inv1.P, inv2.P])

        #inv1.f = inv1.ode_solver.integrate(inv1.ode_solver.t+delta_t)
        #print(inv2.P)


