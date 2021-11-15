import numpy as np
from openmodelica_microgrid_gym.util import nested_map, abc_to_dq0, dq0_to_abc
from typing import List


class Reward:
    def __init__(self, nom, lim, v_DC, gamma, det_run=False, nom_region: float = 1.1, use_gamma_normalization=1,
                 error_exponent: float = 1.0, i_lim: float = np.inf, i_nom: float = np.inf, i_exponent: float = 1.0):
        """

        :param nom: Nominal value for the voltage
        :param lim: Limit value for the voltage
        :param v_DC: DC-Link voltage
        :param gamma: Discount factor to map critic values -> [-1, 1]
        :param use_gamma_normalization: if 0 normalization depending on gamma is not used
        :param nom_region: Defines cliff in the reward landscape where the reward is pulled down because the nominal
                           value is exceeded. nom_region defines how much the nominal value can be exceeded before
                           the cliff (e.g. 1.1 -> cliff @ 1.1*self.nom
        :param error_exponent: defines the used error-function: E.g.: 1 -> Mean absolute error
                                                                2 -> Mean squared error
                                                              0.5 -> Mean root error
        :param i_lim: Limit value for the current
        :param i_nom: Nominal value for the current
        """
        self._idx = None
        self.nom = nom
        self.lim = lim
        self.v_DC = v_DC
        self.use_gamma_normalization = use_gamma_normalization
        if self.use_gamma_normalization == 1:
            self.gamma = gamma
        else:
            self.gamma = 0
        self.det_run = det_run
        self.nom_region = nom_region
        self.exponent = error_exponent
        self.i_lim = i_lim
        self.i_nom = i_nom
        self.i_exponent = i_exponent

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc.inductor{k}.i' for k in '123'], [f'inverter1.i_ref.{k}' for k in '012'],
                 [f'lc.capacitor{k}.v' for k in '123'], [f'inverter1.v_ref.{k}' for k in '012'],
                 'inverter1.phase.0'])

    def rew_fun_dq0(self, cols: List[str], data: np.ndarray, risk) -> float:
        """
        uses the same reward for voltage like defined above

         If v_lim is exceeded, episode abort -> env.abort_reward (should be -1) is given back

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        phase = data[idx[4]]

        vdq0_master = abc_to_dq0(data[idx[2]], phase)  # 3 phase currents at LC inductors

        # set points (sp)
        vsp_dq0_master = data[idx[3]]  # convert dq set-points into three-phase abc coordinates

        # SP = vsp_dq0_master * self.lim
        # mess = vdq0_master * self.lim

        if any(np.abs(data[idx[2]]) > 1):
            if self.det_run:
                return -(1 - self.gamma)
            else:
                return
        else:
            # rew = np.sum(1 - (2 * (np.abs(vsp_dq0_master - vdq0_master)) ** self.exponent)) * (1 - self.gamma) / 3  # /2

            # / 2 da normiert auf v_lim = 1, max abweichung 2*vlim=2, damit worst case bei 0
            rew = np.sum(1 - ((np.abs(vsp_dq0_master - vdq0_master) / 2) ** self.exponent)) * (1 - self.gamma) / 3  # /2
        """
        rew = np.sum(-((np.abs(vsp_dq0_master - vdq0_master)) ** self.exponent)) * (1 - self.gamma) / 3  # /2
        """
        return rew

    def rew_fun_PIPI_MRE(self, cols: List[str], data: np.ndarray, risk) -> float:
        """
        uses the same reward for voltage like defined above but also includes reward depending on the current
        If i_nom is exceeded r_current: f(i_mess) -> [0, 1] is multiplied to the r_voltage
        Before r_voltage is scaled to the region [0,1]:
         - r_voltage = (r_voltage+1)/2
         - r = r_voltage * r_current
         - r = r-1

         If v_lim or i_lim are exceeded, episode abort -> env.abort_reward (should be -1) is given back

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        i_mess = data[idx[0]]  # 3 phase currents at LC inductors
        mess = data[idx[2]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_abc_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates
        SP = data[idx[3]]  # convert dq set-points into three-phase abc coordinates

        # SP = vsp_dq0_master * self.lim
        # mess = vdq0_master * self.lim

        # rew = np.sum(-((np.abs(SP - mess)) ** 0.5)) * (1 - self.gamma) / 3

        phase = data[idx[4]]

        vdq0_master = abc_to_dq0(data[idx[2]], phase) / self.lim  # 3 phase currents at LC inductors

        # set points (sp)
        vsp_dq0_master = abc_to_dq0(data[idx[3]],
                                    phase) / self.lim  # convert dq set-points into three-phase abc coordinates

        # SP = vsp_dq0_master * self.lim
        # mess = vdq0_master * self.lim

        # rew = np.sum(-((np.abs(vsp_dq0_master - vdq0_master)) ** self.exponent)) * (1 - self.gamma) / 3

        rew = np.sum(1 - ((np.abs(vsp_dq0_master - vdq0_master) / 2) ** self.exponent)) * (1 - self.gamma) / 3

        return rew
