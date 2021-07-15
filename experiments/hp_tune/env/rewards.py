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

    def rew_fun(self, cols: List[str], data: np.ndarray, risk) -> float:
        """
        Defines the reward function for the environment. Uses the observations and set-points to evaluate the quality of
        the used parameters.
        Takes current and voltage measurements and set-points to calculate the mean-root control error and uses a
        logarithmic barrier function in case of violating the current limit. Barrier function is adjustable using
        parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        vabc_master = data[idx[2]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_abc_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates
        vsp_abc_master = data[idx[3]]  # convert dq set-points into three-phase abc coordinates

        SP = vsp_abc_master * self.lim
        mess = vabc_master * self.lim

        if all(np.abs(mess) <= self.nom * 1.1):
            # if all(np.abs(mess) <= self.lim*self.nom_region):
            """
            1st area - inside wanted (nom) operation range
            -v_nom -> + v_nom
                rew = 1; if mess = SP
                rew = 1/3; if error = SP-mess = 2*v_nom (worst case without braking out from nom area)
            """
            # devided by 3 because of sums up all 3 phases
            rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * 2 * 1 / 3 +
                         1 / 3) / 3


        elif any(np.abs(mess) > self.lim):
            """
            3rd area - outside valid area - above lim - possible if enough v_DC - DANGEROUS
            +-v_lim -> +-v_DC

            V1:
            @ SP = +v_nom AND mess = -v_DC:
                rew = -1; if error = v_DC + v_nom -> Worst case, +v_nom wanted BUT -v_DC measured
            @ SP = -v_nom AND mess = -v_lim
                rew ~ -1/3 - f[(lim-nom)/(nom+v_DC)]
                rew -> -1 - 2/3*(1 - |lim - nom| / (nom+v_DC))
                The latter fraction is quite small but leads to depending on the system less then 2/3 is
                substracted and we have a gap to the 2nd area! :) 
                
            V2: None is returned to stop the episode (hint: in the env env.abort_reward is given back as reward(?)
            
            V3: rew = -1
            """

            # V1:
            # rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.v_DC)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma)) / 3

            # V2:
            # if return -> rew = None and in env abort_reward is given to agent
            if self.det_run:
                return -(1 - self.gamma)
            else:
                return

            # V3:
            # rew = (1 - gamma)


        else:
            """
            2nd area
            +-v_nom -> +- v_lim

            @ SP = v_nom AND mess = v_nom (-µV), da if mess > v_nom (hier noch Sicherheitsabstand?)
                rew = 1/3
            @ SP = v_nom AND mess = -v_lim
                rew = -1/3

            """
            rew = np.sum(
                (1 - np.abs(SP - mess) / (self.nom + self.lim)) * 2 * 1 / 3 - 1 / 3) / 3

        return rew * (1 - self.gamma)
        # return -np.clip(error.squeeze(), 0, 1e5)

    def rew_fun_include_current(self, cols: List[str], data: np.ndarray, risk) -> float:
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

        iabc_master = data[idx[0]]  # 3 phase currents at LC inductors
        vabc_master = data[idx[2]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_abc_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates
        vsp_abc_master = data[idx[3]]  # convert dq set-points into three-phase abc coordinates

        i_mess = iabc_master * self.i_lim

        SP = vsp_abc_master * self.lim
        mess = vabc_master * self.lim

        if any(np.abs(mess) > self.lim) or any(np.abs(i_mess) > self.i_lim):
            """
            3rd area - outside valid area - above lim - possible if enough v_DC - DANGEROUS
            +-v_lim -> +-v_DC
            Valid for v_lim OR i_lim exceeded

            V1:
            @ SP = +v_nom AND mess = -v_DC:
                rew = -1; if error = v_DC + v_nom -> Worst case, +v_nom wanted BUT -v_DC measured
            @ SP = -v_nom AND mess = -v_lim
                rew ~ -1/3 - f[(lim-nom)/(nom+v_DC)]
                rew -> -1 - 2/3*(1 - |lim - nom| / (nom+v_DC))
                The latter fraction is quite small but leads to depending on the system less then 2/3 is
                substracted and we have a gap to the 2nd area! :) 

            V2: None is returned to stop the episode (hint: in the env env.abort_reward is given back as reward(?)

            V3: rew = -1
            """

            # V1:
            # rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.v_DC)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma)) / 3

            # V2:
            # if return -> rew = None and in env abort_reward is given to agent
            if self.det_run:
                return -(1 - self.gamma)
            else:
                return

            # V3:
            # rew = (1 - gamma)

        elif all(np.abs(mess) <= self.nom * 1.1):
            # if all(np.abs(mess) <= self.lim*self.nom_region):
            """
            1st area - inside wanted (nom) operation range
            -v_nom -> + v_nom
                rew = 1; if mess = SP
                rew = 1/3; if error = SP-mess = 2*v_nom (worst case without braking out from nom area)
            """
            # devided by 3 because of sums up all 3 phases
            # rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * 2 * (1 - self.gamma) / 3 + (
            #        1 - self.gamma) / 3) / 3

            rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * (1 - self.gamma)) / 3




        else:
            """
            2nd area
            +-v_nom -> +- v_lim

            @ SP = v_nom AND mess = v_nom (-µV), da if mess > v_nom (hier noch Sicherheitsabstand?)
                rew = 1/3
            @ SP = v_nom AND mess = -v_lim
                rew = -1/3

            """
            # rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.lim)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma) / 3) / 3
            rew = (1 - np.max(np.abs(SP - mess)) / (self.nom + self.lim)) * (1 - self.gamma) / 2 - (1 - self.gamma) / 2

        if any(abs(i_mess) > ((self.i_nom + self.i_lim) / 2)):
            rew = (rew + 1) / 2  # map rew_voltage -> [0,1]

            # Scale rew_voltage with rew_current
            # rew = rew * np.sum((((self.i_nom - i_mess) / (self.i_lim - self.i_nom))+1) ** self.i_exponent) / 3
            rew = rew * (((self.i_nom - max(abs(i_mess))) / (self.i_lim - self.i_nom)) + 1) ** self.i_exponent

            rew = rew * 2 - 1  # map rew -> [-1, 1]

        if rew < -1:
            asd = 1
        return rew  # * (1-0.9)
        # return -np.clip(error.squeeze(), 0, 1e5)

    def rew_fun_include_current_dq0(self, cols: List[str], data: np.ndarray, risk) -> float:
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

        phase = data[idx[4]]

        idq0_master = abc_to_dq0(data[idx[0]], phase)  # 3 phase currents at LC inductors
        vdq0_master = abc_to_dq0(data[idx[2]], phase)  # 3 phase currents at LC inductors

        # set points (sp)
        # isp_abc_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates
        vsp_dq0_master = data[idx[3]]  # convert dq set-points into three-phase abc coordinates

        vsp_abc = dq0_to_abc(data[idx[3]], phase)

        i_mess = idq0_master * self.i_lim
        i_mess_abc = data[idx[0]] * self.i_lim

        SP = vsp_abc * self.lim
        mess = data[idx[2]] * self.lim

        mess_abc = data[idx[2]] * self.lim

        if any(np.abs(mess_abc) > self.lim) or any(np.abs(i_mess_abc) > self.i_lim):
            """
            3rd area - outside valid area - above lim - possible if enough v_DC - DANGEROUS
            +-v_lim -> +-v_DC
            Valid for v_lim OR i_lim exceeded

            V1:
            @ SP = +v_nom AND mess = -v_DC:
                rew = -1; if error = v_DC + v_nom -> Worst case, +v_nom wanted BUT -v_DC measured
            @ SP = -v_nom AND mess = -v_lim
                rew ~ -1/3 - f[(lim-nom)/(nom+v_DC)]
                rew -> -1 - 2/3*(1 - |lim - nom| / (nom+v_DC))
                The latter fraction is quite small but leads to depending on the system less then 2/3 is
                substracted and we have a gap to the 2nd area! :) 

            V2: None is returned to stop the episode (hint: in the env env.abort_reward is given back as reward(?)

            V3: rew = -1
            """

            # V1:
            # rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.v_DC)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma)) / 3

            # V2:
            # if return -> rew = None and in env abort_reward is given to agent
            if self.det_run:
                return -(1 - self.gamma)
            else:
                return

            # V3:
            # rew = (1 - gamma)

        elif all(np.abs(mess_abc) <= self.nom * 1.1):
            # if all(np.abs(mess) <= self.lim*self.nom_region):
            """
            1st area - inside wanted (nom) operation range
            -v_nom -> + v_nom
                rew = 1; if mess = SP
                rew = 1/3; if error = SP-mess = 2*v_nom (worst case without braking out from nom area)
            """
            # devided by 3 because of sums up all 3 phases
            #rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * 2 * (1 - self.gamma) / 3 + (
            #        1 - self.gamma) / 3) / 3

            rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * (1 - self.gamma) ) / 3




        else:
            """
            2nd area
            +-v_nom -> +- v_lim

            @ SP = v_nom AND mess = v_nom (-µV), da if mess > v_nom (hier noch Sicherheitsabstand?)
                rew = 1/3
            @ SP = v_nom AND mess = -v_lim
                rew = -1/3

            """
            #rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.lim)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma) / 3) / 3
            rew = (1 - np.max(np.abs(SP - mess)) / (self.nom + self.lim)) * (1 - self.gamma) / 2 - (1 - self.gamma) / 2

        if any(abs(i_mess_abc) > ((self.i_nom + self.i_lim) / 2)):
            rew = (rew + 1) / 2  # map rew_voltage -> [0,1]

            # Scale rew_voltage with rew_current
            # rew = rew * np.sum((((self.i_nom - i_mess) / (self.i_lim - self.i_nom))+1) ** self.i_exponent) / 3
            rew = rew * (((self.i_nom - max(abs(i_mess_abc))) / (self.i_lim - self.i_nom)) + 1) ** self.i_exponent

            rew = rew * 2 - 1  # map rew -> [-1, 1]

        if rew < -1:
            asd = 1
        return rew  # * (1-0.9)
        # return -np.clip(error.squeeze(), 0, 1e5)

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

        rew = np.sum(-((np.abs(vsp_dq0_master - vdq0_master)) ** self.exponent)) * (1 - self.gamma) / 3

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

        rew = np.sum(-((np.abs(vsp_dq0_master - vdq0_master)) ** 0.5)) * (1 - self.gamma) / 3

        return rew

    def rew_fun_PIPI(self, cols: List[str], data: np.ndarray, risk) -> float:
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

        # i_mess = iabc_master * self.i_lim

        # SP = vsp_abc_master * self.lim
        # mess = vabc_master * self.lim

        if any(np.abs(mess) > self.lim) or any(np.abs(i_mess) > self.i_lim):
            """
            3rd area - outside valid area - above lim - possible if enough v_DC - DANGEROUS
            +-v_lim -> +-v_DC
            Valid for v_lim OR i_lim exceeded

            V1:
            @ SP = +v_nom AND mess = -v_DC:
                rew = -1; if error = v_DC + v_nom -> Worst case, +v_nom wanted BUT -v_DC measured
            @ SP = -v_nom AND mess = -v_lim
                rew ~ -1/3 - f[(lim-nom)/(nom+v_DC)]
                rew -> -1 - 2/3*(1 - |lim - nom| / (nom+v_DC))
                The latter fraction is quite small but leads to depending on the system less then 2/3 is
                substracted and we have a gap to the 2nd area! :) 

            V2: None is returned to stop the episode (hint: in the env env.abort_reward is given back as reward(?)

            V3: rew = -1
            """

            # V1:
            # rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.v_DC)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma)) / 3

            # V2:
            # if return -> rew = None and in env abort_reward is given to agent
            if self.det_run:
                return -(1 - self.gamma)
            else:
                return

            # V3:
            # rew = (1 - gamma)

        elif all(np.abs(mess) <= self.nom * 1.1):
            # if all(np.abs(mess) <= self.lim*self.nom_region):
            """
            1st area - inside wanted (nom) operation range
            -v_nom -> + v_nom
                rew = 1; if mess = SP
                rew = 1/3; if error = SP-mess = 2*v_nom (worst case without braking out from nom area)
            """
            # devided by 3 because of sums up all 3 phases
            # rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * 2 * (1 - self.gamma) / 3 + (
            #        1 - self.gamma) / 3) / 3
            rew = np.sum((1 - (np.abs(SP - mess) / (2 * self.nom)) ** self.exponent) * (1 - self.gamma)) / 3



        else:
            """
            2nd area
            +-v_nom -> +- v_lim

            @ SP = v_nom AND mess = v_nom (-µV), da if mess > v_nom (hier noch Sicherheitsabstand?)
                rew = 1/3
            @ SP = v_nom AND mess = -v_lim
                rew = -1/3

            """
            # rew = np.sum(
            #    (1 - np.abs(SP - mess) / (self.nom + self.lim)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma) / 3) / 3
            rew = (1 - np.max(np.abs(SP - mess)) / (self.nom + self.lim)) * (1 - self.gamma) / 2 - (1 - self.gamma) / 2
        if any(abs(i_mess) > self.i_nom):
            rew = (rew + 1) / 2  # map rew_voltage -> [0,1]

            # Scale rew_voltage with rew_current
            # rew = rew * np.sum((((self.i_nom - i_mess) / (self.i_lim - self.i_nom))+1) ** self.i_exponent) / 3
            rew = rew * (((self.i_nom - max(abs(i_mess))) / (self.i_lim - self.i_nom)) + 1) ** self.i_exponent

            rew = rew * 2 - 1  # map rew -> [-1, 1]

        if rew < -1:
            asd = 1
        return rew  # * (1-0.9)
        # return -np.clip(error.squeeze(), 0, 1e5)
