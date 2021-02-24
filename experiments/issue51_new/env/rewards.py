import numpy as np
from openmodelica_microgrid_gym.util import nested_map
from typing import List


class Reward:
    def __init__(self, nom, lim, v_DC, gamma, det_run=False, nom_region=1.1, use_gamma_normalization=1):
        self._idx = None
        self.nom = nom
        self.lim = lim
        self.v_DC = v_DC
        self.use_gamma_normalization = use_gamma_normalization
        if self.use_gamma_normalization == 1:
            self.gamma = gamma
        else:
            self.gamma = 0
        self.nom_region = nom_region
        self.det_run = det_run

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n),
                [[f'lc.inductor{k}.i' for k in '123'], [f'inverter1.i_ref.{k}' for k in '012'],
                 [f'lc.capacitor{k}.v' for k in '123'], [f'inverter1.v_ref.{k}' for k in '012']])

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

        if all(np.abs(mess) <= self.nom*1.1):
        #if all(np.abs(mess) <= self.lim*self.nom_region):
            """
            1st area - inside wanted (nom) operation range
            -v_nom -> + v_nom
                rew = 1; if mess = SP
                rew = 1/3; if error = SP-mess = 2*v_nom (worst case without braking out from nom area)
            """
            rew = np.sum((1 - np.abs(SP - mess) / (2 * self.nom)) * 2 * (1 - self.gamma) / 3 + (1 - self.gamma) / 3) / 3

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

        # elif any(np.abs(vabc_master) > v_DC):
        #    rew = (1-gamma)

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
                (1 - np.abs(SP - mess) / (self.nom + self.lim)) * 2 * (1 - self.gamma) / 3 - (1 - self.gamma) / 3) / 3

        return rew  # * (1-0.9)
        # return -np.clip(error.squeeze(), 0, 1e5)
