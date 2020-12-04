from typing import List

import numpy as np

from openmodelica_microgrid_gym.util import nested_map, dq0_to_abc


class Reward:

    def __init__(self, i_limit: float = 15, i_nominal: float = 10, v_limit: float = 500,
                 v_nominal: float = 230 * np.sqrt(2), mu_c: float = 1, mu_v: float = 1, max_episode_steps: float = 1000,
                 obs_dict: List = None, funnel: np.ndarray = None):
        """
        Reward class including different reward functions useable for current and voltage controller evaluation
        :param i_limit: Current limit of inverter
        :param i_nominal: Nominal current of inverter
        :param v_limit: Voltage limit of inverter
        :param v_nominal: Nominal voltage of inverter
        :param mu_c: Weighting factor for barrier to punish over-current
        :param mu_v: Weighting factor for barrier to punish over-voltage
        :param max_episode_steps: number of episode steps
        :param funnel: Defines area around setpoint in which a different reward function is valid to punish leaving
                       funnel different
        """

        self._idx = None
        self.i_limit = i_limit
        self.i_nominal = i_nominal
        self.v_limit = v_limit
        self.v_nominal = v_nominal
        self.mu_c = mu_c
        self.mu_v = mu_v
        self.max_episode_steps = max_episode_steps
        self.funnel = funnel
        self.obs_dict = obs_dict

    def set_idx(self, obs):
        if self._idx is None:
            self._idx = nested_map(
                lambda n: obs.index(n), self.obs_dict)
            # [[f'lc.inductor{k}.i' for k in '123'], 'master.phase', [f'master.SPI{k}' for k in 'dq0'],
            # [f'lc.capacitor{k}.v' for k in '123'], [f'master.SPV{k}' for k in 'dq0'],
            # [f'master.CVV{k}' for k in 'dq0']])

    def rew_fun_c(self, cols: List[str], data: np.ndarray) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
        used parameters.
        Takes current measurement and setpoints so calculate the mean-root-error control error and uses a logarithmic
        barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        self.set_idx(cols)
        idx = self._idx

        i_abc_master = data[idx[0]]  # 3 phase currents at LC inductors
        phase = data[idx[1]]  # phase from the master controller needed for transformation

        # setpoints
        is_pdq0_master = data[idx[2]]  # setting dq reference
        i_sp_abc_master = dq0_to_abc(is_pdq0_master,
                                     phase)  # +0.417e-4*50)  # convert dq set-points into three-phase abc coordinates

        # Idq0_master = data[idx[0]]
        # ISPdq0_master = data[idx[1]]  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = (np.sum((np.abs((i_sp_abc_master - i_abc_master)) / self.i_limit) ** 0.5, axis=0)
                 + -np.sum(self.mu_c * np.log(1 - np.maximum(np.abs(i_abc_master) - self.i_nominal, 0) /
                                              (self.i_limit - self.i_nominal)), axis=0)) / self.max_episode_steps

        return -error.squeeze()

    def rew_fun_v(self, cols: List[str], data: np.ndarray, risk) -> float:
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

        phase = data[idx[1]]  # phase from the master controller needed for transformation
        v_abc_master = data[idx[3]]  # 3 phase currents at LC inductors

        # set points (sp)
        vsp_dq0_master = data[idx[4]]  # setting dq voltage reference
        vsp_abc_master = dq0_to_abc(vsp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = (np.sum((np.abs((vsp_abc_master - v_abc_master)) / self.v_limit) ** 0.5, axis=0) + -np.sum(
            self.mu_v * np.log(
                1 - np.maximum(np.abs(v_abc_master) - self.v_nominal, 0) / (self.v_limit - self.v_nominal)),
            axis=0)) \
                / self.max_episode_steps

        return -error.squeeze()

    def rew_fun_vc(self, cols: List[str], data: np.ndarray) -> float:
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
        phase = data[idx[1]]  # phase from the master controller needed for transformation
        vabc_master = data[idx[3]]  # 3 phase currents at LC inductors

        # set points (sp)
        isp_dq0_master = data[idx[2]]  # setting dq current reference
        isp_abc_master = dq0_to_abc(isp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates
        vsp_dq0_master = data[idx[4]]  # setting dq voltage reference
        vsp_abc_master = dq0_to_abc(vsp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint

        error = (np.sum((np.abs((isp_abc_master - iabc_master)) / self.i_limit) ** 0.5, axis=0)
                 + -np.sum(self.mu_c * np.log(1 - np.maximum(np.abs(iabc_master) - self.i_nominal, 0) /
                                              (self.i_limit - self.i_nominal)), axis=0) +
                 np.sum((np.abs((vsp_abc_master - vabc_master)) / self.v_limit) ** 0.5, axis=0) \
                 + -np.sum(self.mu_v * np.log(1 - np.maximum(np.abs(vabc_master) - self.v_nominal, 0) /
                                              (self.v_limit - self.v_nominal)), axis=0)) \
                / self.max_episode_steps

        return -error.squeeze()

    def rew_fun_funnel(self, cols: List[str], data: np.ndarray) -> float:
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

        phase = data[idx[1]]  # phase from the master controller needed for transformation
        vabc_master = data[idx[3]]  # 3 phase currents at LC inductors
        vdq0_master = data[idx[5]]

        # set points (sp)
        vsp_dq0_master = data[idx[4]]  # setting dq voltage reference
        vsp_abc_master = dq0_to_abc(vsp_dq0_master, phase)  # convert dq set-points into three-phase abc coordinates

        # calculate
        err = vdq0_master - vsp_dq0_master

        if (abs(err) > self.funnel).any():
            # E2 + Offset
            error = (np.sum((np.abs((vsp_abc_master - vabc_master)) / self.v_limit) ** 0.5,
                            axis=0) + 20) / self.max_episode_steps
        else:
            # E1 <! E2 and E1 << Offset
            error = np.sum((np.abs((vsp_abc_master - vabc_master)) / self.v_limit) ** 2,
                           axis=0) / self.max_episode_steps

        return -error.squeeze()
