from math import sqrt

import numpy as np
from scipy.signal import argrelextrema
from sklearn.metrics import mean_squared_error


#####################################
# Calculation of Metrics

class Metrics:
    def __init__(self, quantity, ref_value: float, ts: float, max_episode_steps: int, position_steady_state: int = 0,
                 position_settling_time: int = 0):
        """
        :param quantity: analysed quantity from the history (e.g. current idq0)
        :param ref_value: setpoint of the quantity
        :param position_steady_state: step where quantity reached steady state
        :param position_settling_time: step where quantity reached settling time
        :param ts: absolute time resolution of the env
        :param max_episode_steps: number of simulation steps per episode
        """

        self.quantity = quantity
        self.ts = ts
        self.ref_value = ref_value
        # upper bound --> important for settling time
        self.upper_bound = 1.02 * ref_value
        # lower bound --> important for settling time
        self.lower_bound = 0.98 * ref_value

        self.position_steady_state = position_steady_state
        self.position_settling_time = position_settling_time
        self.max_episode_steps = max_episode_steps

        # creates interval before load steps are applied
        self.interval_before_load_steps = self.quantity.iloc[0:position_steady_state]
        self.max_quantity = 0
        # important for command 'argrelextrema'
        self.n = 5

    def overshoot(self):
        """
        calculation of overshoot
        :return:
        """
        # tries to find all maxima before load steps are implemented
        self.interval_before_load_steps['max'] = \
            self.interval_before_load_steps.iloc[
                argrelextrema(self.interval_before_load_steps.values, np.greater_equal, order=self.n)[0]]
        # return the highest max
        self.max_quantity = self.interval_before_load_steps['max'].max()
        overshoot = (self.max_quantity / self.ref_value) - 1
        if self.max_quantity > self.ref_value:
            return round(overshoot, 4)

    def rise_time(self):
        """
        calculation of rise time
        :return:
        """
        # 10% of its final value
        position_start_rise_time = self.quantity[self.quantity.iloc[:, 0] >= 0.1 * self.ref_value].index[0]
        # 90 % of its final value
        position_end_rise_time = self.quantity[self.quantity.iloc[:, 0] >= 0.9 * self.ref_value].index[0]
        position_rise_time = position_end_rise_time - position_start_rise_time
        rise_time_in_seconds = position_rise_time * self.ts
        return round(rise_time_in_seconds, 4)

    def settling_time(self):
        """
        identification of settling time
        :return:
        """
        if self.position_settling_time == 0:
            raise RuntimeError(
                "Steady State could not be reached. The controller need to be improved. PROGRAM EXECUTION STOP")
        # is calculated in the class LoadstepCallback
        return self.position_settling_time * self.ts

    def settling_time_vd_droop(self):
        """
        identification of settling time, only for vd in in tf_primarylevel_vdq_slavefreq.py
        :return:
        """
        interval_before_steady_state = self.quantity['master.CVVd'].iloc[0:self.position_steady_state]

        # find the beginning of the last period settled period
        is_settled = False
        for index, data in interval_before_steady_state.items():
            if self.lower_bound < data < self.upper_bound and not is_settled:
                is_settled = True
                self.position_settling_time_CVVd = index
            else:
                is_settled = False

        if self.position_settling_time_CVVd == 0:
            raise RuntimeError(
                "Steady State could not be reached. The controller need to be improved. PROGRAM EXECUTION STOP")

        return self.position_settling_time_CVVd * self.ts

    def RMSE(self):
        # converts and reshapes it into an array
        Y_true = self.quantity.to_numpy().reshape(-1)
        # drops nan
        Y_true = Y_true[~np.isnan(Y_true)]
        # creates an list with the set value of the voltage and the length of the real voltages (Y_true)
        Y_pred = [self.ref_value] * (len(Y_true))
        # converts this list into an array
        Y_pred = np.array(Y_pred)
        # returns the RMSE from sklearn
        return round(sqrt(mean_squared_error(Y_true, Y_pred)), 4)

    def steady_state_error(self):
        """
        for a controller with an integral part, steady_state_error may be zero
        :return:
        """
        # the last value of the quantity is stored
        last_value_quantity = self.quantity.iloc[self.max_episode_steps - 1]
        # calculation of the steady-state-error
        steady_state_error = np.abs(self.ref_value - last_value_quantity[0])
        return round(steady_state_error, 4)

    def absolute_peak(self):
        """
        absolute peak is calculated
        :return:
        """
        max_quantity = self.quantity.abs().max()
        return round(max_quantity[0], 4)
