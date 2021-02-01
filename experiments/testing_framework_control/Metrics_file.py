from math import sqrt
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.signal import argrelextrema


#####################################
# Calculation of Metrics

class Metrics:

    def __init__(self, quantity, ref_value: float, position_steady_state: int, position_settling_time: int,
                 ts: float, max_episode_steps: int):
        """
        :param quantity: analysed quantity from the history (e.g. current idq0)
        :param ref_value: setpoint of the quantity
        :param position_steady_state: step where quantity reached steady state
        :param position_settling_time: step where quantity reached settling time
        :param ts: absolute time resolution of the env
        :param max_episode_steps: number of simulation steps per episode
        """

        self.quantity = quantity  # the class is given any quantity to calculate the metrics
        self.ts = ts  # duration of episodes in seconds
        self.ref_value = ref_value  # set value
        self.upper_bound = 1.02 * ref_value  # upper bound --> important for settling time
        self.lower_bound = 0.98 * ref_value  # lower bound --> important for settling time
        self.position_steady_state = position_steady_state  # passes the steady state position
        self.position_settling_time = position_settling_time  # passes the settling_time_position
        self.max_episode_steps = max_episode_steps
        self.interval_before_load_steps = self.quantity.iloc[
                                          0:position_steady_state]  # creates interval before load steps are applied
        self.max_quantity = 0
        self.n = 5  # important for command 'argrelextrema'
        self.overshoot_available = True

    def overshoot(self):  # calculation of overshoot
        self.interval_before_load_steps['max'] = \
            self.interval_before_load_steps.iloc[
                argrelextrema(self.interval_before_load_steps.values, np.greater_equal, order=self.n)[
                    0]]  # tries to find all maxima before load steps are implemented
        self.max_quantity = self.interval_before_load_steps['max'].max()  # return the highest max
        overshoot = (self.max_quantity / self.ref_value) - 1
        if self.max_quantity > self.ref_value:
            return round(overshoot, 4)
        else:
            self.overshoot_available = False
            sentence_error1 = "No"
            return sentence_error1

    def rise_time(self):  # calculation of rise time
        position_start_rise_time = self.quantity[self.quantity.iloc[:, 0] >= 0.1 * self.ref_value].index[
            0]  # 10% of its final value
        position_end_rise_time = self.quantity[self.quantity.iloc[:, 0] >= 0.9 * self.ref_value].index[
            0]  # 90 % of its final value
        position_rise_time = position_end_rise_time - position_start_rise_time
        rise_time_in_seconds = position_rise_time * self.ts
        return round(rise_time_in_seconds, 4)

    def settling_time(self):  # identification of settling time
        if self.position_settling_time == 0:
            sys.exit("Steady State could not be reached. The controller need to be improved. PROGRAM EXECUTION STOP")
        settling_time_value = self.position_settling_time * self.ts  # is calculated in the class LoadstepCallback
        return settling_time_value

    def settling_time_vd_droop(
            self):  # identification of settling time, only for vd in in tf_primarylevel_vdq_slavefreq.py
        interval_before_steady_state = self.quantity['master.CVVd'].iloc[0:self.position_steady_state]
        for index, row in interval_before_steady_state.iteritems():  # iteration
            if row > self.lower_bound and row < self.upper_bound and self.settling_time_check == False:
                self.settling_time_check = True
                self.position_settling_time_CVVd = index
            if row < self.lower_bound or row > self.upper_bound:
                self.settling_time_check = False

        if self.position_settling_time_CVVd == 0:
            sys.exit("Steady State could not be reached. The controller need to be improved. PROGRAM EXECUTION STOP")
        settling_time_value = self.position_settling_time_CVVd * self.ts
        return settling_time_value

    def RMSE(self):
        Y_true = self.quantity.to_numpy().reshape(
            -1)  # converts and reshapes it into an array
        Y_true = Y_true[~np.isnan(Y_true)]  # drops nan
        Y_pred = [self.ref_value] * (len(
            Y_true))  # creates an list with the set value of the voltage and the length of the real voltages (Y_true)
        Y_pred = np.array(Y_pred)  # converts this list into an array
        return round(sqrt(mean_squared_error(Y_true, Y_pred)), 4)  # returns the RMSE from sklearn

    def steady_state_error(self):  # for a controller with an integral part, steady_state_error may be zero
        last_value_quantity = self.quantity.iloc[self.max_episode_steps - 1]  # the last value of the quantity is stored
        steady_state_error = np.abs(self.ref_value - last_value_quantity[0])  # calculation of the steady-state-error
        return round(steady_state_error, 4)

    def absolute_peak(self):  # absolute peak is calculated
        max_quantity = self.quantity.abs().max()
        return round(max_quantity[0], 4)
