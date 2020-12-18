import numpy as np

N_PHASE = 3


class Observer():
    def __init__(self):
        pass

    def cal_estimate(self, y, u):
        """
        Estimtates output values depending on measured outputs y and intputs u

        Our exemplary application:
        x_hat = [iL_abc, vC_abc, iLoad_abc]
        y = [iL_abc_mess, vC_abc_mess, 0]
        C = [[1, 0, 0], [0, 1, 0], [0, 0, 0]] -> y_hat = [iL_hat, vC_hat, 0]
        """
        pass


class Lueneberger(Observer):

    def __init__(self, A, B, C, L, ts, vDC):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        self._ts = ts
        self.vDC = vDC
        self.last_x_estimate = np.zeros([3, 1])

    def cal_estimate(self, y, u):
        """
        Estimtates output values depending on measured outputs y and intputs u

        Our exemplary application:
        x_hat = [iL_abc, vC_abc, iLoad_abc]
        y = [iL_abc_mess, vC_abc_mess, 0]
        C = [[1, 0, 0], [0, 1, 0], [0, 0, 0]] -> y_hat = [iL_hat, vC_hat, 0]
        """

        y_estimate = self.C @ self.last_x_estimate

        uB = self.L @ (np.array(y).reshape([2, 1]) - y_estimate)

        # times vCD because vDC was shifted to the OM-env
        model_input = self.B * u * self.vDC

        dx_estimate = model_input + uB + self.A @ self.last_x_estimate

        self.last_x_estimate = self.last_x_estimate + dx_estimate * self._ts

        return self.last_x_estimate[-1]
