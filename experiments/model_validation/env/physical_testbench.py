import gym
import paramiko
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import strftime, gmtime, sleep


class TestbenchEnv(gym.Env):
    viz_modes = {'episode', 'step', None}
    """Set of all valid visualisation modes"""

    def __init__(self, host: str = '131.234.172.139', username: str = 'root', password: str = 'omg',
                 DT: float = 1 / 20000, executable_script_name: str = 'my_first_hps', num_steps: int = 1000,
                 kP: float = 0.01, kI: float = 5.0, kPV: float = 0.01, kIV: float = 5.0, ref: float = 10.0,
                 ref2: float = 12, f_nom: float = 50.0, i_limit: float = 25,
                 i_nominal: float = 15, v_nominal: float = 20, mu=2):

        """
        Environment to connect the code to an FPGA-Controlled test-bench via SSH.
        Just for demonstration purpose
        """

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.host = host
        self.username = username
        self.password = password
        self.DT = DT
        self.executable_script_name = executable_script_name
        self.max_episode_steps = num_steps
        self.kP = kP
        self.kI = kI
        self.kPV = kPV
        self.kIV = kIV
        self.ref = ref
        self.ref2 = ref2
        self.f_nom = f_nom
        self.data = np.array(list())
        self.current_step = 0
        self.done = False
        self.i_limit = i_limit
        self.i_nominal = i_nominal
        self.v_nominal = v_nominal
        self.mu = mu

    @staticmethod
    def __decode_result(ssh_result):
        """
        Methode to decode the data sent from FPGA
        """

        result = list()
        for line in ssh_result.read().splitlines():

            temp = line.decode("utf-8").split(",")

            if len(temp) == 31:
                # temp.pop(-1)  # Drop the last item

                floats = [float(i) for i in temp]
                # print(floats)
                result.append(floats)
            elif len(temp) != 1:
                print(temp)

        decoded_result = np.array(result)

        return decoded_result

    def rew_fun(self, Iabc_meas, Iabc_SP) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
        used parameters.
        Takes current measurement and setpoints so calculate the mean-root-error control error and uses a logarithmic
        barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

        :param Iabc_meas:
        :param Iabc_SP:
        :return: Error as negative reward
        """
        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        error = (np.sum((np.abs((Iabc_SP - Iabc_meas)) / self.i_limit) ** 0.5, axis=0)
                 + -np.sum(self.mu * np.log(1 - np.maximum(np.abs(Iabc_meas) - self.i_nominal, 0) /
                                            (self.i_limit - self.i_nominal)), axis=0)
                 ) / self.max_episode_steps

        return -error.squeeze()

    def reset(self, kP, kI):
        # toDo: ssh connection not open every episode!
        self.kP = kP
        self.kI = kI

        count_retries = 0
        connected = False

        # for x in range(10):
        while not connected and count_retries < 10:
            count_retries += 1
            try:
                print('I Try!')
                self.ssh.connect(self.host, username=self.username, password=self.password)
                connected = True
                # return True
            except:  # (BadHostKeyException, AuthenticationException,
                # SSHException, socket.gaierror) as e:
                # print(e)
                print('Argh')
                sleep(1)

        if count_retries == 10:
            print('SSH connection not possible!')

        str_command = './{} -u 100 -n {} -i {} -f {} -1 {} -2 {} -E'.format(self.executable_script_name,
                                                                            self.max_episode_steps, self.ref,
                                                                            self.f_nom, self.kP,
                                                                            self.kI
                                                                            )

        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(str_command)

        self.data = self.__decode_result(ssh_stdout)

        self.current_step = 0
        self.done = False

        self.ssh.close()

    def step(self):
        """
        Takes measured data and returns stepwise
        Measured data recorded in reset -> controller part of env
        """
        temp_data = self.data[self.current_step]
        self.current_step += 1

        i_abc_meas = temp_data[[3, 4, 5]]
        i_abc_sp = temp_data[[9, 10, 11]]

        reward = self.rew_fun(i_abc_meas, i_abc_sp)

        if self.current_step == self.max_episode_steps:
            self.done = True

        info = []

        return temp_data, reward, self.done, info

    def render(self, J, save_folder):

        N = (len(self.data))
        t = np.linspace(0, N * self.DT, N)

        V_A = self.data[:, 0]
        V_B = self.data[:, 1]
        V_C = self.data[:, 2]
        I_A = self.data[:, 3]
        I_B = self.data[:, 4]
        I_C = self.data[:, 5]
        I_D = self.data[:, 6]
        I_Q = self.data[:, 7]
        I_0 = self.data[:, 8]
        SP_A = self.data[:, 9]
        SP_B = self.data[:, 10]
        SP_C = self.data[:, 11]
        m_A = self.data[:, 12]
        m_B = self.data[:, 13]
        m_C = self.data[:, 14]
        m_D = self.data[:, 15]
        m_Q = self.data[:, 16]
        m_0 = self.data[:, 17]
        ICont_Out_D = self.data[:, 18]
        ICont_Out_Q = self.data[:, 19]
        ICont_Out_0 = self.data[:, 20]
        UCont_Out_D = self.data[:, 21]
        UCont_Out_Q = self.data[:, 22]
        UCont_Out_0 = self.data[:, 23]
        ISP_A = self.data[:, 24]
        ISP_B = self.data[:, 25]
        ISP_C = self.data[:, 26]
        V_D = self.data[:, 27]
        V_Q = self.data[:, 28]
        V_0 = self.data[:, 29]
        Ph = self.data[:, 30]

        # store measurement to dataframe
        df = pd.DataFrame({
            'V_A': self.data[:, 0],
            'V_B': self.data[:, 1],
            'V_C': self.data[:, 2],
            'I_A': self.data[:, 3],
            'I_B': self.data[:, 4],
            'I_C': self.data[:, 5],
            'I_D': self.data[:, 6],
            'I_Q': self.data[:, 7],
            'I_0': self.data[:, 8],
            'Ph': self.data[:, 30]
        })


        # df.to_pickle('Measurement')
        # df.to_pickle('Noise_measurement')

        """
        fig = plt.figure()
        plt.plot(t, V_A, 'b', label=r'$v_{\mathrm{a}}$')
        plt.plot(t, V_B, 'g')
        plt.plot(t, V_C, 'r')
        plt.plot(t, SP_A, 'b--', label=r'$v_{\mathrm{SP,a}}$')
        plt.plot(t, SP_B, 'g--')
        plt.plot(t, SP_C, 'r--')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        plt.title('{}'.format(J))
        plt.grid()
        plt.legend()
        plt.show()
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # fig.savefig('Paper_CC_meas3/{}_abcInductor_currents' + time + '.pdf'.format(J))
        fig.savefig('Paper_CC_meas3/J_{}_abcvoltage.pdf'.format(J))
        """

        fig = plt.figure()
        plt.plot(t, I_A, 'b', label=r'Measurement')
        plt.plot(t, I_B, 'g')
        plt.plot(t, I_C, 'r')
        plt.plot(t, SP_A, 'b--', label=r'Setpoint')
        plt.plot(t, SP_B, 'g--')
        plt.plot(t, SP_C, 'r--')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        # plt.title('{}'.format(J))
        plt.grid()
        plt.legend()
        plt.show()
        fig.savefig(save_folder + '/J_{}_abcInductor_currents.pdf'.format(J))
        fig.savefig(save_folder + '/J_{}_abcInductor_currents.pgf'.format(J))

        fig = plt.figure()
        plt.plot(t, I_D, t, I_Q, t, I_0)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
        # plt.title('{}'.format(J))
        # plt.ylim([-3, 16])
        plt.grid()
        plt.show()
        fig.savefig(save_folder + '/J_{}_dq0Inductor_currents.pdf'.format(J))
        fig.savefig(save_folder + '/J_{}_dq0Inductor_currents.pgf'.format(J))

        fig = plt.figure()
        plt.plot(t, m_D, t, m_Q, t, m_0)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
        # plt.title('{}'.format(J))
        # plt.ylim([-3, 16])
        plt.grid()
        plt.show()
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # fig.savefig('hardwareTest_plt/dq0Inductor_currents' + time + '.pdf')
        fig.savefig(save_folder + '/J_{}_mdq0.pdf'.format(J))
        fig.savefig(save_folder + '/J_{}_mdq0.pgf'.format(J))
