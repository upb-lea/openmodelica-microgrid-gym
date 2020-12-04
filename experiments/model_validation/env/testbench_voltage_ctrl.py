from socket import socket

import gym
import paramiko
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import strftime, gmtime, sleep

from paramiko import BadHostKeyException, AuthenticationException, SSHException

from openmodelica_microgrid_gym.util import dq0_to_abc


class TestbenchEnvVoltage(gym.Env):

    viz_modes = {'episode', 'step', None}
    """Set of all valid visualisation modes"""

    def __init__(self, host: str = '131.234.172.139', username: str = 'root', password: str = 'omg',
                 DT: float = 1/20000, executable_script_name: str = 'my_first_hps' ,num_steps: int = 1000,
                 kP: float = 0.052346, kI: float = 15.4072 , kPV: float = 0.018619 , kIV: float = 10.0,
                 ref: float = 10.0, ref2: float =12, f_nom: float = 50.0, i_limit: float = 16,
                 i_nominal: float = 12, v_nominal: float = 20, v_limit = 30, mu=2):

        """
                Environment to connect the code to an FPGA-Controlled test-bench via SSH for votage controll.
                Just for demonstration purpose - different
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
        self.data_obs = np.array(list())
        self.current_step = 0
        self.done = False
        self.i_limit = i_limit
        self.i_nominal = i_nominal
        self.v_nominal = v_nominal
        self.v_limit = v_limit
        self.mu = mu

    @staticmethod
    def __decode_result(ssh_result):

        result = list()
        resultObs = list()
        for line in ssh_result.read().splitlines():

            temp = line.decode("utf-8").split(",")

            if len(temp) == 31:
                #temp.pop(-1)  # Drop the last item

                floats = [float(i) for i in temp]
                # print(floats)
                result.append(floats)
            elif len(temp) == 9:
                # print(temp)
                floatsObs = [float(i) for i in temp]
                # print(floatsObs)
                resultObs.append(floatsObs)
                # count=count+1
            elif len(temp) != 1:
                print(temp)

        N = (len(result))
        decoded_result = np.array(result)

        return [decoded_result, np.array(resultObs)]

    @staticmethod
    def __decode_result_obs(ssh_result):


        resultObs = list()
        for line in ssh_result.read().splitlines():

            temp = line.decode("utf-8").split(",")

            if len(temp) == 9:
                # print(temp)
                floatsObs = [float(i) for i in temp]
                # print(floatsObs)
                resultObs.append(floatsObs)
                # count=count+1
            elif len(temp) != 1:
                print(temp)

        N = (len(resultObs))
        decoded_result = np.array(resultObs)

        return decoded_result

    def rew_fun(self, vabc_meas, vsp_abc) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
        used parameters.
        Takes current measurement and setpoints so calculate the mean-root-error control error and uses a logarithmic
        barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        mu = self.mu

        # setpoints
        #Iabc_SP = dq0_to_abc(Idq0_SP, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        #error = np.sum((np.abs((Vabc_SP - vabc_meas)) / self.v_nominal) ** 0.5, axis=0) #+\
        error = (np.sum((np.abs((vsp_abc - vabc_meas)) / self.v_limit) ** 0.5, axis=0) +\
                -np.sum(mu * np.log(1 - np.maximum(np.abs(vabc_meas) - self.v_nominal, 0) / (self.v_limit - self.v_nominal)), axis=0) \
                 )/ self.max_episode_steps


        return -error.squeeze()

    #def reset(self, kP, kI, kPv, kIv):
    def reset(self, kPv, kIv):
        # toDo: ssh connection not open every episode!
        #self.kP = kP
        #self.kI = kI
        self.kPV = kPv
        self.kIV = kIv

        #self.ssh.connect(self.host, username=self.username, password=self.password)

        count_retries = 0
        connected = False

        #for x in range(10):
        while not connected and count_retries < 10:
            count_retries+=1
            try:
                print('I Try!')
                self.ssh.connect(self.host, username=self.username, password=self.password)
                connected =  True
                #return True
            except:# (BadHostKeyException, AuthenticationException,
                    #SSHException, socket.gaierror) as e:
                #print(e)
                print('Argh')
                sleep(1)

        if count_retries == 10:
            print( 'SSH FUCKED UP!')



        #toDo: get SP and kP/I from agent?
        #str_command = './{} {} {} {} {} {}'.format(self.executable_script_name, self.max_episode_steps, self.kP, self.kI,
        #                                           self.i_ref, self.f_nom)

        str_command = './{} {} {} {} {} {} {} {} {} {}'.format(self.executable_script_name, self.max_episode_steps, np.int(self.max_episode_steps/3-220),
                                                               np.int(self.max_episode_steps *2/ 3-520),
                                                      self.kP, self.kI, self.kPV, self.kIV,
                                                   self.v_nominal,   self.f_nom)

        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(str_command)

        self.data, self.data_obs = self.__decode_result(ssh_stdout)
        #self.data_obs = self.__decode_result_obs(ssh_stdout)

        self.current_step = 0
        self.done = False

        self.ssh.close()


    def rew_fun4D(self, vabc_meas, vsp_abc, iabc_meas, iabc_SP) -> float:
        """
        Defines the reward function for the environment. Uses the observations and setpoints to evaluate the quality of the
        used parameters.
        Takes current measurement and setpoints so calculate the mean-root-error control error and uses a logarithmic
        barrier function in case of violating the current limit. Barrier function is adjustable using parameter mu.

        :param cols: list of variable names of the data
        :param data: observation data from the environment (ControlVariables, e.g. currents and voltages)
        :return: Error as negative reward
        """
        mu = self.mu

        # setpoints
        # Iabc_SP = dq0_to_abc(Idq0_SP, phase)  # convert dq set-points into three-phase abc coordinates

        # control error = mean-root-error (MRE) of reference minus measurement
        # (due to normalization the control error is often around zero -> compared to MSE metric, the MRE provides
        #  better, i.e. more significant,  gradients)
        # plus barrier penalty for violating the current constraint
        # error = np.sum((np.abs((Vabc_SP - vabc_meas)) / self.v_nominal) ** 0.5, axis=0) #+\
        error = (np.sum((np.abs((iabc_SP - iabc_meas)) / self.i_limit) ** 0.5, axis=0) \
                + -np.sum(80 * np.log(1 - np.maximum(np.abs(iabc_meas) - self.i_nominal, 0) / \
                (self.i_limit - self.i_nominal)), axis=0)+
                            np.sum((np.abs((vsp_abc - vabc_meas)) / self.v_limit) ** 0.5, axis=0) + \
                     -np.sum(mu * np.log(
                         1 - np.maximum(np.abs(vabc_meas) - self.v_nominal, 0) / (self.v_limit - self.v_nominal)), axis=0) \
                     ) / self.max_episode_steps

        return -error.squeeze()

    def reset4D(self, kP, kI, kPv, kIv):
        # toDo: ssh connection not open every episode!
        self.kP = kP
        self.kI = kI
        self.kPV = kPv
        self.kIV = kIv

        # self.ssh.connect(self.host, username=self.username, password=self.password)

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
            print('SSH FUCKED UP!')

            # toDo: get SP and kP/I from agent?
            # str_command = './{} {} {} {} {} {}'.format(self.executable_script_name, self.max_episode_steps, self.kP, self.kI,
            #                                           self.i_ref, self.f_nom)

        str_command = './{} {} {} {} {} {} {} {} {} {}'.format(self.executable_script_name, self.max_episode_steps,
                                                                   np.int(self.max_episode_steps / 3 - 220),
                                                                   np.int(self.max_episode_steps * 2 / 3 - 520),
                                                                   self.kP, self.kI, self.kPV, self.kIV,
                                                                   self.v_nominal, self.f_nom)

        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(str_command)

        self.data, self.data_obs = self.__decode_result(ssh_stdout)
            # self.data_obs = self.__decode_result_obs(ssh_stdout)

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


        V_abc_meas = temp_data[[0,1,2]]
        V_abc_SP = temp_data[[10,11,12]]
        I_abc_meas = temp_data[[3,4,5]]
        I_abc_SP = temp_data[[24,25,26]]
        #phase = temp_data[9]

        #reward = self.rew_fun(V_abc_meas, V_abc_SP)
        reward = self.rew_fun4D(V_abc_meas, V_abc_SP, I_abc_meas, I_abc_SP)

        #V_abc_meas = temp_data[[0, 1, 2]]
        # Idq0_SP = np.array([self.i_ref,0,0])
        #V_abc_SP = temp_data[[10, 11, 12]]
        #reward = self.rew_fun(V_abc_meas, V_abc_SP)

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

        """
        If_A = self.data[:, 31];
        If_B = self.data[:, 32];
        If_C = self.data[:, 33];
        Vc_A = self.data[:, 34];
        Vc_B = self.data[:, 35];
        Vc_C = self.data[:, 36];
        """
        Io_A = self.data_obs[:, 6]
        Io_B = self.data_obs[:, 7]
        Io_C = self.data_obs[:, 8]


        # store measurment to dataframe
        df = pd.DataFrame({'V_A': self.data[:, 0],
                           'V_B': self.data[:, 1],
                           'V_C': self.data[:, 2],
                           'I_A': self.data[:, 3],
                           'I_B': self.data[:, 4],
                           'I_C': self.data[:, 5],
                           'I_D': self.data[:, 6],
                           'I_Q': self.data[:, 7],
                           'I_0': self.data[:, 8],
                           'Ph': self.data[:, 9],
                           'SP_A': self.data[:, 10],
                           'SP_B': self.data[:, 11],
                           'SP_C': self.data[:, 12],
                           'm_A': self.data[:, 13],
                           'm_B': self.data[:, 14],
                           'm_C': self.data[:, 15],
                           'm_D': self.data[:, 16],
                           'm_Q': self.data[:, 17],
                           'm_0': self.data[:, 18]})

        #df.to_pickle('Measurement')
        #df.to_pickle('Noise_measurement')

        #plt.plot(t, V_A, t, V_B, t, V_C)
        #plt.ylabel('Voltages (V)')
        #plt.show()

        fig = plt.figure()
        plt.plot(t, V_A, 'b', label=r'$v_{\mathrm{a}}$')
        plt.plot(t, V_B, 'g')
        plt.plot(t, V_C, 'r')
        plt.plot(t, SP_A, 'b--', label=r'$v_{\mathrm{SP,a}}$')
        plt.plot(t, SP_B, 'g--')
        plt.plot(t, SP_C, 'r--')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')
        #plt.title('{}'.format(J))
        plt.grid()
        plt.legend()
        plt.show()
        #time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        #fig.savefig('hardwareTest_plt/{}_abcInductor_currents' + time + '.pdf'.format(J))
        fig.savefig(save_folder +'/J_{}_abcvoltage.pdf'.format(J))
        fig.savefig(save_folder +'/J_{}_abcvoltage.pgf'.format(J))




        fig = plt.figure()
        plt.plot(t, I_A, 'b' , label = r'$i_{\mathrm{a}}$')
        plt.plot(t, I_B, 'g')
        plt.plot(t, I_C, 'r')
        #plt.plot(t, SP_A, 'b--', label = r'$i_{\mathrm{a}}$')
        #plt.plot(t, SP_B, 'g--')
        #plt.plot(t, SP_C, 'r--')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
        #plt.title('{}'.format(J))
        plt.grid()
        plt.legend()
        plt.show()
        #time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        #fig.savefig('hardwareTest_plt/{}_abcInductor_currents' + time + '.pdf'.format(J))
        fig.savefig(save_folder +'/J_{}_abcInductor_currents.pdf'.format(J))
        fig.savefig(save_folder +'/J_{}_abcInductor_currents.pgf'.format(J))


        fig = plt.figure()
        plt.plot(t, I_D, t, I_Q, t, I_0)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        plt.title('{}'.format(J))
        #plt.ylim([-3, 16])
        plt.grid()
        plt.show()
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        #fig.savefig('hardwareTest_plt/dq0Inductor_currents' + time + '.pdf')
        #fig.savefig('Paper_meas/J_{}_dq0Inductor_currents.pdf'.format(J))

        fig = plt.figure()
        plt.plot(t, I_D, t, I_Q, t, I_0)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$i_{\mathrm{dq0}}\,/\,\mathrm{A}$')
        plt.title('{}'.format(J))
        # plt.ylim([-3, 16])
        plt.grid()
        plt.show()
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())


        fig = plt.figure()
        plt.plot(t, V_D, t, V_Q, t, V_0)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        plt.title('{}'.format(J))
        # plt.ylim([-3, 16])
        #plt.xlim([0, 0.025])
        plt.grid()
        plt.show()
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        fig.savefig(save_folder + '/J_{}_dq0voltage.pdf'.format(J))
        fig.savefig(save_folder + '/J_{}_dq0voltage.pgf'.format(J))

        """
        fig = plt.figure()
        plt.plot(t, V_D, t, V_Q, t, V_0)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
        plt.title('{}'.format(J))
        # plt.ylim([-3, 16])
        plt.xlim([0, 0.025])
        plt.grid()
        plt.show()
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        """


        fig = plt.figure()
        plt.plot(t, Io_A, 'b', label=r'$i_{\mathrm{a}}$')
        plt.plot(t, Io_B, 'g')
        plt.plot(t, Io_C, 'r')
        # plt.plot(t, SP_A, 'b--', label = r'$i_{\mathrm{a}}$')
        # plt.plot(t, SP_B, 'g--')
        # plt.plot(t, SP_C, 'r--')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$i_{\mathrm{abc,est}}\,/\,\mathrm{A}$')
        # plt.title('{}'.format(J))
        plt.grid()
        plt.legend()
        plt.show()
