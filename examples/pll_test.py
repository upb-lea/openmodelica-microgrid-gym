#####################################
# Example using a FMU by OpenModelica as gym environment containing two inverters, each connected via an LC-filter to
# supply in parallel a RL load.
# This example uses the available standard controllers as defined in the 'auxiliaries' folder.
# One inverter is set up as voltage forming inverter with a direct droop controller.
# The other controller is used as current sourcing inverter with an inverse droop controller which reacts on the
# frequency and voltage change due to its droop control parameters by a power/reactive power change.

import logging
from functools import partial
from functools import reduce

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openmodelica_microgrid_gym import Runner
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams
from openmodelica_microgrid_gym.net import Network
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import PLLParams, PLL
from typing import List, Mapping, Union

import numpy as np

from openmodelica_microgrid_gym.aux_ctl import Controller

# list_resistance = [13,15,17,26,30] #training
# list_inductance = [0.00065, 0.0012, 0.0014, 0.0017, 0.0023] #training
# list_resistance = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # training
# list_inductance = [0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022, 0.0024]
#
# list_resistance = [12]
# list_inductance = [0.0006]
# list_df=[]
list_resistance = [15,23,28,25] #training
list_inductance = [0.002, 0.0025, 0.0008, 0.0013]
i = 0
# df_all_quantity=pd.DataFrame()

list_factor = [0.75, 1.75]
# list_factor = [0.5, 1.5, 2]
# list_factor=[0.5]

class PllAgent(StaticControlAgent):

    def __init__(self, pllPIParams: PLLParams, ts, ctrls: List[Controller],
                 obs_template: Mapping[str, List[Union[List[str], np.ndarray]]],
                 obs_varnames: List[str] = None, **kwargs):

        self._ts = ts

        super().__init__(ctrls, obs_template, obs_varnames, **kwargs)

        self._pll = PLL(pllPIParams, self._ts)
        self.freq_load_store=[]


    def measure(self, state) -> np.ndarray:

        obs = super().measure(state)
        if len(self.env.history.data)!=0:
            v_load1= self.env.history.df[['rl1.resistor1.v']].to_numpy()[-1]
            v_load2= self.env.history.df[['rl1.resistor2.v']].to_numpy()[-1]
            v_load3= self.env.history.df[['rl1.resistor3.v']].to_numpy()[-1]
            asd = 1
            _,freq,_=self._pll.step(np.array([v_load1, v_load2, v_load3]))
            self.freq_load_store.append(freq)
        return obs



for (resistance, inductance) in zip(list_resistance, list_inductance):
    if list_resistance == list_resistance[0]:
        list_factor[0] = 0.7
    else:
        list_factor[0] = 0.5

    for factor in list_factor:
        # Simulation definitions
        max_episode_steps = 100  # number of simulation steps per episode
        num_episodes = 1  # number of simulation episodes
        # (here, only 1 episode makes sense since simulation conditions don't change in this example)
        DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
        QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V
        net = Network.load('../net/microgrid.yaml')
        delta_t = net.ts  # simulation time step size / s
        freq_nom = net.freq_nom  # nominal grid frequency / Hz
        v_nom = net.v_nom  # nominal grid voltage / V
        v_DC = net['inverter1'].v_DC  # DC-link voltage / V; will be set as model parameter in the FMU
        i_lim = net['inverter1'].i_lim  # inverter current limit / A
        i_nom = net['inverter1'].i_nom  # nominal inverter current / A

        logging.basicConfig()


        def load_step(t, gain):
            """
            Doubles the load parameters
            :param t:
            :param gain: device parameter
            :return: Dictionary with load parameters
            """
            # Defines a load step after 0.2 s
            return 1 * gain if t < .25 else factor * gain


        if __name__ == '__main__':
            ctrl = []  # Empty dict which shall include all controllers

            #####################################
            # Define the voltage forming inverter as master
            # Voltage control PI gain parameters for the voltage sourcing inverter
            voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-i_lim, i_lim))
            # Current control PI gain parameters for the voltage sourcing inverter
            current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
            # Droop characteristic for the active power Watt/Hz, delta_t
            droop_param = DroopParams(DroopGain, 0.005, freq_nom)
            # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
            qdroop_param = DroopParams(QDroopGain, 0.002, v_nom)
            # Add to dict
            ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, droop_param,
                                                    qdroop_param, ts_sim=delta_t, name='master'))

            #####################################
            # Define the current sourcing inverter as slave
            # Current control PI gain parameters for the current sourcing inverter
            current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
            # PI gain parameters for the PLL in the current forming inverter
            pll_params = PLLParams(kP=10, kI=200, limits=None, f_nom=freq_nom)
            # Droop characteristic for the active power Watts/Hz, W.s/Hz
            droop_param = InverseDroopParams(DroopGain, delta_t, freq_nom, tau_filt=0.04)
            # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
            qdroop_param = InverseDroopParams(50, delta_t, v_nom, tau_filt=0.01)
            # Add to dict
            ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, i_lim,
                                                      droop_param, qdroop_param, ts_sim=delta_t, name='slave'))

            # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
            agent = PllAgent(pll_params, delta_t, ctrl, {'master': [[f'lc1.inductor{k}.i' for k in '123'],
                                                         [f'lc1.capacitor{k}.v' for k in '123']],
                                              'slave': [[f'lcl1.inductor{k}.i' for k in '123'],
                                                        [f'lcl1.capacitor{k}.v' for k in '123'],
                                                        np.zeros(3)]})

            # Define the environment
            env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                           viz_mode='episode',
                           # viz_cols=['*.m[dq0]', 'slave.freq', 'lcl1.*'],
                           # viz_cols=['master.inst*', 'slave.inst*', 'lcl1.*', 'lc1.*', 'slave.freq', 'master.freq'],
                           viz_cols=[],
                           log_level=logging.INFO,
                           max_episode_steps=max_episode_steps,
                           model_params={'rl1.resistor1.R': partial(load_step, gain=resistance),
                                         'rl1.resistor2.R': partial(load_step, gain=resistance),
                                         'rl1.resistor3.R': partial(load_step, gain=resistance),
                                         'rl1.inductor1.L': partial(load_step, gain=inductance),
                                         'rl1.inductor2.L': partial(load_step, gain=inductance),
                                         'rl1.inductor3.L': partial(load_step, gain=inductance)
                                         },
                           model_path='../omg_grid/grid.microgrid.fmu',
                           net=net
                           )

            # User runner to execute num_episodes-times episodes of the env controlled by the agent
            runner = Runner(agent, env)
            runner.run(num_episodes, visualise=True)
            plt.close('all')


        slave_freq=env.history.df[['slave.freq']]
        load_freq=agent.freq_load_store

        master_freq = env.history.df[['master.freq']]
        plt.plot(slave_freq, label='slave_freq')
        plt.legend()
        plt.show()
        master_instPow = env.history.df[['master.instPow']]
        master_instPow = master_instPow / 3
        master_instQ = env.history.df[['master.instQ']]
        master_instQ = master_instQ / 3
        slave_freq = env.history.df[['slave.freq']]
        slave_instPow = env.history.df[['slave.instPow']]
        slave_instPow = slave_instPow / 3
        slave_instQ = env.history.df[['slave.instQ']]
        slave_instQ = slave_instQ / 3

        df_v_lcl1_1 = env.history.df[['lcl1.capacitor1.v']]
        df_v_lcl1_2 = env.history.df[['lcl1.capacitor2.v']]
        df_v_lcl1_3 = env.history.df[['lcl1.capacitor3.v']]

        df_v_lc1_1 = env.history.df[['lc1.capacitor1.v']]
        df_v_lc1_2 = env.history.df[['lc1.capacitor2.v']]
        df_v_lc1_3 = env.history.df[['lc1.capacitor3.v']]

        df_rl1_inductor1_v = env.history.df[['rl1.inductor1.v']]
        df_rl1_inductor2_v = env.history.df[['rl1.inductor2.v']]
        df_rl1_inductor3_v = env.history.df[['rl1.inductor3.v']]

        df_rl1_resistor1_v = env.history.df[['rl1.resistor1.v']]
        df_rl1_resistor2_v = env.history.df[['rl1.resistor2.v']]
        df_rl1_resistor3_v = env.history.df[['rl1.resistor3.v']]

        df_rl1_inductor1_i = env.history.df[['rl1.inductor1.i']]
        df_rl1_inductor2_i = env.history.df[['rl1.inductor2.i']]
        df_rl1_inductor3_i = env.history.df[['rl1.inductor3.i']]

        # Calculation of the instRMS ActivePower of the Load#
        instRMS_Voltage_Resistor_Load = (df_rl1_resistor1_v['rl1.resistor1.v']) ** 2 + (
        df_rl1_resistor2_v['rl1.resistor2.v']) ** 2 + (df_rl1_resistor3_v['rl1.resistor3.v']) ** 2
        instRMS_Voltage_Resistor_Load = np.sqrt(instRMS_Voltage_Resistor_Load) / np.sqrt(3)
        instRMS_Current_Inductor_Load = (df_rl1_inductor1_i['rl1.inductor1.i']) ** 2 + (
        df_rl1_inductor2_i['rl1.inductor2.i']) ** 2 + (df_rl1_inductor3_i['rl1.inductor3.i']) ** 2
        instRMS_Current_Inductor_Load = np.sqrt(instRMS_Current_Inductor_Load) / np.sqrt(3)

        load_active_Power = instRMS_Current_Inductor_Load * instRMS_Voltage_Resistor_Load
        load_active_Power = pd.Series(load_active_Power,
                                      name="Load.activePower")
        load_active_Power = load_active_Power.to_frame()

        # Calculation of the instRMS ReactivePower of the Load#
        instRMS_Voltage_Inductor_Load = (df_rl1_inductor1_v['rl1.inductor1.v']) ** 2 + (
            df_rl1_inductor2_v['rl1.inductor2.v']) ** 2 + (df_rl1_inductor3_v['rl1.inductor3.v']) ** 2
        instRMS_Voltage_Inductor_Load = np.sqrt(instRMS_Voltage_Inductor_Load) / np.sqrt(3)
        load_reactive_Power = instRMS_Current_Inductor_Load * instRMS_Voltage_Inductor_Load
        load_reactive_Power = pd.Series(load_reactive_Power,
                                        name="load.reactivePower")
        load_reactive_Power = load_reactive_Power.to_frame()

        # Calculation of the instRMS of the Load Voltage#
        load_voltage = (df_rl1_inductor1_v['rl1.inductor1.v'] + df_rl1_resistor1_v['rl1.resistor1.v']) ** 2 + (
                    df_rl1_inductor2_v['rl1.inductor2.v'] + df_rl1_resistor2_v['rl1.resistor2.v']) ** 2 + (
                                   df_rl1_inductor3_v['rl1.inductor3.v'] + df_rl1_resistor3_v['rl1.resistor3.v']) ** 2
        load_voltage = np.sqrt(load_voltage) / np.sqrt(3)
        load_voltage = pd.Series(load_voltage,
                                 name="load.voltage")
        load_voltage = load_voltage.to_frame()

        ##Calculation of the instRMS of the Master Voltage##
        master_Voltage = df_v_lcl1_1['lcl1.capacitor1.v'] * df_v_lcl1_1['lcl1.capacitor1.v'] + df_v_lcl1_2[
            'lcl1.capacitor2.v'] * df_v_lcl1_2['lcl1.capacitor2.v'] + df_v_lcl1_3['lcl1.capacitor3.v'] * df_v_lcl1_3[
                             'lcl1.capacitor3.v']
        master_Voltage = np.sqrt(master_Voltage) / np.sqrt(3)
        master_Voltage = pd.Series(master_Voltage,
                                   name="master.voltage")
        master_Voltage = master_Voltage.to_frame()

        ##Calculation of the instRMS of the Slave Voltage##
        slave_voltage = df_v_lc1_1['lc1.capacitor1.v'] * df_v_lc1_1['lc1.capacitor1.v'] + df_v_lc1_2[
            'lc1.capacitor2.v'] * df_v_lc1_2['lc1.capacitor2.v'] + df_v_lc1_3['lc1.capacitor3.v'] * df_v_lc1_3[
                            'lc1.capacitor3.v']
        slave_voltage = np.sqrt(slave_voltage) / np.sqrt(3)
        slave_voltage = pd.Series(slave_voltage,
                                  name="slave.voltage")
        slave_voltage = slave_voltage.to_frame()
        df_all_quantity_temp = pd.DataFrame().join(
            [master_freq, master_instPow, master_instQ, master_Voltage, slave_freq, slave_instPow, slave_instQ, slave_voltage,
             load_voltage, load_active_Power, load_reactive_Power], how="outer")

        if i == 0:
            df_all_quantity = df_all_quantity_temp
            i = i + 1
        else:
            df_all_quantity = pd.concat([df_all_quantity, df_all_quantity_temp], axis=1, ignore_index=False)
            # df_all_quantity=pd.DataFrame().join([df_all_quantity, df_all_quantity_temp], how="outer")


df_all_quantity.to_pickle('Microgrid_Data_Test')
