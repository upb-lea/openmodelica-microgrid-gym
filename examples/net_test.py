import gym
import numpy as np
import pandas as pd
from openmodelica_microgrid_gym import Agent, Runner
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import PI_params, DroopParams, MultiPhaseDQ0PIPIController, \
    MultiPhaseDQCurrentController, InverseDroopParams, PLLParams

# Simulation definitions
delta_t = 0.5e-4  # simulation time step size / s
max_episode_steps = 3000  # number of simulation steps per episode
num_episodes = 1  # number of simulation episodes
# (here, only 1 episode makes sense since simulation conditions don't change in this example)
v_DC = 1000  # DC-link voltage / V; will be set as model parameter in the fmu
nomFreq = 50  # grid frequency / Hz
nomVoltPeak = 230 * 1.414  # nominal grid voltage / V
iLimit = 30  # current limit / A
iNominal = 20  # nominal current / A
DroopGain = 40000.0  # virtual droop gain for active power / W/Hz
QDroopGain = 1000.0  # virtual droop gain for reactive power / VAR/V

class RndAgent(Agent):
    def act(self, obs: pd.Series) -> np.ndarray:
        return self.env.action_space.sample()

if __name__ == '__main__':
    ctrl = []  # Empty dict which shall include all controllers

    #####################################
    # Define the voltage forming inverter as master
    # Voltage control PI gain parameters for the voltage sourcing inverter
    voltage_dqp_iparams = PI_params(kP=0.025, kI=60, limits=(-iLimit, iLimit))
    # Current control PI gain parameters for the voltage sourcing inverter
    current_dqp_iparams = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    # Droop characteristic for the active power Watt/Hz, delta_t
    droop_param = DroopParams(DroopGain, 0.005, nomFreq)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    # Add to dict
    ctrl.append(MultiPhaseDQ0PIPIController(voltage_dqp_iparams, current_dqp_iparams, delta_t, droop_param,
                                            qdroop_param, name='master'))

    #####################################
    # Define the current sourcing inverter as slave
    # Current control PI gain parameters for the current sourcing inverter
    current_dqp_iparams = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    # PI gain parameters for the PLL in the current forming inverter
    pll_params = PLLParams(kP=10, kI=200, limits=None, f_nom=nomFreq)
    # Droop characteristic for the active power Watts/Hz, W.s/Hz
    droop_param = InverseDroopParams(DroopGain, delta_t, nomFreq, tau_filt=0.04)
    # Droop characteristic for the reactive power VAR/Volt Var.s/Volt
    qdroop_param = InverseDroopParams(50, delta_t, nomVoltPeak, tau_filt=0.01)
    # Add to dict
    ctrl.append(MultiPhaseDQCurrentController(current_dqp_iparams, pll_params, delta_t, iLimit,
                                              droop_param, qdroop_param, name='slave'))


    # Define the agent as StaticControlAgent which performs the basic controller steps for every environment set
    agent = StaticControlAgent(ctrl, {'master': [[f'lc1.inductor{k}.i' for k in '123'],
                                                 [f'lc1.capacitor{k}.v' for k in '123']],
                                      'slave': [[f'lcl1.inductor{k}.i' for k in '123'],
                                                [f'lcl1.capacitor{k}.v' for k in '123'],
                                                np.zeros(3)]})

    env = gym.make('openmodelica_microgrid_gym:NormalizedEnv_test-v1',
                   net='net_static_droop_controller.yaml',
                   model_path='../fmu/grid.network.fmu',
                   max_episode_steps=max_episode_steps,
                   is_normalized=False)

    #agent = RndAgent()
    #runner = Runner(agent, env)

    #runner.run(1)

    env.reset()
    #action = np.array([0.1, -0.3, 0.7])
    action = env.action_space.sample()
    out = env.step(action)

    #print(out)
    #print(env.net.out_vars(True))

    print(dict(zip(env.net.out_vars(True),out[0])))


    # User runner to execute num_episodes-times episodes of the env controlled by the agent
    runner = Runner(agent, env)
    runner.run(num_episodes, visualise=True)