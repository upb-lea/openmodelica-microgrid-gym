import gym
import time
import matplotlib.pyplot as plt
from gym_microgrid.controllers import *

from gym.envs.registration import register

# clears the environment
env_name = "JModelicaConvEnv-v1"

fcontrol = 1e4
delta_t = 1 / fcontrol
# Time to simulate
T_simulate = 0.1
N = int(T_simulate * fcontrol)

V_dc = 1000
nPhase = 3
nomFreq = 50
nomVoltPeak = 230 * 1.414
iLimit = 30
DroopGain = 40000.0  # W/Hz
QDroopGain = 1000.0  # VAR/V


def grid_simulation(sim_env, max_number_of_steps=N, n_episodes=1, visualize=False):
    """
    Runs one experiment of parameter tuning on the grid environment.

    :param sim_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes to perform.
    :param visualize: flag if experiments should be rendered. (not implemented)

    :return: trained Q-learning agent (not implemented yet), array of actual episodes length, execution time in s
    """

    # Droop of the active power Watt/Hz, delta_t
    droopParam = DroopParams(DroopGain, 0.005, nomFreq)
    # droopParam= DroopParams(0,0,nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroopParam = DroopParams(QDroopGain, 0.002, nomVoltPeak)
    # qdroopParam= DroopParams(0,0,nomVoltPeak)

    # PIPI voltage forming controller in ABC frame
    # TODO: To check by Jarren what is still needed!
    """
    
    
    
    #Voltage loop PI controller parameters for the voltage forming inverter
    voltagePIparams=PI_parameters(kP= 0.33, kI=150.0, uL=iLimit, lL= -iLimit, kB =1)
    #Current loop PI parameters controller for the voltage forming inverter's current loop
    currentPIparams=PI_parameters(kP= 0.24, kI=20.0, uL=1, lL= -1 ,kB =1 )
    controller = MultiPhaseABCPIPIController (voltagePIparams, currentPIparams,
                                              delta_t, droopParam, qdroopParam, 
                                              undersampling=10, n_phase=3)
    """
    # Current PI parameters for the voltage sourcing inverter
    currentDQPIparams = PIParams(kP=0.012, kI=90, uL=1, lL=-1, kB=1)
    # Voltage PI parameters for the current sourcing inverter
    voltageDQPIparams = PIParams(kP=0.025, kI=60, uL=iLimit, lL=-iLimit, kB=1)

    controller = MultiPhaseDQ0PIPIController(voltageDQPIparams, currentDQPIparams,
                                             delta_t, droopParam, qdroopParam,
                                             undersampling=1, n_phase=3)

    # Discrete controller implementation for a DQ based Current controller for the current sourcing inverter
    # Droop of the active power Watts/Hz, W.s/Hz
    droopParam = InverseDroopParams(DroopGain, 0, nomValue=nomFreq, tau_filt=0.04)
    # droopParam=InverseDroopParams(0,0,nomFreq)
    # Droop of the reactive power VAR/Volt Var.s/Volt
    qdroopParam = InverseDroopParams(100, 0, nomVoltPeak, tau_filt=0.01)
    # qdroopParam=InverseDroopParams(0,0,nomVoltPeak)

    # PI params for the PLL in the current forming inverter
    pllparams = PLLParams(kP=10, kI=200, uL=10000, lL=-10000, kB=1, f_nom=50)

    # Current PI parameters for the current sourcing inverter
    currentDQPIparams = PIParams(kP=0.005, kI=200, uL=1, lL=-1, kB=1)

    slave_controller = MultiPhaseDQCurrentController(currentDQPIparams, pllparams,
                                                     delta_t, nomFreq, iLimit, droopParam,
                                                     qdroopParam, undersampling=1)

    episode_lengths = np.array([])

    for episode in range(n_episodes):
        start = time.time()
        sim_time = 0
        cont_time = 0
        obs = sim_env.reset()
        obs = np.array(obs)

        currentHist = []  # List to hold the current values
        voltageHist = []  # List to hold the voltage values
        currentsHist = []  # List to hold the currents for the second inverter
        freqHist = []  # List to hold the voltage values
        timeHist = []
        iteration_s = []
        currentsHistdq0 = []  # Currents for the second inverter in dq0

        CVV1, CVI1, CVV2, CVI2 = _map_CVs(obs)

        # Logging for plotting
        currentHist.append(CVI1)
        voltageHist.append(CVV2)
        currentsHist.append(CVI2)
        timeHist.append(0)
        freqHist.append(nomFreq)
        currentsHistdq0.append([0, 0, 0])

        # SP for the slave inverter in dq0 frame
        for step in range(max_number_of_steps):

            SPidq0 = [0, 0, 0]

            # CVs from the states of the simulation
            CVV1, CVI1, CVV2, CVI2 = _map_CVs(obs)

            # prev_cossine, prev_freq, prev_theta, debug = testPLL.step(CVI1)

            shareRatio = inst_power(CVV1, CVI1) / inst_power(CVV2, CVI2)
            # Logging for plotting
            # voltageHist.append(CVV1)
            timeHist.append(step * delta_t)

            startContSim = time.time()
            # cossin, freq, theta,debug =pll.step(CVV)
            mod_indSlave, freq, Idq0, mod_dq0 = slave_controller.step(CVI2, CVV2, SPidq0)

            # Perform controller calculations
            mod_ind, CVI1dq = controller.step(CVI1, CVV1, nomVoltPeak, nomFreq)

            # Average voltages from modulation indices created by current controller

            # TODO: Extract number of Inverters from FMU, write a function which defines actions for them dynamically
            action1 = _get_average_voltage(mod_ind)
            action2 = _get_average_voltage(mod_indSlave)
            # action1 = [0,50,0]
            # action2=action1
            # Accumulate time spent computing controller actions
            cont_time = cont_time + time.time() - startContSim

            currentHist.append(CVI1dq)
            currentsHist.append(CVI2)
            voltageHist.append(CVV2)

            # TODO: This changes the data. If its only for plotting, than the plotting axis should be cropped instead – sheid
            freq = np.clip(freq, 49, 51)

            freqHist.append(freq)
            currentsHistdq0.append(Idq0)
            # Record the start time of the simulation for performance metrics
            startSim = time.time()

            # print("Action: {}".format(action))

            # Perform a step of simulation
            obs, reward, done, iterations, _ = sim_env.step(np.append(action1, action2))
            obs = np.array(obs)

            iteration_s.append(iterations)
            # Accumulate time spent simulating
            sim_time = sim_time + time.time() - startSim

            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))
                break

    # calculate performance metrics before starting to plot
    end = time.time()
    execution_time = end - start

    plt.plot(timeHist, currentsHist)
    plt.ylabel('Current 2 [A]')
    plt.title('Current of current source inverter')
    plt.show()

    plt.plot(timeHist, currentHist)
    plt.ylabel('Current 1 [A]')
    plt.title('Current of voltage source')
    plt.legend(['d', 'q', '0'])
    plt.show()

    plt.plot(timeHist, currentsHistdq0)
    plt.ylabel('Current [A]')
    plt.title('Current of current source inverter (DQ0)')
    plt.legend(['d', 'q', '0'])
    plt.show()
    """
    """
    plt.plot(timeHist, voltageHist)
    plt.title('Voltage of inverter')
    plt.ylabel('Voltage [V]')
    plt.show()

    plt.plot(timeHist, freqHist)
    plt.ylabel('Frequency [Hz]')
    plt.title('Frequency of current source inverter')
    plt.show()

    return episode_lengths, execution_time, sim_time, cont_time, iteration_s, {}


# Internal logic for state discretization


def _map_CVs(observation):
    # TODO implement dynamic function regarding to the number of inverters

    """
    Takes all the observations from the simulation and rearranges it into the 
    Control variables used for system feedback.

    :return CVV1: Voltages measured for inverter 1
    :return CVI1: Currents measured for inverter 1
    :return CVV2: Voltages measured for inverter 2
    :return CVI2: Currents measured for inverter 2
    """
    CVI1 = observation[0:3]
    CVV1 = observation[3:6]
    CVI2 = observation[6:9]
    CVV2 = observation[9:12]

    # Invert the current feedback values (positive power when absorbing power)
    # CVI1=constMult(CVI1,-1)
    # CVI2=constMult(CVI2,-1)

    return CVV1, CVI1, CVV2, CVI2


def run_rl_experiments(n_experiments=1, n_episodes=1, visualize=False, time_step=delta_t, positive_reward=1,
                       negative_reward=-100, log_level=4):
    """

    Wrapper for running experiment of q-learning training.
    Is responsible for environment creation and closing, sets all necessary parameters of environment.
    Runs n exepriments, where each experiment is training Q-learning agent on the same environment.
    After one agent finisEnvironment did not existhed training, environment is reset to the initial state.
    Parameters of the experiment:

    :param n_episodes: number of episodes to perform in each experiment run
    :param visualize: boolean flag if experiments should be rendered
    :param n_experiments: number of experiments to perform.
    :param log_level: level of logging that should be used by environment during experiments.

    Parameters of the environment:

    :param time_step: time difference between simulation steps.
    :param positive_reward: positive reward for RL agent.
    :param negative_reward: negative reward for RL agent.

    :return: trained Q-learning agent, array of actual episodes length that were returned from train_qlearning()

    """
    config = {
        'time_step': time_step,
        'positive_reward': positive_reward,
        'negative_reward': negative_reward,
        'log_level': log_level,
        'solver_method': 'LSODA'
    }

    register(
        id=env_name,
        entry_point='gym_microgrid.env:JModelicaConvEnv',
        kwargs=config
    )

    trained_agent_s = []
    episodes_length_s = []
    exec_time_s = []
    sim_time_s = []
    cont_time_s = []
    count_iterations_s = []
    env = gym.make(env_name,
                   model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
                   model_output=['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i',
                                 'lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v',
                                 'lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i',
                                 'lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v'])
    for i in range(n_experiments):
        episodes_length, exec_time, sim_time, cont_time, count_iters, _ = grid_simulation(env,
                                                                                          n_episodes=n_episodes,
                                                                                          visualize=visualize)

        episodes_length_s.append(episodes_length)
        exec_time_s.append(exec_time)
        sim_time_s.append(sim_time)
        cont_time_s.append(cont_time)
        count_iterations_s.append(np.asarray(count_iters).mean())
        env.reset()

    env.close()
    # delete registered environment to avoid errors in future runs.
    del gym.envs.registry.env_specs[env_name]
    return trained_agent_s, episodes_length_s, exec_time_s, sim_time_s, cont_time_s, count_iterations_s, {}


def _get_average_voltage(arr):
    return arr * V_dc


if __name__ == "__main__":

    # Try deleting the environment before excecution (needed if previous execution was interupted by a runtime error)
    try:
        del gym.envs.registry.env_specs[env_name]
    except:
        # Would ideally like to do nothing, but will print a message nonetheless
        print("Environment did not exist")

    _, episodes_lengths, exec_times, sim_time, cont_time, count_iter, _ = run_rl_experiments(visualize=False,
                                                                                             log_level=logging.INFO)
    print("Experiment length {} s".format(exec_times[0]))
    print("Simulating time length {} s ({} %)".format(sim_time[0], (100 * sim_time[0] / exec_times[0])))
    print("Control calc time length {} s ({} %)".format(cont_time[0], (100 * cont_time[0] / exec_times[0])))
    print(u"Avg episode performance {} {} {}".format(episodes_lengths[0].mean(),
                                                     chr(177),  # plus minus sign
                                                     episodes_lengths[0].std()))
    print(u"Avg solver steps per DeltaT {} ".format(count_iter))
    print(u"Max episode performance {}".format(episodes_lengths[0].max()))
    print(u"All episodes performance {}".format(episodes_lengths))
