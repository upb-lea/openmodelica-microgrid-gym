# OMG Pythoncode - Installation and general remarks

Following is an introduction to the Pythoncode written for th OMG toolbox.

### installation and requirements

It is recommended to install OMG via pip:

``pip install openmodelica_microgrid_gym
``

Alternatively, you can clone the GitHub repository. A list of [requirements](../../requirements.txt) is provided in the home-directory.

```
gym>=0.15.3
PyFMI>=2.5.7
matplotlib>=3.1.1
scipy>=1.3.1
numpy>=1.17.2
pandas>=1.0.1
tqdm>=4
more_itertools>=7
pytest>=5.2.1[tests]
safeopt>=0.15[safeopt]
```

### Program structure

Heart of the program structure is the creation of the environment via **gym.make()** in the main programm (in the folder example). Nearly every simulation setting can be done directly in here. Some of the most important ones are described in the following:

* **time_step:** step size of the simulation in seconds. Too large timesteps may result in numerical issues, small timesteps result in a high simulation time. 1e-4 seems to be a good compromise as many real controllers operate in timesteps like this.

* **reward_fun:** Reward function for the RL-Algorithm. Minimal value of rewards is for example used as lower bound for the safe bayseian algorithm (see berkenkamp.py)

* **solver_method:** Solver used for the ODE system. Every solver from [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) can be selected. Though it is highly recommended to use the implicit ones.
 Default solver is the "LSODA" with its integrated stiffness detection. Non-stiff systems become solved with the faster "Adams" method, stiff systems with "BDF".
 
* **max_episode_steps:** Number of maximal episodes. Length of the simulation is defined by this number multiplied by with the lengtn of a "**time_step**".

* **model_params:** Parameters for the simulation, which should be changed compared to the default values from the OpenModelica model. 
Also usable for loadsteps as replacement for [switches](OpenModelica.md).

Example which selts 
``` 
def f(t):
    return 20 if t < .2 else 40

model_params={'rl.switch1.R': f, 'rl.switch2.R': f, 'rl.switch3.R': f},```
 
