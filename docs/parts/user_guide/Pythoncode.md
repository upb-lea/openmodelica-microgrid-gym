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

pytest>=5.2.1 [tests]
tables>=3.4.1 [tests]


safeopt>=0.15 [safeopt]
GPy>=1.9.9 [safeopt]

sphinx-autodoc-typehints>1.10 [doc]
```

**Hint:** If you are running a windows, PyFMI might throw some errors while installing via pip.
It can be installed via _conda_ by running:

    conda install -c conda-forge pyfmi 

### Simulation settings

Heart of the program structure is the creation of the environment via **gym.make()** in the main programm (in the folder example). Nearly every simulation setting can be done directly in here. Some of the most important ones are described in the following:

* **time_step:** step size of the simulation in seconds. Too large timesteps may result in numerical issues, small timesteps result in a high simulation time. 1e-4 seems to be a good compromise as many real controllers operate in timesteps like this.

* **reward_fun:** Reward function for the RL-Algorithm. Minimal value of rewards is for example used as lower bound for the safe bayseian algorithm (see berkenkamp.py)

* **solver_method:** Solver used for the ODE system. Every solver from [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) can be selected. Though it is highly recommended to use the implicit ones.
 Default solver is the "LSODA" with its integrated stiffness detection. Non-stiff systems become solved with the faster "Adams" method, stiff systems with "BDF".
 
* **max_episode_steps:** Number of maximal episodes. Length of the simulation is defined by this number multiplied by with the lengtn of a "**time_step**".

* **model_params:** Parameters for the simulation, which should be changed compared to the default values from the OpenModelica model. 
Also usable for loadsteps as replacement for [switches](OpenModelica.md).

Example which increases the resistors in the load after 0.2 seconds from 20 Ohm to 40 Ohm:
    
    def f(t):
        return 20 if t < .2 else 40

    model_params={'rl.resistor1.R': f, 'rl.resistor.R': f, 'rl.resistor.R': f},

 
* **model_input:** Input of the inverter in the fmu. Example according to the standard nomenclature:
    ``` 
    model_input=['i1p1', 'i1p2', 'i1p3', 'i2p1', 'i2p2', 'i2p3'],
    ``` 

* **model_output:** Nested dictionaries containing nested lists of strings.
         The keys of the nested dictionaries will be flattened down and appended to their children and finally prepended
         to the strings in the nested lists. The strings final strings represent variables from the FMU and the nesting
         of the lists conveys structure used in the visualisation

         >>> {'inverter': {'condensator': ['i', 'v']}}

   results in

         >>> ['inverter.condensator.i', 'inverter.condensator.v']
* **model_path:** relative path of the fmu data. Default: ../fmu/grid.network.fmu

* **viz_mode:** specifies how and if to render

       - 'episode': render after the episode is finished
       - 'step': render after each time step
       - None: disable visualization
* **viz_cols:** enables specific columns while plotting

        - None: all columns will be used for vizualization (default)
        - string: will be interpret as regex. all fully matched columns names will be enabled
        - list of strings: Each string might be a unix-shell style wildcard like "*.i"
          to match all data series ending with ".i".


### Data logging      
To enable logging, the the root logger needs to be initialized in the main function. To do so, call:


    import numpy as np

    logging.basicConfig()

    if __name__ == '__main__':
        ctrl = dict()
For further information about logging and the level see the [logging standard library](https://docs.python.org/3/library/logging.html).        

