Installation and General Remarks
================================


In the following, an introduction to the Python code written for th OMG
toolbox is presented.


Installation and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to install OMG via pip:

::

    pip install openmodelica_microgrid_gym

Alternatively, you can clone the GitHub repository. A list of
requirements is provided in the
home-directory.

.. literalinclude:: ../../../requirements.txt


**Hint:** If you are running a windows, PyFMI might throw some errors
while installing via pip. It can be installed via *conda* by running:

::

    conda install -c conda-forge pyfmi 

Simulation Settings
~~~~~~~~~~~~~~~~~~~

Heart of the program structure is the creation of the environment via
**gym.make()** in the main programm (in the folder example). Nearly
every simulation setting can be done directly in here. Some of the most
important ones are described in the following. For further information, see the `API-documentation <../../api/omg.env.modelica.html>`__.

-  **time\_step:** step size of the simulation in seconds. Too large
   timesteps may result in numerical issues, small timesteps result in a
   high simulation time. 1e-4 seems to be a good compromise as many real
   controllers operate in timesteps like this.

-  **reward\_fun:** Callable - Reward function for the RL-algorithm. Minimal value
   of rewards is for example used as lower bound for the safe Bayseian
   algorithm (see single_inverter_current_control_safe_opt.py). Has to be adjusted problem-specific.

-  **solver\_method:** Solver used for the ODE system. Every solver from
   `scipy.integrate.solve\_ivp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`__
   can be selected. Though it is highly recommended to use the implicit
   ones. Default solver is the "LSODA" with its integrated stiffness
   detection. Non-stiff systems become solved with the faster "Adams"
   method, stiff systems with "BDF".


-  **model\_params:** Parameters for the simulation, which should be
   changed compared to the default values from the OpenModelica model.
   Also usable for loadsteps as replacement for
   `switches <OpenModelica.md>`__.

Example which increases the resistors in the load after 0.2 seconds from
20 Ohm to 40 Ohm:

::

    def f(t):
        return 20 if t < .2 else 40

    model_params={'rl.resistor1.R': f, 'rl.resistor.R': f, 'rl.resistor.R': f},




Setting of v\_DC
~~~~~~~~~~~~~~~~

The DC supply voltage v\_DC can be set either directly in the
`OpenModelica model <OpenModelica.html#setting-of-v-dc>`__ or via
Python. The default value is 1000 V. It can be changed in the
environment creation with the line:

::

    model_params={'inverter1.v_DC': 700, 'inverter2.v_DC': 500}, 

It will be set for every of the three phases of the inverter. Take care
to set the param for every inverter which should no have the default
supply voltage of 1000 V.

Data Logging
~~~~~~~~~~~~

To enable logging, the root logger needs to be initialized in the
main function. To do so, call:

::

    import numpy as np

    logging.basicConfig()

    if __name__ == '__main__':
        ctrl = dict()

For further information about logging and the level see the `logging
standard library <https://docs.python.org/3/library/logging.html>`__.
