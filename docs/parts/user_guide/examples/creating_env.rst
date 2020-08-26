Creating an environment
=======================


Following is a minimal example how to set-up and run an environment.
Necessary is the definition of model inputs, in this case the three phases of the first inverter.
The model outputs will be shown as simulation results, and the model path is the relative location of the FMU file, which contains the network.
For any other simulation parameters, for example the step-size, default values will be used.

For the initialisation, the environment needs to be reseted, and env.render will define the output plots.
The simulation will perform 1000 steps. A different random number will be provided to every of the three previously defined model_inputs.
Afterwards, the inductor currents of the LC-filter "lc1"shown in the figure above will be plotted, which should result in three increasing and due to the random function noisy  lines.

.. literalinclude:: ../../../../examples/basic_env.py
   :linenos: