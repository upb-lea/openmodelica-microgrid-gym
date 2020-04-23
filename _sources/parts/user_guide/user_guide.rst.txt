Getting Started
===============

.. figure:: ../../pictures/microgrid.jpg
   :alt: 

This user guide covers all of OpenModelica Microgrids Gym (OMG) toolbox features by
topic area. Each of the following steps is introduced in a more detailed
way in the different chapters of this users guide.

First step is to `create the microgrid <OpenModelica.html>`__, which is
the environment for training reinforcement learning agents in power electronic-driven microgrids.
In addition, the OMG toolbox can be used for pure simulation and classical control purpose using OpenModelica models with a Python interface.

Each microgrid model is built in the open source software
`OpenModelica <https://www.openmodelica.org/>`__ and can be easily adapted.

.. figure:: ../../pictures/network.png
   :alt: 

For the transfer to Python, it needs to get exported as a `Functional
Mock-up Unit (FMU) <https://fmi-standard.org/>`__.

The creation process of the FMU is shown `here <fmu.html>`__. It is used to
build a gym environment like in the examples from `OpenAI
Gym <https://gym.openai.com/>`__. In OMG, the gym environment is defined
for example in (examples/two_inverter_static_droop_control.py).

After creating the environment, the network can be simulated in Python.
On the one hand, it is possible to test predefined, static controller designs
like described `here <examples.html#two-inverter-static-droop-control-py>`__.

.. figure::  ../../pictures/abc.png
   :alt: 

However, the main function of this toolbox is to apply reinforcement learning
approaches by utilizing the OMG interface for optimal microgrid control as shown in this
`example <examples.html#single-inverter-current-control-safe-opt-py>`__.

.. figure::  ../../pictures/kp_kp_J.png
   :alt: 


