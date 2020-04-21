Getting Started
===============

.. figure:: ../pictures/microgrid.jpg
   :alt: 

This User Guide covers all of OpenModelica Microgrids Gym toolboxes by
topic area. Each of the following steps is introduced in a more detailed
way in the different chapters of this users guide.

First step is to `create the microgrid <OpenModelica.html>`__, which is
the environment for the reinforcement learning agent.

It is built in the open source software
`OpenModelica <https://www.openmodelica.org/>`__.

.. figure:: ../pictures/network.png
   :alt: 

For the transfer to python, it needs to get exported as a `Functional
Mock-up Unit (FMU) <https://fmi-standard.org/>`__.

The creation process of the FMU is shown `here <fmu.html>`__. It is used to
build a gym environment like in the examples from `OpenAI
Gym <https://gym.openai.com/>`__. In OMG, the gym environment is defined
for example in (examples/two_inverter_static_droop_control.py, line 72).

After creating the environment, the network can be simulated in python.
On the one hand, it is possible to test predefined controller settings
like described `here <examples.html#two-inverter-static-droop-control-py>`__.

.. figure:: ../pictures/abc.png
   :alt: 

The main function of this toolbox is to apply reinforcement learning
approaches to the network, how to run them is shown
`here <examples.html#single-inverter-current-control-safe-opt-py>`__.

.. figure:: ../pictures/kp_kp_J.png
   :alt: 


