Creating an agent and a runner
==============================

Additionally to the environment, an an agent will be created and a runner will be used. The runner class will take care of initializing and termination
of agents and environments, as well as the execution of multiple episodes. The class will handle all information
exchange between agent and environment like presented in the high level code architecture shown below:

.. figure:: ../../../pictures/highlevel.png
   :width: 400
   :alt:

Since the inputs are used for both the agent and the environment, they are defined in advance. Although the Agent gets information of the environment, in this small example, its action is still a random number.

The environment is the same as above. Afterwards, the agent and the runner get defined, and the runner runs for one episode.

.. literalinclude:: ../../../../examples/simple_agent.py
   :linenos: