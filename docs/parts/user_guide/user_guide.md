#  Getting Started
![](../pictures/microgrid.jpg)

This User Guide covers all of OpenModelica Microgrids Gym toolboxes by topic area. Each of the following steps is introduced in a more detailed way in the different chapters of this users guide. 


First step is to [create the microgrid](OpenModelica.html), which is the environment for the reinforcement learning agent.

It is built in the open source software [OpenModelica](https://www.openmodelica.org/). 

![](../pictures/network.png)

For the transfer to python, it needs to get exported as a [Functional Mock-up Unit (FMU)](https://fmi-standard.org/).

The creation process of the FMU is shown [here](fmu.html). It is used build a gym env like in the examples from [OpenAI Gym](https://gym.openai.com/).
In OMG, the gym environment is defined for example in (examples/berkenkamp.py, line 103).

After creating the environment, the network can be simulated in python. On the one hand, it is possible to test predefined controller settings like described [here](examples.html#staticctrl-py).

![](../pictures/abc.png)
 
The main function of this toolbox is to apply reinforcement learning approaches to the network, how to run them is shown [here](examples.html#berkenkamp-py).
 
![](../pictures/kp_kp_J.png)