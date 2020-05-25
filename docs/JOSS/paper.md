---
title: 'OMG: A Scalable and Flexible Simulation and Testing Environment Toolbox for Intelligent Microgrid Control'
tags:
  - Python
  - OpenModelica
  - Microgrids
  - Reinforcement Learning
  - Energy Systems
  - Simulation
  - Testing
  - Control
authors:
  - name: Stefan Heid
    affiliation: 2
  - name: Daniel Weber
    affiliation: 1
  - name: Henrik Bode
    affiliation: 1
  - name: Eyke Hüllermeier
    affiliation: 2
  - name: Oliver Wallscheid
    affiliation: 1
affiliations:
 - name: Power Electronics and Electrical Drives, University of Paderborn
   index: 1
 - name: Intelligent Systems and Machine Learning, Universitsy of Paderborn 
   index: 2
date: 25 May 2020
bibliography: paper.bib
---

# Summary


Micro- and smart grids (MSG) play an important role both for integrating renewable energy sources in conventiona electricity grids and for providing power supply in remote areas. 
Modern MSGs are largely driven by power electronic converters due to their high efficiency and flexibility. 
Nevertheless, controlling MSGs is a challenging task due to highest requirements on energy availability, safety and voltage quality within a  wide range of different MSG topologies.


``OMG`` is a Python-based package for modeling and simulation of microgrids based on power electronics energy conversion.
An attached OpenModelica library enables the user to define its individual microgrid (i.e. local electricity grid containing arbitrary sources, storages and load) in a flexiable and scalable way or to use predefined example grids and use-cases. 
The API is designed to provide a user-friendly interface to connect a modeled microgrid (environment) with a wide range of control methods such as classical linear feedback control or model predictive control techniques. Moreoever, the standardized OpenAI Gym interface [@OpenAI:2020] is also available for training data-driven control approaches such as from the domain of reinforcement learning (RL).  
In addition, application examples using safe Bayesian optimization [@Berkenkamp:2020] for automated controller tuning are provided in the toolbox among other auxiliaries such as basic controller classes, monitoring wrappers phase-looked loops. 
Following this structure, nearly every control approach including data-driven RL can be implemented and tested with ``OMG``. 



Therefore, ``OMG`` is designed to be used by academics in the field of control and energy engineering as well as reinforcement learning to allow an easy access to microgrid simulation and control investigations.


# Features

The ``OMG`` toolbox provides the following features:


* A library for a scalable and flexible grid design in OpenModelica in provided.
The user can select between a wide range of different grid components and connect them in a plug an play approach.

* Easy exchange between of models between the platforms and fast simulation of the model by using the FMI 2.0 standard with c++ code inside. 

* Calculation and evaluation of every single timestep provides the possibility of an agent driven parameter tuning even during a simulation run

* Manual controller tuning for a conventional approach of optimizing the network and presenting the benefits of reinforcement learning

* Safe reinforcement learning allows to use the tested approachs in a real world problem with minimizing the risk of incidents

* Detailed evaluation of the result allows a direct comparision between different reinforcement learning approaches


# Availability and implementation
``OMG`` is supported and tested on Linux and Windows. The package can be 
installed by `pip` Python package manager using 
`pip install openmodelica_microgrid_gym` command. The source code, guide and 
datasets are available on the GitHub repository (https://github.com/upb-lea/openmodelica-microgrid-gym). 

![](https://github.com/upb-lea/openmodelica-microgrid-gym/blob/develop/docs/pictures/omg_flow.png)
_Figure 1.  Overview  of  the  interconnections  between  the  different  parts  of  the  OMG  toolbox.  The  OpenModelica  and  OpenAIGym logos are the property of their respective owners._

# Acknowledgements

The authors kindly acknowledge the funding and support ofthis work by the Paderborn 
University research grant. 

# References

