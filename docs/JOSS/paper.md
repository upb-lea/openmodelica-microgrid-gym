---
title: 'OMG: A scalable and flexible simulation and testing environment toolbox for intelligent microgrid control'
tags:
  - Python
  - OpenModelica
  - Microgrids
  - Reinforcement Learning
  - Energy Systems
  - Simulation
authors:
  - name: Stefan Heid
    affiliation: 2
  - name: Daniel Weber
    affiliation: 1
  - name: Henrik Bode
    affiliation: 1
  - name: Oliver Wallscheid
    affiliation: 1
  - name: Eyke Hüllermeier
    affiliation: 2
affiliations:
 - name: Power Electronics and Electrical Drives, University of Paderborn
   index: 1
 - name: Intelligent Systems and Machine Learning, Universitsy of Paderborn 
   index: 2
date: 25 May 2020
bibliography: paper.bib
---

# Summary


Micro-   and   smart   grids   (MSG)   play   an   important   role   both 
for   integrating   renewable   energy   sources   in conventional  electricity 
grids  and  for  providing  power  supply in  remote  areas.  Modern  MSGs 
are largely  driven  by  power electronic  converters  due  to  their  high 
efficiency and  flexibility. Nevertheless, controlling  MSGs  is  a  challenging  task  due 
to highest  requirements on  energy  availability,  safety and voltage quality 
within  a  wide range  of  different  MSG  topologies.


``OMG`` is an python package for modelling and optimizing microgrids with methods of the reinforcement learning.
An attached OpenModelica library enables the user to define its individual microgrid in a wide range in flexibility and 
scalability or use predefined example grids and use-cases. The API was designed 
to provide a user-friendly interface to connect the microgrid (environment) with an reinforcement learning agent (RL). 
First RL applications based on the work of [@Berkenkamp:2020] are provided in the toolbox. 
Following this structure, nearly every RL approach can be implemented and tested with ``OMG``. Besides, the simulation of classical, low-level 
controller tuning for comparision is also supported. 


``OMG`` was designed to be used by academics in the field in RL to test and compare their algorithms 
in a realistic use-case, to help control engineers to inverters in real grids, and for 
engineering students to get in touch with with machine learning. Since OMG is the onliest open source software which 
provides a RL approach on inverter-tuning on the primary level (voltage and current control), 
a direct comparision to other products is not possible.


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

![](https://git.uni-paderborn.de/walli/omg_paper/-/blob/master/pictures/OMG_flowchart_2.pdf)
_Figure 1.  Overview  of  the  interconnections  between  the  different  parts  of  the  OMG  toolbox.  The  OpenModelica  and  OpenAIGym logos are the property of their respective owners._

# Acknowledgements

The authors kindly acknowledge the funding and support ofthis work by the Paderborn 
University research grant. 

# References

