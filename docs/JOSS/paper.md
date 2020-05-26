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
    affiliation: 1
  - name: Daniel Weber
    affiliation: 2
  - name: Henrik Bode
    affiliation: 2
  - name: Eyke Hüllermeier
    affiliation: 1
  - name: Oliver Wallscheid
    orcid: 0000-0001-9362-8777
    affiliation: 2
affiliations:
 - name: Chair of Intelligent Systems and Machine Learning, Paderborn University
   index: 1
 - name: Chair of Power Electronics and Electrical Drives, Paderborn University
   index: 2
date: 25 May 2020
bibliography: paper.bib
---

# Summary


Micro- and smart grids (MSG) play an important role both for integrating renewable energy sources in conventional electricity grids and for providing power supply in remote areas [@Lund2017]. 
Modern MSGs are largely driven by power electronic converters due to their high efficiency and flexibility. 
Nevertheless, controlling MSGs is a challenging task due to the highest requirements on energy availability, safety and voltage quality within a  wide range of different MSG topologies depending on their field of application like industrial campuses, residential areas or remote off-grid electrification [@Kroposki2008].
This results in high demand for comprehensive testing of new control concepts during their development phase and comparisons with the state of the art to ensure their feasibility.
This applies in particular to data-driven control approaches such as reinforcement learning (RL), the stability and operating behavior of which can hardly be evaluated a priori [@Garcia2015].


``OMG`` is a Python-based package for modeling and simulation of microgrids based on power electronic energy conversion.
An attached OpenModelica [@OSMC2020] library enables the user to define its microgrid (i.e. a local electricity grid containing arbitrary sources, storages and loads) in a flexible and scalable way or to use predefined example grids. 
Due to the component-oriented modeling framework based on OpenModelica, dynamic processes on small time scales are focused which allows for accurate control and test investigations during transients and steady-state.
This is an essential difference to already available open-source solutions for the simulation of electrical energy networks, which, in contrast, generally depict large-scale transmission networks with abstracted models in the (quasi)-stationary state (e.g. PyPSA [@Brown2018] or Pandapower [@Thurner2018]). Besides the pure modeling and simulation of microgrids, basic building blocks for setting up a hierarchical control framework on the inner and primary level are provided [@Guerrero2013]. 


The API is designed to provide a user-friendly interface to connect a modeled microgrid (environment) with a wide range of control methods such as classical linear feedback control or model predictive control techniques (cf. Fig. 1). Moreover, the standardized OpenAI Gym interface [@OpenAI2016] is also available for training data-driven control approaches like RL. 
Many auxiliary functionalities for the essential operation of microgrids are shipped with OMG such as coordinate transformations for basic controller classes, monitoring wrappers or phase-locked loops for frequency and phase angle extraction. 
Following this structure, nearly every control approach including data-driven RL can be implemented and tested with ``OMG`` in a relatively short amount of time. To highlight the challenges of data-driven control approaches in safety-critical environments, application examples using safe Bayesian optimization [@Berkenkamp2020] for automated controller design are provided in the toolbox. 


Therefore, ``OMG`` is designed to be used by academics in the field of control and energy engineering as well as data science. The primary objective of the toolbox is to facilitate the entry into the modeling, control and testing of microgrids and to provide a platform based on which different control methods can be compared under defined conditions (benchmarks).


![\label{fig:omg}](omg.png)
_Fig. 1:  Overview of the interconnections between the different parts of the  OMG  toolbox.  The  OpenModelica and  OpenAIGym logos are the property of their respective owners._

# Features

The ``OMG`` toolbox provides the following key features:


* A library for the scalable and flexible design of local electricity grids in OpenModelica.
The user can select between a wide range of different grid components and connect them in a plug and play approach.

* Dynamic simulation of local electricity grids on component level including single and multi-phase systems as well as AC and DC operation. 

* Easy exchange of models between computing platforms and fast simulation of the models by using the FMI 2.0 standard with C++ code inside. Appropriate numeric solvers for the underlying system of ordinary differential equations can be easily chosen within the usual Python packages (e.g. SciPy) due to the usage of co-simulation. 

* Calculation, evaluation and monitoring of every single time step covering states, action and auxiliary quantities provides an interface for manual or automated inspection. The latter is particularly useful for the automatic training of data-driven control approaches such as reinforcement learning.

* Shipped with many software auxiliaries for the control and monitoring of power electronic-driven microgrids.

* Comes with interesting use cases applying safe data-driven learning to highlight the requirement of safety in a delicate control environment.



# Availability and implementation
``OMG`` is supported and tested on Linux and Windows. The package can be 
installed by `pip` Python package manager using 
`pip install openmodelica_microgrid_gym` command. The source code, guide and 
datasets are available on the GitHub repository (https://github.com/upb-lea/openmodelica-microgrid-gym). 


# Acknowledgements

The authors kindly acknowledge the funding and support of this work by the Paderborn 
University research grant. 

# References

