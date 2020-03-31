

Welcome to Reinforcement Learning in Micrigrids (RLMG) documentation!
=====================================================================


The Reinforcement Learning in Micrigrids (RLMG) package is a software toolbox for the simulation of different microgrids to
train and test reinforcement learning agents and to compare them with classical parameter tuning.


Getting started
***************************

A quick start guide can be found in the following Readme-File.

.. toctree::
   :maxdepth: 1
   :caption: RLMG Readme:

   parts/readme





Content
*******

In the examples section all available usecases are presented with their default configuration.
For quick start, one of these can be selected and used out of the box.


The documentation of the base classes is important for the development of own modules like further reward functions or
reference generators. In this part, the basic interfaces of each module are specified.
For the creation of additional grid contellations, Openmodelica (nightly build recommended) can be used. 


.. automodule:: examples
    :members:


.. toctree::
   :maxdepth: 4
   :titlesonly:
   :caption: RLMG Contents:

   parts/modules






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
