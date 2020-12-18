Welcome to OpenModelica Microgrid Gym Toolbox Documentation!
=====================================================================

The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the simulation of power electronics-driven microgrids and to
train and test reinforcement learning agents and to compare them with classical control approaches.

Content
*******

In the examples section all available use cases are presented with their default configuration.
For quick start, one of these can be selected and used out of the box.


The documentation of the base classes is important for the development of own modules like further reward functions or
reference generators. In this part, the basic interfaces of each module are specified.
For the creation of additional grid constellations, Openmodelica (nightly build recommended) can be used.


.. toctree::
   :maxdepth: 4
   :titlesonly:
   :caption: User Guide:

   parts/user_guide/getting_started
   parts/user_guide/OpenModelica
   parts/user_guide/fmu
   parts/user_guide/Pythoncode
   parts/user_guide/examples
   parts/user_guide/controller_tuning


.. toctree::
   :maxdepth: 4
   :titlesonly:
   :caption: API:

   api/omg.agents
   api/omg.aux_ctl
   api/omg.env
   api/omg.execution
   api/omg.net
   api/omg.util


.. GENERATE APIDOC
.. - sphinx-apidoc -o docs/api openmodelica_microgrid_gym/ -e
.. - delete module.rst
.. - remove package and module names:
..   - execute regex '.* package$' ''
..   - execute regex '.* module$' ''


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
