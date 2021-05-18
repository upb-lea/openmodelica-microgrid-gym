=======
History
=======

Next
-------

0.4.0 (2021-04-07)
------------------
Changes
^^^^^^^
* ModelicaEnv:
    - Introduced action clipping
    - model_params: None values are not passed to the OpenModelica env to allow initialization
    - model_params: negative time values are introduced for initialization (fix)
    - Introduced abort reward in env if episode is terminated
    - Introduced obs_output to define a subset of history given as observation to the agent

Fix
^^^
* omg.net.MasterInverter:
    - default values used to overwrite passed values

Add
^^^
* Random Process wrapper
* ObsTempl test
* reset test for initialized env




0.3.0 (2020-12-18)
------------------

API
^^^
* ModelicaEnv:
    - Uses Network
    - __init__:
      - removed: timestep, model_output, model_input
      - added: network
    - Delay buffer
* Network and Components:
    - Specify class structure using config file corresponding to fmu (see net-folder)
    - added noise
* SafeoptAgent:
    - __init__: Performance parameters and calculation
* aux_ctl.Contoller:
    - __init__: timestep and undersampling changed
    - added output clipping
* Plotmanager


Examples
^^^^^^^^
* updated to changed API

Experiments
^^^^^^^^^^^
* model validation:
    - experiment files
    - experiment environment managing testbench connection via SSH

Dependencies
^^^^^^^^^^^^
* Decreased Language Level to Python 3.7





0.2.0 (2020-05-27)
------------------


API
^^^
* ModelicaEnv:
   - reward function parameter
   - vis_cols now also supports Plotting templates

* EmptyHistory and descendant: update(), append()
* Agent: added properties
* StaticControlAgent and descendant: small changes in constructor params, specifically obs_template, added properties
* SafeOptAgent: added properties
* Runner: plotting can be disabled

Examples
^^^^^^^^
* added example for plotting

Performance
^^^^^^^^^^^
* 6.6× speedup

Dependencies
^^^^^^^^^^^^
* Increased Language Level to Python 3.8



0.1.3 (2020-05-13)
------------------

* best parameter set output after termination of SafeOpt agent (`#7`_)
* proper action and observation space (`#14`_)
* resolved problem related to environment :code:`model_params` (`#21`_)

|

* documentation improvements (more examples, installation)

.. _`#7`: https://github.com/upb-lea/openmodelica-microgrid-gym/issues/7
.. _`#14`: https://github.com/upb-lea/openmodelica-microgrid-gym/issues/14
.. _`#21`: https://github.com/upb-lea/openmodelica-microgrid-gym/issues/21


0.1.2 (2020-05-04)
------------------

* corrected pip install requirements


0.1.1 (2020-04-22)
------------------

* First release on PyPI.
