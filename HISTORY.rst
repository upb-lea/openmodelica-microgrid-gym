=======
History
=======

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
