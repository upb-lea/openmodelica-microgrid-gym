==========================
OpenModelica Microgrid Gym
==========================

.. image:: https://travis-ci.org/upb-lea/openmodelica-microgrid-gym.svg?branch=master
    :target: https://travis-ci.org/github/upb-lea/openmodelica-microgrid-gym

.. image:: https://img.shields.io/badge/doc-success-success
    :target: https://upb-lea.github.io/openmodelica-microgrid-gym

.. image:: https://codecov.io/gh/upb-lea/openmodelica-microgrid-gym/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/upb-lea/openmodelica-microgrid-gym

.. image:: https://img.shields.io/pypi/v/openmodelica_microgrid_gym.svg
    :target: https://pypi.python.org/pypi/openmodelica_microgrid_gym

.. image:: https://pyup.io/repos/github/upb-lea/openmodelica_microgrid_gym/shield.svg
     :target: https://pyup.io/repos/github/upb-lea/openmodelica_microgrid_gym/
     :alt: Updates

.. image:: https://img.shields.io/github/license/upb-lea/openmodelica-microgrid-gym
     :target: LICENSE

.. figure:: https://github.com/upb-lea/openmodelica-microgrid-gym/raw/master/docs/pictures/microgrid.jpg

**The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the
simulation and control optimization of microgrids based on energy conversion by power electronic converters.**

The main characteristics of the toolbox are the plug-and-play grid design and simulation in OpenModelica as well as
the ready-to-go approach of intuitive reinfrocement learning (RL) approaches through a Python interface.

The OMG toolbox is built upon the `OpenAI Gym`_ environment definition framework.
Therefore, the toolbox is specifically designed for running reinforcement
learning algorithms to train agents controlling power electronic converters in microgrids. Nevertheless, also arbritary classical control approaches can be combined and tested using the OMG interface.

.. _OpenAI Gym: https://gym.openai.com/

* Free software: GNU General Public License v3
* Documentation: https://upb-lea.github.io/openmodelica-microgrid-gym


Installation
------------


Install Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Install OpenModelica MicrogridGym from PyPI (recommended)::

    $ pip install openmodelica_microgrid_gym

- Or install from Github source::

    $ git clone https://github.com/upb-lea/openmodelica-microgrid-gym.git
    $ cd openmodelica_microgrid_gym
    $ python setup.py install

**Hint:** PyFMI_ might throw some errors while installing via pip.
It can be installed via ``conda`` by running::

    $ conda install -c conda-forge pyfmi

.. _PyFMI: https://github.com/modelon-community/PyFMI

Installation of OpenModelica
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OMG was create by using OMEdit_ v1.16

In this case, try to download the pre-built `virtual machine`_.

.. _OMEdit: https://openmodelica.org/download/download-windows
.. _virtual machine: https://openmodelica.org/download/virtual-machine

Getting started
---------------


OMG uses the `FMI standard`_ for the exchange of the model between OpenModelica and python.

.. _FMI standard: https://fmi-standard.org/

An example network consisting out of two inverters, three filters and an inductive load.

.. figure:: https://github.com/upb-lea/openmodelica-microgrid-gym/raw/master/docs/pictures/omedit.jpg

You can either use one of the provided FMUs (Windows and Linux, 64-bit, both included in the grid.network.fmu) or create your own by running::

    openmodelica_microgrid_gym\fmu> omc create_fmu.mos

Running the ``staticctrl.py`` starts a simulation with a manually tuned cascaded PIPI controller

.. figure:: https://github.com/upb-lea/openmodelica-microgrid-gym/raw/master/docs/pictures/control.jpg
    :scale: 70%
    :align: center

A save Bayesian approach of a reinforcement learning agent is provided under examples/berkamkamp.py.

.. figure:: https://github.com/upb-lea/openmodelica-microgrid-gym/raw/master/docs/pictures/kp_kp_J.png
    :figwidth: 60%
    :align: center

Every user defined settings can be directly done in the example program.

.. code-block:: python

    env = gym.make(environment-id, **kwargs)

Returns an instantiated grid environment. Provide any additional settings right here (see full documentation for all possibilities)

Citation
--------

A whitepaper for this framework will be avaiable soon. Please use the following BibTeX entry for citing us::

    @misc{LEA2020XXXXXXX,
        title={XXXXXXXXXX},
        author={Daniel Weber and Stefan Heid and Henrik Bode and Oliver Wallscheid},
        year={2020},
        eprint={XXXXX},
        archivePrefix={arXiv},
        primaryClass={eess.SY}
    }


Contributing
------------

Please refer to the `contribution guide`_.

.. _`contribution guide`: https://github.com/upb-lea/openmodelica-microgrid-gym/blob/master/CONTRIBUTING.rst


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
