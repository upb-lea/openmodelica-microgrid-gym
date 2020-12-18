==========================
OpenModelica Microgrid Gym
==========================

| |build| |cov| |nbsp| |nbsp| |python| |pypi| |download| |nbsp| |nbsp| |license|
| |doc| |whitepaper| |joss|

.. |nbsp|   unicode:: U+00A0 .. NO-BREAK SPACE

.. |build| image:: https://travis-ci.org/upb-lea/openmodelica-microgrid-gym.svg?branch=master
    :target: https://travis-ci.org/github/upb-lea/openmodelica-microgrid-gym

.. |cov| image:: https://codecov.io/gh/upb-lea/openmodelica-microgrid-gym/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/upb-lea/openmodelica-microgrid-gym

.. |license| image:: https://img.shields.io/github/license/upb-lea/openmodelica-microgrid-gym
    :target: LICENSE

.. |python| image:: https://img.shields.io/pypi/pyversions/openmodelica-microgrid-gym
    :target: https://pypi.python.org/pypi/openmodelica_microgrid_gym

.. |pypi| image:: https://img.shields.io/pypi/v/openmodelica_microgrid_gym
    :target: https://pypi.python.org/pypi/openmodelica_microgrid_gym

.. |download| image:: https://img.shields.io/pypi/dw/openmodelica-microgrid-gym
    :target: https://pypistats.org/packages/openmodelica-microgrid-gym

.. |doc| image:: https://img.shields.io/badge/doc-success-success
    :target: https://upb-lea.github.io/openmodelica-microgrid-gym

.. |whitepaper| image:: https://img.shields.io/badge/arXiv-whitepaper-informational
    :target: https://arxiv.org/pdf/2005.04869.pdf
    
.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.02435/status.svg
   :target: https://doi.org/10.21105/joss.02435



.. figure:: https://github.com/upb-lea/openmodelica-microgrid-gym/raw/develop/docs/pictures/omg_flow.png

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


Install Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the short installation guide for Windows and Linux. OpenModelica_ is hardly supported for Mac, they suggest to install in a Linux VM. For this reason, running OMG in a Linux VM is strongly recommended for Mac users!

Since it is not possible to install PyFMI_, a package which is necessary for the communication between the python interface and the environment, via pip, we recommend to install this package in advance in a conda environment.
As of now, only Windows and Linux are supported officially.

- If conda is NOT installed on your PC, install miniconda_ for python 3.8
- Create a new conda environment (e.g. in PyCharm)
- Install PyFMI from the conda-forge channel in the terminal::

    $ conda install -c conda-forge pyfmi


- Install OpenModelica MicrogridGym from PyPI (recommended)::

    $ pip install openmodelica_microgrid_gym

.. _OpenModelica: https://openmodelica.org/download/download-mac
.. _miniconda: https://conda.io/en/latest/miniconda.html
.. _PyFMI: https://github.com/modelon-community/PyFMI

Installation of OpenModelica
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OMG was create by using OMEdit_ v1.16

In case of installation issues you can resort to their pre-built `virtual machine`_.

.. _OMEdit: https://openmodelica.org/download/download-windows
.. _virtual machine: https://openmodelica.org/download/virtual-machine

Getting started
---------------

The environment is initialized and run like any other OpenAI Gym

.. code-block:: python

    import gym

    if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   max_episode_steps=None,
                   net='../net/net.yaml',
                   model_path='../omg_grid/grid.network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()




OMG uses the `FMI standard`_ for the exchange of the model between OpenModelica and Python.

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

Citation & white paper
----------------------

Please find a white paper on the OMG toolbox including an exemplary usage scenario here:

- https://arxiv.org/abs/2005.04869

Please use the following BibTeX entry for citing us::

    @article{OMG-code2020,
        title = {OMG: A Scalable and Flexible Simulation and Testing Environment Toolbox for Intelligent Microgrid Control},
        author = {Stefan Heid and Daniel Weber and Henrik Bode and Eyke Hüllermeier and Oliver Wallscheid},
        year = {2020},
        doi = {10.21105/joss.02435},
        url = {https://doi.org/10.21105/joss.02435},
        publisher = {The Open Journal},
        volume = {5},
        number = {54},
        pages = {2435},
        journal = {Journal of Open Source Software}
    }

    @article{OMG-whitepaper2020,
        title={Towards a Scalable and Flexible Simulation and
               Testing Environment Toolbox for Intelligent Microgrid Control},
        author={Henrik Bode and Stefan Heid and Daniel Weber and Eyke Hüllermeier and Oliver Wallscheid},
        year={2020},
        eprint={http://arxiv.org/abs/2005.04869},
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
