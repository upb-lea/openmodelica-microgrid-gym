OpenModelica
============

`OpenModelica <https://openmodelica.org/>`__ is an open-source
Modelica-based modeling and simulation environment intended for
industrial and academic usage.

Installation of OpenModelica
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The models shown below were create by using
`OMEdit <https://openmodelica.org/download/download-windows>`__ v1.16.

Using a Linux, sometimes appear some problems by trying to install
OpenModelica. In this case, try to download the pre-built `virtual
machine <https://openmodelica.org/download/virtual-machine>`__.

Creating Microgrids with OpenModelica
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The microgrids are created with a userdefined library provided in the
modelica model *grid.mo* located in the folder fmu.

The package "grid" contains a library with components required for
creating the microgrids.

.. figure:: ../pictures/library.jpg
   :alt: 

It contains several packages (red P) with predefined components,
filters, loads etc. Gridmodels and test-settings, which provide for
example a direct sine-voltage to the inputs, are placed directly in the
package (Blue M, after loading with doubleclick shown as input/output
diagram like "network" in the picture above).

.. figure:: ../pictures/omedit.jpg
   :alt: 

Main component of the grids are the three-phase inverters. They consist
of three input voltages controlled by a cascaded PI-PI controller in
python. Default nomenclature for those inputs is i1p1 for
"**i**\ nverter 1 **p**\ hase 1" etc, but it can be changed in the
python code (model\_input=['one', 'two',...] in the env=gym.make()
call).

**Important**: By using filters/loads with inductors or capacitors,
leave the initialization in the provided settings. Changes are likely to
result in errors while creating the FMU or running the python files.

The provided examples are designed for up to two inverters, more flexibility
is planned for future versions.

Losses
^^^^^^

In the default example "network", no losses in the filters are included.
They can be added by using parts out of the "filter" package instead of
the "ideal\_filter" package. Due to a big increase of components and
equations in the ODE-system, the simulation time will increase.

For larger simulations with a demand of losses, it is recommended to
create user defined filters with only the resistors which are needed for
the calculation. To modify them, create a new package (ctrl+N,
specialization: package), duplicate the part which is to modify in the
new package (right-click on it, duplicate, select the previously created
package in path) and modify it there.

Switches
^^^^^^^^

The possibility of generating loadsteps by adding or removing loads is
desirable, but difficult to implement with the possibilities of
OpenModelica. Switches in OpenModelica - like in many other free
modelling languages - are designed as resistors. A closed switch has a
low resistance, an open switch a high one. "Removed" loads are still
connected to the grid. Connections with resistors in such dimension
cause numerical issues while simulating as the ODE system becomes stiff.
There are solvers available for stiff equation systems like BDF and
Radau or ones with automatic stiffness detection, but using the switches
often ran into non-converging systems and execution errors.

The working alternative is a parameter-variation of the load. It is
possible to change the parameters of any load during a simulation and
apply loadsteps in this way (see the topic
`pythoncode <Pythoncode.html>`__).

Setting of v\_DC
^^^^^^^^^^^^^^^^

The DC Supply Voltage *v*\_DC can be set either directly in the
OpenModelica model or via `Python <Pythoncode.html#setting-of-v-dc>`__.
In the Model, doubleclick in your network on the inverter, and change
the parameter *v*\_ DC to the demanded value. All three phases of the
inverter will be supplied with the same DC voltage. The default value is
1000 V. The default value can be changed with a right-click on an
inverter, *open class*, select *text view* on the top left corner of the
model canvas, and change the number in the following code line to
the demanded default value:

::

      parameter Real v_DC = 1000;
      

PLL
^^^

The PLL blocks are working for simulations in OpenModelica, but out of
structural reasons, the PLL is calculated in Python.
