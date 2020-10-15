within ;
package omg_grid
  extends Modelica.Icons.Package;
  // Import mathematical constants and functions
  import SI = Modelica.SIunits;
  import Modelica.Constants.pi;
  import Modelica.Math;
  import Modelica.ComplexMath.'abs';
  import Modelica.ComplexMath.arg;
  import Modelica.ComplexMath.fromPolar;
  import Modelica.ComplexMath.real;
  import Modelica.ComplexMath.imag;
  import Modelica.ComplexMath.conj;
  import Modelica.ComplexMath.j;


  annotation (
  preferredView="info",
  uses(Modelica(version="3.2.3"), Complex(version="3.2.3"), ModelicaServices(version = "3.2.3")),
  Documentation(info="<html>
  <p><b>omg_grid</b> is a free package that is developed with the Modelica&reg; language from the
  Modelica Association, see <a href=\"https://www.Modelica.org\">https://www.Modelica.org</a>.</p>  It was designed in the scope of the OpenModelica Microgrid Gym (OMG) project  see <a href=\"https://github.com/upb-lea/openmodelica-microgrid-gym\">https://github.com/upb-lea/openmodelica-microgrid-gym</a>
  <p>It provides model components for microgrids.</p> 
  <p><b>Licensed under the Modelica License 2</b><br>
  Copyright &copy; 2020, chair of Power Electronics and Electrical Drives, Paderborn University.</p>  
  <p><i>This Modelica package is <u>free</u> software and the use is completely at <u>your own risk</u>; 
  it can be redistributed and/or modified under the terms of the Modelica License 2. 
  For license conditions (including the disclaimer of warranty) 
  see <a href=\"modelica://Modelica.UsersGuide.ModelicaLicense2\">Modelica.UsersGuide.ModelicaLicense2</a> 
  or visit <a href=\"https://www.modelica.org/licenses/ModelicaLicense2\"> https://www.modelica.org/licenses/ModelicaLicense2</a>.</i></p>  
  <p/>
  </html>"));
end omg_grid;
