within OpenModelica_Microgrids.Components;

model PhaseAngle
  parameter Real freq(start = 50);
  Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
    Placement(visible = true, transformation(origin = {14, 0}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant f_nom(k = freq) annotation(
    Placement(visible = true, transformation(origin = {-18, 0}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput theta annotation(
    Placement(visible = true, transformation(origin = {108, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {108, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(f_nom.y, f2theta.u) annotation(
    Line(points = {{-14, 0}, {-5, 0}}, color = {0, 0, 127}));
  connect(f2theta.y, deg2rad.u) annotation(
    Line(points = {{4, 0}, {9, 0}}, color = {0, 0, 127}));
  connect(deg2rad.y, theta) annotation(
    Line(points = {{18, 0}, {108, 0}}, color = {0, 0, 127}));
end PhaseAngle;
