within OpenModelica_Microgrids.Components;

block StartValues "Output the input signal filtered with a low pass Butterworth filter of any order"
  extends Modelica.Blocks.Interfaces.SISO;
  Real z;
  Real compare;
  Real value;
  parameter Real startTime(start = 0.02);
  Modelica.Blocks.Sources.RealExpression realExpression(y = 1) annotation(
    Placement(visible = true, transformation(origin = {-62, 36}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.Integrator integrator annotation(
    Placement(visible = true, transformation(origin = {-16, 36}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  z = (integrator.y - startTime) * 100000;
  compare = max(0, z);
  value = min(compare, 1);
  y = u * value;
  connect(realExpression.y, integrator.u) annotation(
    Line(points = {{-50, 36}, {-28, 36}, {-28, 36}, {-28, 36}}, color = {0, 0, 127}));
  annotation(
    Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
end StartValues;
