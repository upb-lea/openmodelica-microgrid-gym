within omg_grid.Loads;

model R
  parameter SI.Resistance R1 = 20;
  parameter SI.Resistance R2 = 20;
  parameter SI.Resistance R3 = 20;
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-66, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-66, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-66, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(resistor3.n, ground1.p) annotation(
    Line(points = {{-56, 60}, {0, 60}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(resistor2.n, ground1.p) annotation(
    Line(points = {{-56, 0}, {-56, 0}, {-56, 0}, {0, 0}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(resistor1.n, ground1.p) annotation(
    Line(points = {{-56, -60}, {0, -60}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(pin1, resistor1.p) annotation(
    Line(points = {{-100, -60}, {-76, -60}, {-76, -60}, {-76, -60}}, color = {0, 0, 255}));
  connect(pin2, resistor2.p) annotation(
    Line(points = {{-100, 0}, {-100, 0}, {-100, 0}, {-76, 0}}, color = {0, 0, 255}));
  connect(pin3, resistor3.p) annotation(
    Line(points = {{-100, 60}, {-76, 60}, {-76, 60}, {-76, 60}}, color = {0, 0, 255}));
end R;
