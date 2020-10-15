within omg_grid.Loads;

model L
  parameter SI.Inductance L1 = 0.001;
  parameter SI.Inductance L2 = 0.001;
  parameter SI.Inductance L3 = 0.001;
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {-48, -50}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {0, -16}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {50, 50}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(inductor3.n, ground1.p) annotation(
    Line(points = {{50, 40}, {50, 40}, {50, -60}, {0, -60}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(pin3, inductor3.p) annotation(
    Line(points = {{-100, 60}, {50, 60}, {50, 60}, {50, 60}}, color = {0, 0, 255}));
  connect(inductor2.n, ground1.p) annotation(
    Line(points = {{0, -26}, {0, -26}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(pin2, inductor2.p) annotation(
    Line(points = {{-100, 0}, {0, 0}, {0, -6}}, color = {0, 0, 255}));
  connect(inductor1.n, ground1.p) annotation(
    Line(points = {{-48, -60}, {0, -60}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(pin1, inductor1.p) annotation(
    Line(points = {{-100, -60}, {-78, -60}, {-78, -40}, {-48, -40}, {-48, -40}, {-48, -40}}, color = {0, 0, 255}));
end L;
