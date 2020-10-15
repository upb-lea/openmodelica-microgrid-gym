within omg_grid.Loads;

model C
  parameter SI.Capacitance C1 = 0.00001;
  parameter SI.Capacitance C2 = 0.00001;
  parameter SI.Capacitance C3 = 0.00001;
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {-34, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {4, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {40, 38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {4, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(capacitor3.p, pin3) annotation(
    Line(points = {{40, 48}, {40, 60}, {-100, 60}}, color = {0, 0, 255}));
  connect(pin2, capacitor2.p) annotation(
    Line(points = {{-102, 0}, {4, 0}}, color = {0, 0, 255}));
  connect(capacitor2.n, ground1.p) annotation(
    Line(points = {{4, -20}, {4, -74}}, color = {0, 0, 255}));
  connect(pin1, capacitor1.p) annotation(
    Line(points = {{-100, -60}, {-67, -60}, {-67, -34}, {-34, -34}}, color = {0, 0, 255}));
  connect(capacitor3.n, ground1.p) annotation(
    Line(points = {{40, 28}, {40, -54}, {4, -54}, {4, -74}}, color = {0, 0, 255}));
  connect(capacitor1.n, ground1.p) annotation(
    Line(points = {{-34, -54}, {4, -54}, {4, -74}, {4, -74}}, color = {0, 0, 255}));
end C;
