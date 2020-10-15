within omg_grid.Loads;

model LC
  parameter SI.Capacitance C1(start = 0.00001);
  parameter SI.Capacitance C2(start = 0.00001);
  parameter SI.Capacitance C3(start = 0.00001);
  parameter SI.Inductance L1(start = 0.001);
  parameter SI.Inductance L2(start = 0.001);
  parameter SI.Inductance L3(start = 0.001);
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {-56, -18}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {0, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {56, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {-56, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {0, -24}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {56, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(capacitor3.n, inductor3.p) annotation(
    Line(points = {{56, 32}, {56, 32}, {56, 18}, {56, 18}}, color = {0, 0, 255}));
  connect(inductor3.n, ground1.p) annotation(
    Line(points = {{56, -2}, {56, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(capacitor1.n, inductor1.p) annotation(
    Line(points = {{-56, -28}, {-56, -28}, {-56, -34}, {-56, -34}}, color = {0, 0, 255}));
  connect(inductor1.n, ground1.p) annotation(
    Line(points = {{-56, -54}, {-56, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(capacitor1.p, pin1) annotation(
    Line(points = {{-56, -8}, {-56, 0}, {-88, 0}, {-88, -60}, {-100, -60}}, color = {0, 0, 255}));
  connect(capacitor2.p, pin2) annotation(
    Line(points = {{0, 18}, {0, 24}, {-94, 24}, {-94, 0}, {-100, 0}}, color = {0, 0, 255}));
  connect(capacitor2.n, inductor2.p) annotation(
    Line(points = {{0, -2}, {0, -2}, {0, -14}, {0, -14}}, color = {0, 0, 255}));
  connect(inductor2.n, ground1.p) annotation(
    Line(points = {{0, -34}, {0, -76}}, color = {0, 0, 255}));
  connect(pin3, capacitor3.p) annotation(
    Line(points = {{-100, 60}, {56, 60}, {56, 52}}, color = {0, 0, 255}));
  connect(capacitor3.p, pin3) annotation(
    Line(points = {{56, 52}, {56, 60}, {-100, 60}}, color = {0, 0, 255}));
end LC;
