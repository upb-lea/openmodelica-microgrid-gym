within OpenModelica_Microgrids.Loads;

model RL
  parameter SI.Resistance R1 = 20;
  parameter SI.Resistance R2 = 20;
  parameter SI.Resistance R3 = 20;
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
    Placement(visible = true, transformation(origin = {-40, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {0, -18}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R=R1) annotation(
    Placement(visible = true, transformation(origin = {-40, -20}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R=R2) annotation(
    Placement(visible = true, transformation(origin = {0, 12}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R=R3) annotation(
    Placement(visible = true, transformation(origin = {60, 34}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(resistor1.n, inductor1.p) annotation(
    Line(points = {{-40, -30}, {-40, -30}, {-40, -36}, {-40, -36}}, color = {0, 0, 255}));
  connect(pin3, resistor3.p) annotation(
    Line(points = {{-100, 60}, {60, 60}, {60, 44}, {60, 44}}, color = {0, 0, 255}));
  connect(resistor3.n, inductor3.p) annotation(
    Line(points = {{60, 24}, {60, 24}, {60, 24}, {60, 18}}, color = {0, 0, 255}));
  connect(resistor2.n, inductor2.p) annotation(
    Line(points = {{0, 2}, {0, 2}, {0, -8}, {0, -8}}, color = {0, 0, 255}));
  connect(pin2, resistor2.p) annotation(
    Line(points = {{-100, 0}, {-66, 0}, {-66, 22}, {0, 22}}, color = {0, 0, 255}));
  connect(pin1, resistor1.p) annotation(
    Line(points = {{-100, -60}, {-74, -60}, {-74, -10}, {-40, -10}, {-40, -10}}, color = {0, 0, 255}));
  connect(inductor3.n, ground1.p) annotation(
    Line(points = {{60, -2}, {60, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(inductor1.n, ground1.p) annotation(
    Line(points = {{-40, -56}, {-40, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(inductor2.n, ground1.p) annotation(
    Line(points = {{0, -28}, {0, -76}}, color = {0, 0, 255}));
end RL;
