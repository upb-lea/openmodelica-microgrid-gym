within omg_grid.Loads;

model RLC
  parameter SI.Resistance R1 = 20;
  parameter SI.Resistance R2 = 20;
  parameter SI.Resistance R3 = 20;
  parameter SI.Capacitance C1 = 0.00001;
  parameter SI.Capacitance C2 = 0.00001;
  parameter SI.Capacitance C3 = 0.00001;
  parameter SI.Inductance L1 = 0.001;
  parameter SI.Inductance L2 = 0.001;
  parameter SI.Inductance L3 = 0.001;
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {-74, -68}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {0, -30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {74, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {-74, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {0, 2}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {74, -4}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-74, -20}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R = R2) annotation(
    Placement(visible = true, transformation(origin = {0, 36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R = R3) annotation(
    Placement(visible = true, transformation(origin = {74, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(resistor1.p, pin1) annotation(
    Line(points = {{-74, -10}, {-100, -10}, {-100, -60}}, color = {0, 0, 255}));
  connect(pin2, resistor2.p) annotation(
    Line(points = {{-100, 0}, {-50, 0}, {-50, 46}, {0, 46}}, color = {0, 0, 255}));
  connect(resistor3.p, pin3) annotation(
    Line(points = {{74, 52}, {74, 52}, {74, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
  connect(capacitor2.n, ground1.p) annotation(
    Line(points = {{0, -40}, {0, -76}}, color = {0, 0, 255}));
  connect(capacitor3.n, ground1.p) annotation(
    Line(points = {{74, -56}, {74, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(capacitor1.p, inductor1.n) annotation(
    Line(points = {{-74, -58}, {-74, -58}, {-74, -54}, {-74, -54}}, color = {0, 0, 255}));
  connect(resistor1.n, inductor1.p) annotation(
    Line(points = {{-74, -30}, {-74, -30}, {-74, -30}, {-74, -34}}, color = {0, 0, 255}));
  connect(resistor2.n, inductor2.p) annotation(
    Line(points = {{0, 26}, {0, 26}, {0, 12}, {0, 12}}, color = {0, 0, 255}));
  connect(inductor2.n, capacitor2.p) annotation(
    Line(points = {{0, -8}, {0, -8}, {0, -8}, {0, -20}}, color = {0, 0, 255}));
  connect(resistor3.n, inductor3.p) annotation(
    Line(points = {{74, 32}, {74, 32}, {74, 6}, {74, 6}}, color = {0, 0, 255}));
  connect(capacitor3.p, inductor3.n) annotation(
    Line(points = {{74, -36}, {74, -36}, {74, -14}, {74, -14}}, color = {0, 0, 255}));
  connect(capacitor1.n, ground1.p) annotation(
    Line(points = {{-74, -78}, {-46, -78}, {-46, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
end RLC;
