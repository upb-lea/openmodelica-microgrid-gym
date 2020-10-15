within omg_grid.Loads;

model RC
  parameter SI.Resistance R1 = 20;
  parameter SI.Resistance R2 = 20;
  parameter SI.Resistance R3 = 20;
  parameter SI.Capacitance C1 = 0.00001;
  parameter SI.Capacitance C2 = 0.00001;
  parameter SI.Capacitance C3 = 0.00001;
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {-66, -48}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {-32, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {48, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-50, -48}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R = R2) annotation(
    Placement(visible = true, transformation(origin = {0, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R = R3) annotation(
    Placement(visible = true, transformation(origin = {72, -2}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(resistor2.n, ground1.p) annotation(
    Line(points = {{0, -20}, {0, -20}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(pin2, resistor2.p) annotation(
    Line(points = {{-100, 0}, {0, 0}, {0, 0}, {0, 0}}, color = {0, 0, 255}));
  connect(resistor1.p, pin1) annotation(
    Line(points = {{-50, -38}, {-50, -38}, {-50, -22}, {-90, -22}, {-90, -60}, {-100, -60}, {-100, -60}}, color = {0, 0, 255}));
  connect(resistor1.n, ground1.p) annotation(
    Line(points = {{-50, -58}, {-50, -58}, {-50, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(resistor3.n, ground1.p) annotation(
    Line(points = {{72, -12}, {72, -12}, {72, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
  connect(pin3, resistor3.p) annotation(
    Line(points = {{-100, 60}, {68, 60}, {68, 60}, {72, 60}, {72, 8}, {72, 8}}, color = {0, 0, 255}));
  connect(capacitor1.p, pin1) annotation(
    Line(points = {{-66, -38}, {-66, -22}, {-90, -22}, {-90, -60}, {-100, -60}}, color = {0, 0, 255}));
  connect(capacitor3.p, pin3) annotation(
    Line(points = {{48, 10}, {48, 60}, {-100, 60}}, color = {0, 0, 255}));
  connect(pin3, capacitor3.p) annotation(
    Line(points = {{-100, 60}, {48, 60}, {48, 10}}, color = {0, 0, 255}));
  connect(capacitor3.n, ground1.p) annotation(
    Line(points = {{48, -10}, {48, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(capacitor2.p, pin2) annotation(
    Line(points = {{-32, 0}, {-100, 0}}, color = {0, 0, 255}));
  connect(capacitor2.n, ground1.p) annotation(
    Line(points = {{-32, -20}, {-32, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
  connect(capacitor1.n, ground1.p) annotation(
    Line(points = {{-66, -58}, {-66, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
end RC;
