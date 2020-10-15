within omg_grid.Filter.LossesFilter;

model LC
  parameter SI.Capacitance C1 = 0.00001;
  parameter SI.Capacitance C2 = 0.00001;
  parameter SI.Capacitance C3 = 0.00001;
  parameter SI.Inductance L1 = 0.001;
  parameter SI.Inductance L2 = 0.001;
  parameter SI.Inductance L3 = 0.001;
  parameter SI.Resistance R1 = 0.01;
  parameter SI.Resistance R2 = 0.01;
  parameter SI.Resistance R3 = 0.01;
  parameter SI.Resistance R4 = 0.01;
  parameter SI.Resistance R5 = 0.01;
  parameter SI.Resistance R6 = 0.01;
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {-60, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {-60, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
    Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
    Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
    Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {32, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {-8, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-34, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R = R2) annotation(
    Placement(visible = true, transformation(origin = {-34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R = R3) annotation(
    Placement(visible = true, transformation(origin = {-26, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor4(R = R4) annotation(
    Placement(visible = true, transformation(origin = {32, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor5(R = R5) annotation(
    Placement(visible = true, transformation(origin = {12, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor6(R = R6) annotation(
    Placement(visible = true, transformation(origin = {-8, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(resistor3.n, resistor6.p) annotation(
    Line(points = {{-16, 70}, {-8, 70}, {-8, 2}}, color = {0, 0, 255}));
  connect(resistor3.n, pin6) annotation(
    Line(points = {{-16, 70}, {80, 70}, {80, 60}, {100, 60}}, color = {0, 0, 255}));
  connect(inductor3.n, resistor3.p) annotation(
    Line(points = {{-50, 70}, {-36, 70}}, color = {0, 0, 255}));
  connect(pin4, resistor1.n) annotation(
    Line(points = {{100, -60}, {62, -60}, {62, 20}, {-24, 20}, {-24, 20}}, color = {0, 0, 255}));
  connect(resistor4.n, capacitor1.p) annotation(
    Line(points = {{32, -18}, {32, -18}, {32, -26}, {32, -26}}, color = {0, 0, 255}));
  connect(resistor5.n, capacitor2.p) annotation(
    Line(points = {{12, -18}, {12, -18}, {12, -26}, {12, -26}}, color = {0, 0, 255}));
  connect(resistor6.n, capacitor3.p) annotation(
    Line(points = {{-8, -18}, {-8, -18}, {-8, -26}, {-8, -26}}, color = {0, 0, 255}));
  connect(pin5, resistor2.n) annotation(
    Line(points = {{100, 0}, {78, 0}, {78, 44}, {-24, 44}, {-24, 44}}, color = {0, 0, 255}));
  connect(resistor2.n, resistor5.p) annotation(
    Line(points = {{-24, 44}, {12, 44}, {12, 2}, {12, 2}}, color = {0, 0, 255}));
  connect(resistor1.n, resistor4.p) annotation(
    Line(points = {{-24, 20}, {32, 20}, {32, 2}, {32, 2}}, color = {0, 0, 255}));
  connect(inductor1.n, resistor1.p) annotation(
    Line(points = {{-50, 20}, {-44, 20}, {-44, 20}, {-44, 20}}, color = {0, 0, 255}));
  connect(inductor2.n, resistor2.p) annotation(
    Line(points = {{-50, 44}, {-44, 44}, {-44, 44}, {-44, 44}}, color = {0, 0, 255}));
  connect(capacitor3.n, capacitor2.n) annotation(
    Line(points = {{-8, -46}, {12, -46}, {12, -46}, {12, -46}}, color = {0, 0, 255}));
  connect(capacitor2.n, ground1.p) annotation(
    Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
  connect(capacitor2.n, capacitor1.n) annotation(
    Line(points = {{12, -46}, {32, -46}, {32, -46}, {32, -46}}, color = {0, 0, 255}));
  connect(pin1, inductor1.p) annotation(
    Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
  connect(pin3, inductor3.p) annotation(
    Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
  connect(pin2, inductor2.p) annotation(
    Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
end LC;

