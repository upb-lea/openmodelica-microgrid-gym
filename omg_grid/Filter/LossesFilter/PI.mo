within omg_grid.Filter.LossesFilter;

model PI
  parameter SI.Capacitance C1 = 0.00001;
  parameter SI.Capacitance C2 = 0.00001;
  parameter SI.Capacitance C3 = 0.00001;
  parameter SI.Capacitance C4 = 0.00001;
  parameter SI.Capacitance C5 = 0.00001;
  parameter SI.Capacitance C6 = 0.00001;
  parameter SI.Inductance L1 = 0.001;
  parameter SI.Inductance L2 = 0.001;
  parameter SI.Inductance L3 = 0.001;
  parameter SI.Resistance R1 = 0.01;
  parameter SI.Resistance R2 = 0.01;
  parameter SI.Resistance R3 = 0.01;
  parameter SI.Resistance R4 = 0.01;
  parameter SI.Resistance R5 = 0.01;
  parameter SI.Resistance R6 = 0.01;
  parameter SI.Resistance R7 = 0.01;
  parameter SI.Resistance R8 = 0.01;
  parameter SI.Resistance R9 = 0.01;
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {-14, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {-14, 52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {-14, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {-70, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {-48, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {-26, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {0, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
    Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
    Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
    Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor4(C = C4) annotation(
    Placement(visible = true, transformation(origin = {26, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor5(C = C5) annotation(
    Placement(visible = true, transformation(origin = {46, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor6(C = C6) annotation(
    Placement(visible = true, transformation(origin = {64, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-70, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R = R2) annotation(
    Placement(visible = true, transformation(origin = {-48, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R = R3) annotation(
    Placement(visible = true, transformation(origin = {-26, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor4(R = R4) annotation(
    Placement(visible = true, transformation(origin = {10, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor5(R = R5) annotation(
    Placement(visible = true, transformation(origin = {10, 52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor6(R = R6) annotation(
    Placement(visible = true, transformation(origin = {10, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor7(R = R7) annotation(
    Placement(visible = true, transformation(origin = {26, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor8(R = R8) annotation(
    Placement(visible = true, transformation(origin = {46, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor9(R = R9) annotation(
    Placement(visible = true, transformation(origin = {64, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(inductor1.n, resistor4.p) annotation(
    Line(points = {{-4, 28}, {0, 28}, {0, 28}, {0, 28}}, color = {0, 0, 255}));
  connect(inductor2.n, resistor5.p) annotation(
    Line(points = {{-4, 52}, {-4, 52}, {-4, 52}, {0, 52}}, color = {0, 0, 255}));
  connect(inductor3.n, resistor6.p) annotation(
    Line(points = {{-4, 78}, {0, 78}, {0, 78}, {0, 78}}, color = {0, 0, 255}));
  connect(resistor3.p, pin3) annotation(
    Line(points = {{-26, 2}, {-26, 2}, {-26, 18}, {-36, 18}, {-36, 78}, {-90, 78}, {-90, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
  connect(resistor2.p, pin2) annotation(
    Line(points = {{-48, 2}, {-48, 2}, {-48, 52}, {-84, 52}, {-84, 0}, {-100, 0}, {-100, 0}}, color = {0, 0, 255}));
  connect(resistor1.p, pin1) annotation(
    Line(points = {{-70, 2}, {-80, 2}, {-80, -60}, {-98, -60}, {-98, -60}, {-100, -60}}, color = {0, 0, 255}));
  connect(resistor1.n, capacitor1.p) annotation(
    Line(points = {{-70, -18}, {-70, -18}, {-70, -28}, {-70, -28}}, color = {0, 0, 255}));
  connect(resistor2.n, capacitor2.p) annotation(
    Line(points = {{-48, -18}, {-48, -18}, {-48, -28}, {-48, -28}}, color = {0, 0, 255}));
  connect(resistor3.n, capacitor3.p) annotation(
    Line(points = {{-26, -18}, {-26, -18}, {-26, -28}, {-26, -28}}, color = {0, 0, 255}));
  connect(resistor9.n, capacitor6.p) annotation(
    Line(points = {{64, -18}, {64, -18}, {64, -28}, {64, -28}}, color = {0, 0, 255}));
  connect(capacitor6.n, capacitor5.n) annotation(
    Line(points = {{64, -48}, {46, -48}}, color = {0, 0, 255}));
  connect(capacitor5.p, resistor8.n) annotation(
    Line(points = {{46, -28}, {46, -28}, {46, -18}, {46, -18}}, color = {0, 0, 255}));
  connect(resistor5.n, resistor8.p) annotation(
    Line(points = {{20, 52}, {46, 52}, {46, 2}}, color = {0, 0, 255}));
  connect(resistor7.n, capacitor4.p) annotation(
    Line(points = {{26, -18}, {26, -18}, {26, -28}, {26, -28}}, color = {0, 0, 255}));
  connect(resistor6.n, pin6) annotation(
    Line(points = {{20, 78}, {84, 78}, {84, 60}, {100, 60}}, color = {0, 0, 255}));
  connect(resistor5.n, pin5) annotation(
    Line(points = {{20, 52}, {84, 52}, {84, 0}, {100, 0}}, color = {0, 0, 255}));
  connect(resistor4.n, pin4) annotation(
    Line(points = {{20, 28}, {74, 28}, {74, 28}, {76, 28}, {76, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
  connect(resistor6.n, resistor9.p) annotation(
    Line(points = {{20, 78}, {64, 78}, {64, 2}, {64, 2}}, color = {0, 0, 255}));
  connect(resistor4.n, resistor7.p) annotation(
    Line(points = {{20, 28}, {26, 28}, {26, 2}, {26, 2}}, color = {0, 0, 255}));
  connect(pin1, inductor1.p) annotation(
    Line(points = {{-100, -60}, {-80, -60}, {-80, 28}, {-24, 28}}, color = {0, 0, 255}));
  connect(pin3, inductor3.p) annotation(
    Line(points = {{-100, 60}, {-90, 60}, {-90, 78}, {-24, 78}}, color = {0, 0, 255}));
  connect(capacitor3.n, ground1.p) annotation(
    Line(points = {{-26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
  connect(capacitor4.n, ground1.p) annotation(
    Line(points = {{26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
  connect(inductor2.p, pin2) annotation(
    Line(points = {{-24, 52}, {-84, 52}, {-84, 0}, {-100, 0}, {-100, 0}}, color = {0, 0, 255}));
  connect(capacitor2.n, capacitor3.n) annotation(
    Line(points = {{-48, -48}, {-26, -48}, {-26, -48}, {-26, -48}}, color = {0, 0, 255}));
  connect(capacitor1.n, capacitor2.n) annotation(
    Line(points = {{-70, -48}, {-48, -48}, {-48, -48}, {-48, -48}}, color = {0, 0, 255}));
  connect(capacitor5.n, capacitor4.n) annotation(
    Line(points = {{46, -48}, {26, -48}, {26, -48}, {26, -48}}, color = {0, 0, 255}));
end PI;
