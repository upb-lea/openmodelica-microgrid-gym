within omg_grid.Filter.LossesFilter;

model LCLC
  parameter SI.Capacitance C1 = 0.00001;
  parameter SI.Capacitance C2 = 0.00001;
  parameter SI.Capacitance C3 = 0.00001;
  parameter SI.Capacitance C4 = 0.00001;
  parameter SI.Capacitance C5 = 0.00001;
  parameter SI.Capacitance C6 = 0.00001;
  parameter SI.Inductance L1 = 0.001;
  parameter SI.Inductance L2 = 0.001;
  parameter SI.Inductance L3 = 0.001;
  parameter SI.Inductance L4 = 0.001;
  parameter SI.Inductance L5 = 0.001;
  parameter SI.Inductance L6 = 0.001;
  parameter SI.Resistance R1 = 0.01;
  parameter SI.Resistance R2 = 0.01;
  parameter SI.Resistance R3 = 0.01;
  parameter SI.Resistance R4 = 0.01;
  parameter SI.Resistance R5 = 0.01;
  parameter SI.Resistance R6 = 0.01;
  parameter SI.Resistance R7 = 0.01;
  parameter SI.Resistance R8 = 0.01;
  parameter SI.Resistance R9 = 0.01;
  parameter SI.Resistance R10 = 0.01;
  parameter SI.Resistance R11 = 0.01;
  parameter SI.Resistance R12 = 0.01;
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {-82, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {-82, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {-84, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
    Placement(visible = true, transformation(origin = {-2, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
    Placement(visible = true, transformation(origin = {-22, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
    Placement(visible = true, transformation(origin = {-42, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
    Placement(visible = true, transformation(origin = {16, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
    Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
    Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
    Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
    Placement(visible = true, transformation(origin = {34, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
    Placement(visible = true, transformation(origin = {34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
    Placement(visible = true, transformation(origin = {36, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor4(C = C4) annotation(
    Placement(visible = true, transformation(origin = {72, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor5(C = C5) annotation(
    Placement(visible = true, transformation(origin = {52, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor6(C = C6) annotation(
    Placement(visible = true, transformation(origin = {32, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R = R1) annotation(
    Placement(visible = true, transformation(origin = {-56, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R = R2) annotation(
    Placement(visible = true, transformation(origin = {-56, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor3(R = R3) annotation(
    Placement(visible = true, transformation(origin = {-56, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor4(R = R4) annotation(
    Placement(visible = true, transformation(origin = {-2, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor5(R = R5) annotation(
    Placement(visible = true, transformation(origin = {-22, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor6(R = R6) annotation(
    Placement(visible = true, transformation(origin = {-42, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor7(R = R7) annotation(
    Placement(visible = true, transformation(origin = {10, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor8(R = R8) annotation(
    Placement(visible = true, transformation(origin = {10, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor9(R = R9) annotation(
    Placement(visible = true, transformation(origin = {10, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Resistor resistor10(R = R10) annotation(
    Placement(visible = true, transformation(origin = {72, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor11(R = R11) annotation(
    Placement(visible = true, transformation(origin = {52, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor12(R = R12) annotation(
    Placement(visible = true, transformation(origin = {32, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(resistor10.p, inductor4.n) annotation(
    Line(points = {{72, -4}, {72, -4}, {72, 20}, {44, 20}, {44, 20}}, color = {0, 0, 255}));
  connect(inductor5.n, resistor11.p) annotation(
    Line(points = {{44, 44}, {54, 44}, {54, 44}, {52, 44}, {52, -4}, {52, -4}}, color = {0, 0, 255}));
  connect(resistor12.p, inductor6.n) annotation(
    Line(points = {{32, -4}, {32, -4}, {32, 10}, {58, 10}, {58, 70}, {46, 70}, {46, 70}}, color = {0, 0, 255}));
  connect(resistor11.n, capacitor5.p) annotation(
    Line(points = {{52, -24}, {52, -24}, {52, -28}, {52, -28}}, color = {0, 0, 255}));
  connect(resistor12.n, capacitor6.p) annotation(
    Line(points = {{32, -24}, {32, -24}, {32, -28}, {32, -28}}, color = {0, 0, 255}));
  connect(resistor1.n, resistor4.p) annotation(
    Line(points = {{-46, 20}, {-2, 20}, {-2, -4}, {-2, -4}, {-2, -4}}, color = {0, 0, 255}));
  connect(resistor2.n, resistor5.p) annotation(
    Line(points = {{-46, 44}, {-22, 44}, {-22, -4}, {-22, -4}, {-22, -4}}, color = {0, 0, 255}));
  connect(resistor6.p, resistor3.n) annotation(
    Line(points = {{-42, -4}, {-42, -4}, {-42, 70}, {-46, 70}, {-46, 70}}, color = {0, 0, 255}));
  connect(resistor1.n, resistor7.p) annotation(
    Line(points = {{-46, 20}, {-46, 20}, {-46, 20}, {0, 20}}, color = {0, 0, 255}));
  connect(resistor2.n, resistor8.p) annotation(
    Line(points = {{-46, 44}, {-46, 44}, {-46, 44}, {0, 44}}, color = {0, 0, 255}));
  connect(resistor3.n, resistor9.p) annotation(
    Line(points = {{-46, 70}, {0, 70}, {0, 70}, {0, 70}}, color = {0, 0, 255}));
  connect(resistor7.n, inductor4.p) annotation(
    Line(points = {{20, 20}, {24, 20}, {24, 20}, {24, 20}}, color = {0, 0, 255}));
  connect(resistor6.n, capacitor3.p) annotation(
    Line(points = {{-42, -24}, {-42, -24}, {-42, -28}, {-42, -28}}, color = {0, 0, 255}));
  connect(resistor5.n, capacitor2.p) annotation(
    Line(points = {{-22, -24}, {-22, -24}, {-22, -24}, {-22, -28}}, color = {0, 0, 255}));
  connect(resistor4.n, capacitor1.p) annotation(
    Line(points = {{-2, -24}, {-2, -24}, {-2, -28}, {-2, -28}}, color = {0, 0, 255}));
  connect(resistor10.n, capacitor4.p) annotation(
    Line(points = {{72, -24}, {72, -24}, {72, -28}, {72, -28}}, color = {0, 0, 255}));
  connect(resistor8.n, inductor5.p) annotation(
    Line(points = {{20, 44}, {24, 44}}, color = {0, 0, 255}));
  connect(resistor9.n, inductor6.p) annotation(
    Line(points = {{20, 70}, {26, 70}, {26, 70}, {26, 70}}, color = {0, 0, 255}));
  connect(inductor1.n, resistor1.p) annotation(
    Line(points = {{-72, 20}, {-66, 20}, {-66, 20}, {-66, 20}}, color = {0, 0, 255}));
  connect(inductor3.n, resistor3.p) annotation(
    Line(points = {{-74, 70}, {-74, 70}, {-74, 70}, {-66, 70}}, color = {0, 0, 255}));
  connect(inductor2.n, resistor2.p) annotation(
    Line(points = {{-72, 44}, {-66, 44}, {-66, 44}, {-66, 44}}, color = {0, 0, 255}));
  connect(inductor4.n, pin4) annotation(
    Line(points = {{44, 20}, {92, 20}, {92, -60}, {100, -60}}, color = {0, 0, 255}));
  connect(inductor6.n, pin6) annotation(
    Line(points = {{46, 70}, {76, 70}, {76, 60}, {100, 60}}, color = {0, 0, 255}));
  connect(capacitor6.n, ground1.p) annotation(
    Line(points = {{32, -48}, {16, -48}, {16, -60}}, color = {0, 0, 255}));
  connect(capacitor6.n, capacitor5.n) annotation(
    Line(points = {{32, -48}, {52, -48}, {52, -48}, {52, -48}}, color = {0, 0, 255}));
  connect(capacitor5.n, capacitor4.n) annotation(
    Line(points = {{52, -48}, {72, -48}, {72, -48}, {72, -48}}, color = {0, 0, 255}));
  connect(capacitor1.n, ground1.p) annotation(
    Line(points = {{-2, -48}, {16, -48}, {16, -60}, {16, -60}}, color = {0, 0, 255}));
  connect(inductor5.n, pin5) annotation(
    Line(points = {{44, 44}, {94, 44}, {94, 0}, {100, 0}}, color = {0, 0, 255}));
  connect(capacitor3.n, capacitor2.n) annotation(
    Line(points = {{-42, -48}, {-22, -48}, {-22, -48}, {-22, -48}}, color = {0, 0, 255}));
  connect(capacitor1.n, capacitor2.n) annotation(
    Line(points = {{-2, -48}, {-22, -48}, {-22, -48}, {-22, -48}}, color = {0, 0, 255}));
  connect(pin3, inductor3.p) annotation(
    Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-94, 70}}, color = {0, 0, 255}));
  connect(pin2, inductor2.p) annotation(
    Line(points = {{-100, 0}, {-95, 0}, {-95, 44}, {-92, 44}}, color = {0, 0, 255}));
  connect(pin1, inductor1.p) annotation(
    Line(points = {{-100, -60}, {-93, -60}, {-93, 20}, {-92, 20}}, color = {0, 0, 255}));
end LCLC;
