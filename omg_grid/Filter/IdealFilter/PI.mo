within OpenModelica_Microgrids.Filter.IdealFilter;

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
  Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
    Placement(visible = true, transformation(origin = {2, -16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
    Placement(visible = true, transformation(origin = {2, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
    Placement(visible = true, transformation(origin = {0, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
    Placement(visible = true, transformation(origin = {66, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
equation
  connect(inductor3.n, capacitor6.p) annotation(
    Line(points = {{10, 60}, {66, 60}, {66, -28}, {66, -28}}, color = {0, 0, 255}));
  connect(capacitor3.p, pin3) annotation(
    Line(points = {{-26, -28}, {-26, -28}, {-26, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
  connect(inductor1.n, pin4) annotation(
    Line(points = {{12, -16}, {70, -16}, {70, -60}, {100, -60}}, color = {0, 0, 255}));
  connect(inductor1.n, capacitor4.p) annotation(
    Line(points = {{12, -16}, {26, -16}, {26, -28}}, color = {0, 0, 255}));
  connect(pin1, inductor1.p) annotation(
    Line(points = {{-100, -60}, {-80, -60}, {-80, -16}, {-8, -16}}, color = {0, 0, 255}));
  connect(inductor3.n, pin6) annotation(
    Line(points = {{10, 60}, {100, 60}}, color = {0, 0, 255}));
  connect(pin3, inductor3.p) annotation(
    Line(points = {{-100, 60}, {-10, 60}}, color = {0, 0, 255}));
  connect(inductor2.n, pin5) annotation(
    Line(points = {{12, 0}, {100, 0}}, color = {0, 0, 255}));
  connect(inductor2.n, capacitor5.p) annotation(
    Line(points = {{12, 0}, {46, 0}, {46, -28}}, color = {0, 0, 255}));
  connect(inductor2.p, pin2) annotation(
    Line(points = {{-8, 0}, {-100, 0}}, color = {0, 0, 255}));
  connect(capacitor1.p, pin1) annotation(
    Line(points = {{-70, -28}, {-80, -28}, {-80, -60}, {-100, -60}}, color = {0, 0, 255}));
  connect(capacitor2.p, pin2) annotation(
    Line(points = {{-48, -28}, {-48, -28}, {-48, 0}, {-100, 0}, {-100, 0}}, color = {0, 0, 255}));
  connect(capacitor3.n, ground1.p) annotation(
    Line(points = {{-26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
  connect(capacitor4.n, ground1.p) annotation(
    Line(points = {{26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
  connect(capacitor2.n, capacitor3.n) annotation(
    Line(points = {{-48, -48}, {-26, -48}, {-26, -48}, {-26, -48}}, color = {0, 0, 255}));
  connect(capacitor1.n, capacitor2.n) annotation(
    Line(points = {{-70, -48}, {-48, -48}, {-48, -48}, {-48, -48}}, color = {0, 0, 255}));
  connect(capacitor5.n, capacitor4.n) annotation(
    Line(points = {{46, -48}, {26, -48}, {26, -48}, {26, -48}}, color = {0, 0, 255}));
  connect(capacitor6.n, capacitor5.n) annotation(
    Line(points = {{66, -48}, {46, -48}, {46, -48}, {46, -48}}, color = {0, 0, 255}));
end PI;
