package grid
  import SI = Modelica.SIunits;

  package filter
    model pi
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
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-70, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-48, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-26, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor4(R = R4) annotation(
        Placement(visible = true, transformation(origin = {10, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor5(R = R5) annotation(
        Placement(visible = true, transformation(origin = {10, 52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor6(R = R6) annotation(
        Placement(visible = true, transformation(origin = {10, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor7(R = R7) annotation(
        Placement(visible = true, transformation(origin = {26, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor8(R = R8) annotation(
        Placement(visible = true, transformation(origin = {46, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor9(R = R9) annotation(
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
    end pi;

    model lcl
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
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
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-64, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-72, 86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {38, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-28, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
        Placement(visible = true, transformation(origin = {68, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
        Placement(visible = true, transformation(origin = {70, 40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
        Placement(visible = true, transformation(origin = {74, 62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1 annotation(
        Placement(visible = true, transformation(origin = {-36, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2 annotation(
        Placement(visible = true, transformation(origin = {-32, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3 annotation(
        Placement(visible = true, transformation(origin = {-30, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor4 annotation(
        Placement(visible = true, transformation(origin = {42, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor5 annotation(
        Placement(visible = true, transformation(origin = {8, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor6 annotation(
        Placement(visible = true, transformation(origin = {-22, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor7 annotation(
        Placement(visible = true, transformation(origin = {32, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor8 annotation(
        Placement(visible = true, transformation(origin = {40, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor9 annotation(
        Placement(visible = true, transformation(origin = {34, 68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(resistor2.n, resistor5.p) annotation(
        Line(points = {{-22, 48}, {8, 48}, {8, 0}}, color = {0, 0, 255}));
      connect(resistor8.p, resistor2.n) annotation(
        Line(points = {{30, 44}, {2, 44}, {2, 48}, {-22, 48}}, color = {0, 0, 255}));
      connect(resistor2.p, inductor2.n) annotation(
        Line(points = {{-42, 48}, {-50, 48}, {-50, 58}, {-54, 58}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 58}, {-74, 58}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 86}, {-82, 86}}, color = {0, 0, 255}));
      connect(resistor3.p, inductor3.n) annotation(
        Line(points = {{-40, 82}, {-47, 82}, {-47, 86}, {-62, 86}}, color = {0, 0, 255}));
      connect(resistor3.n, resistor9.p) annotation(
        Line(points = {{-20, 82}, {3, 82}, {3, 68}, {24, 68}}, color = {0, 0, 255}));
      connect(resistor6.p, resistor3.n) annotation(
        Line(points = {{-22, 2}, {-22, 41}, {-20, 41}, {-20, 82}}, color = {0, 0, 255}));
      connect(inductor6.n, pin6) annotation(
        Line(points = {{84, 62}, {84, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(resistor9.n, inductor6.p) annotation(
        Line(points = {{44, 68}, {55, 68}, {55, 62}, {64, 62}}, color = {0, 0, 255}));
      connect(inductor5.n, pin5) annotation(
        Line(points = {{80, 40}, {88, 40}, {88, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(resistor8.n, inductor5.p) annotation(
        Line(points = {{50, 44}, {55, 44}, {55, 40}, {60, 40}}, color = {0, 0, 255}));
      connect(resistor7.n, inductor4.p) annotation(
        Line(points = {{42, 30}, {54, 30}, {54, 6}, {58, 6}}, color = {0, 0, 255}));
      connect(resistor4.p, resistor7.p) annotation(
        Line(points = {{42, 0}, {42, 14.5}, {22, 14.5}, {22, 30}}, color = {0, 0, 255}));
      connect(resistor1.n, resistor7.p) annotation(
        Line(points = {{-26, 20}, {2, 20}, {2, 30}, {22, 30}}, color = {0, 0, 255}));
      connect(inductor4.n, pin4) annotation(
        Line(points = {{78, 6}, {80, 6}, {80, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor1.n) annotation(
        Line(points = {{12, -46}, {23, -46}, {23, -56}, {38, -56}}, color = {0, 0, 255}));
      connect(resistor4.n, capacitor1.p) annotation(
        Line(points = {{42, -20}, {42, -24}, {38, -24}, {38, -36}}, color = {0, 0, 255}));
      connect(resistor5.n, capacitor2.p) annotation(
        Line(points = {{8, -20}, {8, -24}, {12, -24}, {12, -26}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-28, -46}, {12, -46}}, color = {0, 0, 255}));
      connect(resistor6.n, capacitor3.p) annotation(
        Line(points = {{-22, -18}, {-22, -24}, {-28, -24}, {-28, -26}}, color = {0, 0, 255}));
      connect(resistor1.p, inductor1.n) annotation(
        Line(points = {{-46, 20}, {-50, 20}, {-50, 20}, {-50, 20}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
    end lcl;

    model lc
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
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-34, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-26, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor4(R = R4) annotation(
        Placement(visible = true, transformation(origin = {32, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor5(R = R5) annotation(
        Placement(visible = true, transformation(origin = {12, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor6(R = R6) annotation(
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
    end lc;

    model lclc
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
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-56, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-56, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-56, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor4(R = R4) annotation(
        Placement(visible = true, transformation(origin = {-2, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor5(R = R5) annotation(
        Placement(visible = true, transformation(origin = {-22, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor6(R = R6) annotation(
        Placement(visible = true, transformation(origin = {-42, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor7(R = R7) annotation(
        Placement(visible = true, transformation(origin = {10, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor8(R = R8) annotation(
        Placement(visible = true, transformation(origin = {10, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor9(R = R9) annotation(
        Placement(visible = true, transformation(origin = {10, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor10(R = R10) annotation(
        Placement(visible = true, transformation(origin = {72, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor11(R = R11) annotation(
        Placement(visible = true, transformation(origin = {52, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor12(R = R12) annotation(
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
    end lclc;

    model l
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Resistance R1 = 0.01;
      parameter SI.Resistance R2 = 0.01;
      parameter SI.Resistance R3 = 0.01;
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
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-32, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-32, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-32, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(resistor1.n, pin4) annotation(
        Line(points = {{-22, 20}, {14, 20}, {14, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(resistor2.n, pin5) annotation(
        Line(points = {{-22, 44}, {70, 44}, {70, 0}, {100, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(resistor3.n, pin6) annotation(
        Line(points = {{-22, 70}, {80, 70}, {80, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor1.n, resistor1.p) annotation(
        Line(points = {{-50, 20}, {-50, 20}, {-50, 20}, {-42, 20}}, color = {0, 0, 255}));
      connect(inductor2.n, resistor2.p) annotation(
        Line(points = {{-50, 44}, {-42, 44}, {-42, 44}, {-42, 44}}, color = {0, 0, 255}));
      connect(inductor3.n, resistor3.p) annotation(
        Line(points = {{-50, 70}, {-42, 70}, {-42, 70}, {-42, 70}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
    end l;
  end filter;

  package loads
    model rc
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
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-50, -48}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {0, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
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
    end rc;

    model c
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
    end c;

    model r
      parameter SI.Resistance R1 = 20;
      parameter SI.Resistance R2 = 20;
      parameter SI.Resistance R3 = 20;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(resistor3.n, ground1.p) annotation(
        Line(points = {{-56, 60}, {0, 60}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(resistor2.n, ground1.p) annotation(
        Line(points = {{-56, 0}, {-56, 0}, {-56, 0}, {0, 0}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(resistor1.n, ground1.p) annotation(
        Line(points = {{-56, -60}, {0, -60}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin1, resistor1.p) annotation(
        Line(points = {{-100, -60}, {-76, -60}, {-76, -60}, {-76, -60}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {-100, 0}, {-100, 0}, {-76, 0}}, color = {0, 0, 255}));
      connect(pin3, resistor3.p) annotation(
        Line(points = {{-100, 60}, {-76, 60}, {-76, 60}, {-76, 60}}, color = {0, 0, 255}));
    end r;

    model l
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
    end l;

    model rlc
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
        Placement(visible = true, transformation(origin = {-58, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {0, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {56, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-76, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-20, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {38, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-40, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {18, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {74, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(inductor2.n, ground1.p) annotation(
        Line(points = {{-20, -2}, {-20, -2}, {-20, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(inductor1.p, pin1) annotation(
        Line(points = {{-76, -34}, {-76, -22}, {-90, -22}, {-90, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(inductor1.n, ground1.p) annotation(
        Line(points = {{-76, -54}, {-76, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(ground1.p, resistor1.n) annotation(
        Line(points = {{0, -76}, {0, -62}, {-40, -62}, {-40, -54}}, color = {0, 0, 255}));
      connect(resistor1.p, pin1) annotation(
        Line(points = {{-40, -34}, {-40, -22}, {-90, -22}, {-90, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(resistor2.n, ground1.p) annotation(
        Line(points = {{18, -2}, {18, -2}, {18, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {-50, 0}, {-50, 24}, {18, 24}, {18, 18}, {18, 18}, {18, 18}}, color = {0, 0, 255}));
      connect(resistor3.p, pin3) annotation(
        Line(points = {{74, 52}, {74, 52}, {74, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(ground1.p, resistor3.n) annotation(
        Line(points = {{0, -76}, {0, -76}, {0, -62}, {74, -62}, {74, 32}, {74, 32}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-50, 0}, {-50, 24}, {-20, 24}, {-20, 18}}, color = {0, 0, 255}));
      connect(capacitor2.p, pin2) annotation(
        Line(points = {{0, 18}, {0, 24}, {-50, 24}, {-50, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{0, -2}, {0, -76}}, color = {0, 0, 255}));
      connect(inductor3.n, ground1.p) annotation(
        Line(points = {{38, 32}, {38, 32}, {38, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {38, 60}, {38, 52}, {38, 52}}, color = {0, 0, 255}));
      connect(capacitor3.n, ground1.p) annotation(
        Line(points = {{56, 32}, {56, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(pin3, capacitor3.p) annotation(
        Line(points = {{-100, 60}, {56, 60}, {56, 52}}, color = {0, 0, 255}));
      connect(capacitor3.p, pin3) annotation(
        Line(points = {{56, 52}, {56, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(capacitor1.n, ground1.p) annotation(
        Line(points = {{-58, -54}, {-58, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor1.p, pin1) annotation(
        Line(points = {{-58, -34}, {-58, -22}, {-90, -22}, {-90, -60}, {-100, -60}}, color = {0, 0, 255}));
    end rlc;

    model lc
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
    end lc;

    model rl
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
      grid.components.resistor resistor1 annotation(
        Placement(visible = true, transformation(origin = {-40, -20}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2 annotation(
        Placement(visible = true, transformation(origin = {0, 12}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3 annotation(
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
    end rl;
  end loads;

  package inverters
    model inverter
      //  input Real regler1;
      //  input Real regler2;
      //  input Real regler3;
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {-74, -82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage1 annotation(
        Placement(visible = true, transformation(origin = {-74, -42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage2 annotation(
        Placement(visible = true, transformation(origin = {-74, 20}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage3 annotation(
        Placement(visible = true, transformation(origin = {-76, 70}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealInput u1 annotation(
        Placement(visible = true, transformation(origin = {-104, -60}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, -60}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u3 annotation(
        Placement(visible = true, transformation(origin = {-104, 60}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 60}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u2 annotation(
        Placement(visible = true, transformation(origin = {-104, 4.44089e-16}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 4.44089e-16}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(u1, signalVoltage1.v) annotation(
        Line(points = {{-104, -60}, {-54, -60}, {-54, -42}, {-62, -42}}, color = {0, 0, 127}));
      connect(signalVoltage2.p, pin2) annotation(
        Line(points = {{-74, 30}, {80, 30}, {80, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(signalVoltage3.p, pin3) annotation(
        Line(points = {{-76, 80}, {80, 80}, {80, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(signalVoltage1.p, pin1) annotation(
        Line(points = {{-74, -32}, {80, -32}, {80, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(signalVoltage3.n, ground1.p) annotation(
        Line(points = {{-76, 60}, {-76, 48}, {-82, 48}, {-82, -72}, {-74, -72}}, color = {0, 0, 255}));
      connect(signalVoltage3.v, u3) annotation(
        Line(points = {{-64, 70}, {-56, 70}, {-56, 60}, {-104, 60}}, color = {0, 0, 127}));
      connect(u2, signalVoltage2.v) annotation(
        Line(points = {{-104, 0}, {-56, 0}, {-56, 20}, {-62, 20}}, color = {0, 0, 127}));
      connect(signalVoltage2.n, ground1.p) annotation(
        Line(points = {{-74, 10}, {-82, 10}, {-82, -72}, {-74, -72}, {-74, -72}}, color = {0, 0, 255}));
      connect(signalVoltage1.n, ground1.p) annotation(
        Line(points = {{-74, -52}, {-74, -72}}, color = {0, 0, 255}));
/*  connect(signalVoltage1.v, regler1) annotation(
        Line);
      connect(signalVoltage2.v, regler2) annotation(
        Line);
      connect(signalVoltage3.v, regler3) annotation(
        Line);
    */
      annotation(
        uses(Modelica(version = "3.2.3")));
    end inverter;
  end inverters;

  package ideal_filter
    model pi
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
    end pi;

    model lcl
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Inductance L4 = 0.001;
      parameter SI.Inductance L5 = 0.001;
      parameter SI.Inductance L6 = 0.001;
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
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {32, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-8, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
        Placement(visible = true, transformation(origin = {68, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
        Placement(visible = true, transformation(origin = {74, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
        Placement(visible = true, transformation(origin = {64, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(inductor2.n, inductor5.p) annotation(
        Line(points = {{-50, 44}, {-50, 44}, {-50, 44}, {64, 44}}, color = {0, 0, 255}));
      connect(inductor2.n, capacitor2.p) annotation(
        Line(points = {{-50, 44}, {12, 44}, {12, -26}, {12, -26}}, color = {0, 0, 255}));
      connect(inductor1.n, inductor4.p) annotation(
        Line(points = {{-50, 20}, {-50, 20}, {-50, 20}, {58, 20}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor1.p) annotation(
        Line(points = {{-50, 20}, {32, 20}, {32, -26}, {32, -26}}, color = {0, 0, 255}));
      connect(inductor3.n, capacitor3.p) annotation(
        Line(points = {{-50, 70}, {-8, 70}, {-8, -26}, {-8, -26}}, color = {0, 0, 255}));
      connect(inductor3.n, inductor6.p) annotation(
        Line(points = {{-50, 70}, {54, 70}, {54, 70}, {54, 70}}, color = {0, 0, 255}));
      connect(inductor4.n, pin4) annotation(
        Line(points = {{78, 20}, {80, 20}, {80, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor6.n, pin6) annotation(
        Line(points = {{74, 70}, {84, 70}, {84, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(inductor5.n, pin5) annotation(
        Line(points = {{84, 44}, {88, 44}, {88, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor1.n) annotation(
        Line(points = {{12, -46}, {32, -46}, {32, -46}, {32, -46}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-8, -46}, {12, -46}, {12, -46}, {12, -46}}, color = {0, 0, 255}));
    end lcl;

    model lc
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
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
    equation
      connect(inductor1.n, pin4) annotation(
        Line(points = {{-50, 20}, {54, 20}, {54, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor2.n, pin5) annotation(
        Line(points = {{-50, 44}, {68, 44}, {68, 0}, {100, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(inductor3.n, pin6) annotation(
        Line(points = {{-50, 70}, {80, 70}, {80, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor3.n, capacitor3.p) annotation(
        Line(points = {{-50, 70}, {-8, 70}, {-8, -26}, {-8, -26}}, color = {0, 0, 255}));
      connect(inductor2.n, capacitor2.p) annotation(
        Line(points = {{-50, 44}, {12, 44}, {12, -26}, {12, -26}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor1.p) annotation(
        Line(points = {{-50, 20}, {32, 20}, {32, -26}, {32, -26}}, color = {0, 0, 255}));
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
    end lc;

    model lclc
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
    equation
      connect(inductor4.n, capacitor4.p) annotation(
        Line(points = {{44, 20}, {72, 20}, {72, -28}, {72, -28}}, color = {0, 0, 255}));
      connect(inductor6.n, capacitor6.p) annotation(
        Line(points = {{46, 70}, {64, 70}, {64, 4}, {32, 4}, {32, -28}}, color = {0, 0, 255}));
      connect(capacitor3.p, inductor3.n) annotation(
        Line(points = {{-42, -28}, {-42, -28}, {-42, 70}, {-74, 70}, {-74, 70}}, color = {0, 0, 255}));
      connect(capacitor2.p, inductor2.n) annotation(
        Line(points = {{-22, -28}, {-22, -28}, {-22, 44}, {-72, 44}, {-72, 44}}, color = {0, 0, 255}));
      connect(inductor5.n, capacitor5.p) annotation(
        Line(points = {{44, 44}, {52, 44}, {52, -28}, {52, -28}}, color = {0, 0, 255}));
      connect(inductor3.n, inductor6.p) annotation(
        Line(points = {{-74, 70}, {26, 70}, {26, 70}, {26, 70}}, color = {0, 0, 255}));
      connect(inductor2.n, inductor5.p) annotation(
        Line(points = {{-72, 44}, {-72, 44}, {-72, 44}, {24, 44}}, color = {0, 0, 255}));
      connect(inductor1.n, inductor4.p) annotation(
        Line(points = {{-72, 20}, {24, 20}, {24, 20}, {24, 20}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor1.p) annotation(
        Line(points = {{-72, 20}, {-2, 20}, {-2, -28}, {-2, -28}}, color = {0, 0, 255}));
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
    end lclc;

    model l
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
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
    equation
      connect(inductor3.n, pin6) annotation(
        Line(points = {{-50, 70}, {80, 70}, {80, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor2.n, pin5) annotation(
        Line(points = {{-50, 44}, {78, 44}, {78, 0}, {100, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(inductor1.n, pin4) annotation(
        Line(points = {{-50, 20}, {60, 20}, {60, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
    end l;
  end ideal_filter;

  package components
    model resistor
      parameter SI.Resistance R(start = 1);
      extends Modelica.Electrical.Analog.Interfaces.OnePort;
    equation
      v = R * i;
      annotation(
        Documentation(info = "<html>
<p>The linear resistor connects the branch voltage <em>v</em> with the branch current <em>i</em> by <em>i*R = v</em>. The Resistance <em>R</em> is allowed to be positive, zero, or negative.</p>
</html>", revisions = "<html>
<ul>
<li><em> August 07, 2009   </em>
       by Anton Haumer<br> temperature dependency of resistance added<br>
       </li>
<li><em> March 11, 2009   </em>
       by Christoph Clauss<br> conditional heat port added<br>
       </li>
<li><em> 1998   </em>
       by Christoph Clauss<br> initially implemented<br>
       </li>
</ul>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Text(extent = {{-150, -40}, {150, -80}}, textString = "R=%R"), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
    end resistor;

    block LimPID "P, PI, PD, and PID controller with limited output, anti-windup compensation, setpoint weighting and optional feed-forward"
      import Modelica.Blocks.Types.InitPID;
      import Modelica.Blocks.Types.Init;
      import Modelica.Blocks.Types.SimpleController;
      extends Modelica.Blocks.Interfaces.SVcontrol;
      output Real controlError = u_s - u_m "Control error (set point - measurement)";
      parameter .Modelica.Blocks.Types.SimpleController controllerType = .Modelica.Blocks.Types.SimpleController.PID "Type of controller";
      parameter Real k(min = 0, unit = "1") = 1 "Gain of controller";
      parameter Modelica.SIunits.Time Ti(min = Modelica.Constants.small) = 0.5 "Time constant of Integrator block" annotation(
        Dialog(enable = controllerType == .Modelica.Blocks.Types.SimpleController.PI or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Modelica.SIunits.Time Td(min = 0) = 0.1 "Time constant of Derivative block" annotation(
        Dialog(enable = controllerType == .Modelica.Blocks.Types.SimpleController.PD or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Real yMax(start = 1) "Upper limit of output";
      parameter Real yMin = -yMax "Lower limit of output";
      parameter Real wp(min = 0) = 1 "Set-point weight for Proportional block (0..1)";
      parameter Real wd(min = 0) = 0 "Set-point weight for Derivative block (0..1)" annotation(
        Dialog(enable = controllerType == .Modelica.Blocks.Types.SimpleController.PD or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Real Ni(min = 100 * Modelica.Constants.eps) = 0.9 "Ni*Ti is time constant of anti-windup compensation" annotation(
        Dialog(enable = controllerType == .Modelica.Blocks.Types.SimpleController.PI or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Real Nd(min = 100 * Modelica.Constants.eps) = 10 "The higher Nd, the more ideal the derivative block" annotation(
        Dialog(enable = controllerType == .Modelica.Blocks.Types.SimpleController.PD or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Boolean withFeedForward = false "Use feed-forward input?" annotation(
        Evaluate = true,
        choices(checkBox = true));
      parameter Real kFF = 1 "Gain of feed-forward input" annotation(
        Dialog(enable = withFeedForward));
      parameter .Modelica.Blocks.Types.InitPID initType = .Modelica.Blocks.Types.InitPID.DoNotUse_InitialIntegratorState "Type of initialization (1: no init, 2: steady state, 3: initial state, 4: initial output)" annotation(
        Evaluate = true,
        Dialog(group = "Initialization"));
      parameter Real xi_start = 0 "Initial or guess value for integrator output (= integrator state)" annotation(
        Dialog(group = "Initialization", enable = controllerType == .Modelica.Blocks.Types.SimpleController.PI or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Real xd_start = 0 "Initial or guess value for state of derivative block" annotation(
        Dialog(group = "Initialization", enable = controllerType == .Modelica.Blocks.Types.SimpleController.PD or controllerType == .Modelica.Blocks.Types.SimpleController.PID));
      parameter Real y_start = 0 "Initial value of output" annotation(
        Dialog(enable = initType == .Modelica.Blocks.Types.InitPID.InitialOutput, group = "Initialization"));
      parameter Modelica.Blocks.Types.LimiterHomotopy homotopyType = Modelica.Blocks.Types.LimiterHomotopy.Linear "Simplified model for homotopy-based initialization" annotation(
        Evaluate = true,
        Dialog(group = "Initialization"));
      parameter Boolean strict = false "= true, if strict limits with noEvent(..)" annotation(
        Evaluate = true,
        choices(checkBox = true),
        Dialog(tab = "Advanced"));
      parameter Boolean limitsAtInit = true "Has no longer an effect and is only kept for backwards compatibility (the implementation uses now the homotopy operator)" annotation(
        Dialog(tab = "Dummy"),
        Evaluate = true,
        choices(checkBox = true));
      constant Modelica.SIunits.Time unitTime = 1 annotation(
        HideResult = true);
      Modelica.Blocks.Interfaces.RealInput u_ff if withFeedForward "Optional connector of feed-forward input signal" annotation(
        Placement(transformation(origin = {60, -120}, extent = {{20, -20}, {-20, 20}}, rotation = 270)));
      Modelica.Blocks.Math.Add addP(k1 = wp, k2 = -1) annotation(
        Placement(transformation(extent = {{-80, 40}, {-60, 60}})));
      Modelica.Blocks.Math.Add addD(k1 = wd, k2 = -1) if with_D annotation(
        Placement(transformation(extent = {{-80, -10}, {-60, 10}})));
      Modelica.Blocks.Math.Gain P(k = 1) annotation(
        Placement(transformation(extent = {{-50, 40}, {-30, 60}})));
      Modelica.Blocks.Continuous.Integrator I(k = unitTime / Ti, y_start = xi_start, initType = if initType == InitPID.SteadyState then Init.SteadyState else if initType == InitPID.InitialState or initType == InitPID.DoNotUse_InitialIntegratorState then Init.InitialState else Init.NoInit) if with_I annotation(
        Placement(transformation(extent = {{-50, -60}, {-30, -40}})));
      Modelica.Blocks.Continuous.Derivative D(k = Td / unitTime, T = max([Td / Nd, 1.e-14]), x_start = xd_start, initType = if initType == InitPID.SteadyState or initType == InitPID.InitialOutput then Init.SteadyState else if initType == InitPID.InitialState then Init.InitialState else Init.NoInit) if with_D annotation(
        Placement(transformation(extent = {{-50, -10}, {-30, 10}})));
      Modelica.Blocks.Math.Gain gainPID(k = k) annotation(
        Placement(transformation(extent = {{20, -10}, {40, 10}})));
      Modelica.Blocks.Math.Add3 addPID annotation(
        Placement(transformation(extent = {{-10, -10}, {10, 10}})));
      Modelica.Blocks.Math.Add3 addI(k2 = -1) if with_I annotation(
        Placement(transformation(extent = {{-80, -60}, {-60, -40}})));
      Modelica.Blocks.Math.Add addSat(k1 = +1, k2 = -1) if with_I annotation(
        Placement(transformation(origin = {80, -50}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
      Modelica.Blocks.Math.Gain gainTrack(k = 1 / (k * Ni)) if with_I annotation(
        Placement(transformation(extent = {{0, -80}, {-20, -60}})));
      Modelica.Blocks.Nonlinear.Limiter limiter(uMax = yMax, uMin = yMin, strict = strict, limitsAtInit = limitsAtInit, homotopyType = homotopyType) annotation(
        Placement(transformation(extent = {{70, -10}, {90, 10}})));
    protected
      parameter Boolean with_I = controllerType == SimpleController.PI or controllerType == SimpleController.PID annotation(
        Evaluate = true,
        HideResult = true);
      parameter Boolean with_D = controllerType == SimpleController.PD or controllerType == SimpleController.PID annotation(
        Evaluate = true,
        HideResult = true);
    public
      Modelica.Blocks.Sources.Constant Dzero(k = 0) if not with_D annotation(
        Placement(transformation(extent = {{-40, 20}, {-30, 30}})));
      Modelica.Blocks.Sources.Constant Izero(k = 0) if not with_I annotation(
        Placement(transformation(extent = {{0, -55}, {-10, -45}})));
      Modelica.Blocks.Sources.Constant FFzero(k = 0) if not withFeedForward annotation(
        Placement(transformation(extent = {{30, -35}, {40, -25}})));
      Modelica.Blocks.Math.Add addFF(k1 = 1, k2 = kFF) annotation(
        Placement(transformation(extent = {{48, -6}, {60, 6}})));
    initial equation
      if initType == InitPID.InitialOutput then
        gainPID.y = y_start;
      end if;
    equation
      if initType == InitPID.InitialOutput and (y_start < yMin or y_start > yMax) then
        Modelica.Utilities.Streams.error("LimPID: Start value y_start (=" + String(y_start) + ") is outside of the limits of yMin (=" + String(yMin) + ") and yMax (=" + String(yMax) + ")");
      end if;
      connect(u_s, addP.u1) annotation(
        Line(points = {{-120, 0}, {-96, 0}, {-96, 56}, {-82, 56}}, color = {0, 0, 127}));
      connect(u_s, addD.u1) annotation(
        Line(points = {{-120, 0}, {-96, 0}, {-96, 6}, {-82, 6}}, color = {0, 0, 127}));
      connect(u_s, addI.u1) annotation(
        Line(points = {{-120, 0}, {-96, 0}, {-96, -42}, {-82, -42}}, color = {0, 0, 127}));
      connect(addP.y, P.u) annotation(
        Line(points = {{-59, 50}, {-52, 50}}, color = {0, 0, 127}));
      connect(addD.y, D.u) annotation(
        Line(points = {{-59, 0}, {-52, 0}}, color = {0, 0, 127}));
      connect(addI.y, I.u) annotation(
        Line(points = {{-59, -50}, {-52, -50}}, color = {0, 0, 127}));
      connect(P.y, addPID.u1) annotation(
        Line(points = {{-29, 50}, {-20, 50}, {-20, 8}, {-12, 8}}, color = {0, 0, 127}));
      connect(D.y, addPID.u2) annotation(
        Line(points = {{-29, 0}, {-12, 0}}, color = {0, 0, 127}));
      connect(I.y, addPID.u3) annotation(
        Line(points = {{-29, -50}, {-20, -50}, {-20, -8}, {-12, -8}}, color = {0, 0, 127}));
      connect(limiter.y, addSat.u1) annotation(
        Line(points = {{91, 0}, {94, 0}, {94, -20}, {86, -20}, {86, -38}}, color = {0, 0, 127}));
      connect(limiter.y, y) annotation(
        Line(points = {{91, 0}, {110, 0}}, color = {0, 0, 127}));
      connect(addSat.y, gainTrack.u) annotation(
        Line(points = {{80, -61}, {80, -70}, {2, -70}}, color = {0, 0, 127}));
      connect(gainTrack.y, addI.u3) annotation(
        Line(points = {{-21, -70}, {-88, -70}, {-88, -58}, {-82, -58}}, color = {0, 0, 127}));
      connect(u_m, addP.u2) annotation(
        Line(points = {{0, -120}, {0, -92}, {-92, -92}, {-92, 44}, {-82, 44}}, color = {0, 0, 127}, thickness = 0.5));
      connect(u_m, addD.u2) annotation(
        Line(points = {{0, -120}, {0, -92}, {-92, -92}, {-92, -6}, {-82, -6}}, color = {0, 0, 127}, thickness = 0.5));
      connect(u_m, addI.u2) annotation(
        Line(points = {{0, -120}, {0, -92}, {-92, -92}, {-92, -50}, {-82, -50}}, color = {0, 0, 127}, thickness = 0.5));
      connect(Dzero.y, addPID.u2) annotation(
        Line(points = {{-29.5, 25}, {-24, 25}, {-24, 0}, {-12, 0}}, color = {0, 0, 127}));
      connect(Izero.y, addPID.u3) annotation(
        Line(points = {{-10.5, -50}, {-20, -50}, {-20, -8}, {-12, -8}}, color = {0, 0, 127}));
      connect(addPID.y, gainPID.u) annotation(
        Line(points = {{11, 0}, {18, 0}}, color = {0, 0, 127}));
      connect(addFF.y, limiter.u) annotation(
        Line(points = {{60.6, 0}, {68, 0}}, color = {0, 0, 127}));
      connect(gainPID.y, addFF.u1) annotation(
        Line(points = {{41, 0}, {44, 0}, {44, 3.6}, {46.8, 3.6}}, color = {0, 0, 127}));
      connect(FFzero.y, addFF.u2) annotation(
        Line(points = {{40.5, -30}, {44, -30}, {44, -3.6}, {46.8, -3.6}}, color = {0, 0, 127}));
      connect(addFF.u2, u_ff) annotation(
        Line(points = {{46.8, -3.6}, {44, -3.6}, {44, -92}, {60, -92}, {60, -120}}, color = {0, 0, 127}));
      connect(addFF.y, addSat.u2) annotation(
        Line(points = {{60.6, 0}, {64, 0}, {64, -20}, {74, -20}, {74, -38}}, color = {0, 0, 127}));
      annotation(
        defaultComponentName = "PID",
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Line(points = {{-80, 78}, {-80, -90}}, color = {192, 192, 192}), Polygon(points = {{-80, 90}, {-88, 68}, {-72, 68}, {-80, 90}}, lineColor = {192, 192, 192}, fillColor = {192, 192, 192}, fillPattern = FillPattern.Solid), Line(points = {{-90, -80}, {82, -80}}, color = {192, 192, 192}), Polygon(points = {{90, -80}, {68, -72}, {68, -88}, {90, -80}}, lineColor = {192, 192, 192}, fillColor = {192, 192, 192}, fillPattern = FillPattern.Solid), Line(points = {{-80, -80}, {-80, -20}, {30, 60}, {80, 60}}, color = {0, 0, 127}), Text(extent = {{-20, -20}, {80, -60}}, lineColor = {192, 192, 192}, textString = "%controllerType"), Line(visible = strict, points = {{30, 60}, {81, 60}}, color = {255, 0, 0})}),
        Diagram(graphics = {Text(extent = {{79, -112}, {129, -102}}, lineColor = {0, 0, 255}, textString = " (feed-forward)")}),
        Documentation(info = "<html>
    <p>
    Via parameter <strong>controllerType</strong> either <strong>P</strong>, <strong>PI</strong>, <strong>PD</strong>,
    or <strong>PID</strong> can be selected. If, e.g., PI is selected, all components belonging to the
    D-part are removed from the block (via conditional declarations).
    The example model
    <a href=\"modelica://Modelica.Blocks.Examples.PID_Controller\">Modelica.Blocks.Examples.PID_Controller</a>
    demonstrates the usage of this controller.
    Several practical aspects of PID controller design are incorporated
    according to chapter 3 of the book:
    </p>
    
    <dl>
    <dt>&Aring;str&ouml;m K.J., and H&auml;gglund T.:</dt>
    <dd> <strong>PID Controllers: Theory, Design, and Tuning</strong>.
       Instrument Society of America, 2nd edition, 1995.
    </dd>
    </dl>
    
    <p>
    Besides the additive <strong>proportional, integral</strong> and <strong>derivative</strong>
    part of this controller, the following features are present:
    </p>
    <ul>
    <li> The output of this controller is limited. If the controller is
       in its limits, anti-windup compensation is activated to drive
       the integrator state to zero.</li>
    <li> The high-frequency gain of the derivative part is limited
       to avoid excessive amplification of measurement noise.</li>
    <li> Setpoint weighting is present, which allows to weight
       the setpoint in the proportional and the derivative part
       independently from the measurement. The controller will respond
       to load disturbances and measurement noise independently of this setting
       (parameters wp, wd). However, setpoint changes will depend on this
       setting. For example, it is useful to set the setpoint weight wd
       for the derivative part to zero, if steps may occur in the
       setpoint signal.</li>
    <li> Optional feed-forward. It is possible to add a feed-forward signal.
       The feed-forward signal is added before limitation.</li>
    </ul>
    
    <p>
    The parameters of the controller can be manually adjusted by performing
    simulations of the closed loop system (= controller + plant connected
    together) and using the following strategy:
    </p>
    
    <ol>
    <li> Set very large limits, e.g., yMax = Modelica.Constants.inf</li>
    <li> Select a <strong>P</strong>-controller and manually enlarge parameter <strong>k</strong>
       (the total gain of the controller) until the closed-loop response
       cannot be improved any more.</li>
    <li> Select a <strong>PI</strong>-controller and manually adjust parameters
       <strong>k</strong> and <strong>Ti</strong> (the time constant of the integrator).
       The first value of Ti can be selected, such that it is in the
       order of the time constant of the oscillations occurring with
       the P-controller. If, e.g., vibrations in the order of T=10 ms
       occur in the previous step, start with Ti=0.01 s.</li>
    <li> If you want to make the reaction of the control loop faster
       (but probably less robust against disturbances and measurement noise)
       select a <strong>PID</strong>-Controller and manually adjust parameters
       <strong>k</strong>, <strong>Ti</strong>, <strong>Td</strong> (time constant of derivative block).</li>
    <li> Set the limits yMax and yMin according to your specification.</li>
    <li> Perform simulations such that the output of the PID controller
       goes in its limits. Tune <strong>Ni</strong> (Ni*Ti is the time constant of
       the anti-windup compensation) such that the input to the limiter
       block (= limiter.u) goes quickly enough back to its limits.
       If Ni is decreased, this happens faster. If Ni=infinity, the
       anti-windup compensation is switched off and the controller works bad.</li>
    </ol>
    
    <p>
    <strong>Initialization</strong>
    </p>
    
    <p>
    This block can be initialized in different
    ways controlled by parameter <strong>initType</strong>. The possible
    values of initType are defined in
    <a href=\"modelica://Modelica.Blocks.Types.InitPID\">Modelica.Blocks.Types.InitPID</a>.
    This type is identical to
    <a href=\"modelica://Modelica.Blocks.Types.Init\">Types.Init</a>,
    with the only exception that the additional option
    <strong>DoNotUse_InitialIntegratorState</strong> is added for
    backward compatibility reasons (= integrator is initialized with
    InitialState whereas differential part is initialized with
    NoInit which was the initialization in version 2.2 of the Modelica
    standard library).
    </p>
    
    <p>
    Based on the setting of initType, the integrator (I) and derivative (D)
    blocks inside the PID controller are initialized according to the following table:
    </p>
    
    <table border=1 cellspacing=0 cellpadding=2>
    <tr><td><strong>initType</strong></td>
        <td><strong>I.initType</strong></td>
        <td><strong>D.initType</strong></td></tr>
    
    <tr><td><strong>NoInit</strong></td>
        <td>NoInit</td>
        <td>NoInit</td></tr>
    
    <tr><td><strong>SteadyState</strong></td>
        <td>SteadyState</td>
        <td>SteadyState</td></tr>
    
    <tr><td><strong>InitialState</strong></td>
        <td>InitialState</td>
        <td>InitialState</td></tr>
    
    <tr><td><strong>InitialOutput</strong><br>
            and initial equation: y = y_start</td>
        <td>NoInit</td>
        <td>SteadyState</td></tr>
    
    <tr><td><strong>DoNotUse_InitialIntegratorState</strong></td>
        <td>InitialState</td>
        <td>NoInit</td></tr>
    </table>
    
    <p>
    In many cases, the most useful initial condition is
    <strong>SteadyState</strong> because initial transients are then no longer
    present. If initType = InitPID.SteadyState, then in some
    cases difficulties might occur. The reason is the
    equation of the integrator:
    </p>
    
    <pre>
     <strong>der</strong>(y) = k*u;
    </pre>
    
    <p>
    The steady state equation \"der(x)=0\" leads to the condition that the input u to the
    integrator is zero. If the input u is already (directly or indirectly) defined
    by another initial condition, then the initialization problem is <strong>singular</strong>
    (has none or infinitely many solutions). This situation occurs often
    for mechanical systems, where, e.g., u = desiredSpeed - measuredSpeed and
    since speed is both a state and a derivative, it is natural to
    initialize it with zero. As sketched this is, however, not possible.
    The solution is to not initialize u_m or the variable that is used
    to compute u_m by an algebraic equation.
    </p>
    
    <p>
    When initializing in steady-state, homotopy-based initialization can help the convergence of the solver,
    by using a simplified model a the beginning of the solution process. Different options are available.
    </p>
    
    <ul>
    <li><strong>homotopyType=Linear</strong> (default): the limitations are removed from the simplified model,
    making it linear. Use this if you know that the controller will not be saturated at steady state.</li>
    <li><strong>homotopyType=UpperLimit</strong>: if it is known a priori the controller will be stuck at the upper
    limit yMax, this option assumes y = yMax as a simplified model.</li>
    <li><strong>homotopyType=LowerLimit</strong>: if it is known a priori the controller will be stuck at the lower
    limit yMin, this option assumes y = yMin as a simplified model.</li>
    <li><strong>homotopyType=NoHomotopy</strong>: this option does not apply any simplification and keeps the
    limiter active throughout the homotopy transformation. Use this if it is unknown whether the controller
    is saturated or not at initialization and if the limitations on the output must be enforced throughout
    the entire homotopy transformation.</li>
    </ul>
    
    <p>
    The parameter <strong>limitAtInit</strong> is obsolete since MSL 3.2.2 and only kept for backwards compatibility.
    </p>
    </html>"));
    end LimPID;

    block PI "Proportional-Integral controller"
      import Modelica.Blocks.Types.Init;
      parameter Real k(unit = "1") = 1 "Gain";
      parameter SIunits.Time T(start = 1, min = Modelica.Constants.small) "Time Constant (T>0 required)";
      parameter Modelica.Blocks.Types.Init initType = Modelica.Blocks.Types.Init.NoInit "Type of initialization (1: no init, 2: steady state, 3: initial state, 4: initial output)" annotation(
        Evaluate = true,
        Dialog(group = "Initialization"));
      parameter Real x_start = 0 "Initial or guess value of state" annotation(
        Dialog(group = "Initialization"));
      parameter Real y_start = 0 "Initial value of output" annotation(
        Dialog(enable = initType == Init.SteadyState or initType == Init.InitialOutput, group = "Initialization"));
      extends Interfaces.SISO;
      output Real x(start = x_start) "State of block";
    initial equation
      if initType == Init.SteadyState then
        der(x) = 0;
      elseif initType == Init.InitialState then
        x = x_start;
      elseif initType == Init.InitialOutput then
        y = y_start;
      end if;
    equation
      der(x) = u / T;
      y = k * (x + u);
      annotation(
        defaultComponentName = "PI",
        Documentation(info = "<html>
    <p>
    This blocks defines the transfer function between the input u and
    the output y as <em>PI</em> system:
    </p>
    <pre>
                   1
     y = k * (1 + ---) * u
                  T*s
             T*s + 1
       = k * ------- * u
               T*s
    </pre>
    <p>
    If you would like to be able to change easily between different
    transfer functions (FirstOrder, SecondOrder, ... ) by changing
    parameters, use the general model class <strong>TransferFunction</strong>
    instead and model a PI SISO system with parameters<br>
    b = {k*T, k}, a = {T, 0}.
    </p>
    <pre>
    Example:
    
     parameter: k = 0.3,  T = 0.4
    
     results in:
                 0.4 s + 1
        y = 0.3 ----------- * u
                   0.4 s
    </pre>
    
    <p>
    It might be difficult to initialize the PI component in steady state
    due to the integrator part.
    This is discussed in the description of package
    <a href=\"modelica://Modelica.Blocks.Continuous#info\">Continuous</a>.
    </p>
    
    </html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Line(points = {{-80, 78}, {-80, -90}}, color = {192, 192, 192}), Polygon(points = {{-80, 90}, {-88, 68}, {-72, 68}, {-80, 90}}, lineColor = {192, 192, 192}, fillColor = {192, 192, 192}, fillPattern = FillPattern.Solid), Line(points = {{-90, -80}, {82, -80}}, color = {192, 192, 192}), Polygon(points = {{90, -80}, {68, -72}, {68, -88}, {90, -80}}, lineColor = {192, 192, 192}, fillColor = {192, 192, 192}, fillPattern = FillPattern.Solid), Line(points = {{-80.0, -80.0}, {-80.0, -20.0}, {60.0, 80.0}}, color = {0, 0, 127}), Text(extent = {{0, 6}, {60, -56}}, lineColor = {192, 192, 192}, textString = "PI"), Text(extent = {{-150, -150}, {150, -110}}, textString = "T=%T")}),
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-60, 60}, {60, -60}}, lineColor = {0, 0, 255}), Text(extent = {{-68, 24}, {-24, -18}}, textString = "k"), Text(extent = {{-32, 48}, {60, 0}}, textString = "T s + 1"), Text(extent = {{-30, -8}, {52, -40}}, textString = "T s"), Line(points = {{-24, 0}, {54, 0}}), Line(points = {{-100, 0}, {-60, 0}}, color = {0, 0, 255}), Line(points = {{62, 0}, {100, 0}}, color = {0, 0, 255})}));
    end PI;
    
    model IdealClosingSwitch "Ideal electrical closer"
      extends Modelica.Electrical.Analog.Interfaces.IdealSwitch;
      Modelica.Blocks.Interfaces.BooleanInput control
        "true => p--n connected, false => switch open" annotation (Placement(
            transformation(
            origin={0,120},
            extent={{-20,-20},{20,20}},
            rotation=270)));
    equation
      off = not control;
      annotation (defaultComponentName="switch",
        Documentation(info="<html>
    <p>
    The switching behaviour of the ideal closing switch is controlled by the input signal control: off = not control.<br>
    For further details, see partial model <a href=\"modelica://Modelica.Electrical.Analog.Interfaces.IdealSwitch\">IdealSwitch</a>.
    </p>
    </html>", revisions="<html>
    <ul>
    <li><em>February 7, 2016   </em>
         by Anton Haumer<br> extending from partial IdealSwitch<br>
         </li>
    <li><em> March 11, 2009   </em>
         by Christoph Clauss<br> conditional heat port added<br>
         </li>
    <li><em> 1998   </em>
         by Christoph Clauss<br> initially implemented<br>
         </li>
    </ul>
    </html>"),
        Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{100,
                100}}), graphics={
            Line(
              visible=useHeatPort,
              points={{0,-100},{0,25}},
              color={127,0,0},
              pattern=LinePattern.Dot)}));
    end IdealClosingSwitch;
  end components;

  package plls
    model pll
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-100, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-100, -14}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 15) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {48, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {64, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {78, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-100, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-100, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, ground.p) annotation(
        Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, ground.p) annotation(
        Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_c.n, ground.p) annotation(
        Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{43, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {43, 20}}, color = {0, 0, 127}));
      connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
        Line(points = {{59, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{74, 22}, {68, 22}, {68, 22}, {68, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -6}, {-6, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {82, 22}, {82, 22}, {82, 22}}, color = {0, 0, 127}));
    end pll;
  end plls;

  package transforms
    model abc2AlphaBeta
      Modelica.Blocks.Interfaces.RealInput a annotation(
        Placement(visible = true, transformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput b annotation(
        Placement(visible = true, transformation(origin = {-104, 12}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 12}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput c annotation(
        Placement(visible = true, transformation(origin = {-104, -18}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, -18}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-40, 78}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = -1 / 3) annotation(
        Placement(visible = true, transformation(origin = {-39, 55}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = -1 / 3) annotation(
        Placement(visible = true, transformation(origin = {-39, 29}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {58, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = -1 / sqrt(3)) annotation(
        Placement(visible = true, transformation(origin = {-32, -18}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain4(k = 1 / sqrt(3)) annotation(
        Placement(visible = true, transformation(origin = {-32, -64}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum1(nu = 2) annotation(
        Placement(visible = true, transformation(origin = {48, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput alpha annotation(
        Placement(visible = true, transformation(origin = {102, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {102, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput beta annotation(
        Placement(visible = true, transformation(origin = {102, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {102, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, gain.u) annotation(
        Line(points = {{-104, 40}, {-77, 40}, {-77, 78}, {-48, 78}}, color = {0, 0, 127}));
      connect(gain1.u, b) annotation(
        Line(points = {{-48, 56}, {-71, 56}, {-71, 12}, {-104, 12}}, color = {0, 0, 127}));
      connect(c, gain2.u) annotation(
        Line(points = {{-104, -18}, {-62, -18}, {-62, 30}, {-48, 30}}, color = {0, 0, 127}));
      connect(gain.y, multiSum.u[1]) annotation(
        Line(points = {{-34, 78}, {48, 78}, {48, 66}}, color = {0, 0, 127}));
      connect(gain1.y, multiSum.u[2]) annotation(
        Line(points = {{-32, 56}, {48, 56}, {48, 66}, {48, 66}}, color = {0, 0, 127}));
      connect(gain2.y, multiSum.u[3]) annotation(
        Line(points = {{-32, 30}, {48, 30}, {48, 66}, {48, 66}}, color = {0, 0, 127}));
      connect(gain4.u, b) annotation(
        Line(points = {{-48, -64}, {-73, -64}, {-73, -62}, {-72, -62}, {-72, 12}, {-104, 12}}, color = {0, 0, 127}));
      connect(gain3.u, c) annotation(
        Line(points = {{-48, -18}, {-96, -18}, {-96, -18}, {-104, -18}}, color = {0, 0, 127}));
      connect(gain3.y, multiSum1.u[1]) annotation(
        Line(points = {{-16, -18}, {38, -18}, {38, -34}, {38, -34}}, color = {0, 0, 127}));
      connect(gain4.y, multiSum1.u[2]) annotation(
        Line(points = {{-16, -64}, {38, -64}, {38, -34}, {38, -34}}, color = {0, 0, 127}));
      connect(multiSum.y, alpha) annotation(
        Line(points = {{70, 66}, {96, 66}, {96, 64}, {102, 64}}, color = {0, 0, 127}));
      connect(multiSum1.y, beta) annotation(
        Line(points = {{60, -34}, {102, -34}}, color = {0, 0, 127}));
    end abc2AlphaBeta;
  end transforms;

  model network
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.rc rc1 annotation(
      Placement(visible = true, transformation(origin = {70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter2 annotation(
      Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc2 annotation(
      Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p1 annotation(
      Placement(visible = true, transformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i2p1 annotation(
      Placement(visible = true, transformation(origin = {-104, -42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, -42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p2 annotation(
      Placement(visible = true, transformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i2p2 annotation(
      Placement(visible = true, transformation(origin = {-104, -30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, -30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i2p3 annotation(
      Placement(visible = true, transformation(origin = {-104, -18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, -18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p3 annotation(
      Placement(visible = true, transformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    ideal_filter.lcl lcl1 annotation(
      Placement(visible = true, transformation(origin = {-30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  grid.plls.pll pll annotation(
      Placement(visible = true, transformation(origin = {0, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(lc2.pin4, rc1.pin1) annotation(
      Line(points = {{40, 24}, {60, 24}, {60, 24}, {60, 24}}, color = {0, 0, 255}));
    connect(lc2.pin5, rc1.pin2) annotation(
      Line(points = {{40, 30}, {60, 30}, {60, 30}, {60, 30}}, color = {0, 0, 255}));
    connect(lc2.pin6, rc1.pin3) annotation(
      Line(points = {{40, 36}, {60, 36}, {60, 36}, {60, 36}}, color = {0, 0, 255}));
    connect(lc1.pin6, lc2.pin3) annotation(
      Line(points = {{-20, 36}, {20, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(lc1.pin5, lc2.pin2) annotation(
      Line(points = {{-20, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(lc1.pin4, lc2.pin1) annotation(
      Line(points = {{-20, 24}, {20, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    connect(inverter1.pin3, lc1.pin3) annotation(
      Line(points = {{-60, 36}, {-40, 36}}, color = {0, 0, 255}));
    connect(inverter1.pin2, lc1.pin2) annotation(
      Line(points = {{-60, 30}, {-40, 30}}, color = {0, 0, 255}));
    connect(inverter1.pin1, lc1.pin1) annotation(
      Line(points = {{-60, 24}, {-40, 24}}, color = {0, 0, 255}));
    connect(i1p1, inverter1.u1) annotation(
      Line(points = {{-104, 18}, {-86, 18}, {-86, 24}, {-80, 24}, {-80, 24}}, color = {0, 0, 127}));
    connect(i1p2, inverter1.u2) annotation(
      Line(points = {{-104, 30}, {-80, 30}, {-80, 30}, {-80, 30}}, color = {0, 0, 127}));
    connect(i1p3, inverter1.u3) annotation(
      Line(points = {{-104, 42}, {-86, 42}, {-86, 36}, {-80, 36}}, color = {0, 0, 127}));
    connect(i2p3, inverter2.u3) annotation(
      Line(points = {{-104, -18}, {-88, -18}, {-88, -24}, {-80, -24}, {-80, -24}}, color = {0, 0, 127}));
    connect(i2p2, inverter2.u2) annotation(
      Line(points = {{-104, -30}, {-80, -30}, {-80, -30}, {-80, -30}}, color = {0, 0, 127}));
    connect(i2p1, inverter2.u1) annotation(
      Line(points = {{-104, -42}, {-90, -42}, {-90, -36}, {-80, -36}, {-80, -36}}, color = {0, 0, 127}));
    connect(inverter2.pin3, lcl1.pin3) annotation(
      Line(points = {{-60, -24}, {-40, -24}, {-40, -24}, {-40, -24}}, color = {0, 0, 255}));
    connect(inverter2.pin2, lcl1.pin2) annotation(
      Line(points = {{-60, -30}, {-40, -30}, {-40, -30}, {-40, -30}}, color = {0, 0, 255}));
    connect(inverter2.pin1, lcl1.pin1) annotation(
      Line(points = {{-60, -36}, {-40, -36}, {-40, -36}, {-40, -36}}, color = {0, 0, 255}));
  connect(pll.a, lcl1.pin6) annotation(
      Line(points = {{-10, -56}, {-14, -56}, {-14, -24}, {-20, -24}, {-20, -24}, {-20, -24}}, color = {0, 0, 255}));
  connect(pll.b, lcl1.pin5) annotation(
      Line(points = {{-10, -58}, {-16, -58}, {-16, -30}, {-20, -30}, {-20, -30}}, color = {0, 0, 255}));
  connect(pll.c, lcl1.pin4) annotation(
      Line(points = {{-10, -62}, {-18, -62}, {-18, -36}, {-20, -36}, {-20, -36}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end network;

  model pll_Test
    transforms.abc2AlphaBeta abc2AlphaBeta annotation(
      Placement(visible = true, transformation(origin = {-14, 4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine(amplitude = 230 * 1.414, freqHz = 50) annotation(
      Placement(visible = true, transformation(origin = {-90, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine1(amplitude = 230 * 1.414, freqHz = 50, phase = -2.0944) annotation(
      Placement(visible = true, transformation(origin = {-88, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine2(amplitude = 230 * 1.414, freqHz = 50, phase = -4.18879) annotation(
      Placement(visible = true, transformation(origin = {-88, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter annotation(
      Placement(visible = true, transformation(origin = {-14, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.plls.pll pll annotation(
      Placement(visible = true, transformation(origin = {28, 56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(abc2AlphaBeta.a, sine.y) annotation(
      Line(points = {{-24, 8}, {-44, 8}, {-44, 34}, {-78, 34}, {-78, 34}}, color = {0, 0, 127}));
    connect(abc2AlphaBeta.b, sine1.y) annotation(
      Line(points = {{-24, 6}, {-77, 6}}, color = {0, 0, 127}));
    connect(sine2.y, abc2AlphaBeta.c) annotation(
      Line(points = {{-76, -26}, {-44, -26}, {-44, 2}, {-24, 2}, {-24, 2}}, color = {0, 0, 127}));
    connect(inverter.u3, sine.y) annotation(
      Line(points = {{-24, 64}, {-70, 64}, {-70, 34}, {-78, 34}}, color = {0, 0, 127}));
    connect(inverter.u2, sine1.y) annotation(
      Line(points = {{-24, 58}, {-60, 58}, {-60, 6}, {-76, 6}}, color = {0, 0, 127}));
    connect(inverter.u1, sine2.y) annotation(
      Line(points = {{-24, 52}, {-54, 52}, {-54, -26}, {-76, -26}}, color = {0, 0, 127}));
    connect(pll.a, inverter.pin3) annotation(
      Line(points = {{18, 60}, {4, 60}, {4, 64}, {-4, 64}, {-4, 64}}, color = {0, 0, 255}));
    connect(pll.b, inverter.pin2) annotation(
      Line(points = {{18, 58}, {-4, 58}, {-4, 58}, {-4, 58}}, color = {0, 0, 255}));
    connect(pll.c, inverter.pin1) annotation(
      Line(points = {{18, 54}, {4, 54}, {4, 52}, {-4, 52}, {-4, 52}}, color = {0, 0, 255}));
  end pll_Test;
  annotation(
    uses(Modelica(version = "3.2.3")));
end grid;