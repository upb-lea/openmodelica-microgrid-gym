within omg_grid.Transformations;

model ABC2DQ_Currents
  Real Pi = 3.14159265;
  Modelica.Electrical.Analog.Interfaces.Pin a annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin b annotation(
    Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin c annotation(
    Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput d annotation(
    Placement(visible = true, transformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Blocks.Interfaces.RealOutput q annotation(
    Placement(visible = true, transformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-62, 60}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain1(k = 2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-61, 37}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain2(k = 2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-61, 11}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression1(y = 4 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-17, -2}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-27, 24}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {-4, 28}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {4, 2}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos2 annotation(
    Placement(visible = true, transformation(origin = {16, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product annotation(
    Placement(visible = true, transformation(origin = {46, 6}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos1 annotation(
    Placement(visible = true, transformation(origin = {24, 2}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos3 annotation(
    Placement(visible = true, transformation(origin = {16, 26}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product2 annotation(
    Placement(visible = true, transformation(origin = {40, 54}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product1 annotation(
    Placement(visible = true, transformation(origin = {34, 30}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
    Placement(visible = true, transformation(origin = {74, 54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.MultiSum multiSum1(nu = 3) annotation(
    Placement(visible = true, transformation(origin = {66, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product product5 annotation(
    Placement(visible = true, transformation(origin = {40, -24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add4(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {-6, -52}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {4, -76}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression2(y = 2 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-27, -54}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-17, -80}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Product product3 annotation(
    Placement(visible = true, transformation(origin = {46, -72}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin3 annotation(
    Placement(visible = true, transformation(origin = {25, -77}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain5(k = -2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-61, -67}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Product product4 annotation(
    Placement(visible = true, transformation(origin = {34, -48}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain3(k = -2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-61, -41}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin1 annotation(
    Placement(visible = true, transformation(origin = {17, -53}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain4(k = -2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-62, -18}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin2 annotation(
    Placement(visible = true, transformation(origin = {11, -31}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput theta annotation(
    Placement(visible = true, transformation(origin = {0, 112}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {0, 112}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
  Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor3 annotation(
    Placement(visible = true, transformation(origin = {-90, -50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
  Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor1 annotation(
    Placement(visible = true, transformation(origin = {-88, 70}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
  Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor2 annotation(
    Placement(visible = true, transformation(origin = {-90, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
equation
  connect(realExpression.y, add1.u2) annotation(
    Line(points = {{-19, 24}, {-11, 24}}, color = {0, 0, 127}));
  connect(realExpression1.y, add2.u2) annotation(
    Line(points = {{-9, -2}, {-3, -2}}, color = {0, 0, 127}));
  connect(gain2.y, product.u1) annotation(
    Line(points = {{-53, 11}, {39, 11}, {39, 10}}, color = {0, 0, 127}));
  connect(add2.y, cos1.u) annotation(
    Line(points = {{11, 2}, {17, 2}}, color = {0, 0, 127}));
  connect(cos1.y, product.u2) annotation(
    Line(points = {{31, 2}, {39, 2}}, color = {0, 0, 127}));
  connect(add1.y, cos3.u) annotation(
    Line(points = {{3, 28}, {4, 28}, {4, 26}, {9, 26}}, color = {0, 0, 127}));
  connect(cos2.y, product2.u2) annotation(
    Line(points = {{23, 50}, {33, 50}}, color = {0, 0, 127}));
  connect(gain.y, product2.u1) annotation(
    Line(points = {{-55, 60}, {-11, 60}, {-11, 58}, {33, 58}}, color = {0, 0, 127}));
  connect(gain1.y, product1.u1) annotation(
    Line(points = {{-53, 37}, {27, 37}, {27, 34}}, color = {0, 0, 127}));
  connect(cos3.y, product1.u2) annotation(
    Line(points = {{23, 26}, {27, 26}}, color = {0, 0, 127}));
  connect(multiSum.y, d) annotation(
    Line(points = {{86, 54}, {86, -94}, {-40, -94}, {-40, -110}}, color = {0, 0, 127}));
  connect(product1.y, multiSum.u[1]) annotation(
    Line(points = {{41, 30}, {55.5, 30}, {55.5, 54}, {64, 54}}, color = {0, 0, 127}));
  connect(product.y, multiSum.u[2]) annotation(
    Line(points = {{53, 6}, {64, 6}, {64, 54}}, color = {0, 0, 127}));
  connect(product2.y, multiSum.u[3]) annotation(
    Line(points = {{47, 54}, {64, 54}}, color = {0, 0, 127}));
  connect(sin2.y, product5.u2) annotation(
    Line(points = {{19, -31}, {30, -31}, {30, -28}, {33, -28}}, color = {0, 0, 127}));
  connect(product4.y, multiSum1.u[2]) annotation(
    Line(points = {{41, -48}, {56, -48}, {56, -30}}, color = {0, 0, 127}));
  connect(gain3.y, product4.u1) annotation(
    Line(points = {{-53, -41}, {27, -41}, {27, -44}}, color = {0, 0, 127}));
  connect(realExpression3.y, add3.u2) annotation(
    Line(points = {{-9, -80}, {-3, -80}}, color = {0, 0, 127}));
  connect(product5.y, multiSum1.u[3]) annotation(
    Line(points = {{47, -24}, {62.25, -24}, {62.25, -28}, {59.125, -28}, {59.125, -30}, {56, -30}}, color = {0, 0, 127}));
  connect(add3.y, sin3.u) annotation(
    Line(points = {{11, -76}, {28, -76}, {28, -77}, {17, -77}}, color = {0, 0, 127}));
  connect(sin1.u, add4.y) annotation(
    Line(points = {{9, -53}, {18, -53}, {18, -52}, {1, -52}}, color = {0, 0, 127}));
  connect(sin3.y, product3.u2) annotation(
    Line(points = {{33, -77}, {33, -76.5}, {39, -76.5}, {39, -76}}, color = {0, 0, 127}));
  connect(sin1.y, product4.u2) annotation(
    Line(points = {{25, -53}, {25, -50.5}, {27, -50.5}, {27, -52}}, color = {0, 0, 127}));
  connect(gain5.y, product3.u1) annotation(
    Line(points = {{-53, -67}, {39, -67}, {39, -68}}, color = {0, 0, 127}));
  connect(realExpression2.y, add4.u2) annotation(
    Line(points = {{-19, -54}, {-16, -54}, {-16, -56}, {-13, -56}}, color = {0, 0, 127}));
  connect(product3.y, multiSum1.u[1]) annotation(
    Line(points = {{53, -72}, {56, -72}, {56, -30}}, color = {0, 0, 127}));
  connect(gain4.y, product5.u1) annotation(
    Line(points = {{-55, -18}, {-35, -18}, {-35, -20}, {33, -20}}, color = {0, 0, 127}));
  connect(multiSum1.y, q) annotation(
    Line(points = {{78, -30}, {84, -30}, {84, -92}, {40, -92}, {40, -110}, {40, -110}}, color = {0, 0, 127}));
  connect(theta, cos2.u) annotation(
    Line(points = {{0, 112}, {0, 112}, {0, 50}, {8, 50}, {8, 50}}, color = {0, 0, 127}));
  connect(theta, add1.u1) annotation(
    Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, 32}, {-12, 32}, {-12, 32}}, color = {0, 0, 127}));
  connect(theta, add2.u1) annotation(
    Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, 6}, {-4, 6}, {-4, 6}}, color = {0, 0, 127}));
  connect(theta, sin2.u) annotation(
    Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, -32}, {2, -32}, {2, -30}}, color = {0, 0, 127}));
  connect(theta, add4.u1) annotation(
    Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, -48}, {-14, -48}, {-14, -48}}, color = {0, 0, 127}));
  connect(theta, add3.u1) annotation(
    Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, -72}, {-4, -72}, {-4, -72}}, color = {0, 0, 127}));
  connect(c, currentSensor3.p) annotation(
    Line(points = {{-102, -60}, {-90, -60}, {-90, -56}, {-90, -56}}, color = {0, 0, 255}));
  connect(b, currentSensor2.p) annotation(
    Line(points = {{-102, 0}, {-90, 0}, {-90, 4}, {-90, 4}, {-90, 4}}, color = {0, 0, 255}));
  connect(a, currentSensor1.p) annotation(
    Line(points = {{-100, 60}, {-88, 60}, {-88, 64}, {-88, 64}, {-88, 64}}, color = {0, 0, 255}));
  connect(currentSensor1.n, pin3) annotation(
    Line(points = {{-88, 76}, {-88, 76}, {-88, 84}, {94, 84}, {94, 60}, {100, 60}}, color = {0, 0, 255}));
  connect(currentSensor2.n, pin2) annotation(
    Line(points = {{-90, 16}, {-86, 16}, {-86, 82}, {92, 82}, {92, 0}, {100, 0}}, color = {0, 0, 255}));
  connect(currentSensor3.n, pin1) annotation(
    Line(points = {{-90, -44}, {-84, -44}, {-84, 80}, {90, 80}, {90, -60}, {100, -60}}, color = {0, 0, 255}));
  connect(currentSensor1.i, gain.u) annotation(
    Line(points = {{-82, 70}, {-72, 70}, {-72, 60}, {-70, 60}, {-70, 60}}, color = {0, 0, 127}));
  connect(gain4.u, currentSensor1.i) annotation(
    Line(points = {{-70, -18}, {-72, -18}, {-72, 70}, {-82, 70}, {-82, 70}}, color = {0, 0, 127}));
  connect(currentSensor2.i, gain1.u) annotation(
    Line(points = {{-84, 10}, {-80, 10}, {-80, 36}, {-70, 36}, {-70, 38}}, color = {0, 0, 127}));
  connect(currentSensor2.i, gain3.u) annotation(
    Line(points = {{-84, 10}, {-80, 10}, {-80, -42}, {-70, -42}, {-70, -40}}, color = {0, 0, 127}));
  connect(currentSensor3.i, gain2.u) annotation(
    Line(points = {{-84, -50}, {-76, -50}, {-76, 10}, {-70, 10}, {-70, 12}}, color = {0, 0, 127}));
  connect(currentSensor3.i, gain5.u) annotation(
    Line(points = {{-84, -50}, {-76, -50}, {-76, -68}, {-70, -68}, {-70, -66}}, color = {0, 0, 127}));
end ABC2DQ_Currents;
