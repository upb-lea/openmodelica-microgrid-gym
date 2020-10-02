within OpenModelica_Microgrids.PLLs;

model PLL_DQ
  Real Pi = 3.14159265;
  Modelica.Electrical.Analog.Interfaces.Pin a annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin b annotation(
    Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin c annotation(
    Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground annotation(
    Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
  Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
    Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
  Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
    Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
  Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
    Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
  OpenModelica_Microgrids.Transformations.ABC2AlphaBeta abc2AlphaBeta annotation(
    Placement(visible = true, transformation(origin = {-52, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin annotation(
    Placement(visible = true, transformation(origin = {-8, 56}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
  Modelica.Blocks.Math.Cos cos annotation(
    Placement(visible = true, transformation(origin = {-8, 46}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
  Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
    Placement(visible = true, transformation(origin = {-31, 83}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
  Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
    Placement(visible = true, transformation(origin = {-31, 69}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
  Modelica.Blocks.Math.Product alphaSin annotation(
    Placement(visible = true, transformation(origin = {-5, 83}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
  Modelica.Blocks.Math.Product betaCos annotation(
    Placement(visible = true, transformation(origin = {-7, 69}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
  Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
    Placement(visible = true, transformation(origin = {14, 78}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Continuous.PI pi(T = 0.05, k = 20) annotation(
    Placement(visible = true, transformation(origin = {28, 78}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
    Placement(visible = true, transformation(origin = {50, 76}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
    Placement(visible = true, transformation(origin = {30, 58}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
    Placement(visible = true, transformation(origin = {66, 76}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
    Placement(visible = true, transformation(origin = {80, 76}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
  Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {-6, 8}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-64, 40}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain1(k = 2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-63, 17}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Product product2 annotation(
    Placement(visible = true, transformation(origin = {38, 34}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos2 annotation(
    Placement(visible = true, transformation(origin = {14, 30}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
    Placement(visible = true, transformation(origin = {64, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product product1 annotation(
    Placement(visible = true, transformation(origin = {32, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product annotation(
    Placement(visible = true, transformation(origin = {44, -14}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {2, -18}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos1 annotation(
    Placement(visible = true, transformation(origin = {22, -18}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression1(y = 4 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-19, -22}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain2(k = 2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-63, -9}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput d annotation(
    Placement(visible = true, transformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-29, 4}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos3 annotation(
    Placement(visible = true, transformation(origin = {14, 6}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput theta annotation(
    Placement(visible = true, transformation(origin = {110, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product product3 annotation(
    Placement(visible = true, transformation(origin = {44, -84}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product5 annotation(
    Placement(visible = true, transformation(origin = {38, -36}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain3(k = -2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-63, -53}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {2, -88}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression2(y = 2 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-29, -66}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin1 annotation(
    Placement(visible = true, transformation(origin = {15, -65}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain4(k = -2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-64, -30}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Gain gain5(k = -2 / 3) annotation(
    Placement(visible = true, transformation(origin = {-63, -79}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.MultiSum multiSum1(nu = 3) annotation(
    Placement(visible = true, transformation(origin = {64, -42}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput q annotation(
    Placement(visible = true, transformation(origin = {110, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin2 annotation(
    Placement(visible = true, transformation(origin = {9, -43}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * Pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-19, -92}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin3 annotation(
    Placement(visible = true, transformation(origin = {23, -89}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Add add4(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {-8, -64}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product4 annotation(
    Placement(visible = true, transformation(origin = {32, -60}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
equation
  connect(a, voltageSensor_a.p) annotation(
    Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
  connect(b, voltageSensor_b.p) annotation(
    Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
  connect(c, voltageSensor_c.p) annotation(
    Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
  connect(voltageSensor_a.n, ground.p) annotation(
    Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
  connect(voltageSensor_b.n, ground.p) annotation(
    Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
  connect(voltageSensor_c.n, ground.p) annotation(
    Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
  connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
    Line(points = {{-35, 83}, {-42, 83}, {-42, 84}}, color = {0, 0, 127}));
  connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
    Line(points = {{-28, 83}, {-18.5, 83}, {-18.5, 85}, {-9, 85}}, color = {0, 0, 127}));
  connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
    Line(points = {{-28, 69}, {-19.5, 69}, {-19.5, 71}, {-11, 71}}, color = {0, 0, 127}));
  connect(sin.y, alphaSin.u2) annotation(
    Line(points = {{-12, 56}, {-20, 56}, {-20, 81}, {-9, 81}}, color = {0, 0, 127}));
  connect(cos.y, betaCos.u2) annotation(
    Line(points = {{-12, 46}, {-16, 46}, {-16, 67}, {-11, 67}}, color = {0, 0, 127}));
  connect(add.u1, alphaSin.y) annotation(
    Line(points = {{9, 80}, {-0.25, 80}, {-0.25, 83}, {-2, 83}}, color = {0, 0, 127}));
  connect(betaCos.y, add.u2) annotation(
    Line(points = {{-4, 69}, {-1, 69}, {-1, 76}, {9, 76}}, color = {0, 0, 127}));
  connect(pi.u, add.y) annotation(
    Line(points = {{21, 78}, {18, 78}}, color = {0, 0, 127}));
  connect(deg2rad.u, f2theta.y) annotation(
    Line(points = {{75, 76}, {70, 76}}, color = {0, 0, 127}));
  connect(cos.u, deg2rad.y) annotation(
    Line(points = {{-3, 46}, {92, 46}, {92, 76}, {84, 76}}, color = {0, 0, 127}));
  connect(gain.y, product2.u1) annotation(
    Line(points = {{-57, 40}, {-13, 40}, {-13, 38}, {31, 38}}, color = {0, 0, 127}));
  connect(cos3.y, product1.u2) annotation(
    Line(points = {{21, 6}, {25, 6}}, color = {0, 0, 127}));
  connect(cos1.y, product.u2) annotation(
    Line(points = {{29, -18}, {37, -18}}, color = {0, 0, 127}));
  connect(gain1.y, product1.u1) annotation(
    Line(points = {{-55, 17}, {25, 17}, {25, 14}}, color = {0, 0, 127}));
  connect(realExpression.y, add1.u2) annotation(
    Line(points = {{-21, 4}, {-13, 4}}, color = {0, 0, 127}));
  connect(product2.y, multiSum.u[3]) annotation(
    Line(points = {{45, 34}, {55.5, 34}, {55.5, 28}, {54, 28}}, color = {0, 0, 127}));
  connect(product.y, multiSum.u[1]) annotation(
    Line(points = {{51, -14}, {54, -14}, {54, 28}}, color = {0, 0, 127}));
  connect(add1.y, cos3.u) annotation(
    Line(points = {{1, 8}, {4, 8}, {4, 6}, {7, 6}}, color = {0, 0, 127}));
  connect(add2.y, cos1.u) annotation(
    Line(points = {{9, -18}, {15, -18}}, color = {0, 0, 127}));
  connect(multiSum.y, d) annotation(
    Line(points = {{76, 28}, {95, 28}, {95, 0}, {110, 0}}, color = {0, 0, 127}));
  connect(product1.y, multiSum.u[2]) annotation(
    Line(points = {{39, 10}, {54, 10}, {54, 28}}, color = {0, 0, 127}));
  connect(cos2.y, product2.u2) annotation(
    Line(points = {{21, 30}, {31, 30}}, color = {0, 0, 127}));
  connect(realExpression1.y, add2.u2) annotation(
    Line(points = {{-11, -22}, {-5, -22}}, color = {0, 0, 127}));
  connect(gain2.y, product.u1) annotation(
    Line(points = {{-55, -9}, {37, -9}, {37, -10}}, color = {0, 0, 127}));
  connect(voltageSensor_a.v, gain.u) annotation(
    Line(points = {{-80, 50}, {-74, 50}, {-74, 40}, {-71, 40}}, color = {0, 0, 127}));
  connect(deg2rad.y, cos2.u) annotation(
    Line(points = {{84, 76}, {92, 76}, {92, 46}, {7, 46}, {7, 30}}, color = {0, 0, 127}));
  connect(deg2rad.y, add1.u1) annotation(
    Line(points = {{84, 76}, {92, 76}, {92, 46}, {6, 46}, {6, 24}, {-17, 24}, {-17, 12}, {-13, 12}}, color = {0, 0, 127}));
  connect(deg2rad.y, add2.u1) annotation(
    Line(points = {{84, 76}, {92, 76}, {92, 46}, {6, 46}, {6, 24}, {-16, 24}, {-16, -14}, {-5, -14}}, color = {0, 0, 127}));
  connect(sin.u, deg2rad.y) annotation(
    Line(points = {{-4, 56}, {0, 56}, {0, 46}, {92, 46}, {92, 76}, {84, 76}}, color = {0, 0, 127}));
  connect(voltageSensor_c.v, gain2.u) annotation(
    Line(points = {{-82, -8}, {-71, -8}, {-71, -9}}, color = {0, 0, 127}));
  connect(realExpression2.y, add4.u2) annotation(
    Line(points = {{-21, -66}, {-18, -66}, {-18, -68}, {-15, -68}}, color = {0, 0, 127}));
  connect(gain4.y, product5.u1) annotation(
    Line(points = {{-57, -30}, {-13, -30}, {-13, -32}, {31, -32}}, color = {0, 0, 127}));
  connect(gain3.y, product4.u1) annotation(
    Line(points = {{-55, -53}, {25, -53}, {25, -56}}, color = {0, 0, 127}));
  connect(sin1.y, product4.u2) annotation(
    Line(points = {{23, -65}, {23, -60.5}, {25, -60.5}, {25, -64}}, color = {0, 0, 127}));
  connect(realExpression3.y, add3.u2) annotation(
    Line(points = {{-11, -92}, {-5, -92}}, color = {0, 0, 127}));
  connect(sin2.y, product5.u2) annotation(
    Line(points = {{17, -43}, {28, -43}, {28, -40}, {31, -40}}, color = {0, 0, 127}));
  connect(product3.y, multiSum1.u[1]) annotation(
    Line(points = {{51, -84}, {54, -84}, {54, -42}}, color = {0, 0, 127}));
  connect(gain5.y, product3.u1) annotation(
    Line(points = {{-55, -79}, {37, -79}, {37, -80}}, color = {0, 0, 127}));
  connect(product5.y, multiSum1.u[3]) annotation(
    Line(points = {{45, -36}, {75.5, -36}, {75.5, -42}, {54, -42}}, color = {0, 0, 127}));
  connect(add3.y, sin3.u) annotation(
    Line(points = {{9, -88}, {26, -88}, {26, -89}, {15, -89}}, color = {0, 0, 127}));
  connect(sin1.u, add4.y) annotation(
    Line(points = {{7, -65}, {16, -65}, {16, -64}, {-1, -64}}, color = {0, 0, 127}));
  connect(product4.y, multiSum1.u[2]) annotation(
    Line(points = {{39, -60}, {54, -60}, {54, -42}}, color = {0, 0, 127}));
  connect(sin3.y, product3.u2) annotation(
    Line(points = {{31, -89}, {31, -88.5}, {37, -88.5}, {37, -88}}, color = {0, 0, 127}));
  connect(multiSum1.y, q) annotation(
    Line(points = {{76, -42}, {93, -42}, {93, -60}, {110, -60}}, color = {0, 0, 127}));
  connect(abc2AlphaBeta.beta, Norm_U_ref_beta.u) annotation(
    Line(points = {{-42, 74}, {-38, 74}, {-38, 68}, {-34, 68}, {-34, 70}}, color = {0, 0, 127}));
  connect(deg2rad.y, sin2.u) annotation(
    Line(points = {{84, 76}, {92, 76}, {92, 46}, {6, 46}, {6, 24}, {-44, 24}, {-44, -44}, {0, -44}, {0, -42}}, color = {0, 0, 127}));
  connect(voltageSensor_a.v, gain4.u) annotation(
    Line(points = {{-80, 50}, {-74, 50}, {-74, -30}, {-72, -30}, {-72, -30}}, color = {0, 0, 127}));
  connect(voltageSensor_b.v, gain3.u) annotation(
    Line(points = {{-81, 22}, {-76, 22}, {-76, -54}, {-72, -54}, {-72, -52}}, color = {0, 0, 127}));
  connect(add4.u1, deg2rad.y) annotation(
    Line(points = {{-16, -60}, {-44, -60}, {-44, 24}, {6, 24}, {6, 46}, {92, 46}, {92, 76}, {84, 76}, {84, 76}}, color = {0, 0, 127}));
  connect(add3.u1, deg2rad.y) annotation(
    Line(points = {{-6, -84}, {-44, -84}, {-44, 24}, {6, 24}, {6, 46}, {92, 46}, {92, 76}, {84, 76}, {84, 76}}, color = {0, 0, 127}));
  connect(gain5.u, voltageSensor_c.v) annotation(
    Line(points = {{-72, -78}, {-80, -78}, {-80, -8}, {-82, -8}, {-82, -8}}, color = {0, 0, 127}));
  connect(voltageSensor_b.v, gain1.u) annotation(
    Line(points = {{-82, 22}, {-76, 22}, {-76, 16}, {-72, 16}, {-72, 18}}, color = {0, 0, 127}));
  connect(voltageSensor_c.v, abc2AlphaBeta.c) annotation(
    Line(points = {{-82, -8}, {-78, -8}, {-78, 76}, {-62, 76}}, color = {0, 0, 127}));
  connect(voltageSensor_b.v, abc2AlphaBeta.b) annotation(
    Line(points = {{-82, 22}, {-76, 22}, {-76, 78}, {-62, 78}, {-62, 80}}, color = {0, 0, 127}));
  connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
    Line(points = {{-62, 82}, {-80, 82}, {-80, 50}, {-80, 50}}, color = {0, 0, 127}));
  connect(deg2rad.y, theta) annotation(
    Line(points = {{84, 76}, {92, 76}, {92, 60}, {110, 60}, {110, 60}}, color = {0, 0, 127}));
  connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
    Line(points = {{34, 58}, {37.5, 58}, {37.5, 74}, {45, 74}}, color = {0, 0, 127}));
  connect(pi.y, add_freq_nom_delta_f.u1) annotation(
    Line(points = {{34, 78}, {44, 78}, {44, 78}, {46, 78}}, color = {0, 0, 127}));
  connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
    Line(points = {{62, 76}, {54, 76}, {54, 76}, {54, 76}}, color = {0, 0, 127}));
end PLL_DQ;
