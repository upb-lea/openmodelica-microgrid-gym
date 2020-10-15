within omg_grid.Transformations;

model DQ2ABC
  Real pi = 2 * Modelica.Math.asin(1.0);
  Modelica.Blocks.Interfaces.RealInput d annotation(
    Placement(visible = true, transformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput q annotation(
    Placement(visible = true, transformation(origin = {-104, -40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, -40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput b annotation(
    Placement(visible = true, transformation(origin = {106, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression1(y = 2 * pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-7, 6}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {14, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos1 annotation(
    Placement(visible = true, transformation(origin = {34, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos2 annotation(
    Placement(visible = true, transformation(origin = {26, 68}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product annotation(
    Placement(visible = true, transformation(origin = {56, 14}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product1 annotation(
    Placement(visible = true, transformation(origin = {44, 48}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product2 annotation(
    Placement(visible = true, transformation(origin = {50, 72}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product3 annotation(
    Placement(visible = true, transformation(origin = {58, -58}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Product product4 annotation(
    Placement(visible = true, transformation(origin = {46, -34}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {16, -62}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {6, -38}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression2(y = 4 * pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-17, -40}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput c annotation(
    Placement(visible = true, transformation(origin = {106, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-5, -66}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin annotation(
    Placement(visible = true, transformation(origin = {37, -63}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin5 annotation(
    Placement(visible = true, transformation(origin = {17, 43}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Add add annotation(
    Placement(visible = true, transformation(origin = {74, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput a annotation(
    Placement(visible = true, transformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Product product6 annotation(
    Placement(visible = true, transformation(origin = {54, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add4(k2 = -1) annotation(
    Placement(visible = true, transformation(origin = {14, -12}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * pi / 3) annotation(
    Placement(visible = true, transformation(origin = {-11, -16}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
  Modelica.Blocks.Math.Sin sin8 annotation(
    Placement(visible = true, transformation(origin = {35, -13}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
  Modelica.Blocks.Math.Add add5 annotation(
    Placement(visible = true, transformation(origin = {80, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Cos cos annotation(
    Placement(visible = true, transformation(origin = {24, -38}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
  Modelica.Blocks.Math.Add add6 annotation(
    Placement(visible = true, transformation(origin = {82, -52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput theta annotation(
    Placement(visible = true, transformation(origin = {-34, 116}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {-34, 116}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
equation
  connect(realExpression1.y, add1.u2) annotation(
    Line(points = {{1, 6}, {7, 6}}, color = {0, 0, 127}));
  connect(add1.y, cos1.u) annotation(
    Line(points = {{21, 10}, {27, 10}}, color = {0, 0, 127}));
  connect(cos2.y, product2.u2) annotation(
    Line(points = {{32, 68}, {42, 68}, {42, 68}, {42, 68}}, color = {0, 0, 127}));
  connect(cos1.y, product.u2) annotation(
    Line(points = {{41, 10}, {49, 10}}, color = {0, 0, 127}));
  connect(realExpression2.y, add3.u2) annotation(
    Line(points = {{-9, -40}, {-4, -40}, {-4, -42}, {-1, -42}}, color = {0, 0, 127}));
  connect(realExpression3.y, add2.u2) annotation(
    Line(points = {{3, -66}, {9, -66}}, color = {0, 0, 127}));
  connect(add2.y, sin.u) annotation(
    Line(points = {{22, -62}, {26, -62}, {26, -63}, {29, -63}}, color = {0, 0, 127}));
  connect(sin.y, product3.u2) annotation(
    Line(points = {{45, -63}, {45, -62}, {50, -62}}, color = {0, 0, 127}));
  connect(d, product2.u1) annotation(
    Line(points = {{-104, 40}, {-60, 40}, {-60, 76}, {42, 76}, {42, 76}}, color = {0, 0, 127}));
  connect(sin5.y, product1.u2) annotation(
    Line(points = {{24, 44}, {36, 44}, {36, 44}, {36, 44}}, color = {0, 0, 127}));
  connect(q, product1.u1) annotation(
    Line(points = {{-104, -40}, {-50, -40}, {-50, 52}, {36, 52}, {36, 52}}, color = {0, 0, 127}));
  connect(product1.y, add.u2) annotation(
    Line(points = {{50, 48}, {56, 48}, {56, 52}, {62, 52}, {62, 52}}, color = {0, 0, 127}));
  connect(add.y, a) annotation(
    Line(points = {{86, 58}, {92, 58}, {92, 60}, {106, 60}}, color = {0, 0, 127}));
  connect(product2.y, add.u1) annotation(
    Line(points = {{56, 72}, {58, 72}, {58, 64}, {62, 64}, {62, 64}, {62, 64}}, color = {0, 0, 127}));
  connect(sin8.y, product6.u2) annotation(
    Line(points = {{43, -13}, {43, -12.5}, {47, -12.5}, {47, -12}}, color = {0, 0, 127}));
  connect(realExpression.y, add4.u2) annotation(
    Line(points = {{-3, -16}, {7, -16}}, color = {0, 0, 127}));
  connect(sin8.u, add4.y) annotation(
    Line(points = {{27, -13}, {26, -13}, {26, -12}, {21, -12}}, color = {0, 0, 127}));
  connect(d, product.u1) annotation(
    Line(points = {{-104, 40}, {-60, 40}, {-60, 18}, {49, 18}}, color = {0, 0, 127}));
  connect(q, product6.u1) annotation(
    Line(points = {{-104, -40}, {-50, -40}, {-50, -2}, {47, -2}, {47, -4}}, color = {0, 0, 127}));
  connect(product6.y, add5.u2) annotation(
    Line(points = {{61, -8}, {64, -8}, {64, -4}, {68, -4}}, color = {0, 0, 127}));
  connect(product.y, add5.u1) annotation(
    Line(points = {{63, 14}, {64, 14}, {64, 8}, {68, 8}}, color = {0, 0, 127}));
  connect(add5.y, b) annotation(
    Line(points = {{92, 2}, {94, 2}, {94, 0}, {106, 0}, {106, 0}}, color = {0, 0, 127}));
  connect(cos.y, product4.u2) annotation(
    Line(points = {{30, -38}, {38, -38}, {38, -38}, {38, -38}}, color = {0, 0, 127}));
  connect(d, product4.u1) annotation(
    Line(points = {{-104, 40}, {-60, 40}, {-60, -30}, {38, -30}, {38, -30}}, color = {0, 0, 127}));
  connect(add3.y, cos.u) annotation(
    Line(points = {{12, -38}, {16, -38}, {16, -38}, {16, -38}}, color = {0, 0, 127}));
  connect(q, product3.u1) annotation(
    Line(points = {{-104, -40}, {-50, -40}, {-50, -48}, {48, -48}, {48, -54}, {50, -54}, {50, -54}}, color = {0, 0, 127}));
  connect(product3.y, add6.u2) annotation(
    Line(points = {{64, -58}, {68, -58}, {68, -58}, {70, -58}}, color = {0, 0, 127}));
  connect(product4.y, add6.u1) annotation(
    Line(points = {{52, -34}, {58, -34}, {58, -46}, {68, -46}, {68, -46}, {70, -46}}, color = {0, 0, 127}));
  connect(add6.y, c) annotation(
    Line(points = {{94, -52}, {94, -52}, {94, -60}, {106, -60}, {106, -60}}, color = {0, 0, 127}));
  connect(theta, add4.u1) annotation(
    Line(points = {{-34, 116}, {-34, -8}, {7, -8}}, color = {0, 0, 127}));
  connect(theta, sin5.u) annotation(
    Line(points = {{-34, 116}, {-34, 116}, {-34, 42}, {8, 42}, {8, 44}}, color = {0, 0, 127}));
  connect(theta, add3.u1) annotation(
    Line(points = {{-34, 116}, {-34, -32}, {-2.25, -32}, {-2.25, -34}, {-1, -34}}, color = {0, 0, 127}));
  connect(theta, add2.u1) annotation(
    Line(points = {{-34, 116}, {-34, 116}, {-34, -58}, {8, -58}, {8, -58}}, color = {0, 0, 127}));
  connect(theta, cos2.u) annotation(
    Line(points = {{-34, 116}, {-34, 68}, {18, 68}}, color = {0, 0, 127}));
  connect(theta, add1.u1) annotation(
    Line(points = {{-34, 116}, {-34, 14}, {7, 14}}, color = {0, 0, 127}));
end DQ2ABC;
