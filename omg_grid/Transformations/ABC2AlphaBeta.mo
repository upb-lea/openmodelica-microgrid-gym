within omg_grid.Transformations;

model ABC2AlphaBeta
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
end ABC2AlphaBeta;
