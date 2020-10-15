within omg_grid.PLLs;

model PLL
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
  omg_grid.Transformations.ABC2AlphaBeta abc2AlphaBeta annotation(
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
end PLL;
