within omg_grid.ActiveLoads;

model ActiveLoad
  parameter SI.Power p_ref(start = 5000);
  Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
    Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
    Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
    Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.PLLs.PLL_DQ pll_dq annotation(
    Placement(visible = true, transformation(origin = {-50, 76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.Transformations.ABC2DQ_Currents abc2dq_current annotation(
    Placement(visible = true, transformation(origin = {-58, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.Inverters.Inverter inverter annotation(
    Placement(visible = true, transformation(origin = {-4, -12}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Modelica.Blocks.Continuous.LimPID PID2(Td = 0, Ti = 1.33, k = 0.013, limitsAtInit = true, yMax = 1 / 2.8284, yMin = -1 / 2.8284) annotation(
    Placement(visible = true, transformation(origin = {34, -48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.Transformations.DQ2ABC dq2abc annotation(
    Placement(visible = true, transformation(origin = {68, -66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID1(Td = 0, Ti = 1.33, k = 0.013, limitsAtInit = true, yMax = 1 / 2.8284, yMin = -1 / 2.8284) annotation(
    Placement(visible = true, transformation(origin = {34, -78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID pid(Td = 0, Ti = 0.006, k = 0.023, limitsAtInit = true, yMax = 30) annotation(
    Placement(visible = true, transformation(origin = {-38, -48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID(Td = 0, Ti = 0.006, k = 0.023, limitsAtInit = true, yMax = 30) annotation(
    Placement(visible = true, transformation(origin = {-46, -80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = 1000 / 325) annotation(
    Placement(visible = true, transformation(origin = {-84, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression2(y = 1000 / 325) annotation(
    Placement(visible = true, transformation(origin = {-82, -72}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
omg_grid.Filter.IdealFilter.LC lc(L1 = 0.004, L2 = 0.004, L3 = 0.004)  annotation(
    Placement(visible = true, transformation(origin = {-24, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
Modelica.Blocks.Sources.RealExpression realExpression1(y = 1) annotation(
    Placement(visible = true, transformation(origin = {-4, -58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
Modelica.Blocks.Sources.RealExpression realExpression3(y = 0) annotation(
    Placement(visible = true, transformation(origin = {-6, -72}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
omg_grid.Components.StartValues startvalues(startTime = 0.1)  annotation(
    Placement(visible = true, transformation(origin = {38, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
omg_grid.Components.StartValues startvalues1(startTime = 0.1)  annotation(
    Placement(visible = true, transformation(origin = {38, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
omg_grid.Components.StartValues startvalues2(startTime = 0.1)  annotation(
    Placement(visible = true, transformation(origin = {38, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
equation
  connect(pin1, pll_dq.c) annotation(
    Line(points = {{-100, -60}, {-80, -60}, {-80, 70}, {-60, 70}}, color = {0, 0, 255}));
  connect(pin2, pll_dq.b) annotation(
    Line(points = {{-100, 0}, {-84, 0}, {-84, 76}, {-60, 76}}, color = {0, 0, 255}));
  connect(pin3, pll_dq.a) annotation(
    Line(points = {{-100, 60}, {-88, 60}, {-88, 82}, {-60, 82}}, color = {0, 0, 255}));
  connect(pin1, abc2dq_current.c) annotation(
    Line(points = {{-100, -60}, {-80, -60}, {-80, -6}, {-68, -6}, {-68, -6}}, color = {0, 0, 255}));
  connect(pin2, abc2dq_current.b) annotation(
    Line(points = {{-100, 0}, {-68, 0}, {-68, 0}, {-68, 0}}, color = {0, 0, 255}));
  connect(pin3, abc2dq_current.a) annotation(
    Line(points = {{-100, 60}, {-88, 60}, {-88, 6}, {-68, 6}, {-68, 6}}, color = {0, 0, 255}));
  connect(pll_d.theta, abc2dq_current.theta) annotation(
    Line(points = {{-38, 82}, {-30, 82}, {-30, 24}, {-58, 24}, {-58, 12}, {-58, 12}}, color = {0, 0, 127}));
  connect(PID2.y, dq2abc.d) annotation(
    Line(points = {{46, -48}, {52, -48}, {52, -62}, {58, -62}}, color = {0, 0, 127}));
  connect(PID1.y, dq2abc.q) annotation(
    Line(points = {{46, -78}, {52, -78}, {52, -70}, {58, -70}}, color = {0, 0, 127}));
  connect(abc2dq_current.d, pid.u_m) annotation(
    Line(points = {{-62, -10}, {-62, -10}, {-62, -64}, {-38, -64}, {-38, -60}, {-38, -60}}, color = {0, 0, 127}));
  connect(abc2dq_current.q, PID.u_m) annotation(
    Line(points = {{-54, -10}, {-54, -10}, {-54, -16}, {-64, -16}, {-64, -96}, {-46, -96}, {-46, -92}, {-46, -92}}, color = {0, 0, 127}));
connect(realExpression2.y, pid.u_s) annotation(
    Line(points = {{-70, -72}, {-68, -72}, {-68, -48}, {-50, -48}, {-50, -48}}, color = {0, 0, 127}));
connect(realExpression.y, PID.u_s) annotation(
    Line(points = {{-72, -86}, {-70, -86}, {-70, -80}, {-58, -80}, {-58, -80}}, color = {0, 0, 127}));
  connect(pll_dq.d, PID2.u_m) annotation(
    Line(points = {{-38, 76}, {16, 76}, {16, -64}, {34, -64}, {34, -60}, {34, -60}}, color = {0, 0, 127}));
  connect(pll_dq.q, PID1.u_m) annotation(
    Line(points = {{-38, 70}, {14, 70}, {14, -94}, {34, -94}, {34, -90}, {34, -90}}, color = {0, 0, 127}));
  connect(pll_dq.theta, dq2abc.theta) annotation(
    Line(points = {{-38, 82}, {64, 82}, {64, -54}, {64, -54}}, color = {0, 0, 127}));
connect(abc2dq_current.pin1, lc.pin4) annotation(
    Line(points = {{-48, -6}, {-44, -6}, {-44, 14}, {-34, 14}}, color = {0, 0, 255}));
connect(abc2dq_current.pin2, lc.pin5) annotation(
    Line(points = {{-48, 0}, {-42, 0}, {-42, 8}, {-34, 8}}, color = {0, 0, 255}));
connect(abc2dq_current.pin3, lc.pin6) annotation(
    Line(points = {{-48, 6}, {-38, 6}, {-38, 2}, {-34, 2}, {-34, 2}}, color = {0, 0, 255}));
connect(lc.pin3, inverter.pin3) annotation(
    Line(points = {{-14, 2}, {-10, 2}, {-10, -2}, {-10, -2}}, color = {0, 0, 255}));
connect(lc.pin2, inverter.pin2) annotation(
    Line(points = {{-14, 8}, {-4, 8}, {-4, -2}, {-4, -2}}, color = {0, 0, 255}));
connect(lc.pin1, inverter.pin1) annotation(
    Line(points = {{-14, 14}, {2, 14}, {2, -2}, {2, -2}}, color = {0, 0, 255}));
connect(realExpression1.y, PID2.u_s) annotation(
    Line(points = {{8, -58}, {8, -58}, {8, -48}, {22, -48}, {22, -48}}, color = {0, 0, 127}));
connect(realExpression3.y, PID1.u_s) annotation(
    Line(points = {{6, -72}, {16, -72}, {16, -78}, {20, -78}, {20, -78}, {22, -78}}, color = {0, 0, 127}));
connect(dq2abc.c, startvalues2.u) annotation(
    Line(points = {{78, -72}, {92, -72}, {92, 66}, {50, 66}, {50, 66}}, color = {0, 0, 127}));
connect(inverter.u1, startvalues2.y) annotation(
    Line(points = {{2, -22}, {2, -22}, {2, -30}, {18, -30}, {18, 66}, {28, 66}, {28, 66}}, color = {0, 0, 127}));
connect(dq2abc.b, startvalues1.u) annotation(
    Line(points = {{78, -66}, {86, -66}, {86, 34}, {52, 34}, {52, 34}, {50, 34}}, color = {0, 0, 127}));
connect(dq2abc.a, startvalues.u) annotation(
    Line(points = {{78, -60}, {80, -60}, {80, 0}, {50, 0}, {50, 0}}, color = {0, 0, 127}));
connect(startvalues1.y, inverter.u2) annotation(
    Line(points = {{28, 34}, {20, 34}, {20, -32}, {-4, -32}, {-4, -22}, {-4, -22}}, color = {0, 0, 127}));
connect(inverter.u3, startvalues.y) annotation(
    Line(points = {{-10, -22}, {-10, -22}, {-10, -34}, {22, -34}, {22, 0}, {28, 0}, {28, 0}}, color = {0, 0, 127}));
end ActiveLoad;
