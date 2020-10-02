within OpenModelica_Microgrids.Examples;

model ControlledNetworkSC
  OpenModelica_Microgrids.Inverters.Inverter inverter1(v_DC = 5000)  annotation(
    Placement(visible = true, transformation(origin = {-82, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Filter.IdealFilter.LC lc1(L1 = 0.001, L2 = 0.001, L3 = 0.001) annotation(
    Placement(visible = true, transformation(origin = {-26, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.PLLs.PLL_DQ pll_dq annotation(
    Placement(visible = true, transformation(origin = {2, 62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Transformations.ABC2DQ_Currents abc2dq_current annotation(
    Placement(visible = true, transformation(origin = {-54, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = -350) annotation(
    Placement(visible = true, transformation(origin = {2, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression1(y = 230 * 1.41427) annotation(
    Placement(visible = true, transformation(origin = {5, 77}, extent = {{-13, -11}, {13, 11}}, rotation = 0)));
  OpenModelica_Microgrids.Transformations.DQ2ABC dq2abc annotation(
    Placement(visible = true, transformation(origin = {36, -18}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID pid(Td = 0, Ti = 0.006, k = 0.3, limitsAtInit = true, yMax = 150) annotation(
    Placement(visible = true, transformation(origin = {78, 86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID(Td = 0, Ti = 0.06, k = 0.3, limitsAtInit = true, yMax = 50) annotation(
    Placement(visible = true, transformation(origin = {44, 74}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID1(Td = 0, Ti = 1.33, k = 0.013, limitsAtInit = true, yMax = 1 / 2.8284, yMin = -1 / 2.8284) annotation(
    Placement(visible = true, transformation(origin = {-34, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID2(Td = 0, Ti = 1.33, k = 0.013, limitsAtInit = true, yMax = 1 / 2.8284, yMin = -1 / 2.8284) annotation(
    Placement(visible = true, transformation(origin = {-34, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Components.PhaseAngle angle annotation(
    Placement(visible = true, transformation(origin = {19, 5}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
  Modelica.Electrical.Analog.Basic.Ground ground annotation(
    Placement(visible = true, transformation(origin = {18, 24}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(dq2abc.a, inverter1.u3) annotation(
    Line(points = {{46, -12}, {78, -12}, {78, -66}, {-116, -66}, {-116, 36}, {-92, 36}}, color = {0, 0, 127}));
  connect(dq2abc.b, inverter1.u2) annotation(
    Line(points = {{46, -18}, {72, -18}, {72, -60}, {-110, -60}, {-110, 30}, {-92, 30}}, color = {0, 0, 127}));
  connect(inverter1.u1, dq2abc.c) annotation(
    Line(points = {{-92, 24}, {-104, 24}, {-104, -54}, {66, -54}, {66, -24}, {46, -24}}, color = {0, 0, 127}));
  connect(pll_dq.d, PID.u_m) annotation(
    Line(points = {{14, 62}, {32, 62}, {32, 60}, {40, 60}, {40, 58}, {44, 58}, {44, 62}, {44, 62}}, color = {0, 0, 127}));
  connect(realExpression1.y, PID.u_s) annotation(
    Line(points = {{20, 78}, {30, 78}, {30, 74}, {32, 74}}, color = {0, 0, 127}));
  connect(pll_dq.q, pid.u_m) annotation(
    Line(points = {{14, 56}, {78, 56}, {78, 72}, {78, 72}, {78, 74}}, color = {0, 0, 127}));
  connect(realExpression.y, pid.u_s) annotation(
    Line(points = {{14, 90}, {58, 90}, {58, 86}, {66, 86}, {66, 86}}, color = {0, 0, 127}));
  connect(abc2dq_current.d, PID2.u_m) annotation(
    Line(points = {{-58, 19}, {-58, -20}, {-34, -20}, {-34, -16}}, color = {0, 0, 127}));
  connect(abc2dq_current.q, PID1.u_m) annotation(
    Line(points = {{-50, 19}, {-50, -13.5}, {-34, -13.5}, {-34, -46}}, color = {0, 0, 127}));
  connect(pll_dq.theta, abc2dq_current.theta) annotation(
    Line(points = {{14, 68}, {22, 68}, {22, 48}, {-54, 48}, {-54, 41}}, color = {0, 0, 127}));
  connect(pid.y, PID1.u_s) annotation(
    Line(points = {{90, 86}, {94, 86}, {94, 14}, {-60, 14}, {-60, -34}, {-46, -34}}, color = {0, 0, 127}));
  connect(PID2.u_s, PID.y) annotation(
    Line(points = {{-46, -4}, {-56, -4}, {-56, 12}, {60, 12}, {60, 74}, {56, 74}, {56, 74}}, color = {0, 0, 127}));
  connect(PID2.y, dq2abc.d) annotation(
    Line(points = {{-22, -4}, {16, -4}, {16, -14}, {26, -14}, {26, -14}}, color = {0, 0, 127}));
  connect(PID1.y, dq2abc.q) annotation(
    Line(points = {{-22, -34}, {24, -34}, {24, -22}, {24, -22}, {24, -22}, {26, -22}}, color = {0, 0, 127}));
  connect(angle.theta, dq2abc.theta) annotation(
    Line(points = {{24, 6}, {32, 6}, {32, -6}, {32, -6}}, color = {0, 0, 127}));
  connect(lc1.pin4, pll_dq.c) annotation(
    Line(points = {{-16, 24}, {-12, 24}, {-12, 56}, {-8, 56}}, color = {0, 0, 255}));
  connect(lc1.pin5, pll_dq.b) annotation(
    Line(points = {{-16, 30}, {-14, 30}, {-14, 62}, {-8, 62}}, color = {0, 0, 255}));
  connect(lc1.pin6, pll_dq.a) annotation(
    Line(points = {{-16, 36}, {-16, 68}, {-8, 68}}, color = {0, 0, 255}));
  connect(inverter1.pin3, abc2dq_current.a) annotation(
    Line(points = {{-72, 36}, {-66, 36}, {-66, 36}, {-64, 36}}, color = {0, 0, 255}));
  connect(inverter1.pin2, abc2dq_current.b) annotation(
    Line(points = {{-72, 30}, {-64, 30}, {-64, 30}, {-64, 30}}, color = {0, 0, 255}));
  connect(inverter1.pin1, abc2dq_current.c) annotation(
    Line(points = {{-72, 24}, {-64, 24}, {-64, 24}, {-64, 24}}, color = {0, 0, 255}));
  connect(abc2dq_current.pin3, lc1.pin3) annotation(
    Line(points = {{-44, 36}, {-36, 36}, {-36, 36}, {-36, 36}}, color = {0, 0, 255}));
  connect(abc2dq_current.pin2, lc1.pin2) annotation(
    Line(points = {{-44, 30}, {-36, 30}, {-36, 30}, {-36, 30}}, color = {0, 0, 255}));
  connect(abc2dq_current.pin1, lc1.pin1) annotation(
    Line(points = {{-44, 24}, {-36, 24}, {-36, 24}, {-36, 24}}, color = {0, 0, 255}));
  connect(lc1.pin4, ground.p) annotation(
    Line(points = {{-16, 24}, {-2, 24}, {-2, 34}, {18, 34}, {18, 34}}, color = {0, 0, 255}));
  connect(lc1.pin5, ground.p) annotation(
    Line(points = {{-16, 30}, {-2, 30}, {-2, 34}, {18, 34}, {18, 34}}, color = {0, 0, 255}));
  connect(lc1.pin6, ground.p) annotation(
    Line(points = {{-16, 36}, {18, 36}, {18, 34}, {18, 34}}, color = {0, 0, 255}));
  annotation(
    Diagram);
end ControlledNetworkSC;
