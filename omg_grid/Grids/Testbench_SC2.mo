within OpenModelica_Microgrids.Grids;

model Testbench_SC2
  OpenModelica_Microgrids.Inverters.Inverter inverter1(v_DC = 60)  annotation(
    Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput i1p1 annotation(
    Placement(visible = true, transformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput i1p2 annotation(
    Placement(visible = true, transformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput i1p3 annotation(
    Placement(visible = true, transformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
OpenModelica_Microgrids.Filter.LossesFilter.L rl(L1 = 0.0023, L2 = 0.0023, L3 = 0.0023, R1 = 0.170, R2 = 0.170, R3 = 0.170)  annotation(
    Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(i1p1, inverter1.u1) annotation(
    Line(points = {{-104, 18}, {-86, 18}, {-86, 24}, {-80, 24}, {-80, 24}}, color = {0, 0, 127}));
  connect(i1p2, inverter1.u2) annotation(
    Line(points = {{-104, 30}, {-80, 30}, {-80, 30}, {-80, 30}}, color = {0, 0, 127}));
  connect(i1p3, inverter1.u3) annotation(
    Line(points = {{-104, 42}, {-86, 42}, {-86, 36}, {-80, 36}}, color = {0, 0, 127}));
connect(inverter1.pin3, rl.pin3) annotation(
    Line(points = {{-60, 36}, {-40, 36}, {-40, 36}, {-40, 36}}, color = {0, 0, 255}));
connect(inverter1.pin2, rl.pin2) annotation(
    Line(points = {{-60, 30}, {-40, 30}, {-40, 30}, {-40, 30}}, color = {0, 0, 255}));
connect(inverter1.pin1, rl.pin1) annotation(
    Line(points = {{-60, 24}, {-40, 24}, {-40, 24}, {-40, 24}}, color = {0, 0, 255}));
connect(rl.pin6, rl.pin4) annotation(
    Line(points = {{-20, 36}, {-14, 36}, {-14, 24}, {-20, 24}, {-20, 24}}, color = {0, 0, 255}));
connect(rl.pin5, rl.pin6) annotation(
    Line(points = {{-20, 30}, {-14, 30}, {-14, 36}, {-20, 36}, {-20, 36}}, color = {0, 0, 255}));
connect(rl.pin4, rl.pin5) annotation(
    Line(points = {{-20, 24}, {-14, 24}, {-14, 30}, {-20, 30}, {-20, 30}}, color = {0, 0, 255}));
  annotation(
    Diagram);
end Testbench_SC2;
