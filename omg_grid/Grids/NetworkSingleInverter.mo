within omg_grid.Grids;

model NetworkSingleInverter
  omg_grid.Inverters.Inverter inverter1 annotation(
    Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.Filter.IdealFilter.LC lc1(C1 = 0.00002, C2 = 0.00002, C3 = 0.00002, L1 = 0.002, L2 = 0.002, L3 = 0.002)  annotation(
    Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput i1p1 annotation(
    Placement(visible = true, transformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput i1p2 annotation(
    Placement(visible = true, transformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealInput i1p3 annotation(
    Placement(visible = true, transformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
omg_grid.Loads.RL rl1 annotation(
    Placement(visible = true, transformation(origin = {24, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
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
connect(lc1.pin6, rl1.pin3) annotation(
    Line(points = {{-20, 36}, {14, 36}}, color = {0, 0, 255}));
connect(lc1.pin5, rl1.pin2) annotation(
    Line(points = {{-20, 30}, {14, 30}}, color = {0, 0, 255}));
connect(lc1.pin4, rl1.pin1) annotation(
    Line(points = {{-20, 24}, {14, 24}}, color = {0, 0, 255}));
  annotation(
    Diagram);
end NetworkSingleInverter;
