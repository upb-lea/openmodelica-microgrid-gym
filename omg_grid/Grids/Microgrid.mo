within omg_grid.Grids;

model Microgrid
  omg_grid.Inverters.Inverter inverter1 annotation(
    Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.Filter.IdealFilter.LC lc1 annotation(
    Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  omg_grid.Inverters.Inverter inverter2 annotation(
    Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
  omg_grid.Filter.IdealFilter.LCL lcl1 annotation(
    Placement(visible = true, transformation(origin = {-32, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
omg_grid.Loads.RL rl1 annotation(
    Placement(visible = true, transformation(origin = {92, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
omg_grid.Filter.IdealFilter.L l12 annotation(
    Placement(visible = true, transformation(origin = {2, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
omg_grid.Filter.IdealFilter.L l13 annotation(
    Placement(visible = true, transformation(origin = {46, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
omg_grid.Filter.IdealFilter.L l23 annotation(
    Placement(visible = true, transformation(origin = {48, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
  connect(i2p3, inverter2.u3) annotation(
    Line(points = {{-104, -18}, {-88, -18}, {-88, -24}, {-80, -24}, {-80, -24}}, color = {0, 0, 127}));
  connect(i2p2, inverter2.u2) annotation(
    Line(points = {{-104, -30}, {-80, -30}, {-80, -30}, {-80, -30}}, color = {0, 0, 127}));
  connect(i2p1, inverter2.u1) annotation(
    Line(points = {{-104, -42}, {-90, -42}, {-90, -36}, {-80, -36}, {-80, -36}}, color = {0, 0, 127}));
  connect(inverter2.pin3, lcl1.pin3) annotation(
    Line(points = {{-60, -24}, {-42, -24}, {-42, -24}, {-42, -24}}, color = {0, 0, 255}));
  connect(inverter2.pin2, lcl1.pin2) annotation(
    Line(points = {{-60, -30}, {-42, -30}, {-42, -30}, {-42, -30}}, color = {0, 0, 255}));
  connect(inverter2.pin1, lcl1.pin1) annotation(
    Line(points = {{-60, -36}, {-42, -36}, {-42, -36}, {-42, -36}}, color = {0, 0, 255}));
connect(lc1.pin6, l12.pin3) annotation(
    Line(points = {{-20, 36}, {8, 36}, {8, 10}}, color = {0, 0, 255}));
connect(lc1.pin5, l12.pin2) annotation(
    Line(points = {{-20, 30}, {2, 30}, {2, 10}}, color = {0, 0, 255}));
connect(lc1.pin4, l12.pin1) annotation(
    Line(points = {{-20, 24}, {-4, 24}, {-4, 10}}, color = {0, 0, 255}));
connect(l12.pin6, lcl1.pin6) annotation(
    Line(points = {{8, -10}, {8, -24}, {-22, -24}}, color = {0, 0, 255}));
connect(l12.pin5, lcl1.pin5) annotation(
    Line(points = {{2, -10}, {2, -30}, {-22, -30}}, color = {0, 0, 255}));
connect(l12.pin4, lcl1.pin4) annotation(
    Line(points = {{-4, -10}, {-4, -36}, {-22, -36}}, color = {0, 0, 255}));
connect(l13.pin3, lc1.pin6) annotation(
    Line(points = {{36, 36}, {-20, 36}}, color = {0, 0, 255}));
connect(l13.pin2, lc1.pin5) annotation(
    Line(points = {{36, 30}, {-20, 30}, {-20, 30}, {-20, 30}, {-20, 30}}, color = {0, 0, 255}));
connect(l13.pin1, lc1.pin4) annotation(
    Line(points = {{36, 24}, {-20, 24}, {-20, 24}, {-20, 24}}, color = {0, 0, 255}));
connect(l23.pin3, lcl1.pin6) annotation(
    Line(points = {{38, -24}, {-22, -24}, {-22, -24}, {-22, -24}}, color = {0, 0, 255}));
connect(l23.pin2, lcl1.pin5) annotation(
    Line(points = {{38, -30}, {-22, -30}, {-22, -30}, {-22, -30}}, color = {0, 0, 255}));
connect(l23.pin1, lcl1.pin4) annotation(
    Line(points = {{38, -36}, {-22, -36}, {-22, -36}, {-22, -36}}, color = {0, 0, 255}));
connect(l13.pin6, rl1.pin3) annotation(
    Line(points = {{56, 36}, {72, 36}, {72, 8}, {82, 8}, {82, 8}}, color = {0, 0, 255}));
connect(l13.pin5, rl1.pin2) annotation(
    Line(points = {{56, 30}, {66, 30}, {66, 2}, {82, 2}, {82, 2}}, color = {0, 0, 255}));
connect(l13.pin4, rl1.pin1) annotation(
    Line(points = {{56, 24}, {62, 24}, {62, -4}, {82, -4}, {82, -4}}, color = {0, 0, 255}));
connect(l23.pin5, rl1.pin2) annotation(
    Line(points = {{58, -30}, {66, -30}, {66, 2}, {82, 2}, {82, 2}}, color = {0, 0, 255}));
connect(l23.pin6, rl1.pin3) annotation(
    Line(points = {{58, -24}, {72, -24}, {72, 8}, {82, 8}, {82, 8}}, color = {0, 0, 255}));
connect(l23.pin4, rl1.pin1) annotation(
    Line(points = {{58, -36}, {62, -36}, {62, -4}, {82, -4}, {82, -4}}, color = {0, 0, 255}));
  annotation(
    Diagram);
end Microgrid;
