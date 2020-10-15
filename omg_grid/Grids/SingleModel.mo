within omg_grid.Grids;

model SingleModel
  omg_grid.Inverters.Inverter inverter1 annotation(
    Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
omg_grid.Loads.R r(R1 = 100, R2 = 100, R3 = 100)  annotation(
    Placement(visible = true, transformation(origin = {8, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
omg_grid.Filter.IdealFilter.L l annotation(
    Placement(visible = true, transformation(origin = {-40, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
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
  connect(inverter1.pin3, r.pin3) annotation(
    Line(points = {{-60, 36}, {-2, 36}, {-2, 36}, {-2, 36}}, color = {0, 0, 255}));
  connect(inverter1.pin2, r.pin2) annotation(
    Line(points = {{-60, 30}, {-2, 30}, {-2, 30}, {-2, 30}}, color = {0, 0, 255}));
  connect(inverter1.pin1, r.pin1) annotation(
    Line(points = {{-60, 24}, {-2, 24}, {-2, 24}, {-2, 24}}, color = {0, 0, 255}));
connect(inverter2.pin3, l.pin3) annotation(
    Line(points = {{-60, -24}, {-50, -24}}, color = {0, 0, 255}));
connect(inverter2.pin2, l.pin2) annotation(
    Line(points = {{-60, -30}, {-50, -30}}, color = {0, 0, 255}));
connect(inverter2.pin1, l.pin1) annotation(
    Line(points = {{-60, -36}, {-50, -36}}, color = {0, 0, 255}));
connect(l.pin6, r.pin3) annotation(
    Line(points = {{-30, -24}, {-20, -24}, {-20, 36}, {-2, 36}, {-2, 36}, {-2, 36}}, color = {0, 0, 255}));
connect(l.pin5, r.pin2) annotation(
    Line(points = {{-30, -30}, {-14, -30}, {-14, 30}, {-2, 30}, {-2, 30}}, color = {0, 0, 255}));
connect(l.pin4, r.pin1) annotation(
    Line(points = {{-30, -36}, {-8, -36}, {-8, 24}, {-2, 24}, {-2, 24}, {-2, 24}}, color = {0, 0, 255}));
  annotation(
    Diagram);
end SingleModel;
