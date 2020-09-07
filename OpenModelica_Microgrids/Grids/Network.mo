within OpenModelica_Microgrids.Grids;

model Network
  OpenModelica_Microgrids.Inverters.Inverter inverter1 annotation(
    Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Filter.IdealFilter.LC lc1 annotation(
    Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Inverters.Inverter inverter2 annotation(
    Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Filter.IdealFilter.LC lc2 annotation(
    Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
  OpenModelica_Microgrids.Filter.IdealFilter.LCL lcl1 annotation(
    Placement(visible = true, transformation(origin = {-32, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
OpenModelica_Microgrids.Loads.RL rl1 annotation(
    Placement(visible = true, transformation(origin = {70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(lc1.pin6, lc2.pin3) annotation(
    Line(points = {{-20, 36}, {20, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
  connect(lc1.pin5, lc2.pin2) annotation(
    Line(points = {{-20, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
  connect(lc1.pin4, lc2.pin1) annotation(
    Line(points = {{-20, 24}, {20, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
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
    Line(points = {{-60, -24}, {-42, -24}}, color = {0, 0, 255}));
  connect(inverter2.pin2, lcl1.pin2) annotation(
    Line(points = {{-60, -30}, {-42, -30}}, color = {0, 0, 255}));
  connect(inverter2.pin1, lcl1.pin1) annotation(
    Line(points = {{-60, -36}, {-42, -36}}, color = {0, 0, 255}));
  connect(lcl1.pin6, lc2.pin3) annotation(
    Line(points = {{-22, -24}, {-6, -24}, {-6, 36}, {20, 36}}, color = {0, 0, 255}));
  connect(lcl1.pin5, lc2.pin2) annotation(
    Line(points = {{-22, -30}, {0, -30}, {0, 30}, {20, 30}}, color = {0, 0, 255}));
  connect(lcl1.pin4, lc2.pin1) annotation(
    Line(points = {{-22, -36}, {6, -36}, {6, 24}, {20, 24}}, color = {0, 0, 255}));
connect(lc2.pin6, rl1.pin3) annotation(
    Line(points = {{40, 36}, {60, 36}, {60, 36}, {60, 36}}, color = {0, 0, 255}));
connect(lc2.pin5, rl1.pin2) annotation(
    Line(points = {{40, 30}, {60, 30}, {60, 30}, {60, 30}}, color = {0, 0, 255}));
connect(lc2.pin4, rl1.pin1) annotation(
    Line(points = {{40, 24}, {60, 24}, {60, 24}, {60, 24}}, color = {0, 0, 255}));
  annotation(
    Diagram);
end Network;
