within OpenModelica_Microgrids.Examples;

model NetworkSineTest

  Modelica.Blocks.Sources.Sine sine(amplitude = 230, freqHz = 50) annotation(
    Placement(visible = true, transformation(origin = {-76, -82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine1(amplitude = 230, freqHz = 50, phase = 2.0944) annotation(
    Placement(visible = true, transformation(origin = {-76, -48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine2(amplitude = 230, freqHz = 50, phase = 4.18879) annotation(
    Placement(visible = true, transformation(origin = {-76, -16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine3(amplitude = 230, freqHz = 50) annotation(
    Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine4(amplitude = 230, freqHz = 50, phase = 2.0944) annotation(
    Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine5(amplitude = 230, freqHz = 50, phase = 4.18879) annotation(
    Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
OpenModelica_Microgrids.Grids.Network network1 annotation(
    Placement(visible = true, transformation(origin = {38, 2}, extent = {{-46, -46}, {46, 46}}, rotation = 0)));
equation
  connect(sine5.y, network1.i1p3) annotation(
    Line(points = {{-64, 80}, {-34, 80}, {-34, 22}, {-10, 22}, {-10, 22}}, color = {0, 0, 127}));
connect(sine4.y, network1.i1p2) annotation(
    Line(points = {{-64, 48}, {-40, 48}, {-40, 16}, {-10, 16}, {-10, 16}}, color = {0, 0, 127}));
connect(sine3.y, network1.i1p1) annotation(
    Line(points = {{-64, 14}, {-34, 14}, {-34, 10}, {-10, 10}, {-10, 10}}, color = {0, 0, 127}));
connect(sine2.y, network1.i2p3) annotation(
    Line(points = {{-64, -16}, {-64, -16}, {-64, -6}, {-10, -6}, {-10, -6}}, color = {0, 0, 127}));
connect(sine1.y, network1.i2p2) annotation(
    Line(points = {{-64, -48}, {-50, -48}, {-50, -12}, {-10, -12}, {-10, -12}, {-10, -12}}, color = {0, 0, 127}));
connect(sine.y, network1.i2p1) annotation(
    Line(points = {{-64, -82}, {-32, -82}, {-32, -18}, {-10, -18}, {-10, -18}}, color = {0, 0, 127}));
end NetworkSineTest;
