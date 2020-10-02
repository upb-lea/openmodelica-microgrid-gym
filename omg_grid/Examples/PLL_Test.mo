within OpenModelica_Microgrids.Examples;

model PLL_Test

  OpenModelica_Microgrids.Transformations.ABC2AlphaBeta abc2AlphaBeta annotation(
    Placement(visible = true, transformation(origin = {-14, 4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine(amplitude = 230 * 1.414, freqHz = 50) annotation(
    Placement(visible = true, transformation(origin = {-90, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine1(amplitude = 230 * 1.414, freqHz = 50, phase = -2.0944) annotation(
    Placement(visible = true, transformation(origin = {-88, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Sine sine2(amplitude = 230 * 1.414, freqHz = 50, phase = -4.18879) annotation(
    Placement(visible = true, transformation(origin = {-88, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.Inverters.Inverter inverter annotation(
    Placement(visible = true, transformation(origin = {-14, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  OpenModelica_Microgrids.PLLs.PLL pll annotation(
    Placement(visible = true, transformation(origin = {28, 56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(abc2AlphaBeta.a, sine.y) annotation(
    Line(points = {{-24, 8}, {-44, 8}, {-44, 34}, {-78, 34}, {-78, 34}}, color = {0, 0, 127}));
  connect(abc2AlphaBeta.b, sine1.y) annotation(
    Line(points = {{-24, 6}, {-77, 6}}, color = {0, 0, 127}));
  connect(sine2.y, abc2AlphaBeta.c) annotation(
    Line(points = {{-76, -26}, {-44, -26}, {-44, 2}, {-24, 2}, {-24, 2}}, color = {0, 0, 127}));
  connect(inverter.u3, sine.y) annotation(
    Line(points = {{-24, 64}, {-70, 64}, {-70, 34}, {-78, 34}}, color = {0, 0, 127}));
  connect(inverter.u2, sine1.y) annotation(
    Line(points = {{-24, 58}, {-60, 58}, {-60, 6}, {-76, 6}}, color = {0, 0, 127}));
  connect(inverter.u1, sine2.y) annotation(
    Line(points = {{-24, 52}, {-54, 52}, {-54, -26}, {-76, -26}}, color = {0, 0, 127}));
  connect(pll.a, inverter.pin3) annotation(
    Line(points = {{18, 60}, {4, 60}, {4, 64}, {-4, 64}, {-4, 64}}, color = {0, 0, 255}));
  connect(pll.b, inverter.pin2) annotation(
    Line(points = {{18, 58}, {-4, 58}, {-4, 58}, {-4, 58}}, color = {0, 0, 255}));
  connect(pll.c, inverter.pin1) annotation(
    Line(points = {{18, 54}, {4, 54}, {4, 52}, {-4, 52}, {-4, 52}}, color = {0, 0, 255}));
end PLL_Test;
