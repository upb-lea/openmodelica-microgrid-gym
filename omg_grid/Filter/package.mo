within omg_grid;
package Filter 
extends Modelica.Icons.Package;

annotation (Icon(
      coordinateSystem(preserveAspectRatio=true,
        extent={{-100.0,-100.0},{100.0,100.0}}),
        graphics={
      Line(points={{-80.0,80.0},{-80.0,-88.0}},
        color={192,192,192}),
      Polygon(lineColor={192,192,192},
        fillColor={192,192,192},
        fillPattern=FillPattern.Solid,
        points={{-80.0,92.0},{-88.0,70.0},{-72.0,70.0},{-80.0,92.0}}),
      Line(points={{-90.0,-78.0},{82.0,-78.0}},
        color={192,192,192}),
      Polygon(lineColor={192,192,192},
        fillColor={192,192,192},
        fillPattern=FillPattern.Solid,
        points={{90.0,-78.0},{68.0,-70.0},{68.0,-86.0},{90.0,-78.0}}),
      Text(lineColor={192,192,192},
        extent={{-66.0,52.0},{88.0,90.0}},
        textString="%order"),
      Text(
        extent={{-138.0,-140.0},{162.0,-110.0}},
        textString="f_cut=%f_cut"),
      Rectangle(lineColor={160,160,164},
        fillColor={255,255,255},
        fillPattern=FillPattern.Backward,
        extent={{-80.0,-78.0},{22.0,10.0}}),
      Line(origin = {3.333,-6.667}, points = {{-83.333,34.667},{24.667,34.667},{42.667,-71.333}}, color = {0,0,127}, smooth = Smooth.Bezier)}));
end Filter;
