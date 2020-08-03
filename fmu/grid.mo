package grid
  package active_loads
    model r_active
      parameter SI.Power p_ref(start = 5000);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load1(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load2(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_d pll_d annotation(
        Placement(visible = true, transformation(origin = {-50, 76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {-50, 56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {-12, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, active_load1.p) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, -70}, {-60, -70}, {-60, -70}}, color = {0, 0, 255}));
      connect(pin2, active_load2.p) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, -32}, {-60, -32}, {-60, -30}, {-60, -30}}, color = {0, 0, 255}));
      connect(pin3, active_load.p) annotation(
        Line(points = {{-100, 60}, {-72, 60}, {-72, 30}, {-60, 30}, {-60, 30}, {-60, 30}}, color = {0, 0, 255}));
      connect(pin1, pll_d.c) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, 70}, {-60, 70}}, color = {0, 0, 255}));
      connect(pin2, pll_d.b) annotation(
        Line(points = {{-100, 0}, {-84, 0}, {-84, 76}, {-60, 76}}, color = {0, 0, 255}));
      connect(pin3, pll_d.a) annotation(
        Line(points = {{-100, 60}, {-88, 60}, {-88, 82}, {-60, 82}}, color = {0, 0, 255}));
      connect(realExpression.y, division.u2) annotation(
        Line(points = {{-38, 56}, {-28, 56}, {-28, 60}, {-24, 60}, {-24, 60}}, color = {0, 0, 127}));
      connect(pll_d.d, division.u1) annotation(
        Line(points = {{-39, 76}, {-31.5, 76}, {-31.5, 72}, {-24, 72}}, color = {0, 0, 127}));
      connect(division.y, active_load.u) annotation(
        Line(points = {{0, 66}, {10, 66}, {10, 46}, {-50, 46}, {-50, 42}, {-50, 42}}, color = {0, 0, 127}));
      connect(division.y, active_load2.u) annotation(
        Line(points = {{0, 66}, {10, 66}, {10, -8}, {-50, -8}, {-50, -18}, {-50, -18}}, color = {0, 0, 127}));
      connect(division.y, active_load1.u) annotation(
        Line(points = {{0, 66}, {10, 66}, {10, -52}, {-50, -52}, {-50, -58}, {-50, -58}}, color = {0, 0, 127}));
      connect(active_load1.n, ground.p) annotation(
        Line(points = {{-40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_load2.n, ground.p) annotation(
        Line(points = {{-40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_load.n, ground.p) annotation(
        Line(points = {{-40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
    end r_active;

    model rl_active
      parameter SI.Power p_ref(start = 5000);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load1(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load2(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {-50, 56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {-12, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_serial pll_serial annotation(
        Placement(visible = true, transformation(origin = {-50, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l annotation(
        Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l1 annotation(
        Placement(visible = true, transformation(origin = {30, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l2 annotation(
        Placement(visible = true, transformation(origin = {30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {30, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_serial pll_serial1 annotation(
        Placement(visible = true, transformation(origin = {30, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division1 annotation(
        Placement(visible = true, transformation(origin = {70, 68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, active_load1.p) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, -70}, {-60, -70}, {-60, -70}}, color = {0, 0, 255}));
      connect(pin2, active_load2.p) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, -32}, {-60, -32}, {-60, -30}, {-60, -30}}, color = {0, 0, 255}));
      connect(pin3, active_load.p) annotation(
        Line(points = {{-100, 60}, {-72, 60}, {-72, 30}, {-60, 30}, {-60, 30}, {-60, 30}}, color = {0, 0, 255}));
      connect(division.y, active_load.u) annotation(
        Line(points = {{0, 66}, {10, 66}, {10, 46}, {-50, 46}, {-50, 42}, {-50, 42}}, color = {0, 0, 127}));
      connect(division.y, active_load2.u) annotation(
        Line(points = {{0, 66}, {10, 66}, {10, -8}, {-50, -8}, {-50, -18}, {-50, -18}}, color = {0, 0, 127}));
      connect(division.y, active_load1.u) annotation(
        Line(points = {{0, 66}, {10, 66}, {10, -52}, {-50, -52}, {-50, -58}, {-50, -58}}, color = {0, 0, 127}));
      connect(pll_serial.d, division.u1) annotation(
        Line(points = {{-54, 71}, {-54, 66}, {-34, 66}, {-34, 72}, {-24, 72}}, color = {0, 0, 127}));
      connect(active_load.p, pll_serial.a) annotation(
        Line(points = {{-60, 30}, {-72, 30}, {-72, 88}, {-60, 88}, {-60, 88}}, color = {0, 0, 255}));
      connect(active_load2.p, pll_serial.b) annotation(
        Line(points = {{-60, -30}, {-70, -30}, {-70, 82}, {-60, 82}, {-60, 82}}, color = {0, 0, 255}));
      connect(active_load1.p, pll_serial.c) annotation(
        Line(points = {{-60, -70}, {-68, -70}, {-68, 76}, {-60, 76}, {-60, 76}}, color = {0, 0, 255}));
      connect(pll_serial.pin3, active_load.n) annotation(
        Line(points = {{-40, 88}, {-28, 88}, {-28, 30}, {-40, 30}, {-40, 30}, {-40, 30}}, color = {0, 0, 255}));
      connect(pll_serial.pin2, active_load2.n) annotation(
        Line(points = {{-40, 82}, {-30, 82}, {-30, -30}, {-40, -30}, {-40, -30}}, color = {0, 0, 255}));
      connect(realExpression.y, division.u2) annotation(
        Line(points = {{-38, 56}, {-36, 56}, {-36, 60}, {-24, 60}, {-24, 60}}, color = {0, 0, 127}));
      connect(pll_serial.pin1, active_load1.n) annotation(
        Line(points = {{-40, 76}, {-32, 76}, {-32, -70}, {-40, -70}, {-40, -70}}, color = {0, 0, 255}));
      connect(active_load.n, active_l.p) annotation(
        Line(points = {{-40, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
      connect(active_load2.n, active_l2.p) annotation(
        Line(points = {{-40, -30}, {-40, -30}, {-40, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(active_load1.n, active_l1.p) annotation(
        Line(points = {{-40, -70}, {20, -70}, {20, -70}, {20, -70}}, color = {0, 0, 255}));
      connect(active_l.n, ground.p) annotation(
        Line(points = {{40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l2.n, ground.p) annotation(
        Line(points = {{40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l1.n, ground.p) annotation(
        Line(points = {{40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l.p, pll_serial1.a) annotation(
        Line(points = {{20, 30}, {12, 30}, {12, 90}, {20, 90}, {20, 90}}, color = {0, 0, 255}));
      connect(active_l.n, pll_serial1.pin3) annotation(
        Line(points = {{40, 30}, {52, 30}, {52, 90}, {40, 90}, {40, 90}}, color = {0, 0, 255}));
      connect(pll_serial1.b, active_l2.p) annotation(
        Line(points = {{20, 84}, {14, 84}, {14, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin2, active_l2.n) annotation(
        Line(points = {{40, 84}, {50, 84}, {50, -30}, {40, -30}, {40, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin1, active_l1.n) annotation(
        Line(points = {{40, 78}, {48, 78}, {48, -70}, {40, -70}, {40, -70}}, color = {0, 0, 255}));
      connect(active_l1.p, pll_serial1.c) annotation(
        Line(points = {{20, -70}, {16, -70}, {16, 78}, {20, 78}, {20, 78}}, color = {0, 0, 255}));
      connect(pll_serial1.d, division1.u1) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 68}, {54, 68}, {54, 74}, {58, 74}, {58, 74}}, color = {0, 0, 127}));
      connect(realExpression1.y, division1.u2) annotation(
        Line(points = {{42, 58}, {42, 58}, {42, 62}, {58, 62}, {58, 62}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, 46}, {34, 46}, {34, 42}, {34, 42}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l2.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, -14}, {34, -14}, {34, -18}, {34, -18}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l1.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {48, 70}, {48, -54}, {34, -54}, {34, -58}, {34, -58}}, color = {0, 0, 127}));
      connect(division1.y, active_l.u) annotation(
        Line(points = {{82, 68}, {88, 68}, {88, 48}, {26, 48}, {26, 42}, {26, 42}}, color = {0, 0, 127}));
      connect(division1.y, active_l2.u) annotation(
        Line(points = {{82, 68}, {88, 68}, {88, -12}, {26, -12}, {26, -18}, {26, -18}}, color = {0, 0, 127}));
      connect(division1.y, active_l1.u) annotation(
        Line(points = {{82, 68}, {88, 68}, {88, -52}, {26, -52}, {26, -58}, {26, -58}}, color = {0, 0, 127}));
    protected
    end rl_active;

    model l_active
      parameter SI.Power p_ref(start = 5000);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l annotation(
        Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l1 annotation(
        Placement(visible = true, transformation(origin = {30, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l2 annotation(
        Placement(visible = true, transformation(origin = {30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {30, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_serial pll_serial1 annotation(
        Placement(visible = true, transformation(origin = {30, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division1 annotation(
        Placement(visible = true, transformation(origin = {70, 68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(active_l.n, ground.p) annotation(
        Line(points = {{40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l2.n, ground.p) annotation(
        Line(points = {{40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l1.n, ground.p) annotation(
        Line(points = {{40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l.p, pll_serial1.a) annotation(
        Line(points = {{20, 30}, {12, 30}, {12, 90}, {20, 90}, {20, 90}}, color = {0, 0, 255}));
      connect(active_l.n, pll_serial1.pin3) annotation(
        Line(points = {{40, 30}, {52, 30}, {52, 90}, {40, 90}, {40, 90}}, color = {0, 0, 255}));
      connect(pll_serial1.b, active_l2.p) annotation(
        Line(points = {{20, 84}, {14, 84}, {14, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin2, active_l2.n) annotation(
        Line(points = {{40, 84}, {50, 84}, {50, -30}, {40, -30}, {40, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin1, active_l1.n) annotation(
        Line(points = {{40, 78}, {48, 78}, {48, -70}, {40, -70}, {40, -70}}, color = {0, 0, 255}));
      connect(active_l1.p, pll_serial1.c) annotation(
        Line(points = {{20, -70}, {16, -70}, {16, 78}, {20, 78}, {20, 78}}, color = {0, 0, 255}));
      connect(pll_serial1.d, division1.u1) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 68}, {54, 68}, {54, 74}, {58, 74}, {58, 74}}, color = {0, 0, 127}));
      connect(realExpression1.y, division1.u2) annotation(
        Line(points = {{42, 58}, {42, 58}, {42, 62}, {58, 62}, {58, 62}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, 46}, {34, 46}, {34, 42}, {34, 42}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l2.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, -14}, {34, -14}, {34, -18}, {34, -18}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l1.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {48, 70}, {48, -54}, {34, -54}, {34, -58}, {34, -58}}, color = {0, 0, 127}));
      connect(division1.y, active_l.u) annotation(
        Line(points = {{82, 68}, {88, 68}, {88, 48}, {26, 48}, {26, 42}, {26, 42}}, color = {0, 0, 127}));
      connect(division1.y, active_l2.u) annotation(
        Line(points = {{82, 68}, {88, 68}, {88, -12}, {26, -12}, {26, -18}, {26, -18}}, color = {0, 0, 127}));
      connect(division1.y, active_l1.u) annotation(
        Line(points = {{82, 68}, {88, 68}, {88, -52}, {26, -52}, {26, -58}, {26, -58}}, color = {0, 0, 127}));
      connect(pin1, active_l1.p) annotation(
        Line(points = {{-100, -60}, {0, -60}, {0, -70}, {20, -70}, {20, -70}}, color = {0, 0, 255}));
      connect(pin2, active_l2.p) annotation(
        Line(points = {{-100, 0}, {0, 0}, {0, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pin3, active_l.p) annotation(
        Line(points = {{-100, 60}, {0, 60}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    protected
    end l_active;

    model r_active_alpha
      parameter SI.Power p_ref(start = 5000);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load1(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load2(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha pll_alpha annotation(
        Placement(visible = true, transformation(origin = {-50, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, active_load1.p) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, -70}, {-60, -70}, {-60, -70}}, color = {0, 0, 255}));
      connect(pin2, active_load2.p) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, -32}, {-60, -32}, {-60, -30}, {-60, -30}}, color = {0, 0, 255}));
      connect(pin3, active_load.p) annotation(
        Line(points = {{-100, 60}, {-72, 60}, {-72, 30}, {-60, 30}, {-60, 30}, {-60, 30}}, color = {0, 0, 255}));
      connect(active_load1.n, ground.p) annotation(
        Line(points = {{-40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_load2.n, ground.p) annotation(
        Line(points = {{-40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_load.n, ground.p) annotation(
        Line(points = {{-40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(pin1, pll_alpha.c) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, 64}, {-60, 64}, {-60, 64}}, color = {0, 0, 255}));
      connect(pin2, pll_alpha.b) annotation(
        Line(points = {{-100, 0}, {-84, 0}, {-84, 70}, {-60, 70}, {-60, 70}}, color = {0, 0, 255}));
      connect(pin3, pll_alpha.a) annotation(
        Line(points = {{-100, 60}, {-88, 60}, {-88, 76}, {-60, 76}, {-60, 76}}, color = {0, 0, 255}));
      connect(pll_alpha.u_eff, active_load1.u) annotation(
        Line(points = {{-38, 70}, {-22, 70}, {-22, -56}, {-50, -56}, {-50, -58}, {-50, -58}}, color = {0, 0, 127}));
      connect(pll_alpha.u_eff, active_load.u) annotation(
        Line(points = {{-38, 70}, {-22, 70}, {-22, 46}, {-50, 46}, {-50, 42}, {-50, 42}, {-50, 42}}, color = {0, 0, 127}));
      connect(pll_alpha.u_eff, active_load2.u) annotation(
        Line(points = {{-38, 70}, {-22, 70}, {-22, -8}, {-50, -8}, {-50, -18}, {-50, -18}, {-50, -18}}, color = {0, 0, 127}));
    end r_active_alpha;

    model rl_active_alpha
      parameter SI.Power p_ref(start = 5000);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load1(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load2(p_ref = p_ref, r_min = 1) annotation(
        Placement(visible = true, transformation(origin = {-50, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l annotation(
        Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l1 annotation(
        Placement(visible = true, transformation(origin = {30, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l2 annotation(
        Placement(visible = true, transformation(origin = {30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_serial pll_alpha_serial annotation(
        Placement(visible = true, transformation(origin = {-50, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_serial pll_alpha_serial1 annotation(
        Placement(visible = true, transformation(origin = {30, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, active_load1.p) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, -70}, {-60, -70}, {-60, -70}}, color = {0, 0, 255}));
      connect(pin2, active_load2.p) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, -32}, {-60, -32}, {-60, -30}, {-60, -30}}, color = {0, 0, 255}));
      connect(pin3, active_load.p) annotation(
        Line(points = {{-100, 60}, {-72, 60}, {-72, 30}, {-60, 30}, {-60, 30}, {-60, 30}}, color = {0, 0, 255}));
      connect(active_load.n, active_l.p) annotation(
        Line(points = {{-40, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
      connect(active_load2.n, active_l2.p) annotation(
        Line(points = {{-40, -30}, {-40, -30}, {-40, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(active_load1.n, active_l1.p) annotation(
        Line(points = {{-40, -70}, {20, -70}, {20, -70}, {20, -70}}, color = {0, 0, 255}));
      connect(active_l.n, ground.p) annotation(
        Line(points = {{40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l2.n, ground.p) annotation(
        Line(points = {{40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l1.n, ground.p) annotation(
        Line(points = {{40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_load.p, pll_alpha_serial.a) annotation(
        Line(points = {{-60, 30}, {-70, 30}, {-70, 76}, {-60, 76}, {-60, 76}}, color = {0, 0, 255}));
      connect(active_load2.p, pll_alpha_serial.b) annotation(
        Line(points = {{-60, -30}, {-68, -30}, {-68, 70}, {-60, 70}, {-60, 70}}, color = {0, 0, 255}));
      connect(active_load1.p, pll_alpha_serial.c) annotation(
        Line(points = {{-60, -70}, {-66, -70}, {-66, 64}, {-60, 64}, {-60, 64}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin3, active_load.n) annotation(
        Line(points = {{-40, 76}, {-30, 76}, {-30, 30}, {-40, 30}, {-40, 30}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin2, active_load2.n) annotation(
        Line(points = {{-40, 70}, {-32, 70}, {-32, -30}, {-40, -30}, {-40, -30}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin1, active_load1.n) annotation(
        Line(points = {{-40, 64}, {-34, 64}, {-34, -70}, {-40, -70}, {-40, -70}}, color = {0, 0, 255}));
      connect(pll_alpha_serial1.freq, active_l.f) annotation(
        Line(points = {{34, 60}, {34, 60}, {34, 42}, {34, 42}}, color = {0, 0, 127}));
      connect(pll_alpha_serial.u_eff, active_load.u) annotation(
        Line(points = {{-54, 60}, {-54, 60}, {-54, 48}, {-50, 48}, {-50, 42}, {-50, 42}}, color = {0, 0, 127}));
      connect(pll_alpha_serial.u_eff, active_load2.u) annotation(
        Line(points = {{-54, 60}, {-54, 60}, {-54, 48}, {-62, 48}, {-62, -16}, {-50, -16}, {-50, -18}, {-50, -18}}, color = {0, 0, 127}));
      connect(pll_alpha_serial.u_eff, active_load1.u) annotation(
        Line(points = {{-54, 60}, {-54, 60}, {-54, 48}, {-62, 48}, {-62, -56}, {-50, -56}, {-50, -58}, {-50, -58}}, color = {0, 0, 127}));
      connect(active_l.p, pll_alpha_serial1.a) annotation(
        Line(points = {{20, 30}, {10, 30}, {10, 76}, {20, 76}, {20, 76}}, color = {0, 0, 255}));
      connect(pll_alpha_serial1.pin3, active_l.n) annotation(
        Line(points = {{40, 76}, {48, 76}, {48, 76}, {50, 76}, {50, 30}, {40, 30}, {40, 30}}, color = {0, 0, 255}));
      connect(pll_alpha_serial1.pin2, active_l2.n) annotation(
        Line(points = {{40, 70}, {48, 70}, {48, -30}, {40, -30}, {40, -30}, {40, -30}}, color = {0, 0, 255}));
      connect(pll_alpha_serial1.b, active_l2.p) annotation(
        Line(points = {{20, 70}, {12, 70}, {12, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(active_l1.p, pll_alpha_serial1.c) annotation(
        Line(points = {{20, -70}, {14, -70}, {14, 64}, {20, 64}, {20, 64}}, color = {0, 0, 255}));
      connect(active_l.u, pll_alpha_serial1.u_eff) annotation(
        Line(points = {{26, 42}, {26, 42}, {26, 60}, {26, 60}}, color = {0, 0, 127}));
      connect(pll_alpha_serial1.pin1, active_l1.n) annotation(
        Line(points = {{40, 64}, {46, 64}, {46, -70}, {40, -70}, {40, -70}, {40, -70}}, color = {0, 0, 255}));
      connect(pll_alpha_serial1.u_eff, active_l2.u) annotation(
        Line(points = {{26, 60}, {26, 60}, {26, 48}, {16, 48}, {16, -14}, {26, -14}, {26, -18}, {26, -18}}, color = {0, 0, 127}));
      connect(pll_alpha_serial1.u_eff, active_l1.u) annotation(
        Line(points = {{26, 60}, {26, 60}, {26, 48}, {16, 48}, {16, -54}, {26, -54}, {26, -58}, {26, -58}}, color = {0, 0, 127}));
      connect(pll_alpha_serial1.freq, active_l2.f) annotation(
        Line(points = {{34, 60}, {34, 60}, {34, 48}, {44, 48}, {44, -14}, {34, -14}, {34, -18}, {34, -18}}, color = {0, 0, 127}));
      connect(pll_alpha_serial1.freq, active_l1.f) annotation(
        Line(points = {{34, 60}, {34, 60}, {34, 48}, {44, 48}, {44, -54}, {34, -54}, {34, -58}, {34, -58}}, color = {0, 0, 127}));
    protected
    end rl_active_alpha;

    model l_active_alpha
      parameter SI.Power q_ref(start = 500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l(q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l1(q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {30, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l2(q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_serial pll_serial1 annotation(
        Placement(visible = true, transformation(origin = {30, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load annotation(
        Placement(visible = true, transformation(origin = {-50, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load1 annotation(
        Placement(visible = true, transformation(origin = {-50, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_load active_load2 annotation(
        Placement(visible = true, transformation(origin = {-50, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 100) annotation(
        Placement(visible = true, transformation(origin = {-78, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(active_l.n, ground.p) annotation(
        Line(points = {{40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l2.n, ground.p) annotation(
        Line(points = {{40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l1.n, ground.p) annotation(
        Line(points = {{40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l.p, pll_serial1.a) annotation(
        Line(points = {{20, 30}, {12, 30}, {12, 90}, {20, 90}, {20, 90}}, color = {0, 0, 255}));
      connect(active_l.n, pll_serial1.pin3) annotation(
        Line(points = {{40, 30}, {52, 30}, {52, 90}, {40, 90}, {40, 90}}, color = {0, 0, 255}));
      connect(pll_serial1.b, active_l2.p) annotation(
        Line(points = {{20, 84}, {14, 84}, {14, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin2, active_l2.n) annotation(
        Line(points = {{40, 84}, {50, 84}, {50, -30}, {40, -30}, {40, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin1, active_l1.n) annotation(
        Line(points = {{40, 78}, {48, 78}, {48, -70}, {40, -70}, {40, -70}}, color = {0, 0, 255}));
      connect(active_l1.p, pll_serial1.c) annotation(
        Line(points = {{20, -70}, {16, -70}, {16, 78}, {20, 78}, {20, 78}}, color = {0, 0, 255}));
      connect(pll_serial1.freq, active_l.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, 46}, {34, 46}, {34, 42}, {34, 42}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l2.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, -14}, {34, -14}, {34, -18}, {34, -18}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l1.f) annotation(
        Line(points = {{34, 74}, {34, 70}, {46, 70}, {46, -54}, {34, -54}, {34, -58}}, color = {0, 0, 127}));
      connect(pll_serial1.u_eff, active_l.u) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 42}, {26, 42}}, color = {0, 0, 127}));
      connect(pll_serial1.u_eff, active_l2.u) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 50}, {18, 50}, {18, -14}, {26, -14}, {26, -18}, {26, -18}}, color = {0, 0, 127}));
      connect(pll_serial1.u_eff, active_l1.u) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 50}, {18, 50}, {18, -54}, {26, -54}, {26, -58}, {26, -58}, {26, -58}}, color = {0, 0, 127}));
      connect(pin2, active_load2.p) annotation(
        Line(points = {{-100, 0}, {-60, 0}, {-60, 0}, {-60, 0}}, color = {0, 0, 255}));
      connect(active_load2.n, active_l2.p) annotation(
        Line(points = {{-40, 0}, {0, 0}, {0, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pin1, active_load1.p) annotation(
        Line(points = {{-100, -60}, {-60, -60}, {-60, -60}, {-60, -60}}, color = {0, 0, 255}));
      connect(active_load1.n, active_l1.p) annotation(
        Line(points = {{-40, -60}, {0, -60}, {0, -70}, {20, -70}, {20, -70}}, color = {0, 0, 255}));
      connect(pin3, active_load.p) annotation(
        Line(points = {{-100, 60}, {-60, 60}, {-60, 60}, {-60, 60}}, color = {0, 0, 255}));
      connect(active_load.n, active_l.p) annotation(
        Line(points = {{-40, 60}, {0, 60}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
      connect(realExpression.y, active_load.u) annotation(
        Line(points = {{-67, 80}, {-57.5, 80}, {-57.5, 72}, {-50, 72}}, color = {0, 0, 127}));
      connect(realExpression.y, active_load2.u) annotation(
        Line(points = {{-67, 80}, {-64, 80}, {-64, 14}, {-50, 14}, {-50, 12}}, color = {0, 0, 127}));
      connect(realExpression.y, active_load1.u) annotation(
        Line(points = {{-67, 80}, {-64, 80}, {-64, -46}, {-50, -46}, {-50, -48}}, color = {0, 0, 127}));
    protected
    end l_active_alpha;

    model l_active_alphaR
      parameter SI.Power q_ref(start = 500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {60, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l(q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l1(q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {30, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.active_l active_l2(q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_serial pll_serial1 annotation(
        Placement(visible = true, transformation(origin = {30, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(active_l.n, ground.p) annotation(
        Line(points = {{40, 30}, {60, 30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l2.n, ground.p) annotation(
        Line(points = {{40, -30}, {60, -30}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l1.n, ground.p) annotation(
        Line(points = {{40, -70}, {60, -70}, {60, -74}, {60, -74}}, color = {0, 0, 255}));
      connect(active_l.p, pll_serial1.a) annotation(
        Line(points = {{20, 30}, {12, 30}, {12, 90}, {20, 90}, {20, 90}}, color = {0, 0, 255}));
      connect(active_l.n, pll_serial1.pin3) annotation(
        Line(points = {{40, 30}, {52, 30}, {52, 90}, {40, 90}, {40, 90}}, color = {0, 0, 255}));
      connect(pll_serial1.b, active_l2.p) annotation(
        Line(points = {{20, 84}, {14, 84}, {14, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin2, active_l2.n) annotation(
        Line(points = {{40, 84}, {50, 84}, {50, -30}, {40, -30}, {40, -30}}, color = {0, 0, 255}));
      connect(pll_serial1.pin1, active_l1.n) annotation(
        Line(points = {{40, 78}, {48, 78}, {48, -70}, {40, -70}, {40, -70}}, color = {0, 0, 255}));
      connect(active_l1.p, pll_serial1.c) annotation(
        Line(points = {{20, -70}, {16, -70}, {16, 78}, {20, 78}, {20, 78}}, color = {0, 0, 255}));
      connect(pll_serial1.freq, active_l.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, 46}, {34, 46}, {34, 42}, {34, 42}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l2.f) annotation(
        Line(points = {{34, 74}, {34, 74}, {34, 70}, {46, 70}, {46, -14}, {34, -14}, {34, -18}, {34, -18}}, color = {0, 0, 127}));
      connect(pll_serial1.freq, active_l1.f) annotation(
        Line(points = {{34, 74}, {34, 70}, {46, 70}, {46, -54}, {34, -54}, {34, -58}}, color = {0, 0, 127}));
      connect(pll_serial1.u_eff, active_l.u) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 42}, {26, 42}}, color = {0, 0, 127}));
      connect(pll_serial1.u_eff, active_l2.u) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 50}, {18, 50}, {18, -14}, {26, -14}, {26, -18}, {26, -18}}, color = {0, 0, 127}));
      connect(pll_serial1.u_eff, active_l1.u) annotation(
        Line(points = {{26, 74}, {26, 74}, {26, 50}, {18, 50}, {18, -54}, {26, -54}, {26, -58}, {26, -58}, {26, -58}}, color = {0, 0, 127}));
      connect(pin3, active_l.p) annotation(
        Line(points = {{-100, 60}, {0, 60}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
      connect(pin2, active_l2.p) annotation(
        Line(points = {{-100, 0}, {0, 0}, {0, -30}, {20, -30}, {20, -30}}, color = {0, 0, 255}));
      connect(pin1, active_l1.p) annotation(
        Line(points = {{-100, -60}, {0, -60}, {0, -70}, {20, -70}, {20, -70}}, color = {0, 0, 255}));
    protected
    end l_active_alphaR;

    model l_ctrl
      parameter SI.Inductance L_start(start = 0.005);
      parameter SI.Power q_ref(start = 500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_l ctrl_l(L_start = L_start) annotation(
        Placement(visible = true, transformation(origin = {-6, 12}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {86, -76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_l ctrl_l1(L_start = L_start) annotation(
        Placement(visible = true, transformation(origin = {-6, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_l ctrl_l2(L_start = L_start) annotation(
        Placement(visible = true, transformation(origin = {-6, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_f_serial_test pll_alpha_serial annotation(
        Placement(visible = true, transformation(origin = {-6, 86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.controller_l controller_l(L_start = L_start, q_ref = 500) annotation(
        Placement(visible = true, transformation(origin = {-6, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 100) annotation(
        Placement(visible = true, transformation(origin = {-74, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(ctrl_l.n, ground.p) annotation(
        Line(points = {{4, 12}, {86, 12}, {86, -66}}, color = {0, 0, 255}));
      connect(ctrl_l2.n, ground.p) annotation(
        Line(points = {{4, -26}, {86, -26}, {86, -66}}, color = {0, 0, 255}));
      connect(ctrl_l1.n, ground.p) annotation(
        Line(points = {{4, -60}, {86, -60}, {86, -66}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.freq, controller_l.f) annotation(
        Line(points = {{-2, 75}, {-2, 68}, {-24, 68}, {-24, 48}, {-17, 48}}, color = {0, 0, 127}));
      connect(controller_l.y, ctrl_l.L_Ctrl) annotation(
        Line(points = {{4, 48}, {12, 48}, {12, 28}, {-6, 28}, {-6, 23}}, color = {0, 0, 127}));
      connect(controller_l.y, ctrl_l2.L_Ctrl) annotation(
        Line(points = {{4, 48}, {12, 48}, {12, -8}, {-6, -8}, {-6, -15}}, color = {0, 0, 127}));
      connect(controller_l.y, ctrl_l1.L_Ctrl) annotation(
        Line(points = {{4, 48}, {12, 48}, {12, -44}, {-6, -44}, {-6, -49}}, color = {0, 0, 127}));
      connect(ctrl_l1.p, pll_alpha_serial.c) annotation(
        Line(points = {{-16, -60}, {-26, -60}, {-26, 80}, {-16, 80}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.b, ctrl_l2.p) annotation(
        Line(points = {{-16, 86}, {-28, 86}, {-28, -26}, {-16, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.a, ctrl_l.p) annotation(
        Line(points = {{-16, 92}, {-30, 92}, {-30, 12}, {-16, 12}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin1, ctrl_l1.n) annotation(
        Line(points = {{4, 80}, {16, 80}, {16, -60}, {4, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin2, ctrl_l2.n) annotation(
        Line(points = {{4, 86}, {18, 86}, {18, -26}, {4, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin3, ctrl_l.n) annotation(
        Line(points = {{4, 92}, {20, 92}, {20, 12}, {4, 12}}, color = {0, 0, 255}));
      connect(pin3, ctrl_l.p) annotation(
        Line(points = {{-100, 60}, {-46, 60}, {-46, 12}, {-16, 12}, {-16, 12}}, color = {0, 0, 255}));
      connect(pin2, ctrl_l2.p) annotation(
        Line(points = {{-100, 0}, {-54, 0}, {-54, -26}, {-16, -26}, {-16, -26}, {-16, -26}}, color = {0, 0, 255}));
      connect(pin1, ctrl_l1.p) annotation(
        Line(points = {{-100, -60}, {-16, -60}, {-16, -60}, {-16, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.u_eff, controller_l.u) annotation(
        Line(points = {{-10, 76}, {-10, 76}, {-10, 64}, {-18, 64}, {-18, 54}, {-16, 54}}, color = {0, 0, 127}));
    protected
    end l_ctrl;

    model rl_ctrl
      parameter SI.Inductance L_start(start = 0.005);
      parameter SI.Power q_ref(start = 500);
      parameter SI.Power p_ref(start = 2500);
      parameter SI.Resistance R_start(start = 20);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_l ctrl_l3(L_start = L_start) annotation(
        Placement(visible = true, transformation(origin = {-6, 12}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {86, -76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_l ctrl_l1(L_start = L_start) annotation(
        Placement(visible = true, transformation(origin = {-6, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_l ctrl_l2(L_start = L_start) annotation(
        Placement(visible = true, transformation(origin = {-6, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_f_serial_test pll_alpha_f_serial annotation(
        Placement(visible = true, transformation(origin = {-6, 86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.controller_l controller_l(L_start = L_start, q_ref = q_ref) annotation(
        Placement(visible = true, transformation(origin = {-6, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_serial pll_alpha_serial annotation(
        Placement(visible = true, transformation(origin = {-60, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.controller_r controller_r(R_start = R_start, p_ref = p_ref) annotation(
        Placement(visible = true, transformation(origin = {-58, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_r ctrl_r annotation(
        Placement(visible = true, transformation(origin = {-58, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_r ctrl_r1 annotation(
        Placement(visible = true, transformation(origin = {-58, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_r ctrl_r2 annotation(
        Placement(visible = true, transformation(origin = {-58, 12}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(ctrl_l3.n, ground.p) annotation(
        Line(points = {{4, 12}, {86, 12}, {86, -66}}, color = {0, 0, 255}));
      connect(ctrl_l2.n, ground.p) annotation(
        Line(points = {{4, -26}, {86, -26}, {86, -66}}, color = {0, 0, 255}));
      connect(ctrl_l1.n, ground.p) annotation(
        Line(points = {{4, -60}, {86, -60}, {86, -66}}, color = {0, 0, 255}));
      connect(pll_alpha_f_serial.u_eff, controller_l.u) annotation(
        Line(points = {{-10, 75}, {-10, 62}, {-20, 62}, {-20, 54}, {-17, 54}}, color = {0, 0, 127}));
      connect(pll_alpha_f_serial.freq, controller_l.f) annotation(
        Line(points = {{-2, 75}, {-2, 68}, {-24, 68}, {-24, 48}, {-17, 48}}, color = {0, 0, 127}));
      connect(controller_l.y, ctrl_l3.L_Ctrl) annotation(
        Line(points = {{4, 48}, {12, 48}, {12, 28}, {-6, 28}, {-6, 23}}, color = {0, 0, 127}));
      connect(controller_l.y, ctrl_l2.L_Ctrl) annotation(
        Line(points = {{4, 48}, {12, 48}, {12, -8}, {-6, -8}, {-6, -15}}, color = {0, 0, 127}));
      connect(controller_l.y, ctrl_l1.L_Ctrl) annotation(
        Line(points = {{4, 48}, {12, 48}, {12, -44}, {-6, -44}, {-6, -49}}, color = {0, 0, 127}));
      connect(ctrl_l1.p, pll_alpha_f_serial.c) annotation(
        Line(points = {{-16, -60}, {-26, -60}, {-26, 80}, {-16, 80}}, color = {0, 0, 255}));
      connect(pll_alpha_f_serial.b, ctrl_l2.p) annotation(
        Line(points = {{-16, 86}, {-28, 86}, {-28, -26}, {-16, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_f_serial.a, ctrl_l3.p) annotation(
        Line(points = {{-16, 92}, {-30, 92}, {-30, 12}, {-16, 12}}, color = {0, 0, 255}));
      connect(pll_alpha_f_serial.pin1, ctrl_l1.n) annotation(
        Line(points = {{4, 80}, {16, 80}, {16, -60}, {4, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_f_serial.pin2, ctrl_l2.n) annotation(
        Line(points = {{4, 86}, {18, 86}, {18, -26}, {4, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_f_serial.pin3, ctrl_l3.n) annotation(
        Line(points = {{4, 92}, {20, 92}, {20, 12}, {4, 12}}, color = {0, 0, 255}));
      connect(pin1, ctrl_r1.p) annotation(
        Line(points = {{-100, -60}, {-68, -60}, {-68, -60}, {-68, -60}}, color = {0, 0, 255}));
      connect(ctrl_r1.n, ctrl_l1.p) annotation(
        Line(points = {{-48, -60}, {-16, -60}, {-16, -60}, {-16, -60}}, color = {0, 0, 255}));
      connect(pin2, ctrl_r.p) annotation(
        Line(points = {{-100, 0}, {-92, 0}, {-92, -26}, {-68, -26}}, color = {0, 0, 255}));
      connect(pin3, ctrl_r2.p) annotation(
        Line(points = {{-100, 60}, {-92, 60}, {-92, 12}, {-68, 12}}, color = {0, 0, 255}));
      connect(ctrl_r.n, ctrl_l2.p) annotation(
        Line(points = {{-48, -26}, {-16, -26}}, color = {0, 0, 255}));
      connect(ctrl_r2.n, ctrl_l3.p) annotation(
        Line(points = {{-48, 12}, {-16, 12}, {-16, 12}, {-16, 12}}, color = {0, 0, 255}));
      connect(controller_r.u, pll_alpha_serial.u_eff) annotation(
        Line(points = {{-68, 54}, {-72, 54}, {-72, 66}, {-64, 66}, {-64, 74}, {-64, 74}}, color = {0, 0, 127}));
      connect(pll_alpha_serial.c, ctrl_r1.p) annotation(
        Line(points = {{-70, 78}, {-78, 78}, {-78, -60}, {-68, -60}, {-68, -60}, {-68, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.b, ctrl_r.p) annotation(
        Line(points = {{-70, 84}, {-80, 84}, {-80, -26}, {-68, -26}, {-68, -26}, {-68, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.a, ctrl_r2.p) annotation(
        Line(points = {{-70, 90}, {-82, 90}, {-82, 12}, {-68, 12}, {-68, 12}, {-68, 12}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin1, ctrl_r1.n) annotation(
        Line(points = {{-50, 78}, {-42, 78}, {-42, -60}, {-48, -60}, {-48, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin2, ctrl_r.n) annotation(
        Line(points = {{-50, 84}, {-40, 84}, {-40, -26}, {-48, -26}, {-48, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin3, ctrl_r2.n) annotation(
        Line(points = {{-50, 90}, {-38, 90}, {-38, 12}, {-48, 12}, {-48, 12}, {-48, 12}}, color = {0, 0, 255}));
      connect(controller_r.y, ctrl_r2.R_ctrl) annotation(
        Line(points = {{-48, 48}, {-44, 48}, {-44, 28}, {-58, 28}, {-58, 24}, {-58, 24}}, color = {0, 0, 127}));
      connect(controller_r.y, ctrl_r.R_ctrl) annotation(
        Line(points = {{-48, 48}, {-44, 48}, {-44, -10}, {-58, -10}, {-58, -14}, {-58, -14}, {-58, -14}}, color = {0, 0, 127}));
      connect(controller_r.y, ctrl_r1.R_ctrl) annotation(
        Line(points = {{-48, 48}, {-44, 48}, {-44, -46}, {-58, -46}, {-58, -48}, {-58, -48}, {-58, -48}}, color = {0, 0, 127}));
    protected
    end rl_ctrl;

    model r_ctrl
      parameter SI.Resistance R_start(start = 20);
      parameter SI.Power p_ref(start = 2500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {86, -76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_alpha_serial pll_alpha_serial annotation(
        Placement(visible = true, transformation(origin = {-60, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.controller_r controller_r(R_start = R_start, p_ref = p_ref) annotation(
        Placement(visible = true, transformation(origin = {-58, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_r ctrl_r annotation(
        Placement(visible = true, transformation(origin = {-58, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_r ctrl_r1 annotation(
        Placement(visible = true, transformation(origin = {-58, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.ctrl_r ctrl_r2 annotation(
        Placement(visible = true, transformation(origin = {-58, 12}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, ctrl_r1.p) annotation(
        Line(points = {{-100, -60}, {-68, -60}, {-68, -60}, {-68, -60}}, color = {0, 0, 255}));
      connect(pin2, ctrl_r.p) annotation(
        Line(points = {{-100, 0}, {-92, 0}, {-92, -26}, {-68, -26}}, color = {0, 0, 255}));
      connect(pin3, ctrl_r2.p) annotation(
        Line(points = {{-100, 60}, {-92, 60}, {-92, 12}, {-68, 12}}, color = {0, 0, 255}));
      connect(controller_r.u, pll_alpha_serial.u_eff) annotation(
        Line(points = {{-68, 54}, {-72, 54}, {-72, 66}, {-64, 66}, {-64, 74}, {-64, 74}}, color = {0, 0, 127}));
      connect(pll_alpha_serial.c, ctrl_r1.p) annotation(
        Line(points = {{-70, 78}, {-78, 78}, {-78, -60}, {-68, -60}, {-68, -60}, {-68, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.b, ctrl_r.p) annotation(
        Line(points = {{-70, 84}, {-80, 84}, {-80, -26}, {-68, -26}, {-68, -26}, {-68, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.a, ctrl_r2.p) annotation(
        Line(points = {{-70, 90}, {-82, 90}, {-82, 12}, {-68, 12}, {-68, 12}, {-68, 12}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin1, ctrl_r1.n) annotation(
        Line(points = {{-50, 78}, {-42, 78}, {-42, -60}, {-48, -60}, {-48, -60}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin2, ctrl_r.n) annotation(
        Line(points = {{-50, 84}, {-40, 84}, {-40, -26}, {-48, -26}, {-48, -26}}, color = {0, 0, 255}));
      connect(pll_alpha_serial.pin3, ctrl_r2.n) annotation(
        Line(points = {{-50, 90}, {-38, 90}, {-38, 12}, {-48, 12}, {-48, 12}, {-48, 12}}, color = {0, 0, 255}));
      connect(controller_r.y, ctrl_r2.R_ctrl) annotation(
        Line(points = {{-48, 48}, {-44, 48}, {-44, 28}, {-58, 28}, {-58, 24}, {-58, 24}}, color = {0, 0, 127}));
      connect(controller_r.y, ctrl_r.R_ctrl) annotation(
        Line(points = {{-48, 48}, {-44, 48}, {-44, -10}, {-58, -10}, {-58, -14}, {-58, -14}, {-58, -14}}, color = {0, 0, 127}));
      connect(controller_r.y, ctrl_r1.R_ctrl) annotation(
        Line(points = {{-48, 48}, {-44, 48}, {-44, -46}, {-58, -46}, {-58, -48}, {-58, -48}, {-58, -48}}, color = {0, 0, 127}));
      connect(ctrl_r1.n, ground.p) annotation(
        Line(points = {{-48, -60}, {86, -60}, {86, -66}, {86, -66}}, color = {0, 0, 255}));
      connect(ctrl_r.n, ground.p) annotation(
        Line(points = {{-48, -26}, {86, -26}, {86, -66}, {86, -66}}, color = {0, 0, 255}));
      connect(ctrl_r2.n, ground.p) annotation(
        Line(points = {{-48, 12}, {86, 12}, {86, -66}, {86, -66}}, color = {0, 0, 255}));
    protected
    end r_ctrl;

    model activ_z
      parameter Real p_ref(start = 5000);
      parameter Real q_ref(start = 2500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {86, -76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.transforms.dq2abc dq2abc annotation(
        Placement(visible = true, transformation(origin = {74, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_ueff pll_ueff annotation(
        Placement(visible = true, transformation(origin = {-54, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent3 annotation(
        Placement(visible = true, transformation(origin = {-20, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent1 annotation(
        Placement(visible = true, transformation(origin = {-20, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent2 annotation(
        Placement(visible = true, transformation(origin = {-20, -28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = q_ref * 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {4, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = p_ref * 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {4, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {38, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division1 annotation(
        Placement(visible = true, transformation(origin = {38, 54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Min min annotation(
        Placement(visible = true, transformation(origin = {-2, -42}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Min min1 annotation(
        Placement(visible = true, transformation(origin = {-2, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Min min2 annotation(
        Placement(visible = true, transformation(origin = {-2, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 30) annotation(
        Placement(visible = true, transformation(origin = {-24, -80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max annotation(
        Placement(visible = true, transformation(origin = {40, -44}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Gain gain(k = -1) annotation(
        Placement(visible = true, transformation(origin = {30, -80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max1 annotation(
        Placement(visible = true, transformation(origin = {40, -2}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Max max2 annotation(
        Placement(visible = true, transformation(origin = {40, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
    equation
      connect(pin1, pll_ueff.c) annotation(
        Line(points = {{-100, -60}, {-70, -60}, {-70, 76}, {-64, 76}, {-64, 76}}, color = {0, 0, 255}));
      connect(pin2, pll_ueff.b) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, 82}, {-64, 82}, {-64, 82}}, color = {0, 0, 255}));
      connect(pin3, pll_ueff.a) annotation(
        Line(points = {{-100, 60}, {-74, 60}, {-74, 88}, {-64, 88}, {-64, 88}}, color = {0, 0, 255}));
      connect(pll_ueff.theta, dq2abc.theta) annotation(
        Line(points = {{-44, 82}, {-12, 82}, {-12, 98}, {71, 98}, {71, 92}}, color = {0, 0, 127}));
      connect(pll_ueff.u_eff, division.u2) annotation(
        Line(points = {{-44, 76}, {-14, 76}, {-14, 80}, {20, 80}, {20, 78}, {26, 78}, {26, 78}}, color = {0, 0, 127}));
      connect(realExpression1.y, division.u1) annotation(
        Line(points = {{16, 90}, {24, 90}, {24, 90}, {26, 90}}, color = {0, 0, 127}));
      connect(division.y, dq2abc.d) annotation(
        Line(points = {{50, 84}, {64, 84}, {64, 84}, {64, 84}}, color = {0, 0, 127}));
      connect(realExpression.y, division1.u1) annotation(
        Line(points = {{16, 70}, {20, 70}, {20, 60}, {26, 60}, {26, 60}}, color = {0, 0, 127}));
      connect(pll_ueff.u_eff, division1.u2) annotation(
        Line(points = {{-44, 76}, {-14, 76}, {-14, 48}, {26, 48}, {26, 48}, {26, 48}}, color = {0, 0, 127}));
      connect(division1.y, dq2abc.q) annotation(
        Line(points = {{50, 54}, {60, 54}, {60, 76}, {62, 76}, {62, 76}, {64, 76}}, color = {0, 0, 127}));
      connect(pin1, signalCurrent1.p) annotation(
        Line(points = {{-100, -60}, {-30, -60}}, color = {0, 0, 255}));
      connect(pin2, signalCurrent2.p) annotation(
        Line(points = {{-100, 0}, {-42, 0}, {-42, -26}, {-30, -26}, {-30, -28}}, color = {0, 0, 255}));
      connect(pin3, signalCurrent3.p) annotation(
        Line(points = {{-100, 60}, {-36, 60}, {-36, 16}, {-30, 16}}, color = {0, 0, 255}));
      connect(signalCurrent3.n, ground.p) annotation(
        Line(points = {{-10, 16}, {86, 16}, {86, -66}}, color = {0, 0, 255}));
      connect(signalCurrent2.n, ground.p) annotation(
        Line(points = {{-10, -28}, {28, -28}, {28, -30}, {86, -30}, {86, -66}}, color = {0, 0, 255}));
      connect(signalCurrent1.n, ground.p) annotation(
        Line(points = {{-10, -60}, {86, -60}, {86, -66}}, color = {0, 0, 255}));
      connect(min2.y, signalCurrent3.i) annotation(
        Line(points = {{-13, 32}, {-20, 32}, {-20, 28}}, color = {0, 0, 127}));
      connect(min1.y, signalCurrent2.i) annotation(
        Line(points = {{-13, -4}, {-20, -4}, {-20, -16}}, color = {0, 0, 127}));
      connect(min.y, signalCurrent1.i) annotation(
        Line(points = {{-13, -42}, {-20, -42}, {-20, -48}}, color = {0, 0, 127}));
      connect(realExpression2.y, min.u1) annotation(
        Line(points = {{-13, -80}, {14, -80}, {14, -48}, {10, -48}}, color = {0, 0, 127}));
      connect(realExpression2.y, min1.u1) annotation(
        Line(points = {{-13, -80}, {14, -80}, {14, -10}, {10, -10}}, color = {0, 0, 127}));
      connect(realExpression2.y, min2.u1) annotation(
        Line(points = {{-13, -80}, {14, -80}, {14, 26}, {10, 26}}, color = {0, 0, 127}));
      connect(realExpression2.y, gain.u) annotation(
        Line(points = {{-13, -80}, {18, -80}}, color = {0, 0, 127}));
      connect(gain.y, max.u1) annotation(
        Line(points = {{41, -80}, {76, -80}, {76, -50}, {52, -50}}, color = {0, 0, 127}));
      connect(gain.y, max1.u1) annotation(
        Line(points = {{41, -80}, {76, -80}, {76, -8}, {52, -8}}, color = {0, 0, 127}));
      connect(gain.y, max2.u1) annotation(
        Line(points = {{41, -80}, {76, -80}, {76, 26}, {52, 26}}, color = {0, 0, 127}));
      connect(max.y, min.u2) annotation(
        Line(points = {{29, -44}, {20, -44}, {20, -36}, {10, -36}}, color = {0, 0, 127}));
      connect(max1.y, min1.u2) annotation(
        Line(points = {{29, -2}, {20, -2}, {20, 2}, {10, 2}}, color = {0, 0, 127}));
      connect(max2.y, min2.u2) annotation(
        Line(points = {{29, 32}, {18, 32}, {18, 38}, {10, 38}}, color = {0, 0, 127}));
      connect(dq2abc.a, max2.u2) annotation(
        Line(points = {{84, 86}, {98, 86}, {98, 38}, {52, 38}}, color = {0, 0, 127}));
      connect(dq2abc.b, max1.u2) annotation(
        Line(points = {{84, 80}, {96, 80}, {96, 4}, {52, 4}}, color = {0, 0, 127}));
      connect(dq2abc.c, max.u2) annotation(
        Line(points = {{84, 74}, {94, 74}, {94, -38}, {52, -38}}, color = {0, 0, 127}));
    protected
    end activ_z;

    model activ_z_test
      parameter Real p_ref(start = 5000);
      parameter Real q_ref(start = 2500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {86, -76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.transforms.dq2abc dq2abc annotation(
        Placement(visible = true, transformation(origin = {74, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_ueff_test pll_ueff annotation(
        Placement(visible = true, transformation(origin = {-54, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent3 annotation(
        Placement(visible = true, transformation(origin = {0, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent1 annotation(
        Placement(visible = true, transformation(origin = {0, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent2 annotation(
        Placement(visible = true, transformation(origin = {0, -28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = q_ref * 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {4, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = p_ref * 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {4, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {38, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division1 annotation(
        Placement(visible = true, transformation(origin = {38, 54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 230) annotation(
        Placement(visible = true, transformation(origin = {-26, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime = 0.0001) annotation(
        Placement(visible = true, transformation(origin = {66, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Nonlinear.FixedDelay fixedDelay1(delayTime = 0.0001) annotation(
        Placement(visible = true, transformation(origin = {44, -6}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Nonlinear.FixedDelay fixedDelay2(delayTime = 0.0001) annotation(
        Placement(visible = true, transformation(origin = {50, -44}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
    equation
      connect(pin1, pll_ueff.c) annotation(
        Line(points = {{-100, -60}, {-70, -60}, {-70, 76}, {-64, 76}, {-64, 76}}, color = {0, 0, 255}));
      connect(pin2, pll_ueff.b) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, 82}, {-64, 82}, {-64, 82}}, color = {0, 0, 255}));
      connect(pin3, pll_ueff.a) annotation(
        Line(points = {{-100, 60}, {-74, 60}, {-74, 88}, {-64, 88}, {-64, 88}}, color = {0, 0, 255}));
      connect(pll_ueff.theta, dq2abc.theta) annotation(
        Line(points = {{-44, 82}, {-12, 82}, {-12, 98}, {71, 98}, {71, 92}}, color = {0, 0, 127}));
      connect(pin1, signalCurrent1.p) annotation(
        Line(points = {{-100, -60}, {-10, -60}, {-10, -60}, {-10, -60}}, color = {0, 0, 255}));
      connect(pin2, signalCurrent2.p) annotation(
        Line(points = {{-100, 0}, {-38, 0}, {-38, -26}, {-10, -26}, {-10, -28}, {-10, -28}}, color = {0, 0, 255}));
      connect(pin3, signalCurrent3.p) annotation(
        Line(points = {{-100, 60}, {-40, 60}, {-40, 16}, {-10, 16}, {-10, 16}}, color = {0, 0, 255}));
      connect(signalCurrent1.n, ground.p) annotation(
        Line(points = {{10, -60}, {86, -60}, {86, -66}, {86, -66}}, color = {0, 0, 255}));
      connect(signalCurrent2.n, ground.p) annotation(
        Line(points = {{10, -28}, {86, -28}, {86, -66}, {86, -66}}, color = {0, 0, 255}));
      connect(signalCurrent3.n, ground.p) annotation(
        Line(points = {{10, 16}, {86, 16}, {86, -66}, {86, -66}}, color = {0, 0, 255}));
      connect(realExpression1.y, division.u1) annotation(
        Line(points = {{16, 90}, {24, 90}, {24, 90}, {26, 90}}, color = {0, 0, 127}));
      connect(division.y, dq2abc.d) annotation(
        Line(points = {{50, 84}, {64, 84}, {64, 84}, {64, 84}}, color = {0, 0, 127}));
      connect(realExpression.y, division1.u1) annotation(
        Line(points = {{16, 70}, {20, 70}, {20, 60}, {26, 60}, {26, 60}}, color = {0, 0, 127}));
      connect(division1.y, dq2abc.q) annotation(
        Line(points = {{50, 54}, {60, 54}, {60, 76}, {62, 76}, {62, 76}, {64, 76}}, color = {0, 0, 127}));
      connect(realExpression2.y, division1.u2) annotation(
        Line(points = {{-14, 66}, {-10, 66}, {-10, 48}, {26, 48}, {26, 48}}, color = {0, 0, 127}));
      connect(realExpression2.y, division.u2) annotation(
        Line(points = {{-14, 66}, {-10, 66}, {-10, 80}, {22, 80}, {22, 78}, {26, 78}, {26, 78}}, color = {0, 0, 127}));
      connect(dq2abc.a, fixedDelay.u) annotation(
        Line(points = {{84, 86}, {98, 86}, {98, 34}, {78, 34}, {78, 34}}, color = {0, 0, 127}));
      connect(dq2abc.b, fixedDelay1.u) annotation(
        Line(points = {{84, 80}, {94, 80}, {94, -4}, {56, -4}, {56, -6}}, color = {0, 0, 127}));
      connect(dq2abc.c, fixedDelay2.u) annotation(
        Line(points = {{84, 74}, {88, 74}, {88, -42}, {64, -42}, {64, -44}, {62, -44}}, color = {0, 0, 127}));
      connect(fixedDelay2.y, signalCurrent1.i) annotation(
        Line(points = {{40, -44}, {0, -44}, {0, -48}, {0, -48}}, color = {0, 0, 127}));
      connect(fixedDelay1.y, signalCurrent2.i) annotation(
        Line(points = {{34, -6}, {0, -6}, {0, -16}, {0, -16}, {0, -16}}, color = {0, 0, 127}));
      connect(fixedDelay.y, signalCurrent3.i) annotation(
        Line(points = {{54, 34}, {0, 34}, {0, 26}, {0, 26}, {0, 28}}, color = {0, 0, 127}));
    protected
    end activ_z_test;

    model activ_z_sicherung
      parameter Real p_ref(start = 5000);
      parameter Real q_ref(start = 2500);
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {86, -76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.transforms.dq2abc dq2abc annotation(
        Placement(visible = true, transformation(origin = {74, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll_ueff pll_ueff annotation(
        Placement(visible = true, transformation(origin = {-54, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent3 annotation(
        Placement(visible = true, transformation(origin = {-20, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent1 annotation(
        Placement(visible = true, transformation(origin = {-20, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalCurrent signalCurrent2 annotation(
        Placement(visible = true, transformation(origin = {-20, -28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = q_ref * 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {4, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = p_ref * 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {4, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {38, 84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division1 annotation(
        Placement(visible = true, transformation(origin = {38, 54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Min min annotation(
        Placement(visible = true, transformation(origin = {-2, -42}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Min min1 annotation(
        Placement(visible = true, transformation(origin = {-2, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Min min2 annotation(
        Placement(visible = true, transformation(origin = {-2, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 35) annotation(
        Placement(visible = true, transformation(origin = {-24, -80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max annotation(
        Placement(visible = true, transformation(origin = {40, -44}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Gain gain(k = -1) annotation(
        Placement(visible = true, transformation(origin = {30, -80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max1 annotation(
        Placement(visible = true, transformation(origin = {40, -2}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Math.Max max2 annotation(
        Placement(visible = true, transformation(origin = {40, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth(f = 100) annotation(
        Placement(visible = true, transformation(origin = {67, 39}, extent = {{-5, -5}, {5, 5}}, rotation = 180)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth1(f = 100) annotation(
        Placement(visible = true, transformation(origin = {61, -39}, extent = {{-5, -5}, {5, 5}}, rotation = 180)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth2(f = 100) annotation(
        Placement(visible = true, transformation(origin = {64, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
    equation
      connect(pin1, pll_ueff.c) annotation(
        Line(points = {{-100, -60}, {-70, -60}, {-70, 76}, {-64, 76}, {-64, 76}}, color = {0, 0, 255}));
      connect(pin2, pll_ueff.b) annotation(
        Line(points = {{-100, 0}, {-72, 0}, {-72, 82}, {-64, 82}, {-64, 82}}, color = {0, 0, 255}));
      connect(pin3, pll_ueff.a) annotation(
        Line(points = {{-100, 60}, {-74, 60}, {-74, 88}, {-64, 88}, {-64, 88}}, color = {0, 0, 255}));
      connect(pll_ueff.theta, dq2abc.theta) annotation(
        Line(points = {{-44, 82}, {-12, 82}, {-12, 98}, {71, 98}, {71, 92}}, color = {0, 0, 127}));
      connect(pll_ueff.u_eff, division.u2) annotation(
        Line(points = {{-44, 76}, {-14, 76}, {-14, 80}, {20, 80}, {20, 78}, {26, 78}, {26, 78}}, color = {0, 0, 127}));
      connect(realExpression1.y, division.u1) annotation(
        Line(points = {{16, 90}, {24, 90}, {24, 90}, {26, 90}}, color = {0, 0, 127}));
      connect(division.y, dq2abc.d) annotation(
        Line(points = {{50, 84}, {64, 84}, {64, 84}, {64, 84}}, color = {0, 0, 127}));
      connect(realExpression.y, division1.u1) annotation(
        Line(points = {{16, 70}, {20, 70}, {20, 60}, {26, 60}, {26, 60}}, color = {0, 0, 127}));
      connect(pll_ueff.u_eff, division1.u2) annotation(
        Line(points = {{-44, 76}, {-14, 76}, {-14, 48}, {26, 48}, {26, 48}, {26, 48}}, color = {0, 0, 127}));
      connect(division1.y, dq2abc.q) annotation(
        Line(points = {{50, 54}, {60, 54}, {60, 76}, {62, 76}, {62, 76}, {64, 76}}, color = {0, 0, 127}));
      connect(pin1, signalCurrent1.p) annotation(
        Line(points = {{-100, -60}, {-30, -60}}, color = {0, 0, 255}));
      connect(pin2, signalCurrent2.p) annotation(
        Line(points = {{-100, 0}, {-42, 0}, {-42, -26}, {-30, -26}, {-30, -28}}, color = {0, 0, 255}));
      connect(pin3, signalCurrent3.p) annotation(
        Line(points = {{-100, 60}, {-36, 60}, {-36, 16}, {-30, 16}}, color = {0, 0, 255}));
      connect(signalCurrent3.n, ground.p) annotation(
        Line(points = {{-10, 16}, {86, 16}, {86, -66}}, color = {0, 0, 255}));
      connect(signalCurrent2.n, ground.p) annotation(
        Line(points = {{-10, -28}, {28, -28}, {28, -30}, {86, -30}, {86, -66}}, color = {0, 0, 255}));
      connect(signalCurrent1.n, ground.p) annotation(
        Line(points = {{-10, -60}, {86, -60}, {86, -66}}, color = {0, 0, 255}));
      connect(min2.y, signalCurrent3.i) annotation(
        Line(points = {{-13, 32}, {-20, 32}, {-20, 28}}, color = {0, 0, 127}));
      connect(min1.y, signalCurrent2.i) annotation(
        Line(points = {{-13, -4}, {-20, -4}, {-20, -16}}, color = {0, 0, 127}));
      connect(min.y, signalCurrent1.i) annotation(
        Line(points = {{-13, -42}, {-20, -42}, {-20, -48}}, color = {0, 0, 127}));
      connect(realExpression2.y, min.u1) annotation(
        Line(points = {{-13, -80}, {14, -80}, {14, -48}, {10, -48}}, color = {0, 0, 127}));
      connect(realExpression2.y, min1.u1) annotation(
        Line(points = {{-13, -80}, {14, -80}, {14, -10}, {10, -10}}, color = {0, 0, 127}));
      connect(realExpression2.y, min2.u1) annotation(
        Line(points = {{-13, -80}, {14, -80}, {14, 26}, {10, 26}}, color = {0, 0, 127}));
      connect(realExpression2.y, gain.u) annotation(
        Line(points = {{-13, -80}, {18, -80}}, color = {0, 0, 127}));
      connect(gain.y, max.u1) annotation(
        Line(points = {{41, -80}, {76, -80}, {76, -50}, {52, -50}}, color = {0, 0, 127}));
      connect(gain.y, max1.u1) annotation(
        Line(points = {{41, -80}, {76, -80}, {76, -8}, {52, -8}}, color = {0, 0, 127}));
      connect(gain.y, max2.u1) annotation(
        Line(points = {{41, -80}, {76, -80}, {76, 26}, {52, 26}}, color = {0, 0, 127}));
      connect(max.y, min.u2) annotation(
        Line(points = {{29, -44}, {20, -44}, {20, -36}, {10, -36}}, color = {0, 0, 127}));
      connect(max1.y, min1.u2) annotation(
        Line(points = {{29, -2}, {20, -2}, {20, 2}, {10, 2}}, color = {0, 0, 127}));
      connect(max2.y, min2.u2) annotation(
        Line(points = {{29, 32}, {18, 32}, {18, 38}, {10, 38}}, color = {0, 0, 127}));
      connect(max1.u2, lowpassButterworth2.y) annotation(
        Line(points = {{52, 4}, {60, 4}, {60, 4}, {60, 4}}, color = {0, 0, 127}));
      connect(lowpassButterworth2.u, dq2abc.b) annotation(
        Line(points = {{68, 4}, {92, 4}, {92, 80}, {84, 80}, {84, 80}, {84, 80}}, color = {0, 0, 127}));
      connect(dq2abc.c, lowpassButterworth1.u) annotation(
        Line(points = {{84, 74}, {90, 74}, {90, -40}, {68, -40}, {68, -38}}, color = {0, 0, 127}));
      connect(lowpassButterworth1.y, max.u2) annotation(
        Line(points = {{56, -38}, {52, -38}, {52, -38}, {52, -38}}, color = {0, 0, 127}));
      connect(max2.u2, lowpassButterworth.y) annotation(
        Line(points = {{52, 38}, {62, 38}, {62, 40}, {62, 40}}, color = {0, 0, 127}));
      connect(lowpassButterworth.u, dq2abc.a) annotation(
        Line(points = {{74, 40}, {94, 40}, {94, 86}, {84, 86}, {84, 86}, {84, 86}}, color = {0, 0, 127}));
    protected
    end activ_z_sicherung;
  end active_loads;

  import SI = Modelica.SIunits;

  package filter
    model pi
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Capacitance C4 = 0.00001;
      parameter SI.Capacitance C5 = 0.00001;
      parameter SI.Capacitance C6 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Resistance R1 = 0.01;
      parameter SI.Resistance R2 = 0.01;
      parameter SI.Resistance R3 = 0.01;
      parameter SI.Resistance R4 = 0.01;
      parameter SI.Resistance R5 = 0.01;
      parameter SI.Resistance R6 = 0.01;
      parameter SI.Resistance R7 = 0.01;
      parameter SI.Resistance R8 = 0.01;
      parameter SI.Resistance R9 = 0.01;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-14, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-14, 52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-14, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-70, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {-48, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-26, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor4(C = C4) annotation(
        Placement(visible = true, transformation(origin = {26, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor5(C = C5) annotation(
        Placement(visible = true, transformation(origin = {46, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor6(C = C6) annotation(
        Placement(visible = true, transformation(origin = {64, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-70, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-48, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-26, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor4(R = R4) annotation(
        Placement(visible = true, transformation(origin = {10, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor5(R = R5) annotation(
        Placement(visible = true, transformation(origin = {10, 52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor6(R = R6) annotation(
        Placement(visible = true, transformation(origin = {10, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor7(R = R7) annotation(
        Placement(visible = true, transformation(origin = {26, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor8(R = R8) annotation(
        Placement(visible = true, transformation(origin = {46, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor9(R = R9) annotation(
        Placement(visible = true, transformation(origin = {64, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(inductor1.n, resistor4.p) annotation(
        Line(points = {{-4, 28}, {0, 28}, {0, 28}, {0, 28}}, color = {0, 0, 255}));
      connect(inductor2.n, resistor5.p) annotation(
        Line(points = {{-4, 52}, {-4, 52}, {-4, 52}, {0, 52}}, color = {0, 0, 255}));
      connect(inductor3.n, resistor6.p) annotation(
        Line(points = {{-4, 78}, {0, 78}, {0, 78}, {0, 78}}, color = {0, 0, 255}));
      connect(resistor3.p, pin3) annotation(
        Line(points = {{-26, 2}, {-26, 2}, {-26, 18}, {-36, 18}, {-36, 78}, {-90, 78}, {-90, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(resistor2.p, pin2) annotation(
        Line(points = {{-48, 2}, {-48, 2}, {-48, 52}, {-84, 52}, {-84, 0}, {-100, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(resistor1.p, pin1) annotation(
        Line(points = {{-70, 2}, {-80, 2}, {-80, -60}, {-98, -60}, {-98, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(resistor1.n, capacitor1.p) annotation(
        Line(points = {{-70, -18}, {-70, -18}, {-70, -28}, {-70, -28}}, color = {0, 0, 255}));
      connect(resistor2.n, capacitor2.p) annotation(
        Line(points = {{-48, -18}, {-48, -18}, {-48, -28}, {-48, -28}}, color = {0, 0, 255}));
      connect(resistor3.n, capacitor3.p) annotation(
        Line(points = {{-26, -18}, {-26, -18}, {-26, -28}, {-26, -28}}, color = {0, 0, 255}));
      connect(resistor9.n, capacitor6.p) annotation(
        Line(points = {{64, -18}, {64, -18}, {64, -28}, {64, -28}}, color = {0, 0, 255}));
      connect(capacitor6.n, capacitor5.n) annotation(
        Line(points = {{64, -48}, {46, -48}}, color = {0, 0, 255}));
      connect(capacitor5.p, resistor8.n) annotation(
        Line(points = {{46, -28}, {46, -28}, {46, -18}, {46, -18}}, color = {0, 0, 255}));
      connect(resistor5.n, resistor8.p) annotation(
        Line(points = {{20, 52}, {46, 52}, {46, 2}}, color = {0, 0, 255}));
      connect(resistor7.n, capacitor4.p) annotation(
        Line(points = {{26, -18}, {26, -18}, {26, -28}, {26, -28}}, color = {0, 0, 255}));
      connect(resistor6.n, pin6) annotation(
        Line(points = {{20, 78}, {84, 78}, {84, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(resistor5.n, pin5) annotation(
        Line(points = {{20, 52}, {84, 52}, {84, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(resistor4.n, pin4) annotation(
        Line(points = {{20, 28}, {74, 28}, {74, 28}, {76, 28}, {76, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(resistor6.n, resistor9.p) annotation(
        Line(points = {{20, 78}, {64, 78}, {64, 2}, {64, 2}}, color = {0, 0, 255}));
      connect(resistor4.n, resistor7.p) annotation(
        Line(points = {{20, 28}, {26, 28}, {26, 2}, {26, 2}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, 28}, {-24, 28}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-90, 60}, {-90, 78}, {-24, 78}}, color = {0, 0, 255}));
      connect(capacitor3.n, ground1.p) annotation(
        Line(points = {{-26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
      connect(capacitor4.n, ground1.p) annotation(
        Line(points = {{26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
      connect(inductor2.p, pin2) annotation(
        Line(points = {{-24, 52}, {-84, 52}, {-84, 0}, {-100, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor3.n) annotation(
        Line(points = {{-48, -48}, {-26, -48}, {-26, -48}, {-26, -48}}, color = {0, 0, 255}));
      connect(capacitor1.n, capacitor2.n) annotation(
        Line(points = {{-70, -48}, {-48, -48}, {-48, -48}, {-48, -48}}, color = {0, 0, 255}));
      connect(capacitor5.n, capacitor4.n) annotation(
        Line(points = {{46, -48}, {26, -48}, {26, -48}, {26, -48}}, color = {0, 0, 255}));
    end pi;

    model lcl
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Inductance L4 = 0.001;
      parameter SI.Inductance L5 = 0.001;
      parameter SI.Inductance L6 = 0.001;
      parameter SI.Resistance R1 = 0.01;
      parameter SI.Resistance R2 = 0.01;
      parameter SI.Resistance R3 = 0.01;
      parameter SI.Resistance R4 = 0.01;
      parameter SI.Resistance R5 = 0.01;
      parameter SI.Resistance R6 = 0.01;
      parameter SI.Resistance R7 = 0.01;
      parameter SI.Resistance R8 = 0.01;
      parameter SI.Resistance R9 = 0.01;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-64, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-72, 86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {38, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-28, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
        Placement(visible = true, transformation(origin = {68, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
        Placement(visible = true, transformation(origin = {70, 40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
        Placement(visible = true, transformation(origin = {74, 62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1 annotation(
        Placement(visible = true, transformation(origin = {-36, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2 annotation(
        Placement(visible = true, transformation(origin = {-32, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3 annotation(
        Placement(visible = true, transformation(origin = {-30, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor4 annotation(
        Placement(visible = true, transformation(origin = {42, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor5 annotation(
        Placement(visible = true, transformation(origin = {8, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor6 annotation(
        Placement(visible = true, transformation(origin = {-22, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor7 annotation(
        Placement(visible = true, transformation(origin = {32, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor8 annotation(
        Placement(visible = true, transformation(origin = {40, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor9 annotation(
        Placement(visible = true, transformation(origin = {34, 68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(resistor2.n, resistor5.p) annotation(
        Line(points = {{-22, 48}, {8, 48}, {8, 0}}, color = {0, 0, 255}));
      connect(resistor8.p, resistor2.n) annotation(
        Line(points = {{30, 44}, {2, 44}, {2, 48}, {-22, 48}}, color = {0, 0, 255}));
      connect(resistor2.p, inductor2.n) annotation(
        Line(points = {{-42, 48}, {-50, 48}, {-50, 58}, {-54, 58}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 58}, {-74, 58}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 86}, {-82, 86}}, color = {0, 0, 255}));
      connect(resistor3.p, inductor3.n) annotation(
        Line(points = {{-40, 82}, {-47, 82}, {-47, 86}, {-62, 86}}, color = {0, 0, 255}));
      connect(resistor3.n, resistor9.p) annotation(
        Line(points = {{-20, 82}, {3, 82}, {3, 68}, {24, 68}}, color = {0, 0, 255}));
      connect(resistor6.p, resistor3.n) annotation(
        Line(points = {{-22, 2}, {-22, 41}, {-20, 41}, {-20, 82}}, color = {0, 0, 255}));
      connect(inductor6.n, pin6) annotation(
        Line(points = {{84, 62}, {84, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(resistor9.n, inductor6.p) annotation(
        Line(points = {{44, 68}, {55, 68}, {55, 62}, {64, 62}}, color = {0, 0, 255}));
      connect(inductor5.n, pin5) annotation(
        Line(points = {{80, 40}, {88, 40}, {88, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(resistor8.n, inductor5.p) annotation(
        Line(points = {{50, 44}, {55, 44}, {55, 40}, {60, 40}}, color = {0, 0, 255}));
      connect(resistor7.n, inductor4.p) annotation(
        Line(points = {{42, 30}, {54, 30}, {54, 6}, {58, 6}}, color = {0, 0, 255}));
      connect(resistor4.p, resistor7.p) annotation(
        Line(points = {{42, 0}, {42, 14.5}, {22, 14.5}, {22, 30}}, color = {0, 0, 255}));
      connect(resistor1.n, resistor7.p) annotation(
        Line(points = {{-26, 20}, {2, 20}, {2, 30}, {22, 30}}, color = {0, 0, 255}));
      connect(inductor4.n, pin4) annotation(
        Line(points = {{78, 6}, {80, 6}, {80, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor1.n) annotation(
        Line(points = {{12, -46}, {23, -46}, {23, -56}, {38, -56}}, color = {0, 0, 255}));
      connect(resistor4.n, capacitor1.p) annotation(
        Line(points = {{42, -20}, {42, -24}, {38, -24}, {38, -36}}, color = {0, 0, 255}));
      connect(resistor5.n, capacitor2.p) annotation(
        Line(points = {{8, -20}, {8, -24}, {12, -24}, {12, -26}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-28, -46}, {12, -46}}, color = {0, 0, 255}));
      connect(resistor6.n, capacitor3.p) annotation(
        Line(points = {{-22, -18}, {-22, -24}, {-28, -24}, {-28, -26}}, color = {0, 0, 255}));
      connect(resistor1.p, inductor1.n) annotation(
        Line(points = {{-46, 20}, {-50, 20}, {-50, 20}, {-50, 20}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
    end lcl;

    model lc
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Resistance R1 = 0.01;
      parameter SI.Resistance R2 = 0.01;
      parameter SI.Resistance R3 = 0.01;
      parameter SI.Resistance R4 = 0.01;
      parameter SI.Resistance R5 = 0.01;
      parameter SI.Resistance R6 = 0.01;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-60, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-60, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {32, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-8, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-34, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-26, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor4(R = R4) annotation(
        Placement(visible = true, transformation(origin = {32, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor5(R = R5) annotation(
        Placement(visible = true, transformation(origin = {12, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor6(R = R6) annotation(
        Placement(visible = true, transformation(origin = {-8, -8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(resistor3.n, resistor6.p) annotation(
        Line(points = {{-16, 70}, {-8, 70}, {-8, 2}}, color = {0, 0, 255}));
      connect(resistor3.n, pin6) annotation(
        Line(points = {{-16, 70}, {80, 70}, {80, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor3.n, resistor3.p) annotation(
        Line(points = {{-50, 70}, {-36, 70}}, color = {0, 0, 255}));
      connect(pin4, resistor1.n) annotation(
        Line(points = {{100, -60}, {62, -60}, {62, 20}, {-24, 20}, {-24, 20}}, color = {0, 0, 255}));
      connect(resistor4.n, capacitor1.p) annotation(
        Line(points = {{32, -18}, {32, -18}, {32, -26}, {32, -26}}, color = {0, 0, 255}));
      connect(resistor5.n, capacitor2.p) annotation(
        Line(points = {{12, -18}, {12, -18}, {12, -26}, {12, -26}}, color = {0, 0, 255}));
      connect(resistor6.n, capacitor3.p) annotation(
        Line(points = {{-8, -18}, {-8, -18}, {-8, -26}, {-8, -26}}, color = {0, 0, 255}));
      connect(pin5, resistor2.n) annotation(
        Line(points = {{100, 0}, {78, 0}, {78, 44}, {-24, 44}, {-24, 44}}, color = {0, 0, 255}));
      connect(resistor2.n, resistor5.p) annotation(
        Line(points = {{-24, 44}, {12, 44}, {12, 2}, {12, 2}}, color = {0, 0, 255}));
      connect(resistor1.n, resistor4.p) annotation(
        Line(points = {{-24, 20}, {32, 20}, {32, 2}, {32, 2}}, color = {0, 0, 255}));
      connect(inductor1.n, resistor1.p) annotation(
        Line(points = {{-50, 20}, {-44, 20}, {-44, 20}, {-44, 20}}, color = {0, 0, 255}));
      connect(inductor2.n, resistor2.p) annotation(
        Line(points = {{-50, 44}, {-44, 44}, {-44, 44}, {-44, 44}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-8, -46}, {12, -46}, {12, -46}, {12, -46}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor1.n) annotation(
        Line(points = {{12, -46}, {32, -46}, {32, -46}, {32, -46}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
    end lc;

    model lclc
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Capacitance C4 = 0.00001;
      parameter SI.Capacitance C5 = 0.00001;
      parameter SI.Capacitance C6 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Inductance L4 = 0.001;
      parameter SI.Inductance L5 = 0.001;
      parameter SI.Inductance L6 = 0.001;
      parameter SI.Resistance R1 = 0.01;
      parameter SI.Resistance R2 = 0.01;
      parameter SI.Resistance R3 = 0.01;
      parameter SI.Resistance R4 = 0.01;
      parameter SI.Resistance R5 = 0.01;
      parameter SI.Resistance R6 = 0.01;
      parameter SI.Resistance R7 = 0.01;
      parameter SI.Resistance R8 = 0.01;
      parameter SI.Resistance R9 = 0.01;
      parameter SI.Resistance R10 = 0.01;
      parameter SI.Resistance R11 = 0.01;
      parameter SI.Resistance R12 = 0.01;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-82, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-82, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-84, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-2, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {-22, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-42, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {16, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
        Placement(visible = true, transformation(origin = {34, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
        Placement(visible = true, transformation(origin = {34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
        Placement(visible = true, transformation(origin = {36, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor4(C = C4) annotation(
        Placement(visible = true, transformation(origin = {72, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor5(C = C5) annotation(
        Placement(visible = true, transformation(origin = {52, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor6(C = C6) annotation(
        Placement(visible = true, transformation(origin = {32, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-56, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-56, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-56, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor4(R = R4) annotation(
        Placement(visible = true, transformation(origin = {-2, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor5(R = R5) annotation(
        Placement(visible = true, transformation(origin = {-22, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor6(R = R6) annotation(
        Placement(visible = true, transformation(origin = {-42, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor7(R = R7) annotation(
        Placement(visible = true, transformation(origin = {10, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor8(R = R8) annotation(
        Placement(visible = true, transformation(origin = {10, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor9(R = R9) annotation(
        Placement(visible = true, transformation(origin = {10, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor10(R = R10) annotation(
        Placement(visible = true, transformation(origin = {72, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor11(R = R11) annotation(
        Placement(visible = true, transformation(origin = {52, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor12(R = R12) annotation(
        Placement(visible = true, transformation(origin = {32, -14}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(resistor10.p, inductor4.n) annotation(
        Line(points = {{72, -4}, {72, -4}, {72, 20}, {44, 20}, {44, 20}}, color = {0, 0, 255}));
      connect(inductor5.n, resistor11.p) annotation(
        Line(points = {{44, 44}, {54, 44}, {54, 44}, {52, 44}, {52, -4}, {52, -4}}, color = {0, 0, 255}));
      connect(resistor12.p, inductor6.n) annotation(
        Line(points = {{32, -4}, {32, -4}, {32, 10}, {58, 10}, {58, 70}, {46, 70}, {46, 70}}, color = {0, 0, 255}));
      connect(resistor11.n, capacitor5.p) annotation(
        Line(points = {{52, -24}, {52, -24}, {52, -28}, {52, -28}}, color = {0, 0, 255}));
      connect(resistor12.n, capacitor6.p) annotation(
        Line(points = {{32, -24}, {32, -24}, {32, -28}, {32, -28}}, color = {0, 0, 255}));
      connect(resistor1.n, resistor4.p) annotation(
        Line(points = {{-46, 20}, {-2, 20}, {-2, -4}, {-2, -4}, {-2, -4}}, color = {0, 0, 255}));
      connect(resistor2.n, resistor5.p) annotation(
        Line(points = {{-46, 44}, {-22, 44}, {-22, -4}, {-22, -4}, {-22, -4}}, color = {0, 0, 255}));
      connect(resistor6.p, resistor3.n) annotation(
        Line(points = {{-42, -4}, {-42, -4}, {-42, 70}, {-46, 70}, {-46, 70}}, color = {0, 0, 255}));
      connect(resistor1.n, resistor7.p) annotation(
        Line(points = {{-46, 20}, {-46, 20}, {-46, 20}, {0, 20}}, color = {0, 0, 255}));
      connect(resistor2.n, resistor8.p) annotation(
        Line(points = {{-46, 44}, {-46, 44}, {-46, 44}, {0, 44}}, color = {0, 0, 255}));
      connect(resistor3.n, resistor9.p) annotation(
        Line(points = {{-46, 70}, {0, 70}, {0, 70}, {0, 70}}, color = {0, 0, 255}));
      connect(resistor7.n, inductor4.p) annotation(
        Line(points = {{20, 20}, {24, 20}, {24, 20}, {24, 20}}, color = {0, 0, 255}));
      connect(resistor6.n, capacitor3.p) annotation(
        Line(points = {{-42, -24}, {-42, -24}, {-42, -28}, {-42, -28}}, color = {0, 0, 255}));
      connect(resistor5.n, capacitor2.p) annotation(
        Line(points = {{-22, -24}, {-22, -24}, {-22, -24}, {-22, -28}}, color = {0, 0, 255}));
      connect(resistor4.n, capacitor1.p) annotation(
        Line(points = {{-2, -24}, {-2, -24}, {-2, -28}, {-2, -28}}, color = {0, 0, 255}));
      connect(resistor10.n, capacitor4.p) annotation(
        Line(points = {{72, -24}, {72, -24}, {72, -28}, {72, -28}}, color = {0, 0, 255}));
      connect(resistor8.n, inductor5.p) annotation(
        Line(points = {{20, 44}, {24, 44}}, color = {0, 0, 255}));
      connect(resistor9.n, inductor6.p) annotation(
        Line(points = {{20, 70}, {26, 70}, {26, 70}, {26, 70}}, color = {0, 0, 255}));
      connect(inductor1.n, resistor1.p) annotation(
        Line(points = {{-72, 20}, {-66, 20}, {-66, 20}, {-66, 20}}, color = {0, 0, 255}));
      connect(inductor3.n, resistor3.p) annotation(
        Line(points = {{-74, 70}, {-74, 70}, {-74, 70}, {-66, 70}}, color = {0, 0, 255}));
      connect(inductor2.n, resistor2.p) annotation(
        Line(points = {{-72, 44}, {-66, 44}, {-66, 44}, {-66, 44}}, color = {0, 0, 255}));
      connect(inductor4.n, pin4) annotation(
        Line(points = {{44, 20}, {92, 20}, {92, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor6.n, pin6) annotation(
        Line(points = {{46, 70}, {76, 70}, {76, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(capacitor6.n, ground1.p) annotation(
        Line(points = {{32, -48}, {16, -48}, {16, -60}}, color = {0, 0, 255}));
      connect(capacitor6.n, capacitor5.n) annotation(
        Line(points = {{32, -48}, {52, -48}, {52, -48}, {52, -48}}, color = {0, 0, 255}));
      connect(capacitor5.n, capacitor4.n) annotation(
        Line(points = {{52, -48}, {72, -48}, {72, -48}, {72, -48}}, color = {0, 0, 255}));
      connect(capacitor1.n, ground1.p) annotation(
        Line(points = {{-2, -48}, {16, -48}, {16, -60}, {16, -60}}, color = {0, 0, 255}));
      connect(inductor5.n, pin5) annotation(
        Line(points = {{44, 44}, {94, 44}, {94, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-42, -48}, {-22, -48}, {-22, -48}, {-22, -48}}, color = {0, 0, 255}));
      connect(capacitor1.n, capacitor2.n) annotation(
        Line(points = {{-2, -48}, {-22, -48}, {-22, -48}, {-22, -48}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-94, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-95, 0}, {-95, 44}, {-92, 44}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-93, -60}, {-93, 20}, {-92, 20}}, color = {0, 0, 255}));
    end lclc;

    model l
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Resistance R1 = 0.01;
      parameter SI.Resistance R2 = 0.01;
      parameter SI.Resistance R3 = 0.01;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-60, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-60, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-32, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {-32, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {-32, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(resistor1.n, pin4) annotation(
        Line(points = {{-22, 20}, {14, 20}, {14, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(resistor2.n, pin5) annotation(
        Line(points = {{-22, 44}, {70, 44}, {70, 0}, {100, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(resistor3.n, pin6) annotation(
        Line(points = {{-22, 70}, {80, 70}, {80, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor1.n, resistor1.p) annotation(
        Line(points = {{-50, 20}, {-50, 20}, {-50, 20}, {-42, 20}}, color = {0, 0, 255}));
      connect(inductor2.n, resistor2.p) annotation(
        Line(points = {{-50, 44}, {-42, 44}, {-42, 44}, {-42, 44}}, color = {0, 0, 255}));
      connect(inductor3.n, resistor3.p) annotation(
        Line(points = {{-50, 70}, {-42, 70}, {-42, 70}, {-42, 70}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
    end l;

    model r
      parameter SI.Resistance R1 = 20;
      parameter SI.Resistance R2 = 20;
      parameter SI.Resistance R3 = 20;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, resistor1.p) annotation(
        Line(points = {{-100, -60}, {-76, -60}, {-76, -60}, {-76, -60}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {-100, 0}, {-100, 0}, {-76, 0}}, color = {0, 0, 255}));
      connect(pin3, resistor3.p) annotation(
        Line(points = {{-100, 60}, {-76, 60}, {-76, 60}, {-76, 60}}, color = {0, 0, 255}));
      connect(resistor3.n, pin6) annotation(
        Line(points = {{-56, 60}, {98, 60}, {98, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(resistor2.n, pin5) annotation(
        Line(points = {{-56, 0}, {102, 0}, {102, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(resistor1.n, pin4) annotation(
        Line(points = {{-56, -60}, {102, -60}, {102, -60}, {100, -60}}, color = {0, 0, 255}));
    end r;
  end filter;

  package loads
    model rc
      parameter SI.Resistance R1 = 20;
      parameter SI.Resistance R2 = 20;
      parameter SI.Resistance R3 = 20;
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-66, -48}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {-32, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {48, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-50, -48}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {0, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {72, -2}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(resistor2.n, ground1.p) annotation(
        Line(points = {{0, -20}, {0, -20}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {0, 0}, {0, 0}, {0, 0}}, color = {0, 0, 255}));
      connect(resistor1.p, pin1) annotation(
        Line(points = {{-50, -38}, {-50, -38}, {-50, -22}, {-90, -22}, {-90, -60}, {-100, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(resistor1.n, ground1.p) annotation(
        Line(points = {{-50, -58}, {-50, -58}, {-50, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(resistor3.n, ground1.p) annotation(
        Line(points = {{72, -12}, {72, -12}, {72, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin3, resistor3.p) annotation(
        Line(points = {{-100, 60}, {68, 60}, {68, 60}, {72, 60}, {72, 8}, {72, 8}}, color = {0, 0, 255}));
      connect(capacitor1.p, pin1) annotation(
        Line(points = {{-66, -38}, {-66, -22}, {-90, -22}, {-90, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(capacitor3.p, pin3) annotation(
        Line(points = {{48, 10}, {48, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(pin3, capacitor3.p) annotation(
        Line(points = {{-100, 60}, {48, 60}, {48, 10}}, color = {0, 0, 255}));
      connect(capacitor3.n, ground1.p) annotation(
        Line(points = {{48, -10}, {48, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor2.p, pin2) annotation(
        Line(points = {{-32, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{-32, -20}, {-32, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor1.n, ground1.p) annotation(
        Line(points = {{-66, -58}, {-66, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
    end rc;

    model c
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-34, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {4, -10}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {40, 38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {4, -84}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(capacitor3.p, pin3) annotation(
        Line(points = {{40, 48}, {40, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(pin2, capacitor2.p) annotation(
        Line(points = {{-102, 0}, {4, 0}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{4, -20}, {4, -74}}, color = {0, 0, 255}));
      connect(pin1, capacitor1.p) annotation(
        Line(points = {{-100, -60}, {-67, -60}, {-67, -34}, {-34, -34}}, color = {0, 0, 255}));
      connect(capacitor3.n, ground1.p) annotation(
        Line(points = {{40, 28}, {40, -54}, {4, -54}, {4, -74}}, color = {0, 0, 255}));
      connect(capacitor1.n, ground1.p) annotation(
        Line(points = {{-34, -54}, {4, -54}, {4, -74}, {4, -74}}, color = {0, 0, 255}));
    end c;

    model r
      parameter SI.Resistance R1 = 20;
      parameter SI.Resistance R2 = 20;
      parameter SI.Resistance R3 = 20;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor2(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.resistor resistor3(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-66, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(resistor3.n, ground1.p) annotation(
        Line(points = {{-56, 60}, {0, 60}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(resistor2.n, ground1.p) annotation(
        Line(points = {{-56, 0}, {-56, 0}, {-56, 0}, {0, 0}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(resistor1.n, ground1.p) annotation(
        Line(points = {{-56, -60}, {0, -60}, {0, -76}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin1, resistor1.p) annotation(
        Line(points = {{-100, -60}, {-76, -60}, {-76, -60}, {-76, -60}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {-100, 0}, {-100, 0}, {-76, 0}}, color = {0, 0, 255}));
      connect(pin3, resistor3.p) annotation(
        Line(points = {{-100, 60}, {-76, 60}, {-76, 60}, {-76, 60}}, color = {0, 0, 255}));
    end r;

    model l
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-48, -50}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {0, -16}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {50, 50}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(inductor3.n, ground1.p) annotation(
        Line(points = {{50, 40}, {50, 40}, {50, -60}, {0, -60}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {50, 60}, {50, 60}, {50, 60}}, color = {0, 0, 255}));
      connect(inductor2.n, ground1.p) annotation(
        Line(points = {{0, -26}, {0, -26}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {0, 0}, {0, -6}}, color = {0, 0, 255}));
      connect(inductor1.n, ground1.p) annotation(
        Line(points = {{-48, -60}, {0, -60}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-78, -60}, {-78, -40}, {-48, -40}, {-48, -40}, {-48, -40}}, color = {0, 0, 255}));
    end l;

    model rlc
      parameter SI.Resistance R1 = 20;
      parameter SI.Resistance R2 = 20;
      parameter SI.Resistance R3 = 20;
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-74, -68}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {0, -30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {74, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-74, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {0, 2}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {74, -4}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-74, -20}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {0, 36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {74, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(resistor1.p, pin1) annotation(
        Line(points = {{-74, -10}, {-100, -10}, {-100, -60}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {-50, 0}, {-50, 46}, {0, 46}}, color = {0, 0, 255}));
      connect(resistor3.p, pin3) annotation(
        Line(points = {{74, 52}, {74, 52}, {74, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{0, -40}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor3.n, ground1.p) annotation(
        Line(points = {{74, -56}, {74, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor1.p, inductor1.n) annotation(
        Line(points = {{-74, -58}, {-74, -58}, {-74, -54}, {-74, -54}}, color = {0, 0, 255}));
      connect(resistor1.n, inductor1.p) annotation(
        Line(points = {{-74, -30}, {-74, -30}, {-74, -30}, {-74, -34}}, color = {0, 0, 255}));
      connect(resistor2.n, inductor2.p) annotation(
        Line(points = {{0, 26}, {0, 26}, {0, 12}, {0, 12}}, color = {0, 0, 255}));
      connect(inductor2.n, capacitor2.p) annotation(
        Line(points = {{0, -8}, {0, -8}, {0, -8}, {0, -20}}, color = {0, 0, 255}));
      connect(resistor3.n, inductor3.p) annotation(
        Line(points = {{74, 32}, {74, 32}, {74, 6}, {74, 6}}, color = {0, 0, 255}));
      connect(capacitor3.p, inductor3.n) annotation(
        Line(points = {{74, -36}, {74, -36}, {74, -14}, {74, -14}}, color = {0, 0, 255}));
      connect(capacitor1.n, ground1.p) annotation(
        Line(points = {{-74, -78}, {-46, -78}, {-46, -62}, {0, -62}, {0, -76}, {0, -76}}, color = {0, 0, 255}));
    end rlc;

    model lc
      parameter SI.Capacitance C1(start = 0.00001);
      parameter SI.Capacitance C2(start = 0.00001);
      parameter SI.Capacitance C3(start = 0.00001);
      parameter SI.Inductance L1(start = 0.001);
      parameter SI.Inductance L2(start = 0.001);
      parameter SI.Inductance L3(start = 0.001);
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-56, -18}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {0, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {56, 42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-56, -44}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {0, -24}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {56, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(capacitor3.n, inductor3.p) annotation(
        Line(points = {{56, 32}, {56, 32}, {56, 18}, {56, 18}}, color = {0, 0, 255}));
      connect(inductor3.n, ground1.p) annotation(
        Line(points = {{56, -2}, {56, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor1.n, inductor1.p) annotation(
        Line(points = {{-56, -28}, {-56, -28}, {-56, -34}, {-56, -34}}, color = {0, 0, 255}));
      connect(inductor1.n, ground1.p) annotation(
        Line(points = {{-56, -54}, {-56, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(capacitor1.p, pin1) annotation(
        Line(points = {{-56, -8}, {-56, 0}, {-88, 0}, {-88, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(capacitor2.p, pin2) annotation(
        Line(points = {{0, 18}, {0, 24}, {-94, 24}, {-94, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(capacitor2.n, inductor2.p) annotation(
        Line(points = {{0, -2}, {0, -2}, {0, -14}, {0, -14}}, color = {0, 0, 255}));
      connect(inductor2.n, ground1.p) annotation(
        Line(points = {{0, -34}, {0, -76}}, color = {0, 0, 255}));
      connect(pin3, capacitor3.p) annotation(
        Line(points = {{-100, 60}, {56, 60}, {56, 52}}, color = {0, 0, 255}));
      connect(capacitor3.p, pin3) annotation(
        Line(points = {{56, 52}, {56, 60}, {-100, 60}}, color = {0, 0, 255}));
    end lc;

    model rl
      parameter SI.Resistance R1 = 20;
      parameter SI.Resistance R2 = 20;
      parameter SI.Resistance R3 = 20;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-38, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {0, -18}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor1(R = R1) annotation(
        Placement(visible = true, transformation(origin = {-40, -20}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor2(R = R2) annotation(
        Placement(visible = true, transformation(origin = {0, 12}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      grid.components.resistor resistor3(R = R3) annotation(
        Placement(visible = true, transformation(origin = {60, 34}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(resistor1.n, inductor1.p) annotation(
        Line(points = {{-40, -30}, {-40, -33}, {-38, -33}, {-38, -36}}, color = {0, 0, 255}));
      connect(pin3, resistor3.p) annotation(
        Line(points = {{-100, 60}, {60, 60}, {60, 44}, {60, 44}}, color = {0, 0, 255}));
      connect(resistor3.n, inductor3.p) annotation(
        Line(points = {{60, 24}, {60, 24}, {60, 24}, {60, 18}}, color = {0, 0, 255}));
      connect(resistor2.n, inductor2.p) annotation(
        Line(points = {{0, 2}, {0, 2}, {0, -8}, {0, -8}}, color = {0, 0, 255}));
      connect(pin2, resistor2.p) annotation(
        Line(points = {{-100, 0}, {-66, 0}, {-66, 22}, {0, 22}}, color = {0, 0, 255}));
      connect(pin1, resistor1.p) annotation(
        Line(points = {{-100, -60}, {-74, -60}, {-74, -10}, {-40, -10}, {-40, -10}}, color = {0, 0, 255}));
      connect(inductor3.n, ground1.p) annotation(
        Line(points = {{60, -2}, {60, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(inductor1.n, ground1.p) annotation(
        Line(points = {{-38, -56}, {-38, -62}, {0, -62}, {0, -76}}, color = {0, 0, 255}));
      connect(inductor2.n, ground1.p) annotation(
        Line(points = {{0, -28}, {0, -76}}, color = {0, 0, 255}));
    end rl;
  end loads;

  package inverters
    model inverter
      //  input Real regler1;
      //  input Real regler2;
      //  input Real regler3;
      parameter Real v_DC = 1000;
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {-74, -82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage1 annotation(
        Placement(visible = true, transformation(origin = {-74, -42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage2 annotation(
        Placement(visible = true, transformation(origin = {-74, 18}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage3 annotation(
        Placement(visible = true, transformation(origin = {-74, 78}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealInput u1 annotation(
        Placement(visible = true, transformation(origin = {-104, -60}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, -60}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u3 annotation(
        Placement(visible = true, transformation(origin = {-104, 60}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 60}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u2 annotation(
        Placement(visible = true, transformation(origin = {-104, 4.44089e-16}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 4.44089e-16}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = v_DC) annotation(
        Placement(visible = true, transformation(origin = {-26, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = v_DC) annotation(
        Placement(visible = true, transformation(origin = {-26, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = v_DC) annotation(
        Placement(visible = true, transformation(origin = {-26, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(signalVoltage2.p, pin2) annotation(
        Line(points = {{-74, 28}, {80, 28}, {80, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(signalVoltage3.p, pin3) annotation(
        Line(points = {{-74, 88}, {80, 88}, {80, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(signalVoltage1.p, pin1) annotation(
        Line(points = {{-74, -32}, {80, -32}, {80, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(signalVoltage3.n, ground1.p) annotation(
        Line(points = {{-74, 68}, {-74, 48}, {-82, 48}, {-82, -72}, {-74, -72}}, color = {0, 0, 255}));
      connect(signalVoltage2.n, ground1.p) annotation(
        Line(points = {{-74, 8}, {-82, 8}, {-82, -72}, {-74, -72}}, color = {0, 0, 255}));
      connect(signalVoltage1.n, ground1.p) annotation(
        Line(points = {{-74, -52}, {-74, -72}}, color = {0, 0, 255}));
/*  connect(signalVoltage1.v, regler1) annotation(
        Line);
      connect(signalVoltage2.v, regler2) annotation(
        Line);
      connect(signalVoltage3.v, regler3) annotation(
        Line);
    */
      connect(u1, gain1.u) annotation(
        Line(points = {{-104, -60}, {-38, -60}, {-38, -60}, {-38, -60}}, color = {0, 0, 127}));
      connect(gain1.y, signalVoltage1.v) annotation(
        Line(points = {{-14, -60}, {-6, -60}, {-6, -42}, {-60, -42}, {-60, -42}, {-62, -42}}, color = {0, 0, 127}));
      connect(u2, gain2.u) annotation(
        Line(points = {{-104, 0}, {-38, 0}, {-38, 0}, {-38, 0}}, color = {0, 0, 127}));
      connect(gain2.y, signalVoltage2.v) annotation(
        Line(points = {{-14, 0}, {-6, 0}, {-6, 18}, {-62, 18}, {-62, 18}}, color = {0, 0, 127}));
      connect(u3, gain3.u) annotation(
        Line(points = {{-104, 60}, {-38, 60}, {-38, 60}, {-38, 60}}, color = {0, 0, 127}));
      connect(gain3.y, signalVoltage3.v) annotation(
        Line(points = {{-14, 60}, {-6, 60}, {-6, 78}, {-62, 78}, {-62, 78}}, color = {0, 0, 127}));
      annotation(
        uses(Modelica(version = "3.2.3")));
    end inverter;

    model inverter_ctrl
      //  input Real regler1;
      //  input Real regler2;
      //  input Real regler3;
      parameter Real v_DC = 1000;
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {-74, -82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage1 annotation(
        Placement(visible = true, transformation(origin = {-74, -42}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage2 annotation(
        Placement(visible = true, transformation(origin = {-74, 18}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage3 annotation(
        Placement(visible = true, transformation(origin = {-74, 78}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealInput u1 annotation(
        Placement(visible = true, transformation(origin = {-104, -60}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, -60}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u3 annotation(
        Placement(visible = true, transformation(origin = {-104, 60}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 60}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u2 annotation(
        Placement(visible = true, transformation(origin = {-104, 4.44089e-16}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 4.44089e-16}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = v_DC) annotation(
        Placement(visible = true, transformation(origin = {-26, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = v_DC) annotation(
        Placement(visible = true, transformation(origin = {-26, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = v_DC) annotation(
        Placement(visible = true, transformation(origin = {-26, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(signalVoltage2.p, pin2) annotation(
        Line(points = {{-74, 28}, {80, 28}, {80, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(signalVoltage3.p, pin3) annotation(
        Line(points = {{-74, 88}, {80, 88}, {80, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(signalVoltage1.p, pin1) annotation(
        Line(points = {{-74, -32}, {80, -32}, {80, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(signalVoltage3.n, ground1.p) annotation(
        Line(points = {{-74, 68}, {-74, 48}, {-82, 48}, {-82, -72}, {-74, -72}}, color = {0, 0, 255}));
      connect(signalVoltage2.n, ground1.p) annotation(
        Line(points = {{-74, 8}, {-82, 8}, {-82, -72}, {-74, -72}}, color = {0, 0, 255}));
      connect(signalVoltage1.n, ground1.p) annotation(
        Line(points = {{-74, -52}, {-74, -72}}, color = {0, 0, 255}));
/*  connect(signalVoltage1.v, regler1) annotation(
        Line);
      connect(signalVoltage2.v, regler2) annotation(
        Line);
      connect(signalVoltage3.v, regler3) annotation(
        Line);
    */
      connect(u1, gain1.u) annotation(
        Line(points = {{-104, -60}, {-38, -60}, {-38, -60}, {-38, -60}}, color = {0, 0, 127}));
      connect(gain1.y, signalVoltage1.v) annotation(
        Line(points = {{-14, -60}, {-6, -60}, {-6, -42}, {-60, -42}, {-60, -42}, {-62, -42}}, color = {0, 0, 127}));
      connect(u2, gain2.u) annotation(
        Line(points = {{-104, 0}, {-38, 0}, {-38, 0}, {-38, 0}}, color = {0, 0, 127}));
      connect(gain2.y, signalVoltage2.v) annotation(
        Line(points = {{-14, 0}, {-6, 0}, {-6, 18}, {-62, 18}, {-62, 18}}, color = {0, 0, 127}));
      connect(u3, gain3.u) annotation(
        Line(points = {{-104, 60}, {-38, 60}, {-38, 60}, {-38, 60}}, color = {0, 0, 127}));
      connect(gain3.y, signalVoltage3.v) annotation(
        Line(points = {{-14, 60}, {-6, 60}, {-6, 78}, {-62, 78}, {-62, 78}}, color = {0, 0, 127}));
      annotation(
        uses(Modelica(version = "3.2.3")));
    end inverter_ctrl;
  end inverters;

  package ideal_filter
    model pi
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Capacitance C4 = 0.00001;
      parameter SI.Capacitance C5 = 0.00001;
      parameter SI.Capacitance C6 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {2, -16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {2, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {0, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-70, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {-48, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-26, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {0, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor4(C = C4) annotation(
        Placement(visible = true, transformation(origin = {26, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor5(C = C5) annotation(
        Placement(visible = true, transformation(origin = {46, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor6(C = C6) annotation(
        Placement(visible = true, transformation(origin = {66, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(inductor3.n, capacitor6.p) annotation(
        Line(points = {{10, 60}, {66, 60}, {66, -28}, {66, -28}}, color = {0, 0, 255}));
      connect(capacitor3.p, pin3) annotation(
        Line(points = {{-26, -28}, {-26, -28}, {-26, 60}, {-100, 60}, {-100, 60}}, color = {0, 0, 255}));
      connect(inductor1.n, pin4) annotation(
        Line(points = {{12, -16}, {70, -16}, {70, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor4.p) annotation(
        Line(points = {{12, -16}, {26, -16}, {26, -28}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-80, -60}, {-80, -16}, {-8, -16}}, color = {0, 0, 255}));
      connect(inductor3.n, pin6) annotation(
        Line(points = {{10, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-10, 60}}, color = {0, 0, 255}));
      connect(inductor2.n, pin5) annotation(
        Line(points = {{12, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(inductor2.n, capacitor5.p) annotation(
        Line(points = {{12, 0}, {46, 0}, {46, -28}}, color = {0, 0, 255}));
      connect(inductor2.p, pin2) annotation(
        Line(points = {{-8, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(capacitor1.p, pin1) annotation(
        Line(points = {{-70, -28}, {-80, -28}, {-80, -60}, {-100, -60}}, color = {0, 0, 255}));
      connect(capacitor2.p, pin2) annotation(
        Line(points = {{-48, -28}, {-48, -28}, {-48, 0}, {-100, 0}, {-100, 0}}, color = {0, 0, 255}));
      connect(capacitor3.n, ground1.p) annotation(
        Line(points = {{-26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
      connect(capacitor4.n, ground1.p) annotation(
        Line(points = {{26, -48}, {0, -48}, {0, -60}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor3.n) annotation(
        Line(points = {{-48, -48}, {-26, -48}, {-26, -48}, {-26, -48}}, color = {0, 0, 255}));
      connect(capacitor1.n, capacitor2.n) annotation(
        Line(points = {{-70, -48}, {-48, -48}, {-48, -48}, {-48, -48}}, color = {0, 0, 255}));
      connect(capacitor5.n, capacitor4.n) annotation(
        Line(points = {{46, -48}, {26, -48}, {26, -48}, {26, -48}}, color = {0, 0, 255}));
      connect(capacitor6.n, capacitor5.n) annotation(
        Line(points = {{66, -48}, {46, -48}, {46, -48}, {46, -48}}, color = {0, 0, 255}));
    end pi;

    model lcl
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Inductance L4 = 0.001;
      parameter SI.Inductance L5 = 0.001;
      parameter SI.Inductance L6 = 0.001;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-60, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-60, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {32, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-8, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
        Placement(visible = true, transformation(origin = {68, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
        Placement(visible = true, transformation(origin = {74, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
        Placement(visible = true, transformation(origin = {64, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(inductor2.n, inductor5.p) annotation(
        Line(points = {{-50, 44}, {-50, 44}, {-50, 44}, {64, 44}}, color = {0, 0, 255}));
      connect(inductor2.n, capacitor2.p) annotation(
        Line(points = {{-50, 44}, {12, 44}, {12, -26}, {12, -26}}, color = {0, 0, 255}));
      connect(inductor1.n, inductor4.p) annotation(
        Line(points = {{-50, 20}, {-50, 20}, {-50, 20}, {58, 20}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor1.p) annotation(
        Line(points = {{-50, 20}, {32, 20}, {32, -26}, {32, -26}}, color = {0, 0, 255}));
      connect(inductor3.n, capacitor3.p) annotation(
        Line(points = {{-50, 70}, {-8, 70}, {-8, -26}, {-8, -26}}, color = {0, 0, 255}));
      connect(inductor3.n, inductor6.p) annotation(
        Line(points = {{-50, 70}, {54, 70}, {54, 70}, {54, 70}}, color = {0, 0, 255}));
      connect(inductor4.n, pin4) annotation(
        Line(points = {{78, 20}, {80, 20}, {80, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor6.n, pin6) annotation(
        Line(points = {{74, 70}, {84, 70}, {84, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(inductor5.n, pin5) annotation(
        Line(points = {{84, 44}, {88, 44}, {88, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor1.n) annotation(
        Line(points = {{12, -46}, {32, -46}, {32, -46}, {32, -46}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-8, -46}, {12, -46}, {12, -46}, {12, -46}}, color = {0, 0, 255}));
    end lcl;

    model lc
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-60, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-60, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {32, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {12, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {12, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-8, -36}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(inductor1.n, pin4) annotation(
        Line(points = {{-50, 20}, {54, 20}, {54, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor2.n, pin5) annotation(
        Line(points = {{-50, 44}, {68, 44}, {68, 0}, {100, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(inductor3.n, pin6) annotation(
        Line(points = {{-50, 70}, {80, 70}, {80, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor3.n, capacitor3.p) annotation(
        Line(points = {{-50, 70}, {-8, 70}, {-8, -26}, {-8, -26}}, color = {0, 0, 255}));
      connect(inductor2.n, capacitor2.p) annotation(
        Line(points = {{-50, 44}, {12, 44}, {12, -26}, {12, -26}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor1.p) annotation(
        Line(points = {{-50, 20}, {32, 20}, {32, -26}, {32, -26}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-8, -46}, {12, -46}, {12, -46}, {12, -46}}, color = {0, 0, 255}));
      connect(capacitor2.n, ground1.p) annotation(
        Line(points = {{12, -46}, {12, -46}, {12, -58}, {12, -58}}, color = {0, 0, 255}));
      connect(capacitor2.n, capacitor1.n) annotation(
        Line(points = {{12, -46}, {32, -46}, {32, -46}, {32, -46}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
    end lc;

    model lclc
      parameter SI.Capacitance C1 = 0.00001;
      parameter SI.Capacitance C2 = 0.00001;
      parameter SI.Capacitance C3 = 0.00001;
      parameter SI.Capacitance C4 = 0.00001;
      parameter SI.Capacitance C5 = 0.00001;
      parameter SI.Capacitance C6 = 0.00001;
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      parameter SI.Inductance L4 = 0.001;
      parameter SI.Inductance L5 = 0.001;
      parameter SI.Inductance L6 = 0.001;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-82, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-82, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-84, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C = C1) annotation(
        Placement(visible = true, transformation(origin = {-2, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C = C2) annotation(
        Placement(visible = true, transformation(origin = {-22, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor3(C = C3) annotation(
        Placement(visible = true, transformation(origin = {-42, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Ground ground1 annotation(
        Placement(visible = true, transformation(origin = {16, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor4(L = L4) annotation(
        Placement(visible = true, transformation(origin = {34, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor5(L = L5) annotation(
        Placement(visible = true, transformation(origin = {34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor6(L = L6) annotation(
        Placement(visible = true, transformation(origin = {36, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor4(C = C4) annotation(
        Placement(visible = true, transformation(origin = {72, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor5(C = C5) annotation(
        Placement(visible = true, transformation(origin = {52, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Basic.Capacitor capacitor6(C = C6) annotation(
        Placement(visible = true, transformation(origin = {32, -38}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    equation
      connect(inductor4.n, capacitor4.p) annotation(
        Line(points = {{44, 20}, {72, 20}, {72, -28}, {72, -28}}, color = {0, 0, 255}));
      connect(inductor6.n, capacitor6.p) annotation(
        Line(points = {{46, 70}, {64, 70}, {64, 4}, {32, 4}, {32, -28}}, color = {0, 0, 255}));
      connect(capacitor3.p, inductor3.n) annotation(
        Line(points = {{-42, -28}, {-42, -28}, {-42, 70}, {-74, 70}, {-74, 70}}, color = {0, 0, 255}));
      connect(capacitor2.p, inductor2.n) annotation(
        Line(points = {{-22, -28}, {-22, -28}, {-22, 44}, {-72, 44}, {-72, 44}}, color = {0, 0, 255}));
      connect(inductor5.n, capacitor5.p) annotation(
        Line(points = {{44, 44}, {52, 44}, {52, -28}, {52, -28}}, color = {0, 0, 255}));
      connect(inductor3.n, inductor6.p) annotation(
        Line(points = {{-74, 70}, {26, 70}, {26, 70}, {26, 70}}, color = {0, 0, 255}));
      connect(inductor2.n, inductor5.p) annotation(
        Line(points = {{-72, 44}, {-72, 44}, {-72, 44}, {24, 44}}, color = {0, 0, 255}));
      connect(inductor1.n, inductor4.p) annotation(
        Line(points = {{-72, 20}, {24, 20}, {24, 20}, {24, 20}}, color = {0, 0, 255}));
      connect(inductor1.n, capacitor1.p) annotation(
        Line(points = {{-72, 20}, {-2, 20}, {-2, -28}, {-2, -28}}, color = {0, 0, 255}));
      connect(inductor4.n, pin4) annotation(
        Line(points = {{44, 20}, {92, 20}, {92, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(inductor6.n, pin6) annotation(
        Line(points = {{46, 70}, {76, 70}, {76, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(capacitor6.n, ground1.p) annotation(
        Line(points = {{32, -48}, {16, -48}, {16, -60}}, color = {0, 0, 255}));
      connect(capacitor6.n, capacitor5.n) annotation(
        Line(points = {{32, -48}, {52, -48}, {52, -48}, {52, -48}}, color = {0, 0, 255}));
      connect(capacitor5.n, capacitor4.n) annotation(
        Line(points = {{52, -48}, {72, -48}, {72, -48}, {72, -48}}, color = {0, 0, 255}));
      connect(capacitor1.n, ground1.p) annotation(
        Line(points = {{-2, -48}, {16, -48}, {16, -60}, {16, -60}}, color = {0, 0, 255}));
      connect(inductor5.n, pin5) annotation(
        Line(points = {{44, 44}, {94, 44}, {94, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(capacitor3.n, capacitor2.n) annotation(
        Line(points = {{-42, -48}, {-22, -48}, {-22, -48}, {-22, -48}}, color = {0, 0, 255}));
      connect(capacitor1.n, capacitor2.n) annotation(
        Line(points = {{-2, -48}, {-22, -48}, {-22, -48}, {-22, -48}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-94, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-95, 0}, {-95, 44}, {-92, 44}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-93, -60}, {-93, 20}, {-92, 20}}, color = {0, 0, 255}));
    end lclc;

    model l
      parameter SI.Inductance L1 = 0.001;
      parameter SI.Inductance L2 = 0.001;
      parameter SI.Inductance L3 = 0.001;
      Modelica.Electrical.Analog.Basic.Inductor inductor1(L = L1) annotation(
        Placement(visible = true, transformation(origin = {-60, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor2(L = L2) annotation(
        Placement(visible = true, transformation(origin = {-60, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Inductor inductor3(L = L3) annotation(
        Placement(visible = true, transformation(origin = {-60, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin6 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(inductor3.n, pin6) annotation(
        Line(points = {{-50, 70}, {80, 70}, {80, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(inductor2.n, pin5) annotation(
        Line(points = {{-50, 44}, {78, 44}, {78, 0}, {100, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(inductor1.n, pin4) annotation(
        Line(points = {{-50, 20}, {60, 20}, {60, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(pin1, inductor1.p) annotation(
        Line(points = {{-100, -60}, {-85, -60}, {-85, 20}, {-70, 20}}, color = {0, 0, 255}));
      connect(pin3, inductor3.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 70}, {-70, 70}}, color = {0, 0, 255}));
      connect(pin2, inductor2.p) annotation(
        Line(points = {{-100, 0}, {-91, 0}, {-91, 44}, {-70, 44}}, color = {0, 0, 255}));
    end l;
  end ideal_filter;

  package components
    model resistor
      parameter SI.Resistance R(start = 1);
      extends Modelica.Electrical.Analog.Interfaces.OnePort;
    equation
      v = R * i;
      annotation(
        Documentation(info = "<html>
<p>The linear resistor connects the branch voltage <em>v</em> with the branch current <em>i</em> by <em>i*R = v</em>. The Resistance <em>R</em> is allowed to be positive, zero, or negative.</p>
</html>", revisions = "<html>
<ul>
<li><em> August 07, 2009   </em>
       by Anton Haumer<br> temperature dependency of resistance added<br>
       </li>
<li><em> March 11, 2009   </em>
       by Christoph Clauss<br> conditional heat port added<br>
       </li>
<li><em> 1998   </em>
       by Christoph Clauss<br> initially implemented<br>
       </li>
</ul>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Text(extent = {{-150, -40}, {150, -80}}, textString = "R=%R"), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
    end resistor;

    model RMS
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll pll annotation(
        Placement(visible = true, transformation(origin = {-60, 88}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.transforms.abc2dq abc2dq annotation(
        Placement(visible = true, transformation(origin = {30, 70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {0, -86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor annotation(
        Placement(visible = true, transformation(origin = {-50, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor1 annotation(
        Placement(visible = true, transformation(origin = {-50, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor2 annotation(
        Placement(visible = true, transformation(origin = {-50, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {67, 77}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {67, 65}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Math.Add add annotation(
        Placement(visible = true, transformation(origin = {85, 71}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {52, 46}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression sqrt2(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {52, 32}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {73, 39}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput rms annotation(
        Placement(visible = true, transformation(origin = {106, 40}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput freq annotation(
        Placement(visible = true, transformation(origin = {106, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, pll.c) annotation(
        Line(points = {{-100, -60}, {-74, -60}, {-74, 82}, {-70, 82}}, color = {0, 0, 255}));
      connect(pin2, pll.b) annotation(
        Line(points = {{-100, 0}, {-80, 0}, {-80, 88}, {-70, 88}}, color = {0, 0, 255}));
      connect(pin3, pll.a) annotation(
        Line(points = {{-100, 60}, {-86, 60}, {-86, 94}, {-70, 94}}, color = {0, 0, 255}));
      connect(pll.theta, abc2dq.u) annotation(
        Line(points = {{-50, 82}, {27, 82}}, color = {0, 0, 127}));
      connect(pin1, voltageSensor1.p) annotation(
        Line(points = {{-100, -60}, {-60, -60}, {-60, -60}, {-60, -60}}, color = {0, 0, 255}));
      connect(voltageSensor1.n, ground.p) annotation(
        Line(points = {{-40, -60}, {0, -60}, {0, -76}}, color = {0, 0, 255}));
      connect(pin2, voltageSensor2.p) annotation(
        Line(points = {{-100, 0}, {-60, 0}, {-60, 0}, {-60, 0}}, color = {0, 0, 255}));
      connect(voltageSensor2.n, ground.p) annotation(
        Line(points = {{-40, 0}, {0, 0}, {0, -76}}, color = {0, 0, 255}));
      connect(pin3, voltageSensor.p) annotation(
        Line(points = {{-100, 60}, {-60, 60}, {-60, 60}, {-60, 60}}, color = {0, 0, 255}));
      connect(voltageSensor.n, ground.p) annotation(
        Line(points = {{-40, 60}, {0, 60}, {0, -76}}, color = {0, 0, 255}));
      connect(voltageSensor1.v, abc2dq.c) annotation(
        Line(points = {{-50, -70}, {16, -70}, {16, 68}, {20, 68}, {20, 68}}, color = {0, 0, 127}));
      connect(voltageSensor2.v, abc2dq.b) annotation(
        Line(points = {{-50, -10}, {12, -10}, {12, 70}, {20, 70}, {20, 72}}, color = {0, 0, 127}));
      connect(voltageSensor.v, abc2dq.a) annotation(
        Line(points = {{-50, 48}, {8, 48}, {8, 74}, {20, 74}, {20, 74}}, color = {0, 0, 127}));
      connect(abc2dq.d, product.u1) annotation(
        Line(points = {{40, 76}, {48, 76}, {48, 80}, {60, 80}, {60, 80}}, color = {0, 0, 127}));
      connect(abc2dq.d, product.u2) annotation(
        Line(points = {{40, 76}, {48, 76}, {48, 74}, {60, 74}, {60, 74}}, color = {0, 0, 127}));
      connect(abc2dq.q, product1.u1) annotation(
        Line(points = {{40, 68}, {61, 68}}, color = {0, 0, 127}));
      connect(abc2dq.q, product1.u2) annotation(
        Line(points = {{40, 68}, {48, 68}, {48, 62}, {61, 62}}, color = {0, 0, 127}));
      connect(product1.y, add.u2) annotation(
        Line(points = {{72, 66}, {76, 66}, {76, 68}, {78, 68}, {78, 68}}, color = {0, 0, 127}));
      connect(product.y, add.u1) annotation(
        Line(points = {{72, 78}, {74, 78}, {74, 74}, {78, 74}, {78, 74}, {78, 74}}, color = {0, 0, 127}));
      connect(add.y, sqrt1.u) annotation(
        Line(points = {{90, 72}, {96, 72}, {96, 58}, {38, 58}, {38, 46}, {44, 46}, {44, 46}}, color = {0, 0, 127}));
      connect(sqrt1.y, division.u1) annotation(
        Line(points = {{58, 46}, {64, 46}, {64, 42}, {67, 42}}, color = {0, 0, 127}));
      connect(sqrt2.y, division.u2) annotation(
        Line(points = {{58, 32}, {62, 32}, {62, 36}, {67, 36}}, color = {0, 0, 127}));
      connect(pll.f, freq) annotation(
        Line(points = {{-50, 94}, {-20, 94}, {-20, -40}, {106, -40}, {106, -40}}, color = {0, 0, 127}));
      connect(division.y, rms) annotation(
        Line(points = {{78, 40}, {98, 40}, {98, 40}, {106, 40}}, color = {0, 0, 127}));
    end RMS;

    model active_load
      parameter SI.Power p_ref(start = 1);
      parameter SI.Resistance r_min(start = 1);
      Real u_eff;
      SI.Resistance R;
      extends Modelica.Electrical.Analog.Interfaces.OnePort;
      Modelica.Blocks.Interfaces.RealInput u annotation(
        Placement(visible = true, transformation(origin = {0, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {0, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
    equation
      u_eff = abs(u);
      R = max(u_eff * u_eff / p_ref, 0.1);
      v = R * i;
      annotation(
        Documentation(info = "<html>
  <p>The linear resistor connects the branch voltage <em>v</em> with the branch current <em>i</em> by <em>i*R = v</em>. The Resistance <em>R</em> is allowed to be positive, zero, or negative.</p>
  </html>", revisions = "<html>
  <ul>
  <li><em> March 11, 2009   </em>
       by Christoph Clauss<br> conditional heat port added<br>
       </li>
  <li><em> 1998   </em>
       by Christoph Clauss<br> initially implemented<br>
       </li>
  </ul>
  </html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Text(extent = {{-150, -40}, {150, -80}}, textString = "R=%R"), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
    end active_load;

    model RMS_i
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.plls.pll pll annotation(
        Placement(visible = true, transformation(origin = {-60, 88}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.transforms.abc2dq abc2dq annotation(
        Placement(visible = true, transformation(origin = {6, 76}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {43, 83}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {43, 71}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Math.Add add annotation(
        Placement(visible = true, transformation(origin = {61, 77}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {6, 46}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression sqrt2(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {6, 32}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {27, 39}, extent = {{-5, -5}, {5, 5}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput rms annotation(
        Placement(visible = true, transformation(origin = {40, -106}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {40, -106}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealOutput freq annotation(
        Placement(visible = true, transformation(origin = {-40, -106}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -106}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor annotation(
        Placement(visible = true, transformation(origin = {-50, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor1 annotation(
        Placement(visible = true, transformation(origin = {-50, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor2 annotation(
        Placement(visible = true, transformation(origin = {-50, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin4 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin5 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(pin1, pll.c) annotation(
        Line(points = {{-100, -60}, {-74, -60}, {-74, 82}, {-70, 82}}, color = {0, 0, 255}));
      connect(pin2, pll.b) annotation(
        Line(points = {{-100, 0}, {-80, 0}, {-80, 88}, {-70, 88}}, color = {0, 0, 255}));
      connect(pin3, pll.a) annotation(
        Line(points = {{-100, 60}, {-86, 60}, {-86, 94}, {-70, 94}}, color = {0, 0, 255}));
      connect(pll.theta, abc2dq.u) annotation(
        Line(points = {{-50, 82}, {-24.5, 82}, {-24.5, 88}, {3, 88}}, color = {0, 0, 127}));
      connect(abc2dq.d, product.u1) annotation(
        Line(points = {{16, 82}, {24, 82}, {24, 86}, {37, 86}}, color = {0, 0, 127}));
      connect(abc2dq.d, product.u2) annotation(
        Line(points = {{16, 82}, {26.5, 82}, {26.5, 80}, {37, 80}}, color = {0, 0, 127}));
      connect(abc2dq.q, product1.u1) annotation(
        Line(points = {{16, 74}, {37, 74}}, color = {0, 0, 127}));
      connect(abc2dq.q, product1.u2) annotation(
        Line(points = {{16, 74}, {24, 74}, {24, 68}, {37, 68}}, color = {0, 0, 127}));
      connect(product1.y, add.u2) annotation(
        Line(points = {{48.5, 71}, {52, 71}, {52, 74}, {55, 74}}, color = {0, 0, 127}));
      connect(product.y, add.u1) annotation(
        Line(points = {{48.5, 83}, {50, 83}, {50, 80}, {55, 80}}, color = {0, 0, 127}));
      connect(add.y, sqrt1.u) annotation(
        Line(points = {{66.5, 77}, {72, 77}, {72, 64}, {-6, 64}, {-6, 46}, {-1, 46}}, color = {0, 0, 127}));
      connect(sqrt1.y, division.u1) annotation(
        Line(points = {{13, 46}, {18, 46}, {18, 42}, {21, 42}}, color = {0, 0, 127}));
      connect(sqrt2.y, division.u2) annotation(
        Line(points = {{13, 32}, {16, 32}, {16, 36}, {21, 36}}, color = {0, 0, 127}));
      connect(pll.f, freq) annotation(
        Line(points = {{-50, 94}, {-16, 94}, {-16, -80}, {-40, -80}, {-40, -106}}, color = {0, 0, 127}));
      connect(pin1, currentSensor1.p) annotation(
        Line(points = {{-100, -60}, {-60, -60}, {-60, -60}, {-60, -60}}, color = {0, 0, 255}));
      connect(currentSensor1.n, pin4) annotation(
        Line(points = {{-40, -60}, {100, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(pin2, currentSensor2.p) annotation(
        Line(points = {{-100, 0}, {-60, 0}, {-60, 0}, {-60, 0}}, color = {0, 0, 255}));
      connect(pin3, currentSensor.p) annotation(
        Line(points = {{-100, 60}, {-60, 60}, {-60, 60}, {-60, 60}}, color = {0, 0, 255}));
      connect(currentSensor.n, pin) annotation(
        Line(points = {{-40, 60}, {100, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(currentSensor2.n, pin5) annotation(
        Line(points = {{-40, 0}, {102, 0}, {102, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(division.y, rms) annotation(
        Line(points = {{32, 40}, {40, 40}, {40, -106}, {40, -106}}, color = {0, 0, 127}));
      connect(currentSensor.i, abc2dq.a) annotation(
        Line(points = {{-50, 48}, {-36, 48}, {-36, 80}, {-4, 80}, {-4, 80}}, color = {0, 0, 127}));
      connect(currentSensor2.i, abc2dq.b) annotation(
        Line(points = {{-50, -10}, {-32, -10}, {-32, 76}, {-4, 76}, {-4, 78}}, color = {0, 0, 127}));
      connect(currentSensor1.i, abc2dq.c) annotation(
        Line(points = {{-50, -70}, {-24, -70}, {-24, 74}, {-4, 74}, {-4, 74}}, color = {0, 0, 127}));
    end RMS_i;

    model active_l
      parameter SI.Power q_ref(start = 10);
      Real u_eff;
      SI.Inductance L;
      Real freq;
      extends Modelica.Electrical.Analog.Interfaces.OnePort;
      Modelica.Blocks.Interfaces.RealInput u annotation(
        Placement(visible = true, transformation(origin = {-40, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {-40, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealInput f annotation(
        Placement(visible = true, transformation(origin = {40, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {40, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
    equation
      u_eff = max(u, 0.1);
      freq = max(f, 0.1);
      L = max(abs(u_eff * u_eff / q_ref / 2 / 3.1415 / freq), 0.000001);
      L * der(i) = v;
      annotation(
        Documentation(info = "<html>
    <p>Reactive load, q_ref can be either set positiv (induvtiv load) or negativ (capacitive load).</p>
    </html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Text(extent = {{-150, -40}, {150, -80}}, textString = "R=%R"), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
    end active_l;

    block Mean "Calculate mean over period 1/f"
      extends Modelica.Blocks.Interfaces.SISO;
      parameter Modelica.SIunits.Frequency f(start = 50) "Base frequency";
      parameter Real x0 = 0 "Start value of integrator state";
      parameter Boolean yGreaterOrEqualZero = false "=true, if output y is guaranteed to be >= 0 for the exact solution" annotation(
        Evaluate = true,
        Dialog(tab = "Advanced"));
    protected
      parameter Modelica.SIunits.Time t0(fixed = false) "Start time of simulation";
      Real x "Integrator state";
    initial equation
      t0 = time;
      x = x0;
      y = u;
    equation
      der(x) = u;
      when sample(t0 + 1 / f, 1 / f) then
        y = if time < 1 / f then u else if not yGreaterOrEqualZero then f * pre(x) else max(0.0, f * pre(x));
        reinit(x, 0);
      end when;
      annotation(
        Documentation(info = "<html>
    <p>
    This block calculates the mean of the input signal u over the given period 1/f:
    </p>
    <pre>
    1 T
    - &int; u(t) dt
    T 0
    </pre>
    <p>
    Note: The output is updated after each period defined by 1/f.
    </p>
    
    <p>
    If parameter <strong>yGreaterOrEqualZero</strong> in the Advanced tab is <strong>true</strong> (default = <strong>false</strong>),
    then the modeller provides the information that the mean of the input signal is guaranteed
    to be &ge; 0 for the exact solution. However, due to inaccuracies in the numerical integration scheme,
    the output might be slightly negative. If this parameter is set to true, then the output is
    explicitly set to 0.0, if the mean value results in a negative value.
    </p>
    </html>"),
        Icon(graphics = {Text(extent = {{-80, 60}, {80, 20}}, textString = "mean"), Text(extent = {{-80, -20}, {80, -60}}, textString = "f=%f")}));
    end Mean;

    model controller
      parameter SI.Power q_ref;
      parameter SI.Frequency f_ref;
      Modelica.Blocks.Math.RootMeanSquare rootMeanSquare1 annotation(
        Placement(visible = true, transformation(origin = {-50, -50}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.RootMeanSquare rootMeanSquare annotation(
        Placement(visible = true, transformation(origin = {-50, 50}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Feedback feedback annotation(
        Placement(visible = true, transformation(origin = {40, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-14, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = q_ref) annotation(
        Placement(visible = true, transformation(origin = {9, 70}, extent = {{-9, -10}, {9, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI PI(T = 5, k = 0.004) annotation(
        Placement(visible = true, transformation(origin = {72, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u annotation(
        Placement(visible = true, transformation(origin = {-110, 50}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-110, 50}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u1 annotation(
        Placement(visible = true, transformation(origin = {-110, -50}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-110, -50}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput y annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(feedback.y, PI.u) annotation(
        Line(points = {{49, 20}, {60, 20}}, color = {0, 0, 127}));
      connect(rootMeanSquare1.y, product.u2) annotation(
        Line(points = {{-39, -50}, {-39, -4}, {-26, -4}}, color = {0, 0, 127}));
      connect(product.y, feedback.u2) annotation(
        Line(points = {{-3, 2}, {40, 2}, {40, 12}}, color = {0, 0, 127}));
      connect(rootMeanSquare.y, product.u1) annotation(
        Line(points = {{-39, 50}, {-39, 42}, {-26, 42}, {-26, 8}}, color = {0, 0, 127}));
      connect(u1, rootMeanSquare1.u) annotation(
        Line(points = {{-110, -50}, {-62, -50}}, color = {0, 0, 127}));
      connect(u, rootMeanSquare.u) annotation(
        Line(points = {{-110, 50}, {-62, 50}, {-62, 50}, {-62, 50}}, color = {0, 0, 127}));
      connect(realExpression.y, feedback.u1) annotation(
        Line(points = {{18, 70}, {24, 70}, {24, 20}, {32, 20}, {32, 20}}, color = {0, 0, 127}));
      connect(PI.y, y) annotation(
        Line(points = {{84, 20}, {88, 20}, {88, 0}, {92, 0}, {92, 0}, {100, 0}}, color = {0, 0, 127}));
    end controller;

    model ctrl_l
      parameter SI.Power q_ref(start = 10);
      parameter SI.Inductance L_start(start = 0.005);
      SI.Inductance L;
      extends Modelica.Electrical.Analog.Interfaces.OnePort;
      Modelica.Blocks.Interfaces.RealInput L_Ctrl annotation(
        Placement(visible = true, transformation(origin = {0, 108}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {0, 108}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
    equation
      L = max(L_start - L_Ctrl, 0.0000001);
      L * der(i) = v;
      annotation(
        Documentation(info = "<html>
    <p>Reactive load, q_ref can be either set positiv (induvtiv load) or negativ (capacitive load).</p>
    </html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Text(extent = {{-150, -40}, {150, -80}}, textString = "R=%R"), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
    end ctrl_l;

    model controller_l
      parameter SI.Power q_ref;
      parameter Real L_start(start = L_start);
      Modelica.Blocks.Math.Feedback feedback annotation(
        Placement(visible = true, transformation(origin = {40, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-18, -6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = q_ref) annotation(
        Placement(visible = true, transformation(origin = {9, 70}, extent = {{-9, -10}, {9, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u annotation(
        Placement(visible = true, transformation(origin = {-108, 60}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-108, 60}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput f annotation(
        Placement(visible = true, transformation(origin = {-108, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-108, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput y annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-48, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product2 annotation(
        Placement(visible = true, transformation(origin = {-54, -6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression TwoPi(y = 6.283185307) annotation(
        Placement(visible = true, transformation(origin = {-90, -14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {20, -6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = L_start) annotation(
        Placement(visible = true, transformation(origin = {-88, -28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-52, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator integrator(k = 0.01) annotation(
        Placement(visible = true, transformation(origin = {72, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max annotation(
        Placement(visible = true, transformation(origin = {-16, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 0.0000001) annotation(
        Placement(visible = true, transformation(origin = {-50, -52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max1 annotation(
        Placement(visible = true, transformation(origin = {-44, 32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression3(y = 30) annotation(
        Placement(visible = true, transformation(origin = {-84, 36}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(realExpression.y, feedback.u1) annotation(
        Line(points = {{18, 70}, {24, 70}, {24, 20}, {32, 20}, {32, 20}}, color = {0, 0, 127}));
      connect(u, product1.u1) annotation(
        Line(points = {{-108, 60}, {-84, 60}, {-84, 66}, {-60, 66}}, color = {0, 0, 127}));
      connect(u, product1.u2) annotation(
        Line(points = {{-108, 60}, {-84, 60}, {-84, 54}, {-60, 54}}, color = {0, 0, 127}));
      connect(TwoPi.y, product2.u2) annotation(
        Line(points = {{-78, -14}, {-72, -14}, {-72, -12}, {-66, -12}, {-66, -12}}, color = {0, 0, 127}));
      connect(product2.y, product.u1) annotation(
        Line(points = {{-42, -6}, {-40, -6}, {-40, 0}, {-30, 0}, {-30, 0}}, color = {0, 0, 127}));
      connect(product.y, division.u2) annotation(
        Line(points = {{-6, -6}, {2, -6}, {2, -14}, {8, -14}, {8, -12}}, color = {0, 0, 127}));
      connect(product1.y, division.u1) annotation(
        Line(points = {{-36, 60}, {0, 60}, {0, 0}, {8, 0}, {8, 0}}, color = {0, 0, 127}));
      connect(division.y, feedback.u2) annotation(
        Line(points = {{32, -6}, {40, -6}, {40, 12}, {40, 12}}, color = {0, 0, 127}));
      connect(realExpression1.y, add.u1) annotation(
        Line(points = {{-76, -28}, {-68, -28}, {-68, -28}, {-66, -28}}, color = {0, 0, 127}));
      connect(feedback.y, integrator.u) annotation(
        Line(points = {{50, 20}, {60, 20}, {60, 20}, {60, 20}}, color = {0, 0, 127}));
      connect(integrator.y, y) annotation(
        Line(points = {{84, 20}, {88, 20}, {88, 0}, {92, 0}, {92, 0}, {100, 0}}, color = {0, 0, 127}));
      connect(integrator.y, add.u2) annotation(
        Line(points = {{84, 20}, {88, 20}, {88, -60}, {-72, -60}, {-72, -40}, {-64, -40}, {-64, -40}}, color = {0, 0, 127}));
      connect(add.y, max.u1) annotation(
        Line(points = {{-40, -34}, {-30, -34}, {-30, -34}, {-28, -34}}, color = {0, 0, 127}));
      connect(realExpression2.y, max.u2) annotation(
        Line(points = {{-38, -52}, {-34, -52}, {-34, -46}, {-30, -46}, {-30, -46}, {-28, -46}}, color = {0, 0, 127}));
      connect(max.y, product.u2) annotation(
        Line(points = {{-4, -40}, {4, -40}, {4, -18}, {-34, -18}, {-34, -12}, {-30, -12}, {-30, -12}}, color = {0, 0, 127}));
      connect(f, max1.u2) annotation(
        Line(points = {{-108, 0}, {-80, 0}, {-80, 26}, {-56, 26}, {-56, 26}}, color = {0, 0, 127}));
      connect(realExpression3.y, max1.u1) annotation(
        Line(points = {{-72, 36}, {-58, 36}, {-58, 38}, {-56, 38}}, color = {0, 0, 127}));
      connect(max1.y, product2.u1) annotation(
        Line(points = {{-32, 32}, {-26, 32}, {-26, 14}, {-72, 14}, {-72, 0}, {-66, 0}, {-66, 0}}, color = {0, 0, 127}));
    end controller_l;

    model ctrl_r
      parameter SI.Power p_ref(start = 2500);
      parameter SI.Resistance R_start(start = 20);
      SI.Resistance R;
      extends Modelica.Electrical.Analog.Interfaces.OnePort;
      Modelica.Blocks.Interfaces.RealInput R_ctrl annotation(
        Placement(visible = true, transformation(origin = {0, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {0, 110}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
    equation
      R = max(R_start - R_ctrl, 0.0000001);
      v = R * i;
      annotation(
        Documentation(info = "<html>
    <p>The linear resistor connects the branch voltage <em>v</em> with the branch current <em>i</em> by <em>i*R = v</em>. The Resistance <em>R</em> is allowed to be positive, zero, or negative.</p>
    </html>", revisions = "<html>
    <ul>
    <li><em> March 11, 2009   </em>
       by Christoph Clauss<br> conditional heat port added<br>
       </li>
    <li><em> 1998   </em>
       by Christoph Clauss<br> initially implemented<br>
       </li>
    </ul>
    </html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Rectangle(extent = {{-70, 30}, {70, -30}}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Line(points = {{-90, 0}, {-70, 0}}, color = {0, 0, 255}), Line(points = {{70, 0}, {90, 0}}, color = {0, 0, 255}), Text(extent = {{-150, -40}, {150, -80}}, textString = "R=%R"), Line(visible = useHeatPort, points = {{0, -100}, {0, -30}}, color = {127, 0, 0}, pattern = LinePattern.Dot), Text(extent = {{-150, 90}, {150, 50}}, textString = "%name", lineColor = {0, 0, 255})}));
    end ctrl_r;

    model controller_r
      parameter SI.Power p_ref;
      parameter Real R_start(start = R_start);
      Modelica.Blocks.Math.Feedback feedback annotation(
        Placement(visible = true, transformation(origin = {40, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = p_ref) annotation(
        Placement(visible = true, transformation(origin = {9, 70}, extent = {{-9, -10}, {9, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u annotation(
        Placement(visible = true, transformation(origin = {-108, 60}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-108, 60}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput y annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-48, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {20, -6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = R_start) annotation(
        Placement(visible = true, transformation(origin = {-88, -28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-52, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator integrator(k = 1) annotation(
        Placement(visible = true, transformation(origin = {72, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max annotation(
        Placement(visible = true, transformation(origin = {-16, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 0.0000001) annotation(
        Placement(visible = true, transformation(origin = {-50, -52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(realExpression.y, feedback.u1) annotation(
        Line(points = {{18, 70}, {24, 70}, {24, 20}, {32, 20}, {32, 20}}, color = {0, 0, 127}));
      connect(u, product1.u1) annotation(
        Line(points = {{-108, 60}, {-84, 60}, {-84, 66}, {-60, 66}}, color = {0, 0, 127}));
      connect(u, product1.u2) annotation(
        Line(points = {{-108, 60}, {-84, 60}, {-84, 54}, {-60, 54}}, color = {0, 0, 127}));
      connect(product1.y, division.u1) annotation(
        Line(points = {{-36, 60}, {0, 60}, {0, 0}, {8, 0}, {8, 0}}, color = {0, 0, 127}));
      connect(division.y, feedback.u2) annotation(
        Line(points = {{32, -6}, {40, -6}, {40, 12}, {40, 12}}, color = {0, 0, 127}));
      connect(realExpression1.y, add.u1) annotation(
        Line(points = {{-76, -28}, {-68, -28}, {-68, -28}, {-66, -28}}, color = {0, 0, 127}));
      connect(feedback.y, integrator.u) annotation(
        Line(points = {{50, 20}, {60, 20}, {60, 20}, {60, 20}}, color = {0, 0, 127}));
      connect(integrator.y, y) annotation(
        Line(points = {{84, 20}, {88, 20}, {88, 0}, {92, 0}, {92, 0}, {100, 0}}, color = {0, 0, 127}));
      connect(integrator.y, add.u2) annotation(
        Line(points = {{84, 20}, {88, 20}, {88, -60}, {-72, -60}, {-72, -40}, {-64, -40}, {-64, -40}}, color = {0, 0, 127}));
      connect(add.y, max.u1) annotation(
        Line(points = {{-40, -34}, {-30, -34}, {-30, -34}, {-28, -34}}, color = {0, 0, 127}));
      connect(realExpression2.y, max.u2) annotation(
        Line(points = {{-38, -52}, {-34, -52}, {-34, -46}, {-30, -46}, {-30, -46}, {-28, -46}}, color = {0, 0, 127}));
      connect(max.y, division.u2) annotation(
        Line(points = {{-4, -40}, {0, -40}, {0, -12}, {8, -12}, {8, -12}}, color = {0, 0, 127}));
    end controller_r;
  end components;

  package plls
    model pll
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 150) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {48, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {64, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {78, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput f annotation(
        Placement(visible = true, transformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput theta annotation(
        Placement(visible = true, transformation(origin = {106, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, ground.p) annotation(
        Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, ground.p) annotation(
        Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_c.n, ground.p) annotation(
        Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{43, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {43, 20}}, color = {0, 0, 127}));
      connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
        Line(points = {{59, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{74, 22}, {68, 22}, {68, 22}, {68, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -6}, {-6, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {82, 22}, {82, 22}, {82, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, theta) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -60}, {106, -60}}, color = {0, 0, 127}));
      connect(f, add_freq_nom_delta_f.y) annotation(
        Line(points = {{106, 60}, {56, 60}, {56, 22}, {52, 22}, {52, 22}}, color = {0, 0, 127}));
    end pll;

    model pll_d
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-52, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-8, 56}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-8, 46}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-31, 83}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-31, 69}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-5, 83}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-7, 69}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {14, 78}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.00005, k = 25) annotation(
        Placement(visible = true, transformation(origin = {28, 78}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {50, 76}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {30, 58}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {66, 76}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {80, 76}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-6, 8}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-64, 40}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-63, 17}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Product product2 annotation(
        Placement(visible = true, transformation(origin = {38, 34}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos2 annotation(
        Placement(visible = true, transformation(origin = {14, 30}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {64, 28}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {32, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {44, -14}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {2, -18}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos1 annotation(
        Placement(visible = true, transformation(origin = {22, -18}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 4 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-19, -22}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-63, -9}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput d annotation(
        Placement(visible = true, transformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-29, 4}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos3 annotation(
        Placement(visible = true, transformation(origin = {14, 6}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput theta annotation(
        Placement(visible = true, transformation(origin = {110, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product3 annotation(
        Placement(visible = true, transformation(origin = {44, -84}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product5 annotation(
        Placement(visible = true, transformation(origin = {38, -36}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-63, -53}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {2, -88}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 2 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-29, -66}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin1 annotation(
        Placement(visible = true, transformation(origin = {15, -65}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain4(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-64, -30}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain5(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-63, -79}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum1(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {64, -42}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput q annotation(
        Placement(visible = true, transformation(origin = {110, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin2 annotation(
        Placement(visible = true, transformation(origin = {9, -43}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-19, -92}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin3 annotation(
        Placement(visible = true, transformation(origin = {23, -89}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Add add4(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-8, -64}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product4 annotation(
        Placement(visible = true, transformation(origin = {32, -60}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, ground.p) annotation(
        Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, ground.p) annotation(
        Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_c.n, ground.p) annotation(
        Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-35, 83}, {-42, 83}, {-42, 84}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-28, 83}, {-18.5, 83}, {-18.5, 85}, {-9, 85}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-28, 69}, {-19.5, 69}, {-19.5, 71}, {-11, 71}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-12, 56}, {-20, 56}, {-20, 81}, {-9, 81}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-12, 46}, {-16, 46}, {-16, 67}, {-11, 67}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{9, 80}, {-0.25, 80}, {-0.25, 83}, {-2, 83}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-4, 69}, {-1, 69}, {-1, 76}, {9, 76}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{21, 78}, {18, 78}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{45, 78}, {35, 78}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{75, 76}, {70, 76}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-3, 46}, {92, 46}, {92, 76}, {84, 76}}, color = {0, 0, 127}));
      connect(gain.y, product2.u1) annotation(
        Line(points = {{-57, 40}, {-13, 40}, {-13, 38}, {31, 38}}, color = {0, 0, 127}));
      connect(cos3.y, product1.u2) annotation(
        Line(points = {{21, 6}, {25, 6}}, color = {0, 0, 127}));
      connect(cos1.y, product.u2) annotation(
        Line(points = {{29, -18}, {37, -18}}, color = {0, 0, 127}));
      connect(gain1.y, product1.u1) annotation(
        Line(points = {{-55, 17}, {25, 17}, {25, 14}}, color = {0, 0, 127}));
      connect(realExpression.y, add1.u2) annotation(
        Line(points = {{-21, 4}, {-13, 4}}, color = {0, 0, 127}));
      connect(product2.y, multiSum.u[3]) annotation(
        Line(points = {{45, 34}, {55.5, 34}, {55.5, 28}, {54, 28}}, color = {0, 0, 127}));
      connect(product.y, multiSum.u[1]) annotation(
        Line(points = {{51, -14}, {54, -14}, {54, 28}}, color = {0, 0, 127}));
      connect(add1.y, cos3.u) annotation(
        Line(points = {{1, 8}, {4, 8}, {4, 6}, {7, 6}}, color = {0, 0, 127}));
      connect(add2.y, cos1.u) annotation(
        Line(points = {{9, -18}, {15, -18}}, color = {0, 0, 127}));
      connect(multiSum.y, d) annotation(
        Line(points = {{76, 28}, {95, 28}, {95, 0}, {110, 0}}, color = {0, 0, 127}));
      connect(product1.y, multiSum.u[2]) annotation(
        Line(points = {{39, 10}, {54, 10}, {54, 28}}, color = {0, 0, 127}));
      connect(cos2.y, product2.u2) annotation(
        Line(points = {{21, 30}, {31, 30}}, color = {0, 0, 127}));
      connect(realExpression1.y, add2.u2) annotation(
        Line(points = {{-11, -22}, {-5, -22}}, color = {0, 0, 127}));
      connect(gain2.y, product.u1) annotation(
        Line(points = {{-55, -9}, {37, -9}, {37, -10}}, color = {0, 0, 127}));
      connect(voltageSensor_a.v, gain.u) annotation(
        Line(points = {{-80, 50}, {-74, 50}, {-74, 40}, {-71, 40}}, color = {0, 0, 127}));
      connect(deg2rad.y, cos2.u) annotation(
        Line(points = {{84, 76}, {92, 76}, {92, 46}, {7, 46}, {7, 30}}, color = {0, 0, 127}));
      connect(deg2rad.y, add1.u1) annotation(
        Line(points = {{84, 76}, {92, 76}, {92, 46}, {6, 46}, {6, 24}, {-17, 24}, {-17, 12}, {-13, 12}}, color = {0, 0, 127}));
      connect(deg2rad.y, add2.u1) annotation(
        Line(points = {{84, 76}, {92, 76}, {92, 46}, {6, 46}, {6, 24}, {-16, 24}, {-16, -14}, {-5, -14}}, color = {0, 0, 127}));
      connect(sin.u, deg2rad.y) annotation(
        Line(points = {{-4, 56}, {0, 56}, {0, 46}, {92, 46}, {92, 76}, {84, 76}}, color = {0, 0, 127}));
      connect(voltageSensor_c.v, gain2.u) annotation(
        Line(points = {{-82, -8}, {-71, -8}, {-71, -9}}, color = {0, 0, 127}));
      connect(realExpression2.y, add4.u2) annotation(
        Line(points = {{-21, -66}, {-18, -66}, {-18, -68}, {-15, -68}}, color = {0, 0, 127}));
      connect(gain4.y, product5.u1) annotation(
        Line(points = {{-57, -30}, {-13, -30}, {-13, -32}, {31, -32}}, color = {0, 0, 127}));
      connect(gain3.y, product4.u1) annotation(
        Line(points = {{-55, -53}, {25, -53}, {25, -56}}, color = {0, 0, 127}));
      connect(sin1.y, product4.u2) annotation(
        Line(points = {{23, -65}, {23, -60.5}, {25, -60.5}, {25, -64}}, color = {0, 0, 127}));
      connect(realExpression3.y, add3.u2) annotation(
        Line(points = {{-11, -92}, {-5, -92}}, color = {0, 0, 127}));
      connect(sin2.y, product5.u2) annotation(
        Line(points = {{17, -43}, {28, -43}, {28, -40}, {31, -40}}, color = {0, 0, 127}));
      connect(product3.y, multiSum1.u[1]) annotation(
        Line(points = {{51, -84}, {54, -84}, {54, -42}}, color = {0, 0, 127}));
      connect(gain5.y, product3.u1) annotation(
        Line(points = {{-55, -79}, {37, -79}, {37, -80}}, color = {0, 0, 127}));
      connect(product5.y, multiSum1.u[3]) annotation(
        Line(points = {{45, -36}, {75.5, -36}, {75.5, -42}, {54, -42}}, color = {0, 0, 127}));
      connect(add3.y, sin3.u) annotation(
        Line(points = {{9, -88}, {26, -88}, {26, -89}, {15, -89}}, color = {0, 0, 127}));
      connect(sin1.u, add4.y) annotation(
        Line(points = {{7, -65}, {16, -65}, {16, -64}, {-1, -64}}, color = {0, 0, 127}));
      connect(product4.y, multiSum1.u[2]) annotation(
        Line(points = {{39, -60}, {54, -60}, {54, -42}}, color = {0, 0, 127}));
      connect(sin3.y, product3.u2) annotation(
        Line(points = {{31, -89}, {31, -88.5}, {37, -88.5}, {37, -88}}, color = {0, 0, 127}));
      connect(multiSum1.y, q) annotation(
        Line(points = {{76, -42}, {93, -42}, {93, -60}, {110, -60}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, Norm_U_ref_beta.u) annotation(
        Line(points = {{-42, 74}, {-38, 74}, {-38, 68}, {-34, 68}, {-34, 70}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin2.u) annotation(
        Line(points = {{84, 76}, {92, 76}, {92, 46}, {6, 46}, {6, 24}, {-44, 24}, {-44, -44}, {0, -44}, {0, -42}}, color = {0, 0, 127}));
      connect(voltageSensor_a.v, gain4.u) annotation(
        Line(points = {{-80, 50}, {-74, 50}, {-74, -30}, {-72, -30}, {-72, -30}}, color = {0, 0, 127}));
      connect(voltageSensor_b.v, gain3.u) annotation(
        Line(points = {{-81, 22}, {-76, 22}, {-76, -54}, {-72, -54}, {-72, -52}}, color = {0, 0, 127}));
      connect(add4.u1, deg2rad.y) annotation(
        Line(points = {{-16, -60}, {-44, -60}, {-44, 24}, {6, 24}, {6, 46}, {92, 46}, {92, 76}, {84, 76}, {84, 76}}, color = {0, 0, 127}));
      connect(add3.u1, deg2rad.y) annotation(
        Line(points = {{-6, -84}, {-44, -84}, {-44, 24}, {6, 24}, {6, 46}, {92, 46}, {92, 76}, {84, 76}, {84, 76}}, color = {0, 0, 127}));
      connect(gain5.u, voltageSensor_c.v) annotation(
        Line(points = {{-72, -78}, {-80, -78}, {-80, -8}, {-82, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(voltageSensor_b.v, gain1.u) annotation(
        Line(points = {{-82, 22}, {-76, 22}, {-76, 16}, {-72, 16}, {-72, 18}}, color = {0, 0, 127}));
      connect(voltageSensor_c.v, abc2AlphaBeta.c) annotation(
        Line(points = {{-82, -8}, {-78, -8}, {-78, 76}, {-62, 76}}, color = {0, 0, 127}));
      connect(voltageSensor_b.v, abc2AlphaBeta.b) annotation(
        Line(points = {{-82, 22}, {-76, 22}, {-76, 78}, {-62, 78}, {-62, 80}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-62, 82}, {-80, 82}, {-80, 50}, {-80, 50}}, color = {0, 0, 127}));
  connect(deg2rad.y, theta) annotation(
        Line(points = {{84, 76}, {92, 76}, {92, 60}, {110, 60}, {110, 60}}, color = {0, 0, 127}));
  connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{34, 58}, {37.5, 58}, {37.5, 74}, {45, 74}}, color = {0, 0, 127}));
  connect(add_freq_nom_delta_f.y, f2theta.u) annotation(
        Line(points = {{54, 76}, {60, 76}, {60, 76}, {62, 76}}, color = {0, 0, 127}));
    end pll_d;

    model pll_serial
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 150) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {48, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {64, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {78, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput d annotation(
        Placement(visible = true, transformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealOutput freq annotation(
        Placement(visible = true, transformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-70, -26}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-69, -49}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-69, -75}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 4 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-25, -88}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-35, -62}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-12, -58}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-4, -84}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos2 annotation(
        Placement(visible = true, transformation(origin = {8, -36}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {38, -80}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos1 annotation(
        Placement(visible = true, transformation(origin = {16, -84}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos3 annotation(
        Placement(visible = true, transformation(origin = {8, -60}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product2 annotation(
        Placement(visible = true, transformation(origin = {32, -32}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {26, -56}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {58, -38}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{43, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {43, 20}}, color = {0, 0, 127}));
      connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
        Line(points = {{59, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{74, 22}, {68, 22}, {68, 22}, {68, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -6}, {-6, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {82, 22}, {82, 22}, {82, 22}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.y, freq) annotation(
        Line(points = {{52, 22}, {56, 22}, {56, 36}, {86, 36}, {86, -96}, {40, -96}, {40, -110}}, color = {0, 0, 127}));
      connect(voltageSensor_c.n, pin1) annotation(
        Line(points = {{-88, -2}, {-78, -2}, {-78, 38}, {88, 38}, {88, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, pin2) annotation(
        Line(points = {{-88, 28}, {-80, 28}, {-80, 40}, {90, 40}, {90, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, pin3) annotation(
        Line(points = {{-86, 56}, {-86, 56}, {-86, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(voltageSensor_a.v, gain.u) annotation(
        Line(points = {{-80, 50}, {-56, 50}, {-56, -18}, {-78, -18}, {-78, -26}, {-78, -26}}, color = {0, 0, 127}));
      connect(voltageSensor_b.v, gain1.u) annotation(
        Line(points = {{-82, 22}, {-62, 22}, {-62, -14}, {-82, -14}, {-82, -48}, {-78, -48}, {-78, -48}}, color = {0, 0, 127}));
      connect(voltageSensor_c.v, gain2.u) annotation(
        Line(points = {{-82, -8}, {-82, -8}, {-82, -10}, {-84, -10}, {-84, -76}, {-78, -76}, {-78, -74}}, color = {0, 0, 127}));
      connect(realExpression.y, add1.u2) annotation(
        Line(points = {{-27, -62}, {-19, -62}}, color = {0, 0, 127}));
      connect(deg2rad.y, add1.u1) annotation(
        Line(points = {{82, 22}, {84, 22}, {84, -6}, {4, -6}, {4, -18}, {-2, -18}, {-2, -42}, {-22, -42}, {-22, -54}, {-20, -54}, {-20, -54}}, color = {0, 0, 127}));
      connect(deg2rad.y, add2.u1) annotation(
        Line(points = {{82, 22}, {84, 22}, {84, -6}, {4, -6}, {4, -18}, {-2, -18}, {-2, -42}, {-22, -42}, {-22, -80}, {-12, -80}}, color = {0, 0, 127}));
      connect(realExpression1.y, add2.u2) annotation(
        Line(points = {{-17, -88}, {-11, -88}}, color = {0, 0, 127}));
      connect(deg2rad.y, cos2.u) annotation(
        Line(points = {{82, 22}, {84, 22}, {84, -6}, {4, -6}, {4, -18}, {-2, -18}, {-2, -36}, {0, -36}, {0, -36}}, color = {0, 0, 127}));
      connect(gain2.y, product.u1) annotation(
        Line(points = {{-61, -75}, {31, -75}, {31, -76}}, color = {0, 0, 127}));
      connect(add2.y, cos1.u) annotation(
        Line(points = {{3, -84}, {9, -84}}, color = {0, 0, 127}));
      connect(cos1.y, product.u2) annotation(
        Line(points = {{23, -84}, {31, -84}}, color = {0, 0, 127}));
      connect(add1.y, cos3.u) annotation(
        Line(points = {{-5, -58}, {-4, -58}, {-4, -60}, {1, -60}}, color = {0, 0, 127}));
      connect(cos2.y, product2.u2) annotation(
        Line(points = {{15, -36}, {25, -36}}, color = {0, 0, 127}));
      connect(gain.y, product2.u1) annotation(
        Line(points = {{-63, -26}, {-21, -26}, {-21, -28}, {25, -28}}, color = {0, 0, 127}));
      connect(gain1.y, product1.u1) annotation(
        Line(points = {{-61, -49}, {19, -49}, {19, -52}}, color = {0, 0, 127}));
      connect(cos3.y, product1.u2) annotation(
        Line(points = {{15, -60}, {19, -60}}, color = {0, 0, 127}));
      connect(multiSum.y, d) annotation(
        Line(points = {{70, -38}, {78, -38}, {78, -94}, {-40, -94}, {-40, -110}, {-40, -110}}, color = {0, 0, 127}));
      connect(product1.y, multiSum.u[1]) annotation(
        Line(points = {{33, -56}, {48, -56}, {48, -38}}, color = {0, 0, 127}));
      connect(product.y, multiSum.u[2]) annotation(
        Line(points = {{45, -80}, {48, -80}, {48, -38}}, color = {0, 0, 127}));
      connect(product2.y, multiSum.u[3]) annotation(
        Line(points = {{39, -32}, {61.5, -32}, {61.5, -38}, {48, -38}}, color = {0, 0, 127}));
    end pll_serial;

    model pll_alpha
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput u_eff annotation(
        Placement(visible = true, transformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-34, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-34, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {26, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add add annotation(
        Placement(visible = true, transformation(origin = {-6, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {28, -2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {58, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.Mean mean(f = 50) annotation(
        Placement(visible = true, transformation(origin = {86, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, ground.p) annotation(
        Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, ground.p) annotation(
        Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_c.n, ground.p) annotation(
        Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u1) annotation(
        Line(points = {{-52, 26}, {-46, 26}, {-46, 36}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u2) annotation(
        Line(points = {{-52, 26}, {-46, 26}, {-46, 24}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u1) annotation(
        Line(points = {{-52, 16}, {-46, 16}, {-46, 12}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u2) annotation(
        Line(points = {{-52, 16}, {-46, 16}, {-46, 0}}, color = {0, 0, 127}));
      connect(product.y, add.u1) annotation(
        Line(points = {{-23, 30}, {-18, 30}, {-18, 26}}, color = {0, 0, 127}));
      connect(product1.y, add.u2) annotation(
        Line(points = {{-23, 6}, {-18, 6}, {-18, 14}}, color = {0, 0, 127}));
      connect(add.y, sqrt1.u) annotation(
        Line(points = {{5, 20}, {14, 20}}, color = {0, 0, 127}));
      connect(realExpression.y, division.u2) annotation(
        Line(points = {{39, -2}, {39, 1}, {41, 1}, {41, 4}, {46, 4}}, color = {0, 0, 127}));
      connect(sqrt1.y, division.u1) annotation(
        Line(points = {{37, 20}, {42, 20}, {42, 16}, {46, 16}}, color = {0, 0, 127}));
      connect(division.y, mean.u) annotation(
        Line(points = {{70, 10}, {72, 10}, {72, 10}, {74, 10}}, color = {0, 0, 127}));
      connect(mean.y, u_eff) annotation(
        Line(points = {{98, 10}, {98, 10}, {98, 0}, {102, 0}, {102, 0}, {110, 0}}, color = {0, 0, 127}));
    end pll_alpha;

    model pll_alpha_serial
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput u_eff annotation(
        Placement(visible = true, transformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-64, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {-4, -56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {-2, -78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.Mean mean(f = 50) annotation(
        Placement(visible = true, transformation(origin = {56, -66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1 annotation(
        Placement(visible = true, transformation(origin = {-36, -56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-64, -46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {28, -66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(voltageSensor_c.n, pin1) annotation(
        Line(points = {{-88, -2}, {-78, -2}, {-78, 38}, {88, 38}, {88, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, pin2) annotation(
        Line(points = {{-88, 28}, {-80, 28}, {-80, 40}, {90, 40}, {90, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, pin3) annotation(
        Line(points = {{-86, 56}, {-86, 56}, {-86, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(product1.y, add1.u2) annotation(
        Line(points = {{-53, -70}, {-48, -70}, {-48, -62}}, color = {0, 0, 127}));
      connect(product.y, add1.u1) annotation(
        Line(points = {{-53, -46}, {-48, -46}, {-48, -50}}, color = {0, 0, 127}));
      connect(sqrt1.y, division.u1) annotation(
        Line(points = {{7, -56}, {16, -56}, {16, -60}}, color = {0, 0, 127}));
      connect(realExpression.y, division.u2) annotation(
        Line(points = {{9, -78}, {9, -77}, {11, -77}, {11, -72}, {16, -72}}, color = {0, 0, 127}));
      connect(division.y, mean.u) annotation(
        Line(points = {{39, -66}, {44, -66}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u1) annotation(
        Line(points = {{-52, 26}, {-48, 26}, {-48, -28}, {-80, -28}, {-80, -40}, {-76, -40}, {-76, -40}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u2) annotation(
        Line(points = {{-52, 26}, {-48, 26}, {-48, -28}, {-80, -28}, {-80, -52}, {-76, -52}, {-76, -52}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u1) annotation(
        Line(points = {{-52, 16}, {-50, 16}, {-50, -26}, {-82, -26}, {-82, -64}, {-76, -64}, {-76, -64}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u2) annotation(
        Line(points = {{-52, 16}, {-50, 16}, {-50, -26}, {-82, -26}, {-82, -76}, {-76, -76}, {-76, -76}}, color = {0, 0, 127}));
      connect(mean.y, u_eff) annotation(
        Line(points = {{68, -66}, {72, -66}, {72, -92}, {-40, -92}, {-40, -110}, {-40, -110}}, color = {0, 0, 127}));
      connect(add1.y, sqrt1.u) annotation(
        Line(points = {{-25, -56}, {-16, -56}}, color = {0, 0, 127}));
    end pll_alpha_serial;

    model pll_alpha_f_serial
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {48, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {64, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {78, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput u_eff annotation(
        Placement(visible = true, transformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealOutput freq annotation(
        Placement(visible = true, transformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-64, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {-4, -56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 1.41421356) annotation(
        Placement(visible = true, transformation(origin = {-2, -78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      grid.components.Mean mean(f = 50) annotation(
        Placement(visible = true, transformation(origin = {56, -66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add addition annotation(
        Placement(visible = true, transformation(origin = {-36, -56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-64, -46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Division division annotation(
        Placement(visible = true, transformation(origin = {28, -66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 150) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {43, 20}}, color = {0, 0, 127}));
      connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
        Line(points = {{59, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{74, 22}, {68, 22}, {68, 22}, {68, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -6}, {-6, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {82, 22}, {82, 22}, {82, 22}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.y, freq) annotation(
        Line(points = {{52, 22}, {56, 22}, {56, 36}, {86, 36}, {86, -96}, {40, -96}, {40, -110}}, color = {0, 0, 127}));
      connect(voltageSensor_c.n, pin1) annotation(
        Line(points = {{-88, -2}, {-78, -2}, {-78, 38}, {88, 38}, {88, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, pin2) annotation(
        Line(points = {{-88, 28}, {-80, 28}, {-80, 40}, {90, 40}, {90, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, pin3) annotation(
        Line(points = {{-86, 56}, {-86, 56}, {-86, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(product1.y, addition.u2) annotation(
        Line(points = {{-53, -70}, {-48, -70}, {-48, -62}}, color = {0, 0, 127}));
      connect(product.y, addition.u1) annotation(
        Line(points = {{-53, -46}, {-48, -46}, {-48, -50}}, color = {0, 0, 127}));
      connect(sqrt1.y, division.u1) annotation(
        Line(points = {{7, -56}, {16, -56}, {16, -60}}, color = {0, 0, 127}));
      connect(realExpression.y, division.u2) annotation(
        Line(points = {{9, -78}, {9, -77}, {11, -77}, {11, -72}, {16, -72}}, color = {0, 0, 127}));
      connect(division.y, mean.u) annotation(
        Line(points = {{39, -66}, {44, -66}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u1) annotation(
        Line(points = {{-52, 26}, {-48, 26}, {-48, -28}, {-80, -28}, {-80, -40}, {-76, -40}, {-76, -40}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u2) annotation(
        Line(points = {{-52, 26}, {-48, 26}, {-48, -28}, {-80, -28}, {-80, -52}, {-76, -52}, {-76, -52}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u1) annotation(
        Line(points = {{-52, 16}, {-50, 16}, {-50, -26}, {-82, -26}, {-82, -64}, {-76, -64}, {-76, -64}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u2) annotation(
        Line(points = {{-52, 16}, {-50, 16}, {-50, -26}, {-82, -26}, {-82, -76}, {-76, -76}, {-76, -76}}, color = {0, 0, 127}));
      connect(mean.y, u_eff) annotation(
        Line(points = {{68, -66}, {72, -66}, {72, -92}, {-40, -92}, {-40, -110}, {-40, -110}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{43, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(addition.y, sqrt1.u) annotation(
        Line(points = {{-25, -56}, {-16, -56}}, color = {0, 0, 127}));
    end pll_alpha_f_serial;

    model pll_alpha_f_serial_test
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {48, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {64, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {78, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput u_eff annotation(
        Placement(visible = true, transformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealOutput freq annotation(
        Placement(visible = true, transformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 150) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-18, -68}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add addition annotation(
        Placement(visible = true, transformation(origin = {10, -54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-18, -44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {40, -54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 0.7071067811865) annotation(
        Placement(visible = true, transformation(origin = {70, -54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {43, 20}}, color = {0, 0, 127}));
      connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
        Line(points = {{59, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{74, 22}, {68, 22}, {68, 22}, {68, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -6}, {-6, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {82, 22}, {82, 22}, {82, 22}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.y, freq) annotation(
        Line(points = {{52, 22}, {56, 22}, {56, 36}, {86, 36}, {86, -96}, {40, -96}, {40, -110}}, color = {0, 0, 127}));
      connect(voltageSensor_c.n, pin1) annotation(
        Line(points = {{-88, -2}, {-78, -2}, {-78, 38}, {88, 38}, {88, -60}, {100, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, pin2) annotation(
        Line(points = {{-88, 28}, {-80, 28}, {-80, 40}, {90, 40}, {90, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, pin3) annotation(
        Line(points = {{-86, 56}, {-86, 56}, {-86, 60}, {100, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{43, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(product.y, addition.u1) annotation(
        Line(points = {{-7, -44}, {-2, -44}, {-2, -48}}, color = {0, 0, 127}));
      connect(product1.y, addition.u2) annotation(
        Line(points = {{-7, -68}, {-2, -68}, {-2, -60}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u1) annotation(
        Line(points = {{-52, 26}, {-40, 26}, {-40, -38}, {-30, -38}, {-30, -38}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.alpha, product.u2) annotation(
        Line(points = {{-52, 26}, {-40, 26}, {-40, -50}, {-30, -50}, {-30, -50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u1) annotation(
        Line(points = {{-52, 16}, {-46, 16}, {-46, -62}, {-32, -62}, {-32, -62}, {-30, -62}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u2) annotation(
        Line(points = {{-52, 16}, {-46, 16}, {-46, -74}, {-30, -74}, {-30, -74}}, color = {0, 0, 127}));
      connect(addition.y, sqrt1.u) annotation(
        Line(points = {{22, -54}, {26, -54}, {26, -54}, {28, -54}}, color = {0, 0, 127}));
      connect(sqrt1.y, gain.u) annotation(
        Line(points = {{52, -54}, {58, -54}, {58, -54}, {58, -54}}, color = {0, 0, 127}));
      connect(gain.y, u_eff) annotation(
        Line(points = {{82, -54}, {84, -54}, {84, -84}, {-40, -84}, {-40, -110}}, color = {0, 0, 127}));
    end pll_alpha_f_serial_test;

    model pll_ueff
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 150) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {48, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {64, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {78, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput f annotation(
        Placement(visible = true, transformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput theta annotation(
        Placement(visible = true, transformation(origin = {108, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {108, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 0.7071067811865) annotation(
        Placement(visible = true, transformation(origin = {64, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {-24, -74}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput u_eff annotation(
        Placement(visible = true, transformation(origin = {108, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {108, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Max max annotation(
        Placement(visible = true, transformation(origin = {84, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 1) annotation(
        Placement(visible = true, transformation(origin = {58, -16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sqrt sqrt1 annotation(
        Placement(visible = true, transformation(origin = {32, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {-24, -46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1 annotation(
        Placement(visible = true, transformation(origin = {4, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth(f = 70) annotation(
        Placement(visible = true, transformation(origin = {56, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, ground.p) annotation(
        Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, ground.p) annotation(
        Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_c.n, ground.p) annotation(
        Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{43, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {43, 20}}, color = {0, 0, 127}));
      connect(f2theta.u, add_freq_nom_delta_f.y) annotation(
        Line(points = {{59, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{74, 22}, {68, 22}, {68, 22}, {68, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, -6}, {-6, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {82, 22}, {82, 22}, {82, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, theta) annotation(
        Line(points = {{82, 22}, {92, 22}, {92, 0}, {108, 0}}, color = {0, 0, 127}));
      connect(f, add_freq_nom_delta_f.y) annotation(
        Line(points = {{106, 60}, {56, 60}, {56, 22}, {52, 22}, {52, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u1) annotation(
        Line(points = {{-52, 16}, {-46, 16}, {-46, -68}, {-36, -68}, {-36, -68}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.beta, product1.u2) annotation(
        Line(points = {{-52, 16}, {-46, 16}, {-46, -80}, {-36, -80}, {-36, -80}}, color = {0, 0, 127}));
      connect(realExpression.y, max.u1) annotation(
        Line(points = {{69, -16}, {72, -16}, {72, -20}}, color = {0, 0, 127}));
      connect(sqrt1.y, gain.u) annotation(
        Line(points = {{44, -60}, {52, -60}, {52, -60}, {52, -60}}, color = {0, 0, 127}));
      connect(product.u1, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-36, -40}, {-44, -40}, {-44, 26}, {-52, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(product.u2, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-36, -52}, {-44, -52}, {-44, 26}, {-52, 26}, {-52, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(product1.y, add1.u2) annotation(
        Line(points = {{-12, -74}, {-10, -74}, {-10, -66}, {-8, -66}, {-8, -66}}, color = {0, 0, 127}));
      connect(product.y, add1.u1) annotation(
        Line(points = {{-12, -46}, {-10, -46}, {-10, -54}, {-8, -54}, {-8, -54}, {-8, -54}}, color = {0, 0, 127}));
      connect(add1.y, sqrt1.u) annotation(
        Line(points = {{16, -60}, {20, -60}, {20, -60}, {20, -60}}, color = {0, 0, 127}));
      connect(gain.y, lowpassButterworth.u) annotation(
        Line(points = {{76, -60}, {84, -60}, {84, -48}, {40, -48}, {40, -34}, {44, -34}, {44, -34}}, color = {0, 0, 127}));
      connect(lowpassButterworth.y, max.u2) annotation(
        Line(points = {{68, -34}, {70, -34}, {70, -32}, {72, -32}}, color = {0, 0, 127}));
      connect(max.y, u_eff) annotation(
        Line(points = {{96, -26}, {96, -26}, {96, -60}, {108, -60}, {108, -60}}, color = {0, 0, 127}));
    end pll_ueff;

    model pll_ueff_test
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Basic.Ground ground annotation(
        Placement(visible = true, transformation(origin = {-86, 62}, extent = {{-6, -6}, {6, 6}}, rotation = 180)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_c annotation(
        Placement(visible = true, transformation(origin = {-88, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_a annotation(
        Placement(visible = true, transformation(origin = {-86, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor_b annotation(
        Placement(visible = true, transformation(origin = {-88, 22}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      grid.transforms.abc2AlphaBeta abc2AlphaBeta annotation(
        Placement(visible = true, transformation(origin = {-62, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {-10, -6}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {-10, -18}, extent = {{-4, -4}, {4, 4}}, rotation = 180)));
      Modelica.Blocks.Math.Gain Norm_U_ref_alpha(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Gain Norm_U_ref_beta(k = 1 / (230 * 1.414)) annotation(
        Placement(visible = true, transformation(origin = {-33, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product alphaSin annotation(
        Placement(visible = true, transformation(origin = {-7, 29}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Product betaCos annotation(
        Placement(visible = true, transformation(origin = {-9, 15}, extent = {{-3, -3}, {3, 3}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {12, 24}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.PI pi(T = 0.2, k = 150) annotation(
        Placement(visible = true, transformation(origin = {26, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add_freq_nom_delta_f(k1 = +1, k2 = +1) annotation(
        Placement(visible = true, transformation(origin = {42, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Sources.Constant f_nom(k = 50) annotation(
        Placement(visible = true, transformation(origin = {28, 4}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator f2theta(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {70, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Math.Gain deg2rad(k = 2 * 3.1416) annotation(
        Placement(visible = true, transformation(origin = {84, 22}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput theta annotation(
        Placement(visible = true, transformation(origin = {108, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {108, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth(f = 70) annotation(
        Placement(visible = true, transformation(origin = {62, 54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth1(f = 70) annotation(
        Placement(visible = true, transformation(origin = {-36, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.Integrator integrator(y_start = 0) annotation(
        Placement(visible = true, transformation(origin = {84, 54}, extent = {{-4, -4}, {4, 4}}, rotation = 0)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth2(f = 70) annotation(
        Placement(visible = true, transformation(origin = {-60, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Continuous.LowpassButterworth lowpassButterworth3(f = 70) annotation(
        Placement(visible = true, transformation(origin = {-60, -56}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, voltageSensor_a.p) annotation(
        Line(points = {{-100, 60}, {-93, 60}, {-93, 44}, {-86, 44}}, color = {0, 0, 255}));
      connect(b, voltageSensor_b.p) annotation(
        Line(points = {{-102, 0}, {-94, 0}, {-94, 16}, {-88, 16}}, color = {0, 0, 255}));
      connect(c, voltageSensor_c.p) annotation(
        Line(points = {{-102, -60}, {-94, -60}, {-94, -14}, {-88, -14}}, color = {0, 0, 255}));
      connect(voltageSensor_a.n, ground.p) annotation(
        Line(points = {{-86, 56}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_b.n, ground.p) annotation(
        Line(points = {{-88, 28}, {-88, 42}, {-86, 42}, {-86, 56}}, color = {0, 0, 255}));
      connect(voltageSensor_c.n, ground.p) annotation(
        Line(points = {{-88, -2}, {-88, 27}, {-86, 27}, {-86, 56}}, color = {0, 0, 255}));
      connect(abc2AlphaBeta.b, voltageSensor_b.v) annotation(
        Line(points = {{-72, 21}, {-74, 21}, {-74, 22}, {-82, 22}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.a, voltageSensor_a.v) annotation(
        Line(points = {{-72, 24}, {-76, 24}, {-76, 50}, {-80, 50}}, color = {0, 0, 127}));
      connect(abc2AlphaBeta.c, voltageSensor_c.v) annotation(
        Line(points = {{-72, 18}, {-76, 18}, {-76, -8}, {-82, -8}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.u, abc2AlphaBeta.alpha) annotation(
        Line(points = {{-37, 29}, {-40, 29}, {-40, 26}, {-52, 26}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.u, abc2AlphaBeta.beta) annotation(
        Line(points = {{-37, 15}, {-42, 15}, {-42, 17}, {-52, 17}}, color = {0, 0, 127}));
      connect(Norm_U_ref_alpha.y, alphaSin.u1) annotation(
        Line(points = {{-30, 30}, {-11, 30}, {-11, 31}}, color = {0, 0, 127}));
      connect(Norm_U_ref_beta.y, betaCos.u1) annotation(
        Line(points = {{-30, 16}, {-13, 16}, {-13, 17}}, color = {0, 0, 127}));
      connect(sin.y, alphaSin.u2) annotation(
        Line(points = {{-14, -6}, {-22, -6}, {-22, 27}, {-11, 27}}, color = {0, 0, 127}));
      connect(cos.y, betaCos.u2) annotation(
        Line(points = {{-14, -18}, {-18, -18}, {-18, 13}, {-13, 13}}, color = {0, 0, 127}));
      connect(add.u1, alphaSin.y) annotation(
        Line(points = {{7, 26}, {3.5, 26}, {3.5, 30}, {-4, 30}}, color = {0, 0, 127}));
      connect(betaCos.y, add.u2) annotation(
        Line(points = {{-6, 16}, {4, 16}, {4, 22}, {7, 22}}, color = {0, 0, 127}));
      connect(pi.u, add.y) annotation(
        Line(points = {{19, 24}, {16, 24}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.u1, pi.y) annotation(
        Line(points = {{37, 24}, {33, 24}}, color = {0, 0, 127}));
      connect(f_nom.y, add_freq_nom_delta_f.u2) annotation(
        Line(points = {{32, 4}, {36, 4}, {36, 20}, {37, 20}}, color = {0, 0, 127}));
      connect(deg2rad.u, f2theta.y) annotation(
        Line(points = {{79, 22}, {74, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, sin.u) annotation(
        Line(points = {{88, 22}, {92, 22}, {92, -6}, {-6, -6}}, color = {0, 0, 127}));
      connect(cos.u, deg2rad.y) annotation(
        Line(points = {{-6, -18}, {4, -18}, {4, -6}, {92, -6}, {92, 22}, {88, 22}}, color = {0, 0, 127}));
      connect(deg2rad.y, theta) annotation(
        Line(points = {{88, 22}, {92, 22}, {92, 0}, {108, 0}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.y, f2theta.u) annotation(
        Line(points = {{46, 22}, {64, 22}, {64, 22}, {66, 22}}, color = {0, 0, 127}));
      connect(add_freq_nom_delta_f.y, lowpassButterworth.u) annotation(
        Line(points = {{46, 22}, {48, 22}, {48, 54}, {50, 54}, {50, 54}}, color = {0, 0, 127}));
      connect(voltageSensor_a.v, lowpassButterworth1.u) annotation(
        Line(points = {{-80, 50}, {-64, 50}, {-64, 64}, {-48, 64}, {-48, 64}}, color = {0, 0, 127}));
      connect(lowpassButterworth.y, integrator.u) annotation(
        Line(points = {{74, 54}, {78, 54}, {78, 54}, {80, 54}}, color = {0, 0, 127}));
      connect(voltageSensor_b.v, lowpassButterworth2.u) annotation(
        Line(points = {{-82, 22}, {-80, 22}, {-80, -26}, {-72, -26}, {-72, -26}}, color = {0, 0, 127}));
      connect(voltageSensor_c.v, lowpassButterworth3.u) annotation(
        Line(points = {{-82, -8}, {-82, -8}, {-82, -56}, {-72, -56}, {-72, -56}}, color = {0, 0, 127}));
    end pll_ueff_test;
  end plls;

  package transforms
    model abc2AlphaBeta
      Modelica.Blocks.Interfaces.RealInput a annotation(
        Placement(visible = true, transformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput b annotation(
        Placement(visible = true, transformation(origin = {-104, 12}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 12}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput c annotation(
        Placement(visible = true, transformation(origin = {-104, -18}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, -18}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-40, 78}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = -1 / 3) annotation(
        Placement(visible = true, transformation(origin = {-39, 55}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = -1 / 3) annotation(
        Placement(visible = true, transformation(origin = {-39, 29}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {58, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = -1 / sqrt(3)) annotation(
        Placement(visible = true, transformation(origin = {-32, -18}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain4(k = 1 / sqrt(3)) annotation(
        Placement(visible = true, transformation(origin = {-32, -64}, extent = {{-14, -14}, {14, 14}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum1(nu = 2) annotation(
        Placement(visible = true, transformation(origin = {48, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput alpha annotation(
        Placement(visible = true, transformation(origin = {102, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {102, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput beta annotation(
        Placement(visible = true, transformation(origin = {102, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {102, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(a, gain.u) annotation(
        Line(points = {{-104, 40}, {-77, 40}, {-77, 78}, {-48, 78}}, color = {0, 0, 127}));
      connect(gain1.u, b) annotation(
        Line(points = {{-48, 56}, {-71, 56}, {-71, 12}, {-104, 12}}, color = {0, 0, 127}));
      connect(c, gain2.u) annotation(
        Line(points = {{-104, -18}, {-62, -18}, {-62, 30}, {-48, 30}}, color = {0, 0, 127}));
      connect(gain.y, multiSum.u[1]) annotation(
        Line(points = {{-34, 78}, {48, 78}, {48, 66}}, color = {0, 0, 127}));
      connect(gain1.y, multiSum.u[2]) annotation(
        Line(points = {{-32, 56}, {48, 56}, {48, 66}, {48, 66}}, color = {0, 0, 127}));
      connect(gain2.y, multiSum.u[3]) annotation(
        Line(points = {{-32, 30}, {48, 30}, {48, 66}, {48, 66}}, color = {0, 0, 127}));
      connect(gain4.u, b) annotation(
        Line(points = {{-48, -64}, {-73, -64}, {-73, -62}, {-72, -62}, {-72, 12}, {-104, 12}}, color = {0, 0, 127}));
      connect(gain3.u, c) annotation(
        Line(points = {{-48, -18}, {-96, -18}, {-96, -18}, {-104, -18}}, color = {0, 0, 127}));
      connect(gain3.y, multiSum1.u[1]) annotation(
        Line(points = {{-16, -18}, {38, -18}, {38, -34}, {38, -34}}, color = {0, 0, 127}));
      connect(gain4.y, multiSum1.u[2]) annotation(
        Line(points = {{-16, -64}, {38, -64}, {38, -34}, {38, -34}}, color = {0, 0, 127}));
      connect(multiSum.y, alpha) annotation(
        Line(points = {{70, 66}, {96, 66}, {96, 64}, {102, 64}}, color = {0, 0, 127}));
      connect(multiSum1.y, beta) annotation(
        Line(points = {{60, -34}, {102, -34}}, color = {0, 0, 127}));
    end abc2AlphaBeta;

    model abc2dq
      Real pi = 2 * Modelica.Math.asin(1.0);
      Modelica.Blocks.Interfaces.RealInput a annotation(
        Placement(visible = true, transformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput b annotation(
        Placement(visible = true, transformation(origin = {-104, 12}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 12}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput c annotation(
        Placement(visible = true, transformation(origin = {-104, -18}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, -18}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-52, 78}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-51, 55}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-51, 29}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {76, 66}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput d annotation(
        Placement(visible = true, transformation(origin = {102, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {102, 64}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput u annotation(
        Placement(visible = true, transformation(origin = {-34, 116}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {-34, 116}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {26, 44}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-17, 42}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 4 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-7, 16}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Add add(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {6, 46}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {14, 20}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos1 annotation(
        Placement(visible = true, transformation(origin = {34, 20}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos2 annotation(
        Placement(visible = true, transformation(origin = {26, 68}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {56, 24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {44, 48}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product2 annotation(
        Placement(visible = true, transformation(origin = {50, 72}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum1(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {78, -16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product3 annotation(
        Placement(visible = true, transformation(origin = {58, -58}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product4 annotation(
        Placement(visible = true, transformation(origin = {46, -34}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-49, -27}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {16, -62}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain4(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-50, -4}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {6, -38}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 2 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-15, -40}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput q annotation(
        Placement(visible = true, transformation(origin = {104, -18}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {104, -18}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain5(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-49, -53}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Product product5 annotation(
        Placement(visible = true, transformation(origin = {52, -10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-5, -66}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {37, -77}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin1 annotation(
        Placement(visible = true, transformation(origin = {29, -39}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin2 annotation(
        Placement(visible = true, transformation(origin = {23, -17}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
    equation
      connect(a, gain.u) annotation(
        Line(points = {{-104, 40}, {-77, 40}, {-77, 78}, {-59, 78}}, color = {0, 0, 127}));
      connect(gain1.u, b) annotation(
        Line(points = {{-59, 55}, {-71, 55}, {-71, 12}, {-104, 12}}, color = {0, 0, 127}));
      connect(c, gain2.u) annotation(
        Line(points = {{-104, -18}, {-62, -18}, {-62, 29}, {-59, 29}}, color = {0, 0, 127}));
      connect(multiSum.y, d) annotation(
        Line(points = {{88, 66}, {96, 66}, {96, 64}, {102, 64}}, color = {0, 0, 127}));
      connect(u, add1.u1) annotation(
        Line(points = {{-34, 116}, {-34, 24}, {6, 24}}, color = {0, 0, 127}));
      connect(u, add.u1) annotation(
        Line(points = {{-34, 116}, {-34, 50}, {-2, 50}}, color = {0, 0, 127}));
      connect(u, cos2.u) annotation(
        Line(points = {{-34, 116}, {-34, 68}, {18, 68}}, color = {0, 0, 127}));
      connect(realExpression.y, add.u2) annotation(
        Line(points = {{-10, 42}, {-2, 42}, {-2, 42}, {-2, 42}}, color = {0, 0, 127}));
      connect(realExpression1.y, add1.u2) annotation(
        Line(points = {{0, 16}, {6, 16}, {6, 16}, {6, 16}}, color = {0, 0, 127}));
      connect(add1.y, cos1.u) annotation(
        Line(points = {{20, 20}, {26, 20}, {26, 20}, {26, 20}}, color = {0, 0, 127}));
      connect(add.y, cos.u) annotation(
        Line(points = {{12, 46}, {16, 46}, {16, 44}, {18, 44}, {18, 44}}, color = {0, 0, 127}));
      connect(gain.y, product2.u1) annotation(
        Line(points = {{-46, 78}, {42, 78}, {42, 76}, {42, 76}}, color = {0, 0, 127}));
      connect(cos2.y, product2.u2) annotation(
        Line(points = {{32, 68}, {42, 68}, {42, 68}, {42, 68}}, color = {0, 0, 127}));
      connect(cos.y, product1.u2) annotation(
        Line(points = {{32, 44}, {36, 44}, {36, 44}, {36, 44}}, color = {0, 0, 127}));
      connect(gain1.y, product1.u1) annotation(
        Line(points = {{-44, 56}, {36, 56}, {36, 52}, {36, 52}}, color = {0, 0, 127}));
      connect(gain2.y, product.u1) annotation(
        Line(points = {{-44, 30}, {48, 30}, {48, 28}, {48, 28}}, color = {0, 0, 127}));
      connect(product.y, multiSum.u[1]) annotation(
        Line(points = {{62, 24}, {68, 24}, {68, 50}, {66, 50}, {66, 66}, {66, 66}}, color = {0, 0, 127}));
      connect(product1.y, multiSum.u[2]) annotation(
        Line(points = {{50, 48}, {60, 48}, {60, 62}, {66, 62}, {66, 66}, {66, 66}}, color = {0, 0, 127}));
      connect(product2.y, multiSum.u[3]) annotation(
        Line(points = {{56, 72}, {62, 72}, {62, 68}, {68, 68}, {68, 66}, {66, 66}}, color = {0, 0, 127}));
      connect(cos1.y, product.u2) annotation(
        Line(points = {{40, 20}, {48, 20}, {48, 20}, {48, 20}}, color = {0, 0, 127}));
      connect(gain5.y, product3.u1) annotation(
        Line(points = {{-41, -53}, {51, -53}, {51, -54}}, color = {0, 0, 127}));
      connect(product4.y, multiSum1.u[2]) annotation(
        Line(points = {{53, -34}, {68, -34}, {68, -16}}, color = {0, 0, 127}));
      connect(realExpression2.y, add3.u2) annotation(
        Line(points = {{-7, -40}, {-4, -40}, {-4, -42}, {-1, -42}}, color = {0, 0, 127}));
      connect(gain3.y, product4.u1) annotation(
        Line(points = {{-41, -27}, {39, -27}, {39, -30}}, color = {0, 0, 127}));
      connect(gain4.y, product5.u1) annotation(
        Line(points = {{-43, -4}, {5, -4}, {5, -6}, {45, -6}}, color = {0, 0, 127}));
      connect(realExpression3.y, add2.u2) annotation(
        Line(points = {{3, -66}, {9, -66}}, color = {0, 0, 127}));
      connect(multiSum1.y, q) annotation(
        Line(points = {{90, -16}, {101, -16}, {101, -18}, {104, -18}}, color = {0, 0, 127}));
      connect(product3.y, multiSum1.u[1]) annotation(
        Line(points = {{65, -58}, {68, -58}, {68, -16}}, color = {0, 0, 127}));
      connect(product5.y, multiSum1.u[3]) annotation(
        Line(points = {{59, -10}, {75.5, -10}, {75.5, -16}, {68, -16}}, color = {0, 0, 127}));
      connect(a, gain4.u) annotation(
        Line(points = {{-104, 40}, {-76, 40}, {-76, -4}, {-58, -4}, {-58, -4}}, color = {0, 0, 127}));
      connect(b, gain3.u) annotation(
        Line(points = {{-104, 12}, {-70, 12}, {-70, -26}, {-58, -26}, {-58, -26}}, color = {0, 0, 127}));
      connect(c, gain5.u) annotation(
        Line(points = {{-104, -18}, {-62, -18}, {-62, -52}, {-58, -52}, {-58, -52}}, color = {0, 0, 127}));
      connect(add2.y, sin.u) annotation(
        Line(points = {{22, -62}, {26, -62}, {26, -77}, {29, -77}}, color = {0, 0, 127}));
      connect(sin.y, product3.u2) annotation(
        Line(points = {{45, -77}, {45, -62}, {50, -62}}, color = {0, 0, 127}));
      connect(sin1.y, product4.u2) annotation(
        Line(points = {{36, -38}, {38, -38}, {38, -38}, {38, -38}, {38, -38}}, color = {0, 0, 127}));
      connect(sin2.y, product5.u2) annotation(
        Line(points = {{31, -17}, {40, -17}, {40, -14}, {44, -14}}, color = {0, 0, 127}));
      connect(u, add2.u1) annotation(
        Line(points = {{-34, 116}, {-34, 116}, {-34, -58}, {8, -58}, {8, -58}}, color = {0, 0, 127}));
      connect(u, add3.u1) annotation(
        Line(points = {{-34, 116}, {-34, -34}, {-1, -34}}, color = {0, 0, 127}));
      connect(u, sin2.u) annotation(
        Line(points = {{-34, 116}, {-34, -17}, {15, -17}}, color = {0, 0, 127}));
      connect(sin1.u, add3.y) annotation(
        Line(points = {{20, -38}, {14, -38}, {14, -38}, {12, -38}}, color = {0, 0, 127}));
    end abc2dq;

    model dq2abc
      Real pi = 2 * Modelica.Math.asin(1.0);
      Modelica.Blocks.Interfaces.RealInput d annotation(
        Placement(visible = true, transformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, 40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput q annotation(
        Placement(visible = true, transformation(origin = {-104, -40}, extent = {{-12, -12}, {12, 12}}, rotation = 0), iconTransformation(origin = {-104, -40}, extent = {{-12, -12}, {12, 12}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput b annotation(
        Placement(visible = true, transformation(origin = {106, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput theta annotation(
        Placement(visible = true, transformation(origin = {-34, 116}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {-34, 116}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 2 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-7, 6}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {14, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos1 annotation(
        Placement(visible = true, transformation(origin = {34, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos2 annotation(
        Placement(visible = true, transformation(origin = {26, 68}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {56, 14}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {44, 48}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product2 annotation(
        Placement(visible = true, transformation(origin = {50, 72}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product3 annotation(
        Placement(visible = true, transformation(origin = {58, -58}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product4 annotation(
        Placement(visible = true, transformation(origin = {46, -34}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {16, -62}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {6, -38}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 4 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-17, -40}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput c annotation(
        Placement(visible = true, transformation(origin = {106, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-5, -66}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin annotation(
        Placement(visible = true, transformation(origin = {37, -63}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin5 annotation(
        Placement(visible = true, transformation(origin = {17, 43}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Add add annotation(
        Placement(visible = true, transformation(origin = {74, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput a annotation(
        Placement(visible = true, transformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {106, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product6 annotation(
        Placement(visible = true, transformation(origin = {54, -8}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add4(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {14, -12}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-11, -16}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin8 annotation(
        Placement(visible = true, transformation(origin = {35, -13}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Add add5 annotation(
        Placement(visible = true, transformation(origin = {80, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos annotation(
        Placement(visible = true, transformation(origin = {24, -38}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add6 annotation(
        Placement(visible = true, transformation(origin = {82, -52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    equation
      connect(theta, add1.u1) annotation(
        Line(points = {{-34, 116}, {-34, 14}, {7, 14}}, color = {0, 0, 127}));
      connect(theta, cos2.u) annotation(
        Line(points = {{-34, 116}, {-34, 68}, {18, 68}}, color = {0, 0, 127}));
      connect(realExpression1.y, add1.u2) annotation(
        Line(points = {{1, 6}, {7, 6}}, color = {0, 0, 127}));
      connect(add1.y, cos1.u) annotation(
        Line(points = {{21, 10}, {27, 10}}, color = {0, 0, 127}));
      connect(cos2.y, product2.u2) annotation(
        Line(points = {{32, 68}, {42, 68}, {42, 68}, {42, 68}}, color = {0, 0, 127}));
      connect(cos1.y, product.u2) annotation(
        Line(points = {{41, 10}, {49, 10}}, color = {0, 0, 127}));
      connect(realExpression2.y, add3.u2) annotation(
        Line(points = {{-9, -40}, {-4, -40}, {-4, -42}, {-1, -42}}, color = {0, 0, 127}));
      connect(realExpression3.y, add2.u2) annotation(
        Line(points = {{3, -66}, {9, -66}}, color = {0, 0, 127}));
      connect(add2.y, sin.u) annotation(
        Line(points = {{22, -62}, {26, -62}, {26, -63}, {29, -63}}, color = {0, 0, 127}));
      connect(sin.y, product3.u2) annotation(
        Line(points = {{45, -63}, {45, -62}, {50, -62}}, color = {0, 0, 127}));
      connect(theta, add2.u1) annotation(
        Line(points = {{-34, 116}, {-34, 116}, {-34, -58}, {8, -58}, {8, -58}}, color = {0, 0, 127}));
      connect(theta, add3.u1) annotation(
        Line(points = {{-34, 116}, {-34, -32}, {-2.25, -32}, {-2.25, -34}, {-1, -34}}, color = {0, 0, 127}));
      connect(d, product2.u1) annotation(
        Line(points = {{-104, 40}, {-60, 40}, {-60, 76}, {42, 76}, {42, 76}}, color = {0, 0, 127}));
      connect(theta, sin5.u) annotation(
        Line(points = {{-34, 116}, {-34, 116}, {-34, 42}, {8, 42}, {8, 44}}, color = {0, 0, 127}));
      connect(sin5.y, product1.u2) annotation(
        Line(points = {{24, 44}, {36, 44}, {36, 44}, {36, 44}}, color = {0, 0, 127}));
      connect(q, product1.u1) annotation(
        Line(points = {{-104, -40}, {-50, -40}, {-50, 52}, {36, 52}, {36, 52}}, color = {0, 0, 127}));
      connect(product1.y, add.u2) annotation(
        Line(points = {{50, 48}, {56, 48}, {56, 52}, {62, 52}, {62, 52}}, color = {0, 0, 127}));
      connect(add.y, a) annotation(
        Line(points = {{86, 58}, {92, 58}, {92, 60}, {106, 60}}, color = {0, 0, 127}));
      connect(product2.y, add.u1) annotation(
        Line(points = {{56, 72}, {58, 72}, {58, 64}, {62, 64}, {62, 64}, {62, 64}}, color = {0, 0, 127}));
      connect(sin8.y, product6.u2) annotation(
        Line(points = {{43, -13}, {43, -12.5}, {47, -12.5}, {47, -12}}, color = {0, 0, 127}));
      connect(realExpression.y, add4.u2) annotation(
        Line(points = {{-3, -16}, {7, -16}}, color = {0, 0, 127}));
      connect(sin8.u, add4.y) annotation(
        Line(points = {{27, -13}, {26, -13}, {26, -12}, {21, -12}}, color = {0, 0, 127}));
      connect(theta, add4.u1) annotation(
        Line(points = {{-34, 116}, {-34, -8}, {7, -8}}, color = {0, 0, 127}));
      connect(d, product.u1) annotation(
        Line(points = {{-104, 40}, {-60, 40}, {-60, 18}, {49, 18}}, color = {0, 0, 127}));
      connect(q, product6.u1) annotation(
        Line(points = {{-104, -40}, {-50, -40}, {-50, -2}, {47, -2}, {47, -4}}, color = {0, 0, 127}));
      connect(product6.y, add5.u2) annotation(
        Line(points = {{61, -8}, {64, -8}, {64, -4}, {68, -4}}, color = {0, 0, 127}));
      connect(product.y, add5.u1) annotation(
        Line(points = {{63, 14}, {64, 14}, {64, 8}, {68, 8}}, color = {0, 0, 127}));
      connect(add5.y, b) annotation(
        Line(points = {{92, 2}, {94, 2}, {94, 0}, {106, 0}, {106, 0}}, color = {0, 0, 127}));
      connect(cos.y, product4.u2) annotation(
        Line(points = {{30, -38}, {38, -38}, {38, -38}, {38, -38}}, color = {0, 0, 127}));
      connect(d, product4.u1) annotation(
        Line(points = {{-104, 40}, {-60, 40}, {-60, -30}, {38, -30}, {38, -30}}, color = {0, 0, 127}));
      connect(add3.y, cos.u) annotation(
        Line(points = {{12, -38}, {16, -38}, {16, -38}, {16, -38}}, color = {0, 0, 127}));
      connect(q, product3.u1) annotation(
        Line(points = {{-104, -40}, {-50, -40}, {-50, -48}, {48, -48}, {48, -54}, {50, -54}, {50, -54}}, color = {0, 0, 127}));
      connect(product3.y, add6.u2) annotation(
        Line(points = {{64, -58}, {68, -58}, {68, -58}, {70, -58}}, color = {0, 0, 127}));
      connect(product4.y, add6.u1) annotation(
        Line(points = {{52, -34}, {58, -34}, {58, -46}, {68, -46}, {68, -46}, {70, -46}}, color = {0, 0, 127}));
      connect(add6.y, c) annotation(
        Line(points = {{94, -52}, {94, -52}, {94, -60}, {106, -60}, {106, -60}}, color = {0, 0, 127}));
    end dq2abc;

    model abc2dq_current
      Real Pi = 3.14159265;
      Modelica.Electrical.Analog.Interfaces.Pin a annotation(
        Placement(visible = true, transformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin b annotation(
        Placement(visible = true, transformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin c annotation(
        Placement(visible = true, transformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-102, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput d annotation(
        Placement(visible = true, transformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {-40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Blocks.Interfaces.RealOutput q annotation(
        Placement(visible = true, transformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90), iconTransformation(origin = {40, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
      Modelica.Electrical.Analog.Interfaces.Pin pin3 annotation(
        Placement(visible = true, transformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin1 annotation(
        Placement(visible = true, transformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Electrical.Analog.Interfaces.Pin pin2 annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-62, 60}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain1(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-61, 37}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain2(k = 2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-61, 11}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression1(y = 4 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-17, -2}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression(y = 2 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-27, 24}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Add add1(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-4, 28}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add2(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {4, 2}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos2 annotation(
        Placement(visible = true, transformation(origin = {16, 50}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product annotation(
        Placement(visible = true, transformation(origin = {46, 6}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos1 annotation(
        Placement(visible = true, transformation(origin = {24, 2}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Cos cos3 annotation(
        Placement(visible = true, transformation(origin = {16, 26}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product2 annotation(
        Placement(visible = true, transformation(origin = {40, 54}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Product product1 annotation(
        Placement(visible = true, transformation(origin = {34, 30}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {74, 54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.MultiSum multiSum1(nu = 3) annotation(
        Placement(visible = true, transformation(origin = {66, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      Modelica.Blocks.Math.Product product5 annotation(
        Placement(visible = true, transformation(origin = {40, -24}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add4(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {-6, -52}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Add add3(k2 = -1) annotation(
        Placement(visible = true, transformation(origin = {4, -76}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression2(y = 2 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-27, -54}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Sources.RealExpression realExpression3(y = 4 * Pi / 3) annotation(
        Placement(visible = true, transformation(origin = {-17, -80}, extent = {{-7, -8}, {7, 8}}, rotation = 0)));
      Modelica.Blocks.Math.Product product3 annotation(
        Placement(visible = true, transformation(origin = {46, -72}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin3 annotation(
        Placement(visible = true, transformation(origin = {25, -77}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain5(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-61, -67}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Product product4 annotation(
        Placement(visible = true, transformation(origin = {34, -48}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain3(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-61, -41}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin1 annotation(
        Placement(visible = true, transformation(origin = {17, -53}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Math.Gain gain4(k = -2 / 3) annotation(
        Placement(visible = true, transformation(origin = {-62, -18}, extent = {{-6, -6}, {6, 6}}, rotation = 0)));
      Modelica.Blocks.Math.Sin sin2 annotation(
        Placement(visible = true, transformation(origin = {11, -31}, extent = {{-7, -7}, {7, 7}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealInput theta annotation(
        Placement(visible = true, transformation(origin = {0, 112}, extent = {{-20, -20}, {20, 20}}, rotation = -90), iconTransformation(origin = {0, 112}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
      Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor3 annotation(
        Placement(visible = true, transformation(origin = {-90, -50}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor1 annotation(
        Placement(visible = true, transformation(origin = {-88, 70}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
      Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor2 annotation(
        Placement(visible = true, transformation(origin = {-90, 10}, extent = {{-6, -6}, {6, 6}}, rotation = 90)));
    equation
      connect(realExpression.y, add1.u2) annotation(
        Line(points = {{-19, 24}, {-11, 24}}, color = {0, 0, 127}));
      connect(realExpression1.y, add2.u2) annotation(
        Line(points = {{-9, -2}, {-3, -2}}, color = {0, 0, 127}));
      connect(gain2.y, product.u1) annotation(
        Line(points = {{-53, 11}, {39, 11}, {39, 10}}, color = {0, 0, 127}));
      connect(add2.y, cos1.u) annotation(
        Line(points = {{11, 2}, {17, 2}}, color = {0, 0, 127}));
      connect(cos1.y, product.u2) annotation(
        Line(points = {{31, 2}, {39, 2}}, color = {0, 0, 127}));
      connect(add1.y, cos3.u) annotation(
        Line(points = {{3, 28}, {4, 28}, {4, 26}, {9, 26}}, color = {0, 0, 127}));
      connect(cos2.y, product2.u2) annotation(
        Line(points = {{23, 50}, {33, 50}}, color = {0, 0, 127}));
      connect(gain.y, product2.u1) annotation(
        Line(points = {{-55, 60}, {-11, 60}, {-11, 58}, {33, 58}}, color = {0, 0, 127}));
      connect(gain1.y, product1.u1) annotation(
        Line(points = {{-53, 37}, {27, 37}, {27, 34}}, color = {0, 0, 127}));
      connect(cos3.y, product1.u2) annotation(
        Line(points = {{23, 26}, {27, 26}}, color = {0, 0, 127}));
      connect(multiSum.y, d) annotation(
        Line(points = {{86, 54}, {86, -94}, {-40, -94}, {-40, -110}}, color = {0, 0, 127}));
      connect(product1.y, multiSum.u[1]) annotation(
        Line(points = {{41, 30}, {55.5, 30}, {55.5, 54}, {64, 54}}, color = {0, 0, 127}));
      connect(product.y, multiSum.u[2]) annotation(
        Line(points = {{53, 6}, {64, 6}, {64, 54}}, color = {0, 0, 127}));
      connect(product2.y, multiSum.u[3]) annotation(
        Line(points = {{47, 54}, {64, 54}}, color = {0, 0, 127}));
      connect(sin2.y, product5.u2) annotation(
        Line(points = {{19, -31}, {30, -31}, {30, -28}, {33, -28}}, color = {0, 0, 127}));
      connect(product4.y, multiSum1.u[2]) annotation(
        Line(points = {{41, -48}, {56, -48}, {56, -30}}, color = {0, 0, 127}));
      connect(gain3.y, product4.u1) annotation(
        Line(points = {{-53, -41}, {27, -41}, {27, -44}}, color = {0, 0, 127}));
      connect(realExpression3.y, add3.u2) annotation(
        Line(points = {{-9, -80}, {-3, -80}}, color = {0, 0, 127}));
      connect(product5.y, multiSum1.u[3]) annotation(
        Line(points = {{47, -24}, {62.25, -24}, {62.25, -28}, {59.125, -28}, {59.125, -30}, {56, -30}}, color = {0, 0, 127}));
      connect(add3.y, sin3.u) annotation(
        Line(points = {{11, -76}, {28, -76}, {28, -77}, {17, -77}}, color = {0, 0, 127}));
      connect(sin1.u, add4.y) annotation(
        Line(points = {{9, -53}, {18, -53}, {18, -52}, {1, -52}}, color = {0, 0, 127}));
      connect(sin3.y, product3.u2) annotation(
        Line(points = {{33, -77}, {33, -76.5}, {39, -76.5}, {39, -76}}, color = {0, 0, 127}));
      connect(sin1.y, product4.u2) annotation(
        Line(points = {{25, -53}, {25, -50.5}, {27, -50.5}, {27, -52}}, color = {0, 0, 127}));
      connect(gain5.y, product3.u1) annotation(
        Line(points = {{-53, -67}, {39, -67}, {39, -68}}, color = {0, 0, 127}));
      connect(realExpression2.y, add4.u2) annotation(
        Line(points = {{-19, -54}, {-16, -54}, {-16, -56}, {-13, -56}}, color = {0, 0, 127}));
      connect(product3.y, multiSum1.u[1]) annotation(
        Line(points = {{53, -72}, {56, -72}, {56, -30}}, color = {0, 0, 127}));
      connect(gain4.y, product5.u1) annotation(
        Line(points = {{-55, -18}, {-35, -18}, {-35, -20}, {33, -20}}, color = {0, 0, 127}));
      connect(multiSum1.y, q) annotation(
        Line(points = {{78, -30}, {84, -30}, {84, -92}, {40, -92}, {40, -110}, {40, -110}}, color = {0, 0, 127}));
      connect(theta, cos2.u) annotation(
        Line(points = {{0, 112}, {0, 112}, {0, 50}, {8, 50}, {8, 50}}, color = {0, 0, 127}));
      connect(theta, add1.u1) annotation(
        Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, 32}, {-12, 32}, {-12, 32}}, color = {0, 0, 127}));
      connect(theta, add2.u1) annotation(
        Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, 6}, {-4, 6}, {-4, 6}}, color = {0, 0, 127}));
      connect(theta, sin2.u) annotation(
        Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, -32}, {2, -32}, {2, -30}}, color = {0, 0, 127}));
      connect(theta, add4.u1) annotation(
        Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, -48}, {-14, -48}, {-14, -48}}, color = {0, 0, 127}));
      connect(theta, add3.u1) annotation(
        Line(points = {{0, 112}, {0, 112}, {0, 86}, {-44, 86}, {-44, -72}, {-4, -72}, {-4, -72}}, color = {0, 0, 127}));
      connect(c, currentSensor3.p) annotation(
        Line(points = {{-102, -60}, {-90, -60}, {-90, -56}, {-90, -56}}, color = {0, 0, 255}));
      connect(b, currentSensor2.p) annotation(
        Line(points = {{-102, 0}, {-90, 0}, {-90, 4}, {-90, 4}, {-90, 4}}, color = {0, 0, 255}));
      connect(a, currentSensor1.p) annotation(
        Line(points = {{-100, 60}, {-88, 60}, {-88, 64}, {-88, 64}, {-88, 64}}, color = {0, 0, 255}));
      connect(currentSensor1.n, pin3) annotation(
        Line(points = {{-88, 76}, {-88, 76}, {-88, 84}, {94, 84}, {94, 60}, {100, 60}}, color = {0, 0, 255}));
      connect(currentSensor2.n, pin2) annotation(
        Line(points = {{-90, 16}, {-86, 16}, {-86, 82}, {92, 82}, {92, 0}, {100, 0}}, color = {0, 0, 255}));
      connect(currentSensor3.n, pin1) annotation(
        Line(points = {{-90, -44}, {-84, -44}, {-84, 80}, {90, 80}, {90, -60}, {100, -60}}, color = {0, 0, 255}));
      connect(currentSensor1.i, gain.u) annotation(
        Line(points = {{-82, 70}, {-72, 70}, {-72, 60}, {-70, 60}, {-70, 60}}, color = {0, 0, 127}));
      connect(gain4.u, currentSensor1.i) annotation(
        Line(points = {{-70, -18}, {-72, -18}, {-72, 70}, {-82, 70}, {-82, 70}}, color = {0, 0, 127}));
      connect(currentSensor2.i, gain1.u) annotation(
        Line(points = {{-84, 10}, {-80, 10}, {-80, 36}, {-70, 36}, {-70, 38}}, color = {0, 0, 127}));
      connect(currentSensor2.i, gain3.u) annotation(
        Line(points = {{-84, 10}, {-80, 10}, {-80, -42}, {-70, -42}, {-70, -40}}, color = {0, 0, 127}));
      connect(currentSensor3.i, gain2.u) annotation(
        Line(points = {{-84, -50}, {-76, -50}, {-76, 10}, {-70, 10}, {-70, 12}}, color = {0, 0, 127}));
      connect(currentSensor3.i, gain5.u) annotation(
        Line(points = {{-84, -50}, {-76, -50}, {-76, -68}, {-70, -68}, {-70, -66}}, color = {0, 0, 127}));
    end abc2dq_current;
  end transforms;

  model network
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter2 annotation(
      Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc2 annotation(
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
    ideal_filter.lcl lcl1 annotation(
      Placement(visible = true, transformation(origin = {-32, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.rl rl1 annotation(
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
      Line(points = {{-60, -24}, {-42, -24}, {-42, -24}, {-42, -24}}, color = {0, 0, 255}));
    connect(inverter2.pin2, lcl1.pin2) annotation(
      Line(points = {{-60, -30}, {-42, -30}, {-42, -30}, {-42, -30}}, color = {0, 0, 255}));
    connect(inverter2.pin1, lcl1.pin1) annotation(
      Line(points = {{-60, -36}, {-42, -36}, {-42, -36}, {-42, -36}}, color = {0, 0, 255}));
    connect(lcl1.pin6, lc2.pin3) annotation(
      Line(points = {{-22, -24}, {-6, -24}, {-6, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(lcl1.pin5, lc2.pin2) annotation(
      Line(points = {{-22, -30}, {0, -30}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(lcl1.pin4, lc2.pin1) annotation(
      Line(points = {{-22, -36}, {6, -36}, {6, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    connect(lc2.pin6, rl1.pin3) annotation(
      Line(points = {{40, 36}, {60, 36}, {60, 36}, {60, 36}}, color = {0, 0, 255}));
    connect(lc2.pin5, rl1.pin2) annotation(
      Line(points = {{40, 30}, {60, 30}, {60, 30}, {60, 30}}, color = {0, 0, 255}));
    connect(lc2.pin4, rl1.pin1) annotation(
      Line(points = {{40, 24}, {60, 24}, {60, 24}, {60, 24}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end network;

  model pll_network
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.rc rc1 annotation(
      Placement(visible = true, transformation(origin = {70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter2 annotation(
      Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc2 annotation(
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
    ideal_filter.lcl lcl1 annotation(
      Placement(visible = true, transformation(origin = {-30, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.plls.pll pll annotation(
      Placement(visible = true, transformation(origin = {20, -62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(lc2.pin4, rc1.pin1) annotation(
      Line(points = {{40, 24}, {60, 24}, {60, 24}, {60, 24}}, color = {0, 0, 255}));
    connect(lc2.pin5, rc1.pin2) annotation(
      Line(points = {{40, 30}, {60, 30}, {60, 30}, {60, 30}}, color = {0, 0, 255}));
    connect(lc2.pin6, rc1.pin3) annotation(
      Line(points = {{40, 36}, {60, 36}, {60, 36}, {60, 36}}, color = {0, 0, 255}));
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
      Line(points = {{-60, -24}, {-40, -24}, {-40, -24}, {-40, -24}}, color = {0, 0, 255}));
    connect(inverter2.pin2, lcl1.pin2) annotation(
      Line(points = {{-60, -30}, {-40, -30}, {-40, -30}, {-40, -30}}, color = {0, 0, 255}));
    connect(inverter2.pin1, lcl1.pin1) annotation(
      Line(points = {{-60, -36}, {-40, -36}, {-40, -36}, {-40, -36}}, color = {0, 0, 255}));
    connect(lcl1.pin6, lc2.pin3) annotation(
      Line(points = {{-20, -24}, {-4, -24}, {-4, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(lcl1.pin5, lc2.pin2) annotation(
      Line(points = {{-20, -30}, {0, -30}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(lcl1.pin4, lc2.pin1) annotation(
      Line(points = {{-20, -36}, {6, -36}, {6, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    connect(pll.a, lcl1.pin6) annotation(
      Line(points = {{10, -58}, {-14, -58}, {-14, -24}, {-20, -24}}, color = {0, 0, 255}));
    connect(pll.b, lcl1.pin5) annotation(
      Line(points = {{10, -60}, {-16, -60}, {-16, -30}, {-20, -30}}, color = {0, 0, 255}));
    connect(pll.c, lcl1.pin4) annotation(
      Line(points = {{10, -63}, {-18, -63}, {-18, -36}, {-20, -36}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end pll_network;

  model rlc_network
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter2 annotation(
      Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc2 annotation(
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
    ideal_filter.lcl lcl1 annotation(
      Placement(visible = true, transformation(origin = {-32, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.rlc rlc1 annotation(
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
      Line(points = {{-60, -24}, {-42, -24}, {-42, -24}, {-42, -24}}, color = {0, 0, 255}));
    connect(inverter2.pin2, lcl1.pin2) annotation(
      Line(points = {{-60, -30}, {-42, -30}, {-42, -30}, {-42, -30}}, color = {0, 0, 255}));
    connect(inverter2.pin1, lcl1.pin1) annotation(
      Line(points = {{-60, -36}, {-42, -36}, {-42, -36}, {-42, -36}}, color = {0, 0, 255}));
    connect(lcl1.pin6, lc2.pin3) annotation(
      Line(points = {{-22, -24}, {-6, -24}, {-6, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(lcl1.pin5, lc2.pin2) annotation(
      Line(points = {{-22, -30}, {0, -30}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(lcl1.pin4, lc2.pin1) annotation(
      Line(points = {{-22, -36}, {6, -36}, {6, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    connect(lc2.pin4, rlc1.pin1) annotation(
      Line(points = {{40, 24}, {60, 24}, {60, 24}, {60, 24}}, color = {0, 0, 255}));
    connect(lc2.pin5, rlc1.pin2) annotation(
      Line(points = {{40, 30}, {60, 30}, {60, 30}, {60, 30}}, color = {0, 0, 255}));
    connect(lc2.pin6, rlc1.pin3) annotation(
      Line(points = {{40, 36}, {60, 36}, {60, 36}, {60, 36}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end rlc_network;

  model pll_Test
    transforms.abc2AlphaBeta abc2AlphaBeta annotation(
      Placement(visible = true, transformation(origin = {-14, 4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine(amplitude = 230 * 1.414, freqHz = 50) annotation(
      Placement(visible = true, transformation(origin = {-90, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine1(amplitude = 230 * 1.414, freqHz = 50, phase = -2.0944) annotation(
      Placement(visible = true, transformation(origin = {-88, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine2(amplitude = 230 * 1.414, freqHz = 50, phase = -4.18879) annotation(
      Placement(visible = true, transformation(origin = {-88, -26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter annotation(
      Placement(visible = true, transformation(origin = {-14, 58}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.plls.pll pll annotation(
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
  end pll_Test;

  model sine_Test
    Modelica.Blocks.Sources.Sine sine(amplitude = 0.325, freqHz = 50) annotation(
      Placement(visible = true, transformation(origin = {-76, -82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine1(amplitude = 0.325, freqHz = 50, phase = 2.0944) annotation(
      Placement(visible = true, transformation(origin = {-76, -48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine2(amplitude = 0.325, freqHz = 50, phase = 4.18879) annotation(
      Placement(visible = true, transformation(origin = {-76, -16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine3(amplitude = 0.325, freqHz = 50) annotation(
      Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine4(amplitude = 0.325, freqHz = 50, phase = 2.0944) annotation(
      Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine5(amplitude = 0.325, freqHz = 50, phase = 4.18879) annotation(
      Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.network network1 annotation(
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
  end sine_Test;

  model network_singleInverter
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1(C1 = 0.00002, C2 = 0.00002, C3 = 0.00002, L1 = 0.002, L2 = 0.002, L3 = 0.002) annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p1 annotation(
      Placement(visible = true, transformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p2 annotation(
      Placement(visible = true, transformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p3 annotation(
      Placement(visible = true, transformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    grid.loads.rl rl1 annotation(
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
  end network_singleInverter;

  model sinepll
    Modelica.Blocks.Sources.Sine sine3(amplitude = 325.7, freqHz = 50, phase = 1.5708) annotation(
      Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine4(amplitude = 325.7, freqHz = 50, phase = 3.66519) annotation(
      Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine5(amplitude = 325.7, freqHz = 50, phase = 5.75959) annotation(
      Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter(v_DC = 1) annotation(
      Placement(visible = true, transformation(origin = {-20, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.active_loads.activ_z activ_z(p_ref = 1000, q_ref = 2000) annotation(
      Placement(visible = true, transformation(origin = {54, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.filter.lc lc(L1 = 0.0001, L2 = 0.0001, L3 = 0.0001, R1 = 0.1, R2 = 0.1, R3 = 0.1, R4 = 0.1, R5 = 0.1, R6 = 0.1) annotation(
      Placement(visible = true, transformation(origin = {18, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(sine5.y, inverter.u3) annotation(
      Line(points = {{-65, 80}, {-50, 80}, {-50, 54}, {-30, 54}}, color = {0, 0, 127}));
    connect(sine4.y, inverter.u2) annotation(
      Line(points = {{-64, 48}, {-30, 48}}, color = {0, 0, 127}));
    connect(sine3.y, inverter.u1) annotation(
      Line(points = {{-64, 14}, {-40, 14}, {-40, 42}, {-30, 42}, {-30, 42}}, color = {0, 0, 127}));
    connect(inverter.pin1, lc.pin1) annotation(
      Line(points = {{-10, 42}, {8, 42}, {8, 42}, {8, 42}}, color = {0, 0, 255}));
    connect(inverter.pin2, lc.pin2) annotation(
      Line(points = {{-10, 48}, {8, 48}, {8, 48}, {8, 48}}, color = {0, 0, 255}));
    connect(inverter.pin3, lc.pin3) annotation(
      Line(points = {{-10, 54}, {8, 54}, {8, 54}, {8, 54}}, color = {0, 0, 255}));
    connect(lc.pin6, activ_z.pin3) annotation(
      Line(points = {{28, 54}, {44, 54}, {44, 54}, {44, 54}}, color = {0, 0, 255}));
    connect(lc.pin5, activ_z.pin2) annotation(
      Line(points = {{28, 48}, {44, 48}, {44, 48}, {44, 48}}, color = {0, 0, 255}));
    connect(lc.pin4, activ_z.pin1) annotation(
      Line(points = {{28, 42}, {44, 42}, {44, 42}, {44, 42}}, color = {0, 0, 255}));
  protected
  end sinepll;

  model sine_l
    Modelica.Blocks.Sources.Sine sine3(amplitude = 325.7, freqHz = 50, phase = 1.5708) annotation(
      Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine4(amplitude = 325.7, freqHz = 50, phase = 3.66519) annotation(
      Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine5(amplitude = 325.7, freqHz = 50, phase = 5.75959) annotation(
      Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter(v_DC = 1) annotation(
      Placement(visible = true, transformation(origin = {-12, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc annotation(
      Placement(visible = true, transformation(origin = {20, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.r r annotation(
      Placement(visible = true, transformation(origin = {62, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(sine5.y, inverter.u3) annotation(
      Line(points = {{-65, 80}, {-50, 80}, {-50, 54}, {-22, 54}}, color = {0, 0, 127}));
    connect(sine4.y, inverter.u2) annotation(
      Line(points = {{-64, 48}, {-22, 48}}, color = {0, 0, 127}));
    connect(sine3.y, inverter.u1) annotation(
      Line(points = {{-64, 14}, {-22, 14}, {-22, 42}}, color = {0, 0, 127}));
    connect(inverter.pin3, lc.pin3) annotation(
      Line(points = {{-2, 54}, {10, 54}, {10, 54}, {10, 54}}, color = {0, 0, 255}));
    connect(inverter.pin2, lc.pin2) annotation(
      Line(points = {{-2, 48}, {10, 48}, {10, 48}, {10, 48}}, color = {0, 0, 255}));
    connect(inverter.pin1, lc.pin1) annotation(
      Line(points = {{-2, 42}, {10, 42}, {10, 42}, {10, 42}}, color = {0, 0, 255}));
    connect(lc.pin4, r.pin1) annotation(
      Line(points = {{30, 42}, {52, 42}, {52, 42}, {52, 42}}, color = {0, 0, 255}));
    connect(lc.pin5, r.pin2) annotation(
      Line(points = {{30, 48}, {52, 48}, {52, 48}, {52, 48}}, color = {0, 0, 255}));
    connect(lc.pin6, r.pin3) annotation(
      Line(points = {{30, 54}, {30, 54}, {30, 54}, {52, 54}}, color = {0, 0, 255}));
  protected
  end sine_l;

  model testbase
    Modelica.Electrical.Analog.Basic.Ground ground annotation(
      Placement(visible = true, transformation(origin = {72, -8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Electrical.Analog.Sources.SineVoltage sineVoltage annotation(
      Placement(visible = true, transformation(origin = {56, 42}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Electrical.Analog.Basic.Resistor resistor annotation(
      Placement(visible = true, transformation(origin = {50, 22}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(sineVoltage.p, resistor.p) annotation(
      Line(points = {{46, 42}, {30, 42}, {30, 22}, {40, 22}, {40, 22}}, color = {0, 0, 255}));
    connect(sineVoltage.n, resistor.n) annotation(
      Line(points = {{66, 42}, {76, 42}, {76, 22}, {60, 22}, {60, 22}}, color = {0, 0, 255}));
    connect(resistor.n, ground.p) annotation(
      Line(points = {{60, 22}, {72, 22}, {72, 2}, {72, 2}, {72, 2}}, color = {0, 0, 255}));
  end testbase;

  model sinepll_2
    Modelica.Blocks.Sources.Sine sine3(amplitude = 325.7, freqHz = 50, phase = 1.5708) annotation(
      Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine4(amplitude = 325.7, freqHz = 50, phase = 3.66519) annotation(
      Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine5(amplitude = 325.7, freqHz = 50, phase = 5.75959) annotation(
      Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter(v_DC = 1) annotation(
      Placement(visible = true, transformation(origin = {-20, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc(L1 = 0.002, L2 = 0.002, L3 = 0.002) annotation(
      Placement(visible = true, transformation(origin = {18, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.active_loads.activ_z activ_z(p_ref = 1000, q_ref = 100) annotation(
      Placement(visible = true, transformation(origin = {54, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(sine5.y, inverter.u3) annotation(
      Line(points = {{-65, 80}, {-50, 80}, {-50, 54}, {-30, 54}}, color = {0, 0, 127}));
    connect(sine4.y, inverter.u2) annotation(
      Line(points = {{-64, 48}, {-30, 48}}, color = {0, 0, 127}));
    connect(sine3.y, inverter.u1) annotation(
      Line(points = {{-64, 14}, {-40, 14}, {-40, 42}, {-30, 42}, {-30, 42}}, color = {0, 0, 127}));
    connect(inverter.pin3, lc.pin3) annotation(
      Line(points = {{-10, 54}, {8, 54}, {8, 54}, {8, 54}}, color = {0, 0, 255}));
    connect(inverter.pin2, lc.pin2) annotation(
      Line(points = {{-10, 48}, {8, 48}, {8, 48}, {8, 48}}, color = {0, 0, 255}));
    connect(inverter.pin1, lc.pin1) annotation(
      Line(points = {{-10, 42}, {8, 42}, {8, 42}, {8, 42}}, color = {0, 0, 255}));
    connect(lc.pin6, activ_z.pin3) annotation(
      Line(points = {{28, 54}, {44, 54}, {44, 54}, {44, 54}}, color = {0, 0, 255}));
    connect(lc.pin5, activ_z.pin2) annotation(
      Line(points = {{28, 48}, {44, 48}, {44, 48}, {44, 48}}, color = {0, 0, 255}));
    connect(lc.pin4, activ_z.pin1) annotation(
      Line(points = {{28, 42}, {44, 42}, {44, 42}, {44, 42}}, color = {0, 0, 255}));
  protected
  end sinepll_2;

  model network2
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.inverters.inverter inverter2 annotation(
      Placement(visible = true, transformation(origin = {-70, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc2 annotation(
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
    ideal_filter.lcl lcl1 annotation(
      Placement(visible = true, transformation(origin = {-32, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.active_loads.activ_z activ_z(p_ref = 2000, q_ref = 100) annotation(
      Placement(visible = true, transformation(origin = {66, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
      Line(points = {{-60, -24}, {-42, -24}, {-42, -24}, {-42, -24}}, color = {0, 0, 255}));
    connect(inverter2.pin2, lcl1.pin2) annotation(
      Line(points = {{-60, -30}, {-42, -30}, {-42, -30}, {-42, -30}}, color = {0, 0, 255}));
    connect(inverter2.pin1, lcl1.pin1) annotation(
      Line(points = {{-60, -36}, {-42, -36}, {-42, -36}, {-42, -36}}, color = {0, 0, 255}));
    connect(lcl1.pin6, lc2.pin3) annotation(
      Line(points = {{-22, -24}, {-6, -24}, {-6, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(lcl1.pin5, lc2.pin2) annotation(
      Line(points = {{-22, -30}, {0, -30}, {0, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(lcl1.pin4, lc2.pin1) annotation(
      Line(points = {{-22, -36}, {6, -36}, {6, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    connect(lc2.pin6, activ_z.pin3) annotation(
      Line(points = {{40, 36}, {56, 36}, {56, 36}, {56, 36}}, color = {0, 0, 255}));
    connect(lc2.pin5, activ_z.pin2) annotation(
      Line(points = {{40, 30}, {56, 30}, {56, 30}, {56, 30}}, color = {0, 0, 255}));
    connect(lc2.pin4, activ_z.pin1) annotation(
      Line(points = {{40, 24}, {56, 24}, {56, 24}, {56, 24}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end network2;

  model sine_Test2
    Modelica.Blocks.Sources.Sine sine3(amplitude = 0.325, freqHz = 50) annotation(
      Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine4(amplitude = 0.325, freqHz = 50, phase = 2.0944) annotation(
      Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine5(amplitude = 0.325, freqHz = 50, phase = 4.18879) annotation(
      Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  grid.network_ctrl network_ctrl1 annotation(
      Placement(visible = true, transformation(origin = {46, 46}, extent = {{-34, -34}, {34, 34}}, rotation = 0)));
  equation
  connect(sine3.y, network_ctrl1.i1p1) annotation(
      Line(points = {{-64, 14}, {0, 14}, {0, 50}, {10, 50}, {10, 52}}, color = {0, 0, 127}));
  connect(sine4.y, network_ctrl1.i1p2) annotation(
      Line(points = {{-64, 48}, {-8, 48}, {-8, 56}, {10, 56}, {10, 56}, {10, 56}}, color = {0, 0, 127}));
  connect(sine5.y, network_ctrl1.i1p3) annotation(
      Line(points = {{-64, 80}, {0, 80}, {0, 60}, {10, 60}, {10, 60}, {10, 60}}, color = {0, 0, 127}));
  protected
  end sine_Test2;

  model network_singleInverter2
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p1 annotation(
      Placement(visible = true, transformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p2 annotation(
      Placement(visible = true, transformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p3 annotation(
      Placement(visible = true, transformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    grid.active_loads.activ_z activ_z(p_ref = 500, q_ref = 2000) annotation(
      Placement(visible = true, transformation(origin = {10, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.filter.lc lc1(L1 = 0.0001, L2 = 0.0001, L3 = 0.0001, R1 = 0.1, R2 = 0.1, R3 = 0.1, R4 = 0.1, R5 = 0.1, R6 = 0.1) annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(i1p1, inverter1.u1) annotation(
      Line(points = {{-104, 18}, {-86, 18}, {-86, 24}, {-80, 24}, {-80, 24}}, color = {0, 0, 127}));
    connect(i1p2, inverter1.u2) annotation(
      Line(points = {{-104, 30}, {-80, 30}, {-80, 30}, {-80, 30}}, color = {0, 0, 127}));
    connect(i1p3, inverter1.u3) annotation(
      Line(points = {{-104, 42}, {-86, 42}, {-86, 36}, {-80, 36}}, color = {0, 0, 127}));
    connect(inverter1.pin1, lc1.pin1) annotation(
      Line(points = {{-60, 24}, {-40, 24}, {-40, 24}, {-40, 24}}, color = {0, 0, 255}));
    connect(inverter1.pin2, lc1.pin2) annotation(
      Line(points = {{-60, 30}, {-60, 30}, {-60, 30}, {-40, 30}}, color = {0, 0, 255}));
    connect(inverter1.pin3, lc1.pin3) annotation(
      Line(points = {{-60, 36}, {-40, 36}, {-40, 36}, {-40, 36}}, color = {0, 0, 255}));
    connect(lc1.pin6, activ_z.pin3) annotation(
      Line(points = {{-20, 36}, {-20, 36}, {-20, 36}, {0, 36}}, color = {0, 0, 255}));
    connect(lc1.pin5, activ_z.pin2) annotation(
      Line(points = {{-20, 30}, {0, 30}, {0, 30}, {0, 30}}, color = {0, 0, 255}));
    connect(lc1.pin4, activ_z.pin1) annotation(
      Line(points = {{-20, 24}, {-20, 24}, {-20, 24}, {0, 24}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end network_singleInverter2;

  model network_ctrl
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.rl rl1 annotation(
      Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.plls.pll_d pll_d annotation(
      Placement(visible = true, transformation(origin = {2, 62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.transforms.abc2dq_current abc2dq_current annotation(
      Placement(visible = true, transformation(origin = {0, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression annotation(
      Placement(visible = true, transformation(origin = {2, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression1(y = 230 * 1.41427)  annotation(
      Placement(visible = true, transformation(origin = {5, 77}, extent = {{-13, -11}, {13, 11}}, rotation = 0)));
  Modelica.Blocks.Continuous.PI pi(T = 0.000417, k = 0.025)  annotation(
      Placement(visible = true, transformation(origin = {-76, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Feedback feedback annotation(
      Placement(visible = true, transformation(origin = {34, 78}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Feedback feedback1 annotation(
      Placement(visible = true, transformation(origin = {68, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.PI PI(T = 0.0041666, k = 0.025)  annotation(
      Placement(visible = true, transformation(origin = {-76, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.PI pi1 annotation(
      Placement(visible = true, transformation(origin = { 2, -32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Feedback feedback2 annotation(
      Placement(visible = true, transformation(origin = {-36, -32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Math.Feedback feedback3 annotation(
      Placement(visible = true, transformation(origin = {-36, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.PI PI1(T = 0.000133, k = 0.012)  annotation(
      Placement(visible = true, transformation(origin = {2, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  grid.transforms.dq2abc dq2abc annotation(
      Placement(visible = true, transformation(origin = {36, -18}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID pid annotation(
      Placement(visible = true, transformation(origin = {94, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(inverter1.pin3, lc1.pin3) annotation(
      Line(points = {{-60, 36}, {-40, 36}}, color = {0, 0, 255}));
    connect(inverter1.pin2, lc1.pin2) annotation(
      Line(points = {{-60, 30}, {-40, 30}}, color = {0, 0, 255}));
    connect(inverter1.pin1, lc1.pin1) annotation(
      Line(points = {{-60, 24}, {-40, 24}}, color = {0, 0, 255}));
    connect(lc1.pin6, abc2dq_current.a) annotation(
      Line(points = {{-20, 36}, {-10, 36}, {-10, 36}, {-10, 36}}, color = {0, 0, 255}));
    connect(lc1.pin5, abc2dq_current.b) annotation(
      Line(points = {{-20, 30}, {-10, 30}, {-10, 30}, {-10, 30}}, color = {0, 0, 255}));
    connect(lc1.pin4, abc2dq_current.c) annotation(
      Line(points = {{-20, 24}, {-10, 24}, {-10, 24}, {-10, 24}}, color = {0, 0, 255}));
    connect(abc2dq_current.pin3, rl1.pin3) annotation(
      Line(points = {{10, 36}, {20, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(abc2dq_current.pin2, rl1.pin2) annotation(
      Line(points = {{10, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(abc2dq_current.pin1, rl1.pin1) annotation(
      Line(points = {{10, 24}, {20, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    connect(pll_d.y, abc2dq_current.theta) annotation(
      Line(points = {{13, 68}, {20, 68}, {20, 48}, {0, 48}, {0, 42}}, color = {0, 0, 127}));
    connect(lc1.pin6, pll_d.a) annotation(
      Line(points = {{-20, 36}, {-16, 36}, {-16, 68}, {-8, 68}}, color = {0, 0, 255}));
    connect(lc1.pin5, pll_d.b) annotation(
      Line(points = {{-20, 30}, {-14, 30}, {-14, 62}, {-8, 62}, {-8, 62}, {-8, 62}}, color = {0, 0, 255}));
    connect(lc1.pin4, pll_d.c) annotation(
      Line(points = {{-20, 24}, {-12, 24}, {-12, 56}, {-8, 56}, {-8, 56}, {-8, 56}}, color = {0, 0, 255}));
    connect(realExpression1.y, feedback.u1) annotation(
      Line(points = {{19, 77}, {21.5, 77}, {21.5, 78}, {26, 78}}, color = {0, 0, 127}));
    connect(pll_d.d, feedback.u2) annotation(
      Line(points = {{14, 62}, {34, 62}, {34, 70}}, color = {0, 0, 127}));
    connect(pll_d.q, feedback1.u2) annotation(
      Line(points = {{14, 58}, {68, 58}, {68, 82}}, color = {0, 0, 127}));
    connect(realExpression.y, feedback1.u1) annotation(
      Line(points = {{14, 90}, {58, 90}, {58, 90}, {60, 90}}, color = {0, 0, 127}));
    connect(feedback.y, pi.u) annotation(
      Line(points = {{44, 78}, {48, 78}, {48, 12}, {-92, 12}, {-92, -4}, {-88, -4}, {-88, -4}}, color = {0, 0, 127}));
    connect(feedback1.y, PI.u) annotation(
      Line(points = {{78, 90}, {78, 54}, {50, 54}, {50, 10}, {-60, 10}, {-60, -22}, {-92, -22}, {-92, -34}, {-88, -34}}, color = {0, 0, 127}));
    connect(abc2dq_current.d, feedback3.u2) annotation(
      Line(points = {{-4, 20}, {-4, 20}, {-4, 16}, {-20, 16}, {-20, -16}, {-36, -16}, {-36, -12}, {-36, -12}}, color = {0, 0, 127}));
    connect(pi.y, feedback3.u1) annotation(
      Line(points = {{-64, -4}, {-46, -4}, {-46, -4}, {-44, -4}}, color = {0, 0, 127}));
    connect(abc2dq_current.q, feedback2.u2) annotation(
      Line(points = {{4, 20}, {4, 20}, {4, 14}, {-18, 14}, {-18, -46}, {-36, -46}, {-36, -40}, {-36, -40}, {-36, -40}}, color = {0, 0, 127}));
    connect(PI.y, feedback2.u1) annotation(
      Line(points = {{-64, -34}, {-50, -34}, {-50, -32}, {-44, -32}, {-44, -32}}, color = {0, 0, 127}));
    connect(feedback2.y, pi1.u) annotation(
      Line(points = {{-26, -32}, {-10, -32}}, color = {0, 0, 127}));
    connect(feedback3.y, PI1.u) annotation(
      Line(points = {{-26, -4}, {-10, -4}}, color = {0, 0, 127}));
    connect(pll_d.y, dq2abc.theta) annotation(
      Line(points = {{14, 68}, {20, 68}, {20, 48}, {42, 48}, {42, 4}, {32, 4}, {32, -6}, {32, -6}}, color = {0, 0, 127}));
    connect(pi1.y, dq2abc.q) annotation(
      Line(points = {{14, -32}, {20, -32}, {20, -22}, {24, -22}, {24, -22}, {26, -22}}, color = {0, 0, 127}));
    connect(PI1.y, dq2abc.d) annotation(
      Line(points = {{14, -4}, {22, -4}, {22, -14}, {26, -14}, {26, -14}}, color = {0, 0, 127}));
  connect(dq2abc.a, inverter1.u3) annotation(
      Line(points = {{46, -12}, {78, -12}, {78, -66}, {-116, -66}, {-116, 36}, {-80, 36}, {-80, 36}}, color = {0, 0, 127}));
  connect(dq2abc.b, inverter1.u2) annotation(
      Line(points = {{46, -18}, {72, -18}, {72, -58}, {72, -58}, {72, -60}, {-110, -60}, {-110, 30}, {-80, 30}, {-80, 30}}, color = {0, 0, 127}));
  connect(inverter1.u1, dq2abc.c) annotation(
      Line(points = {{-80, 24}, {-102, 24}, {-102, 24}, {-104, 24}, {-104, -54}, {66, -54}, {66, -24}, {46, -24}, {46, -24}}, color = {0, 0, 127}));
    annotation(
      Diagram);
  end network_ctrl;

  model sine_Test3
    Modelica.Blocks.Sources.Sine sine3(amplitude = 0.325, freqHz = 50) annotation(
      Placement(visible = true, transformation(origin = {-76, 14}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine4(amplitude = 0.325, freqHz = 50, phase = 2.0944) annotation(
      Placement(visible = true, transformation(origin = {-76, 48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Sources.Sine sine5(amplitude = 0.325, freqHz = 50, phase = 4.18879) annotation(
      Placement(visible = true, transformation(origin = {-76, 80}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.network_ctrl2 network_ctrl25 annotation(
      Placement(visible = true, transformation(origin = {62, 14}, extent = {{-50, -50}, {50, 50}}, rotation = 0)));
  equation
    connect(sine3.y, network_ctrl25.i1p1) annotation(
      Line(points = {{-64, 14}, {-12, 14}, {-12, 22}, {10, 22}, {10, 24}}, color = {0, 0, 127}));
    connect(sine4.y, network_ctrl25.i1p2) annotation(
      Line(points = {{-64, 48}, {-38, 48}, {-38, 28}, {8, 28}, {8, 30}, {10, 30}}, color = {0, 0, 127}));
    connect(sine5.y, network_ctrl25.i1p3) annotation(
      Line(points = {{-64, 80}, {-12, 80}, {-12, 36}, {10, 36}, {10, 36}, {10, 36}}, color = {0, 0, 127}));
  end sine_Test3;

  model network_ctrl2
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p1 annotation(
      Placement(visible = true, transformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 18}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p2 annotation(
      Placement(visible = true, transformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 30}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput i1p3 annotation(
      Placement(visible = true, transformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0), iconTransformation(origin = {-104, 42}, extent = {{-8, -8}, {8, 8}}, rotation = 0)));
    grid.loads.rl rl1 annotation(
      Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
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
      Line(points = {{-20, 36}, {20, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(lc1.pin5, rl1.pin2) annotation(
      Line(points = {{-20, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(lc1.pin4, rl1.pin1) annotation(
      Line(points = {{-20, 24}, {20, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
    annotation(
      Diagram);
  end network_ctrl2;
  
  model network_ctrl_lim
    grid.inverters.inverter inverter1 annotation(
      Placement(visible = true, transformation(origin = {-70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.ideal_filter.lc lc1 annotation(
      Placement(visible = true, transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.loads.rl rl1 annotation(
      Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.plls.pll_d pll_dq annotation(
      Placement(visible = true, transformation(origin = {2, 62}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    grid.transforms.abc2dq_current abc2dq_current annotation(
      Placement(visible = true, transformation(origin = {0, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression annotation(
      Placement(visible = true, transformation(origin = {2, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression1(y = 230 * 1.41427)  annotation(
      Placement(visible = true, transformation(origin = {5, 77}, extent = {{-13, -11}, {13, 11}}, rotation = 0)));
  grid.transforms.dq2abc dq2abc annotation(
      Placement(visible = true, transformation(origin = {36, -18}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID pid(Td = 0, Ti = 4.17, k = 0.025, limitsAtInit = true, yMax = 35)  annotation(
      Placement(visible = true, transformation(origin = {78, 86}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID(Td = 0, Ti = 4.17, k = 0.025, limitsAtInit = true, yMax = 30) annotation(
      Placement(visible = true, transformation(origin = {44, 74}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID1(Td = 0, Ti = 1.33, k = 0.012, limitsAtInit = true, yMax = 1 / 2.8284, yMin = -1 / 2.8284) annotation(
      Placement(visible = true, transformation(origin = {-34, -34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Continuous.LimPID PID2(Td = 0, Ti = 1.33, k = 0.012, limitsAtInit = true, yMax = 1 / 2.8284, yMin = -1 / 2.8284) annotation(
      Placement(visible = true, transformation(origin = {-34, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(inverter1.pin3, lc1.pin3) annotation(
      Line(points = {{-60, 36}, {-40, 36}}, color = {0, 0, 255}));
    connect(inverter1.pin2, lc1.pin2) annotation(
      Line(points = {{-60, 30}, {-40, 30}}, color = {0, 0, 255}));
    connect(inverter1.pin1, lc1.pin1) annotation(
      Line(points = {{-60, 24}, {-40, 24}}, color = {0, 0, 255}));
    connect(lc1.pin6, abc2dq_current.a) annotation(
      Line(points = {{-20, 36}, {-10, 36}, {-10, 36}, {-10, 36}}, color = {0, 0, 255}));
    connect(lc1.pin5, abc2dq_current.b) annotation(
      Line(points = {{-20, 30}, {-10, 30}, {-10, 30}, {-10, 30}}, color = {0, 0, 255}));
    connect(lc1.pin4, abc2dq_current.c) annotation(
      Line(points = {{-20, 24}, {-10, 24}, {-10, 24}, {-10, 24}}, color = {0, 0, 255}));
    connect(abc2dq_current.pin3, rl1.pin3) annotation(
      Line(points = {{10, 36}, {20, 36}, {20, 36}, {20, 36}}, color = {0, 0, 255}));
    connect(abc2dq_current.pin2, rl1.pin2) annotation(
      Line(points = {{10, 30}, {20, 30}, {20, 30}, {20, 30}}, color = {0, 0, 255}));
    connect(abc2dq_current.pin1, rl1.pin1) annotation(
      Line(points = {{10, 24}, {20, 24}, {20, 24}, {20, 24}}, color = {0, 0, 255}));
  connect(lc1.pin6, pll_dq.a) annotation(
      Line(points = {{-20, 36}, {-16, 36}, {-16, 68}, {-8, 68}}, color = {0, 0, 255}));
  connect(lc1.pin5, pll_dq.b) annotation(
      Line(points = {{-20, 30}, {-14, 30}, {-14, 62}, {-8, 62}, {-8, 62}, {-8, 62}}, color = {0, 0, 255}));
  connect(lc1.pin4, pll_dq.c) annotation(
      Line(points = {{-20, 24}, {-12, 24}, {-12, 56}, {-8, 56}, {-8, 56}, {-8, 56}}, color = {0, 0, 255}));
    connect(dq2abc.a, inverter1.u3) annotation(
      Line(points = {{46, -12}, {78, -12}, {78, -66}, {-116, -66}, {-116, 36}, {-80, 36}, {-80, 36}}, color = {0, 0, 127}));
    connect(dq2abc.b, inverter1.u2) annotation(
      Line(points = {{46, -18}, {72, -18}, {72, -58}, {72, -58}, {72, -60}, {-110, -60}, {-110, 30}, {-80, 30}, {-80, 30}}, color = {0, 0, 127}));
    connect(inverter1.u1, dq2abc.c) annotation(
      Line(points = {{-80, 24}, {-102, 24}, {-102, 24}, {-104, 24}, {-104, -54}, {66, -54}, {66, -24}, {46, -24}, {46, -24}}, color = {0, 0, 127}));
  connect(pll_dq.d, PID.u_m) annotation(
      Line(points = {{14, 62}, {32, 62}, {32, 60}, {40, 60}, {40, 58}, {44, 58}, {44, 62}, {44, 62}}, color = {0, 0, 127}));
    connect(realExpression1.y, PID.u_s) annotation(
      Line(points = {{20, 78}, {30, 78}, {30, 74}, {32, 74}}, color = {0, 0, 127}));
  connect(pll_dq.q, pid.u_m) annotation(
      Line(points = {{14, 56}, {78, 56}, {78, 72}, {78, 72}, {78, 74}}, color = {0, 0, 127}));
    connect(realExpression.y, pid.u_s) annotation(
      Line(points = {{14, 90}, {58, 90}, {58, 86}, {66, 86}, {66, 86}}, color = {0, 0, 127}));
    connect(pid.y, PID2.u_s) annotation(
      Line(points = {{90, 86}, {92, 86}, {92, 12}, {-56, 12}, {-56, -4}, {-46, -4}, {-46, -4}}, color = {0, 0, 127}));
    connect(abc2dq_current.d, PID2.u_m) annotation(
      Line(points = {{-4, 20}, {-4, 20}, {-4, 16}, {-58, 16}, {-58, -20}, {-34, -20}, {-34, -16}, {-34, -16}}, color = {0, 0, 127}));
    connect(PID2.y, dq2abc.d) annotation(
      Line(points = {{-22, -4}, {16, -4}, {16, -14}, {26, -14}, {26, -14}}, color = {0, 0, 127}));
    connect(PID.y, PID1.u_s) annotation(
      Line(points = {{56, 74}, {60, 74}, {60, 14}, {-60, 14}, {-60, -34}, {-46, -34}, {-46, -34}}, color = {0, 0, 127}));
    connect(abc2dq_current.q, PID1.u_m) annotation(
      Line(points = {{4, 20}, {4, 20}, {4, 10}, {-20, 10}, {-20, -50}, {-34, -50}, {-34, -46}, {-34, -46}, {-34, -46}}, color = {0, 0, 127}));
  connect(PID1.y, dq2abc.q) annotation(
      Line(points = {{-22, -34}, {10, -34}, {10, -22}, {26, -22}, {26, -22}}, color = {0, 0, 127}));
  connect(pll_dq.theta, abc2dq_current.theta) annotation(
      Line(points = {{14, 68}, {22, 68}, {22, 48}, {0, 48}, {0, 42}, {0, 42}, {0, 42}}, color = {0, 0, 127}));
  connect(pll_dq.theta, dq2abc.theta) annotation(
      Line(points = {{14, 68}, {22, 68}, {22, 48}, {44, 48}, {44, 0}, {32, 0}, {32, -6}, {32, -6}}, color = {0, 0, 127}));
    annotation(
      Diagram);
  end network_ctrl_lim;

  model testpi
  Modelica.Blocks.Continuous.PI pi(T = 0.001, k = 1)  annotation(
      Placement(visible = true, transformation(origin = {36, 36}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = 3)  annotation(
      Placement(visible = true, transformation(origin = {-34, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    connect(realExpression.y, pi.u) annotation(
      Line(points = {{-22, 44}, {8, 44}, {8, 36}, {24, 36}, {24, 36}}, color = {0, 0, 127}));
  end testpi;
  annotation(
    uses(Modelica(version = "3.2.3")));
end grid;
