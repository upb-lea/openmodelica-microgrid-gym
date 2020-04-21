Controller Tuning Hints
=======================

1. Current controller of primary inverter

-  With no droop, in other words constant mains frequency, apply a short
   circuit to inverter and tune Kp, Ki of the current controller. Can
   tune Kp then Ki, finally Kp again if both can’t be tuned at the same
   time.

   -  Try tune for 90-95% of max current. Not peak current limit, but
      maximum allowed during nominal operation.
   -  Ensure that the tuning does not allow the current to exceed the
      peak current limit. In real world scenarios this is when an
      inverter will shutoff or explode.

2. Voltage controller of primary inverter

-  With no droop and an open circuit load of just the inverter tune Kp,
   Ki of the voltage controller. Can tune Kp then Ki, finally Kp again
   if both can’t be tuned at the same time.

3. PLL of secondary inverter

-   With primary inverter providing a constant frequency start tuning
    Kp, Ki of the PLL. Noise on the frequency output of the PLL is
    acceptable, as long as the output phasors of the PLL accurately
    match the incoming voltage signal.
-   The secondary inverter power electronics should be
    disconnected/open-circuit for this.
-   If possible inject step changes to the frequency setpoint of the
    primary inverter, watching how accurately the PLL tracks the
    external voltage reference. Continue tuning if necessary.

4. Current controller of secondary inverter

-   With the PLL of the secondary inverter locked to the primary inverter
    tune the Kp, Ki of the current controller. No droops at this stage.

   -  Might need to create a step change for the setpoint for this to
      test accurate tracking.
   -  Try tune for 90-95% of max current. Not peak current limit, but
      maximum allowed during nominal operation.

5. Droop of the primary inverter

-  This isn’t really a tuning step as the parameters won’t affect any
   other system.

6. Droops of the secondary inverter

-  Firstly, the filter for the frequency and the voltage feedback to the
   droop controllers should be set before tuning the droop.
-  Tune the droop controllers of the secondary inverter.

   -  The frequency of the filter (for the frequency feedback from the
      PLL) affects the droop parameters and should thus be considered in
      the tuning.


