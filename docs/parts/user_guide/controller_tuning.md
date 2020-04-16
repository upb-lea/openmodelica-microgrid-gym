#  Controller tuning hints


1. Current controller of primary inverter
    1. With no droop, in other words constant mains frequency, apply a short circuit to inverter and tune K_p,K_i of the current controller. Can tune K_p then〖 K〗_i, finally K_p again if both can’t be tuned at the same time. 
	    1. Try tune for 90-95% of max current. Not peak current limit, but maximum allowed during nominal operation.
	    2. Ensure that the tuning does not allow the current to exceed the peak current limit. In real world scenarios this is when an inverter will shutoff or explode.
2. 	Voltage controller of primary inverter
        1. With no droop and an open circuit load of just the inverter tune K_p,K_i of the current controller. Can tune K_p then〖 K〗_i, finally K_p again if both can’t be tuned at the same time.
3. 	PLL of Secondary inverter
	    1. With primary inverter providing a constant frequency start tuning K_p,K_i of the PLL. Noise on the frequency output of the PLL is acceptable, as long as the output phasors of the PLL accurately match the incoming voltage signal.
	        2. The secondary inverter power electronics should be disconnected/open-circuit for this.
	    2. If possible inject step changes to the frequency setpoint of the primary inverter, watching how accurately the PLL tracks the external voltage reference. Continue tuning if necessary.
4. 	Current controller of secondary inverter
	    1. With the PLL of the secondary inverter locked to the primary inverter tune the K_p,K_i of the current controller. No droops at this stage.
	        1. Might need to create a step change for the setpoint for this to test accurate tracking. 
	        2. Try tune for 90-95% of max current. Not peak current limit, but maximum allowed during nominal operation.
5. 	Droop of the primary inverter
	    1. This isn’t really a tuning step as the parameters won’t affect any other systems. 
6.	Droops of the secondary inverter
	    1. Firstly the filter for the frequency and the voltage feedback to the droop controllers should be set before tuning the droop.
	    2. Tune the droop controllers of the secondary inverter.
	        1. The frequency of the filter (for the frequency feedback from the PLL) affects the droop parameters and should thus be considered in the tuning.
