# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:13:21 2020

@author: jarren

Just contains the definitions of the initialisation parameters used in the 
design of controllers and or the filters.
"""
from .pi_controller import PI_parameters

        
class FiltParams:
    """
    Implements a basic P,f droop controller
    
    """
    def __init__(self, gain, tau):
        """
        :param Droop: The droop gain
        :param tau: The first order time constant [s]
        :param nomValue: An offset to add to the output of the droop
        :param ts: The discrete sampling frequency
        
            
        """
        
        self.gain=gain
        self.tau=tau
            

class DroopParams(FiltParams):
    """
    Implements a basic P,f droop controller
    
    """
    def __init__(self, Droop, tau, nomValue = 0):
        """
        :param Droop: The droop gain
        :param tau: The first order time constant [s]
        :param nomValue: An offset to add to the output of the droop
        :param ts: The discrete sampling frequency
        
        EG for a P-f droop controller (for voltage forming inverter)
        Inverter of 10kW, droop of 10% , timeConstant of 1 sec, 50Hz
            Droop = 1000 [Watt/Hz]
            tau = 1
            nomValue = 50 [Hz]
            
        """
        gain=0
        if Droop!= 0 :
            gain=1/Droop
        
        super().__init__(gain,tau)
        
        self.nom_val = nomValue

class InverseDroopParams(DroopParams):
    """
    Implements a basic P,f droop controller
    
    """
    def __init__(self, Droop, tau, nomValue = 0, tau_filt=0):
        """
        :param Droop: The droop gain
        :param tau: The first order time constant [s]
        :param nomValue: An offset to add to the output of the droop
        :param ts: The discrete sampling frequency
        
        EG for a P-f droop controller (for voltage forming inverter)
        Inverter of 10kW, droop of 10% , timeConstant of 1 sec, 50Hz
            Droop = 1000 [Watt/Hz]
            tau = 1
            nomValue = 50 [Hz]
            
        """
        self.derivativeFiltParams = FiltParams(1,tau_filt)

        super().__init__(Droop,tau,nomValue)
        
               
class PLLParams(PI_parameters):
    """
    The params for a basic PI Controller

    """
    def __init__(self,kP, kI, uL, lL, kB = 1, f_nom = 0, theta_0 = 0):
        """
        :param kP: proportional gain constant
        :param kI: integral gain constant
        :param uL: upper limit of controller output
        :param lL: lower limit of controller output
        :param kB: anti-windup via back calculation gain
        :param ts: The discrete sampling time of the controller
        """
        super().__init__(kP, kI, uL, lL, kB)
        self.f_nom = f_nom
        self.theta_0 = theta_0