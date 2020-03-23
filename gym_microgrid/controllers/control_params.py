# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:13:21 2020

@author: jarren

Just contains the definitions of the initialisation parameters used in the 
design of controllers and or the filters.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FiltParams:
    """
    Implements a basic P,f droop controller
    
    """
    gain: float
    tau: float


class DroopParams(FiltParams):
    """
    Implements a basic P,f droop controller
    """

    def __init__(self, gain, tau, nomValue=0):
        """
        :param gain: The droop gain
        :param tau: The first order time constant [s]
        :param nomValue: An offset to add to the output of the droop
        
        EG for a P-f droop controller (for voltage forming inverter)
        Inverter of 10kW, droop of 10% , timeConstant of 1 sec, 50Hz
            Droop = 1000 [Watt/Hz]
            tau = 1
            nomValue = 50 [Hz]
            
        """
        if gain != 0:
            gain = 1 / gain
        else:
            gain = 0

        super().__init__(gain, tau)
        self.nom_val = nomValue


class InverseDroopParams(DroopParams):
    """
    Implements a basic P,f droop controller
    """

    def __init__(self, droop, tau, nomValue=0, tau_filt=0):
        """
        :param droop: The droop gain
        :param tau: The first order time constant [s]
        :param nomValue: An offset to add to the output of the droop
        :param tau_filt: timeresolution for filter
        
        EG for a P-f droop controller (for voltage forming inverter)
        Inverter of 10kW, droop of 10% , timeConstant of 1 sec, 50Hz
            Droop = 1000 [Watt/Hz]
            tau = 1
            nomValue = 50 [Hz]
        """

        super().__init__(droop, tau, nomValue)
        self.derivativeFiltParams = FiltParams(1, tau_filt)


@dataclass
class PI_params:
    """
    The params for a basic PI Controller
    """
    kP: float
    kI: float
    limits: Tuple[float, float]
    kB: float = 1


@dataclass
class PLLParams(PI_params):
    """
    The params for a basic PI Controller
    """

    f_nom: float = 0
    theta_0: float = 0
