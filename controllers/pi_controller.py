# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:01:39 2020

@author: jarren
"""

class PIController:
    """
    Implements a basic PI controller.
    Uses back calculation for anti-windup 

    """

    def __init__(self,PI_param, ts):
        """
        :param PI_param: The PI_Parameters object with the PI controller
        parameters
        """
        self._params = PI_param
        self.integralSum = 0
        self.windup_compensation = 0
        self._ts =ts

    def reset(self):
        self.integralSum = 0
        self._subsample_count=0
        
    def step(self, error):
        """
        implements a step of a basic PI controller with anti-windup by back-calculation
        
        
        :param error: control error to act on 
        
        :return: the calculated PI controller response to the error, using the
                PI_Parameters provided during initialisation.
        """
                
        self.integralSum = self.integralSum + (self._params.kI*error + self.windup_compensation)*self._ts
        output= self._params.kP*error + self.integralSum
        preout = output
        
        #Limit output to exactly the limit
        if output>self._params.upper_limit:
            output=self._params.upper_limit
        elif output<self._params.lower_limit:
            output=self._params.lower_limit
                
        #Calculate the saturation error (gets added to the integral error in 
        #the next step of the controller)
        self.windup_compensation = (output-preout)*self._params.kB
            
        return output     
        
    
    
class PI_parameters:
    """
    The params for a basic PI Controller

    """
    

    def __init__(self,kP, kI, uL, lL, kB = 1):
        """
        :param kP: proportional gain constant
        :param kI: integral gain constant
        :param uL: upper limit of controller output
        :param lL: lower limit of controller output
        :param kB: anti-windup via back calculation gain
        :param ts: The discrete sampling time of the controller
        """
        self.kP = kP
        self.kI = kI
        self.upper_limit = uL
        self.lower_limit = lL
        self.kB = kB