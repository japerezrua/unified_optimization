#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:52:01 2020

@author: katdyk
"""

# Set up site conditions
from py_wake.site import UniformWeibullSite
from py_wake.site.shear import PowerShear
import numpy as np

# Set up IEA Wind 10 MW Reference Turbine
from py_wake.wind_turbines import OneTypeWindTurbines
rated_power = 3.35 #MW

ct_curve = np.array([[9.8, 8/9]])


class IEA3_5MW(OneTypeWindTurbines):
    '''
    Data from:
    Christian Bak, Frederik Zahle, Robert Bitsche, Taeseong Kim, Anders Yde, Lars Christian Henriksen, Anand Natarajan,
    Morten Hartvig Hansen.“Description of the DTU 10 MW Reference Wind Turbine” DTU Wind Energy Report-I-0092, July 2013. Table 3.5

    '''

    def __init__(self):
        self.cut_in=4
        self.rated=9.8
        self.cut_out=25
        self.rated_rpm=11.75
        OneTypeWindTurbines.__init__(
            self,
            'IEA3_35MW',
            diameter=130,
             hub_height=110.,
            ct_func=self._ct,
            power_func=self._power,
            power_unit='MW')
    def _ct(self, u):
        return np.interp(u, ct_curve[:, 0], ct_curve[:, 1])
    
    def ct_all(self,wind_speeds):
        ct=np.zeros(len(wind_speeds))
        for i in range(len(wind_speeds)):
            ct[i]=np.interp(wind_speeds[i], ct_curve[:, 0], ct_curve[:, 1])
        return ct    

    def _power(self, u):
        if u<self.cut_in:
            power=0
        elif u>=self.cut_in and u<self.rated:
            power=rated_power*((u-self.cut_in)/(self.rated-self.cut_in))**3
        elif u>=self.rated and u<self.cut_out:    
            power=rated_power         
        elif u>=self.cut_out:
            power=0
        return power
    def nominal_power(self):
        nom_power=rated_power
        return nom_power
    
    def power_curve(self):
        u=np.linspace(0,25,100000)
        power_curve=np.zeros((len(u),2))
        for i in range(len(u)):
            power_curve[i,0]=u[i]
            power_curve[i,1]=self._power(u[i])
        return power_curve
    def power_all(self,wind_speeds):
        pc=np.zeros(len(wind_speeds))
        for i in range(len(wind_speeds)):
            pc[i]=self._power(wind_speeds[i])
        return pc  




def main():
    wt = IEA3_5MW()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-')
    plt.show()


if __name__ == '__main__':
    main()
