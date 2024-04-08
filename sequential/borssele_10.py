# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 09:31:19 2022

@author: juru
"""

import numpy as np

# Set up IEA Wind 10 MW Reference Turbine
from py_wake.wind_turbines import OneTypeWindTurbines
rated_power = 10 #MW


ct_curve = np.array([[4, 0.927057057803911], 
                    [5, 0.92026094738932], 
                    [6, 0.874953044625167], 
                    [7, 0.85837496336839], 
                    [8, 0.841138019072282], 
                    [9, 0.79462070404387], 
                    [9.5, 0.75889164978347], 
                    [10, 0.725733585868264], 
                    [10.5, 0.694978605245016], 
                    [11, 0.597310898042545], 
                    [11.5, 0.493702775004447], 
                    [12, 0.420621597422679], 
                    [13, 0.319501163417707], 
                    [14, 0.251839216314048], 
                    [15, 0.203566086419764], 
                    [16, 0.167756545404188], 
                    [18, 0.119172464082921], 
                    [20, 0.0886910505776589], 
                    [25, 0.0489237513003403]])

power_curve = np.array([[0, 0],
                        [3.99, 0],
                        [4, 0.0417775416641181 * rated_power], 
                        [5, 0.100966625080239 * rated_power], 
                        [6, 0.184703860860372 * rated_power], 
                        [7, 0.29363066617047 * rated_power], 
                        [8, 0.437408914882465 * rated_power], 
                        [9, 0.619863166839864 * rated_power], 
                        [9.5, 0.722289307327511 * rated_power], 
                        [10, 0.830738083625031 * rated_power], 
                        [10.5, 0.945267874693394 * rated_power], 
                        [11, rated_power], 
                        [11.5, rated_power], 
                        [12, rated_power], 
                        [13, rated_power], 
                        [14, rated_power], 
                        [15, rated_power], 
                        [16, rated_power], 
                        [18, rated_power], 
                        [20, rated_power], 
                        [24.99, rated_power],
                        [25, 0]])


class IEA10MW(OneTypeWindTurbines):
    '''
    Data from:
    Christian Bak, Frederik Zahle, Robert Bitsche, Taeseong Kim, Anders Yde, Lars Christian Henriksen, Anand Natarajan,
    Morten Hartvig Hansen.“Description of the DTU 10 MW Reference Wind Turbine” DTU Wind Energy Report-I-0092, July 2013. Table 3.5

    '''

    def __init__(self):
        self.cut_in=4
        self.rated=11
        self.cut_out=25
        self.rated_rpm=8.68
        OneTypeWindTurbines.__init__(
            self,
            'IEA10MW',
            diameter=190.6,
             hub_height=119.,
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
        return np.interp(u, power_curve[:, 0], power_curve[:, 1])
    
    def nominal_power(self):
        nom_power=rated_power
        return nom_power
    

    def power_all(self,wind_speeds):
        pc=np.zeros(len(wind_speeds))
        for i in range(len(wind_speeds)):
            pc[i]=self._power(wind_speeds[i])
        return pc  