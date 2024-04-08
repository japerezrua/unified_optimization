# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:48:37 2022

@author: juru
"""




def irr_dtu(aep, electrical_connection_cost,ee_dtu,rated_rpm_array,Drotor_vector,Power_rated_array,hub_height_vector,water_depth_array):
    ee_dtu.calculate_irr(
                    rated_rpm_array, 
                    Drotor_vector, 
                    Power_rated_array,
                    hub_height_vector, 
                    water_depth_array, 
                    aep, 
                    electrical_connection_cost)
    return ee_dtu.IRR
