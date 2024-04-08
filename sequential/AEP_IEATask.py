# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:27:18 2021

@author: juru
"""

import numpy as np
from matrix_rotation import coordinates_rotation
from velocity_deficit import bil
import math

def aep_calculator(X_wt,Y_wt,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter):
    length_points=len(X_wt)
    number_freqs,number_vels=len(freq_input_r),len(vel_set)
    power=0
    for j in range(number_freqs):
        wt_points_freq_t=np.zeros((length_points,2))
        for i in range(length_points):
            xt,yt=coordinates_rotation(freq_input_r[j,0],X_wt[i,0],Y_wt[i,0])
            wt_points_freq_t[i,:]=np.array([xt,yt]).reshape(1,2)      
        index_sorted=wt_points_freq_t[:,0].argsort()
        wt_points_freq_t=wt_points_freq_t[index_sorted,:]
        for k in range(number_vels):
            vel_each_wt=np.zeros(length_points)
            for ind in range(len(index_sorted)):
                affecting=index_sorted[:ind]
                vel_each_wt[ind]=vel_set[k]
                deficit=0
                for wts_affecting in range(len(affecting)):
                    c_t=ct_all[k]#Dependent on wind velocity
                    bil_o=bil(c_t,k_w,wt_points_freq_t[ind,0],wt_points_freq_t[wts_affecting,0],diameter,
                              wt_points_freq_t[ind,1],wt_points_freq_t[wts_affecting,1])
                    deficit+=(vel_set[k]*bil_o)**2
                vel_each_wt[ind]-=math.sqrt(deficit)
                power+=8760*freq_input_r[j,1]*prob_vel_set[j][k]*wt_model._power(vel_each_wt[ind])
    return power

def aep_calculator_mo_wakes(X_wt,Y_wt,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter):
    length_points=len(X_wt)
    number_freqs,number_vels=len(freq_input_r),len(vel_set)
    power=0
    for j in range(number_freqs):
        wt_points_freq_t=np.zeros((length_points,2))
        for i in range(length_points):
            xt,yt=coordinates_rotation(freq_input_r[j,0],X_wt[i,0],Y_wt[i,0])
            wt_points_freq_t[i,:]=np.array([xt,yt]).reshape(1,2)      
        index_sorted=wt_points_freq_t[:,0].argsort()
        wt_points_freq_t=wt_points_freq_t[index_sorted,:]
        for k in range(number_vels):
            vel_each_wt=np.zeros(length_points)
            for ind in range(len(index_sorted)):
                affecting=index_sorted[:ind]
                vel_each_wt[ind]=vel_set[k]
                deficit=0
                for wts_affecting in range(len(affecting)):
                    c_t=ct_all[k]#Dependent on wind velocity
                    bil_o=0
                    deficit+=(vel_set[k]*bil_o)**2
                vel_each_wt[ind]-=math.sqrt(deficit)
                power+=8760*freq_input_r[j,1]*prob_vel_set[j][k]*wt_model._power(vel_each_wt[ind])
    return power
