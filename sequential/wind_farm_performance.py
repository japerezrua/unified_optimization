# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 08:55:37 2021

@author: juru
"""

import numpy as np
import numpy_financial as npf

def aep(solution,WTn,var_power_output_length,freq_input_r,vel_set,prob_vel_set,wt_model):
    active_wts=np.array(solution[:WTn])>0.9 #Indexes of active WTs
    indexes_active_wts=np.argwhere(active_wts==True)
    total_power=np.zeros(len(indexes_active_wts))
    for pos,i in enumerate(indexes_active_wts):
        i=int(i)
        for j in range(len(freq_input_r)):
            for k in range(len(vel_set)):
                velocity=solution[WTn+var_power_output_length+(i*2*len(freq_input_r)*len(vel_set))+(j*len(vel_set))+(k)]
                power=wt_model._power(velocity)
                total_power[pos]+=8760*freq_input_r[j,1]*prob_vel_set[j][k]*power
    """
    Xwt,Ywt=Xwt_i[np.argwhere(active_wts==True)],Ywt_i[np.argwhere(active_wts==True)] #Location actives WTs
    active_wts=np.argwhere(active_wts==True)+OSSn+1 #ID active WTs
    wt_points_active=np.concatenate((Xwt,Ywt),axis=1) #Matrix with location actives WTs 
    wt_points_freq_active,sorted_indexes_active=coordinates_translation(wt_points_active,freq_input_r) #Locations rotated for each freq.
    aep_owf=0
    for j in range(wt_points_freq_active.shape[2]):
        for v in range(len(vel_set)):
            squared_def_j_v=vel_deficit_aep(sorted_indexes_active,wt_points_freq_active,ct_all,d,k,j,v)
            power_j_v=0
            for i in range(wt_points_freq_active.shape[0]):
                tot_def_squared=sum(squared_def_j_v[i,:])
                tot_def=math.sqrt(tot_def_squared)
                vel_i_j_v=vel_set[v]*(1-tot_def)
                power_gen_i_j_v=wt_model._power(vel_i_j_v)
                power_j_v+=power_gen_i_j_v
            aep_owf+=8760*freq_input_r[j,1]*prob_vel_set[j][v]*power_j_v
    """
    nominal_aep_owf=len(indexes_active_wts)*wt_model.nominal_power()*8760 #MWh
    capacity_factor=sum(total_power)*100/nominal_aep_owf    
    return total_power,sum(total_power),capacity_factor

def npv(collection_system_cost,aep_owf,discount_rate,lifetime,price_energy):
    cashflows=[-collection_system_cost]+[aep_owf*price_energy for x in range(lifetime)]
    net_present_value=npf.npv(discount_rate/100,cashflows)
    return net_present_value/1e6

#import matplotlib.pylab as plt
#import numpy as np

#power=np.zeros(len(vel_set))
#for v in range(len(vel_set)):
#    power[v]=wt_model._power(vel_set[v])

#plt.figure()    
#plt.plot(vel_set,power)