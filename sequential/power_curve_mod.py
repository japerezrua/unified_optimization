# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:38:46 2021

@author: juru
"""
import matplotlib.pylab as plt
import numpy as np

def partitioning_cons_pc(wt_model,m,u_in,u_rated,u_out):
    delta_u=(u_rated-u_in)/m
    
    power=wt_model.nominal_power() #nominal power in MW
    power_curve=wt_model.power_curve()
    plt.figure()
    plt.plot(power_curve[:,0],power_curve[:,1],label='Original power curve',color='black')
    
    u_set=np.zeros((m+2,1))
    for i in range(1,len(u_set)-1):
        u_set[i]=u_in+(i-1)*delta_u
    u_set[-1]=u_rated
    ave_wind=np.zeros((m,1))
    for i in range(1,len(ave_wind)+1):
        ave_wind[i-1]=0.5*(u_set[i]+u_set[i+1])
        
    wind_speeds=np.concatenate((u_set,np.array([u_out]).reshape(1,1)))
    pow_vals=wt_model.power_all(ave_wind).reshape(-1,1)   
    power_disc=np.concatenate((np.array([[0]]),pow_vals,np.array([[power]]),np.array([[power]])))
    
    plt.xlabel('Wind speed [m/s]',fontsize=35)
    plt.ylabel('Power [MW]',fontsize=35)
    plt.tick_params(axis="y", labelsize=35)
    plt.tick_params(axis="x", labelsize=35)
    plt.axvline(x=u_in, label='Cut-in speed',linestyle='dashed',color='green')
    plt.axvline(x=u_rated, label='Rated speed',linestyle='dashed',color='blue')
    plt.axvline(x=u_out, label='Cut-out speed',linestyle='dashed',color='red')
    plt.step(wind_speeds,power_disc, label='Step-wise function',where='post',color='gray',linestyle='dashdot')
    plt.scatter(ave_wind, pow_vals,color='grey', alpha=0.3,label='Extracted value')
    plt.legend(fontsize=11) 
    
    wind_speeds
    power_disc=power_disc[:-1]
    return wind_speeds,power_disc
if __name__ == "__main__":
    from borssele import IEA10MW

    
    wt_model=IEA10MW() #Defining object of the wind turbine
    m=5
    u_in=3
    u_rated=11
    u_out=25
    
    wind_speeds,power_disc=partitioning_cons_pc(wt_model,m,u_in,u_rated,u_out)
    