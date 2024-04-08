# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 11:17:57 2021

@author: juru
"""

from scipy.stats import weibull_min
#import matplotlib.pyplot as plt
import numpy as np
def wind_speed_i(shape=2.28,scale=10.08803,min_ws=4,max_ws=25,wind_steps=1):
    #Wind speed PMF: Discretization from PDF
    #c=shape
    #scale=scale
    #x = np.linspace(0.1, max_ws, max_bis_ws)
    x=np.arange(start=min_ws,stop=max_ws+wind_steps,step=wind_steps)
    x=np.concatenate((np.array([0]),x))
    x[-1]=max_ws
    weibull=weibull_min(c=shape,scale=scale)
    #Process
    a=weibull.pdf(x)
    #plt.plot(x,a)
    prob=np.zeros(len(x)-1)
    vel=np.zeros(len(x)-1)
    for i in range(len(x)-1):
        prob[i]=a[i]*(x[i+1]-x[i])
        vel[i]=x[i+1]
    #plt.stem(vel,prob, basefmt=" ",bottom=0)
    #plt.xlim(0, 30)
    #plt.ylim(0, 0.1)
    #if abs(1-sum(prob))<0.0001:
    #    print('Succesful discretization')
    return vel,prob
def wind_direction_t(shape,scale,freq_input,freq_bins):
    #Process
    if len(freq_input)!=freq_bins:
        multiplier=int((360/len(freq_input)))
        if multiplier-(360/len(freq_input))!=0:
            raise Exception('This is programmed for frequency set size multiple of 360')
        freq_input_t,shape_t,scale_t=np.zeros((360,2)),np.zeros((360,1)),np.zeros((360,1))
        for i in range(len(freq_input)):
            f=freq_input[i]/multiplier
            sh_w=shape[i]
            sc_w=scale[i]
            for j in range(i*multiplier,(i+1)*multiplier):
                freq_input_t[j,0],freq_input_t[j,1]=j,f
                shape_t[j],scale_t[j]=sh_w,sc_w
        multiplier_r=(360/freq_bins)   
        int_mult,dec_mult=divmod(multiplier_r,1)
        dec_mult=round(dec_mult,4)
        #num_dec=str(dec_mult)[::-1].find('.')
        shape_r,scale_r,freq_input_r=np.zeros((freq_bins,1)),np.zeros((freq_bins,1)),np.zeros((freq_bins,2))
        for i in range(freq_bins):
            freq_input_r[i,0]=i*(int_mult+dec_mult)
        for i in range(freq_bins):
            a=freq_input_r[i,0]
            a_i,a_d=divmod(a,1)
            a_d=round(a_d,4)
            a_i=int(a_i)
            if a_i!=0:
               a_i+=1
               extra1=freq_input_t[a_i-1,1]*(1-a_d)
               extra11=shape_t[a_i-1]*(1-a_d)
               extra111=scale_t[a_i-1]*(1-a_d)
            else:
                extra1=0
                extra11=0
                extra111=0           
            if i==freq_bins-1:
                b_i,b_d=360,0
                extra2=0
                extra22=0
                extra222=0
            else:
                b=freq_input_r[i+1,0]
                b_i,b_d=divmod(b,1)
                b_i=int(b_i)
                b_d=round(b_d,4) 
                if b_i<=358:
                    extra2=freq_input_t[b_i,1]*(b_d)
                    extra22=shape_t[b_i]*(b_d)
                    extra222=scale_t[b_i]*(b_d)
                else:
                    extra2=0
                    extra22=0
                    extra222=0      
            freq_input_r[i,1]=sum(freq_input_t[a_i:b_i,1])+extra1+extra2
            shape_r[i]=(sum(shape_t[a_i:b_i])+extra11+extra22)/(b_i-a_i+(1-a_d)+b_d)
            scale_r[i]=(sum(scale_t[a_i:b_i])+extra111+extra222)/(b_i-a_i+(1-a_d)+b_d)
            if a_i==0:
                shape_r[i]=(sum(shape_t[a_i:b_i])+extra11+extra22)/(b_i-a_i+b_d)
                scale_r[i]=(sum(scale_t[a_i:b_i])+extra111+extra222)/(b_i-a_i+b_d)
    else:
        shape_r,scale_r,freq_input_r=np.copy(shape).reshape(-1,1),np.copy(scale).reshape(-1,1),np.copy(freq_input)
        freq_input_r=np.concatenate((np.linspace(start=0,stop=360, num=freq_bins, endpoint=False).reshape(-1,1),freq_input_r.reshape(-1,1)),axis=1)
    #Output
    return shape_r,scale_r,freq_input_r 
def wind_speed_t(shape_r,scale_r,min_winds,max_winds,steps):
    prob_vel_set=[]
    for i in range(len(shape_r)):        
        vel_set_i,prob_vel_set_i=wind_speed_i(shape=shape_r[i],scale=scale_r[i],min_ws=min_winds,max_ws=max_winds,wind_steps=steps)
        #print(sum(prob_vel_set_i))
        if i==0:
            vel_set=np.copy(vel_set_i)
        prob_vel_set.append(prob_vel_set_i)
    return vel_set,prob_vel_set
if __name__ == "__main__":
    shape=np.array([2.11,
                2.05,
                2.35,
                2.55,
                2.81,
                2.74,
                2.63,
                2.40,
                2.23,
                2.28,
                2.29,
                2.28])
    scale=np.array([9.176929,
                9.782334,
                9.531809,
                9.909545,
                10.04269,
                9.593921,
                9.584007,
                10.51499,
                11.39895,
                11.68746,
                11.63732,
                10.08803])
    freq_input=np.array([0.051,
                    0.043,
                    0.043,
                    0.066,
                    0.089,
                    0.065,
                    0.087,
                    0.115,
                    0.12,
                    0.111,
                    0.114,
                    0.096])
    freq_bins=16

    shape_r,scale_r,freq_input_r=wind_direction_t(shape,scale,freq_input,freq_bins)
    vel_set,prob_vel_set=wind_speed_t(shape_r,scale_r)
    #Outputs
    freq_input_r
    vel_set
    prob_vel_set