# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:16:18 2021

@author: juru
"""
import numpy as np
import math 

def bil(c_t,k,xi,xl,d,yi,yl):
    if (xi-xl<0):
        raise Exception('Should not be any negative here')
    elif (xi-xl==0):
         bil_o=0
    else:
        sigma_y=(k*(xi-xl))+(d/math.sqrt(8))
        if sigma_y>=math.sqrt((c_t*(d)**2)/(8)):
            bil_o=(1-math.sqrt(1-((c_t)/(8*((sigma_y)**2/(d)**2)))))*(math.exp(-0.5*((yi-yl)/(sigma_y))**2))
        else:
            sigma_y=math.sqrt((c_t*(d)**2)/(8))+1
            bil_o=(1-math.sqrt(1-((c_t)/(8*((sigma_y)**2/(d)**2)))))*(math.exp(-0.5*((yi-yl)/(sigma_y))**2))
    return bil_o

def vel_deficit(sorted_indexes,wt_points_freq,freq_input_r,vel_set,prob_vel_set,ct_all,wt_model,k):
    Bil=np.zeros((wt_points_freq.shape[0],wt_points_freq.shape[0]))
    d=wt_model.diameter()              
    for l in range(wt_points_freq.shape[0]):
       for j in range(wt_points_freq.shape[2]): 
           initial=int(np.where(l==sorted_indexes[:,j])[0][0])
           for i in range(initial+1,wt_points_freq.shape[0]):
               b_accum=0
               affected=int(sorted_indexes[i,j])
               for v in range(len(prob_vel_set[j])):
                   c_t=ct_all[v]#Dependent on wind velocity
                   b_squared=(bil(c_t,k,wt_points_freq[i,0,j],wt_points_freq[initial,0,j],d,wt_points_freq[i,1,j],wt_points_freq[initial,1,j]))**2
                   b_accum+=freq_input_r[j,1]*prob_vel_set[j][v]*vel_set[v]*b_squared
               Bil[affected,l]+=b_accum
       print('WT finished: ',l)
    return Bil

def vel_deficit_aep(sorted_indexes,wt_points_freq,ct_all,d,k,j,v):
    Bil=np.zeros((wt_points_freq.shape[0],wt_points_freq.shape[0]))
    #d=wt_model.diameter()              
    for l in range(wt_points_freq.shape[0]):
       #for j in range(wt_points_freq.shape[2]): 
           initial=int(np.where(l==sorted_indexes[:,j])[0][0])
           for i in range(initial+1,wt_points_freq.shape[0]):
               #b_accum=0
               affected=int(sorted_indexes[i,j])
               #for v in range(len(prob_vel_set[j])):
               c_t=ct_all[v]#Dependent on wind velocity
               eps=0.2*math.sqrt(0.5*((1+math.sqrt(1-c_t))/(math.sqrt(1-c_t)))) #Dependent on wind velocity                       
               b_squared=(bil(c_t,k,wt_points_freq[i,0,j],wt_points_freq[initial,0,j],d,eps,wt_points_freq[i,1,j],wt_points_freq[initial,1,j]))**2
               #b_accum+=b_squared
               Bil[affected,l]=b_squared
    return Bil
def reductions(Bil):
    max_reductions=np.sum(Bil,1).reshape(-1,1)
    return max_reductions
