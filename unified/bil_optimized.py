# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:34:22 2022

@author: juru
"""

import numpy as np

def bil(c_t,k,xi,xl,d,yi,yl):
    #bil_o=np.zeros((len(xi),len(c_t)))
    if (xi-xl<0).any():
        raise Exception('Should not be any negative here')
    #elif (xi-xl==0).any():
    #     bil_o=0
   # else:
    pos_zeroes=np.array(np.argwhere((xi-xl==0)))
    sigma_y=(k*(xi-xl))+(d/np.sqrt(8))
        #if sigma_y>=np.sqrt((c_t*(d)**2)/(8)):
            #sigma_y.reshape(-1,1)>=np.sqrt((c_t*(d)**2)/(8))
    #bil_o=(1-math.sqrt(1-((c_t)/(8*((sigma_y)**2/(d)**2)))))*(math.exp(-0.5*((yi-yl)/(sigma_y))**2))
    bil_o=(1-np.sqrt(1-((c_t)/(8*((sigma_y.reshape(-1,1))**2/(d)**2)))))*(np.exp(-0.5*((yi-yl)/(sigma_y))**2)).reshape(-1,1)
    bil_o[pos_zeroes.ravel(),:]=np.zeros((pos_zeroes.shape[0],c_t.shape[0])) #elif (xi-xl==0).any(): bil_o=0
    wrong=np.argwhere(sigma_y.reshape(-1,1)<np.sqrt((c_t*(d)**2)/(8)))
    if len(wrong)>0:
        print('Correcting negative root squares')
        wrong_rows,wrong_columns=wrong[:,0],wrong[:,1]
        sigma_y=np.sqrt((c_t[wrong_columns]*(d)**2)/(8))+1
        bil_o[list(wrong_rows),list(wrong_columns)]=(1-np.sqrt(1-((c_t[wrong_columns])/(8*((sigma_y)**2/(d)**2)))))*(np.exp(-0.5*((yi-yl)[wrong_rows]/(sigma_y))**2))
    return bil_o

def vel_deficit(sorted_indexes,wt_points_freq,freq_input_r,vel_set,prob_vel_set,ct_all,wt_model,k):
    Bil2=np.zeros((wt_points_freq.shape[0],wt_points_freq.shape[0]))
    d=wt_model.diameter()              
    for l in range(wt_points_freq.shape[0]):
       for j in range(wt_points_freq.shape[2]): 
           initial=int(np.where(l==sorted_indexes[:,j])[0][0])
           #for i in range(initial+1,wt_points_freq.shape[0]):
           i=list(range(initial+1,wt_points_freq.shape[0]))
           affected=sorted_indexes[i,j].astype(int)
               #for v in range(len(prob_vel_set[j])):
           xi,xl,yi,yl=wt_points_freq[i,0,j],wt_points_freq[initial,0,j],wt_points_freq[i,1,j],wt_points_freq[initial,1,j]
           b_squared=(bil(ct_all,k,xi,xl,d,yi,yl))**2
           mult=np.array(freq_input_r[j,1]*prob_vel_set[j][:]*vel_set[:])
           deff=(b_squared*mult).sum(-1)
        #b_accum+=freq_input_r[j,1]*prob_vel_set[j][v]*vel_set[v]*b_squared
           Bil2[affected,l]+=deff
    return Bil2


