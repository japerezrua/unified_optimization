# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:00:57 2022

@author: juru
"""

import numpy as np
from velocity_deficit import reductions
import random
from AEP_IEATask import aep_calculator
import matplotlib.pylab as plt
import math
from proximity_search import neigh_search
from plotting import plotting_simplified
from cs_global_collection import global_optimizer
from cs_plotting_global import plotting_collection_system


def seq_heuristic(X_all,Y_all,Bil,n_min,n_max,min_distance,diameter,organized_points,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model):
    #%% WT layout
    X_wth,Y_wth=X_all,Y_all
    selecting_matrix=Bil+np.transpose(Bil)
    initial_solution,forbidden=np.zeros((selecting_matrix.shape[0],1)),np.zeros((selecting_matrix.shape[0],1))
    selected_wts=[]
    turbines=random.randint(n_min,n_max)
    counter=0
    it=0
    while(True):
        it+=1
        weights_all=np.matmul(selecting_matrix,initial_solution)+np.matmul(np.diagflat(np.diag(selecting_matrix)),forbidden)
        wt_candidate=int(np.argmin(weights_all))
        invalid=False
        for j in selected_wts:
            if math.sqrt((X_wth[j]-X_wth[wt_candidate])**2+(Y_wth[j]-Y_wth[wt_candidate])**2)<min_distance*diameter:
                invalid=True
                print(invalid)
                break
        if not(invalid): 
           selected_wts.append(wt_candidate)
           initial_solution[wt_candidate]=1
           counter+=1
        else:
            forbidden[wt_candidate]=1
        selecting_matrix[wt_candidate,wt_candidate]=10**10
        if counter==turbines:
            break
        if it==1000:
            raise Exception('Couldnt find an initial solution of wind turbine. Check it out')
            break           
    selected_wts.sort()
    X_wt2=X_wth[selected_wts]
    Y_wt2=Y_wth[selected_wts]    
    power=aep_calculator(X_wt2.reshape(-1,1),Y_wt2.reshape(-1,1),vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)    
    
    plt.figure()
    plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
    plt.scatter(X_wt2,Y_wt2)   
    for i in range(len(X_wt2)):
        plt.text(X_wt2[i]+50, Y_wt2[i]+50,str(int(selected_wts[i])))     
    #plt.figure()
    #plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
    #plt.scatter(X_all,Y_all)   
    #for i in range(len(X_all[OSSn:])):
    #    plt.text(X_all[i+OSSn]+50, Y_all[i+OSSn]+50,str(int(i+OSSn+1)))   
    #plt.plot(X_all[:OSSn], Y_all[:OSSn], 'ro',markersize=10, label='OSS') 
        
    #selected_wts.sort()     
    return selected_wts,power

def warm_starting_indices(WTn,OSSn,Bil,var_coll_output,selected_wts,indices_active_connections,connections):
    indices_all=list(range(2*WTn+len(var_coll_output)))
    values_all=[0]*len(indices_all)
    for i in indices_all:
        if i<WTn:
            if i in selected_wts:
               values_all[i]=1
        elif i<2*WTn:
            if i-WTn in selected_wts:
               values_all[i]=np.sum(Bil[i-WTn,:][selected_wts])     
        elif i<2*WTn+OSSn:
            subs=i-2*WTn+1
            numero=(np.argwhere(connections[:,0]==subs))
            if len(numero)!=0: values_all[i]=int(sum(connections[numero,3])) 

        else:
            if i in indices_active_connections:
                values_all[i]=1
    return indices_all,values_all