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


def seq_heuristic(Cables,X_all,Y_all,oss,OSSn,Bil,n_min,n_max,min_distance,diameter,organized_points,var_coll_output,objective_coll,cable_selected,max_reductions,wts_lazy,UL,C,T,coefficient,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model):
    #%% WT layout
    Ti=T.copy()
    X_wth,Y_wth=X_all[OSSn:],Y_all[OSSn:]
    WTn=len(X_wth)
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
    #%% Electrical Network
    """
    network_forbidden=[]
    for i in range(len(var_coll_output)):
        if (var_coll_output[i,0]>OSSn):
            if not((var_coll_output[i,0]-OSSn-1 in selected_wts) and (var_coll_output[i,1]-OSSn-1 in selected_wts)):
               network_forbidden.append(i+2*WTn)  
    model,tc,ts,tp=neigh_search(0,3600,9000,Bil,max_reductions,WTn,n_max,n_min,wts_lazy,min_distance,diameter,[],[],[],0,OSSn,var_coll_output,objective_coll,X_all,Y_all,UL,C,T,coefficient,False,False,True,selected_wts,network_forbidden)
    
    if model.solution.get_status()==103: Exception('Infeasible problem. Check it out')

    solution=model.solution.get_values()
    WTs=np.array(solution[:WTn])>0.9
    wt_positions=np.argwhere(WTs==True)    
    indices_active_connections=(np.argwhere(np.array(solution[2*WTn+OSSn:])>0.9)).ravel()+OSSn
    connections=var_coll_output[indices_active_connections,:]
    indices_cables_selected=cable_selected[indices_active_connections,:]      
    X_all2,Y_all2=np.concatenate((oss[:,0],X_wt2),axis=0),np.concatenate((oss[:,1],Y_wt2),axis=0)  
    plotting_simplified(OSSn,X_all2,Y_all2,Cables,organized_points,connections,indices_cables_selected,wt_positions)
    indices_active_connections+=2*WTn
    indices_active_connections=list(indices_active_connections)
    cost_network=sum(np.multiply(np.array(objective_coll).ravel(),np.array(solution[2*WTn:]).ravel()))  
    """
    #%% Electrical Network 2
    X_all_op,Y_all_op=np.concatenate((oss[:,0],X_wt2),axis=0),np.concatenate((oss[:,1],Y_wt2),axis=0)
    b,cost_network,gap_outputit,time_formulating,time_solving,solutions=global_optimizer(len(X_all_op)-len(oss),X_all_op.reshape(-1,1),Y_all_op.reshape(-1,1),Cables,C=C,OSSc=len(oss),T=[5,15,20,25,30,35,40,45,50],gap=1,iterative=False,time_limit=3600)
    plotting_collection_system(X_all_op,Y_all_op,Cables,b)
    plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
    plt.legend()
    tc,ts,tp=sum(time_formulating)*60,sum(time_solving)*60,0
    
    indices_active_connections=[]
    cont=0
    for i in range(len(b)):
        node1,node2,number_wts=int(b[i,0]),int(b[i,1]),b[i,4]
        if node1>OSSn:
            node1=node1-OSSn-1
            ind_node1=selected_wts[node1]       
            node1=ind_node1+OSSn+1
        if node2>OSSn:
            node2=node2-OSSn-1
            ind_node2=selected_wts[node2] 
            node2=ind_node2+OSSn+1       
        ind1=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==0))[0]
        ind2=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==number_wts))[0]
        if len(ind1)!=1 or len(ind2)!=1:
            print('Need to add a new connection between ',str(node1),' and ', str(node2))
            cont+=1
            ind_x=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,3]==UL-1))[0]
            ind_x=ind_x[-1]+1
            var_output_aux=np.zeros((UL,4))
            cable_selected_aux=np.zeros((UL,1))
            objective_coll_aux=np.zeros((UL,1))
            Ti[node1-OSSn-1]+=1  
            for h in range(len(var_output_aux)):
                var_output_aux[h,0],var_output_aux[h,1],var_output_aux[h,2]=node1,node2,np.sqrt((X_all[node1-1]-X_all[node2-1])**2+(Y_all[node1-1]-Y_all[node2-1])**2)
                var_output_aux[h,3]=h
                             
                if h!=0:
                   for p in range(len(Cables)):
                       if h<=Cables[p,1]:
                          cable_selected_aux[h]=p
                          objective_coll_aux[h]=Cables[p,2]*(var_output_aux[h,2])/1000 
                          break         
            var_coll_output=np.insert(var_coll_output,ind_x,var_output_aux,axis=0)
            cable_selected=np.insert(cable_selected,ind_x,cable_selected_aux,axis=0)
            objective_coll=np.insert(objective_coll,ind_x,objective_coll_aux,axis=0)

    for i in range(len(b)):
        node1,node2,number_wts=int(b[i,0]),int(b[i,1]),b[i,4]
        if node1>OSSn:
            node1=node1-OSSn-1
            ind_node1=selected_wts[node1]       
            node1=ind_node1+OSSn+1
        if node2>OSSn:
            node2=node2-OSSn-1
            ind_node2=selected_wts[node2] 
            node2=ind_node2+OSSn+1 
        ind1=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==0))[0]
        ind2=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==number_wts))[0]
        if len(ind1)!=1 or len(ind2)!=1:
            Exception('Problem. Check it out')
        else:
            ind1=ind1[0]
            ind2=ind2[0]
            indices_active_connections.append(int(ind1+2*WTn))
            indices_active_connections.append(int(ind2+2*WTn))     
    connections=var_coll_output[np.array(indices_active_connections)-2*WTn]    
    return selected_wts,indices_active_connections,connections,power,cost_network,tc,ts,tp,Ti,var_coll_output,objective_coll,cable_selected

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