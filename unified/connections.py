# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:42:06 2022

@author: juru
"""

import numpy as np

def finding_connections(OSSn,WTn,X_all,Y_all,T,Cables,UL,var_coll_output,objective_coll,cable_selected,active_connections,best_sol_ind_wt,initial_length,ini_len_wt):
 #%% Adding potential missing connections   
    Ti=T.copy()
    list_warm_connections=[]
    indices_interest=(np.argwhere(active_connections[:,-1]!=0)).ravel()
    filter_active_connections=active_connections[indices_interest,:]  
    var_coll_output=np.copy(var_coll_output)
    objective_coll=np.copy(objective_coll)
    cable_selected=np.copy(cable_selected)
    for i in range(len(filter_active_connections)):
        node1,node2,number_wts=int(filter_active_connections[i,0]),int(filter_active_connections[i,1]),filter_active_connections[i,3]
        if node1>OSSn:
            node1=node1-OSSn-1
            ind_node1=np.where(best_sol_ind_wt==node1)[0]         
            if len(ind_node1)!=1: 
               Exception('Problem. Check it out')
            else:
                ind_node1=ind_node1[0]
            node1=ind_node1+initial_length+ini_len_wt+OSSn+1
        if node2>OSSn:
            node2=node2-OSSn-1
            ind_node2=np.where(best_sol_ind_wt==node2)[0]  
            if len(ind_node2)!=1: 
               Exception('Problem. Check it out')
            else:
                ind_node2=ind_node2[0]
            node2=ind_node2+initial_length+ini_len_wt+OSSn+1
        ind1=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==0))[0]
        ind2=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==number_wts))[0]
        if len(ind1)!=1 or len(ind2)!=1:
            print('Need to add a new connection between ',str(node1),' and ', str(node2))
  
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
#%% Finding indices
    for i in range(len(filter_active_connections)):
        node1,node2,number_wts=filter_active_connections[i,0],filter_active_connections[i,1],filter_active_connections[i,3]
        if node1>OSSn:
            node1=node1-OSSn-1
            ind_node1=np.where(best_sol_ind_wt==node1)[0]        
            if len(ind_node1)!=1: 
                Exception('Problem. Check it out')
            else:
                ind_node1=ind_node1[0]
            node1=ind_node1+initial_length+ini_len_wt+OSSn+1
        if node2>OSSn:
            node2=node2-OSSn-1
            ind_node2=np.where(best_sol_ind_wt==node2)[0]   
            if len(ind_node2)!=1: 
                Exception('Problem. Check it out')
            else:
                ind_node2=ind_node2[0]
            node2=ind_node2+initial_length+ini_len_wt+OSSn+1
        ind1=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==0))[0]
        ind2=np.where((var_coll_output[:,0]==node1) & (var_coll_output[:,1]==node2) & (var_coll_output[:,3]==number_wts))[0]
        if len(ind1)!=1 or len(ind2)!=1:
            Exception('Problem. Check it out')
        else:
            ind1=ind1[0]
            ind2=ind2[0]
            list_warm_connections.append(int(ind1+2*WTn))
            list_warm_connections.append(int(ind2+2*WTn))        
    return list_warm_connections,Ti,var_coll_output,objective_coll,cable_selected