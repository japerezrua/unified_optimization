# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:13:16 2022

@author: juru
"""


import numpy as np
import math
import matplotlib.pyplot as plt



def plotting(X,Y,Cables,organized_points,solution,WTn,OSSn,var_coll_output,cable_selected):
    active_wts=np.array(solution[:WTn])>0.9
    active_wts=np.argwhere(active_wts==True)+OSSn+1 #Active WTs with identifier
    active_nodes=np.concatenate((np.array(range(OSSn)).reshape(OSSn,1)+1,active_wts),axis=0)
    active_connections=np.array(solution[int(WTn*2+OSSn):int(len(solution))])>0.9
    active_connections=np.argwhere(active_connections==True)
    indices=active_connections[:,0]+OSSn
    WTs=np.array(OSSn*[1]+solution[:WTn])>0.9
    X,Y=X[np.argwhere(WTs==True)],Y[np.argwhere(WTs==True)]
    plt.figure()
    plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
    plt.plot(X[OSSn:], Y[OSSn:], 'r+',markersize=10, label='WTs')
    for i in range(len(X)):
        plt.text(X[i]+50, Y[i]+50,str(int(active_nodes[i][0])))    
    plt.plot(X[:OSSn], Y[:OSSn], 'ro',markersize=10, label='OSS') 
    active_connections_nodes_t=var_coll_output[indices,:]
    active_cables=cable_selected[indices,:]    
    active_connections_nodes=active_connections_nodes_t[active_connections_nodes_t[:,-1]>0,:]
    active_cables=active_cables[active_connections_nodes_t[:,-1]>0,:]
    colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
    for i in range(Cables.shape[0]):
        index = active_cables[:,0]==i
        if index.any():
           n1_node = active_connections_nodes[index,0]
           n2_node = active_connections_nodes[index,1]
           n1xs,n2xs,n1ys,n2ys=[],[],[],[]
           for j in range(len(n1_node)):
               ind_node1=np.where(active_nodes==n1_node[j])[0][0]
               ind_node2=np.where(active_nodes==n2_node[j])[0][0]
               n1xs += [X[ind_node1][0]]
               n2xs += [X[ind_node2][0]]
               n1ys += [Y[ind_node1][0]]
               n2ys += [Y[ind_node2][0]]
           xs = np.vstack([np.array(n1xs),np.array(n2xs)])
           ys = np.vstack([np.array(n1ys),np.array(n2ys)])
           plt.plot(xs,ys,'{}'.format(colors[i]))
           plt.plot([],[],'{}'.format(colors[i]),label='Cable: {}'.format(i+1)) 
    plt.legend()
    return X,Y,active_nodes,active_connections_nodes,active_cables         

def plotting_simplified(OSSn,X_all,Y_all,Cables,organized_points,active_connections_best,cables_best,indices_wts_best):
    indices_all_best=indices_wts_best+OSSn+1
    indices_all_best=np.concatenate((np.array(range(1,OSSn+1,1)).reshape(-1,1),indices_all_best),axis=0)
    indices_interest=(np.argwhere(active_connections_best[:,-1]!=0)).ravel()
    filter_active_connections=active_connections_best[indices_interest,:]
    filter_active_cables=cables_best[indices_interest,:]
    plt.figure()
    plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
    plt.plot(X_all[OSSn:], Y_all[OSSn:], 'r+',markersize=10, label='WTs')
    for i in range(len(X_all[:OSSn])):
        plt.text(X_all[i]+50, Y_all[i]+50,str(int(i+1)))     
    for i in range(len(X_all[OSSn:])):
        plt.text(X_all[OSSn+i]+50, Y_all[OSSn+i]+50,str(int(i+OSSn+1)))    
    plt.plot(X_all[:OSSn], Y_all[:OSSn], 'ro',markersize=10, label='OSS') 
   
    colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
    for i in range(Cables.shape[0]):
        index = filter_active_cables[:,0]==i
        if index.any():
           n1_node = filter_active_connections[index,0]
           n2_node = filter_active_connections[index,1]
           n1xs,n2xs,n1ys,n2ys=[],[],[],[]
           for j in range(len(n1_node)):
               ind_node1=np.where(indices_all_best==n1_node[j])[0][0]
               ind_node2=np.where(indices_all_best==n2_node[j])[0][0]
               n1xs += [X_all[ind_node1]]
               n2xs += [X_all[ind_node2]]
               n1ys += [Y_all[ind_node1]]
               n2ys += [Y_all[ind_node2]]
           xs = np.vstack([np.array(n1xs),np.array(n2xs)])
           ys = np.vstack([np.array(n1ys),np.array(n2ys)])
           plt.plot(xs,ys,'{}'.format(colors[i]))
           plt.plot([],[],'{}'.format(colors[i]),label='Cable: {}'.format(i+1)) 
    plt.legend()
    return filter_active_connections,filter_active_cables,indices_all_best