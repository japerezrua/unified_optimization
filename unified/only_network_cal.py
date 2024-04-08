# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:06:59 2022

@author: juru
"""
import numpy as np
from cs_global_collection import global_optimizer
from cs_plotting_global import plotting_collection_system
X_wt=np.array([0,247,835.624,-380.083,-1243.2,104,-1144,-520,572,-52,1249.64,-1024.41,-818.117,380.083,1131,-195,1010.29])
Y_wt=np.array([0,1118,995.858,1243.2,-380.083,520,208,52,-104,-572,358.329,800.36,-1010.29,-1243.2,-247,-1131,-818.117])

Cables=np.array([[1,4,380000],[2,9,630000]]) #Cables Ormonde Ins 5
C=5 #Number of max main feeders


b,solution_value,gap_outputit,time_formulating,time_solving,solutions=global_optimizer(16,X_wt,Y_wt,Cables,C=C,OSSc=1,T=[5,10,15,25,30,35,40,45,50],gap=1,iterative=True,time_limit=3600)
plotting_collection_system(X_wt,Y_wt,Cables,b)


aep_calculator(X_wt.reshape(-1,1),Y_wt.reshape(-1,1),vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)