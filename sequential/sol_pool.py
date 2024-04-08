# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:29:15 2021

@author: juru
"""

import numpy as np
from AEP_IEATask import aep_calculator
from auxiliaries import dcf
from AEP_IEATask import aep_calculator_mo_wakes



def all_solutions(X_all,Y_all,WTn,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter,model):
    list_solutions_wt_x,list_solutions_wt_y,list_objectives,list_IEA_value,list_positions_wt=[],[],[],[],[]
    number=model.solution.pool.get_num()
    X,Y=X_all,Y_all
    for i in range(number):
        list_objectives.append(model.solution.pool.get_objective_value(i))
        solution=model.solution.pool.get_values(i)
        WTs=np.array(solution[:WTn])>0.9
        wt_positions=np.argwhere(WTs==True)
        list_positions_wt.append(wt_positions)
        X_s,Y_s=X[wt_positions],Y[wt_positions]
        list_solutions_wt_x.append(X_s)
        list_solutions_wt_y.append(Y_s)
        IEA=aep_calculator(X_s,Y_s,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)
        list_IEA_value.append(IEA)
        
    return list_solutions_wt_x,list_solutions_wt_y,np.array(list_objectives),np.array(list_IEA_value),list_positions_wt
