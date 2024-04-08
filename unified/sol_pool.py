# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:29:15 2021

@author: juru
"""

import numpy as np
from AEP_IEATask import aep_calculator
from auxiliaries import dcf
from AEP_IEATask import aep_calculator_mo_wakes
import numpy_financial as np_fin
from auxiliaries import irr_dtu


def all_solutions(X_all,Y_all,WTn,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter,model,OSSn,var_coll_output,objective_coll,cable_selected,cost_energy,dis_rate,lifetime,ee_dtu,rated_rpm,wt_nom_power,hub_height,water_depth):

    list_solutions_wt_x,list_solutions_wt_y,list_full_solutions,list_objectives,list_IEA_value,list_positions_wt,list_connections,list_cost_connections,list_cables_selected,list_npv,list_npv_losses,list_active_con_ind=[],[],[],[],[],[],[],[],[],[],[],[]
    list_first_IRR_simp,list_first_IRR_compr=[],[]
    
    number=model.solution.pool.get_num()
    X,Y=X_all[OSSn:],Y_all[OSSn:]
    dcf_proj=dcf(cost_energy,dis_rate,lifetime)
    for i in range(number):
        list_objectives.append(model.solution.pool.get_objective_value(i))
        solution=model.solution.pool.get_values(i)
        list_full_solutions.append(solution)
        WTs=np.array(solution[:WTn])>0.9
        wt_positions=np.argwhere(WTs==True)
        list_positions_wt.append(wt_positions)
        X_s,Y_s=X[wt_positions],Y[wt_positions]
        list_solutions_wt_x.append(X_s)
        list_solutions_wt_y.append(Y_s)
        IEA=aep_calculator(X_s,Y_s,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)
        list_IEA_value.append(IEA)
        
        indices_active_connections=(np.argwhere(np.array(solution[2*WTn+OSSn:])>0.9)).ravel()+OSSn
        list_active_con_ind.append(indices_active_connections+2*WTn)
        connections=var_coll_output[indices_active_connections,:]
        list_connections.append(connections)
        list_cables_selected.append(cable_selected[indices_active_connections,:])
        cost_collection_system=sum(np.multiply(np.array(objective_coll).ravel(),np.array(solution[2*WTn:]).ravel()))     
        list_cost_connections.append(cost_collection_system)
        npv=-cost_collection_system+dcf_proj*IEA
        list_npv.append(npv)
        list_npv_losses.append(dcf_proj*(aep_calculator_mo_wakes(X_s,Y_s,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)-IEA)+cost_collection_system)
        
        list_first_IRR_simp.append(100*np_fin.irr([-cost_collection_system]+[IEA*cost_energy]*lifetime))
        list_first_IRR_compr.append(irr_dtu(IEA,cost_collection_system,ee_dtu,rated_rpm*np.ones([len(X_s)]),diameter*np.ones([len(X_s)]),wt_nom_power*np.ones([len(X_s)]),hub_height*np.ones([len(X_s)]),water_depth*np.ones([len(X_s)])))
        
    return list_solutions_wt_x,list_solutions_wt_y,np.array(list_objectives),np.array(list_IEA_value),list_positions_wt,list_full_solutions,list_connections,list_cables_selected,np.array(list_cost_connections),np.array(list_npv),np.array(list_npv_losses),list_active_con_ind,np.array(list_first_IRR_simp),np.array(list_first_IRR_compr)
