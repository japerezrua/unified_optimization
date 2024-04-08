# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:30:00 2022

@author: juru

Sequential increasing of both WT candidates set and neigh_max KEEPING THE SAME NEIGH_MAX if solution improved, otherwise INCREAE WT candidates and start from smallest WT candidates
"""

import numpy as np
from points_organizer import rotational_sort,convex_hull
import matplotlib.pylab as plt
from random_points import rand_points_internal,rand_points_boundary
from matrix_rotation import coordinates_translation
from borssele import IEA3_5MW
from borssele_10 import IEA10MW
wts={'3.35':IEA3_5MW,'10':IEA10MW}
import time
from sol_pool import all_solutions
from var import var_coll

from velocity_deficit import vel_deficit,reductions
from bil_optimized import vel_deficit

from auxiliaries import circle_gen,linear_regression
from proximity_search import neigh_search
import dill
from plotting import plotting,plotting_simplified
from heuristics import seq_heuristic,warm_starting_indices
from auxiliaries import dcf
from connections import finding_connections
from wind_rose import wind_speed_t
from cables_cost_function import cost_function_array
from dtu_wind_cm_main import economic_evaluation as EE_DTU
import numpy_financial as np_fin
from auxiliaries import irr_dtu
#%% Inputs
### Inputs related to WFLO----------------------------------------------------------------
freq_input_r=np.array([[0,0.025],
                       [22.5,0.024],
                       [45,.029],
                       [67.5,0.036],
                       [90,0.063],
                       [112.5,0.065,],
                       [135,0.100],
                       [157.5,0.122],
                       [180,0.063],
                       [202.5,0.038],
                       [225,0.039],
                       [247.5,0.083],
                       [270,0.213],
                       [292.5,0.046],
                       [315,0.032],
                       [337.5,0.022]])
k_w=0.0324555 #wake param
#boundaries=np.array([[0,0],[20*190.6,0]])
n_max=74 #Max. number of WTs to install
n_min=74 #Max. number of WTs to install 
min_distance=2# Minimum distance between WTs in diameters
gap=0 #Gap in percentage
memory_limit=100000 #Memory limit in MB
wts_lazy=False #If True than minimum distance constraint between WTs is called as lazy constraints


user_defined_ws=False #If this is True. it is required the three settings below (see line 72)

vel_set=np.array([float(9.8)])
prob_vel_set=[]
#prob_vel_set.append([1])
prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),
prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),
prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),
prob_vel_set.append([1])
#freq_input_r=np.array([[90,1]])

                     #If user_defined_ws=True the two inputs below are required
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
hull=True
### Inputs related to circular boundary generation and sampling boundary------------------
#radius=3000
#radius=2000

radius=1300 #Circular radius
steps_boundary=0.5 #Angle separation to define circular boundary
steps_candidates=5 #Angle separation to sample candidates in the boundary

boundaries = np.array([[484178.55, 5732482.8], [500129.9, 5737534.4], [497318.1, 5731880.24], [488951.0, 5.72794e6],  #General boundary points
                          [488951.0, 5.72794e6], [497318.1, 5731880.24], [503163.37, 5729155.3], [501266.5, 5715990.05]]) #General boundary points

wt_type='10' #Choose between '10' and '3.35' (MW)
### Inputs related to linear regression/economics---------------------------------------------------
number_rep=5000 #Number repetitions for linear regression
cost_energy=25 #Cost of energy in Euro/MWh
dis_rate=5 #Dicount rate in percentage
lifetime=20 #years
distance_from_shore = 10 # [km]
water_depth = 20
### Inputs related to collection system---------------------------------------------------
#oss=np.array([[0,0]])
oss=np.array([[((boundaries[0,0]+boundaries[7,0])/2)-500,((boundaries[0,1]+boundaries[7,1])/2)-500]])
factor=50 #Number of WTs to consider for each WT connection
#Cables=np.array([[1,4,380000],[2,9,630000]]) #Cables Ormonde Ins 5
Cables=np.array([[1,4],[2,6],[3,8]]) 
C=15 #Number of max main feeders
voltage=33 #Voltage level
### Inputs related to Neigh search heuristic-----------------------------------------------
#neigh_pos=[2,4,6,16] #Percentage of required WTn to start neigh search
rand_points_incl=0
r_max=0
r_min=0
neigh_pos=[2,4,6,10]
cand_pot=[r_max,0.40*r_max+0.60*r_min,r_min]
cand_pot=[4.77439664, 2.93284365, 1.70514166]
cand_pot_bound=3
cand_pot_bound=2.046169989506821
timel=[3600,1.5*3600,2*3600]
target_of='IRR_comp' #'IRR_simp': IRR very simplified,'IRR_comp': IRR more elaborated
super_time_limit=5*24*3600 #Super time limit in seconds
#neigh_pos=[2]
#cand_pot=[r_max,r_min]
#timel=[3600,1.5*3600]
outfilename = 'Borselle_74WTs_10MW_int_correctcables_18.dill'
print(outfilename)
#%% Defining WT model
t_initial=time.time()
wt_model=wts[wt_type]() #Defining object of the wind turbine
diameter=wt_model.diameter()
hub_height=wt_model.hub_height()
rated_rpm=wt_model.rated_rpm
wt_nom_power=wt_model.nominal_power()
#%% Calculating cables cost
Cables=cost_function_array(wt_model.nominal_power(),voltage,Cables) 
#%% Polygon available area definition and boundary candidate points sampling
#boundaries,boundaries_cand,ignore=circle_gen(radius,steps_boundary,steps_candidates)
#Physical site: Generate organized_points rotated clockwise from polygon center without repetitions
boundaries=convex_hull(hull,boundaries)
organized_points=rotational_sort(boundaries,(sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
plt.figure()
plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
plt.legend()
boundaries_cand=rand_points_boundary(organized_points,cand_pot_bound*diameter)
boundaries_and_ws=np.copy(boundaries_cand)
#%% Obtaining vel_set and prob_vel_set
if not(user_defined_ws):
    shape_r=np.interp(freq_input_r[:,0],np.linspace(0,360,len(shape),endpoint=False),shape)
    scale_r=np.interp(freq_input_r[:,0],np.linspace(0,360,len(scale),endpoint=False),scale)
    vel_set,prob_vel_set=wind_speed_t(shape_r,scale_r,0,wt_model.cut_out,1)
ct_all=wt_model.ct_all(vel_set) 
#%% Cost model
ee_dtu = EE_DTU(distance_from_shore,cost_energy,lifetime)
#%% Linear regression
#interc,slope,coeff_det=linear_regression(organized_points,number_rep,min_distance,n_min,n_max,radius,hull,k_w,wt_model,freq_input_r,vel_set,prob_vel_set,cost_energy,dis_rate,lifetime)
#slope=12747447.61733501 #Circular farm with only frequency WR
#interc=47492738.439573504 #Circular farm with only frequency WR
#coeff_det=0.7325385827559912 #Circular farm with only frequency WR
#slope=5043054.804336463 #Circular farm with WR
#interc=7748404.393263152 #Circular farm with WR
#coeff_det=0.935409221458792 #Circular farm with WR
#slope=4345780.785844572 #Borssele farm with WR Price of energy= 25 Dis rate=1 Lifetime=20
#interc= 11156605.132807663#Borssele farm with WR
#coeff_det=0.8566133829387699 #Borssele farm with WR
slope=11073296.075771311 #Borssele 10 MW farm with WR Price of energy= 25 Dis rate=15 Lifetime=20
interc=31438667.035207644 #Borssele 10 MW farm with WR Price of energy= 25 Dis rate=15 Lifetime=20
coeff_det=0.8697077551150425 #Borssele 10 MW farm with WR Price of energy= 25 Dis rate=15 Lifetime=20
print(slope)
#%% Main algorithm
best_NPV_coll,best_X_coord_wt_coll,best_Y_coord_wt_coll,time_formu,time_sol,gap_coll,status_coll,WTn_coll,of_coll,neigh_coll,full_solution_coll,time_populating=[],[],[],[],[],[],[],[],[],[],[],[]
active_connections_coll,cost_collect_coll,cables_coll,best_indices_wts_coll,AEP_coll,best_IRR_simp_coll,best_IRR_comp_coll=[],[],[],[],[],[],[]
initial_length=len(boundaries_cand)
old_wt=0
best_sol_ind_wt,best_sol_ind_connec,warm_start_wts,active_connections=0,0,0,0
UL=max(Cables[:,1])
OSSn=len(oss)
final_length=0
cont_cand,cont_neigh=0,0
constraints_all={}
for i in range(1000):
    time_limit=timel[cont_cand]
    new_wt=cand_pot[cont_cand]
    if new_wt!=old_wt:
        rand_points_diag=cand_pot[cont_cand]*diameter #Changing diagonal distance between WTs to meters
        rand_points_ys=cand_pot[cont_cand]*diameter #Changing vertical distance between WTs to meters
        wt_points=rand_points_internal(organized_points,rand_points_incl,rand_points_diag,rand_points_ys) #First seed of WT candidate locations
        ini_len_wt=len(wt_points)        
        wt_points=np.concatenate((wt_points,boundaries_and_ws),axis=0)
        WTn=len(wt_points)     
        wt_points_freq,sorted_indexes=coordinates_translation(wt_points,freq_input_r) #Converting candidate WT location coordinates for each frequency bin
        Bil=vel_deficit(sorted_indexes,wt_points_freq,freq_input_r,vel_set,prob_vel_set,ct_all,wt_model,k_w) #objective function coefficients for WT layout
        max_reductions=reductions(Bil)
        ### Parameters related to collection system
        X_all,Y_all=np.concatenate((oss[:,0],wt_points[:,0]),axis=0),np.concatenate((oss[:,1],wt_points[:,1]),axis=0)
        T=min(factor,WTn-1)
        var_coll_output,objective_coll,cable_selected=var_coll(WTn,OSSn,X_all,Y_all,UL,Cables,T,0)
        T=WTn*[T]
        ###--------------------------------------- Warm-starting
        if i==0: #Warm-starting heuristc first point
           list_warm_wts,list_warm_connections,active_connections,power_first,cost_network_first,tc_first,ts_first,tp_first,T,var_coll_output,objective_coll,cable_selected=seq_heuristic(Cables,X_all,Y_all,oss,OSSn,Bil,n_min,n_max,min_distance,diameter,organized_points,var_coll_output,objective_coll,cable_selected,max_reductions,wts_lazy,UL,C,T,slope,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model)
           dcf_proj=dcf(cost_energy,dis_rate,lifetime)
           npv_first=-cost_network_first+dcf_proj*power_first
           first_IRR=100*np_fin.irr([-cost_network_first]+[power_first*cost_energy]*lifetime)
           first_IRR_compr=irr_dtu(power_first,cost_network_first,ee_dtu, rated_rpm*np.ones([len(list_warm_wts)]),diameter*np.ones([len(list_warm_wts)]),wt_nom_power*np.ones([len(list_warm_wts)]),hub_height*np.ones([len(list_warm_wts)]),water_depth*np.ones([len(list_warm_wts)]))
           irr_dic={'IRR_simp': first_IRR,'IRR_comp': first_IRR_compr}
           incumbent,incumbent_first=irr_dic[target_of],irr_dic[target_of]
           print('First heuristic solution')
           print('IRR Value [%]',incumbent)
        else:
            list_warm_wts = [int(x)+int(ini_len_wt) for x in warm_start_wts] 
            list_warm_connections,T,var_coll_output,objective_coll,cable_selected=finding_connections(OSSn,WTn,X_all,Y_all,T,Cables,UL,var_coll_output,objective_coll,cable_selected,active_connections,best_sol_ind_wt,initial_length,ini_len_wt)           
        ###--------------------------------------- 
    else:
        #wt_points=np.concatenate((wt_points,boundaries_cand),axis=0)
        list_warm_wts = [int(x) for x in best_sol_ind_wt]
        list_warm_connections = [int(x) for x in best_sol_ind_connec]
    #X=wt_points[:,0] #Forming abscissas set 
    #Y=wt_points[:,1] #Forming ordinates set           
    #WTn=len(wt_points)  
    indices_all,values_all=warm_starting_indices(WTn,OSSn,Bil,var_coll_output,list_warm_wts,list_warm_connections,active_connections)
    #%% Optimization model
    model,tc,ts,tp,constraints_all=neigh_search(gap,time_limit,memory_limit,Bil,max_reductions,WTn,n_max,n_min,wts_lazy,min_distance,diameter,list_warm_wts,indices_all,values_all,neigh_pos[cont_neigh],OSSn,var_coll_output,objective_coll,X_all,Y_all,UL,C,T,slope,True,True,[],[],Cables,new_wt!=old_wt,constraints_all,super_time_limit,t_initial)
    #%% Organizing data
    list_solutions_wt_x,list_solutions_wt_y,objective_values,IEA_values,list_positions_wt,list_full_solutions,list_connections,list_cables_selected,list_cost_connections,list_npv,list_npv_losses,list_active_con_ind,list_irr_simp,list_irr_comp=all_solutions(X_all,Y_all,WTn,vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter,model,OSSn,var_coll_output,objective_coll,cable_selected,cost_energy,dis_rate,lifetime,ee_dtu,rated_rpm,wt_nom_power,hub_height,water_depth)
    #%% Indexes best
    best_index={'IRR_simp': np.argmax(list_irr_simp) ,'IRR_comp': np.argmax(list_irr_comp)}
    best_sol={'IRR_simp':max(list_irr_simp) ,'IRR_comp':max(list_irr_comp)}
    
    best_index=best_index[target_of]
    best_sol=best_sol[target_of]
    
    best_sol_ind_wt=list_positions_wt[best_index]
    best_sol_ind_connec=list_active_con_ind[best_index]
    X_wt,Y_wt=list_solutions_wt_x[best_index],list_solutions_wt_y[best_index]
    active_connections=list_connections[best_index]

    WTn_coll.append(WTn)
    full_solution_coll.append(list_full_solutions[best_index])     
    best_indices_wts_coll.append(best_sol_ind_wt)
    cables_coll.append(list_cables_selected[best_index])
    active_connections_coll.append(active_connections)
    cost_collect_coll.append(list_cost_connections[best_index])
    AEP_coll.append(IEA_values[best_index])
    best_X_coord_wt_coll.append(X_wt)
    best_Y_coord_wt_coll.append(Y_wt)    
    of_coll.append(objective_values[best_index])
    best_NPV_coll.append(list_npv[best_index])
    best_IRR_simp_coll.append(list_irr_simp[best_index])
    best_IRR_comp_coll.append(list_irr_comp[best_index])
    gap_coll.append(100*model.solution.MIP.get_mip_relative_gap())
    status_coll.append(model.solution.get_status())
    time_formu.append(tc)
    time_sol.append(ts)
    time_populating.append(tp)
    neigh_coll.append(neigh_pos[cont_neigh])    
    
    boundaries_and_ws=np.concatenate((boundaries_cand,np.concatenate((X_wt.reshape(-1,1),Y_wt.reshape(-1,1)),axis=1)),axis=0)
    final_length=len(boundaries_and_ws)
    warm_start_wts=list(range(initial_length,final_length))
       
    print('Iteration number',i+1)
    print('WTn',WTn)
    print('Neigh size',neigh_pos[cont_neigh])
    print('IRR Value [%]',best_sol)
    print('X coord',X_wt)
    print('Y coord',Y_wt)
    #%% Increasing 
    
    old_wt=cand_pot[cont_cand]
    
    if best_sol>incumbent: #If the solution improved, keep WT candidate set and increase neighboorhood size
        print('Improved solution:',best_sol>incumbent)
        incumbent=best_sol
        #cont_neigh+=1
    #elif cont_cand==len(cand_pot)-1: #But if in the last one WT candidate set, increase anyway neighboorhood size search to avoid early terminations 
    #    cont_neigh+=1
    else: #If no of the previous conditiones are satisfied, restart neighboorhood size to zero and increase candidate set
        cont_neigh+=1
        #cont_cand+=1        
    if cont_neigh==len(neigh_pos): #if exhausted the neighboorhood set, restart it to zero and increase WT candidate set
        cont_neigh=0
        cont_cand+=1
    if cont_cand==len(cand_pot): #If exhausted the WT candidate set, stop the whole algorithm
        break
    del model
    if time.time()-t_initial>super_time_limit*0.97:
        break
#%% Obtaining final result
irr_dic_collector={'IRR_simp':best_IRR_simp_coll ,'IRR_comp':best_IRR_comp_coll}
irr_dic_collector=irr_dic_collector[target_of]
X_wt,Y_wt=best_X_coord_wt_coll[irr_dic_collector.index(max(irr_dic_collector))],best_Y_coord_wt_coll[irr_dic_collector.index(max(irr_dic_collector))]
#plotting(X_wt,Y_wt,organized_points,WTn_coll[best_IEA_coll.index(max(best_IEA_coll))])
WTn_best=WTn_coll[irr_dic_collector.index(max(irr_dic_collector))]
solution_best=full_solution_coll[irr_dic_collector.index(max(irr_dic_collector))]
X_all,Y_all=np.concatenate((oss[:,0],X_wt[:,0]),axis=0),np.concatenate((oss[:,1],Y_wt[:,0]),axis=0)
active_connections_best=active_connections_coll[irr_dic_collector.index(max(irr_dic_collector))]
cables_best=cables_coll[irr_dic_collector.index(max(irr_dic_collector))]
indices_wts_best=best_indices_wts_coll[irr_dic_collector.index(max(irr_dic_collector))]
filter_active_connections,filter_active_cables,indices_all_best=plotting_simplified(OSSn,X_all,Y_all,Cables,organized_points,active_connections_best,cables_best,indices_wts_best)
#%% Plotting performance
total_time=[]
for i in range(len(time_sol)):
    if i>0: 
        total_time.append(time_formu[i]+time_sol[i]+total_time[i-1]) 
    else: 
        total_time.append(time_formu[i]+time_sol[i])
total_time=[tc_first+ts_first+tp_first]+total_time
obj_plot=[incumbent_first]+irr_dic_collector
plt.figure()
plt.plot(np.array(total_time)/3600,np.array(obj_plot))
plt.xlabel('Time [h]',fontsize=35)
plt.ylabel('IRR [%]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)

#%% Refining network design
from cs_global_collection import global_optimizer
from cs_plotting_global import plotting_collection_system
b,solution_value,gap_outputit,time_formulating,time_solving,solutions=global_optimizer(len(X_wt),X_all,Y_all,Cables,C=C,OSSc=len(oss),T=[5,10,15,25,30,35,40,45,50],gap=1,iterative=True,time_limit=3600)
plotting_collection_system(X_all,Y_all,Cables,b)
dcf_proj=dcf(cost_energy,dis_rate,lifetime)
aep_first=power_first
final_aep=AEP_coll[irr_dic_collector.index(max(irr_dic_collector))]
final_NPV=-solution_value+dcf_proj*final_aep
final_IRR=100*np_fin.irr([-solution_value]+[final_aep*cost_energy]*lifetime)
final_IRR_compr=irr_dtu(final_aep,solution_value,ee_dtu, rated_rpm*np.ones([len(X_wt)]),diameter*np.ones([len(X_wt)]),wt_nom_power*np.ones([len(X_wt)]),hub_height*np.ones([len(X_wt)]),water_depth*np.ones([len(X_wt)]))
#%% Saving results
settings={'Frequency_input':freq_input_r,'n_max':n_max,'n_min':n_min,'user_defined_ws':user_defined_ws,'vel_set':vel_set,'prob_vel_set':prob_vel_set,'shape':shape,'scale':scale,'oss':oss,'factor':factor,'Cables':Cables,'C':C,'voltage':voltage,'radius':radius,'steps_boundary':steps_boundary,'steps_candidates':steps_candidates,'number_rep':number_rep,'cost_energy':cost_energy,'dis_rate':dis_rate,'lifetime':lifetime,'r_max':r_max,'r_min':r_min,'neigh_pos':neigh_pos,'cand_pot':cand_pot,'timel':timel}
results = {
    'WT_collection': WTn_coll,
    'Time_formulating_collection': time_formu,
    'Time_solving_collection': time_sol,
    'Time_populating_collection':time_populating,
    'GAP_collection': gap_coll,
    'Status_collection': status_coll,
    'OF_collection': of_coll,
    'Neigh_collection': neigh_coll,
    'X_coll':best_X_coord_wt_coll,
    'Y_coll':best_Y_coord_wt_coll,
    'X_best': X_all,  
    'Y_best': Y_all,  
    'Cost_collection_system':cost_collect_coll,
    'AEP_coll':AEP_coll,
    'NPV_best_collection': best_NPV_coll,
    'IRR_simp_best_collection':best_IRR_simp_coll,
    'IRR_comp_best_collection':best_IRR_comp_coll,
    'Interc_lin_reg': interc,
    'Slope_lin_reg': slope,
    'CD_lin_reg': coeff_det,
    'AEP_first':power_first,
    'Cost_network_first':cost_network_first,
    'Time_constraint_first':tc_first,
    'Time_solving_first':ts_first,
    'NPV_first':npv_first,
    'IRR_first':first_IRR,
    'IRR_first_compr':first_IRR_compr,    
    'AEP_final':final_aep,
    'NPV_final':final_NPV,
    'IRR_final':final_IRR,
    'IRR_final_compr':final_IRR_compr,
    'Final_network_refined':b,
    'Final_network_refined_cost':solution_value,
    'Settings':settings}

with open(outfilename, 'wb') as outfile:
    dill.dump(results, outfile)

print(outfilename, ' written')
#with open(outfilename, 'rb') as outfile:
#    datastruct = dill.load(outfile)

### Sequential: 257355058.68959266 Euros
