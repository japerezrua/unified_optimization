# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:44:55 2022

@author: juru
"""

import dill
import matplotlib.pylab as plt
import numpy as np
from points_organizer import rotational_sort,convex_hull
from auxiliaries import dcf
import numpy_financial as np_fin
from dtu_wind_cm_main import economic_evaluation as EE_DTU
from auxiliaries import irr_dtu
from borssele import IEA3_5MW
from borssele_10 import IEA10MW
wts={'3.35':IEA3_5MW,'10':IEA10MW}
from cs_plotting_global import plotting_collection_system
from cs_global_collection import global_optimizer
from AEP_IEATask import aep_calculator
from cables_cost_function import cost_function_array

outfilename = 'Borselle_74WTs_10MW_int_correctcables_17.dill'
boundaries = np.array([[484178.55, 5732482.8], [500129.9, 5737534.4], [497318.1, 5731880.24], [488951.0, 5.72794e6],  #General boundary points
                          [488951.0, 5.72794e6], [497318.1, 5731880.24], [503163.37, 5729155.3], [501266.5, 5715990.05]]) #General boundary points
wt_type='10'
cost_energy=25 #Cost of energy in Euro/MWh
dis_rate=1 #Dicount rate in percentage
lifetime=20 #years
distance_from_shore = 10 # [km]
water_depth = 20
Cables=np.array([[1,4],[2,6],[3,8]]) 
k_w=0.0324555 #wake param
C=15 #Number of max main feeders
voltage=33 #Voltage level
oss=np.array([[((boundaries[0,0]+boundaries[7,0])/2)-500,((boundaries[0,1]+boundaries[7,1])/2)-500]])

###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
wt_model=wts[wt_type]() #Defining object of the wind turbine
ee_dtu = EE_DTU(distance_from_shore,cost_energy,lifetime)
diameter=wt_model.diameter()
hub_height=wt_model.hub_height()
rated_rpm=wt_model.rated_rpm
wt_nom_power=wt_model.nominal_power()
boundaries=convex_hull(True,boundaries)
organized_points=rotational_sort(boundaries,(sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
Cables=cost_function_array(wt_model.nominal_power(),voltage,Cables) 

with open(outfilename, 'rb') as outfile:
    datastruct = dill.load(outfile)
    
time_formu=datastruct['Time_formulating_collection']
time_sol=datastruct['Time_solving_collection']
irr_first=datastruct['IRR_first_compr']
final_irr_real=datastruct['IRR_final_compr']
best_irr_coll=datastruct['IRR_comp_best_collection']
num_wts=datastruct['WT_collection']
X_best=datastruct['X_best']
Y_best=datastruct['Y_best']
X_coll=datastruct['X_coll']
Y_coll=datastruct['Y_coll']
final_network=datastruct['Final_network_refined']
cost_network=datastruct['Final_network_refined_cost']
vel_set=datastruct['Settings']['vel_set']
prob_vel_set=datastruct['Settings']['prob_vel_set']
freq_input_r=datastruct['Settings']['Frequency_input']
list_AEP_coll=datastruct['AEP_coll']
ct_all=np.array([8/9]*len(vel_set))

total_time=[]
for i in range(len(time_sol)):
    if i>0: 
        total_time.append(time_formu[i]+time_sol[i]+total_time[i-1]) 
    else: 
        total_time.append(time_formu[i]+time_sol[i])
total_time=[0]+total_time
best_irr_coll2=[irr_first]+best_irr_coll


plt.figure()
plt.plot(np.array(total_time)/3600,np.array(best_irr_coll2))
plt.xlabel('Time [h]',fontsize=35)
plt.ylabel('IRR [%]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)

plt.figure()
plt.plot(np.array(total_time)/3600,100*(np.array(best_irr_coll2)-np.array(best_irr_coll2)[0])/(np.array(best_irr_coll2)[0]))
plt.xlabel('Time [h]',fontsize=35)
plt.ylabel('IRR Imp. [%]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)


plt.figure()
plt.plot(np.array(total_time)/3600,[num_wts[0]]+num_wts)
plt.xlabel('Time [h]',fontsize=35)
plt.ylabel('WTs number',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)

plotting_collection_system(X_best,Y_best,Cables,final_network)
plt.plot(organized_points[:,0],organized_points[:,1], 'k-',label='OWF limits')
plt.legend()

