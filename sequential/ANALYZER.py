# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:38:34 2021

@author: juru
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:54:56 2021

@author: juru
"""
import numpy as np
from wind_rose import wind_direction_t,wind_speed_t
from points_organizer import rotational_sort,point_in_polygon,lines_of_polygon,convex_hull
import matplotlib.pylab as plt
from random_points import rand_points
from matrix_rotation import coordinates_translation
from borssele import IEA10MW
from velocity_deficit import vel_deficit,vel_deficit_aep
from var import var_coll,wt_costs
from opt_model import integrated_optimization,plotting
from warmer import initial_solution
from wind_farm_performance import aep,npv
import time
import random
import math 

#%% Inputs
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
user_defined_ws=False #If this is True. it is required the three settings below
vel_set=np.array([12])
prob_vel_set=[]
prob_vel_set.append([1])
freq_input=np.array([0.051,
                    0.043,
                    0.043,
                    0.066,
                    0.089,
                    0.065,
                    0.087,
                    0.115,
                    0.12,
                    0.111,
                    0.114,
                    0.096])
freq_bins=12
#freq_input=np.array([1])
#boundaries=np.array([[484178.55, 5732482.8], [500129.9, 5737534.4], [497318.1, 5731880.24], [488951.0, 5.72794e6], 
#                    [488951.0, 5.72794e6], [497318.1, 5731880.24], [503163.37, 5729155.3], [501266.5, 5715990.05]])

#boundaries=np.array([[0,-3000],[0,3000],[3000,3000],[3000,-3000]])

boundaries=np.array([[471989.864408479,5991227.74995742],[469062.365237162,5993842.58492974],[470927.15211234,5995684.91937665],[473744.434200031,5992886.43846932]])
#boundaries=np.array([[0,0],[20*190.6,0]])

hull=True
rand_points_incl=0
rand_points_diag=0.5
rand_points_ys=0.5
rand_points_border_diag=0.5

k=0.0324555 #wake parameter from pywake
weight_wt=1000
min_winds=0.1#Min wind velocity to consider
max_winds=25 #Max wind velocity to consider
ws_steps=1 #Wind speed steps

Only_WT=False

#oss=np.array([[492222.525,5723736.425]])
oss=np.array([[473094.719489395,5992344.97220128]])
factor=0.3 #Factor for reducing number of variables for colllection system
#Cables=np.array([[1,7,370000],[2,11,390000],[3,13,430000]]) 
Cables=np.array([[1,4,380000],[2,9,630000]]) #Cables Ormonde Ins 5

n_max=100 #Max. number of WTs to install
n_min=100 #Max. number of WTs to install
min_distance=2# Minimum distance between WTs in diameters
number_ws=1000

discount_rate=5  #[%]
price_energy=50 #Euros/MWh
lifetime=20 #Years
#%% Generating random points and calculating Bil matrix
t=time.time()
#Wind resource: Generate required frequency set freq_input_r, and for each bin: vel_set, and prob_vel_set
shape_r,scale_r,freq_input_r=wind_direction_t(shape,scale,freq_input,freq_bins)
if not(user_defined_ws): vel_set,prob_vel_set=wind_speed_t(shape_r,scale_r,min_winds,max_winds,ws_steps)
#Physical site: Generate organized_points rotated clockwise from polygon center without repetitions
boundaries=convex_hull(hull,boundaries)
organized_points=rotational_sort(boundaries,(sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
#Generating candidate WT location points inside designated area
wt_model=IEA10MW() #Defining object of the wind turbine
d=wt_model.diameter() #WT diameter
rand_points_diag=rand_points_diag*wt_model.diameter() #Changing diagonal distance between WTs to meters
rand_points_ys=rand_points_ys*wt_model.diameter() #Changing vertical distance between WTs to meters
rand_points_border_diag=rand_points_border_diag*wt_model.diameter() #Changing diagonal distance between WTs on the borders to meters
wt_points=rand_points(organized_points,rand_points_incl,rand_points_diag,rand_points_ys,rand_points_border_diag) #First seed of WT candidate locations
#Plotting designated area and candidate WT locations
#plt.figure()
#plt.plot(organized_points[:,0],organized_points[:,1], 'k-')
#plt.scatter(wt_points[:,0],wt_points[:,1],marker="o",color='r')
#Calculating Bil matrix
#wt_points_freq,sorted_indexes=coordinates_translation(wt_points,freq_input_r) #Converting candidate WT location coordinates for each frequency bin
ct_all=wt_model.ct_all(vel_set)
#Calculating cost coefficients collection system
X=wt_points[:,0] #Forming abscissas set 
Y=wt_points[:,1] #Forming ordinates set    
WTn=len(wt_points)
#%% Generating random WT layouts
glob_cont=0
v_deficit,power_deficit=[],[]
while(True):    
    n_wts=random.randint(n_min,n_max)
    X1,Y1=X[:],Y[:]
    pot_wts=np.array(range(WTn))
    sel_wts=[]
    counter=1
    pick_up=random.choice(list(enumerate(pot_wts)))
    index,element=int(pick_up[0]),int(pick_up[1])
    sel_wts+=[element]
    pot_wts=np.delete(pot_wts,index)
    post_sec=np.copy(pot_wts)
    sel_wts_sec=sel_wts.copy()
    while(True):
        #wt_i=int(np.random.choice(pot_wts,size=1)[0])
        pick_up=random.choice(list(enumerate(pot_wts)))
        index,element=int(pick_up[0]),int(pick_up[1])
        flag=True
        for j in range(len(sel_wts)):
            dist_wts=np.sqrt((X1[element]-X1[sel_wts[j]])**2+(Y1[element]-Y1[sel_wts[j]])**2)
            if dist_wts<min_distance:
                flag=False
                break
        if flag:
            sel_wts+=[element]
            counter+=1
        pot_wts=np.delete(pot_wts,index)        
        if len(pot_wts)==0:
            #print('Selection factor of [%]',n_wts*100/WTn)
            print('Conflict found while selecting randomly WTs. Restarting process')
            print('Selected up to', counter, 'WTs out of',n_wts)
            counter=1
            pot_wts=post_sec.copy()
            sel_wts=sel_wts_sec.copy()
        if counter==n_wts:
            #t5 print('Selection factor of [%]',n_wts*100/WTn)
            """
            if len(pot_wts)==0:
                print(sel_wts)
                print(counter)
                print(n_wts)
                print('Whats going on')
                plt.figure()
                plt.scatter(X[np.array(sel_wts)+OSSn],Y[np.array(sel_wts)+OSSn],marker="o",color='b')
                """
            break
    sel_wts.sort()
    wt_points_active=np.vstack((X[sel_wts],Y[sel_wts])).transpose() #Matrix with location actives WTs 
    wt_points_freq_active,sorted_indexes_active=coordinates_translation(wt_points_active,freq_input_r) #Locations rotated for each freq.
    total_deficit=0
    for j in range(wt_points_freq_active.shape[2]):
        for v in range(len(vel_set)):
            squared_def_j_v=vel_deficit_aep(sorted_indexes_active,wt_points_freq_active,ct_all,d,k,j,v)
            power_j_v=0
            for i in range(wt_points_freq_active.shape[0]):
                tot_def_squared=sum(squared_def_j_v[i,:])
                total_deficit+=weight_wt*freq_input_r[j,1]*prob_vel_set[j][v]*vel_set[v]*tot_def_squared
    aep_owf=0
    for j in range(wt_points_freq_active.shape[2]):
        for v in range(len(vel_set)):
            squared_def_j_v=vel_deficit_aep(sorted_indexes_active,wt_points_freq_active,ct_all,d,k,j,v)
            power_j_v=0
            for i in range(wt_points_freq_active.shape[0]):
                tot_def_squared=sum(squared_def_j_v[i,:])
                tot_def=math.sqrt(tot_def_squared)
                vel_i_j_v=vel_set[v]*(1-tot_def)
                power_gen_i_j_v=wt_model._power(vel_i_j_v)
                power_j_v+=power_gen_i_j_v
            aep_owf+=8760*freq_input_r[j,1]*prob_vel_set[j][v]*power_j_v
    nominal_aep_owf=len(sel_wts)*wt_model.nominal_power()*8760 #MWh
    capacity_factor=aep_owf/nominal_aep_owf
    total_aep_lost=nominal_aep_owf-aep_owf
    glob_cont+=1 
    print('Point',str(glob_cont), 'calculated')
    v_deficit+=[total_deficit]
    power_deficit+=[total_aep_lost]
    if glob_cont==number_ws:
       break               
plt.figure()
plt.scatter(v_deficit,power_deficit)
plt.text(min(v_deficit),max(power_deficit),'r='+str(round(np.corrcoef(v_deficit,power_deficit)[0,1],2)),fontsize=35)
plt.xlabel('Total wind speed deficit [m/s] - scaled by '+str(weight_wt),fontsize=35)
plt.ylabel('Total AEP deficit [MWh]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)