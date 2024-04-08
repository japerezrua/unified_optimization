# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:39:51 2022

@author: juru
"""

import numpy as np
from points_organizer import rotational_sort,convex_hull
import matplotlib.pylab as plt
from random_points import rand_points
from matrix_rotation import coordinates_translation
from borssele import IEA3_5MW
from power_curve_mod import partitioning_cons_pc
from var import objective_function
from opt_model import integrated_optimization,plotting,LazyConstraints,TimeLimitCallback
#from warmer import initial_solution
from wind_farm_performance import aep,npv
import time
#from mathiasmesh import boundaries,boundaries2,warm_start
from AEP_IEATask import aep_calculator
import cplex
from warmer import initial_solution
from sol_pool import all_solutions

from velocity_deficit import bil,vel_deficit,reductions

from circle import boundaries,boundaries2,warm_start
from proximity_search import neigh_search
import dill
from AEP_IEATask import aep_calculator
import random

#%% Inputs
vel_set=np.array([float(9.8)])
prob_vel_set=[]
#prob_vel_set.append([1])

prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),
prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),
prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),prob_vel_set.append([1]),
prob_vel_set.append([1])


#freq_input_r=np.array([[90,1]])


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



hull=True
rand_points_incl=0

k_w=0.0324555 #wake param

#boundaries=np.array([[0,0],[20*190.6,0]])

n_max=64 #Max. number of WTs to install
n_min=64 #Max. number of WTs to install 

r=1
min_distance=2

number_ws=2000
weight_wt=1000000
#%% Step 1
wt_model=IEA3_5MW()
boundaries=convex_hull(hull,boundaries)
organized_points=rotational_sort(boundaries,(sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
diameter=wt_model.diameter()
rand_points_diag=r*diameter #Changing diagonal distance between WTs to meters
rand_points_ys=r*diameter #Changing vertical distance between WTs to meters
wt_points=rand_points(organized_points,rand_points_incl,rand_points_diag,rand_points_ys) #First seed of WT candidate locations     
wt_points=np.concatenate((wt_points,boundaries2),axis=0)
wt_points_freq,sorted_indexes=coordinates_translation(wt_points,freq_input_r) #Converting candidate WT location coordinates for each frequency bin
ct_all=wt_model.ct_all(vel_set)        
Bil=vel_deficit(sorted_indexes,wt_points_freq,freq_input_r,vel_set,prob_vel_set,ct_all,wt_model,k_w) #objective function coefficients for WT layout
#%% Plot
plt.figure()
plt.plot(organized_points[:,0],organized_points[:,1], 'k-')
plt.scatter(wt_points[:,0],wt_points[:,1],marker="o",color='r')
#%% Generating random WT layouts
X=wt_points[:,0] #Forming abscissas set 
Y=wt_points[:,1] #Forming ordinates set 
WTn=len(wt_points)
glob_cont=0
v_deficit,power_produced=[],[]
n_wts=random.randint(n_min,n_max)
X1,Y1=X[:],Y[:]

while(True):
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
        pick_up=random.choice(list(enumerate(pot_wts)))
        index,element=int(pick_up[0]),int(pick_up[1])
        flag=True
        for j in range(len(sel_wts)):
            dist_wts=np.sqrt((X1[element]-X1[sel_wts[j]])**2+(Y1[element]-Y1[sel_wts[j]])**2)
            if dist_wts<min_distance*diameter:
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
            break

    sel_wts.sort()
    red_Bil=Bil[sel_wts,:][:,sel_wts]
    vel_def=np.sum(red_Bil)
    IEA=aep_calculator(X[sel_wts].reshape(-1,1),Y[sel_wts].reshape(-1,1),vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)
 
    v_deficit+=[vel_def]
    power_produced+=[IEA]

    glob_cont+=1 
    print('Point',str(glob_cont), 'calculated')
    #plt.figure()
    #plt.plot(organized_points[:,0],organized_points[:,1], 'k-')
    #plt.scatter(X[sel_wts].reshape(-1,1),Y[sel_wts].reshape(-1,1),marker="o",color='r')
    if glob_cont==number_ws:
       break             

plt.figure()
plt.scatter(v_deficit,power_produced)
plt.text(min(v_deficit),max(power_produced),'r='+str(round(np.corrcoef(v_deficit,power_produced)[0,1],2)),fontsize=35)
#plt.xlabel('Total wind speed deficit [m/s] - scaled by '+str(weight_wt),fontsize=35)
plt.xlabel('Total wind speed deficit [m/s]',fontsize=35)
plt.ylabel('Total AEP deficit [MWh]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)

