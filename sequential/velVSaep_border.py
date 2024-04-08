# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:17:57 2022

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
#from warmer import initial_solution
from wind_farm_performance import aep,npv
import time
#from mathiasmesh import boundaries,boundaries2,warm_start
from AEP_IEATask import aep_calculator,aep_calculator_mo_wakes
import cplex
from warmer import initial_solution
from sol_pool import all_solutions

from velocity_deficit import bil,vel_deficit,reductions

from circle import boundaries,boundaries2,warm_start
from proximity_search import neigh_search
import dill
from AEP_IEATask import aep_calculator
import random
import dill

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

n_max=16 #Max. number of WTs to install
n_min=16 #Max. number of WTs to install 

r=0.8
min_distance=2

number_rep=50000

min_coeff_bor=0.1 #Only one decimal
max_coeff_bord=0.4 #Only one decimal

cost_energy=50 #Cost of energy in Euro/MWh
dis_rate=5 #Dicount rate in percentage
lifetime=20 #years
#%% Step 1
wt_model=IEA3_5MW()
boundaries=convex_hull(hull,boundaries)
organized_points=rotational_sort(boundaries,(sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
diameter=wt_model.diameter()
rand_points_diag=r*diameter #Changing diagonal distance between WTs to meters
rand_points_ys=r*diameter #Changing vertical distance between WTs to meters
wt_points=rand_points(organized_points,rand_points_incl,rand_points_diag,rand_points_ys) #First seed of WT candidate locations   
ini_wt=len(wt_points)  
wt_points=np.concatenate((wt_points,boundaries2),axis=0)
fin_wt=len(wt_points) 
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
v_deficit,power_produced,power_wasted,money_wasted=[],[],[],[]
X1,Y1=X[:],Y[:]
def dcf(cost_energy,dis_rate,lifetime):  
    dcf_proj=0
    for i in range(lifetime):
        dcf_proj+=(cost_energy)/((1+dis_rate/100)**(i+1))   
    return dcf_proj
projection=dcf(cost_energy,dis_rate,lifetime)
while(True):
    n_wts=random.randint(n_min,n_max)
    num_border=random.randint(int(min_coeff_bor*n_wts),int(max_coeff_bord*n_wts))
    pot_wts_inside=np.array(range(ini_wt))
    pot_wts_border=np.array(range(ini_wt,fin_wt))
    counter=1
    
    #pot_wts=np.array(range(WTn))
    sel_wts=[]
    
    pick_up=random.choice(list(enumerate(pot_wts_border)))
    index,element=int(pick_up[0]),int(pick_up[1])
    sel_wts+=[element]
    pot_wts_border=np.delete(pot_wts_border,index)
    post_sec=np.copy(pot_wts_border)
    sel_wts_sec=sel_wts.copy()
    while(True):
        pick_up=random.choice(list(enumerate(pot_wts_border)))
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
        pot_wts_border=np.delete(pot_wts_border,index)        
        if len(pot_wts_border)==0:
            #print('Selection factor of [%]',n_wts*100/WTn)
            print('Conflict found while selecting randomly WTs in the border. Restarting process in the border')
            print('Selected up to', counter, 'WTs out of',n_wts)
            counter=1
            pot_wts_border=post_sec.copy()
            sel_wts=sel_wts_sec.copy()
        if counter==num_border:
            break    
    
    counter=len(sel_wts)
    post_sec=np.copy(pot_wts_inside)
    sel_wts_sec=sel_wts.copy()
    while(True):
        pick_up=random.choice(list(enumerate(pot_wts_inside)))
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
        pot_wts_inside=np.delete(pot_wts_inside,index)        
        if len(pot_wts_inside)==0:
            #print('Selection factor of [%]',n_wts*100/WTn)
            print('Conflict found while selecting randomly WTs inside of the shape. Restarting process inside of the shape')
            print('Selected up to', counter, 'WTs out of',n_wts)
            counter=len(sel_wts)
            pot_wts_inside=post_sec.copy()
            sel_wts=sel_wts_sec.copy()
        if counter==n_wts:
            break

    sel_wts.sort()
    red_Bil=Bil[sel_wts,:][:,sel_wts]
    vel_def=np.sum(red_Bil)
    IEA=aep_calculator(X[sel_wts].reshape(-1,1),Y[sel_wts].reshape(-1,1),vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)
    AEP_nowakes=aep_calculator_mo_wakes(X[sel_wts].reshape(-1,1),Y[sel_wts].reshape(-1,1),vel_set,prob_vel_set,freq_input_r,ct_all,k_w,wt_model,diameter)
    AEP_lost=AEP_nowakes-IEA
 
    v_deficit+=[vel_def]
    power_produced+=[IEA]
    power_wasted+=[AEP_lost]
    money_wasted+=[projection*AEP_lost]

    glob_cont+=1 
    print('Point',str(glob_cont), 'calculated')
    #plt.figure()
    #plt.plot(organized_points[:,0],organized_points[:,1], 'k-')
    #plt.scatter(X[sel_wts].reshape(-1,1),Y[sel_wts].reshape(-1,1),marker="o",color='r')
    if glob_cont==number_rep:
       break             

plt.figure()
plt.scatter(v_deficit,power_produced)
plt.text(min(v_deficit),max(power_produced),'r='+str(round(np.corrcoef(v_deficit,power_produced)[0,1],2)),fontsize=35)
#plt.xlabel('Total wind speed deficit [m/s] - scaled by '+str(weight_wt),fontsize=35)
plt.xlabel('Total wind speed deficit [m/s]',fontsize=35)
plt.ylabel('Total AEP deficit [MWh]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)

plt.figure()
plt.scatter(v_deficit,money_wasted)
plt.text(min(v_deficit),max(money_wasted),'r='+str(round(np.corrcoef(v_deficit,money_wasted)[0,1],2)),fontsize=35)
#plt.xlabel('Total wind speed deficit [m/s] - scaled by '+str(weight_wt),fontsize=35)
plt.xlabel('Total wind speed deficit [m/s]',fontsize=35)
plt.ylabel('Total money wated due to wakes [Euros]',fontsize=35)
plt.tick_params(axis="y", labelsize=35)
plt.tick_params(axis="x", labelsize=35)


outfilename = 'AEPvsVel_border_16WT.dill'

results = {
    'Velocity_deficit': v_deficit,
    'AEP produced': power_produced,
    'Power wasted': power_wasted,
    'Money wasted':money_wasted}

with open(outfilename, 'wb') as outfile:
    dill.dump(results, outfile)

print(outfilename, ' written')

from sklearn.linear_model import LinearRegression

x = np.array(v_deficit).reshape((-1, 1))
y = np.array(money_wasted)
model = LinearRegression(fit_intercept=True)
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
plt.plot(x,model.predict(x))
