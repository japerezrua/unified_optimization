# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:48:37 2022

@author: juru
"""
import numpy as np
from points_organizer import rotational_sort,convex_hull
import matplotlib.pylab as plt
from random_points import rand_points_internal,rand_points_boundary
from matrix_rotation import coordinates_translation
from AEP_IEATask import aep_calculator,aep_calculator_mo_wakes
from velocity_deficit import vel_deficit
import dill
import random
import math

def dcf(cost_energy,dis_rate,lifetime):  
    dcf_proj=0
    for i in range(lifetime):
        dcf_proj+=(cost_energy)/((1+dis_rate/100)**(i+1))   
    return dcf_proj

def linear_regression(organized_points,number_rep,min_distance,n_min,n_max,radius,hull,k_w,wt_model,freq_input_r,vel_set,prob_vel_set,cost_energy,dis_rate,lifetime,r=2,rand_points_incl=0,min_coeff_bor=0.1,max_coeff_bord=0.4):
    #boundaries,boundaries2,warm_start=circle_gen(radius,0.5,1)
    #boundaries=convex_hull(hull,boundaries)
    #organized_points=rotational_sort(boundaries,(sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
    diameter=wt_model.diameter()
    rand_points_diag=r*diameter #Changing diagonal distance between WTs to meters
    rand_points_ys=r*diameter #Changing vertical distance between WTs to meters
    wt_points=rand_points_internal(organized_points,rand_points_incl,rand_points_diag,rand_points_ys) #First seed of WT candidate locations   
    ini_wt=len(wt_points)  
    boundaries2=rand_points_boundary(organized_points,r*diameter)
    wt_points=np.concatenate((wt_points,boundaries2),axis=0)    
    fin_wt=len(wt_points) 
    print('WTs number linear regression: ',fin_wt)
    wt_points_freq,sorted_indexes=coordinates_translation(wt_points,freq_input_r)
    ct_all=wt_model.ct_all(vel_set) 
    ct_all=np.array([8/9]*len(vel_set))       
    Bil=vel_deficit(sorted_indexes,wt_points_freq,freq_input_r,vel_set,prob_vel_set,ct_all,wt_model,k_w) #objective function coefficients for WT layout
    #%% Generating random WT layouts
    X=wt_points[:,0] #Forming abscissas set 
    Y=wt_points[:,1] #Forming ordinates set 

    glob_cont=0
    v_deficit,power_produced,power_wasted,money_wasted=[],[],[],[]
    X1,Y1=X[:],Y[:]

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
            if flag and num_border>1:
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
            if counter>=num_border:
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
    plt.xlabel('Total wind speed deficit proxy [m/s]',fontsize=35)
    plt.ylabel('Total AEP [MWh]',fontsize=35)
    plt.tick_params(axis="y", labelsize=35)
    plt.tick_params(axis="x", labelsize=35)

    plt.figure()
    plt.scatter(v_deficit,money_wasted)
    plt.text(min(v_deficit),max(money_wasted),'r='+str(round(np.corrcoef(v_deficit,money_wasted)[0,1],2)),fontsize=35)
    #plt.xlabel('Total wind speed deficit [m/s] - scaled by '+str(weight_wt),fontsize=35)
    plt.xlabel('Total wind speed deficit proxy [m/s]',fontsize=35)
    plt.ylabel('Total money wasted due to wakes [Euros]',fontsize=35)
    plt.tick_params(axis="y", labelsize=35)
    plt.tick_params(axis="x", labelsize=35)


    outfilename = 'AEPvsVel_Borselle_site2.dill'

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
    print(f"slope: {model.coef_[0]}")
    plt.plot(x,model.predict(x))
    return model.intercept_,model.coef_[0],r_sq
    
def circle_gen(radius,steps_boundary,steps_candidates):

    boundaries=np.zeros((0,2))

    for i in np.arange(0,360,steps_boundary):
        x=radius*math.cos(i*math.pi/180)
        y=radius*math.sin(i*math.pi/180)
        boundaries=np.concatenate((boundaries,np.array([x,y]).reshape(-1,2)),axis=0)

    boundaries2=np.zeros((0,2))

    warm_start=np.zeros((0))

    fixed_points=np.zeros(0)



    for i in range(0,360,steps_candidates):
        x=radius*math.cos(i*math.pi/180)
        y=radius*math.sin(i*math.pi/180)
        boundaries2=np.concatenate((boundaries2,np.array([x,y]).reshape(-1,2)),axis=0)
    """    
    x_ws=np.array([-650,0,1170,780,130,-260,0,650,630.253,-90.6834,-1158.31,-1298.22,-1102.46,-423.239,1038.23,1296.83]).reshape(-1,1)
    y_ws=np.array([910,780,520,260,-260,-520,-910,-1040,1137.01,1296.83,590.188,68.0367,-688.895,-1229.17,-782.36,-90.6834]).reshape(-1,1)
    initial_length=len(boundaries2)
    points_ws=np.concatenate((x_ws,y_ws),axis=1)
    boundaries2=np.concatenate((boundaries2,points_ws),axis=0)
    final_length=len(boundaries2)
    warm_start=np.array(range(initial_length,final_length))

    """

      
    for i in range(len(boundaries2)):
        if i % 23 == 0:
        #if i % 10 == 0:
        #if i % 5 == 0 and i != 5 and i != 20 and i != 30 and i != 40 and i != 45 and i != 55 and i != 65 and i != 70:
            warm_start=np.concatenate((warm_start,np.array([i])))
            
    """

            
    for i in range(len(boundaries2)):
        if i % 46 == 0:
            fixed_points=np.concatenate((fixed_points,np.array([i])))


    min_r,max_r,num_r=0.30*radius,radius,4
    max_angle,min_angle=75,1

    radius_set=np.linspace(min_r,max_r,num_r)
    for i in radius_set:
        separation=(min_angle-max_angle)/(max_r-min_r)*(i-min_r)+max_angle
        for j in np.arange(0,360,separation):
            x=i*math.cos(j*math.pi/180)
            y=i*math.sin(j*math.pi/180)
            boundaries2=np.concatenate((boundaries2,np.array([x,y]).reshape(-1,2)),axis=0)
    """
    return  boundaries,boundaries2,warm_start
def irr_dtu(aep, electrical_connection_cost,ee_dtu,rated_rpm_array,Drotor_vector,Power_rated_array,hub_height_vector,water_depth_array):
    ee_dtu.calculate_irr(
                    rated_rpm_array, 
                    Drotor_vector, 
                    Power_rated_array,
                    hub_height_vector, 
                    water_depth_array, 
                    aep, 
                    electrical_connection_cost)
    return ee_dtu.IRR
