# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:24:00 2021

@author: juru
"""
import random
import numpy as np
import cplex

def initial_solution(X,Y,WTn,n_min,n_max,min_distance,number_ws):
    glob_cont=0
    war_sols=[]
    while(True):    
        n_wts=random.randint(n_min,n_max)
        X1,Y1=np.copy(X),np.copy(Y)
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
                print('Selection factor of [%]',n_wts*100/WTn)
                print('Conflict found while selecting randomly WTs. Restarting process')
                print('Selected up to', counter, 'WTs out of',n_wts)
                counter=1
                pot_wts=post_sec.copy()
                sel_wts=sel_wts_sec.copy()
            if counter==n_wts:
                print('Selection factor of [%]',n_wts*100/WTn)
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
        pos1=sel_wts.copy()
        val1=[1]*len(pos1)


        glob_cont+=1        
        #warm=cplex.SparsePair(pos+pos5,val1+val3+val4+val2+val5) 
        warm=cplex.SparsePair(pos1,val1) 
        war_sols+=[warm]
        print('Warm-start number ',str(glob_cont), 'obtained')
        if glob_cont==number_ws:
           break               
    return war_sols