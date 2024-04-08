# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 09:19:01 2022

@author: juru
"""
import numpy as np
import cplex
import time
import math
from cplex.callbacks import MIPInfoCallback, LazyConstraintCallback, UserCutCallback
from TwoLinesIntersecting import two_lines_intersecting

#%% Lazy constraints Minimum distance WTs constraints
class LazyConstraintsWTs(LazyConstraintCallback):
      def __call__(self):
        solution=self.get_values()
        num_wts_close=0
        x_wts=solution[:self.WTn]
        for i in range(self.WTn):
            for j in range(i+1,self.WTn):
                if x_wts[i]>=0.9 and x_wts[j]>=0.9:
                    dist_wts=np.sqrt((self.X[i]-self.X[j])**2+(self.Y[i]-self.Y[j])**2) 
                    if dist_wts<self.min_distance:
                       pos,val=[],[]
                       num_wts_close+=1
                       pos1=[int(i)]
                       val1=[1]
                       pos2=[int(j)]
                       val2=[1]
                       pos=pos1+pos2
                       val=val1+val2
                       constraints=cplex.SparsePair(pos,val)
                       self.add(constraint=constraints, sense="L", rhs=1)

        print('Number of pairs of WTs too close to each other',num_wts_close) 
#%% User cut
#class UserCutCallback(UserCutCallback):
#      def __call__(self):
#          print(self.get_values()[:10])

def neigh_search(gap,time_limit,memory_limit,Bil,max_reductions,WTn,n_max,n_min,wts_lazy,min_distance,diameter,list_warm_wts,neigh_max,X_all,Y_all,super_time_limit,t_initial):

    #%% Initialization
    tp=0
    t1=time.time()
    X,Y=X_all,Y_all
    list_warm_wts = [int(x) for x in list_warm_wts]
    #%% TUNING SETTINGS
    model = cplex.Cplex()
    model.parameters.mip.strategy.probe.set(-1)
    model.parameters.threads.set(10)
    model.parameters.parallel.set(-1) #-1 Is Opportunistic, 0 is let cplex decide, 1 is determistic
    model.parameters.mip.tolerances.mipgap.set(gap/100) 
    model.parameters.timelimit.set(time_limit)
    model.parameters.mip.limits.treememory.set(memory_limit)
    model.parameters.mip.strategy.variableselect.set(3)
    #model.parameters.get_all()
    #model.parameters.mip.display.set(4) #Mip display
    model.parameters.emphasis.mip.set(5) #Mip emphasis
    #model.parameters.mip.strategy.heuristicfreq.set(-1) #Mip node heuristic
    #model.parameters.mip.strategy.startalgorithm.set(2) #Algorithm for initial MIP relaxation
    #model.parameters.benders.strategy.set(3)
    #model.parameters.mip.pool.intensity.set(2)
    
    #%% Variables
    model.variables.add(types='B'*WTn+'C'*WTn)
    model.variables.set_upper_bounds([(i,max_reductions[i-WTn,0]) for i in range(WTn,WTn+WTn)]) #Velocity deficit variables
    model.variables.set_lower_bounds([(i,0) for i in range(WTn,WTn+WTn)]) #Velocity deficit variables      
    #model.variables.set_lower_bounds([(0,1)])
    #model.variables.set_lower_bounds([(int(x),1) for x in fixed_points+first_bunch_wt_variables])   
    #model.variables.set_upper_bounds([(i,u_out) for i in range(WTn+var_power_output_length,WTn+var_power_output_length+var_states_length)])
    #model.variables.set_lower_bounds([(i,-u_out) for i in range(WTn+var_power_output_length,WTn+var_power_output_length+var_states_length)])          
    #model.order.set([(int(x),300,model.order.branch_direction.up) for x in wt_variables_boundary])
    #model.order.set([(int(x),1000,model.order.branch_direction.up) for x in wt_variables_interior])
    #%% Objective function
    obj=np.concatenate((np.zeros(WTn),np.ones(WTn)))
    model.objective.set_sense(model.objective.sense.minimize)
    model.objective.set_linear([(i,float(value)) for i,value in enumerate(obj)])
    #%%  Constraints
    #%% C1: Number of selected WTs maximum equal to "n_max" (WTs layout)
    lhsC1 = cplex.SparsePair(list(range(WTn)),[1]*WTn)
    model.linear_constraints.add(lin_expr=[lhsC1], senses=["L"], rhs=[n_max])
    print("Constraint WFLO C1 has been formulated")
    #%% C2: Number of selected WTs minimum equal to "n_min" (WTs layout)
    lhsC2 = cplex.SparsePair(list(range(WTn)),[1]*WTn)
    model.linear_constraints.add(lin_expr=[lhsC2], senses=["G"], rhs=[n_min])
    print("Constraint WFLO C2 has been formulated")    
    #%% C3: Minimum distance between WTs (WTs layout)
    if not wts_lazy:
        lhsC3=[]
        for i in range(WTn):
            for j in range(i+1,WTn):
                dist_wts=np.sqrt((X[i]-X[j])**2+(Y[i]-Y[j])**2)
                if dist_wts<min_distance*diameter:
                   pos,val=[],[]
                   pos1=[int(i)]
                   val1=[1]
                   pos2=[int(j)]
                   val2=[1]
                   pos=pos1+pos2
                   val=val1+val2
                   lhsC3.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC3, senses=["L"]*len(lhsC3), rhs=[1]*len(lhsC3))
        print("Constraint WFLO C3 has been formulated")  
    else:
        print("Constraint WFLO C3 to be generated on the fly") 
        wts_min_di_const=model.register_callback(LazyConstraintsWTs)
        wts_min_di_const.X=X
        wts_min_di_const.Y=Y
        wts_min_di_const.WTn=WTn
        wts_min_di_const.min_distance=min_distance*diameter        
    #%% C4: Expected aggregate kinetic energy deficit for each WT (WTs layout)
    lhsC4=[]
    for i in range(WTn):
        disturbers=list(range(WTn))
        disturbers.remove(i)
        pos1=disturbers.copy()
        val1=list(Bil[i,disturbers])
        pos2=[i+WTn]
        val2=[-1]
        pos3=[i]
        val3=[max_reductions[i,0]]
        pos=pos1+pos2+pos3
        val=val1+val2+val3          
        lhsC4.append(cplex.SparsePair(pos,val))        
    model.linear_constraints.add(lin_expr=lhsC4, senses=["L"]*len(lhsC4), rhs=[float(value) for value in max_reductions])
    print("Constraint WFLO C4 has been formulated")
#    #%% C5: Valid inequalities (WTs layout) - \tao_i>=\xi_i*MINIMUM (closest n_min-1 wind turbines affecting WT i)
#    lhsC5=[]
#    for i in range(WTn):
#        temp=Bil[i,:]  
#        minimum=sum(temp[np.argsort(temp)][:n_min-1])
#        pos1=[i+WTn] #\tao_i
#        val1=[1]
#        pos2=[i] #xi_i
#        val2=[-minimum]
#        pos=pos1+pos2
#        val=val1+val2    
#        lhsC5.append(cplex.SparsePair(pos,val)) 
#    model.linear_constraints.add(lin_expr=lhsC5, senses=["G"]*len(lhsC5), rhs=[0]*len(lhsC5))
#    print("Constraint WFLO C5 has been formulated")       
    #%% C5: Neighborhood search (WTs layout)
    lhsCn=[]
    list_inactive = [x for x in list(range(WTn)) if x not in list_warm_wts]
    list_inactive = [int(x) for x in list_inactive]
    lhsCn.append(cplex.SparsePair(list_inactive+list_warm_wts,[1]*len(list_inactive)+[-1]*len(list_warm_wts)))
    model.linear_constraints.add(lin_expr=lhsCn, senses=["L"]*len(lhsCn), rhs=[neigh_max-len(list_warm_wts)]*len(lhsCn))    
    print("Constraint C5 neighborhood search has been formulated")              
    warm_aux=cplex.SparsePair(list_warm_wts,[1]*len(list_warm_wts))
    #model.MIP_starts.add(warm_aux,model.MIP_starts.effort_level.solve_MIP,'warm start number '+str(1)) 
    model.MIP_starts.add(warm_aux,model.MIP_starts.effort_level.solve_MIP,'warm start number '+str(1)) 
    
    timelim_full=model.register_callback(TimeLimitCallbackFull)
    timelim_full.starttime = model.get_time()
    timelim_full.time_limit=time_limit
    timelim_full.super_limit=super_time_limit
    timelim_full.now=time.time()-t_initial
    #warm=cplex.SparsePair(list_warm,[1]*len(list_warm))
    print("Solving model")
    tc=time.time()-t1
    #my_cuts=model.register_callback(UserCutCallback)
    #%% Running the model
    model.solve()
    ts=time.time()-tc-t1
    #%% Populating even more the pool
    """
    if pop:
        model.unregister_callback(TimeLimitCallbackFull)
        model.parameters.mip.limits.populate.set(30)
        model.parameters.timelimit.set(0.2*time_limit) 
        timelim_full=model.register_callback(TimeLimitCallbackFull)
        timelim_full.starttime = model.get_time() 
        timelim_full.time_limit=0.2*time_limit
        model.populate_solution_pool()
        tp=time.time()-t1-tc-ts  
    """
    return model,tc,ts,tp
class TimeLimitCallbackFull(MIPInfoCallback):
    def __call__(self):
        #newincumbent = False
        #hasincumbent = self.has_incumbent()
        #if hasincumbent:
        #    incobjvalue = self.get_incumbent_objective_value()
        #    if abs(self.lastincumbent - incobjvalue) > 1e-3:
        #        self.lastincumbent = incobjvalue
        #        newincumbent = True
        #if newincumbent:
        #    timeused_inc = self.get_time() - self.starttime
        #    self.accum_inc+=[incobjvalue]
        #    self.accum_time_inc+=[timeused_inc]
            #print("New incumbent variable objective:", incval)
            #print("New incumbent variable time:", timeused)
        #self.accum_db+=[self.get_best_objective_value()]
        #self.accum_time_db+=[self.get_time() - self.starttime]
        #print(self.get_best_objective_value())
        if self.has_incumbent():
            #gap = 100.0 * self.get_MIP_relative_gap()
            timeused = self.get_time() - self.starttime
            if timeused>self.time_limit:
               print("I had enough at", timeused, "sec.")
               #       gap, "%, quitting.")
               self.abort()
            if timeused+self.now>self.super_limit*0.97:
               print("Closing the whole program")
               #       gap, "%, quitting.")
               self.abort()                
            #    print("Good enough solution at", timeused, "sec., gap =",
            #          gap, "%, quittin
"""
class TimeLimitCallback(MIPInfoCallback):
    def __call__(self):
        newincumbent = False
        hasincumbent = self.has_incumbent()
        if hasincumbent:
            incobjvalue = self.get_incumbent_objective_value()
            if fabs(self.lastincumbent - incobjvalue) > 1e-3:
                self.lastincumbent = incobjvalue
                newincumbent = True
        if newincumbent:
            timeused_inc = self.get_time() - self.starttime
            self.accum_inc+=[incobjvalue]
            self.accum_time_inc+=[timeused_inc]
            #print("New incumbent variable objective:", incval)
            #print("New incumbent variable time:", timeused)
        self.accum_db+=[self.get_best_objective_value()]
        self.accum_time_db+=[self.get_time() - self.starttime]
        #print(self.get_best_objective_value())
        #if not self.aborted and self.has_incumbent():
            #gap = 100.0 * self.get_MIP_relative_gap()
            #timeused = self.get_time() - self.starttime
            #if timeused > self.timelimit and gap < self.acceptablegap:
            #    print("Good enough solution at", timeused, "sec., gap =",
            #          gap, "%, quitting.")
            #    self.aborted = Tr
"""

