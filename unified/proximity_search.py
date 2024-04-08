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
#%% Lazy constraints lines crossing constraints
class LazyConstraintsCross(LazyConstraintCallback):
      def __call__(self):
        num_int=0
        solution=self.get_values()
        x_crossings=solution[self.lim_inf:int(len(solution))]
        for i in range(self.length_var):
            if self.var_coll_output[i,0]!=0 and self.var_coll_output[i,1]!=0 and self.var_coll_output[i,3]==0:
                for j in range(i+1,self.length_var):
                    if self.var_coll_output[j,3]==0 and i!=j and x_crossings[i]>=0.9 and x_crossings[j]>=0.9:
                        line1=np.array([[self.X_all[self.var_coll_output[i,0].astype(int)-1],self.Y_all[self.var_coll_output[i,0].astype(int)-1]],\
                               [self.X_all[self.var_coll_output[i,1].astype(int)-1],self.Y_all[self.var_coll_output[i,1].astype(int)-1]]])
                        line2=np.array([[self.X_all[self.var_coll_output[j,0].astype(int)-1],self.Y_all[self.var_coll_output[j,0].astype(int)-1]],\
                                   [self.X_all[self.var_coll_output[j,1].astype(int)-1],self.Y_all[self.var_coll_output[j,1].astype(int)-1]]])
                        if two_lines_intersecting(line1,line2):
                            pos,val=[],[]
                            num_int+=1
                            pos1=[i+self.lim_inf,j+self.lim_inf]
                            val1=[1]*len(pos1)
                            potential1=list(np.where((self.var_coll_output[i,0]==self.var_coll_output[:,1]) & (self.var_coll_output[i,1]==self.var_coll_output[:,0]) & (0==self.var_coll_output[:,3]))[0])
                            pos2=[x.item()+self.lim_inf for x in potential1]
                            val2=[1]*len(pos2)
                            potential2=list(np.where((self.var_coll_output[j,0]==self.var_coll_output[:,1]) & (self.var_coll_output[j,1]==self.var_coll_output[:,0]) & (0==self.var_coll_output[:,3]))[0])
                            pos3=[x.item()+self.lim_inf for x in potential2]
                            val3=[1]*len(pos3)
                            pos=pos1+pos2+pos3
                            val=val1+val2+val3
                            constraints=cplex.SparsePair(pos,val)
                            self.add(constraint=constraints, sense="L", rhs=1)
        print('Intersections detected',num_int)
#%% User cut
#class UserCutCallback(UserCutCallback):
#      def __call__(self):
#          print(self.get_values()[:10])

def neigh_search(gap,time_limit,memory_limit,Bil,max_reductions,WTn,n_max,n_min,wts_lazy,min_distance,diameter,list_warm_wts,indices_all,values_all,neigh_max,OSSn,var_coll_output,objective_coll,X_all,Y_all,UL,C,T,coefficient,ws_active,pop,selected_wts,network_forbidden,Cables,Bigger,constraints_all,super_time_limit,t_initial):
    #%% Initialization
    tp=0
    t1=time.time()
    X,Y=X_all[OSSn:],Y_all[OSSn:]
    if ws_active: list_warm_wts = [int(x) for x in list_warm_wts]
    #if ws_active:list_warm_connections = [int(x) for x in list_warm_connections]
    #if ws_active: list_warm=list_warm_wts+list_warm_connections
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
    var_coll_output_length=len(var_coll_output)
    model.variables.add(types='B'*WTn+'C'*WTn+'C'*OSSn+'B'*(var_coll_output_length-OSSn))
    model.variables.set_upper_bounds([(i,max_reductions[i-WTn,0]) for i in range(WTn,WTn+WTn)]) #Velocity deficit variables
    model.variables.set_lower_bounds([(i,0) for i in range(WTn,WTn+WTn)]) #Velocity deficit variables
    model.variables.set_upper_bounds([(i,math.ceil(WTn/OSSn)) for i in range(2*WTn,2*WTn+OSSn)]) #Sigma variables OSSs
    model.variables.set_lower_bounds([(i,0) for i in range(2*WTn,2*WTn+OSSn)]) #Sigma variables OSSs
    model.variables.set_upper_bounds([(i+2*WTn,0) for i in range(var_coll_output_length) if var_coll_output[i,0]>OSSn and var_coll_output[i,2]<min_distance*diameter]) #Zeroing y vars shorter than min distance excluding OSSs feeders
       
    #model.variables.set_lower_bounds([(0,1)])
    #model.variables.set_lower_bounds([(int(x),1) for x in fixed_points+first_bunch_wt_variables])
    
    #model.variables.set_upper_bounds([(i,u_out) for i in range(WTn+var_power_output_length,WTn+var_power_output_length+var_states_length)])
    #model.variables.set_lower_bounds([(i,-u_out) for i in range(WTn+var_power_output_length,WTn+var_power_output_length+var_states_length)])
           
    #model.order.set([(int(x),300,model.order.branch_direction.up) for x in wt_variables_boundary])
    #model.order.set([(int(x),1000,model.order.branch_direction.up) for x in wt_variables_interior])
    #%% Objective function
    obj=np.concatenate((np.zeros(WTn),coefficient*np.ones(WTn),objective_coll[:,0]))
    model.objective.set_sense(model.objective.sense.minimize)
    model.objective.set_linear([(i,float(value)) for i,value in enumerate(obj)])
    #%%  Constraints
    if Bigger:
        constraints_all={}
        #%% C1: Number of selected WTs maximum equal to "n_max" (WTs layout)
        lhsC1 = cplex.SparsePair(list(range(WTn)),[1]*WTn)
        model.linear_constraints.add(lin_expr=[lhsC1], senses=["L"], rhs=[n_max])
        print("Constraint WFLO C1 has been formulated")
        constraints_all['C1']=lhsC1
        #%% C2: Number of selected WTs minimum equal to "n_min" (WTs layout)
        lhsC2 = cplex.SparsePair(list(range(WTn)),[1]*WTn)
        model.linear_constraints.add(lin_expr=[lhsC2], senses=["G"], rhs=[n_min])
        print("Constraint WFLO C2 has been formulated")    
        constraints_all['C2']=lhsC2
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
            constraints_all['C3']=lhsC3
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
        constraints_all['C4']=lhsC4
        #%% C5: Valid inequalities (WTs layout) - \tao_i>=\xi_i*MINIMUM (closest n_min-1 wind turbines affecting WT i)
        lhsC5=[]
        for i in range(WTn):
            temp=Bil[i,:]  
            minimum=sum(temp[np.argsort(temp)][:n_min-1])
            pos1=[i+WTn] #\tao_i
            val1=[1]
            pos2=[i] #xi_i
            val2=[-minimum]
            pos=pos1+pos2
            val=val1+val2    
            lhsC5.append(cplex.SparsePair(pos,val)) 
        model.linear_constraints.add(lin_expr=lhsC5, senses=["G"]*len(lhsC5), rhs=[0]*len(lhsC5))
        print("Constraint WFLO C5 has been formulated")  
        constraints_all['C5']=lhsC5
        #%% C6: OSSs clustering WTs (Collection system) - Constraints 2 IEEE Trans Power Systems paper
        lhsC6=[]
        for i in range(OSSn):
            initial=i*WTn*(UL+1)+OSSn+2*WTn
            final=(i+1)*WTn*(UL+1)+OSSn+2*WTn
            pos_aux=list(range(initial,final))
            val_aux=list(var_coll_output[np.array(pos_aux)-2*WTn,3])
            pos=[i+2*WTn]+pos_aux
            val=[-1]+val_aux
            lhsC6.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC6, senses=["E"]*len(lhsC6), rhs=[0]*len(lhsC6))
        print("Constraint CSO C6 has been formulated")  
        constraints_all['C6']=lhsC6
        #%% C7: Limiting number of main feeders to the OSSs (Collection system) - Constraints 3 IEEE Trans Power Systems paper
        lhsC7=[]
        for i in range(OSSn):
            initial=i*WTn*(UL+1)+OSSn+2*WTn
            final=(i+1)*WTn*(UL+1)+OSSn+2*WTn
            pos=list(range(initial,final))
            val=[1]*len(pos)
            for j in range(0,len(val),UL+1):
                val[j]=0 
            lhsC7.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC7, senses=["L"]*len(lhsC7), rhs=[C]*len(lhsC7))
        print("Constraint CSO C7 has been formulated") 
        constraints_all['C7']=lhsC7
        #%% C8: Defining crossing arcs (Collection system) - Constraints 6 IEEE Trans Power Systems paper     
        print("Constraint CSO C8 not to be generated") 
        #crossings_const=model.register_callback(LazyConstraintsCross)
        #crossings_const.var_coll_output=var_coll_output
        #crossings_const.length_var=var_coll_output_length    
        #crossings_const.X_all=X_all
        #crossings_const.Y_all=Y_all
        #crossings_const.lim_inf=2*WTn
        #%% C9: Deactivating crossing arcs (Collection system) - Constraints 7 IEEE Trans Power Systems paper     
        lhsC9=[]
        final=OSSn+2*WTn
        buffer=OSSn+2*WTn
        for i in range(WTn+OSSn):
            if (i+1)<=OSSn:
                connections=WTn
            else:
                connections=T[i-1]
            for j in range(connections):
                if (i+1)<=OSSn:
                   initial=final
                   final=((j+1)*(UL+1))+buffer
                   pos=list(range(initial,final))
                   val=[1]*len(pos)
                   val[0]=-1 
                else:
                   initial=final
                   final=((j+1)*(UL))+buffer
                   pos=list(range(initial,final))
                   val=[1]*len(pos)
                   val[0]=-1                     
                lhsC9.append(cplex.SparsePair(pos,val)) 
            buffer=final                   
        model.linear_constraints.add(lin_expr=lhsC9, senses=["E"]*len(lhsC9), rhs=[0]*len(lhsC9))
        print("Constraint CSO C9 has been formulated") 
        constraints_all['C9']=lhsC9
        #%% C10: Valid inequalities (Collection system) - Constraints 8 IEEE Trans Power Systems paper   
        lhsC10=[]   
        for i in range(OSSn+1,WTn+OSSn+1):
            for j in range(2,UL):
                pos1=list(np.where((var_coll_output[:,1]==i) & (var_coll_output[:,3]>=j+1) & (var_coll_output[:,3]<=UL))[0]+2*WTn)
                pos1=[x.item() for x in pos1]
                pos2=list(np.where((var_coll_output[:,0]==i) & (var_coll_output[:,3]>=j) & (var_coll_output[:,3]<=UL-1))[0]+2*WTn)
                pos2=[x.item() for x in pos2] 
                val1=[1]*len(pos1)
                val2=[1]*len(pos2)
                for k in range(len(pos1)):
                    val1[k]=-math.floor((var_coll_output[pos1[k]-2*WTn,3]-1)/j)
                pos=pos1+pos2
                val=val1+val2            
                lhsC10.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC10, senses=["L"]*len(lhsC10), rhs=[0]*len(lhsC10))
        print("Constraint CSO C10 has been formulated")
        constraints_all['C10']=lhsC10
       #%% C11: OSSs supporting all WTs (WTs layout and Collection system) - Constraints 1 IEEE Trans Power Systems paper       
        lhsC11=[]
        pos1=[i for i in range(2*WTn,2*WTn+OSSn)]
        pos2=[i for i in range(WTn)]
        val1=[1]*len(pos1)
        val2=[-1]*len(pos2)
        lhsC11=cplex.SparsePair(pos1+pos2,val1+val2)
        model.linear_constraints.add(lin_expr=[lhsC11], senses=["E"], rhs=[0])   
        print("Constraint WFLO plus CSO C11 has been formulated")
        constraints_all['C11']=lhsC11
        #%% C12: Tree topology (WTs layout and Collection system) - Constraints 4 IEEE Trans Power Systems paper      
        lhsC12=[]
        for i in range(OSSn+1,WTn+OSSn+1):       
            pos1=list(np.where((var_coll_output[:,1]==i) & (var_coll_output[:,3]>0))[0]+2*WTn)
            pos1=[x.item() for x in pos1]
            pos2=[i-OSSn-1]
            val1=[1]*len(pos1)
            val2=[-1]*len(pos2)
            pos=pos1+pos2
            val=val1+val2        
            lhsC12.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC12, senses=["E"]*len(lhsC12), rhs=[0]*len(lhsC12))
        print("Constraint WFLO plus CSO C12 has been formulated")
        constraints_all['C12']=lhsC12
        #%% C13: Flow conservation (WTs layout and Collection system) - Constraints 5 IEEE Trans Power Systems paper 
        lhsC13=[]
        for i in range(OSSn+1,WTn+OSSn+1):
            pos1=list(np.where((var_coll_output[:,1]==i) & (var_coll_output[:,3]>0))[0]+2*WTn)
            pos1=[x.item() for x in pos1]
            pos2=list(np.where((var_coll_output[:,0]==i) & (var_coll_output[:,3]>0))[0]+2*WTn)
            pos2=[x.item() for x in pos2]
            pos3=[i-OSSn-1]
            val1=list(var_coll_output[np.array(pos1)-2*WTn,3])
            val2=list(-1*var_coll_output[np.array(pos2)-2*WTn,3])
            val3=[-1]*len(pos3)           
            pos=pos1+pos2+pos3
            val=val1+val2+val3
            lhsC13.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC13, senses=["E"]*len(lhsC13), rhs=[0]*len(lhsC13))
        print("Constraint WFLO plus CSO C13 has been formulated")
        constraints_all['C13']=lhsC13
        #%% C14: Valid inequalities (WTs layout and Collection system) - New ones: x_ij+x_ji<=0.5\xi_i+0.5\xi_j
        lhsC14=[]    
        forbidden_list=[]
        for i in range(var_coll_output_length):
            if i>OSSn-1 and var_coll_output[i,3]==0:
               first_node=var_coll_output[i,0]
               if first_node<=OSSn:
                   pos1=[i+2*WTn]
                   val1=[1]
                   pos2=[int(var_coll_output[i,1])-OSSn-1]
                   val2=[-1]
                   pos=pos1+pos2
                   val=val1+val2
                   lhsC14.append(cplex.SparsePair(pos,val))
               else:
                   if not(i in forbidden_list):
                       pos1=[i+2*WTn]
                       val1=[1]
                       pos2=[int(var_coll_output[i,0])-OSSn-1]
                       val2=[-0.5]
                       pos3=[int(var_coll_output[i,1])-OSSn-1]
                       val3=[-0.5]                 
                       pos4=list(np.where((var_coll_output[i+1:,1]==var_coll_output[i,0]) & (var_coll_output[i+1:,0]==var_coll_output[i,1]) & (var_coll_output[i+1:,3]==0))[0]+i+1+2*WTn)
                       pos4=[x.item() for x in pos4]
                       if len(pos4)>0: forbidden_list.append(pos4[0]-2*WTn)
                       if len(pos4)>1: raise Exception('Should not be the size larger than one here')
                       val4=[1]*len(pos4)
                       pos=pos1+pos2+pos3+pos4
                       val=val1+val2+val3+val4
                       lhsC14.append(cplex.SparsePair(pos,val))   
        model.linear_constraints.add(lin_expr=lhsC14, senses=["L"]*len(lhsC14), rhs=[0]*len(lhsC14))
        print("Constraint WFLO plus CSO C14 has been formulated")  
        constraints_all['C14'] = lhsC14
    else:
        lhsC1,lhsC2,lhsC3,lhsC4,lhsC5,lhsC6,lhsC7,lhsC9,lhsC10,lhsC11,lhsC12,lhsC13,lhsC14=constraints_all['C1'],constraints_all['C2'],constraints_all['C3'],constraints_all['C4'],constraints_all['C5'],constraints_all['C6'],constraints_all['C7'],constraints_all['C9'],constraints_all['C10'],constraints_all['C11'],constraints_all['C12'],constraints_all['C13'],constraints_all['C14']                     
        model.linear_constraints.add(lin_expr=[lhsC1], senses=["L"], rhs=[n_max])
        model.linear_constraints.add(lin_expr=[lhsC2], senses=["G"], rhs=[n_min])
        model.linear_constraints.add(lin_expr=lhsC3, senses=["L"]*len(lhsC3), rhs=[1]*len(lhsC3))
        model.linear_constraints.add(lin_expr=lhsC4, senses=["L"]*len(lhsC4), rhs=[float(value) for value in max_reductions])
        model.linear_constraints.add(lin_expr=lhsC5, senses=["G"]*len(lhsC5), rhs=[0]*len(lhsC5))
        model.linear_constraints.add(lin_expr=lhsC6, senses=["E"]*len(lhsC6), rhs=[0]*len(lhsC6))
        model.linear_constraints.add(lin_expr=lhsC7, senses=["L"]*len(lhsC7), rhs=[C]*len(lhsC7))
        model.linear_constraints.add(lin_expr=lhsC9, senses=["E"]*len(lhsC9), rhs=[0]*len(lhsC9))
        model.linear_constraints.add(lin_expr=lhsC10, senses=["L"]*len(lhsC10), rhs=[0]*len(lhsC10))
        model.linear_constraints.add(lin_expr=[lhsC11], senses=["E"], rhs=[0])  
        model.linear_constraints.add(lin_expr=lhsC12, senses=["E"]*len(lhsC12), rhs=[0]*len(lhsC12))
        model.linear_constraints.add(lin_expr=lhsC13, senses=["E"]*len(lhsC13), rhs=[0]*len(lhsC13))
        model.linear_constraints.add(lin_expr=lhsC14, senses=["L"]*len(lhsC14), rhs=[0]*len(lhsC14))
    #%% C15: Neighborhood search (WTs layout)
    if ws_active:
        lhsCn=[]
        list_inactive = [x for x in list(range(WTn)) if x not in list_warm_wts]
        list_inactive = [int(x) for x in list_inactive]
        lhsCn.append(cplex.SparsePair(list_inactive+list_warm_wts,[1]*len(list_inactive)+[-1]*len(list_warm_wts)))
        model.linear_constraints.add(lin_expr=lhsCn, senses=["L"]*len(lhsCn), rhs=[neigh_max-len(list_warm_wts)]*len(lhsCn))    
        print("Constraint C15 neighborhood search has been formulated")              
        warm_aux=cplex.SparsePair(indices_all,values_all)
        #model.MIP_starts.add(warm_aux,model.MIP_starts.effort_level.solve_MIP,'warm start number '+str(1)) 
        model.MIP_starts.add(warm_aux,model.MIP_starts.effort_level.no_check,'warm start number '+str(1)) 
    else:
        timelim_init=model.register_callback(TimeLimitCallbackIni)
        timelim_init.starttime = model.get_time()
        #timelim_cb.lastincumbent = 1e+15
        #timelim_cb.accum_inc=[]
        #timelim_cb.accum_time_inc=[]
        #timelim_cb.accum_db=[]
        #timelim_cb.accum_time_db=[]    
        ### Solving model
    timelim_full=model.register_callback(TimeLimitCallbackFull)
    timelim_full.starttime = model.get_time()
    timelim_full.time_limit=time_limit
    timelim_full.super_limit=super_time_limit
    timelim_full.now=time.time()-t_initial
    #warm=cplex.SparsePair(list_warm,[1]*len(list_warm))
    #model.MIP_starts.add(warm,model.MIP_starts.effort_level.solve_MIP,'warm start number 1')
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
    return model,tc,ts,tp,constraints_all
class TimeLimitCallbackIni(MIPInfoCallback):
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
            gap = 100.0 * self.get_MIP_relative_gap()
            timeused = self.get_time() - self.starttime
            if gap<10 and timeused>600:
               print("Good enough solution at", timeused, "sec., gap =",
                      gap, "%, quitting.")
               self.abort()
            #    print("Good enough solution at", timeused, "sec., gap =",
            #          gap, "%, quitting.")
            #    self.aborted = Tr

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

