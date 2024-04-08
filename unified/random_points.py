# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:54:29 2021

@author: juru
"""
from points_organizer import point_in_polygon,lines_of_polygon
import math
import numpy as np
import matplotlib.pyplot as plt

def rand_points_internal(organized_points,rand_points_incl,rand_points_xs,rand_points_ys):
    #%% Internal points inside of the polygon
    inf_slope,zero_slope,pos_slope,neg_slope=False, False, False, False
    slope=math.tan(rand_points_incl*math.pi/180)
    lines=lines_of_polygon(organized_points)
    min_x,max_x,min_y,max_y=min(organized_points[:,0]),max(organized_points[:,0]),min(organized_points[:,1]),max(organized_points[:,1])
    if abs(slope)>1e10:    #infinite slope
        inf_slope=True
    elif abs(slope)<1e-10: #zero slope
        zero_slope=True
    elif slope>0:          #positive slope
        pos_slope=True
    elif slope<0:          #negative slope
        neg_slope=True
    dist=abs(rand_points_xs*math.cos(rand_points_incl*math.pi/180))
    if pos_slope:
       o_points=np.zeros((0,2))
       max_y_c=max_y
       b=(-slope*min_x)+(max_y_c)
       x_upper=(max_y-b)/(slope)
       x_lower=(min_y-b)/(slope)
       x_upper=min(x_upper,max_x)
       x_lower=max(x_lower,min_x)
       x_set=np.arange(start=x_lower,stop=x_upper+dist,step=dist)
       y_set=slope*x_set+b  
       while(True):
          for i in range(len(x_set)):
              flag=point_in_polygon(organized_points,lines,x_set[i],y_set[i])
              if flag:
                  o_points=np.concatenate((o_points,np.array([x_set[i],y_set[i]]).reshape(1,2)),axis=0)
          max_y_c-=rand_points_ys
          b=(-slope*min_x)+(max_y_c)
          x_upper=(max_y-b)/(slope)
          inters=(min_y-b)/(slope)
          if inters>max_x:
              break
          x_lower=inters
          x_upper=min(x_upper,max_x)
          x_lower=max(x_lower,min_x)
          x_set=np.arange(start=x_lower,stop=x_upper+dist,step=dist)
          y_set=slope*x_set+b   
    if neg_slope:
       o_points=np.zeros((0,2))
       max_y_c=max_y
       b=(-slope*max_x)+(max_y_c)
       x_upper=(min_y-b)/(slope)
       x_lower=(max_y-b)/(slope)
       x_upper=min(x_upper,max_x)
       x_lower=max(x_lower,min_x)
       x_set=np.arange(start=x_lower,stop=x_upper+dist,step=dist)
       y_set=slope*x_set+b
       while(True):
           for i in range(len(x_set)):
               flag=point_in_polygon(organized_points,lines,x_set[i],y_set[i])
               if flag:
                  o_points=np.concatenate((o_points,np.array([x_set[i],y_set[i]]).reshape(1,2)),axis=0)
           max_y_c-=rand_points_ys
           b=(-slope*max_x)+(max_y_c)
           x_lower=(max_y-b)/(slope)
           inters=(min_y-b)/(slope)
           if inters<min_x:
              break
           x_upper=inters
           x_upper=min(x_upper,max_x)
           x_lower=max(x_lower,min_x)
           x_set=np.arange(start=x_lower,stop=x_upper+dist,step=dist)
           y_set=slope*x_set+b         
    if zero_slope:
       o_points=np.zeros((0,2))
       max_y_c=max_y
       x_set=np.arange(start=min_x,stop=max_x+dist,step=dist)
       length_x=len(x_set)
       y_set=max_y_c
       while(True):
           for i in range(length_x):
               flag=point_in_polygon(organized_points,lines,x_set[i],y_set)
               if flag:
                  o_points=np.concatenate((o_points,np.array([x_set[i],y_set]).reshape(1,2)),axis=0)
           max_y_c-=rand_points_ys
           if max_y_c<min_y:
              break
           y_set=max_y_c
    if inf_slope:
       o_points=np.zeros((0,2))
       min_x_c=min_x
       y_set=np.arange(start=min_y,stop=max_y+rand_points_ys,step=rand_points_ys)
       length_y=len(y_set)
       x_set=min_x_c
       while(True):
           for i in range(length_y):
               flag=point_in_polygon(organized_points,lines,x_set,y_set[i])
               if flag:
                  o_points=np.concatenate((o_points,np.array([x_set,y_set[i]]).reshape(1,2)),axis=0)
           min_x_c+=rand_points_xs
           if min_x_c>max_x:
              break
           x_set=min_x_c
    return o_points

def rand_points_boundary(organized_points,rand_points_border_diag):
    lines=lines_of_polygon(organized_points)
    o_points=np.zeros((0,2))
    for i,line in enumerate(lines):
        if line[0]!=np.inf:
           dist=abs(rand_points_border_diag*math.cos(math.atan(line[0])))
           min_x=min(organized_points[i,0],organized_points[i+1,0]) 
           max_x=max(organized_points[i,0],organized_points[i+1,0])
           if organized_points[i+1,0]>=organized_points[i,0]:
               x_set=np.arange(start=min_x,stop=max_x,step=dist)
           else:
               x_set=np.arange(start=max_x,stop=min_x,step=-dist)
           y_set=line[0]*x_set+line[1]
           o_points=np.concatenate((o_points,np.concatenate((x_set.reshape(-1,1),y_set.reshape(-1,1)),axis=1)),axis=0)
        else:
           min_y=min(organized_points[i,1],organized_points[i+1,1]) 
           max_y=max(organized_points[i,1],organized_points[i+1,1])
           if organized_points[i+1,1]==max_y:
               y_set=np.arange(start=min_y,stop=max_y,step=rand_points_border_diag)
           else:
               y_set=np.arange(start=max_y,stop=min_y,step=-rand_points_border_diag)
           x_set=organized_points[i,0]*np.ones((len(y_set),1))
           o_points=np.concatenate((o_points,np.concatenate((x_set,y_set.reshape(-1,1)),axis=1)),axis=0)
    return o_points
    
    