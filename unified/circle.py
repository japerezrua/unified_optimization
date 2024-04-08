# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:42:02 2021

@author: juru
"""
import numpy as np
import math

#radius=3000
#radius=2000
radius=1300
steps_boundary=0.5
steps_candidates=5

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