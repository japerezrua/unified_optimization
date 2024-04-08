# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:00:15 2021

@author: juru
"""
import math
import numpy as np

def coordinates_rotation(angle,x,y):
    angle=-(90+angle)
    xt=x*math.cos(angle*math.pi/180)+y*math.sin(angle*math.pi/180)
    yt=-x*math.sin(angle*math.pi/180)+y*math.cos(angle*math.pi/180)
    return xt,yt

def coordinates_translation(wt_points,freq_input_r):
    length_points,length_freq=len(wt_points),len(freq_input_r)
    wt_points_freq=np.zeros((length_points,2,length_freq))
    sorted_indexes=np.zeros((length_points,length_freq))
    for i in range(length_freq):
        wt_points_freq_t=np.zeros((len(wt_points),2))
        for j in range(length_points):
            #if i>0:
            xt,yt=coordinates_rotation(freq_input_r[i,0],wt_points[j,0],wt_points[j,1])
            wt_points_freq_t[j,:]=np.array([xt,yt]).reshape(1,2)
        #if i>0:
        index_sorted=wt_points_freq_t[:,0].argsort()
        wt_points_freq[:,:,i]=wt_points_freq_t[index_sorted,:]
        sorted_indexes[:,i]=index_sorted
        #else:
            #wt_points_freq[:,:,i]=wt_points
            #sorted_indexes[:,i]=np.array(range(length_points))
    return wt_points_freq,sorted_indexes

if __name__ == "__main__":
    x1=7
    y1=7
    x2=7
    y2=5    
    angle=-90
    x1t,y1t=coordinates_rotation(angle,x1,y1)
    x2t,y2t=coordinates_rotation(angle,x2,y2)