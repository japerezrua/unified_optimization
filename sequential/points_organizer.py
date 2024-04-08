# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:37:11 2021

@author: juru
"""

from math import atan2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.spatial import ConvexHull

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(list_of_xy_coords, centre_of_rotation_xy_coord, clockwise=True):
    cx,cy=centre_of_rotation_xy_coord
    angles = [atan2(x-cx, y-cy) for x,y in list_of_xy_coords]
    indices = argsort(angles)
    output=np.zeros((0,2))
    if clockwise:
        for i in indices:
            output=np.concatenate((output,list_of_xy_coords[i].reshape(1,2)))        
    else:
        for i in indices[::-1]:
            output=np.concatenate((output,list_of_xy_coords[i].reshape(1,2)))
    indexes = sorted(np.unique(output, return_index=True,axis=0)[1])
    output=np.array([output[index] for index in indexes])
    output=np.concatenate((output,output[0,:].reshape(1,2)))
    return output
def point_in_polygon(organized_points,lines,pointx,pointy):
    #plt.figure()
    #plt.plot(organized_points[:,0],organized_points[:,1], 'k-')
    #plt.plot(pointx,pointy,marker="o")
    poly_path = mplPath.Path(organized_points)
    if not(poly_path.contains_point(tuple((pointx,pointy)))):
        flag=False
        for i in range(len(lines)):
            if abs(lines[i,0])<1e-10:
                if (pointx>=min(organized_points[i,0],organized_points[i+1,0])) and (pointx<=max(organized_points[i,0],organized_points[i+1,0])) and (abs(pointy-organized_points[i,1])<=1e-6):
                    flag=True
                    break
            elif lines[i,0]!=np.inf:
                y1=lines[i,0]*pointx+lines[i,1]
                if abs(y1-pointy)<=1e-6:
                    flag=True
                    break
            else:
                if (pointy>=min(organized_points[i,1],organized_points[i+1,1])) and (pointy<=max(organized_points[i,1],organized_points[i+1,1])) and (abs(pointx-organized_points[i,0])<=1e-6):
                    flag=True
                    break
    else:
        flag=True
    return flag
def lines_of_polygon(organized_points):
    lines=np.zeros((0,2))
    for i in range(len(organized_points)-1):
        if (organized_points[i+1,0]-organized_points[i,0])!=0:
            m=(organized_points[i+1,1]-organized_points[i,1])/(organized_points[i+1,0]-organized_points[i,0])
            b=-m*organized_points[i+1,0]+organized_points[i+1,1]
            lines=np.concatenate((lines,np.array([m,b]).reshape(1,2)),axis=0)
        else:
            lines=np.concatenate((lines,np.array([float('inf'),float('inf')]).reshape(1,2)),axis=0)
    return lines
def convex_hull(convex_hull,boundaries):
    if convex_hull:
        hull=ConvexHull(boundaries)
        boundaries=np.concatenate((boundaries[hull.simplices,0].flatten(order='F').reshape(-1,1),boundaries[hull.simplices,1].flatten(order='F').reshape(-1,1)),axis=1)
    return boundaries
if __name__ == "__main__":
    plt.figure()
    boundaries=np.array([[484178.55, 5732482.8], [500129.9, 5737534.4], [497318.1, 5731880.24], [488951.0, 5.72794e6], 
                    [488951.0, 5.72794e6], [497318.1, 5731880.24], [503163.37, 5729155.3], [501266.5, 5715990.05]])
    organized_points=rotational_sort(boundaries, (sum(boundaries[:,0]/len(boundaries[:,0])),sum(boundaries[:,1]/len(boundaries[:,1]))),True)
    organized_points=np.concatenate((organized_points,organized_points[0,:].reshape(1,2)))
    plt.plot(organized_points[:,0],organized_points[:,1], 'k-')


