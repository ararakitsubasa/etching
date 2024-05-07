import numpy as np
import torch
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import math
from math import pi


class reflect:
    def __init__(self, center_with_direction, range3D, yield_hist = None):
        self.center_with_direction = center_with_direction
        # boundary x1x2 y1y2 z1z2
        self.range3D = range3D 
        if yield_hist == None:
            self.yield_hist = np.array([[1.0, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0], \
                                        [  0,   10,   20,   30,   40,   50,   60,   70,   80, 90]])
        else:
            self.yield_hist = yield_hist

    def scanZ(film):
        xshape = film.shape[0]
        yshape = film.shape[1]
        zshape = film.shape[2]
        surface_sparse = torch.zeros((xshape, yshape, zshape))
        print(zshape)
        for i in range(zshape-1):
            for j in range(xshape-1):
                for k in range(yshape-1):
                    if (film[j, k, i] == 0 and film[j, k, i-1] != 0) or (film[j, k, i] == 0 and film[j, k, i+1] != 0) \
                    or (film[j, k, i] == 0 and film[j-1, k, i] != 0) or (film[j, k, i] == 0 and film[j+1, k, i] != 0) \
                    or (film[j, k, i] == 0 and film[j, k-1, i] != 0) or (film[j, k, i] == 0 and film[j, k+1, i] != 0):
                        surface_sparse[j,k,i] = 1
        return surface_sparse.to_sparse()
    
    def normalconsistency_3D_real(self, planes):
        
        """
        
        This function checks wherer the normals are oriented towards the outside of the surface, i.e., it 
        checks the consistency of the normals.
        The function changes the direction of the normals that do not point towards the outside of the shape
        The function checks whether the normals are oriented towards the centre of the ellipsoid, 
        and if YES, then, it turns their orientation
        
        INPUTS:
            planes: Vector N x 6, where M is the number of points whose normals and 
            centroid have been calculated. the columns are the coordinates of the normals and the centroids
            
        OUTPUTS:
            planesconsist: N x 6 array, where N is the number of points whose planes have been calculated. This array 
            has all the planes normals pointing outside the surface.
            
        """
        
        nbnormals = np.size(planes, 0)
        # planes_consist=np.zeros((nbnormals,6))
        planes_consist = []
        # planes_consist[:, 3:6] = planes[:, 3:6] # We just copy the columns corresponding to the coordinates of the centroids (from 3th to 5th)
        
        """ Try the atan2 function : https://uk.mathworks.com/help/vision/ref/pcnormals.html#buxdmoj"""
        
        # sensorcentre=np.array([0,0,0])
        sensorcentre=np.array([100,100,0]) # vertax shape

        for c in range(self.center_with_direction.shape[0]):
            sensorcentre = self.center_with_direction[c]
            sensorrange = self.range3D[c]

            planes_in_range_indice  = np.logical_and(planes[:, 3] < sensorrange[0], planes[:, 3] >= sensorrange[1])
            planes_in_range_indice  |= np.logical_and(planes[:, 4] < sensorrange[2], planes[:, 4] >= sensorrange[3])
            planes_in_range_indice  |= np.logical_and(planes[:, 5] < sensorrange[4], planes[:, 5] >= sensorrange[5])
            planes_in_range = planes[~planes_in_range_indice]

            nbnormals_in_range = np.size(planes_in_range, 0)
            planes_in_range_consist=np.zeros((nbnormals_in_range,6))
            planes_in_range_consist[:, 3:6] = planes_in_range[:, 3:6]

            for i in range(nbnormals_in_range):
            
                p1 = (sensorcentre - planes_in_range[i,3:6]) / np.linalg.norm(sensorcentre - planes_in_range[i,3:6]) # Vector from the centroid to the centre of the ellipsoid (here the sensor is placed)
                p2 = planes_in_range[i,0:3]
                
                angle = math.atan2(np.linalg.norm(np.cross(p1,p2)), np.dot(p1,p2) ) # Angle between the centroid-sensor and plane normal
            
                
                if (angle >= -pi/2 and angle <= pi/2): # (angle >= -pi/2 and angle <= pi/2):
                    
                    planes_in_range_consist[i,0] = -planes_in_range[i,0]
                    planes_in_range_consist[i,1] = -planes_in_range[i,1]
                    planes_in_range_consist[i,2] = -planes_in_range[i,2]  
                    
                else:
                    
                    planes_in_range_consist[i,0] = planes_in_range[i,0]
                    planes_in_range_consist[i,1] = planes_in_range[i,1]
                    planes_in_range_consist[i,2] = planes_in_range[i,2]
                
            planes_consist.append(planes_in_range_consist)
            
            return np.array(planes_consist)