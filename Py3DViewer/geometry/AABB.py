import numpy as np
from numba import float64
from numba.experimental import jitclass

spec = [('min', float64[:]),
       ('max', float64[:]),
       ('delta_x', float64),
       ('delta_y', float64),
       ('delta_z', float64)]

@jitclass(spec)
class AABB(object):
    def __init__(self, vertices):
        xmin = vertices[:,0].min()
        ymin = vertices[:,1].min()
        zmin = vertices[:,2].min()
        
        xmax = vertices[:,0].max()
        ymax = vertices[:,1].max()
        zmax = vertices[:,2].max()
        
        self.min = np.array([xmin,ymin,zmin],dtype='float64')
        self.max = np.array([xmax,ymax,zmax],dtype='float64')
        self.delta_x = self.max[0]-self.min[0]
        self.delta_y = self.max[1]-self.min[1]
        self.delta_z = self.max[2]-self.min[2]
        
    @property
    def center(self):
        return (self.min + self.max)*0.5
    
    
    def contains(self, points, strict=False):
        points=np.array(points).reshape(-1,3)
        
        if(strict):
            x_check =  np.logical_and(points[:,0] > self.min[0], points[:,0] < self.max[0])
            y_check =  np.logical_and(points[:,1] > self.min[1], points[:,1] < self.max[1])
            z_check =  np.logical_and(points[:,2] > self.min[2], points[:,2] < self.max[2])
            return np.logical_and(x_check, y_check, z_check)
            
        else:
            x_check =  np.logical_and(points[:,0] >= self.min[0], points[:,0] <= self.max[0])
            y_check =  np.logical_and(points[:,1] >= self.min[1], points[:,1] <= self.max[1])
            z_check =  np.logical_and(points[:,2] >= self.min[2], points[:,2] <= self.max[2])
            return np.logical_and(x_check, y_check, z_check)
        
        
    def intersects_box(self, aabb):
        if(self.max[0] <= aabb.min[0] or self.min[0] >= aabb.max[0]):
            return False;
        if(self.max[1] <= aabb.min[1] or self.min[1] >= aabb.max[1]):
            return False;
        if(self.max[2] <= aabb.min[2] or self.min[2] >= aabb.max[2]):
            return False;
        return True
    
    
    def push_aabb(self,aabb):
        if (self.min is not None or self.max is not None):
            self.min = np.minimum(self.min,aabb.min)
            self.max = np.maximum(self.max,aabb.max)
        else:
            self.min = aabb.min
            self.max = aabb.max
            
        self.delta_x = self.max[0]-self.min[0]
        self.delta_y = self.max[1]-self.min[1]
        self.delta_z = self.max[2]-self.min[2]