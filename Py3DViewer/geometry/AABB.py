import numpy as np

class AABB:
    
    def __init__(self, vertices=None):
        if(vertices is not None):
            self.min = vertices.min(axis=0)
            self.max = vertices.max(axis=0)
            self.delta_x = self.max[0]-self.min[0]
            self.delta_y = self.max[1]-self.min[1]
            self.delta_z = self.max[2]-self.min[2]
        else:
            self.min = None
            self.max = None
            self.delta_x = None
            self.delta_y = None
            self.delta_z = None
    
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