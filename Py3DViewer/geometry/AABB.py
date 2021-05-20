import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba.typed import List
from numba import njit





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
    
    def get_vertices_and_edges(self, starting_index=0):
        bottom_left_back_v = np.array([self.min[0],self.min[1],self.min[2]],dtype=np.float64)
        bottom_right_back_v = np.array([self.max[0],self.min[1],self.min[2]],dtype=np.float64)
        bottom_left_front_v = np.array([self.min[0],self.min[1],self.max[2]],dtype=np.float64)
        bottom_right_front_v = np.array([self.max[0],self.min[1],self.max[2]],dtype=np.float64)
        top_left_back_v = np.array([self.min[0],self.max[1],self.min[2]],dtype=np.float64)
        top_right_back_v = np.array([self.max[0],self.max[1],self.min[2]],dtype=np.float64)
        top_left_front_v = np.array([self.min[0],self.max[1],self.max[2]],dtype=np.float64)
        top_right_front_v = np.array([self.max[0],self.max[1],self.max[2]],dtype=np.float64)
        
        
        v = [bottom_left_back_v,
            bottom_right_back_v,
            bottom_left_front_v,
            bottom_right_front_v,
            top_left_back_v,
            top_right_back_v,
            top_left_front_v,
            top_right_front_v]
        
        e1 = (starting_index+0,starting_index+1)
        e2 = (starting_index+0,starting_index+2)
        e3 = (starting_index+0,starting_index+4)
        e4 = (starting_index+2,starting_index+3)
        e5 = (starting_index+2,starting_index+6)
        e6 = (starting_index+1,starting_index+3)
        e7 = (starting_index+1,starting_index+5)
        e8 = (starting_index+3,starting_index+7)
        e9 = (starting_index+4,starting_index+6)
        e10 = (starting_index+4,starting_index+5)
        e11 = (starting_index+5,starting_index+7)
        e12 = (starting_index+6,starting_index+7)
        edges=[e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12]
        
        return v,edges
    
    @staticmethod
    def get_all_vertices_and_edges(aabbs):
        vertices=List()
        edges=List()
        i=0
        for a in aabbs:
            v,e = a.get_vertices_and_edges(i)
            vertices.append(v)
            edges.append(e)
            i=i+8

            #vertices=np.vstack(vertices)
            #e=np.array(edges)
        return vertices,edges