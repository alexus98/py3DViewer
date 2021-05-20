import numpy as np
from .AABB import AABB
from numba import float64,jit
from numba.experimental import jitclass
import math

spec = [('vertices', float64[:,:])]

@jitclass(spec)
class SpaceObject:
    def __init__(self, vertices):
        self.vertices = vertices

    def magnitude_triangle(self,a,b,c):
        mag = 0
        ab = b-a
        ac = c-a
        abac = np.cross(ab,ac)
        for element in abac:
            mag=mag+pow(element, 2)
        mag=math.sqrt(mag)
        return mag    
        
        
    def triangle_contains_point(self, point):
        if len(self.vertices) == 3:
            a = self.vertices[0]
            b = self.vertices[1]
            c = self.vertices[2]
            if (not np.array_equal(a,b) and not np.array_equal(a,c) and not np.array_equal(b,c)):
                area_triangolo = self.magnitude_triangle(a,b,c)/2
                
                alfa = self.magnitude_triangle(point,b,c)/(2*area_triangolo)
                beta = self.magnitude_triangle(point,c,a)/(2*area_triangolo)
                gamma = 1 - alfa - beta
                
                if alfa <= 1 and alfa >=0 and beta <= 1 and beta >=0 and gamma <= 1 and gamma >=0:
                    return True
                else:
                    return False
            else:
                print('Due o piu vertici sono nella stessa posizione')
            
            
    