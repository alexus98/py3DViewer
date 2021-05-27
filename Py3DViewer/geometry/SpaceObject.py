import numpy as np
from .AABB import AABB
from numba import float64,njit,jit
from numba.experimental import jitclass
import math

@njit
def distance_between_points(p1,p2):
        d = math.sqrt(pow((p2[0] - p1[0]),2) + pow((p2[1] - p1[1]),2) + pow((p2[2] - p1[2]),2))
        return d

spec = [('vertices', float64[:,:])]

@jitclass(spec)
class SpaceObject:
    def __init__(self, vertices):
        self.vertices = vertices

    @staticmethod
    def closest_point(vert,p):
        d = np.zeros(len(vert))
        m = p
        i=0
        for v in vert:
            d[i] = distance_between_points(vert[p],v)
            if m != p and i!=p:        

                if d[i] < d[m]:
                    m = i
            elif m == p and i!=p:
                m = i
            i=i+1
        return m
    
    
    @staticmethod
    def magnitude_triangle(a,b,c):
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


    def quad_contains_point(self):# point):
        if len(self.vertices) == 4:
            a = self.vertices[0]
            b = self.vertices[1]
            c = self.vertices[2]
            d = self.vertices[3]

            if (not np.array_equal(a,b)
                and not np.array_equal(a,c)
                and not np.array_equal(a,d)
                and not np.array_equal(b,c)
                and not np.array_equal(b,d)
                and not np.array_equal(c,d)):
                
                v=[a,b,c,d]
                closest=self.closest_point(self.vertices,0)
                v=v[1:]
                v=v[:closest-1] + v[closest:]

                """
                x1=np.array([a,self.vertices[closest],v[0]])
                x2=np.array([a,self.vertices[closest],v[1]])
                tri1 = SpaceObject(np.array([a,self.vertices[closest],v[0]]))
                tri2 = SpaceObject(np.array([a,self.vertices[closest],v[1]]))
                return np.logical_or(tri1.triangle_contains_point(point),tri2.triangle_contains_point(point))
                """
            else:
                print('Due o piu vertici sono nella stessa posizione')