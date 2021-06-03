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
        
        
    def all_vertices_are_different(self):
        for i,v in enumerate(self.vertices):
            for vert in self.vertices[i+1:]:
                if np.array_equal(vert,v):
                    return False
        return True
    
    
    def triangle_contains_point(self, point):
        if len(self.vertices) == 3:
            #if (not np.array_equal(a,b) and not np.array_equal(a,c) and not np.array_equal(b,c)):
            if self.all_vertices_are_different():
                a = self.vertices[0]
                b = self.vertices[1]
                c = self.vertices[2]

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
                
                
    def quad_contains_point(self, point):
        if len(self.vertices) == 4:
            #if (not np.array_equal(a,b)
            #    and not np.array_equal(a,c)
            #    and not np.array_equal(a,d)
            #    and not np.array_equal(b,c)
            #    and not np.array_equal(b,d)
            #    and not np.array_equal(c,d)):
            if self.all_vertices_are_different():
                
                a = self.vertices[0]
                b = self.vertices[1]
                c = self.vertices[2]
                d = self.vertices[3]
                
                v1=np.zeros((3,3),dtype='float64')
                v1[0]=a
                v1[1]=b
                v1[2]=d
                
                v2=np.zeros((3,3),dtype='float64')
                v2[0]=b
                v2[1]=c
                v2[2]=d
                
                tmp = self.vertices
                self.vertices=v1
                tri1 = self.triangle_contains_point(point)
                self.vertices=v2
                tri2 = self.triangle_contains_point(point)
                self.vertices = tmp
                
                return (tri1 or tri2)
            else:
                print('Due o piu vertici sono nella stessa posizione')
                
                
    def tet_contains_point(self, point):
        if len(self.vertices) == 4:
            a = self.vertices[0]
            b = self.vertices[1]
            c = self.vertices[2]
            d = self.vertices[3]

            if self.all_vertices_are_different():
                
            #if (not np.array_equal(a,b)
            #    and not np.array_equal(a,c)
            #    and not np.array_equal(a,d)
            #    and not np.array_equal(b,c)
            #    and not np.array_equal(b,d)
            #    and not np.array_equal(c,d)):
            
                v1=np.zeros((3,3),dtype='float64')
                v1[0]=a
                v1[1]=b
                v1[2]=c
                
                v2=np.zeros((3,3),dtype='float64')
                v2[0]=a
                v2[1]=b
                v2[2]=d
                
                v3=np.zeros((3,3),dtype='float64')
                v3[0]=a
                v3[1]=c
                v3[2]=d
                
                v4=np.zeros((3,3),dtype='float64')
                v4[0]=c
                v4[1]=b
                v4[2]=d
                
                tmp = self.vertices
                self.vertices=v1
                tri1 = self.triangle_contains_point(point)
                self.vertices=v2
                tri2 = self.triangle_contains_point(point)
                self.vertices=v3
                tri3 = self.triangle_contains_point(point)
                self.vertices=v4
                tri4 = self.triangle_contains_point(point)
                self.vertices = tmp
                
                return (tri1 or tri2 or tri3 or tri4)
            else:
                print('Due o piu vertici sono nella stessa posizione')
            
    def hex_contains_point(self, point):
        if len(self.vertices) == 8:
            if self.all_vertices_are_different():
                a = self.vertices[0]
                b = self.vertices[1]
                c = self.vertices[2]
                d = self.vertices[3]
                e = self.vertices[4]
                f = self.vertices[5]
                g = self.vertices[6]
                h = self.vertices[7]
                
                v1=np.zeros((4,3),dtype='float64')
                v1[0]=a
                v1[1]=b
                v1[2]=c
                v1[3]=d
                
                v2=np.zeros((4,3),dtype='float64')
                v2[0]=e
                v2[1]=f
                v2[2]=g
                v2[3]=h
                
                v3=np.zeros((4,3),dtype='float64')
                v3[0]=a
                v3[1]=b
                v3[2]=e
                v3[3]=f
                
                v4=np.zeros((4,3),dtype='float64')
                v4[0]=c
                v4[1]=d
                v4[2]=g
                v4[3]=h
                
                v5=np.zeros((4,3),dtype='float64')
                v5[0]=a
                v5[1]=e
                v5[2]=d
                v5[3]=h
                
                v6=np.zeros((4,3),dtype='float64')
                v6[0]=b
                v6[1]=f
                v6[2]=c
                v6[3]=g
                
                tmp = self.vertices
                self.vertices=v1
                quad1 = self.quad_contains_point(point)
                self.vertices=v2
                quad2 = self.quad_contains_point(point)
                self.vertices=v3
                quad3 = self.quad_contains_point(point)
                self.vertices=v4
                quad4 = self.quad_contains_point(point)
                self.vertices=v5
                quad5 = self.quad_contains_point(point)
                self.vertices=v6
                quad6 = self.quad_contains_point(point)
                
                return (quad1 or quad2 or quad3 or quad4 or quad5 or quad6)
            else:
                print('Due o piu vertici sono nella stessa posizione')