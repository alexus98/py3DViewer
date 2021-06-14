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
    
    
    @staticmethod
    def same_side(a, b, c, d, point):
        normal = np.cross(b - a, c - a)
        dot_d = np.dot(normal, d - a)
        dot_point = np.dot(normal, point - a)
        return np.sign(dot_d) == np.sign(dot_point) or dot_point == 0
                
    def tet_contains_point(self, point):
        v1 = self.vertices[0]
        v2 = self.vertices[1]
        v3 = self.vertices[2]
        v4 = self.vertices[3]
        return (self.same_side(v1, v2, v3, v4, point)
                and self.same_side(v2, v3, v4, v1, point)
                and self.same_side(v3, v4, v1, v2, point)
                and self.same_side(v4, v1, v2, v3, point))
    
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
                v1[3]=f
                
                v2=np.zeros((4,3),dtype='float64')
                v2[0]=a
                v2[1]=c
                v2[2]=h
                v2[3]=f
                
                v3=np.zeros((4,3),dtype='float64')
                v3[0]=a
                v3[1]=c
                v3[2]=d
                v3[3]=h
                
                v4=np.zeros((4,3),dtype='float64')
                v4[0]=a
                v4[1]=f
                v4[2]=h
                v4[3]=e
                
                v5=np.zeros((4,3),dtype='float64')
                v5[0]=c
                v5[1]=h
                v5[2]=f
                v5[3]=g
                
                tmp = self.vertices
                self.vertices=v1
                quad1 = self.tet_contains_point(point)
                self.vertices=v2
                quad2 = self.tet_contains_point(point)
                self.vertices=v3
                quad3 = self.tet_contains_point(point)
                self.vertices=v4
                quad4 = self.tet_contains_point(point)
                self.vertices=v5
                quad5 = self.tet_contains_point(point)
                
                return (quad1 or quad2 or quad3 or quad4 or quad5)
            else:
                print('Due o piu vertici sono nella stessa posizione')


    def ray_interesects_triangle(self, r_origin, r_dir):
        v0=self.vertices[0]
        v1=self.vertices[1]
        v2=self.vertices[2]
    
        EPSILON = 0.0000001;

        e0 = v1 - v0;
        e1 = v2 - v0;
        p = np.cross(r_dir, e1)
        det = np.dot(e0, p)
        
        if(abs(det) < EPSILON):
            return False;
    
        invDet = 1.0/det
        t = r_origin - v0
        u = np.dot(t, p) * invDet
        if(u < 0.0 or u > 1.0):
            return False

        q = np.cross(t, e0)
        v = np.dot(r_dir, q) * invDet
        
        if(v < 0.0 or u + v > 1.0):
            return False
    
        t = np.dot(e1, q) * invDet
        return True