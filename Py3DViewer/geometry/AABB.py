import numpy as np

class AABB:
    
    def __init__(self, vertices):
        
        self.vertices = None
        if(np.array(vertices).shape == (2,3)):
            self.vertices = np.array(vertices)
            print(self.vertices)
        else:
            print('Wrong number of vertices')
    
    
    def centre_point(self):
        return (self.vertices[0] + self.vertices[1])*0.5
    
    
    def contains(self, point):
        if(point[0] > self.vertices[0][0] and point[0] < self.vertices[1][0] and
          point[1] > self.vertices[0][1] and point[1] < self.vertices[1][1] and
          point[2] > self.vertices[0][2] and point[2] < self.vertices[1][2]):
            return True
        else:
            return False
        
        
    def intersect_box(self, aabb):
        if(self.vertices[1][0] <= aabb.vertices[0][0] or self.vertices[0][0] >= aabb.vertices[1][0]):
            return False;
        if(self.vertices[1][1] <= aabb.vertices[0][1] or self.vertices[0][1] >= aabb.vertices[1][1]):
            return False;
        if(self.vertices[1][2] <= aabb.vertices[0][2] or self.vertices[0][2] >= aabb.vertices[1][2]):
            return False;
        return True
    
            
