import numpy as np
from .AABB import AABB

class SpaceObject:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)
        aabb = np.amin(vertices, axis=0)
        aabb = np.append(aabb,np.amax(vertices, axis = 0)).reshape(2,3)
        self.aabb = AABB(aabb)
        
    def print(self):
        print(self.vertices,"\n")
        print(self.aabb.vertices,"\n\n")