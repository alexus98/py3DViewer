import numpy as np
from .AABB import AABB

class SpaceObject:
    def __init__(self, vertices):
        self.vertices = vertices
        self.aabb = AABB(self.vertices)
        
    def print(self):
        print(self.vertices,"\n")
        print(self.aabb.min)
        print(self.aabb.max,"\n\n")