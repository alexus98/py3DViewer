import numpy as np

class SpaceObject:
    def __init__(self, vertices, aabb):
        self.aabb = np.array(aabb)
        self.vertices = np.array(vertices)
        
    def print(self):
        print(self.vertices)
        print(self.aabb)