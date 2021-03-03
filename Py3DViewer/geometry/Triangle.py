import numpy as np
from .SpaceObject import SpaceObject

class Triangle(SpaceObject):
    
    def __init__(self, vertices,aabb):
        if(np.array(vertices).shape == (3,3) and np.array(aabb).shape == (2,3)):
            super(Triangle, self).__init__(vertices, aabb)
        else:
            print('Wrong number of vertices')