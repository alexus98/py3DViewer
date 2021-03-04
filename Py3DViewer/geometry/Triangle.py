import numpy as np
from .SpaceObject import SpaceObject

class Triangle(SpaceObject):
    
    def __init__(self, vertices):
        v = np.array(vertices)
        if(v.shape == (3,3)):
            super(Triangle, self).__init__(v)
        else:
            print('Wrong number of vertices')