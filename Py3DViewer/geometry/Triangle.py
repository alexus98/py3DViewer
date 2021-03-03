import numpy as np
from .SpaceObject import SpaceObject

class Triangle(SpaceObject):
    
    def __init__(self, vertices):
        if(np.array(vertices).shape == (3,3)):
            super(Triangle, self).__init__(vertices)
        else:
            print('Wrong number of vertices')