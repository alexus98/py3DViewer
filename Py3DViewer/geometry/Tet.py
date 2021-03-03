import numpy as np
from .SpaceObject import SpaceObject

class Tet(SpaceObject):
    
    def __init__(self, vertices):
        if(np.array(vertices).shape == (4,3)):
            super(Tet, self).__init__(vertices)
        else:
            print('Wrong number of vertices')