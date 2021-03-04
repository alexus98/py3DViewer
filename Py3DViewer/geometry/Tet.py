import numpy as np
from .SpaceObject import SpaceObject

class Tet(SpaceObject):
    
    def __init__(self, vertices):
        v = np.array(vertices)

        if(v.shape == (4,3)):
            super(Tet, self).__init__(v)
        else:
            print('Wrong number of vertices')