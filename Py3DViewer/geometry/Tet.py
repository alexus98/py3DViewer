import numpy as np
from .SpaceObject import SpaceObject

class Tet(SpaceObject):
    
    def __init__(self, vertices,aabb):
        if(np.array(vertices).shape == (4,3) and np.array(aabb).shape == (2,3)):
            super(Tet, self).__init__(vertices, aabb)
        else:
            print('Wrong number of vertices')