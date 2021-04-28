import numpy as np
from numba.experimental import jitclass
from numba.types import int64

spec = [('father', int64),
        ('depth', int64),
        ('items', int64[:]),
        ('children', int64[:])
       ]
@jitclass(spec)
class NOctreeNode:
    def __init__(self, father, depth, items):
        self.father=father
        self.depth=depth
        self.items=items
        self.children=np.zeros(8,dtype=int64)
    