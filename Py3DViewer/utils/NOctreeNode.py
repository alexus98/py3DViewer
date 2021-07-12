import numpy as np
from numba.experimental import jitclass
from numba.types import int64, ListType

spec = [('father', int64),
        ('depth', int64),
        ('items', ListType(int64)),
        ('children', int64[:])]

#jitclass that represent a Node of an Octree.
#father: index of the father node contained in the List 'nodes' in Noctree
#items: list of items' indexes contained in the List 'shapes' in NOctree
#children: list of children nodes' indexes contained in the List 'nodes' in Noctree
@jitclass(spec)
class NOctreeNode:
    def __init__(self, father, depth, items):
        self.father = father
        self.depth = depth
        self.items = items
        self.children = np.zeros(8, dtype = int64)