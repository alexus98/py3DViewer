import numpy as np
from numba.experimental import jitclass
from numba.types import int64, ListType, deferred_type
from numba.typed import List

spec = [('father', int64),
        ('depth', int64),
        ('items', ListType(int64)),
        ('children', ListType(int64))
       ]
@jitclass(spec)
class NOctreeNode:
    def __init__(self, father, depth, items, children):
        self.father=father
        self.depth=depth
        self.items=items
        self.children=children
    