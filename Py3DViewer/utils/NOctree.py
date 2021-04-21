import numpy as np
import numba
from numba.experimental import jitclass
from numba.types import int64, ListType, deferred_type

from numba.typed import List
from ..geometry.AABB import AABB
from ..geometry.SpaceObject import SpaceObject
from .NOctreeNode import NOctreeNode


aabb_type = AABB.class_type.instance_type

space_object_type = SpaceObject.class_type.instance_type

space_object_type = SpaceObject.class_type.instance_type

noctreenode_type = NOctreeNode.class_type.instance_type

spec = [('items_per_leaf', int64),
        ('max_depth', int64),
        ('aabbs', ListType(aabb_type)),
        #('shapes', ListType(space_object_type)),
        ('nodes', ListType(noctreenode_type)),
        ('depth', int64)]

@jitclass(spec)
class NOctree:
    def __init__(self,items_per_leaf,max_depth,shapes):
        self.items_per_leaf = 1
        self.max_depth = 1
        self.aabbs = List.empty_list(aabb_type)
        #self.shapes = shapes
        self.nodes = List.empty_list(noctreenode_type)
        self.depth = 0
        