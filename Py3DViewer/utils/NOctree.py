import numpy as np
from numba.experimental import jitclass
from numba.types import int64, ListType, float64
from numba.typed import List
from numba import njit
from ..geometry.AABB import AABB
from ..geometry.SpaceObject import SpaceObject
from .NOctreeNode import NOctreeNode


aabb_type = AABB.class_type.instance_type

space_object_type = SpaceObject.class_type.instance_type

noctreenode_type = NOctreeNode.class_type.instance_type

spec = [('items_per_leaf', int64),
        ('max_depth', int64),
        ('aabbs', ListType(aabb_type)),
        ('shapes', ListType(space_object_type)),
        ('aabb_shapes', ListType(aabb_type)),
        ('nodes', ListType(noctreenode_type)),
        ('depth', int64),
        ('vertices', float64[:,:])]

#Method to get all the AABBs of every shape in the list
@njit
def get_aabbs(shapes):
    aabbs = List()
    for s in shapes:
        aabbs.append(AABB(s.vertices))
    return aabbs


#Jitclass that contains:
#items_per_leaf: max number of items per node
#max_depth: max depth of the octree
#aabbs: list with AABBs of octree nodes
#shapes: list of every shape
#aabb_shapes: aabbs of every shape
#nodes: list of every octree node
#depth: current octree depth
#vertices: list of all shapes' vertices
@jitclass(spec)
class NOctree(object):
    def __init__(self, items_per_leaf, max_depth, shapes, vertices):
        all_aabb = get_aabbs(shapes)
        self.items_per_leaf = items_per_leaf
        self.max_depth = max_depth
        self.aabbs = List.empty_list(aabb_type)
        self.shapes = shapes
        self.aabb_shapes = all_aabb
        self.nodes = List.empty_list(noctreenode_type)
        self.depth = 0
        self.vertices = vertices