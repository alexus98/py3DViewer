import numpy as np
from .AABB import AABB
from numba import float64,jit
from numba.experimental import jitclass

#aabb_type = AABB.class_type.instance_type

spec = [('vertices', float64[:,:])]

@jitclass(spec)
class SpaceObject:
    def __init__(self, vertices):
        self.vertices = vertices