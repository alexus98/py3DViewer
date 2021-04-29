import numpy as np
from .AABB import AABB
from numba import float64,jit
from numba.experimental import jitclass

spec = [('vertices', float64[:,:])]

@jitclass(spec)
class SpaceObject:
    def __init__(self, vertices):
        self.vertices = vertices