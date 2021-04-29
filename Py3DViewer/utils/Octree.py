import numpy as np
from numba import njit
from ..geometry import AABB
from .NOctree import NOctree
from .NOctreeNode import NOctreeNode
from numba.typed import List
from numba import jit
import copy

@njit
def split(nOctree, nodes_to_split, nOctreeNodeIndex):
    aabbNode = nOctree.aabbs[nOctreeNodeIndex]

    min = aabbNode.min
    max = aabbNode.max
    center = aabbNode.center
    node = nOctree.nodes[nOctreeNodeIndex]
    depth = node.depth+1
    children_index_queue = len(nOctree.aabbs)
    
    #print('--- Split ',nOctreeNodeIndex,' ---')
    #print('FATHER MIN',min,'FATHER MAX',max)
    #print('Depth: ', depth)
    #print('ITEMS:',node.items)
    
    aabb1=AABB(np.array([[min[0], min[1], min[2]], [center[0], center[1], center[2]]]))
    nOctree.aabbs.append(aabb1)
    aabb2=AABB(np.array([[center[0], min[1], min[2]], [max[0], center[1], center[2]]]))    
    nOctree.aabbs.append(aabb2)
    aabb3=AABB(np.array([[center[0], center[1], min[2]], [max[0], max[1], center[2]]]))
    nOctree.aabbs.append(aabb3)
    aabb4=AABB(np.array([[min[0], center[1], min[2]], [center[0], max[1], center[2]]]))
    nOctree.aabbs.append(aabb4)
    aabb5=AABB(np.array([[min[0], min[1], center[2]], [center[0], center[1], max[2]]]))
    nOctree.aabbs.append(aabb5)
    aabb6=AABB(np.array([[center[0], min[1], center[2]], [max[0], center[1], max[2]]]))
    nOctree.aabbs.append(aabb6)
    aabb7=AABB(np.array([[center[0], center[1], center[2]], [max[0], max[1], max[2]]]))
    nOctree.aabbs.append(aabb7)
    aabb8=AABB(np.array([[min[0], center[1], center[2]], [center[0], max[1], max[2]]]))
    nOctree.aabbs.append(aabb8)
    
    
    children_index = 0
    for aabb in nOctree.aabbs[children_index_queue:len(nOctree.aabbs)]:
        
        #print('Index assigning: ',children_index_queue)
        
        i=np.empty(0,dtype='int64')
        for item in node.items:
            
            #print('MIN CHILDREN:',aabb.min,'MAX CHILDREN:',aabb.max)
            #print('ITEM', item ,'MIN: ',nOctree.aabb_shapes[item].min, 'ITEM MAX: ',nOctree.aabb_shapes[item].max)
            
            if(aabb.intersects_box(nOctree.aabb_shapes[item])):
                #print('APPEND ITEM', item ,'MIN: ',nOctree.aabb_shapes[item].min, 'ITEM MAX: ',nOctree.aabb_shapes[item].max)
                i = np.append(i,item)
            #else:
                #print('NO ITEM', item ,'MIN: ',nOctree.aabb_shapes[item].min, 'ITEM MAX: ',nOctree.aabb_shapes[item].max)

        nOctree.nodes.append(NOctreeNode(nOctreeNodeIndex,depth,i))
        
        #print('PROFONDITA',nOctree.nodes[children_index_queue].depth)
        #print('ITEM',i)
        nOctree.nodes[nOctreeNodeIndex].children[children_index]=children_index_queue
        
        #print('Items: ',nOctree.nodes[children_index_queue].items,'Depth: ',nOctree.nodes[children_index_queue].depth)
        if(len(nOctree.nodes[children_index_queue].items) > nOctree.items_per_leaf
           and
           nOctree.nodes[children_index_queue].depth < nOctree.max_depth):
            #print('SPLITTING ',children_index_queue)
            nodes_to_split.append(children_index_queue)
        
        children_index_queue += 1
        children_index += 1
        

    nOctree.nodes[nOctreeNodeIndex].items = np.empty(0,dtype='int64')
    #print(nOctree.nodes[nOctreeNodeIndex+1].children)
        
@njit
def build_octree(nOctree):
    i=np.empty(0,dtype='int64')
    
    
    nOctree.aabbs.append(AABB(nOctree.shapes[0].vertices))
    i = np.append(i,0)

    for item_idx,aabb in enumerate(nOctree.aabb_shapes[1:]):
        nOctree.aabbs[0].push_aabb(aabb)
        i = np.append(i,item_idx+1)
    
    nOctree.nodes.append(NOctreeNode(0,0,i))
    
    nodes_to_split = List()
    nodes_to_split.append(0)
    
    while len(nodes_to_split)>0:
        node_idx = nodes_to_split.pop(0)
        if(len(nOctree.nodes[node_idx].items) > nOctree.items_per_leaf and nOctree.nodes[node_idx].depth < nOctree.max_depth):
            split(nOctree, nodes_to_split, node_idx)

class Octree:
    def __init__(self, items_per_leaf, max_depth, shapes):
        self.n = NOctree(items_per_leaf, max_depth, shapes)
        build_octree(self.n)