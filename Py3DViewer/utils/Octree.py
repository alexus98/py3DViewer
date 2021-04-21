import numpy as np
from numba import jit
from ..geometry import AABB
from .NOctree import NOctree
import copy

class Octree:
    def __init__(self, shapes, max_depth, items_per_leaf):
        n = NOctree(shapes, max_depth, items_per_leaf)
        #build(n)
        
    def split(nOctree, nOctreeNodeIndex):
        aabbNode = nOctree.aabb[nOctreeNodeIndex]
        
        min = aabbNode.min
        max = aabbNode.max
        center = aabbNode.center
        
        node = nOctree.nodes[nOctreeNodeIndex]
        depth = node.depth+1
        
        NOctreeNode(nOctreeNodeIndex,depth,None,None)
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
        
        c_index = nOctreeNodeIndex+1
        
        for aabb in nOctree.aabbs[nOctreeNodeIndex+1:len(nOctree.aabb)]:
            i = List()
            for item in node.items:
                if(aabb.intersects_box(nOctree.aabbs[item])):
                    i.append(item)
                nOctree.nodes.append(NOctreeNode(nOctreeNodeIndex,depth,i,List.empty_list(int64)))
                node.children.append(c_index)
                c_index+=1

    def build_octree(nOctree):
        i=List()
        for item_idx,it in enumerate(self.root.items):
            rootAABB = nOctree.aabbs.push(it.aabb)
            i.append(item_idx)
        nOctree.aabbs.append(rootAABB)
        nOctree.nodes.append(NOctreeNode(0,0,i,List.empty_list(int64)))
        
        node_idx = 0
        while node_idx<len(a):
            if(len(Noctree.nodes[node_idx].items) > nOctree.items_per_leaf and Noctree.nodes[node_idx].depth < nOctree.max_depth):
                split(nOctree,node_idx)
            i+=1
        

        
"""    
class OctreeNode:
    def __init__(self, aabb, items, father = None):
        self.aabb = aabb
        self.father = fathter
        self.is_inner = False
        self.children = np.full(8,None)
        if(items is None):
            self.items= None
        else:
            self.items = np.array(items)
        
    def subdivide(self):
        #print('Suddivisione')
        min = self.aabb.min
        max = self.aabb.max
        center = self.aabb.center
        print('min:',min)
        print('max:',max)
        print('centro:',center)
        
        
        self.children[0] = OctreeNode(AABB(np.array([[min[0], min[1], min[2]], [center[0], center[1], center[2]]])), None, self)
        self.children[1] = OctreeNode(AABB(np.array([[center[0], min[1], min[2]], [max[0], center[1], center[2]]])), None, self)
        self.children[2] = OctreeNode(AABB(np.array([[center[0], center[1], min[2]], [max[0], max[1], center[2]]])), None, self)
        self.children[3] = OctreeNode(AABB(np.array([[min[0], center[1], min[2]], [center[0], max[1], center[2]]])), None, self)
        self.children[4] = OctreeNode(AABB(np.array([[min[0], min[1], center[2]], [center[0], center[1], max[2]]])), None, self)
        self.children[5] = OctreeNode(AABB(np.array([[center[0], min[1], center[2]], [max[0], center[1], max[2]]])), None, self)
        self.children[6] = OctreeNode(AABB(np.array([[center[0], center[1], center[2]], [max[0], max[1], max[2]]])), None, self)
        self.children[7] = OctreeNode(AABB(np.array([[min[0], center[1], center[2]], [center[0], max[1], max[2]]])), None, self)
        
        
        for item in self.items:
            for child in self.children:
                if(child.aabb.intersects_box(item.aabb)):
                    if(child.items is None):
                        child.items=np.array(item)
                    else:
                        child.items=np.append(child.items,item)
                        
        self.is_inner = True

        
class Octree:
    def __init__(self, max_depth, items_per_leaf):
        self.max_depth = max_depth
        self.items_per_leaf = items_per_leaf
        self.root = None
        self.leaves = None
        self.tree_depth = 0
        
        
    def build_octree(self,items):
        if(items is None or len(np.array(items)) == 0):
            return;
        
        self.root = OctreeNode(AABB(),items,None)
        
        for it in self.root.items:
            self.root.aabb.push_aabb(it.aabb)
            
        if(len(self.root.items) < self.items_per_leaf or self.max_depth == 1):
            #print('Primo if')
            self.leaves = np.array(self.root)
            self.tree_depth = 1
        else:
            self.root.subdivide()
            self.root.items=None
            if(self.max_depth == 2):
                self.tree_depth = 2;
                self.leaves = np.array(self.root.children)
            else:
                #nodes_queue = np.full((8,2),(None,2))
                #nodes_queue = np.full((8,1,2),(None,2))
                
                queue1=np.empty((0,2))
                queue2=np.empty((0,2))
                queue3=np.empty((0,2))
                queue4=np.empty((0,2))
                queue5=np.empty((0,2))
                queue6=np.empty((0,2))
                queue7=np.empty((0,2))
                queue8=np.empty((0,2))
                
                queue_array = [queue1,queue2,queue3,queue4,queue5,queue6,queue7,queue8]
                
                for i,(child,queue) in enumerate(zip(self.root.children,queue_array)):
                    if child.items is not None:
                        if child.items.size > self.items_per_leaf :
                            #queue=np.append(queue,(child,2))
                            #queue = np.append(queue[1:],[[child,2]],axis=0)
                            queue_array[i] = np.append(queue, [[child,2]], axis=0)
                            #print('Aggiunto alla coda')
                            
                        elif self.leaves is not None:
                            self.leaves = np.append(self.leaves, child)
                            #print('Aggiunta nuova foglia ')
                        else:
                            self.leaves = np.array(child)
                            #print('Aggiunta prima foglia ')
                            
                            
                
                for i,queue in enumerate(queue_array):
                    while(queue_array[i].size!=0):
                        depth = queue_array[i][0][1]+1
                        node = queue_array[i][0][0]
                        queue_array[i] = queue[1:]
                        self.tree_depth = max(depth, self.tree_depth)
                        
                        node.subdivide()
                        node.items = None
                        for child in node.children:
                            #print(depth)
                            if(child.items is not None):
                                if depth < self.max_depth and child.items.size > self.items_per_leaf :
                                    queue_array[i] = np.append(queue_array[i], [[child,depth]], axis=0)
                                    #print('Aggiunto alla coda for')
                                else:
                                    self.leaves = np.append(self.leaves,child)
                                    #print('Aggiunto alle foglie for')
                            else:
                                self.leaves = np.append(self.leaves,child)
                                #print('Aggiunto alle foglie for')"""