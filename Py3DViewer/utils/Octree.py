import numpy as np
from numba import njit
from ..geometry import AABB
from .NOctree import NOctree
from numba.types import int64
from .NOctreeNode import NOctreeNode
from numba.typed import List


#Method to split a node in 8 children.
#We split the AABB of the node in 8 octants and assign all the items of the father to the nodes that intersect them.
#Then we add the node to the list of octree nodes, assign the children index of the list to the list of children of the father.
#If the he numbers of items in the node is higher than the max number of items per leaf and the depth is lower than the max #depth we add it to the list of nodes to split.
@njit
def split(nOctree, nodes_to_split, nOctreeNodeIndex):
    aabbNode = nOctree.aabbs[nOctreeNodeIndex]

    min = aabbNode.min
    max = aabbNode.max
    center = aabbNode.center
    node = nOctree.nodes[nOctreeNodeIndex]
    depth = node.depth+1
    children_index_queue = len(nOctree.aabbs)
    
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

        i = List()
        for item in node.items:
            if(aabb.intersects_box(nOctree.aabb_shapes[item])):
                i.append(item)
        nOctree.nodes.append(NOctreeNode(nOctreeNodeIndex,depth,i))
        nOctree.nodes[nOctreeNodeIndex].children[children_index]=children_index_queue
        
        if(len(nOctree.nodes[children_index_queue].items) > nOctree.items_per_leaf
           and
           nOctree.nodes[children_index_queue].depth < nOctree.max_depth):
            nodes_to_split.append(children_index_queue)
        
        children_index_queue += 1
        children_index += 1
        
    nOctree.nodes[nOctreeNodeIndex].items = List.empty_list(int64)

    
#Method that build the octree given a list of shapes.
#We initalize the list of items indices and initialize the root and its AABB
#We add the root to the list of nodes to split and if the numbers of items in the node is higher than the max number of items 
#per leaf and the depth is lower than the max depth we split the node in 8 children.
#The same method is applied to all the nodes added to the list of nodes to split.
@njit
def build_octree(nOctree):
    i = List(range(0,len(nOctree.shapes)))
    nOctree.aabbs.append(AABB(nOctree.vertices))
    nOctree.nodes.append(NOctreeNode(0,0,i))
    
    if(len(nOctree.nodes[0].items) > nOctree.items_per_leaf and nOctree.nodes[0].depth < nOctree.max_depth):
    
        nodes_to_split = List()
        nodes_to_split.append(0)
    
        while len(nodes_to_split)>0:
            node_idx = nodes_to_split.pop(0)
            split(nOctree, nodes_to_split, node_idx)

            
#Method to search a point inside a face of the mesh represented as an Octree.
#We check if the point lies inside the father's AABB, then we find the first octant that contains the point and recursevly it #goes deep down to find a leave that may contain the point. If the leave contain the point we check every item inside
#and if an item contains the point the method returns a tuple with True and the item's index otherwise False and -1.
@njit            
def search_p(nodes,shapes,aabbs,point,type_mesh,index=0):
        
        aabb = aabbs[index]
        
        if (point[0] < aabb.min[0]
            or point[0] > aabb.max[0]
            or point[1] < aabb.min[1]
            or point[1] > aabb.max[1]
            or point[2] < aabb.min[2]
            or point[2] > aabb.max[2]):
                print('Fuori dall aabb')
                return False,-1
        
        if point[0] <= aabb.center[0]:
            if point[1] <= aabb.center[1]:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 0 #left_back_bottom
                else:
                    inner_child_index = 4 #left_front_bottom
            
            else:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 3 #left_back_top
                else:
                    inner_child_index = 7 #left_front_top
        else:
            if point[1] <= aabb.center[1]:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 1 #right_back_bottom
                else:
                    inner_child_index = 5 #right_front_bottom
            else:
                if point[2] <= aabb.center[2]:
                    inner_child_index = 2 #right_back_top
                else:
                    inner_child_index = 6 #right_front_top
                    
        nodes_child_index = nodes[index].children[inner_child_index]
        if len(nodes[nodes_child_index].items) == 0 and not np.array_equal(nodes[nodes_child_index].children, np.zeros(8)):
            return search_p(nodes, shapes, aabbs, point,type_mesh,nodes_child_index)
        else:
            if type_mesh == 'Trimesh':
                for i in nodes[nodes_child_index].items:
                    if shapes[i].triangle_contains_point(point):
                        return True,i
            
            elif type_mesh == 'Quadmesh':
                for i in nodes[nodes_child_index].items:
                    if shapes[i].quad_contains_point(point):
                        return True,i
                    
            elif type_mesh == 'Hexmesh':
                for i in nodes[nodes_child_index].items:
                    if shapes[i].hex_contains_point(point):
                        return True,i
                    
            elif type_mesh == 'Tetmesh':
                for i in nodes[nodes_child_index].items:
                    if shapes[i].tet_contains_point(point):
                        return True,i
                
            print('Dentro l aabb')
            return False,-1

#Method to get the items intersected by a ray.
#We check if the ray intersects the AABB and if true we add the node to the queue of nodes and check which octants it #intersects. If the node intersected is a leave we add it to the queue. If the node is not a leave we add every item #intersected by the ray to a set
@njit
def intersects_ray(nodes, shapes, aabbs, r_origin, r_dir, type_mesh):
    shapes_hit = set()
    
    if(not aabbs[0].intersects_ray(r_origin, r_dir)):
        return False, shapes_hit
    
    q = List()
    q.append(0)

    while(len(q)>0):
        index = q.pop(0)
        node = nodes[index]
        
        if not(node.children == np.zeros(8)).all():
            for c in node.children:
                child = nodes[c]
                if(aabbs[c].intersects_ray(r_origin, r_dir)):
                    if(len(child.items) == 0):
                        q.append(c)
                    else:
                        if type_mesh == 'Trimesh':
                            for i in child.items:
                                if(shapes[i].ray_interesects_triangle(r_origin, r_dir)):
                                    shapes_hit.add(i)
                        elif type_mesh == 'Quadmesh':
                            for i in child.items:
                                if(shapes[i].ray_interesects_quad(r_origin, r_dir)):
                                    shapes_hit.add(i)
                        elif type_mesh == 'Hexmesh':
                            for i in child.items:
                                if(shapes[i].ray_interesects_hex(r_origin, r_dir)):
                                    shapes_hit.add(i)
                        elif type_mesh == 'Tetmesh':
                            for i in child.items:
                                if(shapes[i].ray_interesects_tet(r_origin, r_dir)):
                                    shapes_hit.add(i)
    if(len(shapes_hit) == 0):
        return False,shapes_hit
    return True,shapes_hit


class Octree:
    def __init__(self, items_per_leaf, max_depth, shapes, vertices):
        self.n = NOctree(items_per_leaf, max_depth, shapes, vertices)
        build_octree(self.n)
        
    def search_point(self,type_mesh,point):
        return search_p(self.n.nodes,self.n.shapes,self.n.aabbs,point,type_mesh,index=0)
    
    def intersects_ray(self, type_mesh, r_origin, r_dir):
        return intersects_ray(self.n.nodes, self.n.shapes, self.n.aabbs, r_origin, r_dir, type_mesh)