import numpy as np
from numba import njit
from ..geometry import AABB
from .NOctree import NOctree
from .NOctreeNode import NOctreeNode
from numba.typed import List
from ..structures import Trimesh, Quadmesh, Tetmesh, Hexmesh

@njit
def split(nOctree, nodes_to_split, nOctreeNodeIndex, leaves):
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
        #print('MIN CHILDREN:',aabb.min,'MAX CHILDREN:',aabb.max)

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
        #print('ITEM',len(i))
        nOctree.nodes[nOctreeNodeIndex].children[children_index]=children_index_queue
        
        #print('Items: ',nOctree.nodes[children_index_queue].items,'Depth: ',nOctree.nodes[children_index_queue].depth)
        if(len(nOctree.nodes[children_index_queue].items) > nOctree.items_per_leaf
           and
           nOctree.nodes[children_index_queue].depth < nOctree.max_depth):
            #print('SPLITTING ',children_index_queue)
            #print('Depth',nOctree.nodes[children_index_queue].depth,'Items',len(nOctree.nodes[children_index_queue].items))
            nodes_to_split.append(children_index_queue)
        else:
            leaves=leaves+1
        
        children_index_queue += 1
        children_index += 1
        

    nOctree.nodes[nOctreeNodeIndex].items = np.empty(0,dtype='int64')
    #print(nOctree.nodes[nOctreeNodeIndex+1].children)
    return leaves
        
@njit
def build_octree(nOctree):
    leaves=0
    
    #i=np.empty(0,dtype='int64')
    
    i = np.array(list(range(0,len(nOctree.shapes))))
    nOctree.aabbs.append(AABB(nOctree.vertices))
    
    #nOctree.aabbs.append(AABB(nOctree.shapes[0].vertices))
    #i = np.append(i,0)
    
    #for item_idx,aabb in enumerate(nOctree.aabb_shapes[1:]):
    #    nOctree.aabbs[0].push_aabb(aabb)
    #    i = np.append(i,item_idx+1)
    
    nOctree.nodes.append(NOctreeNode(0,0,i))
    
    nodes_to_split = List()
    nodes_to_split.append(0)
    
    while len(nodes_to_split)>0:
        node_idx = nodes_to_split.pop(0)
        if(len(nOctree.nodes[node_idx].items) > nOctree.items_per_leaf and nOctree.nodes[node_idx].depth < nOctree.max_depth):
            leaves=split(nOctree, nodes_to_split, node_idx,leaves)
        else:
            leaves=leaves+1
    print(leaves)

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
                print('Prova')
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
                        for i in child.items:
                            if(shapes[i].ray_interesects_triangle(r_origin, r_dir)):
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