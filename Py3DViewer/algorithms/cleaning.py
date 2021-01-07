import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple
from ..utils.utilities import mesh_is_surface

@njit(Tuple((float64[:,::1], int64[:,::1]))(float64[:,::1],int64[:,::1]), cache=True)
def __remove_duplicated_vertices(vertices, polys):
    
    vtx_dictionary = dict()
    support_set = set()
    vtx_dictionary[(-1.,-1.,-1.)] = -1.
    support_set.add((-1.,-1.,-1.))
    new_vertices = np.zeros(vertices.shape, dtype=np.float64)
    
    j=0
    for i in range(vertices.shape[0]):
        
        v = (vertices[i][0], vertices[i][1], vertices[i][2])
        
        if v not in support_set:
            
            vtx_dictionary[v] = i
            support_set.add(v)
            new_vertices[j] = vertices[i]
            j+=1
        
        else:
            idx = vtx_dictionary[v]
            r = np.where(polys==i)
            for k in zip(r[0], r[1]):
                polys[k[0]][k[1]] = idx
    
    
    return new_vertices[:j], polys


def remove_isolated_vertices(mesh):
    used_vertices = set(mesh.faces.flatten())
    all_vertices = set(range(mesh.num_vertices))
    isolated_vertices = np.array(list(all_vertices.difference(used_vertices)))
    mesh.remove_vertices(isolated_vertices)


def remove_duplicated_vertices(mesh):
    vertices = mesh.vertices
    polys = None
    if mesh_is_surface(mesh):
        nv, ns = __remove_duplicated_vertices(vertices, mesh.faces)
        mesh.vertices = nv
        mesh.faces = ns
        if mesh.faces.shape[1] == 3:
            mesh._Trimesh__load_operations()
        else:
            mesh._Quadmesh__load_operations() 

    elif hasattr(mesh, 'hexes'):
        nv, ns = __remove_duplicated_vertices(vertices, mesh.hexes)
        mesh.vertices = nv
        mesh.hexes = ns
        mesh._Hexmesh__load_operations() 
    else:
        nv, ns = __remove_duplicated_vertices(vertices, mesh.tets)
        mesh.vertices = nv
        mesh.tets = ns
        mesh._Tetmesh__load_operations() 
    
    

    
    