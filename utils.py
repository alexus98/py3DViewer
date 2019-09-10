import numpy as np

def read_obj(filename):
    """
    Imports the data from the given OBJ file
    Parameters
    ----------
    filename : str
        name of the file
    Returns
    -------
    (Array,Array,Array,Array)
        the mesh vertices, normals and the topology arrays
    """
    
    with open(filename) as f:
        
        tmpVtx     = []
        tmpFaces   = []
        tmpNormals = []
        
        for line in f.readlines():
            if line[0:2] == 'v ':
                vtx = line.split()
                tmpVtx.append([float(vtx[1]), float(vtx[2]), float(vtx[3])])
            
            if line[0:2] == 'f ':
                face = line.split()
                
                tmpFaces.append([int(f) -1 for f in face[1:]])
            
            
            if line[0:2] == 'fn':
                    face_normal = line.split()
                    tmpNormals.append([float(face_normal[1]), float(face_normal[2]), float(face_normals[3])])
                    
        tmpVtx = np.array(tmpVtx)
        tmpFaces = np.array(tmpFaces)
        tmpNormals = np.array(tmpNormals)
            
        return tmpVtx, tmpFaces, tmpNormals