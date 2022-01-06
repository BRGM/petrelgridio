import numpy as np

from geometry.primitives import Tetrahedron


class TetMesh:
    @classmethod
    def make(cls, vertices, tetrahedra):
        print("[WARNING] This is ongoing work, and should probably be completed before being used") # FIXME
        mesh = TetMesh()
        mesh.vertices = vertices
        mesh.cells = tetrahedra
        mesh.update_connectivity_from_cellnodes()
        return mesh

    def __init__(self):
        self._vertices = None # np.array((NbVertices, Tetrahedron.DIMENSION), dtype = ?)
        self._cells = None # np.array((NbTets, Tetrahedron.NB_VERTICES), dtype = int64)

    @property
    def vertices(self):
        return self._vertices

    @property
    def cells(self):
        return self._cells

    @vertices.setter
    def vertices(self, vertices):
        assert vertices.ndim == 2 
        assert vertices.shape[1] == Tetrahedron.DIMENSION 
        self._vertices = np.copy(vertices) # FIXME Necessary? Keep for security?
    
    @cells.setter
    def cells(self, tetrahedra):
        assert tetrahedra.ndim == 2
        assert tetrahedra.shape[1] == Tetrahedron.NB_VERTICES
        self._cells = np.copy(tetrahedra) # FIXME Necessary? Keep for security?

    def update_connectivity_from_cellnodes(self):
        print("[WARNING] In TetMesh: this function should be implemented once dealing with faults") # FIXME
