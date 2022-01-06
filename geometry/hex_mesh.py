import numpy as np

from geometry.primitives import Hexahedron


class HexMesh:
    @classmethod
    def make(cls, vertices, hexahedra):
        # FIXME Add assert about type/shape/datatype/...
        mesh = HexMesh()
        mesh.vertices = vertices
        mesh.cells = hexahedra
        mesh.update_connectivity_from_cellnodes()
        return mesh

    def __init__(self):
        self._vertices = None # np.array((NbVertices, Hexahedron.DIMENSION), dtype = ?)
        self._cells = None  # np.array((NbHexs, Hexahedron.NB_VERTICES), dtype = int64)

    @property
    def vertices(self):
        return self._vertices

    @property
    def cells(self):
        return self._cells

    @vertices.setter
    def vertices(self, vertices):
        assert vertices.ndim == 2
        assert vertices.shape[1] == Hexahedron.DIMENSION
        self._vertices = np.copy(vertices) # FIXME Necessary? Keep for security?
    
    @cells.setter
    def cells(self, hexahedra):
        assert hexahedra.ndim == 2
        assert hexahedra.shape[1] == Hexahedron.NB_VERTICES
        self._cells = np.copy(hexahedra) # FIXME Necessary? Keep for security?

    def update_connectivity_from_cellnodes(self):
        print("[WARNING] In HexMesh: this function should be implemented once dealing with faults") # FIXME

    def cells_vtk_ids(self):
        return np.full(self._cells.shape[0], Hexahedron.VTK_ELEMENT_ID)

    def cells_nodes_as_COC(self):
        return Hexahedron.NB_VERTICES * np.arange(1, self._cells.shape[0] + 1), self._cells
    