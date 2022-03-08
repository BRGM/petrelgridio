import numpy as np

from .primitives import Hexahedron, Tetrahedron


class UniformMesh:
    BASE_ELEMENT = None  # Hexahedron or Tetrahedron

    def __init__(self, vertices, cells):
        self.vertices = vertices
        self.cells = cells
        self.update_connectivity_from_cells_nodes()

    @property
    def vertices(self):
        return self._vertices

    @property
    def nb_vertices(self):
        return self.vertices.shape[0]

    @property
    def cells(self):
        return self._cells

    @property
    def nb_cells(self):
        return self.cells.shape[0]

    @vertices.setter
    def vertices(self, vertices):
        assert vertices.ndim == 2
        assert vertices.shape[1] == self.BASE_ELEMENT.DIMENSION
        self._vertices = np.copy(vertices)  # FIXME Necessary? Keep for security?

    @cells.setter
    def cells(self, cells):
        assert cells.ndim == 2
        assert cells.shape[1] == self.BASE_ELEMENT.NB_VERTICES
        self._cells = np.copy(cells)  # FIXME Necessary? Keep for security?

    def update_connectivity_from_cells_nodes(self):
        assert False, "Derived classes must imlpement there own version"

    def cells_vtk_ids(self):
        return np.full(self.nb_cells, self.BASE_ELEMENT.VTK_ELEMENT_ID)

    def cells_nodes_as_COC(self):
        return (
            self.BASE_ELEMENT.NB_VERTICES * np.arange(1, self.nb_cells + 1),
            self._cells,
        )


class HexMesh(UniformMesh):
    BASE_ELEMENT = Hexahedron

    def __init__(self, vertices, hexahedra):
        super().__init__(vertices, hexahedra)

    def update_connectivity_from_cells_nodes(self):
        print(
            "[WARNING] In HexMesh: this function should be implemented once dealing with faults"
        )  # FIXME


class TetMesh(UniformMesh):
    BASE_ELEMENT = Tetrahedron

    def __init__(self, vertices, hexahedra):
        print(
            "[WARNING] Class TetMesh is ongoing work and has never been tested."
        )  # FIXME
        print("          It probably needs to be completed before being used")
        super().__init__(vertices, hexahedra)

    def update_connectivity_from_cells_nodes(self):
        print(
            "[WARNING] In TetMesh: this function should be implemented once dealing with faults"
        )  # FIXME
