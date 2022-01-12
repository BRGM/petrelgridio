import numpy as np

from geometry.primitives import * 


class Connectivity:
    def __init__(self):
        self._cells_nodes = [] # List of Heaxahedra/Tetrahedra/Wede/Pyramid
        self._cells_faces = []
        self._faces_ids = {} # Dico <Face : UID>
        self._faces_nodes = []
        self._faces_cells = [] # For each face in the model, pair<cell1_id, cell2_id>
    
    @property
    def cells_nodes(self):
        return self._cells_nodes

    @property
    def nb_cells(self):
        return len(self.cells_nodes)
    
    @property
    def faces_nodes(self):
        return self._faces_nodes

    @property
    def nb_faces(self):
        return len(self._faces_cells)

    def set_cells_nodes(self, cells):
        self._cells_nodes = [Primitive3d.builder(c) for c in cells]
        self._update_from_cell_nodes()
        
    def _update_from_cell_nodes(self):
        self._update_faces_from_cell_nodes()
        self._collect_cells_faces()

    def _update_faces_from_cell_nodes(self):
        """
        A face is shared by 2 siblings cells (except faces on borders). Here, we
        build a list storing for each face in the model the IDs of the 2 cells it
        separates (faces on borders store twice the ID of their unique cell).

        WARNING:
         * a "facet" belongs to a cell, and the order of its vertices is depends
           on the cell which built it as it determines the polarity of the facet.
         * a "face" does not depend on a cell (its a 2D model element living on
           its own). A face matches 1 or 2 cells facets in the model (depending
           on whether it is on a border or inside the model). The order of its
           vertices is chosen such that 1) the 1st one has the minimum UID,
           2) the 2nd one is its neighbor with the lowest UID and 3) continue
           rotating in this direction
        """
        self._faces_ids = {} # Dico : [Face : Face_UID]
        self._faces_cells = [] # List : Pair(Cell1_UID, Cell2_UID) for each Face
        for cur_cell_id, cell in enumerate(self.cells_nodes):
            for facet in cell.facets():
                assert len(self._faces_ids) == len(self._faces_cells)
                # Creates a new face from the facet (assumes the face does not exist yet)
                face = Primitive2d.get_face_from_facet(facet)
                face_id = len(self._faces_ids)
                # Checks if the face already exists (if so, retrieves its true UID)
                original_face_id = self._faces_ids.get(face, None)
                # Case 1: 1st time we meet the face
                if original_face_id is None:
                    self._faces_ids[face] = face_id # Registers the face
                    self._faces_cells.append([cur_cell_id, cur_cell_id]) # Creates an "on border" face
                # Case 2: We already met the face once
                else: 
                    assert self._faces_cells[original_face_id][0] == self._faces_cells[original_face_id][1]
                    self._faces_cells[original_face_id][1] = cur_cell_id # Updates the ID of the 2nd cell of the face
                    assert self._faces_cells[original_face_id][0] != self._faces_cells[original_face_id][1]
                assert len(self._faces_ids) == len(self._faces_cells)
        # Checks redundant face_id values
        assert len(self._faces_ids) == len({f_id: f for (f, f_id) in self._faces_ids.items()})
        # List of faces, ordered by increasing face_id
        self._faces_nodes = [f for f, _ in sorted(self._faces_ids.items(), key=lambda t: t[1])]
    
    def _collect_cells_faces(self):
        self._cells_faces = [[self._faces_ids[Primitive2d.get_face_from_facet(f)] for f in c.facets()] 
                             for c in self.cells_nodes]


class HybridMesh:
    def __init__(self, vertices, cells):
        self.vertices = vertices
        self._connectivity = Connectivity()
        self._connectivity.set_cells_nodes(cells)

    @property
    def vertices(self):
        return self._vertices
    
    @property
    def nb_vertices(self):
        return len(self.vertices)

    @property
    def nb_cells(self):
        return self.connectivity.nb_cells

    @property
    def nb_faces(self):
        return self.connectivity.nb_faces

    @property
    def connectivity(self):
        return self._connectivity

    @vertices.setter
    def vertices(self, vertices):
        assert vertices.ndim == 2
        assert vertices.shape[1] == Primitive.DIMENSION
        self._vertices = np.copy(vertices) # FIXME Is copy necessary?

    def cells_nodes_as_COC(self):
        # Step 1: Compute cells "pointers" (offset of the 1st cell vertex)
        pointers = np.zeros(self.connectivity.nb_cells, dtype=np.int64)
        counter_vertices = 0
        for cell_idx, cell in enumerate(self.connectivity.cells_nodes):
            counter_vertices += cell.nb_vertices
            pointers[cell_idx] = counter_vertices
        #Step 2: Write cells vertices indices
        nodes = np.zeros(counter_vertices, dtype=np.int64)
        assert self.connectivity.nb_cells == pointers.size
        cur_offset = 0
        for cell, cell_offset in zip(self.connectivity.cells_nodes, pointers):
            nodes[cur_offset:cell_offset] = cell.vertices
            cur_offset = cell_offset

        return pointers, nodes

    def cells_vtk_ids(self):
        vtk_ids = np.zeros(self.connectivity.nb_cells, dtype=np.int64)
        for cell_idx, cell in enumerate(self.connectivity.cells_nodes):
            vtk_ids[cell_idx] = cell.VTK_ELEMENT_ID
        return vtk_ids
