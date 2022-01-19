from itertools import chain
import numpy as np

from geometry.hybrid_mesh import HybridMesh
from geometry.uniform_mesh import TetMesh


def _align_face_and_edge(F, e):
    #    print(F, '-<', e, '>-', end=' ')
    assert all([n in F for n in e])
    k = F.index(e[0])
    newF = F[k:] + F[:k]
    #    print('@', newF, '@', end=' ')
    if newF[1] != e[1]:
        assert newF[-1] == e[1]
        newF = [e[0],] + newF[:0:-1]
    assert newF[1] == e[1]
    return newF


def _reprocess_pyramid_nodes(faces):
    assert len(faces) == 5
    fsize = [len(face) for face in faces]
    assert fsize.count(4) == 1 and fsize.count(3) == 4
    base = faces[fsize.index(4)]
    top = set(faces[fsize.index(3)]) - set(base)
    assert len(top) == 1
    return base + list(top)


def _reprocess_wedge_nodes(faces):
    assert len(faces) == 5
    fsize = [len(face) for face in faces]
    assert fsize.count(4) == 3 and fsize.count(3) == 2
    i = fsize.index(3)
    base = faces[i]
    top = faces[i + 1 + fsize[i + 1 :].index(3)]
    assert len(set(top) & set(base)) == 0
    e0 = base[:2]
    F = faces[fsize.index(4)]
    F = _align_face_and_edge(F, e0)
    e1 = F[2:][::-1]
    top = _align_face_and_edge(top, e1)
    return base + list(top)


def _reprocess_hexahedron_nodes(faces):
    # FIXME: do we need to work with numpy arrays???
    faces = np.asarray(faces)
    assert faces.shape == (6, 4)
    nodes = np.unique(np.hstack(faces))
    assert nodes.shape == (8,)
    # find the face opposite to the first one, i.e. faces[0]
    F0 = faces[0]
    for fi in range(1, 7):
        if not np.any(np.isin(F0, faces[fi])):
            break
    assert fi != 6
    Fi = faces[fi]
    # find the face that share the first edge of faces[0]
    e0 = faces[0][:2]
    for fj in range(1, 7):
        if np.all(np.isin(e0, faces[fj])):
            break
    assert fj != 6
    assert fj != fi
    Fj = faces[fj]
    Fj = _align_face_and_edge(list(Fj), e0)
    #    print(Fj)
    assert np.all(Fj[:2] == e0)
    # cycle around Fi such that orientation is ok
    e1 = Fj[2:][::-1]
    Fi = _align_face_and_edge(list(Fi), e1)
    #    print(Fi)
    assert np.all(Fi[:2] == e1)
    return np.hstack([F0, Fi])


def _reprocess_cell_nodes(faces):
    nf = len(faces)
    if nf == 6:
        assert all([len(nodes) == 4 for nodes in faces])
        return _reprocess_hexahedron_nodes(faces)
    elif nf == 5:
        fsizes = [len(nodes) for nodes in faces]
        if fsizes.count(4) == 1:
            assert fsizes.count(3) == 4
            return _reprocess_pyramid_nodes(faces)
        else:
            assert fsizes.count(4) == 3 and fsizes.count(3) == 2
            return _reprocess_wedge_nodes(faces)
    else:
        assert nf == 4
        assert all([len(nodes) == 3 for nodes in faces])
        return np.hstack(faces)


class RawMesh:
    def __init__(self, **kwargs):
        assert "vertices" in kwargs
        assert "cell_faces" in kwargs
        assert "face_nodes" in kwargs
        assert all([len(faces) > 3 for faces in kwargs["cell_faces"]]), "degenerated cell"
        assert all([len(nodes) > 2 for nodes in kwargs["face_nodes"]]), "degenerated face"
        self._cell_nodes = None
        for name, value in kwargs.items():
            if name == "cell_nodes":
                name = "_cell_nodes"
            elif name == "vertices":
                value = np.array(value, copy=False)
                assert value.ndim == 2 and value.shape[1] == 3
            setattr(self, name, value)
        self.reprocessed_cellnodes = False

    @classmethod
    def convert(cls, other):
        info = {
            name: getattr(other, name)
            for name in ["vertices", "cell_faces", "face_nodes"]
        }
        if hasattr(other, "cell_nodes"):
            info["cell_nodes"] = getattr(other, "cell_nodes")
        return cls(**info)

    @property
    def nb_vertices(self):
        return self.vertices.shape[0]

    @property
    def nb_faces(self):
        return len(self.face_nodes)

    @property
    def nb_cells(self):
        return len(self.cell_faces)

    def collect_cell_nodes(self, reprocess=False):
        assert self._cell_nodes is None
        # WARNING this is ok for convex cells
        face_nodes, cell_faces = self.face_nodes, self.cell_faces
        if not reprocess:
            return [
                np.unique(np.hstack([face_nodes[fi] for fi in faces]))
                for faces in cell_faces
            ]
        cell_nodes = []
        for faces in cell_faces:
            nf = len(faces)
            fnodes = [face_nodes[fi] for fi in faces]
            if nf == 6:
                assert all([len(nodes) == 4 for nodes in fnodes])
                cell_nodes.append(_reprocess_hexahedron_nodes(fnodes))
            elif nf == 5:
                fsizes = [len(nodes) for nodes in fnodes]
                if fsizes.count(4) == 1:
                    assert fsizes.count(3) == 4
                    cell_nodes.append(_reprocess_pyramid_nodes(fnodes))
                else:
                    assert fsizes.count(4) == 3 and fsizes.count(3) == 2
                    cell_nodes.append(_reprocess_wedge_nodes(fnodes))
            else:
                assert nf == 4
                assert all([len(nodes) == 3 for nodes in fnodes])
                cell_nodes.append(np.hstack(fnodes))
        return cell_nodes

    def cell_nodes(self, reprocess=False):
        if self._cell_nodes is not None and reprocess == self.reprocessed_cellnodes:
            return self._cell_nodes
        self._cell_nodes = self.collect_cell_nodes(reprocess)
        self.reprocessed_cellnodes = reprocess
        return self._cell_nodes # FIXME Renvoyer directement l'attribut ne change rien ? (pas de copie de toute façon ?)

    def _specific_faces(self, nbnodes):
        face_nodes = self.face_nodes
        return np.array([len(nodes) == nbnodes for nodes in face_nodes], dtype=np.bool)

    def triangle_faces(self):
        return self._specific_faces(3)

    def quadrangle_faces(self):
        return self._specific_faces(4)

    def tetrahedron_cells(self):
        cell_faces = self.cell_faces
        triangles = self.triangle_faces()
        return np.array(
            [
                # WARNING: no check on geometry consistency is performed
                len(faces) == 4 and all(triangles[face] for face in faces)
                for faces in cell_faces
            ],
            dtype=np.bool,
        )

    def pyramid_cells(self):
        cell_faces = self.cell_faces
        return np.array(
            [
                # WARNING: no check on geometry consistency is performed
                (
                    len(faces) == 4
                    and np.sum([len(face) == 3 for face in faces]) == 4
                    and np.sum([len(face) == 4 for face in faces]) == 1
                )
                for faces in cell_faces
            ],
            dtype=np.bool,
        )

    def wedge_cells(self):
        cell_faces = self.cell_faces
        return np.array(
            [
                # WARNING: no check on geometry consistency is performed
                (
                    len(faces) == 5
                    and np.sum([len(face) == 3 for face in faces]) == 2
                    and np.sum([len(face) == 4 for face in faces]) == 3
                )
                for faces in cell_faces
            ],
            dtype=np.bool,
        )

    def hexahedron_cells(self):
        cell_faces = self.cell_faces
        quadrangles = self.quadrangle_faces()
        return np.array(
            [
                # WARNING: no check on geometry consistency is performed
                len(faces) == 6 and all(quadrangles[face] for face in faces)
                for faces in cell_faces
            ],
            dtype=np.bool,
        )

    def _centers(self, element_nodes):
        # CHECKME: returns gravity center, is this the most useful?
        # WARNING: ok for convex elements, no geometry check is performed
        vertices = self.vertices
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        return np.array(
            [np.mean([vertices[n] for n in nodes], axis=0) for nodes in element_nodes]
        )

    def cell_centers(self, cell_nodes=None):
        if cell_nodes is None:
            cell_nodes = self.cell_nodes #FIXME It's not self._cell_nodes?
        return self._centers(cell_nodes)

    def face_centers(self, face_nodes=None):
        if face_nodes is None:
            face_nodes = self.face_nodes
        return self._centers(face_nodes)

    def _new_vertices(
        self, cell_centers, kept_cells, face_nodes, kept_faces, face_centers=None,
    ):
        splitted_cells = np.logical_not(kept_cells)
        splitted_faces = np.logical_not(kept_faces)
        if face_centers is None:
            face_centers = self.face_centers(
                [face_nodes[fi] for fi in np.nonzero(splitted_faces)[0]]
            )
        else:
            face_centers = face_centers[splitted_faces]
        new_vertices = np.vstack(
            [
                self.vertices,
                np.reshape(cell_centers[splitted_cells], (-1, 3)),
                np.reshape(face_centers, (-1, 3)),
            ]
        )
        # cell center (valid only for splitted cells)
        cc = (self.nb_vertices - 1) + np.cumsum(splitted_cells)
        cc[kept_cells] = np.iinfo(cc.dtype).max  # just to generate exeception if used
        # face center (valid only for splitted faces)
        fc = (self.nb_vertices + np.sum(splitted_cells) - 1) + np.cumsum(splitted_faces)
        fc[kept_faces] = np.iinfo(fc.dtype).max  # just to generate exeception if used
        return new_vertices, cc, fc

    def _new_cells(
        self,
        kept_cells,
        kept_faces,
        cell_centers=None,
        face_centers=None,
        reprocess_cellnodes=False,
    ):
        face_nodes = self.face_nodes
        cell_faces = self.cell_faces
        cell_nodes = self.cell_nodes(reprocess_cellnodes)
        if cell_centers is None:
            cell_centers = self._centers(cell_nodes)
        if face_centers is None:
            face_centers = self._centers(face_nodes)
        new_vertices, cc, fc = self._new_vertices(
            cell_centers, kept_cells, face_nodes, kept_faces, face_centers
        )
        new_cells = []
        for ci, kept in enumerate(kept_cells):
            if kept:
                new_cells.append(
                    [_reprocess_cell_nodes([face_nodes[fi] for fi in cell_faces[ci]])]
                )
            else:
                parts = []
                cci = cc[ci]
                faces = cell_faces[ci]
                for fi in faces:
                    if kept_faces[fi]:
                        parts.append(
                            list(face_nodes[fi]) + [cci,]
                        )
                    else:
                        fci = fc[fi]
                        nodes = face_nodes[fi]
                        for k in range(len(nodes)):
                            parts.append([nodes[k - 1], nodes[k], fci, cci])
                new_cells.append(parts)
        assert len(new_cells) == self.nb_cells
        original_cell = np.fromiter(
            chain.from_iterable(
                [ci,] * len(parts) for ci, parts in enumerate(new_cells)
            ),
            dtype=np.int64, # FIXME Ok ?
        )
        new_cells = list(chain.from_iterable(new_cells))
        new_cells = [np.array(c, dtype=np.int64) for c in new_cells] # FIXME Not optimal...
        return new_vertices, new_cells, original_cell

    def as_tets(self, cell_centers=None):
        vertices, cells, original = self._new_cells(
            self.tetrahedron_cells(), self.triangle_faces(), cell_centers=cell_centers,
        )
        cells = np.array(cells, dtype=np.int64) # FIXME Ok ?
        assert cells.ndim == 2 and cells.shape[1] == 4
        return TetMesh(vertices, cells), original

    def as_hybrid_mesh(self, cell_centers=None, face_centers=None):
        """
            Rewritten... To test! # FIXME
        """
        vertices, cells, original = self._new_cells(
            kept_cells=self.tetrahedron_cells() | self.hexahedron_cells(), # On ne garde que les tet & hex?
            kept_faces=self.triangle_faces() | self.quadrangle_faces(),
            cell_centers=cell_centers,
            face_centers=face_centers,
        )
        return HybridMesh(vertices, cells), original
