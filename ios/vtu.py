import numpy as np

import vtkwriters as vtkw


def to_vtu(mesh, filename, **kwargs):
    if type(mesh) is RawMesh:
        cell_faces = mesh.cell_faces
        face_nodes = mesh.face_nodes
        vtu = vtkw.polyhedra_vtu_doc(
            mesh.vertices,
            [[face_nodes[face] for face in faces] for faces in cell_faces],
            **kwargs
        )
    else:
        offsets, cellsnodes = mesh.cells_nodes_as_COC()
        vtu = my_custom_vtu_doc_from_COC(
        # vtu = vtkw.vtu_doc_from_COC( # FIXME
            mesh.vertices, # Shape = (NbVertices * 3)
            offsets, # Size = NbCells
            cellsnodes, # Shape = (NbCells, 8), Values = vertices ids
            mesh.cells_vtk_ids(), # Size = NbCells, Values = 12 (VTK id for hexahedron)
            **kwargs
        )
    vtkw.write_vtu(vtu, filename)


def my_custom_vtu_doc_from_COC(
    vertices,
    offsets,
    connectivity,
    celltypes,
    fielddata=None,
    pointdata=None,
    celldata=None,
    ofmt="binary",
    integer_type=np.int32,
):
    """
        :param integer_type: Type to be used for cell types, connectivity and offsets.
    """
    offsets = offsets.astype(integer_type)
    celltypes = celltypes.astype(integer_type)
    connectivity = connectivity.astype(integer_type)
    doc = vtkw.vtk_doc("UnstructuredGrid", version="1.0")
    grid = vtkw.create_childnode(doc.documentElement, "UnstructuredGrid")
    piece = vtkw.create_childnode(
        grid,
        "Piece",
        {
            "NumberOfPoints": "%d" % vertices.shape[0],
            "NumberOfCells": "%d" % celltypes.shape[0],
        },
    )
    points = vtkw.create_childnode(piece, "Points")
    vtkw.add_dataarray(
        points, vertices.ravel(order="C"), "Points", nbcomp=3, ofmt=ofmt
    )
    cells = vtkw.create_childnode(piece, "Cells")
    vtkw.add_dataarray(
        cells, connectivity.ravel(order="C"), "connectivity", ofmt=ofmt
    )
    vtkw.add_dataarray(cells, offsets, "offsets", ofmt=ofmt)
    vtkw.add_dataarray(cells, celltypes, "types", ofmt=ofmt)
    _add_all_data(piece, pointdata=pointdata, celldata=celldata)
    _add_field_data(grid, fielddata, ofmt=ofmt)
    return doc


def _add_all_data(node, pointdata=None, celldata=None, ofmt="binary"):
    vtkw.add_piece_data(node, "PointData", pointdata, ofmt=ofmt)
    vtkw.add_piece_data(node, "CellData", celldata, ofmt=ofmt)


def _add_field_data(node, data, ofmt="binary"):
    if data is not None:
        vtkw.add_piece_data(node, "FieldData", data, ofmt=ofmt, isfield=True)
