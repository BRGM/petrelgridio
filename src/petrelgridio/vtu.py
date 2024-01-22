import vtkwriters as vtkw

from .raw_mesh import RawMesh


def to_vtu(mesh, filename, **kwargs):
    filename = str(filename)
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
        vtu = vtkw.vtu_doc_from_COC(
            mesh.vertices,  # Shape = (NbVertices * 3)
            offsets,  # Size = NbCells
            cellsnodes,  # Shape = (NbCells, 8), Values = vertices ids
            mesh.cells_vtk_ids(),  # Size = NbCells, Values = 12 (VTK id for hexahedron)
            **kwargs
        )
    vtkw.write_vtu(vtu, filename)
