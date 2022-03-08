from petrelgridio.petrel_grid import PetrelGrid
from petrelgridio.raw_mesh import RawMesh
from petrelgridio.vtu import to_vtu

from utils import DATA_FOLDER, OUTPUT_FOLDER


def test_export_to_vtu():
    # Test model
    name = "Simple20x20x5_Fault.grdecl"

    # Creates a PetrelGrid object
    grid = PetrelGrid.build_from_files(DATA_FOLDER + name)
    hexa, vertices, cell_faces, face_nodes = grid.process()

    # Without faults
    ## Creates and exports RawMesh (from unsplitted PetrelGrid)
    mesh = RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
    print(
        f"Original {name} mesh with: {mesh.nb_vertices} vertices, {mesh.nb_cells} hexaedra, {mesh.nb_faces} faces"
    )
    to_vtu(mesh, OUTPUT_FOLDER + name + "_rawmesh")

    # Split at faults
    # Creates and exports RawMesh (from splitted PetrelGrid)
    vertices, cells_faces, faces_nodes = grid.process_faults(hexa)
    mesh = RawMesh(vertices=vertices, face_nodes=faces_nodes, cell_faces=cells_faces)
    print(
        f"Splitted {name} mesh with: {mesh.nb_vertices} vertices, {mesh.nb_cells} cells, {mesh.nb_faces} faces"
    )
    to_vtu(mesh, OUTPUT_FOLDER + name + "_rawmesh_faulted")
