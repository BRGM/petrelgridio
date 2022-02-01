from petrelgridio.petrel_grid import PetrelGrid
from petrelgridio.uniform_mesh import HexMesh
from petrelgridio.raw_mesh import RawMesh
from petrelgridio.vtu import to_vtu
from utils import *


def test_export_to_vtu():
    ##### Test models #####

    # name = "20X20X6.GRDECL"
    # name = "Simple3x3x1.grdecl"
    # name = "Simple3x3x3.grdecl"
    # name = "Simple20x20x5.grdecl"
    name = "Simple20x20x5_Fault.grdecl"

    # Creates a PetrelGrid object
    grid = PetrelGrid.build_from_files(data_folder + name)
    hexa, vertices, cell_faces, face_nodes = grid.process()

    # Without faults
    ## Creates and exports HexMesh
    mesh = HexMesh(vertices, hexa)
    to_vtu(mesh, output_folder + name + "_hexmesh")
    ## Creates and exports RawMesh
    mesh = RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
    print(
        f"Original {name} mesh with: {mesh.nb_vertices} vertices, {mesh.nb_cells} hexaedra, {mesh.nb_faces} faces"
    )
    to_vtu(mesh, output_folder + name + "_rawmesh")

    # Split at faults
    # Creates and exports HybridMesh
    vertices, cells_faces, faces_nodes = grid.process_faults(hexa)
    mesh = RawMesh(vertices=vertices, face_nodes=faces_nodes, cell_faces=cells_faces)
    mesh, original_cell = mesh.as_hybrid_mesh()
    print(
        f"Splitted {name} mesh with: {mesh.nb_vertices} vertices, {mesh.nb_cells} cells, {mesh.nb_faces} faces"
    )
    to_vtu(
        mesh,
        output_folder + name + "_hybridmesh",
        celldata={"original_cell": original_cell},
    )  # FIXME celldata ?


def test_dummy_grid():
    for test in BasicTest:
        run_basic_test(test)


def test_dummy_grid_fault_ramp():
    run_basic_test(BasicTest.RAMP)
