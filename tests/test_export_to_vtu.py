from geometry.petrel_grid import PetrelGrid
from geometry.uniform_mesh import HexMesh
from geometry.raw_mesh import RawMesh
from ios.vtu import to_vtu
from tests.utils import *


def test_export_to_vtu():
    ##### Test models #####

    # name = "20X20X6.GRDECL"
    # name = "Simple3x3x1.grdecl"
    # name = "Simple3x3x3.grdecl"
    # name = "Simple20x20x5.grdecl"
    name = "Simple20x20x5_Fault.grdecl"

    ##### Build grid #####
    grid = PetrelGrid.build_from_files(data_folder + name)

    ##### Build hex mesh #####
    hexa, vertices, cell_faces, face_nodes = grid.process()
    mesh = HexMesh(vertices, hexa)
    to_vtu(mesh, name)

    ##### Build raw mesh #####
    raw_mesh = RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
    hybrid_mesh, original_cell = raw_mesh.as_hybrid_mesh()
    print(
        f"Splitted {name} mesh with: {hybrid_mesh.nb_vertices} vertices, {hybrid_mesh.nb_cells} cells, {hybrid_mesh.nb_faces} faces"
    )
    to_vtu(hybrid_mesh, f"{name}_splitted", celldata={"original_cell": original_cell})

    ##### Build with faults #####
    # TODO 


def test_dummy_grid():
    for test in BasicTest:
        run_basic_test(test)
