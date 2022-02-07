from petrelgridio.petrel_grid import PetrelGrid
from petrelgridio.uniform_mesh import HexMesh
from petrelgridio.vtu import to_vtu

from utils import *


def test_export_to_vtu():
    # Test model
    name = "Simple20x20x5_Fault.grdecl"

    # Creates a PetrelGrid object
    grid = PetrelGrid.build_from_files(DATA_FOLDER + name)
    hexa, vertices, _, _ = grid.process()

    ## Creates and exports HexMesh
    mesh = HexMesh(vertices, hexa)
    to_vtu(mesh, OUTPUT_FOLDER + name + "_hexmesh")
