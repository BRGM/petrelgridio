from pathlib import Path

from petrelgridio.petrel_grid import PetrelGrid
from petrelgridio.raw_mesh import RawMesh
from petrelgridio.vtu import to_vtu

dataroot = Path("data")
filename = Path("Simple20x20x5_Fault.grdecl")

grid = PetrelGrid.build_from_files(dataroot / filename)
hexa, vertices, cell_faces, face_nodes = grid.process()

vertices, cells_faces, faces_nodes = grid.process_faults(hexa)

mesh = RawMesh(vertices=vertices, face_nodes=faces_nodes, cell_faces=cells_faces)
to_vtu(mesh, filename.stem)
