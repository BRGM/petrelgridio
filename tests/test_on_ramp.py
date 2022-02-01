from petrelgridio.petrel_grid import PetrelGrid
from petrelgridio.raw_mesh import RawMesh
from petrelgridio.vtu import to_vtu
import petrelgridio.dummy_petrel_grids as dg

hexaedra = dg.faulted_ramp((8, 2, 1), begin=0.33)
pillars = dg.pillars(hexaedra)
grid = PetrelGrid.build_from_arrays__for_dummy_grids(
    pillars[..., :3], pillars[..., 3:], hexaedra[..., 2]
)
hexa, vertices, cell_faces, face_nodes = grid.process()
vertices, cells_faces, faces_nodes = grid.process_faults(hexa)
mesh = RawMesh(vertices=vertices, face_nodes=faces_nodes, cell_faces=cells_faces)
to_vtu(mesh, "ramp")
