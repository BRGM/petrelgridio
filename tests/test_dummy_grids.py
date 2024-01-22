from enum import Enum

from petrelgridio.petrel_grid import PetrelGrid
import petrelgridio.dummy_petrel_grids as dg
from petrelgridio.uniform_mesh import HexMesh
from petrelgridio.raw_mesh import RawMesh
from petrelgridio.vtu import to_vtu

from utils import OUTPUT_FOLDER


class BasicTest(Enum):
    COMMON_NODE = "common_node"
    SUGAR_BOX = "sugar_box"
    STAIRS = "stairs"
    RAMP = "ramp"

    @property
    def name(self):
        return self.value

    @property
    def file_name(self):
        """Returns the name to give to the output VTU file (without the".vtu" extension)"""
        return OUTPUT_FOLDER / self.name

    def create_data(self):
        """Returns the input list of hexahedra associated with the test"""
        if self is BasicTest.COMMON_NODE:
            return dg.common_node()
        elif self is BasicTest.SUGAR_BOX:
            return dg.grid_of_heaxaedra((4, 3, 2))
        elif self is BasicTest.STAIRS:
            return dg.four_cells_stairs()
        elif self is BasicTest.RAMP:
            return dg.faulted_ramp((8, 2, 1), begin=0.33)
        else:
            raise AttributeError("Test in not known in BasicTest.create_data()")


def run_basic_test(test):
    assert isinstance(test, BasicTest)
    hexahedra = test.create_data()
    pillars = dg.pillars(hexahedra)

    # Creates a PetrelGrid object
    grid = PetrelGrid.build_from_arrays__for_dummy_grids(  # FIXME pillars[..., 3:] is empty?
        pillars[..., :3], pillars[..., 3:], hexahedra[..., 2]
    )
    hexa, vertices, cells_faces, faces_nodes = grid.process()
    dg.depth_to_elevation(vertices)

    # Without faults
    ## Creates and exports HexMesh
    mesh = HexMesh(vertices, hexa)
    to_vtu(mesh, str(test.file_name) + "_hexmesh")
    ## Creates and exports RawMesh
    mesh = RawMesh(vertices=vertices, face_nodes=faces_nodes, cell_faces=cells_faces)
    print(
        f"Original {test.name} mesh with: {mesh.nb_vertices} vertices, {mesh.nb_cells} hexaedra, {mesh.nb_faces} faces"
    )
    to_vtu(mesh, str(test.file_name) + "_rawmesh")

    # Split at faults
    # Creates and exports HybridMesh
    vertices, cells_faces, faces_nodes = grid.process_faults(hexa)
    mesh = RawMesh(vertices=vertices, face_nodes=faces_nodes, cell_faces=cells_faces)
    mesh, original_cell = mesh.as_hybrid_mesh()
    print(
        f"Splitted {test.name} mesh with: {mesh.nb_vertices} vertices, {mesh.nb_cells} cells, {mesh.nb_faces} faces"
    )
    to_vtu(
        mesh,
        str(test.file_name) + "_hybridmesh",
        celldata={"original_cell": original_cell},
    )


def test_dummy_grids():
    for test in BasicTest:
        run_basic_test(test)
