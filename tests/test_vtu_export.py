import numpy as np

from petrelgridio.vtu import to_vtu
from petrelgridio.raw_mesh import RawMesh

from utils import OUTPUT_FOLDER


def test_export_vtu():
    class MyMesh:
        pass

    mesh = RawMesh(
        vertices=np.array(
            [
                (-1, -1, 0),
                (1, -1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, -1, 0),
            ],
            dtype="d",
        ),
        face_nodes=[
            (0, 4, 1, 2),
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
        ],
        cell_faces=[(0, 1, 2, 3)],
    )

    to_vtu(mesh, OUTPUT_FOLDER + "foo.vtu")
