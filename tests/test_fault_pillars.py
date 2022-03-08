import numpy as np

from petrelgridio.petrel_mesh import mesh
from data.fault_pillars import get_fault_pillars_data


def test_fault_pillars():
    v, e = get_fault_pillars_data()
    vertices, faces, faces_comp_ids, comp_contours = mesh(v, e)
    nb_vertices = vertices.shape[0]
    assert nb_vertices == 1 + np.max(faces)
    assert all(nb_vertices >= 1 + np.max(c) for c in comp_contours)
    assert len(comp_contours) == 1 + np.max(faces_comp_ids)
    assert np.array_equal(
        v,
        vertices[
            : v.shape[0],
        ],
    )
    print("Vertices:")
    for i, v in enumerate(vertices):
        print(f"{i} : {v}")
    print("Faces [3 x vertices ids] (component id)")
    for f, c_id in zip(faces, faces_comp_ids):
        print(f"{f} ({c_id})")
    print("Compnents contours:")
    for c_id, c in enumerate(comp_contours):
        print(f"{c_id}: {c}")
